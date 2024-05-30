from typing import Tuple, List, Dict, Any
from copy import deepcopy
import hydra
import math
from hydra_run import schedule_runs
import omegaconf
from omegaconf import OmegaConf, DictConfig, open_dict, ListConfig
import os
import yaml

def expand_opt_cfg(opt_cfg: DictConfig, cfg: DictConfig) -> None:
    with open_dict(opt_cfg):
        if opt_cfg.recompute_per_n_layers and opt_cfg.recompute_per_n_layers > 0:
            opt_cfg.recompute_granularity = "full"
            opt_cfg.recompute_method = "uniform"
            opt_cfg.recompute_num_layers = opt_cfg.recompute_per_n_layers
        del opt_cfg.recompute_per_n_layers
        
        if opt_cfg.tp_overlap:
            opt_cfg.tp_comm_overlap = True
            opt_cfg.tp_comm_overlap_rs_dgrad = True
        elif opt_cfg.tp_overlap is not None:
            opt_cfg.tp_comm_overlap = False
            opt_cfg.disable_tp_comm_overlap_ag = True
            opt_cfg.disable_tp_comm_overlap_rs = True
            opt_cfg.disable_tp_comm_bulk_dgrad = True
            opt_cfg.disable_tp_comm_bulk_wgrad = True
            opt_cfg.disable_tp_comm_split_ag = True
            opt_cfg.disable_tp_comm_split_rs = True
        del opt_cfg.tp_overlap
        
        if opt_cfg.dp_overlap:
            opt_cfg.overlap_grad_reduce = True
            opt_cfg.overlap_param_gather = True
        del opt_cfg.dp_overlap
        
        if opt_cfg.interleaved_pp:
            num_layers = cfg.framework.framework_setting.network.num_layers
            num_pp = cfg.framework.framework_setting.distributed.pipeline_model_parallel_size
            opt_cfg.num_layers_per_virtual_pipeline_stage = num_layers // num_pp // 2
        del opt_cfg.interleaved_pp

def estimate_memory_usage_in_gb(tp: int, pp: int, dp: int, act_layers: int, model_size_in_B: int, vocab_size: int, network_cfg: DictConfig) -> Tuple[float, float]:
    # assume bf16, zero1, 1f1b
    # according to https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/theoretical_memory_usage.py#L92
    
    seq_length = network_cfg.seq_length
    hidden_size = network_cfg.hidden_size
    ffn_hidden_size = network_cfg.ffn_hidden_size
    model_size = model_size_in_B * 10e9
    num_layers = network_cfg.num_layers
    
    model_memory = (6 + 12.0 / dp) * model_size / tp / pp
    const_no_recompute = 36
    
    if act_layers == 0:
        # no recompute
        act_memory_per_mbs = const_no_recompute * seq_length * hidden_size * num_layers
    elif act_layers > 0:
        act_memory_per_mbs = (const_no_recompute * act_layers + 2 * num_layers / act_layers) * seq_length * hidden_size
    else:
        raise ValueError("act_layers should be non-negative.")
    
    # input to embedding
    act_memory_per_mbs += 8 * seq_length * pp
    # embedding dropout?
    # act_memory_per_mbs += seq_length * hidden_size * mbs * pp
    
    if pp == 1:
        act_memory_per_mbs += seq_length * hidden_size * 4 * (1 + vocab_size / hidden_size)
    
    act_memory_per_mbs /= tp
    print(f"model_memory: {model_memory / 10e9}, act_memory_per_mbs: {act_memory_per_mbs / 10e9}")
    return model_memory / 10e9, act_memory_per_mbs / 10e9

def generate_candidate_configs(train_cfg: DictConfig, network_cfg: DictConfig, model_size_in_b: int, vocab_size: int) -> List[Dict[str, int]]:
    valid_configs = []
    
    tensor_parallel_sizes = train_cfg.tensor_parallel_sizes
    pipeline_parallel_sizes = train_cfg.pipeline_parallel_sizes
    max_pp_size = train_cfg.max_pipeline_parallel_size
    max_dp_size = train_cfg.max_data_parallel_size
    micro_batch_sizes = train_cfg.micro_batch_sizes
    act_ckpt_layers = train_cfg.act_ckpt_layers
    num_layers = network_cfg.num_layers
    
    nodes = train_cfg.get("num_nodes")
    gpus_per_node = train_cfg.get("gpus_per_node")
    assert gpus_per_node == 4
    gpu_memory_gb = train_cfg.get("gpu_memory_gb")    
    gpu_count = nodes * gpus_per_node    
    
    if tensor_parallel_sizes == "auto":
        tp_l = [1, 2, 4]
    elif isinstance(tensor_parallel_sizes, ListConfig):
        tp_l = OmegaConf.to_container(tensor_parallel_sizes, resolve=True)
    elif isinstance(tensor_parallel_sizes, int):
        tp_l = [tensor_parallel_sizes]
        
    if pipeline_parallel_sizes == "auto":
        pp_l = [x for x in range(1, num_layers + 1) if num_layers % x == 0]
        if max_pp_size:
            pp_l = [x for x in pp_l if x <= max_pp_size]
    elif isinstance(pipeline_parallel_sizes, ListConfig):
        pp_l = OmegaConf.to_container(pipeline_parallel_sizes, resolve=True)
    elif isinstance(pipeline_parallel_sizes, int):
        pp_l = [pipeline_parallel_sizes]
        
    if micro_batch_sizes == "auto":
        mbs_l = None
    elif isinstance(micro_batch_sizes, ListConfig):
        mbs_l = OmegaConf.to_container(micro_batch_sizes, resolve=True)
    elif isinstance(micro_batch_sizes, int):
        mbs_l = [micro_batch_sizes]
    
    # print(type(act_ckpt_layers))
    if act_ckpt_layers == "auto":
        act_layers_l = [0, 1]
    elif isinstance(act_ckpt_layers, ListConfig):
        act_layers_l = OmegaConf.to_container(act_ckpt_layers, resolve=True)
    elif isinstance(act_ckpt_layers, int):
        act_layers_l = [act_ckpt_layers]
    
    for tp in tp_l:
        for pp in pp_l:
            if gpu_count % (tp * pp) != 0:
                continue
            dp = gpu_count // (tp * pp)
            if max_dp_size and dp > max_dp_size:
                continue
            for act_layers in act_layers_l:
                if act_layers >= num_layers // pp:
                    continue
                old_mbs_l = mbs_l
                if mbs_l is None:
                    # auto select mbs 
                    safe_factor = 4
                    model_mem, act_mem_per_mbs = estimate_memory_usage_in_gb(tp, pp, dp, act_layers, model_size_in_b, vocab_size, network_cfg)
                    max_mbs = math.floor((gpu_memory_gb - model_mem) / act_mem_per_mbs / safe_factor)
                    if max_mbs <= 1:
                        mbs_l = [1]
                    elif max_mbs <= 2:
                        mbs_l = [1, 2]
                    else:
                        log_max_mbs = int(math.log2(max_mbs))
                        mbs_l = [2 ** i for i in [0, log_max_mbs//2, log_max_mbs]]
                for mbs in mbs_l:
                    gbs = dp * mbs * pp * 4
                    valid_configs.append({"tp": tp, "pp": pp, "dp": dp, "mbs": mbs, "gbs": gbs, "act_layers": act_layers})
                mbs_l = old_mbs_l

    return valid_configs                  
    
    
@hydra.main(config_path="conf", config_name="config")
def run_megatron_configs(cfg: omegaconf.dictconfig.DictConfig) -> None:
    """
    Main function in the entire pipeline, it reads the config using hydra and calls search_config.
    :param omegaconf.dictconfig.DictConfig cfg: OmegaConf object, read using the @hydra.main decorator.
    :return: None
    """
    model_type: str = cfg.model.model_type
    vocab_size: int = cfg.model.vocab_size
    model_name, model_size = model_type.split("_")
    model_size = int(model_size[:-1])
    assert model_name in ["llama3",], "Now only llama3 is supported."
    
    model_framework_cfg: DictConfig = cfg.framework.model_framework[model_name]
    megatron_network_cfg: DictConfig = cfg.framework.framework_setting.network
    model_size_cfg: DictConfig = cfg.model
    
    for k in megatron_network_cfg.keys():
        if k in model_size_cfg.keys():
            megatron_network_cfg[k] = model_size_cfg[k]
        if k in model_framework_cfg.keys():
            megatron_network_cfg[k] = model_framework_cfg[k]
        
    megatron_network_cfg.max_position_embeddings = megatron_network_cfg.seq_length
    megatron_network_cfg.group_query_attention = False if megatron_network_cfg.num_attention_heads == megatron_network_cfg.num_query_groups else True
    # del cfg.model
    # del cfg.model_framework
    
    train_cfg: DictConfig = cfg.train_settings
    gpus_per_node = train_cfg.get("gpus_per_node")
    assert gpus_per_node == 4
    
    # Logging config
    log_dir = cfg.logs_dir
    os.makedirs(log_dir, exist_ok=True)
    
    valid_configs = generate_candidate_configs(train_cfg, megatron_network_cfg, model_size, vocab_size)
    run_configs = []
    for config in valid_configs:
        new_cfg = deepcopy(cfg)
        new_cfg.framework.framework_setting.distributed.tensor_model_parallel_size = config["tp"]
        new_cfg.framework.framework_setting.distributed.pipeline_model_parallel_size = config["pp"]
        new_cfg.framework.framework_setting.distributed.use_distributed_optimizer = True if config["dp"] > 1 else False
        new_cfg.framework.framework_setting.training.micro_batch_size = config["mbs"]
        new_cfg.framework.framework_setting.training.global_batch_size = config["gbs"]
        new_cfg.framework.framework_setting.optimization.recompute_per_n_layers = config["act_layers"]
        
        # after setting distributed cfg, we need to expand the optimization cfg
        # TODO: check if args are valid according to the arguments.py
        
        megatron_opt_cfg: DictConfig = new_cfg.framework.framework_setting.optimization
        expand_opt_cfg(megatron_opt_cfg, megatron_network_cfg)
        run_configs.append(deepcopy(OmegaConf.to_container(new_cfg, resolve=True)))
    
    schedule_runs(run_configs, gpu_budget=4*32)

if __name__ == "__main__":
    run_megatron_configs()
