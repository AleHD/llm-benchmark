import math
import toml
from pathlib import Path
from typing import TypeVar, Literal

from .runconfig import Model, RunConfig


Range = tuple[int, int | Literal[math.inf]]

def generate(out: Path, tps: Range, dps: Range, pps: Range, gpus: Range,
             mbzs: Range,  models: set[Model] = {Model.llama_3_8b},
             seqs: set[int] = {4096}):

    # Initial configuration.
    min_tp, max_tp = tps
    min_dp, max_dp = dps
    min_pp, max_pp = pps
    min_gpu, max_gpu = gpus
    min_mbz, max_mbz = mbzs
    max_gpu = min(max_gpu, max_tp*max_dp*max_pp)
    assert max_gpu < math.inf, "tp,dp,pp,gpu specification range is not finite"
    assert max_mbz < math.inf, "micro_batch_size range must be finite"

    # All combinations.
    configs = []
    for model in models:
        for seq in seqs:
            for gpu in range(min_gpu, max_gpu + 1):
                for tp in range(min_tp, min(max_tp, max_gpu) + 1):
                    for dp in range(min_dp, min(max_dp, max_gpu) + 1):
                        for pp in range(min_pp, min(max_pp, max_gpu) + 1):
                            for mbz in range(min_mbz, max_mbz + 1):
                                if tp*dp*pp == gpu and 256 % (mbz*dp) == 0:
                                    configs.append({"tp": tp, "dp": dp, "pp": pp, "model": model.value,
                                                    "sequence_length": seq, "micro_batch_size": mbz})

    # Save as toml.
    with open(out, "w+") as f:
        toml.dump({"configs": configs}, f)

def generate_mbz_adapt(out: Path, models: set[Model] = {Model.llama_3_8b}, 
                       gpus_per_node: int=4, seqs: set[int] = {4096}) -> list[dict]:
    assert gpus_per_node == 4
    tps = [1, 2, 4]
    dps = [2, 4]
    pps = [1, 2, 4, 8]
    num_mb_factors = [2,4]
    min_gpus, max_gpus = 16, 32

    configs = []
    for model in models:
        for seq in seqs:
            for tp in tps:
                for dp in dps:
                    for pp in pps:
                        if tp*dp*pp < min_gpus or tp*dp*pp > max_gpus:
                            continue
                        for factor in num_mb_factors:
                            num_mb = factor * pp
                            configs.append({"tp": tp, "dp": dp, "pp": pp, "model": model.value,
                                            "sequence_length": seq, "num_micro_batch": num_mb})    
    # Save as toml.
    with open(out, "w") as f:
        toml.dump({"configs": configs}, f)
    return configs
