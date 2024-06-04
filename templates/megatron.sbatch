#!/bin/bash

#SBATCH --job-name=megatron-{{ run_id }}
#SBATCH --nodes={{ n_nodes }}
#SBATCH --ntasks-per-node=1
#SBATCH --time={{ max_time }}
#SBATCH --output logs/%x.%j.out
#SBATCH --error logs/%x.%j.err
#SBATCH --chdir {{ run_root }}

# Misc initializations.
echo "Start: $(date)"
set -x
cat $0
#srun -ul --ntasks=$SLURM_NNODES hostname

# Set some variables.
export MASTER_PORT=25678
export MASTER_ADDR=$(hostname)
export WANDB_API_KEY=$(cat "{{ wandb_api_key }}")
export WANDB_DIR={{ run_root }}
export CWD={{ run_root }}
unset UENV_MOUNT_LIST
NAME="megatron-{{ run_id }}"
ARGS="-ul --environment={{ container_toml }}"
{{ env_vars }}

# Run main script.
srun $ARGS bash -c "
  # Variables.
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export PYTHONPATH={{ megatron_path }}
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export OMP_NUM_THREADS=32

  # Change cwd.
  cd {{ megatron_path }}

  # Main script.
  torchrun --node-rank=\${SLURM_PROCID} --rdzv_endpoint \${MASTER_ADDR}:\${MASTER_PORT} --rdzv_backend c10d --nnodes {{ n_nodes }} --nproc-per-node {{ n_proc_per_node }} {{ megatron_path }}/pretrain_gpt.py {{ megatron_args }} --wandb-save-dir {{ run_root }} --tensorboard-dir {{ run_root }} && echo completed > \${CWD}/status.txt || echo failed > \${CWD}/status.txt
"
