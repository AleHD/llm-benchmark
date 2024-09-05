#!/bin/bash

#SBATCH --job-name=nanotron-{{ run_id }}
#SBATCH --nodes={{ n_nodes }}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node={{ n_proc_per_node }}
#SBATCH --time=2:00:00
#SBATCH --output logs/%x_%j.out
#SBATCH --error logs/%x_%j.err
#SBATCH --chdir {{ run_root }}
##SBATCH --contiguous

# Misc initializations.
echo "Start: $(date)"
echo "Used nodes: $SLURM_NODELIST"
set -x
cat $0
#srun -ul --ntasks=$SLURM_NNODES hostname

# Set some variables.
export MASTER_PORT=25678
export MASTER_ADDR=$(hostname)
export HF_HOME=/capstor/scratch/cscs/ahernnde/huggingface
export WANDB_API_KEY=$(cat /users/ahernnde/workspace/repos/llm-benchmark/wandb_key)
export WANDB_DIR={{ run_root }}
export CWD={{ run_root }}
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ENV
NAME="nanotron-{{ run_id }}"
ARGS="-ul --environment=llm"

# Run main script.
export ENROOT_LIBRARY_PATH=/capstor/scratch/cscs/fmohamed/enroot-lib
srun $ARGS bash -c "
  # Variables.
  export PYTHONPATH=/users/ahernnde/project/repos/nanotron/src
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  #export OMP_NUM_THREADS=32

  # Change cwd.
  export PYTHONPATH=/users/ahernnde/project/repos/nanotron-swissai/src
  cd /users/ahernnde/project/repos/nanotron-swissai

  # Main script.
  torchrun --node-rank=\${SLURM_PROCID} --rdzv_endpoint \${MASTER_ADDR}:\${MASTER_PORT} --rdzv_backend c10d --nnodes {{ n_nodes }} --nproc-per-node {{ n_proc_per_node }} run_train.py --config-file $CWD/nanotron_config.yaml && echo completed > \${CWD}/status.txt || echo failed > \${CWD}/status.txt
"
