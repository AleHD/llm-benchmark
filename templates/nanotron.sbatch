#!/bin/bash

#SBATCH --job-name=nanotron-{{ run_id }}
#SBATCH --nodes={{ n_nodes }}
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
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
export HF_HOME=/bret/scratch/cscs/ahernnde/huggingface
export WANDB_API_KEY=$(cat /users/ahernnde/workspace/repos/llm-benchmark/wandb_key)
export CWD={{ run_root }}
NAME="nanotron-{{ run_id }}"
ARGS="-ul --environment=$PWD/slurm.toml"

# Run main script.
srun $ARGS bash -c "
  # Variables.
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export PYTHONPATH=/users/ahernnde/workspace/repos/nanotron/src
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export OMP_NUM_THREADS=32

  # Change cwd.
  cd /users/ahernnde/workspace/repos/nanotron/

  # Main script.
  torchrun --node-rank=\${SLURM_PROCID} --rdzv_endpoint \${MASTER_ADDR}:\${MASTER_PORT} --rdzv_backend c10d --nnodes {{ n_nodes }} --nproc-per-node {{ n_proc_per_node }} run_train.py --config-file $CWD/nanotron_config.yaml && echo completed > \${CWD}/status.txt || echo failed > \${CWD}/status.txt
"
