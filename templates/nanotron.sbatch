#!/bin/bash

#SBATCH --job-name=nanotron-{{ run_id }}
#SBATCH --nodes={{ n_nodes }}
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

# Misc initializations.
set -x
cat $0
srun -ul --ntasks=$SLURM_NNODES hostname

# Set some variables.
export MASTER_PORT=25678
export MASTER_ADDR=$(hostname)
export HF_HOME=/bret/scratch/cscs/ctianche/huggingface
export WANDB_API_KEY=$(cat /bret/scratch/cscs/ctianche/playground/llm-benchmark/wandb_key)
HOME={{ run_root }}
NAME="nanotron-{{ run_id }}"
ARGS="-ul --output=${HOME}/logs/%x_${NAME}_%j.log --container-writable --uenv=/bret/scratch/cscs/ctianche/nanotron-env/nanotron.squashfs:/user-environment"

# Run main script.
srun $ARGS bash -c "
  # Initialization.
  set -x
  uenv view default

  # Variables.
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export PYTHONPATH=/bret/scratch/cscs/ctianche/playground/nanotron/src
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export OMP_NUM_THREADS=16

  # Change cwd.
  cd /bret/scratch/cscs/ctianche/playground/nanotron/

  # Main script.
  torchrun --node-rank=\${SLURM_PROCID} --master-addr=\${MASTER_ADDR} --master-port=\${MASTER_PORT} --nnodes {{ n_nodes }} --nproc-per-node {{ n_proc_per_node }} run_train.py --config-file $HOME/nanotron_config.yaml && echo completed > \${HOME}/status.txt || echo failed > \${HOME}/status.txt
"
