#!/bin/bash
#SBATCH --job-name=prime-rl-wordle
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1          # one launcher per node
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64           # you had 80; 64 is usually enough unless you need more dataloading
#SBATCH --time=72:00:00
#SBATCH -A epor32
#SBATCH -q acc_ehpc


# ---------------- Modules & env ----------------
module load bsc intel impi mkl hdf5 cuda/12.6 nccl/2.20.5 tensorrt/10.0.0-cuda12
module load gcc/11
#export CC=gcc CXX=g++

# Your offline venv
source ~/prime-rl_ver/bin/activate

# Scratch-first caches (avoid $HOME quota)
#export TMPDIR=/gpfs/scratch/epor32/amsimplicio/tmp
#export HF_HOME=/gpfs/scratch/epor32/amsimplicio/hf_home
#export HF_DATASETS_CACHE=/gpfs/scratch/epor32/amsimplicio/hf_datasets
#export TRANSFORMERS_CACHE=/gpfs/scratch/epor32/amsimplicio/hf_models
#export XDG_CACHE_HOME=/gpfs/scratch/epor32/amsimplicio/.cache
#mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" logs
#


export VLLM_WORKER_MULTIPROC_METHOD=spawn   # <- key fix

export VLLM_LOGGING_LEVEL=INFO

export OUTPUT_DIR=/gpfs/scratch/epor32/amsimplicio/prime-rl/prime-rl-runs/47-mix_$SLURM_JOB_ID
export INFERENCE_SERVER_API_KEY=your_secret_key
export VLLM_USE_RAY=0
export HF_HUB_OFFLINE=1
export HF_HOME=/gpfs/scratch/epor32/amsimplicio/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets

# ---------------- Run ----------------
N=${SLURM_NNODES:-1}
NUM_WORKERS=$(( 4 * (N - 1) ))

srun python prime_rl_multinode_launcher.py \
  --nproc-per-node 4 \
  --local-rank-filter "" \
  --inference-extra    "@ configs/mix/infer.toml --dtype bfloat16" \
  --orchestrator-extra "@ configs/mix/orch.toml --num_train_workers ${NUM_WORKERS}" \
  --trainer-extra      "@ configs/mix/train.toml"
