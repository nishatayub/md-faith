#!/bin/bash
# SLURM template (sbatch). Fill in the partition, resources and time for your cluster; the values below are placeholders.
# Submit:  sbatch --export=ALL,ARGS="reextract --run <run-id> --extractor ollama:qwen2.5:32b" scripts/hpc/job.slurm
#SBATCH --job-name=mdfaith
#SBATCH --partition=CHANGE_ME_gpu_partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=mdfaith-%j.out

cd "$SLURM_SUBMIT_DIR"
# module load cuda    # uncomment / adjust if your cluster needs it for Ollama's bundled CUDA libraries
bash scripts/hpc/run_job.sh $ARGS
