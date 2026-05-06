#!/bin/bash
#SBATCH --job-name=airscape_metrics
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=evc22
#SBATCH --time=3:00:00
#SBATCH --export=ALL

mkdir -p logs

module load ffmpeg
module load anaconda
source activate cog

nvidia-smi
echo "Starting job on $(hostname)"
echo "Date: $(date)"

cd /home/ab575577/projects_fall_2025/video-gen-models/common_metrics_on_video_quality/

PYTHONPATH=. python evaluate_aerobench_pairs.py \
  --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/airscape \
  --data_root /home/ab575577/projects_spring_2026/AeroBench \
  --size 384 \
  --frames 81 \
  --max_pairs 250
