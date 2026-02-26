#!/bin/bash
#SBATCH --job-name=demo_output
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --export=ALL

# 1. Create logs dir just in case
mkdir -p logs

# 2. Load environment
module load ffmpeg
module load anaconda
source activate cog

# 3. Sanity check
nvidia-smi
echo "Starting job on $(hostname)"
echo "Date: $(date)"

# 4. Navigate to Repo (VERIFY THIS PATH)
cd /home/ab575577/projects_fall_2025/video-gen-models/common_metrics_on_video_quality/

# 5. Run Training (Cleaned up spaces)
PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/physics_with_actions --id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81

PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/physics_with_captions--id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81

PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/baseline_with_actions --id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81

PYTHONPATH=. python demo_us.py   --generated_dir /home/ab575577/projects_fall_2025/video-gen-models/CogVideo/outputs/eval/baseline_with_captions --id_manifest /home/ab575577/projects_spring_2026/AeroBench/validation_set_full/val_subset.txt   --data_root /home/ab575577/projects_spring_2026/AeroBench   --size 384 --frames 81