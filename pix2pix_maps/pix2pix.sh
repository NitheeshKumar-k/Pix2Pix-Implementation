#!/bin/bash
#SBATCH --job-name=maps-pix2pix
#SBATCH --time=24:0:0
#SBATCH --account=def-ravanelm
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lihaoyuwilson@gmail.com
#SBATCH --output=logs/%x-%j.out
set -x
set -euo pipefail

# ------------------- CONFIG -------------------
# Path to your tar archive of the maps dataset
MAPS_TAR="$HOME/projects/def-ravanelm/datasets/maps.tar.gz"

# Path to your pix2pix code folder (train.py + receipt.yaml)
CODE_DIR="/home/lihaoyu/Pix2Pix-Implementation/pix2pix_maps"

# Output directory for logs/checkpoints/samples
OUT_ROOT="/home/lihaoyu/scratch/results/pix2pix/job_${SLURM_JOB_ID}"
# ----------------------------------------------

# Activate Python environment
source "$HOME/myenv/bin/activate"

# Move to node-local scratch for speed
cd "$SLURM_TMPDIR"

# Copy dataset archive & extract
cp -v "$MAPS_TAR" .
tar -xzf "$(basename "$MAPS_TAR")"

# After extraction, dataset should be in $SLURM_TMPDIR/maps
DATA_DIR="$SLURM_TMPDIR/maps"

# Copy training code
mkdir -p pix2pix_run
cp -rv "$CODE_DIR"/* pix2pix_run/
cd pix2pix_run

# Run training
python train.py receipt.yaml \
    --data_folder="$DATA_DIR" \
    --output_folder="$OUT_ROOT" \
    --batch_size=16 \
    --lr=0.0002 \
    --lambda_L1=100.0 \
    --number_of_epochs=200

echo "Training completed successfully."
