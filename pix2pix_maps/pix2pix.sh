#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=rrg-ravanelm
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --nodes=1
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=lihaoyuwilson@gmail.com
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -x
set -euo pipefail

# ------------------- CONFIG -------------------
# Path to your tar archive of the maps dataset
MAPS_TAR="$HOME/links/projects/def-ravanelm/datasets/maps.tar.gz"

# Path to your pix2pix code folder (train.py + receipt.yaml)
CODE_DIR="/home/lihaoyu/pix2pix_maps"

# Output directory for logs/checkpoints/samples
OUT_ROOT="$HOME/lihaoyu/links/scratch/pix2pix/maps_unet_patchgan_run1"
/home/lihaoyu/links/scratch
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
