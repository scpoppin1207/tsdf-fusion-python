#!/usr/bin/env bash

set -e

# Simple wrapper to run run_tsdf.py
# Usage:
#   ./run_tsdf.sh IMAGE_DIR DEPTH_NPY INTRINSICS_NPY EXTRINSICS_NPY [OUTPUT_DIR] [VOXEL_SIZE]
export CUDA_VISIBLE_DEVICES=3
SCENE_ID=158
ROOT_DIR="/home/FFGS/experiments/${SCENE_ID}_cam0/worldmirror_output"
IMAGE_DIR="${ROOT_DIR}/images"
DEPTH_NPY="${ROOT_DIR}/depth/depth.npy"
INTRINSICS_NPY="${ROOT_DIR}/cameras/pred_intrinsics.npy"
EXTRINSICS_NPY="${ROOT_DIR}/cameras/pred_extrinsics.npy"
OUTPUT_DIR="${ROOT_DIR}/tsdf_fusion"
VOXEL_SIZE=0.005
SDF_TRUNC=($VOXEL_SIZE * 10)
BLOCK_COUNT=800000



# Use Open3D headless CPU rendering (no display / no GPU graphics)
export EGL_PLATFORM=surfaceless

python run_tsdf.py \
  --image_dir "$IMAGE_DIR" \
  --depth_npy "$DEPTH_NPY" \
  --intrinsics_npy "$INTRINSICS_NPY" \
  --extrinsics_npy "$EXTRINSICS_NPY" \
  --output_dir "$OUTPUT_DIR" \
  --voxel_size "$VOXEL_SIZE" \
  --sdf_trunc "$SDF_TRUNC" \
  --block_count "$BLOCK_COUNT" \
  --render_write
