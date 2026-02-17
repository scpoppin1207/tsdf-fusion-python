#!/usr/bin/env bash

set -e

# Simple wrapper to run run_tsdf.py
# Usage:
#   ./run_tsdf.sh IMAGE_DIR DEPTH_NPY INTRINSICS_NPY EXTRINSICS_NPY [OUTPUT_DIR] [VOXEL_SIZE]
export CUDA_VISIBLE_DEVICES=0
SCENE_ID=158
ROOT_DIR="/home/FFGS/experiments/${SCENE_ID}_cam0/worldmirror_output"
IMAGE_DIR="${ROOT_DIR}/images"
DEPTH_NPY="${ROOT_DIR}/depth/depth.npy"
INTRINSICS_NPY="${ROOT_DIR}/cameras/pred_intrinsics.npy"
EXTRINSICS_NPY="${ROOT_DIR}/cameras/pred_extrinsics.npy"
OUTPUT_DIR="${ROOT_DIR}/tsdf_fusion_dynamic"
VOXEL_SIZE=0.002
BLOCK_COUNT=300000
SDF_TRUNC_MULTIPLIER=15
DEPTH_MAX=2.5
DYNAMIC_ITERS=20
DEBUG_VIS_DIR="${OUTPUT_DIR}/debug_vis_morphology/${SDF_TRUNC_MULTIPLIER}"
DEBUG_VIS_EVERY=1
# Dynamic-mask morphology:
# - OPEN: remove thin streaks / tiny speckles.
# - CLOSE: fill small holes and boundary gaps.
# - MIN_AREA: remove tiny connected components after morphology.
MASK_OPEN_KERNEL=3
MASK_OPEN_ITERS=1
MASK_CLOSE_KERNEL=10
MASK_CLOSE_ITERS=1
MASK_MIN_AREA=200



# Use Open3D headless CPU rendering (no display / no GPU graphics)
export EGL_PLATFORM=surfaceless

python run_tsdf.py \
  --image_dir "$IMAGE_DIR" \
  --depth_npy "$DEPTH_NPY" \
  --intrinsics_npy "$INTRINSICS_NPY" \
  --extrinsics_npy "$EXTRINSICS_NPY" \
  --output_dir "$OUTPUT_DIR" \
  --voxel_size "$VOXEL_SIZE" \
  --block_count "$BLOCK_COUNT" \
  --sdf_trunc_multiplier "$SDF_TRUNC_MULTIPLIER" \
  --depth_max "$DEPTH_MAX" \
  --dynamic_iters "$DYNAMIC_ITERS" \
  --mask_open_kernel "$MASK_OPEN_KERNEL" \
  --mask_open_iters "$MASK_OPEN_ITERS" \
  --mask_close_kernel "$MASK_CLOSE_KERNEL" \
  --mask_close_iters "$MASK_CLOSE_ITERS" \
  --mask_min_area "$MASK_MIN_AREA" \
  --debug_vis_dir "$DEBUG_VIS_DIR" \
  --debug_vis_every "$DEBUG_VIS_EVERY" \
  --render_write
