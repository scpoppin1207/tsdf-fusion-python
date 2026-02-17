import argparse
import ctypes
import ctypes.util
import glob
import itertools
import os
import time

import cv2
import numpy as np

# Must be set before importing open3d.visualization.rendering
os.environ.setdefault("OPEN3D_RENDERING_HEADLESS", "true")

import open3d as o3d


def parse_args():
  parser = argparse.ArgumentParser(description="Run TSDF fusion from RGB-D and camera parameters.")
  parser.add_argument("--image_dir", required=True, help="Directory containing RGB images (PNG). Order will follow sorted filenames.")
  parser.add_argument("--depth_npy", required=True, help="Path to depth numpy file of shape (N, H, W, 1) or (N, H, W), depth in meters.")
  parser.add_argument("--intrinsics_npy", required=True, help="Path to intrinsics numpy file of shape (N, 3, 3).")
  parser.add_argument("--extrinsics_npy", required=True, help="Path to extrinsics numpy file of shape (N, 4, 4).")
  parser.add_argument("--output_dir", default=".", help="Directory to save mesh.ply and renders (default: current directory).")
  parser.add_argument("--voxel_size", type=float, default=0.02, help="TSDF voxel size in meters (default: 0.02).")
  parser.add_argument("--sdf_trunc", type=float, default=None, help="TSDF truncation distance in meters (optional override).")
  parser.add_argument("--sdf_trunc_multiplier", type=float, default=10.0, help="Final sdf_trunc = sdf_trunc_multiplier * voxel_size (default: 10.0).")
  parser.add_argument("--depth_max", type=float, default=50.0, help="Max depth used by TSDF integration in meters (default: 50.0).")
  parser.add_argument("--block_count", type=int, default=400000, help="VoxelBlockGrid block capacity (default: 400000).")
  parser.add_argument("--block_resolution", type=int, default=16, help="Voxels per block side (default: 16).")
  parser.add_argument("--dynamic_iters", type=int, default=2, help="Total TSDF iterations. Iter-0 is unfiltered; later iterations use dynamic masks.")
  parser.add_argument("--mask_open_kernel", type=int, default=3, help="Morphology OPEN kernel size for removing thin/noisy structures.")
  parser.add_argument("--mask_open_iters", type=int, default=1, help="Number of OPEN iterations.")
  parser.add_argument("--mask_close_kernel", type=int, default=7, help="Morphology CLOSE kernel size for filling small holes/gaps.")
  parser.add_argument("--mask_close_iters", type=int, default=1, help="Number of CLOSE iterations.")
  parser.add_argument("--mask_min_area", type=int, default=200, help="Remove connected components smaller than this area (pixels). Set 0 to disable.")
  parser.add_argument("--debug_vis_dir", type=str, default=None, help="Optional directory to save per-frame d_obs / d_pred / error_map visualizations.")
  parser.add_argument("--debug_vis_every", type=int, default=1, help="Save debug visualization every N frames (default: 1).")
  parser.add_argument("--render_write", action="store_true", help="If set, write rendered images to output_dir.")
  return parser.parse_args()


def load_inputs(args):
  # Load depth volume
  depth = np.load(args.depth_npy)
  if depth.ndim == 4 and depth.shape[-1] == 1:
    depth = depth[..., 0]
  if depth.ndim != 3:
    raise ValueError("depth_npy must have shape (N, H, W, 1) or (N, H, W)")
  
  for i in range(depth.shape[0]):
    print(f"Max Depth in frame {i}: {np.max(depth[i])} meters")

  # Load intrinsics and extrinsics
  intrinsics = np.load(args.intrinsics_npy)
  extrinsics = np.load(args.extrinsics_npy)

  if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
    raise ValueError("intrinsics_npy must have shape (N, 3, 3)")
  if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
    raise ValueError("extrinsics_npy must have shape (N, 4, 4)")

  n_from_depth = depth.shape[0]
  if intrinsics.shape[0] != n_from_depth or extrinsics.shape[0] != n_from_depth:
    raise ValueError("N mismatch among depth ({}), intrinsics ({}), extrinsics ({})".format(
      n_from_depth, intrinsics.shape[0], extrinsics.shape[0]
    ))

  # Collect and sort image paths (kept for interface compatibility; not used for TSDF when NoColor)
  image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
  if len(image_paths) != n_from_depth:
    raise ValueError("Number of images ({}) does not match N in depth ({})".format(
      len(image_paths), n_from_depth
    ))

  return image_paths, depth, intrinsics, extrinsics


def _get_device():
  if o3d.core.cuda.is_available():
    return o3d.core.Device("CUDA:0")
  print("[WARN] CUDA not available, falling back to CPU.")
  return o3d.core.Device("CPU:0")


def _make_o3d_intrinsics(intr, width, height):
  fx = float(intr[0, 0])
  fy = float(intr[1, 1])
  cx = float(intr[0, 2])
  cy = float(intr[1, 2])
  return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def _color_image_iter(image_paths):
  for p in image_paths:
    color_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if color_bgr is None:
      raise RuntimeError("Failed to read image: {}".format(p))
    yield cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)


def _get_active_block_count(volume):
  try:
    hm = volume.hashmap()
    # Open3D HashMap API differs slightly across versions/builds.
    if hasattr(hm, "size"):
      return int(hm.size())
    if hasattr(hm, "active_size"):
      return int(hm.active_size())
  except Exception:
    return None
  return None


def _compute_dynamic_mask(depth_obs, depth_pred):
  valid = np.logical_and(depth_obs > 0, depth_pred > 0)
  if np.count_nonzero(valid) < 64:
    return np.zeros_like(depth_obs, dtype=bool), None, 0.0

  valid_obs = depth_obs[depth_obs > 0]
  if valid_obs.size < 64:
    return np.zeros_like(depth_obs, dtype=bool), None, 0.0

  q05 = float(np.percentile(valid_obs, 5))
  q95 = float(np.percentile(valid_obs, 95))
  scene_span = max(q95 - q05, 1e-6)
  scene_scale = max(float(np.median(valid_obs)), scene_span)
  invalid_depth_range = (depth_obs>q95)

  # 深度自适应阈值：场景尺度提供统一底线，远距离像素获得更宽松的容差。
  tau_floor = float(0.015 * scene_scale)
  depth_term = 0.1 * depth_pred + 0.001 * np.square(depth_pred)
  depth_term = np.clip(depth_term, 0.0, None)
  tau = tau_floor + depth_term

  depth_diff = depth_obs - depth_pred
  dynamic = np.zeros_like(depth_obs, dtype=bool)
  # Dynamic is defined as foreground-appearing pixels:
  # observed depth is significantly closer than predicted depth.
  dynamic[valid] = depth_diff[valid] < -tau[valid]
  dynamic[invalid_depth_range]=0
  ratio = float(np.count_nonzero(dynamic & valid)) / float(np.count_nonzero(valid))
  return dynamic, tau, ratio


def _depth_to_color(depth, valid_mask):
  vis = np.zeros_like(depth, dtype=np.float32)
  if np.any(valid_mask):
    d = depth[valid_mask]
    lo = float(np.percentile(d, 1))
    hi = float(np.percentile(d, 99))
    scale = max(hi - lo, 1e-6)
    vis[valid_mask] = np.clip((depth[valid_mask] - lo) / scale, 0.0, 1.0)
  vis_u8 = (vis * 255.0).astype(np.uint8)
  return cv2.applyColorMap(vis_u8, cv2.COLORMAP_TURBO)


def _error_to_color(error, valid_mask):
  vis = np.zeros_like(error, dtype=np.float32)
  if np.any(valid_mask):
    e = error[valid_mask]
    hi = float(np.percentile(e, 99))
    scale = max(hi, 1e-6)
    vis[valid_mask] = np.clip(error[valid_mask] / scale, 0.0, 1.0)
  vis_u8 = (vis * 255.0).astype(np.uint8)
  return cv2.applyColorMap(vis_u8, cv2.COLORMAP_INFERNO)


def _postprocess_dynamic_mask(
  dynamic_mask,
  open_kernel=3,
  open_iters=1,
  close_kernel=7,
  close_iters=1,
  min_area=200,
):
  """Morphology cleanup for dynamic mask.

  Tuning guidance:
  - `open_kernel` / `open_iters`:
    Use to suppress thin streaks and scattered tiny noise blobs.
    Increase gradually if elongated noise remains.
  - `close_kernel` / `close_iters`:
    Use to fill small holes and broken boundaries inside true dynamic regions.
    Increase if objects look perforated.
  - `min_area`:
    Remove tiny connected components after morphology.
    Increase if many micro-blobs remain.
  """
  mask_u8 = (dynamic_mask.astype(np.uint8) * 255)

  if close_kernel > 1 and close_iters > 0:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_kernel), int(close_kernel)))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=int(close_iters))

  if open_kernel > 1 and open_iters > 0:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_kernel), int(open_kernel)))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=int(open_iters))

 

  if min_area > 0:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(mask_u8)
    for comp_id in range(1, num_labels):
      area = stats[comp_id, cv2.CC_STAT_AREA]
      if area >= int(min_area):
        cleaned[labels == comp_id] = 255
    mask_u8 = cleaned

  return mask_u8 > 0


def _save_debug_maps(
  debug_vis_dir,
  frame_idx,
  rgb_obs,
  rgb_pred,
  depth_obs,
  depth_pred,
  dynamic_mask=None,
  dynamic_mask_post=None,
):
  if rgb_obs is None:
    rgb_obs_bgr = np.zeros((depth_obs.shape[0], depth_obs.shape[1], 3), dtype=np.uint8)
    cv2.putText(rgb_obs_bgr, "rgb_obs unavailable", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
  else:
    rgb_obs_bgr = cv2.cvtColor(rgb_obs, cv2.COLOR_RGB2BGR)

  if rgb_pred is None:
    rgb_pred_bgr = np.zeros((depth_obs.shape[0], depth_obs.shape[1], 3), dtype=np.uint8)
    cv2.putText(rgb_pred_bgr, "rgb_pred unavailable", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
  else:
    rgb_pred_bgr = cv2.cvtColor(rgb_pred, cv2.COLOR_RGB2BGR)

  valid_obs = depth_obs > 0
  obs_vis = _depth_to_color(depth_obs, valid_obs)

  if depth_pred is None:
    pred_vis = np.zeros((depth_obs.shape[0], depth_obs.shape[1], 3), dtype=np.uint8)
    err_vis = pred_vis.copy()
    cv2.putText(pred_vis, "d_pred unavailable", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(err_vis, "error_map unavailable", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
  else:
    valid_pred = depth_pred > 0
    pred_vis = _depth_to_color(depth_pred, valid_pred)
    valid = np.logical_and(valid_obs, valid_pred)
    error = np.zeros_like(depth_obs, dtype=np.float32)
    error[valid] = np.abs(depth_obs[valid] - depth_pred[valid])
    err_vis = _error_to_color(error, valid)

  if dynamic_mask is None:
    mask_vis = np.zeros((depth_obs.shape[0], depth_obs.shape[1], 3), dtype=np.uint8)
    cv2.putText(mask_vis, "dynamic_mask unavailable", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
  else:
    mask_vis = np.zeros((depth_obs.shape[0], depth_obs.shape[1], 3), dtype=np.uint8)
    mask_vis[dynamic_mask] = (0, 0, 255)

  if dynamic_mask_post is None:
    mask_post_vis = np.zeros((depth_obs.shape[0], depth_obs.shape[1], 3), dtype=np.uint8)
    cv2.putText(mask_post_vis, "dynamic_mask_post unavailable", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
  else:
    mask_post_vis = np.zeros((depth_obs.shape[0], depth_obs.shape[1], 3), dtype=np.uint8)
    mask_post_vis[dynamic_mask_post] = (0, 255, 0)

  panel = np.concatenate([rgb_obs_bgr, rgb_pred_bgr, obs_vis, pred_vis, err_vis, mask_vis, mask_post_vis], axis=1)
  width = depth_obs.shape[1]
  cv2.putText(panel, "rgb_obs", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
  cv2.putText(panel, "rgb_pred", (width + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
  cv2.putText(panel, "d_obs", (2 * width + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
  cv2.putText(panel, "d_pred", (3 * width + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
  cv2.putText(panel, "error_map", (4 * width + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
  cv2.putText(panel, "dynamic_mask", (5 * width + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
  cv2.putText(panel, "dynamic_mask_post", (6 * width + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
  cv2.imwrite(os.path.join(debug_vis_dir, "debug_{:06d}.png".format(frame_idx)), panel)


def _prepare_raycast_cache(mesh):
  if len(mesh.vertices) == 0:
    return None
  mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
  scene = o3d.t.geometry.RaycastingScene()
  scene.add_triangles(mesh_t)
  tri = np.asarray(mesh.triangles)
  vcols = np.asarray(mesh.vertex_colors)
  return scene, tri, vcols


def _render_depth_rgb_from_mesh(mesh, cam_intr, cam_pose, width, height, raycast_cache=None):
  if len(mesh.vertices) == 0:
    return None, None

  if raycast_cache is None:
    raycast_cache = _prepare_raycast_cache(mesh)
  if raycast_cache is None:
    return None, None
  scene, tri, vcols = raycast_cache

  world_to_cam = np.linalg.inv(cam_pose)
  rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
    intrinsic_matrix=o3d.core.Tensor(cam_intr, dtype=o3d.core.Dtype.Float64),
    extrinsic_matrix=o3d.core.Tensor(world_to_cam, dtype=o3d.core.Dtype.Float64),
    width_px=int(width),
    height_px=int(height),
  )
  ans = scene.cast_rays(rays)
  pred_depth = ans["t_hit"].numpy().astype(np.float32)
  pred_depth[~np.isfinite(pred_depth)] = 0.0

  pred_rgb = np.zeros((height, width, 3), dtype=np.uint8)
  if tri.size > 0 and vcols.size > 0:
    mask = np.isfinite(ans["t_hit"].numpy())
    prim_ids = ans["primitive_ids"].numpy().astype(np.int64)
    uv = ans["primitive_uvs"].numpy()
    valid_mask = np.logical_and(mask, prim_ids >= 0)
    if np.any(valid_mask):
      tri_vids = tri[prim_ids[valid_mask]]
      c0 = vcols[tri_vids[:, 0]]
      c1 = vcols[tri_vids[:, 1]]
      c2 = vcols[tri_vids[:, 2]]
      u = uv[..., 0][valid_mask]
      v = uv[..., 1][valid_mask]
      w0 = 1.0 - u - v
      rgb = (w0[:, None] * c0 + u[:, None] * c1 + v[:, None] * c2)
      pred_rgb[valid_mask] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
  return pred_depth, pred_rgb


def compute_dynamic_masks_from_mesh(
  mesh,
  image_paths,
  depth,
  intrinsics,
  extrinsics,
  depth_max,
  debug_vis_dir=None,
  debug_vis_every=1,
  iter_idx=1,
  mask_open_kernel=3,
  mask_open_iters=1,
  mask_close_kernel=7,
  mask_close_iters=1,
  mask_min_area=200,
):
  n_imgs = depth.shape[0]
  masks = []
  debug_vis_every = max(1, int(debug_vis_every))
  iter_debug_dir = None
  if debug_vis_dir:
    iter_debug_dir = os.path.join(debug_vis_dir, "iter_{}".format(iter_idx))
    os.makedirs(iter_debug_dir, exist_ok=True)
  raycast_cache = _prepare_raycast_cache(mesh)
  for i in range(n_imgs):
    depth_obs = depth[i].astype(np.float32)
    depth_obs[np.logical_or(np.isnan(depth_obs), np.isinf(depth_obs))] = 0
    depth_obs[np.logical_or(depth_obs <= 0, depth_obs > depth_max)] = 0

    h, w = depth_obs.shape
    pred_depth, pred_rgb = _render_depth_rgb_from_mesh(
      mesh, intrinsics[i], extrinsics[i], w, h, raycast_cache=raycast_cache
    )
    if pred_depth is None:
      dynamic_mask = np.zeros_like(depth_obs, dtype=bool)
      dynamic_mask_post = dynamic_mask
      dyn_ratio = 0.0
    else:
      dynamic_mask_raw, _, dyn_ratio = _compute_dynamic_mask(depth_obs, pred_depth)
      dynamic_mask_post = _postprocess_dynamic_mask(
        dynamic_mask_raw,
        open_kernel=mask_open_kernel,
        open_iters=mask_open_iters,
        close_kernel=mask_close_kernel,
        close_iters=mask_close_iters,
        min_area=mask_min_area,
      )
      dynamic_mask = dynamic_mask_post
      valid_pair = np.logical_and(depth_obs > 0, pred_depth > 0)
      if np.count_nonzero(valid_pair) > 0:
        dyn_ratio = float(np.count_nonzero(dynamic_mask & valid_pair)) / float(np.count_nonzero(valid_pair))
      else:
        dyn_ratio = 0.0

    masks.append(dynamic_mask)
    # print("[MASK-ITER{}] frame {}/{} dynamic_mask_ratio={:.3f}".format(
    #   iter_idx, i + 1, n_imgs, dyn_ratio
    # ))
    if iter_debug_dir and (i % debug_vis_every == 0):
      rgb_obs = None
      if image_paths is not None:
        color_bgr = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
        if color_bgr is not None:
          rgb_obs = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
      if pred_depth is None:
        dynamic_mask_raw = None
      _save_debug_maps(
        iter_debug_dir,
        i,
        rgb_obs,
        pred_rgb,
        depth_obs,
        pred_depth,
        dynamic_mask=dynamic_mask_raw,
        dynamic_mask_post=dynamic_mask,
      )
  return masks


def integrate_tsdf(
  depth_iter,
  intrinsics,
  extrinsics,
  voxel_size,
  sdf_trunc,
  sdf_trunc_multiplier=10.0,
  depth_scale=1.0,
  depth_max=50.0,
  color_iter=None,
  block_count=400000,
  block_resolution=16,
  log_prefix="TSDF",
  dynamic_masks=None,
):
  device = _get_device()
  safety_depth_cap = 500.0

  if depth_max is None:
    depth_max = 50.0
  if float(depth_max) > safety_depth_cap:
    print(
      "[WARN] depth_max={} is too large for stable VBG integration. "
      "Clamping to safety cap {}m.".format(depth_max, safety_depth_cap)
    )
  depth_max = min(float(depth_max), safety_depth_cap)

  # Open3D 0.18 tensor TSDF API uses VoxelBlockGrid (GPU-capable).
  volume = o3d.t.geometry.VoxelBlockGrid(
    attr_names=("tsdf", "weight", "color"),
    attr_dtypes=(o3d.core.Dtype.Float32, o3d.core.Dtype.Float32, o3d.core.Dtype.Float32),
    attr_channels=((1), (1), (3)),
    voxel_size=float(voxel_size),
    block_resolution=int(block_resolution),
    block_count=int(block_count),
    device=device,
  )
  if sdf_trunc is None:
    sdf_trunc = float(sdf_trunc_multiplier) * float(voxel_size)
  trunc_voxel_multiplier = float(max(1.0, sdf_trunc / voxel_size))

  t0_elapse = time.time()
  n_imgs = 0
  prev_active_blocks = 0

  color_iter = iter(color_iter) if color_iter is not None else None
  for i, depth in enumerate(depth_iter):
    depth_im = depth.astype(np.float32)
    depth_im[np.logical_or(np.isnan(depth_im), np.isinf(depth_im))] = 0
    # Keep integration numerically stable for outdoor long-range depth tails.
    depth_im[np.logical_or(depth_im <= 0, depth_im > depth_max)] = 0
    if not np.any(depth_im > 0):
      print("[{}] frame {}/{} skipped (no valid depth after range filtering)".format(
        log_prefix, i + 1, intrinsics.shape[0]
      ))
      continue

    if dynamic_masks is not None:
      dyn_mask = dynamic_masks[i]
      if dyn_mask.shape != depth_im.shape:
        raise ValueError("dynamic mask shape mismatch at frame {}: mask {}, depth {}".format(
          i, dyn_mask.shape, depth_im.shape
        ))
      depth_im[dyn_mask] = 0
      if not np.any(depth_im > 0):
        print("[{}] frame {}/{} skipped (all valid depth filtered by dynamic mask)".format(
          log_prefix, i + 1, intrinsics.shape[0]
        ))
        continue

    h, w = depth_im.shape

    depth_t = o3d.t.geometry.Image(
      o3d.core.Tensor(depth_im, dtype=o3d.core.Dtype.Float32, device=device)
    )
    if color_iter is None:
      color_np = np.zeros((h, w, 3), dtype=np.float32)
    else:
      try:
        color_np = next(color_iter).astype(np.float32) / 255.0
      except StopIteration:
        raise ValueError("color_iter has fewer frames than depth_iter")
      if color_np.shape[:2] != (h, w):
        raise ValueError(
          "Color/depth shape mismatch at frame {}: color {}, depth {}".format(
            i, color_np.shape[:2], (h, w)
          )
        )
    color_t = o3d.t.geometry.Image(
      o3d.core.Tensor(color_np, dtype=o3d.core.Dtype.Float32, device=device)
    )
    # Open3D 0.18 expects camera matrices on CPU for these VBG APIs.
    intr_t = o3d.core.Tensor(intrinsics[i], dtype=o3d.core.Dtype.Float64, device=o3d.core.Device("CPU:0"))
    # Input extrinsics are fixed camera-to-world; Open3D integrate expects world-to-camera.
    world_to_cam = np.linalg.inv(extrinsics[i])
    extr_t = o3d.core.Tensor(world_to_cam, dtype=o3d.core.Dtype.Float64, device=o3d.core.Device("CPU:0"))

    block_coords = volume.compute_unique_block_coordinates(
      depth_t,
      intr_t,
      extr_t,
      float(depth_scale),
      float(depth_max),
      trunc_voxel_multiplier,
    )
    # obs_weight is kept for interface compatibility; VoxelBlockGrid does not expose per-frame weights.
    volume.integrate(
      block_coords,
      depth_t,
      color_t,
      intr_t,
      extr_t,
      float(depth_scale),
      float(depth_max),
      trunc_voxel_multiplier,
    )
    n_imgs += 1

  fps = n_imgs / max(1e-6, (time.time() - t0_elapse))
  return volume, fps


def extract_mesh(volume):
  try:
    mesh = volume.extract_triangle_mesh(weight_threshold=1.0)
  except RuntimeError as err:
    err_msg = str(err).lower()
    if "illegal memory access" not in err_msg:
      raise
    print("[WARN] CUDA mesh extraction failed with illegal memory access. Retrying on CPU copy.")
    if hasattr(volume, "cpu"):
      mesh = volume.cpu().extract_triangle_mesh(weight_threshold=1.0)
    else:
      raise
  mesh_legacy = mesh.to_legacy()
  mesh_legacy.compute_vertex_normals()
  return mesh_legacy


def _has_egl_runtime():
  lib = ctypes.util.find_library("EGL")
  if lib is None:
    return False
  try:
    ctypes.CDLL(lib)
  except OSError:
    return False
  return True


def _render_mesh_raycast_batch(mesh, intrinsics, extrinsics, width, height):
  mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
  scene = o3d.t.geometry.RaycastingScene()
  scene.add_triangles(mesh_t)
  tri = np.asarray(mesh.triangles)
  vcols = np.asarray(mesh.vertex_colors)

  rendered_images = []
  for i in range(extrinsics.shape[0]):
    cam_intr = intrinsics[i]
    cam_pose = extrinsics[i]
    world_to_cam = np.linalg.inv(cam_pose)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
      intrinsic_matrix=o3d.core.Tensor(cam_intr, dtype=o3d.core.Dtype.Float64),
      extrinsic_matrix=o3d.core.Tensor(world_to_cam, dtype=o3d.core.Dtype.Float64),
      width_px=int(width),
      height_px=int(height),
    )
    ans = scene.cast_rays(rays)

    t_hit = ans["t_hit"].numpy()
    mask = np.isfinite(t_hit)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if vcols.shape[0] > 0:
      prim_ids = ans["primitive_ids"].numpy().astype(np.int64)
      uv = ans["primitive_uvs"].numpy()
      valid_ids = prim_ids[mask]
      tri_vids = tri[valid_ids]
      c0 = vcols[tri_vids[:, 0]]
      c1 = vcols[tri_vids[:, 1]]
      c2 = vcols[tri_vids[:, 2]]
      u = uv[..., 0][mask]
      v = uv[..., 1][mask]
      w0 = 1.0 - u - v
      rgb = w0[:, None] * c0 + u[:, None] * c1 + v[:, None] * c2
      img[mask] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    else:
      normals = ans["primitive_normals"].numpy()
      dirs = rays.numpy()[..., 3:6]
      lambert = np.clip(-np.sum(normals * dirs, axis=2), 0.0, 1.0)
      gray = (lambert * 200.0 + 30.0).astype(np.uint8)
      img[mask, 0] = gray[mask]
      img[mask, 1] = gray[mask]
      img[mask, 2] = gray[mask]
    rendered_images.append(img)

  return rendered_images


def render_mesh_offscreen_batch(mesh, intrinsics, extrinsics, width, height):
  renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
  renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

  material = o3d.visualization.rendering.MaterialRecord()
  material.shader = "defaultLit"
  material.base_color = [1.0, 1.0, 1.0, 1.0]

  renderer.scene.add_geometry("mesh", mesh, material)

  rendered_images = []
  for i in range(extrinsics.shape[0]):
    cam_intr = intrinsics[i]
    cam_pose = extrinsics[i]

    # Open3D rendering expects extrinsic as world-to-camera. Our input is camera-to-world.
    world_to_cam = np.linalg.inv(cam_pose)
    intrinsic = _make_o3d_intrinsics(cam_intr, width, height)
    renderer.setup_camera(intrinsic, world_to_cam)

    img = renderer.render_to_image()
    img_np = np.asarray(img)
    if img_np.ndim == 3 and img_np.shape[2] == 4:
      img_np = img_np[:, :, :3]
    rendered_images.append(img_np)

  return rendered_images


def render_mesh_batch(mesh, intrinsics, extrinsics, width, height):
  if _has_egl_runtime():
    return render_mesh_offscreen_batch(mesh, intrinsics, extrinsics, width, height)

  print("[WARN] libEGL.so.1 not found. Falling back to RaycastingScene software headless renderer.")
  return _render_mesh_raycast_batch(mesh, intrinsics, extrinsics, width, height)

def run_fusion(args):
  image_paths, depth, intrinsics, extrinsics = load_inputs(args)
  if args.debug_vis_dir:
    os.makedirs(args.debug_vis_dir, exist_ok=True)

  total_iters = max(1, int(args.dynamic_iters))
  dynamic_masks = None
  volume = None
  mesh = None
  for it in range(total_iters):
    use_mask = dynamic_masks is not None
    print("TSDF iteration {}/{} (filtered={})".format(it + 1, total_iters, use_mask))
    volume, fps = integrate_tsdf(
      depth,
      intrinsics,
      extrinsics,
      voxel_size=args.voxel_size,
      sdf_trunc=args.sdf_trunc,
      sdf_trunc_multiplier=args.sdf_trunc_multiplier,
      depth_scale=1.0,
      depth_max=args.depth_max,
      block_count=args.block_count,
      block_resolution=args.block_resolution,
      log_prefix="TSDF-ITER{:02d}".format(it + 1),
      dynamic_masks=dynamic_masks,
      color_iter=_color_image_iter(image_paths),
    )
    print("Average FPS (iter {}): {:.2f}".format(it + 1, fps))
    mesh = extract_mesh(volume)
    next_masks = compute_dynamic_masks_from_mesh(
      mesh,
      image_paths,
      depth,
      intrinsics,
      extrinsics,
      depth_max=args.depth_max,
      debug_vis_dir=args.debug_vis_dir,
      debug_vis_every=args.debug_vis_every,
      iter_idx=it + 1,
      mask_open_kernel=args.mask_open_kernel,
      mask_open_iters=args.mask_open_iters,
      mask_close_kernel=args.mask_close_kernel,
      mask_close_iters=args.mask_close_iters,
      mask_min_area=args.mask_min_area,
    )
    if dynamic_masks is not None:
      change_ratios = []
      for k in range(len(next_masks)):
        prev = dynamic_masks[k]
        curr = next_masks[k]
        if prev.shape != curr.shape:
          continue
        change_ratios.append(float(np.count_nonzero(np.logical_xor(prev, curr))) / float(prev.size))
      if len(change_ratios) > 0:
        print("Mask update ratio (iter {} -> {}): mean={:.4f} max={:.4f}".format(
          it, it + 1, float(np.mean(change_ratios)), float(np.max(change_ratios))
        ))
    if it < total_iters - 1:
      dynamic_masks = next_masks

  os.makedirs(args.output_dir, exist_ok=True)

  print("Saving mesh to mesh.ply...")
  mesh_path = os.path.join(args.output_dir, "mesh.ply")
  o3d.io.write_triangle_mesh(mesh_path, mesh)

  # print("Rendering mesh from camera views (headless)...")
  # H, W = depth.shape[1], depth.shape[2]
  # rendered_images = render_mesh_batch(mesh, intrinsics, extrinsics, W, H)

  # if args.render_write:
  #   for i, rendered in enumerate(rendered_images):
  #     render_name = "render_%06d.png" % i
  #     render_path = os.path.join(args.output_dir, render_name)
  #     cv2.imwrite(render_path, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))

  return mesh


if __name__ == "__main__":
  args = parse_args()
  run_fusion(args)
