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
  parser.add_argument("--obs_weight", type=float, default=1.0, help="Observation weight for TSDF integration (default: 1.0).")
  parser.add_argument("--render_write", action="store_true", help="If set, write rendered images to output_dir.")
  return parser.parse_args()


def load_inputs(args):
  # Load depth volume
  depth = np.load(args.depth_npy)
  if depth.ndim == 4 and depth.shape[-1] == 1:
    depth = depth[..., 0]
  if depth.ndim != 3:
    raise ValueError("depth_npy must have shape (N, H, W, 1) or (N, H, W)")

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


def integrate_tsdf(
  depth_iter,
  intrinsics,
  extrinsics,
  voxel_size,
  sdf_trunc,
  sdf_trunc_multiplier=10.0,
  obs_weight=1.0,
  depth_scale=1.0,
  depth_max=50.0,
  color_iter=None,
  block_count=400000,
  block_resolution=16,
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
    # Force CUDA sync so kernel errors surface at the correct frame.
    if device.get_type() == o3d.core.Device.DeviceType.CUDA:
      o3d.core.cuda.synchronize()
    active_blocks = _get_active_block_count(volume)
    if active_blocks is not None:
      growth = active_blocks - prev_active_blocks
      print(
        "[TSDF] frame {}/{} active_blocks={} growth={:+d} capacity={}".format(
          i + 1, intrinsics.shape[0], active_blocks, growth, int(block_count)
        )
      )
      if active_blocks >= int(block_count):
        print("[WARN] VoxelBlockGrid reached block_count capacity; new regions may be dropped.")
      prev_active_blocks = active_blocks
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

  volume, fps = integrate_tsdf(
    depth,
    intrinsics,
    extrinsics,
    voxel_size=args.voxel_size,
    sdf_trunc=args.sdf_trunc,
    sdf_trunc_multiplier=args.sdf_trunc_multiplier,
    obs_weight=args.obs_weight,
    depth_scale=1.0,
    depth_max=args.depth_max,
    block_count=args.block_count,
    block_resolution=args.block_resolution,
    color_iter=_color_image_iter(image_paths),
  )
  print("Average FPS: {:.2f}".format(fps))

  os.makedirs(args.output_dir, exist_ok=True)

  print("Extracting mesh...")
  mesh = extract_mesh(volume)

  print("Saving mesh to mesh.ply...")
  mesh_path = os.path.join(args.output_dir, "mesh.ply")
  o3d.io.write_triangle_mesh(mesh_path, mesh)

  print("Rendering mesh from camera views (headless)...")
  H, W = depth.shape[1], depth.shape[2]
  rendered_images = render_mesh_batch(mesh, intrinsics, extrinsics, W, H)

  if args.render_write:
    for i, rendered in enumerate(rendered_images):
      render_name = "render_%06d.png" % i
      render_path = os.path.join(args.output_dir, render_name)
      cv2.imwrite(render_path, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))

  return mesh, rendered_images


if __name__ == "__main__":
  args = parse_args()
  run_fusion(args)
