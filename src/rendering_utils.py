import torch
import math
import os
import numpy as np
import imageio
import cv2
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    FoVPerspectiveCameras,
    AlphaCompositor,
    look_at_view_transform,
)

def generate_circular_cameras(
    n_frames: int = 60,
    deg_per_frame: float = 2.0,
    device: str = "cuda",
    base_dist: float = 1.0,
    base_elev: float = 10.0,
    base_azim: float = -90.0,
):
    """
    Generate a list of (R, T) camera transforms that revolve around a horizontal circle,
    maintaining the same distance=base_dist and elev=base_elev as the base viewpoint
    produced by look_at_view_transform(dist=1.0, elev=10, azim=-90).

    Args:
      n_frames: total number of frames to produce.
      deg_per_frame: how many degrees to shift azimuth each frame (constant speed).
      device: e.g. "cuda" or "cpu"
      base_dist: same as 'dist' in look_at_view_transform
      base_elev: same as 'elev' in look_at_view_transform
      base_azim: the initial azimuth angle at frame 0
    Returns:
      A list of length n_frames, each is (R, T) with shape (3,3), (3,) in PyTorch.
    """

    out = []
    for i in range(n_frames):
        # The azimuth for frame i is base_azim + i * deg_per_frame
        azim_i = base_azim + i * deg_per_frame

        # Construct the camera transform with that azimuth,
        # and keep the same dist, elev, up=(0,1,0) by default
        R, T = look_at_view_transform(
            dist=base_dist,
            elev=base_elev,
            azim=azim_i,
            device=device
        )
        # By default, look_at_view_transform returns shape (1,3,3) & (1,3)
        out.append((R[0], T[0]))

    return out

def render_ply(pts: torch.Tensor, cols: torch.Tensor, image_size=512, device="cuda"):
    """
    pts: shape [N,3], float
    cols: shape [N,3], float in [0..1] or [0..255], we can scale to [0..1].
    Returns: a [image_size, image_size, 3] CPU numpy
    """
    pts = pts.to(device)
    cols = cols.to(device)
    print(f"[PCLRenderer] Constructing point cloud with {pts.shape[0]} points.")
    point_cloud = Pointclouds(points=[pts], features=[cols])
    R, T = look_at_view_transform(dist=1.0, elev=10, azim=-90, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.003,  # might need adjusting
        points_per_pixel=10, 
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    compositor= AlphaCompositor(background_color=(1., 1., 1.))
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

    print(f"[PCLRenderer] Rendering point cloud.")
    rend = renderer(point_cloud)
    rend = rend[0]  # [H, W, 3]
    rend_cpu = rend.detach().cpu().numpy()
    return rend_cpu

def render_sequence_spiral(
    sequence,  
    out_dir="results/render_out",
    device="cuda",
    deg_per_frame: float = 2.0,
    base_dist: float = 1.0,
    base_elev: float = 10.0,
    base_azim: float = -90.0,
    n_frames=30,
    image_size=512,
    filename=None
):
    os.makedirs(out_dir, exist_ok=True)

    # Generate rendering trajectory
    poses = generate_circular_cameras(
        n_frames=n_frames, deg_per_frame=deg_per_frame, device=device,
        base_dist=base_dist, base_elev=base_elev, base_azim=base_azim
    )

    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.005,
        points_per_pixel=8,
    )
    compositor = AlphaCompositor(background_color=(1., 1., 1.))

    if n_frames != len(sequence):
        raise ValueError(f"n_frames {n_frames} != sequence length {len(sequence)}")

    video_path = os.path.join(out_dir, f"video.mp4")
    writer = imageio.get_writer(video_path, fps=30)

    for i in range(n_frames):
        (pts, cols) = sequence[i]
        pts = pts.to(device)
        cols = cols.to(device)
     
        pcl = Pointclouds(points=[pts], features=[cols])

        R, T = poses[i]  # (3,3) and (3,)
        cameras = FoVPerspectiveCameras(device=device, R=R[None], T=T[None])

        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

        # render
        rend = renderer(pcl)  # shape [1, H, W, 3]
        rend = rend[0]        # [H, W, 3]
        rend_cpu = rend.detach().cpu().numpy()
        rend_img = (rend_cpu * 255).astype(np.uint8)
        if filename is not None:
            imageio.imwrite(os.path.join(out_dir, f"{filename}_{i:04d}.png"), rend_img)
        else:
            imageio.imwrite(os.path.join(out_dir, f"frame_{i:04d}.png"), rend_img)
        print(f"[render_sequence_spiral] Frame {i}/{n_frames} rendered and saved to {out_dir}.")
        writer.append_data(rend_img)

    writer.close()
    print(f"[render_sequence_spiral] Video saved to {out_dir}.")