import torch
import numpy as np
import os
import glob
from plyfile import PlyData, PlyElement
from tqdm import tqdm

def load_ply(path: str):
    """
    vertex_data[i]: 
    np.void((252.0, 247.0, 121.0, 155, 79, 12), 
    dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    """
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
    vertex_data = plydata['vertex'].data
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    r = vertex_data['red']
    g = vertex_data['green']
    b = vertex_data['blue']

    pts_np = np.stack([x, y, z], axis=-1)  
    cols_np= np.stack([r, g, b], axis=-1) 
    pts   = torch.from_numpy(pts_np).float()
    colors= torch.from_numpy(cols_np).clone()
    mins = pts.min(dim=0)[0]
    maxs = pts.max(dim=0)[0]

    # Normalize colors to [0..1]
    colors = colors.float()
    colors = colors / 255.0

    # Normalize points to [-1..1]
    center = (mins + maxs) / 2
    pts = pts - center  # recenter so the bounding box is around the origin
    scale = (maxs - mins).max().item()
    pts = pts / scale   # or do 1 / (some scale factor)

    return pts, colors

def save_ply(path: str, pts: np.ndarray, colors: np.ndarray):
    """
    Saves a point cloud to a PLY file.
    Args:
      pts: numpy array of shape [N,3] (float32)
      colors: numpy array of shape [N,3] (uint8)
    """
    # Build a structured array with fields for position and color.
    if colors is None:
        colors = np.zeros_like(pts)
    vertices = np.array(
        [(pts[i, 0], pts[i, 1], pts[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
         for i in range(pts.shape[0])],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el], text=True).write(path)
    print(f"[save_ply] Saved downsampled point cloud with {pts.shape[0]} points to {path}.")

def load_ply_sequence(folder: str,
                      nframes: int = 30,
                      cache_name='longdress',
                      start: int = 0):
    """
    Returns:
        a list of (pts, colors).
    """
    cache_path = f"cache/{cache_name}_{start}_{nframes}.pt"
    if os.path.exists(cache_path):
        print(f"[PCLRenderer] Loading sequence from cache/{cache_name}_{start}_{nframes}.pt")
        return torch.load(cache_path)
    
    ply_paths = sorted(glob.glob(os.path.join(folder, "*.ply")))
    if len(ply_paths) == 0:
        raise ValueError(f"No PLY files found in {folder}")
    sequence = []
    for path in tqdm(ply_paths[start:start+nframes]):
        pts, cols = load_ply(path)
        sequence.append((pts, cols))

    os.makedirs("cache", exist_ok=True)
    torch.save(sequence, cache_path)
    print(f"[PCLRenderer] Saved sequence to cache/{cache_name}_{start}_{nframes}.pt")
    return sequence

def downsample(pts: torch.Tensor, colors: torch.Tensor, num_samples: int, seed: int = 42):
    """
    Randomly subsamples the point cloud to at most num_samples points.
    A fixed random seed ensures reproducibility.
    
    Args:
      pts: [N, 3] tensor.
      colors: [N, 3] tensor.
      num_samples: int, maximum number of points to sample.
      seed: random seed.
    
    Returns:
      pts_down, colors_down: subsampled tensors.
    """
    N = pts.shape[0]
    if N > num_samples:
        generator = torch.Generator(device=pts.device)
        generator.manual_seed(seed)
        idx = torch.randperm(N, generator=generator)[:num_samples]
        return pts[idx], colors[idx]
    else:
        return pts, colors