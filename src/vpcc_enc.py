from __future__ import annotations
import os, struct, math, cv2, imageio, shutil, tempfile
from pathlib import Path
from typing import List, Tuple, BinaryIO, NamedTuple, Dict, Iterable, Optional
import numpy as np
import torch
import scipy.sparse
from sklearn.neighbors import KDTree
from collections import deque

from src.cfg import _HDR_FMT, GLOBAL_HDR
from src.patch import Patch, PatchHeader
from src.patch import select_ppi, ppi_smooth, generate_patches
from src.patch import estimate_normals, propagate_orientation, orient_normals_outward
from src.packing import pack_patches, maps_for_frame

###############################################################################
# ╭──────────────────  BITSTREAM WRITER  ─────────────────────────╮
################################################################################

def write_patch_data(patches: List[Patch], fp: BinaryIO):
    for p in patches:
        h=PatchHeader(int(p.ppi), *map(int,p.shift3d), *map(int,p.atlas_xy),
                      int(p.size_uv[0]), int(p.size_uv[1]))
        fp.write(struct.pack(_HDR_FMT, *h))


def encode_frame(pts: torch.Tensor,
                 cols: torch.Tensor,
                 out_dir: Path,
                 frame_idx: int,
                 bit_depth: int = 10):
    # ------------------------------------------------------------------ setup
    pts_np  = pts.cpu().numpy().astype(np.float32)
    cols_np = cols.cpu().numpy().astype(np.float32)

    # global quantisation parameters -----------------------------------------
    Q        = 1 << bit_depth
    xyz_min  = pts_np.min(0)
    xyz_max  = pts_np.max(0)
    scale    = np.maximum(xyz_max - xyz_min, 1e-6)          # avoid /0

    # ------------------------------------------------------------------ 1) normals
    normals = estimate_normals(pts_np)
    propagate_orientation(pts_np, normals)
    orient_normals_outward(pts_np, normals)

    # ------------------------------------------------------------------ 2) PPI
    ppi = select_ppi(normals)
    ppi_smooth(pts_np, ppi)

    # ------------------------------------------------------------------ 3–5) patches → atlas
    patches = generate_patches(pts_np, cols_np, ppi,
                               bit_depth=bit_depth)
    pack_patches(patches)
    
    # # ---------- optional one-shot RD refinement ---------------------------------
    # patches = refine_patches_once(pts_np,
    #                             cols_np,
    #                             ppi,
    #                             patches,
    #                             bit_depth=bit_depth,
    #                             lod=2,                # same as generate_patches
    #                             err_thresh=0.8)       # half-voxel in 10-bit

    # ------------------------------------------------------------------ 6) maps
    occ, g0, g1, c0, c1 = maps_for_frame(patches,
                                         bit_depth=bit_depth)

    # ------------------------------------------------------------------ 7) write
    out_dir.mkdir(parents=True, exist_ok=True)

    # open **new file** for this frame – write global header first
    with open(out_dir / "patch_data.bin", "wb") as fp:
        fp.write(GLOBAL_HDR.pack(*(xyz_min.tolist() + scale.tolist())))
        write_patch_data(patches, fp)

    imageio.imwrite(out_dir / f"occ_{frame_idx:04d}.png",  occ * 255)
    cv2.imwrite(str(out_dir / f"geom0_{frame_idx:04d}.png"), g0)
    cv2.imwrite(str(out_dir / f"geom1_{frame_idx:04d}.png"), g1)
    imageio.imwrite(out_dir / f"attr0_{frame_idx:04d}.png", c0)
    imageio.imwrite(out_dir / f"attr1_{frame_idx:04d}.png", c1)

    loaded_g0 = cv2.imread(str(out_dir / f"geom0_{frame_idx:04d}.png"), cv2.IMREAD_UNCHANGED)
    # ensure the loaded image is 16-bit and the same as the original
    assert loaded_g0.dtype == np.uint16, f"Loaded g0 is not 16-bit: {loaded_g0.dtype}"
    assert loaded_g0.shape == g0.shape, f"Loaded g0 shape {loaded_g0.shape} does not match original {g0.shape}"
    assert np.array_equal(loaded_g0, g0), "Loaded g0 does not match original"

    g0_255 = (g0 >> 2).astype(np.uint8)  # downscale to 8-bit for display
    g1_255 = (g1 >> 2).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"geom0_{frame_idx:04d}_8bit.png"), g0_255)
    cv2.imwrite(str(out_dir / f"geom1_{frame_idx:04d}_8bit.png"), g1_255)