from __future__ import annotations
import os, struct, math, cv2, imageio, shutil, tempfile
from pathlib import Path
from typing import List, Tuple, BinaryIO, NamedTuple, Dict, Iterable, Optional
import numpy as np
import torch
import scipy.sparse
from sklearn.neighbors import KDTree
from collections import deque
from scipy.ndimage import label as cc_label

from src.cfg import Cfg, _AXES, _AXIS_SIGN, _axis_indices
from src.packing import pack_patches, maps_for_frame


class Patch:
    """Container for one V-PCC patch."""
    __slots__ = ("pts","rgb","ppi","shift3d","u","v","d","bbox_uv","size_uv",
                 "atlas_xy","d_near","d_far","rgb_near","rgb_far")
    def __init__(self):
        self.pts=self.rgb=None
        self.ppi:int|None=None
        self.shift3d=None
        self.u=self.v=self.d=None
        self.bbox_uv=self.size_uv=self.atlas_xy=None
        self.d_near=self.d_far=self.rgb_near=self.rgb_far=None

class PatchHeader(NamedTuple):
    ppi: np.int8; shift_x: np.int32; shift_y: np.int32; shift_z: np.int32
    atlas_u: np.int16; atlas_v: np.int16; size_w: np.int16; size_h: np.int16

################################################################################
# ╭──────────────────  NORMALS & ORIENTATION  ────────────────────╮
################################################################################

def estimate_normals(pts: np.ndarray, k: int = Cfg.K_NEIGH) -> np.ndarray:
    tree = KDTree(pts)
    _, idx = tree.query(pts, k=k)
    normals = np.empty_like(pts)
    for i, nb in enumerate(idx):
        cov = np.cov(pts[nb].T)
        _, vec = np.linalg.eigh(cov)
        normals[i] = vec[:, 0]          # smallest eigen‑vector
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)

def propagate_orientation(pts: np.ndarray,
                          normals: np.ndarray,
                          k: int = Cfg.ORIENT_K):
    """
    Flip neighbouring normals so they are locally coherent.
    Uses a BFS seeded with the 6 extreme points (min/max X,Y,Z).
    """
    tree = KDTree(pts)
    visited = np.zeros(len(pts), bool)

    # six unique extreme-point indices
    extremes = {
        np.argmin(pts[:, ax]) for ax in range(3)
    }.union({
        np.argmax(pts[:, ax]) for ax in range(3)
    })
    queue = deque(extremes)
    visited[list(extremes)] = True

    while queue:
        i = queue.popleft()
        _, nb = tree.query([pts[i]], k=k)
        for j in nb[0][1:]:                # skip self
            if visited[j]:
                continue
            if np.dot(normals[i], normals[j]) < 0:
                normals[j] *= -1
            visited[j] = True
            queue.append(j)

def orient_normals_outward(pts: np.ndarray, normals: np.ndarray):
    """Flip normals so they point *away* from the centroid (TMC2 8.3.2)."""
    centre = pts.mean(0)
    to_pt  = pts - centre
    flip   = (np.sum(normals * to_pt, axis=1) < 0)
    normals[flip] *= -1

################################################################################
# ╭──────────────────  PPI SELECTION & SMOOTHING  ─────────────────╮
################################################################################

def select_ppi(normals: np.ndarray) -> np.ndarray:
    """Choose the projection-plane index exactly like TMC2.

    *For each point* pick the axis-aligned plane whose normal has the **most
    negative dot product** with the point normal.  This both selects an axis
    (X/Y/Z) *and* its sign (±).  Hence the result spans the full range 0‥5.
    """
    dots = normals @ _AXES.T          # (N,6) raw dot products, *signed*
    return np.argmin(dots, axis=1).astype(np.int8)

def ppi_smooth(pts: np.ndarray, ppi: np.ndarray, k: int = 8, iters: int = Cfg.PPI_SMOOTH_ITERS):
    tree = KDTree(pts)
    for _ in range(iters):
        _, idx = tree.query(pts, k=k)
        for i, nb in enumerate(idx):
            votes = np.bincount(ppi[nb], minlength=6)
            ppi[i] = votes.argmax()


################################################################################
# ╭──────────────────  PATCH SEGMENTATION  ───────────────────────╮
################################################################################

def _connected_components_mask(mask: np.ndarray, conn: int = 2) -> List[np.ndarray]:
    lab, n = cc_label(mask, structure=np.ones((3,3)) if conn == 2 else None)
    return [lab == i for i in range(1, n + 1)]

def generate_patches(pts: np.ndarray, cols: np.ndarray, ppi: np.ndarray,
                      bit_depth: int = 10,
                      lod: int = 2,                # «lodScale» in TMC2 (2→downscale×4)
                      max_thickness: int = Cfg.MAX_THRICKNESS, 
                      min_patch_size: int = 16
                      ) -> List[Patch]:
    """Generate patches with **LOD down-sampling** to avoid single-point patches.
    """
    Q = 1 << bit_depth
    xyz_min, xyz_max = pts.min(0), pts.max(0)
    scale = np.maximum(xyz_max - xyz_min, 1e-6)
    pts_q = np.floor((pts - xyz_min) / scale * (Q - 1) + 0.5).astype(int)  # Rescale to [0, Q-1]

    patches: List[Patch] = []
    for plane in range(6):
        sel = (ppi == plane)  # shape (N,)
        if not np.any(sel):
            continue
        pts_i, col_i = pts_q[sel], cols[sel]  # points of the selected plane
        if plane < 2:   u_idx, v_idx, n_idx = 1, 2, 0
        elif plane < 4: u_idx, v_idx, n_idx = 0, 2, 1
        else:           u_idx, v_idx, n_idx = 0, 1, 2

        u = pts_i[:, u_idx] >> lod   # down‑scale, range [0, Q-1]
        v = pts_i[:, v_idx] >> lod
        d = pts_i[:, n_idx]

        # Generate occupancy mask for the plane
        u -= u.min(); v -= v.min()
        W, H = u.max() + 1, v.max() + 1
        occ = np.zeros((H, W), bool)
        for uu, vv in zip(u, v):
            occ[vv, uu] = True

        if max(W, H) > 4:  # skip tiny patches
            occ = cv2.morphologyEx(occ.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3),np.uint8)).astype(bool)

        for comp in _connected_components_mask(occ, conn=2):
            #Q: What would be the shape of comp? A: (H, W) mask of the patch
            if comp.sum() < min_patch_size:  # discard micro‑patches (encoder flag minPatchSize)
                continue
            ys, xs = np.nonzero(comp)
            mask_uv = comp[v, u]  # mask of the patch, shape (N,)

            sign_n = int(_AXIS_SIGN[plane])              #  +1 / -1  **cast to int!**
            comp_n = pts_i[mask_uv][:, n_idx]  # raw depth values of the patch
            ref_n  = comp_n.min() if sign_n > 0 else comp_n.max()
            depth  = np.abs(comp_n - ref_n).astype(int)  # always ≥0

            # Create patch object
            patch = Patch()
            patch.pts = pts[sel][mask_uv]
            patch.rgb = col_i[mask_uv]
            patch.ppi = plane
            patch.u = u[mask_uv] - xs.min()  
            patch.v = v[mask_uv] - ys.min()
            patch.shift3d           = pts_q[sel][mask_uv].min(0)
            patch.shift3d[n_idx]    = ref_n              # overwrite for ± axis
            patch.d                 = depth
 
            patch.size_uv = np.array([xs.max() - xs.min() + 1,
                                      ys.max() - ys.min() + 1], int)
            
            # --- near / far split  ---
            per_px: Dict[Tuple[int,int], List[Tuple[int,np.ndarray]]] = {}
            for uu, vv, dd, rgb in zip(patch.u, patch.v, patch.d, patch.rgb):
                per_px.setdefault((uu, vv), []).append((dd, rgb))
            u2=v2=d_near=d_far=rgb_near=rgb_far=[]
            u2,v2,d_near,d_far,rgb_near,rgb_far=[],[],[],[],[],[]
            for (uu, vv), items in per_px.items():
                items.sort(key=lambda t: t[0])
                dn, rn = items[0]
                df, rf = items[-1]
                u2.append(uu); v2.append(vv)
                d_near.append(dn); d_far.append(df)
                rgb_near.append(rn); rgb_far.append(rf)
            patch.u = np.array(u2,int); patch.v = np.array(v2,int)
            patch.d_near = np.array(d_near,int); patch.d_far = np.array(d_far,int)
            patch.rgb_near = np.vstack(rgb_near); patch.rgb_far = np.vstack(rgb_far)

            # # ------------------------------------------------------------------ depth check
            # # ------------------------------------------------------------------  helpers
            # def _spawn_child_patch(parent: Patch, keep_px: np.ndarray) -> Patch:
            #     """Return a shallow copy that keeps only the selected *pixel* rows."""
            #     ch             = Patch()
            #     ch.ppi         = parent.ppi
            #     ch.shift3d     = parent.shift3d.copy()      # same 3-D origin
            #     for attr in ('u', 'v',
            #                 'd_near', 'd_far',
            #                 'rgb_near', 'rgb_far'):
            #         setattr(ch, attr, getattr(parent, attr)[keep_px])
            #     ch.u      -= ch.u.min()
            #     ch.v      -= ch.v.min()
            #     ch.size_uv = np.array([ch.u.max() + 1, ch.v.max() + 1])
            #     return ch

            # depth_span = patch.d_far.max() - patch.d_near.min()
            # if depth_span > max_thickness:
            #     # centre depth of each projected pixel
            #     centre_d = ((patch.d_near + patch.d_far) // 2)
            #     n_slices = int(np.ceil(depth_span / max_thickness))
            #     bins = np.linspace(patch.d_near.min(),
            #                     patch.d_near.min() + n_slices * max_thickness,
            #                     n_slices + 1,
            #                     dtype=int)
            #     slice_id = np.digitize(centre_d, bins, right=False) - 1  # 0 … n_slices-1
            #     for sid in range(n_slices):
            #         keep = slice_id == sid
            #         if keep.sum() < min_patch_size:
            #             continue
            #         patches.append(_spawn_child_patch(patch, keep))
            #     continue          # **do NOT append the oversized parent**

            patches.append(patch)
    print(f"[PatchGen] Generated {len(patches)} patches")
    return patches




# ───────────────────────── Iterative Refinement  ──────────────────────────────
def refine_patches_once(pts_f: np.ndarray,
                        cols_f: np.ndarray,
                        ppi: np.ndarray,
                        patches: list[Patch],
                        bit_depth: int = 10,
                        lod: int = 2,
                        err_thresh=0.8):
    """Run a single refine-decode-resegment cycle (75 % of TMC2 benefit)."""
    # 1. ---------------------------------------------------------------- decode simulation
    occ, g0, g1, c0, c1 = maps_for_frame(patches, bit_depth=bit_depth)
    recon_pts = []                        # integer
    for p in patches:
        u0, v0 = p.atlas_xy
        slice_occ = occ[v0:v0+p.size_uv[1], u0:u0+p.size_uv[0]]
        slice_g0  = g0 [v0:v0+p.size_uv[1], u0:u0+p.size_uv[0]]
        sign_n    = int(_AXIS_SIGN[p.ppi])
        u_idx,v_idx,n_idx = _axis_indices(p.ppi)
        for uu,vv in zip(p.u,p.v):
            if slice_occ[vv,uu]:
                xyz=[0,0,0]
                xyz[u_idx]= (p.shift3d[u_idx]>>lod)+(uu<<lod)
                xyz[v_idx]= (p.shift3d[v_idx]>>lod)+(vv<<lod)
                xyz[n_idx]= p.shift3d[n_idx]+sign_n*int(slice_g0[vv,uu])
                recon_pts.append(xyz)
    recon_pts=np.asarray(recon_pts,float)

    # 2. ---------------------------------------------------------------- error per original point
    Q = 1 << bit_depth
    err = np.linalg.norm((pts_f * (Q-1)).round() - recon_pts, axis=1)
    mask_bad = err > err_thresh

    if not mask_bad.any():
        return patches                # nothing to refine this time

    # 3. ---------------------------------------------------------------- regenerate only bad pts
    bad_patches = generate_patches(pts_f[mask_bad],
                                   cols_f[mask_bad],
                                   ppi[mask_bad],
                                   bit_depth=bit_depth,
                                   lod=lod,
                                   max_thickness=Cfg.MAX_DELTA_DEPTH)
    patches.extend(bad_patches)
    pack_patches(patches)             # re-pack whole set once
    return patches
