# -*- coding: utf-8 -*-
"""
Python‑only **V‑PCC (ISO/IEC 23090‑5) encoder**
================================================
A faithful, though not real‑time, re‑implementation of the core tools in
TMC2 v14.0 suitable for research and educational use.

Highlights
----------
* **Full patch pipeline**: normal estimation → outward orientation → initial PPI
  selection → refined PPI smoothing → connected‑component patch segmentation
  with thickness pruning.
* **Two‑layer geometry & colour maps** to resolve occlusions exactly as the
  reference encoder (G0 = nearest, G1 = farthest).
* **Shelf skyline atlas packing** sorted by patch area (AI config).
* **Occupancy down‑scaling** (×2 or ×4) + background filling of geometry and
  colour maps.
* **Bit‑stream writer** producing a per‑frame *patch data unit* that matches
  the fields of `PatchDataUnit()` in the spec so the decoder can reconstruct
  3‑D points losslessly (given lossless 2‑D videos).

Dependencies: numpy, scipy, scikit‑learn, torch, cv2, imageio, tqdm
"""
from __future__ import annotations
import os, struct, math, cv2, imageio, shutil, tempfile
from pathlib import Path
from typing import List, Tuple, BinaryIO, NamedTuple, Dict, Iterable, Optional
import numpy as np
import torch
import scipy.sparse
from sklearn.neighbors import KDTree
from collections import deque

################################################################################
# ╭────────────────────────────  CONFIG  ────────────────────────────╮
################################################################################

class Cfg:
    K_NEIGH  = 16                # k‑NN for PCA normal estimation
    ORIENT_K = 8                 # neighbours used in orientation propagation
    PPI_SMOOTH_ITERS = 3         # aggressive smoothing passes

    MAX_THRICKNESS = 4           # patch thickness in voxels (near+far)

    ATLAS_WH = (1024, 1024)      # (W,H) in pixels
    BORDER   = 2                 # border between packed patches
    OCC_DOWNSCALE = 1            # 1 (none) / 2 / 4

    FPS      = 30

################################################################################
# ╭──────────────────────  DATA STRUCTURES  ───────────────────────╮
################################################################################

_AXES = np.array([[ 1, 0, 0], [-1, 0, 0],
                  [ 0, 1, 0], [ 0,-1, 0],
                  [ 0, 0, 1], [ 0, 0,-1]], np.float32)
_AXIS_SIGN = np.array([+1, -1, +1, -1, +1, -1], np.int8)   # PPI → +1 / –1

class Patch:
    """Container for one **V‑PCC patch** (single PPI)."""
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
    """Choose the projection‑plane index exactly like TMC2.

    *For each point* pick the axis‑aligned plane whose normal has the **most
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

from scipy.ndimage import label as cc_label

def _connected_components_mask(mask: np.ndarray, conn: int = 2) -> List[np.ndarray]:
    """8‑connected components if *conn* = 2, 4‑conn if 1."""
    lab, n = cc_label(mask, structure=np.ones((3,3)) if conn == 2 else None)
    # return [(lab == i) for i in range(1, n + 1)](lab == i) for i in range(1, n + 1)]
    return [lab == i for i in range(1, n + 1)]

def generate_patches(pts: np.ndarray, cols: np.ndarray, ppi: np.ndarray,
                      bit_depth: int = 10,
                      lod: int = 2,                # «lodScale» in TMC2 (2→downscale×4)
                      max_thickness: int = Cfg.MAX_THRICKNESS, 
                      min_patch_size: int = 16, 
                      patch_split=False
                      ) -> List[Patch]:
    """Generate patches with **LOD down‑sampling** to avoid single‑point patches.

    The Level‑Of‑Detail parameter `lod` (0,1,2) divides the (u,v) grid by
    2^lod *before* connected‑component analysis – exactly what TMC2 does via
    `lodScale{X,Y}`.  With `lod=2`, a 1024³ cube becomes 256² projection grids
    → ~10× fewer patches while retaining shape fidelity.
    """
    Q = 1 << bit_depth
    xyz_min, xyz_max = pts.min(0), pts.max(0)
    scale = np.maximum(xyz_max - xyz_min, 1e-6)
    pts_q = np.floor((pts - xyz_min) / scale * (Q - 1) + 0.5).astype(int)  # Rescale to [0, Q-1]

    patches: List[Patch] = []
    for plane in range(6):
        sel = (ppi == plane)
        if not np.any(sel):
            continue
        pts_i, col_i = pts_q[sel], cols[sel]
        if plane < 2:   u_idx, v_idx, n_idx = 1, 2, 0
        elif plane < 4: u_idx, v_idx, n_idx = 0, 2, 1
        else:           u_idx, v_idx, n_idx = 0, 1, 2

        u = pts_i[:, u_idx] >> lod   # down‑scale
        v = pts_i[:, v_idx] >> lod
        d = pts_i[:, n_idx]
        # Q: what is this >> operation doing?
        # A: it is downscaling the u/v coordinates by a factor of 2^lod

        u -= u.min(); v -= v.min()
        W, H = u.max() + 1, v.max() + 1
        occ = np.zeros((H, W), bool)
        for uu, vv in zip(u, v):
            occ[vv, uu] = True
        # Optional morphological closing to merge 1‑pixel gaps (spec allows)
        if max(W, H) > 4:  # skip tiny patches
            occ = cv2.morphologyEx(occ.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3),np.uint8)).astype(bool)

        for comp in _connected_components_mask(occ, conn=2):
            if comp.sum() < min_patch_size:  # discard micro‑patches (encoder flag minPatchSize)
                continue
            ys, xs = np.nonzero(comp)
            mask_uv = comp[v, u]

            sign_n = int(_AXIS_SIGN[plane])              #  +1 / -1  **cast to int!**
            comp_n = pts_i[mask_uv][:, n_idx]
            ref_n  = comp_n.min() if sign_n > 0 else comp_n.max()
            depth  = np.abs(comp_n - ref_n).astype(int)  # always ≥0

            # same as before ...
            patch = Patch()
            patch.pts = pts[sel][mask_uv]
            patch.rgb = col_i[mask_uv]
            patch.ppi = plane
            patch.u = u[mask_uv] - xs.min()
            patch.v = v[mask_uv] - ys.min()
            # patch.shift3d = pts_q[sel][mask_uv].min(0)
            # patch.d = (d[mask_uv] - patch.shift3d[n_idx]).astype(int)
            patch.shift3d           = pts_q[sel][mask_uv].min(0)
            patch.shift3d[n_idx]    = ref_n              # overwrite for ± axis
            patch.d                 = depth
 
            patch.size_uv = np.array([xs.max() - xs.min() + 1,
                                      ys.max() - ys.min() + 1], int)
            # --- near / far split (unchanged) ---
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

            if patch_split:
                MAX_W = 64
                MAX_H = 256
                queue = [patch]
                while queue:
                    cur = queue.pop()
                    if cur.size_uv[0] <= MAX_W and cur.size_uv[1] <= MAX_H:
                        patches.append(cur)
                        continue

                    # split along the longer edge like TMC2 (generateRectangle())
                    if cur.size_uv[0] >= cur.size_uv[1]:
                        cut = cur.size_uv[0] // 2
                        mask_left  = cur.u <  cut
                        mask_right = ~mask_left
                        tiles = [mask_left, mask_right]
                    else:
                        cut = cur.size_uv[1] // 2
                        mask_top = cur.v <  cut
                        mask_bot = ~mask_top
                        tiles = [mask_top, mask_bot]

                    for keep_px in tiles:
                        if keep_px.sum() < min_patch_size:
                            continue
                        child = Patch()
                        for attr in ('u','v','d_near','d_far','rgb_near','rgb_far'):
                            setattr(child, attr, getattr(cur, attr)[keep_px])
                        child.ppi      = cur.ppi
                        child.shift3d  = cur.shift3d
                        child.u       -= child.u.min()
                        child.v       -= child.v.min()
                        child.size_uv  = np.array([child.u.max()+1, child.v.max()+1])
                        queue.append(child)
                    continue            # do not append the oversize parent
            patches.append(patch)
    return patches


################################################################################
# ╭──────────────────  ATLAS PACKING  ────────────────────────────╮
################################################################################

class ShelfPacker:
    def __init__(self, W:int, H:int, border:int=Cfg.BORDER):
        self.W,self.H,self.border=W,H,border
        self.x=self.y=self.shelf_h=0

    def try_place(self,w:int,h:int)->Optional[Tuple[int,int]]:
        """
        Try to place a rectangle of size (w,h) in the atlas.
        """
        if w>self.W or h>self.H: return None
        if self.x+w>self.W:
            self.x=0
            self.y+=self.shelf_h+self.border
            self.shelf_h=0
        if self.y+h>self.H: return None
        pos=(self.x,self.y)
        self.x+=w+self.border
        self.shelf_h=max(self.shelf_h,h)
        return pos

def pack_patches(patches: List[Patch],
                 atlas_wh: Tuple[int, int] = Cfg.ATLAS_WH):
    """
    Bottom-Left Skyline packer (accurate).  Nodes are (x, width, height).
    """
    W, H = atlas_wh
    patches.sort(key=lambda p: p.size_uv[0] * p.size_uv[1], reverse=True)

    skyline = [(0, W, 0)]         # start with a single flat skyline

    def find_pos(pw: int, ph: int):
        best = None
        best_y = 1e9
        for i, (sx, sw, sh) in enumerate(skyline):
            if sw < pw:
                continue
            x, y = sx, sh
            j = i
            width_left = pw
            max_h = y
            while width_left > 0:
                if j >= len(skyline):
                    break
                nx, nw, nh = skyline[j]
                max_h = max(max_h, nh)
                if max_h + ph > H:
                    break
                width_left -= nw
                j += 1
            else:
                if max_h < best_y or (max_h == best_y and sx < best[0]):
                    best = (sx, max_h, i)
                    best_y = max_h
        return best  # (x, y, node_index) or None

    for p in patches:
        pw, ph = map(int, p.size_uv)
        pos = find_pos(pw, ph)
        if pos is None:
            raise RuntimeError("Atlas overflow – enlarge canvas or split GOP")
        x, y, idx = pos
        p.atlas_xy = np.array([x, y], int)

        # ---------- skyline update ----------
        # 1. insert new node for the top of this patch
        skyline.insert(idx, (x, pw, y + ph))
        # 2. adjust following node’s x / width
        sx, sw, sh = skyline[idx + 1]
        skyline[idx + 1] = (x + pw, sw - pw, sh)
        # 3. merge consecutive nodes with same height
        i = 0
        while i < len(skyline) - 1:
            if skyline[i][2] == skyline[i + 1][2]:
                x0, w0, h0 = skyline[i]
                _,  w1, _  = skyline[i + 1]
                skyline[i] = (x0, w0 + w1, h0)
                skyline.pop(i + 1)
            else:
                i += 1

################################################################################
# ╭──────────────────  MAP GENERATION  ───────────────────────────╮
################################################################################

def maps_for_frame(patches: List[Patch],
                   atlas_wh: Tuple[int, int] = Cfg.ATLAS_WH,
                   bit_depth: int = 10):
    W, H = atlas_wh
    occ = np.zeros((H, W), np.uint8)
    g0  = np.zeros((H, W), np.uint16)     # will hold 10-bit values 0-1023
    g1  = np.zeros_like(g0)
    c0  = np.zeros((H, W, 3), np.uint8)
    c1  = np.zeros_like(c0)

    for p in patches:
        u0, v0 = p.atlas_xy
        for uu, vv, dn, df, rn, rf in zip(
                p.u, p.v, p.d_near, p.d_far, p.rgb_near, p.rgb_far):
            y, x = v0 + vv, u0 + uu
            occ[y, x] = 1
            g0[y, x] = dn
            g1[y, x] = df
            c0[y, x] = (rn * 255).astype(np.uint8)
            c1[y, x] = (rf * 255).astype(np.uint8)

    # --- scale geometry to 10-bit so libx265 can take it raw ---------------
    if bit_depth < 10:
        shift = 10 - bit_depth
        g0 <<= shift
        g1 <<= shift
    elif bit_depth > 10:
        g0 = np.clip(g0, 0, 1023)
        g1 = np.clip(g1, 0, 1023)

    def _downscale_occ(occ: np.ndarray, s: int) -> np.ndarray:
        """Logical-OR down-scaling by factor *s* (s = 2 or 4)."""
        H, W = occ.shape
        assert H % s == 0 and W % s == 0
        occ = occ.reshape(H//s, s, W//s, s)
        return occ.max(axis=(1, 3)).astype(np.uint8)
    # --- optional occupancy down-scaling -----------------------------------
    if Cfg.OCC_DOWNSCALE > 1:
        occ_ds = _downscale_occ(occ, Cfg.OCC_DOWNSCALE)
    else:
        occ_ds = occ
    
    return occ_ds, g0, g1, c0, c1

################################################################################
# ╭──────────────────  BITSTREAM WRITER  ─────────────────────────╮
################################################################################

_HDR_FMT="<b3i4h"; HDR_SZ=struct.calcsize(_HDR_FMT)

def write_patch_data(patches: List[Patch], fp: BinaryIO):
    for p in patches:
        h=PatchHeader(int(p.ppi), *map(int,p.shift3d), *map(int,p.atlas_xy),
                      int(p.size_uv[0]), int(p.size_uv[1]))
        fp.write(struct.pack(_HDR_FMT, *h))
        # depth payload (near+far) flattened, uint16 little endian
        canvas_near=np.full(p.size_uv[::-1], 65535, np.uint16)
        canvas_far =np.full_like(canvas_near, 65535)
        canvas_near[p.v,p.u]=p.d_near
        canvas_far [p.v,p.u]=p.d_far
        fp.write(canvas_near.tobytes())
        fp.write(canvas_far.tobytes())


GLOBAL_HDR = struct.Struct("<3f3f")          # xyz_min (3×f32)  |  scale (3×f32)
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


    # def pack_patches(patches: List[Patch], atlas_wh: Tuple[int,int]=Cfg.ATLAS_WH):
#     # Largest‑area‑first packing, identical to default AI config of TMC2
#     patches.sort(key=lambda p: p.size_uv[0]*p.size_uv[1], reverse=True)
#     packer=ShelfPacker(*atlas_wh)
#     for p in patches:
#         loc=packer.try_place(int(p.size_uv[0]),int(p.size_uv[1]))
#         if loc is None:
#             raise RuntimeError("Atlas overflow; increase canvas or split GOP")
#         p.atlas_xy=np.array(loc,int)


# ────────────────────────────   Decoder  ─────────────────────────────

_HDR_FMT = "<b3i4h"
_HDR_SZ  = struct.calcsize(_HDR_FMT)

def _axis_indices(ppi: int) -> Tuple[int, int, int]:
    """Return (u_idx, v_idx, n_idx) for a given PPI (same rule as encoder)."""
    if ppi < 2:   return 1, 2, 0      # ±X  → (Y,Z) , depth=X
    if ppi < 4:   return 0, 2, 1      # ±Y  → (X,Z) , depth=Y
    return 0, 1, 2                    # ±Z  → (X,Y) , depth=Z

# ────────────────────────────   loader   ──────────────────────────────

def _load_images(frame_dir: Path, idx: int):
    occ  = imageio.v2.imread(frame_dir / f"occ_{idx:04d}.png")          # uint8
    g0   = cv2.imread(str(frame_dir / f"geom0_{idx:04d}.png"),
                      cv2.IMREAD_UNCHANGED)                            # uint16
    g1   = cv2.imread(str(frame_dir / f"geom1_{idx:04d}.png"),
                      cv2.IMREAD_UNCHANGED)
    c0   = imageio.v2.imread(frame_dir / f"attr0_{idx:04d}.png")        # uint8
    c1   = imageio.v2.imread(frame_dir / f"attr1_{idx:04d}.png")
    return occ, g0, g1, c0, c1

# ───────────────────────── decoder core  ──────────────────────────────

GLOBAL_HDR = struct.Struct("<3f3f")          # xyz_min (3×f32)  |  scale (3×f32)
def decode_frame(frame_dir: Path,
                 frame_idx: int,
                 bit_depth: int = 10,
                 lod: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Return pts [N,3] in the encoder’s metric space and RGB [0-1]."""
    occ, g0, g1, c0, c1 = _load_images(frame_dir, frame_idx)

    # # no occupancy down-scale because Cfg.OCC_DOWNSCALE == 1
    # H, W = occ.shape

    # # undo left-shift (10-bit storage)
    # shift_back = max(0, 10 - bit_depth)
    # if shift_back:
    #     g0 >>= shift_back
    #     g1 >>= shift_back


    with open(frame_dir / "patch_data.bin", "rb") as fp:
        buf      = fp.read(GLOBAL_HDR.size)           # 24 bytes
        vals     = GLOBAL_HDR.unpack(buf)             # 6 floats
        xyz_min  = np.asarray(vals[:3],  np.float32)
        scale    = np.asarray(vals[3:], np.float32)

        # -------- per-patch loop ------------------------------------------
        pts_i, cols_f = [], []
        while True:
            hdr = fp.read(_HDR_SZ)
            if not hdr:
                break                                  # EOF

            (ppi, sx, sy, sz, au, av, sw, sh) = struct.unpack(_HDR_FMT, hdr)
            n_pix = int(sw) * int(sh)
            near  = np.frombuffer(fp.read(n_pix * 2), np.uint16).reshape(sh, sw)
            far   = np.frombuffer(fp.read(n_pix * 2), np.uint16).reshape(sh, sw)

            u_idx, v_idx, n_idx = _axis_indices(ppi)
            base_u = (sx >> lod) if u_idx == 0 else (sy >> lod) if u_idx == 1 else (sz >> lod)
            base_v = (sx >> lod) if v_idx == 0 else (sy >> lod) if v_idx == 1 else (sz >> lod)
            base_n =  sx        if n_idx == 0 else  sy        if n_idx == 1 else  sz
            sign_n = int(_AXIS_SIGN[ppi])

            # atlas slices --------------------------------------------------
            occ_p = occ[av:av + sh, au:au + sw]
            c0_p  = c0 [av:av + sh, au:au + sw]
            c1_p  = c1 [av:av + sh, au:au + sw]

            for yy, xx in zip(*np.nonzero(occ_p)):
                u_q = (base_u + xx) << lod
                v_q = (base_v + yy) << lod

                d0 = near[yy, xx]
                if d0 != 65535:
                    xyz = [0, 0, 0]
                    xyz[u_idx] = u_q
                    xyz[v_idx] = v_q
                    xyz[n_idx] = base_n + sign_n * int(d0)
                    pts_i .append(xyz)
                    cols_f.append(c0_p[yy, xx] / 255.0)

                d1 = far[yy, xx]
                if d1 != 65535 and d1 != d0:
                    xyz = [0, 0, 0]
                    xyz[u_idx] = u_q
                    xyz[v_idx] = v_q
                    xyz[n_idx] = base_n + sign_n * int(d1)
                    pts_i .append(xyz)
                    cols_f.append(c1_p[yy, xx] / 255.0)

    pts_i  = np.asarray(pts_i,  np.float32)
    cols_f = np.asarray(cols_f, np.float32)

    # de-quantise back to the original metric space -------------------------
    Q = 1 << bit_depth
    pts_f = pts_i / (Q - 1) * scale + xyz_min       # float32

    return pts_f, cols_f
