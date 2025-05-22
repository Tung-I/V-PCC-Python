from __future__ import annotations
import os, struct, math, cv2, imageio, shutil, tempfile
from pathlib import Path
from typing import List, Tuple, BinaryIO, NamedTuple, Dict, Iterable, Optional
import numpy as np
import torch
import scipy.sparse
from sklearn.neighbors import KDTree
from collections import deque

from src.cfg import Cfg, _AXES, _AXIS_SIGN, _HDR_FMT, _HDR_SZ, _axis_indices
# ────────────────────────────   Decoder  ─────────────────────────────

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
    """Return pts [N,3] in the encoder metric space and RGB [0-1]."""
    occ, g0, g1, c0, c1 = _load_images(frame_dir, frame_idx)

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

            u_idx, v_idx, n_idx = _axis_indices(ppi)
            base_u = (sx >> lod) if u_idx == 0 else (sy >> lod) if u_idx == 1 else (sz >> lod)
            base_v = (sx >> lod) if v_idx == 0 else (sy >> lod) if v_idx == 1 else (sz >> lod)
            base_n =  sx        if n_idx == 0 else  sy        if n_idx == 1 else  sz
            sign_n = int(_AXIS_SIGN[ppi])

            # atlas slices --------------------------------------------------
            occ_p = occ[av:av + sh, au:au + sw]
            c0_p  = c0 [av:av + sh, au:au + sw]
            c1_p  = c1 [av:av + sh, au:au + sw]
            g0_p  = g0 [av:av + sh, au:au + sw]
            g1_p  = g1 [av:av + sh, au:au + sw]

            for yy, xx in zip(*np.nonzero(occ_p)):
                u_q = (base_u + xx) << lod
                v_q = (base_v + yy) << lod

                d0 = int(g0_p[yy, xx])          
                d1 = int(g1_p[yy, xx])       

                # d0 = near[yy, xx]
                if d0 != 65535:
                    xyz = [0, 0, 0]
                    xyz[u_idx] = u_q
                    xyz[v_idx] = v_q
                    xyz[n_idx] = base_n + sign_n * int(d0)
                    pts_i .append(xyz)
                    cols_f.append(c0_p[yy, xx] / 255.0)

                # d1 = far[yy, xx]
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
