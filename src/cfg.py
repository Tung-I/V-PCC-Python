import numpy as np
from typing import List, Tuple
import os, struct

################################################################################
# ╭────────────────────────────  CONFIG  ────────────────────────────╮
################################################################################

class Cfg:
    K_NEIGH  = 16                # k‑NN for PCA normal estimation
    ORIENT_K = 8                 # neighbours used in orientation propagation
    PPI_SMOOTH_ITERS = 4         # aggressive smoothing passes
    MAX_THRICKNESS = 8           # patch thickness in voxels (near+far)
    ATLAS_WH = (2048, 2048)      # (W,H) in pixels
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


GLOBAL_HDR = struct.Struct("<3f3f")          # xyz_min (3×f32)  |  scale (3×f32)
_HDR_FMT = "<b3i4h"
_HDR_SZ  = struct.calcsize(_HDR_FMT)

def _axis_indices(ppi: int) -> Tuple[int, int, int]:
    """Return (u_idx, v_idx, n_idx) for a given PPI (same rule as encoder)."""
    if ppi < 2:   return 1, 2, 0      # ±X  → (Y,Z) , depth=X
    if ppi < 4:   return 0, 2, 1      # ±Y  → (X,Z) , depth=Y
    return 0, 1, 2                    # ±Z  → (X,Y) , depth=Z