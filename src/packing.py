from __future__ import annotations
import os, struct, math, cv2, imageio, shutil, tempfile
from pathlib import Path
from typing import List, Tuple, BinaryIO, NamedTuple, Dict, Iterable, Optional
import numpy as np
import torch
import scipy.sparse
from sklearn.neighbors import KDTree
from collections import deque

from src.cfg import Cfg, _AXES, _AXIS_SIGN
# from src.patch import Patch

################################################################################
# ╭──────────────────  ATLAS PACKING  ────────────────────────────╮
################################################################################

# ───────────────────────  Max-Rects helpers  ────────────────────────────
def _split_free_node(free: list[tuple[int,int,int,int]],
                     node: tuple[int,int,int,int],
                     w: int,
                     h: int,
                     border: int = 0):
    """Carve (w,h) out of *node* and push the two remainders to *free*."""
    x, y, fw, fh = node
    # right slice
    rem_w = fw - w - border
    if rem_w > 0:
        free.append((x + w + border, y, rem_w, h))
    # bottom slice
    rem_h = fh - h - border
    if rem_h > 0:
        free.append((x, y + h + border, fw, rem_h))


def _merge_free_rects(free: list[tuple[int,int,int,int]], *,
                      max_iter: int = 3) -> list[tuple[int,int,int,int]]:
    """Very light-weight rectangle union (enough for a small free list)."""
    for _ in range(max_iter):
        merged = False
        j = 0
        while j < len(free):
            x0, y0, w0, h0 = free[j]
            k = j + 1
            while k < len(free):
                x1, y1, w1, h1 = free[k]
                # horizontal merge
                if y0 == y1 and h0 == h1:
                    if x0 + w0 == x1:               # left+right
                        free[j] = (x0, y0, w0 + w1, h0);  free.pop(k); merged = True; continue
                    if x1 + w1 == x0:               # right+left
                        free[j] = (x1, y1, w1 + w0, h1);  free.pop(k); merged = True; continue
                # vertical merge
                if x0 == x1 and w0 == w1:
                    if y0 + h0 == y1:               # top+bottom
                        free[j] = (x0, y0, w0, h0 + h1);  free.pop(k); merged = True; continue
                    if y1 + h1 == y0:               # bottom+top
                        free[j] = (x1, y1, w1, h1 + h0);  free.pop(k); merged = True; continue
                k += 1
            j += 1
        if not merged:
            break
    return free

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

# # ─────────────────────────  Max-Rects packer  ────────────────────────────
# def pack_patches(patches: list[Patch],
#                  atlas_wh: tuple[int, int] = Cfg.ATLAS_WH,
#                  *,
#                  border: int = Cfg.BORDER,
#                  allow_rotate: bool = True):
#     """
#     Max-Rects / Best-Short-Side-Fit (BSSF) identical to TMC2 ‘AI’ profile.

#     • searches every free rectangle,  
#     • chooses the one with the smallest short-side residual, then area,  
#     • allows optional 90° rotation,  
#     • stores the chosen (x,y) in `patch.atlas_xy` and updates u/v when rotated.
#     """
#     W, H = atlas_wh
#     # sort largest-area first – dramatically speeds up Max-Rects
#     patches.sort(key=lambda p: p.size_uv[0] * p.size_uv[1], reverse=True)

#     free: list[tuple[int,int,int,int]] = [(0, 0, W, H)]   # (x, y, w, h)

#     for p in patches:
#         pw, ph = map(int, p.size_uv)
#         best_node  = None      # (idx_in_free, x, y, rot?)
#         best_short = 1e9
#         best_area  = 1e9

#         for idx, (fx, fy, fw, fh) in enumerate(free):
#             # try 0° ────────────────────────────────────────────────────────
#             if pw + border <= fw and ph + border <= fh:
#                 short = min(fw - pw - border, fh - ph - border)
#                 area  = (fw * fh) - (pw + border) * (ph + border)
#                 if short < best_short or (short == best_short and area < best_area):
#                     best_node = (idx, fx, fy, False)
#                     best_short, best_area = short, area
#             # try 90° ───────────────────────────────────────────────────────
#             if allow_rotate and ph + border <= fw and pw + border <= fh:
#                 short = min(fw - ph - border, fh - pw - border)
#                 area  = (fw * fh) - (ph + border) * (pw + border)
#                 if short < best_short or (short == best_short and area < best_area):
#                     best_node = (idx, fx, fy, True)
#                     best_short, best_area = short, area

#         if best_node is None:
#             raise RuntimeError("Atlas overflow — enlarge canvas or split GOP")

#         idx, x, y, rot = best_node
#         if rot:
#             # swap the axes
#             p.u, p.v = p.v.copy(), p.u.copy()
#             p.size_uv = p.size_uv[::-1]
#             pw, ph = ph, pw

#         p.atlas_xy = np.array([x, y], np.int32)

#         # -------- update free list ---------------------------------------
#         chosen = free.pop(idx)
#         _split_free_node(free, chosen, pw, ph, border=border)
#         free = _merge_free_rects(free)

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