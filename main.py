import argparse, json
import os
from pathlib import Path
import imageio
import torch
import numpy as np
from tqdm import tqdm
from src.patch_utils import encode_frame, decode_frame
from src.ptcl_utils import load_ply_sequence, downsample
from src.rendering_utils import render_sequence_spiral


if __name__=="__main__":
    """
    Usage: python main.py --seq $WORK/datasets/8ivfb/longdress/Ply --out results/longdress --n 1 --num_samples 100000
    """
    p=argparse.ArgumentParser("Python V‑PCC encoder (framewise)")
    p.add_argument("--seq", required=True, help="path to folder of frame‑wise PLY")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--n", type=int, default=-1, help="#frames (‑1=all)")
    p.add_argument("--num_samples", type=int, default=-1, help="number of points to sample")
    p.add_argument("--decode_only", action="store_true", help="decode only")
    args=p.parse_args()

    out_dir=Path(args.out)
    if not args.decode_only:
        pts_seq=load_ply_sequence(args.seq, cache_name="", start=0, nframes=args.n)
        for fi,(pts,cols) in enumerate(tqdm(pts_seq)):
            if args.num_samples > 0:
                pts, cols = downsample(pts, cols, args.num_samples)
                print(f"Scale: pts max {pts.max()}, min {pts.min()}, cols max {cols.max()}, min {cols.min()}")
                sequence = [(pts, cols)]
                render_sequence_spiral(
                    sequence,
                    out_dir=out_dir,
                    device="cuda",
                    n_frames=len(sequence),
                    image_size=512,
                    filename="input",
                )
            encode_frame(pts, cols, out_dir, fi)

        print("Encoding finished. Use ffmpeg to compress PNGs into HEVC/AV1 videos.")


    # --- dummy reconstruction-------------------------------
    in_dir = Path("results/myseq/d8/frame0000")   # folder that contains the pngs
    pts, cols = decode_frame(out_dir, frame_idx=0)
    print("decoded", pts.shape[0], "points")
    # Convert pts, cols to tensor
    pts = torch.tensor(pts, dtype=torch.float32)
    cols = torch.tensor(cols, dtype=torch.float32)
    sequence = [(pts, cols)]
    render_sequence_spiral(
        sequence,
        out_dir=out_dir,
        device="cuda",
        n_frames=len(sequence),
        image_size=512,
        filename="dummy_recon"
    )