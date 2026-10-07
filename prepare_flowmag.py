"""Optical-flow magnitude "videos" so that the same AEAN + RX pipeline can run on the motion modality.

    python prepare_flowmag.py --src cache/ped2 --out cache/ped2_flowmag --gap 1

Every frame t becomes |flow(t-g -> t)| (DIS optical flow, causal), quantised to uint8 as
min(255, mag * scale). The scale is set on the NORMAL training videos only: their 99.9th percentile of the
magnitude maps to 96, leaving head-room for faster-than-normal motion. gt.json is copied unchanged.
"""
import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def flow_mag(frames: np.ndarray, gap: int) -> np.ndarray:
    import cv2
    cv2.setNumThreads(1)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    out = np.zeros(frames.shape, np.float32)
    for t in range(gap, len(frames)):
        f = dis.calc(frames[t - gap], frames[t], None)
        out[t] = np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2)
    out[:gap] = out[gap] if len(frames) > gap else 0
    return out


def _job(args):
    src, gap = args
    return flow_mag(np.load(src), gap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    mags = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for split in ("train", "test"):
            files = sorted((src / split).glob("*.npy"))
            for p, m in zip(files, ex.map(_job, [(p, args.gap) for p in files])):
                mags[(split, p.stem)] = m
    train = np.concatenate([m[::5].ravel() for (s, _), m in mags.items() if s == "train"])
    scale = 96.0 / max(float(np.quantile(train, 0.999)), 1e-3)
    for (split, name), m in mags.items():
        (out / split).mkdir(parents=True, exist_ok=True)
        np.save(out / split / f"{name}.npy", np.minimum(255, np.round(m * scale)).astype(np.uint8))
    shutil.copy(src / "gt.json", out / "gt.json")
    (out / "flowmag.json").write_text(json.dumps({"gap": args.gap, "scale": scale}))
    print(f"{out}: scale {scale:.2f} (train 99.9% -> 96)")


if __name__ == "__main__":
    main()
