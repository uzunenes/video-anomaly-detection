"""UCSD Ped1/Ped2 -> cached uint8 arrays + frame-level ground truth.

Layout produced:
    <out>/<ped>/train/TrainXXX.npy   (N, H, W) uint8
    <out>/<ped>/test/TestXXX.npy
    <out>/<ped>/gt.json               {"TestXXX": [0/1 per frame]}
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def load_video(folder: Path) -> np.ndarray:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".tif" and not p.name.startswith("."))
    frames = []
    for f in files:
        try:
            frames.append(np.array(Image.open(f).convert("L")))
        except OSError:  # UCSDped1/Test/Test017/142.tif is truncated in v1p2
            print(f"warning: unreadable {f}, repeating previous frame to keep GT alignment")
            frames.append(frames[-1].copy())
    return np.stack(frames)


def parse_gt(mfile: Path) -> list[list[tuple[int, int]]]:
    """Read `TestVideoFile{..}.gt_frame = [a:b, c:d];` ranges (1-based, inclusive)."""
    entries = re.findall(r"gt_frame\s*=\s*\[([^\]]*)\]", mfile.read_text())
    return [[(int(a), int(b)) for a, b in re.findall(r"(\d+)\s*:\s*(\d+)", e)] for e in entries]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="UCSD_Anomaly_Dataset.v1p2 folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--peds", nargs="+", default=["UCSDped2", "UCSDped1"])
    args = ap.parse_args()

    for ped in args.peds:
        src, dst = Path(args.root) / ped, Path(args.out) / ped.lower().replace("ucsd", "")
        for split in ("Train", "Test"):
            (dst / split.lower()).mkdir(parents=True, exist_ok=True)
            vids = sorted(d for d in (src / split).iterdir()
                          if d.is_dir() and not d.name.endswith("_gt") and not d.name.startswith("."))
            for v in vids:
                np.save(dst / split.lower() / f"{v.name}.npy", load_video(v))
            print(f"{ped} {split}: {len(vids)} videos")

        ranges = parse_gt(src / "Test" / f"{ped}.m")
        tests = sorted(p.stem for p in (dst / "test").glob("*.npy"))
        assert len(ranges) == len(tests), (len(ranges), len(tests))
        gt = {}
        for name, rr in zip(tests, ranges):
            n = np.load(dst / "test" / f"{name}.npy", mmap_mode="r").shape[0]
            lab = np.zeros(n, dtype=np.uint8)
            for a, b in rr:
                lab[a - 1:b] = 1
            gt[name] = lab.tolist()
        (dst / "gt.json").write_text(json.dumps(gt))
        n_all = sum(len(v) for v in gt.values())
        n_pos = sum(sum(v) for v in gt.values())
        print(f"{ped} GT: {len(gt)} test videos, {n_all} frames, {n_pos} anomalous ({100 * n_pos / n_all:.1f}%)")


if __name__ == "__main__":
    main()
