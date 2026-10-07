"""CUHK Avenue -> cached uint8 arrays + frame-level ground truth (same layout as prepare_ucsd.py).

    python prepare_avenue.py --zip data/avenue/Avenue_Dataset.zip \
        --gt-zip data/avenue/ground_truth_demo.zip --out cache/avenue

Videos are 640x360 at 25 fps; frames are converted to grayscale and resized to --size (default 180x320,
half resolution, aspect kept); --color keeps RGB (N, H, W, 3), e.g. for vision-language models. A test frame is anomalous when its pixel mask in
ground_truth_demo/testing_label_mask/<i>_label.mat (volLabel) is non-empty.
"""
import argparse
import io
import json
import re
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat


def read_video(zf: zipfile.ZipFile, name: str, size: tuple[int, int], color: bool = False) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".avi") as tmp:     # OpenCV reads from a file path
        tmp.write(zf.read(name))
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(f, cv2.COLOR_BGR2RGB if color else cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(g, (size[1], size[0]), interpolation=cv2.INTER_AREA))
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--gt-zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="180x320", help="HxW")
    ap.add_argument("--color", action="store_true", help="keep RGB frames")
    args = ap.parse_args()
    size = tuple(int(v) for v in args.size.lower().split("x"))
    out = Path(args.out)

    zf = zipfile.ZipFile(args.zip)
    for split, folder in (("train", "training_videos"), ("test", "testing_videos")):
        (out / split).mkdir(parents=True, exist_ok=True)
        names = sorted(n for n in zf.namelist() if f"/{folder}/" in n and n.lower().endswith(".avi")
                       and not Path(n).name.startswith("."))
        for n in names:
            np.save(out / split / f"{int(Path(n).stem):02d}.npy", read_video(zf, n, size, args.color))
        print(f"{split}: {len(names)} videos")

    gz = zipfile.ZipFile(args.gt_zip)
    masks = {int(re.match(r"(\d+)_label", Path(n).name).group(1)): n for n in gz.namelist()
             if "testing_label_mask" in n and n.endswith("_label.mat")}
    gt = {}
    for p in sorted((out / "test").glob("*.npy")):
        vol = loadmat(io.BytesIO(gz.read(masks[int(p.stem)])))["volLabel"].ravel()
        lab = np.array([int(np.asarray(m).any()) for m in vol], dtype=np.uint8)
        n = np.load(p, mmap_mode="r").shape[0]
        if len(lab) != n:
            print(f"warning: {p.stem} has {n} frames but {len(lab)} labels; truncating to the shorter")
            k = min(n, len(lab))
            np.save(p, np.load(p)[:k])
            lab = lab[:k]
        gt[p.stem] = lab.tolist()
    (out / "gt.json").write_text(json.dumps(gt))
    n_all = sum(len(v) for v in gt.values())
    n_pos = sum(sum(v) for v in gt.values())
    print(f"GT: {len(gt)} test videos, {n_all} frames, {n_pos} anomalous ({100 * n_pos / n_all:.1f}%)")


if __name__ == "__main__":
    main()
