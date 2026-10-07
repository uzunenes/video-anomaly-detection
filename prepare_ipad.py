"""IPAD (Liu et al. 2024) -> the same cache layout as UCSD, one folder per scene (= one fixed camera):

    <out>/ipad_<scene>/train/<video>.npy   (N, H, W) uint8 grayscale
    <out>/ipad_<scene>/test/<video>.npy
    <out>/ipad_<scene>/gt.json            {"<video>": [0/1 per frame]}

Frames are read straight from IPAD_dataset.zip (no extraction), in parallel, one video per task.
Zip layout: IPAD_dataset/<scene>/{training,testing}/frames/<video>/<nnn>.jpg and
            IPAD_dataset/<scene>/test_label/<video:03d>.npy (per-frame labels of testing video <video>).
"""
import argparse
import io
import json
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ZIP = None


def _init(path):
    global ZIP
    ZIP = zipfile.ZipFile(path)


def _load_video(args):
    frames, size = args
    out = np.empty((len(frames), size, size), np.uint8)
    for i, n in enumerate(frames):
        im = Image.open(io.BytesIO(ZIP.read(n))).convert("L").resize((size, size), Image.BILINEAR)
        out[i] = np.asarray(im)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)
    videos = defaultdict(list)                       # (scene, split, video) -> frame names
    labels = {}
    for n in z.namelist():
        p = n.split("/")
        if n.endswith(".jpg") and len(p) == 6:
            videos[(p[1], p[2], p[4])].append(n)
        elif n.endswith(".npy") and p[2] == "test_label":
            labels[(p[1], int(Path(p[3]).stem))] = n
    scenes = sorted({k[0] for k in videos}) if not args.scenes else args.scenes

    with Pool(args.workers, initializer=_init, initargs=(args.zip,)) as pool:
        for sc in scenes:
            dst = Path(args.out) / f"ipad_{sc}"
            gt, n_frames = {}, {"training": 0, "testing": 0}
            for split, short in (("training", "train"), ("testing", "test")):
                (dst / short).mkdir(parents=True, exist_ok=True)
                keys = sorted(k for k in videos if k[0] == sc and k[1] == split)
                tasks = [(sorted(videos[k]), args.size) for k in keys]
                for k, arr in zip(keys, pool.imap(_load_video, tasks)):
                    vid = k[2]
                    if split == "testing":
                        ln = labels.get((sc, int(vid)))
                        if ln is None:
                            print(f"warning: {sc} test video {vid} has no label file, skipped")
                            continue
                        lab = np.load(io.BytesIO(z.read(ln))).astype(np.uint8).ravel()
                        if len(lab) != len(arr):
                            print(f"warning: {sc}/{vid}: {len(arr)} frames vs {len(lab)} labels, truncated")
                            m = min(len(lab), len(arr))
                            arr, lab = arr[:m], lab[:m]
                        gt[vid] = lab.tolist()
                    np.save(dst / short / f"{vid}.npy", arr)
                    n_frames[split] += len(arr)
            (dst / "gt.json").write_text(json.dumps(gt))
            n_pos = sum(sum(v) for v in gt.values())
            n_te = sum(len(v) for v in gt.values())
            print(f"{sc}: train {n_frames['training']} frames | test {n_te} frames in {len(gt)} videos, "
                  f"{100 * n_pos / max(n_te, 1):.1f}% anomalous", flush=True)


if __name__ == "__main__":
    main()
