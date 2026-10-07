"""ShanghaiTech Campus -> per-scene caches (same layout as prepare_ucsd.py / prepare_ipad.py), read from the zip.

    python prepare_shanghaitech.py --zip data/shanghaitech.zip --out cache --size 180x320

Expected inside the zip (any top folder): training/videos/<SS>_<VVV>.avi (normal only),
testing/frames/<SS>_<VVVV>/<NNN>.jpg and testing/test_frame_mask/<SS>_<VVVV>.npy (frame labels).
Scene SS (01..13) -> cache/shtech_SS/{train,test}/<video>.npy + gt.json. Frames grayscale, resized to --size.
The per-scene models follow the unsupervised protocol (13 fixed cameras, normal training videos only); results are
pooled over all test frames (micro AUC), which our per-scene Fisher calibration makes comparable across scenes.
"""
import argparse
import io
import json
import re
import tempfile
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ZIP = None


def _init(path):
    global ZIP
    ZIP = zipfile.ZipFile(path)


def _video(args):
    name, size, out = args
    import cv2
    cv2.setNumThreads(1)
    with tempfile.NamedTemporaryFile(suffix=".avi") as t:
        t.write(ZIP.read(name))
        t.flush()
        cap = cv2.VideoCapture(t.name)
        fr = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            fr.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (size[1], size[0]), interpolation=cv2.INTER_AREA))
    np.save(out, np.stack(fr))
    return Path(out).stem, len(fr)


def _frames(args):
    names, size, out = args
    import cv2
    cv2.setNumThreads(1)
    fr = [cv2.resize(cv2.imdecode(np.frombuffer(ZIP.read(n), np.uint8), cv2.IMREAD_GRAYSCALE), (size[1], size[0]),
                     interpolation=cv2.INTER_AREA) for n in names]
    np.save(out, np.stack(fr))
    return Path(out).stem, len(fr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="180x320", help="HxW")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    size = tuple(int(v) for v in args.size.lower().split("x"))
    out = Path(args.out)
    names = zipfile.ZipFile(args.zip).namelist()
    vids = sorted(n for n in names if re.search(r"training/videos/\d\d_\d+\.avi$", n))
    frames = defaultdict(list)
    for n in names:
        m = re.search(r"testing/frames/(\d\d_\d+)/[^/]+\.(jpg|png)$", n)
        if m:
            frames[m.group(1)].append(n)
    masks = {Path(n).stem: n for n in names if re.search(r"testing/test_frame_mask/\d\d_\d+\.npy$", n)}
    print(f"zip: {len(vids)} training videos, {len(frames)} test videos, {len(masks)} frame masks")
    jobs_v, jobs_f = [], []
    for n in vids:
        sc = Path(n).stem[:2]
        d = out / f"shtech_{sc}" / "train"
        d.mkdir(parents=True, exist_ok=True)
        if not (d / f"{Path(n).stem}.npy").exists():
            jobs_v.append((n, size, str(d / f"{Path(n).stem}.npy")))
    for v, fl in frames.items():
        d = out / f"shtech_{v[:2]}" / "test"
        d.mkdir(parents=True, exist_ok=True)
        if not (d / f"{v}.npy").exists():
            jobs_f.append((sorted(fl), size, str(d / f"{v}.npy")))
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.zip,)) as ex:
        for i, r in enumerate(ex.map(_video, jobs_v)):
            if i % 50 == 0:
                print("train", i, r, flush=True)
        for i, r in enumerate(ex.map(_frames, jobs_f)):
            if i % 20 == 0:
                print("test", i, r, flush=True)
    zf = zipfile.ZipFile(args.zip)
    gts = defaultdict(dict)
    for v in frames:
        lab = np.load(io.BytesIO(zf.read(masks[v]))).astype(np.uint8).ravel()
        n = np.load(out / f"shtech_{v[:2]}" / "test" / f"{v}.npy", mmap_mode="r").shape[0]
        if len(lab) != n:
            print(f"warning: {v}: {n} frames, {len(lab)} labels -> truncated")
            k = min(n, len(lab))
            lab = lab[:k]
            arr = np.load(out / f"shtech_{v[:2]}" / "test" / f"{v}.npy")[:k]
            np.save(out / f"shtech_{v[:2]}" / "test" / f"{v}.npy", arr)
        gts[v[:2]][v] = lab.tolist()
    for sc, g in sorted(gts.items()):
        (out / f"shtech_{sc}" / "gt.json").write_text(json.dumps(g))
        n_all = sum(len(x) for x in g.values())
        print(f"scene {sc}: {len(g)} test videos, {n_all} frames, {sum(sum(x) for x in g.values())} anomalous")


if __name__ == "__main__":
    main()
