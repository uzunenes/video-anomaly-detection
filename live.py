"""Live CCTV use of a trained AEAN model.

    # 1) record normal footage of the camera (grayscale, resized, fixed fps)
    python live.py record --src rtsp://user:pass@ip/stream --minutes 10 --out cache/cam1/train/rec001.npy
    # 2) train on it exactly like the datasets
    python train.py --data cache/cam1 --variant 3d --out runs/cam1_3d
    # 3) run: background stats + alarm threshold from the recording, then score the stream
    python live.py run --src rtsp://... --run runs/cam1_3d --data cache/cam1 --save cam1_out.mp4

--src accepts an RTSP/HTTP URL, a video file, an image-sequence pattern (e.g. Test001/%03d.tif) or a webcam index.
The camera is sampled at --fps (default 10, the UCSD frame rate) so that T frames cover the same time span.
"""
import argparse
import collections
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from aean.data import load_split, to_unit
from aean.models import build
from aean.scoring import BackgroundStats, reconstruct, score_frame


def open_source(src: str):
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {src}")
    return cap


def frames(cap, fps: float, size: tuple[int, int], every: int = 0):
    """Yield grayscale frames resized to (H, W) at ~`fps`.

    Files are subsampled by frame count (`every`, auto from the file fps); live streams by wall clock.
    """
    is_file = cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    if is_file and every <= 0:
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        every = max(1, round(src_fps / fps)) if src_fps > 0 else 1
    period, last, i = 1.0 / fps, 0.0, -1
    while True:
        ok, f = cap.read()
        if not ok:
            return
        i += 1
        if is_file:
            if i % every:
                continue
        else:
            now = time.monotonic()
            if now - last < period:
                continue
            last = now
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        yield cv2.resize(g, (size[1], size[0]), interpolation=cv2.INTER_AREA)


def causal_median(s, w: int) -> list[float]:
    return [float(np.median(s[max(0, i - w + 1):i + 1])) for i in range(len(s))]


def record(args):
    cap = open_source(args.src)
    n_max = int(args.minutes * 60 * args.fps)
    buf = []
    for g in frames(cap, args.fps, args.size, args.every):
        buf.append(g)
        if len(buf) % 100 == 0:
            print(f"{len(buf)}/{n_max} frames")
        if len(buf) >= n_max:
            break
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.stack(buf))
    print(f"saved {len(buf)} frames -> {out}")


@torch.no_grad()
def run(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "args.json").read_text())
    ae, _, _ = build(cfg["variant"], cfg["T"])
    C = 1 if cfg["variant"] == "2d" else cfg["T"]          # residual vector length per pixel
    ae.load_state_dict(torch.load(run_dir / "model.pt", map_location=dev, weights_only=True)["ae"])
    ae = ae.to(dev).eval()
    P, T, variant = cfg["patch"], cfg["T"], cfg["variant"]
    ts = cfg.get("tstride", 1)
    clip_len = 1 if variant == "2d" else (T - 1) * ts + 1

    def residual(clip):
        x = clip[-1:] if variant == "2d" else clip[::ts]
        return x - reconstruct(ae, x, variant, P, P)          # stride = P: real-time setting

    # background statistics + alarm threshold from the normal recording(s)
    train = load_split(Path(args.data), "train")
    H, W = next(iter(train.values())).shape[1:]
    stats = BackgroundStats(C, H, W, P, dev)
    calib = []
    for v in train.values():
        vt = to_unit(torch.from_numpy(v).to(dev))
        for t in range(clip_len - 1, len(vt), 2):
            stats.update(residual(vt[t - clip_len + 1:t + 1]))
    stats.finalize()
    for v in train.values():
        vt = to_unit(torch.from_numpy(v).to(dev))
        s = [score_frame(residual(vt[t - clip_len + 1:t + 1]), stats, P)[0][args.detector]
             for t in range(clip_len - 1, len(vt), 2)]
        calib += causal_median(s, max(1, args.median // 2))  # calibration frames are every 2nd frame
    thr = float(np.quantile(calib, args.quantile))
    print(f"detector={args.detector} threshold={thr:.4g} (q={args.quantile} of normal scores)")

    cap = open_source(args.src)
    writer = None
    window = collections.deque(maxlen=clip_len)
    recent = collections.deque(maxlen=args.median)
    for i, g in enumerate(frames(cap, args.fps, (H, W), args.every)):
        window.append(torch.from_numpy(g).to(dev))
        if len(window) < clip_len:
            continue
        t0 = time.perf_counter()
        scores, maps = score_frame(residual(to_unit(torch.stack(list(window)))), stats, P)
        recent.append(scores[args.detector])
        smoothed = float(np.median(recent))                  # causal temporal median
        alarm = smoothed > thr
        ms = 1000 * (time.perf_counter() - t0)

        heat = maps[args.detector].clamp(min=0).sqrt().cpu().numpy()
        heat = cv2.applyColorMap(np.uint8(255 * np.clip(heat / (np.sqrt(thr) + 1e-9) / 2, 0, 1)), cv2.COLORMAP_JET)
        vis = cv2.addWeighted(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), 0.6, heat, 0.4, 0)
        cv2.putText(vis, f"{'ALARM ' if alarm else ''}score {smoothed / thr:.2f}x thr  {ms:.0f} ms",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if alarm else (0, 255, 0), 1)
        if alarm:
            print(f"[{time.strftime('%H:%M:%S')}] frame {i}: ALARM score/thr={smoothed / thr:.2f}")
        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
            writer.write(vis)
        if args.show:
            cv2.imshow("cctv_aean", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if writer:
        writer.release()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    size = lambda s: tuple(int(v) for v in s.lower().split("x"))
    r = sub.add_parser("record")
    r.add_argument("--src", required=True)
    r.add_argument("--minutes", type=float, default=10)
    r.add_argument("--fps", type=float, default=10)
    r.add_argument("--size", type=size, default=(240, 360), help="HxW")
    r.add_argument("--every", type=int, default=0, help="keep every n-th frame of a file (0 = auto)")
    r.add_argument("--out", required=True)
    l = sub.add_parser("run")
    l.add_argument("--src", required=True)
    l.add_argument("--run", required=True)
    l.add_argument("--data", required=True, help="folder with train/*.npy normal recordings")
    l.add_argument("--detector", default="frx_global")
    l.add_argument("--quantile", type=float, default=0.995)
    l.add_argument("--median", type=int, default=9)
    l.add_argument("--fps", type=float, default=10)
    l.add_argument("--every", type=int, default=0, help="keep every n-th frame of a file (0 = auto)")
    l.add_argument("--save", default="")
    l.add_argument("--show", action="store_true")
    args = ap.parse_args()
    record(args) if args.cmd == "record" else run(args)


if __name__ == "__main__":
    main()
