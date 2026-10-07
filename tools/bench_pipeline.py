"""Per-frame latency of the full final pipeline, frame by frame as on a live camera (batch 1).

    python tools/bench_pipeline.py --data cache/ped2 --run runs/ped2_2d_noadv --device cpu --threads 4

Stages timed per frame: DIS optical flow (preset medium, 1 thread) + cell flow features, AE 2d reconstruction with
non-overlapping patches + FRX-Bayes map, flow-speed GMM-UBM map, flow-histogram RX map, frame scores and Fisher
fusion. Models are fitted on a subset of the normal training frames first (not timed); only latency is measured here.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aean.channels import CellRX, CellUBM, flow_video, frame_scores_from_maps, speed  # noqa: E402
from aean.data import load_split, to_unit  # noqa: E402
from aean.models import build  # noqa: E402
from aean.scoring import BackgroundStats, reconstruct, frame_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--cell", type=int, default=8)
    ap.add_argument("--pool", type=int, default=48)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    import cv2
    cv2.setNumThreads(1)
    dev = args.device
    cfg = json.loads((Path(args.run) / "args.json").read_text())
    ae, _, _ = build(cfg["variant"], cfg["T"])
    ae.load_state_dict(torch.load(Path(args.run) / "model.pt", map_location=dev, weights_only=True)["ae"])
    ae = ae.to(dev).eval()
    P = cfg["patch"]
    train = load_split(Path(args.data), "train")
    test = load_split(Path(args.data), "test")
    vid = next(iter(test.values()))
    H, W = vid.shape[1:]

    # fit (not timed): residual stats, flow channels on a subset of normal frames
    stats = BackgroundStats(1, H, W, P, dev)
    tr = [v[::10] for v in train.values()]
    with torch.no_grad():
        for v in tr:
            vt = to_unit(torch.from_numpy(v).to(dev))
            for t in range(len(vt)):
                x = vt[t:t + 1]
                stats.update(x - reconstruct(ae, x, "2d", P, P))
    stats.finalize()
    feats = [flow_video(v[:60], gap=args.gap, cell=args.cell, preset="medium") for v in train.values()]
    ubm = CellUBM(dev).fit([speed(f) for f in feats])
    C, gy, gx = feats[0].shape[1:]
    rx = CellRX(C, gy, gx, dev).fit(feats)

    times = {"flow": [], "ae_frx": [], "flow_models": [], "total": []}
    with torch.no_grad():
        for t in range(args.gap, args.gap + args.frames + 5):
            t0 = time.perf_counter()
            pair = vid[t - args.gap:t + 1:args.gap]
            f = flow_video(pair, gap=1, cell=args.cell, preset="medium")[-1:]   # same features as the cached ones
            t1 = time.perf_counter()
            x = to_unit(torch.from_numpy(vid[t:t + 1]).to(dev))
            d = x - reconstruct(ae, x, "2d", P, P)
            m = stats.bayes_delta(d).reshape(H, W)
            s_ae = frame_score(m, args.pool)
            t2 = time.perf_counter()
            s_u = frame_scores_from_maps(ubm.maps(speed(f)), H, W, args.pool)
            s_r = frame_scores_from_maps(rx.maps(f), H, W, args.pool)
            _ = s_ae + float(s_u[0]) + float(s_r[0])              # stands in for the Fisher sum (same cost order)
            t3 = time.perf_counter()
            if t >= args.gap + 5:                                  # warm-up excluded
                times["flow"].append(t1 - t0); times["ae_frx"].append(t2 - t1)
                times["flow_models"].append(t3 - t2); times["total"].append(t3 - t0)
    res = {k: round(1000 * float(np.median(v)), 2) for k, v in times.items()}
    res["fps"] = round(1000 / res["total"], 1)
    res.update({"device": dev, "threads": args.threads, "frame": f"{H}x{W}"})
    print(json.dumps(res))


if __name__ == "__main__":
    main()
