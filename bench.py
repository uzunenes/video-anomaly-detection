"""Per-frame latency of the deployed path: reconstruction with non-overlapping patches (stride = P)
plus all detectors, batch 1, on an otherwise idle device. CPU numbers are a proxy for embedded targets.

    python bench.py --data cache/ped2 --runs runs/ped2_3d_noadv runs/ped2_2d_noadv --device cpu --threads 4
"""
import argparse
import json
import time
from pathlib import Path

import torch

from aean.data import load_split, to_unit
from aean.models import build, n_params
from aean.scoring import BackgroundStats, reconstruct, score_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--out", default="runs/bench.json")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    dev = args.device
    train, test = load_split(Path(args.data), "train"), load_split(Path(args.data), "test")
    first_train = to_unit(torch.from_numpy(next(iter(train.values()))).to(dev))
    video = to_unit(torch.from_numpy(next(iter(test.values()))).to(dev))
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)

    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() else {}
    for run in args.runs:
        cfg = json.loads((Path(run) / "args.json").read_text())
        ae, _, _ = build(cfg["variant"], cfg["T"])
        ae.load_state_dict(torch.load(Path(run) / "model.pt", map_location=dev, weights_only=True)["ae"])
        ae = ae.to(dev).eval()
        P, T, v, ts = cfg["patch"], cfg["T"], cfg["variant"], cfg.get("tstride", 1)
        span = (T - 1) * ts
        C = 1 if v == "2d" else T

        def res(clip):
            x = clip[-1:] if v == "2d" else clip[::ts]
            return x - reconstruct(ae, x, v, P, P)

        H, W = video.shape[1:]
        stats = BackgroundStats(C, H, W, P, dev)
        for t in range(span, len(first_train), 5):
            stats.update(res(first_train[t - span:t + 1]))
        stats.finalize()
        with torch.no_grad():
            for t in range(span + 1, span + 11):
                score_frame(res(video[t - span:t + 1]), stats, P)
            sync()
            t0 = time.perf_counter()
            for t in range(span + 1, span + 1 + args.frames):
                score_frame(res(video[t - span:t + 1]), stats, P)
            sync()
        ms = 1000 * (time.perf_counter() - t0) / args.frames
        name = torch.cuda.get_device_name() if dev == "cuda" else f"CPU x{torch.get_num_threads()}"
        key = f"{Path(run).name}@{dev}{args.threads or ''}"
        results[key] = {"run": Path(run).name, "device": name, "ms_per_frame": round(ms, 2), "fps": round(1000 / ms, 1),
                        "ae_params_M": round(n_params(ae) / 1e6, 3), "frame": f"{H}x{W}"}
        print(json.dumps(results[key]))
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
