"""Final multi-channel score (Fisher fusion of I2 + Fs + Fd) under different temporal post-processing / protocols.

    python tools/temporal_protocols.py ped2 ped1 avenue

    centred-median9      : what evaluate.py reported so far (scipy median_filter is centred: 4 frames look-ahead)
    causal-median9       : median of frames t-8..t (strictly causal)
    causal-kalman        : local-level Kalman filter (x_t = x_{t-1} + w, F_t = x_t + v), steady-state gain set so the
                           filter's effective memory matches the 9-frame median (q/r fixed in advance)
    fixedlag-kalman(4)   : Rauch-Tung-Striebel smoothing with a 4-frame lag (same latency as the centred median)
    literature           : per-video min-max + centred Gaussian smoothing (sigma = 3 frames), the usual protocol
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fuse_runs as fr  # noqa: E402

CHANNELS = tuple(__import__("os").environ.get("CHANNELS", "I2,Fsm,Fdm").split(","))


def causal_median(s, w=9):
    p = np.concatenate([np.full(w - 1, s[0]), s])
    return np.median(np.lib.stride_tricks.sliding_window_view(p, w), axis=1)


def kalman_gain(q_over_r):
    # steady-state Kalman gain of the local-level model: P = (P + q) r / (P + q + r)
    q, r = q_over_r, 1.0
    P = q
    for _ in range(200):
        P = (P + q) * r / (P + q + r)
    return (P + q) / (P + q + r)


def causal_kalman(s, g):
    x, out = s[0], np.empty_like(s, dtype=float)
    for t, z in enumerate(s):
        x = x + g * (z - x)
        out[t] = x
    return out


def fixedlag(s, g, lag=4):
    f = causal_kalman(s, g)
    out = np.empty_like(f)
    for t in range(len(s)):
        e = min(len(s) - 1, t + lag)
        out[t] = causal_kalman(s[t:e + 1][::-1], g)[-1] * 0.5 + f[t] * 0.5 if e > t else f[t]
    return out


def main():
    g = kalman_gain(0.05)          # effective memory ~ 1/g frames; q/r fixed a priori (~ 9-frame median)
    res = {}
    for ds in sys.argv[1:] or ["ped2", "ped1", "avenue"]:
        vids, y, Zs, _ = fr.load(ds)
        F = {v: sum(Zs[(n, "fisher")][v] for n in CHANNELS) for v in vids}
        post = {
            "centred-median9": lambda s: median_filter(s, 9, mode="nearest"),
            "causal-median9": causal_median,
            "causal-kalman": lambda s: causal_kalman(s, g),
            "fixedlag-kalman(4)": lambda s: fixedlag(s, g),
            "raw (no temporal filter)": lambda s: s,
            "literature (video min-max + centred gauss 3)": lambda s: gaussian_filter1d(
                (s - s.min()) / (s.max() - s.min() + 1e-12), 3, mode="nearest"),
        }
        res[ds] = {k: round(roc_auc_score(y, np.concatenate([f(F[v]) for v in vids])), 4) for k, f in post.items()}
    print(f"kalman gain {g:.3f} (effective memory ~{1 / g:.1f} frames)")
    keys = list(next(iter(res.values())))
    print(f"{'protocol':<46}" + "".join(f"{d:>9}" for d in res))
    for k in keys:
        print(f"{k:<46}" + "".join(f"{res[d][k]:>9.4f}" for d in res))


if __name__ == "__main__":
    main()
