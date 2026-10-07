"""ShanghaiTech: pool the per-scene final-method scores (AE 2d + flow-speed GMM-UBM + flow RX, Fisher calibrated per
scene on its own normal training videos) over all test frames: micro AUC (standard), and the mean per-video AUC."""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuse_runs import normaliser  # noqa: E402

KEYS = ("frx_bayes", "flow_ubm_m", "flow_rx_m")


def causal(s, w=9):
    p = np.concatenate([np.full(w - 1, s[0]), s])
    return np.median(np.lib.stride_tricks.sliding_window_view(p, w), axis=1)


ys, out = [], {"causal": [], "centred": [], "single": {k: [] for k in KEYS}}
per_video = []
for ev in sorted(glob.glob("runs/shtech_*_2d_noadv/eval_shtech_*_pool48_mc5")):
    sc = ev.split("eval_shtech_")[1][:2]
    S = np.load(Path(ev) / "scores.npz")
    gt = {k: np.asarray(v) for k, v in json.loads(Path(os.environ.get("VAD_CACHE", "cache"), f"shtech_{sc}/gt.json").read_text()).items()}
    N = {k: normaliser(S[f"ref/{k}"], "fisher") for k in KEYS}
    for v in sorted(gt):
        F = sum(N[k](S[f"{k}/{v}"]) for k in KEYS)
        ys.append(gt[v])
        out["causal"].append(causal(F))
        out["centred"].append(median_filter(F, 9, mode="nearest"))
        for k in KEYS:
            out["single"][k].append(causal(N[k](S[f"{k}/{v}"])))
        if 0 < gt[v].sum() < len(gt[v]):
            per_video.append(roc_auc_score(gt[v], causal(F)))
y = np.concatenate(ys)
res = {"n_videos": len(ys), "n_frames": int(len(y)), "anomalous": int(y.sum()),
       "micro_auc_causal": round(roc_auc_score(y, np.concatenate(out["causal"])), 4),
       "micro_auc_centred": round(roc_auc_score(y, np.concatenate(out["centred"])), 4),
       "macro_auc_causal (videos with both classes)": round(float(np.mean(per_video)), 4),
       "single_channels_causal": {k: round(roc_auc_score(y, np.concatenate(v)), 4) for k, v in out["single"].items()}}
print(json.dumps(res, indent=1))
Path("runs/shtech_pooled.json").write_text(json.dumps(res, indent=1))
