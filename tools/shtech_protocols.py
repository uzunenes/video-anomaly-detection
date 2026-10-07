"""ShanghaiTech (12 test scenes, per-scene models): the final method and its channels under the live and literature
protocols, pooled over all test frames (micro AUC, the standard), plus the mean per-video AUC.

    python tools/shtech_protocols.py --runs runs --cache cache [--vlm runs/vlm]

Final method = I2 + Fsm + Fdm, Fisher-calibrated per scene on its own normal training videos (fixed beforehand on
UCSD/Avenue; nothing is tuned here). With --vlm, the per-scene VLM scores (vlm_score.py) are evaluated the same way.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuse_runs import normaliser  # noqa: E402

KEYS = {"I2": "frx_bayes", "Fsm": "flow_ubm_m", "Fdm": "flow_rx_m"}


def causal_median(s, w=9):
    p = np.concatenate([np.full(w - 1, s[0]), s])
    return np.median(np.lib.stride_tricks.sliding_window_view(p, w), axis=1)


def literature(s):
    return gaussian_filter1d((s - s.min()) / (s.max() - s.min() + 1e-12), 3)


def evaluate(scores, gt):
    """scores, gt: {video: array} -> live / literature micro AUC and mean per-video AUC (live)."""
    vids = sorted(gt)
    y = np.concatenate([gt[v] for v in vids])
    live = {v: causal_median(np.asarray(scores[v], float)) for v in vids}
    lit = {v: literature(np.asarray(scores[v], float)) for v in vids}
    per_video = [roc_auc_score(gt[v], live[v]) for v in vids if 0 < gt[v].sum() < len(gt[v])]
    return {"live_micro": round(float(roc_auc_score(y, np.concatenate([live[v] for v in vids]))), 4),
            "literature_micro": round(float(roc_auc_score(y, np.concatenate([lit[v] for v in vids]))), 4),
            "live_mean_per_video": round(float(np.mean(per_video)), 4), "videos": len(vids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--vlm", default="")
    args = ap.parse_args()
    gt, chan = {}, {k: {} for k in KEYS}
    for ev in sorted(glob.glob(f"{args.runs}/shtech_*_2d_noadv/eval_shtech_*_pool48_mc5/scores.npz")):
        sc = Path(ev).parent.name.split("eval_shtech_")[1][:2]
        S = np.load(ev)
        g = {k: np.asarray(v) for k, v in json.loads((Path(args.cache) / f"shtech_{sc}" / "gt.json").read_text()).items()}
        for name, key in KEYS.items():
            f = normaliser(S[f"ref/{key}"], "fisher")              # per-scene calibration on normal training video
            for v in g:
                chan[name][v] = f(np.asarray(S[f"{key}/{v}"], float))
        gt.update(g)
    res = {"scenes": len({v[:2] for v in gt}), "frames": int(sum(len(x) for x in gt.values())),
           "anomalous": int(sum(int(x.sum()) for x in gt.values()))}
    res["final_I2+Fsm+Fdm"] = evaluate({v: sum(chan[k][v] for k in KEYS) for v in gt}, gt)
    res["flow_only_Fsm+Fdm (post hoc)"] = evaluate({v: chan["Fsm"][v] + chan["Fdm"][v] for v in gt}, gt)
    for k in KEYS:
        res[f"single_{k}"] = evaluate(chan[k], gt)
    if args.vlm:
        vlm = {}
        for f in sorted(glob.glob(f"{args.vlm}/shtech_*_qwen3vl_*/scores.npz")):
            V = np.load(f)
            vlm.update({k[4:]: V[k] for k in V.files if k.startswith("vlm/")})
        common = {v: gt[v] for v in gt if v in vlm}
        if common:
            res["vlm"] = evaluate(vlm, common)
            res["final_on_same_videos"] = evaluate({v: sum(chan[k][v] for k in KEYS) for v in common}, common)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
