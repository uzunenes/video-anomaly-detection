"""VLM vs. our method vs. the cascade, frame AUC under the live and the literature protocol.

    python tools/vlm_eval.py --ds ped2 --vlm runs/vlm/ped2_q3vl8b --scores ../checkpoints/scores --out res.json

Ours = I2 + Fsm + Fdm with Fisher's method (the thesis method); its channels are read from the evaluation runs under
--scores/runs (tools/fuse_runs.py layout).
Protocols: live = causal median over 9 frames, no per-video statistics; literature = per-video min-max + centred
Gaussian smoothing (sigma 3).
Cascade (live): our fused score gates the VLM. A frame whose smoothed score exceeds a Fisher null quantile (--gates,
Gamma(3, 1) for three calibrated channels; correlated channels make the real false-alarm share larger) is sent to the VLM and keeps its score times 2 P(Yes) (a rejection lowers
it); every other frame keeps our score. Reported: AUC, the share of frames sent to the VLM, and at the gate the share
of anomalous / normal frames raising an alarm before and after the verification (P(Yes) > 0.5).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gamma
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fuse_runs  # noqa: E402

OURS = ("I2", "Fsm", "Fdm")


def causal_median(s, w=9):
    p = np.concatenate([np.full(w - 1, s[0]), s])
    return np.median(np.lib.stride_tricks.sliding_window_view(p, w), axis=1)


def literature(s):
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return gaussian_filter1d(s, 3)


def auc(scores, gt, f):
    vids = sorted(gt)
    return round(float(roc_auc_score(np.concatenate([gt[v] for v in vids]),
                                     np.concatenate([f(np.asarray(scores[v], float)) for v in vids]))), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, help="ped2 | ped1 | avenue (the fuse_runs dataset name)")
    ap.add_argument("--vlm", required=True, help="vlm_score.py output folder")
    ap.add_argument("--scores", required=True, help="folder holding runs/<run>/eval_*/scores.npz")
    ap.add_argument("--gates", default="0.9,0.95,0.99,0.999", help="Fisher null quantiles used as VLM gates")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    V = np.load(Path(args.vlm) / "scores.npz")
    vlm = {k.split("/", 1)[1]: V[k].astype(float) for k in V.files if k.startswith("vlm/")}   # log-odds of "Yes"
    p_yes = {v: 1 / (1 + np.exp(-x)) for v, x in vlm.items()}
    gt = {k.split("/", 1)[1]: V[k].astype(int) for k in V.files if k.startswith("gt/")}
    fuse_runs.R = Path(args.scores) / "runs"
    ours = None
    for c in OURS:
        ch = fuse_runs.channel(args.ds, c)
        if ch is None:
            sys.exit(f"missing channel {c} for {args.ds} under {fuse_runs.R}")
        sc, ref = ch
        f = fuse_runs.normaliser(ref, "fisher")
        part = {v: f(np.asarray(sc[v], float)) for v in vlm}
        ours = part if ours is None else {v: ours[v] + part[v] for v in vlm}
    for v in vlm:
        if len(ours[v]) != len(vlm[v]):
            sys.exit(f"{v}: our scores have {len(ours[v])} frames, the VLM's {len(vlm[v])}")
    live_ours = {v: causal_median(ours[v]) for v in vlm}
    y = np.concatenate([gt[v] for v in sorted(gt)]).astype(bool)
    ident = lambda s: s
    cascades = []
    for q in (float(x) for x in args.gates.split(",")):
        thr = float(gamma.ppf(q, 3))
        gated = {v: live_ours[v] > thr for v in vlm}
        casc = {v: np.where(gated[v], live_ours[v] * 2 * p_yes[v], live_ours[v]) for v in vlm}
        g = np.concatenate([gated[v] for v in sorted(gt)])
        keep = np.concatenate([gated[v] & (p_yes[v] > 0.5) for v in sorted(gt)])
        cascades.append({"gate_quantile": q, "gate_threshold": round(thr, 3), "auc_live": auc(casc, gt, ident),
                         "share_of_frames_to_vlm": round(float(g.mean()), 4),
                         "alarm_rate_anomalous_before": round(float(g[y].mean()), 4),
                         "alarm_rate_anomalous_after": round(float(keep[y].mean()), 4),
                         "alarm_rate_normal_before": round(float(g[~y].mean()), 4),
                         "alarm_rate_normal_after": round(float(keep[~y].mean()), 4)})
    res = {"ds": args.ds, "vlm": str(args.vlm), "frames": int(len(y)), "anomalous": int(y.sum()),
           "auc": {"vlm_live": auc(vlm, gt, causal_median), "vlm_literature": auc(vlm, gt, literature),
                   "vlm_raw": auc(vlm, gt, ident),
                   "ours_live": auc(ours, gt, causal_median), "ours_literature": auc(ours, gt, literature)},
           "cascade": cascades}
    tf = Path(args.vlm) / "timing.json"
    if tf.exists():
        res["timing"] = json.loads(tf.read_text())
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
