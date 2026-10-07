"""Late fusion of channels from different runs, with scales taken from NORMAL training frames only.

    python tools/fuse_runs.py ped2 ped1 avenue

Channels (frame scores and cross-fitted normal-frame reference scores saved in each evaluation's scores.npz):
    I2 / I3 : FRX-Bayes of the intensity AEAN, 2d / 3d          (runs/<ds>_<v>_noadv/eval_<ds>_pool48_mc4)
    Fs      : optical-flow speed GMM-UBM                          (same evaluation of the 2d run, channel flow_ubm)
    Fd      : 9-d flow histogram, per-cell FRX-Bayes              (channel flow_rx)
    D       : frozen ResNet-18 cell features, per-cell FRX-Bayes  (channel deep_rx)
    M2 / M3 : FRX-Bayes of the AEAN trained on flow magnitude     (runs/<ds>_flowmag_<v>_noadv/eval_<ds>_flowmag_pool48_mc4)
Normalisations of a channel's frame score s against its normal reference scores r:
    z      : (s - mean r) / std r
    fisher : -log p with p the upper-tail probability under r: empirical below the 90th percentile of r,
             exponential tail beyond it (peaks over threshold, GPD with xi = 0); channels add up (Fisher's method)
The channel combination and the normalisation are chosen leave-one-dataset-out: chosen on two datasets,
reported on the third. Frame AUC after the causal median 9, no per-video normalisation.
"""
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score

R = Path("runs")
# channel -> (run suffix, evaluation tag, detector key); I3 / M2 / M3 come from other runs than the 2d intensity run
SPEC = {"I2": ("2d", "pool48_mc4", "frx_bayes"), "I3": ("3d", "pool48_mc4", "frx_bayes"),
        "Fs": ("2d", "pool48_mc4", "flow_ubm"), "Fd": ("2d", "pool48_mc4", "flow_rx"), "D": ("2d", "pool48_mc4", "deep_rx"),
        "M2": ("flowmag_2d", "pool48_mc4", "frx_bayes"), "M3": ("flowmag_3d", "pool48_mc4", "frx_bayes"),
        "Fsm": ("2d", "pool48_mc5", "flow_ubm_m"), "Fdm": ("2d", "pool48_mc5", "flow_rx_m"), "Dn": ("2d", "pool48_mc5", "dino_rx"),
        "Fsr": ("2d", "pool48_mc6", "flow_ubm_r"), "Fdr": ("2d", "pool48_mc6", "flow_rx_r"),
        "Fstm": ("2d", "pool48_mc7", "flow_st_m"),
        "Pr": ("pred", "pool48_mc8", "frx_bayes")}
NAMES = tuple(SPEC)


def channel(ds: str, name: str):
    """-> ({video: frame scores}, normal reference scores) or None if missing."""
    run, tag, key = SPEC[name]
    data = f"{ds}_flowmag" if run.startswith("flowmag") else ds
    ev = R / f"{ds}_{run}_noadv" / f"eval_{data}_{tag}"
    if not (ev / "scores.npz").exists():
        return None
    S = np.load(ev / "scores.npz")
    if f"ref/{key}" not in S.files:
        return None
    return {k.split("/", 1)[1]: S[k] for k in S.files if k.startswith(key + "/")}, S[f"ref/{key}"]


def normaliser(ref: np.ndarray, kind: str):
    if kind == "z":
        m, sd = ref.mean(), ref.std() + 1e-12
        return lambda s: (s - m) / sd
    r = np.sort(ref)
    n = len(r)
    u = np.quantile(r, 0.9)
    beta = max(float((r[r > u] - u).mean()) if (r > u).any() else 1e-6, 1e-6)

    def f(s):
        p_emp = (n - np.searchsorted(r, s, side="left") + 1) / (n + 1)        # P(R >= s), smoothed
        nlog_tail = -np.log(0.1) + (s - u) / beta                            # -log of 0.1 exp(-(s - u) / beta)
        return np.where(s > u, nlog_tail, -np.log(p_emp))
    return f


def load(ds: str):
    gt = {k: np.asarray(v) for k, v in json.loads(Path(os.environ.get("VAD_CACHE", "cache"), f"{ds}/gt.json").read_text()).items()}
    vids = sorted(gt)
    ch = {n: c for n in NAMES if (c := channel(ds, n))}
    norm = {(n, k): normaliser(ch[n][1], k) for n in ch for k in ("z", "fisher")}
    Zs = {(n, k): {v: norm[(n, k)](ch[n][0][v]) for v in vids} for (n, k) in norm}
    return vids, np.concatenate([gt[v] for v in vids]), Zs, list(ch)


def main():
    dss = sys.argv[1:] or ["ped2", "ped1", "avenue"]
    data = {ds: load(ds) for ds in dss}
    common = [n for n in NAMES if all(n in data[ds][3] for ds in dss)]
    combos = [c for r in range(1, len(common) + 1) for c in combinations(common, r)]
    res = {}
    for ds, (vids, y, Zs, _) in data.items():
        for k in ("z", "fisher"):
            for c in combos:
                s = np.concatenate([median_filter(sum(Zs[(n, k)][v] for n in c), 9, mode="nearest") for v in vids])
                res[(ds, k, c)] = roc_auc_score(y, s)
    out = {"single": {}, "all_channels": {}, "lodo": {}}
    for ds in dss:
        out["single"][ds] = {n: round(res[(ds, "z", (n,))], 4) for n in common}
    for k in ("z", "fisher"):
        out["all_channels"][k] = {ds: round(res[(ds, k, tuple(common))], 4) for ds in dss}
    if len(dss) > 1:
        for held in dss:
            others = [d for d in dss if d != held]
            k, c = max(((k, c) for k in ("z", "fisher") for c in combos),
                       key=lambda kc: np.mean([res[(d, kc[0], kc[1])] for d in others]))
            out["lodo"][held] = {"norm": k, "channels": "+".join(c), "auc": round(res[(held, k, c)], 4),
                                 "chosen_on_mean": round(float(np.mean([res[(d, k, c)] for d in others])), 4)}
    top = sorted(((k, c) for k in ("z", "fisher") for c in combos),
                 key=lambda kc: -np.mean([res[(d, kc[0], kc[1])] for d in dss]))[:10]
    out["top_posthoc"] = [{"norm": k, "channels": "+".join(c), **{d: round(res[(d, k, c)], 4) for d in dss}}
                          for k, c in top]
    print(json.dumps(out, indent=1))
    (R / "fusion_lodo.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
