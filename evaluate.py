"""Frame-level evaluation of a trained AEAN on UCSD test videos.

1. Background statistics of residuals are fitted on NORMAL training frames (per camera view).
2. Every test frame gets REM / FRX-global / FRX-cell / FRX-Bayes / WRX scores (causal clip for 1d/3d);
   2d/3d also get BGMM (DP-GMM on patch codes, aean/latent.py) and FUSED = z(FRX-Bayes) + z(BGMM),
   both z-scores taken from NORMAL training frames.
   FRX-STATE scores the residual against the background of the frame's scene state (cluster-based RX):
   states = Dirichlet-process mixture on coarse views of normal model inputs (aean/latent.py).
3. Scores are optionally median-filtered in time (cf. the old `result_median` experiments)
   and min-max normalised per video; micro AUC over all frames is reported.
4. Real-time cost is measured with non-overlapping patches (stride = P), one frame at a time.
5. Extra channels (aean/channels.py): optical-flow speed per cell scored by a MAP-adapted GMM-UBM (FLOW_UBM), the
   9-d flow histogram scored by per-cell FRX-Bayes (FLOW_RX) and frozen ResNet-18 cell features (DEEP_RX); MC_* = sum of z-scores of the frame
   scores, scales from NORMAL training frames (cross-fitted by video for the new channels).
"""
import argparse
import json
import time
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score, roc_curve

from aean.channels import (CellRX, CellSkewT, CellUBM, DeepFeatures, DinoFeatures, cached_features, fit_channel,
                           flow_video, frame_scores_from_maps, speed)
from aean.data import load_gt, load_split, to_unit
from aean.models import build, n_params
from aean.latent import LatentDPGMM, patch_latents, patches_to_map, scene_descriptor
from aean.scoring import (BackgroundStats, StateBackground, frame_score, reconstruct, residual_of, score_frame,
                          select_kappa0, set_roi_mask)

BASE_DETECTORS = ("rem", "frx_global", "frx_cell", "frx_bayes", "frx_t", "wrx")
LATENT_DETECTORS = ("bgmm", "fused")
STATE_DETECTORS = ("frx_state",)


def model_input(video: torch.Tensor, t: int, variant: str, T: int, s: int = 1) -> torch.Tensor:
    if variant == "2d":
        return video[t:t + 1]
    span = (T - 1) * s
    # frames before the first full clip use the first full clip (as before); a test video shorter
    # than one clip (IPAD S05 with T=16, s=2) repeats its first frame to fill the clip
    t = min(max(t, span), len(video) - 1)
    idx = torch.arange(t - span, t + 1, s, device=video.device).clamp(min=0)
    return video[idx]


def residual(ae, video, t, cfg, stride):
    x = model_input(video, t, cfg["variant"], cfg["T"], cfg.get("tstride", 1))
    return residual_of(ae, x, cfg["variant"], cfg["patch"], stride)


def cell_map(m: torch.Tensor, cell: int) -> np.ndarray:
    """Pixel map (H, W) -> mean over cell x cell blocks (ceil grid), float16."""
    return F.avg_pool2d(m[None, None], cell, ceil_mode=True)[0, 0].half().cpu().numpy()


def frame_scores(ae, video, t, cfg, args, stats, gmm, znorm, with_maps=False, state=None, cellmaps=None):
    """All detector scores for frame t (and the maps when asked); `state` = (state mixture, StateBackground).
    `cellmaps` (list): the FRX-Bayes map pooled to the channel cell grid is appended (per-cell fusion)."""
    P = cfg["patch"]
    x = model_input(video, t, cfg["variant"], cfg["T"], cfg.get("tstride", 1))
    d = residual_of(ae, x, cfg["variant"], P, args.stride)
    out, maps = score_frame(d, stats, P, args.pool)
    if cellmaps is not None:
        cellmaps.append(cell_map(maps["frx_bayes"], args.cell_ch))
    if state is not None:
        k = int(state[0].assign(scene_descriptor(x)[None])[0])
        maps["frx_state"] = state[1].delta(d, k)
        out["frx_state"] = frame_score(maps["frx_state"], args.pool or P)
    if gmm is not None:
        lat, meta, pos = patch_latents(ae, x, P, args.stride)
        m = patches_to_map(gmm.score(lat, pos), meta, P, args.stride)
        out["bgmm"] = frame_score(m, args.pool or P)
        maps["bgmm"] = m
        if znorm:
            out["fused"] = sum((out[k] - znorm[k][0]) / znorm[k][1] for k in ("frx_bayes", "bgmm"))
    return maps if with_maps else out


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    i = np.nanargmin(np.abs(fpr - (1 - tpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def post(scores: dict[str, np.ndarray], median: int, norm: bool) -> np.ndarray:
    out = []
    for s in scores.values():
        s = median_filter(s, size=median, mode="nearest") if median > 1 else s
        if norm:
            s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        out.append(s)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="training output folder")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--median", type=int, default=9)
    ap.add_argument("--stats-every", type=int, default=3)
    ap.add_argument("--pool", type=int, default=0, help="local-mean window for frame scores (0 = patch size)")
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    ap.add_argument("--no-latent", action="store_true", help="skip the DP-GMM latent detector")
    ap.add_argument("--latent-pos", type=float, default=2.0, help="weight of patch position in the DP-GMM (0 = off)")
    ap.add_argument("--no-state", action="store_true", help="skip the scene-state (cluster-based) FRX")
    ap.add_argument("--channels", default="flow,flowrx,deep",
                    help="extra channels: flow (speed GMM-UBM), flowrx (9-d flow FRX-Bayes), flowm / flowrxm (same with "
                         "the DIS 'medium' preset), deep (ResNet-18), dino (DINOv2 ViT-S/14) ('' = none)")
    ap.add_argument("--cell-ch", type=int, default=8, help="cell size (px) of the extra channels")
    ap.add_argument("--flow-gap", type=int, default=1, help="optical flow between frames t-g and t")
    ap.add_argument("--workers", type=int, default=16, help="CPU processes for optical flow")
    ap.add_argument("--save-cellmaps", action="store_true",
                    help="save per-cell maps (FRX-Bayes and channels) of test frames and cross-fitted normal frames")
    ap.add_argument("--states", type=int, default=10, help="max scene states of the Dirichlet-process mixture")
    ap.add_argument("--kappa0", type=float, default=20.0, help="prior strength (frames) of a state's background "
                    "(0 = choose by cross-validated likelihood on the training videos; tested: it picks 1-3 on both "
                    "Ped2 and IPAD, but on Ped2 detection gets worse, so likelihood does not track detection)")
    args = ap.parse_args()

    dev, run = "cuda", Path(args.run)
    cfg = json.loads((run / "args.json").read_text())
    ae, _, _ = build(cfg["variant"], cfg["T"])
    C = cfg["T"] if cfg["variant"] in ("1d", "3d") else 1   # residual vector length per pixel ('pred': 1 frame)
    ae.load_state_dict(torch.load(run / "model.pt", map_location=dev, weights_only=True)["ae"])
    ae = ae.to(dev).eval()
    P = cfg["patch"]
    data = Path(args.data)

    # 1. background statistics from normal training residuals
    train = load_split(data, "train")
    H, W = next(iter(train.values())).shape[1:]
    if (data / "roi_mask.npy").exists():                    # e.g. a forbidden zone: alarms only from cells inside it
        set_roi_mask(torch.from_numpy(np.load(data / "roi_mask.npy")).to(dev).float())
    stats = BackgroundStats(C, H, W, P, dev)
    ts = cfg.get("tstride", 1)
    state, labels = None, None
    if not args.no_state:                                   # scene states from coarse views of normal inputs
        desc = torch.stack([scene_descriptor(model_input(vt, t, cfg["variant"], cfg["T"], ts))
                            for vt in (to_unit(torch.from_numpy(v).to(dev)) for v in train.values())
                            for t in range(cfg["T"] - 1, len(vt), args.stats_every)])
        states = LatentDPGMM(dim=8, max_components=args.states, pos_weight=0).fit(desc)
        labels = states.assign(desc).tolist()
        state = (states, StateBackground(stats, args.states, args.kappa0))
    i, parts = 0, []                                        # per-video state statistics (for choosing kappa0)
    folds = [BackgroundStats(C, H, W, P, dev) for _ in range(2)]   # video-wise halves, for cross-fitted references
    for vi, v in enumerate(train.values()):
        vt = to_unit(torch.from_numpy(v).to(dev))
        part = StateBackground(stats, args.states) if state else None
        for t in range(cfg["T"] - 1, len(vt), args.stats_every):
            d = residual(ae, vt, t, cfg, args.stride)
            stats.update(d)
            folds[vi % 2].update(d)
            if state:
                part.update(d, labels[i])
            i += 1
        if state:
            parts.append(part)
    stats.finalize()
    folds = [f.finalize() if f.n.sum() > 0 else stats for f in folds]
    kappa_cv = {}
    if state:
        for q in parts:
            state[1].add(q)
        if args.kappa0 <= 0:
            state[1].kappa0, kappa_cv = select_kappa0(parts, stats)
        state[1].finalize()

    # 1a. heavy-tailed background: Student-t degrees of freedom by ML on normal residuals
    gen = torch.Generator(device=dev).manual_seed(1)
    deltas = []
    for v in train.values():
        vt = to_unit(torch.from_numpy(v).to(dev))
        for t in range(cfg["T"] - 1, len(vt), 7 * args.stats_every):
            dl = stats.bayes_delta(residual(ae, vt, t, cfg, args.stride))
            deltas.append(dl[torch.randint(len(dl), (2000,), generator=gen, device=dev)])
    nu = stats.fit_nu(torch.cat(deltas)[:200000], C)
    for f in folds:
        f.nu = nu

    # 1a'. cross-fitted reference scores on NORMAL frames: a video is scored with the statistics of the other half
    ref_cf = {d: [] for d in BASE_DETECTORS}
    cell_ref, cell_test = {"frx_bayes": []}, {}
    for vi, v in enumerate(train.values()):
        vt = to_unit(torch.from_numpy(v).to(dev))
        for t in range(cfg["T"] - 1, len(vt), 3 * args.stats_every):
            r, mp = score_frame(residual(ae, vt, t, cfg, args.stride), folds[1 - vi % 2], P, args.pool)
            for d in BASE_DETECTORS:
                ref_cf[d].append(r[d])
            if args.save_cellmaps:
                cell_ref["frx_bayes"].append(cell_map(mp["frx_bayes"], args.cell_ch))

    # 1b. nonparametric Bayesian background in latent space (2d / 3d only)
    use_latent = cfg["variant"] in ("2d", "3d") and not args.no_latent
    DETECTORS = BASE_DETECTORS + (LATENT_DETECTORS if use_latent else ()) + (STATE_DETECTORS if state else ())
    gmm, znorm = None, {}
    if use_latent:
        g = torch.Generator(device=dev).manual_seed(0)
        feats, poss, cap = [], [], 60000
        for v in train.values():
            vt = to_unit(torch.from_numpy(v).to(dev))
            for t in range(cfg["T"] - 1, len(vt), 2 * args.stats_every):
                lat, _, pos = patch_latents(ae, model_input(vt, t, cfg["variant"], cfg["T"], cfg.get("tstride", 1)), P, args.stride)
                pick = torch.randperm(len(lat), generator=g, device=dev)[:48]
                feats.append(lat[pick]); poss.append(pos[pick])
        feats, poss = torch.cat(feats), torch.cat(poss)
        keep = torch.randperm(len(feats), generator=g, device=dev)[:cap]
        gmm = LatentDPGMM(pos_weight=args.latent_pos).fit(feats[keep], poss[keep])
        ref = {"frx_bayes": [], "bgmm": []}                     # frame-score scale on normal frames
        for v in train.values():
            vt = to_unit(torch.from_numpy(v).to(dev))
            for t in range(cfg["T"] - 1, len(vt), 5 * args.stats_every):
                r = frame_scores(ae, vt, t, cfg, args, stats, gmm, None)
                ref["frx_bayes"].append(r["frx_bayes"]); ref["bgmm"].append(r["bgmm"])
        znorm = {k: (float(np.mean(v)), float(np.std(v)) + 1e-12) for k, v in ref.items()}

    # 2. score every test frame
    test, gt = load_split(data, "test"), load_gt(data)
    scores = {d: {} for d in DETECTORS}
    for name, v in test.items():
        vt = to_unit(torch.from_numpy(v).to(dev))
        cm = [] if args.save_cellmaps else None
        rows = [frame_scores(ae, vt, t, cfg, args, stats, gmm, znorm, state=state, cellmaps=cm) for t in range(len(vt))]
        if cm is not None:
            cell_test[f"frx_bayes/{name}"] = np.stack(cm)
        for d in DETECTORS:
            scores[d][name] = np.array([r[d] for r in rows])
    y = np.concatenate([gt[n] for n in test])

    # 2b. extra channels scored with per-cell FRX-Bayes, and their fusion
    ch_info, zs = {}, dict(znorm)                            # zs: (mean, std) of frame scores on normal frames
    Pc = args.pool or cfg["patch"]
    for c in [c for c in args.channels.split(",") if c]:
        if c == "flowstm":                                   # skewed-t background on the 9-d flow features (DIS medium)
            fname = f"flow_g{args.flow_gap}_c{args.cell_ch}_medium"
            fn, w = partial(flow_video, gap=args.flow_gap, cell=args.cell_ch, preset="medium"), args.workers
            key, make, prep = "flow_st_m", partial(CellSkewT, dev), (lambda f: f)
        elif c in ("flow", "flowrx", "flowm", "flowrxm", "flowr", "flowrxr"):
            preset = {"m": "medium", "r": "raft"}.get(c[-1], "fast")
            fname = f"flow_g{args.flow_gap}_c{args.cell_ch}" + ("" if preset == "fast" else f"_{preset}")
            fn = partial(flow_video, gap=args.flow_gap, cell=args.cell_ch, preset=preset)
            w = 0 if preset == "raft" else args.workers               # RAFT runs on the GPU in this process
            sfx = {"medium": "_m", "raft": "_r"}.get(preset, "")
            key, make, prep = ((f"flow_ubm{sfx}", partial(CellUBM, dev), speed) if c.startswith("flow") and "rx" not in c
                               else (f"flow_rx{sfx}", None, lambda f: f))
        elif c == "deep":
            fname, fn, w = f"deep_c{args.cell_ch}", DeepFeatures(dev, args.cell_ch), 0
            key, make, prep = "deep_rx", None, (lambda f: f)
        elif c == "dino":
            fname, fn, w = f"dino_c{args.cell_ch}", DinoFeatures(dev, args.cell_ch), 0
            key, make, prep = "dino_rx", None, (lambda f: f)
        else:
            raise SystemExit(f"unknown channel {c}")
        ftr = {k: prep(v) for k, v in cached_features(data, fname, "train", train, fn, w).items()}
        fte = {k: prep(v) for k, v in cached_features(data, fname, "test", test, fn, w).items()}
        if make is None:
            C_, gy_, gx_ = next(iter(ftr.values())).shape[1:]
            make = partial(CellRX, C_, gy_, gx_, dev)
        model, zs[key], ref_cf[key], rmaps = fit_channel(ftr, make, H=H, W=W, pool=Pc)
        scores[key] = {}
        for n in test:
            mp = model.maps(fte[n])
            scores[key][n] = frame_scores_from_maps(mp, H, W, Pc)
            if args.save_cellmaps:
                cell_test[f"{key}/{n}"] = mp.half().cpu().numpy()
        if args.save_cellmaps:
            cell_ref[key] = rmaps
        t0 = time.perf_counter()
        fn(next(iter(test.values()))[:41])                    # per-frame feature cost (41 frames, batched for deep)
        ch_info[key] = {"feature": fname, "znorm": zs[key], "feature_ms": round(1000 * (time.perf_counter() - t0) / 41, 2)}
        del ftr, fte
    DETECTORS += tuple(ch_info)
    zsum = lambda keys, n: sum((scores[k][n] - zs[k][0]) / zs[k][1] for k in keys)
    for cname, keys in (("mc_ae_dino", ["frx_bayes", "dino_rx"]), ("mc_ae_flow", ["frx_bayes", "flow_ubm", "flow_rx"]),
                        ("mc_all", ["frx_bayes", "flow_ubm", "flow_rx", "deep_rx"])):
        if all(k in zs for k in keys):
            scores[cname] = {n: zsum(keys, n) for n in test}
            DETECTORS += (cname,)

    # 3. AUC table
    results = {"run": cfg, "eval": vars(args), "n_frames": int(len(y)), "anomalous": int(y.sum()), "auc": {}}
    results["student_t_nu"] = nu
    if ch_info:
        results["channels"] = ch_info
    if gmm is not None:
        results["latent"] = {"dpgmm_active_components": gmm.n_active, "pca_dim": gmm.dim,
                             "pos_weight": gmm.pos_weight, "znorm": znorm}
    if state:
        results["state"] = {"active_states": state[0].n_active, "kappa0": state[1].kappa0,
                            "kappa0_cv_loglik": {str(k): v for k, v in kappa_cv.items()},
                            "train_frames_per_state": [int(f) for f in state[1].frames.tolist()]}
    for d in DETECTORS:
        for tag, med, norm in [("raw", 1, False), (f"median{args.median}", args.median, False),
                               (f"median{args.median}+videonorm", args.median, True)]:
            s = post(scores[d], med, norm)
            results["auc"][f"{d}/{tag}"] = {"auc": round(float(roc_auc_score(y, s)), 4), "eer": round(eer(y, s), 4)}

    # 4. real-time cost: non-overlapping patches, one frame at a time
    name0 = next(iter(test))
    vt = to_unit(torch.from_numpy(test[name0]).to(dev))
    t_lo = min(cfg["T"], max(0, len(vt) - 20))                # short test sets (e.g. short frame blocks)
    for t in range(t_lo, min(len(vt), t_lo + 10)):
        score_frame(residual(ae, vt, t, cfg, P), stats, P)
    torch.cuda.synchronize()
    t0, n = time.perf_counter(), max(1, min(100, len(vt) - t_lo))
    for t in range(t_lo, t_lo + n):
        score_frame(residual(ae, vt, t, cfg, P), stats, P)
    torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0) / n
    results["realtime"] = {"ms_per_frame": round(ms, 2), "fps": round(1000 / ms, 1),
                           "ae_params_M": round(n_params(ae) / 1e6, 3), "device": torch.cuda.get_device_name()}

    out = run / f"eval_{data.name}{'_' + args.tag if args.tag else ''}"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    np.savez(out / "scores.npz", **{f"{d}/{n}": s for d in DETECTORS for n, s in scores[d].items()},
             **{f"gt/{n}": gt[n] for n in test},
             **{f"ref/{d}": np.asarray(v) for d, v in ref_cf.items()})   # normal-frame scores, cross-fitted
    if args.save_cellmaps:
        np.savez_compressed(out / "cellmaps.npz", **cell_test,
                            **{f"ref/{k}": (np.stack(v) if isinstance(v, list) else v) for k, v in cell_ref.items()})

    # 5. figures: best detector curves + localisation maps
    best = max(results["auc"], key=lambda k: results["auc"][k]["auc"])
    det = best.split("/")[0]
    names = list(test)
    cols = 4
    fig, axs = plt.subplots(int(np.ceil(len(names) / cols)), cols, figsize=(16, 2.2 * np.ceil(len(names) / cols)))
    for ax, n in zip(axs.flat, names):
        s = median_filter(scores[det][n], size=args.median, mode="nearest")
        ax.plot((s - s.min()) / (s.max() - s.min() + 1e-12), lw=1)
        ax.fill_between(range(len(s)), 0, gt[n], color="red", alpha=0.15, step="mid")
        ax.set_title(n, fontsize=8)
        ax.set_yticks([])
    fig.suptitle(f"{data.name} | {cfg['variant']} | {det} (median {args.median}) | shaded = ground-truth anomaly")
    fig.tight_layout()
    fig.savefig(out / "score_curves.png", dpi=110)
    plt.close(fig)

    picks = [n for n in names if gt[n].any()][:6]
    fig, axs = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), 5), squeeze=False)
    for j, n in enumerate(picks):
        vt = to_unit(torch.from_numpy(test[n]).to(dev))
        idx = np.where(gt[n])[0]
        t = int(idx[np.argmax(scores[det][n][idx])])
        maps = frame_scores(ae, vt, t, cfg, args, stats, gmm, znorm, with_maps=True, state=state)
        axs[0, j].imshow(test[n][t], cmap="gray")
        axs[0, j].set_title(f"{n} frame {t + 1}", fontsize=8)
        axs[1, j].imshow(test[n][t], cmap="gray")
        m = maps.get(det, maps["frx_bayes"]).cpu().numpy()
        axs[1, j].imshow(np.clip(m, None, np.quantile(m, 0.995)), cmap="jet", alpha=0.45)
        for a in axs[:, j]:
            a.axis("off")
    fig.suptitle(f"{data.name} | {cfg['variant']} | {det} anomaly map")
    fig.tight_layout()
    fig.savefig(out / "anomaly_maps.png", dpi=110)
    plt.close(fig)

    print(json.dumps({"best": best, **results["auc"][best], **results["realtime"]}))


if __name__ == "__main__":
    main()
