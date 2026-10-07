"""Cell-level Fisher fusion with a scan statistic, from the cell maps saved by `evaluate.py --save-cellmaps`.

    python tools/cell_fusion.py ped2 ped1 avenue

For channel k and cell c, the evidence of a test value x is -log p_k,c(x), where p is the upper-tail probability
under the cell's NORMAL reference values (cross-fitted), shrunk towards the scene-wide reference distribution:
    p_c(x) = (n_c * S_c(x) + kappa * S_g(x) + 1) / (n_c + kappa + 1),   S = empirical survival function,
a Bayesian (Dirichlet-process-style) shrinkage of the cell distribution towards the global one (kappa = 20).
Cells add their channels' evidence (Fisher); the frame score is the maximum over 6x6-cell windows (48 px) of the
window sum (a scan statistic). Frame AUC with a causal and a centred median 9, no per-video normalisation.
Frame-level fusion of the same channels (tools/fuse_runs.py) is printed for reference.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score

KAPPA, WIN = 20.0, 6
CHANNELS = {"I2": "frx_bayes", "Fs": "flow_ubm", "Fd": "flow_rx"}


def causal_median(s, w=9):
    p = np.concatenate([np.full(w - 1, s[0]), s])
    return np.median(np.lib.stride_tricks.sliding_window_view(p, w), axis=1)


class CellCalib:
    """Shrunk per-cell empirical survival function from reference maps (N, gy, gx), optionally with an exponential
    tail above the cell's 90th percentile (peaks over threshold) whose scale is shrunk towards the scene-wide one."""

    def __init__(self, ref: np.ndarray, tail: bool = False):
        r = torch.from_numpy(ref.astype(np.float32))
        self.g, self.tail = r.shape[1:], tail
        self.cell = r.flatten(1).T.contiguous().sort(1).values          # (G, N)
        self.glob = r.flatten().sort().values                            # (N*G,)
        n = self.cell.shape[1]
        k = max(1, int(0.1 * n))
        self.u = self.cell[:, n - k - 1]                                  # cell threshold (90th percentile)
        exc = self.cell[:, n - k:] - self.u[:, None]                      # (G, k) excesses
        beta_c, beta_g = exc.mean(1), exc.mean()
        self.beta = (k * beta_c + KAPPA * beta_g) / (k + KAPPA)           # shrunk tail scale
        self.beta = self.beta.clamp(min=float(beta_g) * 1e-2 + 1e-9)

    def evidence(self, x: np.ndarray) -> torch.Tensor:
        """(T, gy, gx) -> -log p (T, gy, gx)."""
        q = torch.from_numpy(x.astype(np.float32)).flatten(1).T.contiguous()   # (G, T)
        n, ng = self.cell.shape[1], len(self.glob)
        s_c = (n - torch.searchsorted(self.cell, q, right=False)).float() / n
        s_g = (ng - torch.searchsorted(self.glob, q.flatten(), right=False).reshape(q.shape)).float() / ng
        e = -torch.log((n * s_c + KAPPA * s_g + 1) / (n + KAPPA + 1))
        if self.tail:
            t = -np.log(0.1) + (q - self.u[:, None]) / self.beta[:, None]
            e = torch.where(q > self.u[:, None], torch.maximum(e, t), e)
        return e.T.reshape(-1, *self.g)


def scan(e: torch.Tensor) -> np.ndarray:
    """(T, gy, gx) evidence -> max over WIN x WIN windows of the sum (zero-padded borders)."""
    return (F.avg_pool2d(e[:, None], WIN, 1, WIN // 2, count_include_pad=True) * WIN * WIN).flatten(1).max(1).values.numpy()


def main():
    for ds in sys.argv[1:] or ["ped2", "ped1", "avenue"]:
        ev = Path(f"runs/{ds}_2d_noadv/eval_{ds}_pool48_cm")
        Z = np.load(ev / "cellmaps.npz")
        gt = {k: np.asarray(v) for k, v in json.loads(Path(os.environ.get("VAD_CACHE", "cache"), f"{ds}/gt.json").read_text()).items()}
        vids = sorted(gt)
        y = np.concatenate([gt[v] for v in vids])
        out = {}
        for tail in (False, True):
            calib = {c: CellCalib(Z[f"ref/{k}"], tail) for c, k in CHANNELS.items()}
            E = {c: {v: calib[c].evidence(Z[f"{k}/{v}"]) for v in vids} for c, k in CHANNELS.items()}
            for combo in (("I2",), ("Fs",), ("Fd",), ("I2", "Fs"), ("I2", "Fd"), ("Fs", "Fd"), ("I2", "Fs", "Fd")):
                s = {v: scan(sum(E[c][v] for c in combo)) for v in vids}
                out[("tail" if tail else "ecdf") + " " + "+".join(combo)] = (
                    roc_auc_score(y, np.concatenate([causal_median(s[v]) for v in vids])),
                    roc_auc_score(y, np.concatenate([median_filter(s[v], 9, mode="nearest") for v in vids])))
        print(f"\n## {ds} (cell-level Fisher + scan; AUC causal-median9 / centred-median9)")
        for k, (a, b) in out.items():
            print(f"{k:<16} {a:.4f} / {b:.4f}")


if __name__ == "__main__":
    main()
