"""Frame reconstruction and the detectors of Arısoy (2021, Sec. 3.3) adapted to a static camera.

For every pixel the residual vector d = x - x_hat has C components:
    '2d' -> C = 1 (current frame),  '1d' / '3d' -> C = T (the causal clip).

Detectors (per-pixel maps, higher = more anomalous):
    rem        : ||d||^2                                  (REM, eq. 3.8)
    frx_global : Mahalanobis of d, background stats from NORMAL training residuals (FRX, eq. 3.11)
    frx_cell   : same, but stats per P x P cell of the fixed camera view (scene-specific background)
    frx_bayes  : per-cell stats shrunk towards the global ones (moment-matched mixture, weight alpha);
                 equivalent to a conjugate prior centred on the global background, in the spirit of the
                 Bayesian covariance regularisation of Şahin (GTU MSc 2019) / Arısoy (2021, Ch. 2)
    frx_t      : FRX-Bayes with a heavy-tailed (multivariate Student-t) background instead of a Gaussian,
                 after the non-Gaussian background models of Aytekin (GTU MSc 2022, advisor Kayabol):
                 0.5 (nu + C) log(1 + delta / (nu - 2)) (no per-cell log-det offset), nu fitted by maximum
                 likelihood on normal training residuals (delta = Mahalanobis^2 under FRX-Bayes stats)
    wrx        : RX on d with pixel weights 1 / closing(REM), estimated on the frame itself (WRX, eq. 3.13-3.17)
    frx_state  : cluster-based RX over scene states (cf. CBAD, Carlotto 2005): the frame is assigned to a scene
                 state (Dirichlet-process mixture on coarse views of normal clips, evaluate.py) and its residual
                 is scored against that state's cell statistics, shrunk towards the FRX-Bayes cell statistics
                 by a conjugate prior (StateBackground, kappa0 = 20 pseudo-frames, or chosen by cross-validated
                 likelihood with select_kappa0). Meant for periodic process scenes: when an anomaly changes the
                 whole view (UCSD) the frame can be assigned to a busy state that absorbs it
Frame score = max over the map of its P x P local mean.
"""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F


def _patch_pad(n: int, P: int, stride: int) -> int:
    return (stride - (n - P) % stride) % stride if n > P else P - n


@torch.no_grad()
def reconstruct(ae, x: torch.Tensor, variant: str, P: int, stride: int, chunk: int = 8192) -> torch.Tensor:
    """x: (C, H, W) in [-1, 1]; returns the reconstruction with the same shape."""
    C, H, W = x.shape
    if variant == "1d":                                     # every pixel's temporal profile
        v = x.permute(1, 2, 0).reshape(-1, 1, C)
        out = torch.cat([ae(v[i:i + chunk]) for i in range(0, len(v), chunk)])
        return out.reshape(H, W, C).permute(2, 0, 1)
    ph, pw = _patch_pad(H, P, stride), _patch_pad(W, P, stride)
    xp = F.pad(x[None], (0, pw, 0, ph), mode="reflect")
    cols = F.unfold(xp, P, stride=stride)                   # (1, C*P*P, L)
    L = cols.shape[-1]
    patches = cols[0].T.reshape(L, C, P, P)
    rec = torch.cat([ae(patches[i:i + chunk]) for i in range(0, L, chunk)])
    rec_cols = rec.reshape(L, -1).T[None]                     # output channels may differ ('pred': 1)
    size = xp.shape[-2:]
    num = F.fold(rec_cols, size, P, stride=stride)
    den = F.fold(torch.ones_like(rec_cols), size, P, stride=stride)
    return (num / den)[0, :, :H, :W]


def residual_of(ae, x: torch.Tensor, variant: str, P: int, stride: int) -> torch.Tensor:
    """Residual of a model input x (C, H, W). 'pred': the last frame minus its prediction from the previous ones."""
    if variant == "pred":
        return x[-1:] - reconstruct(ae, x[:-1], variant, P, stride)
    return x - reconstruct(ae, x, variant, P, stride)


def closing(m: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Grey-level morphological closing (dilation then erosion), as used for FREM."""
    m = F.max_pool2d(m[None, None], k, 1, k // 2)
    return -F.max_pool2d(-m, k, 1, k // 2)[0, 0]


ROI_MASK = None          # optional (H, W) 0/1 mask: only cells inside the region of interest can raise an alarm


def set_roi_mask(mask):
    """Restrict every frame score to a region of interest (e.g. a forbidden zone); None switches it off."""
    global ROI_MASK
    ROI_MASK = mask


def local_mean_max(x: torch.Tensor, P: int) -> torch.Tensor:
    """Max over positions of the P x P local mean, (N, H, W) -> (N,). Same result as
    avg_pool2d(x, P, stride 1, padding P // 2).max(), computed with a summed-area table: the cost no longer grows
    with P (the sliding window was the bottleneck of the CPU pipeline)."""
    if ROI_MASK is not None and tuple(ROI_MASK.shape) == tuple(x.shape[-2:]):
        x = x * ROI_MASK.to(x.device, x.dtype)
    p = P // 2
    s = F.pad(x.double(), (p + 1, p, p + 1, p)).cumsum(1).cumsum(2)        # leading zero row / column
    win = s[:, P:, P:] - s[:, :-P, P:] - s[:, P:, :-P] + s[:, :-P, :-P]
    return (win.flatten(1).max(1).values / (P * P)).float()


def frame_score(m: torch.Tensor, P: int) -> float:
    return local_mean_max(m[None], P).item()


def _mahalanobis(d: torch.Tensor, mu: torch.Tensor, prec: torch.Tensor) -> torch.Tensor:
    """d: (N, C); mu: (N|1, C); prec: (N|1, C, C) -> (N,)"""
    z = d - mu
    return torch.einsum("nc,ncd,nd->n", z, prec.expand(len(z), -1, -1), z)


def _precision(cov: torch.Tensor) -> torch.Tensor:
    C = cov.shape[-1]
    eye = torch.eye(C, device=cov.device)
    tr = cov.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
    return torch.linalg.inv(cov + (1e-3 * tr / C + 1e-8) * eye)


class BackgroundStats:
    """Running first/second moments of normal residual vectors, global and per P x P cell."""

    def __init__(self, C: int, H: int, W: int, cell: int, device: str, alpha: float = 0.5):
        self.cell, self.H, self.W, self.alpha = cell, H, W, alpha
        self.gy, self.gx = math.ceil(H / cell), math.ceil(W / cell)
        cy = (torch.arange(H, device=device) // cell)[:, None]
        cx = (torch.arange(W, device=device) // cell)[None, :]
        self.cell_id = (cy * self.gx + cx).flatten()        # (H*W,)
        G = self.gy * self.gx
        self.n = torch.zeros(G, device=device)
        self.s1 = torch.zeros(G, C, device=device)
        self.s2 = torch.zeros(G, C, C, device=device)

    def update(self, d: torch.Tensor):
        v = d.flatten(1).T                                  # (H*W, C)
        self.n.index_add_(0, self.cell_id, torch.ones(len(v), device=v.device))
        self.s1.index_add_(0, self.cell_id, v)
        self.s2.index_add_(0, self.cell_id, v[:, :, None] * v[:, None, :])

    def finalize(self):
        n = self.n[:, None]
        mu = self.s1 / n
        cov = self.s2 / n[..., None] - mu[:, :, None] * mu[:, None, :]
        N = self.n.sum()
        g_mu = self.s1.sum(0) / N
        g_cov = self.s2.sum(0) / N - torch.outer(g_mu, g_mu)
        self.mu_g, self.prec_g = g_mu[None], _precision(g_cov)[None]
        self.mu_c, self.prec_c = mu, _precision(cov)
        a, diff = self.alpha, mu - g_mu
        cov_b = (1 - a) * cov + a * g_cov + a * (1 - a) * diff[:, :, None] * diff[:, None, :]
        self.mu_b, self.cov_b, self.prec_b = (1 - a) * mu + a * g_mu, cov_b, _precision(cov_b)
        self.logdet_b = -torch.linalg.slogdet(self.prec_b).logabsdet           # log|Sigma| per cell
        self.nu = None
        return self

    def bayes_delta(self, d: torch.Tensor) -> torch.Tensor:
        """Mahalanobis^2 of every pixel's residual under the FRX-Bayes cell statistics, (H*W,)."""
        v = d.flatten(1).T
        return _mahalanobis(v, self.mu_b[self.cell_id], self.prec_b[self.cell_id])

    def fit_nu(self, deltas: torch.Tensor, C: int, grid=(2.5, 3, 4, 5, 6, 8, 12, 20, 50, 200)):
        """ML degrees of freedom of a multivariate t whose covariance is the cell covariance:
        delta * nu / (nu - 2) / C ~ F(C, nu)."""
        from scipy.stats import f as fdist
        x = deltas.double().cpu().numpy()
        best = max(grid, key=lambda nu: np.sum(fdist.logpdf(x * nu / (nu - 2) / C, C, nu) + np.log(nu / (nu - 2) / C)))
        self.nu = float(best)
        return self.nu


class StateBackground:
    """Residual statistics per (scene state, cell). A machine in a fixed view passes through states (phases of
    its cycle) whose normal residuals differ; one background per cell mixes them and lets a busy phase hide
    or mimic an anomaly. Each state's cell statistics are the posterior mean under a normal-inverse-Wishart
    prior centred on the state-free FRX-Bayes statistics with `kappa0` pseudo-frames:
        w = n / (n + kappa0),  mu = w mu_k + (1 - w) mu_0,
        Sigma = w S_k + (1 - w) Sigma_0 + w (1 - w) (mu_k - mu_0)(mu_k - mu_0)^T,
    with n the number of training frames of the state, so rare states fall back to the global background."""

    def __init__(self, base: BackgroundStats, n_states: int, kappa0: float = 20.0):
        self.base, self.K, self.kappa0 = base, n_states, kappa0
        G, C = base.s1.shape
        dev = base.s1.device
        self.frames = torch.zeros(n_states, device=dev)
        self.n = torch.zeros(n_states, G, device=dev)
        self.s1 = torch.zeros(n_states, G, C, device=dev)
        self.s2 = torch.zeros(n_states, G, C, C, device=dev)

    def update(self, d: torch.Tensor, k: int):
        v, ids = d.flatten(1).T, self.base.cell_id
        self.frames[k] += 1
        self.n[k].index_add_(0, ids, torch.ones(len(v), device=v.device))
        self.s1[k].index_add_(0, ids, v)
        self.s2[k].index_add_(0, ids, v[:, :, None] * v[:, None, :])

    def finalize(self):
        """Call after base.finalize()."""
        n = self.n.clamp(min=1)[..., None]
        mu = self.s1 / n
        cov = self.s2 / n[..., None] - mu[..., :, None] * mu[..., None, :]
        mu0, cov0 = self.base.mu_b[None], self.base.cov_b[None]
        w = (self.frames / (self.frames + self.kappa0))[:, None, None]       # (K, 1, 1); kappa0 = inf -> 0
        diff = mu - mu0
        self.mu = w * mu + (1 - w) * mu0
        self.prec = _precision(w[..., None] * cov + (1 - w[..., None]) * cov0
                               + (w * (1 - w))[..., None] * diff[..., :, None] * diff[..., None, :])
        return self

    def delta(self, d: torch.Tensor, k: int) -> torch.Tensor:
        """Mahalanobis^2 of every pixel's residual under state k, (H, W)."""
        ids = self.base.cell_id
        return _mahalanobis(d.flatten(1).T, self.mu[k][ids], self.prec[k][ids]).reshape(d.shape[1:])

    def add(self, other: "StateBackground"):
        for name in ("frames", "n", "s1", "s2"):
            getattr(self, name).add_(getattr(other, name))
        return self

    def loglik(self, held: "StateBackground") -> float:
        """Gaussian log-likelihood (up to a constant) of the residuals summarised in `held`
        under this (finalized) model, from sufficient statistics only."""
        n, s1, s2 = held.n, held.s1, held.s2                                 # (K, G), (K, G, C), (K, G, C, C)
        P, mu = self.prec, self.mu
        quad = (P * s2).sum((-1, -2)) - 2 * torch.einsum("kgc,kgcd,kgd->kg", mu, P, s1) \
            + n * torch.einsum("kgc,kgcd,kgd->kg", mu, P, mu)
        return float((0.5 * n * torch.linalg.slogdet(P).logabsdet - 0.5 * quad).sum())


def select_kappa0(parts: list, template: BackgroundStats, grid=(1, 3, 10, 30, 100, 300, 1000, math.inf)):
    """Choose the prior strength of the state backgrounds by 2-fold cross-validated likelihood over the
    NORMAL training videos (`parts`: one StateBackground of sufficient statistics per video).
    kappa0 = inf means no state conditioning (FRX-Bayes), so a scene without recurring states keeps it."""
    folds = (parts[0::2], parts[1::2])
    total = {k0: 0.0 for k0 in grid}
    for fit_parts, held_parts in (folds, folds[::-1]):
        if not fit_parts or not held_parts:
            continue
        acc = StateBackground(template, parts[0].K)
        for q in fit_parts:
            acc.add(q)
        base = copy.copy(template)
        base.n, base.s1, base.s2 = acc.n.sum(0), acc.s1.sum(0), acc.s2.sum(0)
        base.finalize()
        held = StateBackground(template, parts[0].K)
        for q in held_parts:
            held.add(q)
        for k0 in grid:
            m = StateBackground(base, parts[0].K, k0)
            m.frames, m.n, m.s1, m.s2 = acc.frames, acc.n, acc.s1, acc.s2
            total[k0] += m.finalize().loglik(held)
    return max(total, key=total.get), total


def score_frame(d: torch.Tensor, stats: BackgroundStats, P: int, pool: int = 0) -> dict[str, float]:
    """All detector scores for one residual tensor d (C, H, W); `pool` = local-mean window (default P)."""
    C, H, W = d.shape
    v = d.flatten(1).T
    rem = (d ** 2).sum(0)
    frx_g = _mahalanobis(v, stats.mu_g, stats.prec_g).reshape(H, W)
    ids = stats.cell_id
    frx_c = _mahalanobis(v, stats.mu_c[ids], stats.prec_c[ids]).reshape(H, W)
    frx_b = _mahalanobis(v, stats.mu_b[ids], stats.prec_b[ids]).reshape(H, W)
    maps_t = {}
    if stats.nu is not None:
        nu = stats.nu
        # tail-robust transform of delta only: the per-cell 0.5 log|Sigma| term of the full NLL is left out,
        # because a location-dependent offset makes busy cells win the frame max (tested: AUC 0.70 on Ped2)
        maps_t["frx_t"] = 0.5 * (nu + C) * torch.log1p(frx_b / (nu - 2))
    w = 1.0 / (closing(rem) + 1e-6)
    w = (w / w.sum()).flatten()[:, None]
    mu_w = (w * v).sum(0, keepdim=True)
    z = v - mu_w
    cov_w = (w * z).T @ z
    wrx = _mahalanobis(v, mu_w, _precision(cov_w)[None]).reshape(H, W)
    maps = {"rem": rem, "frx_global": frx_g, "frx_cell": frx_c, "frx_bayes": frx_b, "wrx": wrx, **maps_t}
    return {k: frame_score(m, pool or P) for k, m in maps.items()}, maps
