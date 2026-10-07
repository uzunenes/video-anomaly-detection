"""Extra feature channels scored with the same per-cell RX machinery as the autoencoder residual.

Hyperspectral analogy carried one step further: every image cell gets a feature vector ("spectrum") per
channel, the background statistics of each cell are learned from NORMAL video and shrunk towards the
scene-wide statistics (FRX-Bayes, aean/scoring.py), and a cell is scored by its Mahalanobis distance.

    flow : motion. Dense optical flow (DIS, Kroeger et al. 2016) between frames t-g and t (causal); per cell an
           8-bin histogram of flow magnitude over orientation plus the cell's maximum magnitude, log1p-scaled
           (cached), reduced to speed = (log1p mean |v|, log1p max |v|). Scored with a GMM-UBM: a universal
           Gaussian mixture over all cells, MAP-adapted per cell (weights and means, relevance factor r;
           Reynolds et al. 2000), score = negative log-likelihood. No motion threshold is needed: the static,
           walking, ... modes are mixture components, and a cell with little data stays close to the UBM.
    deep : appearance. Frozen ImageNet ResNet-18, layers 1-3 average-pooled to the cell grid, a fixed random
           subset of 100 of the 448 dimensions (PaDiM, Defard et al. 2021); no training on video.

Features are cached per video under <data>/feat_<name>/<split>/<video>.npy.
"""
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .scoring import BackgroundStats, _mahalanobis, local_mean_max


def _grid(H: int, W: int, cell: int) -> tuple[int, int]:
    return math.ceil(H / cell), math.ceil(W / cell)


def _cell_sum(a: np.ndarray, cell: int) -> np.ndarray:
    """(H, W) -> (gy, gx) mean over cell x cell blocks (zero padded at the border)."""
    H, W = a.shape
    gy, gx = _grid(H, W, cell)
    p = np.zeros((gy * cell, gx * cell), a.dtype)
    p[:H, :W] = a
    return p.reshape(gy, cell, gx, cell).mean((1, 3))


_RAFT = {}


def _raft_flows(frames: np.ndarray, gap: int, batch: int = 8) -> np.ndarray:
    """Dense flow t-gap -> t for every t >= gap with torchvision RAFT-large (C+T+S+K+H weights), (N-gap, H, W, 2)."""
    import torchvision.models.optical_flow as of
    dev = "cuda"
    if "m" not in _RAFT:
        _RAFT["m"] = of.raft_large(weights=of.Raft_Large_Weights.C_T_SKHT_V2).eval().to(dev)
    m = _RAFT["m"]
    N, H, W = frames.shape
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    out = []
    with torch.no_grad():
        for i in range(gap, N, batch):
            j = min(N, i + batch)
            a = torch.from_numpy(frames[i - gap:j - gap]).to(dev).float()[:, None].expand(-1, 3, -1, -1) / 127.5 - 1
            b = torch.from_numpy(frames[i:j]).to(dev).float()[:, None].expand(-1, 3, -1, -1) / 127.5 - 1
            a, b = F.pad(a, (0, pw, 0, ph), mode="replicate"), F.pad(b, (0, pw, 0, ph), mode="replicate")
            f = m(a, b, num_flow_updates=12)[-1][:, :, :H, :W]
            out.append(f.permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, H, W, 2), np.float32)


def flow_video(frames: np.ndarray, gap: int = 1, cell: int = 8, bins: int = 8, preset: str = "fast") -> np.ndarray:
    """uint8 (N, H, W) -> float16 (N, bins + 1, gy, gx) causal motion features.
    preset: DIS 'fast' / 'medium' (CPU) or 'raft' (RAFT-large on the GPU)."""
    import cv2
    cv2.setNumThreads(1)
    N, H, W = frames.shape
    if preset == "raft":
        flows = _raft_flows(frames, gap)
    else:
        dis = cv2.DISOpticalFlow_create({"fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
                                         "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}[preset])
    gy, gx = _grid(H, W, cell)
    out = np.zeros((N, bins + 1, gy, gx), np.float32)
    for t in range(gap, N):
        flow = flows[t - gap] if preset == "raft" else dis.calc(frames[t - gap], frames[t], None)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        b = np.minimum((ang / (2 * np.pi) * bins).astype(np.int64), bins - 1)
        for k in range(bins):
            out[t, k] = _cell_sum(mag * (b == k), cell)
        p = np.zeros((gy * cell, gx * cell), np.float32)
        p[:H, :W] = mag
        out[t, bins] = p.reshape(gy, cell, gx, cell).max((1, 3))
    out[:gap] = out[gap] if N > gap else 0                      # first frames: repeat the first flow
    return np.log1p(out).astype(np.float16)


class DeepFeatures:
    """Frozen ImageNet ResNet-18 trunk (layers 1-3) -> cell features (PaDiM-style random projection)."""

    def __init__(self, device: str, cell: int = 8, dim: int = 100, seed: int = 0):
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.net = m.eval().to(device)
        self.cell, self.device = cell, device
        self.idx = torch.randperm(448, generator=torch.Generator().manual_seed(seed))[:dim].to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

    @torch.no_grad()
    def __call__(self, frames: np.ndarray, batch: int = 64) -> np.ndarray:
        N, H, W = frames.shape
        g = _grid(H, W, self.cell)
        out = []
        for i in range(0, N, batch):
            x = torch.from_numpy(frames[i:i + batch]).to(self.device).float().div(255)
            x = (x[:, None].expand(-1, 3, -1, -1) - self.mean) / self.std
            n = self.net
            x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
            l1 = n.layer1(x)
            l2 = n.layer2(l1)
            l3 = n.layer3(l2)
            f = torch.cat([F.adaptive_avg_pool2d(l, g) for l in (l1, l2, l3)], 1)[:, self.idx]
            out.append(f.half().cpu().numpy())
        return np.concatenate(out)


def _fingerprint(frames: np.ndarray) -> str:
    """Cheap identity of a frame array: shape + CRC of a fixed subsample (guards the cache against reused names)."""
    import zlib
    sub = frames[:: max(1, len(frames) // 16)][:, ::7, ::7]
    return f"{frames.shape}:{zlib.crc32(np.ascontiguousarray(sub).tobytes())}"


def cached_features(data: Path, name: str, split: str, videos: dict, fn, workers: int = 0) -> dict:
    """Compute (or load) features of every video of a split; `fn(frames) -> array`.
    A cached file is reused only if the frames it was computed from have the same fingerprint."""
    import json
    d = data / f"feat_{name}" / split
    d.mkdir(parents=True, exist_ok=True)
    fp_file = d / "_fingerprints.json"
    fps = json.loads(fp_file.read_text()) if fp_file.exists() else {}
    now = {k: _fingerprint(v) for k, v in videos.items()}
    todo = [k for k in videos if not (d / f"{k}.npy").exists() or fps.get(k, now[k]) != now[k]]
    if todo:
        if workers > 1:
            with ProcessPoolExecutor(workers) as ex:
                for k, f in zip(todo, ex.map(fn, [videos[k] for k in todo])):
                    np.save(d / f"{k}.npy", f)
        else:
            for k in todo:
                np.save(d / f"{k}.npy", fn(videos[k]))
    fps.update(now)
    fp_file.write_text(json.dumps(fps))
    return {k: np.load(d / f"{k}.npy") for k in videos}


def speed(f: np.ndarray) -> np.ndarray:
    """Cached flow features (N, 9, gy, gx) -> (N, 2, gy, gx): log1p mean |v|, log1p max |v|."""
    f = f.astype(np.float32)
    return np.concatenate([np.log1p(np.expm1(f[:, :-1]).sum(1, keepdims=True)), f[:, -1:]], 1)


class DinoFeatures:
    """Frozen self-supervised DINOv2 ViT-S/14 patch tokens -> cell features (fixed random subset of dimensions).
    Frames are resized so that both sides are multiples of 14 (about the original size)."""

    def __init__(self, device: str, cell: int = 8, dim: int = 100, seed: int = 0):
        import os
        repo, weights = os.environ.get("DINOV2_REPO"), os.environ.get("DINOV2_WEIGHTS")
        if repo and weights:          # offline: local clone of facebookresearch/dinov2 + dinov2_vits14_pretrain.pth
            net = torch.hub.load(repo, "dinov2_vits14", source="local", pretrained=False)
            net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        else:
            net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.net = net.eval().to(device)
        self.cell, self.device = cell, device
        self.idx = torch.randperm(384, generator=torch.Generator().manual_seed(seed))[:dim].to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

    @torch.no_grad()
    def __call__(self, frames: np.ndarray, batch: int = 32) -> np.ndarray:
        N, H, W = frames.shape
        g = _grid(H, W, self.cell)
        h14, w14 = max(14, round(H / 14) * 14), max(14, round(W / 14) * 14)
        out = []
        for i in range(0, N, batch):
            x = torch.from_numpy(frames[i:i + batch]).to(self.device).float().div(255)[:, None]
            x = F.interpolate(x, size=(h14, w14), mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
            tok = self.net.forward_features((x - self.mean) / self.std)["x_norm_patchtokens"]   # (n, P, 384)
            tok = tok[..., self.idx].transpose(1, 2).reshape(len(x), -1, h14 // 14, w14 // 14)
            out.append(F.adaptive_avg_pool2d(tok, g).half().cpu().numpy())
        return np.concatenate(out)


class CellUBM:
    """Per-cell GMM obtained by MAP adaptation of a universal background mixture (GMM-UBM)."""

    def __init__(self, device: str, K: int = 5, r: float = 16.0, seed: int = 0):
        self.device, self.K, self.r, self.seed = device, K, r, seed

    def _logp(self, V: torch.Tensor, w: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """V (G, N, C); w (G|1, K); mu (G|1, K, C) -> log w_k N(v | mu_k, S_k), (G, N, K)."""
        z = V[:, :, None, :] - mu[:, None]
        m = torch.einsum("gnkc,kcd,gnkd->gnk", z, self.P, z)
        return torch.log(w[:, None]) + 0.5 * self.ldP - 0.5 * m - 0.5 * self.C * math.log(2 * math.pi)

    def fit(self, feats: list) -> "CellUBM":
        from sklearn.mixture import GaussianMixture
        X = torch.from_numpy(np.concatenate(feats)).to(self.device).float()
        N, C, gy, gx = X.shape
        V = X.permute(2, 3, 0, 1).reshape(gy * gx, N, C)
        flat = V.reshape(-1, C)
        g = torch.Generator(device=self.device).manual_seed(self.seed)
        sub = flat[torch.randperm(len(flat), generator=g, device=self.device)[:200000]]
        gm = GaussianMixture(self.K, covariance_type="full", reg_covar=1e-4, random_state=self.seed)
        gm.fit(sub.double().cpu().numpy())
        t = lambda a: torch.tensor(a, device=self.device, dtype=torch.float32)
        w, mu, self.P = t(gm.weights_), t(gm.means_), t(gm.precisions_)
        self.ldP, self.C, self.g = torch.linalg.slogdet(self.P).logabsdet, C, (gy, gx)
        ws, mus = [], []
        for i in range(0, len(V), 64):                            # MAP adaptation, cells in chunks
            Vi = V[i:i + 64]
            gam = torch.softmax(self._logp(Vi, w[None], mu[None]), -1)
            n_k = gam.sum(1)
            Ex = torch.einsum("gnk,gnc->gkc", gam, Vi) / n_k.clamp(min=1e-6)[..., None]
            a = n_k / (n_k + self.r)
            wi = a * n_k / N + (1 - a) * w[None]
            ws.append(wi / wi.sum(-1, keepdim=True))
            mus.append(a[..., None] * Ex + (1 - a[..., None]) * mu[None])
        self.w, self.mu = torch.cat(ws), torch.cat(mus)
        return self

    @torch.no_grad()
    def maps(self, f: np.ndarray, chunk: int = 256) -> torch.Tensor:
        out = []
        for i in range(0, len(f), chunk):
            v = torch.from_numpy(f[i:i + chunk]).to(self.device).float()
            n, C = v.shape[:2]
            V = v.permute(2, 3, 0, 1).reshape(-1, n, C)
            out.append((-torch.logsumexp(self._logp(V, self.w, self.mu), -1)).T.reshape(n, *self.g))
        return torch.cat(out)


class CellSkewT:
    """Per-cell multivariate skewed-t background (normal variance-mean mixture, as in Kayabol et al., IEEE GRSL 2021):
        x = mu + W gamma + sqrt(W) Sigma^(1/2) z,   z ~ N(0, I),   W ~ InvGamma(nu / 2, nu / 2).
    Heavy tails (nu) and a skew direction (gamma) fit the one-sided, bursty flow features better than a Gaussian.
    EM per cell (McNeil, Frey & Embrechts 2005, Alg. 3.14 with chi = nu, psi = 0): the posterior of W is
    GIG(-(nu + d) / 2, nu + Q, gamma' Sigma^-1 gamma), whose moments E[W], E[1/W] are Bessel-K ratios.
    nu is chosen on the pooled (scene-wide) data from a grid; the cell parameters are shrunk half-way towards the
    scene-wide fit, as in FRX-Bayes. Score = negative log-density without the per-cell log|Sigma| offset
    (as for frx_t: a location-dependent constant would let busy cells win the frame maximum)."""

    def __init__(self, device: str, iters: int = 20, alpha: float = 0.5, nus=(3.0, 5.0, 10.0, 20.0, 50.0)):
        self.device, self.iters, self.alpha, self.nus = device, iters, alpha, nus

    @staticmethod
    def _moments(Q, c, nu, d):
        """E[W], E[1/W] for GIG(-(nu + d) / 2, nu + Q, c); Q (G, N) and c (G, 1) as float64 numpy."""
        from scipy.special import kve
        lam, chi = -(nu + d) / 2.0, nu + Q
        psi = np.maximum(c, 1e-12)
        sq = np.sqrt(chi * psi)
        r = np.sqrt(chi / psi)
        k0 = kve(lam, sq)
        eta = r * kve(lam + 1, sq) / k0
        delta = kve(lam - 1, sq) / (r * k0)
        small = (c < 1e-8) | ~np.isfinite(eta) | ~np.isfinite(delta)      # no skew: inverse-gamma moments
        eta = np.where(small, chi / (nu + d - 2), eta)
        delta = np.where(small, (nu + d) / chi, delta)
        return eta, delta

    def _em(self, X: torch.Tensor, nu: float):
        """X (G, N, d) float64 -> mu (G, d), gamma (G, d), Sigma (G, d, d)."""
        G, N, d = X.shape
        mu = X.mean(1)
        Z = X - mu[:, None]
        Sig = Z.transpose(1, 2) @ Z / N
        gam = torch.zeros_like(mu)
        eye = torch.eye(d, dtype=X.dtype, device=X.device)
        for _ in range(self.iters):
            P = torch.linalg.inv(Sig + 1e-6 * eye)
            Z = X - mu[:, None]
            Q = torch.einsum("gnc,gcd,gnd->gn", Z, P, Z)
            c = torch.einsum("gc,gcd,gd->g", gam, P, gam)[:, None]
            eta, delta = self._moments(Q.cpu().numpy(), c.cpu().numpy(), nu, d)
            eta = torch.from_numpy(eta).to(X)
            delta = torch.from_numpy(delta).to(X)
            db, eb, xb = delta.mean(1), eta.mean(1), X.mean(1)
            gam = (delta[..., None] * (xb[:, None] - X)).mean(1) / (db * eb - 1).clamp(min=1e-6)[:, None]
            mu = ((delta[..., None] * X).mean(1) - gam) / db[:, None]
            Z = X - mu[:, None]
            Sig = torch.einsum("gn,gnc,gnd->gcd", delta, Z, Z) / N - eb[:, None, None] * gam[:, :, None] * gam[:, None, :]
            tr = Sig.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
            Sig = Sig + (1e-3 * tr / d + 1e-8) * eye                      # same ridge as FRX
        return mu, gam, Sig

    def _nll(self, X, mu, gam, P, nu):
        """negative log-density up to constants and without log|Sigma|, X (G, N, d)."""
        from scipy.special import kve
        d = X.shape[-1]
        Z = X - mu[:, None]
        Q = torch.einsum("gnc,gcd,gnd->gn", Z, P, Z)
        c = torch.einsum("gc,gcd,gd->g", gam, P, gam)[:, None]
        lin = torch.einsum("gnc,gcd,gd->gn", Z, P, gam)
        Qn, cn = Q.double().cpu().numpy(), c.double().cpu().numpy()
        s = np.sqrt((nu + Qn) * np.maximum(cn, 1e-12))
        logk = np.log(np.maximum(kve((nu + d) / 2.0, s), 1e-300)) - s
        skew = logk + (nu + d) / 2.0 * np.log(s)
        stud = np.zeros_like(Qn)                                          # c -> 0 limit: Student-t
        logf = np.where(cn < 1e-8, stud, skew) + lin.double().cpu().numpy() - (nu + d) / 2.0 * np.log1p(Qn / nu)
        return torch.from_numpy(-logf).to(self.device).float()

    def fit(self, feats: list) -> "CellSkewT":
        X = torch.from_numpy(np.concatenate(feats)).to(self.device).double()        # (N, d, gy, gx)
        N, d, gy, gx = X.shape
        self.g = (gy, gx)
        V = X.permute(2, 3, 0, 1).reshape(gy * gx, N, d)
        flat = V.reshape(1, -1, d)
        g = torch.Generator(device=self.device).manual_seed(0)
        sub = flat[:, torch.randperm(flat.shape[1], generator=g, device=self.device)[:200000]]
        best = None
        for nu in self.nus:                                   # nu by maximum likelihood on scene-wide data
            m, ga, Sg = self._em(sub, nu)
            P = torch.linalg.inv(Sg)
            c = float(torch.einsum("gc,gcd,gd->g", ga, P, ga)[0])
            lg = lambda v: float(torch.lgamma(torch.tensor(v, dtype=torch.float64)))
            if c < 1e-8:                                      # Student-t normaliser
                const = lg((nu + d) / 2.0) - lg(nu / 2.0) - d / 2.0 * np.log(np.pi * nu)
            else:                                             # skewed-t normaliser (McNeil et al. 2005, eq. 3.32)
                const = (1 - (nu + d) / 2.0) * np.log(2.0) - lg(nu / 2.0) - d / 2.0 * np.log(np.pi * nu)
            ll = -self._nll(sub, m, ga, P, nu).double().mean() - 0.5 * torch.linalg.slogdet(Sg).logabsdet[0] + const
            if best is None or ll > best[0]:
                best = (float(ll), nu, m, ga, Sg)
        self.nu, mg, gg, Sgg = best[1], best[2], best[3], best[4]
        mus, gams, Ps = [], [], []
        a = self.alpha
        for i in range(0, len(V), 128):                       # cells in chunks
            m, ga, Sg = self._em(V[i:i + 128], self.nu)
            m = (1 - a) * m + a * mg
            ga = (1 - a) * ga + a * gg
            Sg = (1 - a) * Sg + a * Sgg
            mus.append(m); gams.append(ga); Ps.append(torch.linalg.inv(Sg))
        self.mu, self.gam, self.P = torch.cat(mus), torch.cat(gams), torch.cat(Ps)
        return self

    @torch.no_grad()
    def maps(self, f: np.ndarray, chunk: int = 256) -> torch.Tensor:
        out = []
        for i in range(0, len(f), chunk):
            v = torch.from_numpy(f[i:i + chunk]).to(self.device).double()
            n, d = v.shape[:2]
            V = v.permute(2, 3, 0, 1).reshape(-1, n, d)
            out.append(self._nll(V, self.mu, self.gam, self.P, self.nu).T.reshape(n, *self.g))
        return torch.cat(out)


class CellRX:
    """FRX-Bayes on a cell grid: per-cell Gaussian of the feature vector, shrunk to the scene-wide one."""

    def __init__(self, C: int, gy: int, gx: int, device: str, alpha: float = 0.5):
        self.stats = BackgroundStats(C, gy, gx, 1, device, alpha)
        self.device, self.G = device, gy * gx

    def fit(self, feats: list) -> "CellRX":
        s = self.stats
        for f in feats:                                            # f: (N, C, gy, gx)
            v = torch.from_numpy(f).to(self.device).float()
            N, C = v.shape[:2]
            v = v.permute(0, 2, 3, 1).reshape(-1, C)
            ids = s.cell_id.repeat(N)
            s.n.index_add_(0, ids, torch.ones(len(v), device=self.device))
            s.s1.index_add_(0, ids, v)
            for i in range(0, len(v), 4096):                       # second moments in chunks
                vi = v[i:i + 4096]
                s.s2.index_add_(0, ids[i:i + 4096], vi[:, :, None] * vi[:, None, :])
        s.finalize()
        return self

    @torch.no_grad()
    def maps(self, f: np.ndarray, chunk: int = 256) -> torch.Tensor:
        """(N, C, gy, gx) -> Mahalanobis^2 maps (N, gy, gx)."""
        s, out = self.stats, []
        for i in range(0, len(f), chunk):
            v = torch.from_numpy(f[i:i + chunk]).to(self.device).float()
            n, C, gy, gx = v.shape
            z = v.permute(0, 2, 3, 1).reshape(n, -1, C) - s.mu_b[None]
            out.append(torch.einsum("ngc,gcd,ngd->ng", z, s.prec_b, z).reshape(n, gy, gx))
        return torch.cat(out)


def frame_scores_from_maps(m: torch.Tensor, H: int, W: int, pool: int) -> np.ndarray:
    """Cell maps (N, gy, gx) -> pixel maps -> max of the pool x pool local mean, as for the residual maps."""
    out = []
    cell_h, cell_w = math.ceil(H / m.shape[1]), math.ceil(W / m.shape[2])
    for i in range(0, len(m), 128):
        x = m[i:i + 128].repeat_interleave(cell_h, 1).repeat_interleave(cell_w, 2)[:, :H, :W]
        out.append(local_mean_max(x, pool).cpu().numpy())
    return np.concatenate(out)


def fit_channel(feats_train: dict, make, folds: int = 2, every: int = 3, H: int = 0, W: int = 0, pool: int = 48):
    """Full model on all normal videos + the scale (mean, std) of its frame scores on held-out normal videos
    (2-fold cross-fitting by video, so the fusion scale is not fitted on the scored frames).
    `make()` returns an unfitted CellRX / CellUBM. Also returns the held-out cell maps (N, gy, gx) as references
    for per-cell calibration."""
    names = list(feats_train)
    model = make().fit(list(feats_train.values()))
    ref, refmaps = [], []
    for k in range(folds):
        held = names[k::folds]
        fit = [n for n in names if n not in held]
        if not fit or not held:
            continue
        m = make().fit([feats_train[n] for n in fit])
        for n in held:
            mp = m.maps(feats_train[n][::every])
            ref.append(frame_scores_from_maps(mp, H, W, pool))
            refmaps.append(mp.half().cpu().numpy())
    ref = np.concatenate(ref)
    return model, (float(ref.mean()), float(ref.std()) + 1e-12), ref, np.concatenate(refmaps)
