"""Nonparametric Bayesian background model in the autoencoder's latent space.

After Arısoy (GTU PhD 2021, Ch. 2): the background is a Gaussian mixture whose number of components
is inferred with a Dirichlet-process prior, estimated on a PCA-reduced space. Here the "pixels" are
the encoder codes of NORMAL image patches: an industrial scene is multi-modal (a machine passes through
several states in its cycle), so normality is a mixture, not one Gaussian. A patch's anomaly score is its
negative log-likelihood under the mixture; patch scores are spread back onto pixels like the residual maps.
With a fixed camera the patch position is appended (scaled like an average PCA axis times `pos_weight`),
so the mixture learns which states are normal WHERE, the latent counterpart of per-cell background stats.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import BayesianGaussianMixture

from .scoring import _patch_pad


@torch.no_grad()
def patch_latents(ae, x: torch.Tensor, P: int, stride: int, chunk: int = 8192):
    """x: (C, H, W) -> encoder codes of all P x P patches (L, D) and what is needed to fold them back."""
    C, H, W = x.shape
    ph, pw = _patch_pad(H, P, stride), _patch_pad(W, P, stride)
    xp = F.pad(x[None], (0, pw, 0, ph), mode="reflect")
    cols = F.unfold(xp, P, stride=stride)
    L = cols.shape[-1]
    patches = cols[0].T.reshape(L, C, P, P)
    lat = torch.cat([ae.encoder(patches[i:i + chunk]).flatten(1) for i in range(0, L, chunk)])
    ny, nx = (xp.shape[-2] - P) // stride + 1, (xp.shape[-1] - P) // stride + 1
    iy = torch.arange(ny, device=x.device).repeat_interleave(nx).float() / max(ny - 1, 1)
    ix = torch.arange(nx, device=x.device).repeat(ny).float() / max(nx - 1, 1)
    pos = torch.stack([iy, ix], 1)                              # (L, 2) in [0, 1], unfold order
    return lat, (tuple(xp.shape[-2:]), H, W), pos


def patches_to_map(scores: torch.Tensor, meta, P: int, stride: int) -> torch.Tensor:
    """Average every pixel's covering patch scores -> (H, W) map."""
    size, H, W = meta
    cols = scores[None, None, :].expand(1, P * P, -1).contiguous()
    num = F.fold(cols, size, P, stride=stride)
    den = F.fold(torch.ones_like(cols), size, P, stride=stride)
    return (num / den)[0, 0, :H, :W]


class LatentDPGMM:
    """PCA (dim) + Dirichlet-process Gaussian mixture on normal patch codes; scoring runs on the GPU."""

    def __init__(self, dim: int = 16, max_components: int = 20, pos_weight: float = 2.0, seed: int = 0):
        self.dim, self.k, self.pos_weight, self.seed = dim, max_components, pos_weight, seed

    def _features(self, feats, pos):
        z = (feats - self.mu) @ self.V
        if self.pos_weight > 0 and pos is not None:
            z = torch.cat([z, (pos - 0.5) / 0.2887 * self.pos_scale * self.pos_weight], 1)
        return z

    def fit(self, feats: torch.Tensor, pos: torch.Tensor = None):
        self.mu = feats.mean(0, keepdim=True)
        _, _, V = torch.pca_lowrank(feats - self.mu, q=self.dim, center=False)
        self.V = V[:, :self.dim]
        self.pos_scale = float(((feats - self.mu) @ self.V).std(0).mean())
        Zt = self._features(feats, pos)
        self.d = Zt.shape[1]
        Z = Zt.double().cpu().numpy()
        g = BayesianGaussianMixture(n_components=self.k, covariance_type="full",
                                    weight_concentration_prior_type="dirichlet_process",
                                    max_iter=500, reg_covar=1e-4, random_state=self.seed).fit(Z)
        dev = feats.device
        self.n_active = int((g.weights_ > 0.01).sum())
        self.logw = torch.tensor(np.log(g.weights_ + 1e-12), device=dev, dtype=torch.float32)
        self.means = torch.tensor(g.means_, device=dev, dtype=torch.float32)
        self.chol = torch.tensor(g.precisions_cholesky_, device=dev, dtype=torch.float32)   # (K, d, d)
        self.logdet = torch.log(torch.diagonal(self.chol, dim1=-2, dim2=-1)).sum(-1)       # (K,)
        return self

    @torch.no_grad()
    def _logp(self, feats: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        """log w_k + log N(z | m_k, S_k) for every code and component, (N, K)."""
        z = self._features(feats, pos)                                        # (N, d)
        y = torch.einsum("nd,kde->nke", z, self.chol) - torch.einsum("kd,kde->ke", self.means, self.chol)[None]
        return -0.5 * (self.d * math.log(2 * math.pi) + (y ** 2).sum(-1)) + self.logdet[None] + self.logw[None]

    def score(self, feats: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        """Negative log-likelihood of each code under the fitted mixture (higher = more anomalous)."""
        return -torch.logsumexp(self._logp(feats, pos), dim=1)

    def assign(self, feats: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        """MAP component of each code, (N,)."""
        return self._logp(feats, pos).argmax(1)


def scene_descriptor(x: torch.Tensor, pool: int = 8) -> torch.Tensor:
    """Coarse view of a whole model input (C, H, W) -> 1-D vector, used to recognise the scene state
    (e.g. the phase of a machine cycle). For clips it carries both pose and motion."""
    return F.avg_pool2d(x[None], pool).flatten()
