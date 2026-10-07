"""Cached UCSD videos on the GPU and vectorised training-sample extraction."""
import json
from pathlib import Path

import numpy as np
import torch


def to_unit(x: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> float [-1, 1] (the AEAN decoder ends in tanh)."""
    return x.float() / 127.5 - 1.0


def load_split(root: Path, split: str) -> dict[str, np.ndarray]:
    return {p.stem: np.load(p) for p in sorted((root / split).glob("*.npy"))}


def load_gt(root: Path) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.uint8) for k, v in json.loads((root / "gt.json").read_text()).items()}


class ClipSampler:
    """Draws random training inputs from normal videos.

    All frames live in one uint8 tensor; a sample is (frame index t, top-left y, x).
    The clip is frames t-(T-1)*s, ..., t-s, t of the same video (causal, as in a live camera);
    s = `tstride` > 1 widens the time window without more input channels.
    `motion_frac` of each batch is taken from the most dynamic candidates,
    because most of a static-camera frame is unchanging background.
    """

    def __init__(self, videos: dict[str, np.ndarray], variant: str, T: int, patch: int,
                 device: str, motion_frac: float = 0.5, tstride: int = 1):
        self.variant, self.T, self.P, self.device = variant, T, patch, device
        self.motion_frac = motion_frac
        frames, valid, off = [], [], 0
        for v in videos.values():
            frames.append(torch.from_numpy(v))
            if len(v) > (T - 1) * tstride:                  # a video shorter than one clip gives no samples
                valid.append(torch.arange(off + (T - 1) * tstride, off + len(v)))
            off += len(v)
        self.frames = torch.cat(frames).to(device)          # (N, H, W) uint8
        self.valid = torch.cat(valid).to(device)            # frame indices with a full causal clip
        _, self.H, self.W = self.frames.shape
        self.dt = torch.arange(-T + 1, 1, device=device) * tstride

    def _clips(self, n: int, size: int):
        t = self.valid[torch.randint(len(self.valid), (n,), device=self.device)]
        y = torch.randint(self.H - size + 1, (n,), device=self.device)
        x = torch.randint(self.W - size + 1, (n,), device=self.device)
        r = torch.arange(size, device=self.device)
        tt = t[:, None, None, None] + self.dt[None, :, None, None]
        yy = y[:, None, None, None] + r[None, None, :, None]
        xx = x[:, None, None, None] + r[None, None, None, :]
        return to_unit(self.frames[tt, yy, xx])              # (n, T, size, size)

    def sample(self, bs: int) -> torch.Tensor:
        n_cand = 2 * bs
        size = 1 if self.variant == "1d" else self.P
        clips = self._clips(n_cand, size)
        motion = clips.std(dim=1).flatten(1).mean(1)          # temporal variation per candidate
        k = int(bs * self.motion_frac)
        top = motion.topk(k).indices
        rest = torch.randperm(n_cand, device=self.device)
        rest = rest[~torch.isin(rest, top)][: bs - k]
        clips = clips[torch.cat([top, rest])]
        if self.variant == "1d":
            return clips[:, :, 0, 0].unsqueeze(1)             # (bs, 1, T) pixel temporal profile
        if self.variant == "2d":
            return clips[:, -1:]                              # (bs, 1, P, P) current frame only
        return clips                                          # (bs, T, P, P)
