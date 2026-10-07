"""AEAN (autoencoding adversarial network), after Arısoy, GTU PhD thesis 2021, Tables 3.1-3.3.

Encoder : conv 9 -> 64, conv 5 -> 128, conv 3 -> 256   (BN + LeakyReLU, 'valid' padding)
Decoder : deconv 3 -> 128, deconv 5 -> 64, deconv 9 -> C (tanh)
Discrim.: same three convs -> global pool -> linear 256 -> 1

With 16x16 inputs the valid convolutions give 16 -> 8 -> 4 -> 2 and the decoder mirrors it back.

Hyperspectral -> CCTV mapping used here:
    '1d' : spectral vector (L)          -> temporal profile of one pixel, input (B, 1, T)
    '2d' : one-band spatial patch       -> one-frame patch, input (B, 1, P, P)
    '3d' : hypercube P x P x L          -> spatio-temporal cube, T frames as channels, input (B, T, P, P)
    'pred': same network, but the last frame is hidden: input the T-1 previous frames (B, T-1, P, P), output the
            last frame (B, 1, P, P). A temporal blind spot: an anomaly in frame t cannot be copied through, only
            predicted from the past (cf. future-frame prediction, Liu et al. 2018; masking in citing HAD papers).
"""
import torch
import torch.nn as nn

KERNELS = (9, 5, 3)
WIDTHS = (64, 128, 256)


def _layers(dim: int):
    conv = nn.Conv1d if dim == 1 else nn.Conv2d
    deconv = nn.ConvTranspose1d if dim == 1 else nn.ConvTranspose2d
    bn = nn.BatchNorm1d if dim == 1 else nn.BatchNorm2d
    return conv, deconv, bn


def _encoder(dim: int, in_ch: int) -> nn.Sequential:
    conv, _, bn = _layers(dim)
    layers, c = [], in_ch
    for k, w in zip(KERNELS, WIDTHS):
        layers += [conv(c, w, k), bn(w), nn.LeakyReLU(0.2, inplace=True)]
        c = w
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    def __init__(self, dim: int, in_ch: int, out_ch: int = 0):
        super().__init__()
        out_ch = out_ch or in_ch
        _, deconv, bn = _layers(dim)
        self.encoder = _encoder(dim, in_ch)
        dec, c = [], WIDTHS[-1]
        for k, w in zip(KERNELS[::-1][:-1], WIDTHS[::-1][1:]):
            dec += [deconv(c, w, k), bn(w), nn.LeakyReLU(0.2, inplace=True)]
            c = w
        dec += [deconv(c, out_ch, KERNELS[0]), nn.Tanh()]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Discriminator(nn.Module):
    def __init__(self, dim: int, in_ch: int):
        super().__init__()
        pool = nn.AdaptiveAvgPool1d(1) if dim == 1 else nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(_encoder(dim, in_ch), pool, nn.Flatten(), nn.Linear(WIDTHS[-1], 1))

    def forward(self, x):  # logits
        return self.net(x)


def build(variant: str, T: int):
    """Return (autoencoder, discriminator, input channels) for a variant."""
    if variant == "pred":
        return Autoencoder(2, T - 1, 1), Discriminator(2, 1), T - 1
    dim, in_ch = {"1d": (1, 1), "2d": (2, 1), "3d": (2, T)}[variant]
    return Autoencoder(dim, in_ch), Discriminator(dim, in_ch), in_ch


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for v in ("1d", "2d", "3d"):
        ae, d, c = build(v, T=16)
        x = torch.randn(4, c, 16) if v == "1d" else torch.randn(4, c, 16, 16)
        y = ae(x)
        assert y.shape == x.shape, (v, y.shape)
        print(f"{v}: in {tuple(x.shape)} | AE {n_params(ae) / 1e6:.2f} M params | D {n_params(d) / 1e6:.2f} M | D(x) {tuple(d(x).shape)}")
