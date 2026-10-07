"""Train an AEAN variant on the normal training videos of one UCSD subset.

Objective (Arısoy 2021, eq. 3.6):  min_A max_D  L_adv(A, D) + lambda * ||x - A(x)||_1,  lambda = 10.
`--no-adv` drops the discriminator (plain autoencoder) for the ablation.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from aean.data import ClipSampler, load_split
from aean.models import build, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="cache/<ped>")
    ap.add_argument("--variant", choices=["1d", "2d", "3d", "pred"], required=True)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--tstride", type=int, default=1, help="frame step inside a clip (1d/3d)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--motion-frac", type=float, default=0.5)
    ap.add_argument("--no-adv", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    sampler = ClipSampler(load_split(Path(args.data), "train"), args.variant, args.T, args.patch, dev,
                           args.motion_frac, args.tstride)
    ae, disc, _ = build(args.variant, args.T)
    ae, disc = ae.to(dev), disc.to(dev)
    opt_a = torch.optim.Adam(ae.parameters(), args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), args.lr, betas=(0.5, 0.999))
    bce = F.binary_cross_entropy_with_logits
    print(f"{args.variant} | AE {n_params(ae) / 1e6:.2f} M params | frames {len(sampler.frames)} | adv={not args.no_adv}")

    log, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        x = sampler.sample(args.bs)
        if args.variant == "pred":                  # hide the last frame and predict it from the previous ones
            x, inp = x[:, -1:], x[:, :-1]
            xr = ae(inp)
        else:
            xr = ae(x)
        if not args.no_adv:
            real, fake = disc(x), disc(xr.detach())
            d_loss = bce(real, torch.ones_like(real)) + bce(fake, torch.zeros_like(fake))
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
        rec = (xr - x).abs().mean()
        a_loss = args.lam * rec
        if not args.no_adv:
            g = disc(xr)
            adv = bce(g, torch.ones_like(g))
            a_loss = a_loss + adv
        opt_a.zero_grad(set_to_none=True)
        a_loss.backward()
        opt_a.step()

        if step % 500 == 0 or step == args.steps:
            row = {"step": step, "rec_l1": rec.item(), "sec": round(time.time() - t0, 1)}
            if not args.no_adv:
                row |= {"d_loss": d_loss.item(), "g_adv": adv.item()}
            log.append(row)
            print(json.dumps(row), flush=True)

    torch.save({"ae": ae.state_dict(), "disc": disc.state_dict()}, out / "model.pt")
    (out / "train_log.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
