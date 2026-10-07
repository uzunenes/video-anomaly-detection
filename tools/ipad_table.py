"""Per-scene IPAD frame AUC for every detector (paper protocol: AUC per scene, then the mean).

    python tools/ipad_table.py [runs_dir] [variant]
"""
import glob
import json
import sys

import numpy as np

runs = sys.argv[1] if len(sys.argv) > 1 else "runs"
variant = sys.argv[2] if len(sys.argv) > 2 else "2d"
tag = sys.argv[3] if len(sys.argv) > 3 else ""             # evaluation tag, e.g. "state"
dets = ["rem", "frx_global", "frx_cell", "frx_bayes", "frx_t", "wrx", "bgmm", "fused", "frx_state"]
rows = {}
for f in sorted(glob.glob(f"{runs}/ipad_*_{variant}/eval_ipad_*{'_' + tag if tag else ''}/results.json")):
    if f.split("/")[-2].count("_") != (3 if tag else 2):   # untagged run, or exactly this tag
        continue
    rows[f.split("/")[-3].split("_")[1]] = json.loads(open(f).read())

for post in ("raw", "median9", "median9+videonorm"):
    print(f"\n### {post} (frame AUC per scene, {len(rows)} scenes)")
    print("scene " + "".join(f"{d:>11}" for d in dets))
    acc = {d: [] for d in dets}
    for sc, r in rows.items():
        vals = [r["auc"].get(f"{d}/{post}", {}).get("auc", np.nan) for d in dets]
        for d, v in zip(dets, vals):
            acc[d].append(v)
        print(f"{sc:<6}" + "".join(f"{v:>11.3f}" for v in vals))
    print("MEAN  " + "".join(f"{np.nanmean(acc[d]):>11.3f}" for d in dets))
    real = [i for i, sc in enumerate(rows) if sc.startswith("R")]
    synth = [i for i, sc in enumerate(rows) if sc.startswith("S")]
    for name, idx in (("real", real), ("synth", synth)):
        if idx:
            print(f"{name:<6}" + "".join(f"{np.nanmean([acc[d][i] for i in idx]):>11.3f}" for d in dets))

print("\nstudent-t nu:", {sc: r.get("student_t_nu") for sc, r in rows.items()})
print("dp-gmm components:", {sc: r.get("latent", {}).get("dpgmm_active_components") for sc, r in rows.items()})
