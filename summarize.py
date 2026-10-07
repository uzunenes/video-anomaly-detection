"""Collect runs/*/eval_*/results.json into one table (stdout + runs/summary.md). Report text is Turkish."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
rows = []
for f in sorted(root.glob("*/eval_*/results.json")):
    r = json.loads(f.read_text())
    run = f.parent.parent.name + ("" if f.parent.name.count("_") < 2 else "/" + f.parent.name.split("_", 2)[2])
    for key, m in r["auc"].items():
        det, post = key.split("/")
        rows.append((run, r["run"]["variant"], "hayır" if r["run"]["no_adv"] else "evet", det, post,
                     m["auc"], m["eer"], r["realtime"]["ms_per_frame"]))

lines = ["| deney | varyant | çekişmeli | dedektör | son işlem | AUC | EER | ms/kare |", "|---|---|---|---|---|---|---|---|"]
lines += [f"| {a} | {b} | {c} | {d} | {e} | {f:.3f} | {g:.3f} | {h:.1f} |" for a, b, c, d, e, f, g, h in rows]

best = {}
for row in rows:
    k = (row[0], row[4])
    if k not in best or row[5] > best[k][5]:
        best[k] = row
lines += ["", "## Her deney ve son işlem için en iyi dedektör", "",
          "| deney | son işlem | dedektör | AUC |", "|---|---|---|---|"]
lines += [f"| {r[0]} | {r[4]} | {r[3]} | {r[5]:.3f} |" for r in sorted(best.values())]

text = "\n".join(lines)
(root / "summary.md").write_text(text + "\n")
print(text)
