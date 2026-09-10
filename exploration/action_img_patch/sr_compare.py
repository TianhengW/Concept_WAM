"""Fair SR comparison for ap_base vs ap_patch evals.

Only (task, phase) units finished by BOTH sides enter the averages, so
partial progress cannot bias either way.
"""
import re

files = {
    "base": "slurm_logs/eval_ap_base_h100_85969.out",
    "patch": "slurm_logs/eval_ap_patch_h100_85970.out",
}
res = {}
for k, fn in files.items():
    txt = re.sub(r"\x1b\[[0-9;]*m", "", open(fn, errors="ignore").read())
    d = {}
    for m in re.finditer(r"done task=([a-z_0-9]+) phase=(clean|random) .*?success_rate=([\d.]+)", txt):
        d[(m.group(1), m.group(2))] = float(m.group(3))
    res[k] = d

common = sorted(set(res["base"]) & set(res["patch"]))
only_b = len(set(res["base"]) - set(res["patch"]))
only_p = len(set(res["patch"]) - set(res["base"]))
print("common units: %d  (base-only %d, patch-only %d)" % (len(common), only_b, only_p))

for ph in ("clean", "random"):
    cs = [(t, res["base"][(t, p)], res["patch"][(t, p)]) for (t, p) in common if p == ph]
    if not cs:
        continue
    b = 100 * sum(x[1] for x in cs) / len(cs)
    p = 100 * sum(x[2] for x in cs) / len(cs)
    print("%7s: base %5.1f  patch %5.1f  (n=%d)" % (ph, b, p, len(cs)))

allc = [(res["base"][c], res["patch"][c]) for c in common]
b = 100 * sum(x[0] for x in allc) / len(allc)
p = 100 * sum(x[1] for x in allc) / len(allc)
win = sum(1 for x in allc if x[1] > x[0])
lose = sum(1 for x in allc if x[1] < x[0])
tie = sum(1 for x in allc if x[1] == x[0])
print("  total: base %5.1f  patch %5.1f  | per-unit patch W%d/T%d/L%d" % (b, p, win, tie, lose))

print()
print("units with |diff| >= 30pp:")
for (t, ph) in common:
    d = res["patch"][(t, ph)] - res["base"][(t, ph)]
    if abs(d) >= 0.3:
        print("  %-28s %-6s base %4.0f  patch %4.0f  (%+.0f)"
              % (t, ph, 100 * res["base"][(t, ph)], 100 * res["patch"][(t, ph)], 100 * d))
