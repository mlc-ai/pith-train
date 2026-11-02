"""
Loss-curve drift check for the PP4-EP8 A/B (run in a compute step). Per model x dtype, from the rank-0 step lines of
base1, feat1, base2: per step CE of each run, |b1-b2|, |b1-f|, |b2-f|, and feat minus the base mean. Summaries:
first step where any pair differs; max |diff| per pair; signed mean and slope (per 10 steps) of feat - base mean;
how many steps feat lies outside [min(b1,b2), max(b1,b2)] and on which side; the same for grad norm.
Writes loss_curves.md next to this file.
"""

import os
import re
import statistics

K = os.path.dirname(os.path.abspath(__file__))
STEP = re.compile(r"step (\d+)/\d+ \| step-time [0-9.]+ sec \| cross-entropy-loss ([0-9.]+) \| load-balance-loss ([0-9.]+) \| learning-rate \S+ \| gradient-norm ([0-9.]+)")
MODELS = ["qwen3.5-35b-a3b", "qwen3-30b-a3b", "gpt-oss-20b", "gpt-oss-120b"]


def load(name):
    out = {}
    for line in open(os.path.join(K, "runs", f"{name}.node0.log")):
        m = STEP.search(line)
        if m:
            out[int(m[1])] = (float(m[2]), float(m[4]))
    return out


def slope(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


lines = ["# PP4-EP8 loss curves: base1 / feat1 / base2 per step\n"]
summary = ["| model | dtype | first diff step b1-b2 / b1-f | max abs b1-b2 | max abs b1-f | max abs b2-f | mean(f - base mean) | slope per 10 steps | f outside base pair (above/below) | gnorm max rel b-b / b-f |", "|---|---|---|---|---|---|---|---|---|---|"]
for model in MODELS:
    for dtype in ("bf16", "fp8"):
        b1, f, b2 = (load(f"{model}-{dtype}-{a}") for a in ("base1", "feat1", "base2"))
        steps = sorted(set(b1) & set(f) & set(b2))
        ce = lambda r, s: r[s][0]
        gn = lambda r, s: r[s][1]
        first = lambda r: next((s for s in steps if ce(b1, s) != ce(r, s)), None)
        dbb = [abs(ce(b1, s) - ce(b2, s)) for s in steps]
        dbf = [abs(ce(b1, s) - ce(f, s)) for s in steps]
        db2f = [abs(ce(b2, s) - ce(f, s)) for s in steps]
        rel = [ce(f, s) - (ce(b1, s) + ce(b2, s)) / 2 for s in steps]
        above = sum(ce(f, s) > max(ce(b1, s), ce(b2, s)) for s in steps)
        below = sum(ce(f, s) < min(ce(b1, s), ce(b2, s)) for s in steps)
        gbb = max(abs(gn(b1, s) - gn(b2, s)) / gn(b1, s) for s in steps)
        gbf = max(abs(gn(b1, s) - gn(f, s)) / gn(b1, s) for s in steps)
        summary.append(f"| {model} | {dtype} | {first(b2)} / {first(f)} | {max(dbb):.4f} | {max(dbf):.4f} | {max(db2f):.4f} | {statistics.mean(rel):+.5f} | {10 * slope(steps, rel):+.5f} | {above}/{below} of {len(steps)} | {gbb:.3f} / {gbf:.3f} |")
        lines.append(f"\n## {model} {dtype}\n\n| step | base1 | feat1 | base2 | b1-b2 | f-b1 | f-b2 |\n|---|---|---|---|---|---|---|")
        for s in steps:
            lines.append(f"| {s} | {ce(b1, s):.4f} | {ce(f, s):.4f} | {ce(b2, s):.4f} | {ce(b1, s) - ce(b2, s):+.4f} | {ce(f, s) - ce(b1, s):+.4f} | {ce(f, s) - ce(b2, s):+.4f} |")
text = "\n".join(summary) + "\n\n" + "\n".join(lines) + "\n"
open(os.path.join(K, "loss_curves.md"), "w").write(text)
print("\n".join(summary))
