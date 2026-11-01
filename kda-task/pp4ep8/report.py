"""
PP4-EP8 A/B report from runs/<model>-<dtype>-<arm>.node0.log (rank-0 step lines). Run in a compute step.
Per run: median step-time and tokens/s over steps 11-30, peak-gpu-memory (max over ranks, as logged, max over steps).
Per model x dtype: speed = base1 / feat1 median step time (base1 / base2 beside it as noise), peak feat vs base1, and
the loss: max over steps 1-32 of |CE base1 - CE feat1| against the same for base1 vs base2 (flag when feat's exceeds
the base-vs-base one by more than 2x and 0.01). Writes report.md and report.json next to this file.
"""

import json
import os
import re
import statistics

K = os.path.dirname(os.path.abspath(__file__))
STEP = re.compile(r"step (\d+)/\d+ \| step-time ([0-9.]+) sec \| cross-entropy-loss ([0-9.]+) .*tokens-per-second ([0-9,]+) \| peak-gpu-memory ([0-9.]+) GB")
MODELS = ["qwen3.5-35b-a3b", "qwen3-30b-a3b", "gpt-oss-20b", "gpt-oss-120b"]


def run(name):
    path = os.path.join(K, "runs", f"{name}.node0.log")
    if not os.path.exists(os.path.join(K, "runs", f"{name}.done")):
        return None
    steps = {}
    for line in open(path):
        m = STEP.search(line)
        if m:
            steps[int(m[1])] = dict(t=float(m[2]), ce=float(m[3]), tps=float(m[4].replace(",", "")), peak=float(m[5]))
    win = [steps[s] for s in range(11, 31) if s in steps]
    return dict(steps=steps, median_s=statistics.median(x["t"] for x in win), tps=statistics.median(x["tps"] for x in win), peak=max(x["peak"] for x in steps.values()), n=len(win))


def maxdiff(a, b):
    common = sorted(set(a["steps"]) & set(b["steps"]))
    return max(abs(a["steps"][s]["ce"] - b["steps"][s]["ce"]) for s in common), len(common)


rows, out = [], []
for model in MODELS:
    for dtype in ("bf16", "fp8"):
        r = {arm: run(f"{model}-{dtype}-{arm}") for arm in ("base1", "feat1", "base2")}
        if not (r["base1"] and r["feat1"]):
            rows.append(f"| {model} | {dtype} | incomplete ({', '.join(a for a, v in r.items() if not v)} missing) |||||||")
            continue
        b, f, b2 = r["base1"], r["feat1"], r["base2"]
        d_bf, n = maxdiff(b, f)
        d_bb = maxdiff(b, b2)[0] if b2 else None
        flag = d_bb is not None and d_bf > max(2 * d_bb, d_bb + 0.01)
        rec = dict(model=model, dtype=dtype, base_s=b["median_s"], feat_s=f["median_s"], base2_s=b2 and b2["median_s"], speed=b["median_s"] / f["median_s"], noise=b2 and b["median_s"] / b2["median_s"], base_tps=b["tps"], feat_tps=f["tps"], base_peak=b["peak"], feat_peak=f["peak"], loss_maxdiff_bf=d_bf, loss_maxdiff_bb=d_bb, loss_steps=n, loss_flag=flag)
        out.append(rec)
        noise = f"{rec['noise']:.3f}" if b2 else "n/a"
        bb = f"{d_bb:.4f}" if d_bb is not None else "n/a"
        rows.append(f"| {model} | {dtype} | {b['median_s']:.3f} | {f['median_s']:.3f} | **{rec['speed']:.3f}** ({noise}) | {b['tps']:,.0f} / {f['tps']:,.0f} | {b['peak']:.2f} / {f['peak']:.2f} ({100 * (f['peak'] / b['peak'] - 1):+.1f}%) | {d_bf:.4f} ({bb}){' FLAG' if flag else ''} |")
head = "| model | dtype | base s/step | feat s/step | speed base/feat (base1/base2) | tok/s base / feat | peak GB base / feat | loss max abs diff b-f (b-b) |\n|---|---|---|---|---|---|---|---|"
text = "# PP4-EP8 A/B, 4 nodes x 8 H200: base kda/sm90-base b723719 vs feat haok/kda-sm90 1bc9ee6\n\nSteps 11-30 medians of 32-step runs; order base1, feat1, base2 per row; peak = max over ranks and steps.\n\n" + head + "\n" + "\n".join(rows) + "\n"
open(os.path.join(K, "report.md"), "w").write(text)
json.dump(out, open(os.path.join(K, "report.json"), "w"), indent=1)
print(text)
