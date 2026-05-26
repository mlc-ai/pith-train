# Worked examples

Each example shows the question, the script chain to answer it, and what to look for in the
output. Adapt these — they are recipes, not templates.

Setup for all examples:

```bash
source .venv/bin/activate
SQLITE=workspace/capture-nsys-profile/pithtrain_node0.sqlite
SKILL=.claude/skills/analyze-nsys-profile/scripts
nsys export --type=sqlite --force-overwrite=true --output=$SQLITE workspace/capture-nsys-profile/pithtrain_node0.nsys-rep
```

## Q1: How well is each EP phase overlapped with compute?

```bash
python $SKILL/compute_overlap.py $SQLITE
```

One row per `(rank, stage)`. Look at the `overlap` column for `stage2_*` (dispatch) and `stage4_*`
(combine). `overlap_min` / `overlap_max` tell you whether a stage is uniformly bad or has outliers.

Report the **median `overlap` across ranks** for the worst stage as the headline; flag any rank
whose value is much lower than the median.

## Q2: Which EP phase has the worst overlap?

Same script as Q1. Group the rows by `stage`, take the median (or min) of `overlap` across the
8 ranks for each stage. Typical pattern: `stage2_*` (dispatch) is well-hidden; `stage4_f`
(combine forward) is the systemic problem.

## Q3: Are the PP stages balanced?

```bash
python $SKILL/classify_streams.py $SQLITE | grep compute
```

Each rank's compute-stream row carries `comp_ns` — total compute time in the steady-state
window. Group ranks by their PP coordinate (use `show_setup.py` to recover pp_rank from each
pid). If pp=0 ranks consistently have very different `comp_ns` from pp=1 ranks, the layer
partition is uneven.

## Q4: Which (rank, stage) has the highest exposed comm?

```bash
python $SKILL/compute_overlap.py $SQLITE | sort -k 3 -n -r | head -10
```

`exposed_ns` is column 3 (after `pid` and `stage`). The top rows are the biggest contributors
to un-hidden comm wall-clock cost across the trace.
