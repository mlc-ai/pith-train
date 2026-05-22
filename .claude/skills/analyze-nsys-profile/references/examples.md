# Worked examples

Each example shows the question, the script chain to answer it, and what to look for in the output. Adapt these — they are recipes, not templates.

Setup for all examples:

```bash
source .venv/bin/activate
SQLITE=workspace/capture-nsys-profile/pithtrain_node0.sqlite
SKILL=.claude/skills/analyze-nsys-profile/scripts
nsys export --type=sqlite --force-overwrite=true --output=$SQLITE \
    workspace/capture-nsys-profile/pithtrain_node0.nsys-rep
```

## Q1: How much EP all-to-all is hidden by compute, per rank?

```bash
# Identify each rank's EP comm stream.
python $SKILL/classify_streams.py $SQLITE --text | head -40

# Get per-stream overlap stats in the median chunk window.
python $SKILL/compute_overlap.py $SQLITE --text
```

For each rank, find the stream with `purpose: ep_a2a` in the classify-streams output, then look up that stream's row in the `per_comm_stream` array of the overlap output. `hidden_pct` is the answer.

Report the **median across ranks** as the headline; flag any rank whose value is much lower than the median.

**Gotchas:**
- EP a2a may live on more than one stream — take the union when summing.
- A PP P2P stream's `hidden_pct` is expected to be low. Don't include it in the EP analysis.

## Q2: Which EP phase has the worst overlap?

```bash
python $SKILL/summarize_stages.py $SQLITE --view exposed --text
```

The `aggregate` field bucketed by enclosing stage (`stage2_f`, `stage2_b`, `stage4_f`, `stage4_b`) ranks the four EP phases by total exposed time across all ranks. The top of that list is the answer.

The `per_rank.by_stage` field lets you check whether the pattern is consistent across ranks or rank-specific.

## Q3: Which stage has the largest idle gaps between kernels?

```bash
python $SKILL/summarize_stages.py $SQLITE --view gaps --text
```

Same shape as Q2 but on the compute stream — bucketing inter-kernel gaps by the stage of the kernel that precedes each gap. The top of `aggregate` is the answer.

## Q4: Is rank N's long NCCL kernel data movement or a straggler wait?

```bash
# Find the kernel of interest from the exposed list.
python $SKILL/compute_overlap.py $SQLITE --rank 0 --exposed-top 5

# Run cross-rank comparison on its start time.
python $SKILL/compare_kernel.py $SQLITE \
    --rank 0 \
    --kernel-start-ns <start_ns from previous output>
```

The `verdict` field is the human-readable answer (data / straggler / mild skew). The `partners` array shows each rank's matched kernel and its duration for manual inspection.

## Q5: Are the PP stages balanced?

```bash
python $SKILL/compute_overlap.py $SQLITE | python -c "
import json, sys
data = json.load(sys.stdin)
for r in data:
    print(f'rank {r[\"rank\"]:>2}: compute = {r[\"compute_ns_in_window\"]/1e6:7.2f} ms')
"
```

Group ranks by PP stage (PP=2 → ranks 0-3 vs 4-7). Compare the per-PP-stage mean compute time. A consistent gap implies an uneven layer partition.

## Q6: What are the top exposed comm kernels across ALL ranks?

`compute_overlap.py --exposed-top N` returns top-N per rank, not global. To get
one cross-rank sorted list, merge with `jq`:

```bash
python $SKILL/compute_overlap.py $SQLITE --exposed-top 100 | jq '
    [ .[] | (.rank as $r | .top_exposed[] | . + {rank: $r}) ]
    | sort_by(-.exposed_ns)
    | .[0:10]
'
```

Each emitted record now carries the originating `rank` alongside the
existing fields (`name`, `stream_id`, `start_ns`, `duration_ns`, `exposed_ns`,
`enclosing_nvtx`). Useful when you want to dispatch sub-agents on the K longest
exposed kernels regardless of which rank they came from.

## Composing with sub-agents

When `compute_overlap.py --exposed-top N` returns multiple interesting kernels, hand each one to a sub-agent using the [subagent template](subagent-template.md). The sub-agent runs `compare_kernel.py` and reports back. Fan out in parallel; aggregate the verdicts in the parent.
