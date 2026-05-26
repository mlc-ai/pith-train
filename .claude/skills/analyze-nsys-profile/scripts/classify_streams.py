"""
Classify each rank's CUDA streams in the analysis window.

A stream is a compute stream if any of its kernels is non-NCCL, else a communication stream.
Communication streams get a purpose label that every one of their kernels' enclosing NVTX ranges
agrees on:

- ep_a2a  -- stage2_* / stage4_* (EP dispatch / combine)
- cp_ring -- stage1_* (CP ring attention)
- pp_p2p  -- pipeline send/recv
- mixed   -- kernels disagree on the label
- unknown -- enclosing NVTX missing or no marker matched

Compute streams aren't purpose-labelled: PithTrain shares one stream across all stages, so any
per-kernel stage label would just be sampling noise.
"""

import argparse
import sqlite3

from common import innermost_nvtx, kernels_in_window, print_table, streams_in_window
from find_window import extract_window
from show_setup import extract_setup

# (label, needles): a kernel gets the label when its enclosing NVTX range contains any needle as a
# substring. stage1_* also matches DualPipeV's fused stage5+stage1 ranges, which is fine since
# stage5 is pure compute.
PURPOSE_MARKERS = (
    ("ep_a2a", ("stage2_f", "stage2_b", "stage4_f", "stage4_b")),
    ("cp_ring", ("stage1_f", "stage1_b")),
    ("pp_p2p", ("pipeline send/recv",)),
)


def classify_stream(nvtx_names: list[str | None]) -> str:
    """
    Classify a comm stream's purpose by the enclosing-NVTX category across its kernels. Returns
    a single category if every kernel agrees, "mixed" otherwise. Empty input -> "unknown".
    """
    if not nvtx_names:
        return "unknown"
    labels = []
    for name in nvtx_names:
        if name is None:
            labels.append("unknown")
            continue
        for label, needles in PURPOSE_MARKERS:
            if any(n in name for n in needles):
                labels.append(label)
                break
        else:
            labels.append("unknown")
    return labels[0] if all(x == labels[0] for x in labels) else "mixed"


def extract_streams(con: sqlite3.Connection, pid: int, start: int, end: int) -> list[dict]:
    """
    One record per stream active in [start, end] for one rank: counts + a purpose for comm
    streams. NVTX-context lookups use each kernel's CPU-side launch_start (not GPU start),
    because async kernels execute after the launching range has ended.
    """
    stream_rows = streams_in_window(con, pid, start, end)
    out = []
    for row in stream_rows:
        if row["n_comp"] > 0:
            out.append({**row, "purpose": "compute"})
            continue
        kernels = kernels_in_window(con, pid, start, end, stream=row["stream"])
        timestamps = [k["launch_start"] or k["start"] for k in kernels]
        enclosing_nvtx = innermost_nvtx(con, pid, timestamps)
        out.append({**row, "purpose": classify_stream(enclosing_nvtx)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("trace", help="Path to the nsys SQLite trace export.")
    parser.add_argument("--rank", type=int, default=None, help="Scope to one global rank.")
    parser.add_argument("--start", type=int, default=None, help="Custom window start (ns).")
    parser.add_argument("--end", type=int, default=None, help="Custom window end (ns).")
    args = parser.parse_args()

    con = sqlite3.connect(args.trace)
    ranks = extract_setup(con)
    if args.rank is not None:
        ranks = [r for r in ranks if r["rank"] == args.rank]

    rows = []
    for rank in ranks:
        if args.start is not None and args.end is not None:
            start, end = args.start, args.end
        else:
            window = extract_window(con, rank["pid"])
            start, end = window["start"], window["end"]
        for stream in extract_streams(con, rank["pid"], start, end):
            rows.append({"pid": rank["pid"], **stream})
    print_table(rows)


if __name__ == "__main__":
    main()
