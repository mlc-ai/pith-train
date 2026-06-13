"""
Per-stage compute/communication overlap inside each rank's steady-state window.

Emits one row per (rank, enclosing PithTrain stage): exposed_ns (un-hidden wall-clock cost),
overlap (time-weighted hidden fraction across the stage), and overlap_min/overlap_max (extremes
of the per-kernel overlap percentage). Percent cells include a trailing % so the unit is
self-documenting.
"""

import argparse
import sqlite3
from collections import defaultdict

from common import (
    innermost_nvtx,
    kernels_in_window,
    print_table,
    stage_of,
    streams_in_window,
    sweep_overlap,
)
from find_window import extract_window
from show_setup import extract_setup


def extract_overlap(con: sqlite3.Connection, pid: int, start: int, end: int) -> list[dict]:
    """
    Per-stage overlap stats for one rank in [start, end]. Buckets comm-stream kernels by their
    enclosing PithTrain stage and tracks per-kernel overlap percentages for the min/max columns.
    """
    stream_rows = streams_in_window(con, pid, start, end)
    comp_streams = {r["stream"] for r in stream_rows if r["n_comp"] > 0}
    comm_streams = {r["stream"] for r in stream_rows if r["n_comp"] == 0 and r["n_comm"] > 0}

    kernels = kernels_in_window(con, pid, start, end)
    comp_intervals = [
        (k["start"], k["end"])
        for k in kernels
        if k["stream"] in comp_streams and not k["name"].startswith("nccl")
    ]

    comm_kernels = [k for k in kernels if k["stream"] in comm_streams]
    timestamps = [k["launch_start"] or k["start"] for k in comm_kernels]
    enclosing_nvtx = innermost_nvtx(con, pid, timestamps)

    buckets = defaultdict(lambda: {"comm_ns": 0, "exposed_ns": 0, "per_kernel_pct": []})
    for kernel, enclosing in zip(comm_kernels, enclosing_nvtx):
        stage = stage_of(enclosing)
        if stage is None:
            continue
        duration_ns = kernel["end"] - kernel["start"]
        overlap_ns = sweep_overlap([(kernel["start"], kernel["end"])], comp_intervals)
        buckets[stage]["comm_ns"] += duration_ns
        buckets[stage]["exposed_ns"] += max(0, duration_ns - overlap_ns)
        if duration_ns > 0:
            buckets[stage]["per_kernel_pct"].append(100.0 * overlap_ns / duration_ns)

    out = []
    for stage in sorted(buckets):
        stats = buckets[stage]
        overlap_pct = (
            round(100.0 * (stats["comm_ns"] - stats["exposed_ns"]) / stats["comm_ns"], 2)
            if stats["comm_ns"] > 0
            else 0.0
        )
        per_kernel_pct = stats["per_kernel_pct"]
        out.append(
            {
                "stage": stage,
                "exposed_ns": stats["exposed_ns"],
                "overlap": f"{overlap_pct}%",
                "overlap_min": f"{round(min(per_kernel_pct), 2)}%" if per_kernel_pct else "0.0%",
                "overlap_max": f"{round(max(per_kernel_pct), 2)}%" if per_kernel_pct else "0.0%",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("trace", help="Path to the nsys SQLite trace export.")
    parser.add_argument("--start", type=int, default=None, help="Custom window start (ns).")
    parser.add_argument("--end", type=int, default=None, help="Custom window end (ns).")
    args = parser.parse_args()

    con = sqlite3.connect(args.trace)

    rows = []
    for rank in extract_setup(con):
        if args.start is not None and args.end is not None:
            start, end = args.start, args.end
        else:
            window = extract_window(con, rank["pid"])
            start, end = window["start"], window["end"]
        for record in extract_overlap(con, rank["pid"], start, end):
            rows.append({"pid": rank["pid"], **record})
    print_table(rows)


if __name__ == "__main__":
    main()
