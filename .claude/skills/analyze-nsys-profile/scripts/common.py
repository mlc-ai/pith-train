"""Shared helpers for the analyze-nsys-profile skill scripts."""

import bisect
import re
import sqlite3

#: Matches the stage suffix inside any PithTrain stage-NVTX range name.
#: Examples: layer13.stage3_b, layer05_stage5_b_layer04_stage1_b.
STAGE_PATTERN = re.compile(r"stage([1-5])_([fbw])")

#: Matches the per-rank setup label emitted at cudaProfilerStart.
#: Example: rank=3; pp=0/2 dp=0/1 cp=0/1 ep=3/4; mbs=1 seq=2048
patterns = []
patterns.append(r"^rank=\d+")
patterns.append(r"pp=\d+/\d+ dp=\d+/\d+ cp=\d+/\d+ ep=\d+/\d+")
patterns.append(r"mbs=\d+ seq=\d+$")
SETUP_PATTERN = re.compile("; ".join(patterns))


def print_table(rows: list[dict]) -> None:
    """
    Print a list of dicts as a fixed-width table on stdout. Columns are taken from the first row's
    keys (so every row should carry the same keys); widths auto-size to the longest value per
    column. Missing keys in later rows render as empty cells.
    """
    columns = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    print(" ".join(c.ljust(widths[c]) for c in columns))
    print(" ".join("-" * widths[c] for c in columns))
    for row in rows:
        print(" ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))


def streams_in_window(con: sqlite3.Connection, pid: int, start: int, end: int) -> list[dict]:
    """
    Per-stream kernel counts and times within [start, end] for one rank. A kernel counts as
    communication when its short name starts with "nccl", otherwise as compute.
    """
    cur = con.cursor()
    cur.execute(
        """
        SELECT k.streamId,
               SUM(CASE WHEN s.value LIKE 'nccl%' THEN 1 ELSE 0 END) AS n_comm,
               SUM(CASE WHEN s.value NOT LIKE 'nccl%' THEN 1 ELSE 0 END) AS n_comp,
               SUM(CASE WHEN s.value LIKE 'nccl%' THEN k.end - k.start ELSE 0 END) AS comm_ns,
               SUM(CASE WHEN s.value NOT LIKE 'nccl%' THEN k.end - k.start ELSE 0 END) AS comp_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE k.globalPid / 0x1000000 % 0x1000000 = ? AND k.start >= ? AND k.end <= ?
        GROUP BY k.streamId
        ORDER BY k.streamId
        """,
        (pid, start, end),
    )
    out = []
    for sid, n_comm, n_comp, comm_ns, comp_ns in cur.fetchall():
        out.append(
            {
                "stream": sid,
                "n_comm": n_comm,
                "n_comp": n_comp,
                "comm_ns": comm_ns,
                "comp_ns": comp_ns,
            }
        )
    return out


def kernels_in_window(
    con: sqlite3.Connection, pid: int, start: int, end: int, stream: int | None = None
) -> list[dict]:
    """
    All kernels for one rank in [start, end], optionally scoped to one stream. Each record carries
    the kernel's GPU execution interval (start, end), its stream and name, the correlation_id
    linking it to its runtime API call, and launch_start -- the CPU-side cudaLaunchKernel time.

    Use launch_start (not start) for NVTX-context lookups: async kernels execute on the GPU after
    the CPU has moved past the NVTX range that scheduled them, so launch_start is the relevant
    moment for "what was the CPU doing when this kernel got queued?"
    """
    cur = con.cursor()
    sql = """
        SELECT k.start, k.end, k.streamId, s.value AS name, k.correlationId, r.start AS launch_start
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId AND r.globalTid / 0x1000000 % 0x1000000 = k.globalPid / 0x1000000 % 0x1000000
        WHERE k.globalPid / 0x1000000 % 0x1000000 = ? AND k.start >= ? AND k.end <= ?
    """
    args = [pid, start, end]
    if stream is not None:
        sql += " AND k.streamId = ?"
        args.append(stream)
    sql += " ORDER BY k.start"
    cur.execute(sql, args)
    return [
        {
            "start": start,
            "end": end,
            "stream": sid,
            "name": name,
            "correlation_id": cid,
            "launch_start": launch_start,
        }
        for start, end, sid, name, cid, launch_start in cur.fetchall()
    ]


def union_duration(intervals: list[tuple[int, int]]) -> int:
    """Total time covered by the union of (possibly overlapping) intervals."""
    sorted_intervals = sorted(intervals)
    if not sorted_intervals:
        return 0
    merged = [list(sorted_intervals[0])]
    for start, end in sorted_intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def sweep_overlap(set_a: list[tuple[int, int]], set_b: list[tuple[int, int]]) -> int:
    """
    Total time during which both interval sets have at least one active interval. Linear-time sweep
    over the union of endpoints. Either set may contain overlapping intervals internally; the
    implementation tracks active counts rather than assuming disjoint inputs.
    """
    events = []
    for start, end in set_a:
        events.append((start, 0, +1))
        events.append((end, 0, -1))
    for start, end in set_b:
        events.append((start, 1, +1))
        events.append((end, 1, -1))
    events.sort()
    counts = [0, 0]
    overlap = 0
    last_t = None
    for t, side, delta in events:
        if last_t is not None and counts[0] >= 1 and counts[1] >= 1:
            overlap += t - last_t
        counts[side] += delta
        last_t = t
    return overlap


def stage_of(nvtx_name: str | None) -> str | None:
    """
    Canonicalize an NVTX range name to a stage<N>_<f|b|w> key. For DualPipeV's fused boundary
    ranges (e.g. layer05_stage5_b_layer06_stage1_b), returns the LAST stage marker -- the
    comm-launching side of stage5+stage1 fusions is always stage1 (matters once CP ring
    attention adds stage1 comm). Returns None for non-stage ranges.
    """
    if nvtx_name is None:
        return None
    matches = STAGE_PATTERN.findall(nvtx_name)
    if not matches:
        return None
    return f"stage{matches[-1][0]}_{matches[-1][1]}"


def innermost_nvtx(
    con: sqlite3.Connection,
    pid: int,
    timestamps: list[int],
    scan_limit: int = 256,
) -> list[str | None]:
    """
    For each timestamp, the name of the innermost (smallest-span) NVTX range that contains it.
    Fetches all candidate NVTX ranges for pid once, then performs the smallest-span-containing
    search in Python with a bounded backward scan.

    Far cheaper than per-timestamp SQL when looking up thousands of kernels. scan_limit caps the
    backward walk per timestamp; in practice the innermost layer-stage range is found within a
    handful of candidates of the bisect index.

    Excludes the per-rank setup label (matched by SETUP_PATTERN) and NCCL-emitted ranges (the
    "nccl" prefix), so the search only considers PithTrain stage/chunk/p2p ranges.
    """
    cur = con.cursor()
    cur.execute(
        """
        SELECT e.start, COALESCE(e.end, e.start), COALESCE(e.text, s.value)
        FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId = s.id
        WHERE e.start >= 0 AND e.globalTid / 0x1000000 % 0x1000000 = ?
        ORDER BY e.start
        """,
        (pid,),
    )
    candidates = [
        (start, end, name)
        for start, end, name in cur.fetchall()
        if name and not SETUP_PATTERN.match(name) and not name.startswith("nccl")
    ]
    starts = [row[0] for row in candidates]

    results = []
    for t in timestamps:
        idx = bisect.bisect_right(starts, t)
        best_name = None
        best_span = None
        for j in range(idx - 1, max(-1, idx - 1 - scan_limit), -1):
            start, end, name = candidates[j]
            if end >= t:
                span = end - start
                if best_span is None or span < best_span:
                    best_name = name
                    best_span = span
        results.append(best_name)
    return results
