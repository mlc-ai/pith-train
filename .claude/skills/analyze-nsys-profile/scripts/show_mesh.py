"""Show the mesh layout of a trace: one row per rank with pid, mesh, rank."""

import argparse
import sqlite3

from common import print_table


def extract_rank(label: str) -> int:
    """
    Derive the rank from a PithTrain mesh-label string of the form pp=R/S dp=R/S cp=R/S ep=R/S ...,
    folding axes in (pp_rank, dp_rank, cp_rank, ep_rank) order. Non-axis tokens are ignored.
    """
    axes = {}
    for token in label.split():
        key, value = token.split("=", 1)
        if "/" in value:
            r, s = value.split("/", 1)
            axes[f"{key}_rank"] = int(r)
            axes[f"{key}_size"] = int(s)
    rank = axes["pp_rank"]
    rank = rank * axes["dp_size"] + axes["dp_rank"]
    rank = rank * axes["cp_size"] + axes["cp_rank"]
    rank = rank * axes["ep_size"] + axes["ep_rank"]
    return rank


def extract_mesh(con: sqlite3.Connection) -> list[dict]:
    """
    One record per rank: pid, rank, mesh. The mesh is PithTrain's first NVTX range per rank,
    encoding pp/dp/cp/ep/mbs/seq; we recover it by anchoring on the earliest event per pid.

    The globalTid / 0x1000000 % 0x1000000 formula is the nsys-documented PID extraction:
    globalTid packs [TRACE_ID : bits 48+] [PID : bits 24-47] [TID : bits 0-23]; dividing by
    0x1000000 (= 2**24) shifts the PID slot down, and the modulo trims the TRACE_ID above.
    See https://docs.nvidia.com/nsight-systems/2021.5/nsys-exporter/examples.html for more.
    """
    cur = con.cursor()
    cur.execute(
        """
        WITH first_event AS (
            SELECT globalTid / 0x1000000 % 0x1000000 AS pid, MIN(start) AS start
            FROM NVTX_EVENTS
            WHERE start >= 0
            GROUP BY pid
        )
        SELECT e.globalTid / 0x1000000 % 0x1000000 AS pid, COALESCE(e.text, s.value) AS mesh
        FROM NVTX_EVENTS e
        JOIN first_event f ON e.globalTid / 0x1000000 % 0x1000000 = f.pid AND e.start = f.start
        LEFT JOIN StringIds s ON e.textId = s.id
        ORDER BY pid
        """
    )
    records = []
    for pid, label in cur.fetchall():
        records.append({"pid": pid, "rank": extract_rank(label), "mesh": label})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("trace", help="Path to the nsys SQLite trace export.")
    args = parser.parse_args()

    con = sqlite3.connect(args.trace)
    print_table(extract_mesh(con))


if __name__ == "__main__":
    main()
