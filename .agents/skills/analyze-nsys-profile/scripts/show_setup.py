"""Show the per-rank setup of a trace: one row per rank with pid, setup."""

import argparse
import sqlite3

from common import print_table


def extract_setup(con: sqlite3.Connection) -> list[dict]:
    """
    One record per rank: pid, rank, setup. The setup is PithTrain's first NVTX range per rank,
    encoding rank/pp/dp/cp/ep/mbs/seq; we recover it by anchoring on the earliest event per pid.

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
        SELECT e.globalTid / 0x1000000 % 0x1000000 AS pid, COALESCE(e.text, s.value) AS setup
        FROM NVTX_EVENTS e
        JOIN first_event f ON e.globalTid / 0x1000000 % 0x1000000 = f.pid AND e.start = f.start
        LEFT JOIN StringIds s ON e.textId = s.id
        ORDER BY pid
        """
    )
    records = []
    for pid, setup in cur.fetchall():
        tokens = dict(t.strip(";").split("=", 1) for t in setup.split() if "=" in t)
        record = {"pid": pid, "setup": setup, "rank": int(tokens["rank"])}
        for axis in ("pp", "dp", "cp", "ep"):
            r, s = tokens[axis].split("/", 1)
            record[f"{axis}_rank"] = int(r)
            record[f"{axis}_size"] = int(s)
        record["mbs"] = int(tokens["mbs"])
        record["seq"] = int(tokens["seq"])
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("trace", help="Path to the nsys SQLite trace export.")
    args = parser.parse_args()

    con = sqlite3.connect(args.trace)
    rows = [{"pid": r["pid"], "setup": r["setup"]} for r in extract_setup(con)]
    print_table(rows)


if __name__ == "__main__":
    main()
