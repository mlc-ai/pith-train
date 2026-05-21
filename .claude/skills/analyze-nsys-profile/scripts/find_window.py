"""Pick the steady-state analysis window for each rank: the median-indexed DualPipeV chunk."""

import argparse
import sqlite3

from common import print_table
from show_setup import extract_setup


def extract_window(con: sqlite3.Connection, pid: int) -> dict:
    """
    Return the steady-state analysis window for one rank: the median-indexed forward chunk ...
    NVTX range emitted by DualPipeV. Deterministic across re-runs of the same config, so
    windows are comparable for before/after measurements. The name is included for human
    verification (e.g. confirming the chunk in the nsys GUI).
    """
    cur = con.cursor()
    cur.execute(
        """
        SELECT e.start, e.end, COALESCE(e.text, s.value) AS name
        FROM NVTX_EVENTS e
        LEFT JOIN StringIds s ON e.textId = s.id
        WHERE e.start >= 0 AND e.globalTid / 0x1000000 % 0x1000000 = ? AND COALESCE(e.text, s.value) LIKE 'forward chunk%backward chunk%'
        ORDER BY e.start
        """,
        (pid,),
    )
    rows = cur.fetchall()
    start, end, name = rows[len(rows) // 2]
    return {"start": start, "end": end, "name": name}


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("trace", help="Path to the nsys SQLite trace export.")
    args = parser.parse_args()

    con = sqlite3.connect(args.trace)
    rows = []
    for rank in extract_setup(con):
        window = extract_window(con, rank["pid"])
        rows.append({"pid": rank["pid"], **window})
    print_table(rows)


if __name__ == "__main__":
    main()
