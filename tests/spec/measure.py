"""
Recording of the errors the spec-driven tests measure, for calibrating their bounds: with
PITHTRAIN_SPEC_ERRORS=<path> set, every record() call appends one JSON line (case id, output,
metrics). Re-measure on a new architecture or kernel version before changing a bound.
"""

import json
import os


def record(case_id: str, output: str, **metrics: float) -> None:
    path = os.environ.get("PITHTRAIN_SPEC_ERRORS")
    if not path:
        return
    with open(path, "a") as f:
        f.write(json.dumps(dict(case=case_id, output=output, **metrics)) + "\n")
