"""
Appends measured errors as JSON lines to PITHTRAIN_SPEC_ERRORS, for calibrating bounds.
"""

import json
import os


def record(case_id: str, output: str, **metrics: float) -> None:
    path = os.environ.get("PITHTRAIN_SPEC_ERRORS")
    if not path:
        return
    with open(path, "a") as f:
        f.write(json.dumps(dict(case=case_id, output=output, **metrics)) + "\n")
