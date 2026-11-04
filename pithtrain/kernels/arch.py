"""
Device capability of this process, the one place in pithtrain that probes it (SM90 and SM100).
"""

import torch

__all__ = ["ARCH_MAJOR", "NATIVE_E8M0", "SM_COUNT"]

ARCH_MAJOR, _ = torch.cuda.get_device_capability()
"""
Compute-capability major of the current device at import: 9 on SM90, 10 on SM100.
"""

SM_COUNT = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

NATIVE_E8M0 = ARCH_MAJOR >= 10
"""
Whether the E8M0 FP8 scale has a native PTX conversion (SM100+); SM90 emulates it.
"""
