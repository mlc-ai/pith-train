"""
The fused SiLU-mul + FP8 quantizations at the fp8_silu_mul_quant cases, against float64.
"""

import pytest
import torch

from tests.kernels.cases import params
from tests.kernels.common import FP8, check_fp8_expert_activation


@pytest.fixture(autouse=True)
def no_tf32():
    """
    Disable TF32 and empty the allocator cache around each test.
    """
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous
    torch.cuda.empty_cache()


@pytest.mark.parametrize("case", params("fp8_silu_mul_quant", keep=lambda c: FP8(c) and c["axes"]["act"] != "clamped_swiglu"))  # fmt: skip
def test_fp8_silu_mul_quant(case: dict) -> None:
    check_fp8_expert_activation(case)
