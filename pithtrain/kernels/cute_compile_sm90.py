"""
Fake tensors and stream that cute.compile traces the SM90 kernels against; the compiled kernels are
launched later through TVM-FFI.
"""

import cutlass
import cutlass.cute as cute
import torch

__all__ = ["TORCH2CUTE", "fake", "fake_stream"]


def fake(dtype: type, shape: tuple, stride: tuple, align: int = 16) -> object:
    return cute.runtime.make_fake_tensor(dtype, shape, stride=stride, assumed_align=align)


def fake_stream() -> object:
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)


TORCH2CUTE = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16, torch.float32: cutlass.Float32, torch.float8_e4m3fn: cutlass.Float8E4M3FN, torch.uint8: cutlass.Uint8}
