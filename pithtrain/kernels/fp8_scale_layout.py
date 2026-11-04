"""
The MN-major 1x128 FP8 scale layout DeepGEMM's SM90 GEMMs read without a transpose kernel: a (mn,
sf_k) fp32 scale with strides (1, align(mn, 4)), each K block a contiguous column padded to 16
bytes. The SM90 quantization kernels write it directly; 128x128 weight scales stay row-major.
"""

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["act_scale_like", "is_mn_major", "mn_major_scale", "mn_pad", "to_mn_major"]


def mn_pad(mn: int) -> int:
    """
    Row count of one MN-major column: mn rounded up to 16 bytes of fp32.
    """
    return (mn + 3) & ~3


def mn_major_scale(mn: int, sf_k: int, device: torch.device) -> torch.Tensor:
    return torch.empty_strided((mn, sf_k), (1, mn_pad(mn)), dtype=torch.float32, device=device)


def is_mn_major(sf: torch.Tensor) -> bool:
    return sf.dim() == 2 and sf.stride(0) == 1 and sf.stride(1) == mn_pad(sf.shape[0])


def to_mn_major(sf: torch.Tensor) -> torch.Tensor:
    if is_mn_major(sf):
        return sf
    out = mn_major_scale(sf.shape[0], sf.shape[1], sf.device)
    out.copy_(sf)
    return out


def act_scale_like(mn: int, sf_k: int, device: torch.device) -> torch.Tensor:
    """
    Uninitialized fp32 (mn, sf_k) tensor on device (meta included) with the strides this library's
    quantization ops give a 1x128 activation scale (MN-major on SM90, row-major on SM100), for fake
    and meta implementations. The architecture is the one probed at import, not that of device.
    """
    match ARCH_MAJOR:
        case 9:
            return mn_major_scale(mn, sf_k, device)
        case 10:
            return torch.empty((mn, sf_k), dtype=torch.float32, device=device)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
