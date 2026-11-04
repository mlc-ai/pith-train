"""
The SM90 CuTe-DSL kernels reproduce the SM90 branch of compute_fp8_scale bit for bit.
"""

import triton
import triton.language as tl

__all__ = ["compute_fp8_scale"]


@triton.jit
def compute_fp8_scale(amax, NATIVE_E8M0: tl.constexpr):
    """
    (scale, reciprocal_scale) from per-block amax: scale is the power of two at or above max(amax,
    1e-4) / 448, at least 2^-126; both are exact powers of two. NATIVE_E8M0 selects the native PTX
    conversion (SM100) or the bit-manipulation emulation (SM90); the results are equal.
    """
    FP8_MAX_RCP: tl.constexpr = 1.0 / 448.0

    amax_clamped = tl.maximum(amax.to(tl.float32), 1e-4)
    scale_input = amax_clamped * FP8_MAX_RCP

    if NATIVE_E8M0:
        scale_e8m0_biased = tl.inline_asm_elementwise(asm="cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;", constraints="=h,r", args=[scale_input], dtype=tl.uint16, is_pure=True, pack=1).to(tl.uint8)

        scale_fp = (scale_e8m0_biased.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    else:
        # power-of-2 ceil: keep the exponent, bump it if any mantissa bit is set
        bits = scale_input.to(tl.int32, bitcast=True)
        mantissa = bits & 0x007FFFFF
        scale_fp = ((bits & 0x7F800000) + tl.where(mantissa != 0, 0x00800000, 0)).to(tl.float32, bitcast=True)

    fp32_min_normal = tl.exp2(-126.0)
    scale_fp = tl.maximum(scale_fp, fp32_min_normal)

    # Exact reciprocal via exponent flip: 2^(e-127) -> 2^(127-e)
    rcp_bits = (254 << 23) - scale_fp.to(tl.int32, bitcast=True)
    rcp_scale = rcp_bits.to(tl.float32, bitcast=True)

    return scale_fp, rcp_scale
