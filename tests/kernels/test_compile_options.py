"""
stage_compile_options returns the Inductor options the models merge into their compiled stages'
torch.compile options: on SM90 automatic pointwise autotuning off, on SM100 nothing.
"""

import torch

from pithtrain.kernels import silu_mul_bwd, silu_mul_fwd, stage_compile_options


def test_fresh_dict_of_known_options():
    """
    A new dict each call (callers may extend it), holding only real Inductor options, and on the
    architectures that set one, exactly autotune_pointwise off.
    """
    first, second = stage_compile_options(), stage_compile_options()
    assert isinstance(first, dict) and first == second and first is not second
    first["cpp_wrapper"] = True
    assert "cpp_wrapper" not in stage_compile_options()
    assert second in ({}, {"triton.autotune_pointwise": False})
    for key in second:
        group, _, name = key.rpartition(".")
        assert hasattr(
            getattr(torch._inductor.config, group) if group else torch._inductor.config, name
        ), key


def test_compiled_stage_with_options():
    """
    A fullgraph region compiled with the options merged (with and without cpp_wrapper, as the
    stages compile) runs and matches the eager silu_mul to one bf16 ulp per element.
    """
    gen = torch.Generator(device="cuda").manual_seed(0)
    gate, up, do = (
        torch.randn(4096, 512, generator=gen, device="cuda").to(torch.bfloat16) for _ in range(3)
    )

    def body(gate, up, do):
        return silu_mul_fwd(gate, up), *silu_mul_bwd(do, gate, up)

    want = body(gate, up, do)
    for options in ({**stage_compile_options()}, {"cpp_wrapper": True, **stage_compile_options()}):
        torch._dynamo.reset()
        got = torch.compile(body, fullgraph=True, options=options)(gate, up, do)
        for compiled_out, eager_out in zip(got, want):
            bits_c, bits_e = compiled_out.view(torch.int16).int(), eager_out.view(torch.int16).int()
            same_sign = (bits_c < 0) == (bits_e < 0)
            ulps = torch.where(same_sign, (bits_c - bits_e).abs(), bits_c.abs() + bits_e.abs())
            assert ulps.max().item() <= 1
