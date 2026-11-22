"""
Helpers and references shared by the kernel tests.
"""

import json
import os

import torch
import torch.nn.functional as F

from pithtrain.kernels import (
    fp8_clamped_swiglu_bwd,
    fp8_clamped_swiglu_kmajor,
    fp8_clamped_swiglu_rowwise,
    fp8_silu_mul_bwd,
    fp8_silu_mul_fwd,
    fp8_silu_mul_gu_bwd,
    fp8_silu_mul_gu_fwd,
)

CHUNK = 32768
F64 = torch.float64
BF16 = dict(device="cuda", dtype=torch.bfloat16)
BF16_REL = {"bf16": 2.0**-8, "fp32": 2.0**-22}
FP8 = lambda case: case["axes"]["routing"] != "boundary"  # boundary cases are for the BF16 picker
ALPHA = 1.702  # gpt-oss clamped SwiGLU, asserted against the case
LIMIT = 7.0


def record_measurement(case: dict, what: str, **values) -> None:
    """
    Append one measurement to $PITHTRAIN_SPEC_MEASURE.<pid> for calibrating the bounds.
    """
    path = os.environ.get("PITHTRAIN_SPEC_MEASURE")
    if path:
        with open(f"{path}.{os.getpid()}", "a") as f:
            f.write(json.dumps(dict(case=case["id"], what=what, **values)) + "\n")


class Check:
    """
    Running error statistics of one output: calc_diff, the smallest passing coef, finiteness.
    """

    def __init__(self, what: str, rel: float):
        self.what, self.rel = what, rel
        self.xy = self.xx = self.yy = 0.0
        self.coef = 0.0
        self.finite = True

    def add(self, out: torch.Tensor, ref: torch.Tensor, absref: torch.Tensor) -> None:
        if out.numel() == 0:
            return
        out, ref = out.to(F64), ref.to(F64)
        self.finite &= bool(torch.isfinite(out).all())
        self.xy += float((out * ref).sum())
        self.xx += float((out * out).sum())
        self.yy += float((ref * ref).sum())
        excess = ((out - ref).abs() - self.rel * ref.abs()).clamp(min=0)
        self.coef = max(self.coef, float((excess / (absref.to(F64) + 1e-30)).max()))

    @property
    def calc_diff(self) -> float:
        den = self.xx + self.yy
        return 0.0 if den == 0 else 1 - 2 * self.xy / den

    def verify(self, case: dict, coef: float, calc_diff: float | None) -> None:
        record_measurement(case, self.what, calc_diff=self.calc_diff, coef=self.coef)
        assert self.finite, f"{self.what}: non-finite output"
        assert self.coef <= coef, f"{self.what}: per-element error needs coef {self.coef:.3g} > {coef:.3g}"  # fmt: skip
        if calc_diff is not None:
            assert self.calc_diff <= calc_diff, f"{self.what}: calc_diff {self.calc_diff:.3g} > {calc_diff:.3g}"  # fmt: skip


def row_chunks(counts: list[int]):
    """
    Yield (group, group start, chunk start, chunk end) over grouped rows, chunks within a group.
    """
    start = 0
    for group, count in enumerate(counts):
        for chunk in range(start, start + count, CHUNK):
            yield group, start, chunk, min(chunk + CHUNK, start + count)
        start += count


def gate_up_layout(case: dict) -> str:
    return case["axes"]["gate_up_layout"]


def dq_channels(pair, counts: list[int], group_start: int, group: int, start: int, end: int, channels: int):  # fmt: skip
    """
    Rows [start, end) of one group's per-channel FP8, dequantized to fp32 (channels, rows).
    """
    data, scale = pair
    first, last = (start - group_start), (end - group_start)
    rows = counts[group]
    block = data[group_start * channels : (group_start + rows) * channels].view(channels, rows)
    block_scale = scale[:, start // BLOCK : end // BLOCK].repeat_interleave(BLOCK, dim=1)
    return block[:, first:last].float() * block_scale


def grouped_input(rows: int, width: int, counts: list[int], padded: list[int], gen) -> torch.Tensor:
    """
    bf16 (sum(padded), width): random real rows per group, zero padding.
    """
    x = torch.zeros(sum(padded), width, **BF16)
    start = 0
    for count, size in zip(counts, padded):
        if count:
            x[start : start + count] = torch.randn(count, width, generator=gen, **BF16)
        start += size
    return x


def silu_mul_reference(g, u):
    return F.silu(g) * u


def silu_mul_bwd_reference(g, u, dh):
    sig = torch.sigmoid(g)
    return dh * u * sig * (1 + g * (1 - sig)), dh * g * sig


def clamped_swiglu_reference(g, u):
    gc, uc = g.clamp(max=LIMIT), u.clamp(-LIMIT, LIMIT)
    return gc * torch.sigmoid(ALPHA * gc) * (uc + 1)


def clamped_swiglu_bwd_reference(g, u, dh):
    gc, uc = g.clamp(max=LIMIT), u.clamp(-LIMIT, LIMIT)
    sig = torch.sigmoid(ALPHA * gc)
    dg = dh * (uc + 1) * (sig + ALPHA * gc * sig * (1 - sig)) * (g <= LIMIT)
    du = dh * gc * sig * (u.abs() <= LIMIT)
    return dg, du


def e8m0(amax: torch.Tensor) -> torch.Tensor:
    """
    The recipe's E8M0 scale of a block amax.
    """
    raw = amax.float().clamp(min=1e-4) * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    bits = raw.view(torch.int32)
    bump = torch.where((bits & 0x007FFFFF) != 0, 0x00800000, 0).to(torch.int32)
    return ((bits & 0x7F800000) + bump).view(torch.float32)


class QuantCheck:
    """
    An FP8 activation operand against float64: within half an e4m3 ulp, with the recipe's scales.
    """

    def __init__(self, what: str):
        self.what, self.bad, self.bad_scales, self.blocks, self.worst = what, 0, 0, 0, 0.0

    def add(self, deq: torch.Tensor, scale_full: torch.Tensor, ref: torch.Tensor) -> None:
        """
        Check aligned deq and ref; scale_full repeats each block's scale.
        """
        if deq.numel() == 0:
            return
        ref = ref.to(F64)
        tol = (2.0**-4 + 2.0**-8) * ref.abs() + scale_full.to(F64) * 2.0**-10
        ratio = (deq.to(F64) - ref).abs() / tol
        self.worst = max(self.worst, float(ratio.max()))
        self.bad += int((ratio > 1).sum())

    def add_scales(self, scale: torch.Tensor, amax: torch.Tensor) -> None:
        """
        Check per-block scales against the recipe's scale of the bf16 amax.
        """
        lo, hi = e8m0(amax * (1 - 2.0**-7)), e8m0(amax * (1 + 2.0**-7))
        near = lo != hi
        self.blocks += scale.numel()
        self.bad_scales += int(((scale != e8m0(amax)) & ~near).sum())

    def verify(self, case: dict) -> None:
        record_measurement(
            case, self.what, worst=self.worst, bad=self.bad, bad_scales=self.bad_scales
        )
        assert self.bad == 0, f"{self.what}: {self.bad} elements out of bounds (worst {self.worst:.3g}x)"  # fmt: skip
        assert self.bad_scales == 0, f"{self.what}: {self.bad_scales} of {self.blocks} block scales differ from the recipe"  # fmt: skip


def check_rowwise(check: QuantCheck, pair, ref_fn, M: int) -> None:
    """
    Check a 1x128 rowwise (fp8, scale) pair against the float64 rows ref_fn(start, end).
    """
    fp8, scale = pair
    N = fp8.shape[1]
    nb = -(-N // BLOCK)
    for start in range(0, M, CHUNK):
        end = min(start + CHUNK, M)
        ref = ref_fn(start, end)
        rounded = ref.to(torch.bfloat16).float()
        padded = torch.zeros(end - start, nb * BLOCK, device="cuda")
        padded[:, :N] = rounded.abs()
        amax = padded.view(end - start, nb, BLOCK).amax(-1)
        sc = scale[start:end]
        check.add_scales(sc, amax)
        full = sc.repeat_interleave(BLOCK, dim=1)[:, :N]
        check.add(fp8[start:end].float() * full, full, ref)


def check_channels(check: QuantCheck, pair, ref_fn, counts: list[int], channels: int) -> None:
    """
    Check a per-channel (fp8, scale) pair (SM90 K-major) against the float64 rows of ref_fn.
    """
    data, scale = pair
    for group, group_start, start, end in row_chunks(counts):
        ref = ref_fn(start, end)
        rounded = ref.to(torch.bfloat16).float().abs()
        amax = rounded.view(-1, BLOCK, channels).amax(1)
        sc = scale[:, start // BLOCK : end // BLOCK].T
        check.add_scales(sc, amax)
        deq = dq_channels(pair, counts, group_start, group, start, end, channels).T
        check.add(deq, sc.repeat_interleave(BLOCK, dim=0), ref)


def check_fp8_expert_activation(case: dict) -> None:
    """
    The fused activation + FP8 quantization kernels in the model's layout against float64.
    """
    axes = case["axes"]
    G, inter = axes["G"], axes["inter"]
    counts = axes["group_sizes"]
    ks = [-(-count // BLOCK) * BLOCK for count in counts]
    M = sum(ks)
    offs = torch.tensor(ks, dtype=torch.int64).cumsum(0).to(torch.int32).cuda()
    gen = torch.Generator(device="cuda").manual_seed(M + 7 * G)
    layout = gate_up_layout(case)
    gu = grouped_input(M, 2 * inter, counts, ks, gen)
    dh = grouped_input(M, inter, counts, ks, gen)
    checks = []

    if layout == "interleaved":
        assert (case["axes"]["swiglu_alpha"], case["axes"]["swiglu_limit"]) == (ALPHA, LIMIT)
        gu = gu * 4  # reach the clamp
        bias = (torch.randn(G, 2 * inter, device="cuda", generator=gen)).to(torch.bfloat16)
        row_bias = torch.repeat_interleave(bias, torch.tensor(ks, device="cuda"), dim=0, output_size=M)  # fmt: skip

        def z(start, end):
            return (gu[start:end].float() + row_bias[start:end].float()).to(torch.bfloat16).to(F64)  # fmt: skip

        h_fn = lambda s, e: clamped_swiglu_reference(z(s, e)[:, 0::2], z(s, e)[:, 1::2])
        rows = fp8_clamped_swiglu_rowwise(gu, bias, offs, ALPHA, LIMIT)
        ch = fp8_clamped_swiglu_kmajor(gu, bias, offs, ALPHA, LIMIT)
        checks.append(QuantCheck("clamped h rowwise"))
        check_rowwise(checks[-1], rows, h_fn, M)
        checks.append(QuantCheck("clamped h channels"))
        check_channels(checks[-1], ch, h_fn, ks, inter)
        del rows, ch

        def dgu_fn(start, end):
            zz = z(start, end)
            dg, du = clamped_swiglu_bwd_reference(zz[:, 0::2], zz[:, 1::2], dh[start:end].to(F64))
            out = torch.empty(end - start, 2 * inter, device="cuda", dtype=F64)
            out[:, 0::2], out[:, 1::2] = dg, du
            return out

        dgu_quant, d_bias = fp8_clamped_swiglu_bwd(dh, gu, bias, offs, ALPHA, LIMIT)
        checks.append(QuantCheck("clamped dgate_up rowwise"))
        check_rowwise(checks[-1], dgu_quant[:2], dgu_fn, M)
        checks.append(QuantCheck("clamped dgate_up channels"))
        check_channels(checks[-1], dgu_quant[2:], dgu_fn, ks, 2 * inter)
        bias_check = Check("clamped d_bias", BF16_REL["bf16"])
        group_start = 0
        for group, size in enumerate(ks):
            ref = torch.zeros(2 * inter, device="cuda", dtype=F64)
            absref = torch.zeros_like(ref)
            for start in range(group_start, group_start + size, CHUNK):
                part = dgu_fn(start, min(start + CHUNK, group_start + size))
                ref += part.sum(0)
                absref += part.abs().sum(0)
            # one bf16 rounding per summand
            bias_check.add(d_bias[group], ref, absref)
            group_start += size
        bias_check.verify(case, 2.0**-8, None)
        assert bool((gu.float() > LIMIT).any()), "the clamp must be exercised"
    else:
        g = gu[:, :inter].contiguous() if layout == "halves" else gu[:, 0::2].contiguous()
        u = gu[:, inter:].contiguous() if layout == "halves" else gu[:, 1::2].contiguous()
        h_fn = lambda s, e: silu_mul_reference(g[s:e].to(F64), u[s:e].to(F64))
        quant = fp8_silu_mul_gu_fwd(gu, offs) if layout == "halves" else fp8_silu_mul_fwd(g, u, offs)  # fmt: skip
        checks.append(QuantCheck("silu h rowwise"))
        check_rowwise(checks[-1], quant[:2], h_fn, M)
        checks.append(QuantCheck("silu h channels"))
        check_channels(checks[-1], quant[2:], h_fn, ks, inter)
        del quant
        grads = lambda s, e: silu_mul_bwd_reference(g[s:e].to(F64), u[s:e].to(F64), dh[s:e].to(F64))
        if layout == "halves":
            dgu_quant = fp8_silu_mul_gu_bwd(dh, gu, offs)
            cat_fn = lambda s, e: torch.cat(grads(s, e), dim=1)
            checks.append(QuantCheck("silu dgate_up rowwise"))
            check_rowwise(checks[-1], dgu_quant[:2], cat_fn, M)
            checks.append(QuantCheck("silu dgate_up channels"))
            check_channels(checks[-1], dgu_quant[2:], cat_fn, ks, 2 * inter)
        else:
            dg_quant, du_quant = fp8_silu_mul_bwd(dh, g, u, offs)
            for lane, quant in ((0, dg_quant), (1, du_quant)):
                fn = lambda s, e, lane=lane: grads(s, e)[lane]
                name = ("dgate", "dup")[lane]
                checks.append(QuantCheck(f"silu {name} rowwise"))
                check_rowwise(checks[-1], quant[:2], fn, M)
                checks.append(QuantCheck(f"silu {name} channels"))
                check_channels(checks[-1], quant[2:], fn, ks, inter)
    for check in checks:
        check.verify(case)


ROUTING_CHUNK = 16384  # rows per reference chunk: a multiple of 128 and of 256
I64 = dict(device="cuda", dtype=torch.int64)
REALISTIC = lambda c: c["axes"]["routing"] != "boundary"


def pad(count: int) -> int:
    return -(-count // BLOCK) * BLOCK


def cumsum_list(values: list[int]) -> list[int]:
    out = [0]
    for value in values:
        out.append(out[-1] + value)
    return out


def group_topk(case: dict, gen: torch.Generator) -> torch.Tensor:
    """
    Top-k expert ids (T * ep, top_k) of the EP group; rank 0's experts get exactly group_sizes.
    """
    a = case["axes"]
    T, ep, E, k, G = a["T"], a["ep"], a["E"], a["top_k"], a["G"]
    N, local = T * ep, a["group_sizes"]
    rest, others = N * k - sum(local), E - G
    if others == 0:
        assert rest == 0, (rest, case["id"])
        spread = []
    else:
        base, extra = divmod(rest, others)
        spread = [base + 1] * extra + [base] * (others - extra)
    counts = local + spread
    assert max(counts) <= N and sum(counts) == N * k, case["id"]
    ids = torch.repeat_interleave(torch.arange(E, **I64), torch.tensor(counts, **I64), output_size=N * k)  # fmt: skip
    topk = ids.view(k, N).T  # a run of one expert (<= N slots) hits distinct tokens
    topk = topk[torch.randperm(N, generator=gen, device="cuda")]
    cols = torch.rand(N, k, generator=gen, device="cuda").argsort(1)
    return topk.gather(1, cols).contiguous()


def receive(topk: torch.Tensor, T: int, G: int) -> dict:
    """
    Rank 0's received slots: local expert, token, row among the deduplicated tokens, their count.
    """
    N, _ = topk.shape
    local = topk < G
    token, col = local.nonzero(as_tuple=True)
    expert = topk[token, col]
    order = torch.argsort(((token // T) * G + expert) * N + token)
    token, expert = token[order], expert[order]
    visit = local.any(1)
    dedup_pos = visit.long().cumsum(0) - 1
    return dict(expert=expert, token=token, expand=dedup_pos[token], num_tokens=int(visit.sum()))


def stage3_input(case: dict, gen: torch.Generator) -> dict:
    """
    Rank 0's stage-3 input as the EP dispatch delivers it.
    """
    a = case["axes"]
    T, ep, G = a["T"], a["ep"], a["G"]
    topk = group_topk(case, gen)
    if ep == 1:
        expert_idxs = topk.view(-1)
        out = dict(expert_idxs=expert_idxs, expand_idx=None, num_src=expert_idxs.numel(), group_counts=None)  # fmt: skip
    else:
        recv = receive(topk, T, G)
        out = dict(expert_idxs=recv["expert"], expand_idx=recv["expand"], num_src=recv["num_tokens"], group_counts=list(a["group_sizes"]))  # fmt: skip
    counts = torch.bincount(out["expert_idxs"], minlength=G)
    assert counts.tolist() == a["group_sizes"], case["id"]
    return out


def stable_rank(expert_idxs: torch.Tensor, G: int) -> torch.Tensor:
    """
    Each slot's position among the slots of its expert, in slot order.
    """
    order = torch.argsort(expert_idxs, stable=True)
    first = torch.searchsorted(expert_idxs[order], torch.arange(G, **I64))
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel(), **I64) - first[expert_idxs[order]]
    return rank


def check_placement(reverse: torch.Tensor, expert_idxs: torch.Tensor, counts: list[int], ks: list[int], M: int) -> None:  # fmt: skip
    """
    Each slot in its expert's first count rows, one slot per row, in slot order within an expert.
    """
    assert reverse.dtype == torch.int64 and reverse.shape == expert_idxs.shape
    starts = torch.tensor(cumsum_list(ks)[:-1], **I64)
    pos = reverse - starts[expert_idxs]
    assert bool((pos >= 0).all()) and bool((pos < torch.tensor(counts, **I64)[expert_idxs]).all())
    if reverse.numel():
        assert int(torch.bincount(reverse, minlength=M).max()) <= 1, "two slots on one row"
    assert torch.equal(pos, stable_rank(expert_idxs, len(counts))), "not in slot order"


def check_layout(offs: torch.Tensor, ks_tensor: torch.Tensor, ks: list[int], counts: list[int]) -> None:  # fmt: skip
    want = [pad(c) for c in counts]
    assert ks == want
    assert ks_tensor.dtype == torch.int32 and ks_tensor.tolist() == want
    assert offs.dtype == torch.int32 and offs.tolist() == cumsum_list(want)[1:]


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


def grouped_rows(counts: list[int], width: int, gen: torch.Generator, scale: float = 1.0) -> torch.Tensor:  # fmt: skip
    """
    bf16 (sum of padded counts, width): random real rows per group, zero padding.
    """
    ks = [pad(c) for c in counts]
    x = torch.zeros(sum(ks), width, **BF16)
    for count, start in zip(counts, cumsum_list(ks)):
        if count:
            x[start : start + count] = torch.randn(count, width, generator=gen, device="cuda") * scale  # fmt: skip
    return x


ROWS_PER_BLOCK = 512
IGNORE = -100
GRAD_SLACK = 2.0**-16  # fp32 softmax error, on top of half a bf16 ulp


def log_prob_inputs(case: dict) -> tuple[torch.Tensor, torch.Tensor]:
    rows, vocab = case["axes"]["rows"], case["axes"]["vocab"]
    gen = torch.Generator(device=DEV).manual_seed(rows + vocab)
    logits = torch.empty(rows, vocab, dtype=torch.bfloat16, device=DEV)
    for start in range(0, rows, 4096):  # draw in slices to bound the fp32 temporary
        chunk = logits[start : start + 4096]
        chunk.copy_(torch.randn(chunk.shape, device=DEV, generator=gen) * 2.0)
    target = torch.randint(0, vocab, (rows,), device=DEV, generator=gen)
    target[torch.rand(rows, device=DEV, generator=gen) < 0.03] = IGNORE
    return logits, target


def log_prob_reference(logits: torch.Tensor, target: torch.Tensor):
    """
    Per row block: (rows, float64 log-softmax, valid-row mask, safe target).
    """
    for start in range(0, logits.shape[0], ROWS_PER_BLOCK):
        rows = slice(start, start + ROWS_PER_BLOCK)
        logp = torch.log_softmax(logits[rows].double(), dim=-1)
        valid = target[rows] != IGNORE
        yield rows, logp, valid, torch.where(valid, target[rows], 0)


DEV = "cuda"


def randn(gen: torch.Generator, *shape: int, dtype=torch.bfloat16) -> torch.Tensor:
    return torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32).to(dtype)


CHUNK_GDR_HEAD_DIM = 128  # Qwen3.5's key and value head dims
CHUNK_GDR_CHUNK = 64  # the kernels' chunk size
CHUNK_GDR_GRADS = ("dq", "dk", "dv", "dg", "dbeta")
CHUNK_GDR_BOUNDS = {"o": 1.6e-5, "dq": 2e-5, "dk": 2e-5, "dv": 1.4e-5, "dg": 2e-5, "dbeta": 1.4e-5}
CHUNK_GDR_MAX_ABS = 2.8e-2


def chunk_gdr_fwd_pinned_inputs(
    S: int, H: int, hv: int, B: int = 1, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    """
    Random q, k, v, g, beta and do, drawn as the model computes them.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    device, bf16 = "cuda", torch.bfloat16

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, generator=gen)

    q = F.normalize(randn(B, S, H, CHUNK_GDR_HEAD_DIM).to(bf16), dim=-1)
    k = F.normalize(randn(B, S, H, CHUNK_GDR_HEAD_DIM).to(bf16), dim=-1)
    v = randn(B, S, hv, CHUNK_GDR_HEAD_DIM).to(bf16)
    a_log = torch.log(torch.rand(hv, device=device, generator=gen) * 16)
    g = -a_log.exp() * F.softplus(randn(B, S, hv) + 1.0)
    beta = randn(B, S, hv).to(bf16).sigmoid()
    do = randn(B, S, hv, CHUNK_GDR_HEAD_DIM).to(bf16)
    return q, k, v, g, beta, do


def per_value_head(x: torch.Tensor, hv: int) -> torch.Tensor:
    return x.repeat_interleave(hv // x.shape[2], dim=2)


def chunk_gdr_fwd_pinned_reference(
    q, k, v, g, beta, do, chunk: int = CHUNK_GDR_CHUNK
) -> tuple[torch.Tensor, dict]:
    """
    The recurrence's output and autograd gradients in fp64, in chunked form.
    """
    hv = v.shape[2]
    leaves = [per_value_head(q, hv), per_value_head(k, hv), v, g, beta]
    leaves = [x.detach().double().requires_grad_() for x in leaves]
    qd, kd, vd, gd, bd = leaves
    scale = qd.shape[-1] ** -0.5
    state = qd.new_zeros(hv, qd.shape[-1], vd.shape[-1])
    out = []
    for c0 in range(0, qd.shape[1], chunk):
        span = slice(c0, min(c0 + chunk, qd.shape[1]))
        qc, kc, vc = (t[0, span].transpose(0, 1) for t in (qd, kd, vd))
        gc, bc = gd[0, span].transpose(0, 1).cumsum(-1), bd[0, span].transpose(0, 1)
        L = gc.shape[-1]
        lower = torch.ones(L, L, dtype=torch.bool, device=qd.device).tril()
        decay = torch.where(lower, gc[:, :, None] - gc[:, None, :], float("-inf")).exp()
        strict = decay * (~torch.eye(L, dtype=torch.bool, device=qd.device))
        A = bc[:, :, None] * strict * (kc @ kc.transpose(-1, -2))
        rhs = bc[..., None] * (vc - gc.exp()[..., None] * (kc @ state))
        eye = torch.eye(L, dtype=qd.dtype, device=qd.device)
        u = torch.linalg.solve_triangular(eye + A, rhs, upper=False, unitriangular=True)
        o = scale * (gc.exp()[..., None] * (qc @ state) + (decay * (qc @ kc.transpose(-1, -2))) @ u)  # fmt: skip
        out.append(o.transpose(0, 1))
        k_decayed = kc * (gc[:, -1:] - gc).exp()[..., None]
        state = gc[:, -1].exp()[:, None, None] * state + k_decayed.transpose(-1, -2) @ u
    ref = torch.cat(out, 0).unsqueeze(0)
    ref.backward(do.double())
    return ref.detach(), {name: leaf.grad for name, leaf in zip(CHUNK_GDR_GRADS, leaves)}


CHUNK_GDR_OUTPUTS = ("o", "dq", "dk", "dv", "dg", "dbeta")
TOKEN_HEADS_PER_BLOCK = 1 << 21  # caps the fp64 reference's memory


def chunk_gdr_fwd_pinned_reference_by_heads(
    q, k, v, g, beta, do, keep=slice(None), key_grads=False
):
    """
    The fp64 reference's [o, dq, dk, dv, dg, dbeta] for B = 1, a block of value heads at a time.
    """
    S, H, HV = q.shape[1], q.shape[2], v.shape[2]
    group = HV // H
    block = max(group, min(HV, TOKEN_HEADS_PER_BLOCK // S) // group * group)
    outs = [[] for _ in CHUNK_GDR_OUTPUTS]
    for start in range(0, HV, block):
        values, keys = slice(start, start + block), slice(start // group, (start + block) // group)
        ref, grads = chunk_gdr_fwd_pinned_reference(q[:, :, keys], k[:, :, keys], *(t[:, :, values] for t in (v, g, beta, do)))  # fmt: skip
        dq, dk = grads["dq"], grads["dk"]
        if key_grads:
            dq, dk = (x.view(1, S, -1, group, x.shape[-1]).sum(3) for x in (dq, dk))
        for out, x in zip(outs, (ref, dq, dk, grads["dv"], grads["dg"], grads["dbeta"])):
            out.append(x[:, keep].float())
        del ref, grads, dq, dk
    return [torch.cat(parts, 2) for parts in outs]


def chunk_gdr_errors(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, bool]:
    """
    (calc_diff, max abs error / the reference's max magnitude, all finite).
    """
    scale = float(ref.abs().max())
    err = float((got.float() - ref).abs().max())
    return calc_diff(got, ref), err / scale if scale else err, bool(torch.isfinite(got).all())  # fmt: skip


def chunk_gdr_failures(label: str, got: list, ref: list) -> list[str]:
    failures = []
    for name, x, y in zip(CHUNK_GDR_OUTPUTS, got, ref):
        assert x.shape == y.shape, (label, name, x.shape, y.shape)
        diff, rel_max, finite = chunk_gdr_errors(x, y)
        print(f"MEASURED {label} {name} calc_diff={diff:.3e} max_abs/ref_max={rel_max:.3e}")
        if not (finite and diff < CHUNK_GDR_BOUNDS[name] and rel_max <= CHUNK_GDR_MAX_ABS):
            failures.append(f"{label} {name}: calc_diff {diff:.2e} (bound {CHUNK_GDR_BOUNDS[name]:.0e}), max abs {rel_max:.2e} of ref max (bound {CHUNK_GDR_MAX_ABS:.0e}), finite {finite}")  # fmt: skip
    return failures


MAX_BLOCK_ELEMENTS = 1 << 26  # caps each fp32 (heads x rows x keys) score block's memory


def flash_attn_memoize_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool,
    window: int | None = None,
    sink: torch.Tensor | None = None,
    dout: torch.Tensor | None = None,
    q_positions: torch.Tensor | None = None,
    o_in: torch.Tensor | None = None,
    lse_in: torch.Tensor | None = None,
) -> dict:
    """
    Attention of one sequence in fp32: out and lse, plus dq, dk, dv (and dsink) when dout is given.
    """
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return flash_attn_memoize_blocks(q, k, v, scale, causal, window, sink, dout, q_positions, o_in, lse_in)  # fmt: skip
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def flash_attn_memoize_blocks(
    q, k, v, scale, causal, window, sink, dout, q_positions, o_in, lse_in
) -> dict:
    Sq, Hq, D = q.shape
    Skv, Hkv, Dv = v.shape
    group = Hq // Hkv
    f32 = torch.float32
    qh = q.to(f32).transpose(0, 1).contiguous()
    kh = k.to(f32).repeat_interleave(group, dim=1).transpose(0, 1).contiguous()
    vh = v.to(f32).repeat_interleave(group, dim=1).transpose(0, 1).contiguous()
    if q_positions is None:
        q_positions = torch.arange(Sq, device=q.device) + (Skv - Sq)
    q_positions = q_positions.to(q.device)
    out = torch.empty(Hq, Sq, Dv, dtype=f32, device=q.device)
    lse = torch.empty(Hq, Sq, dtype=f32, device=q.device)
    grads = dout is not None
    if grads:
        doh = dout.to(f32).transpose(0, 1).contiguous()
        o_used_all = o_in.to(f32).transpose(0, 1) if o_in is not None else None
        dq = torch.zeros_like(qh)
        dk = torch.zeros_like(kh)
        dv = torch.zeros_like(vh)
        dsink = torch.zeros(Hq, dtype=f32, device=q.device) if sink is not None else None
    sink_f = sink.to(f32)[:, None] if sink is not None else None
    rows = max(16, MAX_BLOCK_ELEMENTS // (Hq * Skv) // 16 * 16)
    key_index = torch.arange(Skv, device=q.device)
    for r0 in range(0, Sq, rows):
        r1 = min(Sq, r0 + rows)
        pos = q_positions[r0:r1]
        lo, hi = 0, Skv
        if causal:
            hi = min(Skv, int(pos.max()) + 1)
            if window is not None:
                lo = max(0, int(pos.min()) - window)
        if hi <= lo:
            raise ValueError("a query block sees no key")
        qb, kb, vb = qh[:, r0:r1], kh[:, lo:hi], vh[:, lo:hi]
        s = torch.matmul(qb, kb.transpose(1, 2)).mul_(scale)
        if causal:
            j = key_index[lo:hi][None, :]
            allowed = j <= pos[:, None]
            if window is not None:
                allowed &= j >= pos[:, None] - window
            s.masked_fill_(~allowed[None], float("-inf"))
        m = torch.logsumexp(s, dim=-1)
        if sink_f is not None:
            m = torch.logaddexp(m, sink_f.expand_as(m))
        p = torch.exp(s - m[..., None])
        o = torch.matmul(p, vb)
        out[:, r0:r1], lse[:, r0:r1] = o, m
        if not grads:
            continue
        if lse_in is not None:
            m_used = lse_in[:, r0:r1].to(f32)
            p = torch.exp(s - m_used[..., None])
        else:
            m_used = m
        o_used = o_used_all[:, r0:r1] if o_used_all is not None else o
        dob = doh[:, r0:r1]
        delta = (dob * o_used).sum(-1, keepdim=True)
        dp = torch.matmul(dob, vb.transpose(1, 2))
        ds = p * (dp - delta)
        del dp, s
        dq[:, r0:r1] = torch.matmul(ds, kb).mul_(scale)
        dk[:, lo:hi] += torch.matmul(ds.transpose(1, 2), qb).mul_(scale)
        dv[:, lo:hi] += torch.matmul(p.transpose(1, 2), dob)
        if dsink is not None:
            dsink -= (torch.exp(sink_f - m_used) * delta.squeeze(-1)).sum(-1)
    result = dict(out=out.transpose(0, 1), lse=lse)
    if grads:
        result["dq"] = dq.transpose(0, 1)
        result["dk"] = dk.reshape(Hkv, group, Skv, D).sum(1).transpose(0, 1)
        result["dv"] = dv.reshape(Hkv, group, Skv, Dv).sum(1).transpose(0, 1)
        if dsink is not None:
            result["dsink"] = dsink
    return result


BLOCK = 128
INV_448 = torch.tensor(1.0 / 448.0, dtype=torch.float32)


def e8m0_scale(amax: torch.Tensor) -> torch.Tensor:
    """
    The E8M0 scale: 2^ceil(log2(max(amax, 1e-4) / 448)) in fp32.
    """
    raw = amax.float().clamp(min=1e-4) * INV_448.to(amax.device)
    mantissa, exponent = torch.frexp(raw)  # mantissa in [0.5, 1)
    exponent = torch.where(mantissa == 0.5, exponent - 1, exponent)
    return torch.ldexp(torch.ones_like(raw), exponent)


def cast(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x.float() * scale.reciprocal()).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def blocks(n: int) -> int:
    return -(-n // BLOCK)


def fp8_quant_grad_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    1x128 blocks along each row of a 2-D x: fp8 (M, K), scale (M, ceil(K / 128)).
    """
    M, K = x.shape
    padded = x.new_zeros(M, blocks(K) * BLOCK, dtype=torch.float32)
    padded[:, :K] = x.float()
    tiles = padded.view(M, blocks(K), BLOCK)
    scale = e8m0_scale(tiles.abs().amax(-1))
    q = cast(tiles, scale[..., None]).view(M, -1)[:, :K].contiguous()
    return q, scale


def fp8_quant_weight_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    128x128 tiles of x (..., R, C): fp8 (..., R, C), scale (..., ceil(R / 128), ceil(C / 128)).
    """
    *lead, R, C = x.shape
    padded = x.new_zeros(*lead, blocks(R) * BLOCK, blocks(C) * BLOCK, dtype=torch.float32)
    padded[..., :R, :C] = x.float()
    tiles = padded.view(*lead, blocks(R), BLOCK, blocks(C), BLOCK)
    scale = e8m0_scale(tiles.abs().amax(dim=(-3, -1)))
    q = cast(tiles, scale[..., :, None, :, None]).view(*lead, blocks(R) * BLOCK, blocks(C) * BLOCK)
    return q[..., :R, :C].contiguous(), scale


def assert_fp8_equal(got: torch.Tensor, want: torch.Tensor, name: str) -> None:
    assert got.dtype == torch.float8_e4m3fn and got.shape == want.shape, (name, got.dtype, got.shape, want.shape)  # fmt: skip
    differ = (
        (got.contiguous().view(torch.uint8) != want.contiguous().view(torch.uint8)).sum().item()
    )
    assert differ == 0, f"{name}: {differ} of {got.numel()} fp8 bytes differ"


def assert_scale_equal(got: torch.Tensor, want: torch.Tensor, name: str) -> None:
    assert got.dtype == torch.float32 and got.shape == want.shape, (name, got.shape, want.shape)
    differ = (got != want).sum().item()
    assert differ == 0, f"{name}: {differ} of {got.numel()} scales differ"


def half_ulp_bf16(x: torch.Tensor) -> torch.Tensor:
    x = x.double().abs()
    _, exponent = torch.frexp(x)  # ulp = 2^(exponent - 8)
    exponent = exponent.clamp(min=-125)
    return torch.ldexp(torch.full_like(x, 0.5), exponent - 8)


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    1 - 2 <x, y> / (|x|^2 + |y|^2) in float64.
    """
    x, y = x.double(), y.double()
    den = (x * x + y * y).sum()
    return 0.0 if den == 0 else float(1 - 2 * (x * y).sum() / den)


def rel_l2(out: torch.Tensor, ref: torch.Tensor) -> float:
    """
    |out - ref| / |ref| in float64 (0 when both are zero).
    """
    out, ref = out.double(), ref.double()
    den = ref.norm()
    num = (out - ref).norm()
    return float(num / den) if den > 0 else float(num)


def record(case_id: str, output: str, **metrics: float) -> None:
    path = os.environ.get("PITHTRAIN_SPEC_ERRORS")
    if not path:
        return
    with open(path, "a") as f:
        f.write(json.dumps(dict(case=case_id, output=output, **metrics)) + "\n")
