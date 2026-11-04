"""
segment_sum_sm100: per-group column sums of rows sorted by group, in TIRx-lite (B200).

Groups are cut into 128-row tiles from their own start; tile j of group g lives at slot f(g) + j
with f(g) = g + start_g // 128 (strictly increasing, collision-free, no scan). A tile's fp32 column
sums (rows in a fixed per-tile order) are computed once, then every group adds its tile sums in row
order starting from zero and rounds once to out_dtype. Dispatch:
- many groups (at least DIRECT_MIN_GROUPS) whose (group, chunk) CTAs fit one wave or are short: one
  CTA per (group, chunk) folds the group's tiles in registers (no second kernel);
- otherwise (rows 16-byte aligned, at most SMEM_GROUPS groups): a compact persistent TMA ring (only
  real tiles, interleaved over the fewest CTAs that give every one the same unit count, each CTA's
  last ~TAIL_UNITS units claimed from a shared tag-validated counter so slow SMs shed tail work; for
  bf16 H=2880 128B-swizzled 256-column chunks) publishes each tile sum as tagged 64-bit words, and a
  PDL-launched reduce detects each chain's last word by its tag (no griddepcontrol.wait) and folds
  in order;
- rows not 16-byte aligned or more than SMEM_GROUPS groups: plain-LDG tile sums + ordered reduce.
Every TMA kernel sums a tile column by column with rows in classes (tile-relative row mod 4)
accumulated in row order and the classes folded 0, 1, 2, 3, so a group's bits do not depend on the
kernel its call is dispatched to.

The tagged scratch is a buffer this module owns per (device, stream), zero-filled when allocated: a
word counts as written only when its high half carries this call's tag, a 16-bit per-call epoch of
that buffer spread into an inf/NaN pattern (word_tag), so consecutive calls always differ. A zero
word carries no tag, and only this module's earlier calls on the same stream ever wrote the buffer;
when the epoch would wrap (every 65535 calls) the buffer is zeroed again, in stream order before
the call, so a stale word can never carry the current tag. The tag is a host argument fixed per
launch, so this path must not be captured in a CUDA graph (replays would reuse one tag over the
previous replay's words).
"""

import functools
import math

import torch
import tvm_tirx

import tirx_kernels.tirx_lite as txl

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})

__all__ = ["segment_sum_sm100"]


EXECUTABLES = []  # keep the compiled modules alive behind the cached entry functions


def entry_function(executable):
    """
    The compiled kernel's entry function, looked up once (Executable.__call__ re-resolves it on
    every call, a few us of host time); the executable stays alive in EXECUTABLES.
    """
    EXECUTABLES.append(executable)
    return executable["main"]


TILE = 128
DTYPE_NAME = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}
INDEX_DTYPE_NAME = {torch.int32: "int32", torch.int64: "int64"}
STAGE_ROWS = 32  # rows per ring stage (one bulk copy per producer lane)
CONSUMER_WARPS = 4  # consumer warps; each owns two of the row classes
EVICT_FIRST = 0x12F0000000000000
EVICT_LAST = 0x14F0000000000000
SMEM_BUDGET = 100000  # compact ring stage budget in bytes
DIRECT_SMEM = 66000  # direct kernel ring budget: 4 stages of 16 KB, three CTAs per SM
LONG_TILES = 16  # average tiles per group from which the chunked reduce is used
CTAS_PER_SM = 1  # persistent ring CTAs per SM
DIRECT_MIN_GROUPS = 16  # groups from which the direct kernel may be used (fewer: compact ring)
SHORT_ROWS = 512  # average rows per group up to which the direct kernel is used regardless of waves
TAIL_UNITS = 6  # ring: last ~TAIL_UNITS units per CTA claimed dynamically from a shared pool (0 = fully static)
CLAIM_LEAD = 2  # static units before the end at which the first claim is issued
IDLE_JOIN = 0  # CTAs beyond the busy ones also claim pool units
SMEM_GROUPS = 2048  # groups whose clamped ends are staged in shared memory (more: global ballot search)
TMA_LOAD_2D = "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
TMA_LOAD_3D = "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
TMA_BOX_MAX = 256  # TMA box extent limit (elements)
SWIZZLE_ATOM = 64  # bf16 columns of one 128-byte swizzle atom
REDUCE_COLS, REDUCE_LANES, REDUCE_UNROLL = 32, 8, 8  # reduce: columns per CTA, tile lanes, tiles per lane per round
TAG_REDUCE_LONG = (1, 8, 8)  # tag reduce for long groups: (column warps, tile lanes, tiles per lane per round)
TAG_REDUCE_SHORT = (4, 2, 8)  # short groups: wider column blocks, fewer tile lanes
POLL_SLEEP_NS = 64  # ns between tag polls
POLL_LIMIT = 1 << 20  # poll rounds before a word is given up on (NaN result, never a hang)


TAGGED_SCRATCH = {}  # (device type, index, stream) -> [int64 buffer (uint64 words), epoch of its last call]


def word_tag(epoch):
    """
    The 32-bit word tag of an epoch (1 .. 0xFFFF): its 16 bits spread under a mask that makes the
    tag half an inf/NaN pattern as fp32, as both bf16 / fp16 halves and as FP8 bytes, never zero.
    """
    tag_bits = 0x7F807F80 | (epoch & 0x7F) | ((epoch & 0x80) << 8) | ((epoch & 0x7F00) << 8) | ((epoch & 0x8000) << 16)
    return tag_bits - (1 << 32) if tag_bits >= 1 << 31 else tag_bits


def tagged_scratch(num_words, device):
    """
    (the first num_words words of this stream's tagged scratch, this call's tag). The buffer is
    zero-filled when allocated or grown, and zeroed again (stream-ordered, before this call's
    kernels) when its epoch would wrap, so no word in it can carry the returned tag before this call
    writes it. Kept per (device, stream): launches on two streams never share words or epochs.
    """
    key = (device.type, device.index, torch.cuda.current_stream(device).cuda_stream)
    entry = TAGGED_SCRATCH.get(key)
    if entry is None or entry[0].numel() < num_words:
        entry = TAGGED_SCRATCH[key] = [torch.zeros(max(num_words, 1 << 16), dtype=torch.int64, device=device), 0]
    entry[1] += 1
    if entry[1] > 0xFFFF:
        entry[0].zero_()
        entry[1] = 1
    return entry[0][:num_words].view(torch.uint64), word_tag(entry[1])


def cdiv(dividend, divisor):
    return (dividend + divisor - 1) // divisor


def vector_width(dtype, H):
    """
    (vec, elem_bytes): elements per load vector (at most 16 bytes, dividing H) and bytes per
    element.
    """
    elem_bytes = 4 if dtype == "float32" else 2
    return math.gcd(H, 16 // elem_bytes), elem_bytes


class View:
    """
    dst[i] -> buf[base + i].
    """

    def __init__(self, buf, base):
        self.buf, self.base = buf, base

    def __getitem__(self, i):
        return self.buf[self.base + i]


def load_vec(x, idx, dtype, vec, elem_bytes, dst, load_ok):
    """
    dst[0:vec] = fp32 of x[idx : idx + vec] if load_ok else 0.
    """
    num_bytes = vec * elem_bytes
    num_words = max(1, num_bytes // 4)
    if num_bytes >= 4:
        words = txl.alloc_local([num_words], "uint32")
        for i in range(num_words):
            txl.assign(words[i], txl.uint32(0))
        with txl.if_(load_ok), txl.then_():
            if num_words == 4:
                txl.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], x.ptr_to([idx]))
            elif num_words == 2:
                txl.ptx.ld.global_.v2.b32(words[0], words[1], x.ptr_to([idx]))
            else:
                txl.ptx.ld.global_.b32(words[0], x.ptr_to([idx]))
        for i in range(num_words):
            if dtype == "float32":
                txl.assign(dst[i], txl.reinterpret(words[i], ty="float32"))
            elif dtype == "bfloat16":
                txl.assign(dst[2 * i], txl.reinterpret(words[i] << txl.uint32(16), ty="float32"))
                txl.assign(dst[2 * i + 1], txl.reinterpret(words[i] & txl.uint32(0xFFFF0000), ty="float32"))
            else:
                txl.idioms.cast_f16x2_to_f32x2(dst, i, words[i])
    else:  # one 16-bit element
        half = txl.alloc_local([1], "uint16")
        txl.assign(half[0], txl.uint16(0))
        with txl.if_(load_ok), txl.then_():
            txl.ptx.ld.global_.b16(half[0], x.ptr_to([idx]))
        if dtype == "bfloat16":
            txl.assign(dst[0], txl.reinterpret(txl.cast(half[0], "uint32") << txl.uint32(16), ty="float32"))
        else:
            txl.ptx.cvt.f32.f16(dst[0], half[0])


def store_vec(out, idx, dtype, vec, src):
    """
    out[idx : idx + vec] = src[0:vec] rounded to dtype.
    """
    if dtype == "float32":
        words = [txl.reinterpret(src[i], ty="uint32") for i in range(vec)]
    else:
        if vec == 1:
            half = txl.alloc_local([1], "uint16")
            if dtype == "bfloat16":
                txl.ptx.cvt.rn.bf16.f32(half[0], src[0])
            else:
                txl.ptx.cvt.rn.f16.f32(half[0], src[0])
            txl.ptx.st.global_.b16(out.ptr_to([idx]), half[0])
            return
        packed = txl.alloc_local([vec // 2], "uint32")
        for i in range(vec // 2):
            if dtype == "bfloat16":
                txl.ptx.cvt.rn.bf16x2.f32(packed[i], src[2 * i + 1], src[2 * i])
            else:
                txl.ptx.cvt.rn.f16x2.f32(packed[i], src[2 * i + 1], src[2 * i])
        words = [packed[i] for i in range(vec // 2)]
    num_left = len(words)
    while num_left > 0:
        width = 4 if num_left >= 4 else (2 if num_left >= 2 else 1)
        word_idx = len(words) - num_left
        elem_off = word_idx * (vec // len(words))
        addr = out.ptr_to([idx + elem_off])
        if width == 4:
            txl.ptx.st.global_.v4.b32(addr, words[word_idx], words[word_idx + 1], words[word_idx + 2], words[word_idx + 3])
        elif width == 2:
            txl.ptx.st.global_.v2.b32(addr, words[word_idx], words[word_idx + 1])
        else:
            txl.ptx.st.global_.b32(addr, words[word_idx])
        num_left -= width


def store_tagged(part, idx, vec, src, tag):
    """
    part[idx : idx + vec] = (tag << 32) | bits(src[i]): one 64-bit word per column, written with
    relaxed gpu-scope stores. A 64-bit scalar access is single-copy atomic, so a reader that sees
    this call's tag in a word also sees that word's value: every word validates itself and no fence
    or flag is needed.
    """
    tag_hi = txl.cast(txl.cast(tag, "uint32"), "uint64") << txl.uint64(32)
    for i in range(0, vec, 2):
        first_word = tag_hi | txl.cast(txl.reinterpret(src[i], ty="uint32"), "uint64")
        second_word = tag_hi | txl.cast(txl.reinterpret(src[i + 1], ty="uint32"), "uint64")
        txl.ptx["st.relaxed.gpu.global.L2::cache_hint.v2.b64"](part.ptr_to([idx + i]), first_word, second_word, txl.uint64(EVICT_LAST))


def load_offset(offs, group_idx, idtype):
    """
    offs[group_idx] as int32 (one global load).
    """
    value = txl.alloc_local([1], idtype)
    if idtype == "int64":
        txl.ptx.ld.global_.s64(value[0], offs.ptr_to([group_idx]))
    else:
        txl.ptx.ld.global_.s32(value[0], offs.ptr_to([group_idx]))
    return txl.cast(value[0], "int32") if idtype == "int64" else value[0]


def group_bounds(offs, group_idx, M, idtype):
    """
    (start, end) of a group, ends clamped to M, start 0 for group 0, end >= start.
    """
    group_end = txl.min(load_offset(offs, group_idx, idtype), M)
    group_start = txl.Select(group_idx > 0, txl.min(load_offset(offs, txl.max(group_idx - 1, 0), idtype), M), txl.int32(0))
    return group_start, txl.max(group_end, group_start)


def tile_sum_layout(H, vec):
    """
    Plain tile sums: (vectors per row, column chunks, vectors per chunk, threads per row class).
    """
    num_vecs = H // vec
    num_chunks = cdiv(num_vecs, 128)
    chunk_vecs = cdiv(num_vecs, num_chunks)
    class_threads = cdiv(chunk_vecs, 32) * 32
    return num_vecs, num_chunks, chunk_vecs, class_threads


def find_slot_group(offs, part_slot, lane, M, G, idtype):
    """
    The last group g with f(g) <= part_slot (f is strictly increasing, f(0) = 0), counted by one
    warp ballot per 32 groups.
    """
    num_before = txl.local_scalar("int32")
    txl.assign(num_before, txl.int32(0))
    with txl.serial((G + 31) // 32) as group_round:
        group_idx = group_round * 32 + lane
        safe_group = txl.Select(group_idx < G, group_idx, txl.int32(0))
        group_start, _ = group_bounds(offs, safe_group, M, idtype)
        is_before = txl.And(group_idx < G, group_idx + group_start // TILE <= part_slot)
        ballot = txl.alloc_local([1], "uint32")
        txl.ptx.vote_sync.ballot.b32(ballot[0], is_before, txl.uint32(0xFFFFFFFF))
        txl.assign(num_before, num_before + txl.cast(txl.popcount(ballot[0]), "int32"))
    return num_before - 1


@functools.cache
def build_tile_sum(dtype, idtype, H):
    """
    Plain-LDG tile sums: one CTA per (tile slot, column chunk) writes the tile's fp32 column sums to
    its slot of part; slots no tile occupies are left untouched.
    """
    vec, elem_bytes = vector_width(dtype, H)
    NUM_VECS, NUM_CHUNKS, CHUNK_VECS, CLASS_THREADS = tile_sum_layout(H, vec)
    ROW_CLASSES = max(1, 256 // CLASS_THREADS)
    while TILE % ROW_CLASSES:
        ROW_CLASSES -= 1
    CLASS_ROWS = TILE // ROW_CLASSES
    UNROLL = min(CLASS_ROWS, 16 if vec * elem_bytes >= 8 else 32)
    NUM_ROUNDS = CLASS_ROWS // UNROLL
    THREADS = CLASS_THREADS * ROW_CLASSES

    @txl.kernel()
    def tile_sum(x: txl.Tensor((txl.int64(),), dtype, data_alignment=1), offs: txl.Tensor((txl.int64(),), idtype, data_alignment=1), part: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), M: txl.int32, G: txl.int32, n_units: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=n_units, block=(THREADS // 32) * 32))
        unit = txl.cta_id()
        tidx = txl.thread_id()
        lane = tidx & 31
        txl.ptx.griddepcontrol.launch_dependents()
        part_slot, chunk_idx = unit // NUM_CHUNKS, unit % NUM_CHUNKS
        smem = txl.smem_pool()
        s_class_sums = smem.alloc([max(ROW_CLASSES - 1, 1) * CLASS_THREADS * vec], "float32")

        group_idx = find_slot_group(offs, part_slot, lane, M, G, idtype)
        group_start, group_end = group_bounds(offs, group_idx, M, idtype)
        row_start = group_start + (part_slot - group_idx - group_start // TILE) * TILE
        num_valid_rows = group_end - row_start
        with txl.if_(num_valid_rows > 0), txl.then_():
            row_class, class_col = tidx // CLASS_THREADS, tidx % CLASS_THREADS
            vec_idx = chunk_idx * CHUNK_VECS + class_col
            col_ok = txl.And(class_col < CHUNK_VECS, vec_idx < NUM_VECS)
            acc_sum = txl.alloc_local([vec], "float32")
            for k in range(vec):
                txl.assign(acc_sum[k], txl.float32(0))
            row_vals = txl.alloc_local([UNROLL * vec], "float32")
            with txl.serial(NUM_ROUNDS) as round_idx:
                round_row = row_class * CLASS_ROWS + round_idx * UNROLL
                for i in range(UNROLL):
                    row = round_row + i
                    x_idx = txl.cast(row_start + row, "int64") * H + txl.cast(vec_idx * vec, "int64")
                    load_vec(x, x_idx, dtype, vec, elem_bytes, View(row_vals, i * vec), txl.And(col_ok, row < num_valid_rows))
                for i in range(UNROLL):
                    for k in range(vec):
                        txl.assign(acc_sum[k], acc_sum[k] + row_vals[i * vec + k])

            if ROW_CLASSES > 1:
                with txl.if_(row_class > 0), txl.then_():
                    for k in range(vec):
                        txl.ptx.st.shared.f32(s_class_sums.ptr_to([((row_class - 1) * CLASS_THREADS + class_col) * vec + k]), acc_sum[k])
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(THREADS))
            with txl.if_(txl.And(row_class == 0, col_ok)), txl.then_():
                partial = txl.alloc_local([1], "float32")
                for class_idx in range(1, ROW_CLASSES):
                    for k in range(vec):
                        txl.ptx.ld.shared.f32(partial[0], s_class_sums.ptr_to([((class_idx - 1) * CLASS_THREADS + class_col) * vec + k]))
                        txl.assign(acc_sum[k], acc_sum[k] + partial[0])
                store_vec(part, txl.cast(part_slot, "int64") * H + txl.cast(vec_idx * vec, "int64"), "float32", vec, acc_sum)

    return tile_sum


def ring_layout(dtype, H):
    """
    None unless rows are 16-byte aligned (TMA strides); else (vec, vectors per row, column chunks,
    vectors per chunk, ring stages).

    Column chunks are TMA boxes of chunk_vecs * vec columns: the widest divisor of H that is a
    multiple of vec and at most 256, else 256 with the tail chunk's box running past H (zero fill).
    """
    vec, elem_bytes = vector_width(dtype, H)
    if vec * elem_bytes != 16:
        return None
    num_vecs = H // vec
    if H <= TMA_BOX_MAX:
        chunk_cols = H
    else:
        chunk_cols = next((width for width in range(TMA_BOX_MAX, 63, -vec) if H % width == 0), TMA_BOX_MAX)
    chunk_vecs = chunk_cols // vec
    num_chunks = cdiv(H, chunk_cols)
    num_stages = max(2, min(16, SMEM_BUDGET // (STAGE_ROWS * chunk_cols * elem_bytes + 8)))
    return vec, num_vecs, num_chunks, chunk_vecs, num_stages


def group_ring_layout(dtype, H):
    """
    The direct kernel uses the ring's column chunks, so both share one per-column tile arithmetic
    (rows by tile-relative index mod 256 // chunk_vecs, classes folded in order) for every H: a
    group's bits do not depend on which kernel a call is dispatched to.
    """
    ring = ring_layout(dtype, H)
    if ring is None:
        return None
    vec, num_vecs, num_chunks, chunk_vecs, _ = ring
    num_stages = max(2, min(16, DIRECT_SMEM // (STAGE_ROWS * chunk_vecs * 16 + 8)))
    return vec, num_vecs, num_chunks, chunk_vecs, num_stages


class RingShape:
    """
    Compile-time shape the TMA ring phases share: input dtype, vec elements per 16-byte vector,
    chunk_vecs vectors per column chunk, row_classes consumer row classes, rows_per_thread rows per
    consumer thread per stage, use_swizzle and num_atoms for the 128B-swizzled stage layout, and
    stage_row_bytes per stage row otherwise.
    """

    def __init__(self, dtype, vec, chunk_vecs, row_classes, rows_per_thread, use_swizzle, num_atoms, stage_row_bytes):
        self.dtype, self.vec, self.chunk_vecs, self.row_classes = dtype, vec, chunk_vecs, row_classes
        self.rows_per_thread, self.use_swizzle, self.num_atoms, self.stage_row_bytes = rows_per_thread, use_swizzle, num_atoms, stage_row_bytes


def start_iket_range(name):
    """
    Open a named profiling range; the returned token closes it (txl.cuda.iket.range_end(token[0])).
    """
    token = txl.alloc_local([1], "uint32")
    txl.assign(token[0], txl.cuda.iket.range_start(name))
    return token


def encode_x_tensormap(x, M, dtype, H, elem_bytes, box_cols, use_swizzle, num_atoms):
    """
    Host-side TMA descriptor of the (M, H) rows of x: boxes of STAGE_ROWS rows by box_cols
    columns, or with use_swizzle a 3-d view (column in atom, row, atom) of 128B-swizzled atoms,
    num_atoms atoms per box.
    """
    xmap = txl.stack_alloca("tensormap", 1)
    # arguments: tensormap, dtype, rank, base, sizes, strides in bytes (rank - 1), box, element strides, interleave, swizzle, L2 promotion, oob fill
    if use_swizzle:
        txl.call_packed("runtime.cuTensorMapEncodeTiled", xmap, dtype, 3, x.data, SWIZZLE_ATOM, M, H // SWIZZLE_ATOM, H * elem_bytes, SWIZZLE_ATOM * elem_bytes, SWIZZLE_ATOM, STAGE_ROWS, num_atoms, 1, 1, 1, 0, txl.SwizzleMode.SWIZZLE_128B_ATOM.value, 3, 0)
    else:
        txl.call_packed("runtime.cuTensorMapEncodeTiled", xmap, dtype, 2, x.data, H, M, H * elem_bytes, box_cols, STAGE_ROWS, 1, 1, 0, 0, 3, 0)
    return xmap


def init_ring_barriers(s_full, s_empty, tidx, num_stages):
    """
    Thread 0 initializes every stage's full barrier (one producer arrival) and empty barrier (one
    arrival per consumer warp).
    """
    with txl.if_(tidx == 0), txl.then_():
        for stage in range(num_stages):
            txl.ptx.mbarrier.init.shared.b64(s_full.ptr_to([stage]), txl.uint32(1))
            txl.ptx.mbarrier.init.shared.b64(s_empty.ptr_to([stage]), txl.uint32(CONSUMER_WARPS))
        txl.ptx.fence.mbarrier_init.release.cluster()


def copy_stage(xmap, s_ring, s_full, stage, k, row_start, num_valid_rows, chunk_idx, shape):
    """
    Stage k of a tile: the TMA copy of its STAGE_ROWS rows into ring stage `stage`, completing
    s_full[stage], or a plain arrival when the tile has no rows there.
    """
    vec, chunk_vecs, use_swizzle, num_atoms, stage_row_bytes = shape.vec, shape.chunk_vecs, shape.use_swizzle, shape.num_atoms, shape.stage_row_bytes
    with txl.if_(num_valid_rows > k * STAGE_ROWS):
        with txl.then_():
            txl.ptx.mbarrier.arrive.expect_tx.shared.b64(s_full.ptr_to([stage]), txl.uint32(STAGE_ROWS * chunk_vecs * 16))
            if use_swizzle:
                txl.ptx[TMA_LOAD_3D](s_ring.ptr_to([stage, 0, 0, 0]), txl.address_of(xmap), txl.int32(0), row_start + k * STAGE_ROWS, chunk_idx * num_atoms, txl.cuda.cvta_generic_to_shared(s_full.ptr_to([stage])), txl.uint64(EVICT_FIRST))
            else:
                txl.ptx[TMA_LOAD_2D](s_ring.ptr_to([stage * STAGE_ROWS * (stage_row_bytes // 4)]), txl.address_of(xmap), chunk_idx * (chunk_vecs * vec), row_start + k * STAGE_ROWS, txl.cuda.cvta_generic_to_shared(s_full.ptr_to([stage])), txl.uint64(EVICT_FIRST))
        with txl.else_():
            txl.ptx.mbarrier.arrive.shared.b64(s_full.ptr_to([stage]), txl.uint32(1))


def consumer_coords(tidx, warp, lane, shape):
    """
    (row class, vector column in the chunk, whether the thread owns one) of a consumer thread: one
    warp per row class when the classes and the chunk fit the warps, else row-major over the
    consumer threads.
    """
    chunk_vecs, row_classes = shape.chunk_vecs, shape.row_classes
    if row_classes <= CONSUMER_WARPS and chunk_vecs <= 32:
        col_vec = lane
        row_class = warp
        thread_ok = txl.And(row_class < row_classes, col_vec < chunk_vecs)
    else:
        col_vec = tidx % chunk_vecs
        row_class = tidx // chunk_vecs
        thread_ok = row_class < row_classes
    return row_class, col_vec, thread_ok


def load_stage_rows(words, s_ring, stage, k, row_class, col_vec, thread_ok, num_chunk_vecs, num_valid_rows, shape):
    """
    Load the thread's 16-byte vector of each row of ring stage `stage` in its row class; rows past
    the tile's valid rows read as zero.
    """
    vec, row_classes, rows_per_thread, use_swizzle, stage_row_bytes = shape.vec, shape.row_classes, shape.rows_per_thread, shape.use_swizzle, shape.stage_row_bytes
    # rows i of this stage with (k * STAGE_ROWS + i) % row_classes == row_class, in increasing order
    first_row = (row_class + row_classes - (k * STAGE_ROWS) % row_classes) % row_classes
    for j in range(rows_per_thread):
        stage_row = first_row + j * row_classes
        row_ok = txl.And(txl.And(thread_ok, col_vec < num_chunk_vecs), txl.And(stage_row < STAGE_ROWS, k * STAGE_ROWS + stage_row < num_valid_rows))
        for word_idx in range(4):
            txl.assign(words[j * 4 + word_idx], txl.uint32(0))
        with txl.if_(row_ok), txl.then_():
            if use_swizzle:
                txl.ptx.ld.shared.v4.b32(words[j * 4], words[j * 4 + 1], words[j * 4 + 2], words[j * 4 + 3], s_ring.ptr_to([stage, (col_vec * vec) // SWIZZLE_ATOM, stage_row, (col_vec * vec) % SWIZZLE_ATOM]))
            else:
                txl.ptx.ld.shared.v4.b32(words[j * 4], words[j * 4 + 1], words[j * 4 + 2], words[j * 4 + 3], s_ring.ptr_to([(stage * STAGE_ROWS + stage_row) * (stage_row_bytes // 4) + col_vec * 4]))


def add_pair(packed_add, acc, elem, src, src_idx):
    """
    acc[elem : elem + 2] += src[src_idx : src_idx + 2] with one packed f32x2 add.
    """
    txl.ptx.add.rn.f32x2(packed_add, txl.cuda.make_float2(acc[elem], acc[elem + 1]), txl.cuda.make_float2(src[src_idx], src[src_idx + 1]))
    txl.ptx.mov.b64(acc[elem], acc[elem + 1], packed_add)


def accumulate_rows(acc_sum, row_vals, words, packed_add, shape):
    """
    Widen the loaded stage rows to fp32 and add them to acc_sum in row order.
    """
    dtype, vec, rows_per_thread = shape.dtype, shape.vec, shape.rows_per_thread
    for j in range(rows_per_thread):
        for word_idx in range(4):
            word = words[j * 4 + word_idx]
            if dtype == "float32":
                txl.assign(row_vals[j * vec + word_idx], txl.reinterpret(word, ty="float32"))
            elif dtype == "bfloat16":
                txl.assign(row_vals[j * vec + 2 * word_idx], txl.reinterpret(word << txl.uint32(16), ty="float32"))
                txl.assign(row_vals[j * vec + 2 * word_idx + 1], txl.reinterpret(word & txl.uint32(0xFFFF0000), ty="float32"))
            else:
                txl.idioms.cast_f16x2_to_f32x2(View(row_vals, j * vec), word_idx, word)
        for elem in range(0, vec, 2):
            add_pair(packed_add, acc_sum, elem, row_vals, j * vec + elem)


def store_class_sums(s_class_sums, class_base, acc_sum, row_class, col_vec, thread_ok, shape):
    """
    Row classes after the first leave their column sums in shared memory for row class 0 (half
    class_base of a buffer that alternates per tile), then the consumer warps synchronize.
    """
    vec, chunk_vecs, row_classes = shape.vec, shape.chunk_vecs, shape.row_classes
    if row_classes > 1:
        with txl.if_(txl.And(row_class > 0, thread_ok)), txl.then_():
            for elem in range(vec):
                txl.ptx.st.shared.f32(s_class_sums.ptr_to([class_base + ((row_class - 1) * chunk_vecs + col_vec) * vec + elem]), acc_sum[elem])
        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(CONSUMER_WARPS * 32))


def add_class_sums(acc_sum, s_class_sums, class_base, col_vec, packed_add, shape):
    """
    Row class 0 adds the other classes' column sums to its own, in class order.
    """
    vec, chunk_vecs, row_classes = shape.vec, shape.chunk_vecs, shape.row_classes
    partial_pair = txl.alloc_local([2], "float32")
    for class_idx in range(1, row_classes):
        for elem in range(0, vec, 2):
            txl.ptx.ld.shared.v2.b32(partial_pair[0], partial_pair[1], s_class_sums.ptr_to([class_base + ((class_idx - 1) * chunk_vecs + col_vec) * vec + elem]))
            add_pair(packed_add, acc_sum, elem, partial_pair, 0)


def prefetch_first_unit(xmap, cta_idx, n_ctas, M, num_chunks, chunk_cols, use_swizzle, num_atoms):
    """
    L2 prefetch of the CTA's first unit, guessing compact tile t at row t * TILE (exact when every
    group before it is a multiple of 128 rows); a wrong guess only costs bandwidth.
    """
    # One unit: two or more, or only part of one, start the stream slower.
    for prefetch_idx in range(1):
        unit = cta_idx + prefetch_idx * n_ctas
        with txl.if_((unit // num_chunks) * TILE < M), txl.then_():
            for k in range(TILE // STAGE_ROWS):
                if use_swizzle:
                    txl.ptx["cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint"](txl.address_of(xmap), txl.int32(0), (unit // num_chunks) * TILE + k * STAGE_ROWS, (unit % num_chunks) * num_atoms, txl.uint64(EVICT_FIRST))
                else:
                    txl.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint"](txl.address_of(xmap), (unit % num_chunks) * chunk_cols, (unit // num_chunks) * TILE + k * STAGE_ROWS, txl.uint64(EVICT_FIRST))


def stage_group_ends(s_ends, offs, tidx, M, G, idtype, threads):
    """
    All threads stage the groups' ends, clamped to M, in shared memory.
    """
    with txl.serial((G + threads - 1) // threads) as group_round:
        group_idx = group_round * threads + tidx
        with txl.if_(group_idx < G), txl.then_():
            txl.ptx.st.shared.s32(s_ends.ptr_to([group_idx]), txl.min(load_offset(offs, group_idx, idtype), M))


def prefix_real_tiles(s_ends, s_tile_prefix, warp, lane, G):
    """
    Warp 0 numbers the real tiles compactly: s_tile_prefix[g] = real tiles of the groups before g,
    s_tile_prefix[G] = all of them (the empty slots get no units).
    """
    with txl.if_(warp == 0), txl.then_():
        running = txl.local_scalar("int32")
        group_end, group_start, incl_sum, peer = txl.local_scalar("int32"), txl.local_scalar("int32"), txl.local_scalar("int32"), txl.local_scalar("int32")
        txl.assign(running, txl.int32(0))
        with txl.serial((G + 31) // 32) as group_round:
            group_idx = group_round * 32 + lane
            txl.assign(group_end, txl.int32(0))
            txl.assign(group_start, txl.int32(0))
            with txl.if_(group_idx < G), txl.then_():
                txl.ptx.ld.shared.s32(group_end, s_ends.ptr_to([group_idx]))
                with txl.if_(group_idx > 0), txl.then_():
                    txl.ptx.ld.shared.s32(group_start, s_ends.ptr_to([txl.max(group_idx - 1, 0)]))
            num_group_tiles = (txl.max(group_end, group_start) - group_start + (TILE - 1)) // TILE
            txl.assign(incl_sum, num_group_tiles)
            for dist in (1, 2, 4, 8, 16):
                txl.ptx.shfl_sync.up.b32(peer, incl_sum, txl.uint32(dist), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                txl.assign(incl_sum, incl_sum + txl.Select(lane >= dist, peer, txl.int32(0)))
            with txl.if_(group_idx < G), txl.then_():
                txl.ptx.st.shared.s32(s_tile_prefix.ptr_to([group_idx]), running + incl_sum - num_group_tiles)
            txl.ptx.shfl_sync.idx.b32(peer, incl_sum, txl.uint32(31), txl.uint32(31), txl.uint32(0xFFFFFFFF))
            txl.assign(running, running + peer)
        with txl.if_(lane == 0), txl.then_():
            txl.ptx.st.shared.s32(s_tile_prefix.ptr_to([G]), running)


def load_group_start(s_ends, group_idx, group_start):
    """
    group_start = the previous group's staged end, 0 for group 0.
    """
    with txl.if_(group_idx > 0):
        with txl.then_():
            txl.ptx.ld.shared.s32(group_start, s_ends.ptr_to([txl.max(group_idx - 1, 0)]))
        with txl.else_():
            txl.assign(group_start, txl.int32(0))


def locate_tile(s_ends, s_tile_prefix, tile_idx, lane, G):
    """
    (first row, valid rows, partial slot f(g) + j) of compact tile tile_idx, from the staged group
    ends and real-tile prefix.
    """
    row_start, num_valid_rows = txl.local_scalar("int32"), txl.local_scalar("int32")
    lo, hi, mid, group_start = txl.local_scalar("int32"), txl.local_scalar("int32"), txl.local_scalar("int32"), txl.local_scalar("int32")
    txl.assign(lo, txl.int32(0))
    txl.assign(hi, txl.int32(SMEM_GROUPS))  # replaced by G - 1 before the binary search
    prefix_val, group_idx = txl.local_scalar("int32"), txl.local_scalar("int32")
    # Above the small-G crossover, load one warp of the prefix table in parallel and derive the
    # upper-bound index with one ballot. Binary search is cheaper for G < 16 and handles G > 32.
    with txl.if_(txl.And(G >= 16, G <= 32)):
        with txl.then_():
            lane_prefix = txl.local_scalar("int32")
            txl.assign(lane_prefix, txl.int32(0))
            with txl.if_(lane < G), txl.then_():
                txl.ptx.ld.shared.s32(lane_prefix, s_tile_prefix.ptr_to([lane]))
            ballot = txl.alloc_local([1], "uint32")
            txl.ptx.vote_sync.ballot.b32(ballot[0], txl.And(lane < G, lane_prefix <= tile_idx), txl.uint32(0xFFFFFFFF))
            txl.assign(group_idx, txl.cast(txl.popcount(ballot[0]), "int32") - 1)
        with txl.else_():
            txl.assign(hi, G - 1)
            with txl.while_(lo < hi):
                txl.assign(mid, (lo + hi + 1) >> 1)
                txl.ptx.ld.shared.s32(prefix_val, s_tile_prefix.ptr_to([mid]))
                with txl.if_(prefix_val <= tile_idx):
                    with txl.then_():
                        txl.assign(lo, mid)
                    with txl.else_():
                        txl.assign(hi, mid - 1)
            txl.assign(group_idx, lo)
    load_group_start(s_ends, group_idx, group_start)
    group_end = txl.local_scalar("int32")
    txl.ptx.ld.shared.s32(group_end, s_ends.ptr_to([group_idx]))
    txl.ptx.ld.shared.s32(prefix_val, s_tile_prefix.ptr_to([group_idx]))
    txl.assign(row_start, group_start + (tile_idx - prefix_val) * TILE)
    txl.assign(num_valid_rows, txl.min(group_end - row_start, txl.int32(TILE)))
    part_slot = group_idx + group_start // TILE + (tile_idx - prefix_val)
    return row_start, num_valid_rows, part_slot


@functools.cache
def build_tile_sum_ring(dtype, idtype, H, tagged=False):
    """
    Compact persistent TMA ring: units (real tile, column chunk) interleaved over the CTAs; each
    tile sum goes to its collision-free slot f(g) + j, as fp32 or (tagged) as tag-validated 64-bit
    words for build_tile_reduce_tag.
    """
    vec, NUM_VECS, NUM_CHUNKS, CHUNK_VECS, NUM_STAGES = ring_layout(dtype, H)
    USE_SWIZZLE = dtype == "bfloat16" and H == 2880  # 128B-swizzled 256-column chunks (conflict-free smem reads)
    if USE_SWIZZLE:
        CHUNK_COLS = 256
        CHUNK_VECS = CHUNK_COLS // vec
        NUM_CHUNKS = cdiv(H, CHUNK_COLS)
        NUM_STAGES = max(2, min(16, SMEM_BUDGET // (STAGE_ROWS * CHUNK_COLS * 2 + 8)))
    NUM_ATOMS = (CHUNK_VECS * vec) // SWIZZLE_ATOM if USE_SWIZZLE else 0
    ROW_CLASSES = (CONSUMER_WARPS * 32) // CHUNK_VECS
    STAGE_ROW_BYTES = CHUNK_VECS * 16
    STAGES_PER_TILE = TILE // STAGE_ROWS
    ROWS_PER_THREAD = cdiv(STAGE_ROWS, ROW_CLASSES)  # rows per consumer thread per stage (max)
    THREADS = (CONSUMER_WARPS + 1) * 32
    elem_bytes = 16 // vec
    shape = RingShape(dtype, vec, CHUNK_VECS, ROW_CLASSES, ROWS_PER_THREAD, USE_SWIZZLE, NUM_ATOMS, STAGE_ROW_BYTES)

    @txl.kernel()
    def tile_sum_ring(x: txl.Tensor((txl.int64(),), dtype, data_alignment=1), offs: txl.Tensor((txl.int64(),), idtype, data_alignment=1), part: txl.Tensor((txl.int64(),), txl.u64 if tagged else txl.f32, data_alignment=1), M: txl.int32, G: txl.int32, n_units: txl.int32, n_ctas: txl.int32, tag: txl.int32):
        xmap = encode_x_tensormap(x, M, dtype, H, elem_bytes, CHUNK_VECS * vec, USE_SWIZZLE, NUM_ATOMS)
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=n_ctas, block=(CONSUMER_WARPS + 1) * 32))
        cta_idx = txl.cta_id()
        tidx = txl.thread_id()
        warp, lane = tidx >> 5, tidx & 31

        txl.ptx.griddepcontrol.launch_dependents()
        smem = txl.smem_pool()
        s_full, s_empty = smem.alloc([NUM_STAGES], "uint64", align=8), smem.alloc([NUM_STAGES], "uint64", align=8)
        s_num_valid, s_part_slot, s_unit = smem.alloc([NUM_STAGES], "int32", align=4), smem.alloc([NUM_STAGES], "int32", align=4), smem.alloc([NUM_STAGES], "int32", align=4)
        s_ends, s_tile_prefix = smem.alloc([SMEM_GROUPS], "int32", align=4), smem.alloc([SMEM_GROUPS + 1], "int32", align=4)
        s_class_sums = smem.alloc([2 * max(ROW_CLASSES - 1, 1) * CHUNK_VECS * vec], "float32", align=16)
        if USE_SWIZZLE:
            s_ring = smem.alloc((NUM_STAGES, NUM_ATOMS, STAGE_ROWS, SWIZZLE_ATOM), dtype, swizzle=txl.SwizzleMode.SWIZZLE_128B_ATOM, align=1024)
        else:
            s_ring = smem.alloc([NUM_STAGES * STAGE_ROWS * STAGE_ROW_BYTES // 4], "uint32", align=128)
        pool_counter = part.ptr_to([txl.cast(G + (M + (TILE - 1)) // TILE, "int64") * H])
        tag_hi = txl.cast(txl.cast(tag, "uint32"), "uint64") << txl.uint64(32)
        if TAIL_UNITS > 0:
            with txl.if_(txl.And(cta_idx == 0, tidx == CONSUMER_WARPS * 32)), txl.then_():
                # pool counter, (tag << 32) | claims: a claim that sees another tag (stale scratch, or this store not
                # yet performed) is retried, so the counter needs no zeroed scratch
                txl.ptx["st.relaxed.gpu.global.b64"](pool_counter, tag_hi)
        with txl.if_(tidx == CONSUMER_WARPS * 32), txl.then_():
            txl.ptx.prefetch.tensormap(txl.address_of(xmap))
            prefetch_first_unit(xmap, cta_idx, n_ctas, M, NUM_CHUNKS, CHUNK_VECS * vec, USE_SWIZZLE, NUM_ATOMS)
        init_ring_barriers(s_full, s_empty, tidx, NUM_STAGES)
        stage_group_ends(s_ends, offs, tidx, M, G, idtype, THREADS)
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(THREADS))
        prefix_real_tiles(s_ends, s_tile_prefix, warp, lane, G)
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(THREADS))

        num_real_tiles = txl.local_scalar("int32")
        txl.ptx.ld.shared.s32(num_real_tiles, s_tile_prefix.ptr_to([G]))
        # Equal unit counts: only the fewest CTAs that still take ceil(units / n_ctas) units each stream, so no
        # partial last wave of single units runs alone after the others finish (e.g. 3264 units on 148 SMs: 142).
        num_real_units = num_real_tiles * NUM_CHUNKS
        units_per_cta = txl.max((num_real_units + n_ctas - 1) // n_ctas, txl.int32(1))
        num_busy_ctas = txl.max((num_real_units + units_per_cta - 1) // units_per_cta, txl.int32(1))
        num_my_units = txl.Select(cta_idx < num_busy_ctas, (num_real_units - cta_idx + num_busy_ctas - 1) // num_busy_ctas, txl.int32(0))
        if TAIL_UNITS > 0:
            # static units cta_idx + i * num_busy_ctas for i < num_static, then the pool [pool_start, num_real_units)
            # is claimed one unit at a time
            num_static = txl.max(units_per_cta - TAIL_UNITS, txl.int32(0))
            pool_start = num_busy_ctas * num_static
            my_num_static = txl.Select(cta_idx < num_busy_ctas, num_static, txl.int32(0))
            joins_pool = txl.Or(cta_idx < num_busy_ctas, txl.int32(IDLE_JOIN) == 1)

        with txl.if_(warp == CONSUMER_WARPS):
            with txl.then_():  # producer warp
                role_token = start_iket_range("ring-producer")
                if TAIL_UNITS > 0:
                    unit_iter, unit = txl.local_scalar("int32"), txl.local_scalar("int32")
                    pending_claim, polls = txl.local_scalar("uint64"), txl.local_scalar("int32")
                    claim_hi, claim_lo = txl.local_scalar("uint32"), txl.local_scalar("uint32")

                    def issue_claim():
                        with txl.if_(lane == 0), txl.then_():
                            txl.ptx["atom.relaxed.gpu.global.add.u64"](pending_claim, pool_counter, txl.uint64(1))

                    def broadcast_claim():
                        txl.ptx.shfl_sync.idx.b32(claim_hi, txl.cast(pending_claim >> txl.uint64(32), "uint32"), txl.uint32(0), txl.uint32(31), txl.uint32(0xFFFFFFFF))
                        txl.ptx.shfl_sync.idx.b32(claim_lo, txl.cast(pending_claim & txl.uint64(0xFFFFFFFF), "uint32"), txl.uint32(0), txl.uint32(31), txl.uint32(0xFFFFFFFF))

                    def take_claim():  # the pending claim's unit (warp-uniform), retrying claims that saw a stale tag
                        txl.assign(polls, txl.int32(0))
                        broadcast_claim()
                        with txl.while_(txl.And(claim_hi != txl.cast(tag, "uint32"), polls < POLL_LIMIT)):
                            txl.cuda.nano_sleep(txl.uint32(POLL_SLEEP_NS))
                            issue_claim()
                            broadcast_claim()
                            txl.assign(polls, polls + 1)
                        txl.assign(unit, txl.Select(polls < POLL_LIMIT, pool_start + txl.cast(claim_lo, "int32"), num_real_units))
                        with txl.if_(unit < num_real_units), txl.then_():
                            issue_claim()  # one claim ahead

                    txl.assign(unit_iter, txl.int32(0))
                    txl.assign(unit, num_real_units)
                    with txl.if_(my_num_static > 0):
                        with txl.then_():
                            txl.assign(unit, cta_idx)
                        with txl.else_():
                            with txl.if_(joins_pool), txl.then_():
                                issue_claim()
                                take_claim()
                    loop = txl.while_(unit < num_real_units)
                else:
                    loop = txl.serial(num_my_units)
                with loop as loop_iter:
                    if TAIL_UNITS > 0:
                        with txl.if_(txl.And(my_num_static > 0, unit_iter == txl.max(my_num_static - CLAIM_LEAD, txl.int32(0)))), txl.then_():
                            issue_claim()
                    else:
                        unit_iter = loop_iter
                        unit = cta_idx + unit_iter * num_busy_ctas
                    tile_idx, chunk_idx = unit // NUM_CHUNKS, unit % NUM_CHUNKS
                    row_start, num_valid_rows, part_slot = locate_tile(s_ends, s_tile_prefix, tile_idx, lane, G)
                    for k in range(STAGES_PER_TILE):
                        stage_seq = unit_iter * STAGES_PER_TILE + k
                        stage, phase = stage_seq % NUM_STAGES, (stage_seq // NUM_STAGES) & 1
                        txl.cuda.mbarrier_wait(s_empty.ptr_to([stage]), phase ^ 1)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        with txl.if_(lane == 0), txl.then_():
                            txl.ptx.st.shared.s32(s_num_valid.ptr_to([stage]), num_valid_rows)
                            txl.ptx.st.shared.s32(s_part_slot.ptr_to([stage]), part_slot)
                            txl.ptx.st.shared.s32(s_unit.ptr_to([stage]), unit)
                            copy_stage(xmap, s_ring, s_full, stage, k, row_start, num_valid_rows, chunk_idx, shape)
                    if TAIL_UNITS > 0:
                        txl.assign(unit_iter, unit_iter + 1)
                        with txl.if_(unit_iter < my_num_static):
                            with txl.then_():
                                txl.assign(unit, cta_idx + unit_iter * num_busy_ctas)
                            with txl.else_():
                                take_claim()
                if TAIL_UNITS > 0:  # end marker for the consumers
                    stage_seq = unit_iter * STAGES_PER_TILE
                    stage = stage_seq % NUM_STAGES
                    txl.cuda.mbarrier_wait(s_empty.ptr_to([stage]), ((stage_seq // NUM_STAGES) & 1) ^ 1)
                    with txl.if_(lane == 0), txl.then_():
                        txl.ptx.st.shared.s32(s_unit.ptr_to([stage]), txl.int32(-1))
                        txl.ptx.mbarrier.arrive.shared.b64(s_full.ptr_to([stage]), txl.uint32(1))
                txl.cuda.iket.range_end(role_token[0])
            with txl.else_():  # consumer warps
                role_token = start_iket_range("ring-consumer")
                row_class, col_vec, thread_ok = consumer_coords(tidx, warp, lane, shape)
                acc_sum, row_vals, words = txl.alloc_local([vec], "float32"), txl.alloc_local([ROWS_PER_THREAD * vec], "float32"), txl.alloc_local([ROWS_PER_THREAD * 4], "uint32")
                num_valid_rows, part_slot, packed_add = txl.local_scalar("int32"), txl.local_scalar("int32"), txl.local_scalar("uint64")
                if TAIL_UNITS > 0:
                    unit_iter, unit = txl.local_scalar("int32"), txl.local_scalar("int32")
                    txl.assign(unit_iter, txl.int32(0))
                    txl.cuda.mbarrier_wait(s_full.ptr_to([0]), 0)
                    txl.ptx.ld.shared.s32(unit, s_unit.ptr_to([0]))
                    loop = txl.while_(unit >= 0)
                else:
                    loop = txl.serial(num_my_units)
                with loop as loop_iter:
                    unit_token = start_iket_range("ring-unit")
                    if TAIL_UNITS == 0:
                        unit_iter = loop_iter
                        unit = cta_idx + unit_iter * num_busy_ctas
                    chunk_idx = unit % NUM_CHUNKS
                    num_chunk_vecs = txl.min(txl.int32(CHUNK_VECS), NUM_VECS - chunk_idx * CHUNK_VECS)
                    for elem in range(vec):
                        txl.assign(acc_sum[elem], txl.float32(0))
                    for k in range(STAGES_PER_TILE):
                        stage_seq = unit_iter * STAGES_PER_TILE + k
                        stage, phase = stage_seq % NUM_STAGES, (stage_seq // NUM_STAGES) & 1
                        txl.cuda.mbarrier_wait(s_full.ptr_to([stage]), phase)
                        txl.ptx.ld.shared.s32(num_valid_rows, s_num_valid.ptr_to([stage]))
                        if k == 0:
                            txl.ptx.ld.shared.s32(part_slot, s_part_slot.ptr_to([stage]))
                        load_stage_rows(words, s_ring, stage, k, row_class, col_vec, thread_ok, num_chunk_vecs, num_valid_rows, shape)
                        txl.cuda.warp_sync()
                        with txl.if_(lane == 0), txl.then_():
                            txl.ptx.mbarrier.arrive.shared.b64(s_empty.ptr_to([stage]), txl.uint32(1))
                        accumulate_rows(acc_sum, row_vals, words, packed_add, shape)

                    class_base = (unit_iter & 1) * (max(ROW_CLASSES - 1, 1) * CHUNK_VECS * vec)
                    store_class_sums(s_class_sums, class_base, acc_sum, row_class, col_vec, thread_ok, shape)
                    with txl.if_(txl.And(txl.And(row_class == 0, col_vec < num_chunk_vecs), num_valid_rows > 0)), txl.then_():
                        add_class_sums(acc_sum, s_class_sums, class_base, col_vec, packed_add, shape)
                        if tagged:
                            store_tagged(part, txl.cast(part_slot, "int64") * H + txl.cast((chunk_idx * CHUNK_VECS + col_vec) * vec, "int64"), vec, acc_sum, tag)
                        else:
                            store_vec(part, txl.cast(part_slot, "int64") * H + txl.cast((chunk_idx * CHUNK_VECS + col_vec) * vec, "int64"), "float32", vec, acc_sum)
                    txl.cuda.iket.range_end(unit_token[0])
                    if TAIL_UNITS > 0:
                        txl.assign(unit_iter, unit_iter + 1)
                        next_seq = unit_iter * STAGES_PER_TILE
                        txl.cuda.mbarrier_wait(s_full.ptr_to([next_seq % NUM_STAGES]), (next_seq // NUM_STAGES) & 1)
                        txl.ptx.ld.shared.s32(unit, s_unit.ptr_to([next_seq % NUM_STAGES]))
                txl.cuda.iket.range_end(role_token[0])

    return tile_sum_ring


@functools.cache
def build_group_sum_ring(dtype, idtype, odtype, H):
    """
    One CTA per (group, column chunk), accumulating 128-row tile sums in order.
    """
    vec, NUM_VECS, NUM_CHUNKS, CHUNK_VECS, NUM_STAGES = group_ring_layout(dtype, H)
    USE_SWIZZLE = dtype == "bfloat16" and H == 2880
    if USE_SWIZZLE:
        CHUNK_COLS = 256
        CHUNK_VECS = CHUNK_COLS // vec
        NUM_CHUNKS = cdiv(H, CHUNK_COLS)
        NUM_STAGES = max(2, min(16, DIRECT_SMEM // (STAGE_ROWS * CHUNK_COLS * 2 + 8)))
    NUM_ATOMS = (CHUNK_VECS * vec) // SWIZZLE_ATOM if USE_SWIZZLE else 0
    ROW_CLASSES = (CONSUMER_WARPS * 32) // CHUNK_VECS
    STAGE_ROW_BYTES = CHUNK_VECS * 16
    STAGES_PER_TILE = TILE // STAGE_ROWS
    ROWS_PER_THREAD = cdiv(STAGE_ROWS, ROW_CLASSES)
    THREADS = (CONSUMER_WARPS + 1) * 32
    CHUNK_COLS = CHUNK_VECS * vec
    elem_bytes = 16 // vec
    shape = RingShape(dtype, vec, CHUNK_VECS, ROW_CLASSES, ROWS_PER_THREAD, USE_SWIZZLE, NUM_ATOMS, STAGE_ROW_BYTES)

    @txl.kernel()
    def group_sum_ring(x: txl.Tensor((txl.int64(),), dtype, data_alignment=1), offs: txl.Tensor((txl.int64(),), idtype, data_alignment=1), out: txl.Tensor((txl.int64(),), odtype, data_alignment=1), M: txl.int32, G: txl.int32):
        xmap = encode_x_tensormap(x, M, dtype, H, elem_bytes, CHUNK_COLS, USE_SWIZZLE, NUM_ATOMS)
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=G * NUM_CHUNKS, block=(CONSUMER_WARPS + 1) * 32))
        unit = txl.cta_id()
        group_idx, chunk_idx = unit // NUM_CHUNKS, unit % NUM_CHUNKS
        tidx = txl.thread_id()
        warp, lane = tidx >> 5, tidx & 31

        smem = txl.smem_pool()
        s_full, s_empty = smem.alloc([NUM_STAGES], "uint64", align=8), smem.alloc([NUM_STAGES], "uint64", align=8)
        s_num_valid = smem.alloc([NUM_STAGES], "int32", align=4)
        s_class_sums = smem.alloc([2 * max(ROW_CLASSES - 1, 1) * CHUNK_VECS * vec], "float32", align=16)
        if USE_SWIZZLE:
            s_ring = smem.alloc((NUM_STAGES, NUM_ATOMS, STAGE_ROWS, SWIZZLE_ATOM), dtype, swizzle=txl.SwizzleMode.SWIZZLE_128B_ATOM, align=1024)
        else:
            s_ring = smem.alloc([NUM_STAGES * STAGE_ROWS * STAGE_ROW_BYTES // 4], "uint32", align=128)
        group_start, group_end = group_bounds(offs, group_idx, M, idtype)  # issued before the barrier setup: overlaps its latency
        with txl.if_(tidx == CONSUMER_WARPS * 32), txl.then_():
            txl.ptx.prefetch.tensormap(txl.address_of(xmap))
        init_ring_barriers(s_full, s_empty, tidx, NUM_STAGES)
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(THREADS))
        num_tiles = (group_end - group_start + (TILE - 1)) // TILE

        with txl.if_(warp == CONSUMER_WARPS):
            with txl.then_():  # producer warp
                role_token = start_iket_range("group-producer")
                with txl.serial(num_tiles) as tile_iter:
                    row_start = group_start + tile_iter * TILE
                    num_valid_rows = txl.min(group_end - row_start, txl.int32(TILE))
                    for k in range(STAGES_PER_TILE):
                        stage_seq = tile_iter * STAGES_PER_TILE + k
                        stage, phase = stage_seq % NUM_STAGES, (stage_seq // NUM_STAGES) & 1
                        txl.cuda.mbarrier_wait(s_empty.ptr_to([stage]), phase ^ 1)
                        txl.ptx.fence.proxy.async_.shared__cta()
                        with txl.if_(lane == 0), txl.then_():
                            txl.ptx.st.shared.s32(s_num_valid.ptr_to([stage]), num_valid_rows)
                            copy_stage(xmap, s_ring, s_full, stage, k, row_start, num_valid_rows, chunk_idx, shape)
                txl.cuda.iket.range_end(role_token[0])
            with txl.else_():  # consumer warps
                role_token = start_iket_range("group-consumer")
                row_class, col_vec, thread_ok = consumer_coords(tidx, warp, lane, shape)
                num_chunk_vecs = txl.min(txl.int32(CHUNK_VECS), NUM_VECS - chunk_idx * CHUNK_VECS)
                acc_sum, acc_group = txl.alloc_local([vec], "float32"), txl.alloc_local([vec], "float32")
                row_vals, words = txl.alloc_local([ROWS_PER_THREAD * vec], "float32"), txl.alloc_local([ROWS_PER_THREAD * 4], "uint32")
                num_valid_rows, packed_add = txl.local_scalar("int32"), txl.local_scalar("uint64")
                for elem in range(vec):
                    txl.assign(acc_group[elem], txl.float32(0))
                with txl.serial(num_tiles) as tile_iter:
                    tile_token = start_iket_range("group-tile")
                    for elem in range(vec):
                        txl.assign(acc_sum[elem], txl.float32(0))
                    for k in range(STAGES_PER_TILE):
                        stage_seq = tile_iter * STAGES_PER_TILE + k
                        stage, phase = stage_seq % NUM_STAGES, (stage_seq // NUM_STAGES) & 1
                        txl.cuda.mbarrier_wait(s_full.ptr_to([stage]), phase)
                        txl.ptx.ld.shared.s32(num_valid_rows, s_num_valid.ptr_to([stage]))
                        load_stage_rows(words, s_ring, stage, k, row_class, col_vec, thread_ok, num_chunk_vecs, num_valid_rows, shape)
                        txl.cuda.warp_sync()
                        with txl.if_(lane == 0), txl.then_():
                            txl.ptx.mbarrier.arrive.shared.b64(s_empty.ptr_to([stage]), txl.uint32(1))
                        accumulate_rows(acc_sum, row_vals, words, packed_add, shape)

                    class_base = (tile_iter & 1) * (max(ROW_CLASSES - 1, 1) * CHUNK_VECS * vec)
                    store_class_sums(s_class_sums, class_base, acc_sum, row_class, col_vec, thread_ok, shape)
                    with txl.if_(txl.And(row_class == 0, col_vec < num_chunk_vecs)), txl.then_():
                        add_class_sums(acc_sum, s_class_sums, class_base, col_vec, packed_add, shape)
                        for elem in range(0, vec, 2):
                            add_pair(packed_add, acc_group, elem, acc_sum, elem)
                    txl.cuda.iket.range_end(tile_token[0])
                with txl.if_(txl.And(row_class == 0, col_vec < num_chunk_vecs)), txl.then_():
                    store_vec(out, txl.cast(group_idx, "int64") * H + txl.cast((chunk_idx * CHUNK_VECS + col_vec) * vec, "int64"), odtype, vec, acc_group)
                txl.cuda.iket.range_end(role_token[0])

    return group_sum_ring


@functools.cache
def sm_count():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


@functools.cache
def build_tile_reduce(idtype, odtype, H):
    """
    Ordered reduce of fp32 tile sums for long groups: per round each CTA stages REDUCE_LANES *
    REDUCE_UNROLL tiles of its columns in shared memory and tile lane 0 adds them in row order. One
    column per lane is faster for the long low-G rows.
    """
    COLS_PER_LANE = 1
    CTA_COLS = REDUCE_COLS * COLS_PER_LANE
    NUM_COL_BLOCKS = cdiv(H, CTA_COLS)
    TILES_PER_ROUND = REDUCE_LANES * REDUCE_UNROLL

    @txl.kernel()
    def tile_reduce(part: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), offs: txl.Tensor((txl.int64(),), idtype, data_alignment=1), out: txl.Tensor((txl.int64(),), odtype, data_alignment=1), M: txl.int32, G: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=G * NUM_COL_BLOCKS, block=REDUCE_LANES * 32, programmatic_stream_serialization=True))
        unit = txl.cta_id()
        group_idx, col_block = unit // NUM_COL_BLOCKS, unit % NUM_COL_BLOCKS
        tidx = txl.thread_id()
        tile_lane = tidx >> 5
        col = col_block * CTA_COLS + (tidx & 31) * COLS_PER_LANE
        smem = txl.smem_pool()
        s_tiles = smem.alloc([TILES_PER_ROUND * CTA_COLS], "float32", align=16)
        txl.ptx.griddepcontrol.wait()
        group_start, group_end = group_bounds(offs, group_idx, M, idtype)
        num_tiles = (group_end - group_start + (TILE - 1)) // TILE
        first_slot = group_idx + group_start // TILE
        acc_sum = txl.alloc_local([COLS_PER_LANE], "float32")
        for k in range(COLS_PER_LANE):
            txl.assign(acc_sum[k], txl.float32(0))
        tile_vals, partial = txl.alloc_local([REDUCE_UNROLL * COLS_PER_LANE], "float32"), txl.alloc_local([COLS_PER_LANE], "float32")
        packed_add = txl.local_scalar("uint64")

        with txl.serial((num_tiles + TILES_PER_ROUND - 1) // TILES_PER_ROUND) as round_idx:
            tile_base = round_idx * TILES_PER_ROUND
            for i in range(REDUCE_UNROLL):
                group_tile = tile_base + tile_lane * REDUCE_UNROLL + i
                load_vec(part, txl.cast(first_slot + group_tile, "int64") * H + txl.cast(col, "int64"), "float32", COLS_PER_LANE, 4, View(tile_vals, i * COLS_PER_LANE), txl.And(group_tile < num_tiles, col < H))
            for i in range(REDUCE_UNROLL):
                s_addr = s_tiles.ptr_to([(tile_lane * REDUCE_UNROLL + i) * CTA_COLS + (tidx & 31) * COLS_PER_LANE])
                if COLS_PER_LANE == 2:
                    txl.ptx.st.shared.v2.b32(s_addr, txl.reinterpret(tile_vals[i * COLS_PER_LANE], ty="uint32"), txl.reinterpret(tile_vals[i * COLS_PER_LANE + 1], ty="uint32"))
                else:
                    txl.ptx.st.shared.f32(s_addr, tile_vals[i])
            txl.ptx.bar.sync(txl.uint32(1), txl.uint32(REDUCE_LANES * 32))
            with txl.if_(tile_lane == 0), txl.then_():
                for i in range(TILES_PER_ROUND):
                    with txl.if_(tile_base + i < num_tiles), txl.then_():
                        s_addr = s_tiles.ptr_to([i * CTA_COLS + (tidx & 31) * COLS_PER_LANE])
                        if COLS_PER_LANE == 2:
                            txl.ptx.ld.shared.v2.b32(partial[0], partial[1], s_addr)
                            add_pair(packed_add, acc_sum, 0, partial, 0)
                        else:
                            txl.ptx.ld.shared.f32(partial[0], s_addr)
                            txl.assign(acc_sum[0], acc_sum[0] + partial[0])
            txl.ptx.bar.sync(txl.uint32(1), txl.uint32(REDUCE_LANES * 32))
        with txl.if_(txl.And(tile_lane == 0, col < H)), txl.then_():
            store_vec(out, txl.cast(group_idx, "int64") * H + txl.cast(col, "int64"), odtype, COLS_PER_LANE, acc_sum)

    return tile_reduce


@functools.cache
def build_tile_reduce_wide(idtype, odtype, H):
    """
    Ordered reduce of fp32 tile sums for short groups: one thread per column pair loads UNROLL tiles
    at a time and adds them in row order.
    """
    vec = math.gcd(H, 2)
    NUM_VECS = H // vec
    THREADS = 128
    NUM_CHUNKS = cdiv(NUM_VECS, THREADS)
    UNROLL = 32

    @txl.kernel()
    def tile_reduce(part: txl.Tensor((txl.int64(),), txl.f32, data_alignment=1), offs: txl.Tensor((txl.int64(),), idtype, data_alignment=1), out: txl.Tensor((txl.int64(),), odtype, data_alignment=1), M: txl.int32, G: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=G * NUM_CHUNKS, block=(THREADS // 32) * 32, programmatic_stream_serialization=True))
        unit = txl.cta_id()
        group_idx, chunk_idx = unit // NUM_CHUNKS, unit % NUM_CHUNKS
        tidx = txl.thread_id()
        vec_idx = chunk_idx * THREADS + tidx
        txl.ptx.griddepcontrol.wait()
        with txl.if_(vec_idx < NUM_VECS), txl.then_():
            group_start, group_end = group_bounds(offs, group_idx, M, idtype)
            num_tiles = (group_end - group_start + (TILE - 1)) // TILE
            first_slot = group_idx + group_start // TILE
            acc_sum = txl.alloc_local([vec], "float32")
            for k in range(vec):
                txl.assign(acc_sum[k], txl.float32(0))
            tile_vals = txl.alloc_local([UNROLL * vec], "float32")
            with txl.serial((num_tiles + UNROLL - 1) // UNROLL) as round_idx:
                for i in range(UNROLL):
                    group_tile = round_idx * UNROLL + i
                    part_idx = txl.cast(first_slot + group_tile, "int64") * H + txl.cast(vec_idx * vec, "int64")
                    load_vec(part, part_idx, "float32", vec, 4, View(tile_vals, i * vec), group_tile < num_tiles)
                for i in range(UNROLL):
                    group_tile = round_idx * UNROLL + i
                    with txl.if_(group_tile < num_tiles), txl.then_():
                        for k in range(vec):
                            txl.assign(acc_sum[k], acc_sum[k] + tile_vals[i * vec + k])
            store_vec(out, txl.cast(group_idx, "int64") * H + txl.cast(vec_idx * vec, "int64"), odtype, vec, acc_sum)

    return tile_reduce


@functools.cache
def build_tile_reduce_tag(idtype, odtype, H, COL_WARPS, TILE_LANES, UNROLL):
    """
    Ordered reduce over tagged tile words, gated by tag detection instead of griddepcontrol.wait:
    one thread polls the chain's last tile word (the ring writes tiles in row order); then the CTA
    loads up to TILE_LANES * UNROLL tiles at once (one column per lane, every word's tag checked, a
    rare late word re-polled), stages them in shared memory and tile lane 0 adds them in row order
    from zero. Absent tiles stage +0.0, which leaves the running sum unchanged (it starts at +0 and
    can never be -0), so the fold needs no per-tile guard.
    """
    CTA_COLS = 32 * COL_WARPS
    NUM_COL_BLOCKS = cdiv(H, CTA_COLS)
    TILES_PER_ROUND = TILE_LANES * UNROLL
    THREADS = CTA_COLS * TILE_LANES

    @txl.kernel()
    def tile_reduce_tag(part: txl.Tensor((txl.int64(),), txl.u64, data_alignment=1), offs: txl.Tensor((txl.int64(),), idtype, data_alignment=1), out: txl.Tensor((txl.int64(),), odtype, data_alignment=1), M: txl.int32, G: txl.int32, tag: txl.int32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=G * NUM_COL_BLOCKS, block=(COL_WARPS * TILE_LANES) * 32, programmatic_stream_serialization=True))
        unit = txl.cta_id()
        group_idx, col_block = unit // NUM_COL_BLOCKS, unit % NUM_COL_BLOCKS
        tidx = txl.thread_id()
        col_lane, tile_lane = tidx % CTA_COLS, tidx // CTA_COLS
        col = col_block * CTA_COLS + col_lane
        tag_u64 = txl.cast(txl.cast(tag, "uint32"), "uint64")
        smem = txl.smem_pool()
        s_tiles = smem.alloc([TILES_PER_ROUND * CTA_COLS], "float32", align=16)
        group_start, group_end = group_bounds(offs, group_idx, M, idtype)
        num_tiles = (group_end - group_start + (TILE - 1)) // TILE
        first_slot = group_idx + group_start // TILE
        polls, has_stale = txl.local_scalar("int32"), txl.local_scalar("int32")

        with txl.if_(txl.And(tidx == 0, num_tiles > 0)), txl.then_():
            last_word = txl.local_scalar("uint64")
            last_idx = txl.cast(first_slot + num_tiles - 1, "int64") * H + txl.cast(col_block * CTA_COLS, "int64")
            txl.ptx["ld.relaxed.gpu.global.b64"](last_word, part.ptr_to([last_idx]))
            txl.assign(polls, txl.int32(0))
            with txl.while_(txl.And((last_word >> txl.uint64(32)) != tag_u64, polls < POLL_LIMIT)):
                txl.cuda.nano_sleep(txl.uint32(POLL_SLEEP_NS))
                txl.ptx["ld.relaxed.gpu.global.b64"](last_word, part.ptr_to([last_idx]))
                txl.assign(polls, polls + 1)
        txl.ptx.bar.sync(txl.uint32(1), txl.uint32(THREADS))

        acc_sum = txl.alloc_local([1], "float32")
        txl.assign(acc_sum[0], txl.float32(0))
        words, tile_vals = txl.alloc_local([UNROLL], "uint64"), txl.alloc_local([TILES_PER_ROUND], "float32")
        with txl.serial((num_tiles + TILES_PER_ROUND - 1) // TILES_PER_ROUND) as round_idx:
            tile_base = round_idx * TILES_PER_ROUND
            for i in range(UNROLL):
                group_tile = tile_base + tile_lane * UNROLL + i
                txl.assign(words[i], tag_u64 << txl.uint64(32))  # absent tiles / columns: +0.0 with a valid tag
                with txl.if_(txl.And(group_tile < num_tiles, col < H)), txl.then_():
                    txl.ptx["ld.relaxed.gpu.global.b64"](words[i], part.ptr_to([txl.cast(first_slot + group_tile, "int64") * H + txl.cast(col, "int64")]))
            # words whose tag is not this call's yet are re-loaded together, one round trip per round
            txl.assign(polls, txl.int32(0))
            txl.assign(has_stale, txl.int32(1))
            with txl.while_(txl.And(has_stale == 1, polls < POLL_LIMIT)):
                txl.assign(has_stale, txl.int32(0))
                for i in range(UNROLL):
                    with txl.if_((words[i] >> txl.uint64(32)) != tag_u64), txl.then_():
                        txl.assign(has_stale, txl.int32(1))
                with txl.if_(has_stale == 1), txl.then_():
                    txl.cuda.nano_sleep(txl.uint32(POLL_SLEEP_NS))
                    for i in range(UNROLL):
                        group_tile = tile_base + tile_lane * UNROLL + i
                        with txl.if_((words[i] >> txl.uint64(32)) != tag_u64), txl.then_():
                            txl.ptx["ld.relaxed.gpu.global.b64"](words[i], part.ptr_to([txl.cast(first_slot + group_tile, "int64") * H + txl.cast(col, "int64")]))
                    txl.assign(polls, polls + 1)
            for i in range(UNROLL):
                value = txl.reinterpret(txl.cast(words[i] & txl.uint64(0xFFFFFFFF), "uint32"), ty="float32")
                value = txl.Select(polls < POLL_LIMIT, value, txl.reinterpret(txl.uint32(0x7FC00000), ty="float32"))
                if TILE_LANES > 1:
                    txl.ptx.st.shared.f32(s_tiles.ptr_to([(tile_lane * UNROLL + i) * CTA_COLS + col_lane]), value)
                else:
                    txl.assign(tile_vals[i], value)
            if TILE_LANES > 1:
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(THREADS))
                with txl.if_(tile_lane == 0), txl.then_():
                    for i in range(TILES_PER_ROUND):
                        txl.ptx.ld.shared.f32(tile_vals[i], s_tiles.ptr_to([i * CTA_COLS + col_lane]))
                    for i in range(TILES_PER_ROUND):
                        txl.assign(acc_sum[0], acc_sum[0] + tile_vals[i])
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(THREADS))
            else:
                for i in range(TILES_PER_ROUND):
                    txl.assign(acc_sum[0], acc_sum[0] + tile_vals[i])
        with txl.if_(txl.And(tile_lane == 0, col < H)), txl.then_():
            store_vec(out, txl.cast(group_idx, "int64") * H + txl.cast(col, "int64"), odtype, 1, acc_sum)

    return tile_reduce_tag


@functools.cache
def compiled_kernel(which, *key):
    kernel = {"sum": build_tile_sum, "ring": build_tile_sum_ring, "group": build_group_sum_ring, "reduce": build_tile_reduce, "reduce_wide": build_tile_reduce_wide, "reduce_tag": build_tile_reduce_tag}[which](*key)
    # the reduce kernels ("reduce", "reduce_wide", "reduce_tag") declare programmatic dependent launch in
    # their LaunchConfig: resident while the ring streams
    return entry_function(kernel.compile(TARGET))


def segment_sum_sm100(x: torch.Tensor, grouped_mm_offs: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """
    Per-group column sums of the group-sorted rows of x (see the picker's contract), as a new (G, H)
    tensor of out_dtype.
    """
    assert x.is_contiguous(), "x must be contiguous"
    M, H = x.shape
    G = grouped_mm_offs.shape[0]
    out = torch.empty((G, H), dtype=out_dtype, device=x.device)
    if G == 0 or H == 0:
        return out
    if x.data_ptr() % 16:
        x = x.clone()  # the kernels read 16-byte vectors from the base: an aligned copy, the same bits
    if grouped_mm_offs.dtype != torch.int32 or not grouped_mm_offs.is_contiguous():
        # The kernels read contiguous int32 or int64 ends below 2^31; any other integer dtype or stride,
        # and int64 ends past the int32 range, become contiguous int32 ends clamped to M (one small
        # device-side conversion, no host synchronization).
        wide = grouped_mm_offs.to(torch.int64)
        if grouped_mm_offs.dtype == torch.uint64:
            wide = torch.where(wide < 0, M, wide)  # ends at or past 2^63 wrapped negative: past M
        grouped_mm_offs = wide.clamp(max=M).to(torch.int32)
    x_name, index_name, out_name = DTYPE_NAME[x.dtype], INDEX_DTYPE_NAME[grouped_mm_offs.dtype], DTYPE_NAME[out_dtype]
    ring = ring_layout(x_name, H)
    x_flat = x.view(-1) if M > 0 else torch.empty(1, dtype=x.dtype, device=x.device)
    # Many groups: one CTA per (group, chunk) accumulates the group's tiles in order (no second kernel) when those
    # CTAs fit one wave (two per SM) or the groups are short; otherwise long, uneven groups would leave a tail
    # and the compact tile ring + ordered reduce balances better.
    if M > 0 and ring is not None and G >= DIRECT_MIN_GROUPS:
        num_direct_chunks = group_ring_layout(x_name, H)[2]
        if G * num_direct_chunks <= 2 * sm_count() or M <= SHORT_ROWS * G:
            compiled_kernel("group", x_name, index_name, out_name, H)(x_flat, grouped_mm_offs, out.view(-1), M, G)
            return out
    num_slots = G + cdiv(M, TILE)
    if M > 0 and ring is not None and G <= SMEM_GROUPS:
        # compact ring publishes tagged tile words; the reduce (PDL, resident early) starts on each chain's last word
        part, tag = tagged_scratch(num_slots * H + 1, x.device)  # + the pool counter
        reduce_shape = TAG_REDUCE_LONG if M >= LONG_TILES * TILE * G else TAG_REDUCE_SHORT
        num_units = num_slots * ring[2]  # upper bound (the kernel schedules only real tiles)
        compiled_kernel("ring", x_name, index_name, H, True)(x_flat, grouped_mm_offs, part.view(-1), M, G, num_units, min(num_units, CTAS_PER_SM * sm_count()), tag)
        compiled_kernel("reduce_tag", index_name, out_name, H, *reduce_shape)(part.view(-1), grouped_mm_offs, out.view(-1), M, G, tag)
        return out
    part = torch.empty((max(num_slots, 1), H), dtype=torch.float32, device=x.device)
    # long groups: smem-staged chunked reduce; short groups: one thread per 2 columns
    reduce_kernel = compiled_kernel("reduce" if M >= LONG_TILES * TILE * G else "reduce_wide", index_name, out_name, H)
    if M > 0:
        vec, _ = vector_width(x_name, H)
        compiled_kernel("sum", x_name, index_name, H)(x_flat, grouped_mm_offs, part.view(-1), M, G, num_slots * tile_sum_layout(H, vec)[1])
    reduce_kernel(part.view(-1), grouped_mm_offs, out.view(-1), M, G)
    return out
