"""Thread-safety regression test for the shared pinned-buffer race in
``get_pinned_buffer`` / ``scatter_for_grouped_gemm``.

Under DualPipeV, the MoE forward (``scatter_for_grouped_gemm`` and
``moe_ep_prepare_dispatch``) can run on the autograd backward worker thread (activation
recomputation) concurrently with the main-thread forward. Both read host-side metadata
(``ks``, the EP splits) back from the GPU through the process-wide cached pinned buffer
returned by ``get_pinned_buffer``. If that buffer is shared across threads, the two
concurrent callers copy into and read from the *same* host buffer, so one caller's
``.tolist()`` observes the other's data -> corrupted ``ks`` -> invalid grouped-GEMM
offsets -> out-of-bounds GPU access downstream.

These tests fail on the shared-buffer implementation and pass once the buffer (and the
copy) is per-thread.
"""

import threading

import pytest
import torch

from pithtrain.operators.token_scatter import get_pinned_buffer, scatter_for_grouped_gemm

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

PADDING_ALIGNMENT = 128


def _expected_ks(expert_idxs: torch.Tensor, num_groups: int) -> list[int]:
    """Ground-truth per-group padded sizes: round each group's token count up to 128."""
    counts = torch.bincount(expert_idxs, minlength=num_groups).cpu()
    a = PADDING_ALIGNMENT
    return [((int(c) + a - 1) // a) * a for c in counts.tolist()]


@requires_cuda
def test_get_pinned_buffer_is_thread_local():
    """Two threads requesting the same (name, dtype, numel) must get DISTINCT buffers,
    otherwise concurrent callers clobber each other's host data."""
    name, numel, dtype = "regr_thread_local", 32, torch.int32
    main_ptr = get_pinned_buffer(name, numel, dtype).data_ptr()

    worker_ptr: dict[str, int] = {}

    def grab() -> None:
        worker_ptr["p"] = get_pinned_buffer(name, numel, dtype).data_ptr()

    t = threading.Thread(target=grab)
    t.start()
    t.join()

    assert worker_ptr["p"] != main_ptr, (
        "get_pinned_buffer handed the same buffer to two threads; concurrent MoE "
        "forwards (main thread + autograd recompute thread) would clobber it"
    )


@requires_cuda
def test_scatter_for_grouped_gemm_concurrent_ks_is_correct():
    """Two threads run scatter_for_grouped_gemm concurrently (mimicking DualPipeV's
    main-thread forward overlapping an autograd-thread recompute). The host-side ``ks``
    each thread gets back must always match its own ground-truth per-group padded counts;
    a shared pinned buffer makes one thread read the other's ks."""
    num_groups, hidden, iters = 32, 256, 200
    errors: dict[int, int] = {}

    def worker(tid: int) -> None:
        gen = torch.Generator(device="cuda").manual_seed(1000 + tid)
        drop = (tid * 7 + 3) % num_groups  # each thread leaves a DIFFERENT group empty
        bad = 0
        for _ in range(iters):
            m = int(torch.randint(200, 1200, (1,), generator=gen, device="cuda").item())
            idxs = torch.randint(0, num_groups, (m,), generator=gen, device="cuda")
            idxs = idxs[idxs != drop].to(torch.int64)
            x = torch.randn(idxs.shape[0], hidden, device="cuda", dtype=torch.bfloat16)
            _, _, _, ks, _ = scatter_for_grouped_gemm(x, idxs, num_groups)
            if ks != _expected_ks(idxs, num_groups):
                bad += 1
        errors[tid] = bad

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(errors.values()) == 0, (
        f"ks corrupted under concurrent scatter (shared pinned buffer): "
        f"per-thread mismatch counts = {errors}"
    )
