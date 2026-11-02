import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["ep_dispatch_recv_idx", "ep_dispatch_send_idx"]


def ep_dispatch_send_idx(topk_ids: torch.Tensor, num_experts: int, ep_size: int, experts_per_rank: int, meta_out: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
    """
    Sender-side metadata of a deduplicated EP dispatch (each token is sent once to every EP rank
    hosting one of its experts). Returns (tokens_per_ep_rank, dedup_tokens_per_gpu,
    dispatch_token_idxs, idxs, expand_idx, send_meta), all int64.

    - topk_ids: (m, k), values in [0, num_experts), not modified. A token's k experts must be
      distinct. num_experts == ep_size * experts_per_rank (expert e lives on rank e //
      experts_per_rank), ep_size <= 32. Integer dtype: SM100 (TIRx) reads int8/16/32/64 and uint8
      with unit inner stride; SM90 also takes uint16/32/64 and any strides.
    - meta_out: optional int64 buffer of at least 2 * ep_size elements; the first two outputs are
      views of its first 2 * ep_size elements (a fresh buffer when None).
    - tokens_per_ep_rank (ep_size,): (token, expert) slots routed to each rank.
    - dedup_tokens_per_gpu (ep_size,): distinct tokens sent to each rank.
    - dispatch_token_idxs (m * ep_size,), over-allocated: the first sum(dedup_tokens_per_gpu)
      entries are the token rows to send, rank-major.
    - idxs (m * k,): the slots token * k + j (topk_ids[token, j], row-major whatever the strides),
      sorted by expert.
    - expand_idx (m * k,): for each sorted slot, the position of its token within the token's dedup
      chunk for that expert's rank.
    - send_meta (ep_size * (experts_per_rank + 1),), viewed (ep_size, experts_per_rank + 1): per
      rank, its per-expert slot counts then its dedup count (the payload of the metadata
      all-to-all). Valid until the next call on the same stream (it may be reused scratch).

    Order within a bucket (a rank's chunk of dispatch_token_idxs, an expert's run of idxs) is token
    order on SM90 (deterministic, every input) and atomic order on SM100 (TIRx). Counts and
    per-bucket sets are identical on both. m == 0 gives zero counts and empty index outputs.

    SM90 limits, checked before anything is allocated (ValueError): ep_size * (experts_per_rank +
    1) <= 5236 (shared-memory bin tables; at ep_size 32, experts_per_rank <= 162) and m * (k +
    min(k, ep_size)) < 2^31 (32-bit slot positions); also a topk_ids that is not a 2-D integer
    tensor.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.ep_dispatch_sm90 import ep_dispatch_send_idx_sm90
            return ep_dispatch_send_idx_sm90(topk_ids, num_experts, ep_size, experts_per_rank, meta_out)
        case 10:
            from pithtrain.kernels.ep_dispatch_sm100 import ep_dispatch_send_idx_sm100
            return ep_dispatch_send_idx_sm100(topk_ids, num_experts, ep_size, experts_per_rank, meta_out)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


def ep_dispatch_recv_idx(recv_meta: torch.Tensor, received_expand_idx: torch.Tensor, ep_size: int, experts_per_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Receiver-side indices of the received token slots: (expert_idxs, expand_idx), fresh (total,)
    with total = received_expand_idx.shape[0]: int64 on SM90, received_expand_idx's dtype on SM100.
    Deterministic, identical on both paths.

    - recv_meta: int64 (ep_size * (experts_per_rank + 1),), the send_meta rows every sender
      addressed to this rank, sender-major: per sender, its per-local-expert slot counts then its
      dedup token count. Not modified; any stride (SM90 also takes any integer dtype).
    - received_expand_idx: int64 (total,), the senders' expand_idx for this rank, sender-major and
      expert-minor; total is at most the sum of the slot counts in recv_meta (the operator passes
      exactly the sum; a shorter one gets the indices of the first total slots, on SM90 exactly for
      any non-negative counts), and the dedup counts sum to fewer than 2^31. Not modified; any
      stride (SM90 also takes any integer dtype, values below 2^31).
    - expert_idxs[i]: the local expert (< experts_per_rank) of slot i.
    - expand_idx[i]: received_expand_idx[i] plus the dedup counts of the preceding senders, i.e.
      slot i's row in the concatenated received deduplicated tokens.

    SM90 limits, checked before anything is allocated (ValueError): total < 2^31 (32-bit slot
    indices) and ceil4(ep_size * experts_per_rank) + ceil4(ep_size) <= 57600 (shared-memory segment
    tables, e.g. experts_per_rank <= 1799 at ep_size 32); also inputs that are not 1-D integer
    tensors or a recv_meta of another length.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.ep_dispatch_sm90 import ep_dispatch_recv_idx_sm90
            return ep_dispatch_recv_idx_sm90(recv_meta, received_expand_idx, ep_size, experts_per_rank)
        case 10:
            from pithtrain.kernels.ep_dispatch_sm100 import ep_dispatch_recv_idx_sm100
            return ep_dispatch_recv_idx_sm100(recv_meta, received_expand_idx, ep_size, experts_per_rank)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
