"""
Activation memory profiling: per-chunk, per-layer, per-stage tensor sizes.

Computes the saved tensors at each of the 5 DualPipeV stage boundaries
(matching execution.py Stage*Args/Outs) plus autograd-internal overhead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .model_profile import ModelConfig, ModelMemoryProfile, ParallelismConfig, layer_is_moe
from .tensor_spec import MemoryBucket


@dataclass(slots=True)
class TokenCounts:
    """Token counts for one microbatch, possibly adjusted for EP imbalance."""

    m: int  # B * local_seq_len (tokens per microbatch)
    k: int  # num_experts_per_tok
    m_times_k: int  # m * k (expanded tokens)
    m_dedup: int  # unique tokens after EP dedup (sent from one source to one dest EP rank)
    m_recv: int  # avg token-expert pairs per EP rank (= m*k/ep, kept for compat)

    # Corrected MLP dimensions (used for stage 3 activation sizing):
    # With EP, each rank processes a different microbatch and exchanges tokens
    # via all-to-all.  Each rank receives tokens from all ep_size source ranks,
    # yielding ~= m * k total (token, expert) pairs on the receiving rank.
    m_sorted: int  # total dedup tokens on sending side (= m_dedup * ep_size)
    m_expanded: int  # total (token, expert) pairs after expansion (~= m * k)


def compute_token_counts(
    model_cfg: ModelConfig,
    parallel_cfg: ParallelismConfig,
    ep_imbalance_factor: float = 1.0,
) -> TokenCounts:
    """Compute expected token counts for MoE dispatch/combine."""
    B = parallel_cfg.micro_batch_size
    S = parallel_cfg.sequence_length // parallel_cfg.cp_size
    m = B * S
    k = model_cfg.num_experts_per_tok
    ep_size = parallel_cfg.ep_size

    m_times_k = m * k

    if ep_size <= 1:
        # No EP: all tokens stay local, experts compute on m*k pairs
        return TokenCounts(
            m=m,
            k=k,
            m_times_k=m_times_k,
            m_dedup=m,
            m_recv=m_times_k,
            m_sorted=m,
            m_expanded=m_times_k,
        )

    # Under uniform routing, the probability a token is NOT sent to a given EP rank
    # across k expert slots is ((ep_size - 1) / ep_size)^k.
    # Expected unique tokens sent per EP rank = m * (1 - ((ep_size-1)/ep_size)^k)
    prob_not_sent = ((ep_size - 1) / ep_size) ** k
    m_dedup_avg = m * (1.0 - prob_not_sent)
    m_dedup = int(math.ceil(m_dedup_avg * ep_imbalance_factor))

    # Each EP rank receives m*k/ep_size tokens on average (total expanded / ep_size)
    m_recv_avg = m_times_k / ep_size
    m_recv = int(math.ceil(m_recv_avg * ep_imbalance_factor))

    # m_sorted: total dedup tokens on sending side across all destination EP ranks.
    # Each source rank sends m_dedup unique tokens to EACH destination rank.
    # Total sent = m_dedup * ep_size.
    m_sorted = m_dedup * ep_size

    # m_expanded: total (token, expert) pairs after expansion on the receiving rank.
    # Each EP rank processes a different microbatch (EP acts as data parallelism
    # for non-expert parts).  The all-to-all gathers tokens from all ep_size
    # sources, each contributing its share.  The total expanded pairs on the
    # receiving rank ~= m * k.
    #
    # Routing variance: finite-sample randomness in the router causes the
    # worst-case layer to have ~20% more expanded tokens than the uniform
    # average.  Measured on Qwen3-30B-A3B with top-8 / 128 experts / 4096
    # tokens: analytical m*k=32768 vs real 38801 (factor 1.18).  We use 1.2
    # as a conservative worst-case default.
    routing_overhead = 1.2
    m_expanded = int(math.ceil(m_times_k * ep_imbalance_factor * routing_overhead))

    return TokenCounts(
        m=m,
        k=k,
        m_times_k=m_times_k,
        m_dedup=m_dedup,
        m_recv=m_recv,
        m_sorted=m_sorted,
        m_expanded=m_expanded,
    )


class ActivationEstimator:
    """Compute activation memory for one microbatch chunk on one module."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        parallel_cfg: ParallelismConfig,
        profile: ModelMemoryProfile,
        token_counts: TokenCounts,
    ):
        self.model_cfg = model_cfg
        self.parallel_cfg = parallel_cfg
        self.profile = profile
        self.tc = token_counts

        self.B = parallel_cfg.micro_batch_size
        self.S = parallel_cfg.sequence_length // parallel_cfg.cp_size
        self.H = model_cfg.hidden_size

    def _stage_activations_moe(
        self, layer_idx: int, prefix: str
    ) -> tuple[MemoryBucket, MemoryBucket]:
        """
        Compute LayerRecord activation sizes and autograd overhead for one MoE layer.

        Returns (records_bucket, autograd_bucket).
        """
        B, S, H = self.B, self.S, self.H
        m = self.tc.m
        k = self.tc.k
        m_times_k = self.tc.m_times_k
        m_expanded = self.tc.m_expanded
        moe_inter = self.model_cfg.moe_intermediate_size
        nheads = self.model_cfg.num_attention_heads
        nkv = self.model_cfg.num_key_value_heads
        hd = self.model_cfg.head_dim
        ep_size = self.parallel_cfg.ep_size

        records = MemoryBucket(f"{prefix} records")
        autograd = MemoryBucket(f"{prefix} autograd")

        # === Stage boundary records ===
        # Each .detach().requires_grad_() at a stage boundary creates a tensor
        # that shares storage with the previous stage's output.  We count each
        # unique tensor ONCE - the outs from the producing stage.  The next
        # stage's args.X is just a detached alias (same storage, 0 extra bytes).
        #
        # s1.args.hidden shares storage with prev layer's s5.outs.hidden_states
        # (or prolog output for the first layer).  NOT counted here - already
        # counted in the producing stage.
        #
        # s1.outs.sorted_tokens: FREED at stage3 start via deferred_free
        # (overlap.py: ctx.fwd_comm_deferred_free.append(sorted_tokens)).
        # Does NOT persist beyond the layer - not counted.
        #
        # s2.outs.gathered_tokens: FREED immediately in stage3_f
        # (execution.py: gathered_tokens.untyped_storage().resize_(0) when ep>1).
        # Does NOT persist - not counted.
        #
        # s3.outs.moe_outs: FREED at stage5 start via deferred_free
        # (overlap.py: ctx.fwd_comm_deferred_free.append(s3.outs.moe_outs)).
        # Does NOT persist beyond the layer - not counted.

        # Persistent stage 1 outs:
        records.add(f"{prefix}.s1.outs.topk_weight", (m, k), "fp32")
        records.add(f"{prefix}.s1.outs.topk_ids", (m, k), "int64")
        records.add(f"{prefix}.s1.outs.residual", (B, S, H), "bf16")

        # Stage 1 autograd (compiled forward_stage1_compute + gate)
        # NOTE: input_ln_input shares storage with s1.args.hidden (= prev
        # layer's hidden_states). NOT counted to avoid double-counting.
        autograd.add(f"{prefix}.s1.ag.qkv_proj_input", (B, S, H), "bf16")
        # Flash Attention 4 saves Q, K, V, O, and softmax_lse for backward.
        # K, V are at nkv heads (no GQA expansion - FA4 handles GQA internally).
        autograd.add(f"{prefix}.s1.ag.flash_Q", (B, S, nheads, hd), "bf16")
        autograd.add(f"{prefix}.s1.ag.flash_K", (B, S, nkv, hd), "bf16")
        autograd.add(f"{prefix}.s1.ag.flash_V", (B, S, nkv, hd), "bf16")
        autograd.add(f"{prefix}.s1.ag.flash_O", (B, S, nheads, hd), "bf16")
        autograd.add(f"{prefix}.s1.ag.flash_lse", (B, nheads, S), "fp32")
        # NOTE: o_proj_input = flash_O.reshape(B, S, nheads*hd) - view shares
        # storage with flash_O.  NOT counted to avoid double-counting.
        autograd.add(f"{prefix}.s1.ag.post_attn_ln_input", (B, S, H), "bf16")
        # Gate (router) input: nn.Linear saves input for weight grad (bf16).
        autograd.add(f"{prefix}.s1.ag.gate_input", (m, H), "bf16")

        # Stage 3 autograd (forward_stage3 - not compiled)
        #
        # The MLP operates at the SCATTERED dimension:
        #   m_scattered = m_expanded + experts_per_rank * 127  (GEMM 128-row alignment)
        #
        # With EP, each rank processes a different microbatch.  The all-to-all
        # gathers tokens from all ep_size sources.  After expand_idx
        # (restoring per-expert entries), the total pairs ~= m * k.  After
        # scatter_for_grouped_gemm (padding for grouped GEMM alignment), the M
        # dimension = m_scattered.
        experts_per_rank = self.model_cfg.num_experts // ep_size
        scatter_padding = experts_per_rank * 127  # 128-row alignment per group
        m_scattered = m_expanded + scatter_padding

        # Expert MLP saves these tensors for backward:
        # - gate_proj input (= scattered tokens): shared by gate_proj and up_proj
        # - gate_proj output (saved by SiLU backward)
        # - SiLU output (saved by element-wise multiply backward)
        # - up_proj output (saved by element-wise multiply backward)
        # - down_proj input (= gate_out * up_out, saved by down_proj backward)
        # - expand_idx, reverse_shuffle_idxs (indices)
        autograd.add(f"{prefix}.s3.ag.scattered_tokens", (m_scattered, H), "bf16")
        autograd.add(f"{prefix}.s3.ag.gate_proj_output", (m_scattered, moe_inter), "bf16")
        autograd.add(f"{prefix}.s3.ag.silu_output", (m_scattered, moe_inter), "bf16")
        autograd.add(f"{prefix}.s3.ag.up_proj_output", (m_scattered, moe_inter), "bf16")
        autograd.add(f"{prefix}.s3.ag.down_proj_input", (m_scattered, moe_inter), "bf16")
        autograd.add(f"{prefix}.s3.ag.expand_idx", (m_expanded,), "int64")
        autograd.add(f"{prefix}.s3.ag.reverse_shuffle_idxs", (m_expanded,), "int64")

        # Stage 4: combine all-to-all output (reverse of dispatch).
        # Output = results sent back to source ranks at m_times_k dimension.
        # Stored as s5.args.moe_outs; also saved by forward_stage5 autograd
        # (shared storage - counted once here in records, not in autograd).
        records.add(f"{prefix}.s5.args.combine_output", (m_times_k, H), "bf16")

        # Stage 5 outs
        records.add(f"{prefix}.s5.outs.hidden_states", (B, S, H), "bf16")

        # Stage 5 autograd (compiled forward_stage5)
        # forward_stage5 saves moe_outs and topk_weight for backward,
        # but both already counted in records (s5.args.combine_output and
        # s1.outs.topk_weight share storage).  Only moe_local_idxs is new.
        autograd.add(f"{prefix}.s5.ag.moe_local_idxs", (m_times_k,), "int64")

        return records, autograd

    def _stage_activations_dense(
        self, layer_idx: int, prefix: str
    ) -> tuple[MemoryBucket, MemoryBucket]:
        """Compute activation sizes for a dense (non-MoE) layer."""
        B, S, H = self.B, self.S, self.H
        inter = self.model_cfg.intermediate_size
        nheads = self.model_cfg.num_attention_heads
        nkv = self.model_cfg.num_key_value_heads
        hd = self.model_cfg.head_dim

        records = MemoryBucket(f"{prefix} records")
        autograd = MemoryBucket(f"{prefix} autograd")

        # Dense layer records - deduplicated (see MoE comments above)
        records.add(f"{prefix}.s1.args.hidden", (B, S, H), "bf16")
        records.add(f"{prefix}.s1.outs.sorted_tokens", (B * S, H), "bf16")
        records.add(f"{prefix}.s1.outs.residual", (B, S, H), "bf16")

        # Stage 1 autograd
        autograd.add(f"{prefix}.s1.ag.input_ln_input", (B, S, H), "bf16")
        autograd.add(f"{prefix}.s1.ag.qkv_proj_input", (B, S, H), "bf16")
        autograd.add(f"{prefix}.s1.ag.flash_Q", (B, S, nheads, hd), "bf16")
        autograd.add(f"{prefix}.s1.ag.flash_K", (B, S, nkv, hd), "bf16")
        autograd.add(f"{prefix}.s1.ag.flash_V", (B, S, nkv, hd), "bf16")
        autograd.add(f"{prefix}.s1.ag.flash_O", (B, S, nheads, hd), "bf16")
        autograd.add(f"{prefix}.s1.ag.flash_lse", (B, nheads, S), "fp32")
        autograd.add(f"{prefix}.s1.ag.o_proj_input", (B, S, nheads * hd), "bf16")
        autograd.add(f"{prefix}.s1.ag.post_attn_ln_input", (B, S, H), "bf16")

        # Stage 2/3: passthrough for dense, no dispatch. Only count outs.
        records.add(f"{prefix}.s3.outs.moe_outs", (B * S, H), "bf16")

        # Stage 3 autograd (dense MLP)
        autograd.add(f"{prefix}.s3.ag.gate_proj_input", (B * S, H), "bf16")
        autograd.add(f"{prefix}.s3.ag.gate_proj_output_silu", (B * S, inter), "bf16")
        autograd.add(f"{prefix}.s3.ag.up_proj_output", (B * S, inter), "bf16")
        autograd.add(f"{prefix}.s3.ag.down_proj_input", (B * S, inter), "bf16")

        # Stage 4/5 outs only
        records.add(f"{prefix}.s5.outs.hidden_states", (B, S, H), "bf16")

        return records, autograd

    def compute_chunk_activations(
        self, module_idx: int, phase: int, chunk_id: int
    ) -> tuple[MemoryBucket, MemoryBucket]:
        """
        Compute total activation memory for one microbatch chunk on one module.

        Returns (records_bucket, autograd_bucket).
        """
        layer_indices = self.profile.get_module_layer_indices(module_idx)
        has_embed = self.profile.module_has_embed(module_idx)
        has_head = self.profile.module_has_head(module_idx)
        cfg = self.model_cfg
        B, S, H = self.B, self.S, self.H

        all_records = MemoryBucket(f"p{phase}.c{chunk_id} records")
        all_autograd = MemoryBucket(f"p{phase}.c{chunk_id} autograd")

        # Prolog (embed_tokens)
        if has_embed:
            all_records.add(f"p{phase}.c{chunk_id}.prolog.outs.hidden", (B, S, H), "bf16")

        for layer_idx in layer_indices:
            prefix = f"p{phase}.c{chunk_id}.L{layer_idx}"
            is_moe = layer_is_moe(
                layer_idx,
                cfg.num_experts,
                cfg.decoder_sparse_step,
                cfg.mlp_only_layers,
            )
            if is_moe:
                records, autograd = self._stage_activations_moe(layer_idx, prefix)
            else:
                records, autograd = self._stage_activations_dense(layer_idx, prefix)

            all_records.specs.extend(records.specs)
            all_autograd.specs.extend(autograd.specs)

        # Epilog (norm + lm_head)
        if has_head:
            all_records.add(f"p{phase}.c{chunk_id}.epilog.args.hidden", (B, S, H), "bf16")
            all_records.add(
                f"p{phase}.c{chunk_id}.epilog.outs.logits",
                (B, S, cfg.vocab_size),
                "bf16",
            )
            # Autograd: norm input saved for backward
            all_autograd.add(f"p{phase}.c{chunk_id}.epilog.ag.norm_input", (B, S, H), "bf16")
            # Autograd: lm_head input (= norm output)
            all_autograd.add(f"p{phase}.c{chunk_id}.epilog.ag.lm_head_input", (B, S, H), "bf16")
            # Note: nn.Linear saves weight for backward (F.linear grad_weight),
            # but lm_head weight is already counted in static params_unsharded
            # (module[1] params).  Not counted here to avoid double-counting.

        return all_records, all_autograd

    def compute_chunk_total_bytes(self, module_idx: int) -> int:
        """Quick total bytes for one chunk (records + autograd)."""
        records, autograd = self.compute_chunk_activations(module_idx, 0, 0)
        return records.total_bytes + autograd.total_bytes

    def compute_wgrad_store_bytes_per_chunk(self, module_idx: int) -> int:
        """
        Estimate memory held by WeightGradStore closures for one chunk.

        When enable_zb=True, stage3_b defers weight gradient computation.
        The closures capture references to the MLP input and grad tensors
        at the scattered dimension.
        """
        layer_indices = self.profile.get_module_layer_indices(module_idx)
        cfg = self.model_cfg
        total = 0
        for layer_idx in layer_indices:
            is_moe = layer_is_moe(
                layer_idx,
                cfg.num_experts,
                cfg.decoder_sparse_step,
                cfg.mlp_only_layers,
            )
            if is_moe:
                # Captured: input and grad_output for each expert linear layer.
                # These are at the expanded dimension (m * k after EP all-to-all).
                m_expanded = self.tc.m_expanded
                H = self.H
                # Two tensors: input and grad, both (m_expanded, H) bf16
                total += 2 * m_expanded * H * 2
            else:
                # Dense: input for wgrad
                m = self.B * self.S
                H = self.H
                total += 2 * m * H * 2
        return total

    def compute_comm_buffer_bytes(self) -> int:
        """
        Estimate P2P communication buffer sizes.

        Each recv allocates (B, local_seq_len, H) in bf16.
        At most ~2-4 buffers alive at once (forward + backward recv).
        """
        B, S, H = self.B, self.S, self.H
        per_buffer = B * S * H * 2  # bf16
        # Conservative: 2 active recv buffers (one forward, one backward)
        return 2 * per_buffer

    def compute_a2a_buffer_bytes_per_layer(self) -> int:
        """
        Estimate all-to-all buffer overhead per layer.

        Dispatch and combine each allocate a padded output buffer.
        Padding to 512-row alignment.
        """
        H = self.H
        m_dedup = self.tc.m_dedup
        m_recv = self.tc.m_recv

        def padded(rows: int) -> int:
            return ((rows + 511) // 512) * 512

        # Dispatch: output = (padded(m_recv), H) bf16
        dispatch_buf = padded(m_recv) * H * 2
        # Combine: output = (padded(m_dedup), H) bf16
        combine_buf = padded(m_dedup) * H * 2
        return dispatch_buf + combine_buf

    def compute_pp_transfer_bytes(self) -> int:
        """
        Memory for one PP transfer tensor: detached output from phase0 to phase1
        at the last PP rank.
        """
        B, S, H = self.B, self.S, self.H
        return B * S * H * 2  # bf16
