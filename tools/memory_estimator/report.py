"""
Report formatting for the memory estimator.

Produces multi-level output: summary, detail, timeline, and optimization suggestions.
"""

from __future__ import annotations

from .schedule_simulator import ScheduleResult
from .tensor_spec import MemoryBucket, format_bytes, format_gb


def _bar(width: int = 70) -> str:
    return "\u2500" * width


def _format_bucket_summary(name: str, bucket: MemoryBucket, indent: int = 2) -> str:
    pad = " " * indent
    return f"{pad}{name:<50s} {format_gb(bucket.total_bytes):>8s} GB"


def _format_bytes_line(name: str, b: int, indent: int = 2) -> str:
    pad = " " * indent
    return f"{pad}{name:<50s} {format_gb(b):>8s} GB"


def print_summary(
    result: ScheduleResult,
    model_desc: str,
    parallel_desc: str,
    training_desc: str,
    ep_desc: str,
    gpu_memory_gb: float,
) -> None:
    """Print the Level 1 summary table."""
    peak_event = result.timeline[result.peak_event_idx]
    snap = peak_event.snapshot

    print()
    print("=" * 70)
    print("  PithTrain Memory Estimator")
    print("=" * 70)
    print(f"  {model_desc}")
    print(f"  {parallel_desc}")
    print(f"  {training_desc}")
    print(f"  {ep_desc}")
    print()

    # Static memory
    print("--- Static Memory (constant during DualPipeV.step) ---")
    print(f"{'  Component':<52s} {'GB':>8s}")
    print(f"  {_bar(58)}")
    for i, bucket in enumerate(snap.params_unsharded):
        print(_format_bucket_summary(bucket.name, bucket))
    print(_format_bucket_summary(snap.fsdp_sharded_params.name, snap.fsdp_sharded_params))
    print(_format_bucket_summary(snap.optimizer_states.name, snap.optimizer_states))
    print(f"  {_bar(58)}")
    print(_format_bytes_line("Total static", snap.static_bytes))
    print()

    # Peak dynamic memory
    p0, p1 = snap.live_chunk_counts
    print("--- Peak Dynamic Memory ---")
    print(f'  Peak at: event #{result.peak_event_idx} "{peak_event.name}"')
    print(f"  Live chunks: phase0={p0}, phase1={p1}")
    print()
    print(f"{'  Component':<52s} {'GB':>8s}")
    print(f"  {_bar(58)}")
    print(_format_bytes_line("Activations (LayerRecord tensors)", snap.activation_bytes))
    print(_format_bytes_line("Autograd saved tensors", snap.autograd_bytes))
    print(_format_bytes_line("Gradients (accumulated)", snap.gradients.total_bytes))
    print(_format_bytes_line("Comm buffers (P2P + all-to-all)", snap.comm_buffers.total_bytes))
    print(
        _format_bytes_line("WeightGradStore deferred tensors", snap.weight_grad_store.total_bytes)
    )
    print(_format_bytes_line("PP transfer buffers", snap.pp_transfer_buffers.total_bytes))
    print(f"  {_bar(58)}")
    print(_format_bytes_line("Total dynamic at peak", snap.dynamic_bytes))
    print()

    # Grand total
    print("--- Grand Total ---")
    base = snap.static_bytes + snap.dynamic_bytes
    print(_format_bytes_line("Estimated peak (model tensors)", base))
    if snap.non_pytorch_bytes > 0:
        print(
            _format_bytes_line(
                "Non-PyTorch (CUDA ctx, NCCL, compile)",
                snap.non_pytorch_bytes,
            )
        )
    pct = int(snap.fragmentation_factor * 100)
    print(_format_bytes_line(f"CUDA allocator fragmentation (~{pct}%)", snap.fragmentation_bytes))
    print(f"  {_bar(58)}")
    print(_format_bytes_line("TOTAL", snap.total_bytes))
    print(_format_bytes_line("GPU capacity", int(gpu_memory_gb * 1024**3)))
    headroom = int(gpu_memory_gb * 1024**3) - snap.total_bytes
    headroom_pct = headroom / (gpu_memory_gb * 1024**3) * 100
    print(_format_bytes_line(f"Headroom ({headroom_pct:.1f}%)", headroom))
    print()

    if headroom < 0:
        print(f"  *** STATUS: OOM - exceeds GPU capacity by {format_bytes(-headroom)} ***")
    elif headroom_pct < 5:
        print("  *** STATUS: TIGHT - less than 5% headroom ***")
    else:
        print("  STATUS: OK")
    print()


def print_detail(result: ScheduleResult) -> None:
    """Print Level 2 detail: per-parameter and per-chunk activation breakdown at peak."""
    peak_event = result.timeline[result.peak_event_idx]
    snap = peak_event.snapshot

    print()
    print("=" * 70)
    print("  Detailed Breakdown at Peak")
    print("=" * 70)

    # Module parameters
    for bucket in snap.params_unsharded:
        print()
        print(f"--- {bucket.name} ---")
        for spec in bucket.specs:
            shape_str = ",".join(str(d) for d in spec.shape)
            print(
                f"    {spec.name:<55s} [{shape_str:<20s}] {spec.dtype:<5s} {format_bytes(spec.bytes):>10s}"
            )
        print(f"    {'Total:':<55s} {'':>28s} {format_bytes(bucket.total_bytes):>10s}")

    # FSDP sharded
    print()
    print(f"--- {snap.fsdp_sharded_params.name} ---")
    print(
        f"    Total: {format_bytes(snap.fsdp_sharded_params.total_bytes)} ({len(snap.fsdp_sharded_params.specs)} shards)"
    )

    # Optimizer
    print()
    print(f"--- {snap.optimizer_states.name} ---")
    print(
        f"    Total: {format_bytes(snap.optimizer_states.total_bytes)} ({len(snap.optimizer_states.specs)} state tensors)"
    )

    # Live chunk activations
    for key in sorted(snap.activations.keys()):
        phase, chunk_id = key
        records = snap.activations[key]
        autograd = snap.autograd_overhead.get(key)

        print()
        print(f"--- Live chunk phase={phase}, chunk={chunk_id} ---")
        print(f"  LayerRecord tensors ({format_bytes(records.total_bytes)}):")

        # Group by layer
        layer_specs: dict[str, list] = {}
        for spec in records.specs:
            # Extract layer identifier (e.g., "L0", "prolog", "epilog")
            parts = spec.name.split(".")
            if len(parts) >= 3:
                layer_key = parts[2]  # e.g., "L0" or "prolog" or "epilog"
            else:
                layer_key = "other"
            layer_specs.setdefault(layer_key, []).append(spec)

        for layer_key in sorted(layer_specs.keys(), key=_layer_sort_key):
            specs = layer_specs[layer_key]
            subtotal = sum(s.bytes for s in specs)
            print(f"    {layer_key} ({format_bytes(subtotal)}):")
            for spec in specs:
                short_name = spec.name.split(".", 3)[-1] if "." in spec.name else spec.name
                shape_str = ",".join(str(d) for d in spec.shape)
                print(
                    f"      {short_name:<45s} [{shape_str:<15s}] {spec.dtype:<5s} {format_bytes(spec.bytes):>10s}"
                )

        if autograd and autograd.total_bytes > 0:
            print(f"  Autograd overhead ({format_bytes(autograd.total_bytes)}):")
            # Group by layer too
            ag_layer_specs: dict[str, list] = {}
            for spec in autograd.specs:
                parts = spec.name.split(".")
                if len(parts) >= 3:
                    layer_key = parts[2]
                else:
                    layer_key = "other"
                ag_layer_specs.setdefault(layer_key, []).append(spec)

            for layer_key in sorted(ag_layer_specs.keys(), key=_layer_sort_key):
                specs = ag_layer_specs[layer_key]
                subtotal = sum(s.bytes for s in specs)
                print(f"    {layer_key} ({format_bytes(subtotal)}):")
                for spec in specs:
                    short_name = spec.name.split(".", 3)[-1] if "." in spec.name else spec.name
                    shape_str = ",".join(str(d) for d in spec.shape)
                    print(
                        f"      {short_name:<45s} [{shape_str:<15s}] {spec.dtype:<5s} {format_bytes(spec.bytes):>10s}"
                    )

        chunk_total = records.total_bytes + (autograd.total_bytes if autograd else 0)
        print(f"  Chunk total: {format_bytes(chunk_total)}")

    # Gradients
    if snap.gradients.total_bytes > 0:
        print()
        print(f"--- Gradients ({format_bytes(snap.gradients.total_bytes)}) ---")
        print(f"    {len(snap.gradients.specs)} gradient tensors")

    # WeightGradStore
    if snap.weight_grad_store.total_bytes > 0:
        print()
        print(f"--- WeightGradStore ({format_bytes(snap.weight_grad_store.total_bytes)}) ---")

    # PP transfers
    if snap.pp_transfer_buffers.total_bytes > 0:
        print()
        print(f"--- PP Transfer Buffers ({format_bytes(snap.pp_transfer_buffers.total_bytes)}) ---")
        for spec in snap.pp_transfer_buffers.specs:
            print(f"    {spec.name}")

    print()


def _layer_sort_key(key: str) -> tuple[int, str]:
    """Sort layer keys: prolog first, then L0, L1, ..., then epilog."""
    if key == "prolog":
        return (-1, key)
    if key == "epilog":
        return (99999, key)
    if key.startswith("L"):
        try:
            return (int(key[1:]), key)
        except ValueError:
            pass
    return (50000, key)


def print_timeline(result: ScheduleResult, max_events: int = 0) -> None:
    """Print Level 3 full timeline of memory at each schedule event."""
    print()
    print("=" * 100)
    print("  Schedule Timeline")
    print("=" * 100)
    print(
        f"  {'#':<5s} {'Event':<50s} {'Total GB':>10s} {'Delta GB':>10s} "
        f"{'Live(p0,p1)':>12s} {'Notes':>10s}"
    )
    print(f"  {_bar(97)}")

    events = result.timeline
    if max_events > 0 and len(events) > max_events:
        # Show first max_events/2 and last max_events/2, plus peak
        half = max_events // 2
        show_indices = set(range(half))
        show_indices.update(range(len(events) - half, len(events)))
        show_indices.add(result.peak_event_idx)
        show_indices = sorted(show_indices)
    else:
        show_indices = list(range(len(events)))

    prev_idx = -1
    for idx in show_indices:
        if prev_idx >= 0 and idx > prev_idx + 1:
            print(f"  {'...':>5s}")
        prev_idx = idx

        event = events[idx]
        snap = event.snapshot
        p0, p1 = snap.live_chunk_counts
        total_gb = snap.total_bytes / 1024**3
        delta_gb = event.delta_bytes / 1024**3
        delta_str = f"{delta_gb:+.3f}" if idx > 0 else f"{delta_gb:.3f}"
        peak_marker = " *** PEAK" if idx == result.peak_event_idx else ""
        print(
            f"  {idx:<5d} {event.name:<50s} {total_gb:>10.3f} {delta_str:>10s} "
            f"{'(%d,%d)' % (p0, p1):>12s}{peak_marker}"
        )

    print()


def print_suggestions(result: ScheduleResult, gpu_memory_gb: float) -> None:
    """Print Level 4 optimization suggestions based on the breakdown."""
    peak_event = result.timeline[result.peak_event_idx]
    snap = peak_event.snapshot
    total = snap.total_bytes
    gpu_bytes = int(gpu_memory_gb * 1024**3)

    print()
    print("=" * 70)
    print("  Optimization Suggestions")
    print("=" * 70)

    # Rank components by size
    components = [
        ("Model parameters (unsharded)", sum(b.total_bytes for b in snap.params_unsharded)),
        ("FSDP sharded params", snap.fsdp_sharded_params.total_bytes),
        ("Optimizer states", snap.optimizer_states.total_bytes),
        ("Activations (records)", snap.activation_bytes),
        ("Autograd saved tensors", snap.autograd_bytes),
        ("Gradients", snap.gradients.total_bytes),
        ("Comm buffers", snap.comm_buffers.total_bytes),
        ("WeightGradStore", snap.weight_grad_store.total_bytes),
        ("PP transfer buffers", snap.pp_transfer_buffers.total_bytes),
        ("Fragmentation overhead", snap.fragmentation_bytes),
    ]
    components.sort(key=lambda x: x[1], reverse=True)

    print()
    print("  Memory breakdown by component (sorted by size):")
    for name, b in components:
        pct = b / total * 100 if total > 0 else 0
        print(f"    {name:<40s} {format_bytes(b):>12s} ({pct:5.1f}%)")
    print()

    # Specific suggestions
    suggestions = []
    idx = 1

    # Check activations
    act_total = snap.activation_bytes + snap.autograd_bytes
    if total > 0 and act_total / total > 0.3:
        suggestions.append(
            f"{idx}. Activations + autograd ({act_total / 1024**3:.2f} GB, "
            f"{act_total / total * 100:.1f}% of peak) dominate.\n"
            f"     Consider: reducing micro_batch_size, reducing sequence_length,\n"
            f"     or enabling activation checkpointing for stage 1 (attention)."
        )
        idx += 1

    # Check epilog logits
    for key, bucket in snap.activations.items():
        for spec in bucket.specs:
            if "epilog.outs.logits" in spec.name and spec.bytes > 100 * 1024**2:
                suggestions.append(
                    f"{idx}. Epilog logits tensor ({format_bytes(spec.bytes)}) is large.\n"
                    f"     Consider: fusing cross-entropy loss with lm_head to avoid\n"
                    f"     materializing the full [B, S, vocab_size] logit tensor."
                )
                idx += 1
                break
        break  # only check one chunk

    # Check optimizer states
    opt_bytes = snap.optimizer_states.total_bytes
    if total > 0 and opt_bytes / total > 0.2:
        suggestions.append(
            f"{idx}. Optimizer states ({opt_bytes / 1024**3:.2f} GB, "
            f"{opt_bytes / total * 100:.1f}% of peak) are significant.\n"
            f"     Consider: 8-bit Adam (bitsandbytes) or Adafactor to reduce\n"
            f"     optimizer state from 2x fp32 to 1x fp32 or less."
        )
        idx += 1

    # Check live chunks
    p0, p1 = snap.live_chunk_counts
    total_live = p0 + p1
    if total_live > 4:
        suggestions.append(
            f"{idx}. {total_live} live activation chunks at peak ({p0} phase0, {p1} phase1).\n"
            f"     This is inherent to the DualPipeV schedule. Increasing pp_size\n"
            f"     reduces layers per rank but may increase live chunk count."
        )
        idx += 1

    # Check params
    param_bytes = sum(b.total_bytes for b in snap.params_unsharded)
    if total > 0 and param_bytes / total > 0.3:
        suggestions.append(
            f"{idx}. Unsharded parameters ({param_bytes / 1024**3:.2f} GB, "
            f"{param_bytes / total * 100:.1f}% of peak) are large.\n"
            f"     This is from FSDP reshard_after_forward=False keeping full params.\n"
            f"     Increasing pp_size would reduce layers per rank."
        )
        idx += 1

    # Check gradients
    grad_bytes = snap.gradients.total_bytes
    if total > 0 and grad_bytes / total > 0.1:
        suggestions.append(
            f"{idx}. Gradient memory ({grad_bytes / 1024**3:.2f} GB) grows as backward\n"
            f"     progresses. This is proportional to unsharded parameter size."
        )
        idx += 1

    if not suggestions:
        print("  No specific optimization suggestions - memory usage looks balanced.")
    else:
        for s in suggestions:
            print(f"  {s}")
            print()

    # Overall verdict
    if total > gpu_bytes:
        deficit = total - gpu_bytes
        print(
            f"  To fit in {gpu_memory_gb:.0f} GB, need to reduce by at least {format_bytes(deficit)}."
        )
    print()
