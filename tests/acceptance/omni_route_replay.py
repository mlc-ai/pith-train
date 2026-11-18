"""Route-conditioned MoE parity for the full-size gates.

Test-only helpers; no production code changes.

Top-k expert selection is discontinuous. A rounding-level difference upstream of a
router, such as FA4 instead of HF eager attention, flips the selected expert set of
tokens whose k-th and (k+1)-th probabilities nearly tie. Every expert, router row and
preceding norm that such a token reaches then receives a legitimately different
gradient, so a free-running per-parameter comparison measures route flips as well as
arithmetic. Replaying the reference's selected experts through the production
``router_replay`` hook removes only that discrete choice: the native router still
computes its own probabilities and routing weights from its own hidden states.

Audio key-bias gradients, which are structurally zero, are judged separately
(omni_structural_zero).
"""

import contextlib
import re
import statistics

import torch
import torch.nn.functional as F

from tests.acceptance.omni_structural_zero import CRITERION

_ROUTER = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")
_GROUPS = (
    ("vision", re.compile(r"^visual\.")),
    ("audio", re.compile(r"^audio_tower\.")),
    ("router", re.compile(r"\.mlp\.gate\.weight$")),
    ("experts", re.compile(r"\.mlp\.experts\.")),
    ("post_attention_norm", re.compile(r"\.post_attention_layernorm\.")),
    ("attention", re.compile(r"\.self_attn\.")),
    ("input_norm", re.compile(r"\.input_layernorm\.")),
)


def moe_routers(model):
    """Decoder-layer routers by layer index; encoder towers have none."""
    routers = {}
    for name, module in model.named_modules():
        match = _ROUTER.search(name)
        if match:
            index = int(match.group(1))
            if index in routers:
                raise AssertionError(f"Duplicate router for layer {index}")
            routers[index] = module
    if not routers:
        raise AssertionError("No MoE routers found")
    return dict(sorted(routers.items()))


@contextlib.contextmanager
def capture_routes(model, native):
    """Record each router's selected experts and FP32 probabilities, outside autograd."""
    routes, handles = {}, []

    def recorder(index):
        def record(module, inputs, output):
            with torch.no_grad():
                if native:
                    # Native router returns (indices, weights, lb_loss); recompute its logits.
                    hidden = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                    logits, indices = F.linear(hidden, module.weight.detach()), output[0]
                else:
                    # HF router returns (router_logits, router_scores, router_indices).
                    logits, indices = output[0], output[2]
                if index in routes:
                    raise AssertionError(f"Router {index} ran twice in one capture")
                routes[index] = {
                    "indices": indices.detach().to("cpu", copy=True),
                    "probabilities": logits.detach().float().softmax(-1).cpu(),
                }

        return record

    try:
        # Registered inside the cleanup scope, so a failure partway releases the earlier hooks.
        for index, router in moe_routers(model).items():
            handles.append(router.register_forward_hook(recorder(index)))
        yield routes
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def replay_routes(model, routes):
    """Install the reference's selected experts on every native router (router_replay)."""
    routers = moe_routers(model)
    if routers.keys() != routes.keys():
        raise AssertionError("Reference and native router layers differ")
    installed = []
    try:
        for index, router in routers.items():
            if router.router_replay is not None:
                raise AssertionError("A router replay is already installed")
            indices = routes[index]["indices"].to(router.weight.device)

            def replay(num_tokens, top_k, indices=indices):
                if tuple(indices.shape) != (num_tokens, top_k):
                    raise AssertionError("Replayed routes do not match the router input")
                return indices

            router.router_replay = replay
            installed.append(router)
        yield
    finally:
        for router in installed:
            router.router_replay = None


def _summary(values):
    if not values:
        return None
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def route_statistics(reference, actual):
    """Per-layer selected-set changes; margins are the reference's p_k - p_(k+1)."""
    if reference.keys() != actual.keys():
        raise AssertionError("Route captures cover different layers")
    per_layer, flipped, kept = [], [], []
    for index in reference:
        expected, got = reference[index]["indices"], actual[index]["indices"]
        if expected.shape != got.shape:
            raise AssertionError("Route captures have different shapes")
        top_k = expected.shape[-1]
        probabilities = reference[index]["probabilities"]
        if top_k >= probabilities.shape[-1]:
            raise AssertionError("Dense routing has no top-k margin")
        changed = (expected.sort(-1).values != got.sort(-1).values).any(-1)
        top = probabilities.topk(top_k + 1, dim=-1).values.double()
        margins = (top[:, top_k - 1] - top[:, top_k]).tolist()
        for flag, margin in zip(changed.tolist(), margins, strict=True):
            (flipped if flag else kept).append(margin)
        per_layer.append(int(changed.sum()))
    return {
        "token_layers": len(flipped) + len(kept),
        "flipped_token_layers": len(flipped),
        "layers_with_flips": sum(count > 0 for count in per_layer),
        "flipped_reference_margin": _summary(flipped),
        "unflipped_reference_margin": _summary(kept),
        "flipped_tokens_per_layer": per_layer,
    }


def parameter_group(name):
    for group, pattern in _GROUPS:
        if pattern.search(name):
            return group
    return "other"


def group_summary(rows):
    """Compared count, failures and worst NSE per parameter group.

    Rows judged by the structural-zero criterion are counted separately and left out of
    max_nse, which would otherwise report the NSE between two rounding residues.
    """
    groups = {}
    for row in rows:
        entry = groups.setdefault(
            parameter_group(row["name"]),
            {"parameters": 0, "fail": 0, "max_nse": 0.0, "structural_zero": 0},
        )
        entry["parameters"] += 1
        entry["fail"] += row["status"] != "PASS"
        if row.get("criterion") == CRITERION:
            entry["structural_zero"] += 1
        elif row.get("difference") is not None:
            entry["max_nse"] = max(entry["max_nse"], row["difference"])
    return dict(sorted(groups.items()))
