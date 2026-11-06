"""The HuggingFace safetensors shard writer shared by the checkpoint converters."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import save_file

__all__ = ["write_hf_shards"]


def write_hf_shards(hf_state_dict, save_path, max_shard_size, stdout):
    """Shared shard writer; family-specific validation precedes this function."""
    shards: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    current_shard: Dict[str, torch.Tensor] = dict()
    current_size, shard_idx = 0, 0

    for key, tensor in hf_state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(("model-%05d.safetensors" % shard_idx, current_shard))
            current_shard, current_size, shard_idx = dict(), 0, shard_idx + 1
        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(("model-%05d.safetensors" % shard_idx, current_shard))

    weight_map, total_size = dict(), 0
    save_path.mkdir(parents=True, exist_ok=True)
    for i, (_, shard_tensors) in enumerate(shards):
        shard_name = "model-%05d-of-%05d.safetensors" % (i, len(shards))
        stdout.info("Writing shard %d/%d: %s" % (i + 1, len(shards), shard_name))
        save_file(shard_tensors, str(Path(save_path, shard_name)))
        for key in shard_tensors:
            weight_map[key] = shard_name
        total_size += sum(t.numel() * t.element_size() for t in shard_tensors.values())

    with open(Path(save_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)
    stdout.info(
        "Saved HF checkpoint to %s (%d weights, %d shards)"
        % (save_path, len(weight_map), len(shards))
    )
