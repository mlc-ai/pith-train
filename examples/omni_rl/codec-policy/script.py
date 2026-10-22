"""Run bounded local codec GRPO from explicit native weights and frozen prompts.

No downloads or reward service are implicit. See the adjacent README for the
weights_only bundle format and the discrete/frozen component boundary.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

from pithtrain.models.qwen3_omni_talker import Qwen3OmniMoeTalkerForConditionalGeneration
from pithtrain.tasks import omni_codec_rl as codec
from pithtrain.tasks import omni_rl


def build_trainer(bundle, *, max_updates, seed, learning_rate, max_weight_bytes, device):
    required = {"config", "talker_state", "reference_state", "prompts", "upstream_contract"}
    if set(bundle) != required:
        raise ValueError(
            "Explicit config, initial/fixed-reference weights, prompts and upstream identity required"
        )
    for state in (bundle["talker_state"], bundle["reference_state"]):
        if (
            not isinstance(state, dict)
            or any(
                not isinstance(t, torch.Tensor) or not torch.isfinite(t).all()
                for t in state.values()
            )
            or sum(t.numel() * t.element_size() for t in state.values()) > max_weight_bytes
        ):
            raise ValueError("Initial/reference weights violate finite per-copy byte admission")
    config = Qwen3OmniMoeConfig.from_dict(bundle["config"])
    prompts = [codec.CodecPrompt(**record) for record in bundle["prompts"]]
    if not prompts:
        raise ValueError("At least one explicit local prompt required")
    # Policy frame/prompt bounds are part of its sealed model identity.
    upstream = dict(bundle["upstream_contract"])
    max_frames = upstream.pop("max_frames")
    max_prompt_tokens = upstream.pop("max_prompt_tokens")
    speaker = upstream.pop("speaker")
    if set(upstream) != {"chat_template_sha256", "processor_sha256", "position_config_sha256"}:
        raise ValueError("Incomplete upstream processor/template/position identity")
    native = Qwen3OmniMoeTalkerForConditionalGeneration(copy.deepcopy(config.talker_config)).float()
    native.load_state_dict(bundle["talker_state"], strict=True)
    policy = codec.CodecPolicy(
        native.to(device),
        config,
        max_frames=max_frames,
        max_prompt_tokens=max_prompt_tokens,
        speaker=speaker,
    )
    reference = copy.deepcopy(policy)
    reference.talker.load_state_dict(bundle["reference_state"], strict=True)
    rollout = codec.LocalCodecRollout(
        copy.deepcopy(policy), reference, prompts, max_weight_bytes=max_weight_bytes
    )
    reward = codec.ExactCodecReward(prompts, policy)
    cfg = omni_rl.OmniRLCfg()
    cfg.max_updates, cfg.max_requests = max_updates, len(prompts)
    cfg.group_size = cfg.max_samples = 2
    cfg.max_sequence_length = max_frames * 16 + 1
    cfg.micro_batch_size, cfg.seed = 1, seed
    backend = codec.CodecPolicyAdapter(
        policy, torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=0)
    )
    contract = omni_rl.RolloutContract(
        **upstream,
        reference_policy_sha256=rollout.reference_sha256,
        reward_config_sha256=omni_rl.fingerprint(reward.config),
    )
    return codec.create_trainer(
        cfg, contract=contract, backend=backend, rollout=rollout, reward=reward
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-updates", type=int, required=True)
    parser.add_argument("--stop-after-requests", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-weight-bytes", type=int, default=64 << 20)
    parser.add_argument("--max-input-bytes", type=int, default=256 << 20)
    args = parser.parse_args()
    if args.max_input_bytes < 1 or args.max_weight_bytes < 1 or args.stop_after_requests < 1:
        raise ValueError("Positive explicit local bounds required")
    # Use the same open descriptor for size admission and safe loading.
    with args.input.open("rb") as stream:
        stream.seek(0, 2)
        if stream.tell() > args.max_input_bytes:
            raise ValueError("Input exceeds admitted file bytes")
        stream.seek(0)
        bundle = torch.load(stream, map_location="cpu", weights_only=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    trainer = build_trainer(
        bundle,
        max_updates=args.max_updates,
        seed=args.seed,
        learning_rate=args.learning_rate,
        max_weight_bytes=args.max_weight_bytes,
        device=torch.device(args.device),
    )
    if args.resume is not None:
        trainer.backend.load(args.resume, trainer)
    reports = []
    while (
        trainer.requests < min(args.stop_after_requests, trainer.cfg.max_requests)
        and trainer.updates < trainer.cfg.max_updates
    ):
        reports.append(trainer.train_step())
        trainer.backend.save(args.output, trainer)
    print(
        json.dumps(
            dict(
                requests=trainer.requests,
                updates=trainer.updates,
                policy_version=trainer.policy_version,
                reports=reports,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
