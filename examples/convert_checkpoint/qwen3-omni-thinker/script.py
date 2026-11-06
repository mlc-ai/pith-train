"""Convert complete Thinker checkpoints locally; never download model weights."""

import argparse
from pathlib import Path

from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg, launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("hf2dcp", "dcp2hf"))
    parser.add_argument("load_path", type=Path)
    parser.add_argument("save_path", type=Path)
    parser.add_argument("--thinker-config", type=Path)
    parser.add_argument("--prefix", choices=("thinker.", ""), default="thinker.")
    parser.add_argument("--expert-layout", choices=("split", "fused"), default="fused")
    parser.add_argument(
        "--thinker-only",
        action="store_true",
        help="Explicitly exclude and report Talker/code2wav weights",
    )
    parser.add_argument("--restore-resume", action="store_true")
    parser.add_argument("--max-shard-size", type=int, default=8 * 1024**3)
    args = parser.parse_args()
    cfg = ConvertCheckpointCfg()
    cfg.operation, cfg.load_path, cfg.save_path = args.operation, args.load_path, args.save_path
    cfg.omni_multimodal, cfg.omni_thinker_config = True, args.thinker_config
    cfg.omni_prefix, cfg.omni_expert_layout = args.prefix, args.expert_layout
    cfg.omni_thinker_only, cfg.omni_restore_resume = args.thinker_only, args.restore_resume
    cfg.max_shard_size = args.max_shard_size
    launch(cfg)


if __name__ == "__main__":
    main()
