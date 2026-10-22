"""Local CPU conversion only; this entrypoint never downloads weights."""

import argparse
from pathlib import Path

from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg, launch
from pithtrain.tasks.convert_checkpoint.qwen3_omni_moe import SIDECAR, Qwen3OmniTextConverter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("hf2dcp", "dcp2hf"))
    parser.add_argument("load_path", type=Path)
    parser.add_argument("save_path", type=Path)
    parser.add_argument(
        "--text-config", type=Path, help="Required for native DCP without converter sidecar"
    )
    parser.add_argument("--prefix", choices=("thinker.", ""), default="thinker.")
    parser.add_argument(
        "--expert-layout", choices=("split", "fused"), default="fused", help="HF export layout"
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Explicitly exclude known media/Talker keys on import",
    )
    parser.add_argument(
        "--restore-resume", action="store_true", help="HF companion to original strict container"
    )
    parser.add_argument("--max-shard-size", type=int, default=8 * 1024**3)
    args = parser.parse_args()
    if (
        args.operation == "dcp2hf"
        and not args.text_config
        and not (args.load_path / SIDECAR).exists()
        and not (args.load_path / "strict-manifest.json").exists()
    ):
        parser.error("Omni native DCP export requires --text-config (or converter sidecar)")
    if args.operation == "hf2dcp" and not Qwen3OmniTextConverter().detect_hf(args.load_path):
        parser.error("Omni import requires an Omni/Thinker/text config.json")
    if args.restore_resume and args.operation != "hf2dcp":
        parser.error("--restore-resume requires hf2dcp")
    cfg = ConvertCheckpointCfg()
    cfg.operation, cfg.load_path, cfg.save_path = args.operation, args.load_path, args.save_path
    cfg.omni_text_config, cfg.omni_prefix = args.text_config, args.prefix
    cfg.omni_expert_layout, cfg.omni_text_only = args.expert_layout, args.text_only
    cfg.omni_restore_resume = args.restore_resume
    cfg.max_shard_size = args.max_shard_size
    launch(cfg)


if __name__ == "__main__":
    main()
