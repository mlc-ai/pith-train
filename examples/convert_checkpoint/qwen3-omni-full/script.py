"""Convert complete Qwen3-Omni components locally; never download weights."""

import argparse
from pathlib import Path

from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg, launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("hf2dcp", "dcp2hf"))
    parser.add_argument("load_path", type=Path)
    parser.add_argument("save_path", type=Path)
    parser.add_argument(
        "--components", required=True, choices=("thinker", "talker", "code2wav", "all")
    )
    parser.add_argument(
        "--config", type=Path, help="Exact full/component config for native model-only DCP export"
    )
    parser.add_argument("--namespace", choices=("component", "standalone"), default="component")
    parser.add_argument("--expert-layout", choices=("split", "fused"), default="fused")
    parser.add_argument("--max-shard-size", type=int, default=8 * 1024**3)
    args = parser.parse_args()
    if args.components == "all" and args.namespace == "standalone":
        parser.error("--components all requires --namespace component")
    cfg = ConvertCheckpointCfg()
    cfg.operation, cfg.load_path, cfg.save_path = args.operation, args.load_path, args.save_path
    cfg.omni_components, cfg.omni_component_config = args.components, args.config
    cfg.omni_component_namespace = args.namespace
    cfg.omni_expert_layout, cfg.max_shard_size = args.expert_layout, args.max_shard_size
    launch(cfg)


if __name__ == "__main__":
    main()
