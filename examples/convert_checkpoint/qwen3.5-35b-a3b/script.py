from pathlib import Path

from huggingface_hub import snapshot_download

from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg, launch

# Qwen3.5-35B-A3B ships as a vision-language checkpoint; only the text tower
# (``model.language_model.*`` + ``lm_head``) is converted. The vision tower and
# the multi-token-prediction (``mtp.*``) head are skipped.
cfg = ConvertCheckpointCfg()
cfg.operation = "hf2dcp"
cfg.load_path = Path("workspace/checkpoints/qwen3.5-35b-a3b/hf-import")
cfg.save_path = Path("workspace/checkpoints/qwen3.5-35b-a3b/torch-dcp/step-00000000")

if __name__ == "__main__":
    snapshot_download(repo_id="Qwen/Qwen3.5-35B-A3B", local_dir=cfg.load_path)
    launch(cfg)

cfg = ConvertCheckpointCfg()
cfg.operation = "dcp2hf"
cfg.load_path = Path("workspace/checkpoints/qwen3.5-35b-a3b/torch-dcp/step-00000000")
cfg.save_path = Path("workspace/checkpoints/qwen3.5-35b-a3b/hf-export")

if __name__ == "__main__":
    launch(cfg)
