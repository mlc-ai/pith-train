from pathlib import Path

from huggingface_hub import snapshot_download

from pithtrain.modules.checkpoint import find_checkpoint
from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg, launch

# The released checkpoint also holds a multi-token-prediction layer (model.layers.47), which the
# model does not load and an export does not write.
root = Path("workspace/checkpoints/glm-4.7-flash")

cfg = ConvertCheckpointCfg()
cfg.save_path = Path(root, "hf-export")

if __name__ == "__main__":
    # The released weights, loaded by setting training.hf_import_path to this directory.
    snapshot_download(repo_id="zai-org/GLM-4.7-Flash", local_dir=Path(root, "hf-import"))
    # Export the newest training checkpoint, if there is one.
    if (step := find_checkpoint(root)) is not None:
        cfg.load_path = Path(root, "torch-dcp", "%08d" % step)
        launch(cfg)
