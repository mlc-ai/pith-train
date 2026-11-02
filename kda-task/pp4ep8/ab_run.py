"""
PP4-EP8 A/B driver: load examples/pretrain_lm/<model>/script.py from the arm's worktree (the cwd) unchanged and
override only the run, not the model: PP=4, EP=8, fp8 per arm, absolute dataset path, a fresh per-run save_location
(so no checkpoint is resumed), no checkpoint save, and max_steps = STEPS. The WSD scheduler keeps the script's own
max_steps (4096) so the learning rate of the first STEPS steps equals the real config's.
Usage (inside torchrun, PYTHONPATH = the arm's worktree): ab_run.py <model> <bf16|fp8> <steps> <save_dir>
"""

import importlib.util
import sys
import types
from functools import partial
from pathlib import Path

model, dtype, steps, save_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
spec = importlib.util.spec_from_file_location("ab_cfg_module", f"examples/pretrain_lm/{model}/script.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cfg = module.cfg

cfg.dataset = Path("/mnt/weka/shrd/ci/haok/haok-pith-train") / cfg.dataset
cfg.distributed.pipeline_parallel_size = 4
cfg.distributed.expert_parallel_size = 8
training = cfg.training
training.fp8 = dtype == "fp8"
script_max_steps = training.max_steps
sched = training.scheduler
if isinstance(sched, partial):
    shadow = types.SimpleNamespace(max_steps=script_max_steps, lr=training.lr)
    training.scheduler = lambda cfg_, sched=sched: sched.func(shadow, *sched.args, **sched.keywords)
training.max_steps = steps
training.save_interval = 10**9
training.save_location = save_dir

if __name__ == "__main__":
    print(f"ab_run model={model} dtype={dtype} steps={steps} script_max_steps={script_max_steps} pp=4 ep=8 fp8={training.fp8}", flush=True)
    module.launch(cfg)
