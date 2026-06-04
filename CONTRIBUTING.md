# Contributing to PithTrain

Thanks for your interest in PithTrain. This guide covers the principles that
keep the framework coherent, the development workflow, and step-by-step recipes
for the most common extensions.

New to the codebase? Read [`docs/architecture.md`](docs/architecture.md) first —
this guide assumes you know the three-layer structure and the 5-stage decoder
layer abstraction.

---

## Design principles (please read before opening a PR)

PithTrain's value is that an agent *or* a human can read the whole codebase and
trust what they read. Contributions are reviewed against four principles. They
are not bureaucracy — they are the property the project exists to demonstrate.

1. **Compact over comprehensive.** We do not aim for broad model/feature/hardware
   coverage. Prefer the smallest change that solves the problem. Adding a feature
   that only some users need? Consider whether it belongs here at all. Growth is
   fine; growth that respects the other three principles is the bar.

2. **Python-native.** Keep the framework navigable in one language. Custom
   kernels go through a Python DSL (Triton) where possible; reach for external
   compiled libraries (DeepGEMM, FlashAttention) only when they are the right
   tool, and wrap them behind a thin Python operator. Avoid in-tree C++/CUDA
   build steps.

3. **Minimal implicit indirection.** Aim for code where what runs at a call site
   is discoverable by reading it. Avoid introducing plugin registries,
   string-keyed dispatch tables, or runtime "spec" objects that resolve which
   submodule to build; instantiate directly where you reasonably can. Each model
   being a self-contained file that builds its own layers is the pattern to
   follow — we accept a little duplication across model files in exchange for
   local readability.

4. **Ship skills for recurring procedures.** If you add a workflow that will be
   repeated (a new kind of profiling, a migration, a validation), consider
   shipping it as an agent skill under `.claude/skills/` with a scoped
   description, explicit prerequisites, and a verifiable PASS/FAIL check.

When in doubt, optimize for the next reader (human or agent) being able to
understand your change by local reading alone.

---

## Reporting issues & proposing changes

**Before a large or architectural change, open an issue first.** Anything that
touches the pipeline schedule, the parallelism mesh, the model contract, or adds
a dependency is worth a short design discussion before you write the code — both
to confirm it fits the "compact over comprehensive" principle and to save you
from a large PR that has to be reworked. Small, self-contained fixes can go
straight to a PR.

**When filing a bug, make it reproducible.** Training-framework bugs are often
mesh- or hardware-specific, so include:

- the **parallelism mesh** (PP / EP / CP degrees, GPU count) and the model;
- the **GPU architecture** (Hopper / Blackwell) and CUDA / PyTorch versions;
- a **minimal repro** — the smallest config or command that triggers it, plus
  the full traceback;
- whether it reproduces with `fp8_training="disabled"` (to isolate FP8-path issues).

---

## Development setup

Requires an NVIDIA Hopper (SM90) or Blackwell (SM100) GPU, CUDA >= 13.0, and
Python ≥ 3.12. We use [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mlc-ai/pith-train.git && cd pith-train
uv venv
uv sync                  # installs dev + runtime deps (contributors use this, not `uv pip install .`)
pre-commit install       # optional but recommended: run hooks on commit
```

### Linting & formatting (Ruff)

CI and pre-commit enforce these. Run before pushing:

```bash
ruff check --fix pithtrain/
ruff format pithtrain/
pre-commit run --all-files
```

Style: 100-char lines, double quotes, `py312` target. Rule sets `E, F, I, W`
(ignoring `E501`, `E731`). First-party import root is `pithtrain`.

---

## Testing

**Test whatever you changed** — match the test surface to your change rather
than running everything. Kernel, operator, and layer changes have single-GPU
unit tests under `tests/` (run them with `pytest`); changes that touch the
engine (the scheduler, parallelism, checkpointing) should also go through the
multi-GPU integration test, which boots DualPipeV with FSDP:

```bash
bash tests/test_fsdp.sh    # multi-GPU integration (torchrun); boots DualPipeV + FSDP
```

Tests skip gracefully when their dependencies aren't met (no CUDA, an optional
package missing, too few GPUs), so run the ones relevant to your change on
hardware that can actually exercise them.

**Gotchas worth knowing (they will bite you otherwise):**

- `F.grouped_mm` may write NaN to padding rows beyond `grouped_mm_offs[-1]`.
  Always truncate to `[:actual_M]` before comparing outputs.
- FP8 correctness uses normalized squared error (`calc_diff`), typically with a
  `< 1e-3` threshold — not exact equality.
- Multi-GPU tests need `torchrun` (see `tests/test_fsdp.sh`).

### Validating training correctness

Numerical changes (kernels, the schedule, FP8, precision) should not silently
change the loss. The `validate-correctness` skill compares per-step loss curves
between your branch and a base branch at a chosen model and PP/EP/CP mesh. Use it
for anything that could perturb numerics, and report the result in your PR.

---

## Extension recipes

### Add a new model

Use the **`add-new-model`** skill — it covers the full scope below and runs the
tests. The shape of the work:

1. **Model file** — `pithtrain/models/<model>.py`, self-contained, implementing
   `ModelProtocol` and `DecoderLayerProtocol` from
   [`models/interface.py`](pithtrain/models/interface.py). The decoder layer must
   expose `forward_attn` (stage 1), `forward_mlp` (stage 3), `forward_aggregate`
   (stage 5), and `reference_forward` (a plain forward for correctness
   validation). Build linears via `layers/factory.py` so the model is FP8/BF16
   agnostic. Copy an existing model (e.g. `qwen3_moe.py`) as a template
   rather than abstracting a shared base — this is the no-indirection principle
   in action.
2. **Wiring** — register the model where models are constructed (`setup_model`),
   add FSDP wrapping (`apply_fsdp`), and a config under
   `examples/pretrain_lm/<model>/` (`script.py` + `config.json`).
3. **Checkpoint conversion** (optional) — a converter under
   `tasks/convert_checkpoint/` if you want HuggingFace import/export.
4. **Tests** — add the model to `tests/test_fsdp.sh` and bring it up from
   pp=1/ep=1 to pp=2/ep=2, plus a single-GPU inference test.

### Add a new operator / kernel

1. Implement the kernel in `pithtrain/operators/<op>.py` (Triton preferred).
2. **Ship a PyTorch reference implementation in the same module** — this is the
   layer's contract and what the test compares against.
3. Add `tests/test_<op>.py` comparing fused vs reference (use `calc_diff` and an
   appropriate tolerance for reduced precision; remember the `grouped_mm`
   padding gotcha).
4. Optionally add a benchmark under `benchmarks/operators/`.

### Add a training feature or subsystem

Follow the **Cfg / Ctx / context-manager** pattern (see
[`docs/architecture.md` §10](docs/architecture.md)):

- A `*Cfg` dataclass for the user-facing knobs (`@dataclass(init=False,
  slots=True)`, inheriting `SlottedDefault`).
- A `*Ctx` for derived runtime state.
- A `*_context` manager that sets up and tears it down, entered from the task's
  `ExitStack`.

Keep new knobs documented with field docstrings (as in `DistributedCfg`).

### Change the pipeline schedule or parallelism

These live in `pithtrain/dualpipe/` and `pithtrain/modules/distributed.py` and
touch correctness and performance broadly. Expect to:

- run `bash tests/test_fsdp.sh`,
- validate loss-curve parity (`validate-correctness`), and
- ideally capture an nsys profile (`capture-nsys-profile` /
  `analyze-nsys-profile`) to confirm overlap didn't regress.

---

## Pull request checklist

- [ ] Change is the smallest that solves the problem; avoids unnecessary new implicit indirection.
- [ ] `ruff check` and `ruff format` pass (or `pre-commit run --all-files`).
- [ ] Relevant unit tests added/updated and passing.
- [ ] Multi-GPU path exercised (`tests/test_fsdp.sh`) if you touched the engine.
- [ ] Numerical changes validated for loss-curve parity (`validate-correctness`).
- [ ] New config knobs have field docstrings; user-facing changes noted in the PR.
- [ ] New recurring workflow shipped as a skill, if applicable.

---

## License

By contributing, you agree your contributions are licensed under the
[Apache 2.0 License](LICENSE). Note that `pithtrain/dualpipe/` contains code
derived from DeepSeek's DualPipe (MIT) — see `pithtrain/dualpipe/LICENSE` and the
project-root `NOTICE`; keep attribution intact when modifying those files.
