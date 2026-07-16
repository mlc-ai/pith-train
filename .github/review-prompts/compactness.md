Reviewing ${GITHUB_REPOSITORY} PR #${GITHUB_EVENT_ISSUE_NUMBER} for COMPACTNESS only. The PR branch is already checked out in the current working directory.

### How to post

- Post each finding as its own inline comment with `mcp__github_inline_comment__create_inline_comment` (`confirmed: true`), anchored to the exact line(s) it concerns, so each becomes an independently resolvable thread; open its body with a bold **Compactness:** tag so its review dimension is clear at a glance.
- When a fix is small, local, and a confident drop-in, include it as a GitHub ```suggestion block scoped to exactly the commented lines (it replaces them verbatim, so it must be complete and valid). If the fix needs validation or judgment, describe the direction in prose instead of a one-click suggestion.
- Also post a quick top-level summary with `gh pr comment` under a `### Compactness Review` heading: a few sentences of high-level storyline over the issues, so a developer gets the overview before opening the threads. This is a summary, not a place to batch findings; the per-finding detail lives only in the inline threads. The one exception is a finding whose line falls outside this PR's diff, which has no inline thread, so state that one here. If no issue is identified, say so here in one line.
- Most importantly, developers are busy. Write short, direct comments: lead with the issue and its consequence, skip preamble, and do not spend words praising what the PR does well. Hold a strict bar in as few words as the point needs.

### What to look for

PithTrain is a deliberately compact, agent-native codebase. Its stated design (see AGENTS.md and docs/architecture.md) favors local readability over cross-model reuse: implement only what training needs, keep per-model code self-contained, and avoid indirection a reader has to chase. The runtime is also fixed: NVIDIA H100 (SM90) or B200 (SM100) GPUs are required, and every dependency in pyproject.toml is assumed installed, so guarding for their absence is dead defensiveness. Judge the change against that bar.

Flag only changes this PR introduces that add abstraction it did not need. Look for:
- New layers of indirection (wrappers, factories, registries, base classes, managers) where a direct call or inline code would read better.
- Parameters, config fields, or generality added for cases the codebase does not actually train (inference-only knobs, unused flexibility, speculative hooks).
- Helpers hoisted into shared modules that would read better duplicated locally per the codebase's local-editability convention.
- Re-wrapping or renaming that adds surface without removing any.
- Defensive guards for conditions that cannot occur here: `try`/`except ImportError` fallbacks or availability checks for required dependencies (e.g. deep_gemm), and CUDA-presence or GPU-architecture guards.

Be concrete and conservative. Only raise a point if you can name the simpler alternative.
