Reviewing ${GITHUB_REPOSITORY} PR #${GITHUB_EVENT_ISSUE_NUMBER} for CONSISTENCY only: documentation and comments that this change makes out of date. The PR branch is already checked out in the current working directory. Use `gh pr comment` for top-level feedback, and `mcp__github_inline_comment__create_inline_comment` (with `confirmed: true`) to highlight specific code issues.

This is an agent-native repo where docs and comments must stay greppable-accurate: AGENTS.md, CLAUDE.md, docs/, .agents/skills/, module and function docstrings, and inline comments are load-bearing. A renamed symbol, moved/deleted file, changed signature, or altered behavior in this PR that any of these still describes the old way is a defect.

For every symbol renamed, file moved or deleted, signature changed, or behavior altered in the diff, grep the repo for prose or comments that still describe the old state, and flag each stale reference. Also flag: new docs/comments added by this PR that are already inaccurate, and example or config snippets that no longer match the code.

Flag post-mortem phrasing in docstrings or comments this PR adds or edits: they must read as if the code was always in its new state, describing what the code is, not narrating the change that produced it. Phrasings like "previously X", "used to", "renamed from", "now does Y", "changed to", or "no longer" in a docstring or comment are a defect even when technically accurate, because the next reader has no memory of the old state.

Separately, when you cite a fix, describe the corrected text as it should read, never narrate the change ("was X, now Y").

Post findings with `gh pr comment` under a "### Consistency Review" heading; cite each with a permalink (full commit SHA + line range). If nothing is stale, say so in one line.
