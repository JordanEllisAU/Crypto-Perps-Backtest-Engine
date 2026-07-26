# AGENTS.md — Agent Directives

## Anti-circular debugging and hygiene directives

These directives are mandatory for all agents working in this repository. They replace previous ad-hoc orchestration tooling and prevent circular debugging.

- Stop on a blocker; do not re-run the same failing command more than twice without a new hypothesis.
- Run `make ci` before every commit; it compiles all Python files, runs the diff-based slop sentinel, and runs `pytest`.
- No bare `print()` / `console.log()` / `printf()` in gate scripts; use structured logging.
- Generated files must be deterministic (no timestamps, strip trailing whitespace).
- Do not modify tests to pass; fix the code or skip with a reason.
- Use feature branches and PRs; never `git reset --hard` on the default branch.
- No session/chat artifacts in commits (`.claude/*.json`, `.vscode/`, `.playwright-mcp/`, scratch files).
- Document root cause, not just the fix.

## Workspace process

- Batch related changes into one PR; see `.github/PULL_REQUEST_TEMPLATE.md` and `CONTRIBUTING.md`.
- Delete merged remote branches promptly; use `python scripts/workspace_health_report.py` to find candidates.
- Reusable trading primitives that overlap with `DeceptionLeaderBot` should be proposed as a shared library, not copy-pasted.

Full rule files:
- `.cursor/rules/anti-circular-debug.mdc`
- `.claude/rules/anti-circular-debug.md`
