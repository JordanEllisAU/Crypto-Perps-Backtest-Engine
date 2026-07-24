# AGENTS.md — Agent Directives

## Anti-circular debugging and hygiene directives

These directives are mandatory for all agents working in this repository. They replace previous ad-hoc orchestration tooling and prevent circular debugging.

- Stop on a blocker; do not re-run the same failing command more than twice without a new hypothesis.
- Run the repo's local lint/test/build gates before every commit (e.g., `make ci`, `npm test`, `pytest`, `npm run build`).
- No bare `print()` / `console.log()` / `printf()` in gate scripts; use structured logging.
- Generated files must be deterministic (no timestamps, strip trailing whitespace).
- Do not modify tests to pass; fix the code or skip with a reason.
- Use feature branches and PRs; never `git reset --hard` on the default branch.
- No session/chat artifacts in commits (`.claude/*.json`, `.vscode/`, `.playwright-mcp/`, scratch files).
- Document root cause, not just the fix.

Full rule files:
- `.cursor/rules/anti-circular-debug.mdc`
- `.claude/rules/anti-circular-debug.md`
