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

## Research and data integrity

- Start every research task with the backtest research spine: `.devin/skills/backtest-research-spine/SKILL.md`.
- Search the memory MCP (`.devin/skills/memory/SKILL.md`) for prior findings on the same signal, invariant, or parameter before starting new research.
- Run `scripts/validate_data_integrity.py` and `make ci` before wiring a new signal or parameter into the engine.
- Use MCP tools (`context7` for library docs, `deepwiki` for repo architecture, `parallel` for live web research, `memory` for prior findings) and record a source corpus in every research artifact.
- Cross-reference DeceptionLeaderBot research findings in `DeceptionLeaderBot/docs/research/` before changing shared primitives.

## Quality sweep

- Run the local A+ sweep before every PR: `python scripts/quality_sweep.py --ci-runs 2`.
- See `.devin/skills/code-quality/quality-sweep/SKILL.md` for the full sweep protocol, rule table, and auto-fix guidance.
- Do not auto-fix `src/` modules that touch accounting invariants, margin math, fee models, or fill simulation without a regression test.

## Workspace process

- Batch related changes into one PR; see `.github/PULL_REQUEST_TEMPLATE.md` and `CONTRIBUTING.md`.
- Delete merged remote branches promptly; use `python scripts/workspace_health_report.py` to find candidates.
- Reusable trading primitives that overlap with `DeceptionLeaderBot` should be proposed as a shared library, not copy-pasted.

## Parallel & batch tool-calling

- Batch independent reads, searches, dry-run checks, and MCP lookups in parallel. Keep ordered gates (`make ci`, `validate_data_integrity.py`, baseline runs) sequential.
- Use `asyncio.gather` or `ThreadPoolExecutor` with bounded concurrency for independent I/O in scripts.
- Never parallelize destructive or data-modifying operations.

Full rule files:
- `.cursor/rules/anti-circular-debug.mdc`
- `.claude/rules/anti-circular-debug.md`
