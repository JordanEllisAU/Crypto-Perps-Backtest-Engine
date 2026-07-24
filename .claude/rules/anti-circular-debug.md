# Anti-Circular Debugging and Hygiene Directive

**Applies to:** All agent work in this repository.

These directives are mandatory. They replace Taskmaster/GitHub Actions slop and prevent circular debugging.

## Stop on a blocker, do not loop

- If the same command fails three times, change the approach. Do not run it again without a new hypothesis.
- Do not re-run lint/test/build gates as a ritual. Read the error and fix the cause.
- Platform-level failures (e.g., GitHub Actions `startup_failure`, API 403) must be reported via `report_blocker` and `message_user`. Do not spawn more PRs hoping the problem disappears.

## Run the repo's local gates before every commit

Identify the project's lint/test/build commands (e.g., `make ci`, `npm test`, `pytest`, `npm run build`) and run them before committing.

## No bare logging in gate scripts

Any validation/build/reporting script must use a shared logger or structured logging. Bare `print()` / `console.log()` / `printf()` is forbidden.

## Generated files must be deterministic

- No timestamps or random values in build outputs.
- Strip trailing whitespace in generated text files.
- Do not stage regenerated artifacts unless the source of truth changed.

## Do not modify tests to pass

Fix the code, or add a `skipif` / `skip` marker with a reason if a runtime is missing.

## Use feature branches and PRs

- Branch from the default branch.
- Never `git reset --hard` on the default branch or `main`.
- Open a PR for any non-trivial change.

## No session/chat artifacts in commits

Ignore and never commit `.claude/*.json`, `.taskmaster/`, `.vscode/`, `.playwright-mcp/`, `.cursor/learning/`, scratch files, or chat logs.

## Document root cause, not just the fix

When fixing a bug, update the project log or `AGENTS.md` / `CLAUDE.md` with symptom, root cause, fix, and verification command.
