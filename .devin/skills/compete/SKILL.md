---
id: compete
name: compete
description: |
  Friendly but fiercely competitive 3-agent build-off. Spawns three lite/free subagents with the
  same goal — build the best system for a stated topic — using the entire repo and the open web,
  then judges the winner and lands a champion PR. Clean git only.
tier: standard
scope: workspace
---

# /compete — Competitive 3-Agent Build-Off

**When to invoke:** `/compete <topic>` or "Run a compete on `<topic>`".

**Goal:** Three `lite`/`swe-1-7` subagents race to build the best system for a topic. Same goal, different lenses, clean git, full research freedom. A final judging round picks the winner and can open a synthesis PR.

## Inputs

- `topic` — what to build/improve.
- `repo` — auto-detected from `git remote -v` in the current directory; do not override except to fix a wrong detection.
- `run_tag` — optional unique tag if the same topic is run again (default: none).
- `timebox` — default `2 hours`.
- `constraints` — optional guardrails (read `AGENTS.md` / `README.md` and list `NEVER TOUCH`/protected paths and safety rules).

## 1. Pre-flight

1. Capture `topic` from the slash command.
2. Detect repo: `git remote -v` or current working directory.
3. Read `AGENTS.md` and `README.md` if they exist; record repo-specific `NEVER TOUCH` values, protected paths, and safety rules in `constraints`.
4. Check workspace cleanliness: `git status --short` before creating any campaign files. If there are uncommitted user changes outside `docs/competitions/`, stop and ask before continuing.
5. Derive a deterministic slug: `{repo}-{topic}` normalized (lowercase, alphanumerics/hyphens). If `run_tag` is given, append `-{run_tag}`. If the directory already exists, append `-2`, `-3`, etc.
6. Create `docs/competitions/{slug}/` and write `BRIEF.md` with:
   - Topic, repo, timebox, constraints, slug
   - Judging rubric:
     | Criterion | Weight |
     |---|---|
     | Correctness / safety | 30% |
     | Performance / throughput | 20% |
     | Code clarity & maintainability | 20% |
     | Evidence & citations | 20% |
     | Git cleanliness | 10% |
7. Create and switch to branch `devin/compete-{slug}-judge`. Commit `docs/competitions/{slug}/` with message `Start /compete campaign: {topic}`. This branch holds all campaign docs and is separate from the competitors' branches.

## 2. Spawn the three competitors

Use `devin_mcp` → `devin_session_create` with `devin_mode="lite"` (or `run_subagent` with `model="swe-1-7"` if available). Launch in parallel. Each subagent runs in an isolated session with its own repo clone.

| Agent | Edge |
|---|---|
| **A — Speed Demon** | Fastest working system, throughput/latency first, pragmatic cuts. |
| **B — Safety Sage** | Bulletproof correctness, edge cases, invariants, tests, no regressions. |
| **C — Elegance Architect** | Cleanest design, minimal code, readable architecture, future-proof. |

### Child prompt

```
You are Agent {A|B|C} in a /compete build-off for {repo}.

TOPIC: {topic}
TIMEBOX: {timebox}
CONSTRAINTS: {constraints}

GOAL: Build the best system for the topic. Friendly but fierce — outperform the other two agents on the rubric, stay professional.

RULES:
- Start by invoking the repo's delivery-discipline skill (e.g., `.devin/skills/fact-checked-slop-free-delivery/SKILL.md`) and follow it for the whole task. If none exists, follow repo `AGENTS.md`.
- Work only in the assigned repo. You are on an isolated machine; do not touch the other agents' branches or PRs.
- FULL RESEARCH CARTE BLANCHE: use every Devin tool, every MCP server, and any source on the open web to research and read the repo. You may read any file.
- MODIFICATIONS respect repo `AGENTS.md`, `NEVER TOUCH` values, and protected paths. Do not directly edit protected files (e.g., runtime game files, `.env`, secrets); route changes through the repo's approved builders/ship tools.
- Cite evidence for every design choice and every factual claim.
- WORK CLEAN ON GIT:
  * Branch: `devin/compete-{slug}-{agent}`
  * Run the repo's lint/test/build gates (e.g., `make ci` / `make lint` / `make test`) before each commit
  * Commit only after gates pass; commit often with clear messages
  * Open a PR against the repo default branch when done
  * Never force push, never commit to `main`/`master`, never touch secrets
- Do not modify `docs/competitions/{slug}/` or the other agents' branches/PRs.
- Stop when timebox expires or a PR is ready — whichever comes first.
- Finish with the repo's finish-first stop: run all validators, then report `PASS`/`FAIL`/`UNVERIFIED` and stop.
- Return: PR URL, branch name, summary, self-score against the rubric, finish-first status (`PASS`/`FAIL`/`UNVERIFIED`), and the exact gate commands you ran with exit codes.
```

Record session IDs and branch names in `docs/competitions/{slug}/COMPETITORS.md`.

## 3. Cook

- `devin_session_gather` all three.
- Do not interrupt.
- If an agent is blocked on a decision, relay it to the user or resolve if it's a prompt error.

## 4. Gather submissions

1. `devin_session_interact` → `get_messages` and `get_attachments` for each child.
2. `ROUNDUP.md` with PR URL, branch, files changed, summary, self-score, finish-first status, gate results.
3. For each PR, check out the branch locally and run the repo's lint/test/build gates. Record exit codes.

## 5. Judging

Write `docs/competitions/{slug}/JUDGE.md`:

- Score each agent against the rubric (1-10 per criterion).
- Declare winner.
- Optional: if user asked, create a synthesis branch `devin/compete-{slug}-synthesis` and PR combining the best pieces.
- Label all compete PRs with `compete-{slug}`.

## 6. Final report and verification

Run the repo gates on each competitor branch, the campaign-docs branch, and the synthesis branch (if any). Record every command and exit code.

Return to user:
- Three PR links, the campaign-docs branch link, and the synthesis PR link (if any).
- Gate results per branch.
- Winner and why.
- One-paragraph critique per entry.
- Any risks/follow-ups.
- `PASS` only if the winning PR/synthesis PR passes all repo gates; otherwise `FAIL` or `UNVERIFIED`.

## Hard stops

- Free/lite models only for children (`swe-1-7` / `lite`). No paid/frontier subagents.
- No live trading restarts, no production service mutations.
- Never change repo `NEVER TOUCH` values without explicit user approval.
- No secrets, no `.env`, no direct `main`/`master` commits.
- Do not edit protected paths listed in repo `AGENTS.md`; use approved builder/ship tools.
