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

- `topic` — what to build/improve (e.g., "best MEXC position sync", "cleanest task-ranking UI").
- `repo` — auto-detected from `git remote -v` in the current directory; override with `owner/repo` if wrong.
- `timebox` — default `2 hours`.
- `constraints` — optional guardrails (read `AGENTS.md` / `README.md` and list `NEVER TOUCH` values).

## 1. Pre-flight

1. Capture `topic` from the slash command.
2. Detect repo: `git remote -v` or current working directory.
3. Read `AGENTS.md` and `README.md` if they exist; record repo-specific `NEVER TOUCH` values and safety rules.
4. Create `docs/competitions/{slug}/` (slug = `{repo}-{topic}-YYYYMMDD`) and write `BRIEF.md` with:
   - Topic, repo, timebox, constraints
   - Judging rubric:
     | Criterion | Weight |
     |---|---|
     | Correctness / safety | 30% |
     | Performance / throughput | 20% |
     | Code clarity & maintainability | 20% |
     | Evidence & citations | 20% |
     | Git cleanliness | 10% |
5. `git status --short` must be clean-ish. If there are uncommitted user changes, do not stage them; only the campaign files will be committed by the agents.

## 2. Spawn the three competitors

Use `devin_mcp` → `devin_session_create` with `devin_mode="lite"` (or `run_subagent` with `model="swe-1-7"` if available). Launch in parallel.

| Agent | Persona | Edge |
|---|---|---|
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
- Work only in the assigned repo.
- FULL CARTE BLANCHE: use every Devin tool, every MCP server, every file in the repo, and any source on the open web.
- Cite evidence for every design choice and every factual claim.
- WORK CLEAN ON GIT:
  * Branch: `devin/compete-{slug}-{agent}`
  * Commit often with clear messages
  * Open a PR against the repo default branch when done
  * Never force push, never commit to `main`/`master`, never touch secrets
- Respect repo `AGENTS.md`, `NEVER TOUCH` values, and safety rules.
- Do not modify the other agents' branches or PRs.
- Stop when timebox expires or a PR is ready — whichever comes first.
- Return: PR URL, summary, and self-score against the rubric.
```

Record session IDs and branch names in `docs/competitions/{slug}/AGENTS.md`.

## 3. Cook

- `devin_session_gather` all three.
- Do not interrupt.
- If an agent is blocked on a decision, relay it to the user or resolve if it's a prompt error.

## 4. Gather submissions

1. `devin_session_interact` → `get_messages` and `get_attachments` for each child.
2. `ROUNDUP.md` with PR URL, files changed, summary, self-score.
3. Review each PR diff locally.

## 5. Judging

Write `docs/competitions/{slug}/JUDGE.md`:

- Score each agent against the rubric (1-10 per criterion).
- Declare winner.
- Optional: if user asked, create a synthesis branch `devin/compete-{slug}-synthesis` and PR combining the best pieces.
- Label all compete PRs with `compete-{slug}`.

## 6. Final report

Return to user:
- Three PR links.
- Winner and why.
- One-paragraph critique per entry.
- Any risks/follow-ups.
- `PASS` with winner/synthesis PR link.

## Hard stops

- Free/lite models only for children (`swe-1-7` / `lite`). No paid/frontier subagents.
- No live trading restarts, no production service mutations.
- Never change repo `NEVER TOUCH` values without explicit user approval.
- No secrets, no `.env`, no direct `main`/`master` commits.
