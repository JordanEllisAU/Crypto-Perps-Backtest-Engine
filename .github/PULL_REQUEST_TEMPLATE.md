<!-- Thank you for contributing. Please batch related changes into a single PR; see CONTRIBUTING.md for guidance. -->

## Summary
Briefly describe the change and why it is needed.

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation / process
- [ ] Workspace hygiene

## Related changes
Link related PRs or issues, especially any corresponding change in `JordanEllisAU/DeceptionLeaderBot`.

## Risk / accounting-path checklist
- [ ] I have NOT changed accounting invariants, margin math, or fee models without test coverage.
- [ ] Every new risk or execution branch has a test or is covered by an invariant check.
- [ ] I ran `python -m pytest -q` on the narrowest test set that touches the changed code.

## Verification
- [ ] `python -m compileall src`
- [ ] `python -m pytest -q`
- [ ] `python scripts/workspace_health_report.py --leader-repo ../DeceptionLeaderBot` to confirm no new workspace debt.

## Notes for reviewers
Highlight anything that is not obvious from the diff (e.g., why a fill model assumption changed, why this PR intentionally does not include a test, links to DeceptionLeaderBot signal schema).
