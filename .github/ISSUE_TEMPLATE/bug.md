---
name: Bug report
about: Something isn't working correctly.
title: "bug: <short description>"
labels: ["type:bug"]
---

## Problem
<!-- What's broken? One or two sentences. -->

## Reproduction
```python
# Minimal reproducing example.
```

## Expected Behavior
<!-- What should happen. -->

## Actual Behavior
<!-- What happens instead. Include traceback if relevant. -->

## Environment
- Somax version:
- Python:
- Platform:
- Key dependency versions:

## References & Existing Code
- Related code: `<path>`
- Related issue / PR: #

## Implementation Steps (fix)
- [ ] Reproduce locally
- [ ] Root-cause analysis
- [ ] Fix at `somax/<path>` or adjacent CLI / config / docs glue
- [ ] Add regression test

## Definition of Done
- [ ] Regression test captures the bug
- [ ] Fix lands and regression test is green
- [ ] `make test-cov && make lint && make typecheck` are green

## Testing
- [ ] Regression test at `tests/<path>::<name>` — asserts <what>

## Documentation
- [ ] N/A, or update the relevant page under `content/`

## Relationships
<!--
Apply the native GitHub links after the issue is opened:
  Parent:      make gh-sub PARENT=<parent#> CHILDREN="<this#>"
  Blocked by:  make gh-block ISSUE=<this#> BLOCKED_BY=<other#>
  Blocks:      make gh-block ISSUE=<other#> BLOCKED_BY=<this#>
Helper: `.github/scripts/link-issues.sh`
-->
- Parent (theme epic, if any): #
- Blocked by: #
- Blocks: #