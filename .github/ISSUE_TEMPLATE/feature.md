---
name: Feature / Enhancement
about: A single deliverable that rolls up to a theme epic.
title: "<scope>: <short description>"
labels: ["type:feature"]
---

## Problem / Request
<!-- What's needed? One or two sentences. -->

## User Story
> As a <role>, I want <capability>, so that <outcome>.

## Motivation
<!-- Why now; what it enables; what breaks if we don't have it. -->

## Proposed API
```python
# Signatures, types, or config snippets.
```

## Design Snapshot
<!-- Lead with the exact snippet or config the implementer should reproduce. Delete if not needed. -->
```python
# Example code / config / CLI invocation
```

<!--
REQUIRED FOR ALGORITHMIC / NUMERICAL ISSUES

If the issue implements an algorithm, numerical method, probabilistic
update, optimizer, filter, solver, approximation, linearization, or any
other mathematically-defined behavior, do not delete this section.

Treat this section as part of the spec. Include the equations,
parameterization/sign conventions, approximations, invariants to test,
and any numerical-stability notes another implementer would need to
complete the work without reopening the original design docs.

Use normal GitHub math syntax:
- inline: `$...$`
- display: `$$...$$`
-->
## Mathematical Notes

Suggested prompts for algorithmic issues:
- Defining equations:
  $$ ... $$
- Parameterization / sign conventions:
  $$ ... $$
- Approximation / factorization used:
  $$ ... $$
- Identities or invariants tests should assert:
  $$ ... $$

## References & Existing Code
- Design doc / spec: `<path or URL>`
- Reference impl: `<path:line>`
- Related prior art: `<repo / paper / issue>`

## Implementation Steps
- [ ] Add or update `<symbol>` in `somax/<module>.py` or `somax/_src/<module>.py`
- [ ] Wire CLI / config / docs integration if needed
- [ ] Add or update tests

## Definition of Done
- [ ] Code lands at the intended path
- [ ] Public API exported through `somax/__init__.py` or a public module if user-facing
- [ ] Tests pass: `make test-cov`
- [ ] Lint + typecheck pass: `make lint && make typecheck`
- [ ] Public docstrings or docs page updates land where needed

## Testing
- [ ] Unit test: `<what it asserts>`
- [ ] Regression or integration test: `<what it asserts>`

## Documentation
- [ ] Update the relevant page under `content/`
- [ ] Add notebook or recipe if the change is user-facing
- [ ] Update docstrings if public API changed

## Relationships
<!--
Parent:      make gh-sub PARENT=<parent#> CHILDREN="<this#>"
Blocked by:  make gh-block ISSUE=<this#> BLOCKED_BY=<other#>
Blocks:      make gh-block ISSUE=<other#> BLOCKED_BY=<this#>
Related:     prose only
-->
- Parent (theme epic): #
- Blocked by: #
- Blocks: #
- Related: #
