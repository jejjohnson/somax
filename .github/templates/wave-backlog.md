<!--
WAVE BACKLOG DRAFT — COPY-TO-USE TEMPLATE

Purpose:
  Draft a whole wave of GitHub issues as one reviewable markdown file
  BEFORE opening the issues. Keeps shared context in one place, uses
  stable draft IDs so children can cross-reference each other, and
  lets the whole backlog be reviewed in a single scroll.

How to use:
  1. Copy this file into your project's `.plans/` directory
     (gitignored). Rename to describe the wave.
  2. Pick a short project prefix and number drafts sequentially:
     <PREFIX>-01, <PREFIX>-02, …
  3. Fill in the wave-level shared context once at the top, then
     draft each issue body. Review the whole file as a unit.
  4. When ready, open each draft as a real GitHub issue using the
     matching `.github/ISSUE_TEMPLATE`.

Conventions:
  - For algorithmic / numerical issues, math is part of the spec, not
    decoration. Include update rules, factorization identities,
    parameterization/sign conventions, approximations, and invariants
    tests should pin down.
  - Use normal GitHub math syntax:
    inline `$...$`, display `$$...$$`.
  - Delete sections that do not apply. Rename headings when a more
    specific label fits the issue type better.
-->

# [Wave N] <title>

---

## Shared Context

<!--
One or two paragraphs that are TRUE for every issue in this wave.
The goal is that another contributor can implement this wave without
opening private design docs or asking follow-up questions.
-->

## Design Snapshot

```python
# Cross-cutting signatures, protocols, or conventions.
```

## Intended Package Layout

```text
somax/
  _src/
    ...
```

---

# [Wave N] <wave title>
Draft ID: `<PREFIX>-01`

## Goal

## Why This Wave Exists

## Canonical Epics
- [ ] `<PREFIX>-02` [Epic] N.A <theme title>
- [ ] `<PREFIX>-03` [Epic] N.B <theme title>

## Sequential Dependencies

## Definition of Done
- [ ] All theme epics closed
- [ ] Tests / lint / format / typecheck all pass on `main`

## Relationships
- Blocks `<PREFIX-of-next-wave>-01`
- Related: <design docs / prior art>

---

# [Epic] N.A <theme title>
Draft ID: `<PREFIX>-02`

## Theme

## Parent Wave
`<PREFIX>-01`

## Motivation

## Canonical Child Issues
- [ ] `<PREFIX>-05` <feature title>
- [ ] `<PREFIX>-06` <feature title>

## Execution Notes
<!--
If this theme is algorithmic, record the shared mathematical conventions
all child issues must preserve.
-->

## Definition of Done
- [ ] All child issues closed
- [ ] Tests for the theme's surface land and pass

## Relationships
- Parent wave: `<PREFIX>-01`

---

# <scope>: <feature title>
Draft ID: `<PREFIX>-05`

## Problem / Request

## User Story
> As a <role>, I want <capability>, so that <outcome>.

## Proposed API
```python
# Signatures, types, docstring stubs.
```

## Design Snapshot

## Mathematical Notes
<!--
For algorithmic issues, this section is required.

Recommended contents:
- defining equations
- parameterization / sign conventions
- approximation / factorization used
- identities or invariants tests should pin down
-->

## Implementation Steps
- [ ] Add `<symbol>` at the intended path
- [ ] Add or update tests
- [ ] Wire docs / CLI / config integration if needed

## Definition of Done
- [ ] Code lands at the intended path
- [ ] Tests pass
- [ ] Lint + typecheck pass

## Testing
- [ ] Unit test: `<what it asserts>`
- [ ] Regression / integration test: `<what it asserts>`

## Documentation
- [ ] Update docs or examples if user-facing

## Relationships
- Parent epic: `<PREFIX>-02`
- Blocked by: `<PREFIX>-NN`
- Blocks: `<PREFIX>-NN`
- Related: `<PREFIX>-NN`
