---
name: Epic — Wave (L1)
about: A release-scoped mega-epic grouping theme epics under one milestone. See CONTRIBUTING.md for the two-layer epic model.
title: "[EPIC] Wave N: <title>"
labels: ["type:epic-wave"]
---

## Goal
<!-- One sentence: what outcome does this wave deliver? -->

## Wave / Milestone
- Wave: `wave:N`
- Milestone: `vX.Y-<slug>`

## Motivation
<!-- Why this wave now; what it unlocks; what it blocks. -->

## Theme Epics (parallel-safe)

### Section A — <theme>
- [ ] #<theme-epic-issue>

### Section B — <theme>
- [ ] #<theme-epic-issue>

## Sequential Dependencies
<!-- e.g. Section A -> Section B -->

## Definition of Done (Wave)
- [ ] All theme epics closed
- [ ] Milestone released
- [ ] Tests, lint, format, and typecheck all pass on `main`
- [ ] Docs updated or published as needed

## Relationships
<!--
Sub-issues:  make gh-sub PARENT=<this#> CHILDREN="<theme#> <theme#>"
Blocked by:  make gh-block ISSUE=<this#> BLOCKED_BY=<prior-wave#>
Blocks:      make gh-block ISSUE=<next-wave#> BLOCKED_BY=<this#>
-->
- Blocked by: #
- Blocks: #
- Related: #