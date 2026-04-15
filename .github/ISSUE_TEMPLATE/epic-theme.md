---
name: Epic — Theme (L2)
about: A parallel-safe group of issues under a Wave epic. See CONTRIBUTING.md for the two-layer epic model.
title: "[EPIC] <theme title>"
labels: ["type:epic-theme"]
---

## Theme
<!-- One-sentence outcome for this group. -->

## Parent Wave
- Wave epic: #
- Wave label: `wave:N`
- Milestone: `vX.Y-<slug>`

## Motivation
<!-- Why this group exists; what it ships together. -->

## Issues
- [ ] #<issue> — <short description>
- [ ] #<issue> — <short description>

## Execution Notes
<!-- Delete if the issues are fully parallel. -->

## Parallelism
- Can run in parallel with: #
- Blocked by (inside this wave): #
- Must complete before: #

## Definition of Done
- [ ] All child issues closed
- [ ] Tests for the theme's surface land and pass
- [ ] Docs or API updates land where needed

## Relationships
<!--
Parent (wave):  make gh-sub PARENT=<wave#> CHILDREN="<this#>"
Sub-issues:     make gh-sub PARENT=<this#> CHILDREN="<child1#> <child2#>"
Blocked by:     make gh-block ISSUE=<this#> BLOCKED_BY=<other-theme#>
-->
- Parent: #<wave-epic>
- Related: #