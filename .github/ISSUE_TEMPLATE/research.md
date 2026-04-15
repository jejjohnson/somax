---
name: Research / Comparative Analysis
about: Investigate prior art and map it onto Somax. Produces a prioritized roadmap of follow-up issues.
title: "research: <short topic>"
labels: ["type:research"]
---

# <Title>

## Context

<!-- What is being investigated, and what roadmap or design question does it inform? -->

## 1. What `<subject>` Contains

### Package / Project Structure
```text
<tree or component map>
```

### Core Data Structures

| Structure | Fields | Purpose |
|---|---|---|
| `<name>` | `<fields>` | <role> |

### Core Algorithms / Features

#### A. <Algorithm area>
- `<function>(args)` — <what it does>

#### B. <Algorithm area>
- `<function>(args)` — <what it does>

## 2. Comparison With Somax

### A. Already In Somax

| Subject feature | Somax equivalent | Path | Notes |
|---|---|---|---|
| `<feature>` | `<our name>` | `somax/<path>:<line>` | <gap or divergence> |

### B. Already In Somax But Missing Enhancements

#### B1. <Enhancement name> (HIGH PRIORITY)
- **Subject**: <approach>
- **Somax**: <current state>
- **What is needed**: <concrete change>
- **Impact**: <why it matters>

### C. Missing Completely From Somax

#### C1. <Feature name> (HIGH PRIORITY)
- **What it is**: <description>
- **Why useful**: <motivation>
- **Where in Somax**: proposed module `somax/<path>.py`

## 3. Summary Table

| Feature | Subject | Somax | Status |
|---|---|---|---|
| `<feature>` | ✓ | ✗ | Missing |

## 4. Recommended Integration Priority

### Phase 1
1. <item>

### Phase 2
2. <item>

## 5. Proposed Follow-up Issues

- [ ] `feat(<scope>): <title>` — covers <phase / section>
- [ ] `[Design] <question>` — resolves <open question>

## References
- <paper / repo / doc> — `<url>`

## Relationships
<!--
Parent:      make gh-sub PARENT=<parent#> CHILDREN="<this#>"
Blocked by:  make gh-block ISSUE=<this#> BLOCKED_BY=<other#>
Blocks:      follow-up issues can reference this research issue as blocker
Related:     prose only
-->
- Parent (theme epic, if any): #
- Blocked by: #
- Blocks: #
- Related: #