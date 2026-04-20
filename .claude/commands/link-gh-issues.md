Apply native GitHub issue relationships (sub-issues and blocked-by) via the GraphQL API.

Use this skill when the user asks to link issues as parent/child (sub-issue hierarchy) or to mark one issue as blocked by another. Works on top of the prose `## Relationships` block convention documented in `CONTRIBUTING.md` — the prose stays as a human-readable record, and this skill applies the matching native link so GitHub's UI and automation pick up the hierarchy.

## When to invoke

- "Link issue #42 as a sub-issue of #7"
- "Mark #44 as blocked by #43"
- "Wire up the sub-issues for the Wave 1 theme epic #29"
- "Apply the relationships from this epic's body" — parse checklist items and parent / blocked-by lines, apply each link
- "What's blocking #44?" — read relationships via GraphQL

## GitHub features used

| Feature | Mutations | Read fields |
|---|---|---|
| Sub-issues (parent ↔ child) | `addSubIssue`, `removeSubIssue`, `reprioritizeSubIssue` | `Issue.parent`, `Issue.subIssues`, `Issue.subIssuesSummary` |
| Blocked-by / blocking | `addBlockedBy`, `removeBlockedBy` | `Issue.blockedBy`, `Issue.blocking` |

Both available via the GitHub UI (Issue side-panel → Sub-issues / Development) and the `gh api graphql` CLI path.

## Instructions

### Step 1 — Resolve issue numbers to node IDs

Mutations need GraphQL node IDs, not issue numbers. Resolve with:

```bash
gh api repos/:owner/:repo/issues/<number> --jq .node_id
```

or via GraphQL for cross-repo cases:

```bash
gh api graphql -f query='
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) { id }
  }
}' -f owner=<owner> -f name=<name> -F number=<number> --jq .data.repository.issue.id
```

Cache IDs in bash variables when applying multiple links:

```bash
PARENT_ID=$(gh api repos/:owner/:repo/issues/7 --jq .node_id)
CHILD_ID=$(gh api repos/:owner/:repo/issues/42 --jq .node_id)
```

### Step 2a — Sub-issue link (parent ↔ child)

```bash
gh api graphql -f query='
mutation($parent: ID!, $child: ID!) {
  addSubIssue(input: {issueId: $parent, subIssueId: $child}) {
    subIssue { number title }
  }
}' -f parent="$PARENT_ID" -f child="$CHILD_ID"
```

Notes:
- `addSubIssue` also accepts `subIssueUrl` instead of `subIssueId` for cross-repo links.
- Pass `replaceParent: true` in the input if the child already has a different parent and should be moved.
- The mutation errors if the link already exists — treat that as no-op.

### Step 2b — Blocked-by link

`addBlockedBy` means "issueId is blocked by blockingIssueId":

```bash
gh api graphql -f query='
mutation($this: ID!, $blocker: ID!) {
  addBlockedBy(input: {issueId: $this, blockingIssueId: $blocker}) {
    issue { number }
  }
}' -f this="$BLOCKED_ID" -f blocker="$BLOCKER_ID"
```

If the user says "A blocks B", apply `addBlockedBy(issueId=B, blockingIssueId=A)` — this is the inverse form.

### Step 3 — Bulk-apply from an epic body

When applying the relationships from a theme epic or wave epic:

1. Fetch the epic's body: `gh issue view <N> --json body --jq .body`
2. Parse the `## Issues` (or `## Canonical Child Issues`) checklist: every `#NNN` reference becomes a sub-issue of the epic.
3. Parse the `## Relationships` block for `Blocked by:` and `Blocks:` lines — apply accordingly.
4. Skip `Related:` lines — they stay as prose only (no native feature).

Confirm each link applied with a brief summary to the user.

### Step 4 — Verify / read back

```bash
# Show parent + all sub-issues of an issue
gh api graphql -f query='
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      title
      parent { number title }
      subIssues(first: 50) { nodes { number title state } }
      subIssuesSummary { total completed percentCompleted }
      blockedBy(first: 50) { nodes { number title state } }
      blocking(first: 50) { nodes { number title state } }
    }
  }
}' -f owner=<owner> -f name=<name> -F number=<number>
```

### Step 5 — Use the helper script when applying many at once

For scripted bulk operations, `.github/scripts/link-issues.sh` wraps the same mutations with a thin CLI:

```bash
# Link children under a parent
bash .github/scripts/link-issues.sh sub <parent> <child1> <child2> ...

# Mark one issue blocked by another
bash .github/scripts/link-issues.sh block <issue> <blocker>

# Or via Makefile
make gh-sub PARENT=7 CHILDREN="42 43 44"
make gh-block ISSUE=44 BLOCKED_BY=43
make gh-show ISSUE=44
```

Prefer the script when applying more than two or three links — it handles node-ID resolution and error cases uniformly.

## Common errors

- **`addSubIssue` 422 "already a sub-issue"** — treat as no-op; report to user.
- **`addBlockedBy` 422 "already blocked by"** — same, no-op.
- **Node not found** — the issue number is wrong, or the repo doesn't match. Double-check `gh repo view` matches the expected repo.
- **Missing scope** — the `gh` token needs `repo` scope. `gh auth refresh` if needed.

## What to keep as prose vs native

The prose `## Relationships` block in the issue body is the human-readable record:

```markdown
## Relationships
- Parent: #7
- Blocked by: #42
- Blocks: #55
- Related: #38
```

Leave those lines in place — they're readable in diffs, search, and tools that don't resolve GraphQL. The native link is for GitHub's UI + automation. Both coexist.

`Related:` has no native equivalent; mention only.
