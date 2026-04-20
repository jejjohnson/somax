Open GitHub issues using Somax's issue templates, with the correct labels, milestone, and relationships applied.

Use this skill when the user asks to file an issue, open an issue, create a feature / design / bug / research / epic, or when publishing draft issues from a wave-backlog file into real GitHub issues. It pairs with `link-gh-issues` for applying native sub-issue and blocked-by links after creation.

## When to invoke

- "Open an issue for X" / "file an issue about Y"
- "Create a feature issue for the new multilayer SWM wrapper"
- "Open an epic for Wave 2"
- "Publish the drafts in `.plans/wave-1-backlog.md` as real issues"
- "Convert this research note into a research issue"

## Templates available

Issue templates live in `.github/ISSUE_TEMPLATE/`. Pick the one that matches the work:

| Template | Use when |
|---|---|
| `epic-wave.md` | Opening a release-scoped wave (owns a milestone; groups several Theme epics) |
| `epic-theme.md` | Grouping related feature / design / chore issues within a wave |
| `feature.md` | One substantial deliverable (new API, module, notebook, etc.) |
| `design.md` | Resolving an open design question — ADR-style |
| `bug.md` | Something isn't working |
| `research.md` | Comparative analysis / prior-art survey → prioritized roadmap of follow-up issues |

If unsure, ask the user before proceeding. Default for ambiguous "add X" requests is `feature.md`.

## Decision tree for picking a template

```
Does the user want to resolve an open API / architectural question?      → design.md
Is the user describing broken behaviour + reproduction?                   → bug.md
Is the user asking for a survey of external repos / papers / prior art?  → research.md
Is the user planning a release or multi-issue phase?                      → epic-wave.md
Is the user grouping related issues under a wave?                         → epic-theme.md
Otherwise (substantial single deliverable)                                → feature.md
```

## Step-by-step workflow

### Step 1 — Confirm scope with the user

Before opening any issue, confirm these five inputs with the user (or infer from context and echo back):

1. **Template** — which of the six (see table above)
2. **Title** — follow the template's title convention:
   - Feature / chore: `<scope>: <short description>` — e.g. `feat(models): multilayer QG reparameterization`
   - Design: `[Design] <question>` — e.g. `[Design] stratification profile: layer vs continuous`
   - Research: `research: <topic>` — e.g. `research: semi-Lagrangian advection vs this project`
   - Bug: `bug: <short description>`
   - Epic wave: `[EPIC] Wave N: <title>`
   - Epic theme: `[EPIC] <theme title>`
3. **Parent theme/wave epic** (if any) — the issue number this child rolls up to
4. **Wave / milestone** — which `wave:*` label + which `vX.Y-<slug>` milestone
5. **Additional labels** — which `area:*`, `layer:*`, `priority:*` apply (see Step 4)

### Step 2 — Draft the body

Read the template file at `.github/ISSUE_TEMPLATE/<template>.md` and use it as the scaffold. Strip:

- The YAML frontmatter (`---` block at the top)
- HTML comments that are pure guidance (e.g. `<!-- Delete if not needed. -->`) — keep comments that describe what the section contains if the filled-in body would benefit from the note

Fill in sections according to the template's intent. **Minimum required sections** (never leave empty):

| Template | Required |
|---|---|
| `feature.md` | Problem / Request, User Story, Motivation, Proposed API, Implementation Steps, Definition of Done, Testing, Relationships |
| `design.md` | Problem / Question, Proposed Options, Alternatives Considered, Relationships |
| `bug.md` | Problem, Reproduction, Expected Behavior, Actual Behavior, Environment, Relationships |
| `research.md` | Title, Context, §1 What the subject contains, §2 Comparison, §3 Summary Table, §5 Proposed Follow-up Issues, Relationships |
| `epic-wave.md` | Goal, Motivation, Theme Epics, Definition of Done (Wave), Relationships |
| `epic-theme.md` | Theme, Parent Wave, Issues, Definition of Done, Relationships |

**Optional / renameable sections** on `feature.md` and `design.md`:

- `## Design Snapshot` — rename to fit the content (e.g. `## Demo To Implement`, `## Config Snippet`, `## Reference Trace`, `## Prior Art Snippets`). Lead with the exact code / config / equation the implementer will reproduce. Delete the section if not relevant.
- `## Mathematical Notes` — rename if warranted (`## Numerical Notes`, `## Stability Notes`, `## Equations To Test`). Delete if not relevant. For algorithmic / numerical issues Somax treats this section as required (see `CONTRIBUTING.md`).

### Step 3 — Apply the style conventions

- **Unicode math in prose** — prefer `σ²`, `E₁`, `∑`, `⊗`, `≈`, `Λ⁻¹`, `∂/∂x`, `O(d³)` over LaTeX / MathJax blocks. Keeps the issue readable in the GH UI and plain-text tools.
- **`text`-tagged code fences for multi-line equations** so pseudo-math isn't syntax-highlighted as Python. Wrap the example in a 4-backtick outer fence so the nested triple-backtick block renders intact:

  ````
  ```text
  ∂h/∂t + ∇·(h·u) = 0
  ∂u/∂t + u·∇u + f·k×u = -g·∇h
  ```
  ````

- **Code-first, prose-last** — Design Snapshot and Proposed API should LEAD with the exact snippet the implementer will reproduce; prose goes after.
- **Concrete implementation steps** — each step should name the file path + function, not a generic `...`. Example: `Add \`solve_multilayer\` in \`somax/_src/models/swm/multilayer.py\``

### Step 4 — Pick labels

Every issue carries exactly one `type:*`, one or more `area:*`, at most one `layer:*`, one `wave:*`, and one `priority:*`. Common combinations:

| Work type | Labels |
|---|---|
| Core numerics / primitives | `type:feature`, `area:core`, `layer:0-core`, `wave:N-…`, `priority:p1` |
| Model implementation | `type:feature`, `area:models`, `layer:1-models`, `wave:N-…` |
| CLI / runner orchestration | `type:feature`, `area:cli`, `layer:2-runner`, `wave:N-…` |
| IO / persistence | `type:feature`, `area:io`, `wave:N-…` |
| Test coverage | `type:feature`, `area:testing`, `wave:N-…` |
| Notebook / docs page | `type:docs`, `area:docs`, `wave:N-…` |
| Design / ADR | `type:design`, `area:core` (or `area:engineering`), `wave:N-…` |
| Research / survey | `type:research`, `area:core` (or relevant area) |
| Engineering / CI / packaging | `type:chore`, `area:engineering`, `wave:0-bootstrap` |

Wave labels and milestones are project-specific — check `CONTRIBUTING.md` for the current taxonomy before picking.

### Step 5 — Create the issue

**CLI path (preferred for drafted bodies):**

Write the drafted body to a temporary file, then:

```bash
gh issue create \
  --title "feat(models): multilayer QG reparameterization" \
  --body-file /tmp/issue-body.md \
  --label "type:feature,area:models,layer:1-models,wave:1-qg,priority:p1" \
  --milestone "v0.1-qg"
```

Returns the new issue's URL; extract the number (e.g. `#42`) for the next step.

**UI path (when the user wants to review in the browser):**

```bash
gh issue create --web
```

Opens `/issues/new/choose` so the user picks the template in the UI, fills in the body, and clicks Submit. Note: `gh issue create --web` does NOT respect `--template` — that flag only applies to the non-web flow (where it's used as starting body text). There is no CLI path to pre-select a template for the web flow; the user has to click the template card in the browser.

**Notes:**
- The `--body-file` path must point to a file containing the post-template body (frontmatter + guidance comments stripped).
- If the user has draft body content in a `.plans/*.md` file, extract the single issue's body section and write it to a temp file before passing to `--body-file`.

### Step 6 — Apply relationships

After the issue is created, apply the native GitHub links using the `link-gh-issues` skill (or its underlying helpers):

```bash
# Link the new issue as a sub-issue of its theme epic
make gh-sub PARENT=<theme-epic#> CHILDREN="<new-issue#>"

# If the issue is blocked by another
make gh-block ISSUE=<new-issue#> BLOCKED_BY=<blocker#>
```

Keep the prose `## Relationships` lines in the body — they coexist with the native links.

### Step 7 — Verify

Confirm the filed issue by URL + a one-line summary. If opening multiple issues from a backlog, echo a mapping:

```
Draft ID  → GH issue
SMX-05    → #42   feat(core): ...
SMX-06    → #43   feat(models): ...
```

## Bulk workflow — publishing a wave-backlog

When the user has a `.plans/<wave>-backlog.md` file drafted from `.github/templates/wave-backlog.md`:

1. **Parse** the backlog into discrete issue sections. Each draft starts with `# <title>` followed by `Draft ID: \`<PREFIX>-NN\``. Sections are separated by `---`.
2. **Identify** which template each draft corresponds to (epic-wave → body starts with `## Goal`; epic-theme → `## Theme`; feature → `## Problem / Request`; design → `## Problem / Question`; research → `## Context` with `## 1.` numbered sections).
3. **Open the wave epic first** (so its issue number exists before theme epics reference it).
4. **Open theme epics next** (so their numbers exist before feature / design issues reference them).
5. **Open leaf issues** (feature / design / bug / research).
6. **Record** the mapping `<PREFIX>-NN → #issue-number` as you go.
7. **Replace draft-ID cross-references** in each issue's body with real GH numbers BEFORE creating (so `Parent: SMX-03` becomes `Parent: #41`, etc.). This requires two passes: parse all draft IDs first, create issues, then substitute. OR: create issues with draft IDs as placeholders, then `gh issue edit` each one to rewrite references after all are opened. Either works — the two-pass version keeps each created issue's body already clean.
8. **Apply native links** using `link-gh-issues` for the whole wave once all issues exist.
9. **Update the backlog file** — replace draft IDs with GH issue numbers throughout, or archive the file to `.plans/archive/`.

## Common pitfalls

- **Missing label** — `gh issue create --label` fails silently if the label doesn't exist. Run `make gh-labels` first on a fresh repo.
- **Missing milestone** — same. `gh api repos/:owner/:repo/milestones --method POST -f title="vX.Y-<slug>"` to create.
- **Template frontmatter in the body** — `gh issue create --body-file` renders the frontmatter block as literal text. Strip the `---` block before writing the temp file.
- **Shell-escaped backticks** — do NOT escape backticks when passing a body via `--body`. Use `--body-file` with a temp file for any body containing code fences or backticks. Heredocs with single-quoted delimiters (`<<'EOF'`) also preserve backticks correctly.
- **Wrong milestone number** — milestones are referenced by title (`--milestone "v0.1-qg"`), not number. `gh milestone list` (via `gh api`) to confirm titles exist.

## Related skills

- [`link-gh-issues`](./link-gh-issues.md) — Apply native sub-issue and blocked-by links after issues are created. Invoke after this skill finishes creating issues.
- [`squash-commit`](./squash-commit.md) — Generate a squash commit message when merging an issue's PR.
