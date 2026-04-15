Open GitHub issues using Somax's issue templates, with the correct labels, milestone, and relationships applied.

Use this command when the user asks to file an issue, open a feature or design issue, publish a wave backlog, or convert notes into real GitHub issues.

Templates live in `.github/ISSUE_TEMPLATE/`:

- `epic-wave.md`
- `epic-theme.md`
- `feature.md`
- `design.md`
- `bug.md`
- `research.md`

Somax conventions:

- See `CONTRIBUTING.md` for the label taxonomy and epic model
- Use one `type:*` label, one or more `area:*` labels, at most one `layer:*` label, one `wave:*` label, and one `priority:*` label
- Prefer `--body-file` when creating issues with `gh issue create`
- After the issue is created, apply native sub-issue or blocked-by links with `.github/scripts/link-issues.sh` or the `make gh-sub` and `make gh-block` helpers

When drafting issue bodies, prefer explicit paths like `somax/_src/models/...`, `configs/...`, `content/...`, or `tests/...` so the issue can be implemented without extra context.