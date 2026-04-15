Apply native GitHub issue relationships for Somax issues.

Use this command when the user asks to:

- link an issue as a sub-issue of an epic
- mark one issue as blocked by another
- inspect parent, sub-issue, or blocking relationships

Helpers available in this repo:

```bash
make gh-sub PARENT=<parent#> CHILDREN="<child1#> <child2#>"
make gh-block ISSUE=<issue#> BLOCKED_BY=<other#>
make gh-show ISSUE=<issue#>
```

Underlying script:

```bash
bash .github/scripts/link-issues.sh <subcommand> ...
```

Keep the prose `Relationships` section in the issue body for readability, and apply the native GitHub relationship as well so the UI and dependency graph stay accurate.