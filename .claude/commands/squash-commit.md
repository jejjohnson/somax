Generate a squash commit message for a GitHub PR.

Instructions:

1. If a PR number or URL is provided, fetch that PR. Otherwise detect the PR for the current branch.
2. Collect the individual commit messages.
3. Produce a single Conventional Commit message that summarizes the overall change.

Format:

```text
<type>(<scope>): <concise summary of the overall change>

<1-3 sentence description combining the key changes from all commits. Focus on the why and overall effect, not the incremental history.>

Co-authored-by: <preserve all unique Co-authored-by lines>
```

Rules:

- Keep the summary line under 72 characters
- Use the dominant change type (`feat`, `fix`, `docs`, `chore`, etc.)
- Preserve all unique `Co-authored-by` lines
- Output only the final squash message in a code block