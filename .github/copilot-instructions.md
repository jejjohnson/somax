# Copilot Instructions

## Project Overview

- **Python**: 3.12+
- **Package Managers**: `uv` for Python workflows, `pixi` for full environment workflows
- **CLI Framework**: cyclopts
- **Layout**: flat package layout (`somax/`)
- **Testing**: pytest
- **Docs**: MyST (`content/`) with notebooks and static assets
- **Workflow Tools**: DVC for simulation pipelines

## Build & Test Commands

```bash
make install     # Install all dependencies (uv sync --all-groups)
make test        # Run tests without coverage gate flags
make test-cov    # Run tests with coverage
make lint        # Lint code (ruff check .)
make format      # Format code (ruff format + ruff check --fix .)
make typecheck   # Type check (ty check somax)
make precommit   # Run pre-commit on all files
make docs-serve  # Serve MyST docs locally
```

## Before Every Commit — Mandatory Checklist

All four checks must pass before any commit. CI runs them from the repo root, not just the package directory.

```bash
uv run pytest -v
uv run --group lint ruff check .
uv run --group lint ruff format --check .
uv run --group typecheck ty check somax
```

> Common pitfall: running `ruff check somax/` instead of `ruff check .` misses issues in `tests/`, `configs/`, `scripts/`, and `.github/` glue.

## Key Directories

| Path | Purpose |
|------|---------|
| `somax/` | Main package source code |
| `somax/_src/` | Internal implementation details |
| `configs/` | Authored and generated simulation configs |
| `content/` | MyST docs source |
| `notebooks/` | Exploratory notebooks |
| `scripts/` | Repo automation and helpers |
| `tests/` | Test suite |

## Behavioral Guidelines

### Do Not Nitpick
- Ignore style issues that `ruff` or formatters already enforce
- Match existing numerical and scientific patterns unless there is a clear bug
- Do not refactor adjacent code unless the task requires it

### Always Propose Tests
When implementing features or fixing bugs:
1. Write or update a test that verifies the expected behavior
2. Implement the change
3. Verify the relevant tests pass

### Never Suggest Without A Proposal
Bad: "You should validate the config here"

Good:
```python
if config.output_dir is None:
    raise ValueError("output_dir must be provided for this run mode")
```

### Simplicity First
- Prefer direct code over speculative abstractions
- Keep JAX code pure and side effects isolated to CLI or IO layers
- If the repo already has a pattern for configs, model factories, or DVC wiring, follow it

### Surgical Changes
- Only modify files and lines directly related to the request
- Remove imports or helpers only if your change made them unused
- Do not add comments or docstrings to unrelated code

## Plans

Plans and design documents belong in `.plans/` and should never be committed.

## PR Review Comments

When addressing PR review comments, resolve each addressed review thread after the fix is pushed. Follow the GraphQL workflow documented in `AGENTS.md`.

## Code Review

For all code review tasks, follow the guidance in `/CODE_REVIEW.md`.