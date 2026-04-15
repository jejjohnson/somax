# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Somax is a JAX-based ocean-modeling library and simulation runner. It combines reusable model components, a Cyclopts CLI, authored configs, DVC pipelines, and MyST documentation.

## Common Commands

```bash
make install              # Install all deps (uv sync --all-groups) + pre-commit hooks
make test                 # Run tests: uv run pytest -v -o addopts=
make test-cov             # Tests with coverage
make format               # Auto-fix: ruff format . && ruff check --fix .
make lint                 # Lint code: ruff check .
make typecheck            # Type check: ty check somax
make precommit            # Run pre-commit on all files
make docs-serve           # Local MyST docs server
```

### Running a single test

```bash
uv run pytest tests/test_cli_run.py::test_run_command -v
```

### Alternative pixi tasks

```bash
pixi run test
pixi run lint
pixi run typecheck
pixi run docs-serve
```

### Pre-commit checklist (all four must pass)

```bash
uv run pytest -v
uv run --group lint ruff check .
uv run --group lint ruff format --check .
uv run --group typecheck ty check somax
```

**Critical**: Always lint the entire repo with `.` from the root. Somax includes tests, configs, scripts, and docs glue outside the package directory.

## Architecture

### Package structure

The public package lives in [somax](/home/azureuser/localfiles/somax/somax). Internal implementation details live in [somax/_src](/home/azureuser/localfiles/somax/somax/_src).

### Key directories

| Path | Purpose |
|------|---------|
| `somax/` | Installable library and public exports |
| `somax/_src/core/` | Core numerics, utilities, and shared primitives |
| `somax/_src/domain/` | Domain-specific types and helpers |
| `somax/_src/models/` | Ocean and dynamical-system model implementations |
| `somax/_src/io/` | IO helpers and persistence layer |
| `somax/_src/cli/` | `somax-sim` CLI entrypoints and orchestration |
| `configs/` | Authored and generated simulation configs |
| `scripts/` | Repo automation and config generation helpers |
| `content/` | MyST documentation source |
| `notebooks/` | Ad hoc notebooks and exploratory examples |
| `tests/` | Test suite |

## Documentation Examples

Docs pages live in [content](/home/azureuser/localfiles/somax/content). Notebooks may live in [notebooks](/home/azureuser/localfiles/somax/notebooks) as `.ipynb` files or jupytext percent-format `.py` files. When notebooks produce figures for docs pages:

1. Run them locally
2. Save figures under `content/images/{notebook_name}/`
3. Reference those assets from the relevant MyST page in `content/`
4. Commit the notebook source and the generated assets together

See [.github/instructions/docs-examples.instructions.md](/home/azureuser/localfiles/somax/.github/instructions/docs-examples.instructions.md) for the workflow expectations.

## Coding Conventions

- `from __future__ import annotations` at the top of Python modules
- Type hints on public functions and methods
- Use `pathlib.Path` for filesystem work
- Keep JAX computations pure; isolate IO and CLI side effects
- Match existing numerical style and avoid refactoring unrelated code

## Plans

Plans and scratch implementation docs go in `.plans/` and should not be committed.

## PR Review Comments

When addressing PR review comments, resolve each review thread after fixing it via the GitHub GraphQL API. Use the workflow documented in [AGENTS.md](/home/azureuser/localfiles/somax/AGENTS.md).

## Code Review

Follow the guidance in [/CODE_REVIEW.md](/home/azureuser/localfiles/somax/CODE_REVIEW.md) for all code review tasks.