---
applyTo: "somax/**/*.py,tests/**/*.py,scripts/**/*.py,configs/**/*.py"
---

# Python Coding Standards

## Modern Python (3.12+)

- `from __future__ import annotations` at the top of every module
- Type hints on all public functions, methods, and module-level variables
- Modern union syntax: `X | None` and `X | Y`
- Built-in generics: `list[int]`, `dict[str, str]`
- `pathlib.Path` over `os.path`
- f-strings for string formatting
- Specific exception types and explicit error messages
- Early returns and guard clauses to reduce nesting

## Somax Preferences

| Purpose | Preferred Package |
|---------|-------------------|
| Logging | `loguru` |
| CLI | `cyclopts` |
| Config | `hydra-core`, `hydra-zen`, `omegaconf` |
| Paths | `pathlib` |
| Arrays / numerics | `jax`, `jax.numpy`, `xarray`, `zarr` |
| Testing | `pytest` |

## Scientific Code Guidance

- Keep JAX computation pure and side effects isolated to CLI or IO layers
- Preserve existing numerical conventions and naming where the repo already has them
- Prefer explicit shapes, units, and boundary-condition assumptions when they affect correctness
- Comments should explain why a numerical step exists, not restate the code

## Documentation

- Module-level docstrings explaining purpose
- Public functions and classes should have docstrings
- Include short examples for public APIs when that improves discoverability