---
applyTo: "content/**/*.md,notebooks/**/*.py,notebooks/**/*.ipynb"
---

# Documentation Examples — Standards & Workflow

## Overview

Somax documentation is authored in `content/` and built with **MyST-NB** (MyST Markdown for Sphinx / Jupyter Book). Exploratory or reproducible example notebooks live in `notebooks/` and may be stored as `.ipynb` files or jupytext percent-format `.py` files.

When a notebook generates figures or tables that are referenced from docs pages, the notebook source is the authoring entrypoint and the generated assets should be committed under `content/images/{notebook_name}/`.

## Directory Layout

```
content/
├── api/           # API reference
├── notes/         # Design notes, ADRs
├── tutorials/     # Executed example pages (MyST)
├── images/        # Generated figures / tables
│   └── <notebook_name>/
└── index.md

notebooks/
├── dev/           # scratch / WIP
├── demo_*.ipynb   # standalone demos
└── tutorial_*.py  # jupytext percent-format source for content/tutorials
```

## Authoring Workflow

**Develop in jupytext percent format**, then either execute in place (for notebooks that live in `notebooks/`) or run and collect assets (for pages that live in `content/tutorials/`). Plain `.py` diffs are vastly easier to review than raw `.ipynb` JSON, so do the substantive editing on the `.py` side.

1. Create `notebooks/foo.py` (or `notebooks/dev/foo.py` for WIP) in jupytext percent format (header below).
2. Iterate — edit, smoke-run via `uv run python notebooks/foo.py`, repeat.
3. When the notebook is stable, convert to `.ipynb` for MyST consumption:

   ```bash
   uv run jupytext --to notebook notebooks/foo.py
   ```

4. Execute in place so cell outputs are embedded:

   ```bash
   uv run jupyter nbconvert --to notebook \
     --execute notebooks/foo.ipynb \
     --inplace \
     --ExecutePreprocessor.timeout=180
   ```

5. If the notebook produces figures that appear in `content/` pages, save the images under `content/images/<notebook_name>/` and reference them from the MyST page.
6. Commit the executed `.ipynb` (and the source `.py` if you keep both).

## Jupytext Header (dev-only)

While developing in `.py`, start the file with:

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
```

## Cell Markers (dev-only)

- **Code cells**: `# %%`
- **Markdown cells**: `# %% [markdown]` followed by `#`-prefixed lines

```python
# %% [markdown]
# # Title
#
# Some explanation with LaTeX: $\nabla^2 \psi = f$

# %%
import numpy as np
```

## Markdown Paragraph Wrapping

**Each paragraph in a `# %% [markdown]` block must be a single long line.** Do not soft-wrap paragraph text across multiple `#` lines. jupytext preserves source newlines as soft breaks, which MyST-NB renders as awkward visual breaks.

Right:

```python
# %% [markdown]
# This notebook demonstrates Somax's shallow-water solver. We walk through the staggered-grid setup, boundary conditions, and time integration using a simple canonical testcase so the only thing that differs between runs is the configuration.
```

Wrong:

```python
# %% [markdown]
# This notebook demonstrates Somax's shallow-water solver. We walk
# through the staggered-grid setup, boundary conditions, and time
# integration using a simple canonical testcase.
```

Lines that *must* stay on their own (do not join):

- Headings: `# # Title`, `# ## Section`
- Display math: `# $$...$$` block (one expression per line)
- Table rows: `# | col | col |`
- List item heads: `# - item` or `# 1. item`
- Code-fence delimiters: `# ``` ` and contents inside the fence
- Blockquotes: `# > quote`

## Notebook Structure

1. **Title + overview** (markdown) — motivation, math, what the user will learn
2. **Imports + path setup** — including `content/images/` asset dir if the notebook backs a docs page
3. **Problem setup** — grids, initial conditions, physical constants
4. **Core computation** — the Somax model run or primitive demo
5. **Figures / tables** — saved under `content/images/<notebook_name>/` when backing a docs page
6. **Summary / takeaways**

## Asset Paths

Use `pathlib.Path` and save generated assets under `content/images/{notebook_name}/`.

Example pattern for a notebook in `notebooks/`:

```python
from pathlib import Path

IMG_DIR = Path(__file__).resolve().parent.parent / "content" / "images" / "notebook_name"
IMG_DIR.mkdir(parents=True, exist_ok=True)
```

## Matplotlib Style

**Defaults only** — no `plt.style.use` and no `rcParams` tweaks:

- `C0`, `C1`, `C2` (matplotlib defaults) for main series.
- `"k--"` for truth / reference lines.
- `figsize=(12, 5)` for single plots, `(18, 5)` for 1×3 comparison grids.
- `ax.scatter(...)` for data points.
- `ax.fill_between(..., alpha=0.2)` for uncertainty bands.
- Save figures **before** `plt.show()` when the figure is referenced from a `content/` page.
- Use descriptive lowercase filenames with underscores (e.g. `velocity_field_t0.png`).

## Math in Markdown Cells

Inline: `$\|x - x'\|^2$`.

Display:

```markdown
$$
\partial_t h + \nabla \cdot (h \, \boldsymbol{u}) = 0
$$
```

MyST-NB / MathJax is configured in `myst.yml` (or equivalent) — both inline and display math render in the docs.

## Reproducibility — `watermark`

After imports, print a version readout so readers (and your future self) know which package versions generated the committed outputs. The cell uses `get_ipython()` and an `importlib.util.find_spec` check so a plain `python foo.py` smoke run during dev, and an `nbconvert --execute` on a machine without `watermark` installed, both no-op cleanly instead of raising `UsageError: Line magic function %load_ext not found`:

```python
import importlib.util

try:
    from IPython import get_ipython

    ipython = get_ipython()
except ImportError:
    ipython = None

if ipython is not None and importlib.util.find_spec("watermark") is not None:
    ipython.run_line_magic("load_ext", "watermark")
    ipython.run_line_magic(
        "watermark",
        "-v -m -p numpy,jax,matplotlib,somax",
    )
else:
    print("watermark extension not installed; skipping reproducibility readout.")
```

## Imports + Warnings

```python
import warnings

warnings.filterwarnings("ignore", message=r".*IProgress.*")

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
# ... other imports
```

- Suppress the `IProgress` warning from ipywidgets so the output is clean.

## MyST Pages

Pages in `content/` should reference committed assets with paths relative to the page location. Keep the source notebook path and reproduction notes nearby so readers know where the figure came from.

## Timing And Benchmarks

For JAX benchmarks:

- Warm up compiled functions before timing (the first call pays the trace/compile cost).
- Use `.block_until_ready()` on the output to account for async dispatch.
- Save the final summary figure or table that the docs page references under `content/images/`.

## Checklist for New Notebooks

- [ ] Authored in jupytext `.py` percent format during development
- [ ] First markdown cell: `#`-level title + one-paragraph overview
- [ ] Markdown paragraphs are single long lines (no soft-wrapping)
- [ ] `warnings.filterwarnings(..., IProgress, ...)`
- [ ] `%watermark` version readout
- [ ] Matplotlib defaults only (no `style.use`, no `rcParams`)
- [ ] Converted to `.ipynb` and executed in place
- [ ] Figures saved under `content/images/<notebook_name>/` when backing a docs page
- [ ] MyST page added under `content/tutorials/` or `content/notes/` if user-facing
