---
applyTo: "content/**/*.md,notebooks/**/*.py"
---

# Documentation Examples — Standards & Workflow

## Overview

Somax documentation is authored in `content/` and built with MyST. Exploratory or reproducible example notebooks live in `notebooks/` and may be stored as `.ipynb` files or jupytext percent-format `.py` files.

When a notebook generates figures or tables that are referenced from docs pages, the notebook source is the authoring entrypoint and the generated assets should be committed under `content/images/{notebook_name}/`.

## Recommended Structure For Notebook-Backed Docs

1. Title and overview in markdown
2. Imports and path setup
3. Problem setup and parameters
4. Core computation
5. Figures or tables
6. Saved assets referenced from `content/` pages
7. Short summary or takeaways

## Jupytext Header

If you use a `.py` notebook, start it with the standard percent-format jupytext header.

## Asset Paths

Use `pathlib.Path` and save generated assets under `content/images/{notebook_name}/`.

Example pattern for a notebook in `notebooks/`:

```python
from pathlib import Path

IMG_DIR = Path(__file__).resolve().parent.parent / "content" / "images" / "notebook_name"
IMG_DIR.mkdir(parents=True, exist_ok=True)
```

## Figures

- Save figures before `plt.show()` when using matplotlib
- Use descriptive lowercase filenames with underscores
- Commit the generated assets if they are referenced from docs pages

## MyST Pages

Pages in `content/` should reference committed assets with paths relative to the page location. Keep the source notebook path and reproduction notes nearby so readers know where the figure came from.

## Timing And Benchmarks

For JAX benchmarks:

- Warm up compiled functions before timing
- Use `block_until_ready()` to account for async dispatch
- Save the final summary figure or table that the docs page references