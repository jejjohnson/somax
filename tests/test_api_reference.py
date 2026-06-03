"""Keep the committed API-reference page in sync with the public API.

``content/api/reference.md`` is generated from docstrings by
``scripts/gen_api_reference.py`` and committed as a documentation artifact.
This test regenerates the page in memory and asserts the committed copy matches,
so a public-API change that isn't reflected in the docs fails CI with a clear
"run the generator" message. These are cheap (import + introspection, no
integration), so they stay in the fast PR lane.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
GEN_PATH = REPO_ROOT / "scripts" / "gen_api_reference.py"
PAGE_PATH = REPO_ROOT / "content" / "api" / "reference.md"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_api_reference", GEN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_page_matches_generator() -> None:
    """The committed reference.md equals a fresh generation.

    If this fails, run ``uv run python scripts/gen_api_reference.py`` and
    commit the result.
    """
    gen = _load_generator()
    expected = gen.build()
    actual = PAGE_PATH.read_text()
    assert actual == expected, (
        "content/api/reference.md is out of date with the public API. "
        "Regenerate it: `uv run python scripts/gen_api_reference.py`."
    )


def test_page_has_no_autodoc_directive() -> None:
    """The page must not reintroduce the mystmd-incompatible {autodoc}."""
    text = PAGE_PATH.read_text()
    assert "{autodoc}" not in text


def test_page_covers_core_and_models() -> None:
    """Sanity: the page documents the core, model, domain, and IO surfaces."""
    text = PAGE_PATH.read_text()
    for token in (
        "## Core",
        "## Models",
        "## Domain",
        "`Lorenz63`",
        "`BarotropicQG`",
        "`TermModel`",
        "`SimulationCheckpointer`",
        "`Domain`",
    ):
        assert token in text
