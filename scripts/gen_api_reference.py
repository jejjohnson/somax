"""Generate the strict-MyST API reference from somax docstrings.

mystmd has no built-in Python autodoc directive (the old
``content/api/reference.md`` used an ``{autodoc}`` directive that mystmd does
not understand, so every entry errored at build time). This script introspects
somax's public modules and emits a hand-rollable MyST page —
``content/api/reference.md`` — with one section per public module, each symbol
rendered as a heading, a fenced signature, its summary line, and a collapsible
full docstring.

Run it from the repo root::

    uv run python scripts/gen_api_reference.py

It writes ``content/api/reference.md`` in place. The output is committed (it is
a documentation artifact, not a build-time step), so regenerate and commit when
the public API changes. A CI smoke test (``tests/test_api_reference.py``)
asserts the committed page is in sync with the current public API.

The optional ``somax.da`` surface (which needs the ``da`` dependency group) is
documented from its ``__all__`` without importing it, so the page builds
whether or not ``filterax`` / ``vardax`` are installed.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "content" / "api" / "reference.md"

# (module, human title, one-line blurb). Order defines the page order.
SECTIONS: list[tuple[str, str, str]] = [
    (
        "somax.core",
        "Core",
        "The base types, model contract, term algebra, forcing, stratification, "
        "elliptic caches, and checkpointing that every component builds on "
        "(re-exported at the top level as ``somax`` and ``somax.core``).",
    ),
    (
        "somax.models",
        "Models",
        "Dynamical-system and ocean model classes, with their state, "
        "parameter, and diagnostic companions.",
    ),
    (
        "somax.domain",
        "Domain",
        "Spatial and temporal domain descriptors.",
    ),
    (
        "somax.operators",
        "pipekit Operators",
        "Bridge that exposes somax models as ``pipekit.Operator`` stages.",
    ),
    (
        "somax.eval",
        "Evaluation Metrics",
        "Reference-free field diagnostics computed on a model's own grid.",
    ),
    (
        "somax.guards",
        "In-JIT Guards",
        "Fail-fast tripwires that halt a run at the offending step.",
    ),
    (
        "somax.monitor",
        "Monitors",
        "Chunk-boundary observability for the ``somax-sim`` runner.",
    ),
    (
        "somax.solvers",
        "Solvers",
        "Matrix-free IMEX integration helpers for stiff term models.",
    ),
    (
        "somax.io",
        "IO & Persistence",
        "xarray / zarr helpers that round-trip model states and snapshots "
        "(requires the ``sim`` dependency group).",
    ),
    (
        "somax.da",
        "Data Assimilation",
        "Adapters wiring somax models into the filterax / vardax DA stack "
        "(requires the optional ``da`` dependency group).",
    ),
]

# Modules that may be absent in a minimal install (optional dependency
# groups). They are documented from their ``__all__`` parsed statically from
# source, so the page generates whether or not the deps are installed.
_OPTIONAL_MODULES = {
    "somax.io": ("somax/io.py", "sim"),
    "somax.da": ("somax/da.py", "da"),
}

# Suffixes that mark a model's data companions — listed compactly rather than
# expanded in full, to keep the (large) models section readable.
_COMPANION_SUFFIXES = ("State", "Params", "PhysConsts", "Diagnostics")


def _summary(obj: Any) -> str:
    """First non-empty line of an object's docstring."""
    doc = inspect.getdoc(obj) or ""
    for line in doc.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _signature(obj: Any) -> str:
    """Best-effort call signature string, or empty for non-callables."""
    if not callable(obj):
        return ""
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"


def _public_names(module: Any) -> list[str]:
    names = getattr(module, "__all__", None)
    if names is None:
        names = sorted(n for n in dir(module) if not n.startswith("_"))
    # Skip re-exported submodules (e.g. ``somax.core``): they are documented in
    # their own sections, not as bare objects here.
    return [n for n in names if not inspect.ismodule(getattr(module, n, None))]


def _all_from_source(rel_path: str) -> list[str]:
    """Read a module's ``__all__`` from source without importing it.

    Used for optional-dependency modules (``somax.io`` needs the ``sim``
    group, ``somax.da`` needs the ``da`` group) so the page generates
    whether or not those deps are installed.
    """
    src = (REPO_ROOT / rel_path).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        return [
                            el.value
                            for el in node.value.elts
                            if isinstance(el, ast.Constant)
                        ]
    return []


def _render_symbol(name: str, obj: Any) -> list[str]:
    """MyST block for a single public symbol."""
    lines: list[str] = []
    sig = _signature(obj)
    kind = (
        "class"
        if inspect.isclass(obj)
        else "function"
        if inspect.isfunction(obj)
        else "object"
    )
    lines.append(f"### `{name}`")
    lines.append("")
    lines.append(f"*{kind}*")
    lines.append("")
    if sig:
        lines.append("```python")
        lines.append(f"{name}{sig}")
        lines.append("```")
        lines.append("")
    summary = _summary(obj)
    if summary:
        lines.append(summary)
        lines.append("")
    doc = inspect.getdoc(obj) or ""
    body = doc[len(summary):].strip() if summary and doc.startswith(summary) else ""
    if body:
        lines.append("````{admonition} Details")
        lines.append(":class: dropdown")
        lines.append("")
        lines.append("```text")
        lines.append(body)
        lines.append("```")
        lines.append("````")
        lines.append("")
    return lines


def _render_optional_listing(module_name: str) -> list[str]:
    """Compact bullet listing for an optional module, parsed from source.

    Renders full per-symbol entries when the optional dependency is installed
    (so the module imports), otherwise a name-only listing parsed statically.
    """
    rel_path, group = _OPTIONAL_MODULES[module_name]
    lines: list[str] = []
    try:
        module = importlib.import_module(module_name)
    except Exception:
        module = None
    if module is not None:
        for name in sorted(_public_names(module)):
            lines += _render_symbol(name, getattr(module, name))
        return lines
    lines.append(
        f"These symbols live in `{module_name}` and require the optional "
        f"`{group}` dependency group (`uv sync --group {group}`):"
    )
    lines.append("")
    for name in sorted(_all_from_source(rel_path)):
        lines.append(f"- `{module_name}.{name}`")
    lines.append("")
    return lines


def _render_module(module_name: str, title: str, blurb: str) -> list[str]:
    lines = [f"## {title}", "", blurb, ""]

    if module_name in _OPTIONAL_MODULES:
        return lines + _render_optional_listing(module_name)

    module = importlib.import_module(module_name)
    names = _public_names(module)

    if module_name == "somax.models":
        models = [
            n
            for n in names
            if inspect.isclass(getattr(module, n, None))
            and not n.endswith(_COMPANION_SUFFIXES)
        ]
        companions = sorted(n for n in names if n.endswith(_COMPANION_SUFFIXES))
        functions = [n for n in names if inspect.isfunction(getattr(module, n, None))]
        for name in sorted(models):
            lines += _render_symbol(name, getattr(module, name))
        for name in functions:
            lines += _render_symbol(name, getattr(module, name))
        if companions:
            lines.append("### State / parameter / diagnostic companions")
            lines.append("")
            lines.append(
                "Each model carries dataclass companions for its state, "
                "differentiable parameters, frozen physical constants, and "
                "on-demand diagnostics:"
            )
            lines.append("")
            for name in companions:
                summary = _summary(getattr(module, name))
                lines.append(f"- `{name}` — {summary}" if summary else f"- `{name}`")
            lines.append("")
        return lines

    for name in sorted(names):
        lines += _render_symbol(name, getattr(module, name))
    return lines


def build() -> str:
    """Render the full API-reference page."""
    header = textwrap.dedent(
        """\
        # API Reference

        ```{note}
        This page is generated from the public-API docstrings by
        `scripts/gen_api_reference.py`. Regenerate it with
        `uv run python scripts/gen_api_reference.py` when the public API
        changes; a CI smoke test keeps the committed page in sync.
        ```
        """
    )
    parts = [header]
    for module_name, title, blurb in SECTIONS:
        parts.append("\n".join(_render_module(module_name, title, blurb)))
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    OUTPUT.write_text(build())
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
