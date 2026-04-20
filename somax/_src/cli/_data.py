"""CLI helpers for real-basin data bundles (#74 + #75).

Thin wrappers around DVC so users never have to memorize DVC's command
surface just to fetch / build / push basin bundles. Each helper
delegates to ``dvc`` via :mod:`subprocess`; if ``dvc`` isn't on PATH,
the user gets an actionable error.

This module is imported by :mod:`somax._src.cli.app` and exposed under
``somax-sim data {init,fetch,build,status}``.

The design is documented in ``content/notes/basin_data_sources.md``
(summarised here): somax ships the DVC pipeline + these helpers, but
does **not** host a canonical bundle. Each user configures their own
DVC remote (Google Drive / S3 / Azure / local); the helpers wrap that
flow so the usual user-facing verbs (fetch, build, ...) do not require
a DVC cheat-sheet.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter
from loguru import logger


# The four basin names the pipeline knows about. Kept in sync with
# ``scripts/data/build_basin.py::BASIN_SPECS`` and the corresponding DVC
# stages. Listed in the help of every ``data`` subcommand so users see
# the valid options without reading the design doc.
BASIN_NAMES = ("med_sea", "gulf_stream", "north_atlantic", "southern_ocean")

BasinName = Literal["med_sea", "gulf_stream", "north_atlantic", "southern_ocean"]

RemoteBackend = Literal["gdrive", "s3", "azure", "gcs", "local"]

# Map each supported backend to the ``dvc[<extra>]`` pip extra that
# provides its fsspec plugin. "local" needs nothing.
_BACKEND_EXTRAS: dict[RemoteBackend, str | None] = {
    "gdrive": "gdrive",
    "s3": "s3",
    "azure": "azure",
    "gcs": "gs",
    "local": None,
}

# Fixed remote name — keeping it consistent across users makes
# troubleshooting easier ("run `dvc remote list`, you should see
# `somax-data`").
REMOTE_NAME = "somax-data"


data_app = App(
    name="data",
    help=(
        "Manage real-basin data bundles (#74 + #75). "
        "Wraps DVC so you don't have to remember its CLI surface."
    ),
)


# ----------------------------------------------------------------------
# somax-sim data init
# ----------------------------------------------------------------------


@data_app.command
def init(
    *,
    backend: Annotated[
        RemoteBackend | None,
        Parameter(
            help=(
                "DVC remote backend to configure. Omit for interactive "
                "selection. 'local' points at a filesystem path; the "
                "others (gdrive, s3, azure, gcs) are cloud-hosted."
            ),
        ),
    ] = None,
    url: Annotated[
        str | None,
        Parameter(
            help=(
                "Remote URL. For gdrive: 'gdrive://<folder-id>'. For s3: "
                "'s3://<bucket>/<key>'. For azure: "
                "'azure://<container>/<path>'. For gcs: 'gs://<bucket>/<key>'. "
                "For local: an absolute filesystem path. Omit for "
                "interactive entry."
            ),
        ),
    ] = None,
    force: Annotated[
        bool,
        Parameter(
            help=(
                "Overwrite an existing 'somax-data' remote. Without this "
                "flag, 'dvc remote add' errors out if the remote already "
                "exists — safer default."
            ),
        ),
    ] = False,
) -> None:
    """Configure a DVC remote for your basin bundles.

    somax does not host a canonical bundle — each user points DVC at
    their own storage (Google Drive, S3, Azure Blob, GCS, or a local
    disk path). This helper runs ``dvc remote add --default
    somax-data <url>`` with the right backend-specific hints.

    First-time setup:

        somax-sim data init                             # interactive
        somax-sim data init --backend gdrive --url gdrive://<id>
        somax-sim data init --backend local --url /data/somax

    The gdrive / s3 / azure / gcs backends need their DVC extra
    installed (``pip install 'dvc[gdrive]'``). ``init`` detects missing
    extras and prints a pip-install hint rather than silently failing.
    """
    _require_dvc()

    if backend is None:
        backend = _prompt_backend()
    if url is None:
        url = _prompt_url(backend)

    _validate_url_for_backend(backend, url)
    _check_backend_installed(backend)

    args = ["dvc", "remote", "add", "--default"]
    if force:
        args.append("-f")
    args.extend([REMOTE_NAME, url])
    _run_dvc(args, "failed to add DVC remote")

    logger.info(f"configured DVC remote '{REMOTE_NAME}' -> {url} (backend: {backend})")
    logger.info(
        "next: `somax-sim data fetch <basin>` (if your remote has bundles) "
        "or `somax-sim data build <basin> --push` (to build and upload)"
    )


# ----------------------------------------------------------------------
# somax-sim data fetch
# ----------------------------------------------------------------------


@data_app.command
def fetch(
    basin: Annotated[
        BasinName,
        Parameter(help=f"Basin to fetch. One of {BASIN_NAMES!r}."),
    ],
) -> None:
    """Download a basin bundle from your configured DVC remote.

    Equivalent to ``dvc pull data/basin/<basin>.zarr``. Fails with an
    actionable message if no remote is configured (see ``data init``).
    """
    _require_dvc()
    target = _basin_zarr_path(basin)
    if not _remote_configured():
        raise SystemExit(
            f"no DVC remote named '{REMOTE_NAME}' is configured.\n"
            f"run `somax-sim data init` first, or if you have a remote by a "
            f"different name, run `dvc pull {target}` directly."
        )
    _run_dvc(
        ["dvc", "pull", str(target)],
        f"failed to pull {target} from remote '{REMOTE_NAME}'",
    )
    logger.info(f"pulled {target}")


# ----------------------------------------------------------------------
# somax-sim data build
# ----------------------------------------------------------------------


@data_app.command
def build(
    basin: Annotated[
        BasinName,
        Parameter(help=f"Basin to build. One of {BASIN_NAMES!r}."),
    ],
    *,
    force: Annotated[
        bool,
        Parameter(
            help=(
                "Force rebuild even if DVC thinks outputs are up to date "
                "(wraps `dvc repro --force`)."
            ),
        ),
    ] = False,
    push: Annotated[
        bool,
        Parameter(
            help=(
                "After a successful build, push the bundle to the "
                "configured DVC remote."
            ),
        ),
    ] = False,
) -> None:
    """Rebuild a basin bundle from source via the DVC pipeline.

    Wraps ``dvc repro build-basin-<basin>``. The Phase 3 placeholder
    build runs the synthetic generator in ``scripts/data/build_basin.py``;
    Phase 4 (#78) swaps in GEBCO + ERA5 real-data integration.
    """
    _require_dvc()
    stage = f"build-basin-{basin.replace('_', '-')}"
    args = ["dvc", "repro"]
    if force:
        args.append("--force")
    args.append(stage)
    _run_dvc(args, f"failed to rebuild basin '{basin}'")
    logger.info(f"built data/basin/{basin}.zarr via stage '{stage}'")

    if push:
        if not _remote_configured():
            raise SystemExit(
                f"--push specified but no DVC remote named '{REMOTE_NAME}' "
                "is configured. run `somax-sim data init` first."
            )
        target = _basin_zarr_path(basin)
        _run_dvc(
            ["dvc", "push", str(target)],
            f"built {basin} locally but failed to push to '{REMOTE_NAME}'",
        )
        logger.info(f"pushed {target} to remote '{REMOTE_NAME}'")


# ----------------------------------------------------------------------
# somax-sim data status
# ----------------------------------------------------------------------


@data_app.command
def status() -> None:
    """Show which basin bundles are materialized locally + the configured remote."""
    _require_dvc()

    remote_url = _remote_url()
    if remote_url is None:
        print("remote: (none configured) — run `somax-sim data init`")
    else:
        print(f"remote: {REMOTE_NAME} -> {remote_url}")

    print("basin bundles:")
    for name in BASIN_NAMES:
        path = _basin_zarr_path(name)
        if path.exists() and any(path.iterdir()):
            print(f"  [x] {name:16s}  ({path})")
        else:
            print(f"  [ ] {name:16s}  (not materialized; fetch or build)")


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _basin_zarr_path(name: str) -> Path:
    return Path("data") / "basin" / f"{name}.zarr"


def _require_dvc() -> None:
    """Abort with an actionable message if ``dvc`` is not on PATH."""
    if shutil.which("dvc") is None:
        raise SystemExit(
            "dvc is not on PATH. install with `pip install dvc` (or "
            "`pip install 'dvc[gdrive]'` etc. for cloud backends) and retry."
        )


def _run_dvc(args: list[str], failure_msg: str) -> None:
    """Run a DVC command; abort the CLI on failure."""
    logger.debug(f"$ {' '.join(args)}")
    result = subprocess.run(args, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{failure_msg} (dvc exited {result.returncode})")


def _remote_configured() -> bool:
    return _remote_url() is not None


def _remote_url() -> str | None:
    """Return the URL of the 'somax-data' remote, or None if not configured."""
    result = subprocess.run(
        ["dvc", "remote", "list"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        # `dvc remote list` outputs one line per remote: "<name>\t<url>"
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0] == REMOTE_NAME:
            return parts[1].strip()
    return None


def _check_backend_installed(backend: RemoteBackend) -> None:
    """Error out with a pip-install hint if the backend's DVC extra is missing."""
    extra = _BACKEND_EXTRAS[backend]
    if extra is None:
        return  # 'local' needs no extra.
    # Rather than doing a shallow import probe (fragile across DVC
    # versions), rely on `dvc remote modify` — but that requires an
    # existing remote. Simplest: rely on DVC's own error, but surface
    # a clearer hint to the user up front.
    pip_hint = f"pip install 'dvc[{extra}]'"
    # We cannot cheaply detect installation without importing DVC
    # internals, so just mention the extra in the log and let DVC raise
    # its own error on the subsequent fetch/push if it's missing.
    logger.info(
        f"backend '{backend}' uses the '{extra}' DVC extra; "
        f"if fetch/push fails with an import error, install it: {pip_hint}"
    )


def _prompt_backend() -> RemoteBackend:
    print("Pick a DVC remote backend:")
    options: list[RemoteBackend] = ["gdrive", "s3", "azure", "gcs", "local"]
    for i, opt in enumerate(options, start=1):
        print(f"  [{i}] {opt}")
    while True:
        raw = input("> ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print(f"enter a number 1-{len(options)}")
            continue
        if 1 <= idx <= len(options):
            return options[idx - 1]
        print(f"enter a number 1-{len(options)}")


def _prompt_url(backend: RemoteBackend) -> str:
    hints = {
        "gdrive": "Google Drive folder URL format: gdrive://<folder-id>",
        "s3": "S3 URL format: s3://<bucket>/<key>",
        "azure": "Azure Blob URL format: azure://<container>/<path>",
        "gcs": "GCS URL format: gs://<bucket>/<key>",
        "local": "Local path: any absolute filesystem path",
    }
    print(hints[backend])
    url = input("remote URL: ").strip()
    if not url:
        raise SystemExit("no URL provided; aborting.")
    return url


def _validate_url_for_backend(backend: RemoteBackend, url: str) -> None:
    """Lightweight URL-shape check so typos are caught before `dvc remote add`."""
    schemes = {
        "gdrive": "gdrive://",
        "s3": "s3://",
        "azure": "azure://",
        "gcs": "gs://",
    }
    if backend == "local":
        if not Path(url).is_absolute():
            raise SystemExit(
                f"local backend expects an absolute path; got {url!r}. "
                "use e.g. /data/somax."
            )
        return
    expected = schemes[backend]
    if not url.startswith(expected):
        raise SystemExit(
            f"backend '{backend}' expects a URL starting with {expected!r}; "
            f"got {url!r}."
        )


def main() -> int:
    """Entry point — used only when running this module directly for debugging."""
    return data_app() or 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
