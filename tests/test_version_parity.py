"""Guards against the exact version-drift bug this project shipped once already:
`ownvoice/__init__.py` hardcoded a version string that fell out of sync with
`pyproject.toml` (and, by extension, the published PyPI release), and separately
`npm/package.json` fell out of sync with both.

`ownvoice/__init__.py` no longer hardcodes a version at all -- it reads
`importlib.metadata.version("ownvoice-cli")`, so it can never drift from
whatever was actually installed. The one remaining place a human has to
remember to bump a second file by hand is `npm/package.json` (the npm
wrapper pins `uvx --from ownvoice-cli==<version>`, so a stale version there
silently runs an old PyPI release). This test is the enforcement: it fails
the build the moment `npm/package.json`'s version disagrees with
`pyproject.toml`'s version, instead of relying on a human to notice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - project requires-python >=3.11
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    pyproject = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data["project"]["version"]


def _npm_package_version() -> str:
    package_json = REPO_ROOT / "npm" / "package.json"
    data = json.loads(package_json.read_text(encoding="utf-8"))
    return data["version"]


def test_npm_package_version_matches_pyproject() -> None:
    """npm/package.json must be bumped in lockstep with pyproject.toml.

    The npm wrapper (npm/bin/ownvoice.js) pins `uvx --from
    ownvoice-cli==<PACKAGE_VERSION>`, using this exact field. If it drifts
    behind pyproject.toml/PyPI, `npx ownvoice-cli` silently runs a stale,
    already-superseded release instead of the one PyPI currently ships.
    """
    pyproject_version = _pyproject_version()
    npm_version = _npm_package_version()

    assert npm_version == pyproject_version, (
        f"npm/package.json version ({npm_version!r}) has drifted from "
        f"pyproject.toml version ({pyproject_version!r}). Bump npm/package.json "
        "to match -- the npm wrapper pins the exact PyPI release it bootstraps "
        "from this field, so a stale value here silently runs an old release."
    )


def test_installed_version_matches_pyproject() -> None:
    """ownvoice.__version__ (from installed package metadata) must match
    pyproject.toml's declared version for this checkout.

    This only holds when the package is installed from this exact checkout
    (editable or otherwise); it's skipped if metadata is unavailable, which
    happens only for tooling that imports ownvoice without installing it.
    """
    from ownvoice import __version__

    if __version__ == "0.0.0+unknown":
        import pytest

        pytest.skip("ownvoice-cli is not installed; __version__ has no metadata to compare")

    assert __version__ == _pyproject_version(), (
        f"Installed ownvoice-cli metadata reports version {__version__!r}, but "
        f"pyproject.toml declares {_pyproject_version()!r}. Reinstall "
        "(`pip install -e .` / `uv sync`) after bumping pyproject.toml."
    )
