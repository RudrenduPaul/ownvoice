"""OwnVoice: train a LoRA voice adapter for pocket-tts, own the resulting model.

OwnVoice wraps kyutai-labs/pocket-tts (MIT-licensed, CPU-capable local
text-to-speech) with a LoRA fine-tuning workflow. The output is an adapter
file you keep and run yourself, not an API subscription.
"""

from __future__ import annotations

from importlib import metadata as _metadata

# Single source of truth for the reported version: the installed package's
# own metadata (populated from pyproject.toml's [project].version at build
# time), not a second hardcoded string here. A hardcoded string here
# previously drifted out of sync with pyproject.toml / the published PyPI
# release (see tests/test_version_parity.py, which now fails the build if
# this ever happens again for the other version-bearing files in this repo).
try:
    __version__ = _metadata.version("ownvoice-cli")
except _metadata.PackageNotFoundError:
    # Running from a source checkout without an installed/editable build
    # (e.g. static analysis tooling importing this module directly).
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
