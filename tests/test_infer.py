"""Unit tests for ownvoice/infer.py's resolve_reference_audio().

Covers the containment fix for reference_clip paths read out of a shared
adapter's metadata.json: that field travels alongside adapter.safetensors
and may have been written by whoever trained the adapter, so it must never
be trusted as an arbitrary filesystem path. The explicit --reference-audio
override, by contrast, is the CLI's own direct trusted input and is exempt
from the containment check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ownvoice.infer import AdapterLoadError, resolve_reference_audio


def _write_adapter_and_metadata(adapter_dir: Path, reference_clip: str) -> Path:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = adapter_dir / "adapter.safetensors"
    adapter_path.write_bytes(b"")
    metadata_path = adapter_dir / "metadata.json"
    metadata_path.write_text(json.dumps({"reference_clip": reference_clip}))
    return adapter_path


def test_resolve_reference_audio_rejects_absolute_reference_clip(tmp_path: Path) -> None:
    """An absolute reference_clip in metadata.json must be rejected outright."""
    adapter_dir = tmp_path / "adapter"
    outside_clip = tmp_path / "outside" / "secret.wav"
    outside_clip.parent.mkdir(parents=True)
    outside_clip.write_bytes(b"")

    adapter_path = _write_adapter_and_metadata(adapter_dir, str(outside_clip))

    with pytest.raises(AdapterLoadError, match="absolute path"):
        resolve_reference_audio(adapter_path)


def test_resolve_reference_audio_rejects_traversal_outside_adapter_dir(tmp_path: Path) -> None:
    """A relative reference_clip using ../ to escape the adapter dir must be rejected."""
    adapter_dir = tmp_path / "adapter"
    outside_clip = tmp_path / "outside" / "secret.wav"
    outside_clip.parent.mkdir(parents=True)
    outside_clip.write_bytes(b"")

    adapter_path = _write_adapter_and_metadata(adapter_dir, "../outside/secret.wav")

    with pytest.raises(AdapterLoadError, match="resolves outside of"):
        resolve_reference_audio(adapter_path)


def test_resolve_reference_audio_accepts_relative_path_within_adapter_dir(tmp_path: Path) -> None:
    """The legitimate case: a relative reference_clip that stays inside the adapter dir works."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    clip_path = adapter_dir / "clip_a.wav"
    clip_path.write_bytes(b"")

    adapter_path = _write_adapter_and_metadata(adapter_dir, "clip_a.wav")

    resolved = resolve_reference_audio(adapter_path)

    assert resolved == clip_path


def test_resolve_reference_audio_missing_in_bounds_file_still_raises(tmp_path: Path) -> None:
    """The pre-existing missing-file check still applies to a validated in-bounds path."""
    adapter_dir = tmp_path / "adapter"
    adapter_path = _write_adapter_and_metadata(adapter_dir, "does_not_exist.wav")

    with pytest.raises(AdapterLoadError, match="no longer exists"):
        resolve_reference_audio(adapter_path)


def test_resolve_reference_audio_override_bypasses_containment_check(tmp_path: Path) -> None:
    """An explicit override is the CLI's own trusted input and skips the containment check."""
    adapter_dir = tmp_path / "adapter"
    adapter_path = _write_adapter_and_metadata(adapter_dir, "clip_a.wav")
    override_clip = tmp_path / "elsewhere" / "override.wav"

    resolved = resolve_reference_audio(adapter_path, override=override_clip)

    assert resolved == override_clip
