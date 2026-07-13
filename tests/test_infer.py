"""Unit tests for ownvoice/infer.py.

Covers resolve_reference_audio()'s containment fix for reference_clip paths
read out of a shared adapter's metadata.json: that field travels alongside
adapter.safetensors and may have been written by whoever trained the
adapter, so it must never be trusted as an arbitrary filesystem path. The
explicit --reference-audio override, by contrast, is the CLI's own direct
trusted input and is exempt from the containment check.

Also covers generate_speech()'s real orchestration logic against a fake
TTSModel (no downloaded weights needed): that it does not reassign
base_model.flow_lm (a real, previously-broken approach against pocket-tts's
actual streaming state machinery, see that function's docstring), and that
it hands get_state_for_audio_prompt() a pre-loaded, pre-resampled tensor
rather than a raw path -- using pocket_tts's own real, lightweight
audio_read()/convert_audio() helpers, which need no downloaded model
weights to run.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ownvoice.infer import AdapterLoadError, generate_speech, resolve_reference_audio


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


class _FakeBaseModelForGenerate:
    """Stands in for pocket-tts's TTSModel: no downloaded weights needed.

    Records what generate_speech() actually does to it, so the test can
    assert on the real orchestration bug this session found and fixed:
    base_model.flow_lm must be left untouched (PEFT's in-place injection
    already carries the trained weights), and get_state_for_audio_prompt()
    must receive a pre-loaded tensor, never the raw reference_audio_path.
    """

    def __init__(self) -> None:
        self.flow_lm = "sentinel-flow-lm"
        self.config = SimpleNamespace(mimi=SimpleNamespace(sample_rate=24000))
        self.calls: dict = {}

    def get_state_for_audio_prompt(self, audio_conditioning):
        self.calls["audio_conditioning"] = audio_conditioning
        return {"state": "sentinel-model-state"}

    def generate_audio(self, model_state, text):
        self.calls["model_state"] = model_state
        self.calls["text"] = text
        return torch.zeros(10)


def test_generate_speech_never_reassigns_base_model_flow_lm(tmp_path: Path) -> None:
    """The real, previously-broken approach reassigned base_model.flow_lm; the fix must not."""
    import numpy as np
    import soundfile as sf

    clip_path = tmp_path / "ref.wav"
    sf.write(str(clip_path), np.zeros(4000, dtype="float32"), 16000)

    base_model = _FakeBaseModelForGenerate()

    generate_speech(base_model, peft_model=object(), text="hello", reference_audio_path=clip_path)

    assert base_model.flow_lm == "sentinel-flow-lm"


def test_generate_speech_passes_a_preloaded_tensor_not_a_raw_path(tmp_path: Path) -> None:
    """A raw Path/URL is rejected by pocket-tts's has_voice_cloning gate on public weights.

    generate_speech() must load and resample the clip itself and hand
    get_state_for_audio_prompt() a torch.Tensor, never the original Path.
    """
    import numpy as np
    import soundfile as sf

    clip_path = tmp_path / "ref.wav"
    sf.write(str(clip_path), np.zeros(4000, dtype="float32"), 16000)

    base_model = _FakeBaseModelForGenerate()

    result = generate_speech(base_model, peft_model=object(), text="hello", reference_audio_path=clip_path)

    assert isinstance(base_model.calls["audio_conditioning"], torch.Tensor)
    assert base_model.calls["text"] == "hello"
    assert base_model.calls["model_state"] == {"state": "sentinel-model-state"}
    assert torch.equal(result, torch.zeros(10))
