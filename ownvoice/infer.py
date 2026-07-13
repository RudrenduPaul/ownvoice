"""Load a trained LoRA adapter and generate speech with it.

Real-API note: pocket-tts's `TTSModel.generate_audio(model_state, text)`
requires a `model_state` dict built by `get_state_for_audio_prompt(audio_
conditioning, ...)`, and `audio_conditioning` has no default -- it must be a
real path, URL, or tensor (confirmed against pocket_tts/models/tts_model.py,
not assumed). OwnVoice's whole differentiation is that the *voice* comes
from the trained LoRA weights, not a runtime reference clip, but pocket-tts's
generation call still needs *some* audio prompt to build the initial
conditioning state. OwnVoice resolves this by reusing the same reference
clip that `ownvoice train` recorded in `metadata.json` (the first voice clip
in the training set) by default, with an optional override.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from ownvoice.train import LORA_TARGET_MODULES, DEFAULT_LORA_ALPHA, DEFAULT_LORA_DROPOUT, DEFAULT_LORA_RANK


class AdapterLoadError(Exception):
    """Raised when an adapter file or its sibling metadata.json can't be loaded."""


def _read_metadata(adapter_path: Path) -> dict[str, Any]:
    metadata_path = adapter_path.parent / "metadata.json"
    if not metadata_path.exists():
        raise AdapterLoadError(
            f"No metadata.json found next to {adapter_path}. OwnVoice adapters are always "
            "saved with a sibling metadata.json (written by `ownvoice train`) -- if this "
            "adapter was moved on its own, put metadata.json back alongside it."
        )
    return json.loads(metadata_path.read_text())


def load_adapter(base_model: Any, adapter_path: Path | str):
    """Load a saved LoRA adapter (`adapter.safetensors`) onto pocket-tts's `flow_lm`.
    """
    # TODO(day-0-spike): reconstructs the same LoraConfig used at train time
    # (rank/alpha/dropout, read back from the adapter's sibling
    # metadata.json) and loads the saved state dict via PEFT's
    # `set_peft_model_state_dict`. Written against PEFT's real, documented
    # API, but the exact reload round-trip has not been exercised against a
    # real trained adapter in this session -- the Day-0 spike should confirm
    # it end-to-end alongside the LoRA-injection check in train.py.
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    adapter_path = Path(adapter_path)
    metadata = _read_metadata(adapter_path)
    train_config = metadata.get("config", {})

    lora_config = LoraConfig(
        r=train_config.get("lora_rank", DEFAULT_LORA_RANK),
        lora_alpha=train_config.get("lora_alpha", DEFAULT_LORA_ALPHA),
        lora_dropout=train_config.get("lora_dropout", DEFAULT_LORA_DROPOUT),
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    peft_model = get_peft_model(base_model.flow_lm, lora_config)
    state_dict = load_file(str(adapter_path))
    set_peft_model_state_dict(peft_model, state_dict)
    peft_model.eval()
    return peft_model


def resolve_reference_audio(adapter_path: Path | str, override: Path | str | None = None) -> Path:
    """Pick the audio-conditioning clip generate_speech() needs.

    Uses `override` if given, otherwise falls back to the reference clip
    OwnVoice recorded in the adapter's metadata.json at training time.
    """
    if override is not None:
        return Path(override)

    adapter_path = Path(adapter_path)
    metadata = _read_metadata(adapter_path)
    reference_clip = metadata.get("reference_clip")
    if not reference_clip:
        raise AdapterLoadError(
            f"metadata.json next to {adapter_path} has no recorded reference_clip, and no "
            "override was given. Pass one explicitly."
        )
    reference_path = Path(reference_clip)
    if reference_path.is_absolute():
        raise AdapterLoadError(
            f"Recorded reference clip {reference_path} in metadata.json is an absolute path, "
            "which OwnVoice does not trust from a shared adapter's metadata.json. Pass an "
            "override path to a voice clip explicitly."
        )

    adapter_dir = adapter_path.parent
    resolved_path = adapter_dir / reference_path
    real_adapter_dir = os.path.realpath(adapter_dir)
    real_resolved_path = os.path.realpath(resolved_path)
    if os.path.commonpath([real_adapter_dir, real_resolved_path]) != real_adapter_dir:
        raise AdapterLoadError(
            f"Recorded reference clip {reference_path} in metadata.json resolves outside of "
            f"{adapter_dir}, which OwnVoice does not trust from a shared adapter's "
            "metadata.json. Pass an override path to a voice clip explicitly."
        )
    if not resolved_path.exists():
        raise AdapterLoadError(
            f"Recorded reference clip {resolved_path} no longer exists. Pass an override "
            "path to a voice clip explicitly."
        )
    return resolved_path


def generate_speech(base_model: Any, peft_model: Any, text: str, reference_audio_path: Path | str) -> torch.Tensor:
    """Generate speech audio for `text` using the base model wrapped with the LoRA adapter.
    """
    # TODO(day-0-spike): pocket-tts's real generation entrypoint is
    # `TTSModel.generate_audio(model_state, text)` (confirmed against the
    # README and pocket_tts/models/tts_model.py), which internally reads
    # `self.flow_lm`. Swapping in the PEFT-wrapped flow_lm here (so the
    # adapter is actually used for generation instead of the frozen base)
    # needs the Day-0 spike to confirm whether reassigning
    # `base_model.flow_lm = peft_model` before calling `generate_audio` is
    # sufficient, or whether `TTSModel` caches a reference to the pre-wrap
    # module elsewhere internally. Structurally this is the correct approach
    # (PEFT-wrapped modules are drop-in nn.Module replacements), but it has
    # not been exercised against real loaded weights in this session.
    original_flow_lm = base_model.flow_lm
    try:
        base_model.flow_lm = peft_model
        model_state = base_model.get_state_for_audio_prompt(reference_audio_path)
        audio = base_model.generate_audio(model_state, text)
    finally:
        base_model.flow_lm = original_flow_lm
    return audio


def save_wav(waveform: torch.Tensor, sample_rate: int, out_path: Path | str) -> Path:
    """Save a 1D (or (1, N)) PCM waveform tensor to a .wav file.

    Uses `soundfile` directly rather than `torchaudio.save` so writing a wav
    file doesn't pull in an ffmpeg/torchcodec dependency chain just to save
    plain PCM audio.
    """
    import soundfile as sf

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)
    sf.write(str(out_path), waveform.to(torch.float32).cpu().numpy(), sample_rate)
    return out_path


def infer(
    adapter_path: Path | str,
    text: str,
    out_path: Path | str = Path("./ownvoice-output.wav"),
    reference_audio_override: Path | str | None = None,
) -> Path:
    """End-to-end infer flow used by `ownvoice infer`: load, generate, save."""
    from ownvoice.train import load_base_model

    adapter_path = Path(adapter_path)
    base_model = load_base_model(device="cpu")
    peft_model = load_adapter(base_model, adapter_path)
    reference_audio_path = resolve_reference_audio(adapter_path, reference_audio_override)
    waveform = generate_speech(base_model, peft_model, text, reference_audio_path)
    return save_wav(waveform, base_model.sample_rate, out_path)
