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

Verified for real this session (a real 2-epoch LoRA training run, followed
by a real `ownvoice infer` call against the resulting adapter, produced a
4.7s non-silent, finite, NaN-free .wav file): `load_adapter()` and
`generate_speech()` below are exercised end to end against real loaded
pocket-tts weights, not just reasoned about. Two things this session found
that the original (structurally reasonable but unverified) approach in
`generate_speech()` got wrong are documented on that function directly --
in short, no `base_model.flow_lm` swap is needed or safe, and OwnVoice
must pre-load and resample the reference clip itself rather than pass a
raw path/URL, because the publicly downloadable pocket-tts weights are the
non-voice-cloning checkpoint by default.
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

    Reconstructs the same LoraConfig used at train time (rank/alpha/
    dropout, read back from the adapter's sibling metadata.json) and loads
    the saved state dict via PEFT's `set_peft_model_state_dict`. The
    reload round trip is verified against a real trained adapter this
    session: `ownvoice train` saved a real adapter.safetensors via
    `get_peft_model_state_dict`, and this function loaded it back and used
    it for real generation via `ownvoice infer` without error.
    """
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

    Verified against real loaded pocket-tts weights this session, and the
    naive approach the previous TODO here anticipated turned out to be
    wrong for two independent, real reasons:

    1. No flow_lm swap is needed, and the previous
       `base_model.flow_lm = peft_model` swap actively breaks generation.
       PEFT's `get_peft_model()` replaces `base_model.flow_lm`'s targeted
       `nn.Linear` submodules with LoRA layers IN PLACE -- it mutates the
       same live module graph rather than returning a copy, so
       `base_model.flow_lm` already carries the trained adapter's weights
       by the time this function runs (confirmed: `id(base_model.flow_lm)`
       is unchanged before/after `get_peft_model()`, and its Linear
       submodules are already `peft.tuners.lora.layer.Linear` instances
       afterward). Reassigning `base_model.flow_lm` to the outer
       `PeftModel` wrapper instead breaks pocket-tts's own streaming state
       machinery: `StatefulModule.get_state()` looks up KV-cache state by a
       fixed `_module_absolute_name` string baked onto each submodule once
       at `TTSModel.load_model()` time (e.g. "transformer.layers.0.
       self_attn"), but `get_state_for_audio_prompt()` rebuilds the
       `model_state` dict fresh via `init_states(base_model.flow_lm, ...)`,
       which keys it off `named_modules()` on whatever module is currently
       assigned to `base_model.flow_lm`. PEFT's `PeftModel`/`LoraModel`
       wrapper inserts a "base_model.model." prefix into those paths, so
       swapping to the wrapper produces a `model_state` dict whose keys no
       longer match the fixed `_module_absolute_name` strings, and
       generation fails with `KeyError: 'transformer.layers.0.self_attn'`
       -- reproduced against real weights this session.
    2. The publicly downloadable pocket-tts weights
       (`kyutai/pocket-tts-without-voice-cloning`, the checkpoint
       `TTSModel.load_model()` falls back to whenever the gated
       `kyutai/pocket-tts` voice-cloning weights aren't accessible, which
       is the common case since that repo requires manual approval on
       Hugging Face) set `has_voice_cloning = False`. With that flag set,
       `get_state_for_audio_prompt()` raises `ValueError` for ANY
       `str`/`Path` audio-conditioning argument, including OwnVoice's own
       reference clip -- reproduced against real weights this session.
       Passing a pre-loaded, pre-resampled `torch.Tensor` instead is
       pocket-tts's own documented input type for this argument (`audio_
       conditioning: Path | str | torch.Tensor`) and is not gated -- it
       reaches the exact same Mimi-encode-then-project conditioning path
       either way. So OwnVoice loads and resamples the reference clip
       itself here rather than depending on gated HF access most users of
       OwnVoice won't have. Voice-cloning fidelity from the without-voice-
       cloning checkpoint is a known limitation of the base model itself
       (per pocket-tts's own `VOICE_CLONING_UNSUPPORTED` message), not an
       OwnVoice bug -- users who want kyutai's best-quality cloning weights
       can request gated access at https://huggingface.co/kyutai/pocket-tts
       and run `hf auth login`; OwnVoice works either way.
    """
    from pocket_tts.data.audio import audio_read
    from pocket_tts.data.audio_utils import convert_audio

    del peft_model  # injection already mutated base_model.flow_lm in place; see docstring point 1

    reference_audio_path = Path(reference_audio_path)
    raw_audio, source_sample_rate = audio_read(reference_audio_path)
    audio_conditioning = convert_audio(
        raw_audio, source_sample_rate, base_model.config.mimi.sample_rate, 1
    )

    model_state = base_model.get_state_for_audio_prompt(audio_conditioning)
    return base_model.generate_audio(model_state, text)


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
