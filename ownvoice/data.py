"""Voice-clip loading and validation for OwnVoice training data.

pocket-tts's own sample rate is read from the loaded model at train time
(`TTSModel.sample_rate`); this module only validates raw input files and
normalizes them to whatever target rate the caller asks for.

Audio I/O uses `soundfile` directly rather than `torchaudio.load`/
`torchaudio.info`: as of the torchaudio version this project pins against
(2.9+), both of those now route through an optional `torchcodec`/ffmpeg
backend and raise `ImportError` without it installed. `soundfile` (libsndfile)
reads plain .wav files with no extra runtime dependency. Resampling still
goes through `torchaudio.transforms.Resample` per the locked design decision,
since that's a pure-tensor operation with no I/O backend involved.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

MIN_CLIP_SECONDS = 1.0
MAX_CLIP_SECONDS = 30.0
SUPPORTED_SUFFIXES = {".wav"}


class VoiceClipError(Exception):
    """Raised when a voice-clip directory or file fails validation."""


@dataclasses.dataclass(frozen=True)
class VoiceClip:
    """Metadata for one validated voice-clip file, not yet loaded into memory."""

    path: Path
    sample_rate: int
    num_frames: int
    num_channels: int

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.sample_rate


def _iter_candidate_files(voice_clips_dir: Path) -> list[Path]:
    if not voice_clips_dir.exists():
        raise VoiceClipError(f"Voice clips directory does not exist: {voice_clips_dir}")
    if not voice_clips_dir.is_dir():
        raise VoiceClipError(f"Voice clips path is not a directory: {voice_clips_dir}")
    return sorted(
        p
        for p in voice_clips_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_voice_clips(
    voice_clips_dir: Path | str,
    *,
    min_seconds: float = MIN_CLIP_SECONDS,
    max_seconds: float = MAX_CLIP_SECONDS,
) -> list[VoiceClip]:
    """Validate every .wav file in `voice_clips_dir` and return clip metadata.

    A file that individually fails to decode, is empty, or falls outside the
    duration bounds is skipped (not fatal on its own) -- only an empty result
    set raises. Raises VoiceClipError with an actionable message either way
    the directory itself doesn't exist/isn't a directory, has no .wav files
    at all, or every candidate file was skipped.
    """
    voice_clips_dir = Path(voice_clips_dir)
    candidates = _iter_candidate_files(voice_clips_dir)
    if not candidates:
        raise VoiceClipError(
            f"No .wav files found in {voice_clips_dir}. OwnVoice needs at least one "
            "clean .wav recording of the target voice to train from."
        )

    clips: list[VoiceClip] = []
    skipped: list[str] = []
    for path in candidates:
        try:
            info = sf.info(str(path))
        except Exception as exc:  # noqa: BLE001 - soundfile raises backend-specific errors
            skipped.append(f"{path.name}: could not read audio ({exc})")
            continue

        if info.frames <= 0:
            skipped.append(f"{path.name}: empty audio (0 frames)")
            continue

        duration = info.frames / info.samplerate
        if duration < min_seconds:
            skipped.append(f"{path.name}: too short ({duration:.2f}s, minimum is {min_seconds}s)")
            continue
        if duration > max_seconds:
            skipped.append(f"{path.name}: too long ({duration:.2f}s, maximum is {max_seconds}s per clip)")
            continue

        clips.append(
            VoiceClip(
                path=path,
                sample_rate=info.samplerate,
                num_frames=info.frames,
                num_channels=info.channels,
            )
        )

    if not clips:
        detail = "; ".join(skipped) if skipped else "no readable clips"
        raise VoiceClipError(
            f"No usable voice clips found in {voice_clips_dir} ({detail}). "
            f"Each clip must be a valid mono or stereo .wav file between "
            f"{min_seconds}s and {max_seconds}s."
        )

    return clips


def load_waveform(clip: VoiceClip, target_sample_rate: int) -> torch.Tensor:
    """Load a clip's waveform, downmix to mono, and resample to `target_sample_rate`.

    Returns a 1D float32 tensor. This is the shared normalization path used
    both by the training data pipeline and (via score.py) by evaluation, so
    every clip reaches the model or the scorer in a consistent format
    regardless of its original sample rate or channel count.
    """
    audio, sample_rate = sf.read(str(clip.path), dtype="float32", always_2d=True)
    # soundfile returns (frames, channels); torchaudio's convention is (channels, frames).
    waveform = torch.from_numpy(audio.T).to(torch.float32)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze(0).to(torch.float32)
