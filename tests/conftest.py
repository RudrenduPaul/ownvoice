"""Shared pytest fixtures: synthetic wav-file generation for data/score tests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def write_sine_wav(path: Path, *, duration_seconds: float, sample_rate: int, num_channels: int = 1, freq: float = 220.0) -> None:
    """Write a synthetic sine-wave .wav file for use as a test fixture.

    Uses soundfile directly (not torchaudio.save) so fixture generation
    doesn't require torchcodec/ffmpeg to be installed -- see the same
    reasoning in ownvoice/infer.py's save_wav().
    """
    num_frames = max(1, int(duration_seconds * sample_rate))
    t = np.arange(num_frames, dtype=np.float32) / sample_rate
    tone = 0.1 * np.sin(2 * math.pi * freq * t).astype(np.float32)
    if num_channels > 1:
        tone = np.stack([tone] * num_channels, axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), tone, sample_rate)


@pytest.fixture
def voice_clips_dir(tmp_path: Path) -> Path:
    """A directory with two valid mono 16kHz .wav clips, ~2 seconds each."""
    clips_dir = tmp_path / "voice-clips"
    clips_dir.mkdir()
    write_sine_wav(clips_dir / "clip_a.wav", duration_seconds=2.0, sample_rate=16000, freq=220.0)
    write_sine_wav(clips_dir / "clip_b.wav", duration_seconds=2.5, sample_rate=16000, freq=330.0)
    return clips_dir
