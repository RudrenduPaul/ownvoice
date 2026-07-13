"""Unit tests for ownvoice/data.py: voice-clip loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ownvoice.data import VoiceClipError, load_voice_clips, load_waveform
from tests.conftest import write_sine_wav


def test_load_voice_clips_valid(voice_clips_dir: Path) -> None:
    clips = load_voice_clips(voice_clips_dir)
    assert len(clips) == 2
    names = sorted(c.path.name for c in clips)
    assert names == ["clip_a.wav", "clip_b.wav"]
    for clip in clips:
        assert clip.sample_rate == 16000
        assert clip.duration_seconds > 1.0


def test_load_voice_clips_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(VoiceClipError, match="does not exist"):
        load_voice_clips(missing)


def test_load_voice_clips_path_is_a_file_not_a_directory(tmp_path: Path) -> None:
    a_file = tmp_path / "not-a-dir.wav"
    write_sine_wav(a_file, duration_seconds=1.0, sample_rate=16000)
    with pytest.raises(VoiceClipError, match="not a directory"):
        load_voice_clips(a_file)


def test_load_voice_clips_empty_directory_raises(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(VoiceClipError, match="No .wav files found"):
        load_voice_clips(empty_dir)


def test_load_voice_clips_ignores_non_wav_files(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "notes.txt").write_text("not audio")
    with pytest.raises(VoiceClipError, match="No .wav files found"):
        load_voice_clips(clips_dir)


def test_load_voice_clips_corrupt_file_is_skipped_and_reported(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    corrupt = clips_dir / "corrupt.wav"
    corrupt.write_bytes(b"this is not a real wav file")
    with pytest.raises(VoiceClipError, match="could not read audio"):
        load_voice_clips(clips_dir)


def test_load_voice_clips_corrupt_file_skipped_when_valid_clip_also_present(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "corrupt.wav").write_bytes(b"not audio at all")
    write_sine_wav(clips_dir / "good.wav", duration_seconds=2.0, sample_rate=16000)

    clips = load_voice_clips(clips_dir)

    assert len(clips) == 1
    assert clips[0].path.name == "good.wav"


def test_load_voice_clips_too_short_clip_skipped(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    write_sine_wav(clips_dir / "tiny.wav", duration_seconds=0.1, sample_rate=16000)
    with pytest.raises(VoiceClipError, match="No usable voice clips"):
        load_voice_clips(clips_dir, min_seconds=1.0)


def test_load_voice_clips_too_long_clip_skipped(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    write_sine_wav(clips_dir / "long.wav", duration_seconds=5.0, sample_rate=16000)
    with pytest.raises(VoiceClipError, match="No usable voice clips"):
        load_voice_clips(clips_dir, max_seconds=2.0)


def test_load_voice_clips_records_nonstandard_sample_rate(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    write_sine_wav(clips_dir / "eight_khz.wav", duration_seconds=2.0, sample_rate=8000)

    clips = load_voice_clips(clips_dir)

    assert len(clips) == 1
    assert clips[0].sample_rate == 8000


def test_load_waveform_resamples_to_target_rate(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    path = clips_dir / "eight_khz.wav"
    write_sine_wav(path, duration_seconds=2.0, sample_rate=8000)
    clips = load_voice_clips(clips_dir)
    clip = clips[0]

    waveform = load_waveform(clip, target_sample_rate=24000)

    assert waveform.dim() == 1
    # Resampled length should scale with the sample-rate ratio (24000 / 8000 = 3x).
    expected_length = clip.num_frames * 3
    assert abs(waveform.shape[0] - expected_length) <= 2


def test_load_waveform_downmixes_stereo_to_mono(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    path = clips_dir / "stereo.wav"
    write_sine_wav(path, duration_seconds=1.0, sample_rate=16000, num_channels=2)
    clips = load_voice_clips(clips_dir)
    clip = clips[0]
    assert clip.num_channels == 2

    waveform = load_waveform(clip, target_sample_rate=16000)

    assert waveform.dim() == 1
    assert waveform.dtype == torch.float32
