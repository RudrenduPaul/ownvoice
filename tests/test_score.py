"""Unit tests for ownvoice/score.py: resampling and threshold logic.

Deliberately does not exercise compute_similarity() end-to-end, since that
requires downloading Resemblyzer's pretrained encoder weights -- exactly the
kind of network-dependent integration path the design doc's test plan marks
manual/integration-only. cosine_similarity() (the pure math) and is_usable()
(the threshold logic) are unit-tested directly instead.
"""

from __future__ import annotations

import math

import pytest
import torch

from ownvoice.score import (
    USABLE_THRESHOLD,
    cosine_similarity,
    is_usable,
    resample_to_16k_mono,
)


def test_resample_to_16k_mono_noop_when_already_16k() -> None:
    waveform = torch.sin(torch.linspace(0, 2 * math.pi, steps=1600))
    out = resample_to_16k_mono(waveform, orig_sample_rate=16000)
    assert out.dim() == 1
    assert out.shape[0] == 1600
    assert out.dtype == torch.float32


def test_resample_to_16k_mono_downsamples_32k_to_16k() -> None:
    waveform = torch.sin(torch.linspace(0, 4 * math.pi, steps=3200))
    out = resample_to_16k_mono(waveform, orig_sample_rate=32000)
    assert out.dim() == 1
    # 32kHz -> 16kHz halves the sample count (allow resampler edge slack).
    assert abs(out.shape[0] - 1600) <= 4


def test_resample_to_16k_mono_upsamples_8k_to_16k() -> None:
    waveform = torch.sin(torch.linspace(0, 2 * math.pi, steps=800))
    out = resample_to_16k_mono(waveform, orig_sample_rate=8000)
    assert out.dim() == 1
    assert abs(out.shape[0] - 1600) <= 4


def test_resample_to_16k_mono_downmixes_stereo() -> None:
    left = torch.sin(torch.linspace(0, 2 * math.pi, steps=1600))
    right = torch.cos(torch.linspace(0, 2 * math.pi, steps=1600))
    stereo = torch.stack([left, right], dim=0)
    out = resample_to_16k_mono(stereo, orig_sample_rate=16000)
    assert out.dim() == 1
    assert out.shape[0] == 1600


def test_cosine_similarity_identical_vectors_is_one() -> None:
    a = [1.0, 2.0, 3.0, 4.0]
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_opposite_vectors_is_negative_one() -> None:
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_does_not_raise() -> None:
    a = [0.0, 0.0, 0.0]
    b = [1.0, 2.0, 3.0]
    assert cosine_similarity(a, b) == 0.0


def test_is_usable_at_exact_threshold_is_true() -> None:
    assert is_usable(USABLE_THRESHOLD) is True


def test_is_usable_just_below_threshold_is_false() -> None:
    assert is_usable(USABLE_THRESHOLD - 0.001) is False


def test_is_usable_just_above_threshold_is_true() -> None:
    assert is_usable(USABLE_THRESHOLD + 0.001) is True


def test_is_usable_well_below_threshold_is_false() -> None:
    assert is_usable(0.2) is False


def test_is_usable_well_above_threshold_is_true() -> None:
    assert is_usable(0.95) is True


def test_is_usable_respects_custom_threshold() -> None:
    assert is_usable(0.6, threshold=0.5) is True
    assert is_usable(0.4, threshold=0.5) is False
