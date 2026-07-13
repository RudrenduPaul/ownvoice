"""Speaker-similarity scoring: resample to 16kHz mono, Resemblyzer cosine similarity.

Per the locked design (Approach A), 0.75 cosine similarity between the
reference clip's and the generated clip's Resemblyzer speaker embeddings is
the provisional bar for "usable adapter." Clearing it or not is reported as
a labeled result, never a crash -- see ownvoice/train.py's train() flow.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torchaudio

RESEMBLYZER_SAMPLE_RATE = 16000
USABLE_THRESHOLD = 0.75


def resample_to_16k_mono(waveform: torch.Tensor, orig_sample_rate: int) -> torch.Tensor:
    """Resample a waveform tensor to 16kHz mono for Resemblyzer's embedding model.

    Accepts a 1D (samples,) or 2D (channels, samples) tensor. Returns a 1D
    float32 tensor. A no-op resample (already 16kHz) still runs through
    downmixing and dtype normalization for a consistent return shape.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sample_rate != RESEMBLYZER_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sample_rate, new_freq=RESEMBLYZER_SAMPLE_RATE
        )
        waveform = resampler(waveform)
    return waveform.squeeze(0).to(torch.float32)


def load_and_resample(path: Path | str) -> torch.Tensor:
    """Load a wav file from disk and return it as 16kHz mono float32.

    Reads via `soundfile` (not `torchaudio.load`, see the note in
    ownvoice/data.py for why), then resamples with
    `torchaudio.transforms.Resample`.
    """
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T).to(torch.float32)
    return resample_to_16k_mono(waveform, sample_rate)


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two 1D embedding vectors (numpy or list-like).

    Pure math, no Resemblyzer/model dependency -- kept separate from
    compute_similarity() so the threshold logic is unit-testable without
    downloading Resemblyzer's pretrained encoder weights.
    """
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_similarity(
    reference_waveform_16k: torch.Tensor,
    generated_waveform_16k: torch.Tensor,
) -> float:
    """Cosine similarity between the two waveforms' Resemblyzer speaker embeddings.

    Both inputs must already be 16kHz mono float32 tensors (use
    resample_to_16k_mono or load_and_resample first). Resemblyzer is
    imported lazily here, not at module load, so importing ownvoice.score
    doesn't require Resemblyzer's pretrained weights to be downloaded yet.
    """
    from resemblyzer import VoiceEncoder

    encoder = VoiceEncoder()
    ref_embed = encoder.embed_utterance(reference_waveform_16k.numpy())
    gen_embed = encoder.embed_utterance(generated_waveform_16k.numpy())
    return cosine_similarity(ref_embed, gen_embed)


def is_usable(score: float, threshold: float = USABLE_THRESHOLD) -> bool:
    """True if `score` clears the usable-adapter bar (>= threshold, default 0.75)."""
    return score >= threshold
