"""Unit tests for ownvoice/train.py.

Running `ownvoice train` against pocket-tts's actual downloaded weights on
a GPU is integration/manual-only per the [redacted]. What's
unit-tested here: the caught-PEFT-exception path (mocked), the
compatibility-check orchestration (mocked), the adapter+manifest save/load
round trip (real safetensors + json, no model needed), and
`_compute_flow_matching_loss()`'s real math against a small fake
pocket-tts-shaped model (real nn.Module forward/backward, no downloaded
weights needed -- `pocket_tts` itself is a real, always-installed
dependency of this package, so its lightweight `init_states()` helper is
exercised for real too, only the multi-GB weights are faked out).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from ownvoice.train import (
    _compute_flow_matching_loss,
    CheckResult,
    PeftInjectionError,
    TrainingConfig,
    check_compatibility,
    describe_module_tree,
    inject_lora,
    save_adapter_and_manifest,
)


class _TinyLinearModel(nn.Module):
    """A small real nn.Module standing in for pocket-tts's flow_lm in tests."""

    def __init__(self) -> None:
        super().__init__()
        self.input_linear = nn.Linear(4, 8, bias=False)
        self.block = nn.ModuleDict(
            {
                "self_attn": nn.ModuleDict(
                    {
                        "in_proj": nn.Linear(8, 24, bias=False),
                        "out_proj": nn.Linear(8, 8, bias=False),
                    }
                )
            }
        )
        self.out_eos = nn.Linear(8, 1, bias=False)


def test_describe_module_tree_lists_named_submodules() -> None:
    model = _TinyLinearModel()
    tree = describe_module_tree(model)
    assert "input_linear: Linear" in tree
    assert "out_eos: Linear" in tree
    assert "block.self_attn.in_proj: Linear" in tree


def test_inject_lora_success_returns_peft_model(monkeypatch) -> None:
    model = _TinyLinearModel()
    sentinel = object()

    def fake_get_peft_model(module, config):
        assert module is model
        assert config.target_modules == "all-linear"
        return sentinel

    import peft

    monkeypatch.setattr(peft, "get_peft_model", fake_get_peft_model)

    result = inject_lora(model, rank=4, alpha=8, dropout=0.1)

    assert result is sentinel


def test_inject_lora_failure_raises_peft_injection_error_with_module_tree(monkeypatch) -> None:
    model = _TinyLinearModel()

    def fake_get_peft_model(module, config):
        raise RuntimeError("target_modules resolution failed: no matching layers")

    import peft

    monkeypatch.setattr(peft, "get_peft_model", fake_get_peft_model)

    try:
        inject_lora(model)
        assert False, "expected PeftInjectionError"
    except PeftInjectionError as exc:
        assert "issues/30" in str(exc)
        assert "input_linear: Linear" in exc.module_tree
        assert isinstance(exc.__cause__, RuntimeError)


def test_check_compatibility_success(monkeypatch) -> None:
    import ownvoice.train as train_module

    class _FakeBaseModel:
        flow_lm = _TinyLinearModel()

    monkeypatch.setattr(train_module, "load_base_model", lambda device="cpu": _FakeBaseModel())
    monkeypatch.setattr(train_module, "inject_lora", lambda flow_lm, **kwargs: object())

    result = check_compatibility()

    assert isinstance(result, CheckResult)
    assert result.success is True
    assert result.module_tree is None


def test_check_compatibility_model_load_failure(monkeypatch) -> None:
    import ownvoice.train as train_module

    def _raise(device="cpu"):
        raise FileNotFoundError("weights not found")

    monkeypatch.setattr(train_module, "load_base_model", _raise)

    result = check_compatibility()

    assert result.success is False
    assert "Could not load the pocket-tts base model" in result.message


def test_check_compatibility_peft_injection_failure_is_caught(monkeypatch) -> None:
    """The caught-PEFT-exception path: injection raises, check_compatibility reports it, doesn't crash."""
    import ownvoice.train as train_module

    class _FakeBaseModel:
        flow_lm = _TinyLinearModel()

    def _raise_injection(flow_lm, **kwargs):
        raise PeftInjectionError("injection failed for testing", module_tree="fake tree contents")

    monkeypatch.setattr(train_module, "load_base_model", lambda device="cpu": _FakeBaseModel())
    monkeypatch.setattr(train_module, "inject_lora", _raise_injection)

    result = check_compatibility()

    assert result.success is False
    assert result.message == "injection failed for testing"
    assert result.module_tree == "fake tree contents"


def test_save_adapter_and_manifest_writes_both_files(tmp_path: Path) -> None:
    state_dict = {"lora_A.weight": torch.randn(4, 2), "lora_B.weight": torch.randn(2, 4)}
    config = TrainingConfig(voice_clips_dir=tmp_path / "clips", out_dir=tmp_path / "out")

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    reference_clip_path = clips_dir / "clip_a.wav"
    reference_clip_path.write_bytes(b"fake-wav-bytes-for-copy-test")

    result_dir = save_adapter_and_manifest(
        state_dict,
        config.out_dir,
        config,
        similarity_score=0.82,
        metrics={"final_loss": 0.1, "epochs": 3},
        timestamp="2026-07-12T00:00:00+00:00",
        reference_clip=reference_clip_path,
    )

    assert result_dir == config.out_dir
    adapter_path = config.out_dir / "adapter.safetensors"
    metadata_path = config.out_dir / "metadata.json"
    assert adapter_path.exists()
    assert metadata_path.exists()

    from safetensors.torch import load_file

    loaded_state_dict = load_file(str(adapter_path))
    assert set(loaded_state_dict.keys()) == set(state_dict.keys())
    assert torch.allclose(loaded_state_dict["lora_A.weight"], state_dict["lora_A.weight"])

    metadata = json.loads(metadata_path.read_text())
    assert metadata["similarity_score"] == 0.82
    assert metadata["usable"] is True
    assert metadata["usable_threshold"] == 0.75
    assert metadata["timestamp"] == "2026-07-12T00:00:00+00:00"
    assert metadata["config"]["epochs"] == config.epochs
    assert metadata["config"]["lora_rank"] == config.lora_rank
    assert metadata["metrics"]["final_loss"] == 0.1
    assert metadata["reference_clip"].endswith("clip_a.wav")
    assert (config.out_dir / "clip_a.wav").exists()


def test_save_adapter_and_manifest_below_threshold_is_usable_false(tmp_path: Path) -> None:
    state_dict = {"lora_A.weight": torch.randn(2, 2)}
    config = TrainingConfig(voice_clips_dir=tmp_path / "clips", out_dir=tmp_path / "out")

    save_adapter_and_manifest(state_dict, config.out_dir, config, similarity_score=0.4)

    metadata = json.loads((config.out_dir / "metadata.json").read_text())
    assert metadata["usable"] is False
    assert metadata["similarity_score"] == 0.4


def test_save_adapter_and_manifest_creates_out_dir(tmp_path: Path) -> None:
    state_dict = {"w": torch.randn(2, 2)}
    config = TrainingConfig(voice_clips_dir=tmp_path / "clips", out_dir=tmp_path / "nested" / "out")

    assert not config.out_dir.exists()

    save_adapter_and_manifest(state_dict, config.out_dir, config, similarity_score=0.9)

    assert config.out_dir.exists()
    assert (config.out_dir / "adapter.safetensors").exists()


class _FakeFlowNet(nn.Module):
    """Stands in for pocket-tts's SimpleMLPAdaLN: forward(c, s, t, x) -> predicted velocity."""

    def __init__(self, ldim: int, dim: int) -> None:
        super().__init__()
        self.from_cond = nn.Linear(dim, ldim, bias=False)
        self.from_x = nn.Linear(ldim, ldim, bias=False)

    def forward(self, c: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del s, t  # unused in this fake -- the real flow_net uses them, this one doesn't need to
        return self.from_cond(c) + self.from_x(x)


class _FakeFlowLMForLoss(nn.Module):
    """A small real nn.Module standing in for pocket-tts's FlowLMModel.

    Matches the real class's public surface that `_compute_flow_matching_loss`
    depends on: `.dim`, `.ldim`, `.input_linear`, `.backbone()`, `.flow_net`.
    No `StatefulModule` submodules, so `init_states()` (the real pocket-tts
    helper, not mocked) returns an empty dict for it -- fine, since this
    fake `backbone()` doesn't need cache state to produce a shape-correct
    result.
    """

    def __init__(self, ldim: int = 4, dim: int = 8) -> None:
        super().__init__()
        self.ldim = ldim
        self.dim = dim
        self.input_linear = nn.Linear(ldim, dim, bias=False)
        self._backbone_proj = nn.Linear(dim, dim, bias=False)
        self.flow_net = _FakeFlowNet(ldim, dim)

    def backbone(self, input_, text_embeddings, sequence, model_state):
        del text_embeddings, sequence, model_state  # unused in this fake
        return self._backbone_proj(input_)


class _FakeMimiForLoss:
    """Stands in for pocket-tts's MimiModel.encode_to_latent: [B,1,T] -> [B,ldim,S]."""

    def __init__(self, ldim: int, frame_size: int = 4) -> None:
        self._conv = nn.Conv1d(1, ldim, kernel_size=frame_size, stride=frame_size, bias=False)

    def encode_to_latent(self, waveform: torch.Tensor) -> torch.Tensor:
        return self._conv(waveform)


class _FakeBaseModelForLoss:
    def __init__(self, ldim: int = 4, dim: int = 8) -> None:
        self.flow_lm = _FakeFlowLMForLoss(ldim, dim)
        self.mimi = _FakeMimiForLoss(ldim)


def test_compute_flow_matching_loss_returns_finite_scalar_and_backprops() -> None:
    """Real math, real backward pass, fake (small, undownloaded) pocket-tts model.

    Exercises exactly the real call path `_compute_flow_matching_loss` uses
    against real pocket-tts weights (mimi.encode_to_latent -> input_linear
    -> backbone -> flow_net -> MSE against x1 - x0), just against a tiny
    fake model instead of multi-GB downloaded weights, so this runs in CI.
    """
    base_model = _FakeBaseModelForLoss(ldim=4, dim=8)
    config = TrainingConfig(voice_clips_dir=Path("unused"))
    waveform = torch.randn(64)  # 1D, matching load_waveform()'s real return shape

    loss = _compute_flow_matching_loss(peft_model=object(), base_model=base_model, waveform=waveform, config=config)

    assert loss.dim() == 0
    assert torch.isfinite(loss).item()

    loss.backward()
    trained_params = [base_model.flow_lm._backbone_proj.weight, base_model.flow_lm.flow_net.from_cond.weight]
    for param in trained_params:
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_compute_flow_matching_loss_ignores_peft_model_argument() -> None:
    """`peft_model` is accepted for signature clarity only -- base_model.flow_lm is what's used.

    Passing something with no attributes at all must not matter, since PEFT's
    in-place injection already mutated base_model.flow_lm before this is
    called (see the function's docstring).
    """
    base_model = _FakeBaseModelForLoss()
    config = TrainingConfig(voice_clips_dir=Path("unused"))
    waveform = torch.randn(64)

    loss = _compute_flow_matching_loss(
        peft_model="not-a-model-at-all", base_model=base_model, waveform=waveform, config=config
    )

    assert torch.isfinite(loss).item()
