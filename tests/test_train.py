"""Unit tests for ownvoice/train.py.

Real LoRA injection against pocket-tts's actual weights, and the real
training loop, are integration/manual-only per the [redacted]
(they need the real, multi-GB pocket-tts model and, for training, a GPU).
What's unit-tested here: the caught-PEFT-exception path (mocked), the
compatibility-check orchestration (mocked), and the adapter+manifest
save/load round trip (real safetensors + json, no model needed).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from ownvoice.train import (
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

    result_dir = save_adapter_and_manifest(
        state_dict,
        config.out_dir,
        config,
        similarity_score=0.82,
        metrics={"final_loss": 0.1, "epochs": 3},
        timestamp="2026-07-12T00:00:00+00:00",
        reference_clip=tmp_path / "clips" / "clip_a.wav",
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
