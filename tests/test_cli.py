"""Unit tests for ownvoice/cli.py: argument parsing for check, train, infer.

These tests never load pocket-tts, PEFT, or run any training/inference --
the underlying implementation functions (check_compatibility, train, infer)
are monkeypatched at the ownvoice.cli module boundary so only argument
parsing, exit codes, and output formatting are under test here.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ownvoice.cli import app
from ownvoice.train import CheckResult, TrainResult

runner = CliRunner()


# ---- ownvoice check --------------------------------------------------------


def test_check_command_pass(monkeypatch) -> None:
    import ownvoice.cli as cli_module

    monkeypatch.setattr(
        cli_module, "check_compatibility", lambda: CheckResult(True, "all good", None)
    )

    result = runner.invoke(app, ["check"])

    assert result.exit_code == 0
    assert "PASS" in result.stdout


def test_check_command_fail_prints_module_tree(monkeypatch) -> None:
    import ownvoice.cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "check_compatibility",
        lambda: CheckResult(False, "injection failed", "root: TTSModel\nflow_lm: FlowLMModel"),
    )

    result = runner.invoke(app, ["check"])

    assert result.exit_code == 1
    assert "FAIL" in result.stdout
    assert "flow_lm: FlowLMModel" in result.stdout


def test_check_command_json_output(monkeypatch) -> None:
    import ownvoice.cli as cli_module

    monkeypatch.setattr(cli_module, "check_compatibility", lambda: CheckResult(True, "ok", None))

    result = runner.invoke(app, ["check", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {"success": True, "message": "ok", "module_tree": None}


# ---- ownvoice train ---------------------------------------------------------


def test_train_command_requires_voice_clips_flag() -> None:
    result = runner.invoke(app, ["train"])
    assert result.exit_code != 0


def test_train_command_rejects_nonexistent_voice_clips_dir(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    result = runner.invoke(app, ["train", "--voice-clips", str(missing)])
    assert result.exit_code != 0


def test_train_command_defaults_applied(monkeypatch, tmp_path: Path) -> None:
    import ownvoice.cli as cli_module
    from ownvoice.train import DEFAULT_EPOCHS, DEFAULT_LORA_RANK, DEFAULT_OUT_DIR

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()

    captured = {}

    def fake_train(config):
        captured["config"] = config
        return TrainResult(True, config.out_dir, 0.9, True, "usable", "ownvoice infer ...")

    monkeypatch.setattr(cli_module, "run_train", fake_train)

    result = runner.invoke(app, ["train", "--voice-clips", str(clips_dir)])

    assert result.exit_code == 0
    assert "USABLE ADAPTER" in result.stdout
    config = captured["config"]
    assert config.voice_clips_dir == clips_dir
    assert config.epochs == DEFAULT_EPOCHS
    assert config.lora_rank == DEFAULT_LORA_RANK
    assert config.out_dir == DEFAULT_OUT_DIR


def test_train_command_overrides_hyperparameters(monkeypatch, tmp_path: Path) -> None:
    import ownvoice.cli as cli_module

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    out_dir = tmp_path / "custom-out"

    captured = {}

    def fake_train(config):
        captured["config"] = config
        return TrainResult(True, config.out_dir, 0.9, True, "usable", "ownvoice infer ...")

    monkeypatch.setattr(cli_module, "run_train", fake_train)

    result = runner.invoke(
        app,
        [
            "train",
            "--voice-clips", str(clips_dir),
            "--out", str(out_dir),
            "--epochs", "25",
            "--lora-rank", "16",
        ],
    )

    assert result.exit_code == 0
    config = captured["config"]
    assert config.out_dir == out_dir
    assert config.epochs == 25
    assert config.lora_rank == 16


def test_train_command_below_threshold_still_exits_zero(monkeypatch, tmp_path: Path) -> None:
    import ownvoice.cli as cli_module

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()

    monkeypatch.setattr(
        cli_module,
        "run_train",
        lambda config: TrainResult(True, config.out_dir, 0.4, False, "below threshold", "ownvoice infer ..."),
    )

    result = runner.invoke(app, ["train", "--voice-clips", str(clips_dir)])

    assert result.exit_code == 0
    assert "BELOW THRESHOLD" in result.stdout


def test_train_command_failure_exits_nonzero(monkeypatch, tmp_path: Path) -> None:
    import ownvoice.cli as cli_module

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()

    monkeypatch.setattr(
        cli_module,
        "run_train",
        lambda config: TrainResult(False, None, None, None, "PEFT injection failed", None),
    )

    result = runner.invoke(app, ["train", "--voice-clips", str(clips_dir)])

    assert result.exit_code == 1
    assert "FAILED" in result.stdout


def test_train_command_json_output(monkeypatch, tmp_path: Path) -> None:
    import ownvoice.cli as cli_module

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()

    monkeypatch.setattr(
        cli_module,
        "run_train",
        lambda config: TrainResult(True, config.out_dir, 0.9, True, "usable", "infer cmd"),
    )

    result = runner.invoke(app, ["train", "--voice-clips", str(clips_dir), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["usable"] is True
    assert payload["similarity_score"] == 0.9


# ---- ownvoice infer -----------------------------------------------------------


def test_infer_command_requires_adapter_and_text() -> None:
    result = runner.invoke(app, ["infer"])
    assert result.exit_code != 0


def test_infer_command_rejects_nonexistent_adapter(tmp_path: Path) -> None:
    missing = tmp_path / "no-adapter.safetensors"
    result = runner.invoke(app, ["infer", "--adapter", str(missing), "--text", "hello"])
    assert result.exit_code != 0


def test_infer_command_success(monkeypatch, tmp_path: Path) -> None:
    import ownvoice.cli as cli_module

    adapter_path = tmp_path / "adapter.safetensors"
    adapter_path.write_bytes(b"fake adapter bytes")
    out_wav = tmp_path / "out.wav"

    monkeypatch.setattr(cli_module, "run_infer", lambda *a, **k: out_wav)

    result = runner.invoke(app, ["infer", "--adapter", str(adapter_path), "--text", "hello there"])

    assert result.exit_code == 0
    assert "Wrote" in result.stdout


def test_infer_command_failure_exits_nonzero(monkeypatch, tmp_path: Path) -> None:
    import ownvoice.cli as cli_module

    adapter_path = tmp_path / "adapter.safetensors"
    adapter_path.write_bytes(b"fake adapter bytes")

    def _raise(*a, **k):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(cli_module, "run_infer", _raise)

    result = runner.invoke(app, ["infer", "--adapter", str(adapter_path), "--text", "hello"])

    assert result.exit_code == 1
    assert "FAILED" in result.stdout


def test_infer_command_json_output(monkeypatch, tmp_path: Path) -> None:
    import ownvoice.cli as cli_module

    adapter_path = tmp_path / "adapter.safetensors"
    adapter_path.write_bytes(b"fake adapter bytes")
    out_wav = tmp_path / "out.wav"

    monkeypatch.setattr(cli_module, "run_infer", lambda *a, **k: out_wav)

    result = runner.invoke(
        app, ["infer", "--adapter", str(adapter_path), "--text", "hello", "--json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["out_path"] == str(out_wav)


# ---- top-level ---------------------------------------------------------------


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "ownvoice" in result.stdout
