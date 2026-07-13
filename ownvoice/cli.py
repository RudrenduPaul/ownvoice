"""OwnVoice CLI: `check`, `train`, `infer` subcommands.

Every subcommand supports `--json` for a structured, machine-parseable
output mode alongside the default human-readable text, so a script or an
agent invoking `ownvoice` programmatically can parse the result reliably.
"""

from __future__ import annotations

import json as json_module
from pathlib import Path
from typing import Optional

import typer

from ownvoice import __version__
from ownvoice.infer import infer as run_infer
from ownvoice.train import (
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_TEXT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_OUT_DIR,
    TrainingConfig,
    check_compatibility,
)
from ownvoice.train import train as run_train

app = typer.Typer(
    name="ownvoice",
    help="Train a LoRA voice adapter for pocket-tts and own the resulting model. No API lock-in.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"ownvoice {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the OwnVoice version and exit.",
    ),
) -> None:
    """OwnVoice: own your voice model, no API lock-in."""


@app.command()
def check(
    json_output: bool = typer.Option(
        False, "--json", help="Print machine-readable JSON instead of human-readable text."
    ),
) -> None:
    """Free, CPU-only compatibility check: load pocket-tts and dry-run the LoRA injection.

    No GPU and no training required. Run this before renting a GPU for
    `ownvoice train` -- it is the Day-0 compatibility spike, exposed as a
    command.

    Example:
        ownvoice check
    """
    result = check_compatibility()

    if json_output:
        typer.echo(
            json_module.dumps(
                {
                    "success": result.success,
                    "message": result.message,
                    "module_tree": result.module_tree,
                }
            )
        )
    else:
        if result.success:
            typer.secho(f"[ownvoice check] PASS: {result.message}", fg=typer.colors.GREEN)
        else:
            typer.secho(f"[ownvoice check] FAIL: {result.message}", fg=typer.colors.RED)
            if result.module_tree:
                typer.echo("\nModule tree (for debugging / for the issue #30 blocker post):")
                typer.echo(result.module_tree)

    raise typer.Exit(code=0 if result.success else 1)


@app.command()
def train(
    voice_clips: Path = typer.Option(
        ...,
        "--voice-clips",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory of .wav voice-clip recordings to train from. Required.",
    ),
    out: Path = typer.Option(
        DEFAULT_OUT_DIR, "--out", help="Directory to write adapter.safetensors + metadata.json to."
    ),
    epochs: int = typer.Option(DEFAULT_EPOCHS, "--epochs", min=1, help="Number of training epochs."),
    lora_rank: int = typer.Option(DEFAULT_LORA_RANK, "--lora-rank", min=1, help="LoRA rank."),
    lora_alpha: int = typer.Option(DEFAULT_LORA_ALPHA, "--lora-alpha", min=1, help="LoRA alpha."),
    lora_dropout: float = typer.Option(
        DEFAULT_LORA_DROPOUT, "--lora-dropout", min=0.0, max=1.0, help="LoRA dropout."
    ),
    learning_rate: float = typer.Option(
        DEFAULT_LEARNING_RATE, "--learning-rate", help="Optimizer learning rate."
    ),
    eval_text: str = typer.Option(
        DEFAULT_EVAL_TEXT,
        "--eval-text",
        help="Sentence synthesized after training to score against the reference voice.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print machine-readable JSON instead of human-readable text."
    ),
) -> None:
    """Train a LoRA voice adapter from a directory of .wav voice clips.

    Only --voice-clips is required; every other flag has a sensible default.
    Exits 0 on a successful run whether or not the similarity score clears
    the usable threshold (0.75) -- a below-threshold result is a labeled
    outcome with a next step, not a crash. Only a data-loading failure or a
    caught PEFT-injection failure exits non-zero.

    Example:
        ownvoice train --voice-clips ./my-voice-clips
    """
    config = TrainingConfig(
        voice_clips_dir=voice_clips,
        out_dir=out,
        epochs=epochs,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        learning_rate=learning_rate,
        eval_text=eval_text,
    )
    result = run_train(config)

    if json_output:
        typer.echo(
            json_module.dumps(
                {
                    "success": result.success,
                    "out_dir": str(result.out_dir) if result.out_dir else None,
                    "similarity_score": result.similarity_score,
                    "usable": result.usable,
                    "message": result.message,
                    "infer_command": result.infer_command,
                }
            )
        )
    else:
        if result.success:
            label = "USABLE ADAPTER" if result.usable else "BELOW THRESHOLD"
            color = typer.colors.GREEN if result.usable else typer.colors.YELLOW
            typer.secho(f"[ownvoice train] {label}", fg=color, bold=True)
            typer.echo(result.message)
        else:
            typer.secho("[ownvoice train] FAILED", fg=typer.colors.RED, bold=True)
            typer.echo(result.message)

    raise typer.Exit(code=0 if result.success else 1)


@app.command()
def infer(
    adapter: Path = typer.Option(
        ..., "--adapter", exists=True, help="Path to a trained adapter.safetensors file. Required."
    ),
    text: str = typer.Option(..., "--text", help="Text to synthesize in the trained voice. Required."),
    out: Path = typer.Option(Path("./ownvoice-output.wav"), "--out", help="Output .wav file path."),
    reference_audio: Optional[Path] = typer.Option(
        None,
        "--reference-audio",
        help="Override the reference clip OwnVoice recorded in metadata.json at train time.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print machine-readable JSON instead of human-readable text."
    ),
) -> None:
    """Generate speech in the trained voice from a saved adapter, and save it to a .wav file.

    Example:
        ownvoice infer --adapter ./ownvoice-adapter/adapter.safetensors --text "Hello, this is my own voice."
    """
    try:
        out_path = run_infer(adapter, text, out_path=out, reference_audio_override=reference_audio)
    except Exception as exc:  # noqa: BLE001
        if json_output:
            typer.echo(json_module.dumps({"success": False, "error": str(exc)}))
        else:
            typer.secho(f"[ownvoice infer] FAILED: {exc}", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(json_module.dumps({"success": True, "out_path": str(out_path)}))
    else:
        typer.secho(f"[ownvoice infer] Wrote {out_path}", fg=typer.colors.GREEN)

    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
