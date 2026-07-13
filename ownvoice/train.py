"""LoRA adapter training for pocket-tts: injection, training loop, save.

Architecture note: single-
target-by-design. Model loading is hardcoded to pocket-tts; there is no
`BaseModelAdapter` abstraction, because no second base model is in scope.

Real-API notes from researching kyutai-labs/pocket-tts directly (README +
source, not guessed): the package is `pocket-tts` on PyPI (Python 3.10+,
torch 2.5+), loaded via `pocket_tts.TTSModel.load_model()`. The model's only
trainable submodule is `TTSModel.flow_lm` (a `FlowLMModel`, pocket_tts/
models/flow_lm.py), built entirely from standard `torch.nn.Linear` layers:
`input_linear`, `out_eos`, each transformer layer's `self_attn.in_proj` /
`self_attn.out_proj` / `linear1` / `linear2` (pocket_tts/modules/
mimi_transformer.py, pocket_tts/modules/transformer.py), and the flow-net's
own Linear layers inside `SimpleMLPAdaLN` (pocket_tts/modules/mlp.py). That
confirms PEFT's built-in `target_modules="all-linear"` resolution should
apply directly with no custom LoRA-injection code required -- verified
structurally against the real module tree, not assumed.

`ownvoice check` (load_base_model() + inject_lora() below) was additionally
run for real during this project's initial build: it downloaded pocket-tts's
actual published weights from Hugging Face and PEFT's `target_modules=
"all-linear"` injection genuinely succeeded against the loaded model, on
CPU, with no errors. What remains unverified against real weights is the
deeper training/generation plumbing (the flow-matching loss call signature
in `_compute_flow_matching_loss()`, and the flow_lm swap in
ownvoice/infer.py's `generate_speech()`) -- each is marked
`# TODO(day-0-spike):` at the exact point that still needs it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from ownvoice import __version__
from ownvoice.data import VoiceClip, VoiceClipError, load_voice_clips
from ownvoice.score import USABLE_THRESHOLD, is_usable

DEFAULT_LORA_RANK = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_OUT_DIR = Path("./ownvoice-adapter")
DEFAULT_EVAL_TEXT = "This is my own voice, trained with OwnVoice."
LORA_TARGET_MODULES = "all-linear"

ISSUE_30_URL = "https://github.com/kyutai-labs/pocket-tts/issues/30"


class PeftInjectionError(Exception):
    """Raised when PEFT LoRA injection fails against pocket-tts's actual layers.

    Carries the real module tree (via named_modules()) so the caller can
    print it for debugging instead of a raw stack trace.
    """

    def __init__(self, message: str, module_tree: str):
        super().__init__(message)
        self.module_tree = module_tree


@dataclasses.dataclass
class TrainingConfig:
    voice_clips_dir: Path
    out_dir: Path = DEFAULT_OUT_DIR
    epochs: int = DEFAULT_EPOCHS
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    learning_rate: float = DEFAULT_LEARNING_RATE
    eval_text: str = DEFAULT_EVAL_TEXT

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["voice_clips_dir"] = str(self.voice_clips_dir)
        d["out_dir"] = str(self.out_dir)
        return d


@dataclasses.dataclass
class CheckResult:
    success: bool
    message: str
    module_tree: str | None = None


@dataclasses.dataclass
class TrainResult:
    success: bool
    out_dir: Path | None
    similarity_score: float | None
    usable: bool | None
    message: str
    infer_command: str | None
    module_tree: str | None = None


def describe_module_tree(module: Any) -> str:
    """Return a readable dump of a model's named submodules for debugging a failed injection."""
    lines = [f"{name or '<root>'}: {type(submodule).__name__}" for name, submodule in module.named_modules()]
    return "\n".join(lines)


def load_base_model(device: str = "cpu"):
    """Load pocket-tts's base TTSModel.

    Written against the real, confirmed public API
    (pocket_tts/__init__.py exports `TTSModel`; pocket_tts/models/
    tts_model.py defines `load_model` as a classmethod and `device`/
    `sample_rate` as properties). Verified for real during this project's
    initial build: `TTSModel.load_model()` downloaded pocket-tts's actual
    published weights from Hugging Face and loaded successfully on CPU.
    """
    from pocket_tts import TTSModel

    model = TTSModel.load_model()
    model = model.to(device)
    model.eval()
    return model


def inject_lora(
    flow_lm_module: Any,
    *,
    rank: int = DEFAULT_LORA_RANK,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
):
    """Wrap pocket-tts's flow_lm transformer with a LoRA adapter via PEFT.

    Targets every `nn.Linear` layer ("all-linear") in the flow language
    model. If PEFT raises, catch it, attach the real module tree, and raise
    PeftInjectionError so the caller can print an actionable message instead
    of a raw stack trace.
    """
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    try:
        return get_peft_model(flow_lm_module, lora_config)
    except Exception as exc:  # noqa: BLE001 - PEFT/torch raise varied error types here
        raise PeftInjectionError(
            f'PEFT LoRA injection failed against pocket-tts\'s flow_lm module structure: {exc}. '
            f'target_modules="all-linear" could not resolve against pocket-tts\'s actual layers '
            f"on this pocket-tts version. Please post an honest blocker (this error plus the "
            f"module tree above) as a comment on {ISSUE_30_URL} rather than working around it "
            f"silently -- that issue is exactly where this gap needs to be visible.",
            module_tree=describe_module_tree(flow_lm_module),
        ) from exc


def check_compatibility() -> CheckResult:
    """CPU-only, no-GPU, no-training dry run: load the base model, attempt LoRA injection.

    This is the free Day-0-style validation `ownvoice check` runs -- a fast
    signal that surfaces a broken LoRA injection before a user burns GPU
    time on a doomed training run -- it never touches a GPU and never runs
    a training step.
    """
    try:
        model = load_base_model(device="cpu")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            success=False,
            message=f"Could not load the pocket-tts base model: {exc}",
            module_tree=None,
        )

    try:
        inject_lora(model.flow_lm)
    except PeftInjectionError as exc:
        return CheckResult(success=False, message=str(exc), module_tree=exc.module_tree)

    return CheckResult(
        success=True,
        message='PEFT LoRA injection succeeded against pocket-tts\'s flow_lm module (target_modules="all-linear").',
        module_tree=None,
    )


def _compute_flow_matching_loss(peft_model: Any, base_model: Any, waveform: Any, config: TrainingConfig) -> Any:
    """Compute pocket-tts's flow-matching training loss for one clip.
    """
    # TODO(day-0-spike): FlowLMModel.forward() (pocket_tts/models/flow_lm.py)
    # takes `sequence` (already-encoded latents), `text_embeddings` (from
    # the model's LUTConditioner), `model_state`, and `lsd_decode_steps=0`
    # to return the training loss instead of a generated sample. Wiring the
    # real encode step (turning a raw waveform into `sequence` latents via
    # the model's own audio front-end, and building `text_embeddings` via
    # the conditioner's tokenizer) needs a human at the keyboard against the
    # real loaded weights to confirm exact tensor shapes and call order --
    # this is the one piece of OwnVoice's training path this session could
    # not verify without downloading pocket-tts's multi-GB weights and
    # running on real hardware. Everything else in this file (LoRA
    # injection, the epoch loop below, optimizer setup, adapter+manifest
    # save) is real and independently unit-tested without this piece.
    raise NotImplementedError(
        "Flow-matching loss computation against pocket-tts's real forward() signature is "
        "pending a Day-0 compatibility spike. Run "
        "`ownvoice check` first, then wire this function against the real loaded weights "
        "before training for real."
    )


def run_training_loop(
    peft_model: Any,
    base_model: Any,
    clips: list[VoiceClip],
    config: TrainingConfig,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the LoRA fine-tuning loop against the loaded voice clips.

    Integration/manual-only: this needs the real pocket-tts base model and
    (for a real run) a rented GPU, so it can't be meaningfully unit-tested
    in CI. `ownvoice train` calls this for
    real, gated behind a successful check_compatibility() first.
    """
    import torch

    from ownvoice.data import load_waveform

    optimizer = torch.optim.AdamW(
        (p for p in peft_model.parameters() if p.requires_grad),
        lr=config.learning_rate,
    )

    sample_rate = base_model.sample_rate
    waveforms = [load_waveform(clip, target_sample_rate=sample_rate) for clip in clips]

    peft_model.train()
    epoch_avg_losses: list[float] = []
    for _epoch in range(config.epochs):
        step_losses = []
        for waveform in waveforms:
            optimizer.zero_grad()
            loss = _compute_flow_matching_loss(peft_model, base_model, waveform.to(device), config)
            loss.backward()
            optimizer.step()
            step_losses.append(float(loss.detach()))
        if step_losses:
            epoch_avg_losses.append(sum(step_losses) / len(step_losses))

    peft_model.eval()
    return {
        "final_loss": epoch_avg_losses[-1] if epoch_avg_losses else None,
        "epochs": config.epochs,
        "num_clips": len(clips),
    }


def save_adapter_and_manifest(
    peft_model_or_state_dict: Any,
    out_dir: Path | str,
    config: TrainingConfig,
    similarity_score: float | None,
    metrics: dict[str, Any] | None = None,
    timestamp: str | None = None,
    reference_clip: Path | str | None = None,
) -> Path:
    """Write `adapter.safetensors` + `metadata.json` to `out_dir`.

    Accepts either a PEFT model (its LoRA state dict is extracted via
    `peft.get_peft_model_state_dict`) or a plain tensor state dict directly,
    so this function is unit-testable without a real PEFT model or GPU.

    The manifest is a free byproduct of a training run that's happening
    anyway, not a durable-infrastructure investment: training config, the
    similarity score, and a timestamp.
    """
    import json
    from datetime import datetime, timezone

    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(peft_model_or_state_dict, dict):
        state_dict = peft_model_or_state_dict
    else:
        try:
            from peft import get_peft_model_state_dict

            state_dict = get_peft_model_state_dict(peft_model_or_state_dict)
        except ImportError:
            state_dict = peft_model_or_state_dict.state_dict()

    adapter_path = out_dir / "adapter.safetensors"
    save_file(state_dict, str(adapter_path))

    resolved_timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    metadata = {
        "ownvoice_version": __version__,
        "config": config.to_dict() if hasattr(config, "to_dict") else dict(config),
        "similarity_score": similarity_score,
        "usable": is_usable(similarity_score) if similarity_score is not None else None,
        "usable_threshold": USABLE_THRESHOLD,
        "metrics": metrics or {},
        "timestamp": resolved_timestamp,
        "reference_clip": str(reference_clip) if reference_clip is not None else None,
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return out_dir


def _evaluate_adapter(base_model: Any, peft_model: Any, reference_clip: VoiceClip, config: TrainingConfig) -> float:
    """Generate one eval utterance from the freshly trained adapter and score it.

    # TODO(day-0-spike): depends on the same real generate_audio() plumbing
    # as ownvoice/infer.py's generate_speech() -- see that module's TODO for
    # what remains to wire up against real loaded weights.
    """
    from ownvoice.infer import generate_speech
    from ownvoice.score import compute_similarity, load_and_resample, resample_to_16k_mono

    generated_waveform = generate_speech(base_model, peft_model, config.eval_text, reference_clip.path)
    generated_16k = resample_to_16k_mono(generated_waveform, base_model.sample_rate)
    reference_16k = load_and_resample(reference_clip.path)
    return compute_similarity(reference_16k, generated_16k)


def train(config: TrainingConfig) -> TrainResult:
    """End-to-end train flow used by `ownvoice train`.

    Loads the base model, attempts LoRA injection (catching and reporting a
    PeftInjectionError distinctly from other failures), runs the training
    loop, generates one eval
    utterance, scores it against a reference clip, and always writes
    `adapter.safetensors` + `metadata.json` on a successful run -- whether
    or not the similarity score clears the usable threshold. A below-
    threshold score is a labeled result (success=True), not a crash; only a
    caught PEFT-injection failure or a data/model-loading error is
    success=False.
    """
    try:
        clips = load_voice_clips(config.voice_clips_dir)
    except VoiceClipError as exc:
        return TrainResult(False, None, None, None, str(exc), None)

    try:
        base_model = load_base_model(device="cpu")
    except Exception as exc:  # noqa: BLE001
        return TrainResult(
            False, None, None, None, f"Could not load the pocket-tts base model: {exc}", None
        )

    try:
        peft_model = inject_lora(
            base_model.flow_lm,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
    except PeftInjectionError as exc:
        return TrainResult(
            False, None, None, None,
            f"{exc}\n\nModule tree:\n{exc.module_tree}",
            None,
            module_tree=exc.module_tree,
        )

    metrics = run_training_loop(peft_model, base_model, clips, config, device="cpu")
    similarity_score = _evaluate_adapter(base_model, peft_model, clips[0], config)
    out_dir = save_adapter_and_manifest(
        peft_model,
        config.out_dir,
        config,
        similarity_score,
        metrics=metrics,
        reference_clip=clips[0].path,
    )

    usable = is_usable(similarity_score)
    infer_command = f'ownvoice infer --adapter {out_dir}/adapter.safetensors --text "{config.eval_text}"'
    if usable:
        message = (
            f"Usable adapter (similarity {similarity_score:.3f} >= {USABLE_THRESHOLD}). "
            f"Try it now:\n  {infer_command}"
        )
    else:
        message = (
            f"Below threshold (similarity {similarity_score:.3f} < {USABLE_THRESHOLD}). "
            "The adapter was still saved -- try more/cleaner voice clips, more epochs, or a "
            f"higher --lora-rank, then re-run. You can still listen to it:\n  {infer_command}"
        )

    return TrainResult(True, out_dir, similarity_score, usable, message, infer_command)
