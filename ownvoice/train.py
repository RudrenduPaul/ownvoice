"""LoRA adapter training for pocket-tts: injection, training loop, save.

Architecture note (locked by /plan-[redacted], see design doc): single-
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
CPU, with no errors. The deeper training/generation plumbing -- the
flow-matching loss computation in `_compute_flow_matching_loss()`, and
`ownvoice/infer.py`'s `generate_speech()` -- has since been verified for
real too: a real 2-epoch LoRA training run against the loaded weights
produced a finite, non-NaN loss, and the resulting adapter produced real,
non-silent generated audio via `ownvoice infer`. Two genuine, real findings
came out of that validation (see the docstrings on
`_compute_flow_matching_loss()` and `generate_speech()` for the full
detail): `FlowLMModel.forward()`'s "compute the loss" branch doesn't
actually exist in the published inference-only package despite its own
docstring, so the loss is computed directly from `flow_lm`'s real
submodules instead; and swapping `base_model.flow_lm` to the PEFT wrapper
for generation is not just unnecessary but actively wrong (it breaks
pocket-tts's internal KV-cache state lookup) -- no swap is needed at all,
since PEFT's injection already mutates `base_model.flow_lm` in place.
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
    print it for debugging instead of a raw stack trace, per the locked
    [redacted] decision.
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
    of a raw stack trace (locked [redacted] decision #4).
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

    This is the free Day-0-style validation `ownvoice check` runs (the
    "Champion-tier" DX moment from the design doc's DX review) -- it never
    touches a GPU and never runs a training step.
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

    Verified against real loaded pocket-tts weights (kyutai's published
    `pocket-tts` PyPI package, v2.1.0): `FlowLMModel.forward()`
    (pocket_tts/models/flow_lm.py) does NOT actually support a
    `lsd_decode_steps=0` "return the loss" branch the way its own
    docstring describes -- that docstring is a leftover from kyutai's
    internal (non-public) training codebase. The installed forward() has a
    hard `assert lsd_decode_steps > 0` immediately before its return
    statement, so calling it with `lsd_decode_steps=0` raises an
    AssertionError, not a loss tensor, and nowhere else in the published,
    inference-only package computes this loss either.

    Given that, this function computes a real, structurally-correct
    conditional flow-matching (rectified-flow) loss directly from
    `FlowLMModel`'s real, live submodules instead of going through the
    stripped `forward()`:

    1. Encode the real waveform to latents via the model's own frozen
       Mimi codec (`base_model.mimi.encode_to_latent`), giving the target
       latent sequence `x_1` in the `[B, S, ldim]` shape `forward()`'s
       docstring describes for `sequence`.
    2. Run the same real transformer conditioning pass `forward()` would
       have run (`input_linear` then `backbone()`), teacher-forced over the
       whole sequence in one shot. No text conditioning is used --
       OwnVoice's training clips are audio-only (see data.py), so
       `text_embeddings` is an empty `[B, 0, dim]` tensor, exactly like
       `TTSModel._run_flow_lm_and_increment_step`'s own default when no
       text is given.
    3. Sample Gaussian noise `x_0` and regress `flow_net`'s predicted
       velocity against the real target velocity `x_1 - x_0`, using
       `s=0, t=1` -- the exact same one-shot jump `lsd_decode()` uses at
       generation time for this checkpoint (the published weights load
       with `lsd_decode_steps=1`, i.e. `lsd_decode(..., num_steps=1)`
       resolves to a single `s=0 -> t=1` jump). This is the real
       generation math run in reverse with a known target, not an
       invented objective.

    `peft_model` is accepted for signature clarity (it is the object the
    caller trained), but is not used directly here: PEFT's
    `get_peft_model()` replaces `base_model.flow_lm`'s targeted
    `nn.Linear` submodules with LoRA layers in place, on the same live
    module graph -- `base_model.flow_lm` already reflects the LoRA
    injection by the time this function runs, confirmed against real
    weights this session (see also `generate_speech()` in infer.py, which
    depends on this same fact).
    """
    import torch
    from pocket_tts.modules.stateful_module import init_states

    flow_lm = base_model.flow_lm

    with torch.no_grad():
        latent = base_model.mimi.encode_to_latent(waveform.reshape(1, 1, -1))
    sequence = latent.transpose(-1, -2).to(torch.float32)  # [1, S, ldim]

    text_embeddings = torch.zeros(
        (1, 0, flow_lm.dim), dtype=sequence.dtype, device=sequence.device
    )
    model_state = init_states(flow_lm, batch_size=1, sequence_length=sequence.shape[1])

    input_ = flow_lm.input_linear(sequence)
    conditioning = flow_lm.backbone(input_, text_embeddings, sequence, model_state=model_state)

    batch, steps, ldim = sequence.shape
    x1 = sequence.reshape(batch * steps, ldim)
    c = conditioning.reshape(batch * steps, flow_lm.dim)
    x0 = torch.randn_like(x1)
    start_time = torch.zeros((batch * steps, 1), dtype=x1.dtype, device=x1.device)
    target_time = torch.ones((batch * steps, 1), dtype=x1.dtype, device=x1.device)

    predicted_velocity = flow_lm.flow_net(c, start_time, target_time, x0)
    target_velocity = x1 - x0
    return torch.nn.functional.mse_loss(predicted_velocity, target_velocity)


def run_training_loop(
    peft_model: Any,
    base_model: Any,
    clips: list[VoiceClip],
    config: TrainingConfig,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the LoRA fine-tuning loop against the loaded voice clips.

    Integration/manual-only per the [redacted]: this needs the
    real pocket-tts base model and (for a real run) a rented GPU, so it
    can't be meaningfully unit-tested in CI. `ownvoice train` calls this for
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
    anyway (locked [redacted] decision #3), not a durable-infrastructure
    investment: training config, the similarity score, and a timestamp.

    When `reference_clip` is given, the clip file itself is copied into
    `out_dir` alongside the adapter, not just referenced by its original
    path. Without this, the `ownvoice infer` command this function's caller
    prints on success fails on copy-paste whenever `--voice-clips` isn't
    nested inside `--out` (the common case), because
    `infer.resolve_reference_audio()` deliberately only trusts a
    `reference_clip` path that resolves inside the adapter's own directory.
    Copying the file in makes the adapter directory self-contained and
    satisfies that check by construction instead of by coincidence.
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

    reference_clip_name: str | None = None
    if reference_clip is not None:
        import shutil

        reference_clip = Path(reference_clip)
        reference_clip_name = reference_clip.name
        shutil.copyfile(reference_clip, out_dir / reference_clip_name)

    resolved_timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    metadata = {
        "ownvoice_version": __version__,
        "config": config.to_dict() if hasattr(config, "to_dict") else dict(config),
        "similarity_score": similarity_score,
        "usable": is_usable(similarity_score) if similarity_score is not None else None,
        "usable_threshold": USABLE_THRESHOLD,
        "metrics": metrics or {},
        "timestamp": resolved_timestamp,
        "reference_clip": reference_clip_name,
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return out_dir


def _evaluate_adapter(base_model: Any, peft_model: Any, reference_clip: VoiceClip, config: TrainingConfig) -> float:
    """Generate one eval utterance from the freshly trained adapter and score it.

    Uses the same real `generate_speech()` plumbing as ownvoice/infer.py --
    verified end to end against real loaded pocket-tts weights this
    session (a real 2-epoch training run produced a finite similarity
    score from this function without error). See generate_speech()'s
    docstring for what the real call path actually requires.
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
    PeftInjectionError distinctly from other failures per the locked
    [redacted] decision), runs the training loop, generates one eval
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
