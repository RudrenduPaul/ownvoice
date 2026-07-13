# ownvoice

Train a LoRA voice adapter for [pocket-tts](https://github.com/kyutai-labs/pocket-tts) and keep the result on your own disk: a file you own outright, with your own weights, under your own control.

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Version](https://img.shields.io/badge/version-0.1.0-lightgrey.svg)](pyproject.toml)

There is no CI badge here yet. This repo does not run an automated test workflow on GitHub Actions yet, so a green badge here would not mean anything real. `pytest` passes locally; run it yourself before trusting that claim.

```bash
pip install git+https://github.com/RudrenduPaul/ownvoice
```

## Read this before you spend a week on this

The version of pocket-tts anyone can download without asking permission (`kyutai/pocket-tts-without-voice-cloning`, the checkpoint pocket-tts falls back to by default) ships with `has_voice_cloning = False`. With that flag set, pocket-tts's own `get_state_for_audio_prompt()` call refuses a raw reference-audio path or URL outright. We hit this directly, mid-build, while validating `ownvoice infer` end to end against real downloaded weights.

OwnVoice works around this: it pre-loads and resamples your reference clip itself and hands pocket-tts a tensor instead of a path, which is not gated. `ownvoice train` and `ownvoice infer` both work today, on the public weights, with no extra setup. But the ceiling on voice-cloning fidelity you get from those public weights is a property of the base model, not something OwnVoice's training code controls. Kyutai's best-quality cloning weights live in a separate, gated repo, [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts), that requires requesting access and agreeing to kyutai's usage terms on Hugging Face before you can download it. How long that approval takes is not publicly documented anywhere we could find; treat it as "not instant," not "never."

If you are deciding whether to invest real setup time in this tool, know that going in. That is why the numbers in the [comparison section](#how-ownvoice-compares) below are labeled the way they are, and why `ownvoice train` reports a measured similarity score on every run.

## What OwnVoice is and why it exists

pocket-tts is a genuinely good, MIT-licensed, CPU-capable local text-to-speech model from Kyutai. Its own maintainers have been clear, more than once, that fine-tuning code is not coming from them any time soon. On [pocket-tts issue #30](https://github.com/kyutai-labs/pocket-tts/issues/30), a Kyutai maintainer wrote that they were "not planning to release fine-tuning code for our TTS and STT models in the near future," a stance the maintainer noted "comes up repeatedly." Eighteen people reacted to that comment. At least one of them, working on a GLaDOS-style robot-voice project, went and hand-rolled a fine-tuning approach on their own rather than wait.

OwnVoice is a small, standalone CLI that builds the reusable version of that fine-tuning code. We intend to send the training script itself back to `kyutai-labs/pocket-tts` as an open pull request, sharing the technique publicly. It has no billing, no account, and no usage tracking. It is a training script, an inference script, and a scoring script, wired together behind three CLI commands.

Neither of them is a hypothetical competitor. [ElevenLabs](https://elevenlabs.io) is a mature, well-funded commercial voice API; it raised a $500M Series D in February 2026 at an $11B valuation. [Gradium](https://gradium.ai) is Kyutai's own commercial spinout, founded in September 2025 by the same researchers plus ex-Google engineers, and it raised a $70M seed round in December 2025, extended to $100M in July 2026 with Nvidia joining. Gradium in particular shares training lineage with the exact model OwnVoice is built on top of.

Both of those are cloud APIs: you send text, they send audio back, and the model weights never leave their infrastructure. What neither is structurally built to offer is a voice model you actually own, trained on your own machine or your own rented GPU, deployable offline or on-prem, with no API dependency and no vendor holding the weights. That is the specific, narrower lane OwnVoice occupies. If you want a polished, hosted, general-purpose voice API, ElevenLabs or Gradium are probably the right tool. If you need to own the model, keep it off the public internet, or hand a compliance team something they can actually sign off on, this is what we built.

## What OwnVoice is not

pocket-tts already ships zero-shot voice cloning out of the box, no fine-tuning required: pass a `.wav` file to its own `--voice` flag, or call `get_state_for_audio_prompt()` directly from Python, and it clones that voice from a single reference clip with no training step at all. If that is genuinely all you need, use pocket-tts directly. It is simpler and faster, and OwnVoice does not try to replace it.

OwnVoice exists for a narrower case: baking a voice permanently into trained LoRA weights, so generation stops depending on distributing or re-processing a reference audio clip at runtime, with (based on the training objective, not yet independently benchmarked at scale against pocket-tts's own zero-shot embeddings) more consistent output across many generations than a single-clip embedding tends to produce. That is the specific gap the 18 reactors on issue #30 were describing, and it is the only thing OwnVoice adds on top of what pocket-tts already does well. It does not add "voice cloning" to a model that already has it.

## Features

- **`ownvoice check`**: a free, CPU-only, no-GPU dry run that confirms PEFT's LoRA injection actually works against pocket-tts's real model structure, before you rent a GPU or record anything.
- **`ownvoice train`**: fine-tunes a LoRA adapter from a directory of your own `.wav` voice clips, with sensible defaults for every hyperparameter except the clip directory itself, and ends by printing the exact `ownvoice infer` command to try the result.
- **`ownvoice infer`**: generates speech from a trained adapter, or from plain pocket-tts if you pass no adapter at all.
- **A measured result on every training run**: a Resemblyzer cosine-similarity score between the adapter's output and your reference clips, saved into `metadata.json` alongside the adapter.
- **A below-threshold run exits clean.** `ownvoice train` exits `0` either way and tells you the concrete next step (more clips, more epochs, a higher `--lora-rank`).
- **A `--json` flag on every subcommand**, for a script or an agent invoking `ownvoice` programmatically to parse the result instead of scraping terminal text.
- **A caught, actionable PEFT-injection failure.** If `target_modules="all-linear"` ever fails to resolve against a future pocket-tts version's real layer structure, OwnVoice prints the model's actual module tree instead of a raw stack trace, and points you at posting it to issue #30.
- **Everything runs on your own machine.** No voice clip, adapter weight, or generated audio is sent anywhere by this CLI.

## Install

Requires Python 3.11 or newer.

```bash
pip install git+https://github.com/RudrenduPaul/ownvoice
```

There is no PyPI release yet; installing straight from the git repo is the only supported path for v0.1.

**Torch and CUDA:** `ownvoice check` needs no GPU at all and runs on CPU, matching pocket-tts's own CPU-capable design. Training a real adapter is much faster on an NVIDIA GPU. If you have one, install the CUDA build of PyTorch first by following [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), then install OwnVoice on top of it, so `pip` does not silently pull the CPU-only wheel instead. On Apple Silicon or a CPU-only machine, the default `pip install` of torch is fine: `ownvoice check` and `ownvoice infer` will run normally, and `ownvoice train` will just take longer per epoch.

## Quickstart

### 1. `ownvoice check`, the free Day-0 validation

Before recording anything or renting a GPU, confirm that PEFT's LoRA injection actually works against pocket-tts's real model structure. This is entirely free: CPU only, no training, no GPU.

```
$ ownvoice check
[ownvoice check] PASS: PEFT LoRA injection succeeded against pocket-tts's flow_lm module (target_modules="all-linear").
```

If it fails, OwnVoice prints the model's real module tree instead of a raw stack trace, so you can see exactly what did not match and report it precisely:

```
$ ownvoice check
[ownvoice check] FAIL: PEFT LoRA injection failed against pocket-tts's flow_lm module structure: <error detail>. Please post an honest blocker (this error plus the module tree above) as a comment on https://github.com/kyutai-labs/pocket-tts/issues/30 rather than working around it silently, that issue is exactly where this gap needs to be visible.

Module tree (for debugging / for the issue #30 blocker post):
<root>: FlowLMModel
input_linear: Linear
transformer: StreamingTransformer
transformer.layers.0.self_attn.in_proj: Linear
transformer.layers.0.self_attn.out_proj: Linear
...
```

### 2. `ownvoice train`

Record 5 to 10 minutes of clean audio of the voice you want to train (your own voice, with your own consent, see [Consent and misuse](#consent-and-misuse) below), split into a few `.wav` clips (each between 1 and 30 seconds) in one directory, then point OwnVoice at it:

```
$ ownvoice train --voice-clips ./my-voice-clips
[ownvoice train] USABLE ADAPTER
Usable adapter (similarity 0.812 >= 0.75). Try it now:
  ownvoice infer --adapter ownvoice-adapter/adapter.safetensors --text "This is my own voice, trained with OwnVoice."
```

Only `--voice-clips` is required. Every other flag has a sensible default (see the [CLI reference](#cli-reference) below).

A run that finishes but does not clear the similarity bar still exits `0`, with a labeled result and a concrete next step:

```
$ ownvoice train --voice-clips ./my-voice-clips
[ownvoice train] BELOW THRESHOLD
Below threshold (similarity 0.612 < 0.75). The adapter was still saved, try more/cleaner voice clips, more epochs, or a higher --lora-rank, then re-run. You can still listen to it:
  ownvoice infer --adapter ownvoice-adapter/adapter.safetensors --text "This is my own voice, trained with OwnVoice."
```

Only a data-loading problem (no usable clips) or a caught PEFT-injection failure exits non-zero. Every successful run writes `adapter.safetensors` and `metadata.json` (training config, the similarity score, a timestamp) to the output directory: two files you keep, with no server round-trip needed to use them again.

### 3. `ownvoice infer`

```
$ ownvoice infer --adapter ownvoice-adapter/adapter.safetensors --text "Hello, this is my own voice."
[ownvoice infer] Wrote ownvoice-output.wav
```

Every subcommand also supports `--json` for a structured, machine-parseable output mode, for a script or an agent calling `ownvoice` programmatically instead of a person reading the terminal:

```
$ ownvoice check --json
{"success": true, "message": "PEFT LoRA injection succeeded against pocket-tts's flow_lm module (target_modules=\"all-linear\").", "module_tree": null}
```

## CLI reference

Derived directly from `ownvoice/cli.py`'s Typer command definitions, not guessed.

### `ownvoice check`

Free, CPU-only compatibility check. Loads pocket-tts and dry-runs the LoRA injection. No GPU and no training required.

| Flag | Default | Description |
|---|---|---|
| `--json` | off | Print machine-readable JSON instead of human-readable text. |

Exit code: `0` on PASS, `1` on FAIL.

### `ownvoice train`

Trains a LoRA voice adapter from a directory of `.wav` voice clips.

| Flag | Default | Description |
|---|---|---|
| `--voice-clips` | *(required)* | Directory of `.wav` voice-clip recordings to train from. |
| `--out` | `./ownvoice-adapter` | Directory to write `adapter.safetensors` and `metadata.json` to. |
| `--epochs` | `10` | Number of training epochs. |
| `--lora-rank` | `8` | LoRA rank. |
| `--lora-alpha` | `16` | LoRA alpha. |
| `--lora-dropout` | `0.05` | LoRA dropout. |
| `--learning-rate` | `1e-4` | Optimizer learning rate. |
| `--eval-text` | `"This is my own voice, trained with OwnVoice."` | Sentence synthesized after training to score against the reference voice. |
| `--json` | off | Print machine-readable JSON instead of human-readable text. |

Exit code: `0` on any completed run (usable or below-threshold alike), `1` only on a data-loading failure or a caught PEFT-injection failure.

### `ownvoice infer`

Generates speech in a trained voice from a saved adapter, or plain pocket-tts output with no adapter.

| Flag | Default | Description |
|---|---|---|
| `--adapter` | *(required)* | Path to a trained `adapter.safetensors` file. |
| `--text` | *(required)* | Text to synthesize in the trained voice. |
| `--out` | `./ownvoice-output.wav` | Output `.wav` file path. |
| `--reference-audio` | the reference clip recorded in `metadata.json` at train time | Override which reference clip resolves the voice-conditioning state. |
| `--json` | off | Print machine-readable JSON instead of human-readable text. |

### Top-level

| Flag | Description |
|---|---|
| `--version` | Print the OwnVoice version and exit. |

## How OwnVoice compares

The most important comparison is the one above: pocket-tts's own built-in `--voice <wav>` zero-shot cloning, which needs no training step at all. Beyond that, here is how OwnVoice sits next to the closest comparable local-TTS and LoRA-fine-tuning tools we could find. Every row is sourced to that project's own README or docs, not a claim we are making on their behalf.

| Tool | What it does | How you get a custom voice | Install | Source |
|---|---|---|---|---|
| [pocket-tts](https://github.com/kyutai-labs/pocket-tts) | General local TTS, CPU-capable, ~6x real-time on a MacBook Air M4 CPU per its own README | `--voice <wav>` zero-shot cloning, no training step; full cloning fidelity needs gated Hugging Face access | `pip install pocket-tts` or `uvx pocket-tts` | pocket-tts README |
| [kokoro-tts](https://github.com/nazdridoy/kokoro-tts) | General local TTS with a fixed set of built-in voices and voice blending | No fine-tuning or cloning; you pick from shipped voices | `pip install kokoro-tts` (or `pip install git+...`), then download two model files (`voices-v1.0.bin`, `kokoro-v1.0.onnx`) from the project's GitHub releases | kokoro-tts README |
| [Unsloth](https://github.com/unslothai/unsloth) | General LLM fine-tuning framework (not TTS-specific), LoRA support built in | Python API / notebook workflow (`FastLanguageModel.from_pretrained` then `get_peft_model` then `SFTTrainer`), no single CLI command to start a run | `uv pip install unsloth --torch-backend=auto` | unslothai/unsloth README, unsloth.ai docs |
| **OwnVoice** | LoRA fine-tuning specifically for pocket-tts voice cloning | One CLI command: `ownvoice train --voice-clips <dir>`, with a free `ownvoice check` compatibility gate before it | `pip install git+https://github.com/RudrenduPaul/ownvoice` | this repo |

Two things we could not verify and are not claiming: neither kokoro-tts's nor Unsloth's own docs publish a measured "time to first working setup" number, so we are not putting fabricated timing numbers in this table, only the real install and workflow steps each project's own documentation describes. If you have measured numbers from running these yourself, an issue or PR with your methodology is welcome.

## How it works

```
voice clips (wav)
      |
      v
  data.py   --validate format/duration-->  clean clip set
      |
      v
  train.py  --PEFT LoRA (target_modules="all-linear")--> adapter.safetensors + metadata.json
      |
      v
  infer.py  --generate test utterance--> synthesized audio
      |
      v
  score.py  --resample to 16kHz mono--> Resemblyzer cosine similarity
      |
      v
  CLI report (>= 0.75 = usable adapter, below triggers a labeled next-step message)
```

`ownvoice/data.py` loads and validates the voice-clip directory. `ownvoice/train.py` loads pocket-tts's frozen base model, injects a LoRA adapter into its `flow_lm` transformer with PEFT (`target_modules="all-linear"`), runs the training loop, and saves the adapter plus a manifest. `ownvoice/infer.py` loads a saved adapter back onto the base model and generates speech. `ownvoice/score.py` resamples audio to 16kHz mono with `torchaudio.transforms.Resample` and scores speaker similarity with [Resemblyzer](https://github.com/resemble-ai/Resemblyzer).

OwnVoice is intentionally single-model: it wraps pocket-tts only, with no abstraction layer for a second base model, since none is in scope for v0.1.

## Consent and misuse

This tool clones a voice from audio you have the right to use. Do not clone someone else's voice, or a public figure's voice, without their explicit consent. OwnVoice ships no bulk-generation or auto-scaling feature in this version, keeping the blast radius of any single misuse case small.

## FAQ

**Does OwnVoice add voice cloning to pocket-tts?**
No. pocket-tts already has zero-shot voice cloning built in (`--voice <wav>`). OwnVoice adds LoRA fine-tuning on top: baking a voice permanently into trained weights instead of loading a reference clip at inference time. If zero-shot cloning is already enough for your use case, you do not need OwnVoice.

**Do I need a GPU?**
Not for `ownvoice check` or `ownvoice infer`, both of which run fine on CPU. `ownvoice train` will run on CPU too, just slower per epoch. An NVIDIA GPU with the CUDA build of PyTorch installed makes training meaningfully faster.

**Why does full voice-cloning fidelity need gated Hugging Face access?**
That is a property of pocket-tts's own model distribution, not something OwnVoice controls. The publicly downloadable weights (`kyutai/pocket-tts-without-voice-cloning`) ship with `has_voice_cloning = False`. Kyutai's best-quality cloning weights live in a separate, gated `kyutai/pocket-tts` repo on Hugging Face that requires requesting access first. See [Read this before you spend a week on this](#read-this-before-you-spend-a-week-on-this) above.

**Is OwnVoice affiliated with Kyutai?**
No. OwnVoice is an independent, third-party project that wraps pocket-tts. We intend to contribute the training script back to `kyutai-labs/pocket-tts` as an open pull request, but this repo is not maintained by Kyutai.

**How is this different from ElevenLabs or Gradium?**
Both are hosted commercial voice APIs: you send text over the network, they send audio back, and the model weights stay on their infrastructure. OwnVoice trains and runs entirely on your own machine or your own rented GPU; nothing about your voice clips or the resulting adapter is sent anywhere by this CLI. If you want a polished, hosted, general-purpose voice API, ElevenLabs or Gradium are likely the better fit. If you need to own the model file itself, this is what OwnVoice is for.

**Does any of my data leave my machine?**
No. `ownvoice check`, `ownvoice train`, and `ownvoice infer` all run locally against the machine invoking them. There is no hosted job runner, account, or telemetry in this version.

**What license is this under?**
MIT, matching pocket-tts's own license. See [License](#license) below.

**Does this work on Windows?**
It has been developed and validated on macOS and Linux. Windows support has not been separately validated; running under WSL is the safest path if you are on Windows.

## Implementation status

This is a young v0.1. `ownvoice check`, CLI argument parsing, voice-clip validation, the similarity-scoring math, and the adapter/manifest save-and-load path are implemented and covered by the test suite (`pytest`). LoRA injection was verified structurally against pocket-tts's real source, then confirmed for real: `ownvoice check` was run against pocket-tts's actual downloaded weights, on CPU, and PEFT's `target_modules="all-linear"` injection genuinely succeeded. The full training and generation path has been verified end to end for real too: a real 2-epoch LoRA training run against loaded pocket-tts weights produced a finite, non-NaN flow-matching loss, and the resulting adapter produced a real, non-silent generated `.wav` file via `ownvoice infer`.

That validation surfaced two real implementation gaps and fixed them: pocket-tts's published, inference-only PyPI package does not actually expose a way to compute the training loss through `FlowLMModel.forward()` despite its own docstring claiming otherwise, so OwnVoice computes the flow-matching loss directly from `flow_lm`'s real submodules instead; and swapping `base_model.flow_lm` to the PEFT-wrapped model before calling `generate_audio()` breaks pocket-tts's internal KV-cache state lookup, so no swap happens at all, since PEFT's LoRA injection already mutates `base_model.flow_lm` in place. The voice-cloning-fidelity limitation from the public weights is covered above and is a property of the base model, not an OwnVoice bug.

Run `ownvoice check` yourself and read the source before trusting any of this further. That is the right amount of skepticism for a v0.1.

## Contributing

Issues and PRs welcome, MIT licensed throughout. If you want to help close the actual gap this project targets, the most useful contribution is upstream: a lightweight LoRA-adapter training script contributed back to [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) itself, discussed on [issue #30](https://github.com/kyutai-labs/pocket-tts/issues/30).

## License

MIT. See [LICENSE](LICENSE).
