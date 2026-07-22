# OwnVoice

[![PyPI](https://img.shields.io/pypi/v/ownvoice-cli)](https://pypi.org/project/ownvoice-cli/) [![npm](https://img.shields.io/npm/v/ownvoice-cli)](https://www.npmjs.com/package/ownvoice-cli)

Train a LoRA voice adapter for [pocket-tts](https://github.com/kyutai-labs/pocket-tts) and keep the result: a file on your own disk, not an API subscription.

## Why this exists

pocket-tts is a genuinely good, MIT-licensed, CPU-capable local text-to-speech model from Kyutai. Its own maintainers have been clear that fine-tuning code isn't coming any time soon: on [issue #30](https://github.com/kyutai-labs/pocket-tts/issues/30), maintainer @vvolhejn wrote "We are not planning to release fine-tuning code for our TTS and STT models in the near future," and 18 people reacted to that thread asking for exactly this. OwnVoice is a small, standalone CLI that fills that specific gap: point it at a handful of your own voice recordings, and it trains a LoRA adapter you keep and run yourself.

It is not a hosted service, it has no billing, and it does not track usage. It is a training script, an inference script, and a scoring script, wired together behind three CLI commands.

## What OwnVoice is not

pocket-tts already ships zero-shot voice cloning out of the box: pass a `.wav` file to `--voice` (or call `get_state_for_audio_prompt()` from Python) and it clones that voice with no training step at all. If that is all you need, use pocket-tts directly, it is simpler and faster.

OwnVoice exists for a narrower case: baking a voice permanently into trained weights, so generation no longer depends on distributing or re-processing a reference audio clip at runtime, with (based on the training objective, not yet independently benchmarked at scale) more consistent output across many generations than a single-clip zero-shot embedding tends to produce. That is the specific gap the 18 reactors on issue #30 were describing, and it is the only thing OwnVoice adds on top of what pocket-tts already does well.

## Install

Requires Python 3.11 or newer.

```bash
pip install ownvoice-cli
```

**npx / agent-native environments:** OwnVoice is a Python/PyTorch CLI, so the [npm package](https://www.npmjs.com/package/ownvoice-cli) is a thin wrapper, not a Node reimplementation. It bootstraps into the real CLI via [`uv`](https://docs.astral.sh/uv/) or `pipx`, whichever is already on `PATH` -- useful for coding-agent sandboxes and CI runners that default to a Node toolchain. The npm package was renamed to `ownvoice-cli` (from the old plain `ownvoice`, now deprecated) to match its PyPI counterpart.

```bash
npx ownvoice-cli check
```

Both the npm wrapper and the PyPI package (`ownvoice-cli`) are live, so the command above works today.

**Torch and CUDA:** `ownvoice check` (see below) needs no GPU at all and runs on CPU, matching pocket-tts's own CPU-capable design. Training a real adapter is much faster on an NVIDIA GPU. If you have one, install the CUDA build of PyTorch first by following [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), then install OwnVoice on top of it, so `pip` does not silently pull the CPU-only wheel instead. On Apple Silicon or a CPU-only machine, the default `pip install` of torch is fine: `ownvoice check` and `ownvoice infer` will run normally, `ownvoice train` will just take longer per epoch.

## Quickstart

![Terminal recording of installing ownvoice-cli with pip into a fresh virtual environment, then running `ownvoice --version` and `ownvoice --help` to show the real CLI and its three subcommands.](docs/demo.gif)

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

Record 5 to 10 minutes of clean audio of the voice you want to train (your own voice, with your own consent, see Consent and misuse below), split into a few `.wav` clips in one directory, then point OwnVoice at it:

```
$ ownvoice train --voice-clips ./my-voice-clips
[ownvoice train] USABLE ADAPTER
Usable adapter (similarity 0.812 >= 0.75). Try it now:
  ownvoice infer --adapter ownvoice-adapter/adapter.safetensors --text "This is my own voice, trained with OwnVoice."
```

Only `--voice-clips` is required. Every other flag has a sensible default: `--out` (`./ownvoice-adapter/`), `--epochs` (10), `--lora-rank` (8), `--lora-alpha` (16), `--lora-dropout` (0.05), `--learning-rate` (1e-4), and `--eval-text` (the sentence synthesized to compute the similarity score).

A run that finishes but does not clear the similarity bar still exits `0`. It is a labeled result with a concrete next step, not a crash:

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

Every subcommand also supports `--json` for a structured, machine-parseable output mode, useful if a script or an agent is calling `ownvoice` programmatically instead of a person reading the terminal:

```
$ ownvoice check --json
{"success": true, "message": "PEFT LoRA injection succeeded against pocket-tts's flow_lm module (target_modules=\"all-linear\").", "module_tree": null}
```

![Terminal recording of running `ownvoice check --json` for structured, agent-parseable output, then `ownvoice train --help` to show the real training flags and their defaults.](docs/usage.gif)

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

OwnVoice is intentionally single-model: it wraps pocket-tts only, with no abstraction layer for a second base model, since none is in scope.

## Consent and misuse

This tool clones a voice from audio you have the right to use. Do not clone someone else's voice, or a public figure's voice, without their explicit consent. OwnVoice ships no bulk-generation or auto-scaling feature in this version, keeping the blast radius of any single misuse case small.

## Setup-time benchmark vs comparable tools

| Tool | Time to first working setup | Notable design choice | Source |
|---|---|---|---|
| [kokoro-tts](https://github.com/nazdridoy/kokoro-tts) | under 2 minutes | `pip install git+...`, instant CLI synthesis, no fine-tuning | kokoro-tts README |
| [Unsloth](https://unsloth.ai) | under 1 minute to start a run | one-command training start (`uv pip install`) | Unsloth docs |
| [pocket-tts](https://github.com/kyutai-labs/pocket-tts) | seconds | `--voice <wav>` zero-shot cloning, no training available | pocket-tts README |
| **OwnVoice** | under 2 minutes to a confirmed-working training environment | `ownvoice check`: free, instant, CPU-only PEFT-compatibility validation before spending anything on a GPU | this repo |

OwnVoice's own training run is real GPU time, honestly labeled and not hidden behind a fake progress bar, the same category norm Unsloth uses. What OwnVoice compresses to under two minutes is everything *before* that: confirming your environment actually works.

## Implementation status

This is a young v0.1. `ownvoice check`, the CLI argument parsing, voice-clip validation, the similarity scoring math, and the adapter/manifest save and load path are implemented and covered by the test suite (`pytest`). LoRA injection was verified structurally against pocket-tts's real source and then confirmed for real: `ownvoice check` was run against pocket-tts's actual downloaded weights, on CPU, and PEFT's `target_modules="all-linear"` injection genuinely succeeded. The full training and generation path has since been verified end to end for real too: a real 2-epoch LoRA training run against loaded pocket-tts weights produced a finite, non-NaN flow-matching loss, and the resulting adapter produced a real, non-silent generated `.wav` file via `ownvoice infer`. That validation surfaced two real gaps in the naive approach and fixed them: (1) pocket-tts's published, inference-only PyPI package does not actually expose a way to compute the training loss through `FlowLMModel.forward()` despite its own docstring claiming otherwise, so OwnVoice computes the flow-matching loss directly from `flow_lm`'s real submodules instead; (2) swapping `base_model.flow_lm` to the PEFT-wrapped model before calling `generate_audio()` breaks pocket-tts's internal KV-cache state lookup -- no swap is needed at all, since PEFT's LoRA injection already mutates `base_model.flow_lm` in place. One real, external limitation to know about: the publicly downloadable pocket-tts weights (`kyutai/pocket-tts-without-voice-cloning`) refuse a raw reference-clip path/URL outright; OwnVoice works around this by pre-loading and resampling the clip itself, but voice-cloning fidelity from that checkpoint is a known limitation of the base model, not an OwnVoice bug -- for kyutai's best-quality cloning weights, request gated access at [huggingface.co/kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts). Run `ownvoice check` yourself and read the source before trusting any of it further, that is the right amount of skepticism for a v0.1.

## FAQ

**What is OwnVoice, and why not just use pocket-tts by itself?**
OwnVoice trains a LoRA adapter for [pocket-tts](https://github.com/kyutai-labs/pocket-tts) and saves it to your own disk as `adapter.safetensors` plus `metadata.json`. It exists because pocket-tts's own maintainers have said fine-tuning code is not on their near-term roadmap (see [issue #30](https://github.com/kyutai-labs/pocket-tts/issues/30)). Once you have a trained adapter, you never need OwnVoice again to use it: `ownvoice infer` just loads the adapter back onto the base model.

**How is this different from pocket-tts's own built-in `--voice <wav>` zero-shot cloning?**
pocket-tts already clones a voice from a single reference clip with no training step, `--voice <wav>` at the CLI or `get_state_for_audio_prompt()` in Python. OwnVoice trades that speed for a permanently trained adapter, so generation no longer depends on carrying around a reference clip at runtime, with (based on the training objective, not yet independently benchmarked at scale) more consistent output across repeated generations than a single-clip zero-shot embedding tends to give. If zero-shot is enough for your use case, use pocket-tts directly, it is simpler and faster.

**What do I need to install it, and does it run on Apple Silicon or a CPU-only machine?**
Python 3.11 or newer, then `pip install ownvoice-cli`. `ownvoice check` and `ownvoice infer` need no GPU at all and run fine on Apple Silicon or a CPU-only machine, matching pocket-tts's own CPU-capable design. `ownvoice train` runs on CPU too, it just takes longer per epoch; install the CUDA build of PyTorch first if you have an NVIDIA GPU and want training to go faster.

**How does OwnVoice compare to kokoro-tts and Unsloth?**
[kokoro-tts](https://github.com/nazdridoy/kokoro-tts) gets you synthesizing speech in under 2 minutes but has no fine-tuning step at all. [Unsloth](https://unsloth.ai) gets a training run started in under a minute but is a general LLM fine-tuning framework, not TTS-specific. OwnVoice is narrower than either: one base model (pocket-tts only), one job (a voice adapter), plus a free `ownvoice check` step that confirms PEFT's LoRA injection actually works against your environment before you spend anything on a GPU, a check neither of those tools has an equivalent of.

**My training run finished but printed "BELOW THRESHOLD", is that a bug?**
No. It is a labeled outcome, not a crash, `ownvoice train` exits `0` either way. Below the 0.75 cosine-similarity bar, the adapter is still saved to disk and OwnVoice tells you plainly to try more or cleaner voice clips, more epochs, or a higher `--lora-rank`, then re-run. Only two things actually fail the command with a non-zero exit: no usable clips to load, or a caught PEFT-injection failure.

**Can I use OwnVoice, and the adapters it produces, commercially?**
OwnVoice's own code is MIT (see [LICENSE](LICENSE)). pocket-tts's code package is MIT too, but the model weights OwnVoice actually downloads and trains against, [`kyutai/pocket-tts-without-voice-cloning`](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning) and the gated [`kyutai/pocket-tts`](https://huggingface.co/kyutai/pocket-tts), are licensed CC-BY-4.0, not MIT. CC-BY-4.0 permits commercial use but requires attribution to Kyutai. Since any adapter you train is derived from those weights, check that attribution requirement before shipping a commercial product built on it.

**Whose voice can I actually clone with this?**
Only your own, or someone else's with their explicit, checked consent, never a public figure's voice without it. See [Consent and misuse](#consent-and-misuse) above. OwnVoice ships no bulk-generation or auto-scaling feature in this version, which keeps the blast radius of any single misuse case small.

## Contributing

Issues and PRs welcome, MIT licensed throughout. If you want to help close the actual gap this project targets, the most useful contribution is upstream: a lightweight LoRA-adapter training script contributed back to [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) itself, discussed on [issue #30](https://github.com/kyutai-labs/pocket-tts/issues/30).

## License

MIT. See [LICENSE](LICENSE).

