# Security Policy

OwnVoice trains a LoRA voice adapter locally and loads/saves model
checkpoints on disk. A vulnerability that lets a crafted checkpoint,
audio file, or CLI argument execute code, escape the intended output
directory, or otherwise compromise the machine running `ownvoice` is
taken seriously and handled as a priority.

## Supported versions

| Package | Version | Supported |
| --- | --- | --- |
| `ownvoice-cli` (PyPI) | 0.x | Yes |
| `ownvoice-cli` (npm wrapper) | 0.x | Yes |

Both distributions are pre-1.0 and under active development. Security
fixes land on the latest `0.x` release of each; there is no older
supported line to backport to yet.

## Reporting a vulnerability

**Do not open a public GitHub issue for a security vulnerability.**

Report it privately via
[GitHub Security Advisories](https://github.com/RudrenduPaul/ownvoice/security/advisories/new)
for this repository. Include:

- Which distribution is affected (PyPI package, the npm wrapper, or both).
- A minimal reproduction: the command/flags used and, if relevant, the
  checkpoint or audio input that triggers the issue.
- What you expected OwnVoice to do, and what it actually did.
- Your assessment of impact, e.g. "loading a crafted `--adapter` checkpoint
  with `ownvoice infer` executes arbitrary code via unsafe deserialization."

## What counts as in scope

- Unsafe deserialization when loading a LoRA adapter checkpoint (for
  example, a `torch.load` call on an untrusted file that can lead to
  arbitrary code execution instead of only loading tensor weights).
- Path traversal via `--output`, `--adapter`, or similar path arguments
  that writes or reads outside the intended directory.
- Command or argument injection in the npm wrapper's bootstrap into `uv`
  or `pipx` before it hands off to the real Python CLI.

## What is out of scope

- Voice quality, adapter training convergence, or other output-correctness
  issues, those are normal bugs, please open a regular issue instead.
- Vulnerabilities in [pocket-tts](https://github.com/kyutai-labs/pocket-tts)
  or PEFT themselves, report those to their own maintainers.

## Response

We aim to acknowledge a report within 5 business days and to have a fix or
a mitigation plan within 30 days for a confirmed, in-scope vulnerability.
Credit is given in the release notes unless you ask to remain anonymous.
