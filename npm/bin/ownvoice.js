#!/usr/bin/env node
'use strict';

const { spawnSync } = require('node:child_process');
const path = require('node:path');

// Pinned to this npm package's own version, not left floating: an unqualified
// "uvx ownvoice"/"pipx run ownvoice" always fetches whatever is currently
// newest on PyPI. That decouples what a user actually installed (this npm
// package, a fixed point in time) from what runs on their machine (PyPI's
// latest, which changes underneath them and could be a compromised or
// unintended publish) -- a supply-chain determinism gap. Pinning to this
// package's own version keeps the two registries in lockstep: bump this
// package.json's version when PyPI's ownvoice ships a new release, and every
// subsequent invocation resolves to that exact pinned release, not whatever
// is newest at run time.
const PACKAGE_VERSION = require(path.join(__dirname, '..', 'package.json')).version;

function commandExists(cmd) {
  const probe = process.platform === 'win32' ? 'where' : 'which';
  const result = spawnSync(probe, [cmd], { stdio: 'ignore' });
  return result.status === 0;
}

function run(cmd, args) {
  const result = spawnSync(cmd, args, { stdio: 'inherit' });
  if (result.error) {
    return null;
  }
  return result.status;
}

const args = process.argv.slice(2);

// ownvoice is a Python/PyTorch package (LoRA fine-tuning on top of pocket-tts).
// This wrapper never bundles a platform binary -- there isn't one to bundle --
// it bootstraps into the real ownvoice Python CLI via whichever Python runner
// is already on PATH, preferring uv/uvx since that's the primary documented
// install path (`uvx ownvoice train ...`) and increasingly present by default
// in agent and CI sandboxes.
const runners = [
  { cmd: 'uvx', build: (a) => [`ownvoice==${PACKAGE_VERSION}`, ...a] },
  { cmd: 'pipx', build: (a) => ['run', `ownvoice==${PACKAGE_VERSION}`, ...a] },
];

for (const runner of runners) {
  if (commandExists(runner.cmd)) {
    const status = run(runner.cmd, runner.build(args));
    if (status !== null) {
      process.exit(status);
    }
  }
}

console.error(
  [
    'ownvoice: no Python runner found (checked uvx, pipx).',
    '',
    'ownvoice is a Python/PyTorch CLI; this npm package is a thin wrapper',
    'that bootstraps it, not a standalone Node reimplementation.',
    '',
    'Install one of the following, then re-run this command:',
    '  - uv (recommended):  https://docs.astral.sh/uv/getting-started/installation/',
    '  - pipx:              https://pipx.pypa.io/stable/installation/',
    '',
    'Or install ownvoice directly with pip:',
    '  pip install ownvoice',
  ].join('\n')
);
process.exit(1);
