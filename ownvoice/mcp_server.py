"""OwnVoice MCP server: a generic subprocess wrapper around the `ownvoice` CLI.

Requires the `mcp` extra (`pip install "ownvoice-cli[mcp]"`). Started via
`ownvoice-mcp` (stdio transport). Exposes a single `run` tool that shells
out to the real `ownvoice` CLI and returns its result as structured JSON,
so any MCP-compatible agent runtime can drive `ownvoice check` / `train` /
`infer` directly instead of shelling out and parsing text itself.

Uses `mcp.server.MCPServer`, the official SDK's current high-level server
class (`mcp` 2.0.0+). Earlier `mcp` 1.x releases exposed the same
`.tool()`/`.run()` pattern under `mcp.server.fastmcp.FastMCP`, which the
2.0.0 release removed -- confirmed directly against the installed package.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from typing import Any

from mcp.server import MCPServer

_TOOL_DESCRIPTION = (
    "Run the ownvoice CLI (`check`, `train`, or `infer`) with the given argv "
    "and return its result as structured JSON. Call this to confirm a "
    "machine can run LoRA voice-adapter training against pocket-tts before "
    "committing GPU time, to train an adapter from your own .wav voice "
    "clips, or to synthesize speech in a previously trained voice -- all "
    "local, no hosted API or account involved.\n\n"
    "Call `run(args=[\"check\", \"--json\"])` first, before ever training: "
    "it is free, CPU-only, takes seconds, and confirms PEFT's LoRA "
    "injection actually works against pocket-tts's model structure on this "
    "machine. Do not call `train` until `check` has reported success. "
    "`train` requires an existing directory of .wav voice-clip recordings "
    "(5-10 minutes of clean audio) already on disk; `infer` requires an "
    "`adapter.safetensors` file already produced by a prior `train` call. "
    "No API key or network credential is needed anywhere in this flow.\n\n"
    "This tool is a thin subprocess wrapper: it never makes network calls "
    "itself, but `train` and `infer` load a local pocket-tts model into "
    "memory and are CPU/GPU- and time-intensive (`train` can run minutes "
    "to hours depending on `--epochs`; `check` and `infer` are comparatively "
    "fast). `check` is read-only. `train` and `infer` are mutating: `train` "
    "writes `adapter.safetensors` and `metadata.json` to the `--out` "
    "directory (default `./ownvoice-adapter/`), and `infer` writes a .wav "
    "file to `--out` (default `./ownvoice-output.wav`). Re-running either "
    "with the same args overwrites the same output paths rather than "
    "accumulating state. A `train` run that completes but scores below the "
    "0.75 similarity threshold still exits 0 -- that is a labeled 'usable' "
    "vs 'below threshold' outcome, not a failure. Only a data-loading "
    "problem or a caught PEFT-injection failure exits non-zero. Every "
    "failure mode (launch failure, timeout after 1800s, non-zero exit) is "
    "caught by this tool and returned as {\"error\": ...} rather than "
    "raised.\n\n"
    "`args` is a list[str] of the exact argv to pass after `ownvoice`, e.g.:\n"
    '- ["check", "--json"] -- Day-0 compatibility check.\n'
    '- ["train", "--voice-clips", "./my-voice-clips", "--epochs", "15", '
    '"--json"] -- train an adapter, overriding the default epoch count.\n'
    '- ["infer", "--adapter", "./ownvoice-adapter/adapter.safetensors", '
    '"--text", "Hello, this is my own voice.", "--json"] -- synthesize '
    "speech from a trained adapter.\n\n"
    "Always include `--json` for a structured payload; pass "
    '["<subcommand>", "--help"] (or just ["--help"]) as args to discover '
    "the full flag list for any subcommand rather than guessing. The "
    'returned dict has one of three shapes: {"result": {...}} on a '
    "successful JSON-mode call (`check` returns success/message/"
    "module_tree; `train` returns success/out_dir/similarity_score/usable/"
    "message/infer_command; `infer` returns success/out_path); "
    '{"error": "...", "returncode": N} on a non-zero exit; or '
    '{"stdout": ..., "stderr": ..., "returncode": N} if `--json` was '
    "omitted or stdout was not valid JSON."
)


def _resolve_command() -> list[str]:
    """Prefer the installed `ownvoice` console script; fall back to
    `python -m ownvoice.cli` if it is not on PATH (e.g. an editable
    install invoked before the entry point script is regenerated)."""
    exe = shutil.which("ownvoice")
    if exe:
        return [exe]
    return [sys.executable, "-m", "ownvoice.cli"]


mcp = MCPServer("ownvoice")


@mcp.tool(description=_TOOL_DESCRIPTION)
def run(args: list[str]) -> dict[str, Any]:
    """Shell out to the real `ownvoice` CLI and return its result as a dict.

    `args` is the argv passed after `ownvoice`, e.g. ["check", "--json"] or
    ["train", "--voice-clips", "./clips", "--json"]. Every failure mode
    (launch failure, timeout, non-zero exit, unparsable stdout) is caught
    and returned as a dict -- this tool is never able to raise.
    """
    try:
        completed = subprocess.run(
            [*_resolve_command(), *args],
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except OSError as exc:
        return {"error": f"failed to launch ownvoice: {exc}"}
    except subprocess.TimeoutExpired as exc:
        return {"error": f"ownvoice timed out after {exc.timeout}s"}

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    if completed.returncode != 0:
        return {
            "error": stderr.strip() or stdout.strip() or f"ownvoice exited with code {completed.returncode}",
            "returncode": completed.returncode,
        }

    stripped = stdout.strip()
    if stripped:
        try:
            return {"result": json.loads(stripped)}
        except json.JSONDecodeError:
            pass

    return {"stdout": stdout, "stderr": stderr, "returncode": completed.returncode}


def main() -> None:
    """Start the MCP server on stdio transport -- entry point for `ownvoice-mcp`."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
