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

_STATIC_FALLBACK_DESCRIPTION = (
    "Run the ownvoice CLI with the given argv and return its result as "
    "structured JSON. ownvoice trains a LoRA voice adapter for pocket-tts "
    "and keeps the result on disk, not behind an API. Subcommands: "
    "`check` (free, CPU-only compatibility check, no GPU/training needed), "
    "`train --voice-clips <dir>` (train an adapter from a directory of "
    ".wav clips), `infer --adapter <path> --text <text>` (synthesize "
    "speech in the trained voice). Pass `--json` in args to get a "
    "machine-readable payload back under \"result\"; without it, raw "
    "stdout/stderr is returned instead. Example: "
    'run(args=["check", "--json"]).'
)


def _resolve_command() -> list[str]:
    """Prefer the installed `ownvoice` console script; fall back to
    `python -m ownvoice.cli` if it is not on PATH (e.g. an editable
    install invoked before the entry point script is regenerated)."""
    exe = shutil.which("ownvoice")
    if exe:
        return [exe]
    return [sys.executable, "-m", "ownvoice.cli"]


def _build_tool_description() -> str:
    """Populate the tool description from the real `ownvoice --help` output
    at import time. Falls back to a safe static description if the
    subprocess call fails for any reason -- this must never raise."""
    try:
        result = subprocess.run(
            [*_resolve_command(), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        help_text = (result.stdout or result.stderr or "").strip()
        if not help_text:
            return _STATIC_FALLBACK_DESCRIPTION
        return (
            "Run the ownvoice CLI with the given argv and return its result "
            "as structured JSON. Real `ownvoice --help` output follows:\n\n"
            + help_text
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return _STATIC_FALLBACK_DESCRIPTION


_TOOL_DESCRIPTION = _build_tool_description()

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
