"""KernelOpt GPU runner: stateless JSON-in/JSON-out command dispatcher.

Protocol v1: one JSON request object on stdin -> one JSON response object.
Rust owns all state; this process holds none between invocations.

Large responses (e.g. embedded graphsignal payloads) are written atomically
to `response_file` when the request provides one — stdout pipes can be
truncated if the process dies during CUDA/CUPTI teardown, so the file is the
authoritative channel; stdout remains for backward compatibility.

Usage: python3 -m kernelopt_runner   (from runner/, or with runner/ on PYTHONPATH)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

PROTOCOL_VERSION = 1


def fail(kind: str, message: str, traceback_text: str = "") -> dict:
    """Structured error every consumer (Executor retry loop) relies on."""
    return {"ok": False, "error": {"kind": kind, "message": message, "traceback": traceback_text}}


def _emit(response: dict, response_file: str | None) -> None:
    text = json.dumps(response)
    if response_file:
        tmp = response_file + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, response_file)  # atomic
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except BrokenPipeError:
        # The response file is authoritative; a closed stdout must not turn a
        # successful command into a nonzero exit (the bridge warns on that).
        pass


def _debug(msg: str) -> None:
    """Opt-in exit-path tracing (KERNELOPT_RUNNER_DEBUG=1)."""
    if not os.environ.get("KERNELOPT_RUNNER_DEBUG"):
        return
    path = os.environ.get("KERNELOPT_RUNNER_DEBUG_LOG") or os.path.join(
        tempfile.gettempdir(), "kernelopt_runner_debug.log"
    )
    try:
        with open(path, "a") as f:
            f.write(f"{os.getpid()} {msg}\n")
    except OSError:
        pass


def main() -> int:
    raw = sys.stdin.read()
    try:
        request = json.loads(raw)
    except json.JSONDecodeError as exc:
        _debug("invalid-json")
        _emit(fail("protocol", f"invalid JSON request: {exc}"), None)
        return 1

    response_file = request.get("response_file")
    command = request.get("command", "")
    try:
        if command == "trace":
            from . import trace as cmd
        elif command == "verify":
            from . import verify as cmd
        elif command == "bench":
            from . import bench as cmd
        elif command == "signals":
            from . import signals as cmd
        elif command == "restitch":
            from . import restitch as cmd
        elif command == "e2e_verify":
            from . import e2e as cmd
        elif command == "ncu":
            from . import ncu as cmd
        elif command in ("graphsignal_profile", "graphsignal_setup"):
            from . import graphsignal as cmd
        elif command in ("engine_generate", "engine_perplexity"):
            from . import engine as cmd
        else:
            _debug(f"unknown-command {command!r}")
            _emit(fail("protocol", f"unknown command: {command!r}"), response_file)
            return 1
        response = cmd.run(request)
    except Exception as exc:  # noqa: BLE001 - the protocol boundary is exactly the place to catch
        import traceback as tb

        kind = "runtime"
        text = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, SyntaxError) or "IndentationError" in text:
            kind = "compile"
        _debug(f"except command={command} {text}")
        _emit(fail(kind, text, tb.format_exc()), response_file)
        return 1

    response.setdefault("ok", True)
    response.setdefault("protocol", PROTOCOL_VERSION)
    _debug(f"success command={command}")
    _emit(response, response_file)
    sys.stdout.flush()
    sys.stderr.flush()
    # Deterministic success exit: the response file (or stdout) is already
    # written, so nothing in interpreter shutdown should be able to turn a
    # completed command into a nonzero exit code.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
