"""Lightweight timestamped progress logger for the regime pipeline.

The pipeline runs long multi-phase work (selection grid, CV folds, OOF folds,
final refit, 20 k-session forecast), so a silent process looks like a hang.
This module exposes a single ``log`` function that every stage calls with a
short, tag-prefixed status line so it is obvious from the terminal output
(a) *which* phase is currently running and (b) that it is making progress.

Design:

* Output goes to ``stderr`` so that ``stdout`` can still be captured cleanly
  when the CLI is redirected (the submission path and JSON diagnostics stay on
  ``stdout``).
* Messages are prefixed with ``[<elapsed>s][<tag>]`` for quick grepping.
* ``tick`` is a helper for inner-loop progress (fold 3/5, cluster iter 2, ...)
  that avoids polluting the log with one line per MC path.

The logger has no dependency on the rest of the module and is safe to import
from any file; :func:`set_verbose` lets the CLI flip it off for batch runs.
"""

from __future__ import annotations

import sys
import time
from typing import Optional

_T0: float = time.time()
_ENABLED: bool = True


def set_verbose(enabled: bool) -> None:
    """Enable or disable all progress output at runtime."""
    global _ENABLED
    _ENABLED = bool(enabled)


def reset_clock() -> None:
    """Reset the elapsed-time reference used by :func:`log`."""
    global _T0
    _T0 = time.time()


def log(tag: str, msg: str, *, end: str = "\n") -> None:
    """Emit a timestamped progress line to ``stderr``."""
    if not _ENABLED:
        return
    elapsed = time.time() - _T0
    stream = sys.stderr
    stream.write(f"[{elapsed:7.1f}s][{tag}] {msg}{end}")
    stream.flush()


def tick(tag: str, i: int, total: int, msg: str = "") -> None:
    """Inner-loop step indicator: ``[..][tag] (i/total) msg``."""
    if not _ENABLED:
        return
    base = f"({i}/{total})"
    log(tag, f"{base} {msg}".rstrip())


class Timer:
    """Context manager that logs entry/exit with an elapsed-seconds summary.

    Usage::

        with Timer("phase", "retraining full HMM"):
            ...

    emits ``[.][phase] retraining full HMM`` on entry and
    ``[.][phase] retraining full HMM (done in 12.3s)`` on exit.
    """

    def __init__(self, tag: str, msg: str):
        self.tag = tag
        self.msg = msg
        self._t0: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._t0 = time.time()
        log(self.tag, self.msg)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        dt = time.time() - (self._t0 or time.time())
        if exc is None:
            log(self.tag, f"{self.msg} (done in {dt:.1f}s)")
        else:
            log(self.tag, f"{self.msg} (failed after {dt:.1f}s: {exc_type.__name__})")
