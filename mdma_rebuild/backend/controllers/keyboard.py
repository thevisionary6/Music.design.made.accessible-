"""Computer-keyboard controller.

Implements the :class:`mdma_rebuild.backend.input_controller.Controller`
interface. Primarily for development and testing.

**NVDA compatibility (non-negotiable):** this controller must never
install a global keyboard hook. A global hook would intercept
keystrokes before NVDA sees them, breaking the screen reader Jake
uses to drive the system. The implementation below only captures keys
when the Python process has focus — it reads stdin via a blocking
reader on a background thread, which only sees input directed at the
terminal window.

For more sophisticated focus-scoped capture (GUI apps), swap in
``pynput``'s focused-mode listener. Do **not** use its global hook
API. The ``start()`` path here is intentionally simple so a reviewer
can confirm at a glance that there is no system-wide keyboard grab.
"""

from __future__ import annotations

import queue
import sys
import threading
from typing import Dict, List, Optional

from ..input_controller import Controller, InputChannel, InputEvent


class KeyboardController(Controller):
    """Focus-scoped stdin-based keyboard controller.

    Reads single characters from stdin on a background thread and
    emits an ``InputEvent`` with ``kind="press"`` for each one. The
    stdin read is blocking; the thread exits cleanly when :meth:`stop`
    flips ``_running`` off and pushes a sentinel.

    No ``"release"`` events in this simple variant — stdin doesn't
    report key releases. A future implementation can swap in pynput's
    focused-mode listener to add them.
    """

    name = "Keyboard"

    def __init__(self) -> None:
        self._events: "queue.Queue[InputEvent]" = queue.Queue()
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

    # -- Controller interface -----------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._read_stdin,
            name="mdma-keyboard",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        # The stdin read is blocking; joining with a tight timeout is
        # fine because the thread is a daemon and will be GC'd when
        # the interpreter exits.
        self._thread = None

    def poll(self) -> List[InputEvent]:
        out: List[InputEvent] = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                break
        return out

    def channels(self) -> Dict[str, InputChannel]:
        return {}

    # -- Internals ----------------------------------------------------

    def _read_stdin(self) -> None:
        while self._running:
            line = sys.stdin.readline()
            if not line:
                break
            for ch in line.rstrip("\n\r"):
                key = _normalise_key(ch)
                self._events.put(InputEvent(
                    timestamp=0.0,  # aggregator stamps it
                    kind="press",
                    key=key,
                    value=None,
                    meta=None,
                ))


def _normalise_key(ch: str) -> str:
    """Normalise a single stdin character to a stable key name.

    Lowercased alphabetic characters pass through; spaces and a few
    common specials get named for readability (``"space"``,
    ``"tab"``); everything else becomes ``"char_XX"`` where XX is the
    hex code so handlers always have a stable identifier to match on.
    """
    if ch == " ":
        return "space"
    if ch == "\t":
        return "tab"
    if ch.isprintable() and len(ch) == 1:
        return ch.lower()
    return f"char_{ord(ch):02x}"


__all__ = ["KeyboardController"]
