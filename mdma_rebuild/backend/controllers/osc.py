"""OSC input controller.

Uses :mod:`pythonosc` — install with ``pip install python-osc``.

Per the spec:

- ``kind="trigger"`` for incoming OSC messages (OSC has no natural
  press/release model; single bangs are idiomatic).
- ``key`` is the OSC address with slashes converted to dots:
  ``/synth/cutoff`` -> ``"synth.cutoff"``.
- Continuous channels are created dynamically on the first float
  message to a given address. Each such channel remembers the last
  value received so consumers can pull it via ``channel.read()``.

Like :class:`MidiController`, the ``pythonosc`` import is deferred
until :meth:`start` so the module stays importable without the
external dependency. The bound UDP port defaults to 8000.
"""

from __future__ import annotations

import queue
import threading
from typing import Dict, List, Optional, Tuple

from ..input_controller import Controller, InputChannel, InputEvent


class _OSCChannel(InputChannel):
    """Continuous OSC channel backed by the most recent float value."""

    def __init__(self, key: str) -> None:
        super().__init__(key)
        self._value: float = 0.0

    def read(self) -> float:
        return self._value

    def range(self):
        # OSC is domain-agnostic — declare the widest safe range.
        return (-1.0, 1.0)

    def _write(self, value: float) -> None:
        self._value = float(value)


class OSCController(Controller):
    """Receives OSC messages on a UDP socket.

    ``host`` / ``port`` default to ``"127.0.0.1"`` and ``8000``.
    """

    name = "OSC"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        self.host: str = host
        self.port: int = int(port)
        self._events: "queue.Queue[InputEvent]" = queue.Queue()
        self._channels: Dict[str, _OSCChannel] = {}
        self._server = None
        self._thread: Optional[threading.Thread] = None

    # -- Controller interface -----------------------------------------

    def start(self) -> None:
        if self._server is not None:
            return
        try:
            from pythonosc.dispatcher import Dispatcher  # type: ignore
            from pythonosc.osc_server import ThreadingOSCUDPServer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OSCController requires 'python-osc' "
                "(install with: pip install python-osc)"
            ) from exc

        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self._on_message)
        self._server = ThreadingOSCUDPServer(
            (self.host, self.port), dispatcher
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"mdma-osc-{self.port}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        server = self._server
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass
            self._server = None
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
        return dict(self._channels)

    # -- OSC dispatch -------------------------------------------------

    def _on_message(self, address: str, *args) -> None:
        key = address.lstrip("/").replace("/", ".")
        first_float = _first_float(args)
        if first_float is not None:
            ch = self._channels.get(key)
            if ch is None:
                ch = _OSCChannel(key)
                self._channels[key] = ch
            ch._write(first_float)
        self._events.put(InputEvent(
            timestamp=0.0,
            kind="trigger",
            key=key,
            value=first_float,
            meta={"args": list(args), "address": address},
        ))


def _first_float(args: Tuple) -> Optional[float]:
    """Return the first argument convertible to float, or ``None``."""
    for arg in args:
        try:
            return float(arg)
        except (TypeError, ValueError):
            continue
    return None


__all__ = ["OSCController"]
