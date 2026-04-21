"""MIDI input controller.

Uses :mod:`mido` — install with ``pip install mido python-rtmidi``.

Per the spec:

- ``kind="press"`` for note-on with velocity > 0, ``kind="release"``
  for note-off (or note-on with velocity 0).
- ``key`` is ``"note_{n}"`` for note events.
- Continuous channels: ``"cc_{n}"`` for each CC that has received a
  value since ``start()``, value normalised to ``[0.0, 1.0]``.
- ``meta`` includes the MIDI channel and raw velocity.

The controller does not require the mido import to exist at module
import time: :class:`MidiController` only calls ``import mido`` inside
:meth:`start`, so ``from mdma_rebuild.backend.controllers import
MidiController`` works even without mido installed. ``start()`` raises
a clear :class:`RuntimeError` in that case.
"""

from __future__ import annotations

import queue
import threading
from typing import Dict, List, Optional

from ..input_controller import Controller, InputChannel, InputEvent


class _CCChannel(InputChannel):
    """Continuous channel backed by the most recent MIDI CC value."""

    def __init__(self, key: str) -> None:
        super().__init__(key)
        self._value: float = 0.0

    def read(self) -> float:
        return self._value

    def range(self):
        return (0.0, 1.0)

    def _write(self, value: float) -> None:
        self._value = float(value)


class MidiController(Controller):
    """Reads from a MIDI input port via :mod:`mido`.

    ``port`` may be an integer index (into :func:`mido.get_input_names`)
    or a port name string. Defaults to the first available port.
    """

    name = "Midi"

    def __init__(self, port=None) -> None:
        self._port_spec = port
        self._events: "queue.Queue[InputEvent]" = queue.Queue()
        self._channels: Dict[str, _CCChannel] = {}
        self._port = None
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

    # -- Controller interface -----------------------------------------

    def start(self) -> None:
        if self._running:
            return
        try:
            import mido  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "MidiController requires 'mido' (install with: "
                "pip install mido python-rtmidi)"
            ) from exc

        port_name = self._resolve_port_name(mido)
        self._port = mido.open_input(port_name)
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop,
            name=f"mdma-midi-{port_name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        port = self._port
        if port is not None:
            try:
                port.close()
            except Exception:
                pass
            self._port = None

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

    # -- Internals ----------------------------------------------------

    def _resolve_port_name(self, mido) -> str:
        """Turn the user's ``port`` argument into a mido port name."""
        available = mido.get_input_names()
        if self._port_spec is None:
            if not available:
                raise RuntimeError("no MIDI input ports available")
            return available[0]
        if isinstance(self._port_spec, int):
            return available[self._port_spec]
        return str(self._port_spec)

    def _read_loop(self) -> None:
        port = self._port
        while self._running and port is not None:
            try:
                msg = port.receive(block=True)
            except Exception:
                break
            self._handle_message(msg)

    def _handle_message(self, msg) -> None:
        mtype = getattr(msg, "type", None)
        channel = getattr(msg, "channel", None)
        if mtype == "note_on":
            velocity = getattr(msg, "velocity", 0)
            kind = "press" if velocity > 0 else "release"
            self._events.put(InputEvent(
                timestamp=0.0,
                kind=kind,
                key=f"note_{msg.note}",
                value=float(velocity) / 127.0,
                meta={"channel": channel, "velocity": velocity},
            ))
        elif mtype == "note_off":
            self._events.put(InputEvent(
                timestamp=0.0,
                kind="release",
                key=f"note_{msg.note}",
                value=float(getattr(msg, "velocity", 0)) / 127.0,
                meta={"channel": channel},
            ))
        elif mtype == "control_change":
            key = f"cc_{msg.control}"
            ch = self._channels.get(key)
            if ch is None:
                ch = _CCChannel(key)
                self._channels[key] = ch
            ch._write(float(msg.value) / 127.0)
        elif mtype == "program_change":
            self._events.put(InputEvent(
                timestamp=0.0,
                kind="trigger",
                key=f"program_{msg.program}",
                value=None,
                meta={"channel": channel},
            ))


__all__ = ["MidiController"]
