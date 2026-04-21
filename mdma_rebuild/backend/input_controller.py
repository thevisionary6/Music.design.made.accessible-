"""Agnostic input layer for the MDMA backend.

Implements ``references/input_controller_spec.md`` (Phase 7).

The :class:`InputController` translates physical input — computer
keyboard, MIDI, OSC, gamepad (v1.1), custom hardware — into a unified
internal protocol consumable by the scheduler and the
:class:`ControllerSource` automation source.

Two record types carry the data:

- :class:`InputEvent` — discrete (key press, note-on, OSC bang).
  Pushed through ``poll()``.
- :class:`InputChannel` — continuous (MIDI CC, joystick axis, OSC
  float stream). Pulled via ``channel.read()`` at whatever rate the
  consumer needs.

Every concrete :class:`Controller` implements ``start`` / ``stop`` /
``poll`` / ``channels``. The :class:`InputController` owns a set of
controllers under namespace aliases so events from two MIDI keyboards
can coexist without colliding — ``"mk1.note_60"`` vs ``"mk2.note_60"``.

Threading: the aggregator runs a background poll thread on
:meth:`start`. Handlers are invoked on that thread; anything they call
on the scheduler or Session must be thread-safe.

Handler exceptions are logged (via ``print`` for Phase 7; swap for a
logger later) but never propagated. Live performance cannot afford a
single faulty handler killing the loop.
"""

from __future__ import annotations

import fnmatch
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .automation import AutomationSource


# ---------------------------------------------------------------------------
# Internal protocol
# ---------------------------------------------------------------------------


@dataclass
class InputEvent:
    """Discrete input event.

    Attributes
    ----------
    timestamp : float
        Seconds, clock time at dispatch. Filled in by the
        InputController when it rewrites the event with the namespaced
        key, so concrete controllers can leave it at 0.0 if they want
        to.
    kind : str
        ``"press"`` / ``"release"`` / ``"trigger"`` by convention;
        controllers may emit additional kinds for device-specific
        events, but handlers that want portable behaviour should stick
        to those three.
    key : str
        Stable identifier for this event. Set by the controller; the
        aggregator rewrites it to ``"alias.original_key"`` before
        dispatching.
    value : float, optional
        Scalar payload (e.g., velocity). May be ``None``.
    meta : dict, optional
        Extra data — channel number, raw bytes, device-specific
        info. May be ``None``.
    """

    timestamp: float
    kind: str
    key: str
    value: Optional[float] = None
    meta: Optional[dict] = None


class InputChannel:
    """Continuous input channel.

    Subclasses override :meth:`read` and :meth:`range`. The default
    ``range`` is ``(0.0, 1.0)``.
    """

    def __init__(self, key: str) -> None:
        self.key: str = key

    def read(self) -> float:
        """Return the current value of this channel."""
        raise NotImplementedError

    def range(self) -> Tuple[float, float]:
        """Return the declared ``(low, high)`` range of this channel."""
        return (0.0, 1.0)


# ---------------------------------------------------------------------------
# Controller interface
# ---------------------------------------------------------------------------


class Controller:
    """Base class for concrete input controllers.

    Controllers manage their own resources (ports, sockets, listeners)
    between :meth:`start` and :meth:`stop`, and expose their events and
    continuous channels via :meth:`poll` and :meth:`channels`.
    """

    name: str = "Controller"

    def start(self) -> None:
        """Open the device / socket / listener. Idempotent."""
        return None

    def stop(self) -> None:
        """Close cleanly. Idempotent."""
        return None

    def poll(self) -> List[InputEvent]:
        """Return any events that have arrived since the last poll.

        Must be non-blocking; called on the InputController thread.
        """
        return []

    def channels(self) -> Dict[str, InputChannel]:
        """Return a dict of continuous channels, keyed by stable identifier."""
        return {}


# ---------------------------------------------------------------------------
# Handler handle
# ---------------------------------------------------------------------------


class HandlerHandle:
    """Opaque handle returned by :meth:`InputController.on`.

    Calling :meth:`remove` unregisters the handler. Idempotent; a handle
    that has already been removed does nothing on subsequent calls.
    """

    def __init__(self, controller: "InputController", registration) -> None:
        self._controller = controller
        self._registration = registration

    def remove(self) -> None:
        self._controller._remove_handler(self._registration)


@dataclass
class _HandlerRegistration:
    """Private record for one registered handler."""

    kind: str
    key_pattern: str
    handler: Callable[[InputEvent], None]


# ---------------------------------------------------------------------------
# InputController
# ---------------------------------------------------------------------------


class InputController:
    """Top-level aggregator for every :class:`Controller` in the system.

    Not owned by the scheduler — the user wires the two together
    explicitly. See the spec's "Integration with Scheduler" section.
    """

    def __init__(self, clock=None) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._controllers: Dict[str, Controller] = {}
        self._handlers: List[_HandlerRegistration] = []
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._poll_interval: float = 0.005  # 5 ms; good enough for MIDI
        self._wake = threading.Event()

    # -- Controller registration --------------------------------------

    def add_controller(self, controller: Controller, alias: str) -> None:
        """Register ``controller`` under namespace ``alias``.

        Events and channels from this controller become accessible as
        ``"alias.original_key"``. Re-registering under an existing alias
        raises :class:`ValueError`.
        """
        with self._lock:
            if alias in self._controllers:
                raise ValueError(
                    f"alias {alias!r} is already registered"
                )
            self._controllers[alias] = controller
        if self._running:
            try:
                controller.start()
            except Exception as exc:
                print(f"[input] controller {alias} failed to start: {exc}")

    def remove_controller(self, alias: str) -> None:
        """Unregister a controller. No-op if unknown."""
        with self._lock:
            controller = self._controllers.pop(alias, None)
        if controller is not None:
            try:
                controller.stop()
            except Exception as exc:
                print(f"[input] controller {alias} failed to stop: {exc}")

    # -- Transport ----------------------------------------------------

    def start(self) -> None:
        """Start every registered controller and the poll thread.

        Idempotent. If a controller's :meth:`Controller.start` raises,
        the error is logged and the aggregator continues with the rest.
        """
        with self._lock:
            if self._running:
                return
            self._running = True
            controllers = list(self._controllers.items())
        for alias, controller in controllers:
            try:
                controller.start()
            except Exception as exc:
                print(f"[input] controller {alias} failed to start: {exc}")
        self._thread = threading.Thread(
            target=self._run_loop, name="mdma-input", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the poll thread and every registered controller."""
        with self._lock:
            self._running = False
            controllers = list(self._controllers.items())
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
            self._thread = None
        for alias, controller in controllers:
            try:
                controller.stop()
            except Exception as exc:
                print(f"[input] controller {alias} failed to stop: {exc}")

    # -- Polling ------------------------------------------------------

    def poll(self) -> List[InputEvent]:
        """Aggregate poll across every registered controller.

        Events have their ``key`` rewritten to ``"alias.original_key"``
        before being returned and before handlers are dispatched.
        """
        now = self._current_time()
        with self._lock:
            controllers = list(self._controllers.items())
            handlers = list(self._handlers)

        rewritten: List[InputEvent] = []
        for alias, controller in controllers:
            try:
                events = controller.poll()
            except Exception as exc:
                print(f"[input] {alias}.poll() raised: {exc}")
                continue
            for ev in events:
                namespaced = InputEvent(
                    timestamp=ev.timestamp if ev.timestamp else now,
                    kind=ev.kind,
                    key=f"{alias}.{ev.key}",
                    value=ev.value,
                    meta=ev.meta,
                )
                rewritten.append(namespaced)

        for ev in rewritten:
            self._dispatch(ev, handlers)
        return rewritten

    # -- Channel lookup ----------------------------------------------

    def channel(self, path: str) -> InputChannel:
        """Look up a channel by ``"alias.key"`` path.

        Raises :class:`KeyError` on unknown aliases or unknown channel
        keys within a known alias.
        """
        if "." not in path:
            raise KeyError(f"path must be 'alias.key', got {path!r}")
        alias, _, key = path.partition(".")
        with self._lock:
            controller = self._controllers.get(alias)
        if controller is None:
            raise KeyError(f"unknown alias: {alias!r}")
        channels = controller.channels()
        if key not in channels:
            raise KeyError(f"unknown channel {key!r} on controller {alias!r}")
        return channels[key]

    # -- Handler registration -----------------------------------------

    def on(
        self,
        kind: str,
        key: str,
        handler: Callable[[InputEvent], None],
    ) -> HandlerHandle:
        """Register a handler for events matching ``kind`` and ``key``.

        ``key`` is the full ``"alias.original_key"`` path or a glob
        pattern (:mod:`fnmatch` syntax: ``*`` matches anything,
        ``?`` matches a single character, ``[abc]`` matches a set).
        """
        registration = _HandlerRegistration(
            kind=kind, key_pattern=key, handler=handler
        )
        with self._lock:
            self._handlers.append(registration)
        return HandlerHandle(self, registration)

    def _remove_handler(self, registration: _HandlerRegistration) -> None:
        with self._lock:
            try:
                self._handlers.remove(registration)
            except ValueError:
                return

    # -- Internals ----------------------------------------------------

    def _dispatch(
        self,
        event: InputEvent,
        handlers: List[_HandlerRegistration],
    ) -> None:
        for reg in handlers:
            if reg.kind != event.kind:
                continue
            if not fnmatch.fnmatchcase(event.key, reg.key_pattern):
                continue
            try:
                reg.handler(event)
            except Exception as exc:
                # Handler exceptions must not kill the loop.
                print(
                    f"[input] handler for {reg.kind}/{reg.key_pattern} "
                    f"raised: {exc}"
                )

    def _current_time(self) -> float:
        if self._clock is not None:
            try:
                return float(self._clock.now())
            except Exception:
                pass
        return time.monotonic()

    def _run_loop(self) -> None:
        while True:
            with self._lock:
                running = self._running
            if not running:
                break
            try:
                self.poll()
            except Exception as exc:
                print(f"[input] poll error: {exc}")
            self._wake.wait(timeout=self._poll_interval)
            self._wake.clear()


# ---------------------------------------------------------------------------
# ControllerSource (Phase 4 deferred binding, now live)
# ---------------------------------------------------------------------------


class ControllerSource(AutomationSource):
    """Automation source that reads from an :class:`InputChannel`.

    ``controller`` is the :class:`InputController` instance,
    ``path`` is the ``"alias.key"`` string identifying the channel.
    The channel is looked up at construction (which fails fast on
    typos via :class:`KeyError`) and cached.

    ``value_at`` reads the channel every time; it ignores ``t`` because
    the channel itself is the source of truth for the current value.
    """

    def __init__(self, controller: InputController, path: str) -> None:
        self._controller = controller
        self._path = path
        # Fail fast: if the channel doesn't exist at bind time, the
        # KeyError is a better error than silent zeros.
        self._channel: InputChannel = controller.channel(path)

    def value_at(self, t: float) -> float:
        try:
            return float(self._channel.read())
        except Exception:
            return 0.0


__all__ = [
    "Controller",
    "ControllerSource",
    "HandlerHandle",
    "InputChannel",
    "InputController",
    "InputEvent",
]
