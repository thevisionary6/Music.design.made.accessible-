"""Bridge between core.Session and the V2 backend.

Implements ``references/scheduler_spec.md`` §Render "Interop with
existing Session buffer flow" and the SKILL.md Phase 5b deliverable.
Exposes the following methods on Session via :func:`bind_to`:

- ``session.backend_graph()`` — lazy SignalFlow AudioGraph.
- ``session.backend_clock()`` — lazy :class:`Clock` bound to Session.
- ``session.backend_scheduler()`` — lazy :class:`Scheduler` wired to
  the backend graph + clock.
- ``session.play_pattern(pattern, shape)`` — sequential playback via
  :meth:`Pattern.play`.
- ``session.schedule(pattern, shape, **kwargs)`` — delegate to
  :meth:`Scheduler.add`.
- ``session.backend_loop(pattern, shape, **kwargs)`` — delegate to
  :meth:`Scheduler.loop`. (Named ``backend_loop`` to avoid shadowing
  any future ``Session.loop`` command helpers.)
- ``session.render_pattern(pattern, shape, path, **kwargs)`` — offline
  render; also populates ``session.last_buffer`` so downstream effects
  and AI analysis see the rendered audio.

SignalFlow imports are lazy so environments that never touch the
backend (legacy command-only flows, unit tests) pay no import cost.
The new methods coexist with the old buffer-producing methods like
``session.generate_tone``; the numpy-buffer contract is untouched.
"""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import numpy as np  # type: ignore

from ..backend.clock import Clock
from ..backend.pattern import Pattern
from ..backend.render import render_pattern as _render_pattern
from ..backend.scheduler import LoopHandle, ScheduleHandle, Scheduler

if TYPE_CHECKING:  # pragma: no cover
    from .session import Session


def backend_graph(self: "Session"):
    """Return the session's backend AudioGraph, creating one on demand.

    Lazily initialises a :class:`signalflow.AudioGraph` the first time
    backend playback or render is requested. Subsequent calls return
    the same graph. ``start=False`` so tests and rendering don't boot
    the audio device implicitly; live playback callers can call
    ``graph.start()`` on the returned object.
    """
    graph = getattr(self, "_backend_graph", None)
    if graph is not None:
        return graph
    import signalflow as sf
    graph = sf.AudioGraph(start=False)
    self._backend_graph = graph
    return graph


def attach_backend_graph(self: "Session", graph) -> None:
    """Inject an already-constructed AudioGraph (or test double).

    Useful for tests, for sessions that want to share a graph across
    several Session instances, and for the render path when running
    against an offline graph.
    """
    self._backend_graph = graph


def backend_clock(self: "Session") -> Clock:
    """Return the session's backend Clock, creating one on demand.

    The clock is bound to the Session so ``session.set_bpm(...)`` and
    ``clock.set_tempo(...)`` stay in sync.
    """
    clock = getattr(self, "_backend_clock", None)
    if clock is not None:
        return clock
    clock = Clock(session=self)
    return clock


def backend_scheduler(self: "Session") -> Scheduler:
    """Return the session's backend Scheduler, creating one on demand."""
    sched = getattr(self, "_backend_scheduler", None)
    if sched is not None:
        return sched
    clock = self.backend_clock()
    graph = self.backend_graph()
    sched = Scheduler(clock, graph)
    self._backend_scheduler = sched
    return sched


# -- Playback ------------------------------------------------------------

def play_pattern(self: "Session", pattern: Pattern, shape: Callable) -> None:
    """Play one pattern sequentially via :meth:`Pattern.play`.

    This is the simple single-shot path — use :meth:`schedule` or
    :meth:`backend_loop` for concurrency, live mutation, or looping.
    """
    if not isinstance(pattern, Pattern):
        raise TypeError(
            f"play_pattern requires a Pattern, got {type(pattern).__name__}"
        )
    pattern.play(shape, self.backend_graph())


def schedule(
    self: "Session",
    pattern: Pattern,
    shape: Callable,
    **kwargs,
) -> ScheduleHandle:
    """Schedule a pattern via the session's Scheduler (one-shot)."""
    return self.backend_scheduler().add(pattern, shape, **kwargs)


def backend_loop(
    self: "Session",
    pattern: Pattern,
    shape: Callable,
    **kwargs,
) -> LoopHandle:
    """Loop a pattern via the session's Scheduler."""
    return self.backend_scheduler().loop(pattern, shape, **kwargs)


# -- Render --------------------------------------------------------------

def render_pattern(
    self: "Session",
    pattern: Pattern,
    shape: Callable,
    out_path: str,
    *,
    sample_rate: int = 48000,
    bit_depth: int = 16,
    graph: object | None = None,
) -> str:
    """Render a pattern to ``out_path`` and populate ``session.last_buffer``.

    When ``graph`` is ``None`` the session's backend graph is used.
    For headless / CI flows the caller should inject an offline
    AudioGraph (or test double) here so no live audio device is
    touched.
    """
    if graph is None:
        graph = self.backend_graph()
    buffer = _render_pattern(
        pattern,
        shape,
        out_path,
        graph=graph,
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        return_buffer=True,
    )
    # Promote to float64 to match the legacy buffer contract.
    if buffer is not None:
        self.last_buffer = np.asarray(buffer, dtype=np.float64)
    return out_path


def bind_to(session_cls) -> None:
    """Attach the backend-bridge methods to ``session_cls``.

    Called from ``core/session.py`` after the other helper modules are
    wired up, so the Session class ends up with the full surface
    whether the user loads the backend or not.
    """
    session_cls.backend_graph = backend_graph
    session_cls.attach_backend_graph = attach_backend_graph
    session_cls.backend_clock = backend_clock
    session_cls.backend_scheduler = backend_scheduler
    session_cls.play_pattern = play_pattern
    session_cls.schedule = schedule
    session_cls.backend_loop = backend_loop
    session_cls.render_pattern = render_pattern


__all__ = ["bind_to"]
