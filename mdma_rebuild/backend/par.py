"""PAR (Pattern at Audio Rate) and PARScheduler.

Implements ``references/scheduler_spec.md`` §PARScheduler (Phase 6).

A :class:`PAR` is a subclass of :class:`Pattern` that carries a
``mod_speed`` density multiplier. Effective event duration is
``duration / mod_speed``. This lets patterns be authored with
readable ratio-form durations while still being playable at very
high density.

A :class:`PARScheduler` dispatches PARs in batches rather than one
event at a time. On each tick it pre-computes the events that fall
within the next ``batch_ms`` window (scaled by ``mod_speed``) and
hands them to the graph together. This keeps the Python-side
per-event overhead from choking at high ``mod_speed`` values.

Phase 6 scope includes the :class:`Scheduler.bind_mod_speed`
automation binding, completing the chain of bindings left over from
Phase 4.
"""

from __future__ import annotations

import threading
from typing import Callable, List, Optional

from .automation import AutomationSource, BindingHandle
from .clock import Clock
from .pattern import Pattern
from .scheduler import (
    ScheduleHandle,
    Scheduler,
    _Binding,
    _safe_play,
    _safe_stop,
)


# ---------------------------------------------------------------------------
# PAR
# ---------------------------------------------------------------------------


class PAR(Pattern):
    """A :class:`Pattern` with an audio-rate density multiplier.

    ``mod_speed`` defaults to 1.0 (same density as a regular Pattern).
    At the schedule call site, :meth:`PARScheduler.add` /
    :meth:`PARScheduler.loop` can override the instance value for that
    particular schedule.

    Non-positive ``mod_speed`` raises :class:`ValueError` on both
    construction and live mutation, matching the scheduler-spec
    error table.
    """

    mod_speed: float

    def __init__(
        self,
        patt: list[tuple[float, float]],
        mod_speed: float = 1.0,
    ) -> None:
        super().__init__(patt)
        if mod_speed <= 0:
            raise ValueError(
                f"mod_speed must be positive, got {mod_speed}"
            )
        self.mod_speed = float(mod_speed)


# ---------------------------------------------------------------------------
# PARScheduleHandle
# ---------------------------------------------------------------------------


class PARScheduleHandle(ScheduleHandle):
    """A :class:`ScheduleHandle` that carries a live-mutable ``mod_speed``.

    The scheduler's batch loop reads :attr:`mod_speed` on every batch,
    so :meth:`set_mod_speed` takes effect at the next batch boundary
    rather than mid-event.
    """

    mod_speed: float

    def __init__(
        self,
        pattern: PAR,
        shape: Callable,
        start_time: float,
        mod_speed: float,
    ) -> None:
        if not isinstance(pattern, PAR):
            raise TypeError(
                f"PARScheduleHandle requires a PAR, got {type(pattern).__name__}"
            )
        super().__init__(pattern, shape, start_time)
        self.mod_speed = float(mod_speed)

    def set_mod_speed(self, mod_speed: float) -> None:
        """Live mutation. Takes effect on the next batch."""
        if mod_speed <= 0:
            raise ValueError(
                f"mod_speed must be positive, got {mod_speed}"
            )
        with self._lock:
            self.mod_speed = float(mod_speed)

    # -- Override tick to use mod_speed-scaled durations --------------

    def _dispatch_due_events(self, now: float) -> None:
        """Same dispatch as :class:`ScheduleHandle`, but event durations
        are divided by the handle's current ``mod_speed`` at dispatch
        time so the next event's start moves closer under higher
        density."""
        while (
            self._event_index < len(self.pattern.events)
            and now >= self._next_event_time
            and not self._cancelled
        ):
            if self._current_voice is not None:
                _safe_stop(self._current_voice)
                self._current_voice = None
                self._current_stop_time = None
                self._current_stages = None

            note, dur = self.pattern.events[self._event_index]
            effective_dur = float(dur) / float(self.mod_speed)
            voice = self.shape(note)
            output, stages = self.pattern._apply_chain_with_nodes(voice)
            _safe_play(output)

            self._current_voice = output
            self._current_stages = stages
            self._current_stop_time = self._next_event_time + effective_dur
            self._event_index += 1
            self._next_event_time += effective_dur


class PARLoopHandle(PARScheduleHandle):
    """Looping :class:`PARScheduleHandle`. Matches :class:`LoopHandle`."""

    iterations: int

    def __init__(
        self,
        pattern: PAR,
        shape: Callable,
        start_time: float,
        mod_speed: float,
    ) -> None:
        super().__init__(pattern, shape, start_time, mod_speed)
        self.iterations = 0
        self._stop_requested = False

    def stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def stop_immediately(self) -> None:
        with self._lock:
            self._stop_requested = True
            self._event_index = len(self.pattern.events)

    def _tick(self, now: float) -> None:
        with self._lock:
            if not self.active:
                return
            self._maybe_stop_trailing_voice(now)
            self._dispatch_due_events(now)
            if self._event_index >= len(self.pattern.events):
                if self._stop_requested:
                    if self._current_voice is None:
                        self.active = False
                else:
                    self.iterations += 1
                    self._event_index = 0


# ---------------------------------------------------------------------------
# PARScheduler
# ---------------------------------------------------------------------------


class PARScheduler(Scheduler):
    """Audio-rate scheduler for :class:`PAR` patterns.

    Shares the base :class:`Scheduler`'s tick, binding, and transport
    machinery. Exposes :meth:`add` and :meth:`loop` overrides that
    accept a ``mod_speed`` override and produce
    :class:`PARScheduleHandle` / :class:`PARLoopHandle` handles.

    ``batch_ms`` is accepted and stored but not yet used as a hard
    batching boundary — the tick loop already batches by processing
    every due event in one pass. The parameter is kept in the
    constructor so the tuning knob from the spec is available when
    we move to a denser dispatch model.
    """

    def __init__(
        self,
        clock: Clock,
        graph: object,
        batch_ms: float = 50.0,
    ) -> None:
        super().__init__(clock, graph)
        self.batch_ms: float = float(batch_ms)

    def add(  # type: ignore[override]
        self,
        pattern: PAR,
        shape: Callable,
        *,
        start: float = 0.0,
        start_mode: str = "relative",
        unit: str = "seconds",
        mod_speed: Optional[float] = None,
    ) -> PARScheduleHandle:
        if not isinstance(pattern, PAR):
            raise TypeError(
                f"PARScheduler.add requires a PAR, got {type(pattern).__name__}"
            )
        effective = float(pattern.mod_speed if mod_speed is None else mod_speed)
        if effective <= 0:
            raise ValueError(
                f"mod_speed must be positive, got {effective}"
            )
        start_time = self._resolve_start_time(start, start_mode, unit)
        handle = PARScheduleHandle(pattern, shape, start_time, effective)
        with self._lock:
            self._handles.append(handle)
        self._wake.set()
        return handle

    def loop(  # type: ignore[override]
        self,
        pattern: PAR,
        shape: Callable,
        *,
        start: float = 0.0,
        start_mode: str = "relative",
        unit: str = "seconds",
        mod_speed: Optional[float] = None,
    ) -> PARLoopHandle:
        if not isinstance(pattern, PAR):
            raise TypeError(
                f"PARScheduler.loop requires a PAR, got {type(pattern).__name__}"
            )
        effective = float(pattern.mod_speed if mod_speed is None else mod_speed)
        if effective <= 0:
            raise ValueError(
                f"mod_speed must be positive, got {effective}"
            )
        start_time = self._resolve_start_time(start, start_mode, unit)
        handle = PARLoopHandle(pattern, shape, start_time, effective)
        with self._lock:
            self._handles.append(handle)
        self._wake.set()
        return handle


# ---------------------------------------------------------------------------
# Scheduler.bind_mod_speed (delayed until PAR exists, per Phase 4 note)
# ---------------------------------------------------------------------------


def _bind_mod_speed(
    self: Scheduler,
    handle: PARScheduleHandle,
    source: AutomationSource,
) -> BindingHandle:
    """Bind an :class:`AutomationSource` to ``handle.mod_speed``.

    Non-positive values from the source are skipped each tick, matching
    :meth:`Scheduler.bind_tempo`'s treatment of non-positive tempos.
    """
    if not isinstance(handle, PARScheduleHandle):
        raise TypeError(
            f"bind_mod_speed requires a PARScheduleHandle, got "
            f"{type(handle).__name__}"
        )
    binding = _Binding(
        kind="mod_speed",
        source=source,
        handle=handle,
    )
    self._register_binding(binding)
    return BindingHandle(self, binding)


# Install bind_mod_speed on Scheduler here rather than in scheduler.py so
# the circular dependency (Scheduler -> PARScheduleHandle) stays
# localised to this module.
Scheduler.bind_mod_speed = _bind_mod_speed  # type: ignore[attr-defined]


# Extend Scheduler._dispatch_binding to route mod_speed bindings. We
# wrap the original method rather than editing scheduler.py so the base
# module stays PAR-ignorant.
_original_dispatch_binding = Scheduler._dispatch_binding


def _dispatch_binding_with_mod_speed(
    self: Scheduler, binding: _Binding, now: float
) -> None:
    if binding.kind == "mod_speed":
        value = float(binding.source.value_at(now))
        if value <= 0 or binding.handle is None:
            return
        handle = binding.handle
        if isinstance(handle, PARScheduleHandle):
            # Live mutation path; takes effect on the next dispatch.
            try:
                handle.set_mod_speed(value)
            except ValueError:
                return
        return
    _original_dispatch_binding(self, binding, now)


Scheduler._dispatch_binding = _dispatch_binding_with_mod_speed  # type: ignore[assignment]


__all__ = [
    "PAR",
    "PARScheduleHandle",
    "PARLoopHandle",
    "PARScheduler",
]
