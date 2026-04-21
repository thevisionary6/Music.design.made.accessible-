"""Automation sources for the MDMA backend scheduler.

Implements ``references/scheduler_spec.md`` §Automation (Phase 4).

Every source implements the three-method contract in the spec:

- ``value_at(t)`` — produce the source's value at absolute time ``t``
  (seconds, measured against the Clock).
- ``start(t0)`` — bind the source to an origin time; the scheduler
  calls this when the binding is registered.
- ``stop()`` — tear down any resources; the scheduler calls this on
  :meth:`BindingHandle.unbind` or :meth:`Scheduler.stop`.

This module intentionally stays SignalFlow-free: sources produce plain
floats. The scheduler is the only component that knows what to do with
those floats (push them onto live nodes, the clock, or the graph's
master level). Phase 7 will add :class:`ControllerSource`, which pulls
from an :class:`InputChannel` instead of generating its own values.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class AutomationSource:
    """Base class for time-varying value sources.

    Concrete subclasses override :meth:`value_at`. ``start`` and ``stop``
    default to no-ops so trivial sources (``Constant``) don't have to
    implement them.
    """

    def value_at(self, t: float) -> float:
        """Return the source's value at absolute time ``t`` (seconds)."""
        raise NotImplementedError

    def start(self, t0: float) -> None:
        """Called when the source is bound and the scheduler is running."""
        return None

    def stop(self) -> None:
        """Called when the binding is removed or the scheduler stops."""
        return None


# ---------------------------------------------------------------------------
# Concrete sources
# ---------------------------------------------------------------------------


class Constant(AutomationSource):
    """Fixed value. Useful for testing and for holding a parameter steady
    while a binding is active."""

    def __init__(self, value: float) -> None:
        self.value: float = float(value)

    def value_at(self, t: float) -> float:
        return self.value


class LFO(AutomationSource):
    """Low-frequency oscillator.

    ``shape`` is one of ``"sine"``, ``"saw"``, ``"square"``, ``"triangle"``.
    ``rate`` is in Hz (cycles per second). ``depth`` scales the output and
    ``offset`` shifts it, so e.g. ``LFO("sine", 2.0, depth=400, offset=800)``
    oscillates between 400 and 1200 Hz.

    Output range (before ``depth`` / ``offset``):

    - sine/saw/triangle: ``[-1, 1]``
    - square: ``{-1, +1}``
    """

    SHAPES = ("sine", "saw", "square", "triangle")

    def __init__(
        self,
        shape: str = "sine",
        rate: float = 1.0,
        depth: float = 1.0,
        offset: float = 0.0,
    ) -> None:
        if shape not in self.SHAPES:
            raise ValueError(
                f"LFO shape must be one of {list(self.SHAPES)}, got {shape!r}"
            )
        if rate <= 0:
            raise ValueError(f"LFO rate must be positive, got {rate}")
        self.shape: str = shape
        self.rate: float = float(rate)
        self.depth: float = float(depth)
        self.offset: float = float(offset)
        self._t0: float = 0.0

    def start(self, t0: float) -> None:
        self._t0 = float(t0)

    def value_at(self, t: float) -> float:
        phase = ((t - self._t0) * self.rate) % 1.0
        if self.shape == "sine":
            raw = math.sin(2.0 * math.pi * phase)
        elif self.shape == "saw":
            raw = 2.0 * phase - 1.0
        elif self.shape == "square":
            raw = 1.0 if phase < 0.5 else -1.0
        else:  # triangle — start at 0 rising, peak at 0.25, trough at 0.75
            raw = 4.0 * abs(((phase + 0.75) % 1.0) - 0.5) - 1.0
        return self.offset + self.depth * raw


class Ramp(AutomationSource):
    """Linear interpolation from ``start`` to ``end`` over ``duration``.

    Values before ``start(t0)`` return ``start``; values after
    ``t0 + duration`` return ``end``. This makes ramps safe to bind in
    advance — the output never over-extrapolates.
    """

    def __init__(self, start: float, end: float, duration: float) -> None:
        if duration <= 0:
            raise ValueError(
                f"Ramp duration must be positive, got {duration}"
            )
        self.start_val: float = float(start)
        self.end_val: float = float(end)
        self.duration: float = float(duration)
        self._t0: float = 0.0

    def start(self, t0: float) -> None:
        self._t0 = float(t0)

    def value_at(self, t: float) -> float:
        elapsed = t - self._t0
        if elapsed <= 0:
            return self.start_val
        if elapsed >= self.duration:
            return self.end_val
        frac = elapsed / self.duration
        return self.start_val + (self.end_val - self.start_val) * frac


class Envelope(AutomationSource):
    """Breakpoint envelope with linear interpolation between points.

    ``points`` is a list of ``(time, value)`` pairs. ``time`` is relative
    to the envelope's start (see :meth:`start`). Points are sorted by
    time at construction so callers don't have to keep them in order.

    Behaviour outside the envelope:

    - Before the first point: clamp to that point's value.
    - After the last point: clamp to that point's value.

    Empty ``points`` raises :class:`ValueError` because the envelope
    would be ill-defined.
    """

    def __init__(self, points: Sequence) -> None:
        if not points:
            raise ValueError("Envelope requires at least one breakpoint")
        sorted_points = sorted(
            (float(t), float(v)) for (t, v) in points
        )
        self.points: list[tuple[float, float]] = sorted_points
        self._t0: float = 0.0

    def start(self, t0: float) -> None:
        self._t0 = float(t0)

    def value_at(self, t: float) -> float:
        elapsed = t - self._t0
        pts = self.points
        if elapsed <= pts[0][0]:
            return pts[0][1]
        if elapsed >= pts[-1][0]:
            return pts[-1][1]
        # Linear search — envelopes are typically short. If a caller
        # needs a big one, swap for bisect.
        for i in range(len(pts) - 1):
            t_lo, v_lo = pts[i]
            t_hi, v_hi = pts[i + 1]
            if t_lo <= elapsed <= t_hi:
                frac = (elapsed - t_lo) / (t_hi - t_lo)
                return v_lo + (v_hi - v_lo) * frac
        return pts[-1][1]  # unreachable, for type checker


# ---------------------------------------------------------------------------
# BindingHandle
# ---------------------------------------------------------------------------


class BindingHandle:
    """Opaque handle returned by :meth:`Scheduler.bind_*`.

    Callers hold the handle and call :meth:`unbind` to remove the
    binding. The scheduler keeps its own reference so it can clean up
    on ``.stop()``.
    """

    def __init__(self, scheduler, binding) -> None:
        self._scheduler = scheduler
        self._binding = binding

    def unbind(self) -> None:
        """Remove this binding from the scheduler.

        No-op if the binding was already removed (e.g. by
        :meth:`Scheduler.stop`).
        """
        self._scheduler._unbind_binding(self._binding)


__all__ = [
    "AutomationSource",
    "BindingHandle",
    "Constant",
    "Envelope",
    "LFO",
    "Ramp",
]
