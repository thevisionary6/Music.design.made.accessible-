"""Shared clock for the MDMA backend scheduler.

Implements ``references/scheduler_spec.md`` §Clock (Phase 3).

The clock is seconds-based internally. BPM is a convenience layer on top:
``beats_to_seconds`` converts, and ``set_tempo`` updates the conversion
factor. Changing tempo mid-playback does not retroactively re-time
already-scheduled events — only the conversion for events computed
*after* the change is affected.

Session integration is bidirectional: when the clock is constructed
with a ``session=`` reference, it reads its initial tempo from
``session.bpm`` and :meth:`set_tempo` writes back to ``session.bpm``.
The matching path in reverse is :meth:`Session.set_bpm` (added during
the Phase 3 merge), which calls :meth:`set_tempo` on the bound clock.
Neither call recurses because both paths mutate the backing float
directly rather than re-invoking the other setter.

Standalone usage (no Session, for tests and independent consumers) just
constructs the clock with its own tempo.

The clock is thread-safe: :meth:`set_tempo`, :meth:`start`,
:meth:`stop`, and :meth:`now` are all protected by an internal lock and
safe to call from the scheduler's dispatch thread or any other thread.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional


class Clock:
    """Monotonic clock with a BPM-backed ``beats_to_seconds`` converter.

    Construct in one of two ways:

    - ``Clock()`` — standalone, defaults to 120 BPM.
    - ``Clock(session=session)`` — bound to an MDMA Session; initial
      tempo comes from ``session.bpm`` and :meth:`set_tempo` syncs
      ``session.bpm`` in both directions.

    ``time_fn`` lets tests inject a synthetic time source (defaults to
    :func:`time.monotonic`). The production path never touches this.
    """

    def __init__(
        self,
        tempo: float = 120.0,
        session: Optional[object] = None,
        time_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        if session is not None:
            tempo = float(getattr(session, "bpm", tempo))
        if tempo <= 0:
            raise ValueError(f"tempo must be positive, got {tempo}")

        self._lock = threading.RLock()
        self._tempo: float = float(tempo)
        self._session = session
        self._time_fn: Callable[[], float] = time_fn or time.monotonic
        self._start_wall: Optional[float] = None
        self._running: bool = False

        if session is not None:
            # Seed session.bpm so both directions agree from t=0, and
            # stash a back-reference so ``session.set_bpm(...)`` can find
            # the clock in the opposite direction of the sync.
            try:
                session.bpm = float(tempo)
            except AttributeError:
                pass
            try:
                session._clock = self
            except AttributeError:
                pass

    # -- Public state --------------------------------------------------

    @property
    def tempo(self) -> float:
        """Current tempo in BPM. Always positive."""
        with self._lock:
            return self._tempo

    @property
    def running(self) -> bool:
        """True between :meth:`start` and :meth:`stop`."""
        with self._lock:
            return self._running

    # -- Transport -----------------------------------------------------

    def start(self) -> None:
        """Start the clock. No-op if already running.

        ``now()`` measures seconds from the most recent ``start()``
        call; subsequent ``stop()`` + ``start()`` resets the origin.
        """
        with self._lock:
            if self._running:
                return
            self._start_wall = self._time_fn()
            self._running = True

    def stop(self) -> None:
        """Stop the clock. :meth:`now` returns 0.0 while stopped."""
        with self._lock:
            self._running = False
            self._start_wall = None

    def now(self) -> float:
        """Absolute time in seconds since the most recent :meth:`start`.

        Returns 0.0 while the clock is stopped, so callers that check
        ``clock.now() >= target_time`` degrade cleanly on a stopped
        clock rather than raising.
        """
        with self._lock:
            if not self._running or self._start_wall is None:
                return 0.0
            return float(self._time_fn() - self._start_wall)

    # -- Tempo ---------------------------------------------------------

    def set_tempo(self, bpm: float) -> None:
        """Thread-safe tempo change. Takes effect on the next conversion.

        When bound to a Session, also writes ``session.bpm`` so
        downstream code that still reads the attribute sees the new
        value. The session-side mutation is a direct attribute write so
        this call does not recurse through :meth:`Session.set_bpm`.

        Raises :class:`ValueError` for non-positive values, matching the
        scheduler-spec error table.
        """
        if bpm <= 0:
            raise ValueError(f"tempo must be positive, got {bpm}")
        with self._lock:
            self._tempo = float(bpm)
            if self._session is not None:
                try:
                    self._session.bpm = float(bpm)
                except AttributeError:
                    # Session without a writable ``bpm`` attribute is
                    # legal — callers that pass arbitrary objects get
                    # no sync, and that's the contract.
                    pass

    def beats_to_seconds(self, beats: float) -> float:
        """Convert a duration in beats to seconds at the current tempo.

        ``beats * (60.0 / tempo)``.
        """
        with self._lock:
            return float(beats) * (60.0 / self._tempo)


__all__ = ["Clock"]
