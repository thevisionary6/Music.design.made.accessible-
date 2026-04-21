"""Event-rate Scheduler for the MDMA backend.

Implements ``references/scheduler_spec.md`` §Scheduler (Phase 3, no
automation, no PAR). The Scheduler supersedes :meth:`Pattern.play` for
anything involving concurrency, looping, or live mutation. Single-shot
sequential playback without those needs can still use ``Pattern.play``
directly.

Key architectural points (spec-driven):

- **No ``graph.wait(dur)``.** The scheduler polls the :class:`Clock`
  on every tick and fires voices when their start time arrives /
  stops them when their end time arrives. This is what enables
  concurrent dispatch across many handles.
- **Per-handle event cursor.** Each :class:`ScheduleHandle` owns its
  own ``_event_index`` / ``_next_event_time`` / ``_current_voice`` /
  ``_current_stop_time``, so mutations on one handle never affect
  another.
- **Thread-safe mutation.** ``add``, ``loop``, ``cancel``, ``swap``,
  ``stop`` (loop), ``stop_immediately`` (loop), and the scheduler's own
  ``start`` / ``stop`` all take the relevant lock before touching
  state. The dispatch thread takes the same locks before reading.
- **Explicit tick method.** The inner loop is factored into
  :meth:`Scheduler._tick`, which tests call with synthetic clock
  values so they don't have to start real threads or sleep.

Phase 3 scope excludes automation bindings, PAR / PARScheduler, and
render. Those arrive in Phase 4+.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .automation import AutomationSource, BindingHandle
from .clock import Clock
from .pattern import Pattern


# ---------------------------------------------------------------------------
# Handles
# ---------------------------------------------------------------------------


class ScheduleHandle:
    """One scheduled playthrough of a :class:`Pattern`.

    Returned by :meth:`Scheduler.add`. The scheduler's dispatch loop
    drives :meth:`_tick`. Public callers use :meth:`cancel` and
    :meth:`swap`.

    ``active`` is ``True`` while events (or the currently-playing
    voice) are still pending. Once both are exhausted it flips to
    ``False`` and the scheduler stops ticking the handle.
    """

    # Keeps :meth:`Scheduler.stop` from calling stop() on voices that
    # were never played — ``_current_voice`` stays None in that case.
    pattern: Pattern
    start_time: float
    active: bool

    def __init__(
        self,
        pattern: Pattern,
        shape: Callable,
        start_time: float,
    ) -> None:
        if not isinstance(pattern, Pattern):
            raise TypeError(
                f"ScheduleHandle requires a Pattern, got {type(pattern).__name__}"
            )
        self.pattern: Pattern = pattern
        self.shape: Callable = shape
        self.start_time: float = float(start_time)
        self.active: bool = True

        self._lock = threading.RLock()
        self._event_index: int = 0
        self._next_event_time: float = float(start_time)
        self._current_voice: Optional[object] = None
        self._current_stop_time: Optional[float] = None
        # Per-chain-stage node list for the currently-playing voice, or
        # ``None`` if nothing is playing. The scheduler's chain-param
        # automation walks this list to find the live node for a given
        # chain index.
        self._current_stages: Optional[list] = None
        self._cancelled: bool = False

    # -- Public control ------------------------------------------------

    def cancel(self) -> None:
        """Cancel remaining events.

        Events already dispatched continue to their natural end (the
        scheduler will stop the current voice when its duration
        elapses), but no further events fire.
        """
        with self._lock:
            self._cancelled = True
            # Mark all remaining events consumed so _tick short-circuits
            # the dispatch side of the loop; voice teardown is handled
            # by the same tick when stop time arrives.
            self._event_index = len(self.pattern.events)

    def swap(self, new_pattern: Pattern) -> None:
        """Replace the pattern. Takes effect at the next event boundary.

        The currently-playing voice (if any) finishes naturally. After
        it stops, subsequent events come from ``new_pattern`` starting
        at event 0. If ``new_pattern`` is shorter than the remaining
        events, playback ends when ``new_pattern`` is exhausted.

        Raises :class:`TypeError` if ``new_pattern`` is not a Pattern,
        or is a PAR (reserved for the PAR scheduler in Phase 6). PAR
        detection is deferred until Phase 6 adds the PAR class; for
        now any non-Pattern is rejected.
        """
        if not isinstance(new_pattern, Pattern):
            raise TypeError(
                f"swap requires a Pattern, got {type(new_pattern).__name__}"
            )
        # PAR subclass check lands in Phase 6 when the PAR class exists.
        # In the meantime Pattern's own __class__ covers the common case.
        with self._lock:
            self.pattern = new_pattern
            # Next event comes from new_pattern[0]. _next_event_time stays
            # at the current "next boundary" so the swap respects the
            # spec's "at next event boundary" semantics.
            self._event_index = 0
            self._cancelled = False

    # -- Tick (called by Scheduler) ------------------------------------

    def _tick(self, now: float) -> None:
        """Advance one scheduler tick's worth of state at wall time ``now``.

        This is the per-handle half of the dispatch loop. It:

        1. Stops the currently-playing voice if its duration has
           elapsed.
        2. Dispatches the next event(s) whose start time has arrived.
        3. Marks the handle inactive when both events and the trailing
           voice are exhausted.
        """
        with self._lock:
            if not self.active:
                return

            self._maybe_stop_trailing_voice(now)
            self._dispatch_due_events(now)

            # Inactivity check. Both conditions must hold:
            #    - no more events to dispatch, AND
            #    - no voice is currently playing (cancelled voices are
            #      allowed to finish naturally).
            if (
                self._event_index >= len(self.pattern.events)
                and self._current_voice is None
            ):
                self.active = False

    # -- Shared tick helpers (used by ScheduleHandle and LoopHandle) ---

    def _maybe_stop_trailing_voice(self, now: float) -> None:
        """Stop the currently-playing voice if its natural end has arrived."""
        if (
            self._current_voice is not None
            and self._current_stop_time is not None
            and now >= self._current_stop_time
        ):
            _safe_stop(self._current_voice)
            self._current_voice = None
            self._current_stop_time = None
            self._current_stages = None

    def _dispatch_due_events(self, now: float) -> None:
        """Fire every event whose start time has arrived.

        Loops so a late tick catches up instead of slipping further.
        """
        while (
            self._event_index < len(self.pattern.events)
            and now >= self._next_event_time
            and not self._cancelled
        ):
            # If the previous voice's stop time coincides with the next
            # event's start, the earlier branch may not have caught it.
            if self._current_voice is not None:
                _safe_stop(self._current_voice)
                self._current_voice = None
                self._current_stop_time = None
                self._current_stages = None

            note, dur = self.pattern.events[self._event_index]
            voice = self.shape(note)
            output, stages = self.pattern._apply_chain_with_nodes(voice)
            _safe_play(output)

            self._current_voice = output
            self._current_stages = stages
            self._current_stop_time = self._next_event_time + dur
            self._event_index += 1
            self._next_event_time += dur


class LoopHandle(ScheduleHandle):
    """A :class:`ScheduleHandle` that restarts at event 0 on exhaustion.

    Returned by :meth:`Scheduler.loop`. ``iterations`` increments each
    time the pattern restarts. Graceful exit via :meth:`stop` lets the
    current iteration finish; :meth:`stop_immediately` cancels the
    remaining events in the current iteration but still lets the
    currently-playing voice finish naturally, matching the spec.
    """

    iterations: int

    def __init__(
        self,
        pattern: Pattern,
        shape: Callable,
        start_time: float,
    ) -> None:
        super().__init__(pattern, shape, start_time)
        self.iterations: int = 0
        self._stop_requested: bool = False

    # -- Public control ------------------------------------------------

    def stop(self) -> None:
        """Stop looping at the end of the current iteration (graceful)."""
        with self._lock:
            self._stop_requested = True

    def stop_immediately(self) -> None:
        """Stop looping and cancel remaining events in the current iteration.

        The currently-playing voice still finishes naturally.
        """
        with self._lock:
            self._stop_requested = True
            # Cancel the rest of this iteration's events.
            self._event_index = len(self.pattern.events)

    # -- Tick ----------------------------------------------------------

    def _tick(self, now: float) -> None:
        with self._lock:
            if not self.active:
                return

            self._maybe_stop_trailing_voice(now)
            self._dispatch_due_events(now)

            # Loop or retire based on exhaustion + stop request.
            if self._event_index >= len(self.pattern.events):
                if self._stop_requested:
                    if self._current_voice is None:
                        self.active = False
                    # else: wait for the trailing voice to finish in a
                    # future tick, then flip inactive.
                else:
                    # Restart at event 0 for the next iteration. The
                    # currently-playing voice (last event of the previous
                    # iteration) will be stopped by its own stop_time
                    # branch next tick.
                    self.iterations += 1
                    self._event_index = 0
                    # _next_event_time is already the moment the last
                    # event of the previous iteration ends, which is
                    # also the intended start of the new iteration's
                    # first event.


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class Scheduler:
    """Event-rate pattern dispatcher.

    Construct with a :class:`Clock` and an object that satisfies the
    SignalFlow ``AudioGraph`` duck type (the scheduler stores the
    graph but does not call ``graph.wait`` — the clock replaces it).

    :meth:`add` schedules a pattern for one-shot playback.
    :meth:`loop` schedules a continuously-looping pattern. Both return
    handles that accept live mutation.
    """

    def __init__(self, clock: Clock, graph: object) -> None:
        self._clock: Clock = clock
        self._graph: object = graph

        self._lock = threading.RLock()
        self._handles: List[ScheduleHandle] = []
        self._bindings: List["_Binding"] = []
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._tick_interval: float = 0.001  # 1 ms — plenty of headroom
        self._wake = threading.Event()

    # -- Scheduling ---------------------------------------------------

    def add(
        self,
        pattern: Pattern,
        shape: Callable,
        *,
        start: float = 0.0,
        start_mode: str = "relative",
        unit: str = "seconds",
    ) -> ScheduleHandle:
        """Schedule a pattern for one-shot playback."""
        if not isinstance(pattern, Pattern):
            raise TypeError(
                f"Scheduler.add requires a Pattern, got {type(pattern).__name__}"
            )
        start_time = self._resolve_start_time(start, start_mode, unit)
        handle = ScheduleHandle(pattern, shape, start_time)
        with self._lock:
            self._handles.append(handle)
        self._wake.set()
        return handle

    def loop(
        self,
        pattern: Pattern,
        shape: Callable,
        *,
        start: float = 0.0,
        start_mode: str = "relative",
        unit: str = "seconds",
    ) -> LoopHandle:
        """Schedule a pattern to loop continuously."""
        if not isinstance(pattern, Pattern):
            raise TypeError(
                f"Scheduler.loop requires a Pattern, got {type(pattern).__name__}"
            )
        start_time = self._resolve_start_time(start, start_mode, unit)
        handle = LoopHandle(pattern, shape, start_time)
        with self._lock:
            self._handles.append(handle)
        self._wake.set()
        return handle

    # -- Transport -----------------------------------------------------

    def start(self) -> None:
        """Start the dispatch thread. Also starts the clock if stopped.

        No-op if already running.
        """
        with self._lock:
            if self._running:
                return
            self._running = True
        if not self._clock.running:
            self._clock.start()
        self._thread = threading.Thread(
            target=self._run_loop, name="mdma-scheduler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the dispatch thread, cancel pending events, and tear down bindings.

        Currently-playing voices are stopped so the graph returns to
        silence. Safe to call repeatedly.
        """
        with self._lock:
            self._running = False
            handles = list(self._handles)
            bindings = list(self._bindings)
            self._bindings.clear()
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
            self._thread = None
        # Cancel everything and silence any trailing voices.
        for h in handles:
            h.cancel()
            with h._lock:
                if h._current_voice is not None:
                    _safe_stop(h._current_voice)
                    h._current_voice = None
                    h._current_stop_time = None
                    h._current_stages = None
                h.active = False
        # Tear down automation bindings.
        for b in bindings:
            try:
                b.source.stop()
            except Exception:
                pass

    # -- Tick ----------------------------------------------------------

    def _tick(self, now: float) -> None:
        """Advance every active handle and every active binding by one tick.

        Public-but-underscored: tests call this directly with a
        synthetic ``now`` so they don't need real threads. Production
        callers go through :meth:`start` instead.

        Handles are ticked first so chain-param bindings see fresh stage
        nodes if a new event just fired. Binding errors are swallowed
        so one misbehaving source can't take down the dispatch loop.
        """
        with self._lock:
            handles = list(self._handles)
            bindings = list(self._bindings)
        for h in handles:
            h._tick(now)
        for b in bindings:
            try:
                self._dispatch_binding(b, now)
            except Exception:
                pass

    def _dispatch_binding(self, binding: "_Binding", now: float) -> None:
        """Apply one binding's automation output to its target."""
        value = float(binding.source.value_at(now))
        if binding.kind == "chain_param":
            handle = binding.handle
            if handle is None:
                return
            with handle._lock:
                stages = handle._current_stages
                if stages is None:
                    # Nothing playing on this handle right now — nothing
                    # to route the value to. Will resume on the next
                    # event.
                    return
                if binding.chain_index >= len(stages):
                    # Pattern swapped to something shorter; silently
                    # skip rather than blowing up the loop.
                    return
                node = stages[binding.chain_index]
            _safe_set_input(node, binding.param_name, value)
        elif binding.kind == "tempo":
            if value > 0:
                self._clock.set_tempo(value)
        elif binding.kind == "master_volume":
            setter = getattr(self._graph, "set_output_level", None)
            if setter is not None:
                setter(value)
            else:
                # Fall back to attribute assignment so graphs without a
                # setter still surface the value for inspection.
                try:
                    self._graph.output_level = value  # type: ignore[attr-defined]
                except AttributeError:
                    pass

    def _run_loop(self) -> None:
        while True:
            with self._lock:
                running = self._running
            if not running:
                break
            self._tick(self._clock.now())
            # Sleep-with-wake-up so ``add`` / ``stop`` can nudge the
            # thread without waiting for the next tick interval.
            self._wake.wait(timeout=self._tick_interval)
            self._wake.clear()

    # -- Automation bindings ------------------------------------------

    def bind_chain_param(
        self,
        handle: ScheduleHandle,
        chain_index: int,
        param_name: str,
        source: AutomationSource,
    ) -> BindingHandle:
        """Bind an :class:`AutomationSource` to one chain entry's parameter.

        ``chain_index`` must be a valid index into ``handle.pattern.chain``
        at bind time (later swaps that shorten the pattern will make
        out-of-range indices silently skip). ``param_name`` is the
        SignalFlow node's input name — typical values are ``"cutoff"``,
        ``"resonance"``, or whatever a custom DSP callable exposes via
        ``set_input``.

        Raises :class:`IndexError` for out-of-range indices, matching
        the scheduler-spec error table.
        """
        chain_len = len(handle.pattern.chain)
        if chain_index < 0 or chain_index >= chain_len:
            raise IndexError(
                f"chain_index {chain_index} out of range for "
                f"pattern with {chain_len} chain entries"
            )
        binding = _Binding(
            kind="chain_param",
            source=source,
            handle=handle,
            chain_index=chain_index,
            param_name=param_name,
        )
        self._register_binding(binding)
        return BindingHandle(self, binding)

    def bind_tempo(self, source: AutomationSource) -> BindingHandle:
        """Bind an :class:`AutomationSource` to the clock's tempo.

        The source's output is pushed to :meth:`Clock.set_tempo` every
        tick. Non-positive values are skipped (Clock.set_tempo would
        raise, which would then surface as the binding being silently
        dropped for that tick).
        """
        binding = _Binding(kind="tempo", source=source)
        self._register_binding(binding)
        return BindingHandle(self, binding)

    def bind_master_volume(self, source: AutomationSource) -> BindingHandle:
        """Bind an :class:`AutomationSource` to the graph's master output.

        Calls ``graph.set_output_level(value)`` if available, else falls
        back to assigning ``graph.output_level``. Users with a graph
        type that exposes neither can wire their own hook.
        """
        binding = _Binding(kind="master_volume", source=source)
        self._register_binding(binding)
        return BindingHandle(self, binding)

    def _register_binding(self, binding: "_Binding") -> None:
        binding.source.start(self._clock.now())
        with self._lock:
            self._bindings.append(binding)
        self._wake.set()

    def _unbind_binding(self, binding: "_Binding") -> None:
        """Remove ``binding`` if still registered; called by BindingHandle."""
        with self._lock:
            try:
                self._bindings.remove(binding)
            except ValueError:
                return
        try:
            binding.source.stop()
        except Exception:
            pass

    # -- Helpers -------------------------------------------------------

    def _resolve_start_time(
        self, start: float, start_mode: str, unit: str
    ) -> float:
        if unit == "beats":
            start = self._clock.beats_to_seconds(start)
        elif unit != "seconds":
            raise ValueError(
                f"unit must be 'seconds' or 'beats', got {unit!r}"
            )
        if start_mode == "relative":
            return self._clock.now() + float(start)
        if start_mode == "absolute":
            return float(start)
        raise ValueError(
            f"start_mode must be 'relative' or 'absolute', got {start_mode!r}"
        )


# ---------------------------------------------------------------------------
# Safe play / stop wrappers
# ---------------------------------------------------------------------------

def _safe_play(node: object) -> None:
    """Call ``node.play()`` and swallow exceptions.

    In a live scheduler context, a single misbehaving voice must not
    take down the whole dispatch loop. The handle's own state stays
    consistent regardless of whether the SignalFlow call succeeded.
    """
    try:
        node.play()  # type: ignore[attr-defined]
    except Exception:
        pass


def _safe_stop(node: object) -> None:
    """Call ``node.stop()`` and swallow exceptions (see :func:`_safe_play`)."""
    try:
        node.stop()  # type: ignore[attr-defined]
    except Exception:
        pass


def _safe_set_input(node: object, param_name: str, value: float) -> None:
    """Push an automation value into a node.

    Prefers SignalFlow's ``set_input(name, value)`` so live parameters
    update cleanly. Falls back to attribute assignment for test doubles
    or custom nodes that don't mimic the SignalFlow Node API. Exceptions
    are swallowed so a dead node can't kill the dispatch loop.
    """
    setter = getattr(node, "set_input", None)
    try:
        if callable(setter):
            setter(param_name, value)
            return
        setattr(node, param_name, value)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Binding record
# ---------------------------------------------------------------------------


@dataclass
class _Binding:
    """Private record for one automation binding.

    Kept as a plain dataclass rather than a class hierarchy so the
    scheduler's dispatch code can pattern-match on ``kind`` without
    virtual dispatch through source-owned objects. The :class:`BindingHandle`
    returned to the caller is the public face.
    """

    kind: str  # "chain_param" | "tempo" | "master_volume" | "mod_speed"
    source: AutomationSource
    handle: Optional[ScheduleHandle] = None
    chain_index: int = -1
    param_name: str = ""


__all__ = ["Scheduler", "ScheduleHandle", "LoopHandle"]
