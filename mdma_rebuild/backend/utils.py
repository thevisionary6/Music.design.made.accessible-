"""Miscellaneous SignalFlow integrations and helpers.

Sits between the Pattern/Scheduler/Effects layer and SignalFlow's
raw node API. Everything here is importable without SignalFlow — the
functions that need it do lazy imports and surface a clear
``RuntimeError`` if SignalFlow is missing.

Exports:

- :func:`note_to_freq` / :func:`freq_to_note` — MIDI <-> Hz conversion.
- :func:`build_default_shape` — returns a shape callable suitable for
  ``Pattern.play(shape, graph)`` / ``Session.play_pattern``. When a
  Session is supplied, the shape reads the monolith's current
  configuration and builds a full voice; without a Session, a simple
  :class:`signalflow.SawOscillator` is returned.
- :func:`build_saw_shape` / :func:`build_sine_shape` — convenience
  default shapes for scripting and tests.
- :func:`audiograph` — context-manager style lifecycle for a shared
  :class:`signalflow.AudioGraph` that auto-stops on exit.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Note conversions (no SignalFlow needed)
# ---------------------------------------------------------------------------


def note_to_freq(note: float, a4_hz: float = 440.0) -> float:
    """Convert a MIDI note number (float, so microtones work) to Hz.

    ``note=69`` is A4 which returns ``a4_hz`` (440 Hz by default).
    """
    return float(a4_hz) * (2.0 ** ((float(note) - 69.0) / 12.0))


def freq_to_note(freq: float, a4_hz: float = 440.0) -> float:
    """Inverse of :func:`note_to_freq`. Returns a float MIDI note."""
    if freq <= 0:
        raise ValueError(f"freq must be positive, got {freq}")
    import math
    return 69.0 + 12.0 * math.log2(float(freq) / float(a4_hz))


# ---------------------------------------------------------------------------
# Sample rate lookup
# ---------------------------------------------------------------------------


def current_sample_rate() -> float:
    """Best-effort sample-rate lookup.

    Prefers the active :class:`signalflow.AudioGraph`, falls back to
    SignalFlow's compile-time default, and finally to 44100.0 so the
    helper stays importable even when SignalFlow is missing
    (important during Phase-0 / Phase-1 -only test runs and for the
    effects modules that want a sensible default without forcing a
    SignalFlow dependency at import time).
    """
    try:
        import signalflow as sf
    except ImportError:
        return 44100.0
    graph = sf.AudioGraph.get_shared_graph()
    if graph is not None:
        return float(graph.sample_rate)
    return float(sf.SIGNALFLOW_DEFAULT_SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Default shape factories
# ---------------------------------------------------------------------------


def build_saw_shape() -> Callable[[float], object]:
    """Return a ``shape(note) -> sf.SawOscillator(note_to_freq(note))``.

    Handy when all you want is a bare saw voice per event — ``/patn``
    uses this when the session has no monolith configuration worth
    reading.
    """
    def _shape(note: float):
        import signalflow as sf
        return sf.SawOscillator(note_to_freq(note))
    return _shape


def build_sine_shape() -> Callable[[float], object]:
    """Return a ``shape(note) -> sf.SineOscillator(note_to_freq(note))``."""
    def _shape(note: float):
        import signalflow as sf
        return sf.SineOscillator(note_to_freq(note))
    return _shape


def build_default_shape(session=None) -> Callable[[float], object]:
    """Return a best-effort default shape for ``Pattern.play``.

    If ``session`` is provided and has a backend-configured monolith
    (detected via the presence of ``session.engine`` and at least
    one operator), the returned shape reads the monolith's state per
    call and produces a full :func:`build_voice` graph — so any
    ``/wm``, ``/fr``, ``/am``, ``/fm`` settings in the REPL take
    effect on the next played note.

    Falls back to :func:`build_saw_shape` when no session is
    supplied or the monolith config is empty.
    """
    if session is None:
        return build_saw_shape()

    engine = getattr(session, "engine", None)
    ops = getattr(engine, "operators", None) if engine is not None else None
    if not ops:
        return build_saw_shape()

    # Late import so backend.monolith's SignalFlow-aware helpers
    # don't pull signalflow at package import time.
    from .monolith import build_voice

    def _shape(note: float):
        return build_voice(note, session)

    return _shape


# ---------------------------------------------------------------------------
# AudioGraph lifecycle
# ---------------------------------------------------------------------------


@contextmanager
def audiograph(start: bool = True):
    """Context manager for a shared :class:`signalflow.AudioGraph`.

    Creates one if none exists, starts it on entry (unless
    ``start=False``), stops it on exit. Useful for scripts that want
    tight control over the live graph without leaving a thread
    behind::

        from mdma_rebuild.backend.utils import audiograph

        with audiograph() as g:
            Pattern([(60, 0.5), (64, 0.5)]).play(shape, g)

    Idempotent on entry: if a shared graph is already live, the
    context manager just yields it and leaves teardown to whoever
    created it.
    """
    import signalflow as sf
    existing = sf.AudioGraph.get_shared_graph()
    if existing is not None:
        yield existing
        return
    graph = sf.AudioGraph(start=start)
    try:
        yield graph
    finally:
        try:
            graph.stop()
        except Exception:
            pass
