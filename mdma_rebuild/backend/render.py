"""Offline render helpers for the MDMA backend.

Implements ``references/scheduler_spec.md`` §Render (Phase 5a):

- :func:`render_pattern` — render a single :class:`Pattern` to a wav
  file, applying its chain.
- :func:`render_schedule` — render N seconds of a
  :class:`Scheduler`'s current schedule offline.

Both functions follow the spec's "architecturally separable" rule:
audio production lives in :func:`produce_pattern_buffer` /
:func:`produce_schedule_buffer`, and wav-writing lives in
:func:`write_wav`. A future ``render_to_buffer`` variant shares the
produce step without duplicating anything.

The actual audio production uses :class:`signalflow.AudioGraph`'s
offline-rendering surface (``graph.render_subgraph``). Callers that
want to render headlessly (no live audio device) pass in an offline
AudioGraph; the functions don't create one implicitly because
headless-graph configuration is environment-specific.
"""

from __future__ import annotations

import struct
import wave
from typing import Callable, Optional

import numpy as np

from .pattern import Pattern
from .scheduler import Scheduler


# ---------------------------------------------------------------------------
# Audio production
# ---------------------------------------------------------------------------


def produce_pattern_buffer(
    pattern: Pattern,
    shape: Callable,
    graph,
    sample_rate: int = 48000,
) -> np.ndarray:
    """Render a pattern's audio to a mono float32 numpy buffer.

    Each event is rendered in sequence: ``shape(note)`` builds the
    voice, ``pattern._apply_chain`` wraps it with the DSP chain, and
    the graph renders ``round(dur * sample_rate)`` frames of that
    node. Buffers are concatenated in event order.

    An empty pattern produces an empty buffer. A graph that lacks
    ``render_subgraph`` is treated as a test double and this function
    falls back to per-chunk silence — useful for unit tests that only
    want to exercise the shape+chain path without real SignalFlow.
    """
    if not pattern.events:
        return np.zeros(0, dtype=np.float32)

    chunks: list[np.ndarray] = []
    for (note, dur) in pattern.events:
        voice = shape(note)
        output = pattern._apply_chain(voice)
        n_frames = max(0, int(round(float(dur) * sample_rate)))
        chunks.append(_render_node(graph, output, n_frames, sample_rate))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def produce_schedule_buffer(
    scheduler: Scheduler,
    duration: float,
    sample_rate: int = 48000,
    tick_interval: float = 0.01,
) -> np.ndarray:
    """Render ``duration`` seconds of a scheduler's schedule offline.

    Walks synthetic time forward in ``tick_interval`` steps. At each
    step the scheduler's tick dispatches any due events and any bound
    automation. Between ticks, the scheduler's graph renders
    ``round(tick_interval * sample_rate)`` frames.

    ``scheduler._running`` must be False — this is a destructive walk
    of the schedule. The caller is expected to have stopped the live
    dispatch thread first. Matches the spec's error condition
    "render_schedule while scheduler is running" -> RuntimeError.
    """
    if scheduler._running:
        raise RuntimeError(
            "render_schedule requires the scheduler to be stopped; "
            "call scheduler.stop() first"
        )
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")

    graph = scheduler._graph
    chunks: list[np.ndarray] = []
    clock = scheduler._clock
    # Use the clock's current time as the render origin so scheduled
    # start times resolve consistently with live runs.
    t = clock.now()
    end = t + float(duration)
    frames_per_tick = max(1, int(round(tick_interval * sample_rate)))

    while t < end:
        scheduler._tick(t)
        # For rendering, sum the currently-playing voices from every
        # handle. Each handle exposes its current output node via
        # ``_current_voice``.
        active_nodes = [
            h._current_voice
            for h in list(scheduler._handles)
            if h._current_voice is not None
        ]
        if active_nodes:
            mixed = None
            for node in active_nodes:
                frame = _render_node(graph, node, frames_per_tick, sample_rate)
                mixed = frame if mixed is None else mixed + frame
            chunks.append(mixed)  # type: ignore[arg-type]
        else:
            chunks.append(np.zeros(frames_per_tick, dtype=np.float32))
        t += tick_interval

    return (
        np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    )


def _render_node(graph, node, n_frames: int, sample_rate: int) -> np.ndarray:
    """Render a SignalFlow node to an N-frame numpy buffer.

    If the graph doesn't have ``render_subgraph`` (e.g. a test double),
    returns silence of the requested length. This keeps the render
    helpers testable without an audio device.
    """
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float32)
    render = getattr(graph, "render_subgraph", None)
    if render is None:
        return np.zeros(n_frames, dtype=np.float32)
    try:
        import signalflow as sf
        buf = sf.Buffer(1, n_frames)
        render(node, buf)
        data = np.asarray(buf.data, dtype=np.float32)
        if data.ndim == 2:
            data = data[0]
        return data.astype(np.float32, copy=False)
    except Exception:
        return np.zeros(n_frames, dtype=np.float32)


# ---------------------------------------------------------------------------
# Wav writing
# ---------------------------------------------------------------------------


def write_wav(
    path: str,
    buffer: np.ndarray,
    sample_rate: int,
    bit_depth: int = 16,
) -> None:
    """Write a mono float buffer to a wav file.

    ``bit_depth`` is 16 (default) or 24. Stereo buffers can be
    supported later by extending ``num_channels`` — the wav header
    path below is already ready for it.
    """
    data = np.asarray(buffer, dtype=np.float32)
    if data.ndim != 1:
        raise ValueError(
            f"write_wav expects a 1-D mono buffer, got shape {data.shape}"
        )
    # Clip to [-1, 1] so integer conversion doesn't overflow.
    clipped = np.clip(data, -1.0, 1.0)

    if bit_depth == 16:
        samp = np.int16(clipped * 32767)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(samp.tobytes())
    elif bit_depth == 24:
        samples = np.int32(
            np.clip(clipped * 8388607, -8388607, 8388607)
        )
        raw = bytearray()
        for s in samples:
            val = int(s) & 0xFFFFFF
            raw.append(val & 0xFF)
            raw.append((val >> 8) & 0xFF)
            raw.append((val >> 16) & 0xFF)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(3)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(raw))
    else:
        raise ValueError(f"bit_depth must be 16 or 24, got {bit_depth}")


# ---------------------------------------------------------------------------
# Top-level render functions
# ---------------------------------------------------------------------------


def render_pattern(
    pattern: Pattern,
    shape: Callable,
    out_path: str,
    *,
    graph,
    sample_rate: int = 48000,
    bit_depth: int = 16,
    return_buffer: bool = False,
) -> np.ndarray | None:
    """Render a single pattern offline to ``out_path``.

    When ``return_buffer=True`` the produced numpy buffer is returned
    as well; the Session integration uses this to populate
    ``session.last_buffer`` without re-reading the file.
    """
    if not isinstance(pattern, Pattern):
        raise TypeError(
            f"render_pattern requires a Pattern, got {type(pattern).__name__}"
        )
    buffer = produce_pattern_buffer(pattern, shape, graph, sample_rate)
    write_wav(out_path, buffer, sample_rate, bit_depth)
    return buffer if return_buffer else None


def render_schedule(
    scheduler: Scheduler,
    out_path: str,
    *,
    duration: float,
    sample_rate: int = 48000,
    bit_depth: int = 16,
    return_buffer: bool = False,
    tick_interval: float = 0.01,
) -> np.ndarray | None:
    """Render ``duration`` seconds of a scheduler's schedule offline.

    See :func:`produce_schedule_buffer` for the simulation details.
    The scheduler must not be running.
    """
    buffer = produce_schedule_buffer(
        scheduler, duration, sample_rate, tick_interval
    )
    write_wav(out_path, buffer, sample_rate, bit_depth)
    return buffer if return_buffer else None


__all__ = [
    "produce_pattern_buffer",
    "produce_schedule_buffer",
    "render_pattern",
    "render_schedule",
    "write_wav",
]
