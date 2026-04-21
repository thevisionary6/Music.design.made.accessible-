"""Tests for Phase 5: render helpers + Session integration.

Uses a `_NullGraph` test double so nothing here needs a real
SignalFlow AudioGraph. The render helpers degrade to silence when the
graph doesn't expose ``render_subgraph``, which is exactly what we
want for CI.

Run with::

    python -m unittest tests.test_render
"""

from __future__ import annotations

import os
import tempfile
import unittest
import wave

import numpy as np

from mdma_rebuild.backend.pattern import Pattern
from mdma_rebuild.backend.render import (
    produce_pattern_buffer,
    produce_schedule_buffer,
    render_pattern,
    render_schedule,
    write_wav,
)
from mdma_rebuild.backend.scheduler import Scheduler


class _NullGraph:
    """Graph stub without ``render_subgraph`` — render helpers produce
    silence against it, which is enough to exercise the control flow."""

    pass


class _RecordingVoice:
    def __init__(self, note: float) -> None:
        self.note = note
        self.events: list[str] = []

    def play(self) -> None:
        self.events.append("play")

    def stop(self) -> None:
        self.events.append("stop")


# ---------------------------------------------------------------------------
# produce_pattern_buffer
# ---------------------------------------------------------------------------


class TestProducePatternBuffer(unittest.TestCase):
    def test_empty_pattern_yields_empty_buffer(self):
        buf = produce_pattern_buffer(
            Pattern([]), lambda n: _RecordingVoice(n), _NullGraph(), sample_rate=48000
        )
        self.assertEqual(buf.size, 0)

    def test_length_sums_to_event_durations_in_samples(self):
        p = Pattern([(60, 0.5), (62, 0.25)])
        buf = produce_pattern_buffer(
            p, lambda n: _RecordingVoice(n), _NullGraph(), sample_rate=48000
        )
        expected = int(round(0.5 * 48000)) + int(round(0.25 * 48000))
        self.assertEqual(buf.shape, (expected,))
        # Silence because _NullGraph has no render_subgraph.
        self.assertTrue(np.all(buf == 0.0))

    def test_shape_called_per_event(self):
        shape_calls: list[float] = []

        def shape(note):
            shape_calls.append(note)
            return _RecordingVoice(note)

        p = Pattern([(60, 0.1), (64, 0.1), (67, 0.1)])
        produce_pattern_buffer(p, shape, _NullGraph(), sample_rate=48000)
        self.assertEqual(shape_calls, [60.0, 64.0, 67.0])


# ---------------------------------------------------------------------------
# write_wav
# ---------------------------------------------------------------------------


class TestWriteWav(unittest.TestCase):
    def test_roundtrips_16bit(self):
        # Deterministic signal so we can verify round-trip.
        rng = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.wav")
            write_wav(path, rng, sample_rate=48000, bit_depth=16)
            with wave.open(path, "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), 48000)
                self.assertEqual(wf.getnframes(), 100)

    def test_24bit_produces_larger_file(self):
        signal = np.zeros(100, dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out24.wav")
            write_wav(path, signal, sample_rate=48000, bit_depth=24)
            with wave.open(path, "rb") as wf:
                self.assertEqual(wf.getsampwidth(), 3)

    def test_rejects_bad_bit_depth(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.wav")
            with self.assertRaises(ValueError):
                write_wav(path, np.zeros(10, dtype=np.float32), 48000, bit_depth=32)

    def test_rejects_non_mono(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.wav")
            with self.assertRaises(ValueError):
                write_wav(path, np.zeros((100, 2), dtype=np.float32), 48000)


# ---------------------------------------------------------------------------
# render_pattern
# ---------------------------------------------------------------------------


class TestRenderPattern(unittest.TestCase):
    def test_writes_file_and_optionally_returns_buffer(self):
        p = Pattern([(60, 0.25), (64, 0.25)])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.wav")
            result = render_pattern(
                p,
                lambda n: _RecordingVoice(n),
                path,
                graph=_NullGraph(),
                sample_rate=48000,
                bit_depth=16,
                return_buffer=True,
            )
            self.assertTrue(os.path.exists(path))
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.size, int(round(0.5 * 48000)))

    def test_rejects_non_pattern(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.wav")
            with self.assertRaises(TypeError):
                render_pattern(
                    "not a pattern",
                    lambda n: _RecordingVoice(n),
                    path,
                    graph=_NullGraph(),
                )


# ---------------------------------------------------------------------------
# render_schedule
# ---------------------------------------------------------------------------


class TestRenderSchedule(unittest.TestCase):
    def _make_sched(self):
        from mdma_rebuild.backend.clock import Clock

        clock = Clock()
        clock.start()
        return Scheduler(clock, _NullGraph()), clock

    def test_renders_duration_of_schedule(self):
        sched, clock = self._make_sched()
        p = Pattern([(60, 0.25), (62, 0.25)])
        sched.add(p, lambda n: _RecordingVoice(n))

        buf = produce_schedule_buffer(
            sched, duration=0.5, sample_rate=48000, tick_interval=0.05
        )
        # Duration * sample_rate, within tick rounding.
        expected_min = int(round(0.45 * 48000))
        self.assertGreaterEqual(buf.size, expected_min)

    def test_rejects_running_scheduler(self):
        sched, _clock = self._make_sched()
        sched._running = True
        with self.assertRaises(RuntimeError):
            produce_schedule_buffer(sched, duration=0.5)

    def test_rejects_non_positive_duration(self):
        sched, _clock = self._make_sched()
        with self.assertRaises(ValueError):
            produce_schedule_buffer(sched, duration=0)

    def test_render_schedule_writes_wav(self):
        sched, _clock = self._make_sched()
        p = Pattern([(60, 0.1)])
        sched.add(p, lambda n: _RecordingVoice(n))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.wav")
            render_schedule(
                sched,
                path,
                duration=0.2,
                sample_rate=48000,
                tick_interval=0.05,
            )
            self.assertTrue(os.path.exists(path))


# ---------------------------------------------------------------------------
# Session integration
# ---------------------------------------------------------------------------


class TestSessionBackendBridge(unittest.TestCase):
    def _session(self):
        from mdma_rebuild.core.session import Session

        session = Session()
        session.attach_backend_graph(_NullGraph())
        return session

    def test_play_pattern_rejects_non_pattern(self):
        session = self._session()
        with self.assertRaises(TypeError):
            session.play_pattern("not a pattern", lambda n: _RecordingVoice(n))

    def test_play_pattern_sequential_with_empty_chain(self):
        session = self._session()

        # Pattern.play calls graph.wait(dur); _NullGraph doesn't have it.
        # Add a trivial wait for the test double.
        session._backend_graph.wait = lambda _dur: None

        voices: list[_RecordingVoice] = []

        def shape(n):
            v = _RecordingVoice(n)
            voices.append(v)
            return v

        p = Pattern([(60, 0.01), (62, 0.01)])
        session.play_pattern(p, shape)
        self.assertEqual([v.note for v in voices], [60.0, 62.0])
        for v in voices:
            self.assertEqual(v.events, ["play", "stop"])

    def test_backend_scheduler_is_lazy_singleton(self):
        session = self._session()
        a = session.backend_scheduler()
        b = session.backend_scheduler()
        self.assertIs(a, b)

    def test_render_pattern_populates_last_buffer(self):
        session = self._session()
        p = Pattern([(60, 0.1)])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.wav")
            session.render_pattern(
                p,
                lambda n: _RecordingVoice(n),
                path,
                sample_rate=48000,
                bit_depth=16,
            )
            self.assertTrue(os.path.exists(path))
            self.assertIsNotNone(session.last_buffer)
            self.assertEqual(
                session.last_buffer.size, int(round(0.1 * 48000))
            )
            self.assertEqual(session.last_buffer.dtype, np.float64)

    def test_schedule_and_backend_loop_return_handles(self):
        session = self._session()
        p = Pattern([(60, 0.1)])
        h1 = session.schedule(p, lambda n: _RecordingVoice(n))
        h2 = session.backend_loop(p, lambda n: _RecordingVoice(n))
        from mdma_rebuild.backend.scheduler import (
            LoopHandle,
            ScheduleHandle,
        )

        self.assertIsInstance(h1, ScheduleHandle)
        self.assertIsInstance(h2, LoopHandle)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
