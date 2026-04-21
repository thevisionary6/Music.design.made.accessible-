"""Unit tests for Clock and Scheduler (Phase 3).

Tests drive the scheduler via :meth:`Scheduler._tick(now)` with a
synthetic ``now`` value so they don't have to start real threads or
sleep. Clock is tested with an injected ``time_fn`` for the same reason.

Run with::

    python -m unittest tests.test_scheduler
"""

from __future__ import annotations

import time as _real_time
import threading
import unittest

from mdma_rebuild.backend.clock import Clock
from mdma_rebuild.backend.pattern import Pattern
from mdma_rebuild.backend.scheduler import (
    LoopHandle,
    ScheduleHandle,
    Scheduler,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeClockTime:
    """Mutable float source for Clock's ``time_fn`` injection."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class _RecordingVoice:
    def __init__(self, note: float) -> None:
        self.note = note
        self.events: list[str] = []

    def play(self) -> None:
        self.events.append("play")

    def stop(self) -> None:
        self.events.append("stop")


class _FakeSession:
    """Minimal Session stand-in that only exposes ``bpm`` + ``_clock``."""

    def __init__(self, bpm: float = 128.0) -> None:
        self.bpm = float(bpm)
        self._clock: object | None = None


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------


class TestClock(unittest.TestCase):
    def test_defaults(self):
        c = Clock()
        self.assertEqual(c.tempo, 120.0)
        self.assertFalse(c.running)
        self.assertEqual(c.now(), 0.0)

    def test_start_stop(self):
        ft = _FakeClockTime(start=100.0)
        c = Clock(time_fn=ft)
        c.start()
        self.assertTrue(c.running)
        ft.now = 102.5
        self.assertAlmostEqual(c.now(), 2.5)
        c.stop()
        self.assertFalse(c.running)
        self.assertEqual(c.now(), 0.0)

    def test_start_is_idempotent(self):
        ft = _FakeClockTime(start=10.0)
        c = Clock(time_fn=ft)
        c.start()
        ft.now = 11.0
        self.assertAlmostEqual(c.now(), 1.0)
        # A second start() must not reset the origin.
        c.start()
        self.assertAlmostEqual(c.now(), 1.0)

    def test_restart_resets_origin(self):
        ft = _FakeClockTime(start=10.0)
        c = Clock(time_fn=ft)
        c.start()
        ft.now = 13.0
        self.assertAlmostEqual(c.now(), 3.0)
        c.stop()
        ft.now = 20.0
        c.start()
        ft.now = 22.0
        self.assertAlmostEqual(c.now(), 2.0)

    def test_beats_to_seconds(self):
        c = Clock(tempo=120.0)
        self.assertAlmostEqual(c.beats_to_seconds(1), 0.5)
        self.assertAlmostEqual(c.beats_to_seconds(4), 2.0)
        c.set_tempo(60.0)
        self.assertAlmostEqual(c.beats_to_seconds(1), 1.0)

    def test_set_tempo_rejects_non_positive(self):
        c = Clock()
        with self.assertRaises(ValueError):
            c.set_tempo(0)
        with self.assertRaises(ValueError):
            c.set_tempo(-120)

    def test_constructor_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            Clock(tempo=0)


class TestClockSessionSync(unittest.TestCase):
    def test_reads_initial_tempo_from_session(self):
        s = _FakeSession(bpm=140.0)
        c = Clock(session=s)
        self.assertEqual(c.tempo, 140.0)
        # The back-reference is installed too so the reverse path works.
        self.assertIs(s._clock, c)

    def test_clock_set_tempo_updates_session_bpm(self):
        s = _FakeSession(bpm=100.0)
        c = Clock(session=s)
        c.set_tempo(90.0)
        self.assertEqual(s.bpm, 90.0)
        self.assertEqual(c.tempo, 90.0)

    def test_session_set_bpm_updates_clock(self):
        # Exercises the Session.set_bpm path added in Phase 3.
        from mdma_rebuild.core.session import Session

        session = Session()
        clock = Clock(session=session)
        session.set_bpm(96.0)
        self.assertEqual(session.bpm, 96.0)
        self.assertEqual(clock.tempo, 96.0)
        # And the reverse direction is still live.
        clock.set_tempo(108.0)
        self.assertEqual(session.bpm, 108.0)

    def test_session_set_bpm_rejects_non_positive(self):
        from mdma_rebuild.core.session import Session

        session = Session()
        with self.assertRaises(ValueError):
            session.set_bpm(0)
        with self.assertRaises(ValueError):
            session.set_bpm(-4)

    def test_session_set_bpm_without_clock_only_touches_attribute(self):
        from mdma_rebuild.core.session import Session

        session = Session()
        session.set_bpm(150.0)
        self.assertEqual(session.bpm, 150.0)


# ---------------------------------------------------------------------------
# Scheduler fixtures
# ---------------------------------------------------------------------------


def _make_scheduler():
    """Build a Scheduler + fake time source + voice recorder."""
    ft = _FakeClockTime(start=0.0)
    clock = Clock(time_fn=ft)
    clock.start()
    voices: list[_RecordingVoice] = []

    def shape(note):
        v = _RecordingVoice(note)
        voices.append(v)
        return v

    sched = Scheduler(clock, graph=object())
    return sched, clock, ft, shape, voices


# ---------------------------------------------------------------------------
# Scheduler.add
# ---------------------------------------------------------------------------


class TestSchedulerAdd(unittest.TestCase):
    def test_add_rejects_non_pattern(self):
        sched, *_ = _make_scheduler()
        with self.assertRaises(TypeError):
            sched.add("not a pattern", lambda n: None)

    def test_one_shot_dispatches_each_event_in_turn(self):
        sched, clock, ft, shape, voices = _make_scheduler()
        p = Pattern([(60, 0.5), (62, 0.25)])
        handle = sched.add(p, shape)

        # At t=0, the first event fires.
        sched._tick(0.0)
        self.assertEqual([v.note for v in voices], [60.0])
        self.assertEqual(voices[0].events, ["play"])

        # Halfway through the first note — nothing new.
        sched._tick(0.25)
        self.assertEqual(len(voices), 1)
        self.assertEqual(voices[0].events, ["play"])

        # At t=0.5, the first voice stops and the second starts.
        sched._tick(0.5)
        self.assertEqual([v.note for v in voices], [60.0, 62.0])
        self.assertEqual(voices[0].events, ["play", "stop"])
        self.assertEqual(voices[1].events, ["play"])

        # At t=0.75, the second voice stops and the handle retires.
        sched._tick(0.75)
        self.assertEqual(voices[1].events, ["play", "stop"])
        self.assertFalse(handle.active)

    def test_empty_pattern_retires_immediately(self):
        sched, *_rest, shape, voices = _make_scheduler()
        handle = sched.add(Pattern([]), shape)
        sched._tick(0.0)
        self.assertEqual(voices, [])
        self.assertFalse(handle.active)

    def test_relative_start_respects_clock_now(self):
        sched, clock, ft, shape, voices = _make_scheduler()
        ft.now = 1.0  # clock has been running for 1 second
        p = Pattern([(60, 0.5)])
        handle = sched.add(p, shape, start=0.25)
        self.assertAlmostEqual(handle.start_time, 1.25)
        sched._tick(1.0)
        self.assertEqual(voices, [])
        sched._tick(1.25)
        self.assertEqual([v.note for v in voices], [60.0])

    def test_absolute_start_ignores_clock_now(self):
        sched, clock, ft, shape, voices = _make_scheduler()
        ft.now = 5.0
        p = Pattern([(60, 0.5)])
        handle = sched.add(
            p, shape, start=2.0, start_mode="absolute"
        )
        self.assertEqual(handle.start_time, 2.0)

    def test_beats_unit_uses_clock_tempo(self):
        sched, clock, *_rest = _make_scheduler()
        clock.set_tempo(120.0)
        p = Pattern([(60, 0.25)])
        shape = lambda n: _RecordingVoice(n)
        handle = sched.add(p, shape, start=2, unit="beats")
        # 2 beats @ 120 BPM = 1.0 second.
        self.assertAlmostEqual(handle.start_time, 1.0)

    def test_invalid_unit_raises(self):
        sched, *_ = _make_scheduler()
        with self.assertRaises(ValueError):
            sched.add(Pattern([(60, 0.25)]), lambda n: None, unit="bogus")

    def test_invalid_start_mode_raises(self):
        sched, *_ = _make_scheduler()
        with self.assertRaises(ValueError):
            sched.add(
                Pattern([(60, 0.25)]),
                lambda n: None,
                start_mode="elsewhere",
            )


# ---------------------------------------------------------------------------
# ScheduleHandle.cancel
# ---------------------------------------------------------------------------


class TestCancel(unittest.TestCase):
    def test_cancel_stops_future_events_but_lets_current_voice_finish(self):
        sched, *_rest, shape, voices = _make_scheduler()
        p = Pattern([(60, 0.5), (62, 0.5), (64, 0.5)])
        handle = sched.add(p, shape)

        sched._tick(0.0)
        self.assertEqual([v.note for v in voices], [60.0])

        handle.cancel()

        # Event 1 (62) is never dispatched.
        sched._tick(0.5)
        # The scheduler will stop voice 0 at t=0.5 per the usual rule.
        self.assertEqual(voices[0].events, ["play", "stop"])
        self.assertEqual(len(voices), 1)

        sched._tick(1.0)
        self.assertEqual(len(voices), 1)
        self.assertFalse(handle.active)


# ---------------------------------------------------------------------------
# ScheduleHandle.swap
# ---------------------------------------------------------------------------


class TestSwap(unittest.TestCase):
    def test_swap_replaces_pattern_at_next_event_boundary(self):
        sched, *_rest, shape, voices = _make_scheduler()
        a = Pattern([(60, 0.5), (62, 0.5), (64, 0.5)])
        b = Pattern([(80, 0.25), (81, 0.25)])
        handle = sched.add(a, shape)

        sched._tick(0.0)  # note 60 fires
        self.assertEqual([v.note for v in voices], [60.0])

        # Swap mid-voice: current voice finishes naturally, then b fires.
        handle.swap(b)
        sched._tick(0.5)  # note 60 stops, note 80 (from b[0]) fires
        self.assertEqual([v.note for v in voices], [60.0, 80.0])

        sched._tick(0.75)  # note 80 stops, note 81 fires
        self.assertEqual([v.note for v in voices], [60.0, 80.0, 81.0])

        sched._tick(1.0)  # note 81 stops, pattern b exhausted
        self.assertEqual(voices[-1].events, ["play", "stop"])
        self.assertFalse(handle.active)

    def test_swap_rejects_non_pattern(self):
        sched, *_ = _make_scheduler()
        handle = sched.add(Pattern([(60, 0.5)]), lambda n: _RecordingVoice(n))
        with self.assertRaises(TypeError):
            handle.swap("not a pattern")


# ---------------------------------------------------------------------------
# Concurrent dispatch
# ---------------------------------------------------------------------------


class TestConcurrent(unittest.TestCase):
    def test_two_patterns_play_concurrently(self):
        sched, *_rest, shape, voices = _make_scheduler()
        a = Pattern([(60, 0.5)])
        b = Pattern([(72, 0.25)])
        ha = sched.add(a, shape)
        hb = sched.add(b, shape)

        sched._tick(0.0)
        # Both handles dispatched their first event at t=0.
        self.assertEqual(sorted(v.note for v in voices), [60.0, 72.0])

        sched._tick(0.25)  # b finishes
        self.assertEqual([v.note for v in voices if v.events == ["play", "stop"]], [72.0])
        self.assertFalse(hb.active)

        sched._tick(0.5)  # a finishes
        self.assertFalse(ha.active)


# ---------------------------------------------------------------------------
# Scheduler.loop / LoopHandle
# ---------------------------------------------------------------------------


class TestLoop(unittest.TestCase):
    def test_loop_restarts_and_counts_iterations(self):
        sched, *_rest, shape, voices = _make_scheduler()
        p = Pattern([(60, 0.5), (62, 0.5)])
        handle = sched.loop(p, shape)
        self.assertIsInstance(handle, LoopHandle)

        sched._tick(0.0)  # iter 0 / event 0 -> note 60
        sched._tick(0.5)  # iter 0 / event 1 -> note 62
        sched._tick(1.0)  # iter 0 exhausted -> iter 1 / event 0 -> note 60
        self.assertEqual([v.note for v in voices], [60.0, 62.0, 60.0])
        self.assertEqual(handle.iterations, 1)

        sched._tick(1.5)  # iter 1 / event 1 -> note 62
        sched._tick(2.0)  # iter 1 exhausted -> iter 2 / event 0
        self.assertEqual(
            [v.note for v in voices], [60.0, 62.0, 60.0, 62.0, 60.0]
        )
        self.assertEqual(handle.iterations, 2)
        self.assertTrue(handle.active)

    def test_graceful_stop_lets_current_iteration_finish(self):
        sched, *_rest, shape, voices = _make_scheduler()
        p = Pattern([(60, 0.5), (62, 0.5)])
        handle = sched.loop(p, shape)

        sched._tick(0.0)  # event 0 -> 60
        handle.stop()  # graceful stop mid-iteration
        sched._tick(0.5)  # event 1 -> 62 (iteration completes)
        sched._tick(1.0)  # iteration exhausted; loop exits
        self.assertEqual([v.note for v in voices], [60.0, 62.0])
        self.assertFalse(handle.active)

    def test_stop_immediately_cancels_remaining_events_in_iter(self):
        sched, *_rest, shape, voices = _make_scheduler()
        p = Pattern([(60, 0.5), (62, 0.5)])
        handle = sched.loop(p, shape)

        sched._tick(0.0)  # event 0 -> 60
        handle.stop_immediately()
        sched._tick(0.5)  # voice 60 stops, but no new voice dispatches
        self.assertEqual([v.note for v in voices], [60.0])
        self.assertEqual(voices[0].events, ["play", "stop"])
        self.assertFalse(handle.active)

    def test_loop_rejects_non_pattern(self):
        sched, *_ = _make_scheduler()
        with self.assertRaises(TypeError):
            sched.loop(42, lambda n: None)


# ---------------------------------------------------------------------------
# Scheduler.start / .stop smoke (real thread, no audio)
# ---------------------------------------------------------------------------


class TestSchedulerTransport(unittest.TestCase):
    def test_start_and_stop_are_safe(self):
        """Start the real thread briefly; stop must join without hanging."""
        ft = _FakeClockTime(start=0.0)
        clock = Clock(time_fn=ft)
        sched = Scheduler(clock, graph=object())

        sched.start()
        self.assertTrue(clock.running)
        # Let the thread spin a few iterations.
        _real_time.sleep(0.01)
        sched.stop()
        # After stop, _thread should be cleared.
        self.assertIsNone(sched._thread)

    def test_stop_cancels_handles_and_stops_voices(self):
        ft = _FakeClockTime(start=0.0)
        clock = Clock(time_fn=ft)
        clock.start()
        voices: list[_RecordingVoice] = []

        def shape(note):
            v = _RecordingVoice(note)
            voices.append(v)
            return v

        sched = Scheduler(clock, graph=object())
        p = Pattern([(60, 10.0)])  # long event
        handle = sched.add(p, shape)
        sched._tick(0.0)  # note 60 fires; will want to stop at t=10
        self.assertEqual(voices[0].events, ["play"])

        sched.stop()
        self.assertFalse(handle.active)
        self.assertEqual(voices[0].events, ["play", "stop"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
