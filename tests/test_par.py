"""Tests for Phase 6: PAR + PARScheduler + bind_mod_speed.

Run with::

    python -m unittest tests.test_par
"""

from __future__ import annotations

import unittest

from mdma_rebuild.backend.automation import Constant, LFO
from mdma_rebuild.backend.clock import Clock
from mdma_rebuild.backend.par import (
    PAR,
    PARLoopHandle,
    PARScheduleHandle,
    PARScheduler,
)
from mdma_rebuild.backend.pattern import Pattern
from mdma_rebuild.backend.scheduler import Scheduler


class _FakeClockTime:
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


def _make_par_scheduler():
    ft = _FakeClockTime(start=0.0)
    clock = Clock(time_fn=ft)
    clock.start()
    sched = PARScheduler(clock, graph=object())
    return sched, clock, ft


# ---------------------------------------------------------------------------
# PAR class
# ---------------------------------------------------------------------------


class TestPAR(unittest.TestCase):
    def test_inherits_pattern(self):
        p = PAR([(60, 0.25)], mod_speed=2.0)
        self.assertIsInstance(p, Pattern)
        self.assertEqual(p.mod_speed, 2.0)
        self.assertEqual(p.events, [(60.0, 0.25)])

    def test_default_mod_speed(self):
        p = PAR([(60, 0.1)])
        self.assertEqual(p.mod_speed, 1.0)

    def test_rejects_non_positive_mod_speed(self):
        with self.assertRaises(ValueError):
            PAR([(60, 0.1)], mod_speed=0)
        with self.assertRaises(ValueError):
            PAR([(60, 0.1)], mod_speed=-2)


# ---------------------------------------------------------------------------
# PARScheduler.add / loop
# ---------------------------------------------------------------------------


class TestPARSchedulerAdd(unittest.TestCase):
    def test_add_rejects_non_PAR(self):
        sched, *_ = _make_par_scheduler()
        with self.assertRaises(TypeError):
            sched.add(Pattern([(60, 0.1)]), lambda n: _RecordingVoice(n))

    def test_add_uses_pattern_mod_speed_by_default(self):
        sched, *_ = _make_par_scheduler()
        p = PAR([(60, 0.5)], mod_speed=2.0)
        h = sched.add(p, lambda n: _RecordingVoice(n))
        self.assertIsInstance(h, PARScheduleHandle)
        self.assertEqual(h.mod_speed, 2.0)

    def test_add_override_mod_speed(self):
        sched, *_ = _make_par_scheduler()
        p = PAR([(60, 0.5)], mod_speed=1.0)
        h = sched.add(p, lambda n: _RecordingVoice(n), mod_speed=4.0)
        self.assertEqual(h.mod_speed, 4.0)

    def test_mod_speed_shortens_effective_durations(self):
        sched, clock, ft = _make_par_scheduler()
        voices: list[_RecordingVoice] = []

        def shape(n):
            v = _RecordingVoice(n)
            voices.append(v)
            return v

        # 4 events at 0.5s each => 2.0s pattern at mod_speed=1.
        # At mod_speed=4 each effective duration = 0.125s.
        p = PAR([(60, 0.5), (62, 0.5), (64, 0.5), (65, 0.5)], mod_speed=4.0)
        handle = sched.add(p, shape)

        sched._tick(0.0)
        self.assertEqual([v.note for v in voices], [60.0])
        sched._tick(0.125)
        self.assertEqual([v.note for v in voices], [60.0, 62.0])
        sched._tick(0.25)
        self.assertEqual([v.note for v in voices], [60.0, 62.0, 64.0])
        sched._tick(0.375)
        self.assertEqual([v.note for v in voices], [60.0, 62.0, 64.0, 65.0])
        sched._tick(0.5)
        self.assertFalse(handle.active)

    def test_set_mod_speed_live_mutation(self):
        sched, clock, ft = _make_par_scheduler()
        voices: list[_RecordingVoice] = []

        def shape(n):
            v = _RecordingVoice(n)
            voices.append(v)
            return v

        # Start at mod_speed 1.0, first event dur 0.5s.
        p = PAR([(60, 0.5), (62, 0.5), (64, 0.5)], mod_speed=1.0)
        handle = sched.add(p, shape)

        sched._tick(0.0)  # fire event 0 at dur 0.5
        # After first event dispatches, bump mod_speed to 2.0. The next
        # event's effective duration becomes 0.25s, so event 1 starts at
        # t=0.5 (unchanged — already queued) but event 2 starts at 0.75
        # instead of 1.0.
        handle.set_mod_speed(2.0)
        sched._tick(0.5)  # event 1 fires
        sched._tick(0.75)  # event 2 fires (0.5 + 0.25)
        self.assertEqual([v.note for v in voices], [60.0, 62.0, 64.0])

    def test_set_mod_speed_rejects_non_positive(self):
        sched, *_ = _make_par_scheduler()
        h = sched.add(
            PAR([(60, 0.1)]), lambda n: _RecordingVoice(n), mod_speed=1.0
        )
        with self.assertRaises(ValueError):
            h.set_mod_speed(0)
        with self.assertRaises(ValueError):
            h.set_mod_speed(-3)

    def test_loop_returns_par_loop_handle(self):
        sched, *_ = _make_par_scheduler()
        h = sched.loop(PAR([(60, 0.1)]), lambda n: _RecordingVoice(n))
        self.assertIsInstance(h, PARLoopHandle)


# ---------------------------------------------------------------------------
# bind_mod_speed (Phase 4 deferred binding, now live)
# ---------------------------------------------------------------------------


class TestBindModSpeed(unittest.TestCase):
    def test_lfo_drives_handle_mod_speed(self):
        sched, clock, ft = _make_par_scheduler()
        p = PAR([(60, 1.0)], mod_speed=1.0)
        handle = sched.add(p, lambda n: _RecordingVoice(n))

        # LFO oscillates mod_speed between 1 and 3 (offset 2, depth 1).
        lfo = LFO("sine", rate=1.0, depth=1.0, offset=2.0)
        sched.bind_mod_speed(handle, lfo)

        # Initial tick dispatches event 0 AND writes mod_speed for this
        # tick. sine at t=0 = 0; offset 2 + depth 1 * 0 = 2.
        sched._tick(0.0)
        self.assertAlmostEqual(handle.mod_speed, 2.0)

        # Peak of sine at t=0.25 -> mod_speed 3
        sched._tick(0.25)
        self.assertAlmostEqual(handle.mod_speed, 3.0)

    def test_bind_mod_speed_rejects_non_par_handle(self):
        # Use a regular Scheduler with regular ScheduleHandle, then try
        # to bind mod_speed to it — should TypeError.
        ft = _FakeClockTime()
        clock = Clock(time_fn=ft)
        clock.start()
        sched = Scheduler(clock, graph=object())
        h = sched.add(Pattern([(60, 0.25)]), lambda n: _RecordingVoice(n))
        with self.assertRaises(TypeError):
            sched.bind_mod_speed(h, Constant(2.0))

    def test_non_positive_automation_value_is_ignored(self):
        sched, clock, ft = _make_par_scheduler()
        p = PAR([(60, 1.0)], mod_speed=2.0)
        handle = sched.add(p, lambda n: _RecordingVoice(n))
        # Constant(-5) is invalid; binding should skip the write.
        sched.bind_mod_speed(handle, Constant(-5.0))
        sched._tick(0.0)
        # handle.mod_speed stays at 2.0 despite the bad source.
        self.assertEqual(handle.mod_speed, 2.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
