"""Unit tests for Phase 4 automation.

Covers the :class:`AutomationSource` hierarchy (``Constant``, ``LFO``,
``Ramp``, ``Envelope``) plus the :class:`Scheduler` bindings
(``bind_chain_param``, ``bind_tempo``, ``bind_master_volume``) and the
:class:`BindingHandle` lifecycle.

Run with::

    python -m unittest tests.test_automation
"""

from __future__ import annotations

import math
import unittest

from mdma_rebuild.backend.automation import (
    AutomationSource,
    BindingHandle,
    Constant,
    Envelope,
    LFO,
    Ramp,
)
from mdma_rebuild.backend.clock import Clock
from mdma_rebuild.backend.pattern import Pattern
from mdma_rebuild.backend.scheduler import Scheduler


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeClockTime:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class _RecordingNode:
    """SignalFlow-Node stand-in that records set_input calls."""

    def __init__(self, tag: str = "node") -> None:
        self.tag = tag
        self.set_input_calls: list[tuple[str, float]] = []
        self.events: list[str] = []

    def set_input(self, name: str, value: float) -> None:
        self.set_input_calls.append((name, float(value)))

    def play(self) -> None:
        self.events.append("play")

    def stop(self) -> None:
        self.events.append("stop")


class _RecordingGraph:
    def __init__(self) -> None:
        self.level_calls: list[float] = []
        self.output_level: float | None = None

    def set_output_level(self, level: float) -> None:
        self.level_calls.append(float(level))


class _PlainGraph:
    """Graph that only supports attribute-based output_level writes."""

    def __init__(self) -> None:
        self.output_level: float = 1.0


# ---------------------------------------------------------------------------
# Source-level tests
# ---------------------------------------------------------------------------


class TestConstant(unittest.TestCase):
    def test_returns_fixed_value(self):
        c = Constant(42.0)
        self.assertEqual(c.value_at(0.0), 42.0)
        self.assertEqual(c.value_at(100.0), 42.0)

    def test_start_stop_are_noops(self):
        c = Constant(1.0)
        c.start(5.0)  # should not affect output
        self.assertEqual(c.value_at(10.0), 1.0)
        c.stop()


class TestLFO(unittest.TestCase):
    def test_sine_zero_at_t0(self):
        lfo = LFO("sine", rate=1.0)
        lfo.start(0.0)
        self.assertAlmostEqual(lfo.value_at(0.0), 0.0)
        # Quarter period -> sine == 1
        self.assertAlmostEqual(lfo.value_at(0.25), 1.0)

    def test_square_jumps(self):
        lfo = LFO("square", rate=1.0)
        lfo.start(0.0)
        self.assertEqual(lfo.value_at(0.1), 1.0)
        self.assertEqual(lfo.value_at(0.6), -1.0)

    def test_saw_ramps_from_minus1_to_1(self):
        lfo = LFO("saw", rate=1.0)
        lfo.start(0.0)
        self.assertAlmostEqual(lfo.value_at(0.0), -1.0)
        self.assertAlmostEqual(lfo.value_at(0.5), 0.0)
        # Approach the top just before wrap-around.
        self.assertAlmostEqual(lfo.value_at(0.999), 0.998, places=3)

    def test_triangle_waveform(self):
        """Triangle starts at 0 rising, peaks at phase 0.25, troughs at 0.75."""
        lfo = LFO("triangle", rate=1.0)
        lfo.start(0.0)
        self.assertAlmostEqual(lfo.value_at(0.0), 0.0)
        self.assertAlmostEqual(lfo.value_at(0.25), 1.0)
        self.assertAlmostEqual(lfo.value_at(0.5), 0.0)
        self.assertAlmostEqual(lfo.value_at(0.75), -1.0)

    def test_depth_and_offset(self):
        lfo = LFO("sine", rate=1.0, depth=400.0, offset=800.0)
        lfo.start(0.0)
        # Output at t=0.25 (peak) = offset + depth * 1.0 = 1200
        self.assertAlmostEqual(lfo.value_at(0.25), 1200.0)
        self.assertAlmostEqual(lfo.value_at(0.75), 400.0)

    def test_start_resets_phase(self):
        lfo = LFO("sine", rate=1.0)
        lfo.start(5.0)
        # t=5 is the origin, so sin(0) = 0
        self.assertAlmostEqual(lfo.value_at(5.0), 0.0)
        self.assertAlmostEqual(lfo.value_at(5.25), 1.0)

    def test_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            LFO("noise")

    def test_rejects_non_positive_rate(self):
        with self.assertRaises(ValueError):
            LFO("sine", rate=0)
        with self.assertRaises(ValueError):
            LFO("sine", rate=-1)


class TestRamp(unittest.TestCase):
    def test_linear_interpolation(self):
        r = Ramp(start=0.0, end=10.0, duration=2.0)
        r.start(0.0)
        self.assertEqual(r.value_at(0.0), 0.0)
        self.assertEqual(r.value_at(1.0), 5.0)
        self.assertEqual(r.value_at(2.0), 10.0)

    def test_holds_end_value_after(self):
        r = Ramp(0.0, 5.0, duration=1.0)
        r.start(0.0)
        self.assertEqual(r.value_at(10.0), 5.0)

    def test_holds_start_value_before(self):
        r = Ramp(3.0, 7.0, duration=1.0)
        r.start(5.0)
        self.assertEqual(r.value_at(0.0), 3.0)

    def test_rejects_non_positive_duration(self):
        with self.assertRaises(ValueError):
            Ramp(0.0, 1.0, 0.0)
        with self.assertRaises(ValueError):
            Ramp(0.0, 1.0, -2.0)


class TestEnvelope(unittest.TestCase):
    def test_linear_between_points(self):
        env = Envelope([(0.0, 0.0), (1.0, 10.0), (2.0, 0.0)])
        env.start(0.0)
        self.assertEqual(env.value_at(0.5), 5.0)
        self.assertEqual(env.value_at(1.5), 5.0)
        self.assertEqual(env.value_at(1.0), 10.0)

    def test_clamps_outside_range(self):
        env = Envelope([(1.0, 2.0), (3.0, 6.0)])
        env.start(0.0)
        self.assertEqual(env.value_at(0.0), 2.0)  # before first point
        self.assertEqual(env.value_at(5.0), 6.0)  # after last point

    def test_sorts_input_points(self):
        env = Envelope([(2.0, 5.0), (0.0, 0.0), (1.0, 10.0)])
        env.start(0.0)
        self.assertEqual(env.value_at(0.5), 5.0)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            Envelope([])


# ---------------------------------------------------------------------------
# Scheduler binding tests
# ---------------------------------------------------------------------------


def _scheduler_with_bindings(graph=None):
    ft = _FakeClockTime(start=0.0)
    clock = Clock(time_fn=ft)
    clock.start()
    sched = Scheduler(clock, graph=graph if graph is not None else _RecordingGraph())
    return sched, clock, ft


class TestBindChainParam(unittest.TestCase):
    def test_lfo_writes_to_live_node_per_tick(self):
        sched, clock, ft = _scheduler_with_bindings()

        nodes_built: list[_RecordingNode] = []

        def bf_stub(self, voice, ftype, cutoff, res):
            # Act as Pattern._apply_builtin_filter without SignalFlow.
            n = _RecordingNode(f"bf({ftype})")
            nodes_built.append(n)
            return n

        Pattern._apply_builtin_filter = bf_stub  # type: ignore[method-assign]
        try:
            p = Pattern([(60, 2.0)]).bf("lpf", 800)
            shape = lambda note: _RecordingNode(f"v{note}")
            handle = sched.add(p, shape)

            # First tick dispatches the event, so stage nodes are live.
            sched._tick(0.0)
            self.assertEqual(len(nodes_built), 1)

            lfo = LFO("sine", rate=1.0, depth=400.0, offset=800.0)
            binding = sched.bind_chain_param(handle, 0, "cutoff", lfo)
            self.assertIsInstance(binding, BindingHandle)

            # Peak of sine at t=0.25 -> 1200
            ft.now = 0.25
            sched._tick(0.25)
            self.assertIn(("cutoff", 1200.0), nodes_built[0].set_input_calls)

            # Trough at t=0.75 -> 400
            ft.now = 0.75
            sched._tick(0.75)
            self.assertIn(("cutoff", 400.0), nodes_built[0].set_input_calls)

            binding.unbind()
            before = len(nodes_built[0].set_input_calls)
            ft.now = 1.0
            sched._tick(1.0)
            self.assertEqual(
                len(nodes_built[0].set_input_calls),
                before,
                "unbind must stop further writes",
            )
        finally:
            del Pattern._apply_builtin_filter

    def test_out_of_range_chain_index_raises(self):
        sched, *_rest = _scheduler_with_bindings()
        p = Pattern([(60, 0.5)])  # empty chain
        handle = sched.add(p, lambda n: _RecordingNode())
        with self.assertRaises(IndexError):
            sched.bind_chain_param(handle, 0, "cutoff", Constant(1.0))

    def test_handle_without_live_voice_skips_silently(self):
        """Binding created before the handle dispatches should not crash."""
        sched, clock, ft = _scheduler_with_bindings()

        def bf_stub(self, voice, ftype, cutoff, res):
            return _RecordingNode(f"bf({ftype})")

        Pattern._apply_builtin_filter = bf_stub  # type: ignore[method-assign]
        try:
            p = Pattern([(60, 1.0)]).bf("lpf", 800)
            handle = sched.add(
                p, lambda n: _RecordingNode(), start=5.0
            )
            sched.bind_chain_param(handle, 0, "cutoff", Constant(500.0))
            # t=0 — handle hasn't dispatched yet, binding has no live node
            sched._tick(0.0)  # should not raise
        finally:
            del Pattern._apply_builtin_filter


class TestBindTempo(unittest.TestCase):
    def test_ramp_sweeps_clock_tempo(self):
        sched, clock, ft = _scheduler_with_bindings()
        ramp = Ramp(start=120.0, end=60.0, duration=2.0)
        sched.bind_tempo(ramp)
        ft.now = 0.0
        sched._tick(0.0)
        self.assertEqual(clock.tempo, 120.0)
        ft.now = 1.0
        sched._tick(1.0)
        self.assertAlmostEqual(clock.tempo, 90.0)
        ft.now = 2.0
        sched._tick(2.0)
        self.assertAlmostEqual(clock.tempo, 60.0)

    def test_non_positive_tempo_is_dropped_silently(self):
        sched, clock, ft = _scheduler_with_bindings()
        clock.set_tempo(100.0)
        sched.bind_tempo(Constant(-5.0))
        sched._tick(0.0)
        # Clock keeps its last valid tempo — the binding drop silently.
        self.assertEqual(clock.tempo, 100.0)


class TestBindMasterVolume(unittest.TestCase):
    def test_setter_method_is_called(self):
        graph = _RecordingGraph()
        sched, clock, ft = _scheduler_with_bindings(graph=graph)
        sched.bind_master_volume(Constant(0.7))
        sched._tick(0.0)
        self.assertIn(0.7, graph.level_calls)

    def test_fallback_to_attribute_assignment(self):
        graph = _PlainGraph()
        sched, clock, ft = _scheduler_with_bindings(graph=graph)
        sched.bind_master_volume(Constant(0.5))
        sched._tick(0.0)
        self.assertEqual(graph.output_level, 0.5)


class TestBindingHandleLifecycle(unittest.TestCase):
    def test_unbind_is_idempotent(self):
        sched, *_ = _scheduler_with_bindings()
        b = sched.bind_tempo(Constant(120.0))
        b.unbind()
        b.unbind()  # must not raise

    def test_scheduler_stop_tears_down_bindings(self):
        sched, clock, ft = _scheduler_with_bindings()

        stops: list[bool] = []

        class _Tracking(AutomationSource):
            def value_at(self, t): return 1.0
            def stop(self_inner): stops.append(True)

        sched.bind_tempo(_Tracking())
        sched.stop()
        self.assertEqual(stops, [True])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
