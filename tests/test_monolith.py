"""Tests for backend/monolith.py and backend/utils.py.

Uses a signalflow stub so no audio device is required. The stub
covers every node type the monolith builder needs.
"""

from __future__ import annotations

import math
import sys
import types
import unittest

from mdma_rebuild.backend import monolith, utils
from mdma_rebuild.backend.utils import (
    build_default_shape,
    build_saw_shape,
    build_sine_shape,
    freq_to_note,
    note_to_freq,
)


# ---------------------------------------------------------------------------
# Stub harness (mirrors the one in test_effects, trimmed to what the
# monolith and utils actually construct)
# ---------------------------------------------------------------------------


class _StubNode:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover
        return f"_StubNode({self.name!r})"

    def __mul__(self, other):
        return _StubNode(f"({self.name}*{_tag(other)})")

    def __rmul__(self, other):
        return _StubNode(f"({_tag(other)}*{self.name})")

    def __add__(self, other):
        return _StubNode(f"({self.name}+{_tag(other)})")

    def __radd__(self, other):
        return _StubNode(f"({_tag(other)}+{self.name})")


def _tag(x):
    return x.name if isinstance(x, _StubNode) else str(x)


def _build_signalflow_stub():
    mod = types.ModuleType("signalflow")

    class _Sine(_StubNode):
        def __init__(self, frequency=440, phase_offset=None, reset=None):
            super().__init__(f"Sine(f={_tag(frequency)},ph={_tag(phase_offset)})")

    class _Triangle(_StubNode):
        def __init__(self, frequency=440):
            super().__init__(f"Triangle(f={_tag(frequency)})")

    class _Saw(_StubNode):
        def __init__(self, frequency=440):
            super().__init__(f"Saw(f={_tag(frequency)})")

    class _Square(_StubNode):
        def __init__(self, frequency=440):
            super().__init__(f"Square(f={_tag(frequency)})")

    class _White(_StubNode):
        def __init__(self):
            super().__init__("WhiteNoise()")

    class _ASR(_StubNode):
        def __init__(self, attack=0.01, sustain=0.1, release=0.1):
            super().__init__(f"ASR(a={attack},s={sustain},r={release})")

    class _SVF(_StubNode):
        def __init__(self, input, filter_type, cutoff=440, resonance=0.0):
            super().__init__(
                f"SVF({_tag(input)},{filter_type},cut={cutoff},q={resonance})"
            )

    class _AudioGraph:
        _shared = None

        def __init__(self, start=False, **kwargs):
            self.started = False
            self.stopped = False
            _AudioGraph._shared = self
            if start:
                self.start()

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        @classmethod
        def get_shared_graph(cls):
            return cls._shared

    mod.SineOscillator = _Sine
    mod.TriangleOscillator = _Triangle
    mod.SawOscillator = _Saw
    mod.SquareOscillator = _Square
    mod.WhiteNoise = _White
    mod.ASREnvelope = _ASR
    mod.SVFilter = _SVF
    mod.AudioGraph = _AudioGraph
    return mod


class _SignalFlowStubbed:
    def __enter__(self):
        self._prev = sys.modules.get("signalflow")
        sys.modules["signalflow"] = _build_signalflow_stub()
        return sys.modules["signalflow"]

    def __exit__(self, exc_type, exc, tb):
        if self._prev is None:
            del sys.modules["signalflow"]
        else:
            sys.modules["signalflow"] = self._prev


# ---------------------------------------------------------------------------
# Minimal fake Session for monolith tests
# ---------------------------------------------------------------------------


class _FakeEngine:
    def __init__(self, operators=None, algorithms=None):
        self.operators = operators or {}
        self.algorithms = algorithms or []


class _FakeSession:
    def __init__(self, **kwargs):
        self.engine = _FakeEngine(
            operators=kwargs.pop("operators", {}),
            algorithms=kwargs.pop("algorithms", []),
        )
        self.carrier_count = kwargs.pop("carrier_count", 1)
        self.attack = kwargs.pop("attack", 0.01)
        self.decay = kwargs.pop("decay", 0.1)
        self.sustain = kwargs.pop("sustain", 1.0)
        self.release = kwargs.pop("release", 0.1)
        # Filter state
        self.selected_filter = 0
        self.filter_enabled = {0: False}
        self.filter_types = {0: 0}
        self.filter_cutoffs = {0: 1000.0}
        self.filter_resonances = {0: 50.0}
        self.filter_type_names = {
            0: "lowpass", 1: "highpass", 2: "bandpass", 3: "notch",
            4: "peak", 17: "lowshelf", 18: "highshelf",
            5: "ringmod", 6: "allpass",
        }
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# note_to_freq / freq_to_note
# ---------------------------------------------------------------------------


class TestNoteFreq(unittest.TestCase):
    def test_a4_is_440(self):
        self.assertAlmostEqual(note_to_freq(69), 440.0)

    def test_octave_ratio(self):
        self.assertAlmostEqual(note_to_freq(69 + 12), 880.0)
        self.assertAlmostEqual(note_to_freq(69 - 12), 220.0)

    def test_microtonal(self):
        self.assertAlmostEqual(note_to_freq(69.5), 440.0 * 2 ** (0.5 / 12))

    def test_round_trip(self):
        for midi in (33, 48, 60, 69, 84):
            self.assertAlmostEqual(freq_to_note(note_to_freq(midi)), midi)

    def test_freq_to_note_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            freq_to_note(0)
        with self.assertRaises(ValueError):
            freq_to_note(-220)


# ---------------------------------------------------------------------------
# Shape factories
# ---------------------------------------------------------------------------


class TestShapeFactories(unittest.TestCase):
    def test_saw_shape(self):
        with _SignalFlowStubbed():
            shape = build_saw_shape()
            node = shape(69)
            self.assertIn("Saw", node.name)
            self.assertIn("440", node.name)

    def test_sine_shape(self):
        with _SignalFlowStubbed():
            shape = build_sine_shape()
            node = shape(69)
            self.assertIn("Sine", node.name)

    def test_default_shape_no_session_returns_saw(self):
        with _SignalFlowStubbed():
            shape = build_default_shape(None)
            node = shape(60)
            self.assertIn("Saw", node.name)

    def test_default_shape_session_without_operators_returns_saw(self):
        with _SignalFlowStubbed():
            session = _FakeSession()  # empty operators
            shape = build_default_shape(session)
            node = shape(60)
            self.assertIn("Saw", node.name)

    def test_default_shape_with_operators_uses_monolith(self):
        with _SignalFlowStubbed():
            session = _FakeSession(operators={0: {"wave": "sine"}})
            shape = build_default_shape(session)
            node = shape(69)
            # Monolith builds ASR+SineOsc; distinguishes from plain saw.
            self.assertIn("ASR", node.name)
            self.assertIn("Sine", node.name)


# ---------------------------------------------------------------------------
# audiograph context manager
# ---------------------------------------------------------------------------


class TestAudioGraphContext(unittest.TestCase):
    def test_creates_and_stops_graph(self):
        with _SignalFlowStubbed() as sf:
            # Reset the shared-graph state so the test is isolated.
            sf.AudioGraph._shared = None
            with utils.audiograph(start=True) as g:
                self.assertTrue(g.started)
            self.assertTrue(g.stopped)

    def test_yields_existing_graph_without_stopping_it(self):
        with _SignalFlowStubbed() as sf:
            pre_existing = sf.AudioGraph(start=True)
            self.assertIs(pre_existing, sf.AudioGraph.get_shared_graph())
            with utils.audiograph() as g:
                self.assertIs(g, pre_existing)
            # Leave the pre-existing graph alone; we didn't create it.
            self.assertFalse(pre_existing.stopped)


# ---------------------------------------------------------------------------
# Monolith: operator building
# ---------------------------------------------------------------------------


class TestMonolithOperatorBuild(unittest.TestCase):
    def test_sine_operator(self):
        with _SignalFlowStubbed():
            session = _FakeSession(operators={0: {"wave": "sine"}})
            voice = monolith.build_voice(69, session)
            self.assertIn("Sine", voice.name)
            self.assertIn("440", voice.name)

    def test_wave_aliases_map(self):
        """pwm / square both resolve to pulse ('Square' in SignalFlow)."""
        with _SignalFlowStubbed():
            for alias in ("pulse", "square", "pwm"):
                session = _FakeSession(operators={0: {"wave": alias}})
                voice = monolith.build_voice(69, session)
                self.assertIn("Square", voice.name)

    def test_unsupported_wave_falls_back_to_sine(self):
        with _SignalFlowStubbed():
            session = _FakeSession(operators={0: {"wave": "supersaw"}})
            voice = monolith.build_voice(69, session)
            self.assertIn("Sine", voice.name)

    def test_noise_operator(self):
        with _SignalFlowStubbed():
            session = _FakeSession(operators={0: {"wave": "noise"}})
            voice = monolith.build_voice(69, session)
            self.assertIn("WhiteNoise", voice.name)

    def test_ratio_overrides_freq(self):
        """ratio is a multiplier on note frequency."""
        with _SignalFlowStubbed():
            session = _FakeSession(
                operators={0: {"wave": "sine", "ratio": 2.0}},
            )
            voice = monolith.build_voice(69, session)
            # A4 * 2 = 880 Hz.
            self.assertIn("880", voice.name)


# ---------------------------------------------------------------------------
# Monolith: modulation routing
# ---------------------------------------------------------------------------


class TestMonolithModulation(unittest.TestCase):
    def test_fm_rewires_target_frequency(self):
        with _SignalFlowStubbed():
            session = _FakeSession(
                operators={
                    0: {"wave": "sine", "ratio": 1.0},
                    1: {"wave": "sine", "ratio": 2.0},
                },
                algorithms=[("FM", 1, 0, 100.0)],
                carrier_count=1,
            )
            voice = monolith.build_voice(69, session)
            # FM target (op 0) has its Sine built with an added node
            # for the modulator source. The modulator's "*100.0" scale
            # appears inside the target's name.
            self.assertIn("*100.0", voice.name)

    def test_am_multiplies_target(self):
        with _SignalFlowStubbed():
            session = _FakeSession(
                operators={
                    0: {"wave": "sine"},
                    1: {"wave": "sine", "ratio": 2.0},
                },
                algorithms=[("AM", 1, 0, 0.5)],
                carrier_count=1,
            )
            voice = monolith.build_voice(69, session)
            # AM shape is target * (1 + mod*0.5).
            self.assertIn("*0.5", voice.name)

    def test_rm_ring_modulates(self):
        with _SignalFlowStubbed():
            session = _FakeSession(
                operators={
                    0: {"wave": "sine"},
                    1: {"wave": "sine", "ratio": 2.0},
                },
                algorithms=[("RM", 1, 0, 1.0)],
                carrier_count=1,
            )
            voice = monolith.build_voice(69, session)
            # Ring mod: target * (mod * amount).
            self.assertIn("*1.0", voice.name)

    def test_unknown_algorithm_prints_warning_but_does_not_raise(self):
        with _SignalFlowStubbed():
            session = _FakeSession(
                operators={0: {"wave": "sine"}, 1: {"wave": "sine"}},
                algorithms=[("QUANTUM", 1, 0, 1.0)],
                carrier_count=1,
            )
            # Should not raise — just warn and skip.
            voice = monolith.build_voice(69, session)
            self.assertIn("Sine", voice.name)


# ---------------------------------------------------------------------------
# Monolith: envelope + filter + carrier summing
# ---------------------------------------------------------------------------


class TestMonolithWrap(unittest.TestCase):
    def test_envelope_always_wraps_voice(self):
        with _SignalFlowStubbed():
            session = _FakeSession(operators={0: {"wave": "sine"}})
            voice = monolith.build_voice(69, session)
            self.assertIn("ASR", voice.name)

    def test_filter_applied_when_enabled(self):
        with _SignalFlowStubbed():
            session = _FakeSession(operators={0: {"wave": "sine"}})
            session.filter_enabled[0] = True
            session.filter_types[0] = 0  # lowpass
            session.filter_cutoffs[0] = 800.0
            voice = monolith.build_voice(69, session)
            self.assertIn("SVF", voice.name)
            self.assertIn("low_pass", voice.name)

    def test_filter_skipped_for_unsupported_type(self):
        """Formant / ringmod slots don't map to SVFilter; voice should
        still build without crashing."""
        with _SignalFlowStubbed():
            session = _FakeSession(operators={0: {"wave": "sine"}})
            session.filter_enabled[0] = True
            session.filter_types[0] = 5  # ringmod
            voice = monolith.build_voice(69, session)
            self.assertNotIn("SVF", voice.name)

    def test_multiple_carriers_are_summed(self):
        with _SignalFlowStubbed():
            session = _FakeSession(
                operators={
                    0: {"wave": "sine"},
                    1: {"wave": "saw", "ratio": 2.0},
                },
                carrier_count=2,
            )
            voice = monolith.build_voice(69, session)
            self.assertIn("Sine", voice.name)
            self.assertIn("Saw", voice.name)

    def test_no_carriers_falls_back_to_sine(self):
        with _SignalFlowStubbed():
            session = _FakeSession(
                operators={1: {"wave": "saw"}},  # op 1 isn't a carrier
                carrier_count=1,
            )
            voice = monolith.build_voice(69, session)
            self.assertIn("Sine", voice.name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
