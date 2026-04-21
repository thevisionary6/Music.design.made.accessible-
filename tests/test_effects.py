"""Tests for Phase 8: ported effects.

These tests exercise the ``fn(input_node, **params) -> Node``
convention by putting the ported effects into a Pattern chain against
a lightweight node mock. Real-audio correctness tests for each effect
(actual tanh response, actual delay timing) live under
``tests/audio/`` and are opt-in.

Run with::

    python -m unittest tests.test_effects
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from mdma_rebuild.backend.effects import (
    bitcrush,
    delay,
    foldback,
    hard_clip,
    reverb,
    soft_clip,
)
from mdma_rebuild.backend.pattern import Pattern


# ---------------------------------------------------------------------------
# SignalFlow stub (only active for the tests that need it)
# ---------------------------------------------------------------------------


class _StubNode:
    """Minimal node double that supports arithmetic and plays like a Node."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __mul__(self, other):
        return _StubNode(f"({self.name}*{_tag(other)})")

    def __rmul__(self, other):
        return _StubNode(f"({_tag(other)}*{self.name})")

    def __add__(self, other):
        return _StubNode(f"({self.name}+{_tag(other)})")

    def __radd__(self, other):
        return _StubNode(f"({_tag(other)}+{self.name})")

    def play(self): pass
    def stop(self): pass


def _tag(x):
    return x.name if isinstance(x, _StubNode) else str(x)


def _build_stub_signalflow_module():
    """Return a stub ``signalflow`` module with every node the effects
    need. Stub nodes carry a readable name so tests can verify the
    graph topology without real SignalFlow."""
    mod = types.ModuleType("signalflow")

    class _Tanh(_StubNode):
        def __init__(self, a):
            super().__init__(f"Tanh({_tag(a)})")

    class _OneTapDelay(_StubNode):
        def __init__(self, input, delay_time=0.1, max_delay_time=0.5):
            super().__init__(
                f"OneTapDelay({_tag(input)},{delay_time},{max_delay_time})"
            )

    class _CombDelay(_StubNode):
        def __init__(
            self,
            input,
            delay_time=0.1,
            feedback=0.5,
            max_delay_time=0.5,
        ):
            super().__init__(
                f"CombDelay({_tag(input)},dt={delay_time},fb={feedback})"
            )

    class _AllpassDelay(_StubNode):
        def __init__(
            self,
            input,
            delay_time=0.1,
            feedback=0.5,
            max_delay_time=0.5,
        ):
            super().__init__(
                f"AllpassDelay({_tag(input)},dt={delay_time},fb={feedback})"
            )

    class _Clip(_StubNode):
        def __init__(self, input, lo=-1.0, hi=1.0):
            super().__init__(f"Clip({_tag(input)},{lo},{hi})")

    class _Fold(_StubNode):
        def __init__(self, input, lo=-1.0, hi=1.0):
            super().__init__(f"Fold({_tag(input)},{lo},{hi})")

    class _Resample(_StubNode):
        def __init__(self, input, sample_rate=44100, bit_rate=16):
            super().__init__(
                f"Resample({_tag(input)},sr={sample_rate},bits={bit_rate})"
            )

    mod.Tanh = _Tanh
    mod.OneTapDelay = _OneTapDelay
    mod.CombDelay = _CombDelay
    mod.AllpassDelay = _AllpassDelay
    mod.Clip = _Clip
    mod.Fold = _Fold
    mod.Resample = _Resample
    return mod


class _SignalFlowStubbed:
    """Context manager that temporarily installs the stub module."""

    def __enter__(self):
        self._prev = sys.modules.get("signalflow")
        sys.modules["signalflow"] = _build_stub_signalflow_module()
        return sys.modules["signalflow"]

    def __exit__(self, exc_type, exc, tb):
        if self._prev is None:
            del sys.modules["signalflow"]
        else:
            sys.modules["signalflow"] = self._prev


# ---------------------------------------------------------------------------
# soft_clip
# ---------------------------------------------------------------------------


class TestSoftClip(unittest.TestCase):
    def test_returns_new_node_with_expected_shape(self):
        with _SignalFlowStubbed():
            voice = _StubNode("voice")
            out = soft_clip(voice, amount=3.0)
            # Tanh(voice * 3.0)
            self.assertIn("Tanh", out.name)
            self.assertIn("*3.0", out.name)

    def test_rejects_non_positive_amount(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                soft_clip(_StubNode("v"), amount=0)
            with self.assertRaises(ValueError):
                soft_clip(_StubNode("v"), amount=-1)

    def test_integrates_into_pattern_dist_chain(self):
        with _SignalFlowStubbed():
            p = Pattern([(60, 0.25)]).dist(soft_clip, amount=2.0)
            voice = _StubNode("v")
            out = p._apply_chain(voice)
            # chain is "dist" -> soft_clip(v, amount=2)
            self.assertIn("Tanh", out.name)


# ---------------------------------------------------------------------------
# delay
# ---------------------------------------------------------------------------


class TestDelay(unittest.TestCase):
    def test_returns_mix_of_dry_and_wet(self):
        with _SignalFlowStubbed():
            voice = _StubNode("voice")
            out = delay(voice, time=0.25, feedback=0.4, mix=0.5)
            # Expect a node whose name reflects both dry and wet.
            self.assertIn("voice", out.name)
            self.assertIn("OneTapDelay", out.name)

    def test_rejects_non_positive_time(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                delay(_StubNode("v"), time=0)
            with self.assertRaises(ValueError):
                delay(_StubNode("v"), time=-0.1)

    def test_rejects_negative_feedback(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                delay(_StubNode("v"), feedback=-0.1)

    def test_clamps_feedback_below_unity(self):
        """feedback=0.99 must be clamped to 0.95 so the DSP can't
        diverge; we check this by confirming the call doesn't raise
        and the node is still produced."""
        with _SignalFlowStubbed():
            out = delay(_StubNode("v"), feedback=0.99)
            self.assertIsNotNone(out)

    def test_clamps_mix_to_unit_range(self):
        with _SignalFlowStubbed():
            # Out-of-range mix values must be clamped, not raise.
            out_low = delay(_StubNode("v"), mix=-5.0)
            out_high = delay(_StubNode("v"), mix=5.0)
            self.assertIsNotNone(out_low)
            self.assertIsNotNone(out_high)

    def test_integrates_into_pattern_fx_chain(self):
        with _SignalFlowStubbed():
            p = Pattern([(60, 0.25)]).fx(delay, time=0.2, feedback=0.3)
            voice = _StubNode("v")
            out = p._apply_chain(voice)
            self.assertIn("OneTapDelay", out.name)


# ---------------------------------------------------------------------------
# hard_clip
# ---------------------------------------------------------------------------


class TestHardClip(unittest.TestCase):
    def test_builds_clip_node(self):
        with _SignalFlowStubbed():
            out = hard_clip(_StubNode("v"), drive=2.0, ceiling=0.8)
            self.assertIn("Clip", out.name)
            self.assertIn("*2.0", out.name)
            # ceiling threaded through
            self.assertIn("0.8", out.name)

    def test_rejects_non_positive_drive(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                hard_clip(_StubNode("v"), drive=0)
            with self.assertRaises(ValueError):
                hard_clip(_StubNode("v"), drive=-1)

    def test_rejects_non_positive_ceiling(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                hard_clip(_StubNode("v"), ceiling=0)


# ---------------------------------------------------------------------------
# foldback
# ---------------------------------------------------------------------------


class TestFoldback(unittest.TestCase):
    def test_builds_fold_node(self):
        with _SignalFlowStubbed():
            out = foldback(_StubNode("v"), drive=3.0, threshold=0.5)
            self.assertIn("Fold", out.name)
            self.assertIn("*3.0", out.name)

    def test_rejects_out_of_range_threshold(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                foldback(_StubNode("v"), threshold=0)
            with self.assertRaises(ValueError):
                foldback(_StubNode("v"), threshold=1.5)

    def test_rejects_non_positive_drive(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                foldback(_StubNode("v"), drive=-0.5)


# ---------------------------------------------------------------------------
# bitcrush
# ---------------------------------------------------------------------------


class TestBitcrush(unittest.TestCase):
    def test_builds_resample_node(self):
        with _SignalFlowStubbed():
            out = bitcrush(_StubNode("v"), bit_depth=8, sample_rate=22050)
            self.assertIn("Resample", out.name)
            self.assertIn("sr=22050", out.name)
            self.assertIn("bits=8", out.name)

    def test_rejects_bit_depth_out_of_range(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                bitcrush(_StubNode("v"), bit_depth=0)
            with self.assertRaises(ValueError):
                bitcrush(_StubNode("v"), bit_depth=25)

    def test_rejects_non_positive_sample_rate(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                bitcrush(_StubNode("v"), sample_rate=0)


# ---------------------------------------------------------------------------
# reverb
# ---------------------------------------------------------------------------


class TestReverb(unittest.TestCase):
    def test_builds_schroeder_topology(self):
        """Four parallel combs + two series allpasses; the wet signal's
        name should reflect the allpasses."""
        with _SignalFlowStubbed():
            out = reverb(_StubNode("voice"), room_size=0.8, mix=0.5)
            # Both allpass stages appear in the name of the wet leg.
            self.assertEqual(out.name.count("AllpassDelay"), 2)
            # Four comb taps in the wet leg.
            self.assertEqual(out.name.count("CombDelay"), 4)
            # Dry voice survives into the mix.
            self.assertIn("voice", out.name)

    def test_rejects_negative_params(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                reverb(_StubNode("v"), room_size=-0.1)
            with self.assertRaises(ValueError):
                reverb(_StubNode("v"), damping=-0.1)
            with self.assertRaises(ValueError):
                reverb(_StubNode("v"), mix=-0.1)

    def test_clamps_room_size_below_unity(self):
        """room_size=1.5 must be clamped so the comb feedback doesn't
        diverge; just check the call succeeds."""
        with _SignalFlowStubbed():
            out = reverb(_StubNode("v"), room_size=1.5)
            self.assertIsNotNone(out)

    def test_integrates_into_pattern_ir_chain(self):
        with _SignalFlowStubbed():
            p = Pattern([(60, 0.25)]).ir(reverb, mix=0.3)
            voice = _StubNode("v")
            out = p._apply_chain(voice)
            self.assertIn("AllpassDelay", out.name)


# ---------------------------------------------------------------------------
# Porting convention smoke
# ---------------------------------------------------------------------------


class TestPortingConvention(unittest.TestCase):
    def test_effects_package_exports(self):
        """The package __init__ should expose every ported effect so
        future additions just show up as ``from backend.effects import
        their_name``."""
        from mdma_rebuild.backend import effects as fx

        for name in (
            "soft_clip", "hard_clip", "foldback", "bitcrush", "delay",
            "reverb",
        ):
            self.assertTrue(
                callable(getattr(fx, name)),
                f"effects.{name} is not callable",
            )

    def test_signature_matches_convention(self):
        """Every ported effect must accept ``input_node`` as the first
        positional arg. Everything else must be keyword-only by
        convention, even if Python doesn't enforce it at the signature
        level."""
        import inspect

        for fn in (soft_clip, hard_clip, foldback, bitcrush, delay, reverb):
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            self.assertTrue(len(params) >= 1, f"{fn.__name__} has no params")
            self.assertEqual(
                params[0].name,
                "input_node",
                f"{fn.__name__}: first param must be 'input_node', "
                f"got {params[0].name!r}",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
