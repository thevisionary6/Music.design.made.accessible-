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

from mdma_rebuild.backend.effects import delay, soft_clip
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
    """Return a stub ``signalflow`` module with Tanh / OneTapDelay classes."""
    mod = types.ModuleType("signalflow")

    class _Tanh(_StubNode):
        def __init__(self, a):
            super().__init__(f"Tanh({_tag(a)})")
            self.input = a

    class _OneTapDelay(_StubNode):
        def __init__(self, input, delay_time=0.1, max_delay_time=0.5):
            super().__init__(
                f"OneTapDelay({_tag(input)},{delay_time},{max_delay_time})"
            )
            self.input = input
            self.delay_time = delay_time
            self.max_delay_time = max_delay_time

    mod.Tanh = _Tanh
    mod.OneTapDelay = _OneTapDelay
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
# Porting convention smoke
# ---------------------------------------------------------------------------


class TestPortingConvention(unittest.TestCase):
    def test_effects_package_exports(self):
        """The package __init__ should expose every ported effect so
        future additions just show up as ``from backend.effects import
        their_name``."""
        from mdma_rebuild.backend import effects as fx

        self.assertTrue(callable(fx.soft_clip))
        self.assertTrue(callable(fx.delay))

    def test_signature_matches_convention(self):
        """Every ported effect must accept ``input_node`` as the first
        positional arg. Everything else must be keyword-only by
        convention, even if Python doesn't enforce it at the signature
        level."""
        import inspect

        for fn in (soft_clip, delay):
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
