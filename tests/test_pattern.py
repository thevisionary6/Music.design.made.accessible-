"""Unit tests for the Pattern class.

Every test here corresponds to a test example in
``Backend Docs V2/references/pattern_spec.md``. The spec is the source of
truth; if a test here and the spec disagree, the spec wins.

Run with::

    python -m unittest tests.test_pattern
"""

from __future__ import annotations

import unittest

from mdma_rebuild.backend.pattern import Pattern


# ---------------------------------------------------------------------------
# Construction and immutability
# ---------------------------------------------------------------------------

class TestConstruction(unittest.TestCase):
    def test_coerces_to_float(self):
        p = Pattern([(60, 1), (62, 2)])
        self.assertEqual(p.events, [(60.0, 1.0), (62.0, 2.0)])
        for note, dur in p.events:
            self.assertIsInstance(note, float)
            self.assertIsInstance(dur, float)

    def test_empty_pattern_is_legal(self):
        p = Pattern([])
        self.assertEqual(p.events, [])
        self.assertEqual(p.chain, [])

    def test_chain_starts_empty(self):
        p = Pattern([(60, 0.25)])
        self.assertEqual(p.chain, [])

    def test_transform_does_not_mutate_self(self):
        base = Pattern([(60, 0.25), (64, 0.25), (67, 0.25)])
        base.fast(2)
        base.t_p(5)
        base.bf("lpf", 800)
        self.assertEqual(base.events, [(60.0, 0.25), (64.0, 0.25), (67.0, 0.25)])
        self.assertEqual(base.chain, [])


# ---------------------------------------------------------------------------
# Compositional transforms
# ---------------------------------------------------------------------------

class TestFast(unittest.TestCase):
    def test_doubles_speed(self):
        events = Pattern([(60, 0.5), (62, 0.5)]).fast(2).events
        self.assertEqual(events, [(60.0, 0.25), (62.0, 0.25)])

    def test_halves_speed(self):
        events = Pattern([(60, 0.5)]).fast(0.5).events
        self.assertEqual(events, [(60.0, 1.0)])

    def test_empty_passthrough(self):
        self.assertEqual(Pattern([]).fast(2).events, [])

    def test_rejects_non_positive(self):
        p = Pattern([(60, 0.25)])
        with self.assertRaises(ValueError):
            p.fast(0)
        with self.assertRaises(ValueError):
            p.fast(-1)


class TestSlow(unittest.TestCase):
    def test_doubles_duration(self):
        events = Pattern([(60, 0.25), (62, 0.25)]).slow(2).events
        self.assertEqual(events, [(60.0, 0.5), (62.0, 0.5)])

    def test_halves_duration(self):
        events = Pattern([(60, 0.5)]).slow(0.5).events
        self.assertEqual(events, [(60.0, 0.25)])

    def test_rejects_non_positive(self):
        p = Pattern([(60, 0.25)])
        with self.assertRaises(ValueError):
            p.slow(0)
        with self.assertRaises(ValueError):
            p.slow(-0.5)


class TestTranspose(unittest.TestCase):
    def test_transpose_up(self):
        events = Pattern([(60, 0.25), (64, 0.25)]).t_p(7).events
        self.assertEqual(events, [(67.0, 0.25), (71.0, 0.25)])

    def test_transpose_down(self):
        events = Pattern([(60, 0.25)]).t_p(-12).events
        self.assertEqual(events, [(48.0, 0.25)])

    def test_transpose_microtonal(self):
        events = Pattern([(60, 0.25)]).t_p(0.5).events
        self.assertEqual(events, [(60.5, 0.25)])


class TestLength(unittest.TestCase):
    def test_scale_to_target(self):
        events = Pattern([(60, 0.25), (62, 0.25), (64, 0.5)]).l(2.0).events
        self.assertEqual(events, [(60.0, 0.5), (62.0, 0.5), (64.0, 1.0)])

    def test_compress(self):
        events = Pattern([(60, 1.0), (62, 1.0)]).l(1.0).events
        self.assertEqual(events, [(60.0, 0.5), (62.0, 0.5)])

    def test_empty_returns_empty(self):
        self.assertEqual(Pattern([]).l(2.0).events, [])

    def test_zero_total_returns_copy(self):
        p = Pattern([(60, 0.0), (62, 0.0)])
        out = p.l(2.0)
        self.assertEqual(out.events, [(60.0, 0.0), (62.0, 0.0)])

    def test_rejects_non_positive_target(self):
        p = Pattern([(60, 0.5)])
        with self.assertRaises(ValueError):
            p.l(0)
        with self.assertRaises(ValueError):
            p.l(-1.0)


class TestGrooveSwing(unittest.TestCase):
    def test_alternating_multipliers(self):
        events = Pattern(
            [(60, 0.25), (62, 0.25), (64, 0.25), (65, 0.25)]
        ).gs(1.5, 0.5).events
        self.assertEqual(
            events,
            [(60.0, 0.375), (62.0, 0.125), (64.0, 0.375), (65.0, 0.125)],
        )

    def test_single_event_uses_a(self):
        events = Pattern([(60, 1.0)]).gs(2.0, 0.5).events
        self.assertEqual(events, [(60.0, 2.0)])


class TestExtend(unittest.TestCase):
    def test_extend_nonempty(self):
        events = Pattern([(60, 0.25)]).ex(62, 0.5).events
        self.assertEqual(events, [(60.0, 0.25), (62.0, 0.5)])

    def test_extend_empty(self):
        events = Pattern([]).ex(60, 1.0).events
        self.assertEqual(events, [(60.0, 1.0)])

    def test_extend_coerces_to_float(self):
        events = Pattern([]).ex(60, 1).events
        self.assertEqual(events, [(60.0, 1.0)])
        self.assertIsInstance(events[0][0], float)
        self.assertIsInstance(events[0][1], float)


class TestCat(unittest.TestCase):
    def test_cat_nonempty(self):
        a = Pattern([(60, 0.25)])
        b = Pattern([(62, 0.5)])
        self.assertEqual(a.cat(b).events, [(60.0, 0.25), (62.0, 0.5)])

    def test_cat_empty_right(self):
        a = Pattern([(60, 0.25)])
        self.assertEqual(a.cat(Pattern([])).events, [(60.0, 0.25)])

    def test_cat_empty_left(self):
        a = Pattern([(60, 0.25)])
        self.assertEqual(Pattern([]).cat(a).events, [(60.0, 0.25)])

    def test_cat_uses_left_chain(self):
        def dummy_fx(node, **_):
            return node

        a = Pattern([(60, 0.25)]).mod(dummy_fx, depth=1)
        b = Pattern([(62, 0.5)]).dist(dummy_fx, amount=2)
        result = a.cat(b)
        # Chain comes from ``a`` only; ``b``'s chain is discarded.
        self.assertEqual(len(result.chain), 1)
        self.assertEqual(result.chain[0][0], "mod")


class TestPre(unittest.TestCase):
    def test_pre_prepends(self):
        a = Pattern([(60, 0.25)])
        b = Pattern([(62, 0.5)])
        self.assertEqual(a.pre(b).events, [(62.0, 0.5), (60.0, 0.25)])

    def test_pre_uses_left_chain(self):
        def dummy_fx(node, **_):
            return node

        a = Pattern([(60, 0.25)]).mod(dummy_fx)
        b = Pattern([(62, 0.5)]).dist(dummy_fx)
        result = a.pre(b)
        self.assertEqual(len(result.chain), 1)
        self.assertEqual(result.chain[0][0], "mod")


# ---------------------------------------------------------------------------
# Synthesis transforms
# ---------------------------------------------------------------------------

def _fm_mod(input_node, mod_index: float = 2.0):
    """Stand-in DSP callable for tests. Matches the spec signature
    ``fn(input_node, **params) -> Node`` but returns ``input_node`` so
    the tests don't need SignalFlow to run."""
    return input_node


class TestMod(unittest.TestCase):
    def test_chain_entry_shape(self):
        p = Pattern([(60, 0.25)]).mod(_fm_mod, mod_index=3.0)
        self.assertEqual(p.chain, [("mod", _fm_mod, {"mod_index": 3.0})])

    def test_events_untouched(self):
        p = Pattern([(60, 0.25)]).mod(_fm_mod, mod_index=3.0)
        self.assertEqual(p.events, [(60.0, 0.25)])


class TestDist(unittest.TestCase):
    def test_chain_entry_shape(self):
        p = Pattern([(60, 0.25)]).dist(_fm_mod, amount=0.5)
        self.assertEqual(p.chain, [("dist", _fm_mod, {"amount": 0.5})])


class TestBF(unittest.TestCase):
    def test_lpf_with_default_resonance(self):
        p = Pattern([(60, 0.25)]).bf("lpf", 1000)
        self.assertEqual(p.chain, [("bf", "lpf", 1000.0, 0.0)])

    def test_bpf_with_resonance(self):
        p = Pattern([(60, 0.25)]).bf("bpf", 800, 0.7)
        self.assertEqual(p.chain, [("bf", "bpf", 800.0, 0.7)])

    def test_hpf_and_notch_accepted(self):
        self.assertEqual(
            Pattern([(60, 0.25)]).bf("hpf", 200).chain[0][1], "hpf",
        )
        self.assertEqual(
            Pattern([(60, 0.25)]).bf("notch", 400).chain[0][1], "notch",
        )

    def test_rejects_bad_type(self):
        p = Pattern([(60, 0.25)])
        with self.assertRaises(ValueError):
            p.bf("badtype", 1000)
        # Allpass is explicitly excluded.
        with self.assertRaises(ValueError):
            p.bf("allpass", 1000)


class TestSpec(unittest.TestCase):
    def test_chain_entry_shape(self):
        p = Pattern([(60, 0.25)]).spec(_fm_mod, size=1024)
        self.assertEqual(p.chain, [("spec", _fm_mod, {"size": 1024})])


class TestIR(unittest.TestCase):
    def test_chain_entry_shape(self):
        p = Pattern([(60, 0.25)]).ir(_fm_mod, path="room.wav")
        self.assertEqual(p.chain, [("ir", _fm_mod, {"path": "room.wav"})])


class TestGate(unittest.TestCase):
    def test_chain_entry_shape(self):
        g = Pattern([(1, 0.25), (0, 0.25)])
        p = Pattern([(60, 2.0)]).gate(g)
        self.assertEqual(p.chain, [("gate", g)])


class TestFx(unittest.TestCase):
    def test_chain_entry_shape(self):
        p = Pattern([(60, 0.25)]).fx(_fm_mod, feedback=0.3)
        self.assertEqual(p.chain, [("fx", _fm_mod, {"feedback": 0.3})])


# ---------------------------------------------------------------------------
# Chaining and compound behaviour
# ---------------------------------------------------------------------------

class TestChaining(unittest.TestCase):
    def test_long_chain_order(self):
        p = (Pattern([(60, 0.25)])
             .mod(_fm_mod, depth=1)
             .dist(_fm_mod, amount=0.5)
             .bf("lpf", 800))
        kinds = [entry[0] for entry in p.chain]
        self.assertEqual(kinds, ["mod", "dist", "bf"])

    def test_variations_are_independent(self):
        base = Pattern([(60, 0.25), (64, 0.25), (67, 0.25)])
        a = base.fast(2).t_p(5)
        b = base.slow(2).mod(_fm_mod)
        self.assertEqual(base.events, [(60.0, 0.25), (64.0, 0.25), (67.0, 0.25)])
        self.assertEqual(a.events, [(65.0, 0.125), (69.0, 0.125), (72.0, 0.125)])
        self.assertEqual(b.events, [(60.0, 0.5), (64.0, 0.5), (67.0, 0.5)])
        self.assertEqual(a.chain, [])
        self.assertEqual(len(b.chain), 1)


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

class _RecordingGraph:
    """Minimal test double for an AudioGraph-like object."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def wait(self, dur: float) -> None:
        self.waits.append(dur)


class _RecordingVoice:
    """Minimal test double for a SignalFlow voice node."""

    def __init__(self, note: float) -> None:
        self.note = note
        self.events: list[str] = []

    def play(self) -> None:
        self.events.append("play")

    def stop(self) -> None:
        self.events.append("stop")


class TestPlay(unittest.TestCase):
    def test_empty_pattern_returns_immediately(self):
        graph = _RecordingGraph()
        shape_calls: list[float] = []

        def shape(note):
            shape_calls.append(note)
            return _RecordingVoice(note)

        Pattern([]).play(shape, graph)

        self.assertEqual(shape_calls, [])
        self.assertEqual(graph.waits, [])

    def test_empty_chain_plays_sequentially(self):
        graph = _RecordingGraph()
        voices: list[_RecordingVoice] = []

        def shape(note):
            v = _RecordingVoice(note)
            voices.append(v)
            return v

        Pattern([(60, 0.5), (62, 0.25)]).play(shape, graph)

        self.assertEqual([v.note for v in voices], [60.0, 62.0])
        self.assertEqual(graph.waits, [0.5, 0.25])
        for v in voices:
            self.assertEqual(v.events, ["play", "stop"])

    def test_chain_applied_per_event(self):
        """Custom DSP callables are invoked once per event during .play()."""
        graph = _RecordingGraph()
        calls: list[tuple[float, dict]] = []

        def dummy_mod(node, **params):
            calls.append((node.note, dict(params)))
            return node

        def shape(note):
            return _RecordingVoice(note)

        p = Pattern([(60, 0.25), (62, 0.5)]).mod(dummy_mod, depth=0.7)
        p.play(shape, graph)

        self.assertEqual(
            calls, [(60.0, {"depth": 0.7}), (62.0, {"depth": 0.7})]
        )
        self.assertEqual(graph.waits, [0.25, 0.5])


# ---------------------------------------------------------------------------
# Phase 2: chain application
# ---------------------------------------------------------------------------

class _MarkerNode:
    """Test double for a SignalFlow node. Behaves like the real thing for
    chain-walking purposes: supports ``play``/``stop`` and ``*`` so the
    gate multiplication path works without SignalFlow installed.
    """

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.events: list[str] = []

    def play(self) -> None:
        self.events.append("play")

    def stop(self) -> None:
        self.events.append("stop")

    def __mul__(self, other):
        return _MarkerNode(f"{self.tag}*{getattr(other, 'tag', other)}")

    def __repr__(self) -> str:  # pragma: no cover
        return f"_MarkerNode({self.tag!r})"


class TestApplyChain(unittest.TestCase):
    def test_empty_chain_passes_through(self):
        p = Pattern([(60, 0.25)])
        voice = _MarkerNode("voice")
        self.assertIs(p._apply_chain(voice), voice)

    def test_mod_dist_spec_ir_fx_all_invoke_callable(self):
        seen: list[tuple[str, str, dict]] = []

        def make_wrapper(kind: str):
            def wrapper(node, **params):
                seen.append((kind, node.tag, dict(params)))
                return _MarkerNode(f"{kind}({node.tag})")
            return wrapper

        wrapped_mod = make_wrapper("mod")
        wrapped_dist = make_wrapper("dist")
        wrapped_spec = make_wrapper("spec")
        wrapped_ir = make_wrapper("ir")
        wrapped_fx = make_wrapper("fx")

        p = (Pattern([(60, 0.25)])
             .mod(wrapped_mod, a=1)
             .dist(wrapped_dist, b=2)
             .spec(wrapped_spec, c=3)
             .ir(wrapped_ir, d=4)
             .fx(wrapped_fx, e=5))

        voice = _MarkerNode("voice")
        out = p._apply_chain(voice)

        self.assertEqual(
            [k for (k, _t, _p) in seen],
            ["mod", "dist", "spec", "ir", "fx"],
        )
        # The outer wrapper sees the next-inner wrapper's tag.
        self.assertEqual(
            [t for (_k, t, _p) in seen],
            ["voice", "mod(voice)", "dist(mod(voice))",
             "spec(dist(mod(voice)))", "ir(spec(dist(mod(voice))))"],
        )
        self.assertEqual(out.tag, "fx(ir(spec(dist(mod(voice)))))")

    def test_bf_delegates_to_helper(self):
        p = Pattern([(60, 0.25)]).bf("lpf", 800, 0.3)
        voice = _MarkerNode("v")

        calls: list[tuple] = []

        def fake_bf(self, node, ftype, cutoff, res):
            calls.append((node.tag, ftype, cutoff, res))
            return _MarkerNode(f"bf({ftype},{cutoff},{res})")

        # Patch the bound helper for this instance only — keeps the test
        # independent of SignalFlow's presence / absence.
        p._apply_builtin_filter = fake_bf.__get__(p, Pattern)
        out = p._apply_chain(voice)

        self.assertEqual(calls, [("v", "lpf", 800.0, 0.3)])
        self.assertEqual(out.tag, "bf(lpf,800.0,0.3)")

    def test_gate_delegates_to_helper(self):
        g = Pattern([(1, 0.25), (0, 0.25)])
        p = Pattern([(60, 2.0)]).gate(g)
        voice = _MarkerNode("v")

        calls: list[tuple] = []

        def fake_gate(self, node, rhythm):
            calls.append((node.tag, rhythm.events))
            return _MarkerNode("gated")

        p._apply_gate = fake_gate.__get__(p, Pattern)
        out = p._apply_chain(voice)

        self.assertEqual(calls, [("v", [(1.0, 0.25), (0.0, 0.25)])])
        self.assertEqual(out.tag, "gated")

    def test_unknown_chain_entry_raises(self):
        p = Pattern([(60, 0.25)])
        # Inject a bogus entry — Pattern's public methods never produce
        # one, so this checks the defensive branch in _apply_chain.
        p.chain.append(("bogus", None))
        with self.assertRaises(ValueError):
            p._apply_chain(_MarkerNode("v"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
