"""SignalFlow integration tests for Pattern DSP application.

These tests build real SignalFlow nodes and therefore require:

- The ``signalflow`` package installed.
- A host with a working audio output device (or a SignalFlow build that
  supports a headless backend).

Because SignalFlow segfaults on some headless CI hosts when probing for
an output device, this module is opt-in: set the environment variable
``MDMA_AUDIO_TESTS=1`` to enable it. Default CI runs skip everything
here. Per SKILL.md's testing conventions, audio-dependent tests live
under ``tests/audio/``.

Run with::

    MDMA_AUDIO_TESTS=1 python -m unittest tests.audio.test_pattern_dsp
"""

from __future__ import annotations

import os
import unittest

try:
    import signalflow as sf  # type: ignore
    _HAS_SF = True
except ImportError:  # pragma: no cover
    sf = None  # type: ignore
    _HAS_SF = False


_AUDIO_OPT_IN = os.environ.get("MDMA_AUDIO_TESTS") == "1"


def _graph_available() -> bool:
    """True if we can stand up an :class:`signalflow.AudioGraph`.

    Guarded by ``MDMA_AUDIO_TESTS`` so CI never attempts the probe.
    """
    if not (_HAS_SF and _AUDIO_OPT_IN):
        return False
    existing = sf.AudioGraph.get_shared_graph()
    if existing is not None:
        return True
    try:
        sf.AudioGraph(start=False)
    except Exception:
        return False
    return True


_SKIP_REASON = (
    "signalflow AudioGraph not available (set MDMA_AUDIO_TESTS=1 to "
    "enable these tests on a host with a working audio device)"
)


@unittest.skipUnless(_graph_available(), _SKIP_REASON)
class TestBuiltinFilter(unittest.TestCase):
    def test_lpf_produces_svfilter(self):
        from mdma_rebuild.backend.pattern import Pattern

        p = Pattern([(60, 0.25)]).bf("lpf", 800, 0.3)
        voice = sf.SineOscillator(440)
        out = p._apply_chain(voice)
        self.assertIsInstance(out, sf.SVFilter)

    def test_all_four_types_build(self):
        from mdma_rebuild.backend.pattern import Pattern

        for ftype in ("lpf", "hpf", "bpf", "notch"):
            p = Pattern([(60, 0.25)]).bf(ftype, 1000, 0.2)
            voice = sf.SineOscillator(440)
            out = p._apply_chain(voice)
            self.assertIsInstance(out, sf.SVFilter, f"type {ftype}")


@unittest.skipUnless(_graph_available(), _SKIP_REASON)
class TestGate(unittest.TestCase):
    def test_gate_returns_multiplied_node(self):
        from mdma_rebuild.backend.pattern import Pattern

        rhythm = Pattern([(1, 0.25), (0, 0.25)])
        p = Pattern([(60, 2.0)]).gate(rhythm)
        voice = sf.SineOscillator(440)
        out = p._apply_chain(voice)
        # Multiplying two Nodes in SignalFlow returns a Multiply node.
        self.assertIsInstance(out, sf.Node)
        # The output should incorporate the original voice somewhere in
        # the graph — at minimum it should not simply be the voice.
        self.assertIsNot(out, voice)

    def test_empty_rhythm_is_noop(self):
        from mdma_rebuild.backend.pattern import Pattern

        p = Pattern([(60, 2.0)]).gate(Pattern([]))
        voice = sf.SineOscillator(440)
        out = p._apply_chain(voice)
        self.assertIs(out, voice)

    def test_gate_rhythm_shorter_than_host_is_legal(self):
        """A 0.5-second rhythm driving a 2-second host event is fine —
        the looping BufferPlayer handles the repeat."""
        from mdma_rebuild.backend.pattern import Pattern

        rhythm = Pattern([(1, 0.125), (0, 0.125), (1, 0.125), (0, 0.125)])
        p = Pattern([(60, 4.0)]).gate(rhythm)
        voice = sf.SineOscillator(440)
        out = p._apply_chain(voice)
        self.assertIsNot(out, voice)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
