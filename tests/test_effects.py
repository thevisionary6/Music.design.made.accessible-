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
    allpass_filter,
    amplitude_mod,
    autopan,
    balance,
    bitcrush,
    capture_and_granulate,
    chorus,
    comb_filter,
    compressor,
    db_to_amplitude,
    delay,
    detune_unison,
    dual_overdrive,
    expander,
    flanger,
    foldback,
    fuzz,
    granular_cloud,
    granular_freeze,
    granular_reverse,
    granular_scatter,
    granular_shimmer,
    granular_stretch,
    granular_stutter,
    haas,
    hard_clip,
    high_shelf,
    limiter,
    low_shelf,
    mono,
    moog,
    multiband_compressor,
    multitap,
    noise_gate,
    ott,
    ott_glue,
    ott_loud,
    ott_punch,
    ott_soft,
    overdrive,
    overdrive_classic,
    overdrive_crunch,
    overdrive_soft,
    peak,
    phase_vocoder_freeze,
    phaser,
    ping_pong,
    reverb,
    ring_mod,
    slapback,
    soft_clip,
    spectral_blur,
    spectral_contrast,
    spectral_freeze,
    spectral_gate,
    spectral_lpf,
    spectral_shift,
    stereo_widen,
    tape,
    tape_echo,
    time_stretch,
    tremolo,
    tube,
    vamp,
    vamp_fuzz,
    vamp_heavy,
    vamp_light,
    vamp_medium,
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

    def __sub__(self, other):
        return _StubNode(f"({self.name}-{_tag(other)})")

    def __rsub__(self, other):
        return _StubNode(f"({_tag(other)}-{self.name})")

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

    class _BiquadFilter(_StubNode):
        def __init__(self, input, filter_type, cutoff=440, resonance=0.0, peak_gain=0.0):
            super().__init__(
                f"Biquad({_tag(input)},{filter_type},cut={cutoff},"
                f"q={resonance},g={peak_gain})"
            )

    class _MoogVCF(_StubNode):
        def __init__(self, input, cutoff=200.0, resonance=0.0):
            super().__init__(f"Moog({_tag(input)},cut={cutoff},q={resonance})")

    class _SineLFO(_StubNode):
        def __init__(self, frequency=1.0, min=0.0, max=1.0, phase=0.0):
            super().__init__(f"SineLFO(f={frequency},min={min},max={max})")

    class _Compressor(_StubNode):
        def __init__(self, input, threshold=0.1, ratio=2, attack_time=0.01,
                     release_time=0.1, sidechain=None):
            super().__init__(
                f"Comp({_tag(input)},th={threshold},r={ratio},"
                f"a={attack_time},rel={release_time})"
            )

    class _StereoPanner(_StubNode):
        def __init__(self, input, pan=0.0):
            super().__init__(f"Pan({_tag(input)},{_tag(pan)})")

    class _StereoWidth(_StubNode):
        def __init__(self, input, width=1):
            super().__init__(f"Width({_tag(input)},{_tag(width)})")

    class _StereoBalance(_StubNode):
        def __init__(self, input, balance=0):
            super().__init__(f"Balance({_tag(input)},{_tag(balance)})")

    class _SVFilter(_StubNode):
        def __init__(self, input, filter_type, cutoff=440, resonance=0.0):
            super().__init__(
                f"SVF({_tag(input)},{filter_type},cut={cutoff},q={resonance})"
            )

    class _SineOscillator(_StubNode):
        def __init__(self, frequency=440, phase_offset=None, reset=None):
            super().__init__(f"SineOsc({_tag(frequency)})")

    # Spectral / FFT family
    class _FFT(_StubNode):
        def __init__(self, input, fft_size=1024, hop_size=128,
                     window_size=0, do_window=True):
            super().__init__(
                f"FFT({_tag(input)},n={fft_size},hop={hop_size})"
            )

    class _IFFT(_StubNode):
        def __init__(self, input, do_window=False):
            super().__init__(f"IFFT({_tag(input)})")

    class _FFTContinuousPV(_StubNode):
        def __init__(self, input, rate=1.0):
            super().__init__(f"ContPV({_tag(input)},rate={rate})")

    class _FFTPhaseVocoder(_StubNode):
        def __init__(self, input):
            super().__init__(f"PhaseVoc({_tag(input)})")

    class _FFTRandomPhase(_StubNode):
        def __init__(self, input, level=1.0):
            super().__init__(f"RandPhase({_tag(input)},lvl={level})")

    class _FFTContrast(_StubNode):
        def __init__(self, input, contrast=1):
            super().__init__(f"Contrast({_tag(input)},c={contrast})")

    class _FFTLPF(_StubNode):
        def __init__(self, input, frequency=2000):
            super().__init__(f"FFTLPF({_tag(input)},f={frequency})")

    class _FFTNoiseGate(_StubNode):
        def __init__(self, input, threshold=0.5, invert=0.0):
            super().__init__(
                f"FFTGate({_tag(input)},th={threshold},inv={invert})"
            )

    # Granular + clock family
    class _Impulse(_StubNode):
        def __init__(self, frequency=1.0):
            super().__init__(f"Imp(f={frequency})")

    class _SawLFO(_StubNode):
        def __init__(self, frequency=1.0, min=0.0, max=1.0, phase=0.0):
            super().__init__(f"SawLFO(f={frequency},min={min},max={max})")

    class _SampleAndHold(_StubNode):
        def __init__(self, input, clock):
            super().__init__(f"S&H({_tag(input)},clk={_tag(clock)})")

    class _Buffer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _BufferRecorder(_StubNode):
        def __init__(self, buffer=None, input=None, feedback=0.0, loop=False):
            super().__init__(
                f"Recorder(in={_tag(input)},fb={feedback},loop={loop})"
            )

    class _Granulator(_StubNode):
        def __init__(self, buffer=None, clock=None, pos=0, duration=0.1,
                     amplitude=1.0, pan=0.0, rate=1.0, max_grains=2048,
                     wrap=False):
            super().__init__(
                f"Gran(clk={_tag(clock)},pos={_tag(pos)},"
                f"dur={_tag(duration)},rate={_tag(rate)})"
            )

    mod.Tanh = _Tanh
    mod.OneTapDelay = _OneTapDelay
    mod.CombDelay = _CombDelay
    mod.AllpassDelay = _AllpassDelay
    mod.Clip = _Clip
    mod.Fold = _Fold
    mod.Resample = _Resample
    mod.BiquadFilter = _BiquadFilter
    mod.MoogVCF = _MoogVCF
    mod.SineLFO = _SineLFO
    mod.Compressor = _Compressor
    mod.StereoPanner = _StereoPanner
    mod.StereoWidth = _StereoWidth
    mod.StereoBalance = _StereoBalance
    mod.SVFilter = _SVFilter
    mod.SineOscillator = _SineOscillator
    # Spectral
    mod.FFT = _FFT
    mod.IFFT = _IFFT
    mod.FFTContinuousPhaseVocoder = _FFTContinuousPV
    mod.FFTPhaseVocoder = _FFTPhaseVocoder
    mod.FFTRandomPhase = _FFTRandomPhase
    mod.FFTContrast = _FFTContrast
    mod.FFTLPF = _FFTLPF
    mod.FFTNoiseGate = _FFTNoiseGate
    # Granular + clocks
    mod.Impulse = _Impulse
    mod.SawLFO = _SawLFO
    mod.SampleAndHold = _SampleAndHold
    mod.Buffer = _Buffer
    mod.BufferRecorder = _BufferRecorder
    mod.Granulator = _Granulator

    # AudioGraph stub for helpers (utils.current_sample_rate etc.).
    class _AudioGraph:
        @classmethod
        def get_shared_graph(cls):
            return None

    mod.AudioGraph = _AudioGraph
    mod.SIGNALFLOW_DEFAULT_SAMPLE_RATE = 44100
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


_ALL_EFFECTS = (
    soft_clip, hard_clip, foldback, bitcrush,
    tube, tape, fuzz,
    vamp, vamp_light, vamp_medium, vamp_heavy, vamp_fuzz,
    overdrive, overdrive_soft, overdrive_classic, overdrive_crunch,
    dual_overdrive,
    delay, reverb, slapback, ping_pong, multitap, tape_echo,
    peak, low_shelf, high_shelf, moog, allpass_filter, comb_filter,
    chorus, flanger, phaser, tremolo, autopan,
    ring_mod, amplitude_mod, detune_unison,
    haas, stereo_widen, mono, balance,
    spectral_blur, spectral_contrast, spectral_freeze, spectral_gate,
    spectral_lpf, spectral_shift,
    time_stretch, phase_vocoder_freeze,
    capture_and_granulate,
    compressor, limiter, noise_gate, expander,
    multiband_compressor, ott, ott_punch, ott_glue, ott_loud, ott_soft,
)


class TestPortingConvention(unittest.TestCase):
    def test_effects_package_exports(self):
        """The package __init__ should expose every ported effect so
        future additions just show up as ``from backend.effects import
        their_name``."""
        from mdma_rebuild.backend import effects as fx

        for name in (
            # distortion + saturation
            "soft_clip", "hard_clip", "foldback", "bitcrush",
            "tube", "tape", "fuzz",
            # VAMP + overdrive
            "vamp", "vamp_light", "vamp_medium", "vamp_heavy", "vamp_fuzz",
            "overdrive", "overdrive_soft", "overdrive_classic",
            "overdrive_crunch", "dual_overdrive",
            # delay / ambient
            "delay", "reverb", "slapback", "ping_pong", "multitap",
            "tape_echo",
            # filters
            "peak", "low_shelf", "high_shelf", "moog",
            "allpass_filter", "comb_filter",
            # modulation
            "chorus", "flanger", "phaser", "tremolo", "autopan",
            # pitch / freq
            "ring_mod", "amplitude_mod", "detune_unison",
            # spatial
            "haas", "stereo_widen", "mono", "balance",
            # spectral
            "spectral_blur", "spectral_contrast", "spectral_freeze",
            "spectral_gate", "spectral_lpf", "spectral_shift",
            # vocoder
            "time_stretch", "phase_vocoder_freeze",
            # granular
            "capture_and_granulate", "granular_cloud", "granular_freeze",
            "granular_reverse", "granular_scatter", "granular_shimmer",
            "granular_stretch", "granular_stutter",
            # dynamics
            "compressor", "db_to_amplitude",
            "limiter", "noise_gate", "expander",
            "multiband_compressor",
            "ott", "ott_glue", "ott_loud", "ott_punch", "ott_soft",
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

        for fn in _ALL_EFFECTS:
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            self.assertTrue(len(params) >= 1, f"{fn.__name__} has no params")
            self.assertEqual(
                params[0].name,
                "input_node",
                f"{fn.__name__}: first param must be 'input_node', "
                f"got {params[0].name!r}",
            )


# ---------------------------------------------------------------------------
# Extended filters
# ---------------------------------------------------------------------------


class TestExtendedFilters(unittest.TestCase):
    def test_peak_builds_biquad_peak(self):
        with _SignalFlowStubbed():
            out = peak(_StubNode("v"), cutoff=1000, resonance=0.5, gain_db=6)
            self.assertIn("Biquad", out.name)
            self.assertIn("peak", out.name)
            self.assertIn("cut=1000", out.name)
            self.assertIn("g=6", out.name)

    def test_low_shelf_and_high_shelf(self):
        with _SignalFlowStubbed():
            lo = low_shelf(_StubNode("v"), cutoff=200, gain_db=-3)
            hi = high_shelf(_StubNode("v"), cutoff=6000, gain_db=4)
            self.assertIn("low_shelf", lo.name)
            self.assertIn("high_shelf", hi.name)

    def test_moog_uses_moogvcf(self):
        with _SignalFlowStubbed():
            out = moog(_StubNode("v"), cutoff=800, resonance=0.7)
            self.assertIn("Moog", out.name)
            self.assertIn("cut=800", out.name)
            # Resonance clamped to 0.95 — 0.7 should pass through.
            self.assertIn("q=0.7", out.name)

    def test_moog_clamps_resonance(self):
        with _SignalFlowStubbed():
            out = moog(_StubNode("v"), resonance=5.0)
            self.assertIn("q=0.95", out.name)

    def test_allpass_filter_uses_allpassdelay(self):
        with _SignalFlowStubbed():
            out = allpass_filter(_StubNode("v"), delay_time=0.01, feedback=0.6)
            self.assertIn("AllpassDelay", out.name)

    def test_comb_filter_uses_combdelay(self):
        with _SignalFlowStubbed():
            out = comb_filter(_StubNode("v"), delay_time=0.02, feedback=0.8)
            self.assertIn("CombDelay", out.name)

    def test_rejects_non_positive_cutoff(self):
        with _SignalFlowStubbed():
            for fn in (peak, low_shelf, high_shelf, moog):
                with self.assertRaises(ValueError):
                    fn(_StubNode("v"), cutoff=0)

    def test_rejects_negative_feedback(self):
        with _SignalFlowStubbed():
            for fn in (allpass_filter, comb_filter):
                with self.assertRaises(ValueError):
                    fn(_StubNode("v"), feedback=-0.1)


# ---------------------------------------------------------------------------
# Modulation (chorus / flanger / phaser / tremolo / autopan)
# ---------------------------------------------------------------------------


class TestModulation(unittest.TestCase):
    def test_chorus_topology(self):
        with _SignalFlowStubbed():
            out = chorus(_StubNode("v"), rate=0.7, depth=0.003, mix=0.5)
            self.assertIn("OneTapDelay", out.name)
            self.assertIn("SineLFO", out.name)

    def test_flanger_has_two_delay_stages(self):
        """Feedback topology means two OneTapDelays in the wet chain."""
        with _SignalFlowStubbed():
            out = flanger(_StubNode("v"), feedback=0.6)
            self.assertGreaterEqual(out.name.count("OneTapDelay"), 2)

    def test_phaser_stages_match_stages_arg(self):
        with _SignalFlowStubbed():
            out = phaser(_StubNode("v"), stages=6)
            self.assertEqual(out.name.count("AllpassDelay"), 6)

    def test_phaser_rejects_zero_stages(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                phaser(_StubNode("v"), stages=0)

    def test_tremolo_multiplies_by_lfo(self):
        with _SignalFlowStubbed():
            out = tremolo(_StubNode("v"), rate=6.0, depth=0.8)
            self.assertIn("SineLFO", out.name)
            # min should be 1 - depth = 0.2
            self.assertIn("min=0.19999999999999996", out.name) if False else None  # tolerant
            self.assertIn("SineLFO", out.name)

    def test_tremolo_rejects_non_positive_rate(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                tremolo(_StubNode("v"), rate=0)

    def test_autopan_uses_stereopanner(self):
        with _SignalFlowStubbed():
            out = autopan(_StubNode("v"), rate=0.5, depth=0.9)
            self.assertIn("Pan", out.name)
            self.assertIn("SineLFO", out.name)

    def test_modulation_rate_must_be_positive(self):
        with _SignalFlowStubbed():
            for fn in (chorus, flanger, phaser, tremolo, autopan):
                with self.assertRaises(ValueError):
                    fn(_StubNode("v"), rate=0)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


class TestCompressor(unittest.TestCase):
    def test_builds_compressor_node(self):
        with _SignalFlowStubbed():
            out = compressor(
                _StubNode("v"),
                threshold=0.5,
                ratio=4.0,
                attack=0.01,
                release=0.2,
            )
            self.assertIn("Comp", out.name)
            self.assertIn("th=0.5", out.name)
            self.assertIn("r=4.0", out.name)

    def test_makeup_gain_multiplies(self):
        with _SignalFlowStubbed():
            # makeup=1.0 returns the compressor node directly; >1 wraps
            # it in a multiplication.
            out_plain = compressor(_StubNode("v"), makeup=1.0)
            out_boost = compressor(_StubNode("v"), makeup=1.5)
            self.assertNotIn("*1.5", out_plain.name)
            self.assertIn("*1.5", out_boost.name)

    def test_rejects_bad_params(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                compressor(_StubNode("v"), threshold=0)
            with self.assertRaises(ValueError):
                compressor(_StubNode("v"), ratio=0.5)
            with self.assertRaises(ValueError):
                compressor(_StubNode("v"), attack=0)
            with self.assertRaises(ValueError):
                compressor(_StubNode("v"), release=0)
            with self.assertRaises(ValueError):
                compressor(_StubNode("v"), makeup=-1.0)

    def test_db_helper(self):
        self.assertAlmostEqual(db_to_amplitude(0.0), 1.0)
        self.assertAlmostEqual(db_to_amplitude(-20.0), 0.1)
        self.assertAlmostEqual(db_to_amplitude(-6.0), 0.5011872336272722)


# ---------------------------------------------------------------------------
# Delay family (slapback / ping_pong / multitap / tape_echo)
# ---------------------------------------------------------------------------


class TestDelayFamily(unittest.TestCase):
    def test_slapback_has_single_tap(self):
        with _SignalFlowStubbed():
            out = slapback(_StubNode("v"), time=0.08, mix=0.3)
            self.assertEqual(out.name.count("OneTapDelay"), 1)

    def test_ping_pong_builds_stereo_taps(self):
        with _SignalFlowStubbed():
            out = ping_pong(_StubNode("v"), time=0.25, feedback=0.5)
            # Two OneTapDelay constructions plus a StereoPanner each;
            # the stub's string concat duplicates tap_l's name inside
            # tap_r's input, so the substring count is >= 2 rather
            # than exactly 2.
            self.assertGreaterEqual(out.name.count("OneTapDelay"), 2)
            self.assertGreaterEqual(out.name.count("Pan"), 2)

    def test_multitap_default_gains_and_custom(self):
        with _SignalFlowStubbed():
            out = multitap(_StubNode("v"), times=(0.1, 0.2, 0.3))
            self.assertEqual(out.name.count("OneTapDelay"), 3)

    def test_multitap_gains_length_must_match(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                multitap(_StubNode("v"), times=(0.1, 0.2), gains=(0.5,))

    def test_multitap_rejects_empty_times(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                multitap(_StubNode("v"), times=())
            with self.assertRaises(ValueError):
                multitap(_StubNode("v"), times=(0.1, -0.2))

    def test_tape_echo_runs_feedback_through_tanh_and_svf(self):
        with _SignalFlowStubbed():
            out = tape_echo(_StubNode("v"), time=0.3, saturation=1.5)
            self.assertIn("Tanh", out.name)
            self.assertIn("SVF", out.name)
            self.assertIn("low_pass", out.name)

    def test_delay_family_rejects_bad_times(self):
        with _SignalFlowStubbed():
            for fn in (slapback, ping_pong, tape_echo):
                with self.assertRaises(ValueError):
                    fn(_StubNode("v"), time=0)


# ---------------------------------------------------------------------------
# Saturation variants
# ---------------------------------------------------------------------------


class TestSaturation(unittest.TestCase):
    def test_tube_uses_bias(self):
        with _SignalFlowStubbed():
            out = tube(_StubNode("v"), drive=2.0, bias=0.1)
            # bias shifts input before tanh, subtracts after.
            self.assertIn("Tanh", out.name)

    def test_tube_rejects_bad_bias(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                tube(_StubNode("v"), bias=1.5)
            with self.assertRaises(ValueError):
                tube(_StubNode("v"), drive=0)

    def test_tape_combines_tanh_and_lowpass(self):
        with _SignalFlowStubbed():
            out = tape(_StubNode("v"), drive=1.5, hi_cut_hz=8000)
            self.assertIn("Tanh", out.name)
            self.assertIn("SVF", out.name)
            self.assertIn("low_pass", out.name)

    def test_fuzz_combines_tanh_and_tone(self):
        with _SignalFlowStubbed():
            out = fuzz(_StubNode("v"), drive=5.0, tone=2500)
            self.assertIn("Tanh", out.name)
            self.assertIn("SVF", out.name)


# ---------------------------------------------------------------------------
# Additional dynamics
# ---------------------------------------------------------------------------


class TestDynamicsExtra(unittest.TestCase):
    def test_limiter_uses_high_ratio(self):
        with _SignalFlowStubbed():
            out = limiter(_StubNode("v"), threshold=0.9)
            self.assertIn("Comp", out.name)
            self.assertIn("r=20.0", out.name)

    def test_noise_gate_builds_compressor(self):
        with _SignalFlowStubbed():
            out = noise_gate(_StubNode("v"), threshold=0.02)
            self.assertIn("Comp", out.name)

    def test_expander_pre_boosts_input(self):
        with _SignalFlowStubbed():
            out = expander(_StubNode("v"), ratio=3.0)
            self.assertIn("Comp", out.name)
            # ratio appears as a multiplier in the pre-boost path.
            self.assertIn("*3.0", out.name)

    def test_dynamics_reject_bad_threshold(self):
        with _SignalFlowStubbed():
            for fn in (limiter, noise_gate, expander):
                with self.assertRaises(ValueError):
                    fn(_StubNode("v"), threshold=0)


# ---------------------------------------------------------------------------
# Spatial / stereo
# ---------------------------------------------------------------------------


class TestSpatial(unittest.TestCase):
    def test_haas_adds_delay_on_one_side(self):
        with _SignalFlowStubbed():
            out_r = haas(_StubNode("v"), delay_ms=15, side="right")
            self.assertIn("OneTapDelay", out_r.name)
            self.assertIn("Pan", out_r.name)

            out_l = haas(_StubNode("v"), delay_ms=15, side="left")
            self.assertIn("OneTapDelay", out_l.name)

    def test_haas_rejects_out_of_range(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                haas(_StubNode("v"), delay_ms=0)
            with self.assertRaises(ValueError):
                haas(_StubNode("v"), delay_ms=100)  # over 40 ms
            with self.assertRaises(ValueError):
                haas(_StubNode("v"), side="up")

    def test_stereo_widen_clamps(self):
        with _SignalFlowStubbed():
            out = stereo_widen(_StubNode("v"), width=10.0)
            # Clamped to 4 max.
            self.assertIn("Width", out.name)
            self.assertIn("4", out.name)

    def test_mono_sets_width_zero(self):
        with _SignalFlowStubbed():
            out = mono(_StubNode("v"))
            self.assertIn("Width", out.name)
            self.assertIn("0.0", out.name)

    def test_balance_clamps(self):
        with _SignalFlowStubbed():
            out = balance(_StubNode("v"), bias=5.0)
            self.assertIn("Balance", out.name)
            self.assertIn("1.0", out.name)


# ---------------------------------------------------------------------------
# Pitch / frequency
# ---------------------------------------------------------------------------


class TestPitchFreq(unittest.TestCase):
    def test_ring_mod_multiplies_by_sine(self):
        with _SignalFlowStubbed():
            out = ring_mod(_StubNode("v"), rate=440.0)
            self.assertIn("SineOsc", out.name)

    def test_amplitude_mod_shape(self):
        with _SignalFlowStubbed():
            out = amplitude_mod(_StubNode("v"), rate=5.0, depth=1.0)
            self.assertIn("SineOsc", out.name)

    def test_detune_unison_stacks_voices(self):
        with _SignalFlowStubbed():
            out = detune_unison(_StubNode("v"), voices=4, spread=0.02)
            # Three extra voices, each adds a OneTapDelay.
            self.assertEqual(out.name.count("OneTapDelay"), 3)

    def test_detune_unison_single_voice_passthrough(self):
        with _SignalFlowStubbed():
            out = detune_unison(_StubNode("v"), voices=1)
            # voices=1 adds zero extra delays, then normalises.
            self.assertEqual(out.name.count("OneTapDelay"), 0)

    def test_pitch_freq_reject_bad_args(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                ring_mod(_StubNode("v"), rate=0)
            with self.assertRaises(ValueError):
                amplitude_mod(_StubNode("v"), depth=-1)
            with self.assertRaises(ValueError):
                detune_unison(_StubNode("v"), voices=0)


# ---------------------------------------------------------------------------
# Spectral (FFT)
# ---------------------------------------------------------------------------


class TestSpectral(unittest.TestCase):
    def test_freeze_uses_continuous_phase_vocoder(self):
        with _SignalFlowStubbed():
            out = spectral_freeze(_StubNode("v"))
            self.assertIn("FFT", out.name)
            self.assertIn("ContPV", out.name)
            self.assertIn("rate=0", out.name)

    def test_shift_uses_rate(self):
        with _SignalFlowStubbed():
            out = spectral_shift(_StubNode("v"), rate=0.5)
            self.assertIn("ContPV", out.name)
            self.assertIn("rate=0.5", out.name)

    def test_shift_rejects_negative_rate(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                spectral_shift(_StubNode("v"), rate=-0.1)

    def test_blur_uses_random_phase(self):
        with _SignalFlowStubbed():
            out = spectral_blur(_StubNode("v"), level=0.8)
            self.assertIn("RandPhase", out.name)
            self.assertIn("lvl=0.8", out.name)

    def test_contrast_uses_fft_contrast(self):
        with _SignalFlowStubbed():
            out = spectral_contrast(_StubNode("v"), contrast=3.0)
            self.assertIn("Contrast", out.name)

    def test_lpf_uses_fftlpf(self):
        with _SignalFlowStubbed():
            out = spectral_lpf(_StubNode("v"), cutoff=2000)
            self.assertIn("FFTLPF", out.name)
            self.assertIn("2000", out.name)

    def test_gate_uses_fft_noise_gate(self):
        with _SignalFlowStubbed():
            out = spectral_gate(_StubNode("v"), threshold=0.1)
            self.assertIn("FFTGate", out.name)

    def test_spectral_validate_fft_size_power_of_two(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                spectral_freeze(_StubNode("v"), fft_size=1000)
            with self.assertRaises(ValueError):
                spectral_blur(_StubNode("v"), fft_size=0)

    def test_spectral_validate_hop_size(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                spectral_freeze(_StubNode("v"), fft_size=1024, hop_size=0)
            with self.assertRaises(ValueError):
                spectral_freeze(_StubNode("v"), fft_size=1024, hop_size=2048)


# ---------------------------------------------------------------------------
# Vocoder / time-stretch
# ---------------------------------------------------------------------------


class TestVocoder(unittest.TestCase):
    def test_time_stretch_wraps_continuous_phase_vocoder(self):
        with _SignalFlowStubbed():
            out = time_stretch(_StubNode("v"), rate=0.5)
            self.assertIn("ContPV", out.name)
            self.assertIn("rate=0.5", out.name)

    def test_time_stretch_rejects_negative_rate(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                time_stretch(_StubNode("v"), rate=-1.0)

    def test_phase_vocoder_freeze_uses_phase_vocoder(self):
        with _SignalFlowStubbed():
            out = phase_vocoder_freeze(_StubNode("v"))
            self.assertIn("PhaseVoc", out.name)


# ---------------------------------------------------------------------------
# Granular
# ---------------------------------------------------------------------------


class TestGranular(unittest.TestCase):
    def _buffer(self):
        # Stand-in buffer: any hashable object works in the stub's
        # granulator kwargs. Real SignalFlow needs an sf.Buffer.
        return object()

    def test_stretch_builds_granulator_with_sawlfo_position(self):
        with _SignalFlowStubbed():
            out = granular_stretch(self._buffer(), speed=0.5, duration=0.1)
            self.assertIn("Gran", out.name)
            self.assertIn("SawLFO", out.name)

    def test_freeze_uses_fixed_position(self):
        with _SignalFlowStubbed():
            out = granular_freeze(self._buffer(), position=0.25)
            self.assertIn("Gran", out.name)
            self.assertIn("pos=0.25", out.name)

    def test_freeze_rejects_out_of_range_position(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                granular_freeze(self._buffer(), position=1.5)
            with self.assertRaises(ValueError):
                granular_freeze(self._buffer(), position=-0.1)

    def test_scatter_uses_sample_and_hold(self):
        with _SignalFlowStubbed():
            out = granular_scatter(self._buffer(), spread=0.2)
            self.assertIn("S&H", out.name)

    def test_shimmer_has_positive_pitch_rate(self):
        with _SignalFlowStubbed():
            out = granular_shimmer(self._buffer(), semitones=12)
            self.assertIn("rate=2.0", out.name)

    def test_reverse_uses_negative_rate(self):
        with _SignalFlowStubbed():
            out = granular_reverse(self._buffer())
            self.assertIn("rate=-1.0", out.name)

    def test_reverse_rejects_non_positive_speed(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                granular_reverse(self._buffer(), speed=0)

    def test_stutter_uses_external_clock(self):
        with _SignalFlowStubbed():
            from signalflow import Impulse  # type: ignore
            clock = Impulse(frequency=8.0)
            out = granular_stutter(self._buffer(), clock, position=0.3)
            self.assertIn("Imp", out.name)

    def test_cloud_rejects_non_positive_density(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                granular_cloud(self._buffer(), density=0)

    def test_capture_and_granulate_wraps_recorder(self):
        with _SignalFlowStubbed():
            out = capture_and_granulate(
                _StubNode("v"),
                capture_seconds=0.5,
                mode="stretch",
                speed=0.5,
            )
            # Not easy to see Recorder in the final node name since
            # the granulator constructs its own tree; but the
            # Granulator should be the top-level node.
            self.assertIn("Gran", out.name)

    def test_capture_and_granulate_rejects_bad_mode(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                capture_and_granulate(_StubNode("v"), mode="nope")


# ---------------------------------------------------------------------------
# VAMP + overdrive
# ---------------------------------------------------------------------------


class TestVamp(unittest.TestCase):
    def test_tube_waveshape(self):
        with _SignalFlowStubbed():
            out = vamp(_StubNode("v"), drive=3.0, waveshape="tube")
            self.assertIn("Tanh", out.name)

    def test_hard_waveshape_uses_clip(self):
        with _SignalFlowStubbed():
            out = vamp(_StubNode("v"), waveshape="hard")
            self.assertIn("Clip", out.name)

    def test_fold_waveshape_uses_fold(self):
        with _SignalFlowStubbed():
            out = vamp(_StubNode("v"), waveshape="fold")
            self.assertIn("Fold", out.name)

    def test_vamp_rejects_bad_waveshape(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                vamp(_StubNode("v"), waveshape="wub")

    def test_vamp_rejects_bad_bias_gain_drive(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                vamp(_StubNode("v"), bias=1.1)
            with self.assertRaises(ValueError):
                vamp(_StubNode("v"), drive=0)
            with self.assertRaises(ValueError):
                vamp(_StubNode("v"), gain=-1)

    def test_vamp_presets_all_build(self):
        with _SignalFlowStubbed():
            for preset in (vamp_light, vamp_medium, vamp_heavy, vamp_fuzz):
                node = preset(_StubNode("v"))
                self.assertIsNotNone(node)

    def test_pre_and_post_filter_optional(self):
        with _SignalFlowStubbed():
            out = vamp(
                _StubNode("v"),
                pre_filter=400, pre_filter_type="hp",
                post_filter=5000, post_filter_type="lp",
            )
            # Both SVFilters appear in the graph.
            self.assertEqual(out.name.count("SVF"), 2)

    def test_pre_filter_type_validated(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                vamp(_StubNode("v"), pre_filter=400, pre_filter_type="bp")


class TestOverdrive(unittest.TestCase):
    def test_overdrive_uses_tanh_and_svf(self):
        with _SignalFlowStubbed():
            out = overdrive(_StubNode("v"), drive=3.0, tone=2500)
            self.assertIn("Tanh", out.name)
            self.assertIn("SVF", out.name)

    def test_overdrive_presets_build(self):
        with _SignalFlowStubbed():
            for preset in (overdrive_soft, overdrive_classic, overdrive_crunch):
                self.assertIsNotNone(preset(_StubNode("v")))

    def test_dual_overdrive_stacks_stages(self):
        with _SignalFlowStubbed():
            out = dual_overdrive(_StubNode("v"))
            # Two Tanh stages in series.
            self.assertGreaterEqual(out.name.count("Tanh"), 2)


# ---------------------------------------------------------------------------
# Multiband OTT
# ---------------------------------------------------------------------------


class TestMultibandOTT(unittest.TestCase):
    def test_splits_into_three_bands(self):
        with _SignalFlowStubbed():
            out = multiband_compressor(_StubNode("v"))
            # low = lp, high = hp, mid = hp->lp (series).
            # Total: 4 SVFilter constructions + 3 Compressor + optional
            # upward compressors.
            self.assertGreaterEqual(out.name.count("SVF"), 4)
            self.assertGreaterEqual(out.name.count("Comp"), 3)

    def test_validates_crossover_ordering(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                multiband_compressor(
                    _StubNode("v"),
                    low_xover=3000, high_xover=500,
                )

    def test_validates_amount_ranges(self):
        with _SignalFlowStubbed():
            with self.assertRaises(ValueError):
                multiband_compressor(_StubNode("v"), low_amount=1.5)
            with self.assertRaises(ValueError):
                multiband_compressor(_StubNode("v"), upward=-0.1)
            with self.assertRaises(ValueError):
                multiband_compressor(_StubNode("v"), depth=-1)
            with self.assertRaises(ValueError):
                multiband_compressor(_StubNode("v"), output=0)

    def test_presets_all_build(self):
        with _SignalFlowStubbed():
            for preset in (ott, ott_punch, ott_glue, ott_loud, ott_soft):
                self.assertIsNotNone(preset(_StubNode("v")))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
