"""Extended filter variants, for :meth:`Pattern.fx` / :meth:`Pattern.spec`.

``Pattern.bf`` is deliberately spec-locked to the four transformer
filters (lpf, hpf, bpf, notch). Everything richer lives here as a
DSP callable that follows the Phase 8 porting convention::

    def my_filter(input_node, **params) -> Node:
        ...

Coverage:

- :func:`peak` / :func:`low_shelf` / :func:`high_shelf` — EQ filters
  (``signalflow.BiquadFilter``).
- :func:`moog` — Moog-style ladder VCF with built-in saturation.
- :func:`allpass_filter` — phase-rotator; the non-transformer
  sibling of :meth:`Pattern.bf`.
- :func:`comb_filter` — feedback comb, thin metallic colouring.

Cutoffs are in Hz. Resonance is in SignalFlow's ``[0, 1]`` range —
larger values (up to ``0.95``) are clamped to keep the filter
stable. Peak / shelf gain is in dB.
"""

from __future__ import annotations


def _clamp_resonance(r: float) -> float:
    return max(0.0, min(0.95, float(r)))


def peak(
    input_node,
    cutoff: float = 1000.0,
    resonance: float = 0.0,
    gain_db: float = 6.0,
):
    """Parametric EQ peak filter.

    ``gain_db > 0`` boosts the band; ``gain_db < 0`` cuts it.
    """
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    import signalflow as sf
    return sf.BiquadFilter(
        input_node,
        "peak",
        float(cutoff),
        _clamp_resonance(resonance),
        float(gain_db),
    )


def low_shelf(
    input_node,
    cutoff: float = 200.0,
    gain_db: float = 6.0,
):
    """Low-shelf EQ — gently boosts or cuts frequencies below ``cutoff``."""
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    import signalflow as sf
    return sf.BiquadFilter(
        input_node,
        "low_shelf",
        float(cutoff),
        0.0,
        float(gain_db),
    )


def high_shelf(
    input_node,
    cutoff: float = 6000.0,
    gain_db: float = 6.0,
):
    """High-shelf EQ — gently boosts or cuts frequencies above ``cutoff``."""
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    import signalflow as sf
    return sf.BiquadFilter(
        input_node,
        "high_shelf",
        float(cutoff),
        0.0,
        float(gain_db),
    )


def moog(
    input_node,
    cutoff: float = 1000.0,
    resonance: float = 0.3,
):
    """Moog-ladder VCF with built-in saturation.

    ``resonance`` near ``1.0`` self-oscillates; clamped to ``0.95``
    to stay stable.
    """
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    import signalflow as sf
    return sf.MoogVCF(input_node, float(cutoff), _clamp_resonance(resonance))


def allpass_filter(
    input_node,
    delay_time: float = 0.005,
    feedback: float = 0.5,
    max_delay_time: float = 0.05,
):
    """Single-stage allpass filter. Rotates phase without changing the
    magnitude response. Useful for phaser-style effects when chained.
    """
    if delay_time <= 0:
        raise ValueError(f"delay_time must be positive, got {delay_time}")
    if feedback < 0:
        raise ValueError(f"feedback must be non-negative, got {feedback}")
    import signalflow as sf
    return sf.AllpassDelay(
        input_node,
        delay_time=float(delay_time),
        feedback=min(0.95, float(feedback)),
        max_delay_time=float(max_delay_time),
    )


def comb_filter(
    input_node,
    delay_time: float = 0.01,
    feedback: float = 0.5,
    max_delay_time: float = 0.1,
):
    """Feedback comb filter. Thin, metallic, resonant at
    ``1 / delay_time`` and its harmonics.
    """
    if delay_time <= 0:
        raise ValueError(f"delay_time must be positive, got {delay_time}")
    if feedback < 0:
        raise ValueError(f"feedback must be non-negative, got {feedback}")
    import signalflow as sf
    return sf.CombDelay(
        input_node,
        delay_time=float(delay_time),
        feedback=min(0.95, float(feedback)),
        max_delay_time=float(max_delay_time),
    )
