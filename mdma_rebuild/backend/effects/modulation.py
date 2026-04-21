"""Modulation and stereo effects, for :meth:`Pattern.fx`.

Chorus, flanger, phaser, tremolo, autopan — the standard pedal-chain
kit. Each follows the Phase 8 porting convention::

    def fn(input_node, **params) -> Node:
        ...

All five use :class:`signalflow.SineLFO` as the modulator so the
depth/rate knobs behave the same across the set.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Chorus
# ---------------------------------------------------------------------------


def chorus(
    input_node,
    rate: float = 0.5,
    depth: float = 0.003,
    mix: float = 0.5,
    base_delay: float = 0.015,
):
    """Mono chorus via a short LFO-modulated delay.

    Parameters
    ----------
    rate : float
        LFO rate in Hz.
    depth : float
        LFO-driven delay-time deviation in seconds. Typical range
        ``0.001 .. 0.005``.
    mix : float
        Dry/wet in ``[0, 1]``.
    base_delay : float
        Centre delay time in seconds. Typical range ``0.010 .. 0.025``.
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    if depth < 0:
        raise ValueError(f"depth must be non-negative, got {depth}")
    if base_delay <= 0:
        raise ValueError(f"base_delay must be positive, got {base_delay}")
    mix = max(0.0, min(1.0, float(mix)))

    import signalflow as sf
    lfo = sf.SineLFO(
        frequency=float(rate),
        min=max(0.001, float(base_delay) - float(depth)),
        max=float(base_delay) + float(depth),
    )
    wet = sf.OneTapDelay(
        input_node,
        delay_time=lfo,
        max_delay_time=float(base_delay) + float(depth) + 0.01,
    )
    return input_node * (1.0 - mix) + wet * mix


# ---------------------------------------------------------------------------
# Flanger
# ---------------------------------------------------------------------------


def flanger(
    input_node,
    rate: float = 0.25,
    depth: float = 0.002,
    feedback: float = 0.5,
    mix: float = 0.5,
    base_delay: float = 0.005,
):
    """LFO-modulated short-delay flanger with feedback.

    Same topology as :func:`chorus` plus a feedback path that
    sharpens the comb notches and gives flanging its characteristic
    jet-whoosh sound. ``feedback`` clamped to ``0.95``.
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    if depth < 0:
        raise ValueError(f"depth must be non-negative, got {depth}")
    if feedback < 0:
        raise ValueError(f"feedback must be non-negative, got {feedback}")
    if base_delay <= 0:
        raise ValueError(f"base_delay must be positive, got {base_delay}")
    mix = max(0.0, min(1.0, float(mix)))
    fb = min(0.95, float(feedback))

    import signalflow as sf
    lfo = sf.SineLFO(
        frequency=float(rate),
        min=max(0.0005, float(base_delay) - float(depth)),
        max=float(base_delay) + float(depth),
    )
    # Two-tap feedback topology: the second delay realises the loop.
    first = sf.OneTapDelay(
        input_node,
        delay_time=lfo,
        max_delay_time=float(base_delay) + float(depth) + 0.01,
    )
    with_fb = input_node + first * fb
    wet = sf.OneTapDelay(
        with_fb,
        delay_time=lfo,
        max_delay_time=float(base_delay) + float(depth) + 0.01,
    )
    return input_node * (1.0 - mix) + wet * mix


# ---------------------------------------------------------------------------
# Phaser
# ---------------------------------------------------------------------------


def phaser(
    input_node,
    rate: float = 0.3,
    depth: float = 0.003,
    stages: int = 4,
    mix: float = 0.5,
    base_delay: float = 0.005,
):
    """N-stage allpass phaser with an LFO-swept sweet spot.

    ``stages`` controls how many AllpassDelay stages are chained.
    More stages -> more notches -> thicker phasing. Typical values
    are 2, 4, 6, 8. Must be a positive integer.
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    if depth < 0:
        raise ValueError(f"depth must be non-negative, got {depth}")
    if stages < 1:
        raise ValueError(f"stages must be >= 1, got {stages}")
    if base_delay <= 0:
        raise ValueError(f"base_delay must be positive, got {base_delay}")
    mix = max(0.0, min(1.0, float(mix)))

    import signalflow as sf
    lfo = sf.SineLFO(
        frequency=float(rate),
        min=max(0.0005, float(base_delay) - float(depth)),
        max=float(base_delay) + float(depth),
    )
    wet = input_node
    for _ in range(int(stages)):
        wet = sf.AllpassDelay(
            wet,
            delay_time=lfo,
            feedback=0.5,
            max_delay_time=float(base_delay) + float(depth) + 0.01,
        )
    return input_node * (1.0 - mix) + wet * mix


# ---------------------------------------------------------------------------
# Tremolo
# ---------------------------------------------------------------------------


def tremolo(
    input_node,
    rate: float = 5.0,
    depth: float = 0.5,
):
    """Amplitude modulation at ``rate`` Hz.

    ``depth`` in ``[0, 1]``: 0 is no modulation, 1 takes the signal
    all the way to silence at the trough. Uses a sine LFO that
    oscillates between ``1 - depth`` and ``1.0``, so ``depth=0``
    leaves the dry signal untouched.
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    depth = max(0.0, min(1.0, float(depth)))

    import signalflow as sf
    lfo = sf.SineLFO(frequency=float(rate), min=1.0 - depth, max=1.0)
    return input_node * lfo


# ---------------------------------------------------------------------------
# Autopan
# ---------------------------------------------------------------------------


def autopan(
    input_node,
    rate: float = 0.25,
    depth: float = 1.0,
):
    """Stereo auto-panner.

    ``depth`` in ``[0, 1]`` — 1.0 swings fully left to fully right,
    0.0 stays centre. Uses :class:`signalflow.StereoPanner` driven
    by a sine LFO.
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    depth = max(0.0, min(1.0, float(depth)))

    import signalflow as sf
    lfo = sf.SineLFO(frequency=float(rate), min=-depth, max=depth)
    return sf.StereoPanner(input_node, lfo)
