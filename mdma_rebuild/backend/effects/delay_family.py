"""Extended delay family.

Complements :mod:`.delay` (single-tap feedback delay) with the
standard pedal-chain variants:

- :func:`slapback` — single short repeat, no feedback, classic
  50s-guitar sound.
- :func:`ping_pong` — stereo-bouncing delay; signal alternates
  left/right on each repeat.
- :func:`multitap` — multiple parallel taps at different delay
  times, summed.
- :func:`tape_echo` — delay with internal saturation and a gentle
  high-cut on the feedback path, mimicking magnetic-tape wear.

Every callable follows the Phase 8 convention
``fn(input_node, **params) -> Node`` and imports SignalFlow lazily.
"""

from __future__ import annotations

from typing import Sequence


def slapback(input_node, time: float = 0.08, mix: float = 0.3):
    """Single-repeat delay. No feedback — classic rockabilly slapback."""
    if time <= 0:
        raise ValueError(f"time must be positive, got {time}")
    mix = max(0.0, min(1.0, float(mix)))
    import signalflow as sf
    wet = sf.OneTapDelay(input_node, delay_time=float(time), max_delay_time=max(0.5, float(time) * 2))
    return input_node * (1.0 - mix) + wet * mix


def ping_pong(
    input_node,
    time: float = 0.25,
    feedback: float = 0.5,
    mix: float = 0.4,
):
    """Stereo ping-pong delay.

    Two delay lines offset by ``time`` bounce the signal between left
    and right via :class:`signalflow.StereoPanner`. ``feedback``
    clamped to ``0.95`` to keep the loop stable.
    """
    if time <= 0:
        raise ValueError(f"time must be positive, got {time}")
    if feedback < 0:
        raise ValueError(f"feedback must be non-negative, got {feedback}")
    fb = min(0.95, float(feedback))
    mix = max(0.0, min(1.0, float(mix)))

    import signalflow as sf
    # First tap goes left, second tap goes right; feed each back
    # into the other for alternating bounce.
    tap_l = sf.OneTapDelay(
        input_node,
        delay_time=float(time),
        max_delay_time=float(time) * 2 + 0.1,
    )
    tap_r = sf.OneTapDelay(
        input_node + tap_l * fb,
        delay_time=float(time),
        max_delay_time=float(time) * 2 + 0.1,
    )
    left = sf.StereoPanner(tap_l, -1.0)
    right = sf.StereoPanner(tap_r, 1.0)
    wet = left + right
    return input_node * (1.0 - mix) + wet * mix


def multitap(
    input_node,
    times: Sequence[float] = (0.125, 0.25, 0.375),
    gains: Sequence[float] | None = None,
    mix: float = 0.4,
):
    """Multi-tap delay. Each ``time`` gets its own tap summed into
    the wet output; per-tap ``gains`` default to a decaying series.

    Taps are parallel (no cross-feedback), so the effect is a fixed
    rhythmic pattern rather than a decay tail. Pair with
    :func:`delay` for a combined rhythmic + sustained echo.
    """
    if not times:
        raise ValueError("multitap requires at least one tap time")
    for t in times:
        if t <= 0:
            raise ValueError(f"tap times must be positive, got {t}")
    if gains is None:
        # Default: -3 dB per tap.
        gains = [0.707 ** (i + 1) for i in range(len(times))]
    if len(gains) != len(times):
        raise ValueError(
            f"gains length ({len(gains)}) must match times length ({len(times)})"
        )
    mix = max(0.0, min(1.0, float(mix)))
    max_time = max(float(t) for t in times)

    import signalflow as sf
    wet = None
    for t, g in zip(times, gains):
        tap = sf.OneTapDelay(
            input_node,
            delay_time=float(t),
            max_delay_time=max_time + 0.1,
        ) * float(g)
        wet = tap if wet is None else wet + tap
    return input_node * (1.0 - mix) + wet * mix


def tape_echo(
    input_node,
    time: float = 0.3,
    feedback: float = 0.5,
    mix: float = 0.4,
    saturation: float = 1.2,
    hi_cut_hz: float = 4000.0,
):
    """Feedback delay with tanh saturation + high-cut on the feedback
    path. Simulates the warm, degrading repeats of a tape echo unit.

    ``saturation`` feeds the feedback signal into ``sf.Tanh`` with a
    pre-gain; higher values produce more colour. ``hi_cut_hz`` is the
    cutoff of a lowpass in the feedback path so each repeat loses
    high-frequency content, mimicking tape wear.
    """
    if time <= 0:
        raise ValueError(f"time must be positive, got {time}")
    if feedback < 0:
        raise ValueError(f"feedback must be non-negative, got {feedback}")
    if saturation <= 0:
        raise ValueError(f"saturation must be positive, got {saturation}")
    if hi_cut_hz <= 0:
        raise ValueError(f"hi_cut_hz must be positive, got {hi_cut_hz}")
    fb = min(0.95, float(feedback))
    mix = max(0.0, min(1.0, float(mix)))

    import signalflow as sf
    first = sf.OneTapDelay(
        input_node,
        delay_time=float(time),
        max_delay_time=float(time) * 2 + 0.1,
    )
    # Saturate + high-cut on the feedback path.
    saturated = sf.Tanh(first * float(saturation))
    filtered = sf.SVFilter(
        saturated, "low_pass", float(hi_cut_hz), 0.0
    )
    with_fb = input_node + filtered * fb
    wet = sf.OneTapDelay(
        with_fb,
        delay_time=float(time),
        max_delay_time=float(time) * 2 + 0.1,
    )
    return input_node * (1.0 - mix) + wet * mix
