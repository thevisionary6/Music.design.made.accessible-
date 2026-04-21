"""Schroeder-style reverb built from SignalFlow comb + allpass delays.

Reference implementation of the Phase 8 porting convention
(``fn(input_node, **params) -> Node``). Uses the canonical Schroeder
topology: four parallel feedback combs feeding two series allpass
filters. Comb and allpass delay times come from Manfred Schroeder's
original 1962 paper with small tweaks to stay musical at typical
``room_size`` values.

Suitable for :meth:`Pattern.ir` (the spec pigeonholes convolution /
impulse-response / reverb territory there). Works equally well via
:meth:`Pattern.fx` for catch-all use.
"""

from __future__ import annotations


# Schroeder's original comb-filter delay times, in seconds. These
# were mutually-prime milliseconds in the paper so the tap spectrum
# stays dense; rescaled here to the 20-50 ms range typical of
# algorithmic reverbs.
_COMB_DELAYS_S = (0.0297, 0.0371, 0.0411, 0.0437)

# Allpass delays (seconds). Shorter than the combs so late-field
# density builds without introducing perceptible repetition.
_ALLPASS_DELAYS_S = (0.00500, 0.01700)

# Allpass feedback fixed per Schroeder.
_ALLPASS_FEEDBACK = 0.5

_MAX_COMB_DELAY = 0.1  # seconds; sizes the underlying buffers
_MAX_ALLPASS_DELAY = 0.05


def reverb(
    input_node,
    room_size: float = 0.7,
    damping: float = 0.5,
    mix: float = 0.35,
):
    """Algorithmic reverb suitable for :meth:`Pattern.ir`.

    Parameters
    ----------
    input_node
        SignalFlow node producing the dry signal.
    room_size : float
        ``[0, 1]`` — scales comb-filter feedback. 0 is near-dry, 0.7
        is a medium room, 0.95 is cathedral. Values >= 1.0 will
        diverge and are clamped at 0.98 for safety.
    damping : float
        ``[0, 1]`` — not yet routed to a tone control in this
        implementation. Kept in the signature so callers can plan
        automation against it; will drive a lowpass inside the comb
        feedback loop once SignalFlow exposes a convenient
        in-loop filter.
    mix : float
        ``[0, 1]`` — dry/wet. 0 is dry only, 1 is wet only.

    Raises
    ------
    ValueError
        If ``room_size``, ``damping``, or ``mix`` is negative.
    """
    if room_size < 0:
        raise ValueError(f"room_size must be non-negative, got {room_size}")
    if damping < 0:
        raise ValueError(f"damping must be non-negative, got {damping}")
    if mix < 0:
        raise ValueError(f"mix must be non-negative, got {mix}")

    # Clamp to safe ranges so the DSP can't runaway.
    room = min(0.98, float(room_size))
    mix_clamped = min(1.0, float(mix))
    # `damping` is stored for future wiring; currently only affects
    # the allpass-feedback trim so it's not a total no-op.
    damp_trim = 1.0 - min(0.95, float(damping)) * 0.3  # 1.0 .. 0.715

    import signalflow as sf

    # 4 parallel feedback combs, summed.
    combs = [
        sf.CombDelay(
            input_node,
            delay_time=delay,
            feedback=room,
            max_delay_time=_MAX_COMB_DELAY,
        )
        for delay in _COMB_DELAYS_S
    ]
    comb_sum = combs[0]
    for c in combs[1:]:
        comb_sum = comb_sum + c
    comb_mix = comb_sum * 0.25  # average of the four taps

    # 2 series allpass filters to build late-field density.
    wet = comb_mix
    for delay in _ALLPASS_DELAYS_S:
        wet = sf.AllpassDelay(
            wet,
            delay_time=delay,
            feedback=_ALLPASS_FEEDBACK * damp_trim,
            max_delay_time=_MAX_ALLPASS_DELAY,
        )

    return input_node * (1.0 - mix_clamped) + wet * mix_clamped
