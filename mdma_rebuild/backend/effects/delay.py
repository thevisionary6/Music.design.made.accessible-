"""Feedback delay, for :meth:`Pattern.fx`.

Reference implementation of the Phase 8 porting convention. Uses
:class:`signalflow.OneTapDelay` with an explicit feedback path so
``feedback`` behaves like the usual delay pedal knob (0 = single
repeat, approaching 1 = near-infinite repeats).
"""

from __future__ import annotations


def delay(
    input_node,
    time: float = 0.25,
    feedback: float = 0.4,
    mix: float = 0.5,
    max_time: float = 2.0,
):
    """Single-tap feedback delay.

    Parameters
    ----------
    input_node
        SignalFlow node producing the dry signal.
    time : float
        Delay time in seconds.
    feedback : float
        Feedback coefficient in ``[0, 0.95]`` — clamped at ``0.95`` so
        the DSP can't runaway. Negative values raise
        :class:`ValueError`.
    mix : float
        Dry/wet mix in ``[0, 1]``. ``0`` is dry only, ``1`` is wet
        only. Values outside the range are clamped.
    max_time : float
        Upper bound on ``time`` for the underlying delay buffer; does
        not clamp ``time`` itself but caps the allocation.

    Typical use::

        from mdma_rebuild.backend.effects import delay
        pattern.fx(delay, time=0.375, feedback=0.6, mix=0.3)
    """
    if time <= 0:
        raise ValueError(f"time must be positive, got {time}")
    if feedback < 0:
        raise ValueError(f"feedback must be non-negative, got {feedback}")
    feedback = min(0.95, float(feedback))
    mix = max(0.0, min(1.0, float(mix)))

    import signalflow as sf
    delayed = sf.OneTapDelay(
        input_node, delay_time=float(time), max_delay_time=float(max_time)
    )
    with_feedback = input_node + delayed * feedback
    # Re-route through a second tap to realise the feedback loop.
    wet = sf.OneTapDelay(
        with_feedback, delay_time=float(time), max_delay_time=float(max_time)
    )
    return input_node * (1.0 - mix) + wet * mix
