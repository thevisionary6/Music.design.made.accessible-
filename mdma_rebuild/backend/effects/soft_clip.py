"""Tanh-based soft clipping, for :meth:`Pattern.dist`.

Reference implementation of the Phase 8 porting convention
(``fn(input_node, **params) -> Node``). Intentionally short so it
reads as a template rather than a DSP deep-dive.
"""

from __future__ import annotations


def soft_clip(input_node, amount: float = 1.0):
    """Saturate ``input_node`` through a tanh nonlinearity.

    ``amount`` scales the input before the tanh (higher = more
    clipping). ``amount=1.0`` is identity-like for quiet input and
    saturates progressively for loud input, matching the usual
    "drive" control. Negative or zero ``amount`` raises
    :class:`ValueError`.

    Typical use::

        from mdma_rebuild.backend.effects import soft_clip
        pattern.dist(soft_clip, amount=3.0)
    """
    if amount <= 0:
        raise ValueError(f"amount must be positive, got {amount}")
    import signalflow as sf
    return sf.Tanh(input_node * float(amount))
