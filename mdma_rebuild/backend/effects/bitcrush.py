"""Bit-depth and sample-rate reduction, for :meth:`Pattern.dist`.

Wraps :class:`signalflow.Resample`, which combines sample-rate
reduction (producing aliasing artefacts and the classic "lo-fi"
character) with bit-depth reduction (producing quantisation noise).
"""

from __future__ import annotations


def bitcrush(
    input_node,
    bit_depth: int = 8,
    sample_rate: float = 8000.0,
):
    """Reduce ``input_node`` to ``bit_depth`` bits at ``sample_rate`` Hz.

    Parameters
    ----------
    input_node
        SignalFlow node producing the dry signal.
    bit_depth : int
        Target bit depth in ``[1, 24]``. Lower values produce heavier
        quantisation noise. ``bit_depth=16`` is roughly transparent;
        ``8`` is classic 8-bit crunch; ``4`` is aggressive.
    sample_rate : float
        Target sample rate in Hz. Must be positive. Typical musical
        values: 22050 (mild high-frequency loss), 8000 (telephone),
        4000 (harsh aliasing).

    Typical use::

        pattern.dist(bitcrush, bit_depth=6, sample_rate=11025)
    """
    if bit_depth < 1 or bit_depth > 24:
        raise ValueError(
            f"bit_depth must be in [1, 24], got {bit_depth}"
        )
    if sample_rate <= 0:
        raise ValueError(
            f"sample_rate must be positive, got {sample_rate}"
        )

    import signalflow as sf
    return sf.Resample(
        input_node,
        sample_rate=float(sample_rate),
        bit_rate=int(bit_depth),
    )
