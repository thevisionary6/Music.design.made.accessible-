"""Foldback distortion, for :meth:`Pattern.dist`.

Where :mod:`.hard_clip` flattens peaks above the ceiling, foldback
reflects them back into the legal range. Produces a rich, detuned
aliasing character that's distinct from both soft-clip saturation
and hard-clip harshness. Uses :class:`signalflow.Fold` under the
hood.
"""

from __future__ import annotations


def foldback(input_node, drive: float = 1.5, threshold: float = 0.7):
    """Fold ``input_node`` back into ``[-threshold, threshold]``.

    Parameters
    ----------
    input_node
        SignalFlow node producing the dry signal.
    drive : float
        Pre-fold gain. Higher values drive more folds through the
        threshold and produce more dramatic distortion. Must be
        positive.
    threshold : float
        Absolute value at which folding occurs. Signal beyond
        ``threshold`` reflects back toward zero; repeats as needed for
        sample values well beyond ``drive * threshold``. Must be
        positive and in ``(0, 1]``.

    Typical use::

        pattern.dist(foldback, drive=3.0, threshold=0.5)
    """
    if drive <= 0:
        raise ValueError(f"drive must be positive, got {drive}")
    if threshold <= 0 or threshold > 1:
        raise ValueError(
            f"threshold must be in (0, 1], got {threshold}"
        )

    import signalflow as sf
    driven = input_node * float(drive)
    return sf.Fold(driven, -float(threshold), float(threshold))
