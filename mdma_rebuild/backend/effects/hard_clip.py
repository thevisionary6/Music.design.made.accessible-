"""Hard clipping, for :meth:`Pattern.dist`.

Complements :mod:`.soft_clip` with a harsh, symmetric brick-wall
limiter. Useful for aggressive distortion and as a crude limiter
when combined with a pre-gain.
"""

from __future__ import annotations


def hard_clip(input_node, drive: float = 1.0, ceiling: float = 1.0):
    """Clip ``input_node`` to ``[-ceiling, ceiling]`` after scaling by ``drive``.

    Parameters
    ----------
    input_node
        SignalFlow node producing the dry signal.
    drive : float
        Pre-clip gain. Higher values push the signal further into the
        clipper for a more saturated tone. Must be positive.
    ceiling : float
        Absolute clipping threshold. Typical values are 0.1 (heavy
        clipping even on quiet signals) up to 1.0 (traditional brick
        wall at full scale). Must be positive.

    Typical use::

        pattern.dist(hard_clip, drive=4.0, ceiling=0.8)
    """
    if drive <= 0:
        raise ValueError(f"drive must be positive, got {drive}")
    if ceiling <= 0:
        raise ValueError(f"ceiling must be positive, got {ceiling}")

    import signalflow as sf
    driven = input_node * float(drive)
    # sf.Clip(node, min, max) clamps the signal; it's the cleanest
    # way to get symmetric hard-clipping out of SignalFlow.
    return sf.Clip(driven, -float(ceiling), float(ceiling))
