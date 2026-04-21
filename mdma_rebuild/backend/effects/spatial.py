"""Spatial / stereo effects.

- :func:`haas` — Haas-effect stereo widener: ~15 ms delay on one
  channel gives the illusion of width without phase cancellation
  at mono summation.
- :func:`stereo_widen` — scale the stereo-side component via
  :class:`signalflow.StereoWidth`.
- :func:`mono` — force mono by zeroing the stereo width.
- :func:`balance` — static left/right balance.
"""

from __future__ import annotations


def haas(input_node, delay_ms: float = 15.0, side: str = "right"):
    """Haas-effect widener.

    One channel plays dry, the other plays through a short delay;
    the ear interprets the delay as width. ``delay_ms`` in
    ``(0, 40]`` — past 40 ms the brain starts hearing a distinct
    echo instead of a widened image.
    """
    if delay_ms <= 0 or delay_ms > 40:
        raise ValueError(f"delay_ms must be in (0, 40], got {delay_ms}")
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    import signalflow as sf
    delayed = sf.OneTapDelay(
        input_node,
        delay_time=float(delay_ms) / 1000.0,
        max_delay_time=0.05,
    )
    if side == "right":
        dry = sf.StereoPanner(input_node, -1.0)
        wet = sf.StereoPanner(delayed, 1.0)
    else:
        dry = sf.StereoPanner(input_node, 1.0)
        wet = sf.StereoPanner(delayed, -1.0)
    return dry + wet


def stereo_widen(input_node, width: float = 1.5):
    """Widen a stereo signal's side component.

    ``width=1.0`` is neutral, ``<1`` narrows, ``>1`` widens (``0``
    collapses to mono, ``2`` aggressively wide). Clamped to
    ``[0, 4]`` to avoid degenerate phase artefacts.
    """
    if width < 0:
        raise ValueError(f"width must be non-negative, got {width}")
    w = min(4.0, float(width))
    import signalflow as sf
    return sf.StereoWidth(input_node, w)


def mono(input_node):
    """Collapse to mono via :func:`stereo_widen` with width 0."""
    import signalflow as sf
    return sf.StereoWidth(input_node, 0.0)


def balance(input_node, bias: float = 0.0):
    """Static left/right balance. ``bias`` in ``[-1, 1]``: -1 fully
    left, 0 centre, 1 fully right."""
    b = max(-1.0, min(1.0, float(bias)))
    import signalflow as sf
    return sf.StereoBalance(input_node, b)
