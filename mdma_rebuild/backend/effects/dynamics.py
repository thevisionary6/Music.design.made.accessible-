"""Dynamics processing, for :meth:`Pattern.fx`.

Wraps :class:`signalflow.Compressor`. ``threshold`` is linear
amplitude (0..1, not dB) to match the SignalFlow API; helper
:func:`db_to_amplitude` is exposed for callers that want to think in
dB.
"""

from __future__ import annotations

import math


def db_to_amplitude(db: float) -> float:
    """Convert a value in dB to linear amplitude.

    ``db=0`` returns ``1.0`` (unity), ``db=-6`` returns ``~0.501``,
    ``db=-20`` returns ``0.1``.
    """
    return 10.0 ** (float(db) / 20.0)


def compressor(
    input_node,
    threshold: float = 0.3,
    ratio: float = 4.0,
    attack: float = 0.005,
    release: float = 0.100,
    makeup: float = 1.0,
):
    """Feed-forward compressor with optional makeup gain.

    Parameters
    ----------
    threshold : float
        Linear amplitude threshold in ``(0, 1]``. Signal exceeding
        this level is attenuated by ``ratio``. Use
        :func:`db_to_amplitude` to convert from dB.
    ratio : float
        Compression ratio. ``1.0`` is no compression, ``4.0`` is the
        usual "compressor" setting, ``20+`` approximates a limiter.
        Must be >= 1.
    attack : float
        Attack time in seconds.
    release : float
        Release time in seconds.
    makeup : float
        Linear gain applied after compression so the output isn't
        quieter than the input. Default ``1.0`` is no makeup.
    """
    if threshold <= 0 or threshold > 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if ratio < 1:
        raise ValueError(f"ratio must be >= 1, got {ratio}")
    if attack <= 0:
        raise ValueError(f"attack must be positive, got {attack}")
    if release <= 0:
        raise ValueError(f"release must be positive, got {release}")
    if makeup <= 0:
        raise ValueError(f"makeup must be positive, got {makeup}")

    import signalflow as sf
    comp = sf.Compressor(
        input_node,
        threshold=float(threshold),
        ratio=float(ratio),
        attack_time=float(attack),
        release_time=float(release),
    )
    if makeup == 1.0:
        return comp
    return comp * float(makeup)
