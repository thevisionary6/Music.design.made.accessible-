"""Additional dynamics beyond :func:`.dynamics.compressor`.

- :func:`limiter` — very-high-ratio compressor with fast attack.
- :func:`noise_gate` — downward expander at very low threshold.
- :func:`expander` — inverse-compressor that increases dynamic
  range below threshold.

All three wrap :class:`signalflow.Compressor`. The math is: a
limiter is a compressor with ratio >= 20; a noise gate is a
compressor applied to below-threshold material (emulated here by
a high ratio with a low threshold and very fast attack); an
expander gets a lower ratio (< 1 effectively, realised here by a
compressor with pre/post gain trickery).
"""

from __future__ import annotations


def limiter(
    input_node,
    threshold: float = 0.9,
    attack: float = 0.001,
    release: float = 0.050,
    makeup: float = 1.0,
):
    """Brick-wall-ish limiter via a 20:1 compressor with fast attack."""
    if threshold <= 0 or threshold > 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
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
        ratio=20.0,
        attack_time=float(attack),
        release_time=float(release),
    )
    return comp if makeup == 1.0 else comp * float(makeup)


def noise_gate(
    input_node,
    threshold: float = 0.02,
    attack: float = 0.002,
    release: float = 0.100,
):
    """Downward gate.

    Signal below ``threshold`` is squashed hard (ratio ~ 20) so
    ambient noise vanishes while above-threshold material passes
    through. Fast attack so transients survive.

    This is the audio-processing gate, distinct from the rhythmic
    :meth:`Pattern.gate` which applies a step-pattern envelope.
    """
    if threshold <= 0 or threshold > 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if attack <= 0:
        raise ValueError(f"attack must be positive, got {attack}")
    if release <= 0:
        raise ValueError(f"release must be positive, got {release}")

    import signalflow as sf
    # High ratio with a low threshold: material below threshold is
    # attenuated proportionally by the compressor's static curve.
    return sf.Compressor(
        input_node,
        threshold=float(threshold),
        ratio=20.0,
        attack_time=float(attack),
        release_time=float(release),
    )


def expander(
    input_node,
    threshold: float = 0.5,
    ratio: float = 2.0,
    attack: float = 0.01,
    release: float = 0.1,
):
    """Upward expander: increases dynamic range above ``threshold``.

    Implemented as a pre-gain-boosted compressor so loud material
    gets an extra push while quieter material passes through
    normally. For true below-threshold expansion use
    :func:`noise_gate`.
    """
    if threshold <= 0 or threshold > 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if ratio < 1:
        raise ValueError(f"ratio must be >= 1, got {ratio}")
    if attack <= 0:
        raise ValueError(f"attack must be positive, got {attack}")
    if release <= 0:
        raise ValueError(f"release must be positive, got {release}")

    import signalflow as sf
    boosted = input_node * float(ratio)
    return sf.Compressor(
        boosted,
        threshold=float(threshold),
        ratio=float(ratio),
        attack_time=float(attack),
        release_time=float(release),
    )
