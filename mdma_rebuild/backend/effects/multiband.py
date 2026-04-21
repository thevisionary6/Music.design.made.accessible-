"""OTT-style multiband compressor.

Multi-band feedback compression is not shipped as a single node in
the current SignalFlow release (checked against 0.5.x) — only the
single-band :class:`signalflow.Compressor` is available. This module
builds the compressor manually: the input is split into three
frequency bands via SVFilter, each band runs through a downward
compressor and an upward-compression simulation (pre-gain boost
into the compressor's threshold), and the bands are recombined.

The result matches the "Forever Compression" / OTT preset from the
legacy ``dsp/effects.py``: glued lows, controlled mids, articulate
highs, with the characteristic upward-and-downward pumping that
makes OTT useful on sparse sources.

Shipped:

- :func:`multiband_compressor` — the parameterised 3-band OTT
  compressor.
- :func:`ott_punch`, :func:`ott_glue`, :func:`ott_loud`,
  :func:`ott_soft`, :func:`ott` — preset wrappers matching the
  legacy ``_fc_punch``, ``_fc_glue``, etc. voicings.

All callables honour the Phase 8 convention
``fn(input_node, **params) -> Node``.
"""

from __future__ import annotations


def _split_bands(input_node, low_xover: float, high_xover: float):
    """Split ``input_node`` into (low, mid, high) via SVFilter triples.

    Linkwitz-Riley would give flat summation, but SignalFlow's
    SVFilter is a single-pole state-variable filter; running the
    filters in series (low + high pass for the mid band) is the
    idiomatic SignalFlow pattern and close enough for OTT-style
    loudness work.
    """
    import signalflow as sf
    low = sf.SVFilter(input_node, "low_pass", float(low_xover), 0.0)
    high = sf.SVFilter(input_node, "high_pass", float(high_xover), 0.0)
    # Mid band: bandpass built from series hp + lp so the corner
    # frequencies match the low/high bands exactly.
    mid_hp = sf.SVFilter(input_node, "high_pass", float(low_xover), 0.0)
    mid = sf.SVFilter(mid_hp, "low_pass", float(high_xover), 0.0)
    return low, mid, high


def _ott_band(
    band_node,
    down_thresh: float,
    down_ratio: float,
    up_boost: float,
    attack: float,
    release: float,
    makeup: float,
):
    """One OTT band: downward compression + upward-compression shim.

    Downward: vanilla SignalFlow Compressor at (``down_thresh``,
    ``down_ratio``).

    Upward: boosted input routed through a second compressor at a
    higher threshold so quiet material is amplified toward the
    reference level. The ``up_boost`` multiplier sets how much
    quiet material is lifted before the second compressor clamps
    it.
    """
    import signalflow as sf
    downward = sf.Compressor(
        band_node,
        threshold=float(down_thresh),
        ratio=float(down_ratio),
        attack_time=float(attack),
        release_time=float(release),
    )
    if up_boost > 1.0:
        # Boost the signal then compress again so quiet portions get
        # lifted to the upward-threshold ceiling. Mathematically this
        # is a coarse upward compressor: loud content is already
        # ceiling'd by the downward stage, so the boost only lifts
        # the quiet passages.
        boosted = downward * float(up_boost)
        upward = sf.Compressor(
            boosted,
            threshold=float(down_thresh),
            ratio=float(max(1.5, down_ratio)),
            attack_time=float(attack),
            release_time=float(release),
        )
        result = upward
    else:
        result = downward
    if makeup != 1.0:
        result = result * float(makeup)
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def multiband_compressor(
    input_node,
    low_xover: float = 120.0,
    high_xover: float = 2500.0,
    low_amount: float = 0.5,
    mid_amount: float = 0.5,
    high_amount: float = 0.5,
    upward: float = 0.5,
    downward: float = 0.5,
    depth: float = 0.5,
    attack: float = 0.010,
    release: float = 0.100,
    mix: float = 1.0,
    output: float = 1.0,
):
    """3-band OTT-style compressor.

    Parameters
    ----------
    input_node
        SignalFlow node producing the dry signal.
    low_xover, high_xover : float
        Crossover frequencies (Hz) for the band splits. Low band is
        everything below ``low_xover``; high band is everything
        above ``high_xover``; mid band is the material between.
    low_amount, mid_amount, high_amount : float
        Per-band compression intensity in ``[0, 1]``. Higher values
        drive harder into the downward threshold.
    upward, downward : float
        Global scales (``[0, 1]``) for how much upward and downward
        compression is applied. ``downward=1.0`` and ``upward=0.0``
        is a traditional downward-only multiband; the OTT character
        comes from driving both above 0.5.
    depth : float
        Overall depth multiplier in ``[0, 2]``. Scales all per-band
        amounts together. 1.0 is default, 2.0 is very aggressive.
    attack, release : float
        Per-band envelope times in seconds.
    mix : float
        Dry/wet in ``[0, 1]``.
    output : float
        Final output gain multiplier (positive).
    """
    for name, value in (
        ("low_xover", low_xover), ("high_xover", high_xover),
        ("attack", attack), ("release", release), ("output", output),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if low_xover >= high_xover:
        raise ValueError(
            f"low_xover ({low_xover}) must be less than high_xover ({high_xover})"
        )
    for name, value in (
        ("low_amount", low_amount), ("mid_amount", mid_amount),
        ("high_amount", high_amount),
        ("upward", upward), ("downward", downward),
        ("mix", mix),
    ):
        if value < 0 or value > 1:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if depth < 0:
        raise ValueError(f"depth must be non-negative, got {depth}")

    # Per-band thresholds + ratios come from combining the
    # per-band amount with the global depth and downward scale.
    depth_eff = max(0.05, min(2.0, float(depth)))
    dwn = max(0.0, min(1.0, float(downward)))
    up = max(0.0, min(1.0, float(upward)))

    def band_params(amount: float):
        amt = max(0.0, min(1.0, float(amount))) * depth_eff
        # Downward: threshold falls as amount rises; ratio grows.
        down_thresh = max(0.05, 0.6 - 0.4 * amt) if dwn > 0 else 1.0
        down_ratio = 1.0 + 9.0 * amt * dwn   # 1..~10 at full tilt
        # Upward boost scales with upward global.
        up_boost = 1.0 + 3.0 * amt * up       # 1..~4 at full tilt
        # Per-band makeup so quieter bands don't vanish after dynamics.
        makeup = 1.0 + 0.4 * amt
        return {
            "down_thresh": down_thresh,
            "down_ratio": down_ratio,
            "up_boost": up_boost,
            "makeup": makeup,
        }

    low, mid, high = _split_bands(input_node, low_xover, high_xover)
    low_comp = _ott_band(
        low, attack=attack, release=release, **band_params(low_amount),
    )
    mid_comp = _ott_band(
        mid, attack=attack, release=release, **band_params(mid_amount),
    )
    high_comp = _ott_band(
        high, attack=attack, release=release, **band_params(high_amount),
    )

    wet = (low_comp + mid_comp + high_comp) * float(output)

    mix = max(0.0, min(1.0, float(mix)))
    if mix >= 1.0:
        return wet
    if mix <= 0.0:
        return input_node
    return input_node * (1.0 - mix) + wet * mix


# ---------------------------------------------------------------------------
# Preset wrappers — map to the old _fc_* voicings.
# ---------------------------------------------------------------------------


def ott_punch(input_node):
    """Punchy preset: short attack, moderate depth, balanced bands."""
    return multiband_compressor(
        input_node,
        low_amount=0.7, mid_amount=0.6, high_amount=0.6,
        upward=0.55, downward=0.7, depth=0.8,
        attack=0.003, release=0.080,
    )


def ott_glue(input_node):
    """Glue preset: gentle multiband with slower times."""
    return multiband_compressor(
        input_node,
        low_amount=0.4, mid_amount=0.35, high_amount=0.3,
        upward=0.35, downward=0.55, depth=0.5,
        attack=0.020, release=0.180,
    )


def ott_loud(input_node):
    """Loud preset: aggressive maximisation."""
    return multiband_compressor(
        input_node,
        low_amount=0.85, mid_amount=0.8, high_amount=0.8,
        upward=0.7, downward=0.85, depth=1.2,
        attack=0.002, release=0.060, output=1.3,
    )


def ott_soft(input_node):
    """Soft preset: subtle dynamics control."""
    return multiband_compressor(
        input_node,
        low_amount=0.25, mid_amount=0.2, high_amount=0.2,
        upward=0.2, downward=0.35, depth=0.35,
        attack=0.030, release=0.250,
    )


def ott(input_node):
    """The classic OTT preset: heavy upward + downward on all bands."""
    return multiband_compressor(
        input_node,
        low_amount=0.8, mid_amount=0.8, high_amount=0.8,
        upward=0.85, downward=0.85, depth=1.0,
        attack=0.003, release=0.100,
    )
