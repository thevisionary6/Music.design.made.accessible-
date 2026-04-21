"""VAMP amp/overdrive stack + standalone overdrive variants.

Ports the old ``dsp/effects.py:vamp_process`` and its overdrive
presets to SignalFlow-node form. Keeps the signal path faithful
enough that the VAMP presets (``light`` / ``medium`` / ``heavy`` /
``fuzz``) and the overdrive presets (``soft`` / ``classic`` /
``crunch``) sound recognisably like their numpy originals:

    pre-filter (optional) -> drive -> bias -> waveshape -> post-filter -> gain

For SignalFlow the waveshape step is one of ``sf.Tanh`` (tube),
``sf.Clip`` (hard), ``sf.Fold`` (fold), or a tanh-based composite
(fuzz / cubic / sine) depending on the preset.

Shipped:

- :func:`vamp` — the main programmable VAMP stage.
- :func:`vamp_light`, :func:`vamp_medium`, :func:`vamp_heavy`,
  :func:`vamp_fuzz` — preset wrappers around :func:`vamp`.
- :func:`overdrive` — classic guitar overdrive: mild tanh + tone
  control, less aggressive than :func:`vamp`.
- :func:`overdrive_soft`, :func:`overdrive_classic`,
  :func:`overdrive_crunch` — preset wrappers around :func:`overdrive`.
- :func:`dual_overdrive` — two overdrive stages in series with
  different voicings; mimics the old ``dual_overdrive`` presets.
"""

from __future__ import annotations


_VAMP_WAVESHAPES = ("tube", "hard", "fold", "fuzz", "rectify", "cubic", "sine")


def _apply_waveshape(node, waveshape: str):
    """Internal dispatch from waveshape name to a SignalFlow node shape.

    The tube/hard/fold/fuzz/cubic shapes map to distinct audible
    voicings; rectify and sine fall back to tanh-based variants
    because SignalFlow doesn't expose dedicated primitives for
    those and writing per-sample Python DSP inside a node graph
    isn't idiomatic.
    """
    import signalflow as sf
    ws = waveshape.lower()
    if ws == "tube":
        return sf.Tanh(node)
    if ws == "hard":
        return sf.Clip(node, -1.0, 1.0)
    if ws == "fold":
        return sf.Fold(node, -1.0, 1.0)
    if ws == "fuzz":
        # Double tanh for extra harmonic density.
        return sf.Tanh(sf.Tanh(node) * 2.0)
    if ws == "rectify":
        # Full-wave-rectify-flavoured shape: square, bounded by tanh.
        return sf.Tanh(node * node) * 2.0 - 1.0
    if ws == "cubic":
        # Cubic + tanh blend: asymmetric, warmer than pure tanh.
        return sf.Tanh(node) * 0.6 + sf.Tanh(node * node * node) * 0.4
    if ws == "sine":
        # Tanh-at-lower-gain gives a mellower, sine-ish knee.
        return sf.Tanh(node * 0.5)
    raise ValueError(
        f"waveshape must be one of {_VAMP_WAVESHAPES}, got {waveshape!r}"
    )


def _maybe_filter(node, cutoff: float | None, filter_type: str):
    if cutoff is None or cutoff <= 0:
        return node
    if filter_type not in ("lp", "hp"):
        raise ValueError(
            f"filter_type must be 'lp' or 'hp', got {filter_type!r}"
        )
    import signalflow as sf
    sf_type = "low_pass" if filter_type == "lp" else "high_pass"
    return sf.SVFilter(node, sf_type, float(cutoff), 0.0)


# ---------------------------------------------------------------------------
# VAMP
# ---------------------------------------------------------------------------


def vamp(
    input_node,
    drive: float = 3.0,
    waveshape: str = "tube",
    bias: float = 0.0,
    gain: float = 0.7,
    mix: float = 1.0,
    pre_filter: float | None = None,
    pre_filter_type: str = "hp",
    post_filter: float | None = None,
    post_filter_type: str = "lp",
):
    """Advanced amp / overdrive / waveshaping processor.

    Parameters
    ----------
    input_node
        SignalFlow node producing the dry signal.
    drive : float
        Pre-waveshape gain multiplier. ``1.0`` is neutral; larger
        values push harder into the waveshape.
    waveshape : str
        One of ``'tube'``, ``'hard'``, ``'fold'``, ``'fuzz'``,
        ``'rectify'``, ``'cubic'``, ``'sine'``.
    bias : float
        DC offset added before the waveshape. Use for asymmetric
        harmonic content (tube-style even-order emphasis). Must be
        in ``(-1, 1)``; 0 is neutral.
    gain : float
        Output makeup gain. Typical range ``0.5 .. 2.0``.
    mix : float
        Dry/wet in ``[0, 1]``.
    pre_filter, post_filter : float | None
        Optional filter cutoffs (Hz). ``None`` disables.
    pre_filter_type, post_filter_type : str
        ``'lp'`` or ``'hp'``. Defaults match the old numpy VAMP.
    """
    if drive <= 0:
        raise ValueError(f"drive must be positive, got {drive}")
    if abs(bias) >= 1:
        raise ValueError(f"bias must be in (-1, 1), got {bias}")
    if gain <= 0:
        raise ValueError(f"gain must be positive, got {gain}")
    if waveshape.lower() not in _VAMP_WAVESHAPES:
        raise ValueError(
            f"waveshape must be one of {_VAMP_WAVESHAPES}, got {waveshape!r}"
        )
    mix = max(0.0, min(1.0, float(mix)))

    # Pre-filter (if any) -> pre-gain -> bias -> waveshape -> post-filter -> gain
    node = _maybe_filter(input_node, pre_filter, pre_filter_type)
    node = node * float(drive)
    if bias != 0:
        node = node + float(bias)
    node = _apply_waveshape(node, waveshape)
    if bias != 0:
        node = node - float(bias)
    node = _maybe_filter(node, post_filter, post_filter_type)
    wet = node * float(gain)

    if mix >= 1.0:
        return wet
    if mix <= 0.0:
        return input_node
    return input_node * (1.0 - mix) + wet * mix


def vamp_light(input_node):
    """Mild tube-style warmth; the cleanest VAMP preset."""
    return vamp(input_node, drive=1.8, waveshape="tube", gain=0.8)


def vamp_medium(input_node):
    """Moderate drive, tube character."""
    return vamp(input_node, drive=3.5, waveshape="tube", gain=0.7)


def vamp_heavy(input_node):
    """High-gain hard-clip voicing for aggressive leads."""
    return vamp(input_node, drive=7.0, waveshape="hard", gain=0.55)


def vamp_fuzz(input_node):
    """Double-tanh fuzz voicing."""
    return vamp(input_node, drive=5.0, waveshape="fuzz", gain=0.5)


# ---------------------------------------------------------------------------
# Overdrive
# ---------------------------------------------------------------------------


def overdrive(
    input_node,
    drive: float = 2.5,
    tone: float = 2500.0,
    mix: float = 1.0,
):
    """Classic guitar overdrive.

    Gentler than :func:`vamp_heavy` — tanh with a mild tone-shaping
    low-pass. Good at the end of a chain where you want presence
    without the fizz.
    """
    if drive <= 0:
        raise ValueError(f"drive must be positive, got {drive}")
    if tone <= 0:
        raise ValueError(f"tone must be positive, got {tone}")
    mix = max(0.0, min(1.0, float(mix)))

    import signalflow as sf
    wet = sf.SVFilter(sf.Tanh(input_node * float(drive)), "low_pass", float(tone), 0.2)
    if mix >= 1.0:
        return wet
    if mix <= 0.0:
        return input_node
    return input_node * (1.0 - mix) + wet * mix


def overdrive_soft(input_node):
    return overdrive(input_node, drive=1.6, tone=4000.0)


def overdrive_classic(input_node):
    return overdrive(input_node, drive=2.5, tone=2500.0)


def overdrive_crunch(input_node):
    return overdrive(input_node, drive=4.0, tone=1800.0)


def dual_overdrive(
    input_node,
    drive_low: float = 2.0,
    drive_high: float = 4.0,
    tone_low: float = 3500.0,
    tone_high: float = 1500.0,
    mix: float = 1.0,
):
    """Two-stage overdrive: low-gain + high-gain voicings in series.

    Produces a thicker, more saturated sound than a single stage at
    the same total gain because each stage clips its own harmonics
    rather than piling them all into one waveshape.
    """
    stage1 = overdrive(input_node, drive=drive_low, tone=tone_low, mix=1.0)
    stage2 = overdrive(stage1, drive=drive_high, tone=tone_high, mix=1.0)
    mix = max(0.0, min(1.0, float(mix)))
    if mix >= 1.0:
        return stage2
    if mix <= 0.0:
        return input_node
    return input_node * (1.0 - mix) + stage2 * mix
