"""Saturation variants beyond :mod:`.soft_clip` and :mod:`.hard_clip`.

These are the "warmth" end of the distortion spectrum — subtler,
program-material-friendly nonlinearities rather than aggressive
clip/fold/bitcrush territory.

- :func:`tube` — asymmetric tube-style warmth via tanh with a tiny
  DC offset.
- :func:`tape` — soft tanh saturation plus gentle high-cut,
  matching the self-muffling character of magnetic tape.
- :func:`fuzz` — aggressive hard-knee saturation. Brighter and
  harsher than :func:`soft_clip` but less brutal than
  :func:`hard_clip`.
"""

from __future__ import annotations


def tube(input_node, drive: float = 2.0, bias: float = 0.15):
    """Tube-style asymmetric saturation.

    The DC ``bias`` shifts the input before the tanh so positive and
    negative excursions clip at different thresholds — the
    characteristic asymmetric harmonic signature of triodes.
    """
    if drive <= 0:
        raise ValueError(f"drive must be positive, got {drive}")
    if abs(bias) >= 1:
        raise ValueError(
            f"bias must be in (-1, 1), got {bias}"
        )
    import signalflow as sf
    # Shift, saturate, shift back so the output stays centred.
    return sf.Tanh((input_node + float(bias)) * float(drive)) - float(bias)


def tape(
    input_node,
    drive: float = 1.5,
    hi_cut_hz: float = 8000.0,
):
    """Tape saturation: tanh warmth + gentle high-cut.

    Magnetic tape compresses transients and rolls off the high end;
    the combination here reproduces that character.
    """
    if drive <= 0:
        raise ValueError(f"drive must be positive, got {drive}")
    if hi_cut_hz <= 0:
        raise ValueError(f"hi_cut_hz must be positive, got {hi_cut_hz}")
    import signalflow as sf
    saturated = sf.Tanh(input_node * float(drive))
    return sf.SVFilter(saturated, "low_pass", float(hi_cut_hz), 0.0)


def fuzz(input_node, drive: float = 5.0, tone: float = 2500.0):
    """Hard-knee fuzz saturation.

    Combines heavy pre-gain (``drive``) through tanh with a tone
    control — a gentle lowpass that keeps the output musical
    despite the saturation.
    """
    if drive <= 0:
        raise ValueError(f"drive must be positive, got {drive}")
    if tone <= 0:
        raise ValueError(f"tone must be positive, got {tone}")
    import signalflow as sf
    saturated = sf.Tanh(input_node * float(drive))
    return sf.SVFilter(saturated, "low_pass", float(tone), 0.3)
