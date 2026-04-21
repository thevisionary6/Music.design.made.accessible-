"""Spectral / FFT-based effects.

Wraps SignalFlow's ``FFT``/``IFFT`` family: every effect here feeds
the input through ``sf.FFT``, applies one or more spectral-domain
nodes, then runs ``sf.IFFT`` to return to the time domain. This keeps
the callables composable with the rest of the chain machinery —
a caller can do ``pattern.spec(spectral_blur, depth=0.8)`` and the
Pattern's ``_apply_chain`` will wire the FFT round-trip through the
voice node.

Shipped:

- :func:`spectral_freeze` — phase-vocoder freeze at the current
  magnitude spectrum (rate=0 continuous phase vocoder).
- :func:`spectral_shift` — continuous-phase-vocoder time/pitch
  shift via the ``rate`` parameter.
- :func:`spectral_blur` — phase randomisation to smear the harmonic
  structure (good for pad-like textures).
- :func:`spectral_contrast` — sharpen or soften the spectral peaks
  via :class:`signalflow.FFTContrast`.
- :func:`spectral_lpf` — lowpass in the frequency domain
  (zero-phase, brick-wall feel).
- :func:`spectral_gate` — FFT-bin-level noise gate; brittle but
  effective for cleanup on noisy inputs.

FFT size and hop size are tunable via the ``fft_size`` / ``hop_size``
kwargs on every function; defaults track SignalFlow's own defaults
(1024 / 128) for predictable latency.

Every callable honours the Phase 8 convention
``fn(input_node, **params) -> Node`` and imports SignalFlow lazily.
"""

from __future__ import annotations

from typing import Optional


def _analyse(input_node, fft_size: int, hop_size: int):
    """Forward FFT with sanity-checked sizes."""
    if fft_size <= 0 or fft_size & (fft_size - 1):
        raise ValueError(
            f"fft_size must be a positive power of two, got {fft_size}"
        )
    if hop_size <= 0 or hop_size > fft_size:
        raise ValueError(
            f"hop_size must be in (0, fft_size], got {hop_size}"
        )
    import signalflow as sf
    return sf.FFT(input_node, fft_size=int(fft_size), hop_size=int(hop_size))


def _synthesise(spectral_node):
    """Inverse FFT with the windowing SignalFlow expects."""
    import signalflow as sf
    return sf.IFFT(spectral_node, do_window=True)


def _mix(dry, wet, mix: float):
    """Blend dry + wet. ``mix`` clamped to ``[0, 1]``."""
    mix = max(0.0, min(1.0, float(mix)))
    if mix >= 1.0:
        return wet
    if mix <= 0.0:
        return dry
    return dry * (1.0 - mix) + wet * mix


# ---------------------------------------------------------------------------
# Freeze / shift (phase vocoder)
# ---------------------------------------------------------------------------


def spectral_freeze(
    input_node,
    mix: float = 1.0,
    fft_size: int = 1024,
    hop_size: int = 128,
):
    """Freeze the current spectrum indefinitely.

    Uses :class:`signalflow.FFTContinuousPhaseVocoder` at ``rate=0``
    so the magnitude spectrum at the moment of engagement is held
    forever while phases keep advancing — the canonical "infinite
    sustain" texture.
    """
    import signalflow as sf
    spec = _analyse(input_node, fft_size, hop_size)
    wet = _synthesise(sf.FFTContinuousPhaseVocoder(spec, rate=0.0))
    return _mix(input_node, wet, mix)


def spectral_shift(
    input_node,
    rate: float = 0.5,
    mix: float = 1.0,
    fft_size: int = 1024,
    hop_size: int = 128,
):
    """Time-stretch / compress via continuous phase vocoder.

    ``rate=1.0`` passes through, ``rate=0.5`` stretches to twice the
    original duration, ``rate=2.0`` plays back at double speed.
    Negative rates are not supported by the underlying node and will
    raise.
    """
    if rate < 0:
        raise ValueError(f"rate must be non-negative, got {rate}")
    import signalflow as sf
    spec = _analyse(input_node, fft_size, hop_size)
    wet = _synthesise(sf.FFTContinuousPhaseVocoder(spec, rate=float(rate)))
    return _mix(input_node, wet, mix)


# ---------------------------------------------------------------------------
# Spectral modifiers
# ---------------------------------------------------------------------------


def spectral_blur(
    input_node,
    level: float = 1.0,
    mix: float = 1.0,
    fft_size: int = 1024,
    hop_size: int = 128,
):
    """Phase randomisation — smears transients and harmonic
    relationships into a pad-like texture.

    ``level`` in ``[0, 1]`` scales how much of the original phase is
    replaced; ``0`` is passthrough, ``1`` is full randomisation.
    """
    if level < 0 or level > 1:
        raise ValueError(f"level must be in [0, 1], got {level}")
    import signalflow as sf
    spec = _analyse(input_node, fft_size, hop_size)
    wet = _synthesise(sf.FFTRandomPhase(spec, level=float(level)))
    return _mix(input_node, wet, mix)


def spectral_contrast(
    input_node,
    contrast: float = 2.0,
    mix: float = 1.0,
    fft_size: int = 1024,
    hop_size: int = 128,
):
    """Enhance or soften spectral peaks.

    ``contrast > 1`` exaggerates peaks relative to valleys (sharper,
    more articulated); ``contrast < 1`` flattens the spectrum
    (duller, more midrangey). Must be positive.
    """
    if contrast <= 0:
        raise ValueError(f"contrast must be positive, got {contrast}")
    import signalflow as sf
    spec = _analyse(input_node, fft_size, hop_size)
    wet = _synthesise(sf.FFTContrast(spec, contrast=float(contrast)))
    return _mix(input_node, wet, mix)


def spectral_lpf(
    input_node,
    cutoff: float = 4000.0,
    mix: float = 1.0,
    fft_size: int = 1024,
    hop_size: int = 128,
):
    """Zero-phase lowpass filter in the frequency domain.

    Brick-wall character — sharper cutoff than an IIR filter at the
    same frequency. Good for aggressive high-frequency removal
    without phase smearing.
    """
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    import signalflow as sf
    spec = _analyse(input_node, fft_size, hop_size)
    wet = _synthesise(sf.FFTLPF(spec, frequency=float(cutoff)))
    return _mix(input_node, wet, mix)


def spectral_gate(
    input_node,
    threshold: float = 0.5,
    mix: float = 1.0,
    fft_size: int = 1024,
    hop_size: int = 128,
):
    """FFT-bin-level noise gate.

    Bins below ``threshold`` are silenced; bins above pass through.
    Produces a distinctive "cleaned" sound and is useful for
    suppressing low-level hum or noise on sustained material.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    import signalflow as sf
    spec = _analyse(input_node, fft_size, hop_size)
    wet = _synthesise(sf.FFTNoiseGate(spec, threshold=float(threshold)))
    return _mix(input_node, wet, mix)
