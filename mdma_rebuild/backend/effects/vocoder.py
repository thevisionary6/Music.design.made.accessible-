"""Phase-vocoder based time/pitch manipulation.

SignalFlow exposes :class:`signalflow.FFTPhaseVocoder` (trigger-driven,
fixed-rate) and :class:`signalflow.FFTContinuousPhaseVocoder` (live
rate modulation). This module wraps both with the Phase 8 DSP-callable
convention so they plug into ``Pattern.fx`` / ``Pattern.spec``.

Shipped:

- :func:`time_stretch` — change duration without changing pitch
  (continuous rate, like ``rate=0.5`` doubles duration).
- :func:`phase_vocoder_freeze` — single-shot freeze via
  :class:`FFTPhaseVocoder` (useful when paired with an external
  clock/trigger).

A traditional carrier/modulator channel vocoder (voice over a chord
pad, for example) requires a filterbank analyser per modulator band
and an envelope follower per band. That's a larger build — see the
open questions in the Phase 9 summary.
"""

from __future__ import annotations


def _analyse(input_node, fft_size: int, hop_size: int):
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
    import signalflow as sf
    return sf.IFFT(spectral_node, do_window=True)


def time_stretch(
    input_node,
    rate: float = 1.0,
    mix: float = 1.0,
    fft_size: int = 1024,
    hop_size: int = 128,
):
    """Pitch-preserving time stretch via continuous phase vocoder.

    Parameters
    ----------
    input_node
        SignalFlow node producing the dry signal.
    rate : float
        Playback rate. ``1.0`` is no change. ``0.5`` doubles the
        duration (half speed, same pitch); ``2.0`` halves it.
    mix : float
        Dry/wet in ``[0, 1]``.

    Negative rates are not supported by the underlying SignalFlow
    node; use a reversed :class:`Buffer` + :func:`.granular.reverse`
    for backwards playback.
    """
    if rate < 0:
        raise ValueError(f"rate must be non-negative, got {rate}")
    import signalflow as sf
    mix = max(0.0, min(1.0, float(mix)))
    spec = _analyse(input_node, fft_size, hop_size)
    wet = _synthesise(
        sf.FFTContinuousPhaseVocoder(spec, rate=float(rate))
    )
    if mix >= 1.0:
        return wet
    if mix <= 0.0:
        return input_node
    return input_node * (1.0 - mix) + wet * mix


def phase_vocoder_freeze(
    input_node,
    mix: float = 1.0,
    fft_size: int = 1024,
    hop_size: int = 128,
):
    """Single-shot spectral freeze via the non-continuous phase
    vocoder.

    Holds the spectrum captured on the next FFT frame. Pair with a
    trigger-driven flow to control when the freeze engages.
    """
    import signalflow as sf
    mix = max(0.0, min(1.0, float(mix)))
    spec = _analyse(input_node, fft_size, hop_size)
    wet = _synthesise(sf.FFTPhaseVocoder(spec))
    if mix >= 1.0:
        return wet
    if mix <= 0.0:
        return input_node
    return input_node * (1.0 - mix) + wet * mix
