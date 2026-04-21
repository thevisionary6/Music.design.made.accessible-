"""Pitch / frequency effects built from SignalFlow primitives.

- :func:`ring_mod` — ring modulation (input * sine carrier). Produces
  sum-and-difference sidebands; classic sci-fi / bell sound.
- :func:`amplitude_mod` — amplitude modulation (input * (1 + mod * sine)).
  Softer than ring_mod; tremolo-adjacent when rate is low, sideband-
  generating when rate is high.
- :func:`detune_unison` — stack N detuned copies of the input for a
  thick unison character; useful on its own or as a ``.fx()`` stage
  inside a chain.

Pitch shifting (keeping duration, changing pitch) is deferred until
the SignalFlow version gains a first-class pitch-shifter primitive.
:class:`signalflow.TimeShift` changes duration, not pitch; emulating
a clean pitch shifter from delay lines is a Phase 9+ research task.
"""

from __future__ import annotations


def ring_mod(input_node, rate: float = 440.0):
    """Ring modulation at ``rate`` Hz.

    Output is input * sine(2*pi*rate*t), producing sum-and-difference
    sidebands around the input spectrum. Classic Dalek voice,
    sci-fi robot sounds, bell-like metallics.
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    import signalflow as sf
    return input_node * sf.SineOscillator(float(rate))


def amplitude_mod(
    input_node,
    rate: float = 5.0,
    depth: float = 1.0,
):
    """Amplitude modulation.

    Output is input * (1 + depth * sine(2*pi*rate*t)). Different
    from :func:`ring_mod` in that at ``depth <= 1`` the envelope
    never goes negative, so the effect stays a pure amplitude
    shift; at high rates it generates sidebands like ring mod.
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    if depth < 0:
        raise ValueError(f"depth must be non-negative, got {depth}")
    import signalflow as sf
    mod = sf.SineOscillator(float(rate)) * float(depth) + 1.0
    return input_node * mod


def detune_unison(
    input_node,
    voices: int = 3,
    spread: float = 0.02,
):
    """Stack ``voices`` detuned copies of the input via very-short
    LFO-modulated delays.

    ``spread`` is the delay-time deviation in seconds per voice.
    Not a true pitch detuner (no sample-rate manipulation), but the
    LFO-driven short delays produce the chorus-y detune-character
    most users want from a unison stack.
    """
    if voices < 1:
        raise ValueError(f"voices must be >= 1, got {voices}")
    if spread < 0:
        raise ValueError(f"spread must be non-negative, got {spread}")
    import signalflow as sf
    # Each voice gets a different LFO rate so they don't all sweep
    # together, which would just sound like one modulation.
    total = input_node
    for i in range(int(voices) - 1):
        rate = 0.2 + 0.13 * (i + 1)  # low-rate, unique per voice
        lfo = sf.SineLFO(
            frequency=rate,
            min=0.001,
            max=max(0.002, 0.001 + float(spread)),
        )
        total = total + sf.OneTapDelay(
            input_node,
            delay_time=lfo,
            max_delay_time=max(0.01, float(spread) + 0.005),
        )
    # Normalise so unison doesn't clip — divide by the voice count
    # (approximate; true RMS would be sqrt(N) but linear is fine
    # here for a perceptual match).
    return total * (1.0 / float(voices))
