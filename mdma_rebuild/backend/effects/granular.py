"""Granular synthesis from a pre-recorded buffer.

SignalFlow's :class:`signalflow.Granulator` needs a
:class:`signalflow.Buffer` to read grains from. For real-time input,
the caller captures a short window via
:class:`signalflow.BufferRecorder` first and passes the resulting
buffer to these helpers; for pre-rendered material, the caller hands
in the buffer directly.

Shipped:

- :func:`granular_stretch` — slow-moving position for time-stretch.
- :func:`granular_freeze` — fixed position.
- :func:`granular_scatter` — random per-grain position jitter.
- :func:`granular_shimmer` — scatter + octave-up rate for pitch-
  shifted shimmer.
- :func:`granular_reverse` — negative playback rate.
- :func:`granular_stutter` — grains locked to an external clock for
  rhythmic repetition.
- :func:`granular_cloud` — dense wide-pan grain cloud.

Every helper takes a ``buffer`` (the source material) as the first
argument so they stand alone; the Pattern chain integration is via
a partial::

    from functools import partial
    from mdma_rebuild.backend.effects.granular import granular_stretch

    buffer = sf.Buffer("voice.wav")
    pattern.ir(partial(granular_stretch, buffer=buffer), speed=0.5)

The ``input_node`` signature convention (``fn(input_node, **params)``)
is followed by the :func:`capture_and_granulate` helper, which
records the input into a buffer and grains from it live — suitable
for direct use in ``Pattern.fx`` / ``Pattern.ir``.
"""

from __future__ import annotations

from typing import Optional


def _validate_duration(duration: float) -> float:
    if duration <= 0:
        raise ValueError(f"grain duration must be positive, got {duration}")
    return float(duration)


def _validate_rate(rate: float, allow_negative: bool = False) -> float:
    if not allow_negative and rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    return float(rate)


# ---------------------------------------------------------------------------
# Buffer-based helpers (user supplies the buffer)
# ---------------------------------------------------------------------------


def granular_stretch(
    buffer,
    speed: float = 0.3,
    duration: float = 0.15,
    density: float = 40.0,
    pan_spread: float = 0.4,
):
    """Time-stretch a buffer by moving the grain position slowly.

    ``speed`` is the position-advance rate relative to real-time
    playback; ``1.0`` walks through the buffer at normal speed,
    ``0.5`` at half speed (so 1 s of source plays over 2 s of
    output). ``density`` is grains per second (clock rate).
    ``pan_spread`` is max per-grain pan deviation from centre.
    """
    duration = _validate_duration(duration)
    speed = _validate_rate(speed)
    if density <= 0:
        raise ValueError(f"density must be positive, got {density}")

    import signalflow as sf
    # Position ramps from 0..1 at ``speed`` cycles/sec across the
    # buffer. A SineLFO with min=0 / max=1 oscillates; for stretch
    # we want monotonic — use a phasor-style ramp via the LFO's
    # triangle half-cycle. In practice speed is low enough that an
    # LFO works as a readhead.
    pos_lfo = sf.SawLFO(frequency=speed, min=0.0, max=1.0)
    clock = sf.Impulse(frequency=density)
    pan = sf.SineLFO(
        frequency=density * 0.37,  # decorrelated so pan doesn't march
        min=-pan_spread,
        max=pan_spread,
    )
    return sf.Granulator(
        buffer=buffer,
        clock=clock,
        pos=pos_lfo,
        duration=duration,
        pan=pan,
        rate=1.0,
        wrap=True,
    )


def granular_freeze(
    buffer,
    position: float = 0.5,
    duration: float = 0.15,
    density: float = 40.0,
    pan_spread: float = 0.3,
):
    """Freeze at a fixed ``position`` (``[0, 1]``) in the buffer."""
    if position < 0 or position > 1:
        raise ValueError(f"position must be in [0, 1], got {position}")
    duration = _validate_duration(duration)
    if density <= 0:
        raise ValueError(f"density must be positive, got {density}")

    import signalflow as sf
    clock = sf.Impulse(frequency=density)
    pan = sf.SineLFO(
        frequency=density * 0.41,
        min=-pan_spread,
        max=pan_spread,
    )
    return sf.Granulator(
        buffer=buffer,
        clock=clock,
        pos=float(position),
        duration=duration,
        pan=pan,
        wrap=True,
    )


def granular_scatter(
    buffer,
    duration: float = 0.1,
    density: float = 30.0,
    spread: float = 0.5,
    centre: float = 0.5,
):
    """Randomised per-grain position jitter around ``centre``.

    ``spread`` is the full-swing half-width of the position random
    walk in buffer units (``[0, 1]``).
    """
    duration = _validate_duration(duration)
    if density <= 0:
        raise ValueError(f"density must be positive, got {density}")
    if spread < 0:
        raise ValueError(f"spread must be non-negative, got {spread}")
    if centre < 0 or centre > 1:
        raise ValueError(f"centre must be in [0, 1], got {centre}")

    import signalflow as sf
    # RandomImpulse-driven position for per-grain jumps.
    clock = sf.Impulse(frequency=density)
    pos = sf.SampleAndHold(
        sf.SineLFO(
            frequency=density * 1.7,
            min=max(0.0, centre - spread),
            max=min(1.0, centre + spread),
        ),
        clock,
    )
    return sf.Granulator(
        buffer=buffer,
        clock=clock,
        pos=pos,
        duration=duration,
        wrap=True,
    )


def granular_shimmer(
    buffer,
    duration: float = 0.12,
    density: float = 50.0,
    semitones: float = 12.0,
):
    """Scatter + octave-up (by default) grain rate for shimmer-style
    pitch-shifted clouds."""
    duration = _validate_duration(duration)
    if density <= 0:
        raise ValueError(f"density must be positive, got {density}")

    rate = 2.0 ** (float(semitones) / 12.0)
    import signalflow as sf
    clock = sf.Impulse(frequency=density)
    pos = sf.SampleAndHold(
        sf.SineLFO(frequency=density * 1.9, min=0.0, max=1.0),
        clock,
    )
    return sf.Granulator(
        buffer=buffer,
        clock=clock,
        pos=pos,
        duration=duration,
        rate=rate,
        wrap=True,
    )


def granular_reverse(
    buffer,
    speed: float = 0.3,
    duration: float = 0.2,
    density: float = 40.0,
):
    """Play grains at negative rate for backwards granular playback."""
    duration = _validate_duration(duration)
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    if density <= 0:
        raise ValueError(f"density must be positive, got {density}")

    import signalflow as sf
    pos_lfo = sf.SawLFO(frequency=speed, min=0.0, max=1.0)
    clock = sf.Impulse(frequency=density)
    return sf.Granulator(
        buffer=buffer,
        clock=clock,
        pos=pos_lfo,
        duration=duration,
        rate=-1.0,
        wrap=True,
    )


def granular_stutter(
    buffer,
    clock,
    position: float = 0.5,
    duration: float = 0.06,
):
    """Grains locked to an external ``clock`` node (Impulse, LFO > 0
    threshold, etc.). Produces tight rhythmic stutter.

    Pair with ``session.backend_clock()`` or a ``signalflow.Impulse``
    to sync to tempo.
    """
    duration = _validate_duration(duration)
    if position < 0 or position > 1:
        raise ValueError(f"position must be in [0, 1], got {position}")

    import signalflow as sf
    return sf.Granulator(
        buffer=buffer,
        clock=clock,
        pos=float(position),
        duration=duration,
        wrap=True,
    )


def granular_cloud(
    buffer,
    duration: float = 0.2,
    density: float = 80.0,
    pan_spread: float = 0.9,
):
    """Dense, wide-pan grain cloud across the whole buffer.

    Higher ``density`` costs more CPU; 80 grains/sec is a good
    ambient-cloud starting point.
    """
    duration = _validate_duration(duration)
    if density <= 0:
        raise ValueError(f"density must be positive, got {density}")
    if pan_spread < 0:
        raise ValueError(f"pan_spread must be non-negative, got {pan_spread}")

    import signalflow as sf
    clock = sf.Impulse(frequency=density)
    pos = sf.SampleAndHold(
        sf.SineLFO(frequency=density * 2.3, min=0.0, max=1.0),
        clock,
    )
    pan = sf.SineLFO(
        frequency=density * 0.53,
        min=-pan_spread,
        max=pan_spread,
    )
    return sf.Granulator(
        buffer=buffer,
        clock=clock,
        pos=pos,
        duration=duration,
        pan=pan,
        wrap=True,
    )


# ---------------------------------------------------------------------------
# Input-stream helper (fits the fn(input_node, **params) convention)
# ---------------------------------------------------------------------------


def capture_and_granulate(
    input_node,
    capture_seconds: float = 2.0,
    mode: str = "stretch",
    **kwargs,
):
    """Record ``input_node`` into a buffer and granulate from it live.

    ``mode`` selects the granulator callable:

    - ``"stretch"``  -> :func:`granular_stretch`
    - ``"freeze"``   -> :func:`granular_freeze`
    - ``"scatter"``  -> :func:`granular_scatter`
    - ``"shimmer"``  -> :func:`granular_shimmer`
    - ``"cloud"``    -> :func:`granular_cloud`

    Additional ``**kwargs`` are forwarded to the chosen granulator.
    Rhythmic ``granular_stutter`` and the reverse variant want an
    external clock / specific rate argument and should be called
    directly with a known buffer.

    Captures roll over (``loop=True``) so the granulator always has
    material even on long plays.
    """
    if capture_seconds <= 0:
        raise ValueError(
            f"capture_seconds must be positive, got {capture_seconds}"
        )
    modes = {
        "stretch": granular_stretch,
        "freeze": granular_freeze,
        "scatter": granular_scatter,
        "shimmer": granular_shimmer,
        "cloud": granular_cloud,
    }
    if mode not in modes:
        raise ValueError(
            f"mode must be one of {sorted(modes)}, got {mode!r}"
        )

    import signalflow as sf
    from ..utils import current_sample_rate

    sample_rate = int(round(current_sample_rate()))
    num_frames = max(1, int(sample_rate * float(capture_seconds)))
    buffer = sf.Buffer(1, num_frames)
    sf.BufferRecorder(
        buffer=buffer,
        input=input_node,
        feedback=0.0,
        loop=True,
    )
    return modes[mode](buffer, **kwargs)
