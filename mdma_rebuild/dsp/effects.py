"""Audio effects and filter helper for the MDMA rebuild.

This module defines a collection of audio effects and a helper
function to apply a chain of effects to a one‑dimensional NumPy
array of float64 samples.  Each effect function operates on a
buffer and returns a processed buffer.  The effect names are
registered in the ``_effect_funcs`` dictionary.  Additional
functions can be added here to extend the available palette.

Effects are grouped into several categories:

* Reverb: small, large, plate, spring, cathedral.
* Delay: simple, ping‑pong, multi‑tap, slapback, tape echo.
* Saturation: soft clip, hard clip, overdrive, fuzz, tube.
* Dynamics: mild compressor, hard compressor, limiter,
  expander, soft clipper.
* Gates: five threshold settings (gate1 … gate5).
* Lo‑fi: bitcrusher, chorus, flanger, phaser, lofi filter,
  and halftime (time‑stretch).  These are creative sound
  degraders for adding vintage or textural character.

The ``apply_effects`` function takes an input buffer, a list of
effect names, and optional filter parameters.  After each effect
is applied, if filter parameters are provided a simple filter is
executed using the helper ``_apply_filter``.  This allows the
user to insert the session's filter bank before or after each
effect without having to duplicate logic in the Monolith engine.

The sample rate for all internal processing is fixed at 48 kHz to
match the default sample rate of the Session and MonolithEngine.

PARAMETER SCALING:
-----------------
All abstract effect parameters use the unified 1-100 scaling system
defined in scaling.py:
  - 0 = minimum/off
  - 50 = moderate/default  
  - 100 = general maximum for clean audio
  - >100 = "wacky" territory (allowed but may produce scuffed audio)

Real-world units (Hz, ms) keep their natural scales.

BUILD ID: effects_v14_chunk1.1
"""

from __future__ import annotations

import numpy as np  # type: ignore
from scipy.signal import convolve  # type: ignore
import math

# Import unified scaling system
from .scaling import (
    parse_param,
    clamp_param,
    is_wacky,
    validate_param,
    scale_to_range,
    scale_drive,
    scale_wet,
    scale_dry,
    scale_feedback,
    scale_resonance,
    scale_threshold_db,
    scale_ratio,
    scale_bits,
    scale_modulation_index,
    PARAM_PRESETS,
)

SAMPLE_RATE: int = 48_000


def _normalise(buf: np.ndarray) -> np.ndarray:
    """Normalise a buffer to the range [-1, 1] if necessary."""
    max_val = np.max(np.abs(buf))
    if max_val > 0:
        buf = buf / max_val
    return buf.astype(np.float64)


# ---------------------------------------------------------------------------
# LEGACY COMPATIBILITY WRAPPERS
# ---------------------------------------------------------------------------
# These wrap the new scaling module for backwards compatibility with
# existing effect code. New code should import from scaling.py directly.

def scale_amount(amount: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Scale a 0-100 amount to a specific range (legacy wrapper)."""
    return scale_to_range(clamp_param(amount, allow_wacky=True), min_val, max_val)


def scale_time_ms(amount: float, max_ms: float = 2000.0) -> float:
    """Scale 0-100 to time in milliseconds (legacy wrapper)."""
    return scale_to_range(clamp_param(amount), 0.0, max_ms)


def scale_freq(amount: float, min_hz: float = 20.0, max_hz: float = 20000.0) -> float:
    """Scale 0-100 to frequency (logarithmic).
    
    NOTE: For frequency parameters, prefer using real Hz values.
    This function is for cases where a 0-100 "brightness" style control is needed.
    """
    amount = clamp_param(amount)
    log_min = np.log10(min_hz)
    log_max = np.log10(max_hz)
    log_val = log_min + (amount / 100.0) * (log_max - log_min)
    return 10 ** log_val


def parse_amount(value) -> float:
    """Parse an amount value (legacy wrapper for parse_param)."""
    return parse_param(value, default=50.0)




# ---------------------------------------------------------------------------
# Reverb effects
def _reverb_small(buffer: np.ndarray) -> np.ndarray:
    """Small room reverb using a short exponential decay impulse."""
    impulse_len = int(0.1 * SAMPLE_RATE)
    impulse = np.exp(-5.0 * np.linspace(0.0, 1.0, impulse_len))
    out = convolve(buffer, impulse, mode='full')[: len(buffer)]
    return _normalise(out)


def _reverb_large(buffer: np.ndarray) -> np.ndarray:
    """Large hall reverb with a longer decay."""
    impulse_len = int(0.4 * SAMPLE_RATE)
    impulse = np.exp(-3.0 * np.linspace(0.0, 1.0, impulse_len))
    out = convolve(buffer, impulse, mode='full')[: len(buffer)]
    return _normalise(out)


def _reverb_plate(buffer: np.ndarray) -> np.ndarray:
    """Plate reverb emulation with a medium decay and brighter tone."""
    impulse_len = int(0.25 * SAMPLE_RATE)
    # Slightly brighter impulse (slower decay at high frequencies)
    t = np.linspace(0.0, 1.0, impulse_len)
    impulse = np.exp(-4.0 * t) * (1.0 - 0.3 * t)
    out = convolve(buffer, impulse, mode='full')[: len(buffer)]
    return _normalise(out)


def _reverb_spring(buffer: np.ndarray) -> np.ndarray:
    """Spring reverb approximation with a characteristic bouncy decay."""
    impulse_len = int(0.15 * SAMPLE_RATE)
    t = np.linspace(0.0, 1.0, impulse_len)
    # Sinusoidally damped impulse to simulate spring resonance
    impulse = np.exp(-6.0 * t) * np.sin(10.0 * math.pi * t)
    out = convolve(buffer, impulse, mode='full')[: len(buffer)]
    return _normalise(out)


def _reverb_cathedral(buffer: np.ndarray) -> np.ndarray:
    """Cathedral reverb with a very long decay."""
    impulse_len = int(0.6 * SAMPLE_RATE)
    t = np.linspace(0.0, 1.0, impulse_len)
    impulse = np.exp(-2.5 * t)  # slower decay for large space
    out = convolve(buffer, impulse, mode='full')[: len(buffer)]
    return _normalise(out)

# ---------------------------------------------------------------------------
# CONVOLUTION REVERB MODULE - Big Brain Evolver
# ---------------------------------------------------------------------------

# Baked-in impulse response presets
def _make_ir_hall(duration: float = 2.0, brightness: float = 0.5) -> np.ndarray:
    """Generate a synthetic hall impulse response."""
    samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, samples, dtype=np.float64)
    
    # Multi-stage decay for realistic hall
    ir = np.zeros(samples, dtype=np.float64)
    
    # Early reflections (first 50ms)
    early_samples = int(0.05 * SAMPLE_RATE)
    early_delays = [int(0.01 * SAMPLE_RATE), int(0.023 * SAMPLE_RATE), 
                    int(0.037 * SAMPLE_RATE), int(0.047 * SAMPLE_RATE)]
    for d in early_delays:
        if d < samples:
            ir[d] = 0.7 * np.random.uniform(0.5, 1.0)
    
    # Late reverb tail with exponential decay
    decay_rate = 3.0 / duration  # Adjust decay to duration
    late_start = int(0.05 * SAMPLE_RATE)
    noise = np.random.randn(samples - late_start) * 0.3
    envelope = np.exp(-decay_rate * t[late_start:])
    ir[late_start:] += noise * envelope
    
    # Apply brightness filter (simple high shelf simulation)
    if brightness < 0.5:
        # Darker - reduce highs
        alpha = 0.3 + brightness * 0.4
        for i in range(1, len(ir)):
            ir[i] = ir[i] * alpha + ir[i-1] * (1 - alpha)
    elif brightness > 0.5:
        # Brighter - emphasize transients
        diff = np.diff(ir, prepend=ir[0])
        ir = ir + diff * (brightness - 0.5) * 0.5
    
    # Normalize
    max_val = np.max(np.abs(ir))
    if max_val > 0:
        ir = ir / max_val
    
    return ir.astype(np.float64)


def _make_ir_room(size: float = 0.3) -> np.ndarray:
    """Generate a small room impulse response."""
    duration = 0.1 + size * 0.4  # 100ms to 500ms
    samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, samples, dtype=np.float64)
    
    ir = np.zeros(samples, dtype=np.float64)
    
    # Strong early reflections for small room
    num_reflections = int(5 + size * 10)
    for i in range(num_reflections):
        delay = int(np.random.uniform(0.005, duration * 0.3) * SAMPLE_RATE)
        if delay < samples:
            ir[delay] += np.random.uniform(0.3, 0.8) * (1.0 - i / num_reflections)
    
    # Short tail
    decay = np.exp(-10.0 * t) * np.random.randn(samples) * 0.2
    ir += decay
    
    max_val = np.max(np.abs(ir))
    if max_val > 0:
        ir = ir / max_val
    
    return ir.astype(np.float64)


def _make_ir_plate(decay: float = 2.0, damping: float = 0.5) -> np.ndarray:
    """Generate a plate reverb impulse response."""
    samples = int(decay * SAMPLE_RATE)
    t = np.linspace(0, decay, samples, dtype=np.float64)
    
    # Plate characteristics: dense, bright, metallic
    ir = np.random.randn(samples) * 0.1
    
    # Multiple resonant modes
    modes = [300, 500, 800, 1200, 2000, 3500]
    for mode in modes:
        phase = np.random.uniform(0, 2 * np.pi)
        ir += np.sin(2 * np.pi * mode * t + phase) * np.exp(-5 * damping * t) * 0.1
    
    # Overall envelope
    envelope = np.exp(-3.0 / decay * t)
    ir = ir * envelope
    
    max_val = np.max(np.abs(ir))
    if max_val > 0:
        ir = ir / max_val
    
    return ir.astype(np.float64)


def _make_ir_spring(tension: float = 0.5, length: float = 1.0) -> np.ndarray:
    """Generate a spring reverb impulse response."""
    duration = 0.3 + length * 0.5
    samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, samples, dtype=np.float64)
    
    # Spring characteristics: chirpy, bouncy, with dispersion
    base_freq = 50 + tension * 100
    
    # Chirp (frequency changes over time due to dispersion)
    chirp_rate = 20 + tension * 50
    phase = 2 * np.pi * base_freq * t + chirp_rate * t * t
    ir = np.sin(phase) * np.exp(-8 * t)
    
    # Add some noise
    ir += np.random.randn(samples) * 0.1 * np.exp(-15 * t)
    
    max_val = np.max(np.abs(ir))
    if max_val > 0:
        ir = ir / max_val
    
    return ir.astype(np.float64)


def _make_ir_shimmer(decay: float = 3.0, pitch_shift: float = 12.0) -> np.ndarray:
    """Generate a shimmer reverb impulse response with octave shifts."""
    samples = int(decay * SAMPLE_RATE)
    t = np.linspace(0, decay, samples, dtype=np.float64)
    
    # Base reverb
    ir = np.random.randn(samples) * np.exp(-2.0 * t)
    
    # Add pitched components (shimmer effect)
    shift_ratio = 2 ** (pitch_shift / 12.0)  # Semitones to ratio
    
    # Modulated sine waves at shifted pitch
    for harmonic in [1, shift_ratio, shift_ratio * 2]:
        freq = 200 * harmonic
        modulation = 1 + 0.1 * np.sin(2 * np.pi * 0.5 * t)  # Slow vibrato
        ir += np.sin(2 * np.pi * freq * t * modulation) * np.exp(-3 * t) * 0.1
    
    max_val = np.max(np.abs(ir))
    if max_val > 0:
        ir = ir / max_val
    
    return ir.astype(np.float64)


def _make_ir_reverse(duration: float = 1.5) -> np.ndarray:
    """Generate a reverse reverb impulse response."""
    samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, samples, dtype=np.float64)
    
    # Reversed exponential - builds up instead of decaying
    envelope = 1.0 - np.exp(-5.0 * (duration - t) / duration)
    ir = np.random.randn(samples) * envelope
    
    max_val = np.max(np.abs(ir))
    if max_val > 0:
        ir = ir / max_val
    
    return ir.astype(np.float64)


# IR preset registry
IR_PRESETS = {
    'hall': lambda: _make_ir_hall(2.0, 0.5),
    'hall_long': lambda: _make_ir_hall(4.0, 0.4),
    'hall_bright': lambda: _make_ir_hall(2.0, 0.8),
    'hall_dark': lambda: _make_ir_hall(2.5, 0.2),
    'room': lambda: _make_ir_room(0.3),
    'room_small': lambda: _make_ir_room(0.1),
    'room_large': lambda: _make_ir_room(0.6),
    'plate': lambda: _make_ir_plate(2.0, 0.5),
    'plate_bright': lambda: _make_ir_plate(1.5, 0.3),
    'plate_dark': lambda: _make_ir_plate(2.5, 0.8),
    'spring': lambda: _make_ir_spring(0.5, 1.0),
    'spring_tight': lambda: _make_ir_spring(0.8, 0.5),
    'spring_loose': lambda: _make_ir_spring(0.3, 1.5),
    'shimmer': lambda: _make_ir_shimmer(3.0, 12.0),
    'shimmer_fifth': lambda: _make_ir_shimmer(2.5, 7.0),
    'reverse': lambda: _make_ir_reverse(1.5),
    'reverse_long': lambda: _make_ir_reverse(3.0),
}


def convolve_reverb(
    buffer: np.ndarray,
    ir: np.ndarray = None,
    preset: str = None,
    ir_file: str = None,
    wet: float = 50.0,
    dry: float = 50.0,
    stretch: float = 1.0,
    pre_delay: float = 0.0,
    high_cut: float = 20000.0,
    low_cut: float = 20.0,
) -> np.ndarray:
    """Apply convolution reverb to a buffer.
    
    Abstract parameters use unified 1-100 scaling.
    Real-world parameters (Hz, ms) keep natural units.
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio buffer
    ir : np.ndarray, optional
        Custom impulse response array
    preset : str, optional
        Name of built-in IR preset (hall, room, plate, spring, shimmer, reverse)
    ir_file : str, optional
        Path to WAV file to use as impulse response
    wet : float
        Wet signal level (0-100, default 50)
    dry : float
        Dry signal level (0-100, default 50)
    stretch : float
        Time-stretch factor for IR (0.5 = half length, 2.0 = double) - real multiplier
    pre_delay : float
        Pre-delay in milliseconds (real units)
    high_cut : float
        High frequency cutoff in Hz (real units)
    low_cut : float
        Low frequency cutoff in Hz (real units)
    
    Returns
    -------
    np.ndarray
        Processed audio buffer
    """
    # Scale wet/dry from 1-100 to 0-1
    wet_scaled = scale_wet(wet)
    dry_scaled = scale_wet(dry)
    
    # Get or generate impulse response
    if ir is not None:
        impulse = ir.copy()
    elif preset and preset.lower() in IR_PRESETS:
        impulse = IR_PRESETS[preset.lower()]()
    elif ir_file:
        try:
            import wave
            with wave.open(ir_file, 'rb') as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            
            if sampwidth == 2:
                data = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
            elif sampwidth == 4:
                data = np.frombuffer(frames, dtype=np.int32).astype(np.float64) / 2147483648.0
            else:
                data = np.frombuffer(frames, dtype=np.uint8).astype(np.float64) / 128.0 - 1.0
            
            # Convert to mono if stereo
            if channels == 2:
                data = (data[::2] + data[1::2]) / 2.0
            
            # Resample if needed
            if framerate != SAMPLE_RATE:
                ratio = SAMPLE_RATE / framerate
                new_len = int(len(data) * ratio)
                x_old = np.linspace(0, 1, len(data))
                x_new = np.linspace(0, 1, new_len)
                data = np.interp(x_new, x_old, data)
            
            impulse = data.astype(np.float64)
        except Exception:
            # Fall back to hall preset if file loading fails
            impulse = IR_PRESETS['hall']()
    else:
        # Default to hall preset
        impulse = IR_PRESETS['hall']()
    
    # Apply stretch
    if stretch != 1.0 and stretch > 0:
        new_len = int(len(impulse) * stretch)
        if new_len > 0:
            x_old = np.linspace(0, 1, len(impulse))
            x_new = np.linspace(0, 1, new_len)
            impulse = np.interp(x_new, x_old, impulse)
    
    # Apply frequency cuts to IR
    if high_cut < 20000.0 or low_cut > 20.0:
        try:
            import scipy.signal
            nyq = SAMPLE_RATE / 2
            if high_cut < nyq * 0.99:
                b, a = scipy.signal.butter(2, high_cut / nyq, btype='low')
                impulse = scipy.signal.lfilter(b, a, impulse)
            if low_cut > 20.0:
                b, a = scipy.signal.butter(2, low_cut / nyq, btype='high')
                impulse = scipy.signal.lfilter(b, a, impulse)
        except Exception:
            pass
    
    # Apply pre-delay
    if pre_delay > 0:
        delay_samples = int(pre_delay / 1000.0 * SAMPLE_RATE)
        impulse = np.concatenate([np.zeros(delay_samples), impulse])
    
    # Normalize impulse
    max_ir = np.max(np.abs(impulse))
    if max_ir > 0:
        impulse = impulse / max_ir
    
    # Convolve
    wet_signal = convolve(buffer, impulse, mode='full')[:len(buffer)]
    
    # Normalize wet signal
    max_wet = np.max(np.abs(wet_signal))
    if max_wet > 0:
        wet_signal = wet_signal / max_wet
    
    # Mix using scaled values
    out = dry_scaled * buffer + wet_scaled * wet_signal
    
    # Final normalize if clipping
    max_out = np.max(np.abs(out))
    if max_out > 1.0:
        out = out / max_out
    
    return out.astype(np.float64)


def _conv_hall(buffer: np.ndarray) -> np.ndarray:
    """Convolution reverb with hall preset."""
    return convolve_reverb(buffer, preset='hall', wet=40, dry=70)

def _conv_hall_long(buffer: np.ndarray) -> np.ndarray:
    """Convolution reverb with long hall preset."""
    return convolve_reverb(buffer, preset='hall_long', wet=50, dry=60)

def _conv_room(buffer: np.ndarray) -> np.ndarray:
    """Convolution reverb with room preset."""
    return convolve_reverb(buffer, preset='room', wet=35, dry=80)

def _conv_plate(buffer: np.ndarray) -> np.ndarray:
    """Convolution reverb with plate preset."""
    return convolve_reverb(buffer, preset='plate', wet=45, dry=70)

def _conv_spring(buffer: np.ndarray) -> np.ndarray:
    """Convolution reverb with spring preset."""
    return convolve_reverb(buffer, preset='spring', wet=40, dry=75)

def _conv_shimmer(buffer: np.ndarray) -> np.ndarray:
    """Convolution reverb with shimmer preset."""
    return convolve_reverb(buffer, preset='shimmer', wet=50, dry=60)

def _conv_reverse(buffer: np.ndarray) -> np.ndarray:
    """Convolution reverb with reverse preset."""
    return convolve_reverb(buffer, preset='reverse', wet=50, dry=60)




# ---------------------------------------------------------------------------
# Delay effects
def _delay_simple(buffer: np.ndarray) -> np.ndarray:
    """A basic delay with a single feedback path and moderate wet mix."""
    delay_samples = int(0.12 * SAMPLE_RATE)
    feedback = 0.3
    wet = 0.5
    out = np.copy(buffer)
    # Feedback delay line
    for i in range(delay_samples, len(buffer)):
        out[i] += feedback * out[i - delay_samples]
    mixed = (1.0 - wet) * buffer + wet * out
    return _normalise(mixed)


def _delay_pingpong(buffer: np.ndarray) -> np.ndarray:
    """Ping‑pong delay emulation for mono buffers.

    This alternates the polarity of successive echoes to create
    movement between channels.  In mono it simply flips phase on
    every echo.
    """
    delay_samples = int(0.15 * SAMPLE_RATE)
    feedback = 0.4
    wet = 0.5
    out = np.copy(buffer)
    sign = -1.0
    for i in range(delay_samples, len(buffer)):
        out[i] += feedback * sign * out[i - delay_samples]
        # Flip polarity for next echo
        sign *= -1.0 if (i - delay_samples) % delay_samples == 0 else 1.0
    mixed = (1.0 - wet) * buffer + wet * out
    return _normalise(mixed)


def _delay_multitap(buffer: np.ndarray) -> np.ndarray:
    """Multi‑tap delay with three descending echoes."""
    delays = [int(0.1 * SAMPLE_RATE), int(0.2 * SAMPLE_RATE), int(0.35 * SAMPLE_RATE)]
    gains = [0.5, 0.3, 0.2]
    out = buffer.copy().astype(np.float64)
    for d, g in zip(delays, gains):
        # Shift buffer by d samples with zero padding
        shifted = np.pad(buffer, (d, 0), mode='constant')[: len(buffer)]
        out += g * shifted
    return _normalise(out)


def _delay_slapback(buffer: np.ndarray) -> np.ndarray:
    """Short slapback delay reminiscent of rockabilly vocals."""
    delay_samples = int(0.08 * SAMPLE_RATE)
    feedback = 0.25
    wet = 0.6
    out = np.copy(buffer)
    for i in range(delay_samples, len(buffer)):
        out[i] += feedback * out[i - delay_samples]
    mixed = (1.0 - wet) * buffer + wet * out
    return _normalise(mixed)


def _delay_tape_echo(buffer: np.ndarray) -> np.ndarray:
    """Tape echo emulation with longer delay and gentle feedback."""
    delay_samples = int(0.3 * SAMPLE_RATE)
    feedback = 0.35
    wet = 0.6
    out = buffer.copy().astype(np.float64)
    # Use a simple low‑pass filter on the feedback to simulate tape head damping
    lp_coef = 0.1
    feedback_buffer = np.zeros_like(out)
    for i in range(delay_samples, len(buffer)):
        # Low‑pass the feedback path
        fb = feedback_buffer[i - delay_samples] * (1.0 - lp_coef) + out[i - delay_samples] * lp_coef
        feedback_buffer[i] = fb
        out[i] += feedback * fb
    mixed = (1.0 - wet) * buffer + wet * out
    return _normalise(mixed)


# ---------------------------------------------------------------------------
# Saturation effects
def _saturate_soft(buffer: np.ndarray) -> np.ndarray:
    """Soft clipping via a hyperbolic tangent."""
    out = np.tanh(2.0 * buffer)
    return _normalise(out)


def _saturate_hard(buffer: np.ndarray) -> np.ndarray:
    """Hard clipping with a steeper transfer curve."""
    out = np.tanh(5.0 * buffer)
    return _normalise(out)


def _saturate_overdrive(buffer: np.ndarray) -> np.ndarray:
    """Overdrive emulation using an arctangent transfer curve."""
    out = np.arctan(3.0 * buffer)
    return _normalise(out)


def _saturate_fuzz(buffer: np.ndarray) -> np.ndarray:
    """Fuzz distortion using a sign and exponential curve."""
    out = np.sign(buffer) * (1.0 - np.exp(-np.abs(buffer) * 5.0))
    return _normalise(out)


def _saturate_tube(buffer: np.ndarray) -> np.ndarray:
    """Tube saturation emulation via asymmetric clipping."""
    # Apply different gains to positive and negative halves
    pos = np.tanh(3.0 * np.clip(buffer, 0.0, None))
    neg = 0.8 * np.tanh(3.0 * np.clip(buffer, None, 0.0))
    out = pos + neg
    return _normalise(out)


# ---------------------------------------------------------------------------
# VAMP MODULE - Advanced Overdrive & Waveshaping System
# ---------------------------------------------------------------------------

# Built-in waveshape transfer functions
def _make_waveshape_tanh() -> np.ndarray:
    """Smooth tanh waveshape."""
    x = np.linspace(-1, 1, 2048)
    return np.tanh(x * 3).astype(np.float64)

def _make_waveshape_hard() -> np.ndarray:
    """Hard clipping waveshape."""
    x = np.linspace(-1, 1, 2048)
    return np.clip(x * 2, -1, 1).astype(np.float64)

def _make_waveshape_tube() -> np.ndarray:
    """Asymmetric tube-like waveshape."""
    x = np.linspace(-1, 1, 2048)
    pos = np.tanh(x * 2.5)
    neg = np.tanh(x * 1.5) * 0.8
    return np.where(x >= 0, pos, neg).astype(np.float64)

def _make_waveshape_fuzz() -> np.ndarray:
    """Aggressive fuzz waveshape."""
    x = np.linspace(-1, 1, 2048)
    return (np.sign(x) * (1 - np.exp(-np.abs(x) * 5))).astype(np.float64)

def _make_waveshape_rectify() -> np.ndarray:
    """Full wave rectification waveshape."""
    x = np.linspace(-1, 1, 2048)
    return np.abs(x).astype(np.float64)

def _make_waveshape_fold() -> np.ndarray:
    """Wave folding waveshape."""
    x = np.linspace(-1, 1, 2048)
    # Fold waves back at ±0.5
    y = x.copy()
    y = np.where(np.abs(y) > 0.5, np.sign(y) * (1 - np.abs(y)), y)
    return (y * 2).astype(np.float64)

def _make_waveshape_cubic() -> np.ndarray:
    """Cubic soft saturation."""
    x = np.linspace(-1, 1, 2048)
    return (x - x**3 / 3).astype(np.float64) * 1.5

def _make_waveshape_sine() -> np.ndarray:
    """Sine waveshaper for harmonics."""
    x = np.linspace(-1, 1, 2048)
    return np.sin(x * np.pi / 2 * 3).astype(np.float64)


WAVESHAPE_PRESETS = {
    'tanh': _make_waveshape_tanh,
    'hard': _make_waveshape_hard,
    'tube': _make_waveshape_tube,
    'fuzz': _make_waveshape_fuzz,
    'rectify': _make_waveshape_rectify,
    'fold': _make_waveshape_fold,
    'cubic': _make_waveshape_cubic,
    'sine': _make_waveshape_sine,
}


def _apply_simple_filter(buf: np.ndarray, cutoff: float, filter_type: str = 'lp', q: float = 0.707) -> np.ndarray:
    """Apply a simple biquad filter to a buffer.
    
    Parameters
    ----------
    buf : np.ndarray
        Input buffer
    cutoff : float
        Cutoff frequency in Hz
    filter_type : str
        'lp' for lowpass, 'hp' for highpass
    q : float
        Resonance/Q factor
    
    Returns
    -------
    np.ndarray
        Filtered buffer
    """
    try:
        import scipy.signal
        nyq = SAMPLE_RATE / 2
        norm_cutoff = np.clip(cutoff / nyq, 0.001, 0.999)
        b, a = scipy.signal.butter(2, norm_cutoff, btype='low' if filter_type == 'lp' else 'high')
        return scipy.signal.lfilter(b, a, buf).astype(np.float64)
    except Exception:
        return buf


def vamp_process(
    buffer: np.ndarray,
    drive: float = 25.0,
    waveshape: str = 'tube',
    custom_curve: np.ndarray = None,
    pre_filter: float = None,
    pre_filter_type: str = 'hp',
    post_filter: float = None,
    post_filter_type: str = 'lp',
    gain: float = 50.0,
    bias: float = 0.0,
    mix: float = 100.0,
) -> np.ndarray:
    """Advanced amp/overdrive/waveshaping processor.
    
    All abstract parameters use unified 1-100 scaling:
    - 0-100 = normal operating range
    - >100 = "wacky" territory (allowed, may produce scuffed audio)
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio buffer
    drive : float
        Input gain/drive amount (0-100, default 25)
        0 = minimal drive (1x)
        50 = moderate drive (10x)  
        100 = maximum clean drive (20x)
        >100 = wacky overdrive territory
    waveshape : str
        Waveshape preset name (tanh, hard, tube, fuzz, rectify, fold, cubic, sine)
    custom_curve : np.ndarray, optional
        Custom transfer function array (2048 samples, -1 to 1 range)
    pre_filter : float, optional
        Pre-filter cutoff frequency in Hz (real units)
    pre_filter_type : str
        Pre-filter type ('lp' or 'hp')
    post_filter : float, optional  
        Post-filter cutoff frequency in Hz (real units)
    post_filter_type : str
        Post-filter type ('lp' or 'hp')
    gain : float
        Output gain compensation (0-100, default 50)
        0 = 0.5x, 50 = 1.0x, 100 = 4.0x
    bias : float
        DC offset/bias before waveshaping (-1 to 1, real units)
    mix : float
        Dry/wet mix (0-100, default 100 = fully wet)
    
    Returns
    -------
    np.ndarray
        Processed audio buffer
    """
    # Scale abstract params from 1-100 to internal ranges
    drive_mult = scale_drive(drive)  # 0-100 -> 1-20
    mix_scaled = scale_wet(mix)      # 0-100 -> 0-1
    # Gain: 0-100 maps to 0.5-4.0 (50 = unity gain at 1.0)
    gain_mult = scale_to_range(clamp_param(gain), 0.5, 4.0)
    
    dry = buffer.copy()
    out = buffer.astype(np.float64)
    
    # Pre-filter (Hz stays real)
    if pre_filter is not None and pre_filter > 0:
        out = _apply_simple_filter(out, pre_filter, pre_filter_type)
    
    # Apply drive
    out = out * drive_mult
    
    # Apply bias (real value -1 to 1)
    if bias != 0:
        out = out + bias
    
    # Get waveshape curve
    if custom_curve is not None and len(custom_curve) >= 256:
        curve = custom_curve.astype(np.float64)
    elif waveshape.lower() in WAVESHAPE_PRESETS:
        curve = WAVESHAPE_PRESETS[waveshape.lower()]()
    else:
        curve = WAVESHAPE_PRESETS['tube']()
    
    # Apply waveshaping via interpolation
    # Map input (-inf, inf) to curve index (0, len(curve)-1)
    # Use tanh to soft-limit input range first
    normalized = np.tanh(out)  # Now in range [-1, 1]
    indices = ((normalized + 1) / 2 * (len(curve) - 1)).astype(np.float64)
    indices = np.clip(indices, 0, len(curve) - 1)
    
    # Linear interpolation
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, len(curve) - 1)
    frac = indices - idx_floor
    out = curve[idx_floor] * (1 - frac) + curve[idx_ceil] * frac
    
    # Post-filter (Hz stays real)
    if post_filter is not None and post_filter > 0:
        out = _apply_simple_filter(out, post_filter, post_filter_type)
    
    # Gain makeup
    out = out * gain_mult
    
    # Mix
    if mix_scaled < 1.0:
        out = dry * (1 - mix_scaled) + out * mix_scaled
    
    # Normalize if needed
    max_val = np.max(np.abs(out))
    if max_val > 1.0:
        out = out / max_val
    
    return out.astype(np.float64)


def dual_overdrive(
    buffer: np.ndarray,
    drive_low: float = 15.0,
    drive_high: float = 25.0,
    shape_low: str = 'tube',
    shape_high: str = 'tanh',
    crossover: float = 800.0,
    blend: float = 50.0,
    gain: float = 50.0,
) -> np.ndarray:
    """Dual-stage overdrive with separate processing for low/high frequencies.
    
    All abstract parameters use unified 1-100 scaling.
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio buffer
    drive_low : float
        Drive for low frequency band (0-100, default 15)
    drive_high : float
        Drive for high frequency band (0-100, default 25)
    shape_low : str
        Waveshape for low band
    shape_high : str
        Waveshape for high band
    crossover : float
        Crossover frequency between bands (Hz, real units)
    blend : float
        Blend between bands (0-100, 50 = equal, default 50)
    gain : float
        Output gain compensation (0-100, default 50 = unity)
    
    Returns
    -------
    np.ndarray
        Processed audio buffer
    """
    # Scale abstract params
    blend_scaled = scale_wet(blend)  # 0-100 -> 0-1
    gain_mult = scale_to_range(clamp_param(gain), 0.5, 4.0)
    
    # Split into low and high bands (crossover stays Hz)
    low = _apply_simple_filter(buffer, crossover, 'lp')
    high = buffer - low  # High is remainder
    
    # Process each band with their own drive settings (vamp_process now uses 1-100)
    low_proc = vamp_process(low, drive=drive_low, waveshape=shape_low, mix=100)
    high_proc = vamp_process(high, drive=drive_high, waveshape=shape_high, mix=100)
    
    # Combine bands
    out = low_proc + high_proc
    
    # Apply blend (crossfade between focusing on low vs high)
    if blend_scaled != 0.5:
        low_weight = 1.0 - blend_scaled
        high_weight = blend_scaled
        out = low_proc * low_weight + high_proc * high_weight
        # Normalize to maintain volume
        out = out * 2.0
    
    # Gain makeup
    out = out * gain_mult
    
    # Normalize
    max_val = np.max(np.abs(out))
    if max_val > 1.0:
        out = out / max_val
    
    return out.astype(np.float64)


# Effect wrappers for the effect chain system
# All use 1-100 scaling for abstract parameters

def _vamp_light(buffer: np.ndarray) -> np.ndarray:
    """Light amp warmth."""
    return vamp_process(buffer, drive=10, waveshape='tube', post_filter=8000, gain=50, mix=100)

def _vamp_medium(buffer: np.ndarray) -> np.ndarray:
    """Medium amp overdrive."""
    return vamp_process(buffer, drive=25, waveshape='tube', post_filter=6000, gain=50, mix=100)

def _vamp_heavy(buffer: np.ndarray) -> np.ndarray:
    """Heavy amp distortion."""
    return vamp_process(buffer, drive=50, waveshape='tube', pre_filter=100, pre_filter_type='hp', post_filter=4000, gain=50, mix=100)

def _vamp_fuzz(buffer: np.ndarray) -> np.ndarray:
    """Fuzz pedal style distortion."""
    return vamp_process(buffer, drive=75, waveshape='fuzz', post_filter=3000, gain=40, mix=100)

def _overdrive_soft(buffer: np.ndarray) -> np.ndarray:
    """Soft overdrive."""
    return vamp_process(buffer, drive=15, waveshape='cubic', post_filter=10000, gain=50, mix=100)

def _overdrive_classic(buffer: np.ndarray) -> np.ndarray:
    """Classic overdrive tone."""
    return vamp_process(buffer, drive=30, waveshape='tanh', pre_filter=150, pre_filter_type='hp', post_filter=5000, gain=50, mix=100)

def _overdrive_crunch(buffer: np.ndarray) -> np.ndarray:
    """Crunchy overdrive."""
    return vamp_process(buffer, drive=40, waveshape='tube', post_filter=4000, gain=50, mix=100)

def _dual_od_warm(buffer: np.ndarray) -> np.ndarray:
    """Warm dual-stage overdrive."""
    return dual_overdrive(buffer, drive_low=15, drive_high=20, shape_low='tube', shape_high='cubic', crossover=600)

def _dual_od_bright(buffer: np.ndarray) -> np.ndarray:
    """Bright dual-stage overdrive."""
    return dual_overdrive(buffer, drive_low=10, drive_high=30, shape_low='cubic', shape_high='tanh', crossover=1200)

def _dual_od_heavy(buffer: np.ndarray) -> np.ndarray:
    """Heavy dual-stage overdrive."""
    return dual_overdrive(buffer, drive_low=30, drive_high=50, shape_low='tube', shape_high='fuzz', crossover=400)

def _waveshape_fold(buffer: np.ndarray) -> np.ndarray:
    """Wave folding distortion."""
    return vamp_process(buffer, drive=20, waveshape='fold', gain=60, mix=100)

def _waveshape_rectify(buffer: np.ndarray) -> np.ndarray:
    """Rectifier distortion."""
    return vamp_process(buffer, drive=10, waveshape='rectify', post_filter=6000, mix=100)

def _waveshape_sine(buffer: np.ndarray) -> np.ndarray:
    """Sine waveshaping for harmonics."""
    return vamp_process(buffer, drive=15, waveshape='sine', post_filter=8000, mix=100)



# ---------------------------------------------------------------------------
# Dynamics effects (compression/expansion/limiting)
# ---------------------------------------------------------------------------

def compress(
    buffer: np.ndarray,
    threshold: float = 50.0,
    ratio: float = 50.0,
    makeup: float = 50.0,
) -> np.ndarray:
    """Parameterized compressor with unified 1-100 scaling.
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio buffer
    threshold : float
        Compression threshold (0-100, default 50)
        0 = compress everything (-60dB)
        50 = -30dB threshold
        100 = 0dB (no compression)
    ratio : float
        Compression ratio (0-100, default 50)
        0 = 1:1 (no compression)
        50 = 4:1
        100 = inf:1 (limiting)
    makeup : float
        Makeup gain (0-100, default 50)
        0 = -12dB, 50 = 0dB (unity), 100 = +12dB
    
    Returns
    -------
    np.ndarray
        Compressed audio buffer
    """
    # Scale threshold: 0-100 -> 0.001-1.0 (linear)
    thresh_linear = scale_to_range(clamp_param(threshold), 0.001, 1.0)
    
    # Scale ratio: 0-100 -> 1.0-20.0
    ratio_scaled = scale_ratio(ratio)
    
    # Scale makeup: 0-100 -> 0.25-4.0
    makeup_mult = scale_to_range(clamp_param(makeup), 0.25, 4.0)
    
    out = buffer.copy().astype(np.float64)
    
    # Apply compression
    mask = np.abs(out) > thresh_linear
    if ratio_scaled >= 20.0:
        # Limiting mode
        out[mask] = np.sign(out[mask]) * thresh_linear
    else:
        out[mask] = np.sign(out[mask]) * (thresh_linear + (np.abs(out[mask]) - thresh_linear) / ratio_scaled)
    
    # Apply makeup gain
    out = out * makeup_mult
    
    return _normalise(out)


def gate(
    buffer: np.ndarray,
    threshold: float = 50.0,
    attack: float = 1.0,
    release: float = 10.0,
) -> np.ndarray:
    """Parameterized noise gate with unified 1-100 scaling for threshold.
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio buffer
    threshold : float
        Gate threshold (0-100, default 50)
        0 = gate at -60dB (gate almost nothing)
        50 = gate at -30dB
        100 = gate at 0dB (gate everything below 0dB)
    attack : float
        Attack time in milliseconds (real units, default 1ms)
    release : float
        Release time in milliseconds (real units, default 10ms)
    
    Returns
    -------
    np.ndarray
        Gated audio buffer
    """
    # Scale threshold: 0-100 -> 0.001-1.0
    thresh_linear = scale_to_range(clamp_param(threshold), 0.001, 1.0)
    
    out = buffer.copy().astype(np.float64)
    
    # Simple gate (attack/release smoothing would be added in full implementation)
    out[np.abs(out) < thresh_linear] = 0.0
    
    return out.astype(np.float64)


# Preset compression effects
def _compress_mild(buffer: np.ndarray) -> np.ndarray:
    """Mild compression with medium threshold and ratio."""
    return compress(buffer, threshold=50, ratio=40, makeup=50)


def _compress_hard(buffer: np.ndarray) -> np.ndarray:
    """Hard compression with lower threshold and higher ratio."""
    return compress(buffer, threshold=30, ratio=70, makeup=55)


def _compress_limiter(buffer: np.ndarray) -> np.ndarray:
    """Limiter: absolute clip at threshold to prevent clipping."""
    return compress(buffer, threshold=70, ratio=100, makeup=50)


def _compress_expander(buffer: np.ndarray) -> np.ndarray:
    """Expander: make quiet parts quieter, leaving loud parts unchanged."""
    threshold = 0.2
    ratio = 2.0  # expand below threshold
    out = buffer.copy().astype(np.float64)
    mask = np.abs(out) < threshold
    out[mask] = out[mask] / ratio
    return _normalise(out)


def _compress_softclipper(buffer: np.ndarray) -> np.ndarray:
    """Soft clipper: saturate only above a threshold."""
    threshold = 0.5
    out = buffer.copy().astype(np.float64)
    mask_pos = out > threshold
    mask_neg = out < -threshold
    # For values above threshold apply tanh, leaving lower region untouched
    out[mask_pos] = threshold + np.tanh(out[mask_pos] - threshold)
    out[mask_neg] = -threshold + np.tanh(out[mask_neg] + threshold)
    return _normalise(out)


# ---------------------------------------------------------------------------
# Forever Compression - Multiband OTT-style Compressor
# Module name: forever_compression (Section C2)
# ---------------------------------------------------------------------------

def _ott_band_compress(
    buffer: np.ndarray,
    down_thresh: float,
    down_ratio: float,
    up_thresh: float,
    up_ratio: float,
    makeup: float = 1.0,
) -> np.ndarray:
    """Apply OTT-style upward and downward compression to a single band.
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio buffer
    down_thresh : float
        Downward compression threshold (linear, 0-1)
    down_ratio : float
        Downward compression ratio (1.0 = no compression, higher = more)
    up_thresh : float
        Upward compression threshold (linear, 0-1)
    up_ratio : float
        Upward expansion ratio (1.0 = no expansion, higher = more boost)
    makeup : float
        Output gain multiplier
    
    Returns
    -------
    np.ndarray
        Compressed audio buffer
    """
    out = buffer.copy().astype(np.float64)
    abs_out = np.abs(out)
    
    # Downward compression: reduce levels above down_thresh
    down_mask = abs_out > down_thresh
    if np.any(down_mask):
        excess = abs_out[down_mask] - down_thresh
        compressed_excess = excess / down_ratio
        out[down_mask] = np.sign(out[down_mask]) * (down_thresh + compressed_excess)
    
    # Upward compression: boost levels below up_thresh
    abs_out = np.abs(out)  # Recalculate after downward
    up_mask = (abs_out < up_thresh) & (abs_out > 0.001)
    if np.any(up_mask):
        # How far below threshold (0 to up_thresh)
        deficit = up_thresh - abs_out[up_mask]
        # Reduce the deficit (bring closer to threshold)
        boosted_deficit = deficit / up_ratio
        new_level = up_thresh - boosted_deficit
        out[up_mask] = np.sign(out[up_mask]) * new_level
    
    return out * makeup


def forever_compression(
    buffer: np.ndarray,
    depth: float = 50.0,
    low_xover: float = 120.0,
    high_xover: float = 2500.0,
    low_amount: float = 50.0,
    mid_amount: float = 50.0,
    high_amount: float = 50.0,
    upward: float = 50.0,
    downward: float = 50.0,
    mix: float = 100.0,
    output: float = 50.0,
) -> np.ndarray:
    """Forever Compression - Multiband OTT-style compressor.
    
    Section C2 of MDMA Master Feature List.
    Module name: forever_compression
    
    OTT-style compression with both upward and downward compression
    across three frequency bands.
    
    Parameters (all abstract use unified 1-100 scaling)
    ----------
    buffer : np.ndarray
        Input audio buffer
    depth : float
        Overall compression depth (0-100, default 50)
        0 = gentle, 50 = moderate, 100 = extreme
    low_xover : float
        Low/mid crossover frequency in Hz (real units, default 120)
    high_xover : float
        Mid/high crossover frequency in Hz (real units, default 2500)
    low_amount : float
        Compression amount for low band (0-100, default 50)
    mid_amount : float
        Compression amount for mid band (0-100, default 50)
    high_amount : float
        Compression amount for high band (0-100, default 50)
    upward : float
        Upward compression strength (0-100, default 50)
        Higher = more boost of quiet signals
    downward : float
        Downward compression strength (0-100, default 50)
        Higher = more reduction of loud signals
    mix : float
        Wet/dry mix (0-100, default 100)
    output : float
        Output trim (0-100, 50 = unity, default 50)
    
    Returns
    -------
    np.ndarray
        Processed audio buffer
    """
    # Scale parameters
    depth_scaled = scale_wet(depth)  # 0-1
    mix_scaled = scale_wet(mix)
    low_amt = scale_wet(low_amount) * depth_scaled
    mid_amt = scale_wet(mid_amount) * depth_scaled
    high_amt = scale_wet(high_amount) * depth_scaled
    up_strength = scale_wet(upward)
    down_strength = scale_wet(downward)
    output_mult = scale_to_range(clamp_param(output), 0.25, 4.0)
    
    # Calculate compression parameters based on amounts
    # Higher amount = lower threshold, higher ratio
    def calc_params(amount: float, up_str: float, down_str: float):
        # Downward: threshold 0.7 -> 0.2 as amount increases
        down_thresh = 0.7 - (amount * 0.5)
        # Downward ratio: 1.5 -> 8.0 as amount increases
        down_ratio = 1.5 + (amount * down_str * 6.5)
        # Upward: threshold 0.1 -> 0.4 as amount increases  
        up_thresh = 0.1 + (amount * 0.3)
        # Upward ratio: 1.2 -> 4.0 as amount increases
        up_ratio = 1.2 + (amount * up_str * 2.8)
        return down_thresh, down_ratio, up_thresh, up_ratio
    
    # Split into three bands
    low = _apply_simple_filter(buffer, low_xover, 'lp')
    high = _apply_simple_filter(buffer, high_xover, 'hp')
    mid = buffer - low - high
    
    # Compress each band
    low_params = calc_params(low_amt, up_strength, down_strength)
    mid_params = calc_params(mid_amt, up_strength, down_strength)
    high_params = calc_params(high_amt, up_strength, down_strength)
    
    low_comp = _ott_band_compress(low, *low_params, makeup=1.0 + low_amt * 0.5)
    mid_comp = _ott_band_compress(mid, *mid_params, makeup=1.0 + mid_amt * 0.3)
    high_comp = _ott_band_compress(high, *high_params, makeup=1.0 + high_amt * 0.4)
    
    # Recombine
    wet = low_comp + mid_comp + high_comp
    
    # Apply output gain
    wet = wet * output_mult
    
    # Mix
    if mix_scaled < 1.0:
        out = buffer * (1.0 - mix_scaled) + wet * mix_scaled
    else:
        out = wet
    
    return _normalise(out)


# Forever Compression presets (Section C2 macros)
def _fc_punch(buffer: np.ndarray) -> np.ndarray:
    """Forever Compression: Punch preset.
    
    Emphasizes transients with fast attack recovery.
    Good for drums and percussive elements.
    """
    return forever_compression(
        buffer, 
        depth=60, 
        low_amount=70, mid_amount=50, high_amount=40,
        upward=30, downward=70,
        mix=75
    )


def _fc_glue(buffer: np.ndarray) -> np.ndarray:
    """Forever Compression: Glue preset.
    
    Gentle multiband compression for cohesion.
    Good for bus/group processing.
    """
    return forever_compression(
        buffer,
        depth=35,
        low_amount=45, mid_amount=50, high_amount=45,
        upward=40, downward=40,
        mix=60
    )


def _fc_loud(buffer: np.ndarray) -> np.ndarray:
    """Forever Compression: Loud preset.
    
    Aggressive OTT-style maximizer.
    Classic EDM "in your face" sound.
    """
    return forever_compression(
        buffer,
        depth=85,
        low_amount=80, mid_amount=90, high_amount=85,
        upward=70, downward=80,
        mix=100,
        output=60
    )


def _fc_soft(buffer: np.ndarray) -> np.ndarray:
    """Forever Compression: Soft preset.
    
    Subtle multiband dynamics control.
    Good for vocals, acoustic sources.
    """
    return forever_compression(
        buffer,
        depth=25,
        low_amount=30, mid_amount=35, high_amount=25,
        upward=50, downward=30,
        mix=50
    )


def _fc_ott(buffer: np.ndarray) -> np.ndarray:
    """Forever Compression: OTT preset.
    
    Classic OTT multiband compression sound.
    Heavy upward + downward compression.
    """
    return forever_compression(
        buffer,
        depth=100,
        low_amount=100, mid_amount=100, high_amount=100,
        upward=100, downward=100,
        mix=50
    )


# ---------------------------------------------------------------------------
# Gate effects
# ---------------------------------------------------------------------------

# Preset gate effects (use parameterized gate function)
def _gate1(buffer: np.ndarray) -> np.ndarray:
    """Noise gate with a relatively high threshold (30%)."""
    return gate(buffer, threshold=70)


def _gate2(buffer: np.ndarray) -> np.ndarray:
    """Noise gate with medium-high threshold (20%)."""
    return gate(buffer, threshold=60)


def _gate3(buffer: np.ndarray) -> np.ndarray:
    """Noise gate with medium threshold (10%)."""
    return gate(buffer, threshold=50)


def _gate4(buffer: np.ndarray) -> np.ndarray:
    """Noise gate with low threshold (5%)."""
    return gate(buffer, threshold=40)


def _gate5(buffer: np.ndarray) -> np.ndarray:
    """Noise gate with very low threshold (1%)."""
    return gate(buffer, threshold=25)


# ---------------------------------------------------------------------------
# Giga Gate - Pattern-based Gating and Stutter Engine
# Module name: giga_gate (Section C2)
# ---------------------------------------------------------------------------

def _parse_gate_pattern(pattern: str, steps: int = 16) -> np.ndarray:
    """Parse a gate pattern string into an array of gate states.
    
    Parameters
    ----------
    pattern : str
        Pattern string. Can be:
        - Binary: "1010101010101010" (1=on, 0=off)
        - Shorthand: "x.x.x.x." (x=on, .=off)
        - Named: "four_floor", "offbeat", "tresillo", etc.
    steps : int
        Number of steps to generate (default 16)
    
    Returns
    -------
    np.ndarray
        Array of gate states (0.0 or 1.0) with length `steps`
    """
    # Named patterns
    named_patterns = {
        'four_floor': '1000100010001000',
        'offbeat': '0010001000100010',
        'all': '1111111111111111',
        'half': '1010101010101010',
        'quarter': '1000100010001000',
        'eighth': '1100110011001100',
        'triplet': '100100100100100100100100',
        'tresillo': '1001001010010010',
        'son_clave': '1001001000101000',
        'rumba_clave': '1001000010101000',
        'bossa': '1001001000100100',
        'shuffle': '1010001010001010',
        'stutter2': '1100000000000000',
        'stutter4': '1111000000000000',
        'stutter8': '1111111100000000',
        'glitch1': '1010110010100110',
        'glitch2': '1001101011001010',
        'sparse': '1000000010000000',
        'dense': '1110111011101110',
    }
    
    pattern_lower = pattern.lower().strip()
    
    # Check for named pattern
    if pattern_lower in named_patterns:
        pattern = named_patterns[pattern_lower]
    
    # Convert shorthand notation
    pattern = pattern.replace('x', '1').replace('X', '1')
    pattern = pattern.replace('.', '0').replace('-', '0').replace(' ', '')
    
    # Parse binary pattern
    try:
        gate_states = np.array([float(c) for c in pattern if c in '01'], dtype=np.float64)
    except (ValueError, TypeError):
        gate_states = np.ones(steps, dtype=np.float64)
    
    # Resize to target steps
    if len(gate_states) == 0:
        gate_states = np.ones(steps, dtype=np.float64)
    elif len(gate_states) != steps:
        # Tile or truncate to match steps
        full_cycles = steps // len(gate_states)
        remainder = steps % len(gate_states)
        if full_cycles > 0:
            gate_states = np.tile(gate_states, full_cycles)
        if remainder > 0:
            gate_states = np.concatenate([gate_states, gate_states[:remainder]])
        gate_states = gate_states[:steps]
    
    return gate_states


def _apply_gate_shape(
    buffer: np.ndarray,
    gate_states: np.ndarray,
    shape: str = 'square',
    attack_ms: float = 1.0,
    release_ms: float = 5.0,
    sample_rate: int = 48000,
) -> np.ndarray:
    """Apply shaped gating to buffer based on gate states.
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio buffer
    gate_states : np.ndarray
        Array of gate states (0.0 or 1.0), one per step
    shape : str
        Gate shape: 'square', 'saw', 'ramp', 'triangle', 'sine', 'exp'
    attack_ms : float
        Attack time in milliseconds
    release_ms : float
        Release time in milliseconds
    sample_rate : int
        Sample rate in Hz
    
    Returns
    -------
    np.ndarray
        Gated audio buffer
    """
    n_samples = len(buffer)
    n_steps = len(gate_states)
    samples_per_step = n_samples // n_steps
    
    if samples_per_step < 1:
        return buffer.copy()
    
    # Build amplitude envelope
    envelope = np.zeros(n_samples, dtype=np.float64)
    
    attack_samples = max(1, int(attack_ms * sample_rate / 1000))
    release_samples = max(1, int(release_ms * sample_rate / 1000))
    
    for step_idx, gate_val in enumerate(gate_states):
        start = step_idx * samples_per_step
        end = min(start + samples_per_step, n_samples)
        step_len = end - start
        
        if gate_val > 0:
            # Gate is on - create shaped envelope
            step_env = np.ones(step_len, dtype=np.float64)
            
            if shape == 'square':
                pass  # Already ones
            elif shape == 'saw':
                step_env = np.linspace(1.0, 0.0, step_len)
            elif shape == 'ramp':
                step_env = np.linspace(0.0, 1.0, step_len)
            elif shape == 'triangle':
                half = step_len // 2
                step_env[:half] = np.linspace(0.0, 1.0, half)
                step_env[half:] = np.linspace(1.0, 0.0, step_len - half)
            elif shape == 'sine':
                step_env = np.sin(np.linspace(0, np.pi, step_len))
            elif shape == 'exp':
                step_env = np.exp(-3.0 * np.linspace(0, 1, step_len))
            
            # Apply attack
            if attack_samples < step_len:
                attack_curve = np.linspace(0, 1, attack_samples)
                step_env[:attack_samples] *= attack_curve
            
            # Apply release
            if release_samples < step_len:
                release_curve = np.linspace(1, 0, release_samples)
                step_env[-release_samples:] *= release_curve
            
            envelope[start:end] = step_env * gate_val
        # else: gate is off, envelope stays at 0
    
    return buffer * envelope


def giga_gate(
    buffer: np.ndarray,
    pattern: str = 'half',
    steps: int = 16,
    shape: str = 'square',
    attack: float = 1.0,
    release: float = 5.0,
    mix: float = 100.0,
    sample_rate: int = 48000,
) -> np.ndarray:
    """Giga Gate - Pattern-based gating engine.
    
    Section C2 of MDMA Master Feature List.
    Module name: giga_gate
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio buffer
    pattern : str
        Gate pattern (binary, shorthand, or named preset)
    steps : int
        Number of steps in pattern (default 16)
    shape : str
        Gate shape: 'square', 'saw', 'ramp', 'triangle', 'sine', 'exp'
    attack : float
        Attack time in milliseconds
    release : float
        Release time in milliseconds
    mix : float
        Wet/dry mix (0-100)
    sample_rate : int
        Sample rate in Hz
    
    Returns
    -------
    np.ndarray
        Gated audio buffer
    """
    gate_states = _parse_gate_pattern(pattern, steps)
    
    wet = _apply_gate_shape(
        buffer, gate_states, shape, attack, release, sample_rate
    )
    
    mix_scaled = scale_wet(mix)
    if mix_scaled < 1.0:
        return buffer * (1.0 - mix_scaled) + wet * mix_scaled
    return wet


def giga_stutter(
    buffer: np.ndarray,
    repeats: int = 4,
    decay: float = 0.9,
    pitch_shift: float = 0.0,
    reverse: bool = False,
    mix: float = 100.0,
    sample_rate: int = 48000,
) -> np.ndarray:
    """Giga Gate stutter effect.
    
    Repeats a slice of audio multiple times with optional decay and pitch.
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio buffer
    repeats : int
        Number of repeats (2-32)
    decay : float
        Amplitude decay per repeat (0.0-1.0)
    pitch_shift : float
        Pitch shift per repeat in semitones
    reverse : bool
        Reverse alternate repeats
    mix : float
        Wet/dry mix (0-100)
    sample_rate : int
        Sample rate in Hz
    
    Returns
    -------
    np.ndarray
        Stuttered audio buffer
    """
    repeats = max(2, min(32, repeats))
    decay = max(0.0, min(1.0, decay))
    
    n_samples = len(buffer)
    slice_len = n_samples // repeats
    
    if slice_len < 10:
        return buffer.copy()
    
    # Get first slice
    base_slice = buffer[:slice_len].copy()
    
    # Build output
    out = np.zeros(n_samples, dtype=np.float64)
    
    for i in range(repeats):
        start = i * slice_len
        end = min(start + slice_len, n_samples)
        actual_len = end - start
        
        # Get slice (with decay)
        amp = decay ** i
        current_slice = base_slice[:actual_len].copy() * amp
        
        # Reverse alternate if requested
        if reverse and i % 2 == 1:
            current_slice = current_slice[::-1]
        
        # Pitch shift (simple resampling)
        if pitch_shift != 0:
            ratio = 2.0 ** (pitch_shift * i / 12.0)
            new_len = int(len(current_slice) / ratio)
            if new_len > 0 and new_len != len(current_slice):
                x_old = np.linspace(0, 1, len(current_slice))
                x_new = np.linspace(0, 1, min(new_len, actual_len))
                current_slice = np.interp(x_new, x_old, current_slice)
                # Pad or truncate
                if len(current_slice) < actual_len:
                    current_slice = np.pad(current_slice, (0, actual_len - len(current_slice)))
                else:
                    current_slice = current_slice[:actual_len]
        
        out[start:end] = current_slice[:actual_len]
    
    mix_scaled = scale_wet(mix)
    if mix_scaled < 1.0:
        return buffer * (1.0 - mix_scaled) + out * mix_scaled
    return out


def giga_halftime(
    buffer: np.ndarray,
    mix: float = 100.0,
) -> np.ndarray:
    """Giga Gate halftime effect.
    
    Stretches audio to half speed (one octave down).
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio buffer
    mix : float
        Wet/dry mix (0-100)
    
    Returns
    -------
    np.ndarray
        Halftime audio buffer
    """
    # Simple linear interpolation for 2x stretch
    n_samples = len(buffer)
    x_old = np.linspace(0, 1, n_samples)
    x_new = np.linspace(0, 0.5, n_samples)  # Only read first half
    
    wet = np.interp(x_new, x_old, buffer).astype(np.float64)
    
    mix_scaled = scale_wet(mix)
    if mix_scaled < 1.0:
        return buffer * (1.0 - mix_scaled) + wet * mix_scaled
    return wet


def giga_doubletime(
    buffer: np.ndarray,
    mix: float = 100.0,
) -> np.ndarray:
    """Giga Gate doubletime effect.
    
    Compresses audio to double speed (one octave up), repeated.
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio buffer
    mix : float
        Wet/dry mix (0-100)
    
    Returns
    -------
    np.ndarray
        Doubletime audio buffer
    """
    # Compress to half length, then repeat
    n_samples = len(buffer)
    half_len = n_samples // 2
    
    # Resample to half length
    x_old = np.linspace(0, 1, n_samples)
    x_new = np.linspace(0, 1, half_len)
    compressed = np.interp(x_new, x_old, buffer)
    
    # Repeat to fill
    wet = np.tile(compressed, 2)[:n_samples].astype(np.float64)
    
    mix_scaled = scale_wet(mix)
    if mix_scaled < 1.0:
        return buffer * (1.0 - mix_scaled) + wet * mix_scaled
    return wet


def giga_tape_stop(
    buffer: np.ndarray,
    duration: float = 0.5,
    curve: str = 'exp',
    mix: float = 100.0,
    sample_rate: int = 48000,
) -> np.ndarray:
    """Giga Gate tape stop effect.
    
    Simulates a tape machine slowing to a stop.
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio buffer
    duration : float
        Stop duration in seconds (0.1-2.0)
    curve : str
        Slowdown curve: 'linear', 'exp', 'log'
    mix : float
        Wet/dry mix (0-100)
    sample_rate : int
        Sample rate in Hz
    
    Returns
    -------
    np.ndarray
        Tape-stopped audio buffer
    """
    duration = max(0.1, min(2.0, duration))
    n_samples = len(buffer)
    stop_samples = int(duration * sample_rate)
    stop_samples = min(stop_samples, n_samples)
    
    # Generate speed curve (1.0 -> 0.0)
    t = np.linspace(0, 1, stop_samples)
    if curve == 'exp':
        speed_curve = np.exp(-4 * t)
    elif curve == 'log':
        speed_curve = 1.0 - np.log1p(t * (np.e - 1))
    else:  # linear
        speed_curve = 1.0 - t
    
    # Variable-rate resampling via phase accumulation
    out = np.zeros(n_samples, dtype=np.float64)
    
    # First part: tape stop
    phase = 0.0
    for i in range(stop_samples):
        if phase < n_samples - 1:
            # Linear interpolation
            idx = int(phase)
            frac = phase - idx
            if idx < n_samples - 1:
                out[i] = buffer[idx] * (1 - frac) + buffer[idx + 1] * frac
            else:
                out[i] = buffer[idx]
        phase += speed_curve[i]
    
    # Remaining part: silence (tape stopped)
    # Already zeros
    
    mix_scaled = scale_wet(mix)
    if mix_scaled < 1.0:
        return buffer * (1.0 - mix_scaled) + out * mix_scaled
    return out


def giga_tape_bend(
    buffer: np.ndarray,
    amount: float = 50.0,
    rate: float = 2.0,
    mix: float = 100.0,
    sample_rate: int = 48000,
) -> np.ndarray:
    """Giga Gate tape bend/wobble effect.
    
    Simulates tape speed fluctuation (wow and flutter).
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio buffer
    amount : float
        Bend amount (0-100, higher = more pitch variation)
    rate : float
        Wobble rate in Hz
    mix : float
        Wet/dry mix (0-100)
    sample_rate : int
        Sample rate in Hz
    
    Returns
    -------
    np.ndarray
        Tape-bent audio buffer
    """
    amount_scaled = scale_wet(amount) * 0.1  # Max ±10% speed variation
    
    n_samples = len(buffer)
    t = np.arange(n_samples) / sample_rate
    
    # Generate wobble LFO
    wobble = np.sin(2 * np.pi * rate * t) * amount_scaled
    
    # Variable-rate resampling
    speed = 1.0 + wobble
    
    # Phase accumulation
    phase = np.zeros(n_samples, dtype=np.float64)
    phase[0] = 0.0
    for i in range(1, n_samples):
        phase[i] = phase[i-1] + speed[i-1]
    
    # Wrap phase to buffer length
    phase = np.clip(phase, 0, n_samples - 1)
    
    # Interpolate
    idx = phase.astype(int)
    frac = phase - idx
    idx = np.clip(idx, 0, n_samples - 2)
    
    wet = buffer[idx] * (1 - frac) + buffer[idx + 1] * frac
    
    mix_scaled = scale_wet(mix)
    if mix_scaled < 1.0:
        return buffer * (1.0 - mix_scaled) + wet * mix_scaled
    return wet.astype(np.float64)


# Giga Gate presets
def _gg_half(buffer: np.ndarray) -> np.ndarray:
    """Giga Gate: Half pattern (alternating on/off)."""
    return giga_gate(buffer, pattern='half', shape='square')


def _gg_quarter(buffer: np.ndarray) -> np.ndarray:
    """Giga Gate: Quarter pattern (every 4th step)."""
    return giga_gate(buffer, pattern='quarter', shape='square')


def _gg_tresillo(buffer: np.ndarray) -> np.ndarray:
    """Giga Gate: Tresillo rhythm pattern."""
    return giga_gate(buffer, pattern='tresillo', shape='sine')


def _gg_glitch(buffer: np.ndarray) -> np.ndarray:
    """Giga Gate: Glitch pattern."""
    return giga_gate(buffer, pattern='glitch1', shape='exp')


def _gg_stutter(buffer: np.ndarray) -> np.ndarray:
    """Giga Gate: 4x stutter effect."""
    return giga_stutter(buffer, repeats=4, decay=0.85)


def _gg_halftime(buffer: np.ndarray) -> np.ndarray:
    """Giga Gate: Halftime effect."""
    return giga_halftime(buffer)


def _gg_tape_stop(buffer: np.ndarray) -> np.ndarray:
    """Giga Gate: Tape stop effect."""
    return giga_tape_stop(buffer, duration=0.5)


# ---------------------------------------------------------------------------
# Lo‑fi effects
# ---------------------------------------------------------------------------

def bitcrush(
    buffer: np.ndarray,
    bits: float = 50.0,
    downsample: float = 50.0,
    mix: float = 100.0,
) -> np.ndarray:
    """Parameterized bitcrusher with unified 1-100 scaling.
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio buffer
    bits : float
        Bit depth reduction (0-100, default 50)
        0 = 1 bit (extreme)
        50 = 8 bits (classic lo-fi)
        100 = 16 bits (clean)
    downsample : float
        Sample rate reduction (0-100, default 50)
        0 = no downsampling
        50 = 4x downsample (classic)
        100 = 16x downsample (extreme)
    mix : float
        Wet/dry mix (0-100, default 100 = fully wet)
    
    Returns
    -------
    np.ndarray
        Bitcrushed audio buffer
    """
    # Scale bits: 0-100 -> 1-16
    bit_depth = scale_bits(bits)
    
    # Ensure minimum bit depth of 2 to avoid division by zero
    bit_depth = max(2, int(bit_depth))
    
    # Scale downsample: 0-100 -> 1-16 factor
    downsample_factor = max(1, int(scale_to_range(clamp_param(downsample), 1, 16)))
    
    # Scale mix
    mix_scaled = scale_wet(mix)
    
    dry = buffer.copy()
    
    # Downsample
    reduced = buffer[::downsample_factor]
    
    # Bit reduction
    max_int = 2 ** (bit_depth - 1) - 1
    if max_int < 1:
        max_int = 1
    
    quantised = np.round(reduced * max_int) / max_int
    
    # Upsample back
    out = np.repeat(quantised, downsample_factor)[:len(buffer)]
    
    # Mix
    if mix_scaled < 1.0:
        out = dry * (1 - mix_scaled) + out * mix_scaled
    
    return _normalise(out)


def _lofi_bitcrush(buffer: np.ndarray) -> np.ndarray:
    """Bitcrusher: reduce sample rate and bit depth (classic lo-fi)."""
    return bitcrush(buffer, bits=50, downsample=50, mix=100)


def _lofi_chorus(buffer: np.ndarray) -> np.ndarray:
    """Simple chorus effect using a modulated delay line."""
    mod_rate = 1.5  # Hz
    mod_depth = 0.003  # seconds (~3 ms)
    delay_base = 0.015  # base delay of 15 ms
    n = len(buffer)
    t = np.arange(n) / SAMPLE_RATE
    out = np.zeros_like(buffer, dtype=np.float64)
    # Create modulated delay offsets in samples
    mod = mod_depth * np.sin(2.0 * math.pi * mod_rate * t)
    delay_samples = (delay_base + mod) * SAMPLE_RATE
    for i in range(n):
        d = int(delay_samples[i])
        if i - d >= 0:
            out[i] = buffer[i] + 0.5 * buffer[i - d]
        else:
            out[i] = buffer[i]
    return _normalise(out)


def _lofi_flanger(buffer: np.ndarray) -> np.ndarray:
    """Flanger effect using a shorter, faster modulated delay."""
    mod_rate = 0.5  # Hz
    mod_depth = 0.002  # seconds (~2 ms)
    delay_base = 0.005  # 5 ms
    n = len(buffer)
    t = np.arange(n) / SAMPLE_RATE
    out = np.zeros_like(buffer, dtype=np.float64)
    mod = mod_depth * np.sin(2.0 * math.pi * mod_rate * t)
    delay_samples = (delay_base + mod) * SAMPLE_RATE
    for i in range(n):
        d = int(delay_samples[i])
        if i - d >= 0:
            out[i] = buffer[i] + 0.7 * buffer[i - d]
        else:
            out[i] = buffer[i]
    return _normalise(out)


def _lofi_phaser(buffer: np.ndarray) -> np.ndarray:
    """Basic phaser effect using an all‑pass filter cascade."""
    # Phaser parameters
    rate = 0.4  # Hz
    depth = 0.5
    stages = 4
    n = len(buffer)
    t = np.arange(n) / SAMPLE_RATE
    out = buffer.copy().astype(np.float64)
    # Low frequency oscillator for sweeping
    lfo = (1.0 + depth * np.sin(2.0 * math.pi * rate * t)) * 0.5
    # Apply a simple series of all‑pass filters whose cutoffs move with the LFO
    for stage in range(stages):
        y = np.zeros_like(out)
        xh = 0.0  # past input sample
        yh = 0.0  # past output sample
        for i in range(n):
            f_c = 500.0 + 1500.0 * lfo[i]
            omega = 2.0 * math.pi * f_c / SAMPLE_RATE
            # Use stable all-pass coefficient formula
            tan_half = math.tan(omega / 2.0)
            alpha = (tan_half - 1.0) / (tan_half + 1.0)
            # Clamp alpha to prevent instability
            alpha = max(-0.99, min(0.99, alpha))
            # All‑pass filter difference equation
            y[i] = alpha * out[i] + xh - alpha * yh
            xh = out[i]
            yh = y[i]
        out = y
    return _normalise(out)


def _lofi_filter(buffer: np.ndarray) -> np.ndarray:
    """Lo‑fi filter: a gentle low‑pass to simulate vintage equipment."""
    # Cutoff around 2 kHz
    cutoff_norm = 2000.0 / (0.5 * SAMPLE_RATE)
    import scipy.signal
    b, a = scipy.signal.butter(2, cutoff_norm, btype='low')
    out = scipy.signal.lfilter(b, a, buffer)
    return _normalise(out)


def _lofi_halftime(buffer: np.ndarray) -> np.ndarray:
    """Halftime stretching: slows down playback by a factor of 2."""
    # Downsample by factor 2 then linear interpolate back to original length
    half = buffer[::2]
    # Simple linear interpolation to original length
    idx = np.linspace(0.0, len(half) - 1, len(buffer))
    out = np.interp(idx, np.arange(len(half)), half)
    return _normalise(out)


# ---------------------------------------------------------------------------
# STANDALONE FILTER EFFECTS
# ---------------------------------------------------------------------------

def _filter_lowpass(buffer: np.ndarray) -> np.ndarray:
    """Lowpass filter at 2kHz with moderate resonance."""
    import scipy.signal
    nyq = SAMPLE_RATE / 2
    cutoff = min(0.99, 2000 / nyq)
    b, a = scipy.signal.butter(4, cutoff, btype='low')
    out = scipy.signal.lfilter(b, a, buffer)
    return _normalise(out)


def _filter_lowpass_soft(buffer: np.ndarray) -> np.ndarray:
    """Gentle lowpass at 4kHz."""
    import scipy.signal
    nyq = SAMPLE_RATE / 2
    cutoff = min(0.99, 4000 / nyq)
    b, a = scipy.signal.butter(2, cutoff, btype='low')
    out = scipy.signal.lfilter(b, a, buffer)
    return _normalise(out)


def _filter_lowpass_hard(buffer: np.ndarray) -> np.ndarray:
    """Aggressive lowpass at 800Hz."""
    import scipy.signal
    nyq = SAMPLE_RATE / 2
    cutoff = min(0.99, 800 / nyq)
    b, a = scipy.signal.butter(6, cutoff, btype='low')
    out = scipy.signal.lfilter(b, a, buffer)
    return _normalise(out)


def _filter_highpass(buffer: np.ndarray) -> np.ndarray:
    """Highpass filter at 300Hz."""
    import scipy.signal
    nyq = SAMPLE_RATE / 2
    cutoff = max(0.01, 300 / nyq)
    b, a = scipy.signal.butter(4, cutoff, btype='high')
    out = scipy.signal.lfilter(b, a, buffer)
    return _normalise(out)


def _filter_highpass_soft(buffer: np.ndarray) -> np.ndarray:
    """Gentle highpass at 80Hz (removes sub-bass)."""
    import scipy.signal
    nyq = SAMPLE_RATE / 2
    cutoff = max(0.01, 80 / nyq)
    b, a = scipy.signal.butter(2, cutoff, btype='high')
    out = scipy.signal.lfilter(b, a, buffer)
    return _normalise(out)


def _filter_highpass_hard(buffer: np.ndarray) -> np.ndarray:
    """Aggressive highpass at 1kHz."""
    import scipy.signal
    nyq = SAMPLE_RATE / 2
    cutoff = max(0.01, 1000 / nyq)
    b, a = scipy.signal.butter(6, cutoff, btype='high')
    out = scipy.signal.lfilter(b, a, buffer)
    return _normalise(out)


def _filter_bandpass(buffer: np.ndarray) -> np.ndarray:
    """Bandpass filter centered at 1kHz."""
    import scipy.signal
    nyq = SAMPLE_RATE / 2
    low = max(0.01, 400 / nyq)
    high = min(0.99, 2500 / nyq)
    b, a = scipy.signal.butter(4, [low, high], btype='band')
    out = scipy.signal.lfilter(b, a, buffer)
    return _normalise(out)


def _filter_bandpass_narrow(buffer: np.ndarray) -> np.ndarray:
    """Narrow bandpass (telephone-like)."""
    import scipy.signal
    nyq = SAMPLE_RATE / 2
    low = max(0.01, 300 / nyq)
    high = min(0.99, 3400 / nyq)
    b, a = scipy.signal.butter(6, [low, high], btype='band')
    out = scipy.signal.lfilter(b, a, buffer)
    return _normalise(out)


# ---------------------------------------------------------------------------
# PITCH SHIFTING EFFECTS
# ---------------------------------------------------------------------------

def pitchshift(buffer: np.ndarray, semitones: float = 0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Pitch shift audio by semitones using phase vocoder.
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio
    semitones : float
        Pitch shift in semitones (-12 to +12 typical)
    sr : int
        Sample rate
    
    Returns
    -------
    np.ndarray
        Pitch shifted audio (same length as input)
    """
    if abs(semitones) < 0.01:
        return buffer.copy()
    
    # Calculate pitch ratio
    ratio = 2 ** (semitones / 12.0)
    
    # Phase vocoder pitch shift
    fft_size = 2048
    hop = fft_size // 4
    
    # Pad buffer
    pad_len = fft_size
    padded = np.pad(buffer, (pad_len, pad_len), mode='constant')
    
    # Analysis frames
    num_frames = (len(padded) - fft_size) // hop + 1
    
    # Window
    window = np.hanning(fft_size)
    
    # Analyze
    phases = np.zeros(fft_size // 2 + 1)
    output = np.zeros(int(len(padded) / ratio) + fft_size)
    
    for i in range(num_frames):
        # Get frame
        start = i * hop
        frame = padded[start:start + fft_size] * window
        
        # FFT
        spectrum = np.fft.rfft(frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # Phase accumulation with ratio adjustment
        phase_diff = phase - phases
        phases = phase
        
        # Unwrap and scale
        phase_diff = phase_diff - 2 * np.pi * np.round(phase_diff / (2 * np.pi))
        
        # Synthesize at new position
        out_pos = int(i * hop / ratio)
        if out_pos + fft_size <= len(output):
            synth_phase = phases * ratio
            synth = magnitude * np.exp(1j * synth_phase)
            out_frame = np.fft.irfft(synth, fft_size) * window
            output[out_pos:out_pos + fft_size] += out_frame
    
    # Trim to original length
    output = output[int(pad_len / ratio):int(pad_len / ratio) + len(buffer)]
    
    # Ensure correct length
    if len(output) < len(buffer):
        output = np.pad(output, (0, len(buffer) - len(output)))
    elif len(output) > len(buffer):
        output = output[:len(buffer)]
    
    return _normalise(output)


def _pitch_up_2(buffer: np.ndarray) -> np.ndarray:
    """Pitch up 2 semitones."""
    return pitchshift(buffer, 2)


def _pitch_up_5(buffer: np.ndarray) -> np.ndarray:
    """Pitch up 5 semitones (fourth)."""
    return pitchshift(buffer, 5)


def _pitch_up_7(buffer: np.ndarray) -> np.ndarray:
    """Pitch up 7 semitones (fifth)."""
    return pitchshift(buffer, 7)


def _pitch_up_12(buffer: np.ndarray) -> np.ndarray:
    """Pitch up 12 semitones (octave)."""
    return pitchshift(buffer, 12)


def _pitch_down_2(buffer: np.ndarray) -> np.ndarray:
    """Pitch down 2 semitones."""
    return pitchshift(buffer, -2)


def _pitch_down_5(buffer: np.ndarray) -> np.ndarray:
    """Pitch down 5 semitones (fourth)."""
    return pitchshift(buffer, -5)


def _pitch_down_7(buffer: np.ndarray) -> np.ndarray:
    """Pitch down 7 semitones (fifth)."""
    return pitchshift(buffer, -7)


def _pitch_down_12(buffer: np.ndarray) -> np.ndarray:
    """Pitch down 12 semitones (octave)."""
    return pitchshift(buffer, -12)


def _harmonizer_3rd(buffer: np.ndarray) -> np.ndarray:
    """Add major 3rd harmony (+4 semitones)."""
    harmony = pitchshift(buffer, 4)
    mixed = buffer * 0.7 + harmony * 0.5
    return _normalise(mixed)


def _harmonizer_5th(buffer: np.ndarray) -> np.ndarray:
    """Add perfect 5th harmony (+7 semitones)."""
    harmony = pitchshift(buffer, 7)
    mixed = buffer * 0.7 + harmony * 0.5
    return _normalise(mixed)


def _harmonizer_octave(buffer: np.ndarray) -> np.ndarray:
    """Add octave harmony (+12 semitones)."""
    harmony = pitchshift(buffer, 12)
    mixed = buffer * 0.7 + harmony * 0.4
    return _normalise(mixed)


def _harmonizer_chord(buffer: np.ndarray) -> np.ndarray:
    """Add major chord harmony (3rd + 5th)."""
    h3 = pitchshift(buffer, 4)
    h5 = pitchshift(buffer, 7)
    mixed = buffer * 0.6 + h3 * 0.35 + h5 * 0.35
    return _normalise(mixed)


# ---------------------------------------------------------------------------
# Helper: simple filter application based on MonolithEngine's filter bank.

# Filter type constants for reference
FILTER_TYPES = {
    # Basic filters (0-6)
    0: 'lowpass',
    1: 'highpass', 
    2: 'bandpass',
    3: 'notch',
    4: 'peak',
    5: 'ringmod',
    6: 'allpass',
    # Comb filters (7-9)
    7: 'comb_ff',
    8: 'comb_fb',
    9: 'comb_both',
    # Analog modeled (10-11)
    10: 'analog',
    11: 'acid',
    # Formant filters (12-14)
    12: 'formant_a',
    13: 'formant_e',
    14: 'formant_i',
    # NEW: Extended filters (15-29)
    15: 'formant_o',
    16: 'formant_u',
    17: 'lowshelf',
    18: 'highshelf',
    19: 'moog',
    20: 'svf_lp',
    21: 'svf_hp',
    22: 'svf_bp',
    23: 'bitcrush',
    24: 'downsample',
    25: 'dc_block',
    26: 'tilt',
    27: 'resonant',
    28: 'vocal',
    29: 'telephone',
}

def _apply_filter(buffer: np.ndarray, filter_type: int, cutoff: float, resonance: float) -> np.ndarray:
    """Apply a filter to the buffer using settings similar to MonolithEngine.

    Parameters
    ----------
    buffer : np.ndarray
        Input audio buffer (float64).
    filter_type : int
        Index of the filter to apply (0-29).
    cutoff : float
        Cutoff frequency in Hz.  Must be > 0.
    resonance : float
        Resonance/Q parameter.  Its interpretation depends on the filter.

    Returns
    -------
    np.ndarray
        The filtered buffer.
    """
    import scipy.signal  # type: ignore
    nyq = 0.5 * SAMPLE_RATE
    norm_cutoff = max(0.001, min(0.999, cutoff / nyq))
    q = resonance if resonance is not None else 0.5
    out = buffer.copy().astype(np.float64)
    
    try:
        # === BASIC FILTERS (0-6) ===
        if filter_type == 0:
            # Low pass
            b, a = scipy.signal.butter(2, norm_cutoff, btype='low')
            out = scipy.signal.lfilter(b, a, out)
            
        elif filter_type == 1:
            # High pass
            b, a = scipy.signal.butter(2, norm_cutoff, btype='high')
            out = scipy.signal.lfilter(b, a, out)
            
        elif filter_type == 2:
            # Band pass: 1 octave bandwidth around cutoff
            low = max(0.001, norm_cutoff / np.sqrt(2))
            high = min(0.999, norm_cutoff * np.sqrt(2))
            if high > low:
                b, a = scipy.signal.butter(2, [low, high], btype='band')
                out = scipy.signal.lfilter(b, a, out)
                
        elif filter_type == 3:
            # Notch (band stop)
            low = max(0.001, norm_cutoff / np.sqrt(2))
            high = min(0.999, norm_cutoff * np.sqrt(2))
            if high > low:
                b, a = scipy.signal.butter(2, [low, high], btype='bandstop')
                out = scipy.signal.lfilter(b, a, out)
                
        elif filter_type == 4:
            # Peak filter: narrow band around cutoff
            bw = max(0.001, norm_cutoff * 0.1)
            low = max(0.001, norm_cutoff - bw)
            high = min(0.999, norm_cutoff + bw)
            if high > low:
                b, a = scipy.signal.butter(2, [low, high], btype='band')
                out = scipy.signal.lfilter(b, a, out)
                
        elif filter_type == 5:
            # Ring mod filter: multiply by a sine at cutoff frequency
            t = np.arange(len(out)) / SAMPLE_RATE
            ring = np.sin(2.0 * math.pi * cutoff * t)
            out = out * ring
            
        elif filter_type == 6:
            # All pass: phase shift without amplitude change
            # Using first-order allpass
            d = (1.0 - norm_cutoff) / (1.0 + norm_cutoff)
            y = np.zeros_like(out)
            y[0] = -d * out[0]
            for i in range(1, len(out)):
                y[i] = -d * out[i] + out[i-1] + d * y[i-1]
            out = y
            
        # === COMB FILTERS (7-9) ===
        elif filter_type == 7:
            # Feed-forward comb
            delay_samples = max(1, int(SAMPLE_RATE / max(20, cutoff)))
            g = min(0.99, max(-0.99, q))
            comb_out = out.copy()
            if delay_samples < len(out):
                comb_out[delay_samples:] += g * out[:-delay_samples]
            out = comb_out
            
        elif filter_type == 8:
            # Feed-back comb
            delay_samples = max(1, int(SAMPLE_RATE / max(20, cutoff)))
            g = min(0.95, max(-0.95, q))
            comb_out = out.copy()
            for i in range(delay_samples, len(out)):
                comb_out[i] = out[i] + g * comb_out[i - delay_samples]
            out = comb_out
            
        elif filter_type == 9:
            # Combined feed-forward and feed-back
            delay_samples = max(1, int(SAMPLE_RATE / max(20, cutoff)))
            g = min(0.9, max(-0.9, q))
            comb_out = out.copy()
            for i in range(delay_samples, len(out)):
                comb_out[i] = out[i] + g * out[i - delay_samples] + g * comb_out[i - delay_samples]
            out = comb_out
            
        # === ANALOG MODELED (10-11) ===
        elif filter_type == 10:
            # Analog lowpass (4-pole cascade)
            fc = norm_cutoff
            alpha = fc / (fc + 1.0)
            y = out.copy()
            for stage in range(4):
                y2 = np.zeros_like(y)
                y2[0] = alpha * y[0]
                for i in range(1, len(y)):
                    y2[i] = y2[i - 1] + alpha * (y[i] - y2[i - 1])
                y = y2
            out = y
            
        elif filter_type == 11:
            # Acid (303-style with saturation)
            fc = norm_cutoff
            alpha = fc / (fc + 1.0)
            res_amt = min(4.0, q)  # Resonance adds feedback
            y = out.copy()
            fb = np.zeros_like(y)
            for stage in range(4):
                y2 = np.zeros_like(y)
                y2[0] = alpha * (y[0] - res_amt * fb[0] if stage == 0 else y[0])
                for i in range(1, len(y)):
                    input_val = y[i] - res_amt * fb[i] if stage == 0 else y[i]
                    y2[i] = y2[i - 1] + alpha * (input_val - y2[i - 1])
                if stage == 3:
                    fb = y2
                y2 = np.tanh(y2 * 1.5) / 1.5  # Soft saturation
                y = y2
            out = y
            
        # === FORMANT FILTERS (12-16) ===
        elif filter_type == 12:
            # Formant A (ah)
            formants = [(730, 90), (1090, 110), (2440, 170)]
            out = _apply_formants(out, formants, nyq, q)
            
        elif filter_type == 13:
            # Formant E (eh)
            formants = [(530, 60), (1840, 150), (2480, 200)]
            out = _apply_formants(out, formants, nyq, q)
            
        elif filter_type == 14:
            # Formant I (ee)
            formants = [(270, 60), (2290, 200), (3010, 300)]
            out = _apply_formants(out, formants, nyq, q)
            
        elif filter_type == 15:
            # Formant O (oh)
            formants = [(570, 70), (840, 80), (2410, 170)]
            out = _apply_formants(out, formants, nyq, q)
            
        elif filter_type == 16:
            # Formant U (oo)
            formants = [(300, 50), (870, 90), (2240, 160)]
            out = _apply_formants(out, formants, nyq, q)
            
        # === SHELF FILTERS (17-18) ===
        elif filter_type == 17:
            # Low shelf
            gain_db = (q - 0.5) * 24  # -12 to +12 dB based on Q
            A = 10 ** (gain_db / 40)
            w0 = 2 * np.pi * norm_cutoff
            alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * 2)
            
            b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
            a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
            
            b = np.array([b0/a0, b1/a0, b2/a0])
            a = np.array([1, a1/a0, a2/a0])
            out = scipy.signal.lfilter(b, a, out)
            
        elif filter_type == 18:
            # High shelf
            gain_db = (q - 0.5) * 24  # -12 to +12 dB
            A = 10 ** (gain_db / 40)
            w0 = 2 * np.pi * norm_cutoff
            alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * 2)
            
            b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
            a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
            
            b = np.array([b0/a0, b1/a0, b2/a0])
            a = np.array([1, a1/a0, a2/a0])
            out = scipy.signal.lfilter(b, a, out)
            
        # === MOOG LADDER (19) ===
        elif filter_type == 19:
            # Moog ladder filter (4-pole resonant)
            fc = cutoff / SAMPLE_RATE
            fc = max(0.001, min(0.45, fc))
            k = min(4.0, q * 4)  # Resonance (0-4, self-oscillates at 4)
            
            # State variables
            s = np.zeros(4)
            out_buf = np.zeros_like(out)
            
            for i in range(len(out)):
                x = out[i] - k * s[3]  # Feedback
                x = np.tanh(x)  # Input saturation
                
                # 4-pole cascade
                for j in range(4):
                    s[j] = s[j] + fc * (x - s[j])
                    x = s[j]
                    
                out_buf[i] = s[3]
            out = out_buf
            
        # === STATE VARIABLE (20-22) ===
        elif filter_type == 20:
            # SVF lowpass
            out = _apply_svf(out, cutoff, q, 'lp')
            
        elif filter_type == 21:
            # SVF highpass
            out = _apply_svf(out, cutoff, q, 'hp')
            
        elif filter_type == 22:
            # SVF bandpass
            out = _apply_svf(out, cutoff, q, 'bp')
            
        # === DESTRUCTIVE FILTERS (23-24) ===
        elif filter_type == 23:
            # Bit crusher (Q controls bit depth)
            bits = max(1, int(1 + q * 15))  # 1-16 bits
            levels = 2 ** bits
            out = np.round(out * levels) / levels
            
        elif filter_type == 24:
            # Downsample (cutoff controls sample rate divisor)
            factor = max(1, int(cutoff / 1000))  # 1-20x downsampling
            if factor > 1:
                # Simple sample-and-hold downsampling
                for i in range(len(out)):
                    if i % factor != 0:
                        out[i] = out[i - (i % factor)]
                        
        # === UTILITY FILTERS (25-26) ===
        elif filter_type == 25:
            # DC blocker
            alpha = 0.995
            y = np.zeros_like(out)
            y[0] = out[0]
            for i in range(1, len(out)):
                y[i] = out[i] - out[i-1] + alpha * y[i-1]
            out = y
            
        elif filter_type == 26:
            # Tilt EQ (cutoff = tilt freq, Q = tilt amount)
            # Positive Q boosts highs, negative boosts lows
            tilt = (q - 0.5) * 2  # -1 to +1
            # Low shelf at tilt freq
            b, a = scipy.signal.butter(1, norm_cutoff, btype='low')
            lows = scipy.signal.lfilter(b, a, out)
            b, a = scipy.signal.butter(1, norm_cutoff, btype='high')
            highs = scipy.signal.lfilter(b, a, out)
            out = lows * (1 - tilt * 0.5) + highs * (1 + tilt * 0.5)
            
        # === CHARACTER FILTERS (27-29) ===
        elif filter_type == 27:
            # Resonant filter (strong resonance at cutoff)
            Q_val = 1 + q * 10  # Higher Q for more resonance
            w0 = 2 * np.pi * norm_cutoff
            alpha = np.sin(w0) / (2 * Q_val)
            
            b0 = alpha
            b1 = 0
            b2 = -alpha
            a0 = 1 + alpha
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha
            
            b = np.array([b0/a0, b1/a0, b2/a0])
            a = np.array([1, a1/a0, a2/a0])
            out = scipy.signal.lfilter(b, a, out)
            
        elif filter_type == 28:
            # Vocal filter (parallel formants with resonance control)
            formants = [(500, 80), (1500, 100), (2500, 150), (3500, 200)]
            mix = np.zeros_like(out)
            for freq, bw in formants:
                bw_scaled = bw * (0.5 + q)  # Q affects bandwidth
                low = max(0.001, (freq - bw_scaled/2) / nyq)
                high = min(0.999, (freq + bw_scaled/2) / nyq)
                if high > low:
                    b, a = scipy.signal.butter(2, [low, high], btype='band')
                    mix += scipy.signal.lfilter(b, a, out) * 0.4
            out = mix
            
        elif filter_type == 29:
            # Telephone (bandpass 300-3400 Hz)
            low = max(0.001, 300 / nyq)
            high = min(0.999, 3400 / nyq)
            if high > low:
                b, a = scipy.signal.butter(4, [low, high], btype='band')
                out = scipy.signal.lfilter(b, a, out)
            # Add subtle distortion for character
            out = np.tanh(out * (1 + q))
            
    except Exception:
        pass
        
    return out.astype(np.float64)


def _apply_formants(buffer: np.ndarray, formants: list, nyq: float, q: float) -> np.ndarray:
    """Apply parallel formant filters."""
    import scipy.signal
    mix = np.zeros_like(buffer)
    for freq, bw in formants:
        bw_scaled = bw * (0.5 + q * 0.5)  # Q affects bandwidth
        low = max(0.001, (freq - bw_scaled/2) / nyq)
        high = min(0.999, (freq + bw_scaled/2) / nyq)
        if high > low:
            b, a = scipy.signal.butter(2, [low, high], btype='band')
            mix += scipy.signal.lfilter(b, a, buffer) * 0.5
    return mix


def _apply_svf(buffer: np.ndarray, cutoff: float, q: float, mode: str) -> np.ndarray:
    """Apply state variable filter."""
    # SVF coefficients
    fc = cutoff / SAMPLE_RATE
    fc = max(0.001, min(0.45, fc))
    Q_val = max(0.5, q * 2)
    
    g = np.tan(np.pi * fc)
    k = 1 / Q_val
    
    a1 = 1 / (1 + g * (g + k))
    a2 = g * a1
    a3 = g * a2
    
    # State
    ic1eq = 0.0
    ic2eq = 0.0
    
    out = np.zeros_like(buffer)
    
    for i in range(len(buffer)):
        v0 = buffer[i]
        v3 = v0 - ic2eq
        v1 = a1 * ic1eq + a2 * v3
        v2 = ic2eq + a2 * ic1eq + a3 * v3
        ic1eq = 2 * v1 - ic1eq
        ic2eq = 2 * v2 - ic2eq
        
        if mode == 'lp':
            out[i] = v2
        elif mode == 'hp':
            out[i] = v0 - k * v1 - v2
        elif mode == 'bp':
            out[i] = v1
            
    return out


# =============================================================================
# VOCODER
# =============================================================================

def vocoder(
    carrier: np.ndarray,
    modulator: np.ndarray,
    bands: int = 16,
    sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Vocoder effect - imposes modulator's spectral envelope on carrier.
    
    Parameters
    ----------
    carrier : np.ndarray
        Carrier signal (typically synth, noise, or sawtooth)
    modulator : np.ndarray
        Modulator signal (typically voice)
    bands : int
        Number of frequency bands (8-32)
    sr : int
        Sample rate
    
    Returns
    -------
    np.ndarray
        Vocoded output
    """
    import scipy.signal
    
    bands = max(8, min(32, bands))
    
    # Match lengths
    min_len = min(len(carrier), len(modulator))
    carrier = carrier[:min_len]
    modulator = modulator[:min_len]
    
    # Frequency band edges (logarithmic spacing from 100Hz to 8kHz)
    low_freq = 100
    high_freq = min(8000, sr // 2 - 100)
    band_edges = np.logspace(np.log10(low_freq), np.log10(high_freq), bands + 1)
    
    nyq = sr / 2
    output = np.zeros_like(carrier)
    
    for i in range(bands):
        low = band_edges[i] / nyq
        high = band_edges[i + 1] / nyq
        
        # Clamp to valid range
        low = max(0.001, min(0.999, low))
        high = max(0.001, min(0.999, high))
        
        if high <= low:
            continue
        
        try:
            # Bandpass filter for this band
            b, a = scipy.signal.butter(2, [low, high], btype='band')
            
            # Filter both signals
            mod_band = scipy.signal.lfilter(b, a, modulator)
            car_band = scipy.signal.lfilter(b, a, carrier)
            
            # Extract envelope from modulator (rectify + lowpass)
            mod_env = np.abs(mod_band)
            
            # Smooth envelope
            env_cutoff = 50 / nyq  # 50Hz envelope following
            b_env, a_env = scipy.signal.butter(2, max(0.001, env_cutoff), btype='low')
            mod_env = scipy.signal.lfilter(b_env, a_env, mod_env)
            
            # Apply modulator envelope to carrier band
            output += car_band * mod_env * 2
            
        except Exception:
            continue
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.9
    
    return output


def _vocoder_synth(buffer: np.ndarray) -> np.ndarray:
    """Vocoder with internal synth carrier (sawtooth)."""
    import scipy.signal as sig
    
    # Generate sawtooth carrier at multiple frequencies
    duration = len(buffer) / SAMPLE_RATE
    t = np.linspace(0, duration, len(buffer))
    
    # Rich carrier: stacked saws
    carrier = np.zeros_like(buffer)
    for freq in [110, 220, 330, 440]:
        carrier += sig.sawtooth(2 * np.pi * freq * t) * 0.25
    
    return vocoder(carrier, buffer, bands=16)


def _vocoder_noise(buffer: np.ndarray) -> np.ndarray:
    """Vocoder with noise carrier (robot voice)."""
    # White noise carrier
    carrier = np.random.randn(len(buffer)) * 0.5
    return vocoder(carrier, buffer, bands=24)


def _vocoder_chord(buffer: np.ndarray) -> np.ndarray:
    """Vocoder with chord carrier."""
    import scipy.signal as sig
    
    duration = len(buffer) / SAMPLE_RATE
    t = np.linspace(0, duration, len(buffer))
    
    # Major chord carrier
    carrier = np.zeros_like(buffer)
    for freq in [130.81, 164.81, 196.00, 261.63]:  # C major
        carrier += sig.sawtooth(2 * np.pi * freq * t) * 0.25
    
    return vocoder(carrier, buffer, bands=20)


# =============================================================================
# SPECTRAL PROCESSING
# =============================================================================

def spectral_freeze(buffer: np.ndarray, position: float = 0.5) -> np.ndarray:
    """Freeze spectrum at a point and sustain.
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio
    position : float
        Position in buffer to freeze (0-1)
    
    Returns
    -------
    np.ndarray
        Frozen spectrum sustained for buffer length
    """
    fft_size = 2048
    hop = fft_size // 4
    
    # Get spectrum at position
    pos_sample = int(position * len(buffer))
    pos_sample = max(0, min(len(buffer) - fft_size, pos_sample))
    
    frame = buffer[pos_sample:pos_sample + fft_size]
    if len(frame) < fft_size:
        frame = np.pad(frame, (0, fft_size - len(frame)))
    
    # Get magnitude and phase
    window = np.hanning(fft_size)
    spectrum = np.fft.rfft(frame * window)
    magnitude = np.abs(spectrum)
    
    # Reconstruct with random phase evolution
    output = np.zeros(len(buffer))
    
    for i in range(0, len(buffer) - fft_size, hop):
        # Slowly evolving random phase
        phase = np.random.uniform(-np.pi, np.pi, len(magnitude))
        
        # Reconstruct frame
        frame_spectrum = magnitude * np.exp(1j * phase)
        frame_out = np.fft.irfft(frame_spectrum, fft_size)
        
        # Overlap-add
        output[i:i + fft_size] += frame_out * window
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * np.max(np.abs(buffer))
    
    return output


def spectral_blur(buffer: np.ndarray, amount: float = 50) -> np.ndarray:
    """Blur/smear spectrum over time.
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio
    amount : float
        Blur amount (0-100)
    
    Returns
    -------
    np.ndarray
        Spectrally blurred audio
    """
    fft_size = 2048
    hop = fft_size // 4
    
    blur_factor = scale_amount(amount, 0.0, 0.95)
    
    window = np.hanning(fft_size)
    output = np.zeros(len(buffer))
    
    prev_magnitude = None
    
    for i in range(0, len(buffer) - fft_size, hop):
        frame = buffer[i:i + fft_size]
        
        spectrum = np.fft.rfft(frame * window)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # Blur magnitude with previous frame
        if prev_magnitude is not None:
            magnitude = blur_factor * prev_magnitude + (1 - blur_factor) * magnitude
        
        prev_magnitude = magnitude.copy()
        
        # Reconstruct
        frame_spectrum = magnitude * np.exp(1j * phase)
        frame_out = np.fft.irfft(frame_spectrum, fft_size)
        
        output[i:i + fft_size] += frame_out * window
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * np.max(np.abs(buffer))
    
    return output


def spectral_shift(buffer: np.ndarray, semitones: float = 0) -> np.ndarray:
    """Shift spectrum up or down.
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio
    semitones : float
        Shift amount in semitones
    
    Returns
    -------
    np.ndarray
        Pitch-shifted audio (formant preserved)
    """
    if abs(semitones) < 0.01:
        return buffer
    
    fft_size = 2048
    hop = fft_size // 4
    
    shift_ratio = 2 ** (semitones / 12)
    
    window = np.hanning(fft_size)
    output = np.zeros(len(buffer))
    
    for i in range(0, len(buffer) - fft_size, hop):
        frame = buffer[i:i + fft_size]
        
        spectrum = np.fft.rfft(frame * window)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # Shift bins
        new_magnitude = np.zeros_like(magnitude)
        new_phase = np.zeros_like(phase)
        
        for bin_idx in range(len(magnitude)):
            new_bin = int(bin_idx * shift_ratio)
            if 0 <= new_bin < len(magnitude):
                new_magnitude[new_bin] += magnitude[bin_idx]
                new_phase[new_bin] = phase[bin_idx]
        
        # Reconstruct
        frame_spectrum = new_magnitude * np.exp(1j * new_phase)
        frame_out = np.fft.irfft(frame_spectrum, fft_size)
        
        output[i:i + fft_size] += frame_out * window
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * np.max(np.abs(buffer))
    
    return output


def _spc_freeze(buffer: np.ndarray) -> np.ndarray:
    """Spectral freeze effect preset."""
    return spectral_freeze(buffer, 0.5)


def _spc_blur(buffer: np.ndarray) -> np.ndarray:
    """Spectral blur effect preset."""
    return spectral_blur(buffer, 60)


def _spc_shift_up(buffer: np.ndarray) -> np.ndarray:
    """Spectral shift up 5 semitones."""
    return spectral_shift(buffer, 5)


def _spc_shift_down(buffer: np.ndarray) -> np.ndarray:
    """Spectral shift down 5 semitones."""
    return spectral_shift(buffer, -5)


# =============================================================================
# STEREO PROCESSING
# =============================================================================

def stereo_spread(buffer: np.ndarray, spread: float = 50, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Create stereo spread from mono source.
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio (mono or stereo)
    spread : float
        Spread amount (0-100)
    sr : int
        Sample rate
    
    Returns
    -------
    np.ndarray
        Stereo audio (N, 2)
    """
    import scipy.signal
    
    # Ensure mono input
    if buffer.ndim > 1:
        mono = np.mean(buffer, axis=1)
    else:
        mono = buffer
    
    spread_amt = scale_amount(spread, 0.0, 1.0)
    
    if spread_amt < 0.01:
        # No spread - return mono as stereo
        return np.column_stack([mono, mono])
    
    # Create stereo via multiple methods
    
    # 1. Slight delay difference (Haas effect)
    delay_samples = int(spread_amt * 0.02 * sr)  # Up to 20ms
    left = mono.copy()
    right = np.zeros_like(mono)
    right[delay_samples:] = mono[:-delay_samples] if delay_samples > 0 else mono
    
    # 2. Frequency-dependent panning (low center, high spread)
    nyq = sr / 2
    b_low, a_low = scipy.signal.butter(2, min(0.999, 300 / nyq), btype='low')
    low = scipy.signal.lfilter(b_low, a_low, mono)
    
    # High frequencies (spread)
    b_high, a_high = scipy.signal.butter(2, max(0.001, 300 / nyq), btype='high')
    high = scipy.signal.lfilter(b_high, a_high, mono)
    
    # Mix
    left = low + high * (0.5 + spread_amt * 0.5) + left * spread_amt * 0.3
    right = low + high * (0.5 + spread_amt * 0.5) + right * spread_amt * 0.3
    
    # Normalize
    peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if peak > 1:
        left /= peak
        right /= peak
    
    return np.column_stack([left, right])


def _stereo_wide(buffer: np.ndarray) -> np.ndarray:
    """Wide stereo effect. Returns stereo (N, 2) array."""
    result = stereo_spread(buffer, 80)
    return result


def _stereo_narrow(buffer: np.ndarray) -> np.ndarray:
    """Narrow/mono stereo effect. Returns stereo (N, 2) with identical channels."""
    if buffer.ndim > 1:
        mono = np.mean(buffer, axis=1)
    else:
        mono = buffer
    return np.column_stack([mono, mono])


# =============================================================================
# LFO MODULATION EFFECTS
# =============================================================================

class LFO:
    """Low Frequency Oscillator for parameter modulation."""
    
    def __init__(self, shape: str = 'sin', rate: float = 1.0, depth: float = 50, sr: int = SAMPLE_RATE):
        self.shape = shape
        self.rate = rate
        self.depth = scale_amount(depth, 0.0, 1.0)
        self.sr = sr
        self.phase = 0.0
    
    def generate(self, num_samples: int) -> np.ndarray:
        """Generate LFO waveform.
        
        Returns
        -------
        np.ndarray
            LFO values in range [-depth, +depth]
        """
        t = np.arange(num_samples) / self.sr + self.phase
        self.phase = (self.phase + num_samples / self.sr) % (1 / max(0.001, self.rate))
        
        phase_rad = 2 * np.pi * self.rate * t
        
        if self.shape == 'sin':
            wave = np.sin(phase_rad)
        elif self.shape == 'tri':
            wave = 2 * np.abs(2 * (t * self.rate % 1) - 1) - 1
        elif self.shape == 'saw':
            wave = 2 * (t * self.rate % 1) - 1
        elif self.shape == 'sqr':
            wave = np.sign(np.sin(phase_rad))
        elif self.shape == 'rnd':
            # Sample and hold random
            period_samples = max(1, int(self.sr / max(0.001, self.rate)))
            wave = np.zeros(num_samples)
            value = np.random.uniform(-1, 1)
            for i in range(num_samples):
                if i % period_samples == 0:
                    value = np.random.uniform(-1, 1)
                wave[i] = value
        else:
            wave = np.sin(phase_rad)
        
        return wave * self.depth


def apply_lfo_to_cutoff(buffer: np.ndarray, base_cutoff: float, lfo_rate: float = 1.0, 
                        lfo_depth: float = 50, lfo_shape: str = 'sin') -> np.ndarray:
    """Apply LFO modulation to filter cutoff.
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio
    base_cutoff : float
        Base cutoff frequency in Hz
    lfo_rate : float
        LFO rate in Hz
    lfo_depth : float
        Modulation depth (0-100)
    lfo_shape : str
        LFO shape (sin, tri, saw, sqr, rnd)
    
    Returns
    -------
    np.ndarray
        Filtered audio with modulated cutoff
    """
    import scipy.signal
    
    lfo = LFO(lfo_shape, lfo_rate, lfo_depth)
    mod = lfo.generate(len(buffer))
    
    # Modulate cutoff (in octaves)
    cutoff_mod = base_cutoff * (2 ** (mod * 2))  # +/- 2 octaves at full depth
    cutoff_mod = np.clip(cutoff_mod, 20, SAMPLE_RATE // 2 - 100)
    
    # Time-varying filter (simplified - process in blocks)
    block_size = 256
    output = np.zeros_like(buffer)
    
    nyq = SAMPLE_RATE / 2
    
    for i in range(0, len(buffer), block_size):
        end = min(i + block_size, len(buffer))
        block = buffer[i:end]
        
        # Average cutoff for this block
        avg_cutoff = np.mean(cutoff_mod[i:end])
        
        try:
            wn = max(0.001, min(0.999, avg_cutoff / nyq))
            b, a = scipy.signal.butter(2, wn, btype='low')
            output[i:end] = scipy.signal.lfilter(b, a, block)
        except:
            output[i:end] = block
    
    return output


def _lfo_filter_slow(buffer: np.ndarray) -> np.ndarray:
    """Slow LFO filter sweep."""
    return apply_lfo_to_cutoff(buffer, 1000, lfo_rate=0.5, lfo_depth=70)


def _lfo_filter_fast(buffer: np.ndarray) -> np.ndarray:
    """Fast LFO filter sweep (wah-wah style)."""
    return apply_lfo_to_cutoff(buffer, 800, lfo_rate=4.0, lfo_depth=60)


def _lfo_tremolo(buffer: np.ndarray) -> np.ndarray:
    """Tremolo effect (amplitude LFO)."""
    lfo = LFO('sin', 6.0, 40)
    mod = lfo.generate(len(buffer))
    # Scale modulation to 0.5-1.0 range to prevent clipping
    # mod is in range [-0.4, 0.4], convert to [0.6, 1.0]
    gain = 0.7 + mod * 0.75  # centers around 0.7, varies by ±0.3
    result = buffer * gain
    # Soft limit to prevent any clipping
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result = result / peak
    return result


def _lfo_vibrato(buffer: np.ndarray) -> np.ndarray:
    """Vibrato effect (pitch LFO) - approximation via delay modulation."""
    lfo = LFO('sin', 5.0, 30)
    mod = lfo.generate(len(buffer))
    
    # Variable delay for vibrato effect
    max_delay = int(0.003 * SAMPLE_RATE)  # 3ms max delay
    delay_samples = ((mod + 1) * max_delay / 2).astype(int)
    
    output = np.zeros_like(buffer)
    for i in range(len(buffer)):
        read_pos = i - delay_samples[i]
        if 0 <= read_pos < len(buffer):
            output[i] = buffer[read_pos]
        else:
            output[i] = buffer[i]
    
    return output


# =============================================================================
# GRANULAR EFFECTS (whole-source presets)
# =============================================================================

def _granular_cloud(buffer: np.ndarray) -> np.ndarray:
    """Granular cloud — dense, ethereal texture from whole source."""
    from .granular import granular_process
    dur = len(buffer) / SAMPLE_RATE
    return granular_process(buffer, dur, SAMPLE_RATE,
                           grain_size=60, density=8, position=0.5,
                           spread=1.0, pitch=1.0, envelope='hann')


def _granular_scatter(buffer: np.ndarray) -> np.ndarray:
    """Granular scatter — sparse random grains across source."""
    from .granular import granular_process
    dur = len(buffer) / SAMPLE_RATE
    return granular_process(buffer, dur, SAMPLE_RATE,
                           grain_size=30, density=2, position=0.5,
                           spread=1.0, pitch=1.0, envelope='triangle')


def _granular_stretch(buffer: np.ndarray) -> np.ndarray:
    """Granular time-stretch — slow to 2x duration, preserving pitch."""
    from .granular import granular_process
    dur = len(buffer) / SAMPLE_RATE * 2.0
    return granular_process(buffer, dur, SAMPLE_RATE,
                           grain_size=80, density=6, position=0.5,
                           spread=0.8, pitch=1.0, envelope='hann')


def _granular_freeze(buffer: np.ndarray) -> np.ndarray:
    """Granular freeze — sustains the midpoint of the source."""
    from .granular import granular_process
    dur = len(buffer) / SAMPLE_RATE
    return granular_process(buffer, dur, SAMPLE_RATE,
                           grain_size=40, density=10, position=0.5,
                           spread=0.05, pitch=1.0, envelope='gaussian')


def _granular_shimmer(buffer: np.ndarray) -> np.ndarray:
    """Granular shimmer — pitch-shifted grains for sparkle."""
    from .granular import granular_process
    dur = len(buffer) / SAMPLE_RATE
    a = granular_process(buffer, dur, SAMPLE_RATE,
                         grain_size=35, density=4, position=0.5,
                         spread=0.6, pitch=2.0, envelope='hann')
    # Mix pitched grains with dry
    min_len = min(len(buffer), len(a))
    out = buffer[:min_len] * 0.6 + a[:min_len] * 0.4
    return out


def _granular_reverse(buffer: np.ndarray) -> np.ndarray:
    """Granular reverse — randomly reversed grains from whole source."""
    from .granular import granular_process
    dur = len(buffer) / SAMPLE_RATE
    return granular_process(buffer, dur, SAMPLE_RATE,
                           grain_size=50, density=5, position=0.5,
                           spread=1.0, pitch=1.0, reverse=0.7,
                           envelope='tukey')


def _granular_stutter(buffer: np.ndarray) -> np.ndarray:
    """Granular stutter — tiny grains for glitchy repetition."""
    from .granular import granular_process
    dur = len(buffer) / SAMPLE_RATE
    return granular_process(buffer, dur, SAMPLE_RATE,
                           grain_size=10, density=12, position=0.5,
                           spread=0.3, pitch=1.0, envelope='rect')


# =============================================================================
# UTILITY EFFECTS (normalization, declip, declick, smoothing)
# =============================================================================

def _util_normalize(buffer: np.ndarray) -> np.ndarray:
    """Normalize peak to -1 dB."""
    peak = np.max(np.abs(buffer))
    if peak < 1e-10:
        return buffer
    target = 10 ** (-1.0 / 20.0)  # -1dB
    return buffer * (target / peak)


def _util_normalize_rms(buffer: np.ndarray) -> np.ndarray:
    """RMS normalize to -14 LUFS (approx -14 dB RMS)."""
    rms = np.sqrt(np.mean(buffer ** 2))
    if rms < 1e-10:
        return buffer
    target_rms = 10 ** (-14.0 / 20.0)
    out = buffer * (target_rms / rms)
    # Soft clip to prevent overs
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = np.tanh(out * 0.95)
    return out


def _util_declip(buffer: np.ndarray) -> np.ndarray:
    """De-clip — repair hard-clipped audio using cubic interpolation.

    Detects regions where signal is at ±1.0 (or within 0.001) and
    replaces them with a smooth interpolation from surrounding samples.
    """
    out = buffer.copy()
    threshold = 0.999

    # Find clipped regions
    clipped = np.abs(out) >= threshold
    if not np.any(clipped):
        return out

    # Label contiguous clipped regions
    changes = np.diff(clipped.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    # Handle edge cases
    if clipped[0]:
        starts = np.concatenate([[0], starts])
    if clipped[-1]:
        ends = np.concatenate([ends, [len(out)]])

    for s, e in zip(starts, ends):
        length = e - s
        # Get anchor values from surrounding samples
        pre_val = out[max(0, s - 1)]
        post_val = out[min(len(out) - 1, e)]
        # Cubic interpolation
        t = np.linspace(0, 1, length + 2)[1:-1]
        t2 = t * t
        t3 = t2 * t
        # Hermite basis
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        # Estimate tangents
        m0 = 0.0  # zero slope at clip entry
        m1 = 0.0  # zero slope at clip exit
        interp = h00 * pre_val + h10 * m0 + h01 * post_val + h11 * m1
        # Clamp to prevent new clipping
        interp = np.clip(interp, -0.98, 0.98)
        out[s:e] = interp[:e - s]

    return out


def _util_declick(buffer: np.ndarray) -> np.ndarray:
    """De-click — detect and remove transient clicks/pops.

    Uses a first-derivative threshold to find sudden jumps, then
    cross-fades across them.
    """
    out = buffer.copy()
    diff = np.abs(np.diff(out))
    median_diff = np.median(diff) + 1e-10
    threshold = median_diff * 15  # clicks are 15x the median jump

    click_indices = np.where(diff > threshold)[0]
    if len(click_indices) == 0:
        return out

    # Merge nearby clicks into regions
    fade_len = max(4, int(0.001 * SAMPLE_RATE))  # 1ms crossfade
    for idx in click_indices:
        start = max(0, idx - fade_len)
        end = min(len(out), idx + fade_len + 1)
        region_len = end - start
        # Linear crossfade across the click
        env = np.linspace(1, 0, region_len)
        left_val = out[start]
        right_val = out[min(end, len(out) - 1)]
        out[start:end] = env * left_val + (1 - env) * right_val

    return out


def _util_smooth(buffer: np.ndarray) -> np.ndarray:
    """Gentle smoothing — low-pass at ~8kHz to tame harshness."""
    import scipy.signal
    nyq = SAMPLE_RATE / 2
    cutoff = min(8000 / nyq, 0.999)
    b, a = scipy.signal.butter(2, cutoff, btype='low')
    return scipy.signal.lfilter(b, a, buffer)


def _util_smooth_heavy(buffer: np.ndarray) -> np.ndarray:
    """Heavy smoothing — low-pass at ~4kHz for very muffled/warm tone."""
    import scipy.signal
    nyq = SAMPLE_RATE / 2
    cutoff = min(4000 / nyq, 0.999)
    b, a = scipy.signal.butter(3, cutoff, btype='low')
    return scipy.signal.lfilter(b, a, buffer)


def _util_dc_remove(buffer: np.ndarray) -> np.ndarray:
    """Remove DC offset — high-pass at 10Hz."""
    import scipy.signal
    nyq = SAMPLE_RATE / 2
    cutoff = max(10 / nyq, 0.001)
    b, a = scipy.signal.butter(2, cutoff, btype='high')
    return scipy.signal.lfilter(b, a, buffer)


def _util_fade_in(buffer: np.ndarray) -> np.ndarray:
    """Apply 50ms fade-in to prevent click at start."""
    samples = min(int(0.05 * SAMPLE_RATE), len(buffer))
    out = buffer.copy()
    out[:samples] *= np.linspace(0, 1, samples)
    return out


def _util_fade_out(buffer: np.ndarray) -> np.ndarray:
    """Apply 50ms fade-out to prevent click at end."""
    samples = min(int(0.05 * SAMPLE_RATE), len(buffer))
    out = buffer.copy()
    out[-samples:] *= np.linspace(1, 0, samples)
    return out


def _util_fade_both(buffer: np.ndarray) -> np.ndarray:
    """Apply 50ms fade-in and fade-out."""
    out = _util_fade_in(buffer)
    return _util_fade_out(out)


# Registry of all available effects
_effect_funcs = {
    # Reverbs (basic)
    'reverb_small': _reverb_small,
    'reverb_large': _reverb_large,
    'reverb_plate': _reverb_plate,
    'reverb_spring': _reverb_spring,
    'reverb_cathedral': _reverb_cathedral,
    # Convolution reverbs
    'conv_hall': _conv_hall,
    'conv_hall_long': _conv_hall_long,
    'conv_room': _conv_room,
    'conv_plate': _conv_plate,
    'conv_spring': _conv_spring,
    'conv_shimmer': _conv_shimmer,
    'conv_reverse': _conv_reverse,
    # Delays
    'delay_simple': _delay_simple,
    'delay_pingpong': _delay_pingpong,
    'delay_multitap': _delay_multitap,
    'delay_slapback': _delay_slapback,
    'delay_tape': _delay_tape_echo,
    # Saturations
    'saturate_soft': _saturate_soft,
    'saturate_hard': _saturate_hard,
    'saturate_overdrive': _saturate_overdrive,
    'saturate_fuzz': _saturate_fuzz,
    'saturate_tube': _saturate_tube,
    # Vamp/Overdrive/Waveshaping
    'vamp_light': _vamp_light,
    'vamp_medium': _vamp_medium,
    'vamp_heavy': _vamp_heavy,
    'vamp_fuzz': _vamp_fuzz,
    'overdrive_soft': _overdrive_soft,
    'overdrive_classic': _overdrive_classic,
    'overdrive_crunch': _overdrive_crunch,
    'dual_od_warm': _dual_od_warm,
    'dual_od_bright': _dual_od_bright,
    'dual_od_heavy': _dual_od_heavy,
    'waveshape_fold': _waveshape_fold,
    'waveshape_rectify': _waveshape_rectify,
    'waveshape_sine': _waveshape_sine,
    # Dynamics
    'compress_mild': _compress_mild,
    'compress_hard': _compress_hard,
    'compress_limiter': _compress_limiter,
    'compress_expander': _compress_expander,
    'compress_softclipper': _compress_softclipper,
    # Forever Compression (multiband OTT-style)
    'fc_punch': _fc_punch,
    'fc_glue': _fc_glue,
    'fc_loud': _fc_loud,
    'fc_soft': _fc_soft,
    'fc_ott': _fc_ott,
    # Gates
    'gate1': _gate1,
    'gate2': _gate2,
    'gate3': _gate3,
    'gate4': _gate4,
    'gate5': _gate5,
    # Giga Gate (pattern-based gating/stutter)
    'gg_half': _gg_half,
    'gg_quarter': _gg_quarter,
    'gg_tresillo': _gg_tresillo,
    'gg_glitch': _gg_glitch,
    'gg_stutter': _gg_stutter,
    'gg_halftime': _gg_halftime,
    'gg_tape_stop': _gg_tape_stop,
    # Lo‑fi
    'lofi_bitcrush': _lofi_bitcrush,
    'lofi_chorus': _lofi_chorus,
    'lofi_flanger': _lofi_flanger,
    'lofi_phaser': _lofi_phaser,
    'lofi_filter': _lofi_filter,
    'lofi_halftime': _lofi_halftime,
    # Vocoder
    'vocoder_synth': _vocoder_synth,
    'vocoder_noise': _vocoder_noise,
    'vocoder_chord': _vocoder_chord,
    # Spectral processing
    'spc_freeze': _spc_freeze,
    'spc_blur': _spc_blur,
    'spc_shift_up': _spc_shift_up,
    'spc_shift_down': _spc_shift_down,
    # Stereo
    'stereo_wide': _stereo_wide,
    'stereo_narrow': _stereo_narrow,
    # LFO effects
    'lfo_filter_slow': _lfo_filter_slow,
    'lfo_filter_fast': _lfo_filter_fast,
    'lfo_tremolo': _lfo_tremolo,
    'lfo_vibrato': _lfo_vibrato,
    # Standalone filters
    'filter_lowpass': _filter_lowpass,
    'filter_lowpass_soft': _filter_lowpass_soft,
    'filter_lowpass_hard': _filter_lowpass_hard,
    'filter_highpass': _filter_highpass,
    'filter_highpass_soft': _filter_highpass_soft,
    'filter_highpass_hard': _filter_highpass_hard,
    'filter_bandpass': _filter_bandpass,
    'filter_bandpass_narrow': _filter_bandpass_narrow,
    # Pitch shifting
    'pitch_up_2': _pitch_up_2,
    'pitch_up_5': _pitch_up_5,
    'pitch_up_7': _pitch_up_7,
    'pitch_up_12': _pitch_up_12,
    'pitch_down_2': _pitch_down_2,
    'pitch_down_5': _pitch_down_5,
    'pitch_down_7': _pitch_down_7,
    'pitch_down_12': _pitch_down_12,
    # Harmonizers
    'harmonizer_3rd': _harmonizer_3rd,
    'harmonizer_5th': _harmonizer_5th,
    'harmonizer_octave': _harmonizer_octave,
    'harmonizer_chord': _harmonizer_chord,
    # Granular presets (whole-source)
    'granular_cloud': _granular_cloud,
    'granular_scatter': _granular_scatter,
    'granular_stretch': _granular_stretch,
    'granular_freeze': _granular_freeze,
    'granular_shimmer': _granular_shimmer,
    'granular_reverse': _granular_reverse,
    'granular_stutter': _granular_stutter,
    # Utility (normalization, repair, smoothing)
    'util_normalize': _util_normalize,
    'util_normalize_rms': _util_normalize_rms,
    'util_declip': _util_declip,
    'util_declick': _util_declick,
    'util_smooth': _util_smooth,
    'util_smooth_heavy': _util_smooth_heavy,
    'util_dc_remove': _util_dc_remove,
    'util_fade_in': _util_fade_in,
    'util_fade_out': _util_fade_out,
    'util_fade_both': _util_fade_both,
}


# ---------------------------------------------------------------------------
# EFFECT METADATA: Maps effect names to (base_function, default_params)
#
# Each entry is:  effect_name -> (callable_base_func, {param: default, ...})
#
# The base_func accepts (buffer, **params).  When extra params are provided
# in apply_effects_with_params, they override the defaults and are forwarded
# to the base function.  This makes every effect's DSP parameters tunable
# at runtime via the parameter system.
# ---------------------------------------------------------------------------

_effect_metadata: dict[str, tuple[callable, dict]] = {
    # --- Convolution reverbs -> convolve_reverb ---
    'conv_hall':      (convolve_reverb, {'preset': 'hall', 'wet': 40.0, 'dry': 70.0, 'stretch': 1.0, 'pre_delay': 0.0, 'high_cut': 20000.0, 'low_cut': 20.0}),
    'conv_hall_long': (convolve_reverb, {'preset': 'hall_long', 'wet': 50.0, 'dry': 60.0, 'stretch': 1.0, 'pre_delay': 0.0, 'high_cut': 20000.0, 'low_cut': 20.0}),
    'conv_room':      (convolve_reverb, {'preset': 'room', 'wet': 35.0, 'dry': 80.0, 'stretch': 1.0, 'pre_delay': 0.0, 'high_cut': 20000.0, 'low_cut': 20.0}),
    'conv_plate':     (convolve_reverb, {'preset': 'plate', 'wet': 45.0, 'dry': 70.0, 'stretch': 1.0, 'pre_delay': 0.0, 'high_cut': 20000.0, 'low_cut': 20.0}),
    'conv_spring':    (convolve_reverb, {'preset': 'spring', 'wet': 40.0, 'dry': 75.0, 'stretch': 1.0, 'pre_delay': 0.0, 'high_cut': 20000.0, 'low_cut': 20.0}),
    'conv_shimmer':   (convolve_reverb, {'preset': 'shimmer', 'wet': 50.0, 'dry': 60.0, 'stretch': 1.0, 'pre_delay': 0.0, 'high_cut': 20000.0, 'low_cut': 20.0}),
    'conv_reverse':   (convolve_reverb, {'preset': 'reverse', 'wet': 50.0, 'dry': 60.0, 'stretch': 1.0, 'pre_delay': 0.0, 'high_cut': 20000.0, 'low_cut': 20.0}),

    # --- Vamp / Overdrive / Waveshaping -> vamp_process ---
    'vamp_light':        (vamp_process, {'drive': 10.0, 'waveshape': 'tube', 'post_filter': 8000.0, 'gain': 50.0, 'mix': 100.0}),
    'vamp_medium':       (vamp_process, {'drive': 25.0, 'waveshape': 'tube', 'post_filter': 6000.0, 'gain': 50.0, 'mix': 100.0}),
    'vamp_heavy':        (vamp_process, {'drive': 50.0, 'waveshape': 'tube', 'pre_filter': 100.0, 'pre_filter_type': 'hp', 'post_filter': 4000.0, 'gain': 50.0, 'mix': 100.0}),
    'vamp_fuzz':         (vamp_process, {'drive': 75.0, 'waveshape': 'fuzz', 'post_filter': 3000.0, 'gain': 40.0, 'mix': 100.0}),
    'overdrive_soft':    (vamp_process, {'drive': 15.0, 'waveshape': 'cubic', 'post_filter': 10000.0, 'gain': 50.0, 'mix': 100.0}),
    'overdrive_classic': (vamp_process, {'drive': 30.0, 'waveshape': 'tanh', 'pre_filter': 150.0, 'pre_filter_type': 'hp', 'post_filter': 5000.0, 'gain': 50.0, 'mix': 100.0}),
    'overdrive_crunch':  (vamp_process, {'drive': 40.0, 'waveshape': 'tube', 'post_filter': 4000.0, 'gain': 50.0, 'mix': 100.0}),
    'waveshape_fold':    (vamp_process, {'drive': 20.0, 'waveshape': 'fold', 'gain': 60.0, 'mix': 100.0}),
    'waveshape_rectify': (vamp_process, {'drive': 10.0, 'waveshape': 'rectify', 'post_filter': 6000.0, 'mix': 100.0}),
    'waveshape_sine':    (vamp_process, {'drive': 15.0, 'waveshape': 'sine', 'post_filter': 8000.0, 'mix': 100.0}),

    # --- Dual overdrive -> dual_overdrive ---
    'dual_od_warm':   (dual_overdrive, {'drive_low': 15.0, 'drive_high': 20.0, 'shape_low': 'tube', 'shape_high': 'cubic', 'crossover': 600.0, 'blend': 50.0, 'gain': 50.0}),
    'dual_od_bright': (dual_overdrive, {'drive_low': 10.0, 'drive_high': 30.0, 'shape_low': 'cubic', 'shape_high': 'tanh', 'crossover': 1200.0, 'blend': 50.0, 'gain': 50.0}),
    'dual_od_heavy':  (dual_overdrive, {'drive_low': 30.0, 'drive_high': 50.0, 'shape_low': 'tube', 'shape_high': 'fuzz', 'crossover': 400.0, 'blend': 50.0, 'gain': 50.0}),

    # --- Compressor -> compress ---
    'compress_mild':    (compress, {'threshold': 50.0, 'ratio': 40.0, 'makeup': 50.0}),
    'compress_hard':    (compress, {'threshold': 30.0, 'ratio': 70.0, 'makeup': 55.0}),
    'compress_limiter': (compress, {'threshold': 70.0, 'ratio': 100.0, 'makeup': 50.0}),

    # --- Forever Compression -> forever_compression ---
    'fc_punch': (forever_compression, {'depth': 60.0, 'low_amount': 70.0, 'mid_amount': 50.0, 'high_amount': 40.0, 'upward': 30.0, 'downward': 70.0, 'mix': 75.0, 'output': 50.0}),
    'fc_glue':  (forever_compression, {'depth': 35.0, 'low_amount': 45.0, 'mid_amount': 50.0, 'high_amount': 45.0, 'upward': 40.0, 'downward': 40.0, 'mix': 60.0, 'output': 50.0}),
    'fc_loud':  (forever_compression, {'depth': 85.0, 'low_amount': 80.0, 'mid_amount': 90.0, 'high_amount': 85.0, 'upward': 70.0, 'downward': 80.0, 'mix': 100.0, 'output': 60.0}),
    'fc_soft':  (forever_compression, {'depth': 25.0, 'low_amount': 30.0, 'mid_amount': 35.0, 'high_amount': 25.0, 'upward': 50.0, 'downward': 30.0, 'mix': 50.0, 'output': 50.0}),
    'fc_ott':   (forever_compression, {'depth': 100.0, 'low_amount': 100.0, 'mid_amount': 100.0, 'high_amount': 100.0, 'upward': 100.0, 'downward': 100.0, 'mix': 50.0, 'output': 50.0}),

    # --- Gate -> gate ---
    'gate1': (gate, {'threshold': 70.0, 'attack': 1.0, 'release': 10.0}),
    'gate2': (gate, {'threshold': 60.0, 'attack': 1.0, 'release': 10.0}),
    'gate3': (gate, {'threshold': 50.0, 'attack': 1.0, 'release': 10.0}),
    'gate4': (gate, {'threshold': 40.0, 'attack': 1.0, 'release': 10.0}),
    'gate5': (gate, {'threshold': 25.0, 'attack': 1.0, 'release': 10.0}),

    # --- Giga Gate -> giga_gate ---
    'gg_half':     (giga_gate, {'pattern': 'half', 'shape': 'square', 'steps': 16, 'attack': 1.0, 'release': 5.0, 'mix': 100.0}),
    'gg_quarter':  (giga_gate, {'pattern': 'quarter', 'shape': 'square', 'steps': 16, 'attack': 1.0, 'release': 5.0, 'mix': 100.0}),
    'gg_tresillo': (giga_gate, {'pattern': 'tresillo', 'shape': 'sine', 'steps': 16, 'attack': 1.0, 'release': 5.0, 'mix': 100.0}),
    'gg_glitch':   (giga_gate, {'pattern': 'glitch1', 'shape': 'exp', 'steps': 16, 'attack': 1.0, 'release': 5.0, 'mix': 100.0}),

    # --- Spectral effects ---
    'spc_freeze':     (spectral_freeze, {'position': 0.5}),
    'spc_blur':       (spectral_blur, {'amount': 50.0}),
    'spc_shift_up':   (spectral_shift, {'semitones': 7.0}),
    'spc_shift_down': (spectral_shift, {'semitones': -7.0}),

    # --- Delays (inline — metadata for parameter exposure, wrapper used at runtime) ---
    'delay_simple':   (None, {'delay_time': 0.12, 'feedback': 0.3, 'wet': 0.5}),
    'delay_pingpong': (None, {'delay_time': 0.15, 'feedback': 0.4, 'wet': 0.5}),
    'delay_multitap': (None, {'delay_time': 0.1, 'feedback': 0.5, 'wet': 0.5}),
    'delay_slapback': (None, {'delay_time': 0.08, 'feedback': 0.25, 'wet': 0.6}),
    'delay_tape':     (None, {'delay_time': 0.3, 'feedback': 0.35, 'wet': 0.6}),

    # --- Basic reverbs (inline) ---
    'reverb_small':     (None, {'decay': 5.0, 'wet': 0.5}),
    'reverb_large':     (None, {'decay': 2.0, 'wet': 0.5}),
    'reverb_plate':     (None, {'decay': 3.0, 'wet': 0.4}),
    'reverb_spring':    (None, {'decay': 4.0, 'wet': 0.4}),
    'reverb_cathedral': (None, {'decay': 1.5, 'wet': 0.5}),

    # --- Saturations (inline) ---
    'saturate_soft':      (None, {'drive': 30.0}),
    'saturate_hard':      (None, {'drive': 60.0}),
    'saturate_overdrive': (None, {'drive': 50.0}),
    'saturate_fuzz':      (None, {'drive': 80.0}),
    'saturate_tube':      (None, {'drive': 40.0}),
}


def get_effect_dsp_params(effect_name: str) -> dict:
    """Get the controllable DSP parameters and their defaults for an effect.

    Returns a dict of {param_name: default_value} for all numeric params.
    Only includes params that are meaningful to expose (skips string-type
    params like waveshape/preset which are choices, not continuous).

    For effects without metadata, returns {'amount': 50.0} as the only
    controllable parameter (wet/dry mix).
    """
    meta = _effect_metadata.get(effect_name)
    if meta is None:
        return {'amount': 50.0}

    _base_func, defaults = meta
    # Filter to numeric params only (skip strings like waveshape, preset)
    numeric = {}
    for k, v in defaults.items():
        if isinstance(v, (int, float)):
            numeric[k] = float(v)
    # Always include amount as first param
    result = {'amount': 50.0}
    result.update(numeric)
    return result


def _get_param_range(param_name: str, default: float) -> tuple[float, float]:
    """Infer parameter range from name and default value."""
    n = param_name.lower()
    if any(k in n for k in ('freq', 'cutoff', 'crossover', 'pre_filter', 'post_filter', 'high_cut', 'low_cut')):
        return (20.0, 20000.0)
    if any(k in n for k in ('delay_time', 'pre_delay')):
        return (0.0, 2.0)
    if n in ('stretch',):
        return (0.1, 4.0)
    if any(k in n for k in ('drive', 'amount', 'wet', 'dry', 'blend', 'mix', 'gain',
                              'threshold', 'ratio', 'makeup', 'depth', 'output',
                              'low_amount', 'mid_amount', 'high_amount',
                              'upward', 'downward', 'lfo_depth', 'position',
                              'bias')):
        return (0.0, 100.0)
    if any(k in n for k in ('feedback', 'lp_coef', 'decay')):
        return (0.0, 1.0)
    if any(k in n for k in ('attack', 'release')):
        return (0.1, 100.0)
    if 'steps' in n:
        return (1.0, 64.0)
    if 'semitones' in n:
        return (-24.0, 24.0)
    if 'lfo_rate' in n:
        return (0.01, 20.0)
    if 'bits' in n:
        return (1.0, 100.0)
    if 'downsample' in n:
        return (0.0, 100.0)
    # Fallback
    if 0.0 <= default <= 1.0:
        return (0.0, 1.0)
    if 0.0 <= default <= 100.0:
        return (0.0, 100.0)
    return (min(0.0, default * 0.5), max(default * 2, 1.0))


def apply_effects(
    buffer: np.ndarray,
    effect_names: list[str],
    *,
    filter_type: int | None = None,
    cutoff: float | None = None,
    resonance: float | None = None,
) -> np.ndarray:
    """Apply a sequence of effects and optional filtering to a buffer.

    Parameters
    ----------
    buffer : np.ndarray
        The input audio buffer (1D float64 array).
    effect_names : list[str]
        A list of effect names to apply.  Names must be keys in
        ``_effect_funcs``.  Unknown names are skipped.
    filter_type : int, optional
        If provided along with cutoff, a filter is applied after
        each effect.  This allows pre/post filtering around effects.
    cutoff : float, optional
        Cutoff frequency for the filter in Hz.  Ignored if
        ``filter_type`` is None.
    resonance : float, optional
        Resonance/Q parameter for the filter.  Ignored if
        ``filter_type`` is None.

    Returns
    -------
    np.ndarray
        The processed audio buffer.
    """
    out = buffer.astype(np.float64)
    for name in effect_names:
        func = _effect_funcs.get(name)
        if func is None:
            continue
        try:
            out = func(out)
        except Exception:
            # Skip effect if it fails
            continue
        # Apply filter after each effect if parameters provided
        if filter_type is not None and cutoff is not None and cutoff > 0:
            out = _apply_filter(out, filter_type, cutoff, resonance or 0.0)
    return out


def apply_effects_with_params(
    buffer: np.ndarray,
    effect_names: list[str],
    effect_params: list[dict] = None,
    *,
    filter_type: int | None = None,
    cutoff: float | None = None,
    resonance: float | None = None,
) -> np.ndarray:
    """Apply effects with per-effect parameters.

    Parameters
    ----------
    buffer : np.ndarray
        The input audio buffer (1D float64 array).
    effect_names : list[str]
        A list of effect names to apply.
    effect_params : list[dict], optional
        Parallel list of parameter dicts for each effect.
        - 'amount' (0-100): controls wet/dry mix
        - Any other keys matching the effect's DSP parameters (as listed
          in ``_effect_metadata``) are forwarded to the base function,
          overriding the preset defaults.
    filter_type : int, optional
        Filter type for post-effect filtering.
    cutoff : float, optional
        Cutoff frequency for the filter in Hz.
    resonance : float, optional
        Resonance/Q parameter for the filter.

    Returns
    -------
    np.ndarray
        The processed audio buffer.
    """
    out = buffer.astype(np.float64)

    if effect_params is None:
        effect_params = [{'amount': 50.0}] * len(effect_names)

    # Ensure params list matches effects
    while len(effect_params) < len(effect_names):
        effect_params.append({'amount': 50.0})

    for i, name in enumerate(effect_names):
        func = _effect_funcs.get(name)
        if func is None:
            continue

        params = effect_params[i] if i < len(effect_params) else {'amount': 50.0}
        amount = params.get('amount', 50.0)

        # Convert 0-100 amount to wet/dry mix (0-1)
        wet = amount / 100.0
        dry = 1.0 - wet

        try:
            # Store dry signal
            dry_signal = out.copy()

            # Check if we have metadata with a callable base function and
            # the user has overridden any DSP params beyond 'amount'.
            meta = _effect_metadata.get(name)
            dsp_overrides = {k: v for k, v in params.items()
                            if k != 'amount' and isinstance(v, (int, float))}

            if meta and meta[0] is not None and dsp_overrides:
                # Use base function with merged params
                base_func, defaults = meta
                merged = dict(defaults)
                merged.update(dsp_overrides)
                # Only pass params the base function actually accepts
                import inspect
                try:
                    sig = inspect.signature(base_func)
                    valid_keys = set(sig.parameters.keys()) - {'buffer', 'self'}
                    call_params = {k: v for k, v in merged.items() if k in valid_keys}
                except (ValueError, TypeError):
                    call_params = merged
                wet_signal = base_func(out, **call_params)
            else:
                # Use the preset wrapper function as-is
                wet_signal = func(out)

            # Mix based on amount
            if amount >= 100.0:
                out = wet_signal
            elif amount <= 0.0:
                pass  # Keep dry signal
            else:
                out = dry * dry_signal + wet * wet_signal

        except Exception:
            # Skip effect if it fails
            continue

        # Apply filter after each effect if parameters provided
        if filter_type is not None and cutoff is not None and cutoff > 0:
            out = _apply_filter(out, filter_type, cutoff, resonance or 0.0)
    
    # Normalize if needed
    max_val = np.max(np.abs(out))
    if max_val > 1.0:
        out = out / max_val
    
    return out


# ---------------------------------------------------------------------------
# DIRECT EFFECT APPLICATION (file/buffer helpers)
# ---------------------------------------------------------------------------
# These functions apply effects directly to files or buffers without
# going through the track/session system. Useful for quick conversions.

def fx_buffer(
    buffer: np.ndarray,
    effects: list[str] | str,
    amount: float = 100.0,
    sr: int = None,
) -> np.ndarray:
    """Apply effects directly to a buffer.
    
    Quick helper for instant effect application without tracks.
    
    Parameters
    ----------
    buffer : np.ndarray
        Input audio buffer (1D or 2D). If stereo, processes each channel.
    effects : list[str] or str
        Effect name(s) to apply. Can be single string or list.
    amount : float
        Effect amount/mix (0-100, default 100 = fully wet)
    sr : int, optional
        Sample rate (for reference, not currently used but reserved)
    
    Returns
    -------
    np.ndarray
        Processed buffer (same shape as input)
    
    Examples
    --------
    >>> out = fx_buffer(audio, 'vamp_medium')
    >>> out = fx_buffer(audio, ['compress_mild', 'reverb_plate'], amount=75)
    """
    # Normalize effects to list
    if isinstance(effects, str):
        effects = [effects]
    
    # Handle stereo
    if buffer.ndim == 2:
        out = np.zeros_like(buffer)
        for ch in range(buffer.shape[1]):
            out[:, ch] = fx_buffer(buffer[:, ch], effects, amount, sr)
        return out
    
    # Apply effects with amount control
    params = [{'amount': amount}] * len(effects)
    return apply_effects_with_params(buffer.astype(np.float64), effects, params)


def fx_file(
    input_path: str,
    effects: list[str] | str,
    output_path: str = None,
    amount: float = 100.0,
) -> str:
    """Apply effects directly to an audio file.
    
    Quick helper for instant file conversion without tracks.
    
    Parameters
    ----------
    input_path : str
        Path to input audio file (WAV)
    effects : list[str] or str
        Effect name(s) to apply
    output_path : str, optional
        Path for output file. If None, appends '_fx' to input name.
    amount : float
        Effect amount/mix (0-100, default 100)
    
    Returns
    -------
    str
        Path to output file
    
    Examples
    --------
    >>> fx_file('drums.wav', 'vamp_heavy')
    >>> fx_file('pad.wav', ['reverb_plate', 'compress_mild'], 'pad_processed.wav')
    """
    import wave
    from pathlib import Path
    
    input_path = Path(input_path)
    
    # Read input file
    with wave.open(str(input_path), 'rb') as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    
    # Convert to float
    if sampwidth == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float64) / 128.0 - 1.0
    
    # Reshape for stereo
    if channels == 2:
        data = data.reshape(-1, 2)
    
    # Apply effects
    processed = fx_buffer(data, effects, amount, sr)
    
    # Determine output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_fx{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # Convert back to int16
    out_int = (np.clip(processed, -1.0, 1.0) * 32767).astype(np.int16)
    out_bytes = out_int.tobytes()
    
    # Write output file
    with wave.open(str(output_path), 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # Always 16-bit output
        wf.setframerate(sr)
        wf.writeframes(out_bytes)
    
    return str(output_path)


def list_effects() -> list[str]:
    """Return sorted list of all available effect names."""
    return sorted(_effect_funcs.keys())


def effect_exists(name: str) -> bool:
    """Check if an effect name exists."""
    return name in _effect_funcs


# =============================================================================
# HIGH-QUALITY RENDER CHAIN
# =============================================================================

def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Remove DC offset from audio.
    
    Uses a high-pass filter at very low frequency (1Hz) to remove
    any DC component while preserving all audible content.
    """
    if audio is None or len(audio) == 0:
        return audio
    
    audio = np.asarray(audio, dtype=np.float64)
    
    if audio.ndim == 2:
        # Stereo
        left = audio[:, 0] - np.mean(audio[:, 0])
        right = audio[:, 1] - np.mean(audio[:, 1])
        return np.column_stack([left, right])
    else:
        # Mono
        return audio - np.mean(audio)


def hq_highpass(audio: np.ndarray, cutoff: float = 20.0, sr: int = SAMPLE_RATE, 
                order: int = 4) -> np.ndarray:
    """High-quality highpass filter for subsonic removal.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio
    cutoff : float
        Cutoff frequency in Hz (default 20Hz for subsonic removal)
    sr : int
        Sample rate
    order : int
        Filter order (higher = steeper rolloff)
    
    Returns
    -------
    np.ndarray
        Filtered audio with subsonic frequencies removed
    """
    if audio is None or len(audio) == 0:
        return audio
    
    try:
        from scipy.signal import butter, sosfilt
        
        # Design Butterworth highpass
        nyquist = sr / 2
        normalized_cutoff = cutoff / nyquist
        
        # Ensure cutoff is valid
        if normalized_cutoff >= 1.0:
            return audio
        if normalized_cutoff <= 0.0:
            return audio
        
        sos = butter(order, normalized_cutoff, btype='highpass', output='sos')
        
        audio = np.asarray(audio, dtype=np.float64)
        
        if audio.ndim == 2:
            left = sosfilt(sos, audio[:, 0])
            right = sosfilt(sos, audio[:, 1])
            return np.column_stack([left, right])
        else:
            return sosfilt(sos, audio)
            
    except ImportError:
        # Fallback: simple DC removal
        return remove_dc_offset(audio)


def gentle_highshelf(audio: np.ndarray, freq: float = 16000.0, 
                     gain_db: float = -1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply gentle high-shelf filter for smoother high-end.
    
    Rolls off frequencies above the specified frequency with a subtle
    reduction, creating a more "analog" sound.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio
    freq : float
        Shelf frequency in Hz (default 16kHz)
    gain_db : float
        Gain adjustment in dB (negative for reduction)
    sr : int
        Sample rate
    
    Returns
    -------
    np.ndarray
        Audio with smoothed high-end
    """
    if audio is None or len(audio) == 0:
        return audio
    
    try:
        from scipy.signal import butter, sosfilt
        
        # Simple approach: lowpass blend
        # More sophisticated shelf would use biquad, but this works well
        nyquist = sr / 2
        normalized_freq = min(freq / nyquist, 0.99)
        
        if normalized_freq <= 0:
            return audio
        
        # Create lowpass for blending
        sos = butter(2, normalized_freq, btype='lowpass', output='sos')
        
        audio = np.asarray(audio, dtype=np.float64)
        
        # Amount of high-frequency reduction
        reduction = 10 ** (gain_db / 20)  # Convert dB to linear
        blend = 1.0 - reduction  # How much of the lowpassed signal to mix
        
        if audio.ndim == 2:
            lp_left = sosfilt(sos, audio[:, 0])
            lp_right = sosfilt(sos, audio[:, 1])
            
            # Blend original with lowpassed
            left = audio[:, 0] * reduction + lp_left * blend
            right = audio[:, 1] * reduction + lp_right * blend
            return np.column_stack([left, right])
        else:
            lp = sosfilt(sos, audio)
            return audio * reduction + lp * blend
            
    except ImportError:
        return audio


def soft_saturation(audio: np.ndarray, drive: float = 0.1) -> np.ndarray:
    """Apply soft saturation for analog-style warmth.
    
    Uses a soft-clipping curve (tanh) to add subtle harmonic content
    without harsh digital clipping.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio
    drive : float
        Saturation amount (0.0 = none, 1.0 = heavy)
    
    Returns
    -------
    np.ndarray
        Audio with subtle harmonic saturation
    """
    if audio is None or len(audio) == 0:
        return audio
    
    if drive <= 0:
        return audio
    
    audio = np.asarray(audio, dtype=np.float64)
    
    # Scale drive to usable range
    drive_scaled = 1.0 + drive * 5.0  # 1.0 to 6.0
    
    # Soft clip using tanh
    saturated = np.tanh(audio * drive_scaled) / np.tanh(drive_scaled)
    
    # Blend with original (subtle)
    blend = min(drive, 0.5)  # Max 50% saturation blend
    return audio * (1.0 - blend) + saturated * blend


def soft_limiter(audio: np.ndarray, threshold: float = 0.95, 
                 knee: float = 0.1) -> np.ndarray:
    """Soft limiter to prevent clipping.
    
    Uses a soft-knee limiting curve to gently reduce peaks above
    the threshold without harsh digital clipping.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio
    threshold : float
        Limiting threshold (0.0 to 1.0)
    knee : float
        Soft knee width (0.0 = hard, 0.5 = very soft)
    
    Returns
    -------
    np.ndarray
        Limited audio
    """
    if audio is None or len(audio) == 0:
        return audio
    
    audio = np.asarray(audio, dtype=np.float64)
    
    # Calculate envelope
    abs_audio = np.abs(audio)
    
    # Soft knee limiting
    knee_start = threshold - knee
    knee_end = threshold + knee
    
    # Create gain reduction curve
    gain = np.ones_like(abs_audio)
    
    # Below knee: no reduction
    # In knee: gradual reduction  
    # Above knee: full limiting
    
    in_knee = (abs_audio > knee_start) & (abs_audio <= knee_end)
    above_knee = abs_audio > knee_end
    
    if knee > 0:
        # Soft knee compression
        knee_factor = (abs_audio[in_knee] - knee_start) / (2 * knee)
        gain[in_knee] = 1.0 - knee_factor * (1.0 - threshold / abs_audio[in_knee])
    
    # Hard limiting above knee
    gain[above_knee] = threshold / abs_audio[above_knee]
    
    return audio * gain


def hq_render_chain(audio: np.ndarray, sr: int = SAMPLE_RATE,
                    dc_remove: bool = True,
                    subsonic_filter: bool = True,
                    subsonic_freq: float = 20.0,
                    highend_smooth: bool = True,
                    highend_freq: float = 16000.0,
                    highend_reduction: float = -1.5,
                    saturation: bool = True,
                    saturation_drive: float = 0.1,
                    limiting: bool = True,
                    limit_threshold: float = 0.95) -> np.ndarray:
    """Apply complete high-quality render chain.
    
    Processes audio through a mastering-style chain:
    1. DC offset removal
    2. Subsonic filtering (20Hz highpass)
    3. High-end smoothing (gentle shelf at 16kHz)
    4. Soft saturation (analog warmth)
    5. Soft limiting (prevent clipping)
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio
    sr : int
        Sample rate
    dc_remove : bool
        Enable DC offset removal
    subsonic_filter : bool
        Enable subsonic filtering
    subsonic_freq : float
        Subsonic filter cutoff (Hz)
    highend_smooth : bool
        Enable high-end smoothing
    highend_freq : float
        High-end shelf frequency (Hz)
    highend_reduction : float
        High-end reduction (dB, negative)
    saturation : bool
        Enable soft saturation
    saturation_drive : float
        Saturation amount (0-1)
    limiting : bool
        Enable soft limiting
    limit_threshold : float
        Limiter threshold (0-1)
    
    Returns
    -------
    np.ndarray
        Processed audio
    """
    if audio is None or len(audio) == 0:
        return audio
    
    result = np.asarray(audio, dtype=np.float64)
    
    # 1. DC offset removal
    if dc_remove:
        result = remove_dc_offset(result)
    
    # 2. Subsonic filtering
    if subsonic_filter:
        result = hq_highpass(result, cutoff=subsonic_freq, sr=sr)
    
    # 3. High-end smoothing
    if highend_smooth:
        result = gentle_highshelf(result, freq=highend_freq, 
                                  gain_db=highend_reduction, sr=sr)
    
    # 4. Soft saturation
    if saturation:
        result = soft_saturation(result, drive=saturation_drive)
    
    # 5. Soft limiting
    if limiting:
        result = soft_limiter(result, threshold=limit_threshold)
    
    return result


# Register HQ effects in the effect registry
_effect_funcs['hq_dc'] = remove_dc_offset
_effect_funcs['hq_subsonic'] = lambda buf: hq_highpass(buf, cutoff=20.0)
_effect_funcs['hq_smooth'] = lambda buf: gentle_highshelf(buf, freq=16000, gain_db=-1.5)
_effect_funcs['hq_warm'] = lambda buf: soft_saturation(buf, drive=0.15)
_effect_funcs['hq_limit'] = lambda buf: soft_limiter(buf, threshold=0.95)
_effect_funcs['hq_master'] = lambda buf: hq_render_chain(buf)