"""Monolith Offline Synth Engine - Full Implementation.

This engine implements:
- Operator-based architecture with configurable wave types
- Full modulation routing: FM, TFM, AM, RM, PM
- Voice algorithm system with detune, random, stereo spread, phase offset
- Extended wave models: sine, triangle, saw, pulse/PWM, noise (white/pink), physical modeling
- Phase 2 extensions: supersaw, additive, formant, waveguide (string/tube/membrane/plate),
  harmonic (odd/even independent), wavetable import, compound wave creation
- Deterministic offline rendering
- Per-operator parameters and multi-voice stacking

PARAMETER SCALING:
-----------------
Abstract parameters use unified 1-100 scaling:
- rand (amplitude variation): 0-100
- mod (modulation scaling): 0-100
- stereo_spread: 0-100
- resonance: 0-100

Real-world units preserved:
- dt (detune): Hz
- phase_spread: radians
- cutoff: Hz
- freq: Hz

Section A1-A4 of MDMA Master Feature List v1.1
BUILD ID: monolith_v15_phase2
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, List, Tuple, Any

# Import unified scaling
from .scaling import (
    scale_to_range,
    scale_wet,
    scale_resonance,
    scale_modulation_index,
    clamp_param,
)


# ============================================================================
# WAVE GENERATORS
# ============================================================================

def _generate_sine(t: np.ndarray, freq: float, amp: float, phase: float) -> np.ndarray:
    """Generate sine wave."""
    return amp * np.sin(2 * np.pi * freq * t + phase)


def _generate_triangle(t: np.ndarray, freq: float, amp: float, phase: float) -> np.ndarray:
    """Generate triangle wave using absolute value of sawtooth."""
    # Phase-shifted sawtooth folded into triangle
    saw_phase = (freq * t + phase / (2 * np.pi)) % 1.0
    return amp * (2 * np.abs(2 * saw_phase - 1) - 1)


def _generate_saw(t: np.ndarray, freq: float, amp: float, phase: float) -> np.ndarray:
    """Generate sawtooth wave (carrier-only recommended)."""
    saw_phase = (freq * t + phase / (2 * np.pi)) % 1.0
    return amp * (2 * saw_phase - 1)


def _generate_saw_bandlimited(t: np.ndarray, freq: float, amp: float, phase: float,
                               sr: int = 48000) -> np.ndarray:
    """Generate band-limited sawtooth using additive synthesis.
    
    Creates a smoother, alias-free sawtooth by summing harmonics up to
    the Nyquist frequency. This sounds more natural and analog-like.
    
    Parameters
    ----------
    t : np.ndarray
        Time array
    freq : float
        Fundamental frequency in Hz
    amp : float
        Amplitude (0-1)
    phase : float
        Phase offset in radians
    sr : int
        Sample rate for Nyquist calculation
    """
    nyquist = sr / 2
    max_harmonic = int(nyquist / max(freq, 1)) 
    max_harmonic = min(max_harmonic, 64)  # Cap for performance
    
    if max_harmonic < 1:
        return np.zeros_like(t)
    
    wave = np.zeros_like(t, dtype=np.float64)
    
    # Sawtooth: sum of sin(k*x)/k for k=1,2,3...
    for k in range(1, max_harmonic + 1):
        wave += ((-1) ** (k + 1)) * np.sin(2 * np.pi * k * freq * t + k * phase) / k
    
    # Scale to [-1, 1] range
    return amp * wave * (2 / np.pi)


def _generate_square_bandlimited(t: np.ndarray, freq: float, amp: float, phase: float,
                                  sr: int = 48000) -> np.ndarray:
    """Generate band-limited square wave using additive synthesis.
    
    Creates a smoother, alias-free square wave by summing odd harmonics
    up to the Nyquist frequency.
    
    Parameters
    ----------
    t : np.ndarray
        Time array
    freq : float
        Fundamental frequency in Hz
    amp : float
        Amplitude (0-1)
    phase : float
        Phase offset in radians
    sr : int
        Sample rate for Nyquist calculation
    """
    nyquist = sr / 2
    max_harmonic = int(nyquist / max(freq, 1))
    max_harmonic = min(max_harmonic, 64)  # Cap for performance
    
    if max_harmonic < 1:
        return np.zeros_like(t)
    
    wave = np.zeros_like(t, dtype=np.float64)
    
    # Square: sum of sin((2k-1)*x)/(2k-1) for k=1,2,3... (odd harmonics only)
    for k in range(1, max_harmonic + 1, 2):  # 1, 3, 5, 7...
        wave += np.sin(2 * np.pi * k * freq * t + k * phase) / k
    
    # Scale to [-1, 1] range
    return amp * wave * (4 / np.pi)


def _generate_triangle_bandlimited(t: np.ndarray, freq: float, amp: float, phase: float,
                                    sr: int = 48000) -> np.ndarray:
    """Generate band-limited triangle wave using additive synthesis.
    
    Creates a smoother, alias-free triangle wave by summing odd harmonics
    with alternating signs and 1/k^2 falloff.
    
    Parameters
    ----------
    t : np.ndarray
        Time array
    freq : float
        Fundamental frequency in Hz
    amp : float
        Amplitude (0-1)
    phase : float
        Phase offset in radians
    sr : int
        Sample rate for Nyquist calculation
    """
    nyquist = sr / 2
    max_harmonic = int(nyquist / max(freq, 1))
    max_harmonic = min(max_harmonic, 64)  # Cap for performance
    
    if max_harmonic < 1:
        return np.zeros_like(t)
    
    wave = np.zeros_like(t, dtype=np.float64)
    
    # Triangle: sum of (-1)^((k-1)/2) * sin(k*x)/k^2 for k=1,3,5...
    sign = 1
    for k in range(1, max_harmonic + 1, 2):  # 1, 3, 5, 7...
        wave += sign * np.sin(2 * np.pi * k * freq * t + k * phase) / (k * k)
        sign *= -1
    
    # Scale to [-1, 1] range
    return amp * wave * (8 / (np.pi * np.pi))


def _generate_pulse(t: np.ndarray, freq: float, amp: float, phase: float, 
                    pw: float = 0.5) -> np.ndarray:
    """Generate pulse wave with variable pulse width (carrier-only).
    
    Parameters
    ----------
    pw : float
        Pulse width from 0.0 to 1.0 (0.5 = square wave)
    """
    pw = max(0.01, min(0.99, pw))  # Clamp to avoid DC
    pulse_phase = (freq * t + phase / (2 * np.pi)) % 1.0
    return amp * np.where(pulse_phase < pw, 1.0, -1.0)


def _generate_white_noise(n_samples: int, amp: float, seed: Optional[int] = None) -> np.ndarray:
    """Generate white noise."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    return amp * (rng.random(n_samples) * 2 - 1)


def _generate_pink_noise(n_samples: int, amp: float, seed: Optional[int] = None) -> np.ndarray:
    """Generate pink noise using Voss-McCartney algorithm.
    
    Pink noise has equal energy per octave (-3dB/octave rolloff).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Number of random sources (octaves)
    n_rows = 16
    
    # Initialize
    rows = np.zeros(n_rows)
    running_sum = 0.0
    max_val = n_rows
    
    out = np.zeros(n_samples, dtype=np.float64)
    
    for i in range(n_samples):
        # Determine which rows to update based on binary counter
        idx = i
        for j in range(n_rows):
            if idx & 1:
                running_sum -= rows[j]
                rows[j] = rng.random() * 2 - 1
                running_sum += rows[j]
            idx >>= 1
            if idx == 0:
                break
        
        # Add white noise component for high frequencies
        white = rng.random() * 2 - 1
        out[i] = (running_sum + white) / (max_val + 1)
    
    # Normalize and apply amplitude
    max_out = np.max(np.abs(out))
    if max_out > 0:
        out = out / max_out
    
    return amp * out


def _generate_physical(t: np.ndarray, freq: float, amp: float, phase: float,
                       even_harmonics: int = 8, odd_harmonics: int = 4,
                       even_weight: float = 1.0, decay: float = 0.7) -> np.ndarray:
    """Generate physical modeling wave with harmonic control.
    
    Physical modeling variant 1: Even-harmonics emphasis around fundamental.
    Creates bell-like, marimba-like, or wood-block-like tones.
    
    Parameters
    ----------
    even_harmonics : int
        Number of even harmonics to include (2, 4, 6, 8...)
    odd_harmonics : int
        Number of odd harmonics to include (3, 5, 7...)
    even_weight : float
        Relative weight of even harmonics (1.0 = equal to odd)
    decay : float
        Amplitude decay per harmonic (0.5 = -6dB per harmonic)
    """
    out = np.zeros_like(t)
    
    # Fundamental
    out += np.sin(2 * np.pi * freq * t + phase)
    
    # Even harmonics (2, 4, 6, 8...)
    for i in range(1, even_harmonics + 1):
        harmonic = 2 * i
        harmonic_amp = even_weight * (decay ** (harmonic - 1))
        out += harmonic_amp * np.sin(2 * np.pi * freq * harmonic * t + phase * harmonic)
    
    # Odd harmonics (3, 5, 7...)
    for i in range(1, odd_harmonics + 1):
        harmonic = 2 * i + 1
        harmonic_amp = decay ** (harmonic - 1)
        out += harmonic_amp * np.sin(2 * np.pi * freq * harmonic * t + phase * harmonic)
    
    # Normalize
    max_val = np.max(np.abs(out))
    if max_val > 0:
        out = out / max_val
    
    return amp * out


def _generate_physical2(t: np.ndarray, freq: float, amp: float, phase: float,
                        inharmonicity: float = 0.01, partials: int = 12,
                        decay_curve: str = 'exp') -> np.ndarray:
    """Generate physical modeling wave variant 2 with inharmonicity.
    
    Physical modeling variant 2: Stretched partials like piano strings.
    Creates more metallic, piano-like, or bell-like tones with beating.
    
    Parameters
    ----------
    inharmonicity : float
        Amount of partial stretching (0.0 = harmonic, 0.05 = very stretched)
    partials : int
        Number of partials to generate
    decay_curve : str
        'exp' for exponential decay, 'linear' for linear, 'sqrt' for sqrt
    """
    out = np.zeros_like(t)
    
    for n in range(1, partials + 1):
        # Stretched partial frequency: f_n = n * f0 * sqrt(1 + B * n^2)
        # where B is the inharmonicity coefficient
        stretched_freq = freq * n * np.sqrt(1 + inharmonicity * n * n)
        
        # Amplitude decay based on curve type
        if decay_curve == 'exp':
            partial_amp = np.exp(-0.3 * (n - 1))
        elif decay_curve == 'linear':
            partial_amp = 1.0 / n
        elif decay_curve == 'sqrt':
            partial_amp = 1.0 / np.sqrt(n)
        else:
            partial_amp = np.exp(-0.3 * (n - 1))
        
        # Randomize phase slightly for natural sound
        partial_phase = phase + (n - 1) * 0.1
        
        out += partial_amp * np.sin(2 * np.pi * stretched_freq * t + partial_phase)
    
    # Normalize
    max_val = np.max(np.abs(out))
    if max_val > 0:
        out = out / max_val
    
    return amp * out


# ============================================================================
# PHASE 2: EXTENDED WAVE GENERATORS
# ============================================================================

def _generate_supersaw(t: np.ndarray, freq: float, amp: float, phase: float,
                       num_saws: int = 7, detune_spread: float = 0.5,
                       mix: float = 0.75, sr: int = 48000) -> np.ndarray:
    """Generate supersaw wave (JP-8000 style).

    Stacks multiple detuned sawtooth oscillators for a rich, wide sound.

    Parameters
    ----------
    num_saws : int
        Number of sawtooth oscillators (3-11, default 7)
    detune_spread : float
        Detune spread in semitones (0.0-2.0)
    mix : float
        Balance between center saw and detuned stack (0=center only, 1=stack only)
    """
    num_saws = max(3, min(11, num_saws))
    out = np.zeros_like(t)

    for i in range(num_saws):
        # Spread detuning symmetrically
        offset = (i - (num_saws - 1) / 2) / max(1, (num_saws - 1) / 2)
        detune_semitones = offset * detune_spread
        saw_freq = freq * (2.0 ** (detune_semitones / 12.0))
        saw_phase = phase + i * 0.7  # Spread initial phase

        if i == num_saws // 2:
            # Center oscillator
            weight = 1.0 - mix
        else:
            weight = mix / max(1, num_saws - 1)

        saw_val = (saw_freq * t + saw_phase / (2 * np.pi)) % 1.0
        out += weight * (2 * saw_val - 1)

    max_val = np.max(np.abs(out))
    if max_val > 0:
        out = out / max_val
    return amp * out


def _generate_additive(t: np.ndarray, freq: float, amp: float, phase: float,
                       harmonics: Optional[List[Tuple[int, float, float]]] = None,
                       num_harmonics: int = 16, rolloff: float = 1.0) -> np.ndarray:
    """Generate additive synthesis wave from harmonic specification.

    Parameters
    ----------
    harmonics : list of (harmonic_number, amplitude, phase_offset), optional
        Explicit harmonic specification. If None, uses natural rolloff.
    num_harmonics : int
        Number of harmonics when using auto-rolloff (1-64)
    rolloff : float
        Amplitude rolloff exponent (1.0 = 1/n, 2.0 = 1/n^2, 0.5 = 1/sqrt(n))
    """
    out = np.zeros_like(t)

    if harmonics is not None:
        for h_num, h_amp, h_phase in harmonics:
            out += h_amp * np.sin(2 * np.pi * freq * h_num * t + phase * h_num + h_phase)
    else:
        num_harmonics = max(1, min(64, num_harmonics))
        for n in range(1, num_harmonics + 1):
            h_amp = 1.0 / (n ** rolloff)
            out += h_amp * np.sin(2 * np.pi * freq * n * t + phase * n)

    max_val = np.max(np.abs(out))
    if max_val > 0:
        out = out / max_val
    return amp * out


def _generate_formant_wave(t: np.ndarray, freq: float, amp: float, phase: float,
                           vowel: str = 'a') -> np.ndarray:
    """Generate formant-shaped oscillator (vocal-like timbres).

    Parameters
    ----------
    vowel : str
        Vowel shape: 'a', 'e', 'i', 'o', 'u'
    """
    # Formant frequencies and bandwidths for each vowel
    formant_table = {
        'a': [(800, 80, 1.0), (1150, 90, 0.5), (2900, 120, 0.3)],
        'e': [(350, 60, 1.0), (2000, 100, 0.5), (2800, 120, 0.2)],
        'i': [(270, 60, 1.0), (2300, 100, 0.4), (3000, 120, 0.2)],
        'o': [(450, 70, 1.0), (800, 80, 0.5), (2830, 100, 0.15)],
        'u': [(325, 50, 1.0), (700, 60, 0.4), (2530, 100, 0.1)],
    }
    formants = formant_table.get(vowel.lower(), formant_table['a'])

    # Generate carrier as impulse train at fundamental frequency
    carrier_phase = (freq * t + phase / (2 * np.pi)) % 1.0
    carrier = 2 * carrier_phase - 1  # Sawtooth as carrier

    # Shape using formant resonances (additive approach)
    out = np.zeros_like(t)
    for f_freq, f_bw, f_amp in formants:
        # Each formant as a resonant sine burst at the formant frequency
        # modulated by the carrier fundamental
        n_harm = max(1, int(f_freq / max(freq, 1)))
        envelope = np.exp(-np.pi * f_bw * ((t * freq) % 1.0) / freq) if freq > 0 else np.ones_like(t)
        out += f_amp * envelope * np.sin(2 * np.pi * f_freq * t + phase)

    max_val = np.max(np.abs(out))
    if max_val > 0:
        out = out / max_val
    return amp * out


# ── Phase 2 Feature 2.7: Odd/Even Harmonic Model ──

def _generate_harmonic(t: np.ndarray, freq: float, amp: float, phase: float,
                       odd_level: float = 1.0, even_level: float = 1.0,
                       num_harmonics: int = 16, odd_decay: float = 0.8,
                       even_decay: float = 0.8) -> np.ndarray:
    """Generate waveform with independent odd/even harmonic control.

    Odd harmonics produce clarinet/square-wave-like timbres.
    Even harmonics produce warm/tubular organ-like timbres.

    Parameters
    ----------
    odd_level : float
        Overall level of odd harmonics (0.0-2.0)
    even_level : float
        Overall level of even harmonics (0.0-2.0)
    num_harmonics : int
        Total number of harmonics (1-64)
    odd_decay : float
        Per-harmonic decay for odd harmonics (0.1-1.0)
    even_decay : float
        Per-harmonic decay for even harmonics (0.1-1.0)
    """
    out = np.zeros_like(t)
    num_harmonics = max(1, min(64, num_harmonics))

    # Fundamental is harmonic 1 (odd)
    out += odd_level * np.sin(2 * np.pi * freq * t + phase)

    for n in range(2, num_harmonics + 1):
        if n % 2 == 0:
            # Even harmonic
            h_amp = even_level * (even_decay ** (n // 2 - 1))
        else:
            # Odd harmonic
            h_amp = odd_level * (odd_decay ** (n // 2))
        out += h_amp * np.sin(2 * np.pi * freq * n * t + phase * n)

    max_val = np.max(np.abs(out))
    if max_val > 0:
        out = out / max_val
    return amp * out


# ── Phase 2 Feature 2.6: Waveguide Physical Models ──

def _generate_waveguide_string(t: np.ndarray, freq: float, amp: float, phase: float,
                                damping: float = 0.996, brightness: float = 0.5,
                                position: float = 0.28, sr: int = 48000) -> np.ndarray:
    """Karplus-Strong waveguide string model.

    Parameters
    ----------
    damping : float
        String damping (0.9-0.999, higher = longer sustain)
    brightness : float
        Tone brightness (0.0 = dark, 1.0 = bright)
    position : float
        Pluck position (0.0-0.5, affects harmonic content)
    """
    n_samples = len(t)
    delay_len = max(2, int(sr / max(freq, 20)))

    # Initialize delay line with filtered noise burst
    rng = np.random.default_rng()
    delay_line = rng.random(delay_len) * 2 - 1

    # Shape the excitation based on pluck position
    pos_samples = max(1, int(delay_len * position))
    for i in range(delay_len):
        if i < pos_samples:
            delay_line[i] *= np.sin(np.pi * i / pos_samples)
        else:
            delay_line[i] *= np.sin(np.pi * (delay_len - i) / (delay_len - pos_samples))

    # Brightness filter coefficient
    b_coeff = 0.5 + 0.5 * brightness

    out = np.zeros(n_samples, dtype=np.float64)
    ptr = 0

    for i in range(n_samples):
        out[i] = delay_line[ptr]
        next_ptr = (ptr + 1) % delay_len
        # Two-point averaging filter (Karplus-Strong) with brightness control
        filtered = b_coeff * delay_line[ptr] + (1 - b_coeff) * delay_line[next_ptr]
        delay_line[ptr] = filtered * damping
        ptr = next_ptr

    max_val = np.max(np.abs(out))
    if max_val > 0:
        out = out / max_val
    return amp * out


def _generate_waveguide_tube(t: np.ndarray, freq: float, amp: float, phase: float,
                              damping: float = 0.995, reflection: float = 0.7,
                              bore_shape: float = 0.5, sr: int = 48000) -> np.ndarray:
    """Waveguide tube/pipe model (flute/clarinet-like).

    Parameters
    ----------
    damping : float
        Air column damping (0.9-0.999)
    reflection : float
        End reflection coefficient (0.0 = open, 1.0 = closed)
    bore_shape : float
        Bore taper (0.0 = cylindrical/clarinet, 1.0 = conical/oboe)
    """
    n_samples = len(t)
    delay_len = max(2, int(sr / max(freq, 20) / 2))  # Half-wavelength for tube

    # Forward and backward traveling waves
    forward = np.zeros(delay_len, dtype=np.float64)
    backward = np.zeros(delay_len, dtype=np.float64)

    # Excitation: breath noise filtered through the tube
    rng = np.random.default_rng()
    out = np.zeros(n_samples, dtype=np.float64)
    ptr = 0

    # Bore taper affects high harmonics
    taper_filter = 0.5 + 0.5 * (1.0 - bore_shape)

    for i in range(n_samples):
        # Breath excitation (gentle noise)
        excitation = 0.3 * (rng.random() * 2 - 1) if i < n_samples // 4 else 0.0

        next_ptr = (ptr + 1) % delay_len

        # Forward wave with damping
        forward[ptr] = (excitation + backward[ptr] * reflection) * damping

        # Backward wave (reflected)
        backward[next_ptr] = -forward[ptr] * reflection * taper_filter

        # Output is the sum of traveling waves
        out[i] = forward[ptr] + backward[ptr]
        ptr = next_ptr

    max_val = np.max(np.abs(out))
    if max_val > 0:
        out = out / max_val
    return amp * out


def _generate_waveguide_membrane(t: np.ndarray, freq: float, amp: float, phase: float,
                                  tension: float = 0.5, damping: float = 0.995,
                                  strike_pos: float = 0.3, sr: int = 48000) -> np.ndarray:
    """Waveguide membrane/drum model.

    Parameters
    ----------
    tension : float
        Membrane tension (0.0-1.0, affects pitch bend and inharmonicity)
    damping : float
        Membrane damping (0.9-0.999)
    strike_pos : float
        Strike position (0.0 = center, 1.0 = edge)
    """
    n_samples = len(t)
    out = np.zeros(n_samples, dtype=np.float64)

    # Membrane has inharmonic modes: ratios 1.00, 1.59, 2.14, 2.30, 2.65, 2.92
    mode_ratios = [1.00, 1.59, 2.14, 2.30, 2.65, 2.92, 3.16, 3.50]
    mode_amps = [1.0, 0.8, 0.6, 0.5, 0.3, 0.2, 0.15, 0.1]

    # Strike position affects which modes are excited
    for j, (ratio, base_amp) in enumerate(zip(mode_ratios, mode_amps)):
        # Modes near strike position are more excited
        mode_excite = max(0.1, 1.0 - abs(strike_pos - (j / len(mode_ratios))) * 2)
        mode_amp = base_amp * mode_excite

        # Tension affects frequency (higher tension = closer to harmonic)
        adj_ratio = 1.0 + (ratio - 1.0) * (0.5 + 0.5 * tension)
        mode_freq = freq * adj_ratio

        # Each mode decays independently (higher modes decay faster)
        mode_decay = damping ** (1 + j * 0.5)
        decay_env = mode_decay ** (np.arange(n_samples, dtype=np.float64))
        out += mode_amp * decay_env * np.sin(2 * np.pi * mode_freq * t + phase + j * 0.3)

    max_val = np.max(np.abs(out))
    if max_val > 0:
        out = out / max_val
    return amp * out


def _generate_waveguide_plate(t: np.ndarray, freq: float, amp: float, phase: float,
                               thickness: float = 0.5, damping: float = 0.997,
                               material: float = 0.5, sr: int = 48000) -> np.ndarray:
    """Waveguide plate/bar model (vibraphone/marimba-like).

    Parameters
    ----------
    thickness : float
        Plate thickness (0.0-1.0, affects inharmonicity)
    damping : float
        Material damping (0.9-0.999)
    material : float
        Material type (0.0 = wood/soft, 1.0 = metal/bright)
    """
    n_samples = len(t)
    out = np.zeros(n_samples, dtype=np.float64)

    # Plate modes: f_n ~ n^2 (unlike strings where f_n ~ n)
    # This gives the characteristic metallic/bell sound
    inharmonicity = 0.005 + 0.03 * thickness
    num_partials = 8 + int(8 * material)  # Metal has more audible partials

    for n in range(1, num_partials + 1):
        # Plate frequency: f_n = f0 * n * sqrt(1 + B * n^2)
        p_freq = freq * n * np.sqrt(1 + inharmonicity * n * n)

        # Amplitude: metal sustains high partials, wood rolls off faster
        if material > 0.5:
            p_amp = 1.0 / (n ** 0.8)  # Metal: slow rolloff
        else:
            p_amp = 1.0 / (n ** 1.5)  # Wood: fast rolloff

        # Decay: higher partials decay faster; metal decays slower
        base_damp = damping * (1 - 0.002 * n)
        decay_env = (base_damp ** (np.arange(n_samples, dtype=np.float64)))
        p_phase = phase + n * 0.2
        out += p_amp * decay_env * np.sin(2 * np.pi * p_freq * t + p_phase)

    max_val = np.max(np.abs(out))
    if max_val > 0:
        out = out / max_val
    return amp * out


# ── Phase 2 Feature 2.8: Wavetable Playback ──

def _generate_wavetable(t: np.ndarray, freq: float, amp: float, phase: float,
                        frames: Optional[np.ndarray] = None,
                        frame_pos: float = 0.0,
                        sr: int = 48000) -> np.ndarray:
    """Generate sound from a loaded wavetable.

    Parameters
    ----------
    frames : np.ndarray or None
        Wavetable data as (num_frames, frame_size) array.
        If None, falls back to sine.
    frame_pos : float
        Position in wavetable (0.0-1.0, selects which frame to play)
    """
    if frames is None or len(frames) == 0:
        return _generate_sine(t, freq, amp, phase)

    num_frames, frame_size = frames.shape

    # Select frame (with interpolation between adjacent frames)
    pos = max(0.0, min(1.0, frame_pos)) * (num_frames - 1)
    idx_lo = int(pos)
    idx_hi = min(idx_lo + 1, num_frames - 1)
    frac = pos - idx_lo
    frame = frames[idx_lo] * (1 - frac) + frames[idx_hi] * frac

    # Read through the single-cycle waveform at the desired frequency
    n_samples = len(t)
    out = np.zeros(n_samples, dtype=np.float64)
    phase_inc = freq * frame_size / sr
    read_pos = phase / (2 * np.pi) * frame_size

    for i in range(n_samples):
        idx_f = read_pos % frame_size
        idx_int = int(idx_f)
        idx_next = (idx_int + 1) % frame_size
        frac_s = idx_f - idx_int
        out[i] = frame[idx_int] * (1 - frac_s) + frame[idx_next] * frac_s
        read_pos += phase_inc

    max_val = np.max(np.abs(out))
    if max_val > 0:
        out = out / max_val
    return amp * out


# ── Phase 2 Feature 2.9: Compound Wave Creation ──

def _generate_compound(t: np.ndarray, freq: float, amp: float, phase: float,
                       layers: Optional[List[Dict[str, Any]]] = None,
                       morph: float = 0.0,
                       sr: int = 48000) -> np.ndarray:
    """Generate compound waveform by layering or morphing wave types.

    Parameters
    ----------
    layers : list of dicts, optional
        Each dict: {'wave': str, 'amp': float, 'detune': float, 'phase': float}
        If None, defaults to sine+saw morph.
    morph : float
        Morph position between first two layers (0.0 = layer A, 1.0 = layer B).
        Only used in 2-layer mode for smooth transitions.
    """
    if layers is None or len(layers) == 0:
        layers = [
            {'wave': 'sine', 'amp': 1.0, 'detune': 0.0, 'phase': 0.0},
            {'wave': 'saw', 'amp': 1.0, 'detune': 0.0, 'phase': 0.0},
        ]

    # Simple generators for compound layers
    simple_gens = {
        'sine': lambda tt, f, a, p: a * np.sin(2 * np.pi * f * tt + p),
        'triangle': lambda tt, f, a, p: a * (2 * np.abs(2 * ((f * tt + p / (2 * np.pi)) % 1.0) - 1) - 1),
        'saw': lambda tt, f, a, p: a * (2 * ((f * tt + p / (2 * np.pi)) % 1.0) - 1),
        'pulse': lambda tt, f, a, p: a * np.where(((f * tt + p / (2 * np.pi)) % 1.0) < 0.5, 1.0, -1.0),
    }

    if len(layers) == 2 and morph > 0:
        # Morph mode: crossfade between two layers
        a_wave = layers[0].get('wave', 'sine')
        b_wave = layers[1].get('wave', 'saw')
        a_gen = simple_gens.get(a_wave, simple_gens['sine'])
        b_gen = simple_gens.get(b_wave, simple_gens['sine'])

        a_freq = freq * (2.0 ** (layers[0].get('detune', 0.0) / 12.0))
        b_freq = freq * (2.0 ** (layers[1].get('detune', 0.0) / 12.0))
        a_ph = phase + layers[0].get('phase', 0.0)
        b_ph = phase + layers[1].get('phase', 0.0)

        a_out = a_gen(t, a_freq, layers[0].get('amp', 1.0), a_ph)
        b_out = b_gen(t, b_freq, layers[1].get('amp', 1.0), b_ph)

        out = a_out * (1.0 - morph) + b_out * morph
    else:
        # Layer mode: sum all layers
        out = np.zeros_like(t)
        for layer in layers:
            l_wave = layer.get('wave', 'sine')
            gen = simple_gens.get(l_wave, simple_gens['sine'])
            l_freq = freq * (2.0 ** (layer.get('detune', 0.0) / 12.0))
            l_ph = phase + layer.get('phase', 0.0)
            l_amp = layer.get('amp', 1.0)
            out += gen(t, l_freq, l_amp, l_ph)

    max_val = np.max(np.abs(out))
    if max_val > 0:
        out = out / max_val
    return amp * out


# ============================================================================
# WAVE TYPE REGISTRY
# ============================================================================

# Carrier-only wave types (should not be used as modulators)
CARRIER_ONLY_WAVES = {'saw', 'pulse', 'pwm', 'square', 'supersaw'}

# All supported wave types
WAVE_TYPES = {
    'sine', 'sin',
    'triangle', 'tri',
    'saw', 'sawtooth',
    'pulse', 'pwm', 'square',
    'noise', 'white',
    'pink',
    'physical', 'phys',
    'physical2', 'phys2',
    # Phase 2 extensions
    'supersaw', 'ssaw',
    'additive', 'add',
    'formant', 'vowel',
    'harmonic', 'harm',
    'waveguide_string', 'string', 'pluck',
    'waveguide_tube', 'tube', 'pipe',
    'waveguide_membrane', 'membrane', 'drum',
    'waveguide_plate', 'plate', 'bar',
    'wavetable', 'wt',
    'compound', 'comp', 'layer',
}

# Wave type aliases
WAVE_ALIASES = {
    'sin': 'sine',
    'tri': 'triangle',
    'sawtooth': 'saw',
    'pwm': 'pulse',
    'square': 'pulse',
    'white': 'noise',
    'phys': 'physical',
    'phys2': 'physical2',
    # Phase 2 aliases
    'ssaw': 'supersaw',
    'add': 'additive',
    'vowel': 'formant',
    'harm': 'harmonic',
    'string': 'waveguide_string',
    'pluck': 'waveguide_string',
    'tube': 'waveguide_tube',
    'pipe': 'waveguide_tube',
    'membrane': 'waveguide_membrane',
    'drum': 'waveguide_membrane',
    'plate': 'waveguide_plate',
    'bar': 'waveguide_plate',
    'wt': 'wavetable',
    'comp': 'compound',
    'layer': 'compound',
}


# ============================================================================
# MODULATION TYPES
# ============================================================================

MODULATION_TYPES = {
    'FM': 'Frequency Modulation',
    'TFM': 'Through-Zero Frequency Modulation', 
    'AM': 'Amplitude Modulation',
    'RM': 'Ring Modulation',
    'PM': 'Phase Modulation',
}


# ============================================================================
# MONOLITH ENGINE
# ============================================================================

class MonolithEngine:
    """Operator-based offline synthesis engine with full routing support.
    
    Features:
    - Configurable operators with extended wave types
    - Full modulation routing (FM, TFM, AM, RM, PM)
    - Audio-rate interval modulation per operator
    - Audio-rate filter modulation
    - Voice algorithm system for unison/thickness
    - Deterministic seeded rendering
    - Per-operator envelopes (via external envelope application)
    - Preset bank for routing/settings storage
    - Band-limited oscillators for high-quality output
    """

    def __init__(self, sample_rate: int = 48_000) -> None:
        self.sample_rate: int = sample_rate
        
        # HQ mode: use band-limited oscillators
        self.hq_oscillators: bool = True
        
        # Operators: dict[index] -> {wave, freq, amp, phase, pw, ...}
        self.operators: Dict[int, Dict[str, Any]] = {}
        
        # Modulation algorithms: list of (type, source, target, amount)
        self.algorithms: List[Tuple[str, int, int, float]] = []
        
        # Random seed for deterministic rendering
        self.render_seed: Optional[int] = None
        
        # === AUDIO-RATE MODULATION SOURCES (per operator) ===
        # interval_mod: dict[op_idx] -> np.ndarray (semitones at audio rate)
        self.interval_mod: Dict[int, Optional[np.ndarray]] = {}
        
        # filter_mod: np.ndarray (cutoff modulation at audio rate)
        self.filter_mod: Optional[np.ndarray] = None
        self.filter_mod_depth: float = 1.0  # Octaves
        
        # Per-operator filter modulation sources
        self.op_filter_mod: Dict[int, Optional[np.ndarray]] = {}
        
        # === PRESET BANK ===
        self.preset_bank: Dict[int, Dict[str, Any]] = {}
        
        # === INTERVAL MODULATION SETTINGS ===
        self.interval_mod_rate: float = 0.0  # LFO rate in Hz (0 = disabled)
        self.interval_mod_depth: float = 0.0  # Depth in semitones
        self.interval_mod_wave: str = 'sine'  # LFO waveform

        # === PHASE 2: WAVETABLE STORAGE ===
        # wavetables: dict[name] -> np.ndarray of shape (num_frames, frame_size)
        self.wavetables: Dict[str, np.ndarray] = {}

        # === PHASE 2: COMPOUND WAVE STORAGE ===
        # compound_waves: dict[name] -> list of layer dicts
        self.compound_waves: Dict[str, List[Dict[str, Any]]] = {}

    def set_operator(self, idx: int, wave_type: str = 'sine', freq: float = 440.0,
                     amp: float = 1.0, phase: float = 0.0, **kwargs) -> None:
        """Create or update an operator.
        
        Parameters
        ----------
        idx : int
            Operator index
        wave_type : str
            Wave type (sine, triangle, saw, pulse, noise, pink, physical, physical2)
        freq : float
            Frequency in Hz
        amp : float
            Amplitude (0.0 to 1.0+)
        phase : float
            Initial phase in radians
        **kwargs : dict
            Wave-specific parameters:
            - pw: pulse width for pulse wave (0.0-1.0)
            - even_harmonics, odd_harmonics, even_weight, decay: for physical
            - inharmonicity, partials, decay_curve: for physical2
        """
        # Normalize wave type
        wave = wave_type.lower()
        if wave in WAVE_ALIASES:
            wave = WAVE_ALIASES[wave]
        
        self.operators[idx] = {
            'wave': wave,
            'freq': float(freq),
            'amp': float(amp),
            'phase': float(phase),
            # Wave-specific params with defaults
            'pw': kwargs.get('pw', 0.5),
            'even_harmonics': kwargs.get('even_harmonics', 8),
            'odd_harmonics': kwargs.get('odd_harmonics', 4),
            'even_weight': kwargs.get('even_weight', 1.0),
            'decay': kwargs.get('decay', 0.7),
            'inharmonicity': kwargs.get('inharmonicity', 0.01),
            'partials': kwargs.get('partials', 12),
            'decay_curve': kwargs.get('decay_curve', 'exp'),
            # Phase 2: supersaw params
            'num_saws': kwargs.get('num_saws', 7),
            'detune_spread': kwargs.get('detune_spread', 0.5),
            'mix': kwargs.get('mix', 0.75),
            # Phase 2: additive params
            'harmonics': kwargs.get('harmonics', None),
            'num_harmonics': kwargs.get('num_harmonics', 16),
            'rolloff': kwargs.get('rolloff', 1.0),
            # Phase 2: formant params
            'vowel': kwargs.get('vowel', 'a'),
            # Phase 2: harmonic model params
            'odd_level': kwargs.get('odd_level', 1.0),
            'even_level': kwargs.get('even_level', 1.0),
            'odd_decay': kwargs.get('odd_decay', 0.8),
            'even_decay': kwargs.get('even_decay', 0.8),
            # Phase 2: waveguide params
            'damping': kwargs.get('damping', 0.996),
            'brightness': kwargs.get('brightness', 0.5),
            'position': kwargs.get('position', 0.28),
            'reflection': kwargs.get('reflection', 0.7),
            'bore_shape': kwargs.get('bore_shape', 0.5),
            'tension': kwargs.get('tension', 0.5),
            'strike_pos': kwargs.get('strike_pos', 0.3),
            'thickness': kwargs.get('thickness', 0.5),
            'material': kwargs.get('material', 0.5),
            # Phase 2: wavetable params
            'wavetable_name': kwargs.get('wavetable_name', ''),
            'frame_pos': kwargs.get('frame_pos', 0.0),
            # Phase 2: compound params
            'compound_name': kwargs.get('compound_name', ''),
            'layers': kwargs.get('layers', None),
            'morph': kwargs.get('morph', 0.0),
        }

    def set_wave(self, idx: int, wave_type: str, **kwargs) -> None:
        """Change wave type and optionally other parameters for an operator."""
        if idx not in self.operators:
            self.set_operator(idx, wave_type=wave_type, **kwargs)
        else:
            wave = wave_type.lower()
            if wave in WAVE_ALIASES:
                wave = WAVE_ALIASES[wave]
            self.operators[idx]['wave'] = wave
            
            # Update any provided kwargs
            for key, value in kwargs.items():
                if key in self.operators[idx]:
                    self.operators[idx][key] = value

    def set_param(self, idx: int, param: str, value: Any) -> None:
        """Set a specific parameter on an operator."""
        if idx not in self.operators:
            self.set_operator(idx)
        self.operators[idx][param] = value

    def add_algorithm(self, algo_type: str, source: int, target: int, amount: float) -> None:
        """Add a modulation routing.
        
        Parameters
        ----------
        algo_type : str
            Modulation type: FM, TFM, AM, RM, PM
        source : int
            Source operator index (modulator)
        target : int
            Target operator index (carrier)
        amount : float
            Modulation depth (0-100 scale)
            0 = no modulation
            50 = moderate modulation (index ~5)
            100 = heavy modulation (index ~10)
            >100 = wacky territory (allowed, index >10)
        """
        algo = algo_type.upper()
        if algo not in MODULATION_TYPES:
            raise ValueError(f"Unknown modulation type: {algo_type}. "
                           f"Valid types: {', '.join(MODULATION_TYPES.keys())}")
        # Scale 0-100 to 0-10 modulation index (allow wacky >100)
        scaled_amount = scale_modulation_index(clamp_param(amount, allow_wacky=True))
        self.algorithms.append((algo, int(source), int(target), scaled_amount))

    def add_algorithm_raw(self, algo_type: str, source: int, target: int, amount: float) -> None:
        """Add a modulation routing with raw (unscaled) amount.
        
        For advanced users who want direct control over modulation index.
        
        Parameters
        ----------
        algo_type : str
            Modulation type: FM, TFM, AM, RM, PM
        source : int
            Source operator index (modulator)
        target : int
            Target operator index (carrier)
        amount : float
            Raw modulation index (typical range 0-10, but can go higher)
        """
        algo = algo_type.upper()
        if algo not in MODULATION_TYPES:
            raise ValueError(f"Unknown modulation type: {algo_type}. "
                           f"Valid types: {', '.join(MODULATION_TYPES.keys())}")
        self.algorithms.append((algo, int(source), int(target), float(amount)))

    def clear_algorithms(self) -> None:
        """Clear all modulation routings."""
        self.algorithms.clear()

    # =========================================================================
    # AUDIO-RATE MODULATION (per operator)
    # =========================================================================
    
    def set_interval_mod(self, op_idx: int, mod_signal: Optional[np.ndarray], 
                         depth: float = 12.0) -> None:
        """Set audio-rate interval modulation for an operator.
        
        Parameters
        ----------
        op_idx : int
            Operator index
        mod_signal : np.ndarray or None
            Modulation signal (-1 to 1 range, will be scaled by depth)
        depth : float
            Modulation depth in semitones (default 12 = 1 octave)
        """
        if mod_signal is not None:
            # Store normalized signal and depth in operator dict
            if op_idx not in self.operators:
                self.set_operator(op_idx)
            self.operators[op_idx]['interval_mod'] = mod_signal.copy()
            self.operators[op_idx]['interval_mod_depth'] = depth
        else:
            if op_idx in self.operators:
                self.operators[op_idx].pop('interval_mod', None)
                self.operators[op_idx].pop('interval_mod_depth', None)
        self.interval_mod[op_idx] = mod_signal
    
    def set_interval_lfo(self, op_idx: int, rate: float, depth: float, 
                         wave: str = 'sine') -> None:
        """Configure interval modulation LFO for an operator.
        
        Parameters
        ----------
        op_idx : int
            Operator index
        rate : float
            LFO rate in Hz
        depth : float
            Depth in semitones
        wave : str
            LFO waveform (sine, triangle, saw, square)
        """
        if op_idx not in self.operators:
            self.set_operator(op_idx)
        self.operators[op_idx]['interval_lfo_rate'] = rate
        self.operators[op_idx]['interval_lfo_depth'] = depth
        self.operators[op_idx]['interval_lfo_wave'] = wave
    
    def set_filter_mod(self, mod_signal: Optional[np.ndarray], 
                       depth_octaves: float = 2.0) -> None:
        """Set audio-rate filter cutoff modulation.
        
        Parameters
        ----------
        mod_signal : np.ndarray or None
            Modulation signal (-1 to 1 range)
        depth_octaves : float
            Modulation depth in octaves
        """
        self.filter_mod = mod_signal.copy() if mod_signal is not None else None
        self.filter_mod_depth = depth_octaves
    
    def set_op_filter_mod(self, op_idx: int, mod_signal: Optional[np.ndarray],
                          cutoff: float = 2000.0, resonance: float = 0.5) -> None:
        """Set per-operator filter with audio-rate modulation.
        
        Parameters
        ----------
        op_idx : int
            Operator index
        mod_signal : np.ndarray or None
            Cutoff modulation signal (-1 to 1)
        cutoff : float
            Base cutoff frequency in Hz
        resonance : float
            Filter resonance (0-1)
        """
        if op_idx not in self.operators:
            self.set_operator(op_idx)
        
        self.operators[op_idx]['filter_enabled'] = mod_signal is not None
        self.operators[op_idx]['filter_cutoff'] = cutoff
        self.operators[op_idx]['filter_resonance'] = resonance
        self.op_filter_mod[op_idx] = mod_signal.copy() if mod_signal is not None else None
    
    def clear_modulation(self) -> None:
        """Clear all audio-rate modulation sources."""
        self.interval_mod.clear()
        self.filter_mod = None
        self.op_filter_mod.clear()
        for op in self.operators.values():
            op.pop('interval_mod', None)
            op.pop('interval_mod_depth', None)
            op.pop('interval_lfo_rate', None)
            op.pop('interval_lfo_depth', None)
            op.pop('interval_lfo_wave', None)
            op.pop('filter_enabled', None)
            op.pop('filter_cutoff', None)
            op.pop('filter_resonance', None)

    # =========================================================================
    # PRESET BANK
    # =========================================================================
    
    def save_preset(self, slot: int, name: str = '') -> None:
        """Save current engine state to preset slot.
        
        Parameters
        ----------
        slot : int
            Preset slot (0-127)
        name : str
            Optional preset name
        """
        import copy
        self.preset_bank[slot] = {
            'name': name or f'preset_{slot}',
            'operators': copy.deepcopy(self.operators),
            'algorithms': copy.deepcopy(self.algorithms),
            'interval_mod_rate': self.interval_mod_rate,
            'interval_mod_depth': self.interval_mod_depth,
            'interval_mod_wave': self.interval_mod_wave,
            'filter_mod_depth': self.filter_mod_depth,
        }
    
    def load_preset(self, slot: int) -> bool:
        """Load engine state from preset slot.
        
        Parameters
        ----------
        slot : int
            Preset slot to load
            
        Returns
        -------
        bool
            True if loaded successfully
        """
        import copy
        if slot not in self.preset_bank:
            return False
        
        preset = self.preset_bank[slot]
        self.operators = copy.deepcopy(preset.get('operators', {}))
        self.algorithms = copy.deepcopy(preset.get('algorithms', []))
        self.interval_mod_rate = preset.get('interval_mod_rate', 0.0)
        self.interval_mod_depth = preset.get('interval_mod_depth', 0.0)
        self.interval_mod_wave = preset.get('interval_mod_wave', 'sine')
        self.filter_mod_depth = preset.get('filter_mod_depth', 1.0)
        return True
    
    def list_presets(self) -> List[Tuple[int, str]]:
        """List all saved presets.
        
        Returns
        -------
        list
            List of (slot, name) tuples
        """
        return [(slot, preset.get('name', f'preset_{slot}')) 
                for slot, preset in sorted(self.preset_bank.items())]
    
    def delete_preset(self, slot: int) -> bool:
        """Delete a preset.
        
        Parameters
        ----------
        slot : int
            Preset slot to delete
            
        Returns
        -------
        bool
            True if deleted
        """
        if slot in self.preset_bank:
            del self.preset_bank[slot]
            return True
        return False

    # ── Phase 2: Wavetable Management ──

    def load_wavetable(self, name: str, frames: np.ndarray) -> None:
        """Load a wavetable into the engine.

        Parameters
        ----------
        name : str
            Name to reference this wavetable
        frames : np.ndarray
            Wavetable data, shape (num_frames, frame_size)
        """
        if frames.ndim == 1:
            # Single cycle — wrap as 1 frame
            frames = frames.reshape(1, -1)
        self.wavetables[name] = frames.astype(np.float64)

    def load_wavetable_from_file(self, name: str, filepath: str,
                                  frame_size: int = 2048) -> str:
        """Load a wavetable from a .wav file (Serum format).

        Serum wavetables are single .wav files containing multiple
        single-cycle frames concatenated. Each frame is typically
        2048 samples.

        Parameters
        ----------
        name : str
            Name to reference this wavetable
        filepath : str
            Path to .wav file
        frame_size : int
            Samples per frame (default 2048, Serum standard)

        Returns
        -------
        str
            Status message
        """
        try:
            import soundfile as sf
            data, sr = sf.read(filepath, dtype='float64')
            if data.ndim > 1:
                data = data[:, 0]  # Mono

            # Resample to engine rate if needed
            if sr != self.sample_rate:
                from scipy.signal import resample
                ratio = self.sample_rate / sr
                data = resample(data, int(len(data) * ratio))

            total_samples = len(data)
            num_frames = max(1, total_samples // frame_size)

            # Trim to exact frame boundary
            data = data[:num_frames * frame_size]
            frames = data.reshape(num_frames, frame_size)

            # Normalize each frame
            for i in range(num_frames):
                mx = np.max(np.abs(frames[i]))
                if mx > 0:
                    frames[i] /= mx

            self.wavetables[name] = frames
            return f"Loaded wavetable '{name}': {num_frames} frames x {frame_size} samples"
        except Exception as e:
            return f"ERROR: Failed to load wavetable: {e}"

    def list_wavetables(self) -> List[Tuple[str, int, int]]:
        """List all loaded wavetables.

        Returns list of (name, num_frames, frame_size).
        """
        result = []
        for name, frames in self.wavetables.items():
            result.append((name, frames.shape[0], frames.shape[1]))
        return result

    def delete_wavetable(self, name: str) -> bool:
        """Delete a loaded wavetable."""
        if name in self.wavetables:
            del self.wavetables[name]
            return True
        return False

    # ── Phase 2: Compound Wave Management ──

    def create_compound(self, name: str, layers: List[Dict[str, Any]]) -> None:
        """Create a named compound wave definition.

        Parameters
        ----------
        name : str
            Name to reference this compound
        layers : list of dicts
            Each dict: {'wave': str, 'amp': float, 'detune': float, 'phase': float}
        """
        self.compound_waves[name] = layers

    def list_compounds(self) -> List[Tuple[str, int]]:
        """List all compound wave definitions.

        Returns list of (name, num_layers).
        """
        return [(name, len(layers)) for name, layers in self.compound_waves.items()]

    def delete_compound(self, name: str) -> bool:
        """Delete a compound wave definition."""
        if name in self.compound_waves:
            del self.compound_waves[name]
            return True
        return False

    def get_wave_info(self) -> str:
        """Get human-readable list of all available wave types."""
        lines = ["=== AVAILABLE WAVE TYPES ==="]
        lines.append("Basic: sine, triangle, saw, pulse")
        lines.append("Noise: noise (white), pink")
        lines.append("Physical: physical (harmonic ctrl), physical2 (inharmonic)")
        lines.append("Extended: supersaw, additive, formant, harmonic")
        lines.append("Waveguide: waveguide_string, waveguide_tube, waveguide_membrane, waveguide_plate")
        lines.append("Tables: wavetable, compound")
        if self.wavetables:
            lines.append(f"Loaded wavetables: {', '.join(self.wavetables.keys())}")
        if self.compound_waves:
            lines.append(f"Compound waves: {', '.join(self.compound_waves.keys())}")
        return '\n'.join(lines)

    def get_routing_info(self) -> str:
        """Get human-readable routing information."""
        if not self.algorithms:
            return "No modulation routings defined."
        
        lines = ["=== MODULATION ROUTING ==="]
        for i, (algo_type, src, tgt, amt) in enumerate(self.algorithms):
            lines.append(f"  [{i}] {algo_type}: op{src} -> op{tgt} (amount={amt:.3f})")
        return '\n'.join(lines)

    def _generate_operator_buffer(self, idx: int, t: np.ndarray, 
                                   freq_offset: float = 0.0,
                                   amp_scale: float = 1.0,
                                   phase_offset: float = 0.0,
                                   seed: Optional[int] = None) -> np.ndarray:
        """Generate buffer for a single operator with voice offsets.
        
        Supports audio-rate interval modulation via:
        - interval_mod: explicit signal in operator dict
        - interval_lfo_*: automatic LFO generation
        """
        if idx not in self.operators:
            return np.zeros(len(t), dtype=np.float64)
        
        op = self.operators[idx]
        wave = op['wave']
        base_freq = op['freq'] + freq_offset
        amp = op['amp'] * amp_scale
        phase = op['phase'] + phase_offset
        
        n_samples = len(t)
        
        # === AUDIO-RATE INTERVAL MODULATION ===
        freq_array = None  # None = constant frequency
        
        # Check for explicit interval modulation signal
        interval_mod = op.get('interval_mod')
        interval_depth = op.get('interval_mod_depth', 12.0)
        
        # Check for LFO-based interval modulation
        lfo_rate = op.get('interval_lfo_rate', 0.0)
        lfo_depth = op.get('interval_lfo_depth', 0.0)
        lfo_wave = op.get('interval_lfo_wave', 'sine')
        
        if interval_mod is not None and len(interval_mod) > 0:
            # Resample modulation signal to match output length
            mod_indices = np.linspace(0, len(interval_mod) - 1, n_samples)
            mod_signal = np.interp(mod_indices, np.arange(len(interval_mod)), interval_mod)
            
            # Convert semitones to frequency ratio
            semitones = mod_signal * interval_depth
            freq_ratio = np.power(2.0, semitones / 12.0)
            freq_array = base_freq * freq_ratio
            
        elif lfo_rate > 0 and lfo_depth > 0:
            # Generate LFO for interval modulation
            lfo_phase = 2 * np.pi * lfo_rate * t
            
            if lfo_wave == 'sine':
                mod_signal = np.sin(lfo_phase)
            elif lfo_wave == 'triangle' or lfo_wave == 'tri':
                mod_signal = 2 * np.abs(2 * ((lfo_rate * t) % 1) - 1) - 1
            elif lfo_wave == 'saw':
                mod_signal = 2 * ((lfo_rate * t) % 1) - 1
            elif lfo_wave == 'square':
                mod_signal = np.sign(np.sin(lfo_phase))
            else:
                mod_signal = np.sin(lfo_phase)
            
            # Convert semitones to frequency ratio
            semitones = mod_signal * lfo_depth
            freq_ratio = np.power(2.0, semitones / 12.0)
            freq_array = base_freq * freq_ratio
        
        # === GENERATE WAVEFORM ===
        if freq_array is not None:
            # Audio-rate frequency modulation (interval mod)
            return self._generate_with_variable_freq(wave, t, freq_array, amp, phase, op, seed)
        else:
            # Constant frequency generation
            if wave == 'sine':
                return _generate_sine(t, base_freq, amp, phase)
            elif wave == 'triangle':
                # Use band-limited version if HQ mode enabled
                if self.hq_oscillators:
                    return _generate_triangle_bandlimited(t, base_freq, amp, phase, self.sample_rate)
                return _generate_triangle(t, base_freq, amp, phase)
            elif wave == 'saw':
                # Use band-limited version if HQ mode enabled
                if self.hq_oscillators:
                    return _generate_saw_bandlimited(t, base_freq, amp, phase, self.sample_rate)
                return _generate_saw(t, base_freq, amp, phase)
            elif wave == 'pulse':
                pw = op.get('pw', 0.5)
                # Square wave (pw=0.5) can use band-limited version
                if self.hq_oscillators and abs(pw - 0.5) < 0.01:
                    return _generate_square_bandlimited(t, base_freq, amp, phase, self.sample_rate)
                return _generate_pulse(t, base_freq, amp, phase, pw)
            elif wave == 'noise':
                return _generate_white_noise(n_samples, amp, seed)
            elif wave == 'pink':
                return _generate_pink_noise(n_samples, amp, seed)
            elif wave == 'physical':
                return _generate_physical(
                    t, base_freq, amp, phase,
                    even_harmonics=op.get('even_harmonics', 8),
                    odd_harmonics=op.get('odd_harmonics', 4),
                    even_weight=op.get('even_weight', 1.0),
                    decay=op.get('decay', 0.7)
                )
            elif wave == 'physical2':
                return _generate_physical2(
                    t, base_freq, amp, phase,
                    inharmonicity=op.get('inharmonicity', 0.01),
                    partials=op.get('partials', 12),
                    decay_curve=op.get('decay_curve', 'exp')
                )
            # ── Phase 2 wave types ──
            elif wave == 'supersaw':
                return _generate_supersaw(
                    t, base_freq, amp, phase,
                    num_saws=op.get('num_saws', 7),
                    detune_spread=op.get('detune_spread', 0.5),
                    mix=op.get('mix', 0.75),
                    sr=self.sample_rate
                )
            elif wave == 'additive':
                return _generate_additive(
                    t, base_freq, amp, phase,
                    harmonics=op.get('harmonics'),
                    num_harmonics=op.get('num_harmonics', 16),
                    rolloff=op.get('rolloff', 1.0)
                )
            elif wave == 'formant':
                return _generate_formant_wave(
                    t, base_freq, amp, phase,
                    vowel=op.get('vowel', 'a')
                )
            elif wave == 'harmonic':
                return _generate_harmonic(
                    t, base_freq, amp, phase,
                    odd_level=op.get('odd_level', 1.0),
                    even_level=op.get('even_level', 1.0),
                    num_harmonics=op.get('num_harmonics', 16),
                    odd_decay=op.get('odd_decay', 0.8),
                    even_decay=op.get('even_decay', 0.8)
                )
            elif wave == 'waveguide_string':
                return _generate_waveguide_string(
                    t, base_freq, amp, phase,
                    damping=op.get('damping', 0.996),
                    brightness=op.get('brightness', 0.5),
                    position=op.get('position', 0.28),
                    sr=self.sample_rate
                )
            elif wave == 'waveguide_tube':
                return _generate_waveguide_tube(
                    t, base_freq, amp, phase,
                    damping=op.get('damping', 0.995),
                    reflection=op.get('reflection', 0.7),
                    bore_shape=op.get('bore_shape', 0.5),
                    sr=self.sample_rate
                )
            elif wave == 'waveguide_membrane':
                return _generate_waveguide_membrane(
                    t, base_freq, amp, phase,
                    tension=op.get('tension', 0.5),
                    damping=op.get('damping', 0.995),
                    strike_pos=op.get('strike_pos', 0.3),
                    sr=self.sample_rate
                )
            elif wave == 'waveguide_plate':
                return _generate_waveguide_plate(
                    t, base_freq, amp, phase,
                    thickness=op.get('thickness', 0.5),
                    damping=op.get('damping', 0.997),
                    material=op.get('material', 0.5),
                    sr=self.sample_rate
                )
            elif wave == 'wavetable':
                wt_name = op.get('wavetable_name', '')
                frames = self.wavetables.get(wt_name) if wt_name else None
                return _generate_wavetable(
                    t, base_freq, amp, phase,
                    frames=frames,
                    frame_pos=op.get('frame_pos', 0.0),
                    sr=self.sample_rate
                )
            elif wave == 'compound':
                cmp_name = op.get('compound_name', '')
                layers = self.compound_waves.get(cmp_name) if cmp_name else op.get('layers')
                return _generate_compound(
                    t, base_freq, amp, phase,
                    layers=layers,
                    morph=op.get('morph', 0.0),
                    sr=self.sample_rate
                )
            else:
                return _generate_sine(t, base_freq, amp, phase)
    
    def _generate_with_variable_freq(self, wave: str, t: np.ndarray, 
                                      freq_array: np.ndarray, amp: float,
                                      phase: float, op: dict,
                                      seed: Optional[int] = None) -> np.ndarray:
        """Generate waveform with audio-rate frequency modulation.
        
        Uses phase accumulation for smooth frequency changes.
        """
        n_samples = len(t)
        dt = 1.0 / self.sample_rate
        
        # Noise doesn't need frequency modulation
        if wave == 'noise':
            return _generate_white_noise(n_samples, amp, seed)
        elif wave == 'pink':
            return _generate_pink_noise(n_samples, amp, seed)
        
        # Phase accumulation for smooth frequency modulation
        phase_acc = np.zeros(n_samples, dtype=np.float64)
        phase_acc[0] = phase
        for i in range(1, n_samples):
            phase_acc[i] = phase_acc[i-1] + 2 * np.pi * freq_array[i-1] * dt
        
        # Generate waveform based on accumulated phase
        if wave == 'sine':
            return amp * np.sin(phase_acc)
        elif wave == 'triangle':
            # Convert phase to triangle
            normalized = (phase_acc / (2 * np.pi)) % 1.0
            return amp * (2 * np.abs(2 * normalized - 1) - 1)
        elif wave == 'saw':
            normalized = (phase_acc / (2 * np.pi)) % 1.0
            return amp * (2 * normalized - 1)
        elif wave == 'pulse':
            pw = op.get('pw', 0.5)
            normalized = (phase_acc / (2 * np.pi)) % 1.0
            return amp * np.where(normalized < pw, 1.0, -1.0)
        elif wave in ('waveguide_string', 'waveguide_tube', 'waveguide_membrane',
                       'waveguide_plate', 'physical', 'physical2', 'supersaw',
                       'additive', 'formant', 'harmonic', 'wavetable', 'compound'):
            # These complex wave types don't support per-sample frequency
            # modulation — use mean frequency as approximation
            mean_freq = float(np.mean(freq_array))
            if wave == 'supersaw':
                return _generate_supersaw(t, mean_freq, amp, phase,
                    num_saws=op.get('num_saws', 7), detune_spread=op.get('detune_spread', 0.5),
                    mix=op.get('mix', 0.75), sr=self.sample_rate)
            elif wave == 'additive':
                return _generate_additive(t, mean_freq, amp, phase,
                    harmonics=op.get('harmonics'), num_harmonics=op.get('num_harmonics', 16),
                    rolloff=op.get('rolloff', 1.0))
            elif wave == 'formant':
                return _generate_formant_wave(t, mean_freq, amp, phase, vowel=op.get('vowel', 'a'))
            elif wave == 'harmonic':
                return _generate_harmonic(t, mean_freq, amp, phase,
                    odd_level=op.get('odd_level', 1.0), even_level=op.get('even_level', 1.0),
                    num_harmonics=op.get('num_harmonics', 16),
                    odd_decay=op.get('odd_decay', 0.8), even_decay=op.get('even_decay', 0.8))
            elif wave == 'physical':
                return _generate_physical(t, mean_freq, amp, phase,
                    even_harmonics=op.get('even_harmonics', 8), odd_harmonics=op.get('odd_harmonics', 4),
                    even_weight=op.get('even_weight', 1.0), decay=op.get('decay', 0.7))
            elif wave == 'physical2':
                return _generate_physical2(t, mean_freq, amp, phase,
                    inharmonicity=op.get('inharmonicity', 0.01), partials=op.get('partials', 12),
                    decay_curve=op.get('decay_curve', 'exp'))
            elif wave == 'waveguide_string':
                return _generate_waveguide_string(t, mean_freq, amp, phase,
                    damping=op.get('damping', 0.996), brightness=op.get('brightness', 0.5),
                    position=op.get('position', 0.28), sr=self.sample_rate)
            elif wave == 'waveguide_tube':
                return _generate_waveguide_tube(t, mean_freq, amp, phase,
                    damping=op.get('damping', 0.995), reflection=op.get('reflection', 0.7),
                    bore_shape=op.get('bore_shape', 0.5), sr=self.sample_rate)
            elif wave == 'waveguide_membrane':
                return _generate_waveguide_membrane(t, mean_freq, amp, phase,
                    tension=op.get('tension', 0.5), damping=op.get('damping', 0.995),
                    strike_pos=op.get('strike_pos', 0.3), sr=self.sample_rate)
            elif wave == 'waveguide_plate':
                return _generate_waveguide_plate(t, mean_freq, amp, phase,
                    thickness=op.get('thickness', 0.5), damping=op.get('damping', 0.997),
                    material=op.get('material', 0.5), sr=self.sample_rate)
            else:
                return amp * np.sin(phase_acc)
        else:
            # Fallback to sine
            return amp * np.sin(phase_acc)

    def _apply_modulation(self, buffers: Dict[int, np.ndarray],
                          t: np.ndarray,
                          mod_scale: float = 1.0) -> Dict[int, np.ndarray]:
        """Apply all modulation algorithms to operator buffers.
        
        Parameters
        ----------
        buffers : dict
            Current operator buffers
        t : np.ndarray
            Time array
        mod_scale : float
            Scaling factor for modulation amounts (for voice algorithm)
            
        Returns
        -------
        dict
            Modified buffers with modulation applied
        """
        result = {k: v.copy() for k, v in buffers.items()}
        
        for algo_type, source, target, amount in self.algorithms:
            if source not in result or target not in self.operators:
                continue
            
            src_buf = result[source]
            op = self.operators[target]
            freq = op['freq']
            amp = op['amp']
            phase = op['phase']
            
            scaled_amount = amount * mod_scale
            
            if algo_type == 'FM':
                # Frequency Modulation: modulator affects carrier frequency
                # Classic FM: phase = 2π * (fc * t + I * mod(t))
                mod_phase = 2 * np.pi * freq * t + scaled_amount * src_buf + phase
                result[target] = amp * np.sin(mod_phase)
                
            elif algo_type == 'TFM':
                # Through-Zero FM: allows negative frequencies
                # More aggressive/harsh than standard FM
                mod_phase = 2 * np.pi * freq * t + 0.5 * scaled_amount * src_buf + phase
                result[target] = amp * np.sin(mod_phase)
                
            elif algo_type == 'AM':
                # Amplitude Modulation: modulator affects carrier amplitude
                # AM: out = carrier * (1 + depth * modulator)
                result[target] = result[target] * (1.0 + scaled_amount * src_buf)
                
            elif algo_type == 'RM':
                # Ring Modulation: multiply carrier and modulator
                # RM: out = carrier * modulator * depth
                result[target] = result[target] * src_buf * scaled_amount
                
            elif algo_type == 'PM':
                # Phase Modulation: modulator directly offsets carrier phase
                # PM: out = sin(2π * fc * t + depth * mod(t))
                # Similar to FM but modulator is added directly to phase
                mod_phase = 2 * np.pi * freq * t + phase + scaled_amount * src_buf
                result[target] = amp * np.sin(mod_phase)
        
        return result

    def render(
        self,
        duration_sec: float,
        voice_count: int = 1,
        carrier_count: Optional[int] = None,
        mod_count: Optional[int] = None,
        filter_type: Optional[int] = None,
        cutoff: Optional[float] = None,
        resonance: Optional[float] = None,
        dt: Optional[float] = None,
        rand: Optional[float] = None,
        mod: Optional[float] = None,
        stereo_spread: Optional[float] = None,
        phase_spread: Optional[float] = None,
        voice_algorithm: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Render audio buffer with full voice algorithm support.
        
        Voice algorithms control how multiple voices are spread:
        
        Algorithm 0 / "" / "stack":
            Classic stack — all voices at same phase unless phase_spread
            is explicitly set.  rand only varies amplitude.  Can phase-
            lock without manual phase_spread or detune.
            
        Algorithm 1 / "unison":
            Normal unison — rand automatically spreads voice phase to
            prevent phase-locking.  Even with no detune, voices get
            random initial phase so they don't cancel or constructively
            pile up.  This is what most hardware synths do.

        Algorithm 2 / "wide":
            Like unison but also applies automatic stereo spread and
            slight per-voice detune jitter for maximum width.

        Parameters
        ----------
        duration_sec : float
            Duration in seconds
        voice_count : int
            Number of voices for unison (1 = mono)
        carrier_count : int, optional
            Number of operators that contribute to output
        mod_count : int, optional
            Number of operators designated as modulators (reserved)
        filter_type : int, optional
            Filter type index (0-29)
        cutoff : float, optional
            Filter cutoff frequency in Hz
        resonance : float, optional
            Filter resonance (0-100)
        dt : float, optional
            Detune amount in Hz per voice
        rand : float, optional
            Randomness (0-100).  In algorithm 0 only varies amplitude.
            In algorithm 1/2 also randomizes per-voice phase to prevent
            phase-locking.
        mod : float, optional
            Modulation depth scaling per voice (0-100)
        stereo_spread : float, optional
            Stereo spread (0-100, 0=mono, 100=full width)
        phase_spread : float, optional
            Phase offset per voice in radians
        voice_algorithm : str, optional
            Voice algorithm name or number ("0"/"stack", "1"/"unison",
            "2"/"wide")
        seed : int, optional
            Random seed for deterministic rendering
            
        Returns
        -------
        np.ndarray
            Rendered audio buffer (mono or stereo)
        """
        import scipy.signal
        
        if not self.operators:
            return np.zeros(int(self.sample_rate * duration_sec), dtype=np.float64)
        
        # --- Resolve voice algorithm ---
        va = (voice_algorithm or '').strip().lower()
        if va in ('1', 'unison'):
            algo = 1
        elif va in ('2', 'wide'):
            algo = 2
        else:
            algo = 0  # stack (legacy)
        
        # --- Scale abstract parameters from 0-100 to internal ranges ---
        rand_scaled = scale_wet(rand) if rand is not None else 0.0
        mod_scaled = scale_wet(mod) if mod is not None else 0.0
        stereo_scaled = scale_wet(stereo_spread) if stereo_spread is not None else 0.0
        res_scaled = scale_resonance(resonance) if resonance is not None else 0.707
        
        # dt and phase_spread stay as real units (Hz and radians)
        dt_hz = dt if dt is not None else 0.0
        phase_rad = phase_spread if phase_spread is not None else 0.0
        
        # --- Algorithm-specific defaults ---
        # Algo 1 (unison): if rand > 0, auto-spread phase to prevent lock
        # Algo 2 (wide): auto stereo + detune jitter
        if algo >= 1 and rand_scaled == 0 and phase_rad == 0 and voice_count > 1:
            # Even with no explicit rand, give a mild phase spread
            # so voices don't phase-lock by default
            rand_scaled = 0.15  # ~15% default randomness
        if algo == 2 and stereo_scaled == 0 and voice_count > 1:
            stereo_scaled = 0.7  # 70% auto-spread
        
        # --- Random generator ---
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        
        # --- Time axis ---
        n_samples = int(max(1, self.sample_rate * duration_sec))
        t = np.arange(n_samples, dtype=np.float64) / self.sample_rate
        
        # --- Carrier operators ---
        sorted_indices = sorted(self.operators.keys())
        if carrier_count is None or carrier_count >= len(sorted_indices):
            carrier_indices = sorted_indices
        else:
            carrier_indices = sorted_indices[:max(1, carrier_count)]
        
        # --- Render each voice ---
        voices = []
        
        for voice_idx in range(max(1, voice_count)):
            # --- Detune: symmetric around center ---
            if dt_hz and voice_count > 1:
                detune_offset = dt_hz * (voice_idx - (voice_count - 1) / 2)
            else:
                detune_offset = 0.0
            
            # Algo 2: add random detune jitter on top
            if algo == 2 and voice_count > 1 and voice_idx > 0:
                jitter_hz = rng.uniform(-0.5, 0.5) * max(dt_hz, 1.0)
                detune_offset += jitter_hz
            
            # --- Amplitude variation ---
            if rand_scaled > 0 and voice_idx > 0:
                amp_variation = 1.0 + (rng.random() * 2 - 1) * rand_scaled
            else:
                amp_variation = 1.0
            
            # --- Phase offset ---
            if algo >= 1 and voice_count > 1 and voice_idx > 0:
                # Unison/wide: random phase per voice (prevents phase-lock)
                random_phase = rng.uniform(0, 2 * np.pi)
                # Add any explicit phase_spread on top
                explicit_phase = phase_rad * voice_idx / voice_count if phase_rad else 0.0
                phase_offset = random_phase + explicit_phase
            elif phase_rad and voice_count > 1:
                # Stack: only use explicit phase_spread
                phase_offset = phase_rad * voice_idx / voice_count
            else:
                phase_offset = 0.0
            
            # --- Modulation scaling ---
            if mod_scaled > 0 and voice_count > 1:
                mod_scale = 1.0 + mod_scaled * voice_idx / (voice_count - 1)
            else:
                mod_scale = 1.0
            
            # --- Generate operator buffers ---
            voice_buffers = {}
            for idx in self.operators:
                voice_seed = (seed + voice_idx * 1000) if seed is not None else None
                voice_buffers[idx] = self._generate_operator_buffer(
                    idx, t,
                    freq_offset=detune_offset,
                    amp_scale=amp_variation,
                    phase_offset=phase_offset,
                    seed=voice_seed
                )
            
            # Apply modulation algorithms
            voice_buffers = self._apply_modulation(voice_buffers, t, mod_scale)
            
            # Sum carrier operators
            voice_out = np.zeros(n_samples, dtype=np.float64)
            for idx in carrier_indices:
                if idx in voice_buffers:
                    voice_out += voice_buffers[idx]
            
            voices.append(voice_out)
        
        # Mix voices
        if voice_count == 1:
            out = voices[0]
        else:
            out = np.mean(np.array(voices), axis=0)
        
        # Apply filter if specified
        if filter_type is not None and cutoff is not None and cutoff > 0:
            out = self._apply_filter(out, t, filter_type, cutoff, res_scaled)
        
        # Handle stereo spread
        if stereo_scaled > 0 and voice_count > 1:
            # Create stereo output with voice panning
            left = np.zeros(n_samples, dtype=np.float64)
            right = np.zeros(n_samples, dtype=np.float64)
            
            for i, voice in enumerate(voices):
                # Pan position: spread voices across stereo field
                pan = (i / (voice_count - 1) - 0.5) * 2 * stereo_scaled if voice_count > 1 else 0.0
                left_gain = np.sqrt(0.5 * (1.0 - pan))
                right_gain = np.sqrt(0.5 * (1.0 + pan))
                left += voice * left_gain
                right += voice * right_gain
            
            # Normalize
            left /= voice_count
            right /= voice_count
            
            out = np.column_stack([left, right])
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(out))
        if max_val > 1.0:
            out = out / max_val
        
        return out.astype(np.float64)

    def _apply_filter(self, buf: np.ndarray, t: np.ndarray, 
                      filter_type: int, cutoff: float, resonance: float) -> np.ndarray:
        """Apply filter to buffer."""
        import scipy.signal
        
        try:
            nyq = 0.5 * self.sample_rate
            norm_cutoff = min(0.99, max(0.001, cutoff / nyq))
            q = max(0.1, min(20.0, resonance))
            
            if filter_type == 0:  # Low pass
                b, a = scipy.signal.butter(2, norm_cutoff, btype='low')
                return scipy.signal.lfilter(b, a, buf)
                
            elif filter_type == 1:  # High pass
                b, a = scipy.signal.butter(2, norm_cutoff, btype='high')
                return scipy.signal.lfilter(b, a, buf)
                
            elif filter_type == 2:  # Band pass
                low = max(0.001, norm_cutoff / np.sqrt(2))
                high = min(0.999, norm_cutoff * np.sqrt(2))
                if high > low:
                    b, a = scipy.signal.butter(2, [low, high], btype='band')
                    return scipy.signal.lfilter(b, a, buf)
                    
            elif filter_type == 3:  # Notch
                low = max(0.001, norm_cutoff / np.sqrt(2))
                high = min(0.999, norm_cutoff * np.sqrt(2))
                if high > low:
                    b, a = scipy.signal.butter(2, [low, high], btype='bandstop')
                    return scipy.signal.lfilter(b, a, buf)
                    
            elif filter_type == 4:  # Peak
                bw = max(0.001, norm_cutoff * 0.1)
                low = max(0.001, norm_cutoff - bw)
                high = min(0.999, norm_cutoff + bw)
                if high > low:
                    b, a = scipy.signal.butter(2, [low, high], btype='band')
                    return scipy.signal.lfilter(b, a, buf)
                    
            elif filter_type == 5:  # Ring mod
                ring = np.sin(2 * np.pi * cutoff * t)
                return buf * ring
                
            elif filter_type == 6:  # Allpass
                return buf
                
            elif filter_type in (7, 8, 9):  # Comb filters
                delay_samples = max(1, int(self.sample_rate / cutoff))
                g = min(0.99, q * 0.1)  # Scale resonance for feedback
                
                if filter_type == 7:  # Feed-forward
                    out = buf.copy()
                    out[delay_samples:] += g * buf[:-delay_samples]
                    return out
                elif filter_type == 8:  # Feed-back
                    out = buf.copy()
                    for i in range(delay_samples, len(buf)):
                        out[i] = buf[i] + g * out[i - delay_samples]
                    return out
                else:  # Combined
                    out = buf.copy()
                    for i in range(delay_samples, len(buf)):
                        out[i] = buf[i] + g * buf[i - delay_samples] + g * out[i - delay_samples]
                    return out
                    
            elif filter_type in (10, 11):  # Analog/acid
                fc = norm_cutoff
                alpha = fc / (fc + 1.0)
                y = buf.copy()
                for stage in range(4):
                    y2 = np.zeros_like(y)
                    y2[0] = alpha * y[0]
                    for i in range(1, len(y)):
                        y2[i] = y2[i - 1] + alpha * (y[i] - y2[i - 1])
                    if filter_type == 11:  # Acid: add tanh
                        y2 = np.tanh(y2 * (1 + q * 0.5))
                    y = y2
                return y
                
            elif filter_type in (12, 13, 14, 15, 16):  # Formants
                formant_freqs = {
                    12: [(800, 200), (1150, 150), (2900, 200)],   # A
                    13: [(350, 100), (2000, 150), (2800, 200)],   # E
                    14: [(270, 80), (2300, 150), (3000, 200)],    # I
                    15: [(450, 100), (800, 150), (2830, 200)],    # O
                    16: [(325, 80), (700, 150), (2530, 200)],     # U
                }
                formants = formant_freqs.get(filter_type, [(500, 150)])
                mix = np.zeros_like(buf)
                for freq, bw in formants:
                    low = max(0.001, (freq - bw/2) / nyq)
                    high = min(0.999, (freq + bw/2) / nyq)
                    if high > low:
                        b, a = scipy.signal.butter(2, [low, high], btype='band')
                        mix += scipy.signal.lfilter(b, a, buf)
                return mix
                
            # More filter types can be added here...
            
        except Exception:
            pass
        
        return buf
