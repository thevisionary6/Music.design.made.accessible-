"""Monolith Offline Synth Engine - Full Implementation.

This engine implements:
- Operator-based architecture with configurable wave types
- Full modulation routing: FM, TFM, AM, RM, PM
- Voice algorithm system with detune, random, stereo spread, phase offset
- Extended wave models: sine, triangle, saw, pulse/PWM, noise (white/pink), physical modeling
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
BUILD ID: monolith_v14_chunk1.2
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
# WAVE TYPE REGISTRY
# ============================================================================

# Carrier-only wave types (should not be used as modulators)
CARRIER_ONLY_WAVES = {'saw', 'pulse', 'pwm', 'square'}

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
