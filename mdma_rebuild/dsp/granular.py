"""Granular Synthesis Engine for MDMA.

Full implementation of granular processing with:
- Configurable grain size, density, and overlap
- Random and sequential grain selection
- Pitch shifting and time stretching
- Grain envelope shaping
- Audio-rate modulation of grain parameters

BUILD ID: granular_v15.1
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, List, Tuple, Any


# ============================================================================
# GRAIN ENVELOPES
# ============================================================================

def _hann_window(size: int) -> np.ndarray:
    """Hann (raised cosine) window - smooth attack/release."""
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(size) / (size - 1)))


def _triangular_window(size: int) -> np.ndarray:
    """Triangular window - linear attack/release."""
    half = size // 2
    rising = np.linspace(0, 1, half, endpoint=False)
    falling = np.linspace(1, 0, size - half)
    return np.concatenate([rising, falling])


def _gaussian_window(size: int, sigma: float = 0.4) -> np.ndarray:
    """Gaussian window - bell curve shape."""
    n = np.arange(size) - (size - 1) / 2
    return np.exp(-0.5 * (n / (sigma * (size - 1) / 2)) ** 2)


def _trapezoid_window(size: int, attack_ratio: float = 0.25) -> np.ndarray:
    """Trapezoid window - flat top with linear attack/release."""
    attack = int(size * attack_ratio)
    sustain = size - 2 * attack
    rising = np.linspace(0, 1, attack)
    flat = np.ones(sustain)
    falling = np.linspace(1, 0, attack)
    return np.concatenate([rising, flat, falling])


def _tukey_window(size: int, alpha: float = 0.5) -> np.ndarray:
    """Tukey (tapered cosine) window - adjustable taper ratio."""
    window = np.ones(size)
    taper = int(alpha * size / 2)
    
    # Rising taper
    for i in range(taper):
        window[i] = 0.5 * (1 - np.cos(np.pi * i / taper))
    
    # Falling taper
    for i in range(taper):
        window[size - 1 - i] = 0.5 * (1 - np.cos(np.pi * i / taper))
    
    return window


GRAIN_ENVELOPES = {
    'hann': _hann_window,
    'triangle': _triangular_window,
    'tri': _triangular_window,
    'gaussian': _gaussian_window,
    'gauss': _gaussian_window,
    'trapezoid': _trapezoid_window,
    'trap': _trapezoid_window,
    'tukey': _tukey_window,
    'rect': lambda size: np.ones(size),  # Rectangular (no envelope)
}


# ============================================================================
# GRANULAR ENGINE
# ============================================================================

class GranularEngine:
    """Full granular synthesis/processing engine.
    
    Features:
    - Variable grain size, density, and overlap
    - Random, sequential, or modulated position selection
    - Pitch shifting via playback rate
    - Time stretching via grain density
    - Audio-rate parameter modulation
    - Multiple grain envelope shapes
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # Default grain parameters
        self.grain_size_ms: float = 50.0  # Grain duration in ms
        self.density: float = 4.0  # Grains per grain-duration (overlap factor)
        self.position: float = 0.0  # Normalized position in source (0-1)
        self.position_spread: float = 0.1  # Random spread around position
        self.pitch_ratio: float = 1.0  # Playback rate (1.0 = normal)
        self.envelope: str = 'hann'  # Grain envelope type
        self.reverse_prob: float = 0.0  # Probability of reversed grains
        
        # Modulation sources (audio-rate arrays or None)
        self.position_mod: Optional[np.ndarray] = None
        self.pitch_mod: Optional[np.ndarray] = None
        self.density_mod: Optional[np.ndarray] = None
        self.size_mod: Optional[np.ndarray] = None
        
        # Random seed for determinism
        self.seed: Optional[int] = None
        
        # Grain bank (for preset storage)
        self.grain_bank: Dict[int, Dict[str, Any]] = {}
    
    def set_params(self, **kwargs) -> None:
        """Set granular parameters.
        
        Parameters
        ----------
        grain_size : float
            Grain size in milliseconds (1-500)
        density : float
            Grains per grain-duration (0.5-32)
        position : float
            Normalized position in source (0-1)
        position_spread : float
            Random spread around position (0-1)
        pitch : float
            Pitch ratio (0.25-4.0)
        envelope : str
            Envelope type (hann, triangle, gaussian, trapezoid, tukey, rect)
        reverse_prob : float
            Probability of reversed grains (0-1)
        seed : int
            Random seed for determinism
        """
        if 'grain_size' in kwargs:
            self.grain_size_ms = max(1.0, min(500.0, float(kwargs['grain_size'])))
        if 'density' in kwargs:
            self.density = max(0.5, min(32.0, float(kwargs['density'])))
        if 'position' in kwargs:
            self.position = max(0.0, min(1.0, float(kwargs['position'])))
        if 'position_spread' in kwargs or 'spread' in kwargs:
            self.position_spread = max(0.0, min(1.0, float(kwargs.get('position_spread', kwargs.get('spread', 0.1)))))
        if 'pitch' in kwargs:
            self.pitch_ratio = max(0.25, min(4.0, float(kwargs['pitch'])))
        if 'envelope' in kwargs or 'env' in kwargs:
            env = kwargs.get('envelope', kwargs.get('env', 'hann')).lower()
            if env in GRAIN_ENVELOPES:
                self.envelope = env
        if 'reverse_prob' in kwargs or 'reverse' in kwargs:
            self.reverse_prob = max(0.0, min(1.0, float(kwargs.get('reverse_prob', kwargs.get('reverse', 0.0)))))
        if 'seed' in kwargs:
            self.seed = int(kwargs['seed']) if kwargs['seed'] is not None else None
    
    def set_modulation(self, param: str, mod_signal: Optional[np.ndarray]) -> None:
        """Set audio-rate modulation for a parameter.
        
        Parameters
        ----------
        param : str
            Parameter to modulate (position, pitch, density, size)
        mod_signal : np.ndarray or None
            Modulation signal (will be interpolated to output length)
        """
        if param == 'position':
            self.position_mod = mod_signal
        elif param == 'pitch':
            self.pitch_mod = mod_signal
        elif param == 'density':
            self.density_mod = mod_signal
        elif param == 'size':
            self.size_mod = mod_signal
    
    def _get_modulated_value(self, base: float, mod: Optional[np.ndarray], 
                             sample_idx: int, mod_depth: float = 1.0) -> float:
        """Get parameter value with modulation applied."""
        if mod is None or len(mod) == 0:
            return base
        
        # Map sample index to modulation array
        mod_idx = int((sample_idx / self.sample_rate) * len(mod)) % len(mod)
        return base + mod[mod_idx] * mod_depth
    
    def process(self, source: np.ndarray, duration_sec: float) -> np.ndarray:
        """Process source audio through granular engine.
        
        Parameters
        ----------
        source : np.ndarray
            Source audio buffer
        duration_sec : float
            Output duration in seconds
            
        Returns
        -------
        np.ndarray
            Processed audio
        """
        if len(source) == 0:
            return np.zeros(int(duration_sec * self.sample_rate))
        
        rng = np.random.default_rng(self.seed)
        
        out_samples = int(duration_sec * self.sample_rate)
        output = np.zeros(out_samples, dtype=np.float64)
        
        # Base grain size in samples
        base_grain_samples = int(self.grain_size_ms * self.sample_rate / 1000)
        
        # Grain spacing based on density
        base_spacing = base_grain_samples / self.density
        
        # Get envelope function
        env_func = GRAIN_ENVELOPES.get(self.envelope, _hann_window)
        
        # Generate grains
        current_pos = 0
        grain_count = 0
        
        while current_pos < out_samples:
            # Get modulated parameters for this grain
            grain_samples = base_grain_samples
            if self.size_mod is not None:
                size_mod = self._get_modulated_value(1.0, self.size_mod, current_pos, 0.5)
                grain_samples = int(base_grain_samples * max(0.25, min(4.0, size_mod)))
            
            pitch = self.pitch_ratio
            if self.pitch_mod is not None:
                pitch = self._get_modulated_value(pitch, self.pitch_mod, current_pos, 0.5)
                pitch = max(0.25, min(4.0, pitch))
            
            density = self.density
            if self.density_mod is not None:
                density = self._get_modulated_value(density, self.density_mod, current_pos, 2.0)
                density = max(0.5, min(32.0, density))
            
            position = self.position
            if self.position_mod is not None:
                position = self._get_modulated_value(position, self.position_mod, current_pos, 0.5)
                position = max(0.0, min(1.0, position))
            
            # Add random position spread
            pos_offset = rng.uniform(-self.position_spread, self.position_spread)
            source_pos = max(0.0, min(1.0, position + pos_offset))
            
            # Calculate source start position in samples
            source_start = int(source_pos * (len(source) - grain_samples))
            source_start = max(0, min(len(source) - grain_samples, source_start))
            
            # Extract grain from source
            if pitch == 1.0:
                # No pitch shift - direct extraction
                grain_end = min(source_start + grain_samples, len(source))
                grain = source[source_start:grain_end].copy()
            else:
                # Pitch shift via resampling
                read_samples = int(grain_samples * pitch)
                read_end = min(source_start + read_samples, len(source))
                source_chunk = source[source_start:read_end]
                
                if len(source_chunk) > 1:
                    # Resample to target length
                    indices = np.linspace(0, len(source_chunk) - 1, grain_samples)
                    grain = np.interp(indices, np.arange(len(source_chunk)), source_chunk)
                else:
                    grain = np.zeros(grain_samples)
            
            # Ensure grain is correct length
            if len(grain) < grain_samples:
                grain = np.pad(grain, (0, grain_samples - len(grain)))
            elif len(grain) > grain_samples:
                grain = grain[:grain_samples]
            
            # Reverse grain probabilistically
            if rng.random() < self.reverse_prob:
                grain = grain[::-1]
            
            # Apply envelope
            envelope = env_func(len(grain))
            grain = grain * envelope
            
            # Add grain to output
            out_end = min(current_pos + len(grain), out_samples)
            add_len = out_end - current_pos
            output[current_pos:out_end] += grain[:add_len]
            
            # Advance position based on density
            spacing = grain_samples / density
            spacing = max(1, int(spacing + rng.uniform(-spacing * 0.1, spacing * 0.1)))
            current_pos += spacing
            grain_count += 1
        
        # Normalize output
        peak = np.max(np.abs(output))
        if peak > 0:
            output = output / peak * 0.9
        
        return output
    
    def freeze(self, source: np.ndarray, duration_sec: float, 
               freeze_pos: float = 0.5) -> np.ndarray:
        """Freeze at a single position (sustained grain cloud).
        
        Parameters
        ----------
        source : np.ndarray
            Source audio
        duration_sec : float
            Output duration
        freeze_pos : float
            Position to freeze (0-1)
            
        Returns
        -------
        np.ndarray
            Frozen audio
        """
        old_position = self.position
        old_spread = self.position_spread
        
        self.position = freeze_pos
        self.position_spread = 0.05  # Small spread for frozen texture
        
        result = self.process(source, duration_sec)
        
        self.position = old_position
        self.position_spread = old_spread
        
        return result
    
    def time_stretch(self, source: np.ndarray, stretch_factor: float) -> np.ndarray:
        """Time stretch audio without pitch change.
        
        Parameters
        ----------
        source : np.ndarray
            Source audio
        stretch_factor : float
            Stretch factor (0.5 = half length, 2.0 = double length)
            
        Returns
        -------
        np.ndarray
            Stretched audio
        """
        duration = (len(source) / self.sample_rate) * stretch_factor
        
        # Save and restore pitch
        old_pitch = self.pitch_ratio
        self.pitch_ratio = 1.0  # Keep original pitch
        
        # Adjust density for smooth output
        old_density = self.density
        self.density = max(4.0, self.density)
        
        # Sequential position scanning
        old_pos = self.position
        old_spread = self.position_spread
        self.position_spread = 0.02  # Minimal spread for clean stretch
        
        # Create position modulation to scan through source
        out_samples = int(duration * self.sample_rate)
        self.position_mod = np.linspace(0, 1, out_samples)
        self.position = 0.0
        
        result = self.process(source, duration)
        
        # Restore
        self.pitch_ratio = old_pitch
        self.density = old_density
        self.position = old_pos
        self.position_spread = old_spread
        self.position_mod = None
        
        return result
    
    def pitch_shift(self, source: np.ndarray, semitones: float) -> np.ndarray:
        """Pitch shift audio without time change.
        
        Parameters
        ----------
        source : np.ndarray
            Source audio
        semitones : float
            Pitch shift in semitones (-24 to +24)
            
        Returns
        -------
        np.ndarray
            Pitch-shifted audio
        """
        duration = len(source) / self.sample_rate
        
        # Convert semitones to ratio
        ratio = 2 ** (semitones / 12.0)
        
        old_pitch = self.pitch_ratio
        self.pitch_ratio = ratio
        
        # Sequential position for clean pitch shift
        old_pos = self.position
        old_spread = self.position_spread
        self.position_spread = 0.01
        
        out_samples = int(duration * self.sample_rate)
        self.position_mod = np.linspace(0, 1, out_samples)
        self.position = 0.0
        
        result = self.process(source, duration)
        
        # Restore
        self.pitch_ratio = old_pitch
        self.position = old_pos
        self.position_spread = old_spread
        self.position_mod = None
        
        return result
    
    def save_preset(self, slot: int) -> None:
        """Save current parameters to grain bank slot."""
        self.grain_bank[slot] = {
            'grain_size_ms': self.grain_size_ms,
            'density': self.density,
            'position': self.position,
            'position_spread': self.position_spread,
            'pitch_ratio': self.pitch_ratio,
            'envelope': self.envelope,
            'reverse_prob': self.reverse_prob,
            'seed': self.seed,
        }
    
    def load_preset(self, slot: int) -> bool:
        """Load parameters from grain bank slot."""
        if slot not in self.grain_bank:
            return False
        
        preset = self.grain_bank[slot]
        self.grain_size_ms = preset.get('grain_size_ms', 50.0)
        self.density = preset.get('density', 4.0)
        self.position = preset.get('position', 0.0)
        self.position_spread = preset.get('position_spread', 0.1)
        self.pitch_ratio = preset.get('pitch_ratio', 1.0)
        self.envelope = preset.get('envelope', 'hann')
        self.reverse_prob = preset.get('reverse_prob', 0.0)
        self.seed = preset.get('seed')
        return True
    
    def get_status(self) -> str:
        """Get formatted status string."""
        lines = [
            "GRANULAR ENGINE:",
            f"  Grain size: {self.grain_size_ms:.1f}ms",
            f"  Density: {self.density:.1f}",
            f"  Position: {self.position:.2f}",
            f"  Spread: {self.position_spread:.2f}",
            f"  Pitch: {self.pitch_ratio:.2f}x",
            f"  Envelope: {self.envelope}",
            f"  Reverse prob: {self.reverse_prob:.2f}",
            f"  Seed: {self.seed}",
            f"  Presets: {len(self.grain_bank)}",
        ]
        return '\n'.join(lines)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_granular_engine(sample_rate: int = 48000) -> GranularEngine:
    """Create a new granular engine instance."""
    return GranularEngine(sample_rate)


def granular_process(source: np.ndarray, duration_sec: float,
                     sample_rate: int = 48000, **kwargs) -> np.ndarray:
    """Quick granular processing without creating engine.
    
    Parameters
    ----------
    source : np.ndarray
        Source audio
    duration_sec : float
        Output duration
    sample_rate : int
        Sample rate
    **kwargs
        Granular parameters (grain_size, density, position, pitch, etc.)
        
    Returns
    -------
    np.ndarray
        Processed audio
    """
    engine = GranularEngine(sample_rate)
    engine.set_params(**kwargs)
    return engine.process(source, duration_sec)
