"""MDMA Advanced Audio Operations.

Auto-chunking, remix algorithms, rhythmic patterns, and user variable system.

Features:
- Auto-chunk algorithms for wavetables, sample slicing, reconstruction
- AI and deterministic remix algorithms
- Rhythmic pattern system (RPAT) for drum programming
- Buffer combining and overlaying
- User variable stack system

BUILD ID: advanced_ops_v1.0
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import random


# ============================================================================
# USER VARIABLE STACK
# ============================================================================

class UserStack:
    """Global user variable stack for storing values across commands.
    
    Usage:
        /= name value      Set variable
        /GET name          Get variable
        /GET name.sub      Get nested value (dict key or list index)
    """
    
    _instance: Optional['UserStack'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._vars: Dict[str, Any] = {}
        return cls._instance
    
    def set(self, name: str, value: Any) -> None:
        """Set a variable."""
        self._vars[name.lower()] = value
    
    def get(self, name: str, default: Any = None) -> Any:
        """Get a variable, supports dot notation for nested access."""
        name = name.lower()
        
        # Handle dot notation: var.key or var.0
        if '.' in name:
            parts = name.split('.', 1)
            base = self._vars.get(parts[0])
            if base is None:
                return default
            
            sub = parts[1]
            # Try as dict key first
            if isinstance(base, dict):
                return base.get(sub, default)
            # Try as list index
            if isinstance(base, (list, tuple)):
                try:
                    idx = int(sub)
                    return base[idx] if 0 <= idx < len(base) else default
                except ValueError:
                    return default
            # Try as attribute
            return getattr(base, sub, default)
        
        return self._vars.get(name, default)
    
    def exists(self, name: str) -> bool:
        """Check if variable exists."""
        return name.lower() in self._vars
    
    def delete(self, name: str) -> bool:
        """Delete a variable."""
        name = name.lower()
        if name in self._vars:
            del self._vars[name]
            return True
        return False
    
    def list_vars(self) -> Dict[str, Any]:
        """List all variables."""
        return self._vars.copy()
    
    def clear(self) -> None:
        """Clear all variables."""
        self._vars.clear()


def get_user_stack() -> UserStack:
    """Get the global user stack instance."""
    return UserStack()


# ============================================================================
# AUTO-CHUNKING ALGORITHMS
# ============================================================================

class ChunkAlgorithm(Enum):
    """Chunking algorithms."""
    TRANSIENT = "transient"      # Split at transients
    ZERO_CROSS = "zero_cross"    # Split at zero crossings
    BEAT = "beat"                # Split at detected beats
    EQUAL = "equal"              # Equal size chunks
    WAVETABLE = "wavetable"      # Single-cycle wavetable extraction
    SPECTRAL = "spectral"        # Spectral similarity
    ENERGY = "energy"            # Energy-based segmentation
    AUTO = "auto"                # Automatic best algorithm


@dataclass
class AudioChunk:
    """A chunk of audio with metadata."""
    audio: np.ndarray
    start_sample: int
    end_sample: int
    index: int
    
    # Analysis
    energy: float = 0.0
    zero_crossings: int = 0
    peak: float = 0.0
    
    # Optional labels
    label: str = ""
    tags: List[str] = field(default_factory=list)


def auto_chunk(
    audio: np.ndarray,
    sr: int = 48000,
    algorithm: str = "auto",
    num_chunks: int = 0,
    min_length_ms: float = 50.0,
    max_length_ms: float = 5000.0,
) -> List[AudioChunk]:
    """Auto-chunk audio using specified algorithm.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio buffer to chunk
    sr : int
        Sample rate
    algorithm : str
        Chunking algorithm (auto, transient, zero_cross, beat, equal, wavetable, energy)
    num_chunks : int
        Target number of chunks (0 = auto-detect)
    min_length_ms : float
        Minimum chunk length in milliseconds
    max_length_ms : float
        Maximum chunk length in milliseconds
    
    Returns
    -------
    List[AudioChunk]
        List of audio chunks
    """
    if len(audio) == 0:
        return []
    
    # Convert to mono for analysis
    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio
    
    min_samples = int(min_length_ms * sr / 1000)
    max_samples = int(max_length_ms * sr / 1000)
    
    # Auto-select algorithm based on content
    if algorithm == "auto":
        algorithm = _select_chunk_algorithm(mono, sr)
    
    # Get split points based on algorithm
    if algorithm == "transient":
        splits = _chunk_transient(mono, sr, min_samples)
    elif algorithm == "zero_cross":
        splits = _chunk_zero_cross(mono, sr, min_samples)
    elif algorithm == "beat":
        splits = _chunk_beat(mono, sr, min_samples)
    elif algorithm == "equal":
        splits = _chunk_equal(mono, num_chunks if num_chunks > 0 else 8)
    elif algorithm == "wavetable":
        splits = _chunk_wavetable(mono, sr, num_chunks if num_chunks > 0 else 256)
    elif algorithm == "energy":
        splits = _chunk_energy(mono, sr, min_samples)
    elif algorithm == "spectral":
        splits = _chunk_spectral(mono, sr, min_samples)
    else:
        splits = _chunk_equal(mono, 8)
    
    # Enforce min/max lengths
    splits = _enforce_chunk_limits(splits, len(mono), min_samples, max_samples)
    
    # Create chunks
    chunks = []
    for i, (start, end) in enumerate(zip(splits[:-1], splits[1:])):
        chunk_audio = audio[start:end] if audio.ndim == 1 else audio[start:end, :]
        
        chunk = AudioChunk(
            audio=chunk_audio,
            start_sample=start,
            end_sample=end,
            index=i,
        )
        
        # Analyze chunk
        chunk_mono = mono[start:end]
        chunk.energy = np.sqrt(np.mean(chunk_mono ** 2))
        chunk.zero_crossings = np.sum(np.abs(np.diff(np.signbit(chunk_mono))))
        chunk.peak = np.max(np.abs(chunk_mono))
        
        chunks.append(chunk)
    
    return chunks


def _select_chunk_algorithm(audio: np.ndarray, sr: int) -> str:
    """Auto-select best chunking algorithm based on content."""
    # Analyze content characteristics
    duration = len(audio) / sr
    
    # Very short = wavetable
    if duration < 0.1:
        return "wavetable"
    
    # Check for transients (percussive content)
    env = np.abs(audio)
    env_smooth = np.convolve(env, np.ones(100)/100, mode='same')
    transient_ratio = np.max(env) / (np.mean(env_smooth) + 1e-10)
    
    if transient_ratio > 5:
        return "transient"
    
    # Check for rhythmic content
    if duration > 1.0:
        return "beat"
    
    # Default to energy-based
    return "energy"


def _chunk_transient(audio: np.ndarray, sr: int, min_samples: int) -> List[int]:
    """Chunk at transients (onset detection)."""
    # Simple onset detection
    hop = sr // 100  # 10ms hop
    frame_size = sr // 50  # 20ms frame
    
    # Energy envelope
    num_frames = len(audio) // hop
    energy = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop
        end = min(start + frame_size, len(audio))
        energy[i] = np.sqrt(np.mean(audio[start:end] ** 2))
    
    # Detect onsets (rising edges)
    diff = np.diff(energy)
    threshold = np.std(diff) * 1.5
    onsets = np.where(diff > threshold)[0]
    
    # Convert to sample positions
    splits = [0]
    for onset in onsets:
        sample_pos = onset * hop
        if sample_pos - splits[-1] >= min_samples:
            splits.append(sample_pos)
    splits.append(len(audio))
    
    return splits


def _chunk_zero_cross(audio: np.ndarray, sr: int, min_samples: int) -> List[int]:
    """Chunk at zero crossings."""
    # Find zero crossings
    zero_crosses = np.where(np.diff(np.signbit(audio)))[0]
    
    if len(zero_crosses) == 0:
        return [0, len(audio)]
    
    # Select zero crossings at regular intervals
    target_chunks = max(1, len(audio) // min_samples)
    step = max(1, len(zero_crosses) // target_chunks)
    
    splits = [0]
    for i in range(step, len(zero_crosses), step):
        splits.append(zero_crosses[i])
    splits.append(len(audio))
    
    return splits


def _chunk_beat(audio: np.ndarray, sr: int, min_samples: int) -> List[int]:
    """Chunk at detected beats."""
    # Simple beat detection using energy
    hop = sr // 20  # 50ms hop
    frame_size = sr // 10  # 100ms frame
    
    num_frames = len(audio) // hop
    energy = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop
        end = min(start + frame_size, len(audio))
        # Low-frequency energy (bass/kick)
        frame = audio[start:end]
        # Simple lowpass via averaging
        if len(frame) > 10:
            lp = np.convolve(frame, np.ones(10)/10, mode='same')
            energy[i] = np.sqrt(np.mean(lp ** 2))
    
    # Peak detection
    threshold = np.mean(energy) + np.std(energy)
    peaks = []
    
    for i in range(1, len(energy) - 1):
        if energy[i] > energy[i-1] and energy[i] > energy[i+1] and energy[i] > threshold:
            peaks.append(i)
    
    # Convert to sample positions
    splits = [0]
    for peak in peaks:
        sample_pos = peak * hop
        if sample_pos - splits[-1] >= min_samples:
            splits.append(sample_pos)
    splits.append(len(audio))
    
    return splits


def _chunk_equal(audio: np.ndarray, num_chunks: int) -> List[int]:
    """Chunk into equal-sized pieces."""
    chunk_size = len(audio) // num_chunks
    return [i * chunk_size for i in range(num_chunks)] + [len(audio)]


def _chunk_wavetable(audio: np.ndarray, sr: int, num_cycles: int = 256) -> List[int]:
    """Extract single-cycle waveforms for wavetable."""
    # Detect fundamental frequency
    # Use autocorrelation
    min_period = sr // 2000  # Max 2kHz
    max_period = sr // 20    # Min 20Hz
    
    if len(audio) < max_period * 2:
        return [0, len(audio)]
    
    # Autocorrelation
    corr = np.correlate(audio[:max_period*2], audio[:max_period*2], mode='full')
    corr = corr[len(corr)//2:]
    
    # Find first peak after min_period
    period = min_period
    for i in range(min_period, min(max_period, len(corr) - 1)):
        if corr[i] > corr[i-1] and corr[i] > corr[i+1]:
            period = i
            break
    
    # Generate splits
    splits = []
    pos = 0
    while pos < len(audio) and len(splits) < num_cycles:
        splits.append(pos)
        pos += period
    splits.append(len(audio))
    
    return splits


def _chunk_energy(audio: np.ndarray, sr: int, min_samples: int) -> List[int]:
    """Chunk based on energy changes."""
    hop = sr // 50  # 20ms
    
    num_frames = len(audio) // hop
    energy = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop
        end = min(start + hop, len(audio))
        energy[i] = np.sqrt(np.mean(audio[start:end] ** 2))
    
    # Find significant energy changes
    diff = np.abs(np.diff(energy))
    threshold = np.percentile(diff, 75)
    
    changes = np.where(diff > threshold)[0]
    
    splits = [0]
    for change in changes:
        sample_pos = change * hop
        if sample_pos - splits[-1] >= min_samples:
            splits.append(sample_pos)
    splits.append(len(audio))
    
    return splits


def _chunk_spectral(audio: np.ndarray, sr: int, min_samples: int) -> List[int]:
    """Chunk based on spectral changes."""
    # Simple spectral flux
    hop = sr // 50
    fft_size = 1024
    
    num_frames = (len(audio) - fft_size) // hop
    if num_frames < 2:
        return [0, len(audio)]
    
    flux = np.zeros(num_frames)
    prev_spec = None
    
    for i in range(num_frames):
        start = i * hop
        frame = audio[start:start + fft_size]
        spec = np.abs(np.fft.rfft(frame * np.hanning(fft_size)))
        
        if prev_spec is not None:
            flux[i] = np.sum(np.maximum(0, spec - prev_spec))
        prev_spec = spec
    
    # Find peaks in spectral flux
    threshold = np.mean(flux) + np.std(flux)
    peaks = []
    
    for i in range(1, len(flux) - 1):
        if flux[i] > flux[i-1] and flux[i] > flux[i+1] and flux[i] > threshold:
            peaks.append(i)
    
    splits = [0]
    for peak in peaks:
        sample_pos = peak * hop
        if sample_pos - splits[-1] >= min_samples:
            splits.append(sample_pos)
    splits.append(len(audio))
    
    return splits


def _enforce_chunk_limits(
    splits: List[int],
    total_length: int,
    min_samples: int,
    max_samples: int
) -> List[int]:
    """Enforce minimum and maximum chunk lengths."""
    if len(splits) < 2:
        return [0, total_length]
    
    # Merge chunks that are too small
    merged = [splits[0]]
    for i in range(1, len(splits)):
        if splits[i] - merged[-1] >= min_samples:
            merged.append(splits[i])
    
    if merged[-1] != total_length:
        merged.append(total_length)
    
    # Split chunks that are too large
    final = [merged[0]]
    for i in range(1, len(merged)):
        chunk_size = merged[i] - final[-1]
        if chunk_size > max_samples:
            # Split into smaller chunks
            num_sub = (chunk_size + max_samples - 1) // max_samples
            sub_size = chunk_size // num_sub
            for j in range(1, num_sub):
                final.append(final[-1] + sub_size)
        final.append(merged[i])
    
    return final


# ============================================================================
# REMIX ALGORITHMS
# ============================================================================

class RemixAlgorithm(Enum):
    """Remix algorithms."""
    SHUFFLE = "shuffle"          # Random chunk shuffle
    REVERSE = "reverse"          # Reverse chunks
    STUTTER = "stutter"          # Repeat chunks
    STRETCH = "stretch"          # Time stretch variations
    GLITCH = "glitch"            # Glitch effects
    CHOP = "chop"                # Beat chop/rearrange
    LAYER = "layer"              # Layer chunks
    EVOLVE = "evolve"            # AI-style evolution
    GRANULAR = "granular"        # Granular reconstruction


def remix_audio(
    audio: np.ndarray,
    sr: int = 48000,
    algorithm: str = "shuffle",
    intensity: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Remix audio using specified algorithm.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio to remix
    sr : int
        Sample rate
    algorithm : str
        Remix algorithm
    intensity : float
        Effect intensity 0-1
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Remixed audio
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if algorithm == "shuffle":
        return _remix_shuffle(audio, sr, intensity)
    elif algorithm == "reverse":
        return _remix_reverse(audio, sr, intensity)
    elif algorithm == "stutter":
        return _remix_stutter(audio, sr, intensity)
    elif algorithm == "glitch":
        return _remix_glitch(audio, sr, intensity)
    elif algorithm == "chop":
        return _remix_chop(audio, sr, intensity)
    elif algorithm == "layer":
        return _remix_layer(audio, sr, intensity)
    elif algorithm == "evolve":
        return _remix_evolve(audio, sr, intensity)
    elif algorithm == "granular":
        return _remix_granular(audio, sr, intensity)
    else:
        return _remix_shuffle(audio, sr, intensity)


def _remix_shuffle(audio: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Shuffle audio chunks."""
    chunks = auto_chunk(audio, sr, "beat")
    
    if len(chunks) < 2:
        return audio
    
    # Shuffle based on intensity
    num_swaps = int(len(chunks) * intensity)
    indices = list(range(len(chunks)))
    
    for _ in range(num_swaps):
        i, j = random.sample(range(len(chunks)), 2)
        indices[i], indices[j] = indices[j], indices[i]
    
    # Reconstruct
    result = np.concatenate([chunks[i].audio for i in indices])
    return result


def _remix_reverse(audio: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Reverse some chunks."""
    chunks = auto_chunk(audio, sr, "beat")
    
    result_chunks = []
    for chunk in chunks:
        if random.random() < intensity:
            result_chunks.append(chunk.audio[::-1])
        else:
            result_chunks.append(chunk.audio)
    
    return np.concatenate(result_chunks) if result_chunks else audio


def _remix_stutter(audio: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Add stutter effects."""
    chunks = auto_chunk(audio, sr, "transient", min_length_ms=25)
    
    result_chunks = []
    for chunk in chunks:
        result_chunks.append(chunk.audio)
        
        # Add stutters based on intensity
        if random.random() < intensity * 0.5:
            stutter_len = min(len(chunk.audio), int(sr * 0.05))  # 50ms stutter
            stutter = chunk.audio[:stutter_len]
            repeats = int(2 + intensity * 4)
            for _ in range(repeats):
                result_chunks.append(stutter)
    
    return np.concatenate(result_chunks) if result_chunks else audio


def _remix_glitch(audio: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Apply glitch effects."""
    result = audio.copy()
    
    # Number of glitches
    num_glitches = int(10 * intensity)
    
    for _ in range(num_glitches):
        # Random position and length
        pos = random.randint(0, max(1, len(audio) - sr // 10))
        length = random.randint(sr // 100, sr // 20)  # 10-50ms
        end = min(pos + length, len(audio))
        
        # Random glitch type
        glitch_type = random.choice(['repeat', 'reverse', 'zero', 'noise'])
        
        if glitch_type == 'repeat':
            # Repeat a small section
            chunk = result[pos:pos + length // 4].copy()
            result[pos:end] = np.tile(chunk, (end - pos) // len(chunk) + 1)[:end - pos]
        elif glitch_type == 'reverse':
            result[pos:end] = result[pos:end][::-1]
        elif glitch_type == 'zero':
            result[pos:end] = 0
        elif glitch_type == 'noise':
            result[pos:end] *= np.random.uniform(0.5, 1.5, end - pos)
    
    return result


def _remix_chop(audio: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Beat chop and rearrange."""
    # Chunk at beats
    chunks = auto_chunk(audio, sr, "beat")
    
    if len(chunks) < 4:
        return audio
    
    # Create patterns based on intensity
    result_chunks = []
    
    for i, chunk in enumerate(chunks):
        result_chunks.append(chunk.audio)
        
        # Skip or repeat based on intensity
        if random.random() < intensity * 0.3:
            continue  # Skip next
        
        if random.random() < intensity * 0.2:
            # Repeat current
            result_chunks.append(chunk.audio)
    
    return np.concatenate(result_chunks) if result_chunks else audio


def _remix_layer(audio: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Layer chunks on top of each other."""
    chunks = auto_chunk(audio, sr, "beat")
    
    if len(chunks) < 2:
        return audio
    
    # Start with original
    result = audio.copy()
    
    # Layer some chunks at different positions
    num_layers = int(len(chunks) * intensity * 0.5)
    
    for _ in range(num_layers):
        chunk = random.choice(chunks)
        pos = random.randint(0, max(1, len(result) - len(chunk.audio)))
        end = min(pos + len(chunk.audio), len(result))
        
        # Mix in
        result[pos:end] += chunk.audio[:end - pos] * 0.5
    
    # Normalize
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result /= peak
    
    return result


def _remix_evolve(audio: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """AI-style evolution - gradual transformation."""
    chunks = auto_chunk(audio, sr, "energy")
    
    if len(chunks) < 2:
        return audio
    
    result_chunks = []
    
    # Sort chunks by energy
    sorted_chunks = sorted(chunks, key=lambda c: c.energy)
    
    # Create evolution path
    for i, chunk in enumerate(sorted_chunks):
        progress = i / len(sorted_chunks)
        
        # Apply progressive effects
        chunk_audio = chunk.audio.copy()
        
        # Fade intensity increases
        if progress < 0.5:
            chunk_audio *= 0.5 + progress
        
        # Add harmonic content at end
        if progress > 0.7 and random.random() < intensity:
            # Simple saturation
            chunk_audio = np.tanh(chunk_audio * (1 + intensity))
        
        result_chunks.append(chunk_audio)
    
    return np.concatenate(result_chunks) if result_chunks else audio


def _remix_granular(audio: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Granular reconstruction."""
    grain_size = int(sr * 0.02 * (1 + intensity))  # 20-40ms grains
    hop = grain_size // 2
    
    num_grains = len(audio) // hop
    
    # Extract grains
    grains = []
    for i in range(num_grains):
        start = i * hop
        end = min(start + grain_size, len(audio))
        grain = audio[start:end]
        
        # Apply window
        if len(grain) == grain_size:
            window = np.hanning(grain_size)
            grain = grain * window
            grains.append(grain)
    
    if not grains:
        return audio
    
    # Shuffle grains based on intensity
    if intensity > 0.3:
        random.shuffle(grains)
    
    # Reconstruct with overlap-add
    result = np.zeros(len(audio))
    for i, grain in enumerate(grains):
        start = i * hop
        end = start + len(grain)
        if end <= len(result):
            result[start:end] += grain
    
    # Normalize
    peak = np.max(np.abs(result))
    if peak > 0:
        result /= peak
    
    return result


# ============================================================================
# RHYTHMIC PATTERN (RPAT)
# ============================================================================

@dataclass
class RhythmicPattern:
    """Rhythmic pattern definition."""
    pattern: List[float]  # Velocity values (0-1), 0 = off
    length_beats: float = 1.0
    name: str = ""


def apply_rhythmic_pattern(
    audio: np.ndarray,
    pattern: List[float],
    sr: int = 48000,
    bpm: float = 120.0,
    duration_beats: Optional[float] = None,
) -> np.ndarray:
    """Apply rhythmic pattern to audio.
    
    Plays audio at each pattern position with velocity-based amplitude.
    Does not resample - uses original audio and triggers at pattern positions.
    
    Parameters
    ----------
    audio : np.ndarray
        Source audio (one-shot or loop)
    pattern : List[float]
        Pattern values (0 = off, 0.01-1.0 = velocity/amplitude)
    sr : int
        Sample rate
    bpm : float
        Tempo in BPM
    duration_beats : float, optional
        Total duration in beats (default = pattern length)
    
    Returns
    -------
    np.ndarray
        Audio with pattern applied
    """
    if not pattern or len(audio) == 0:
        return audio
    
    # Calculate timing
    beat_samples = int(sr * 60 / bpm)
    step_samples = beat_samples // len(pattern)
    
    if duration_beats is None:
        duration_beats = 1.0
    
    total_samples = int(duration_beats * beat_samples)
    
    # Create output buffer
    result = np.zeros(total_samples)
    
    # Pattern repeats
    pattern_samples = len(pattern) * step_samples
    num_repeats = (total_samples + pattern_samples - 1) // pattern_samples
    
    for repeat in range(num_repeats):
        for i, velocity in enumerate(pattern):
            if velocity <= 0:
                continue
            
            # Calculate position
            pos = repeat * pattern_samples + i * step_samples
            if pos >= total_samples:
                break
            
            # Calculate end position (don't exceed buffer)
            end_pos = min(pos + len(audio), total_samples)
            chunk_len = end_pos - pos
            
            # Apply velocity as amplitude
            result[pos:end_pos] += audio[:chunk_len] * velocity
    
    # Normalize if clipping
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result /= peak
    
    return result


def parse_pattern_string(pattern_str: str) -> List[float]:
    """Parse pattern string to velocity list.
    
    Formats:
        "1010"           -> [1.0, 0, 1.0, 0]
        "x.x."           -> [1.0, 0, 1.0, 0]
        "1 0.5 0 0.8"    -> [1.0, 0.5, 0, 0.8]
        "x-x-x---"       -> [1.0, 0, 1.0, 0, 1.0, 0, 0, 0]
    """
    pattern_str = pattern_str.strip()
    
    # Check for space-separated floats
    if ' ' in pattern_str:
        parts = pattern_str.split()
        result = []
        for p in parts:
            try:
                result.append(float(p))
            except ValueError:
                result.append(1.0 if p in 'xX1' else 0.0)
        return result
    
    # Character-based pattern
    result = []
    for c in pattern_str:
        if c in 'xX1':
            result.append(1.0)
        elif c in '.-_0':
            result.append(0.0)
        elif c.isdigit():
            result.append(int(c) / 9.0)  # 0-9 maps to 0-1
        elif c in 'lL':
            result.append(0.3)  # Low velocity
        elif c in 'mM':
            result.append(0.6)  # Medium velocity
        elif c in 'hH':
            result.append(0.9)  # High velocity
    
    return result if result else [1.0]


# ============================================================================
# BUFFER COMBINING
# ============================================================================

def combine_buffers(
    buffers: List[np.ndarray],
    mode: str = "overlay",
    sr: int = 48000,
) -> np.ndarray:
    """Combine multiple buffers into one.
    
    Parameters
    ----------
    buffers : List[np.ndarray]
        List of audio buffers
    mode : str
        Combine mode: overlay, append, mix, crossfade
    sr : int
        Sample rate
    
    Returns
    -------
    np.ndarray
        Combined buffer
    """
    if not buffers:
        return np.array([])
    
    if len(buffers) == 1:
        return buffers[0].copy()
    
    if mode == "append":
        return np.concatenate(buffers)
    
    elif mode == "overlay" or mode == "mix":
        # Find max length
        max_len = max(len(b) for b in buffers)
        
        # Pad and sum
        result = np.zeros(max_len)
        for buf in buffers:
            result[:len(buf)] += buf
        
        # Normalize
        result /= len(buffers)
        
        return result
    
    elif mode == "crossfade":
        # Crossfade between sequential buffers
        fade_samples = int(sr * 0.05)  # 50ms crossfade
        
        result = buffers[0].copy()
        
        for buf in buffers[1:]:
            # Create crossfade
            fade_len = min(fade_samples, len(result), len(buf))
            
            # Fade out end of result
            fade_out = np.linspace(1, 0, fade_len)
            result[-fade_len:] *= fade_out
            
            # Fade in start of new buffer
            fade_in = np.linspace(0, 1, fade_len)
            new_buf = buf.copy()
            new_buf[:fade_len] *= fade_in
            
            # Combine
            result[-fade_len:] += new_buf[:fade_len]
            result = np.concatenate([result, new_buf[fade_len:]])
        
        return result
    
    else:
        return np.concatenate(buffers)


# ============================================================================
# WAVETABLE GENERATION
# ============================================================================

def generate_wavetable(
    audio: np.ndarray,
    sr: int = 48000,
    num_frames: int = 256,
    frame_size: int = 2048,
) -> np.ndarray:
    """Generate wavetable from audio.
    
    Parameters
    ----------
    audio : np.ndarray
        Source audio
    sr : int
        Sample rate
    num_frames : int
        Number of wavetable frames
    frame_size : int
        Samples per frame
    
    Returns
    -------
    np.ndarray
        Wavetable array (num_frames x frame_size)
    """
    if len(audio) == 0:
        return np.zeros((num_frames, frame_size))
    
    # Convert to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    wavetable = np.zeros((num_frames, frame_size))
    
    # Extract frames from audio
    if len(audio) >= num_frames * frame_size:
        # Enough audio - extract evenly spaced frames
        step = len(audio) // num_frames
        for i in range(num_frames):
            start = i * step
            wavetable[i] = audio[start:start + frame_size]
    else:
        # Not enough audio - use single-cycle extraction
        chunks = auto_chunk(audio, sr, "wavetable", num_chunks=num_frames)
        
        for i, chunk in enumerate(chunks[:num_frames]):
            # Resample to frame_size
            chunk_audio = chunk.audio
            if len(chunk_audio) != frame_size:
                x_old = np.linspace(0, 1, len(chunk_audio))
                x_new = np.linspace(0, 1, frame_size)
                chunk_audio = np.interp(x_new, x_old, chunk_audio)
            wavetable[i] = chunk_audio
    
    # Normalize each frame
    for i in range(num_frames):
        peak = np.max(np.abs(wavetable[i]))
        if peak > 0:
            wavetable[i] /= peak
    
    return wavetable


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # User stack
    'UserStack',
    'get_user_stack',
    
    # Chunking
    'ChunkAlgorithm',
    'AudioChunk',
    'auto_chunk',
    
    # Remix
    'RemixAlgorithm',
    'remix_audio',
    
    # Rhythmic pattern
    'RhythmicPattern',
    'apply_rhythmic_pattern',
    'parse_pattern_string',
    
    # Buffer combining
    'combine_buffers',
    
    # Wavetable
    'generate_wavetable',
]
