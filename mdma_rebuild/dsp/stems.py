"""MDMA Stem Separation and Audio Sectioning.

Features:
- AI-powered stem separation (Demucs/Spleeter)
- Auto-sectioning with beat/transient detection
- Chopping and slicing
- Stem resynthesis via main engine

All audio is processed in engine-native format: float64 wave.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

if TYPE_CHECKING:
    from ..core.session import Session


# ============================================================================
# STEM TYPES
# ============================================================================

class StemType(Enum):
    VOCALS = "vocals"
    DRUMS = "drums"
    BASS = "bass"
    OTHER = "other"
    PIANO = "piano"
    GUITAR = "guitar"
    # Extended stems (6-stem model)
    MELODY = "melody"
    # Full mix
    MIX = "mix"


@dataclass
class Stem:
    """A separated audio stem."""
    stem_type: StemType
    audio: np.ndarray  # float64
    sample_rate: int
    source_name: str = ""
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate
    
    def normalize(self) -> "Stem":
        """Normalize to -1/+1 range."""
        peak = np.max(np.abs(self.audio))
        if peak > 0:
            self.audio = self.audio / peak * 0.95
        return self
    
    def to_mono(self) -> "Stem":
        """Convert to mono if stereo."""
        if len(self.audio.shape) > 1:
            self.audio = self.audio.mean(axis=1)
        return self


@dataclass
class StemSet:
    """Collection of separated stems from one source."""
    stems: Dict[StemType, Stem] = field(default_factory=dict)
    source_name: str = ""
    sample_rate: int = 48000
    
    def get(self, stem_type: StemType) -> Optional[Stem]:
        return self.stems.get(stem_type)
    
    def add(self, stem: Stem):
        self.stems[stem.stem_type] = stem
    
    def list_stems(self) -> List[str]:
        return [s.value for s in self.stems.keys()]
    
    def remix(self, levels: Dict[StemType, float]) -> np.ndarray:
        """Remix stems with given levels.
        
        Parameters
        ----------
        levels : dict
            StemType -> volume level (0-1+)
        
        Returns
        -------
        np.ndarray
            Mixed audio (float64)
        """
        if not self.stems:
            return np.array([])
        
        # Find max length
        max_len = max(len(s.audio) for s in self.stems.values())
        
        # Mix
        out = np.zeros(max_len, dtype=np.float64)
        for stem_type, stem in self.stems.items():
            level = levels.get(stem_type, 1.0)
            if level > 0:
                audio = stem.audio
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                out[:len(audio)] += audio * level
        
        # Normalize
        peak = np.max(np.abs(out))
        if peak > 1.0:
            out = out / peak * 0.95
        
        return out


# ============================================================================
# STEM SEPARATION ENGINE
# ============================================================================

# Global model cache
_separation_model = None
_model_name = None


def get_separator(model: str = "htdemucs"):
    """Get or load stem separation model.
    
    Models:
    - htdemucs: High-quality 4-stem (vocals, drums, bass, other)
    - htdemucs_6s: 6-stem (adds piano, guitar)
    - mdx_extra: MDX-Net based
    - spleeter: Deezer Spleeter (lighter weight)
    """
    global _separation_model, _model_name
    
    if _separation_model is not None and _model_name == model:
        return _separation_model
    
    try:
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        
        _separation_model = get_model(model)
        _model_name = model
        
        # Move to GPU if available
        if torch.cuda.is_available():
            _separation_model.cuda()
        
        return _separation_model
        
    except ImportError:
        raise ImportError(
            "Stem separation requires: pip install demucs torch\n"
            "Or for lighter weight: pip install spleeter"
        )


def separate_stems(
    audio: np.ndarray,
    sample_rate: int = 48000,
    model: str = "htdemucs",
    device: str = None,
) -> StemSet:
    """Separate audio into stems.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio (float64, mono or stereo)
    sample_rate : int
        Sample rate
    model : str
        Separation model to use
    device : str
        Device (cuda, cpu, auto)
    
    Returns
    -------
    StemSet
        Collection of separated stems
    """
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    
    # Auto device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    separator = get_separator(model)
    
    # Prepare audio - demucs expects stereo
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    
    # Convert to tensor
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    
    if device == 'cuda':
        audio_tensor = audio_tensor.cuda()
    
    # Resample if needed (demucs expects 44100)
    if sample_rate != 44100:
        # Simple resample
        ratio = 44100 / sample_rate
        new_len = int(audio_tensor.shape[-1] * ratio)
        audio_tensor = torch.nn.functional.interpolate(
            audio_tensor, size=new_len, mode='linear', align_corners=False
        )
    
    # Separate
    with torch.no_grad():
        sources = apply_model(separator, audio_tensor, device=device)
    
    # Convert back
    sources = sources.cpu().numpy()[0]  # Remove batch dim
    
    # Resample back if needed
    if sample_rate != 44100:
        ratio = sample_rate / 44100
        new_sources = []
        for src in sources:
            new_len = int(src.shape[-1] * ratio)
            # Simple interpolation
            x_old = np.linspace(0, 1, src.shape[-1])
            x_new = np.linspace(0, 1, new_len)
            new_src = np.array([np.interp(x_new, x_old, ch) for ch in src])
            new_sources.append(new_src)
        sources = np.array(new_sources)
    
    # Build StemSet
    stem_set = StemSet(sample_rate=sample_rate)
    
    # Map model outputs to stem types
    stem_names = separator.sources  # e.g., ['drums', 'bass', 'other', 'vocals']
    
    type_map = {
        'drums': StemType.DRUMS,
        'bass': StemType.BASS,
        'vocals': StemType.VOCALS,
        'other': StemType.OTHER,
        'piano': StemType.PIANO,
        'guitar': StemType.GUITAR,
    }
    
    for i, name in enumerate(stem_names):
        stem_type = type_map.get(name, StemType.OTHER)
        stem_audio = sources[i].mean(axis=0).astype(np.float64)  # Mono, float64
        
        stem = Stem(
            stem_type=stem_type,
            audio=stem_audio,
            sample_rate=sample_rate,
        )
        stem_set.add(stem)
    
    return stem_set


def separate_stems_spleeter(
    audio: np.ndarray,
    sample_rate: int = 48000,
    stems: int = 4,
) -> StemSet:
    """Separate using Spleeter (lighter weight alternative).
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio
    sample_rate : int
        Sample rate
    stems : int
        Number of stems (2, 4, or 5)
    
    Returns
    -------
    StemSet
        Separated stems
    """
    try:
        from spleeter.separator import Separator
        from spleeter.audio.adapter import AudioAdapter
    except ImportError:
        raise ImportError("Spleeter requires: pip install spleeter")
    
    # Initialize separator
    separator = Separator(f'spleeter:{stems}stems')
    
    # Ensure stereo
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio], axis=1)
    elif audio.shape[0] == 2:
        audio = audio.T
    
    # Separate
    prediction = separator.separate(audio)
    
    # Build StemSet
    stem_set = StemSet(sample_rate=sample_rate)
    
    type_map = {
        'vocals': StemType.VOCALS,
        'accompaniment': StemType.OTHER,
        'drums': StemType.DRUMS,
        'bass': StemType.BASS,
        'piano': StemType.PIANO,
        'other': StemType.OTHER,
    }
    
    for name, stem_audio in prediction.items():
        stem_type = type_map.get(name, StemType.OTHER)
        mono = stem_audio.mean(axis=1).astype(np.float64)
        
        stem = Stem(
            stem_type=stem_type,
            audio=mono,
            sample_rate=sample_rate,
        )
        stem_set.add(stem)
    
    return stem_set


# ============================================================================
# AUDIO SECTIONING
# ============================================================================

@dataclass
class Section:
    """A section of audio with metadata."""
    start_sample: int
    end_sample: int
    sample_rate: int
    label: str = ""
    confidence: float = 1.0
    
    @property
    def start_time(self) -> float:
        return self.start_sample / self.sample_rate
    
    @property
    def end_time(self) -> float:
        return self.end_sample / self.sample_rate
    
    @property
    def duration(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate


@dataclass
class Chop:
    """A chopped slice of audio."""
    audio: np.ndarray
    start_time: float
    end_time: float
    label: str = ""
    beat_aligned: bool = False


def detect_onsets(
    audio: np.ndarray,
    sample_rate: int = 48000,
    threshold: float = 0.5,
) -> np.ndarray:
    """Detect onset times in audio.
    
    Returns array of onset sample positions.
    """
    try:
        import librosa
        onset_frames = librosa.onset.onset_detect(
            y=audio.astype(np.float32),
            sr=sample_rate,
            units='samples',
            backtrack=True,
        )
        return onset_frames
    except ImportError:
        pass
    
    # Fallback: simple energy-based detection
    hop = sample_rate // 100  # 10ms hops
    energy = np.array([
        np.sum(audio[i:i+hop]**2) 
        for i in range(0, len(audio) - hop, hop)
    ])
    
    # Normalize
    energy = energy / (np.max(energy) + 1e-10)
    
    # Find peaks
    diff = np.diff(energy)
    peaks = np.where((diff[:-1] > 0) & (diff[1:] < 0) & (energy[1:-1] > threshold))[0]
    
    return peaks * hop


def detect_beats(
    audio: np.ndarray,
    sample_rate: int = 48000,
) -> Tuple[float, np.ndarray]:
    """Detect tempo and beat positions.
    
    Returns (tempo_bpm, beat_sample_positions).
    """
    try:
        import librosa
        tempo, beats = librosa.beat.beat_track(
            y=audio.astype(np.float32),
            sr=sample_rate,
            units='samples',
        )
        return float(tempo), beats
    except ImportError:
        pass
    
    # Fallback: estimate from onsets
    onsets = detect_onsets(audio, sample_rate)
    if len(onsets) < 2:
        return 120.0, np.array([0])
    
    # Estimate tempo from inter-onset intervals
    intervals = np.diff(onsets) / sample_rate
    median_interval = np.median(intervals)
    tempo = 60.0 / median_interval if median_interval > 0 else 120.0
    
    # Clamp to reasonable range
    while tempo > 200:
        tempo /= 2
    while tempo < 60:
        tempo *= 2
    
    return tempo, onsets


def auto_section(
    audio: np.ndarray,
    sample_rate: int = 48000,
    min_section_seconds: float = 4.0,
) -> List[Section]:
    """Automatically detect sections in audio.
    
    Uses spectral analysis to find structural boundaries.
    """
    try:
        import librosa
        
        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(
            y=audio.astype(np.float32),
            sr=sample_rate,
            n_mels=128,
        )
        
        # Self-similarity matrix
        R = librosa.segment.recurrence_matrix(
            S, mode='affinity', sym=True
        )
        
        # Detect boundaries
        boundaries = librosa.segment.agglomerative(
            R, k=max(2, int(len(audio) / sample_rate / 30))  # ~30s sections
        )
        
        # Convert frames to samples
        hop_length = 512
        boundary_samples = librosa.frames_to_samples(boundaries, hop_length=hop_length)
        
    except ImportError:
        # Fallback: use beat-based sectioning
        tempo, beats = detect_beats(audio, sample_rate)
        
        # Every 16 beats = 1 section
        beats_per_section = 16
        boundary_samples = [0]
        for i in range(beats_per_section, len(beats), beats_per_section):
            boundary_samples.append(beats[i])
        boundary_samples.append(len(audio))
        boundary_samples = np.array(boundary_samples)
    
    # Build sections
    sections = []
    min_samples = int(min_section_seconds * sample_rate)
    
    for i in range(len(boundary_samples) - 1):
        start = boundary_samples[i]
        end = boundary_samples[i + 1]
        
        if end - start >= min_samples:
            sections.append(Section(
                start_sample=start,
                end_sample=end,
                sample_rate=sample_rate,
                label=f"section_{i+1}",
            ))
    
    return sections


def chop_audio(
    audio: np.ndarray,
    sample_rate: int = 48000,
    mode: str = "beat",
    divisions: int = 16,
    snap_to_zero: bool = True,
) -> List[Chop]:
    """Chop audio into slices.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio
    sample_rate : int
        Sample rate
    mode : str
        Chop mode: 'beat', 'time', 'transient'
    divisions : int
        Number of divisions (for beat/time mode)
    snap_to_zero : bool
        Snap chop points to zero crossings
    
    Returns
    -------
    list
        List of Chop objects
    """
    chops = []
    
    if mode == "beat":
        tempo, beats = detect_beats(audio, sample_rate)
        
        # Use beats as chop points
        if len(beats) < 2:
            # Fall back to even division
            chop_points = np.linspace(0, len(audio), divisions + 1, dtype=int)
        else:
            # Use every beat or every N beats
            beat_skip = max(1, len(beats) // divisions)
            chop_points = beats[::beat_skip]
            if chop_points[-1] != len(audio):
                chop_points = np.append(chop_points, len(audio))
        
        beat_aligned = True
        
    elif mode == "transient":
        onsets = detect_onsets(audio, sample_rate)
        chop_points = np.concatenate([[0], onsets, [len(audio)]])
        chop_points = np.unique(chop_points)
        beat_aligned = False
        
    else:  # time mode
        chop_points = np.linspace(0, len(audio), divisions + 1, dtype=int)
        beat_aligned = False
    
    # Snap to zero crossings
    if snap_to_zero:
        for i in range(1, len(chop_points) - 1):
            pos = chop_points[i]
            # Search nearby for zero crossing
            window = min(1000, pos, len(audio) - pos)
            segment = audio[pos - window:pos + window]
            crossings = np.where(np.diff(np.signbit(segment)))[0]
            if len(crossings) > 0:
                # Find closest crossing
                closest = crossings[np.argmin(np.abs(crossings - window))]
                chop_points[i] = pos - window + closest
    
    # Create chops
    for i in range(len(chop_points) - 1):
        start = int(chop_points[i])
        end = int(chop_points[i + 1])
        
        chop = Chop(
            audio=audio[start:end].astype(np.float64),
            start_time=start / sample_rate,
            end_time=end / sample_rate,
            label=f"chop_{i+1:02d}",
            beat_aligned=beat_aligned,
        )
        chops.append(chop)
    
    return chops


# ============================================================================
# STEM RESYNTHESIS
# ============================================================================

def resynthesize_stem(
    stem: Stem,
    engine: Any,
    mode: str = "granular",
    **params,
) -> np.ndarray:
    """Resynthesize a stem through the main engine.
    
    Parameters
    ----------
    stem : Stem
        Input stem
    engine : MonolithEngine
        MDMA synthesis engine
    mode : str
        Resynthesis mode: 'granular', 'spectral', 'additive', 'fm'
    **params
        Mode-specific parameters
    
    Returns
    -------
    np.ndarray
        Resynthesized audio (float64)
    """
    audio = stem.audio
    sr = stem.sample_rate
    
    if mode == "granular":
        # Granular resynthesis
        grain_size = params.get('grain_size', 0.05)  # 50ms
        density = params.get('density', 10)
        pitch = params.get('pitch', 1.0)
        
        grain_samples = int(grain_size * sr)
        out_len = int(len(audio) * pitch)
        out = np.zeros(out_len, dtype=np.float64)
        
        # Simple granular - scatter grains
        num_grains = int(out_len / sr * density)
        for _ in range(num_grains):
            # Random grain position
            src_pos = np.random.randint(0, max(1, len(audio) - grain_samples))
            dst_pos = np.random.randint(0, max(1, out_len - grain_samples))
            
            # Extract grain
            grain = audio[src_pos:src_pos + grain_samples]
            
            # Apply window
            window = np.hanning(len(grain))
            grain = grain * window
            
            # Add to output
            end_pos = min(dst_pos + len(grain), out_len)
            out[dst_pos:end_pos] += grain[:end_pos - dst_pos]
        
        # Normalize
        peak = np.max(np.abs(out))
        if peak > 0:
            out = out / peak * 0.95
        
        return out
    
    elif mode == "spectral":
        # Spectral freeze/smear
        try:
            from scipy import signal
            
            freeze_factor = params.get('freeze', 0.5)
            
            # STFT
            f, t, Zxx = signal.stft(audio, sr, nperseg=2048)
            
            # Smear phases
            phase = np.angle(Zxx)
            mag = np.abs(Zxx)
            
            # Random phase deviation
            phase += np.random.randn(*phase.shape) * freeze_factor
            
            # Reconstruct
            Zxx_mod = mag * np.exp(1j * phase)
            _, out = signal.istft(Zxx_mod, sr)
            
            return out.astype(np.float64)
            
        except ImportError:
            return audio
    
    elif mode == "additive":
        # Additive resynthesis from detected peaks
        try:
            import librosa
            
            # Harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(audio.astype(np.float32))
            
            # Simple additive: keep harmonics, filter percussive
            mix = params.get('harmonic_mix', 0.8)
            out = harmonic * mix + percussive * (1 - mix)
            
            return out.astype(np.float64)
            
        except ImportError:
            return audio
    
    elif mode == "fm":
        # FM resynthesis - use audio as modulator
        carrier_freq = params.get('carrier', 200)
        mod_depth = params.get('depth', 500)
        
        t = np.arange(len(audio)) / sr
        
        # Audio as modulator (envelope follower)
        from scipy.ndimage import uniform_filter1d
        envelope = np.abs(audio)
        envelope = uniform_filter1d(envelope, size=int(sr * 0.01))
        
        # FM synthesis
        carrier = np.sin(2 * np.pi * carrier_freq * t + envelope * mod_depth)
        
        # Apply original envelope
        out = carrier * envelope
        
        return out.astype(np.float64)
    
    return audio


# ============================================================================
# ANALYSIS INTEGRATION
# ============================================================================

def analyze_stem(stem: Stem) -> Dict[str, Any]:
    """Analyze a stem and return attributes.
    
    Integrates with MDMA AI analysis system.
    """
    try:
        from ..ai.analysis import analyze_audio
        return analyze_audio(stem.audio, stem.sample_rate)
    except ImportError:
        pass
    
    # Basic analysis fallback
    audio = stem.audio
    sr = stem.sample_rate
    
    # RMS energy
    rms = np.sqrt(np.mean(audio**2))
    
    # Peak
    peak = np.max(np.abs(audio))
    
    # Zero crossing rate
    zcr = np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio)
    
    # Spectral centroid (simple)
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1/sr)
    centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-10)
    
    return {
        'rms': float(rms),
        'peak': float(peak),
        'zcr': float(zcr),
        'spectral_centroid': float(centroid),
        'duration': len(audio) / sr,
        'stem_type': stem.stem_type.value,
    }


def describe_stem(stem: Stem) -> str:
    """Generate text description of a stem."""
    attrs = analyze_stem(stem)
    
    # Map attributes to descriptors
    descriptors = []
    
    if attrs['rms'] > 0.3:
        descriptors.append("loud")
    elif attrs['rms'] < 0.1:
        descriptors.append("quiet")
    
    if attrs['spectral_centroid'] > 3000:
        descriptors.append("bright")
    elif attrs['spectral_centroid'] < 500:
        descriptors.append("dark")
    
    if attrs['zcr'] > 0.1:
        descriptors.append("noisy")
    else:
        descriptors.append("tonal")
    
    stem_name = attrs['stem_type']
    desc_str = ", ".join(descriptors) if descriptors else "neutral"
    
    return f"{stem_name}: {desc_str} ({attrs['duration']:.1f}s)"
