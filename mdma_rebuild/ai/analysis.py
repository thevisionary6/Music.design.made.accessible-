"""MDMA Deep Audio Analysis Module.

Comprehensive attribute detection with weighted attribute pool.
Analyzes samples across multiple dimensions: spectral, temporal, dynamic, harmonic, etc.

Section N2 of MDMA Master Feature List.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json


# ============================================================================
# MASTER ATTRIBUTE POOL
# ============================================================================

# Each attribute has: description, range, category, weight (importance)
ATTRIBUTE_POOL = {
    # === SPECTRAL ATTRIBUTES ===
    'brightness': {
        'desc': 'High frequency energy (>4kHz)',
        'range': (0, 100),
        'category': 'spectral',
        'weight': 1.0,
    },
    'darkness': {
        'desc': 'Low frequency dominance (<300Hz)',
        'range': (0, 100),
        'category': 'spectral',
        'weight': 1.0,
    },
    'warmth': {
        'desc': 'Mid-low presence (200-800Hz)',
        'range': (0, 100),
        'category': 'spectral',
        'weight': 0.9,
    },
    'harshness': {
        'desc': 'Aggressive high-mids (2-6kHz)',
        'range': (0, 100),
        'category': 'spectral',
        'weight': 0.8,
    },
    'airiness': {
        'desc': 'Ultra-high sparkle (>10kHz)',
        'range': (0, 100),
        'category': 'spectral',
        'weight': 0.7,
    },
    'muddiness': {
        'desc': 'Low-mid congestion (200-500Hz)',
        'range': (0, 100),
        'category': 'spectral',
        'weight': 0.6,
    },
    'presence': {
        'desc': 'Vocal presence range (1-5kHz)',
        'range': (0, 100),
        'category': 'spectral',
        'weight': 0.9,
    },
    'spectral_centroid': {
        'desc': 'Center of spectral mass (Hz)',
        'range': (20, 15000),
        'category': 'spectral',
        'weight': 1.0,
    },
    'spectral_spread': {
        'desc': 'Bandwidth around centroid',
        'range': (0, 100),
        'category': 'spectral',
        'weight': 0.8,
    },
    'spectral_rolloff': {
        'desc': 'Frequency containing 85% energy',
        'range': (100, 20000),
        'category': 'spectral',
        'weight': 0.7,
    },
    'spectral_flatness': {
        'desc': 'Noise-like vs tonal (0=tonal, 100=noise)',
        'range': (0, 100),
        'category': 'spectral',
        'weight': 0.9,
    },
    'spectral_flux': {
        'desc': 'Rate of spectral change',
        'range': (0, 100),
        'category': 'spectral',
        'weight': 0.8,
    },
    'spectral_contrast': {
        'desc': 'Peak vs valley difference per band',
        'range': (0, 100),
        'category': 'spectral',
        'weight': 0.7,
    },
    
    # === ENVELOPE / TEMPORAL ATTRIBUTES ===
    'attack': {
        'desc': 'Attack speed (0=slow, 100=instant)',
        'range': (0, 100),
        'category': 'envelope',
        'weight': 1.0,
    },
    'attack_time_ms': {
        'desc': 'Time to reach peak (ms)',
        'range': (0, 1000),
        'category': 'envelope',
        'weight': 0.9,
    },
    'decay': {
        'desc': 'Decay rate after peak',
        'range': (0, 100),
        'category': 'envelope',
        'weight': 0.9,
    },
    'sustain': {
        'desc': 'Sustained body level',
        'range': (0, 100),
        'category': 'envelope',
        'weight': 0.8,
    },
    'release': {
        'desc': 'Release/tail length',
        'range': (0, 100),
        'category': 'envelope',
        'weight': 0.8,
    },
    'transient_strength': {
        'desc': 'Initial transient punch',
        'range': (0, 100),
        'category': 'envelope',
        'weight': 1.0,
    },
    'punch': {
        'desc': 'Percussive impact',
        'range': (0, 100),
        'category': 'envelope',
        'weight': 0.9,
    },
    'snap': {
        'desc': 'Initial click/snap',
        'range': (0, 100),
        'category': 'envelope',
        'weight': 0.8,
    },
    'body': {
        'desc': 'Mid-section fullness',
        'range': (0, 100),
        'category': 'envelope',
        'weight': 0.8,
    },
    'tail': {
        'desc': 'Reverb/decay tail',
        'range': (0, 100),
        'category': 'envelope',
        'weight': 0.7,
    },
    'envelope_smoothness': {
        'desc': 'Smooth vs choppy envelope',
        'range': (0, 100),
        'category': 'envelope',
        'weight': 0.6,
    },
    
    # === DYNAMICS ATTRIBUTES ===
    'dynamic_range': {
        'desc': 'Peak to quiet ratio (dB)',
        'range': (0, 100),
        'category': 'dynamics',
        'weight': 0.9,
    },
    'compression': {
        'desc': 'Perceived compression amount',
        'range': (0, 100),
        'category': 'dynamics',
        'weight': 0.8,
    },
    'loudness': {
        'desc': 'Perceived loudness (LUFS-style)',
        'range': (0, 100),
        'category': 'dynamics',
        'weight': 1.0,
    },
    'peak_level': {
        'desc': 'Maximum amplitude',
        'range': (0, 100),
        'category': 'dynamics',
        'weight': 0.7,
    },
    'rms_level': {
        'desc': 'Average RMS energy',
        'range': (0, 100),
        'category': 'dynamics',
        'weight': 0.8,
    },
    'crest_factor': {
        'desc': 'Peak to RMS ratio',
        'range': (0, 100),
        'category': 'dynamics',
        'weight': 0.7,
    },
    'limiting': {
        'desc': 'Perceived limiting/clipping',
        'range': (0, 100),
        'category': 'dynamics',
        'weight': 0.6,
    },
    
    # === TEXTURE / DENSITY ATTRIBUTES ===
    'density': {
        'desc': 'Event density over time',
        'range': (0, 100),
        'category': 'texture',
        'weight': 0.9,
    },
    'sparsity': {
        'desc': 'Sparse/minimal content',
        'range': (0, 100),
        'category': 'texture',
        'weight': 0.8,
    },
    'complexity': {
        'desc': 'Textural complexity',
        'range': (0, 100),
        'category': 'texture',
        'weight': 0.9,
    },
    'smoothness': {
        'desc': 'Smooth vs granular',
        'range': (0, 100),
        'category': 'texture',
        'weight': 0.7,
    },
    'roughness': {
        'desc': 'Surface roughness/grain',
        'range': (0, 100),
        'category': 'texture',
        'weight': 0.7,
    },
    'noisiness': {
        'desc': 'Noise content amount',
        'range': (0, 100),
        'category': 'texture',
        'weight': 0.8,
    },
    'tonality': {
        'desc': 'Tonal vs noise content',
        'range': (0, 100),
        'category': 'texture',
        'weight': 0.9,
    },
    'granularity': {
        'desc': 'Granular quality',
        'range': (0, 100),
        'category': 'texture',
        'weight': 0.6,
    },
    
    # === HARMONIC ATTRIBUTES ===
    'harmonic_richness': {
        'desc': 'Number/strength of harmonics',
        'range': (0, 100),
        'category': 'harmonic',
        'weight': 0.9,
    },
    'inharmonicity': {
        'desc': 'Inharmonic partial content',
        'range': (0, 100),
        'category': 'harmonic',
        'weight': 0.8,
    },
    'fundamental_strength': {
        'desc': 'Fundamental frequency prominence',
        'range': (0, 100),
        'category': 'harmonic',
        'weight': 0.9,
    },
    'odd_harmonics': {
        'desc': 'Odd harmonic emphasis',
        'range': (0, 100),
        'category': 'harmonic',
        'weight': 0.7,
    },
    'even_harmonics': {
        'desc': 'Even harmonic emphasis',
        'range': (0, 100),
        'category': 'harmonic',
        'weight': 0.7,
    },
    'harmonic_decay': {
        'desc': 'How fast harmonics roll off',
        'range': (0, 100),
        'category': 'harmonic',
        'weight': 0.6,
    },
    'distortion': {
        'desc': 'Harmonic distortion amount',
        'range': (0, 100),
        'category': 'harmonic',
        'weight': 0.8,
    },
    'saturation': {
        'desc': 'Warm saturation level',
        'range': (0, 100),
        'category': 'harmonic',
        'weight': 0.7,
    },
    
    # === PITCH ATTRIBUTES ===
    'pitch_clarity': {
        'desc': 'Clear pitch vs pitched noise',
        'range': (0, 100),
        'category': 'pitch',
        'weight': 0.9,
    },
    'pitch_hz': {
        'desc': 'Estimated fundamental (Hz)',
        'range': (20, 8000),
        'category': 'pitch',
        'weight': 1.0,
    },
    'pitch_midi': {
        'desc': 'MIDI note number',
        'range': (0, 127),
        'category': 'pitch',
        'weight': 0.8,
    },
    'pitch_stability': {
        'desc': 'Pitch consistency over time',
        'range': (0, 100),
        'category': 'pitch',
        'weight': 0.8,
    },
    'vibrato_amount': {
        'desc': 'Vibrato/wobble depth',
        'range': (0, 100),
        'category': 'pitch',
        'weight': 0.6,
    },
    'vibrato_rate': {
        'desc': 'Vibrato speed (Hz)',
        'range': (0, 20),
        'category': 'pitch',
        'weight': 0.5,
    },
    'pitch_bend': {
        'desc': 'Pitch slide/glide amount',
        'range': (0, 100),
        'category': 'pitch',
        'weight': 0.6,
    },
    
    # === RHYTHM ATTRIBUTES ===
    'rhythmic': {
        'desc': 'Rhythmic vs sustained',
        'range': (0, 100),
        'category': 'rhythm',
        'weight': 0.9,
    },
    'periodicity': {
        'desc': 'Periodic/repetitive patterns',
        'range': (0, 100),
        'category': 'rhythm',
        'weight': 0.8,
    },
    'tempo_bpm': {
        'desc': 'Implied tempo (BPM)',
        'range': (40, 200),
        'category': 'rhythm',
        'weight': 0.7,
    },
    'groove': {
        'desc': 'Groove/swing feel',
        'range': (0, 100),
        'category': 'rhythm',
        'weight': 0.6,
    },
    'syncopation': {
        'desc': 'Off-beat emphasis',
        'range': (0, 100),
        'category': 'rhythm',
        'weight': 0.5,
    },
    
    # === CHARACTER / MOOD ATTRIBUTES ===
    'aggressive': {
        'desc': 'Aggressive/harsh character',
        'range': (0, 100),
        'category': 'character',
        'weight': 0.9,
    },
    'gentle': {
        'desc': 'Soft/gentle character',
        'range': (0, 100),
        'category': 'character',
        'weight': 0.8,
    },
    'metallic': {
        'desc': 'Metallic/bell-like quality',
        'range': (0, 100),
        'category': 'character',
        'weight': 0.8,
    },
    'woody': {
        'desc': 'Woody/organic quality',
        'range': (0, 100),
        'category': 'character',
        'weight': 0.7,
    },
    'digital': {
        'desc': 'Digital/synthetic feel',
        'range': (0, 100),
        'category': 'character',
        'weight': 0.8,
    },
    'analog': {
        'desc': 'Analog/warm character',
        'range': (0, 100),
        'category': 'character',
        'weight': 0.8,
    },
    'clean': {
        'desc': 'Clean/pure sound',
        'range': (0, 100),
        'category': 'character',
        'weight': 0.7,
    },
    'dirty': {
        'desc': 'Dirty/gritty sound',
        'range': (0, 100),
        'category': 'character',
        'weight': 0.7,
    },
    'organic': {
        'desc': 'Natural/acoustic feel',
        'range': (0, 100),
        'category': 'character',
        'weight': 0.7,
    },
    'synthetic': {
        'desc': 'Synthesized/electronic',
        'range': (0, 100),
        'category': 'character',
        'weight': 0.7,
    },
    
    # === SPATIAL ATTRIBUTES ===
    'stereo_width': {
        'desc': 'Stereo spread',
        'range': (0, 100),
        'category': 'spatial',
        'weight': 0.7,
    },
    'mono_compatibility': {
        'desc': 'How well it sums to mono',
        'range': (0, 100),
        'category': 'spatial',
        'weight': 0.6,
    },
    'depth': {
        'desc': 'Front-to-back depth',
        'range': (0, 100),
        'category': 'spatial',
        'weight': 0.7,
    },
    'reverberance': {
        'desc': 'Room/reverb amount',
        'range': (0, 100),
        'category': 'spatial',
        'weight': 0.8,
    },
    'room_size': {
        'desc': 'Implied room size',
        'range': (0, 100),
        'category': 'spatial',
        'weight': 0.6,
    },
}

# Total: 80+ attributes across 9 categories


# ============================================================================
# ATTRIBUTE VECTOR
# ============================================================================

@dataclass
class AttributeVector:
    """Container for analyzed audio attributes."""
    
    attributes: Dict[str, float] = field(default_factory=dict)
    sample_rate: int = 48000
    duration: float = 0.0
    source: str = ""  # Filename or description
    
    def get(self, name: str, default: float = 0.0) -> float:
        """Get attribute value."""
        return self.attributes.get(name, default)
    
    def set(self, name: str, value: float) -> None:
        """Set attribute value with range clamping."""
        if name in ATTRIBUTE_POOL:
            min_val, max_val = ATTRIBUTE_POOL[name]['range']
            value = max(min_val, min(max_val, value))
        self.attributes[name] = value
    
    def top(self, n: int = 10, category: Optional[str] = None) -> List[Tuple[str, float]]:
        """Get top N attributes by value.
        
        Parameters
        ----------
        n : int
            Number of attributes to return
        category : str, optional
            Filter by category
        
        Returns
        -------
        list
            List of (name, value) tuples
        """
        items = self.attributes.items()
        if category:
            items = [(k, v) for k, v in items 
                     if ATTRIBUTE_POOL.get(k, {}).get('category') == category]
        
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        return sorted_items[:n]
    
    def by_category(self) -> Dict[str, Dict[str, float]]:
        """Group attributes by category."""
        result = {}
        for name, value in self.attributes.items():
            cat = ATTRIBUTE_POOL.get(name, {}).get('category', 'other')
            if cat not in result:
                result[cat] = {}
            result[cat][name] = value
        return result
    
    def category_averages(self) -> Dict[str, float]:
        """Get weighted average per category."""
        by_cat = self.by_category()
        averages = {}
        for cat, attrs in by_cat.items():
            total_weight = 0
            weighted_sum = 0
            for name, value in attrs.items():
                weight = ATTRIBUTE_POOL.get(name, {}).get('weight', 1.0)
                weighted_sum += value * weight
                total_weight += weight
            if total_weight > 0:
                averages[cat] = weighted_sum / total_weight
        return averages
    
    def similarity(self, other: 'AttributeVector') -> float:
        """Cosine similarity to another vector (0-1)."""
        common = set(self.attributes.keys()) & set(other.attributes.keys())
        if not common:
            return 0.0
        
        vec1 = np.array([self.attributes[k] for k in common])
        vec2 = np.array([other.attributes[k] for k in common])
        
        # Normalize by attribute ranges
        for i, k in enumerate(common):
            if k in ATTRIBUTE_POOL:
                min_v, max_v = ATTRIBUTE_POOL[k]['range']
                vec1[i] = (vec1[i] - min_v) / (max_v - min_v + 0.001)
                vec2[i] = (vec2[i] - min_v) / (max_v - min_v + 0.001)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def distance(self, other: 'AttributeVector') -> float:
        """Euclidean distance to another vector (lower = more similar)."""
        common = set(self.attributes.keys()) & set(other.attributes.keys())
        if not common:
            return float('inf')
        
        sq_diff = 0.0
        for k in common:
            v1, v2 = self.attributes[k], other.attributes[k]
            # Normalize by range
            if k in ATTRIBUTE_POOL:
                min_v, max_v = ATTRIBUTE_POOL[k]['range']
                v1 = (v1 - min_v) / (max_v - min_v + 0.001)
                v2 = (v2 - min_v) / (max_v - min_v + 0.001)
            sq_diff += (v1 - v2) ** 2
        
        return np.sqrt(sq_diff / len(common))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'attributes': self.attributes,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'source': self.source,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttributeVector':
        """Create from dictionary."""
        return cls(
            attributes=data.get('attributes', {}),
            sample_rate=data.get('sample_rate', 48000),
            duration=data.get('duration', 0.0),
            source=data.get('source', ''),
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AttributeVector':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# ============================================================================
# AUDIO ANALYZER
# ============================================================================

class DeepAnalyzer:
    """Deep audio analyzer with comprehensive attribute detection."""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self._librosa_available = None
    
    def _check_librosa(self) -> bool:
        """Check if librosa is available."""
        if self._librosa_available is None:
            try:
                import librosa
                self._librosa_available = True
            except ImportError:
                self._librosa_available = False
        return self._librosa_available
    
    def analyze(
        self,
        audio: np.ndarray,
        detailed: bool = True,
        source: str = "",
    ) -> AttributeVector:
        """Perform deep analysis on audio.
        
        Parameters
        ----------
        audio : np.ndarray
            Audio samples (mono, -1 to 1)
        detailed : bool
            If True, compute all attributes (slower)
        source : str
            Source identifier for the audio
        
        Returns
        -------
        AttributeVector
            Complete attribute analysis
        """
        # Ensure mono and float
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float64)
        
        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        
        result = AttributeVector(
            sample_rate=self.sample_rate,
            duration=len(audio) / self.sample_rate,
            source=source,
        )
        
        # Core analysis (always)
        self._analyze_dynamics(audio, result)
        self._analyze_envelope(audio, result)
        self._analyze_spectral_basic(audio, result)
        
        # Detailed analysis (optional)
        if detailed:
            if self._check_librosa():
                self._analyze_spectral_advanced(audio, result)
                self._analyze_pitch(audio, result)
                self._analyze_rhythm(audio, result)
            self._analyze_harmonic(audio, result)
        
        # Derived attributes
        self._compute_character(result)
        
        return result
    
    def _analyze_dynamics(self, audio: np.ndarray, result: AttributeVector) -> None:
        """Analyze dynamic characteristics."""
        # RMS
        rms = np.sqrt(np.mean(audio ** 2))
        result.set('rms_level', rms * 100)
        
        # Peak
        peak = np.max(np.abs(audio))
        result.set('peak_level', peak * 100)
        
        # Crest factor
        if rms > 0.001:
            crest = peak / rms
            result.set('crest_factor', min(100, (crest - 1) * 15))
        
        # Dynamic range (via frame RMS)
        frame_size = 1024
        n_frames = len(audio) // frame_size
        if n_frames > 2:
            frame_rms = []
            for i in range(n_frames):
                frame = audio[i*frame_size:(i+1)*frame_size]
                r = np.sqrt(np.mean(frame ** 2))
                if r > 0.001:
                    frame_rms.append(r)
            
            if len(frame_rms) > 2:
                dr_db = 20 * np.log10(max(frame_rms) / (min(frame_rms) + 0.001))
                result.set('dynamic_range', min(100, dr_db * 2))
                result.set('compression', 100 - result.get('dynamic_range'))
        
        # Loudness (simplified LUFS-like)
        result.set('loudness', min(100, rms * 150))
        
        # Limiting detection (consecutive peaks at max)
        near_peak = np.abs(audio) > 0.95
        consecutive = np.diff(np.where(np.diff(near_peak.astype(int)))[0])
        if len(consecutive) > 0:
            limiting = min(100, np.mean(consecutive) * 2)
            result.set('limiting', limiting)
    
    def _analyze_envelope(self, audio: np.ndarray, result: AttributeVector) -> None:
        """Analyze envelope/ADSR characteristics."""
        # Amplitude envelope
        env = np.abs(audio)
        
        # Smooth envelope
        win = min(512, len(env) // 20)
        if win > 1:
            env_smooth = np.convolve(env, np.ones(win)/win, mode='same')
        else:
            env_smooth = env
        
        # Peak location
        peak_idx = np.argmax(env_smooth)
        peak_pos = peak_idx / len(env_smooth)
        
        # Attack
        attack_sharpness = 100 * (1 - peak_pos) if peak_pos < 0.5 else 50
        result.set('attack', min(100, attack_sharpness * 2))
        
        # Attack time in ms
        threshold = env_smooth[peak_idx] * 0.1
        attack_samples = np.argmax(env_smooth > threshold)
        attack_ms = (peak_idx - attack_samples) / self.sample_rate * 1000
        result.set('attack_time_ms', max(0, attack_ms))
        
        # Transient strength (first 10ms vs rest)
        trans_samples = int(0.01 * self.sample_rate)
        if len(audio) > trans_samples * 2:
            trans_energy = np.sqrt(np.mean(audio[:trans_samples] ** 2))
            rest_energy = np.sqrt(np.mean(audio[trans_samples:] ** 2))
            if rest_energy > 0:
                trans_ratio = trans_energy / rest_energy
                result.set('transient_strength', min(100, trans_ratio * 30))
                result.set('punch', result.get('transient_strength') * 0.9)
                result.set('snap', min(100, result.get('transient_strength') * 1.1))
        
        # Decay
        if peak_idx < len(env_smooth) - 100:
            decay_section = env_smooth[peak_idx:peak_idx + len(env_smooth)//4]
            if len(decay_section) > 10:
                decay_rate = 1 - (decay_section[-1] / (decay_section[0] + 0.001))
                result.set('decay', min(100, decay_rate * 100))
        
        # Sustain (middle 50%)
        mid_start = len(env_smooth) // 4
        mid_end = 3 * len(env_smooth) // 4
        if mid_end > mid_start:
            sustain_level = np.mean(env_smooth[mid_start:mid_end])
            peak_level = np.max(env_smooth)
            if peak_level > 0:
                result.set('sustain', (sustain_level / peak_level) * 100)
                result.set('body', result.get('sustain'))
        
        # Release/tail
        tail_start = int(len(env_smooth) * 0.75)
        if tail_start < len(env_smooth):
            tail_energy = np.mean(env_smooth[tail_start:])
            total_energy = np.mean(env_smooth)
            if total_energy > 0:
                result.set('release', min(100, (tail_energy / total_energy) * 200))
                result.set('tail', result.get('release'))
        
        # Envelope smoothness (variance of derivative)
        env_diff = np.abs(np.diff(env_smooth))
        smoothness = 100 - min(100, np.std(env_diff) * 5000)
        result.set('envelope_smoothness', smoothness)
    
    def _analyze_spectral_basic(self, audio: np.ndarray, result: AttributeVector) -> None:
        """Basic spectral analysis using FFT."""
        n_fft = min(4096, len(audio))
        spectrum = np.abs(np.fft.rfft(audio, n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)
        
        # Normalize
        spec_sum = np.sum(spectrum) + 0.001
        spec_norm = spectrum / spec_sum
        
        # Spectral centroid
        centroid = np.sum(freqs * spec_norm)
        result.set('spectral_centroid', centroid)
        
        # Spectral spread
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * spec_norm))
        result.set('spectral_spread', min(100, spread / 50))
        
        # Brightness (energy > 4kHz)
        high_mask = freqs > 4000
        brightness = np.sum(spec_norm[high_mask]) * 300
        result.set('brightness', min(100, brightness))
        
        # Darkness (energy < 300Hz)
        low_mask = freqs < 300
        darkness = np.sum(spec_norm[low_mask]) * 500
        result.set('darkness', min(100, darkness))
        
        # Warmth (200-800Hz)
        warm_mask = (freqs >= 200) & (freqs <= 800)
        warmth = np.sum(spec_norm[warm_mask]) * 400
        result.set('warmth', min(100, warmth))
        
        # Harshness (2-6kHz)
        harsh_mask = (freqs >= 2000) & (freqs <= 6000)
        harshness = np.sum(spec_norm[harsh_mask]) * 400
        result.set('harshness', min(100, harshness))
        
        # Airiness (>10kHz)
        air_mask = freqs > 10000
        airiness = np.sum(spec_norm[air_mask]) * 600
        result.set('airiness', min(100, airiness))
        
        # Muddiness (200-500Hz)
        mud_mask = (freqs >= 200) & (freqs <= 500)
        muddiness = np.sum(spec_norm[mud_mask]) * 500
        result.set('muddiness', min(100, muddiness))
        
        # Presence (1-5kHz)
        pres_mask = (freqs >= 1000) & (freqs <= 5000)
        presence = np.sum(spec_norm[pres_mask]) * 300
        result.set('presence', min(100, presence))
        
        # Spectral flatness (noise-like)
        log_spec = np.log(spectrum + 0.001)
        geo_mean = np.exp(np.mean(log_spec))
        arith_mean = np.mean(spectrum)
        flatness = (geo_mean / (arith_mean + 0.001)) * 100
        result.set('spectral_flatness', min(100, flatness))
        
        # Noisiness / tonality
        result.set('noisiness', result.get('spectral_flatness'))
        result.set('tonality', 100 - result.get('noisiness'))
        
        # Spectral rolloff
        cumsum = np.cumsum(spectrum)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        if rolloff_idx < len(freqs):
            result.set('spectral_rolloff', freqs[rolloff_idx])
    
    def _analyze_spectral_advanced(self, audio: np.ndarray, result: AttributeVector) -> None:
        """Advanced spectral with librosa."""
        try:
            import librosa
            
            # Spectral flux
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            result.set('spectral_flux', min(100, np.mean(onset_env) * 15))
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            result.set('spectral_contrast', min(100, np.mean(contrast) + 50))
            
            # Zero crossing rate (granularity indicator)
            zcr = librosa.feature.zero_crossing_rate(audio)
            result.set('granularity', min(100, np.mean(zcr) * 500))
            result.set('roughness', result.get('granularity'))
            result.set('smoothness', 100 - result.get('granularity'))
            
        except Exception:
            pass
    
    def _analyze_pitch(self, audio: np.ndarray, result: AttributeVector) -> None:
        """Analyze pitch characteristics."""
        try:
            import librosa
            
            # Pitch detection
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            
            # Get strongest pitches
            pitch_values = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                p = pitches[idx, t]
                if p > 20:
                    pitch_values.append(p)
            
            if pitch_values:
                mean_pitch = np.mean(pitch_values)
                result.set('pitch_hz', mean_pitch)
                
                # MIDI note
                midi = 12 * np.log2(mean_pitch / 440) + 69
                result.set('pitch_midi', midi)
                
                # Pitch clarity
                pitch_confidence = len(pitch_values) / pitches.shape[1]
                result.set('pitch_clarity', pitch_confidence * 100)
                
                # Pitch stability
                if len(pitch_values) > 5:
                    stability = 100 - min(100, (np.std(pitch_values) / mean_pitch) * 200)
                    result.set('pitch_stability', stability)
                
                # Vibrato (low-frequency pitch variation)
                if len(pitch_values) > 20:
                    pitch_diff = np.diff(pitch_values)
                    vibrato = np.std(pitch_diff) / (mean_pitch + 0.001) * 500
                    result.set('vibrato_amount', min(100, vibrato))
                    
        except Exception:
            pass
    
    def _analyze_rhythm(self, audio: np.ndarray, result: AttributeVector) -> None:
        """Analyze rhythmic characteristics."""
        try:
            import librosa
            
            # Onset detection
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sample_rate)
            
            # Density
            duration = len(audio) / self.sample_rate
            density = len(onsets) / duration * 5
            result.set('density', min(100, density))
            result.set('sparsity', 100 - result.get('density'))
            
            # Rhythmic vs sustained
            result.set('rhythmic', min(100, len(onsets) * 10))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate)
            result.set('tempo_bpm', float(tempo))
            
            # Periodicity
            if len(onset_env) > 100:
                ac = np.correlate(onset_env, onset_env, mode='full')
                ac = ac[len(ac)//2:]
                ac = ac / (ac[0] + 0.001)
                
                peaks = np.where((ac[1:-1] > ac[:-2]) & (ac[1:-1] > ac[2:]))[0] + 1
                if len(peaks) > 0:
                    periodicity = np.mean(ac[peaks[:5]]) * 100
                    result.set('periodicity', min(100, periodicity))
                    result.set('groove', periodicity * 0.7)
                    
        except Exception:
            pass
    
    def _analyze_harmonic(self, audio: np.ndarray, result: AttributeVector) -> None:
        """Analyze harmonic structure."""
        n_fft = min(4096, len(audio))
        spectrum = np.abs(np.fft.rfft(audio, n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)
        
        # Find peaks (harmonics)
        peaks = []
        for i in range(1, len(spectrum) - 1):
            if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                if spectrum[i] > np.mean(spectrum) * 2:
                    peaks.append((freqs[i], spectrum[i]))
        
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        if peaks:
            # Fundamental strength
            if len(peaks) > 1:
                fund_ratio = peaks[0][1] / sum(p[1] for p in peaks)
                result.set('fundamental_strength', fund_ratio * 100)
            
            # Harmonic richness
            result.set('harmonic_richness', min(100, len(peaks) * 5))
            
            # Check harmonic ratios (inharmonicity)
            if len(peaks) >= 3:
                fund = peaks[0][0]
                if fund > 0:
                    inharmonic_score = 0
                    for i, (f, _) in enumerate(peaks[1:6], 2):
                        expected = fund * i
                        deviation = abs(f - expected) / expected
                        inharmonic_score += deviation
                    result.set('inharmonicity', min(100, inharmonic_score * 100))
            
            # Odd vs even harmonics
            odd_energy = sum(p[1] for i, p in enumerate(peaks) if i % 2 == 0)
            even_energy = sum(p[1] for i, p in enumerate(peaks) if i % 2 == 1)
            total = odd_energy + even_energy + 0.001
            result.set('odd_harmonics', (odd_energy / total) * 100)
            result.set('even_harmonics', (even_energy / total) * 100)
        
        # Distortion (high-frequency harmonics)
        high_harmonic_energy = np.sum(spectrum[len(spectrum)//2:])
        total_energy = np.sum(spectrum) + 0.001
        result.set('distortion', min(100, (high_harmonic_energy / total_energy) * 200))
        result.set('saturation', result.get('distortion') * 0.7)
    
    def _compute_character(self, result: AttributeVector) -> None:
        """Compute derived character attributes."""
        # Aggressive
        aggressive = (
            result.get('harshness') * 0.3 +
            result.get('transient_strength') * 0.3 +
            result.get('brightness') * 0.2 +
            result.get('distortion') * 0.2
        )
        result.set('aggressive', min(100, aggressive))
        result.set('gentle', 100 - aggressive)
        
        # Metallic
        metallic = (
            result.get('inharmonicity', 0) * 0.4 +
            result.get('brightness') * 0.3 +
            result.get('harshness') * 0.3
        )
        result.set('metallic', min(100, metallic))
        result.set('woody', 100 - metallic * 0.8)
        
        # Digital vs Analog
        digital = (
            result.get('harshness') * 0.25 +
            (100 - result.get('warmth')) * 0.25 +
            result.get('noisiness') * 0.25 +
            result.get('granularity', 50) * 0.25
        )
        result.set('digital', min(100, digital))
        result.set('analog', 100 - digital)
        
        # Clean vs Dirty
        dirty = (
            result.get('distortion') * 0.4 +
            result.get('noisiness') * 0.3 +
            result.get('harshness') * 0.3
        )
        result.set('dirty', min(100, dirty))
        result.set('clean', 100 - dirty)
        
        # Organic vs Synthetic
        organic = (
            result.get('warmth') * 0.3 +
            result.get('inharmonicity', 50) * 0.2 +
            result.get('envelope_smoothness', 50) * 0.2 +
            (100 - result.get('noisiness')) * 0.3
        )
        result.set('organic', min(100, organic))
        result.set('synthetic', 100 - organic)
        
        # Complexity
        complexity = (
            result.get('harmonic_richness', 50) * 0.25 +
            result.get('spectral_flux', 50) * 0.25 +
            result.get('density', 50) * 0.25 +
            result.get('inharmonicity', 50) * 0.25
        )
        result.set('complexity', min(100, complexity))


# ============================================================================
# FORMATTING
# ============================================================================

def format_analysis(av: AttributeVector, top_n: int = 15) -> str:
    """Format attribute vector as readable text.
    
    Parameters
    ----------
    av : AttributeVector
        Analysis results
    top_n : int
        Number of top attributes to show
    
    Returns
    -------
    str
        Formatted text output
    """
    lines = [
        "=== DEEP AUDIO ANALYSIS ===",
        f"Source: {av.source or 'buffer'}",
        f"Duration: {av.duration:.3f}s",
        f"Sample Rate: {av.sample_rate} Hz",
        "",
        f"TOP {top_n} ATTRIBUTES:",
    ]
    
    for name, value in av.top(top_n):
        info = ATTRIBUTE_POOL.get(name, {})
        desc = info.get('desc', '')
        cat = info.get('category', 'other')
        lines.append(f"  {name}: {value:.1f}  [{cat}] {desc}")
    
    lines.append("")
    lines.append("CATEGORY SUMMARY:")
    for cat, avg in sorted(av.category_averages().items()):
        lines.append(f"  {cat}: {avg:.1f}")
    
    return '\n'.join(lines)


def format_comparison(av1: AttributeVector, av2: AttributeVector) -> str:
    """Format comparison between two analyses.
    
    Parameters
    ----------
    av1, av2 : AttributeVector
        Analyses to compare
    
    Returns
    -------
    str
        Formatted comparison
    """
    lines = [
        "=== ATTRIBUTE COMPARISON ===",
        f"Sample A: {av1.source or 'buffer 1'}",
        f"Sample B: {av2.source or 'buffer 2'}",
        "",
        f"Similarity: {av1.similarity(av2):.1%}",
        f"Distance: {av1.distance(av2):.2f}",
        "",
        "BIGGEST DIFFERENCES:",
    ]
    
    # Find biggest differences
    diffs = []
    common = set(av1.attributes.keys()) & set(av2.attributes.keys())
    for k in common:
        diff = av1.get(k) - av2.get(k)
        diffs.append((k, diff, av1.get(k), av2.get(k)))
    
    diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for name, diff, v1, v2 in diffs[:10]:
        arrow = ">" if diff > 0 else "<"
        lines.append(f"  {name}: {v1:.1f} {arrow} {v2:.1f} (Δ{abs(diff):.1f})")
    
    return '\n'.join(lines)


# Global analyzer
_analyzer: Optional[DeepAnalyzer] = None

def get_analyzer(sample_rate: int = 48000) -> DeepAnalyzer:
    """Get or create global analyzer."""
    global _analyzer
    if _analyzer is None or _analyzer.sample_rate != sample_rate:
        _analyzer = DeepAnalyzer(sample_rate)
    return _analyzer



# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def analyze_buffer(
    buffer: np.ndarray,
    sample_rate: int = 48000,
    top_n: int = 15
) -> str:
    """Analyze an audio buffer and return formatted results.
    
    Parameters
    ----------
    buffer : np.ndarray
        Audio buffer to analyze
    sample_rate : int
        Sample rate of the audio
    top_n : int
        Number of top attributes to show
    
    Returns
    -------
    str
        Formatted analysis string
    """
    analyzer = get_analyzer(sample_rate)
    av = analyzer.analyze(buffer)
    return format_analysis(av, top_n)


def quick_analyze(buffer: np.ndarray, sr: int = 48000) -> dict:
    """Quick analysis returning key metrics as dict.
    
    Parameters
    ----------
    buffer : np.ndarray
        Audio buffer
    sr : int
        Sample rate
    
    Returns
    -------
    dict
        Key metrics (bpm, key, energy, brightness, etc.)
    """
    analyzer = get_analyzer(sr)
    av = analyzer.analyze(buffer)
    
    return {
        'duration': len(buffer) / sr,
        'peak': float(np.max(np.abs(buffer))),
        'rms': float(np.sqrt(np.mean(buffer**2))),
        'bpm': av.get('tempo', 0),
        'key': av.get('key', 'unknown'),
        'energy': av.get('energy', 0),
        'brightness': av.get('brightness', 0),
        'warmth': av.get('warmth', 0),
        'groove': av.get('groove', 0),
    }

