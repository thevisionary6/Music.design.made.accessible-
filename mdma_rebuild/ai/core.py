"""MDMA AI Systems - Audio Generation, Analysis, and Breeding.

Section N of MDMA Master Feature List.

Features:
- GPU auto-detection with 3060 fallback defaults
- AudioLDM2 text-to-audio generation
- Deep attribute analysis with weighted attribute pool
- Genetic algorithm sample breeding

Requirements:
- torch
- diffusers (for AudioLDM2)
- librosa (for analysis)
- scipy
- numpy
"""

from __future__ import annotations

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
import warnings

# Suppress warnings during import
warnings.filterwarnings('ignore')

# ============================================================================
# GPU DETECTION AND CONFIGURATION
# ============================================================================

@dataclass
class GPUConfig:
    """GPU configuration for AI operations."""
    name: str = "Unknown"
    vram_gb: float = 6.0
    compute_capability: str = "8.6"
    is_available: bool = False
    device: str = "cpu"
    
    # AudioLDM2 defaults based on VRAM
    aldm_steps: int = 150
    aldm_cfg_scale: float = 10.0
    aldm_scheduler: str = "discrete"
    aldm_batch_size: int = 1
    
    # Analysis settings
    analysis_chunk_size: int = 48000
    
    def __post_init__(self):
        """Adjust settings based on VRAM."""
        if self.vram_gb >= 12:
            self.aldm_batch_size = 2
            self.aldm_steps = 200
        elif self.vram_gb >= 8:
            self.aldm_batch_size = 1
            self.aldm_steps = 150
        elif self.vram_gb >= 6:
            self.aldm_batch_size = 1
            self.aldm_steps = 100
        else:
            # Low VRAM - reduce quality
            self.aldm_batch_size = 1
            self.aldm_steps = 50


def detect_gpu() -> GPUConfig:
    """Detect available GPU and return configuration.
    
    Returns
    -------
    GPUConfig
        Configuration with detected GPU or 3060 defaults
    """
    config = GPUConfig()
    
    try:
        import torch
        
        if torch.cuda.is_available():
            config.is_available = True
            config.device = "cuda"
            
            # Get GPU info
            gpu_id = torch.cuda.current_device()
            config.name = torch.cuda.get_device_name(gpu_id)
            
            # Get VRAM in GB
            props = torch.cuda.get_device_properties(gpu_id)
            config.vram_gb = props.total_memory / (1024 ** 3)
            
            # Get compute capability
            config.compute_capability = f"{props.major}.{props.minor}"
            
            # Recalculate settings based on actual VRAM
            config.__post_init__()
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon
            config.is_available = True
            config.device = "mps"
            config.name = "Apple Silicon (MPS)"
            config.vram_gb = 8.0  # Assume unified memory
            config.__post_init__()
            
    except ImportError:
        pass
    
    # Default to 3060 specs if no GPU detected
    if not config.is_available:
        config.name = "Default (RTX 3060 specs)"
        config.vram_gb = 12.0
        config.device = "cpu"
        config.__post_init__()
    
    return config


# Global GPU config (lazy loaded)
_gpu_config: Optional[GPUConfig] = None


def get_gpu_config() -> GPUConfig:
    """Get cached GPU configuration."""
    global _gpu_config
    if _gpu_config is None:
        _gpu_config = detect_gpu()
    return _gpu_config


def gpu_info() -> str:
    """Get formatted GPU information string."""
    cfg = get_gpu_config()
    lines = [
        "=== GPU CONFIGURATION ===",
        f"Device: {cfg.name}",
        f"VRAM: {cfg.vram_gb:.1f} GB",
        f"Compute: {cfg.compute_capability}",
        f"Available: {'Yes' if cfg.is_available else 'No (using CPU)'}",
        f"PyTorch Device: {cfg.device}",
        "",
        "AudioLDM2 Settings:",
        f"  Steps: {cfg.aldm_steps}",
        f"  CFG Scale: {cfg.aldm_cfg_scale}",
        f"  Scheduler: {cfg.aldm_scheduler}",
        f"  Batch Size: {cfg.aldm_batch_size}",
    ]
    return '\n'.join(lines)


# ============================================================================
# AUDIOLDM2 GENERATION
# ============================================================================

class AudioGenerator:
    """AudioLDM2-based text-to-audio generator."""
    
    def __init__(self, model_id: str = "cvssp/audioldm2"):
        """Initialize the generator.
        
        Parameters
        ----------
        model_id : str
            HuggingFace model ID for AudioLDM2
        """
        self.model_id = model_id
        self.pipe = None
        self.config = get_gpu_config()
        self._loaded = False
    
    def load(self) -> str:
        """Load the AudioLDM2 model.
        
        Returns
        -------
        str
            Status message
        """
        if self._loaded:
            return "AudioLDM2 already loaded"
        
        try:
            import torch
            from diffusers import AudioLDM2Pipeline, DPMSolverMultistepScheduler
            
            # Load pipeline
            self.pipe = AudioLDM2Pipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            )
            
            # Set scheduler to discrete (DPM++ 2M)
            if self.config.aldm_scheduler == "discrete":
                self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config,
                    algorithm_type="dpmsolver++",
                    solver_order=2,
                )
            
            # Move to device
            if self.config.device != "cpu":
                self.pipe = self.pipe.to(self.config.device)
            
            # Enable memory optimizations
            if self.config.device == "cuda":
                try:
                    self.pipe.enable_model_cpu_offload()
                except Exception:
                    pass
            
            self._loaded = True
            return f"AudioLDM2 loaded on {self.config.device}"
            
        except ImportError as e:
            return f"ERROR: Missing dependency: {e}. Install with: pip install diffusers transformers accelerate"
        except Exception as e:
            return f"ERROR: Could not load AudioLDM2: {e}"
    
    def generate(
        self,
        prompt: str,
        duration: float = 5.0,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
        negative_prompt: str = "low quality, noise, distortion",
    ) -> Tuple[Optional[np.ndarray], str]:
        """Generate audio from text prompt.
        
        Parameters
        ----------
        prompt : str
            Text description of desired audio
        duration : float
            Audio duration in seconds (default 5.0)
        steps : int, optional
            Inference steps (default from GPU config)
        cfg_scale : float, optional
            Classifier-free guidance scale (default from GPU config)
        seed : int, optional
            Random seed for reproducibility
        negative_prompt : str
            What to avoid in generation
        
        Returns
        -------
        tuple
            (audio_array, status_message)
        """
        if not self._loaded:
            status = self.load()
            if status.startswith("ERROR"):
                return None, status
        
        # Use defaults from config
        steps = steps or self.config.aldm_steps
        cfg_scale = cfg_scale or self.config.aldm_cfg_scale
        
        try:
            import torch
            
            # Set seed
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.config.device).manual_seed(seed)
            
            # Calculate audio length (AudioLDM2 uses 16kHz, we'll resample)
            audio_length = duration
            
            # Generate
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                audio_length_in_s=audio_length,
                generator=generator,
            )
            
            # Extract audio
            audio = result.audios[0]
            
            # AudioLDM2 outputs at 16kHz, resample to 48kHz
            if len(audio.shape) > 1:
                audio = audio.mean(axis=0)  # Mono
            
            # Resample from 16kHz to 48kHz
            audio_48k = self._resample(audio, 16000, 48000)
            
            # Normalize
            peak = np.max(np.abs(audio_48k))
            if peak > 0:
                audio_48k = audio_48k / peak * 0.95
            
            return audio_48k.astype(np.float64), f"Generated {duration:.1f}s audio from: '{prompt[:50]}...'"
            
        except Exception as e:
            return None, f"ERROR: Generation failed: {e}"
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
        
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Simple linear interpolation fallback
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            x_old = np.linspace(0, 1, len(audio))
            x_new = np.linspace(0, 1, new_length)
            return np.interp(x_new, x_old, audio)


# Global generator instance (lazy loaded)
_generator: Optional[AudioGenerator] = None


def get_generator() -> AudioGenerator:
    """Get or create the global audio generator."""
    global _generator
    if _generator is None:
        _generator = AudioGenerator()
    return _generator


# ============================================================================
# ATTRIBUTE ANALYSIS SYSTEM
# ============================================================================

# Master attribute pool with descriptions and ranges
ATTRIBUTE_POOL = {
    # Spectral attributes (0-100)
    'brightness': {'desc': 'High frequency content', 'range': (0, 100), 'category': 'spectral'},
    'darkness': {'desc': 'Low frequency emphasis', 'range': (0, 100), 'category': 'spectral'},
    'warmth': {'desc': 'Mid-low frequency warmth', 'range': (0, 100), 'category': 'spectral'},
    'harshness': {'desc': 'Harsh/brittle high frequencies', 'range': (0, 100), 'category': 'spectral'},
    'fullness': {'desc': 'Broadband spectral coverage', 'range': (0, 100), 'category': 'spectral'},
    'thinness': {'desc': 'Narrow spectral content', 'range': (0, 100), 'category': 'spectral'},
    'spectral_centroid': {'desc': 'Center of spectral mass (Hz)', 'range': (20, 10000), 'category': 'spectral'},
    'spectral_spread': {'desc': 'Spectral bandwidth', 'range': (0, 100), 'category': 'spectral'},
    'spectral_rolloff': {'desc': 'Frequency below which 85% energy', 'range': (20, 20000), 'category': 'spectral'},
    'spectral_flux': {'desc': 'Spectral change rate', 'range': (0, 100), 'category': 'spectral'},
    
    # Temporal/Envelope attributes
    'attack': {'desc': 'Attack sharpness', 'range': (0, 100), 'category': 'envelope'},
    'decay': {'desc': 'Decay rate', 'range': (0, 100), 'category': 'envelope'},
    'sustain': {'desc': 'Sustain level', 'range': (0, 100), 'category': 'envelope'},
    'release': {'desc': 'Release length', 'range': (0, 100), 'category': 'envelope'},
    'transient': {'desc': 'Transient strength', 'range': (0, 100), 'category': 'envelope'},
    'punch': {'desc': 'Percussive punch', 'range': (0, 100), 'category': 'envelope'},
    'snap': {'desc': 'Initial snap/click', 'range': (0, 100), 'category': 'envelope'},
    'body': {'desc': 'Sustained body', 'range': (0, 100), 'category': 'envelope'},
    'tail': {'desc': 'Tail/reverb length', 'range': (0, 100), 'category': 'envelope'},
    
    # Dynamics attributes
    'dynamics': {'desc': 'Dynamic range', 'range': (0, 100), 'category': 'dynamics'},
    'compression': {'desc': 'Perceived compression', 'range': (0, 100), 'category': 'dynamics'},
    'loudness': {'desc': 'Perceived loudness (LUFS-based)', 'range': (0, 100), 'category': 'dynamics'},
    'peak_level': {'desc': 'Peak amplitude', 'range': (0, 100), 'category': 'dynamics'},
    'rms_level': {'desc': 'RMS level', 'range': (0, 100), 'category': 'dynamics'},
    'crest_factor': {'desc': 'Peak to RMS ratio', 'range': (0, 100), 'category': 'dynamics'},
    
    # Texture/Density attributes
    'density': {'desc': 'Event density', 'range': (0, 100), 'category': 'texture'},
    'sparsity': {'desc': 'Sparse/minimal content', 'range': (0, 100), 'category': 'texture'},
    'complexity': {'desc': 'Textural complexity', 'range': (0, 100), 'category': 'texture'},
    'smoothness': {'desc': 'Smooth vs grainy', 'range': (0, 100), 'category': 'texture'},
    'granularity': {'desc': 'Granular quality', 'range': (0, 100), 'category': 'texture'},
    'noisiness': {'desc': 'Noise content', 'range': (0, 100), 'category': 'texture'},
    'tonality': {'desc': 'Tonal vs noise', 'range': (0, 100), 'category': 'texture'},
    
    # Harmonic attributes
    'harmonic_content': {'desc': 'Harmonic richness', 'range': (0, 100), 'category': 'harmonic'},
    'inharmonicity': {'desc': 'Inharmonic partials', 'range': (0, 100), 'category': 'harmonic'},
    'fundamental_strength': {'desc': 'Fundamental presence', 'range': (0, 100), 'category': 'harmonic'},
    'odd_harmonics': {'desc': 'Odd harmonic emphasis', 'range': (0, 100), 'category': 'harmonic'},
    'even_harmonics': {'desc': 'Even harmonic emphasis', 'range': (0, 100), 'category': 'harmonic'},
    'distortion': {'desc': 'Distortion/saturation', 'range': (0, 100), 'category': 'harmonic'},
    
    # Rhythm/Temporal pattern
    'rhythmic': {'desc': 'Rhythmic content', 'range': (0, 100), 'category': 'rhythm'},
    'periodic': {'desc': 'Periodic/repetitive', 'range': (0, 100), 'category': 'rhythm'},
    'tempo_feel': {'desc': 'Implied tempo feel', 'range': (40, 200), 'category': 'rhythm'},
    'groove': {'desc': 'Groove/swing feel', 'range': (0, 100), 'category': 'rhythm'},
    
    # Pitch attributes
    'pitch_clarity': {'desc': 'Clear pitch vs noise', 'range': (0, 100), 'category': 'pitch'},
    'pitch_height': {'desc': 'High vs low pitch', 'range': (0, 100), 'category': 'pitch'},
    'pitch_stability': {'desc': 'Pitch stability', 'range': (0, 100), 'category': 'pitch'},
    'vibrato': {'desc': 'Vibrato amount', 'range': (0, 100), 'category': 'pitch'},
    'pitch_bend': {'desc': 'Pitch bend/glide', 'range': (0, 100), 'category': 'pitch'},
    'estimated_pitch': {'desc': 'Estimated fundamental (Hz)', 'range': (20, 5000), 'category': 'pitch'},
    
    # Character/Mood attributes
    'aggressive': {'desc': 'Aggressive character', 'range': (0, 100), 'category': 'character'},
    'gentle': {'desc': 'Gentle/soft character', 'range': (0, 100), 'category': 'character'},
    'metallic': {'desc': 'Metallic quality', 'range': (0, 100), 'category': 'character'},
    'woody': {'desc': 'Woody/organic quality', 'range': (0, 100), 'category': 'character'},
    'digital': {'desc': 'Digital/synthetic', 'range': (0, 100), 'category': 'character'},
    'analog': {'desc': 'Analog/warm character', 'range': (0, 100), 'category': 'character'},
    'clean': {'desc': 'Clean/pure', 'range': (0, 100), 'category': 'character'},
    'dirty': {'desc': 'Dirty/gritty', 'range': (0, 100), 'category': 'character'},
    
    # Spatial attributes
    'stereo_width': {'desc': 'Stereo width', 'range': (0, 100), 'category': 'spatial'},
    'depth': {'desc': 'Front-to-back depth', 'range': (0, 100), 'category': 'spatial'},
    'reverberance': {'desc': 'Reverb amount', 'range': (0, 100), 'category': 'spatial'},
}


@dataclass
class AttributeVector:
    """Vector of analyzed attributes for a sample."""
    attributes: Dict[str, float] = field(default_factory=dict)
    sample_rate: int = 48000
    duration: float = 0.0
    analysis_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttributeVector':
        """Create from dictionary."""
        return cls(**data)
    
    def get(self, attr: str, default: float = 0.0) -> float:
        """Get attribute value."""
        return self.attributes.get(attr, default)
    
    def set(self, attr: str, value: float) -> None:
        """Set attribute value."""
        self.attributes[attr] = value
    
    def top_attributes(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N attributes by value."""
        sorted_attrs = sorted(self.attributes.items(), key=lambda x: x[1], reverse=True)
        return sorted_attrs[:n]
    
    def category_summary(self) -> Dict[str, float]:
        """Get average value per category."""
        categories = {}
        counts = {}
        for attr, value in self.attributes.items():
            if attr in ATTRIBUTE_POOL:
                cat = ATTRIBUTE_POOL[attr]['category']
                categories[cat] = categories.get(cat, 0) + value
                counts[cat] = counts.get(cat, 0) + 1
        
        return {cat: categories[cat] / counts[cat] for cat in categories}
    
    def distance(self, other: 'AttributeVector') -> float:
        """Calculate Euclidean distance to another vector."""
        common_attrs = set(self.attributes.keys()) & set(other.attributes.keys())
        if not common_attrs:
            return float('inf')
        
        sq_diff = sum(
            (self.attributes[a] - other.attributes[a]) ** 2
            for a in common_attrs
        )
        return np.sqrt(sq_diff / len(common_attrs))
    
    def similarity(self, other: 'AttributeVector') -> float:
        """Calculate cosine similarity to another vector (0-1)."""
        common_attrs = set(self.attributes.keys()) & set(other.attributes.keys())
        if not common_attrs:
            return 0.0
        
        vec1 = np.array([self.attributes[a] for a in common_attrs])
        vec2 = np.array([other.attributes[a] for a in common_attrs])
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))


class AudioAnalyzer:
    """Deep audio analyzer with attribute detection."""
    
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
    
    def analyze(self, audio: np.ndarray, detailed: bool = True) -> AttributeVector:
        """Analyze audio and return attribute vector.
        
        Parameters
        ----------
        audio : np.ndarray
            Audio samples (mono, float64)
        detailed : bool
            If True, compute all attributes. If False, fast subset only.
        
        Returns
        -------
        AttributeVector
            Analyzed attributes
        """
        result = AttributeVector(
            sample_rate=self.sample_rate,
            duration=len(audio) / self.sample_rate
        )
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Normalize for analysis
        audio = audio.astype(np.float64)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio_norm = audio / peak
        else:
            audio_norm = audio
        
        # Basic time-domain analysis (always available)
        self._analyze_dynamics(audio_norm, result)
        self._analyze_envelope(audio_norm, result)
        
        # Spectral analysis (requires FFT)
        self._analyze_spectral_basic(audio_norm, result)
        
        # Advanced analysis with librosa if available
        if detailed and self._check_librosa():
            self._analyze_spectral_advanced(audio_norm, result)
            self._analyze_rhythm(audio_norm, result)
            self._analyze_pitch(audio_norm, result)
        
        # Derived/character attributes
        self._analyze_character(result)
        
        return result
    
    def _analyze_dynamics(self, audio: np.ndarray, result: AttributeVector) -> None:
        """Analyze dynamic properties."""
        # RMS level
        rms = np.sqrt(np.mean(audio ** 2))
        result.set('rms_level', min(100, rms * 100))
        
        # Peak level
        peak = np.max(np.abs(audio))
        result.set('peak_level', min(100, peak * 100))
        
        # Crest factor (peak to RMS ratio)
        if rms > 0:
            crest = peak / rms
            result.set('crest_factor', min(100, (crest - 1) * 20))
        
        # Dynamic range (simplified)
        if len(audio) > 1024:
            frame_size = 1024
            frames = len(audio) // frame_size
            rms_frames = [
                np.sqrt(np.mean(audio[i*frame_size:(i+1)*frame_size] ** 2))
                for i in range(frames)
            ]
            rms_frames = [r for r in rms_frames if r > 0.001]
            if rms_frames:
                dynamic_range = max(rms_frames) / (min(rms_frames) + 0.001)
                result.set('dynamics', min(100, np.log10(dynamic_range + 1) * 30))
                result.set('compression', 100 - result.get('dynamics'))
        
        # Perceived loudness (simplified LUFS-like)
        result.set('loudness', min(100, rms * 120))
    
    def _analyze_envelope(self, audio: np.ndarray, result: AttributeVector) -> None:
        """Analyze envelope characteristics."""
        # Get amplitude envelope
        env = np.abs(audio)
        
        # Smooth envelope
        window_size = min(512, len(env) // 10)
        if window_size > 1:
            env_smooth = np.convolve(env, np.ones(window_size)/window_size, mode='same')
        else:
            env_smooth = env
        
        # Find peak position
        peak_idx = np.argmax(env_smooth)
        peak_pos = peak_idx / len(env_smooth)
        
        # Attack (how quickly it reaches peak)
        attack_sharpness = 1.0 - peak_pos
        result.set('attack', min(100, attack_sharpness * 100))
        
        # Transient (first 10ms energy)
        transient_samples = int(0.01 * self.sample_rate)
        if len(audio) > transient_samples:
            transient_energy = np.sqrt(np.mean(audio[:transient_samples] ** 2))
            total_energy = np.sqrt(np.mean(audio ** 2))
            if total_energy > 0:
                result.set('transient', min(100, (transient_energy / total_energy) * 200))
                result.set('punch', result.get('transient') * 0.8)
                result.set('snap', min(100, result.get('transient') * 1.2))
        
        # Decay (energy after peak)
        if peak_idx < len(env_smooth) - 100:
            decay_portion = env_smooth[peak_idx:]
            decay_rate = 1.0 - (np.mean(decay_portion) / (env_smooth[peak_idx] + 0.001))
            result.set('decay', min(100, decay_rate * 100))
        
        # Sustain (middle portion level)
        if len(env_smooth) > 100:
            mid_start = len(env_smooth) // 4
            mid_end = 3 * len(env_smooth) // 4
            mid_level = np.mean(env_smooth[mid_start:mid_end])
            peak_level = np.max(env_smooth)
            if peak_level > 0:
                result.set('sustain', min(100, (mid_level / peak_level) * 100))
                result.set('body', result.get('sustain'))
        
        # Release/tail (last 20%)
        tail_start = int(len(env_smooth) * 0.8)
        if tail_start < len(env_smooth):
            tail_energy = np.mean(env_smooth[tail_start:])
            total_energy = np.mean(env_smooth)
            if total_energy > 0:
                result.set('release', min(100, (tail_energy / total_energy) * 150))
                result.set('tail', result.get('release'))
    
    def _analyze_spectral_basic(self, audio: np.ndarray, result: AttributeVector) -> None:
        """Basic spectral analysis using FFT."""
        # Compute FFT
        n_fft = min(4096, len(audio))
        spectrum = np.abs(np.fft.rfft(audio, n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)
        
        # Normalize spectrum
        spectrum_norm = spectrum / (np.sum(spectrum) + 0.001)
        
        # Spectral centroid
        centroid = np.sum(freqs * spectrum_norm) / (np.sum(spectrum_norm) + 0.001)
        result.set('spectral_centroid', centroid)
        
        # Brightness (energy above 4kHz)
        high_mask = freqs > 4000
        brightness = np.sum(spectrum_norm[high_mask]) * 100
        result.set('brightness', min(100, brightness * 3))
        
        # Darkness (energy below 200Hz)
        low_mask = freqs < 200
        darkness = np.sum(spectrum_norm[low_mask]) * 100
        result.set('darkness', min(100, darkness * 5))
        
        # Warmth (200Hz - 800Hz)
        warm_mask = (freqs >= 200) & (freqs <= 800)
        warmth = np.sum(spectrum_norm[warm_mask]) * 100
        result.set('warmth', min(100, warmth * 4))
        
        # Harshness (2kHz - 6kHz presence)
        harsh_mask = (freqs >= 2000) & (freqs <= 6000)
        harshness = np.sum(spectrum_norm[harsh_mask]) * 100
        result.set('harshness', min(100, harshness * 4))
        
        # Fullness (spectral flatness)
        log_spectrum = np.log(spectrum + 0.001)
        geo_mean = np.exp(np.mean(log_spectrum))
        arith_mean = np.mean(spectrum)
        if arith_mean > 0:
            flatness = geo_mean / arith_mean
            result.set('fullness', min(100, flatness * 100))
            result.set('thinness', 100 - result.get('fullness'))
        
        # Spectral rolloff (85% energy)
        cumsum = np.cumsum(spectrum)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        if rolloff_idx < len(freqs):
            result.set('spectral_rolloff', freqs[rolloff_idx])
        
        # Noisiness vs tonality (simplified)
        # High flatness = noisy, low flatness = tonal
        result.set('noisiness', result.get('fullness') * 0.8)
        result.set('tonality', 100 - result.get('noisiness'))
    
    def _analyze_spectral_advanced(self, audio: np.ndarray, result: AttributeVector) -> None:
        """Advanced spectral analysis with librosa."""
        try:
            import librosa
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            result.set('spectral_centroid', float(np.mean(spectral_centroid)))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
            result.set('spectral_spread', min(100, float(np.mean(spectral_bandwidth)) / 100))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
            result.set('spectral_rolloff', float(np.mean(spectral_rolloff)))
            
            # Spectral flux
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            result.set('spectral_flux', min(100, float(np.mean(onset_env)) * 10))
            
            # Zero crossing rate (texture indicator)
            zcr = librosa.feature.zero_crossing_rate(audio)
            result.set('granularity', min(100, float(np.mean(zcr)) * 500))
            
            # Harmonic/percussive separation
            harmonic, percussive = librosa.effects.hpss(audio)
            h_energy = np.sum(harmonic ** 2)
            p_energy = np.sum(percussive ** 2)
            total = h_energy + p_energy + 0.001
            result.set('harmonic_content', min(100, (h_energy / total) * 100))
            result.set('rhythmic', min(100, (p_energy / total) * 100))
            
        except Exception:
            pass
    
    def _analyze_rhythm(self, audio: np.ndarray, result: AttributeVector) -> None:
        """Analyze rhythmic properties."""
        try:
            import librosa
            
            # Onset detection
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            
            # Tempo estimation
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate)
            result.set('tempo_feel', float(tempo))
            
            # Event density
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sample_rate)
            density = len(onsets) / (len(audio) / self.sample_rate)
            result.set('density', min(100, density * 5))
            result.set('sparsity', 100 - result.get('density'))
            
            # Periodicity (autocorrelation)
            ac = np.correlate(onset_env, onset_env, mode='full')
            ac = ac[len(ac)//2:]
            ac = ac / (ac[0] + 0.001)
            
            # Find peaks in autocorrelation (periodic patterns)
            if len(ac) > 100:
                peaks = np.where((ac[1:-1] > ac[:-2]) & (ac[1:-1] > ac[2:]))[0]
                if len(peaks) > 0:
                    periodicity = np.mean(ac[peaks[:5]]) * 100
                    result.set('periodic', min(100, periodicity))
                    result.set('groove', periodicity * 0.8)
                    
        except Exception:
            pass
    
    def _analyze_pitch(self, audio: np.ndarray, result: AttributeVector) -> None:
        """Analyze pitch properties."""
        try:
            import librosa
            
            # Pitch detection
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            
            # Get strongest pitch per frame
            pitch_per_frame = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                pitch = pitches[idx, t]
                if pitch > 0:
                    pitch_per_frame.append(pitch)
            
            if pitch_per_frame:
                mean_pitch = np.mean(pitch_per_frame)
                pitch_std = np.std(pitch_per_frame)
                
                result.set('estimated_pitch', mean_pitch)
                result.set('pitch_height', min(100, (mean_pitch - 50) / 30))
                
                # Pitch stability
                if mean_pitch > 0:
                    stability = 100 - min(100, (pitch_std / mean_pitch) * 200)
                    result.set('pitch_stability', stability)
                    result.set('pitch_clarity', stability * 0.9)
                
                # Vibrato detection
                if len(pitch_per_frame) > 10:
                    pitch_diff = np.diff(pitch_per_frame)
                    vibrato = np.std(pitch_diff) / (np.mean(pitch_per_frame) + 0.001)
                    result.set('vibrato', min(100, vibrato * 500))
                    result.set('pitch_bend', result.get('vibrato') * 0.5)
                    
        except Exception:
            pass
    
    def _analyze_character(self, result: AttributeVector) -> None:
        """Derive character attributes from other attributes."""
        # Aggressive vs gentle
        aggressive = (
            result.get('transient') * 0.3 +
            result.get('harshness') * 0.3 +
            result.get('brightness') * 0.2 +
            result.get('loudness') * 0.2
        )
        result.set('aggressive', min(100, aggressive))
        result.set('gentle', 100 - aggressive)
        
        # Metallic vs woody
        metallic = (
            result.get('inharmonicity', 0) * 0.4 +
            result.get('brightness') * 0.3 +
            result.get('harshness') * 0.3
        )
        result.set('metallic', min(100, metallic))
        result.set('woody', 100 - metallic * 0.7)
        
        # Digital vs analog
        digital = (
            result.get('harshness') * 0.3 +
            result.get('noisiness') * 0.2 +
            100 - result.get('warmth') * 0.3 +
            result.get('granularity', 50) * 0.2
        )
        result.set('digital', min(100, digital / 2))
        result.set('analog', 100 - result.get('digital'))
        
        # Clean vs dirty
        dirty = (
            result.get('distortion', 0) * 0.4 +
            result.get('noisiness') * 0.3 +
            result.get('harshness') * 0.3
        )
        result.set('dirty', min(100, dirty))
        result.set('clean', 100 - dirty)
        
        # Complexity
        complexity = (
            result.get('spectral_flux', 50) * 0.25 +
            result.get('density', 50) * 0.25 +
            result.get('fullness', 50) * 0.25 +
            result.get('harmonic_content', 50) * 0.25
        )
        result.set('complexity', min(100, complexity))
        result.set('smoothness', 100 - complexity * 0.5)


# Global analyzer instance
_analyzer: Optional[AudioAnalyzer] = None


def get_analyzer(sample_rate: int = 48000) -> AudioAnalyzer:
    """Get or create the global audio analyzer."""
    global _analyzer
    if _analyzer is None or _analyzer.sample_rate != sample_rate:
        _analyzer = AudioAnalyzer(sample_rate)
    return _analyzer


# ============================================================================
# GENETIC SAMPLE BREEDING
# ============================================================================

@dataclass
class BreedingConfig:
    """Configuration for genetic breeding."""
    crossover_rate: float = 0.7
    mutation_rate: float = 0.1
    mutation_strength: float = 0.2
    elite_count: int = 2
    population_size: int = 8
    spectral_crossover: bool = True
    temporal_crossover: bool = True
    amplitude_crossover: bool = True


class SampleBreeder:
    """Genetic algorithm-based sample breeding system."""
    
    def __init__(self, sample_rate: int = 48000, config: Optional[BreedingConfig] = None):
        self.sample_rate = sample_rate
        self.config = config or BreedingConfig()
        self.analyzer = get_analyzer(sample_rate)
    
    def breed(
        self,
        parent_a: np.ndarray,
        parent_b: np.ndarray,
        num_children: int = 4,
        crossover_point: Optional[float] = None,
    ) -> List[np.ndarray]:
        """Breed two parent samples to create children.
        
        Parameters
        ----------
        parent_a : np.ndarray
            First parent audio sample
        parent_b : np.ndarray
            Second parent audio sample
        num_children : int
            Number of children to generate
        crossover_point : float, optional
            Where to crossover (0-1), random if None
        
        Returns
        -------
        list
            List of child audio samples
        """
        # Ensure same length
        max_len = max(len(parent_a), len(parent_b))
        parent_a = self._pad_or_trim(parent_a, max_len)
        parent_b = self._pad_or_trim(parent_b, max_len)
        
        children = []
        
        for i in range(num_children):
            # Determine crossover point
            if crossover_point is None:
                cp = np.random.random()
            else:
                cp = crossover_point + np.random.random() * 0.2 - 0.1
            cp = max(0.1, min(0.9, cp))
            
            # Create child using various breeding methods
            method = i % 4
            
            if method == 0:
                child = self._temporal_crossover(parent_a, parent_b, cp)
            elif method == 1:
                child = self._spectral_crossover(parent_a, parent_b, cp)
            elif method == 2:
                child = self._blend_crossover(parent_a, parent_b, cp)
            else:
                child = self._morphological_crossover(parent_a, parent_b, cp)
            
            # Apply mutation
            if np.random.random() < self.config.mutation_rate:
                child = self._mutate(child)
            
            # Normalize
            peak = np.max(np.abs(child))
            if peak > 0:
                child = child / peak * 0.95
            
            children.append(child.astype(np.float64))
        
        return children
    
    def evolve(
        self,
        population: List[np.ndarray],
        fitness_fn: callable,
        generations: int = 10,
    ) -> List[np.ndarray]:
        """Evolve a population over multiple generations.
        
        Parameters
        ----------
        population : list
            Initial population of samples
        fitness_fn : callable
            Function that takes a sample and returns fitness score (higher = better)
        generations : int
            Number of generations to evolve
        
        Returns
        -------
        list
            Final evolved population
        """
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = [(sample, fitness_fn(sample)) for sample in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Keep elite
            new_population = [s for s, _ in fitness_scores[:self.config.elite_count]]
            
            # Breed to fill population
            while len(new_population) < self.config.population_size:
                # Tournament selection
                parent_a = self._tournament_select(fitness_scores)
                parent_b = self._tournament_select(fitness_scores)
                
                # Breed
                children = self.breed(parent_a, parent_b, num_children=2)
                new_population.extend(children)
            
            population = new_population[:self.config.population_size]
        
        return population
    
    def _pad_or_trim(self, audio: np.ndarray, target_len: int) -> np.ndarray:
        """Pad or trim audio to target length."""
        if len(audio) == target_len:
            return audio
        elif len(audio) > target_len:
            return audio[:target_len]
        else:
            return np.pad(audio, (0, target_len - len(audio)))
    
    def _temporal_crossover(self, a: np.ndarray, b: np.ndarray, cp: float) -> np.ndarray:
        """Time-domain crossover at crossover point."""
        split = int(len(a) * cp)
        
        # Crossfade region
        fade_len = min(1024, split // 4, (len(a) - split) // 4)
        
        child = np.zeros_like(a)
        child[:split-fade_len] = a[:split-fade_len]
        child[split+fade_len:] = b[split+fade_len:]
        
        # Crossfade
        if fade_len > 0:
            fade_in = np.linspace(0, 1, fade_len * 2)
            fade_out = 1 - fade_in
            mid_start = split - fade_len
            mid_end = split + fade_len
            child[mid_start:mid_end] = (
                a[mid_start:mid_end] * fade_out +
                b[mid_start:mid_end] * fade_in
            )
        
        return child
    
    def _spectral_crossover(self, a: np.ndarray, b: np.ndarray, cp: float) -> np.ndarray:
        """Frequency-domain crossover."""
        # FFT both parents
        n_fft = len(a)
        spec_a = np.fft.rfft(a)
        spec_b = np.fft.rfft(b)
        
        # Crossover frequency
        crossover_bin = int(len(spec_a) * cp)
        
        # Create child spectrum
        child_spec = np.zeros_like(spec_a)
        child_spec[:crossover_bin] = spec_a[:crossover_bin]
        child_spec[crossover_bin:] = spec_b[crossover_bin:]
        
        # Smooth transition
        fade_bins = min(100, crossover_bin // 4)
        if fade_bins > 0:
            fade = np.linspace(1, 0, fade_bins)
            child_spec[crossover_bin-fade_bins:crossover_bin] = (
                spec_a[crossover_bin-fade_bins:crossover_bin] * fade +
                spec_b[crossover_bin-fade_bins:crossover_bin] * (1 - fade)
            )
        
        # Inverse FFT
        return np.fft.irfft(child_spec, n_fft)
    
    def _blend_crossover(self, a: np.ndarray, b: np.ndarray, cp: float) -> np.ndarray:
        """Linear blend crossover."""
        return a * cp + b * (1 - cp)
    
    def _morphological_crossover(self, a: np.ndarray, b: np.ndarray, cp: float) -> np.ndarray:
        """Envelope + spectral morphing crossover."""
        # Extract envelopes
        env_a = np.abs(a)
        env_b = np.abs(b)
        
        # Smooth envelopes
        window = 256
        env_a_smooth = np.convolve(env_a, np.ones(window)/window, mode='same')
        env_b_smooth = np.convolve(env_b, np.ones(window)/window, mode='same')
        
        # Blend envelope
        child_env = env_a_smooth * cp + env_b_smooth * (1 - cp)
        
        # Get spectral content via FFT
        spec_a = np.fft.rfft(a)
        spec_b = np.fft.rfft(b)
        
        # Blend magnitudes, mix phases
        mag_a = np.abs(spec_a)
        mag_b = np.abs(spec_b)
        phase_a = np.angle(spec_a)
        phase_b = np.angle(spec_b)
        
        child_mag = mag_a * cp + mag_b * (1 - cp)
        
        # Phase blending (complex)
        if np.random.random() < 0.5:
            child_phase = phase_a
        else:
            child_phase = phase_b
        
        child_spec = child_mag * np.exp(1j * child_phase)
        child = np.fft.irfft(child_spec, len(a))
        
        # Apply blended envelope
        child_current_env = np.abs(child) + 0.001
        child = child * (child_env / child_current_env)
        
        return child
    
    def _mutate(self, audio: np.ndarray) -> np.ndarray:
        """Apply random mutations to audio."""
        mutation_type = np.random.randint(0, 5)
        strength = self.config.mutation_strength
        
        if mutation_type == 0:
            # Pitch shift (resample)
            ratio = 1.0 + (np.random.random() - 0.5) * strength
            new_len = int(len(audio) / ratio)
            x_old = np.linspace(0, 1, len(audio))
            x_new = np.linspace(0, 1, new_len)
            audio = np.interp(x_new, x_old, audio)
            audio = self._pad_or_trim(audio, len(x_old))
            
        elif mutation_type == 1:
            # Time stretch region
            start = int(np.random.random() * len(audio) * 0.5)
            end = start + int(len(audio) * 0.3)
            region = audio[start:end]
            stretch_factor = 1.0 + (np.random.random() - 0.5) * strength
            new_len = int(len(region) * stretch_factor)
            x_old = np.linspace(0, 1, len(region))
            x_new = np.linspace(0, 1, new_len)
            stretched = np.interp(x_new, x_old, region)
            # Reinsert
            audio = np.concatenate([audio[:start], stretched, audio[end:]])
            audio = self._pad_or_trim(audio, len(x_old) + end - start)
            
        elif mutation_type == 2:
            # Add noise
            noise = np.random.randn(len(audio)) * strength * 0.1
            audio = audio + noise
            
        elif mutation_type == 3:
            # Frequency shift
            spec = np.fft.rfft(audio)
            shift = int(len(spec) * strength * 0.1)
            if shift > 0:
                spec = np.roll(spec, shift)
                spec[:shift] = 0
            audio = np.fft.irfft(spec, len(audio))
            
        elif mutation_type == 4:
            # Amplitude envelope mutation
            env_mod = 1.0 + (np.random.rand(len(audio)) - 0.5) * strength
            # Smooth the modulation
            window = 512
            env_mod = np.convolve(env_mod, np.ones(window)/window, mode='same')
            audio = audio * env_mod
        
        return audio
    
    def _tournament_select(self, fitness_scores: List[Tuple[np.ndarray, float]], k: int = 3) -> np.ndarray:
        """Tournament selection."""
        tournament = np.random.choice(len(fitness_scores), min(k, len(fitness_scores)), replace=False)
        best = max(tournament, key=lambda i: fitness_scores[i][1])
        return fitness_scores[best][0]


# Global breeder instance
_breeder: Optional[SampleBreeder] = None


def get_breeder(sample_rate: int = 48000) -> SampleBreeder:
    """Get or create the global sample breeder."""
    global _breeder
    if _breeder is None or _breeder.sample_rate != sample_rate:
        _breeder = SampleBreeder(sample_rate)
    return _breeder


# ============================================================================
# HIGH-LEVEL API FUNCTIONS
# ============================================================================

def generate_audio(
    prompt: str,
    duration: float = 5.0,
    steps: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """Generate audio from text prompt using AudioLDM2.
    
    Parameters
    ----------
    prompt : str
        Text description of desired audio
    duration : float
        Duration in seconds
    steps : int, optional
        Inference steps (default: GPU-appropriate)
    cfg_scale : float, optional
        CFG scale (default: 10.0)
    seed : int, optional
        Random seed
    
    Returns
    -------
    tuple
        (audio_array, status_message)
    """
    gen = get_generator()
    return gen.generate(prompt, duration, steps, cfg_scale, seed)


def analyze_audio(audio: np.ndarray, sample_rate: int = 48000, detailed: bool = True) -> AttributeVector:
    """Analyze audio and return attribute vector.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio samples
    sample_rate : int
        Sample rate
    detailed : bool
        Full analysis if True
    
    Returns
    -------
    AttributeVector
        Analysis results
    """
    analyzer = get_analyzer(sample_rate)
    return analyzer.analyze(audio, detailed)


def breed_samples(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    num_children: int = 4,
    sample_rate: int = 48000,
) -> List[np.ndarray]:
    """Breed two samples to create children.
    
    Parameters
    ----------
    parent_a : np.ndarray
        First parent
    parent_b : np.ndarray
        Second parent
    num_children : int
        Number of children
    sample_rate : int
        Sample rate
    
    Returns
    -------
    list
        Child audio samples
    """
    breeder = get_breeder(sample_rate)
    return breeder.breed(parent_a, parent_b, num_children)


def format_analysis(av: AttributeVector, top_n: int = 15) -> str:
    """Format attribute vector as readable string.
    
    Parameters
    ----------
    av : AttributeVector
        Analysis results
    top_n : int
        Number of top attributes to show
    
    Returns
    -------
    str
        Formatted analysis
    """
    lines = [
        "=== AUDIO ANALYSIS ===",
        f"Duration: {av.duration:.3f}s",
        "",
        "Top Attributes:",
    ]
    
    for attr, value in av.top_attributes(top_n):
        info = ATTRIBUTE_POOL.get(attr, {})
        desc = info.get('desc', '')
        lines.append(f"  {attr}: {value:.1f}  ({desc})")
    
    lines.append("")
    lines.append("Category Summary:")
    for cat, avg in sorted(av.category_summary().items()):
        lines.append(f"  {cat}: {avg:.1f}")
    
    return '\n'.join(lines)
