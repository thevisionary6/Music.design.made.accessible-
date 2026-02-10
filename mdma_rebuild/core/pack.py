"""MDMA Pack System - Sample pack management and generation.

Handles:
- Pack creation and management
- Algorithmic sample generation (drums, bass, leads, pads, fx)
- AI-powered pack generation
- Sample dictionary for quick access
"""

from __future__ import annotations

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass, field, asdict

if TYPE_CHECKING:
    from ..core.session import Session


# ============================================================================
# PACK DIRECTORY MANAGEMENT
# ============================================================================

def get_packs_dir() -> Path:
    """Get the packs directory, creating if needed."""
    packs_dir = Path.home() / 'Documents' / 'MDMA' / 'packs'
    packs_dir.mkdir(parents=True, exist_ok=True)
    return packs_dir


def list_packs() -> List[str]:
    """List all available packs."""
    packs_dir = get_packs_dir()
    return [d.name for d in packs_dir.iterdir() if d.is_dir()]


def get_pack_info(pack_name: str) -> Optional[Dict]:
    """Get pack manifest info."""
    manifest_path = get_packs_dir() / pack_name / 'pack.json'
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def get_pack_samples(pack_name: str) -> List[Path]:
    """Get list of samples in a pack."""
    pack_dir = get_packs_dir() / pack_name
    if not pack_dir.exists():
        return []
    
    extensions = {'.wav', '.mp3', '.flac', '.aiff', '.ogg'}
    samples = []
    for f in pack_dir.iterdir():
        if f.suffix.lower() in extensions:
            samples.append(f)
    
    return sorted(samples, key=lambda p: p.name.lower())


def create_pack_manifest(pack_name: str, author: str = "MDMA User", 
                         description: str = "", tags: List[str] = None) -> Path:
    """Create or update pack manifest."""
    pack_dir = get_packs_dir() / pack_name
    pack_dir.mkdir(parents=True, exist_ok=True)
    
    samples = get_pack_samples(pack_name)
    
    manifest = {
        'name': pack_name,
        'author': author,
        'version': '1.0',
        'description': description,
        'tags': tags or [],
        'sample_count': len(samples),
        'created': __import__('datetime').datetime.now().isoformat(),
    }
    
    manifest_path = pack_dir / 'pack.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest_path


# ============================================================================
# SAMPLE DICTIONARY
# ============================================================================

@dataclass
class SampleEntry:
    """Entry in the sample dictionary."""
    name: str
    path: str
    duration: float
    sample_rate: int
    pack: str = ""
    tags: List[str] = field(default_factory=list)
    attributes: Dict[str, float] = field(default_factory=dict)
    descriptors: List[str] = field(default_factory=list)


class SampleDictionary:
    """Dictionary of samples with metadata for quick access."""
    
    def __init__(self):
        self.entries: Dict[str, SampleEntry] = {}
        self._audio_cache: Dict[str, np.ndarray] = {}
    
    def add(self, name: str, audio: np.ndarray, sample_rate: int = 48000,
            pack: str = "", tags: List[str] = None) -> str:
        """Add a sample to the dictionary.
        
        Parameters
        ----------
        name : str
            Sample name (key)
        audio : np.ndarray
            Audio data
        sample_rate : int
            Sample rate
        pack : str
            Pack name this belongs to
        tags : list
            Optional tags
        
        Returns
        -------
        str
            Final name used (may be modified if duplicate)
        """
        # Handle duplicates
        base_name = name
        counter = 1
        while name in self.entries:
            name = f"{base_name}_{counter}"
            counter += 1
        
        entry = SampleEntry(
            name=name,
            path="",  # In-memory
            duration=len(audio) / sample_rate,
            sample_rate=sample_rate,
            pack=pack,
            tags=tags or [],
        )
        
        self.entries[name] = entry
        self._audio_cache[name] = audio.astype(np.float64)
        
        return name
    
    def get(self, name: str) -> Optional[np.ndarray]:
        """Get audio data for a sample."""
        if name in self._audio_cache:
            return self._audio_cache[name].copy()
        
        # Try loading from path
        entry = self.entries.get(name)
        if entry and entry.path:
            try:
                import soundfile as sf
                audio, sr = sf.read(entry.path)
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                self._audio_cache[name] = audio.astype(np.float64)
                return self._audio_cache[name].copy()
            except Exception:
                pass
        
        return None
    
    def remove(self, name: str) -> bool:
        """Remove a sample from the dictionary."""
        if name in self.entries:
            del self.entries[name]
            if name in self._audio_cache:
                del self._audio_cache[name]
            return True
        return False
    
    def list(self, pack: str = None, tags: List[str] = None) -> List[str]:
        """List samples, optionally filtered."""
        results = []
        for name, entry in self.entries.items():
            if pack and entry.pack != pack:
                continue
            if tags:
                if not any(t in entry.tags for t in tags):
                    continue
            results.append(name)
        return sorted(results)
    
    def search(self, query: str) -> List[str]:
        """Search samples by name or tags."""
        query_lower = query.lower()
        results = []
        for name, entry in self.entries.items():
            if query_lower in name.lower():
                results.append(name)
            elif any(query_lower in t.lower() for t in entry.tags):
                results.append(name)
        return sorted(results)
    
    def clear(self) -> int:
        """Clear all samples. Returns count cleared."""
        count = len(self.entries)
        self.entries.clear()
        self._audio_cache.clear()
        return count


# Global sample dictionary
_sample_dict: Optional[SampleDictionary] = None

def get_sample_dictionary() -> SampleDictionary:
    """Get or create the global sample dictionary."""
    global _sample_dict
    if _sample_dict is None:
        _sample_dict = SampleDictionary()
    return _sample_dict


# ============================================================================
# ALGORITHMIC GENERATION
# ============================================================================

def gen_kick(sr: int = 48000, variation: int = 0) -> np.ndarray:
    """Generate kick drum."""
    dur = 0.3 + (variation * 0.05)
    t = np.arange(int(sr * dur)) / sr
    
    # Pitch envelope (high to low)
    start_freq = 150 + (variation * 20)
    end_freq = 40 + (variation * 5)
    freq = start_freq * np.exp(-t * (8 + variation))
    freq = np.maximum(freq, end_freq)
    
    # Oscillator
    phase = np.cumsum(2 * np.pi * freq / sr)
    osc = np.sin(phase)
    
    # Amplitude envelope
    amp = np.exp(-t * (4 + variation * 0.5))
    
    # Add click transient
    click_dur = 0.005
    click_samples = int(click_dur * sr)
    click = np.random.randn(click_samples) * np.exp(-np.linspace(0, 5, click_samples))
    
    out = osc * amp
    out[:click_samples] += click * 0.3
    
    return np.clip(out, -1, 1).astype(np.float64)


def gen_snare(sr: int = 48000, variation: int = 0) -> np.ndarray:
    """Generate snare drum."""
    dur = 0.25 + (variation * 0.03)
    t = np.arange(int(sr * dur)) / sr
    
    # Body (pitched component)
    body_freq = 180 + (variation * 15)
    body = np.sin(2 * np.pi * body_freq * t) * np.exp(-t * 20)
    
    # Noise (snare wires)
    noise = np.random.randn(len(t))
    noise_env = np.exp(-t * (10 + variation * 2))
    noise = noise * noise_env
    
    # Highpass the noise
    try:
        from scipy import signal
        b, a = signal.butter(2, 2000 / (sr/2), 'high')
        noise = signal.lfilter(b, a, noise)
    except ImportError:
        pass
    
    # Mix
    mix = (0.4 + variation * 0.05)
    out = body * (1 - mix) + noise * mix
    
    return np.clip(out * 0.8, -1, 1).astype(np.float64)


def gen_hihat(sr: int = 48000, variation: int = 0, is_open: bool = False) -> np.ndarray:
    """Generate hi-hat."""
    dur = 0.5 if is_open else 0.1 + (variation * 0.02)
    t = np.arange(int(sr * dur)) / sr
    
    # Multiple high-frequency oscillators
    freqs = [4000 + (variation * 500), 7000 + (variation * 300), 10000]
    osc = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    
    # Noise component
    noise = np.random.randn(len(t)) * 0.5
    
    # Mix
    out = osc * 0.3 + noise * 0.7
    
    # Envelope
    decay = 5 if is_open else (30 + variation * 5)
    out = out * np.exp(-t * decay)
    
    # Highpass
    try:
        from scipy import signal
        b, a = signal.butter(2, 5000 / (sr/2), 'high')
        out = signal.lfilter(b, a, out)
    except ImportError:
        pass
    
    return np.clip(out, -1, 1).astype(np.float64)


def gen_perc(sr: int = 48000, variation: int = 0) -> np.ndarray:
    """Generate percussion one-shot."""
    dur = 0.15 + (variation * 0.05)
    t = np.arange(int(sr * dur)) / sr
    
    # Pitched metallic component
    freq = 300 + (variation * 100)
    freq2 = freq * (1.4 + variation * 0.1)  # Inharmonic
    
    osc1 = np.sin(2 * np.pi * freq * t)
    osc2 = np.sin(2 * np.pi * freq2 * t) * 0.5
    
    # Envelope
    env = np.exp(-t * (12 + variation * 2))
    
    out = (osc1 + osc2) * env
    
    return np.clip(out * 0.7, -1, 1).astype(np.float64)


def gen_bass(sr: int = 48000, variation: int = 0, note: float = 55) -> np.ndarray:
    """Generate bass synth hit."""
    dur = 0.4 + (variation * 0.1)
    t = np.arange(int(sr * dur)) / sr
    
    # Base frequency
    freq = note * (2 ** (variation / 12))  # Different notes
    
    # FM synthesis
    mod_ratio = 1 + (variation * 0.5)
    mod_depth = 2 + (variation * 0.5)
    
    mod = np.sin(2 * np.pi * freq * mod_ratio * t) * mod_depth
    carrier = np.sin(2 * np.pi * freq * t + mod)
    
    # Add sub
    sub = np.sin(2 * np.pi * freq * 0.5 * t) * 0.5
    
    # Envelope
    env = np.exp(-t * 3) * (1 - np.exp(-t * 50))  # Attack + decay
    
    out = (carrier + sub) * env
    
    return np.clip(out, -1, 1).astype(np.float64)


def gen_lead(sr: int = 48000, variation: int = 0, note: float = 440, 
             waveform: str = 'saw') -> np.ndarray:
    """Generate lead synth tone."""
    dur = 0.5 + (variation * 0.1)
    t = np.arange(int(sr * dur)) / sr
    
    freq = note
    
    if waveform == 'saw':
        osc = 2 * ((freq * t) % 1) - 1
    elif waveform == 'square':
        osc = np.sign(np.sin(2 * np.pi * freq * t))
    elif waveform == 'fm':
        mod = np.sin(2 * np.pi * freq * 2 * t) * 2
        osc = np.sin(2 * np.pi * freq * t + mod)
    else:
        osc = np.sin(2 * np.pi * freq * t)
    
    # Detune for thickness
    detune = 1.005 + (variation * 0.002)
    osc2 = 2 * ((freq * detune * t) % 1) - 1 if waveform == 'saw' else np.sin(2 * np.pi * freq * detune * t)
    osc = (osc + osc2 * 0.5) / 1.5
    
    # Envelope
    attack = 0.01
    release = 0.1 + variation * 0.05
    env = np.minimum(t / attack, 1.0) * np.exp(-(t - 0.3) * (1 / release))
    env = np.maximum(env, 0)
    
    out = osc * env
    
    return np.clip(out * 0.8, -1, 1).astype(np.float64)


def gen_pad(sr: int = 48000, variation: int = 0, note: float = 220) -> np.ndarray:
    """Generate pad/atmosphere."""
    dur = 2.0 + (variation * 0.5)
    t = np.arange(int(sr * dur)) / sr
    
    freq = note
    
    # 5 detuned oscillators
    detunes = [-0.006, -0.003, 0, 0.003, 0.006]
    osc = np.zeros_like(t)
    for d in detunes:
        osc += np.sin(2 * np.pi * freq * (1 + d + variation * 0.001) * t)
    osc = osc / len(detunes)
    
    # Fade in/out envelope
    fade_time = 0.3
    fade_samples = int(fade_time * sr)
    env = np.ones_like(t)
    env[:fade_samples] = np.linspace(0, 1, fade_samples)
    env[-fade_samples:] = np.linspace(1, 0, fade_samples)
    
    # Add subtle movement
    lfo = 1 + 0.1 * np.sin(2 * np.pi * 0.5 * t)
    
    out = osc * env * lfo
    
    return np.clip(out * 0.6, -1, 1).astype(np.float64)


def gen_riser(sr: int = 48000, variation: int = 0, dur: float = 2.0) -> np.ndarray:
    """Generate riser/sweep up."""
    t = np.arange(int(sr * dur)) / sr
    
    # Exponential frequency sweep
    start_freq = 100 + variation * 50
    end_freq = 2000 + variation * 500
    
    freq = start_freq * (end_freq / start_freq) ** (t / dur)
    phase = np.cumsum(2 * np.pi * freq / sr)
    
    osc = np.sin(phase)
    
    # Add noise
    noise = np.random.randn(len(t)) * 0.2
    
    # Rising amplitude
    amp = t / dur
    
    out = (osc + noise) * amp
    
    return np.clip(out * 0.7, -1, 1).astype(np.float64)


def gen_impact(sr: int = 48000, variation: int = 0) -> np.ndarray:
    """Generate impact/hit."""
    dur = 0.5 + variation * 0.2
    t = np.arange(int(sr * dur)) / sr
    
    # Low frequency thump
    freq = 60 + 200 * np.exp(-t * 10)
    phase = np.cumsum(2 * np.pi * freq / sr)
    thump = np.sin(phase) * np.exp(-t * 5)
    
    # Noise burst
    noise = np.random.randn(len(t)) * np.exp(-t * 15)
    
    # Layered crash
    crash_freq = 1000 + variation * 200
    crash = np.sin(2 * np.pi * crash_freq * t) * np.exp(-t * 8)
    
    out = thump * 0.6 + noise * 0.3 + crash * 0.2
    
    return np.clip(out, -1, 1).astype(np.float64)


def gen_sweep(sr: int = 48000, variation: int = 0, direction: str = 'down') -> np.ndarray:
    """Generate filter sweep."""
    dur = 1.0 + variation * 0.5
    t = np.arange(int(sr * dur)) / sr
    
    # Noise source
    noise = np.random.randn(len(t))
    
    # Time-varying filter (simulated)
    if direction == 'down':
        cutoff_curve = np.linspace(1.0, 0.1, len(t))
    else:
        cutoff_curve = np.linspace(0.1, 1.0, len(t))
    
    # Simple lowpass approximation via smoothing
    window = int(sr * 0.001 * (1 + variation))
    if window > 0:
        out = np.convolve(noise * cutoff_curve, np.ones(window)/window, mode='same')
    else:
        out = noise * cutoff_curve
    
    # Envelope
    env = np.exp(-t * 2)
    out = out * env
    
    return np.clip(out * 0.8, -1, 1).astype(np.float64)


# ============================================================================
# PACK GENERATION
# ============================================================================

GENERATOR_MAP = {
    'kicks': (gen_kick, 5, ['kick', 'drum']),
    'snares': (gen_snare, 5, ['snare', 'drum']),
    'hats': (gen_hihat, 5, ['hihat', 'drum', 'cymbal']),
    'percs': (gen_perc, 5, ['percussion', 'drum']),
    'bass': (gen_bass, 5, ['bass', 'synth']),
    'leads': (gen_lead, 5, ['lead', 'synth']),
    'pads': (gen_pad, 4, ['pad', 'synth', 'ambient']),
    'risers': (gen_riser, 3, ['riser', 'fx', 'transition']),
    'impacts': (gen_impact, 4, ['impact', 'fx', 'hit']),
    'sweeps': (gen_sweep, 3, ['sweep', 'fx', 'transition']),
}


def generate_pack(
    pack_name: str,
    gen_types: List[str],
    sample_rate: int = 48000,
    add_to_dict: bool = True,
) -> Tuple[List[str], str]:
    """Generate a sample pack.
    
    Parameters
    ----------
    pack_name : str
        Name for the pack
    gen_types : list
        Types to generate: kicks, snares, hats, percs, bass, leads, pads, risers, impacts, sweeps, all
    sample_rate : int
        Sample rate for generated samples
    add_to_dict : bool
        Also add to global sample dictionary
    
    Returns
    -------
    tuple
        (list of generated filenames, status message)
    """
    # Create pack directory
    pack_dir = get_packs_dir() / pack_name
    try:
        pack_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return [], f"ERROR: could not create pack directory: {e}"
    
    # Expand 'all'
    if 'all' in gen_types:
        gen_types = list(GENERATOR_MAP.keys())
    
    generated = []
    sample_dict = get_sample_dictionary() if add_to_dict else None
    
    for gen_type in gen_types:
        if gen_type not in GENERATOR_MAP:
            continue
        
        gen_func, count, tags = GENERATOR_MAP[gen_type]
        
        for i in range(count):
            try:
                # Generate audio
                if gen_type == 'hats' and i >= 3:
                    audio = gen_func(sample_rate, i, is_open=True)
                    prefix = 'hat_open'
                elif gen_type == 'hats':
                    audio = gen_func(sample_rate, i, is_open=False)
                    prefix = 'hat_closed'
                elif gen_type == 'leads':
                    waveforms = ['saw', 'square', 'fm', 'sine', 'saw']
                    audio = gen_func(sample_rate, i, waveform=waveforms[i])
                    prefix = f'lead_{waveforms[i]}'
                elif gen_type == 'sweeps':
                    direction = 'down' if i % 2 == 0 else 'up'
                    audio = gen_func(sample_rate, i, direction=direction)
                    prefix = f'sweep_{direction}'
                else:
                    audio = gen_func(sample_rate, i)
                    prefix = gen_type.rstrip('s')  # Remove trailing 's'
                
                # Save to file
                filename = f"{prefix}_{i+1:02d}.wav"
                out_path = pack_dir / filename
                
                try:
                    import soundfile as sf
                    sf.write(str(out_path), audio, sample_rate)
                except ImportError:
                    from scipy.io import wavfile
                    wavfile.write(str(out_path), sample_rate, (audio * 32767).astype(np.int16))
                
                generated.append(filename)
                
                # Add to dictionary
                if sample_dict:
                    dict_name = f"{pack_name}/{prefix}_{i+1:02d}"
                    sample_dict.add(dict_name, audio, sample_rate, pack_name, tags)
                    
            except Exception as e:
                generated.append(f"{gen_type}_{i+1:02d}.wav (ERROR: {e})")
    
    # Create manifest
    create_pack_manifest(
        pack_name,
        author='MDMA Generator',
        description=f'Auto-generated pack with {len(generated)} samples',
        tags=gen_types,
    )
    
    return generated, f"OK: generated pack '{pack_name}' with {len(generated)} samples"


# ============================================================================
# AI-POWERED PACK GENERATION
# ============================================================================

def generate_pack_ai(
    pack_name: str,
    prompts: List[str],
    session: "Session",
    samples_per_prompt: int = 1,
    duration: float = 3.0,
) -> Tuple[List[str], str]:
    """Generate pack using AI text-to-audio.
    
    Parameters
    ----------
    pack_name : str
        Pack name
    prompts : list
        Text prompts for generation
    session : Session
        Current session (for sample rate)
    samples_per_prompt : int
        Number of samples per prompt
    duration : float
        Duration for each sample
    
    Returns
    -------
    tuple
        (list of generated filenames, status message)
    """
    try:
        from ..ai.generation import generate_audio, detect_gpu, get_optimal_settings
    except ImportError:
        return [], "ERROR: AI generation not available. Install torch, diffusers."
    
    # Ensure GPU mode
    gpu_info = detect_gpu()
    settings = get_optimal_settings(gpu_info)
    
    # Force CUDA if available, even with fallback
    device = 'cuda' if gpu_info.get('cuda_available', False) else gpu_info.get('device', 'cpu')
    
    # Create pack directory
    pack_dir = get_packs_dir() / pack_name
    try:
        pack_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return [], f"ERROR: could not create pack directory: {e}"
    
    generated = []
    sample_dict = get_sample_dictionary()
    
    for prompt_idx, prompt in enumerate(prompts):
        for var_idx in range(samples_per_prompt):
            try:
                # Generate with unique seed
                seed = 42 + prompt_idx * 100 + var_idx
                
                audio, sr = generate_audio(
                    prompt=prompt,
                    duration=duration,
                    steps=settings.get('steps', 150),
                    cfg_scale=10.0,
                    seed=seed,
                    device=device,
                )
                
                # Resample to session rate if needed
                if sr != session.sample_rate:
                    ratio = session.sample_rate / sr
                    new_len = int(len(audio) * ratio)
                    x_old = np.linspace(0, 1, len(audio))
                    x_new = np.linspace(0, 1, new_len)
                    audio = np.interp(x_new, x_old, audio)
                    sr = session.sample_rate
                
                # Normalize
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = (audio / peak * 0.95).astype(np.float64)
                
                # Create filename from prompt
                safe_prompt = prompt[:30].replace(' ', '_').replace('/', '-')
                safe_prompt = ''.join(c for c in safe_prompt if c.isalnum() or c in '_-')
                filename = f"{safe_prompt}_{var_idx+1:02d}.wav"
                out_path = pack_dir / filename
                
                # Save
                try:
                    import soundfile as sf
                    sf.write(str(out_path), audio, sr)
                except ImportError:
                    from scipy.io import wavfile
                    wavfile.write(str(out_path), sr, (audio * 32767).astype(np.int16))
                
                generated.append(filename)
                
                # Add to dictionary
                dict_name = f"{pack_name}/{safe_prompt}_{var_idx+1:02d}"
                sample_dict.add(dict_name, audio, sr, pack_name, ['ai', 'generated'])
                
            except Exception as e:
                generated.append(f"prompt_{prompt_idx}_{var_idx}.wav (ERROR: {e})")
    
    # Create manifest
    create_pack_manifest(
        pack_name,
        author='MDMA AI',
        description=f'AI-generated pack from {len(prompts)} prompts',
        tags=['ai', 'generated'],
    )
    
    return generated, f"OK: AI generated pack '{pack_name}' with {len(generated)} samples"


# ============================================================================
# SESSION INTEGRATION
# ============================================================================

def load_sample_to_session(
    session: "Session",
    name: str,
    buffer_idx: int = None,
) -> Tuple[bool, str]:
    """Load a sample from dictionary to session buffer.
    
    Parameters
    ----------
    session : Session
        Current session
    name : str
        Sample name in dictionary
    buffer_idx : int, optional
        Buffer index (None = last_buffer)
    
    Returns
    -------
    tuple
        (success, message)
    """
    sample_dict = get_sample_dictionary()
    audio = sample_dict.get(name)
    
    if audio is None:
        return False, f"ERROR: sample '{name}' not found in dictionary"
    
    if buffer_idx is not None:
        if not hasattr(session, 'buffers'):
            session.buffers = {}
        session.buffers[buffer_idx] = audio
        msg = f"OK: loaded '{name}' to buffer {buffer_idx}"
    else:
        session.last_buffer = audio
        msg = f"OK: loaded '{name}' to working buffer"
    
    entry = sample_dict.entries.get(name)
    if entry:
        msg += f" ({entry.duration:.3f}s)"
    
    return True, msg


def save_buffer_to_dictionary(
    session: "Session",
    name: str,
    buffer_idx: int = None,
    pack: str = "",
    tags: List[str] = None,
) -> Tuple[bool, str]:
    """Save session buffer to dictionary.
    
    Parameters
    ----------
    session : Session
        Current session
    name : str
        Name for the sample
    buffer_idx : int, optional
        Buffer index (None = last_buffer)
    pack : str
        Pack to associate with
    tags : list
        Tags for the sample
    
    Returns
    -------
    tuple
        (success, message)
    """
    if buffer_idx is not None:
        if not hasattr(session, 'buffers') or buffer_idx not in session.buffers:
            return False, f"ERROR: buffer {buffer_idx} not found"
        audio = session.buffers[buffer_idx]
    else:
        audio = session.last_buffer
    
    if audio is None or len(audio) == 0:
        return False, "ERROR: no audio in buffer"
    
    sample_dict = get_sample_dictionary()
    final_name = sample_dict.add(name, audio, session.sample_rate, pack, tags)
    
    dur = len(audio) / session.sample_rate
    return True, f"OK: saved to dictionary as '{final_name}' ({dur:.3f}s)"
