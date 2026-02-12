"""Advanced Convolution & Impulse Response Engine - Phase 3.

Implements:
- Advanced convolution reverb with early/late reflection split
- Impulse-to-LFO waveshape conversion (Feature 3.1)
- Impulse-to-envelope conversion (Feature 3.2)
- Enhanced convolution with pre-delay, stereo width, EQ (Feature 3.3)
- Neural-inspired IR enhancement: extend, denoise, fill gaps (Feature 3.4)
- AI-descriptor IR transformation: semantic spectral reshaping (Feature 3.5)
- Granular IR processing: stretch, morph, redesign tails (Feature 3.6)

BUILD ID: convolution_v1.0_phase3
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from scipy import signal


SAMPLE_RATE = 48000


# ============================================================================
# UTILITY
# ============================================================================

def _normalize(buf: np.ndarray) -> np.ndarray:
    """Peak-normalize an array to [-1, 1]."""
    peak = np.max(np.abs(buf))
    if peak > 0:
        return buf / peak
    return buf


def _envelope_follower(audio: np.ndarray, sr: int = SAMPLE_RATE,
                       attack_ms: float = 5.0, release_ms: float = 50.0
                       ) -> np.ndarray:
    """Extract amplitude envelope from audio using attack/release smoothing."""
    rectified = np.abs(audio)
    attack_coeff = np.exp(-1.0 / (attack_ms * 0.001 * sr))
    release_coeff = np.exp(-1.0 / (release_ms * 0.001 * sr))
    env = np.zeros_like(rectified)
    env[0] = rectified[0]
    for i in range(1, len(rectified)):
        if rectified[i] > env[i - 1]:
            env[i] = attack_coeff * env[i - 1] + (1 - attack_coeff) * rectified[i]
        else:
            env[i] = release_coeff * env[i - 1] + (1 - release_coeff) * rectified[i]
    return env


# ============================================================================
# FEATURE 3.1: IMPULSE → LFO WAVESHAPE
# ============================================================================

def impulse_to_lfo(audio: np.ndarray, sr: int = SAMPLE_RATE,
                   cycle_samples: int = 4096) -> np.ndarray:
    """Convert an audio impulse into a single-cycle LFO waveshape.

    The impulse is resampled to *cycle_samples* length and peak-normalised
    to [-1, 1].  The result can be used as one cycle of a modulation
    oscillator via table look-up.

    Parameters
    ----------
    audio : np.ndarray
        Source impulse audio (mono).
    sr : int
        Sample rate of source (used for documentation; resampling is
        purely sample-count based).
    cycle_samples : int
        Length of the output single-cycle waveshape.

    Returns
    -------
    np.ndarray
        Single-cycle waveshape, length *cycle_samples*, range [-1, 1].
    """
    # Ensure mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample to cycle length
    x_old = np.linspace(0, 1, len(audio), endpoint=False)
    x_new = np.linspace(0, 1, cycle_samples, endpoint=False)
    waveshape = np.interp(x_new, x_old, audio)
    return _normalize(waveshape)


def lfo_from_waveshape(waveshape: np.ndarray, duration_sec: float,
                       rate: float = 1.0, depth: float = 1.0,
                       sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a modulation signal from a single-cycle waveshape.

    Parameters
    ----------
    waveshape : np.ndarray
        Single-cycle waveshape (from impulse_to_lfo).
    duration_sec : float
        Total modulation duration in seconds.
    rate : float
        LFO rate in Hz.
    depth : float
        Modulation depth multiplier (0-1).
    sr : int
        Sample rate.

    Returns
    -------
    np.ndarray
        Modulation signal in [-depth, depth] range.
    """
    n_samples = int(duration_sec * sr)
    cycle_len = len(waveshape)
    # Phase accumulator at *rate* Hz
    phase = np.arange(n_samples, dtype=np.float64) * rate / sr
    indices = (phase * cycle_len).astype(np.int64) % cycle_len
    return waveshape[indices] * depth


# ============================================================================
# FEATURE 3.2: IMPULSE → ENVELOPE SHAPE
# ============================================================================

def impulse_to_envelope(audio: np.ndarray, sr: int = SAMPLE_RATE,
                        attack_ms: float = 2.0, release_ms: float = 30.0,
                        target_samples: Optional[int] = None
                        ) -> np.ndarray:
    """Convert an audio impulse into an amplitude envelope contour.

    Follows the amplitude envelope of the impulse, then optionally
    resamples it to a target length so it can be applied to any sound.

    Parameters
    ----------
    audio : np.ndarray
        Source impulse audio (mono).
    sr : int
        Source sample rate.
    attack_ms : float
        Envelope follower attack (ms).
    release_ms : float
        Envelope follower release (ms).
    target_samples : int, optional
        If given, resample envelope to this length.

    Returns
    -------
    np.ndarray
        Amplitude envelope in [0, 1] range.
    """
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    env = _envelope_follower(audio, sr, attack_ms, release_ms)

    # Normalise to 0-1
    peak = np.max(env)
    if peak > 0:
        env = env / peak

    if target_samples is not None and target_samples != len(env):
        x_old = np.linspace(0, 1, len(env), endpoint=False)
        x_new = np.linspace(0, 1, target_samples, endpoint=False)
        env = np.interp(x_new, x_old, env)

    return env


# ============================================================================
# FEATURE 3.3: ADVANCED CONVOLUTION REVERB ENGINE
# ============================================================================

class ConvolutionEngine:
    """Advanced convolution reverb with early/late split, stereo width,
    pre-delay, decay control, and multi-band processing.

    This wraps and extends the existing convolve_reverb() in effects.py
    with features required by Phase 3.3.
    """

    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = sr
        # Current impulse response
        self.ir: Optional[np.ndarray] = None
        self.ir_name: str = ''
        # Split components
        self.early_ir: Optional[np.ndarray] = None
        self.late_ir: Optional[np.ndarray] = None
        # Parameters
        self.wet = 50.0        # 0-100
        self.dry = 50.0        # 0-100
        self.pre_delay_ms = 0.0
        self.decay = 1.0       # multiplier on IR tail length
        self.stereo_width = 50.0  # 0-100
        self.early_level = 50.0   # 0-100
        self.late_level = 50.0    # 0-100
        self.high_cut = 20000.0
        self.low_cut = 20.0
        self.early_late_split_ms = 80.0  # border between early and late
        # IR bank (named IRs)
        self.ir_bank: Dict[str, np.ndarray] = {}

    # ---- IR management ----

    def load_ir(self, ir: np.ndarray, name: str = '') -> None:
        """Load an impulse response array."""
        if ir.ndim == 2:
            ir = ir.mean(axis=1)
        self.ir = ir.astype(np.float64)
        self.ir_name = name
        self._split_ir()
        if name:
            self.ir_bank[name] = self.ir.copy()

    def load_ir_from_file(self, path: str, name: str = '') -> str:
        """Load an IR from a WAV file. Returns status message."""
        try:
            import wave as wav_mod
            with wav_mod.open(path, 'rb') as wf:
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                fr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())

            if sw == 2:
                data = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
            elif sw == 4:
                data = np.frombuffer(frames, dtype=np.int32).astype(np.float64) / 2147483648.0
            elif sw == 3:
                # 24-bit
                raw = np.frombuffer(frames, dtype=np.uint8)
                samples_24 = np.zeros(len(raw) // 3, dtype=np.int32)
                samples_24 = (raw[2::3].astype(np.int32) << 24 |
                              raw[1::3].astype(np.int32) << 16 |
                              raw[0::3].astype(np.int32) << 8) >> 8
                data = samples_24.astype(np.float64) / 8388608.0
            else:
                data = np.frombuffer(frames, dtype=np.uint8).astype(np.float64) / 128.0 - 1.0

            if ch == 2:
                data = (data[::2] + data[1::2]) / 2.0

            # Resample if needed
            if fr != self.sr:
                ratio = self.sr / fr
                new_len = int(len(data) * ratio)
                x_old = np.linspace(0, 1, len(data))
                x_new = np.linspace(0, 1, new_len)
                data = np.interp(x_new, x_old, data)

            if not name:
                import os
                name = os.path.splitext(os.path.basename(path))[0]
            self.load_ir(data, name)
            dur = len(self.ir) / self.sr
            return f"Loaded IR '{name}' ({dur:.2f}s, {len(self.ir)} samples)"
        except Exception as e:
            return f"ERROR loading IR: {e}"

    def load_preset(self, preset_name: str) -> str:
        """Load a built-in IR preset."""
        try:
            from ..dsp.effects import IR_PRESETS
            if preset_name.lower() in IR_PRESETS:
                ir = IR_PRESETS[preset_name.lower()]()
                self.load_ir(ir, preset_name)
                return f"Loaded preset IR '{preset_name}' ({len(self.ir)/self.sr:.2f}s)"
            return f"ERROR: Unknown preset '{preset_name}'. Available: {', '.join(sorted(IR_PRESETS.keys()))}"
        except ImportError:
            return "ERROR: Could not import IR presets"

    def save_ir(self, name: str) -> str:
        """Save current IR to the bank."""
        if self.ir is None:
            return "ERROR: No IR loaded"
        self.ir_bank[name] = self.ir.copy()
        return f"Saved IR as '{name}'"

    def list_irs(self) -> List[str]:
        """List all IRs in the bank."""
        return sorted(self.ir_bank.keys())

    def delete_ir(self, name: str) -> bool:
        """Delete an IR from the bank."""
        if name in self.ir_bank:
            del self.ir_bank[name]
            return True
        return False

    # ---- IR splitting ----

    def _split_ir(self) -> None:
        """Split IR into early reflections and late reverb tail."""
        if self.ir is None:
            self.early_ir = None
            self.late_ir = None
            return

        split_sample = int(self.early_late_split_ms * 0.001 * self.sr)
        split_sample = min(split_sample, len(self.ir))

        self.early_ir = np.zeros_like(self.ir)
        self.early_ir[:split_sample] = self.ir[:split_sample]

        self.late_ir = np.zeros_like(self.ir)
        self.late_ir[split_sample:] = self.ir[split_sample:]

    # ---- Parameter setters ----

    def set_params(self, **kwargs) -> None:
        """Set convolution parameters."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        # Re-split if split point changed
        if 'early_late_split_ms' in kwargs:
            self._split_ir()

    # ---- Processing ----

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply advanced convolution reverb to audio.

        Returns processed audio with wet/dry mix and stereo width.
        """
        if self.ir is None:
            return audio.copy()

        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        n = len(mono)

        # Optionally modify IR decay
        ir = self.ir.copy()
        if self.decay != 1.0:
            decay_env = np.exp(-np.linspace(0, 3 / max(self.decay, 0.01),
                                            len(ir)))
            ir = ir * decay_env

        # Apply frequency band cuts
        ir = self._apply_ir_eq(ir)

        # Pre-delay
        pre_samples = int(self.pre_delay_ms * 0.001 * self.sr)
        if pre_samples > 0:
            ir = np.concatenate([np.zeros(pre_samples), ir])

        # Early / late split convolution
        early_wet = np.zeros(n)
        late_wet = np.zeros(n)

        if self.early_ir is not None:
            early_ir = self.early_ir.copy()
            if self.decay != 1.0:
                early_ir = early_ir * np.exp(-np.linspace(0, 3 / max(self.decay, 0.01),
                                                           len(early_ir)))
            early_ir = self._apply_ir_eq(early_ir)
            conv = signal.fftconvolve(mono, early_ir, mode='full')[:n]
            early_wet = conv * (self.early_level / 100.0)

        if self.late_ir is not None:
            late_ir = self.late_ir.copy()
            if self.decay != 1.0:
                late_ir = late_ir * np.exp(-np.linspace(0, 3 / max(self.decay, 0.01),
                                                         len(late_ir)))
            late_ir = self._apply_ir_eq(late_ir)
            if pre_samples > 0:
                late_ir = np.concatenate([np.zeros(pre_samples), late_ir])
            conv = signal.fftconvolve(mono, late_ir, mode='full')[:n]
            late_wet = conv * (self.late_level / 100.0)

        wet_signal = early_wet + late_wet

        # Normalise wet to prevent clipping
        wet_peak = np.max(np.abs(wet_signal))
        if wet_peak > 1.0:
            wet_signal = wet_signal / wet_peak

        # Wet/dry mix
        wet_g = self.wet / 100.0
        dry_g = self.dry / 100.0
        mixed = dry_g * mono + wet_g * wet_signal

        # Stereo width (decorrelation)
        if audio.ndim == 2 or self.stereo_width > 0:
            width = self.stereo_width / 100.0
            # Create stereo from mono reverb with slight delay decorrelation
            delay_samps = int(0.012 * self.sr)  # 12ms Haas offset
            right_wet = np.roll(wet_signal, delay_samps) * wet_g
            left = dry_g * mono + wet_g * wet_signal
            right = dry_g * mono + right_wet
            # Cross-blend based on width
            mid = (left + right) * 0.5
            side = (left - right) * 0.5
            left_out = mid + side * width
            right_out = mid - side * width
            return np.column_stack([left_out, right_out])

        return mixed

    def _apply_ir_eq(self, ir: np.ndarray) -> np.ndarray:
        """Apply high/low cut filters to the IR."""
        nyquist = self.sr / 2.0
        modified = False

        if self.low_cut > 20.0:
            freq = min(self.low_cut, nyquist * 0.9) / nyquist
            if 0 < freq < 1:
                b, a = signal.butter(2, freq, 'high')
                ir = signal.filtfilt(b, a, ir)
                modified = True

        if self.high_cut < 20000.0:
            freq = min(self.high_cut, nyquist * 0.9) / nyquist
            if 0 < freq < 1:
                b, a = signal.butter(2, freq, 'low')
                ir = signal.filtfilt(b, a, ir)
                modified = True

        return ir

    def get_info(self) -> Dict[str, Any]:
        """Return current engine state."""
        return {
            'ir_loaded': self.ir is not None,
            'ir_name': self.ir_name,
            'ir_duration': len(self.ir) / self.sr if self.ir is not None else 0,
            'ir_samples': len(self.ir) if self.ir is not None else 0,
            'wet': self.wet,
            'dry': self.dry,
            'pre_delay_ms': self.pre_delay_ms,
            'decay': self.decay,
            'stereo_width': self.stereo_width,
            'early_level': self.early_level,
            'late_level': self.late_level,
            'high_cut': self.high_cut,
            'low_cut': self.low_cut,
            'early_late_split_ms': self.early_late_split_ms,
            'bank_count': len(self.ir_bank),
            'bank_names': sorted(self.ir_bank.keys()),
        }


# ============================================================================
# FEATURE 3.4: NEURAL-ENHANCED CONVOLUTION
# ============================================================================

def ir_extend(ir: np.ndarray, target_duration: float,
              sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extend an IR by learning its decay profile and extrapolating.

    Analyses the exponential decay rate and noise texture of the IR tail,
    then synthesises additional tail that matches. This is a
    spectral-domain approach inspired by neural resynthesis.

    Parameters
    ----------
    ir : np.ndarray
        Input impulse response.
    target_duration : float
        Target duration in seconds.
    sr : int
        Sample rate.

    Returns
    -------
    np.ndarray
        Extended IR.
    """
    current_dur = len(ir) / sr
    if target_duration <= current_dur:
        return ir.copy()

    # Analyse last quarter of IR for decay characteristics
    tail_start = max(len(ir) // 2, len(ir) - int(1.0 * sr))
    tail = ir[tail_start:]

    # Estimate decay rate from tail RMS in overlapping windows
    win_size = int(0.05 * sr)  # 50ms windows
    rms_vals = []
    for i in range(0, len(tail) - win_size, win_size // 2):
        rms = np.sqrt(np.mean(tail[i:i + win_size] ** 2))
        rms_vals.append(rms)

    if len(rms_vals) < 2 or rms_vals[0] <= 0:
        # Can't analyse, just zero-pad
        extend = int((target_duration - current_dur) * sr)
        return np.concatenate([ir, np.zeros(extend)])

    rms_vals = np.array(rms_vals)
    # Fit exponential: log(rms) = -rate * t + const
    times = np.arange(len(rms_vals)) * (win_size / 2) / sr
    log_rms = np.log(np.maximum(rms_vals, 1e-10))
    # Linear regression on log domain
    coeffs = np.polyfit(times, log_rms, 1)
    decay_rate = -coeffs[0]  # positive = decaying
    level_at_end = rms_vals[-1]

    # Spectral characteristics of tail (average spectrum)
    n_fft = min(2048, len(tail))
    spectrum = np.abs(np.fft.rfft(tail[-n_fft:]))
    spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum

    # Synthesise extension
    extend_samples = int((target_duration - current_dur) * sr)
    t_ext = np.arange(extend_samples) / sr

    # Decay envelope continuing from end
    env = level_at_end * np.exp(-decay_rate * t_ext)

    # Shaped noise with matching spectrum
    noise = np.random.randn(extend_samples)
    # Apply spectral shape
    noise_fft = np.fft.rfft(noise[:n_fft])
    target_freqs = len(noise_fft)
    spec_interp = np.interp(
        np.linspace(0, 1, target_freqs),
        np.linspace(0, 1, len(spectrum)),
        spectrum
    )
    noise_fft = noise_fft * spec_interp
    shaped_block = np.fft.irfft(noise_fft, n_fft)

    # Tile shaped noise to fill extension
    extension = np.tile(shaped_block, extend_samples // n_fft + 1)[:extend_samples]
    extension = extension * env

    # Crossfade at the join point
    xfade = min(int(0.02 * sr), len(ir) // 4, extend_samples // 4)
    if xfade > 0:
        fade_out = np.linspace(1, 0, xfade)
        fade_in = np.linspace(0, 1, xfade)
        ir_end = ir[-xfade:] * fade_out
        ext_start = extension[:xfade] * fade_in
        join = ir_end + ext_start
        return np.concatenate([ir[:-xfade], join, extension[xfade:]])

    return np.concatenate([ir, extension])


def ir_denoise(ir: np.ndarray, sr: int = SAMPLE_RATE,
               threshold_db: float = -60.0) -> np.ndarray:
    """Remove noise floor from a recorded IR.

    Uses spectral gating: estimates noise floor from the quietest
    portion of the IR, then gates the full IR spectrum against it.

    Parameters
    ----------
    ir : np.ndarray
        Input IR.
    sr : int
        Sample rate.
    threshold_db : float
        Noise floor threshold in dB.

    Returns
    -------
    np.ndarray
        Denoised IR.
    """
    # Estimate noise floor from last 10% (usually just noise)
    noise_region = ir[int(len(ir) * 0.9):]
    if len(noise_region) < 256:
        return ir.copy()

    # Spectral gate
    n_fft = 2048
    hop = n_fft // 4
    n_frames = (len(ir) - n_fft) // hop + 1

    if n_frames < 1:
        return ir.copy()

    # Estimate noise spectrum
    noise_spec = np.zeros(n_fft // 2 + 1)
    n_noise_frames = max(1, (len(noise_region) - n_fft) // hop + 1)
    window = np.hanning(n_fft)
    for i in range(n_noise_frames):
        start = i * hop
        if start + n_fft > len(noise_region):
            break
        frame = noise_region[start:start + n_fft] * window
        noise_spec += np.abs(np.fft.rfft(frame))
    noise_spec /= max(n_noise_frames, 1)
    noise_spec *= 2.0  # Safety margin

    # Process full IR with spectral subtraction
    output = np.zeros_like(ir, dtype=np.float64)
    window_sum = np.zeros_like(ir, dtype=np.float64)

    for i in range(n_frames):
        start = i * hop
        end = start + n_fft
        if end > len(ir):
            break

        frame = ir[start:end] * window
        spec = np.fft.rfft(frame)
        mag = np.abs(spec)
        phase = np.angle(spec)

        # Spectral subtraction
        clean_mag = np.maximum(mag - noise_spec, 0)
        clean_spec = clean_mag * np.exp(1j * phase)
        clean_frame = np.fft.irfft(clean_spec, n_fft)

        output[start:end] += clean_frame * window
        window_sum[start:end] += window ** 2

    # Normalise overlap-add
    mask = window_sum > 1e-8
    output[mask] /= window_sum[mask]

    return output


def ir_fill_gaps(ir: np.ndarray, sr: int = SAMPLE_RATE,
                 gap_threshold_db: float = -40.0) -> np.ndarray:
    """Fill gaps or dropouts in a recorded IR.

    Detects sudden dips in the IR envelope and interpolates across
    them using surrounding spectral content.
    """
    env = _envelope_follower(ir, sr, attack_ms=1.0, release_ms=5.0)
    threshold = 10 ** (gap_threshold_db / 20.0)

    # Find gap regions
    in_gap = env < threshold
    output = ir.copy()

    # Simple approach: linear interpolate across gaps
    gap_start = None
    for i in range(len(in_gap)):
        if in_gap[i] and gap_start is None:
            gap_start = i
        elif not in_gap[i] and gap_start is not None:
            gap_end = i
            gap_len = gap_end - gap_start
            if gap_len < int(0.1 * sr):  # Only fill gaps < 100ms
                before = max(0, gap_start - 1)
                after = min(len(ir) - 1, gap_end)
                interp = np.linspace(ir[before], ir[after], gap_len)
                output[gap_start:gap_end] = interp
            gap_start = None

    return output


# ============================================================================
# FEATURE 3.5: AI-DESCRIPTOR IMPULSE TRANSFORMATION
# ============================================================================

# Descriptor → spectral modification mapping
_DESCRIPTOR_TRANSFORMS = {
    'bigger': {'low_boost': 3.0, 'decay_mult': 1.5, 'stretch': 1.3},
    'smaller': {'high_boost': 2.0, 'decay_mult': 0.6, 'stretch': 0.7},
    'brighter': {'high_boost': 4.0, 'low_cut_hz': 300},
    'darker': {'low_boost': 3.0, 'high_cut_hz': 4000},
    'warmer': {'low_boost': 2.0, 'high_cut_hz': 8000, 'saturation': 0.3},
    'metallic': {'resonances': [800, 2200, 4500, 7000], 'res_q': 8.0},
    'wooden': {'resonances': [250, 700, 2000], 'res_q': 4.0, 'high_cut_hz': 6000},
    'glass': {'resonances': [1200, 3500, 6800], 'res_q': 12.0, 'high_boost': 2.0},
    'cathedral': {'decay_mult': 3.0, 'stretch': 2.0, 'pre_delay_ms': 40},
    'intimate': {'decay_mult': 0.4, 'stretch': 0.5, 'high_cut_hz': 10000},
    'ethereal': {'shimmer_shift': 12.0, 'decay_mult': 2.0, 'high_boost': 2.0},
    'haunted': {'reverse_mix': 0.4, 'decay_mult': 1.8, 'low_boost': 2.0},
    'telephone': {'bandpass_low': 300, 'bandpass_high': 3400, 'decay_mult': 0.3},
    'underwater': {'low_boost': 5.0, 'high_cut_hz': 1500, 'stretch': 1.5},
    'vintage': {'saturation': 0.4, 'high_cut_hz': 8000, 'low_boost': 1.5},
}


def ir_transform(ir: np.ndarray, descriptor: str,
                 intensity: float = 1.0,
                 sr: int = SAMPLE_RATE) -> np.ndarray:
    """Transform an IR using a semantic descriptor.

    Parameters
    ----------
    ir : np.ndarray
        Input IR.
    descriptor : str
        Transformation descriptor (e.g. 'bigger', 'darker', 'metallic').
    intensity : float
        Intensity of transformation (0-1, default 1.0).
    sr : int
        Sample rate.

    Returns
    -------
    np.ndarray
        Transformed IR.
    """
    desc = descriptor.lower().strip()
    if desc not in _DESCRIPTOR_TRANSFORMS:
        return ir.copy()

    params = _DESCRIPTOR_TRANSFORMS[desc]
    result = ir.copy().astype(np.float64)
    nyquist = sr / 2.0

    # Decay multiplier
    if 'decay_mult' in params:
        mult = 1.0 + (params['decay_mult'] - 1.0) * intensity
        if mult > 1.0:
            result = ir_extend(result, len(result) / sr * mult, sr)
        elif mult < 1.0:
            new_len = int(len(result) * mult)
            fade = np.linspace(1, 0, len(result) - new_len)
            result[new_len:] *= fade[:len(result) - new_len]

    # Stretch
    if 'stretch' in params:
        stretch = 1.0 + (params['stretch'] - 1.0) * intensity
        new_len = int(len(result) * stretch)
        x_old = np.linspace(0, 1, len(result))
        x_new = np.linspace(0, 1, new_len)
        result = np.interp(x_new, x_old, result)

    # Low boost
    if 'low_boost' in params:
        boost = 1.0 + (params['low_boost'] - 1.0) * intensity
        freq = 300.0 / nyquist
        if 0 < freq < 1:
            b, a = signal.butter(2, freq, 'low')
            low = signal.filtfilt(b, a, result)
            result = result + low * (boost - 1.0)

    # High boost
    if 'high_boost' in params:
        boost = 1.0 + (params['high_boost'] - 1.0) * intensity
        freq = 3000.0 / nyquist
        if 0 < freq < 1:
            b, a = signal.butter(2, freq, 'high')
            high = signal.filtfilt(b, a, result)
            result = result + high * (boost - 1.0)

    # High cut
    if 'high_cut_hz' in params:
        cut = params['high_cut_hz']
        freq = min(cut, nyquist * 0.9) / nyquist
        if 0 < freq < 1:
            b, a = signal.butter(3, freq, 'low')
            filtered = signal.filtfilt(b, a, result)
            result = result * (1 - intensity) + filtered * intensity

    # Low cut
    if 'low_cut_hz' in params:
        cut = params['low_cut_hz']
        freq = min(cut, nyquist * 0.9) / nyquist
        if 0 < freq < 1:
            b, a = signal.butter(3, freq, 'high')
            filtered = signal.filtfilt(b, a, result)
            result = result * (1 - intensity) + filtered * intensity

    # Bandpass
    if 'bandpass_low' in params and 'bandpass_high' in params:
        lo = min(params['bandpass_low'], nyquist * 0.9) / nyquist
        hi = min(params['bandpass_high'], nyquist * 0.9) / nyquist
        if 0 < lo < hi < 1:
            b, a = signal.butter(3, [lo, hi], 'band')
            filtered = signal.filtfilt(b, a, result)
            result = result * (1 - intensity) + filtered * intensity

    # Resonances
    if 'resonances' in params:
        q = params.get('res_q', 8.0)
        for freq_hz in params['resonances']:
            w0 = freq_hz / nyquist
            if 0 < w0 < 1:
                bw = w0 / q
                b, a = signal.iirpeak(w0, q)
                resonant = signal.filtfilt(b, a, result)
                result = result + resonant * 0.3 * intensity

    # Saturation
    if 'saturation' in params:
        sat = params['saturation'] * intensity
        result = np.tanh(result * (1.0 + sat * 3.0)) / (1.0 + sat * 2.0)

    # Shimmer (pitch-shift up and add)
    if 'shimmer_shift' in params:
        shift = params['shimmer_shift']
        ratio = 2 ** (shift / 12.0)
        # Simple pitch shift by resampling
        shifted_len = int(len(result) / ratio)
        x_old = np.linspace(0, 1, shifted_len)
        x_new = np.linspace(0, 1, len(result))
        shifted = np.interp(x_old, x_new, result)
        # Pad back to original length with decay
        if len(shifted) < len(result):
            shifted = np.concatenate([shifted,
                                       np.zeros(len(result) - len(shifted))])
        else:
            shifted = shifted[:len(result)]
        result = result + shifted * 0.3 * intensity

    # Reverse mix
    if 'reverse_mix' in params:
        mix = params['reverse_mix'] * intensity
        result = result * (1 - mix) + result[::-1] * mix

    # Pre-delay
    if 'pre_delay_ms' in params:
        delay_samps = int(params['pre_delay_ms'] * intensity * sr / 1000.0)
        if delay_samps > 0:
            result = np.concatenate([np.zeros(delay_samps), result])

    return _normalize(result)


def list_descriptors() -> List[str]:
    """Return all available transformation descriptors."""
    return sorted(_DESCRIPTOR_TRANSFORMS.keys())


# ============================================================================
# FEATURE 3.6: GRANULAR IMPULSE TOOLS
# ============================================================================

def granular_ir_stretch(ir: np.ndarray, stretch_factor: float = 2.0,
                        grain_ms: float = 40.0, density: float = 8.0,
                        sr: int = SAMPLE_RATE) -> np.ndarray:
    """Stretch an IR using granular processing.

    Uses overlapping grains to extend the reverb tail without
    changing the tonal character.
    """
    try:
        from .granular import GranularEngine
        engine = GranularEngine(sr)
        engine.set_params(
            grain_size_ms=grain_ms,
            density=density,
            envelope='hann',
            position_spread=0.05,
            pitch_ratio=1.0,
        )
        target_dur = len(ir) / sr * stretch_factor
        return engine.time_stretch(ir, stretch_factor)
    except Exception:
        # Fallback: simple interpolation stretch
        new_len = int(len(ir) * stretch_factor)
        return np.interp(
            np.linspace(0, 1, new_len),
            np.linspace(0, 1, len(ir)),
            ir
        )


def granular_ir_morph(ir_a: np.ndarray, ir_b: np.ndarray,
                      morph_pos: float = 0.5,
                      grain_ms: float = 30.0,
                      sr: int = SAMPLE_RATE) -> np.ndarray:
    """Morph between two IRs using granular interleaving.

    Alternates grains from each IR based on morph_pos (0 = all A,
    1 = all B).
    """
    # Match lengths
    max_len = max(len(ir_a), len(ir_b))
    a = np.zeros(max_len)
    b = np.zeros(max_len)
    a[:len(ir_a)] = ir_a
    b[:len(ir_b)] = ir_b

    grain_samples = int(grain_ms * 0.001 * sr)
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(grain_samples) /
                                 (grain_samples - 1)))

    output = np.zeros(max_len)
    pos = 0
    grain_idx = 0

    while pos + grain_samples <= max_len:
        # Decide which source to use for this grain
        use_b = (np.random.random() < morph_pos)
        source = b if use_b else a
        grain = source[pos:pos + grain_samples] * window
        output[pos:pos + grain_samples] += grain
        pos += grain_samples // 2  # 50% overlap
        grain_idx += 1

    return _normalize(output)


def granular_ir_redesign(ir: np.ndarray, grain_ms: float = 20.0,
                         density: float = 6.0, scatter: float = 0.3,
                         reverse_prob: float = 0.2,
                         sr: int = SAMPLE_RATE) -> np.ndarray:
    """Redesign an IR using granular decomposition and re-synthesis.

    Breaks the IR into grains and re-assembles them with randomised
    positioning, optional reversal, and density control.  Produces
    a new IR with similar spectral character but different time structure.
    """
    try:
        from .granular import GranularEngine
        engine = GranularEngine(sr)
        engine.set_params(
            grain_size_ms=grain_ms,
            density=density,
            position_spread=scatter,
            reverse_prob=reverse_prob,
            envelope='hann',
            pitch_ratio=1.0,
        )
        target_dur = len(ir) / sr
        return engine.process(ir, target_dur)
    except Exception:
        return ir.copy()


# ============================================================================
# GLOBAL CONVOLUTION ENGINE INSTANCE
# ============================================================================

_conv_engine: Optional[ConvolutionEngine] = None


def get_convolution_engine() -> ConvolutionEngine:
    """Get the global ConvolutionEngine instance."""
    global _conv_engine
    if _conv_engine is None:
        _conv_engine = ConvolutionEngine()
    return _conv_engine
