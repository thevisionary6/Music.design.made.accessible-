"""Pattern Module - Enhanced Audio-Rate Pattern Modulation for MDMA.

This module provides robust pattern-based audio manipulation including:
- Adaptive note lengths based on audio/clip duration
- Beat-based duration notation (D1=1 beat, D0.5=half beat, R1=rest 1 beat)
- Multiple pattern algorithms with index selection
- Anti-artifact processing (crossfades, smoothing, click removal)
- Audio-rate mode for granular repetition at detected frequency
- Pattern blocks that continue until /end

DURATION NOTATION:
-----------------
D1   = hold for 1 beat
D0.5 = hold for half beat (1/8 note at standard)
D2   = hold for 2 beats
D4   = hold for 4 beats (1 bar at 4/4)

R1   = rest for 1 beat
R0.5 = rest for half beat
R2   = rest for 2 beats

ADAPTIVE MODE:
-------------
If no durations specified, notes divide audio equally.
Pattern "0 7 12" on 3-second audio = 1 second per note.

ALGORITHMS:
----------
0: general     - Equal division, smooth crossfades
1: staccato    - Short notes with gaps
2: legato      - Overlapping notes, no gaps
3: audiorate   - Granular repetition at source frequency
4: glide       - Smooth pitch transitions
5: stutter     - Rhythmic repetition
6: reverse     - Play notes in reverse order
7: bounce      - Forward then backward
8: random      - Randomize note order (seeded)
9: humanize    - Add timing/velocity variation

BUILD ID: pattern_v14.2_chunk3_enhanced
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import re

from .scaling import parse_param, scale_wet, clamp_param


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_BPM = 128.0
DEFAULT_STEP = 1.0  # 1 beat (was 0.125 for 1/32)

# Anti-artifact settings
XFADE_SAMPLES = 512          # Crossfade for smooth transitions
ATTACK_SAMPLES = 128         # Attack fade to prevent clicks
RELEASE_SAMPLES = 256        # Release fade
MIN_SEGMENT_SAMPLES = 64     # Minimum segment size

# Audio-rate detection
FREQ_DETECT_MIN_HZ = 20.0
FREQ_DETECT_MAX_HZ = 4000.0


# ============================================================================
# ENUMS
# ============================================================================

class PlaybackMode(Enum):
    """Pattern playback modes."""
    ONESHOT = "oneshot"
    LOOP = "loop"
    PINGPONG = "pingpong"
    REVERSE = "reverse"


class NoteType(Enum):
    """Note type in a pattern."""
    PITCH = "pitch"
    REST = "rest"
    HOLD = "hold"
    GLIDE = "glide"


class PatternAlgorithm(Enum):
    """Pattern processing algorithms."""
    GENERAL = 0       # Equal division, smooth crossfades
    STACCATO = 1      # Short notes with gaps
    LEGATO = 2        # Overlapping notes, no gaps
    AUDIORATE = 3     # Granular repetition at source frequency
    GLIDE = 4         # Smooth pitch transitions
    STUTTER = 5       # Rhythmic repetition
    REVERSE = 6       # Play notes in reverse order
    BOUNCE = 7        # Forward then backward
    RANDOM = 8        # Randomize note order
    HUMANIZE = 9      # Add timing/velocity variation


# Algorithm info for PAG command
ALGORITHM_INFO: Dict[int, Dict[str, str]] = {
    0: {"name": "general", "desc": "Equal division with smooth crossfades"},
    1: {"name": "staccato", "desc": "Short notes (70%) with silent gaps"},
    2: {"name": "legato", "desc": "Overlapping notes for smooth flow"},
    3: {"name": "audiorate", "desc": "Granular repetition at detected/known frequency"},
    4: {"name": "glide", "desc": "Smooth pitch transitions between notes"},
    5: {"name": "stutter", "desc": "Rhythmic repetition of each note"},
    6: {"name": "reverse", "desc": "Play pattern notes in reverse order"},
    7: {"name": "bounce", "desc": "Forward then backward (ping-pong)"},
    8: {"name": "random", "desc": "Randomize note order (deterministic seed)"},
    9: {"name": "humanize", "desc": "Add timing and velocity variation"},
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PatternNote:
    """Single note in a pattern sequence.
    
    Attributes
    ----------
    note_type : NoteType
        Type of note (pitch, rest, hold, glide)
    semitone : float
        Pitch offset in semitones (0 = no change)
    duration : float
        Duration in beats (1.0 = one beat, 0.5 = half beat)
        If None/0, uses adaptive mode (equal division)
    velocity : float
        Velocity/amplitude (0-100 scale)
    """
    note_type: NoteType = NoteType.PITCH
    semitone: float = 0.0
    duration: float = 0.0  # 0 = adaptive (equal division)
    velocity: float = 100.0
    
    def __post_init__(self):
        self.velocity = clamp_param(self.velocity, allow_wacky=False)


@dataclass
class PatternClip:
    """Pattern clip containing a note sequence and source audio.
    
    Attributes
    ----------
    name : str
        Clip identifier
    source_buffer : np.ndarray
        Source audio buffer to pitch-shift
    notes : List[PatternNote]
        Sequence of notes to apply
    sample_rate : int
        Audio sample rate
    bpm : float
        Tempo in beats per minute
    algorithm : int
        Pattern algorithm index (0-9)
    source_frequency : float
        Known source frequency (Hz) for audio-rate mode, 0 = auto-detect
    """
    name: str = "pattern_clip"
    source_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0))
    notes: List[PatternNote] = field(default_factory=list)
    sample_rate: int = DEFAULT_SAMPLE_RATE
    bpm: float = DEFAULT_BPM
    algorithm: int = 0  # PatternAlgorithm.GENERAL
    source_frequency: float = 0.0  # 0 = auto-detect
    mode: PlaybackMode = PlaybackMode.LOOP
    
    # Computed
    _rendered_buffer: Optional[np.ndarray] = field(default=None, repr=False)
    _detected_frequency: float = field(default=0.0, repr=False)
    
    @property
    def total_duration_beats(self) -> float:
        """Total duration in beats from explicit note durations."""
        total = 0.0
        for note in self.notes:
            if note.duration > 0:
                total += note.duration
        return total
    
    @property
    def source_duration_seconds(self) -> float:
        """Duration of source audio in seconds."""
        if len(self.source_buffer) == 0:
            return 0.0
        return len(self.source_buffer) / self.sample_rate
    
    def beats_to_samples(self, beats: float) -> int:
        """Convert beats to samples."""
        if self.bpm <= 0:
            return int(beats * self.sample_rate)
        return int(beats * 60.0 / self.bpm * self.sample_rate)
    
    def samples_to_beats(self, samples: int) -> float:
        """Convert samples to beats."""
        if self.bpm <= 0:
            return samples / self.sample_rate
        return samples / self.sample_rate * self.bpm / 60.0


# ============================================================================
# PATTERN PARSING - ENHANCED
# ============================================================================

def parse_pattern_token(token: str) -> PatternNote:
    """Parse a single pattern token into a PatternNote.
    
    Enhanced Token Formats:
    ----------------------
    <number>          Pitch offset in semitones (0, 7, -5, 3.5)
    <number>/<dur>    Pitch with explicit beat duration (7/1 = 7 semitones, 1 beat)
    
    R or R<dur>       Rest (R = adaptive, R1 = 1 beat, R0.5 = half beat)
    D or D<dur>       Hold previous note (D = adaptive, D1 = 1 beat)
    G<pitch>          Glide to pitch (smooth transition)
    G<pitch>/<dur>    Glide with duration
    
    V<vel>:<pitch>    Velocity prefix (V80:7 = velocity 80%, pitch +7)
    
    Duration shortcuts:
    /1   = 1 beat (quarter note)
    /2   = 2 beats (half note)
    /4   = 4 beats (whole note)
    /0.5 = half beat (eighth note)
    /0.25 = quarter beat (sixteenth note)
    """
    token = token.strip()
    if not token:
        return PatternNote(NoteType.REST, 0.0, 0.0, 100.0)
    
    upper = token.upper()
    
    # Parse duration suffix if present
    duration = 0.0  # 0 = adaptive
    if '/' in token:
        parts = token.rsplit('/', 1)
        try:
            duration = float(parts[1])
        except ValueError:
            duration = 0.0
        token = parts[0]
        upper = token.upper()
    
    # Rest: R or R<duration>
    if upper.startswith('R'):
        rest_dur = upper[1:]
        if rest_dur and duration == 0.0:
            try:
                duration = float(rest_dur)
            except ValueError:
                pass
        return PatternNote(NoteType.REST, 0.0, duration, 0.0)
    
    # Hold/Duration extend: D or D<duration>
    if upper.startswith('D') or upper.startswith('H'):
        hold_dur = upper[1:]
        if hold_dur and duration == 0.0:
            try:
                duration = float(hold_dur)
            except ValueError:
                pass
        return PatternNote(NoteType.HOLD, 0.0, duration, 100.0)
    
    # Glide: G<pitch> or G<pitch>/<dur>
    if upper.startswith('G'):
        semi_str = token[1:]
        try:
            semitone = float(semi_str) if semi_str else 0.0
        except ValueError:
            semitone = 0.0
        return PatternNote(NoteType.GLIDE, semitone, duration, 100.0)
    
    # Velocity prefix: V<vel>:<pitch>
    if upper.startswith('V') and ':' in token:
        parts = token[1:].split(':')
        try:
            velocity = float(parts[0])
            semitone = float(parts[1]) if len(parts) > 1 else 0.0
        except ValueError:
            velocity = 100.0
            semitone = 0.0
        return PatternNote(NoteType.PITCH, semitone, duration, velocity)
    
    # Standard pitch (possibly with dots for simple hold notation)
    try:
        semitone = float(token)
        return PatternNote(NoteType.PITCH, semitone, duration, 100.0)
    except ValueError:
        # Unknown token, treat as rest
        return PatternNote(NoteType.REST, 0.0, duration, 0.0)


def parse_pattern(
    tokens: Union[str, List[str]],
    adaptive_to_length: float = 0.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    bpm: float = DEFAULT_BPM,
) -> List[PatternNote]:
    """Parse a pattern string or token list into PatternNotes.
    
    Parameters
    ----------
    tokens : str or List[str]
        Pattern as space-separated string or list of tokens
    adaptive_to_length : float
        If > 0, adjust adaptive note durations to fit this length in seconds
    sample_rate : int
        Sample rate for calculations
    bpm : float
        BPM for beat calculations
        
    Returns
    -------
    List[PatternNote]
        List of parsed notes with durations calculated
    """
    if isinstance(tokens, str):
        # Split on whitespace, commas, pipes, or newlines
        tokens = re.split(r'[\s,|\n]+', tokens)
    
    notes = []
    for tok in tokens:
        tok = tok.strip()
        if tok and not tok.startswith('#') and tok.lower() not in ('/end', 'end'):
            notes.append(parse_pattern_token(tok))
    
    if not notes:
        return notes
    
    # Calculate adaptive durations
    if adaptive_to_length > 0:
        # Count notes with adaptive (0) duration
        adaptive_notes = [n for n in notes if n.duration <= 0]
        explicit_duration = sum(n.duration for n in notes if n.duration > 0)
        
        # Convert explicit duration (beats) to seconds
        explicit_seconds = explicit_duration * 60.0 / bpm if bpm > 0 else explicit_duration
        
        # Remaining time for adaptive notes
        remaining_seconds = max(0, adaptive_to_length - explicit_seconds)
        
        if adaptive_notes and remaining_seconds > 0:
            # Equal division among adaptive notes
            seconds_per_note = remaining_seconds / len(adaptive_notes)
            beats_per_note = seconds_per_note * bpm / 60.0 if bpm > 0 else seconds_per_note
            
            for note in adaptive_notes:
                note.duration = beats_per_note
        elif adaptive_notes:
            # No remaining time, give minimum duration
            for note in adaptive_notes:
                note.duration = 0.125  # 1/32 note minimum
    else:
        # No adaptive target - give adaptive notes 1 beat each
        for note in notes:
            if note.duration <= 0:
                note.duration = 1.0
    
    return notes


def pattern_to_string(notes: List[PatternNote]) -> str:
    """Convert pattern notes back to string representation."""
    tokens = []
    for note in notes:
        dur_suffix = f"/{note.duration:g}" if note.duration != 1.0 else ""
        
        if note.note_type == NoteType.REST:
            tokens.append(f"R{note.duration:g}" if note.duration > 0 else "R")
        elif note.note_type == NoteType.HOLD:
            tokens.append(f"D{note.duration:g}" if note.duration > 0 else "D")
        elif note.note_type == NoteType.GLIDE:
            tokens.append(f"G{note.semitone:g}{dur_suffix}")
        else:
            if note.velocity != 100.0:
                tokens.append(f"V{note.velocity:.0f}:{note.semitone:g}{dur_suffix}")
            else:
                tokens.append(f"{note.semitone:g}{dur_suffix}")
    return ' '.join(tokens)


# ============================================================================
# ANTI-ARTIFACT PROCESSING
# ============================================================================

def apply_attack_release(
    buffer: np.ndarray,
    attack_samples: int = ATTACK_SAMPLES,
    release_samples: int = RELEASE_SAMPLES,
) -> np.ndarray:
    """Apply attack and release fades to prevent clicks."""
    if len(buffer) < attack_samples + release_samples:
        return buffer
    
    out = buffer.astype(np.float64)
    
    # Attack fade
    if attack_samples > 0:
        attack = np.linspace(0, 1, attack_samples)
        out[:attack_samples] *= attack
    
    # Release fade
    if release_samples > 0:
        release = np.linspace(1, 0, release_samples)
        out[-release_samples:] *= release
    
    return out


def crossfade_segments(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    xfade_samples: int = XFADE_SAMPLES,
) -> np.ndarray:
    """Crossfade two segments together."""
    if len(seg_a) == 0:
        return seg_b
    if len(seg_b) == 0:
        return seg_a
    
    xfade = min(xfade_samples, len(seg_a), len(seg_b))
    
    if xfade < 2:
        return np.concatenate([seg_a, seg_b])
    
    # Create crossfade
    fade_out = np.linspace(1, 0, xfade)
    fade_in = np.linspace(0, 1, xfade)
    
    # Apply crossfade
    a_end = seg_a[-xfade:].astype(np.float64) * fade_out
    b_start = seg_b[:xfade].astype(np.float64) * fade_in
    crossfaded = a_end + b_start
    
    # Build output
    return np.concatenate([
        seg_a[:-xfade],
        crossfaded,
        seg_b[xfade:]
    ])


def remove_dc_offset(buffer: np.ndarray) -> np.ndarray:
    """Remove DC offset from buffer."""
    return buffer - np.mean(buffer)


def smooth_discontinuities(
    buffer: np.ndarray,
    window_size: int = 64,
) -> np.ndarray:
    """Smooth any sharp discontinuities in the buffer."""
    if len(buffer) < window_size * 2:
        return buffer
    
    out = buffer.astype(np.float64)
    
    # Simple moving average to smooth discontinuities
    # Only apply at potential problem points
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(out, kernel, mode='same')
    
    # Blend original with smoothed at potential click points
    # (where derivative is very high)
    diff = np.abs(np.diff(out, prepend=out[0]))
    threshold = np.std(diff) * 3
    
    click_mask = diff > threshold
    # Dilate mask
    for _ in range(window_size // 4):
        click_mask = np.logical_or(click_mask, np.roll(click_mask, 1))
        click_mask = np.logical_or(click_mask, np.roll(click_mask, -1))
    
    out[click_mask] = smoothed[click_mask]
    
    return out


def anti_artifact_process(
    buffer: np.ndarray,
    attack: int = ATTACK_SAMPLES,
    release: int = RELEASE_SAMPLES,
) -> np.ndarray:
    """Apply full anti-artifact processing pipeline."""
    if len(buffer) == 0:
        return buffer
    
    out = buffer.astype(np.float64)
    
    # Remove DC offset
    out = remove_dc_offset(out)
    
    # Smooth any discontinuities
    out = smooth_discontinuities(out)
    
    # Apply attack/release
    out = apply_attack_release(out, attack, release)
    
    # Normalize to prevent clipping
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak * 0.99
    
    return out


# ============================================================================
# FREQUENCY DETECTION
# ============================================================================

def detect_fundamental_frequency(
    buffer: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    min_freq: float = FREQ_DETECT_MIN_HZ,
    max_freq: float = FREQ_DETECT_MAX_HZ,
) -> float:
    """Detect fundamental frequency using autocorrelation.
    
    Returns 0 if frequency cannot be reliably detected.
    """
    if len(buffer) < sample_rate // int(max_freq) * 2:
        return 0.0
    
    # Use a segment from the middle of the buffer
    seg_len = min(len(buffer), sample_rate // 2)  # Max 0.5 seconds
    start = max(0, (len(buffer) - seg_len) // 2)
    segment = buffer[start:start + seg_len].astype(np.float64)
    
    # Normalize
    segment = segment - np.mean(segment)
    if np.max(np.abs(segment)) < 0.001:
        return 0.0
    segment = segment / np.max(np.abs(segment))
    
    # Autocorrelation
    n = len(segment)
    autocorr = np.correlate(segment, segment, mode='full')[n-1:]
    
    # Find period (first peak after initial decay)
    min_lag = int(sample_rate / max_freq)
    max_lag = int(sample_rate / min_freq)
    
    if max_lag > len(autocorr) - 1:
        max_lag = len(autocorr) - 1
    
    if min_lag >= max_lag:
        return 0.0
    
    search_region = autocorr[min_lag:max_lag]
    if len(search_region) == 0:
        return 0.0
    
    peak_idx = np.argmax(search_region) + min_lag
    
    # Verify it's a good peak
    if autocorr[peak_idx] < 0.3 * autocorr[0]:
        return 0.0  # Not a reliable peak
    
    # Refine with parabolic interpolation
    if 0 < peak_idx < len(autocorr) - 1:
        alpha = autocorr[peak_idx - 1]
        beta = autocorr[peak_idx]
        gamma = autocorr[peak_idx + 1]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-10:
            peak_refined = peak_idx + 0.5 * (alpha - gamma) / denom
            return sample_rate / peak_refined
    
    return sample_rate / peak_idx if peak_idx > 0 else 0.0


# ============================================================================
# PITCH SHIFTING
# ============================================================================

def pitch_shift_segment(
    segment: np.ndarray,
    semitones: float,
    target_length: int,
) -> np.ndarray:
    """Pitch-shift a segment using high-quality resampling."""
    try:
        import scipy.signal
    except ImportError:
        # Fallback without scipy
        return _pitch_shift_simple(segment, semitones, target_length)
    
    if len(segment) == 0:
        return np.zeros(target_length, dtype=np.float64)
    
    # Calculate pitch ratio
    ratio = 2.0 ** (semitones / 12.0)
    
    if abs(ratio - 1.0) < 0.001:
        # No pitch change needed
        if len(segment) >= target_length:
            return segment[:target_length].astype(np.float64)
        else:
            out = np.zeros(target_length, dtype=np.float64)
            out[:len(segment)] = segment
            return out
    
    # Resample to achieve pitch shift
    new_length = int(len(segment) / ratio)
    if new_length < 1:
        new_length = 1
    
    resampled = scipy.signal.resample(segment.astype(np.float64), new_length)
    
    # Fit to target length
    out = np.zeros(target_length, dtype=np.float64)
    
    if len(resampled) >= target_length:
        out[:] = resampled[:target_length]
    else:
        # Loop with crossfade
        pos = 0
        xfade = min(XFADE_SAMPLES, len(resampled) // 4)
        while pos < target_length:
            chunk_len = min(len(resampled), target_length - pos)
            out[pos:pos + chunk_len] = resampled[:chunk_len]
            
            if pos > 0 and xfade > 0 and pos < target_length:
                xf_len = min(xfade, pos, chunk_len)
                fade_in = np.linspace(0, 1, xf_len)
                fade_out = np.linspace(1, 0, xf_len)
                out[pos:pos + xf_len] = (
                    out[pos:pos + xf_len] * fade_in +
                    out[pos - xf_len:pos] * fade_out
                )
            
            pos += len(resampled)
    
    return out


def _pitch_shift_simple(
    segment: np.ndarray,
    semitones: float,
    target_length: int,
) -> np.ndarray:
    """Simple pitch shift without scipy (linear interpolation)."""
    if len(segment) == 0:
        return np.zeros(target_length, dtype=np.float64)
    
    ratio = 2.0 ** (semitones / 12.0)
    
    out = np.zeros(target_length, dtype=np.float64)
    src = segment.astype(np.float64)
    
    for i in range(target_length):
        src_pos = (i * ratio) % len(src)
        idx = int(src_pos)
        frac = src_pos - idx
        
        if idx + 1 < len(src):
            out[i] = src[idx] * (1 - frac) + src[idx + 1] * frac
        else:
            out[i] = src[idx]
    
    return out


# ============================================================================
# PATTERN ALGORITHMS
# ============================================================================

def algorithm_general(
    clip: PatternClip,
) -> np.ndarray:
    """Algorithm 0: General - Equal division with smooth crossfades."""
    if len(clip.source_buffer) == 0 or len(clip.notes) == 0:
        return np.zeros(1, dtype=np.float64)
    
    source = clip.source_buffer.astype(np.float64)
    source_len = len(source)
    
    # Calculate total output samples
    total_beats = sum(max(n.duration, 0.125) for n in clip.notes)
    total_samples = clip.beats_to_samples(total_beats)
    
    if total_samples <= 0:
        total_samples = source_len
    
    out = np.zeros(total_samples, dtype=np.float64)
    pos = 0
    last_semitone = 0.0
    last_velocity = 100.0
    
    for note in clip.notes:
        dur_beats = max(note.duration, 0.125)
        seg_samples = clip.beats_to_samples(dur_beats)
        
        if pos + seg_samples > total_samples:
            seg_samples = total_samples - pos
        
        if seg_samples <= 0:
            break
        
        # Get source segment (loop if needed)
        src_pos = pos % source_len
        if src_pos + seg_samples <= source_len:
            src_seg = source[src_pos:src_pos + seg_samples]
        else:
            src_seg = np.zeros(seg_samples, dtype=np.float64)
            remaining = seg_samples
            sp = src_pos
            op = 0
            while remaining > 0:
                chunk = min(remaining, source_len - sp)
                src_seg[op:op + chunk] = source[sp:sp + chunk]
                op += chunk
                remaining -= chunk
                sp = 0
        
        # Process based on note type
        if note.note_type == NoteType.REST:
            out[pos:pos + seg_samples] = 0.0
        elif note.note_type == NoteType.HOLD:
            shifted = pitch_shift_segment(src_seg, last_semitone, seg_samples)
            out[pos:pos + seg_samples] = shifted * (last_velocity / 100.0)
        elif note.note_type == NoteType.GLIDE:
            # Smooth glide
            for i in range(seg_samples):
                progress = i / seg_samples
                semi = last_semitone + (note.semitone - last_semitone) * progress
                ratio = 2.0 ** (semi / 12.0)
                src_idx = int((src_pos + i / ratio) % source_len)
                out[pos + i] = source[src_idx] * (note.velocity / 100.0)
            last_semitone = note.semitone
            last_velocity = note.velocity
        else:  # NoteType.PITCH
            shifted = pitch_shift_segment(src_seg, note.semitone, seg_samples)
            out[pos:pos + seg_samples] = shifted * (note.velocity / 100.0)
            last_semitone = note.semitone
            last_velocity = note.velocity
        
        pos += seg_samples
    
    return anti_artifact_process(out)


def algorithm_staccato(clip: PatternClip) -> np.ndarray:
    """Algorithm 1: Staccato - Short notes (70%) with silent gaps."""
    if len(clip.source_buffer) == 0 or len(clip.notes) == 0:
        return np.zeros(1, dtype=np.float64)
    
    source = clip.source_buffer.astype(np.float64)
    source_len = len(source)
    
    total_beats = sum(max(n.duration, 0.125) for n in clip.notes)
    total_samples = clip.beats_to_samples(total_beats)
    
    out = np.zeros(total_samples, dtype=np.float64)
    pos = 0
    
    for note in clip.notes:
        dur_beats = max(note.duration, 0.125)
        seg_samples = clip.beats_to_samples(dur_beats)
        
        if pos + seg_samples > total_samples:
            seg_samples = total_samples - pos
        
        if seg_samples <= 0:
            break
        
        # Staccato: only play 70% of duration
        play_samples = int(seg_samples * 0.7)
        
        if note.note_type == NoteType.REST or play_samples < MIN_SEGMENT_SAMPLES:
            out[pos:pos + seg_samples] = 0.0
        else:
            src_pos = pos % source_len
            src_seg = np.zeros(play_samples, dtype=np.float64)
            remaining = play_samples
            sp = src_pos
            op = 0
            while remaining > 0:
                chunk = min(remaining, source_len - sp)
                src_seg[op:op + chunk] = source[sp:sp + chunk]
                op += chunk
                remaining -= chunk
                sp = 0
            
            shifted = pitch_shift_segment(src_seg, note.semitone, play_samples)
            shifted = apply_attack_release(shifted, ATTACK_SAMPLES, RELEASE_SAMPLES * 2)
            out[pos:pos + play_samples] = shifted * (note.velocity / 100.0)
        
        pos += seg_samples
    
    return anti_artifact_process(out)


def algorithm_legato(clip: PatternClip) -> np.ndarray:
    """Algorithm 2: Legato - Overlapping notes for smooth flow."""
    if len(clip.source_buffer) == 0 or len(clip.notes) == 0:
        return np.zeros(1, dtype=np.float64)
    
    source = clip.source_buffer.astype(np.float64)
    source_len = len(source)
    
    total_beats = sum(max(n.duration, 0.125) for n in clip.notes)
    overlap_beats = 0.1  # 0.1 beat overlap
    total_samples = clip.beats_to_samples(total_beats)
    overlap_samples = clip.beats_to_samples(overlap_beats)
    
    out = np.zeros(total_samples, dtype=np.float64)
    pos = 0
    
    for note in clip.notes:
        dur_beats = max(note.duration, 0.125)
        seg_samples = clip.beats_to_samples(dur_beats + overlap_beats)
        
        end_pos = min(pos + seg_samples, total_samples)
        actual_samples = end_pos - pos
        
        if actual_samples <= 0:
            break
        
        if note.note_type == NoteType.REST:
            # Fade out existing content
            fade_len = min(overlap_samples, end_pos - pos)
            if fade_len > 0:
                fade = np.linspace(1, 0, fade_len)
                out[pos:pos + fade_len] *= fade
        else:
            src_pos = pos % source_len
            src_seg = np.zeros(actual_samples, dtype=np.float64)
            remaining = actual_samples
            sp = src_pos
            op = 0
            while remaining > 0:
                chunk = min(remaining, source_len - sp)
                src_seg[op:op + chunk] = source[sp:sp + chunk]
                op += chunk
                remaining -= chunk
                sp = 0
            
            shifted = pitch_shift_segment(src_seg, note.semitone, actual_samples)
            shifted *= (note.velocity / 100.0)
            
            # Crossfade with existing
            if pos > 0 and overlap_samples > 0:
                xf_len = min(overlap_samples, pos, actual_samples)
                fade_in = np.linspace(0, 1, xf_len)
                fade_out = np.linspace(1, 0, xf_len)
                out[pos:pos + xf_len] = out[pos:pos + xf_len] * fade_out + shifted[:xf_len] * fade_in
                out[pos + xf_len:end_pos] = shifted[xf_len:]
            else:
                out[pos:end_pos] = shifted
        
        pos = pos + clip.beats_to_samples(dur_beats)
    
    return anti_artifact_process(out)


def algorithm_audiorate(clip: PatternClip) -> np.ndarray:
    """Algorithm 3: Audio-rate - Granular repetition at source frequency.
    
    This algorithm duplicates each note at audio rate (at the detected or
    known fundamental frequency) throughout the note duration. Creates
    a granular/drone effect where the pitch is maintained through rapid
    repetition of a single cycle.
    """
    if len(clip.source_buffer) == 0 or len(clip.notes) == 0:
        return np.zeros(1, dtype=np.float64)
    
    source = clip.source_buffer.astype(np.float64)
    source_len = len(source)
    
    # Detect or use known frequency
    freq = clip.source_frequency
    if freq <= 0:
        freq = detect_fundamental_frequency(source, clip.sample_rate)
    if freq <= 0:
        freq = 100.0  # Default to 100Hz if detection fails
    
    # Period in samples
    period_samples = int(clip.sample_rate / freq)
    if period_samples < MIN_SEGMENT_SAMPLES:
        period_samples = MIN_SEGMENT_SAMPLES
    
    # Get one period of source (grain)
    grain = source[:min(period_samples, source_len)]
    grain = apply_attack_release(grain, period_samples // 8, period_samples // 8)
    
    total_beats = sum(max(n.duration, 0.125) for n in clip.notes)
    total_samples = clip.beats_to_samples(total_beats)
    
    out = np.zeros(total_samples, dtype=np.float64)
    pos = 0
    
    for note in clip.notes:
        dur_beats = max(note.duration, 0.125)
        seg_samples = clip.beats_to_samples(dur_beats)
        
        if pos + seg_samples > total_samples:
            seg_samples = total_samples - pos
        
        if seg_samples <= 0:
            break
        
        if note.note_type == NoteType.REST:
            out[pos:pos + seg_samples] = 0.0
        else:
            # Pitch-shift the grain
            ratio = 2.0 ** (note.semitone / 12.0)
            new_period = int(period_samples / ratio)
            if new_period < MIN_SEGMENT_SAMPLES:
                new_period = MIN_SEGMENT_SAMPLES
            
            try:
                import scipy.signal
                shifted_grain = scipy.signal.resample(grain, new_period)
            except ImportError:
                shifted_grain = _pitch_shift_simple(grain, note.semitone, new_period)
            
            shifted_grain = apply_attack_release(shifted_grain, new_period // 8, new_period // 8)
            
            # Repeat grain at audio rate
            gp = 0
            for sp in range(seg_samples):
                out[pos + sp] = shifted_grain[gp % len(shifted_grain)] * (note.velocity / 100.0)
                gp += 1
        
        pos += seg_samples
    
    return anti_artifact_process(out)


def algorithm_glide(clip: PatternClip) -> np.ndarray:
    """Algorithm 4: Glide - Smooth pitch transitions between all notes."""
    if len(clip.source_buffer) == 0 or len(clip.notes) == 0:
        return np.zeros(1, dtype=np.float64)
    
    source = clip.source_buffer.astype(np.float64)
    source_len = len(source)
    
    total_beats = sum(max(n.duration, 0.125) for n in clip.notes)
    total_samples = clip.beats_to_samples(total_beats)
    
    out = np.zeros(total_samples, dtype=np.float64)
    pos = 0
    last_semitone = 0.0
    
    for note in clip.notes:
        dur_beats = max(note.duration, 0.125)
        seg_samples = clip.beats_to_samples(dur_beats)
        
        if pos + seg_samples > total_samples:
            seg_samples = total_samples - pos
        
        if seg_samples <= 0:
            break
        
        target_semitone = note.semitone if note.note_type == NoteType.PITCH else last_semitone
        
        if note.note_type == NoteType.REST:
            # Glide to silence
            for i in range(seg_samples):
                progress = i / seg_samples
                amp = 1.0 - progress
                semi = last_semitone + (target_semitone - last_semitone) * progress
                ratio = 2.0 ** (semi / 12.0)
                src_idx = int((pos + i / ratio) % source_len)
                out[pos + i] = source[src_idx] * amp * (note.velocity / 100.0)
        else:
            # Smooth glide between pitches
            for i in range(seg_samples):
                progress = i / seg_samples
                semi = last_semitone + (target_semitone - last_semitone) * progress
                ratio = 2.0 ** (semi / 12.0)
                src_idx = int((pos + i / ratio) % source_len)
                out[pos + i] = source[src_idx] * (note.velocity / 100.0)
            last_semitone = target_semitone
        
        pos += seg_samples
    
    return anti_artifact_process(out)


def algorithm_stutter(clip: PatternClip) -> np.ndarray:
    """Algorithm 5: Stutter - Rhythmic repetition of each note."""
    if len(clip.source_buffer) == 0 or len(clip.notes) == 0:
        return np.zeros(1, dtype=np.float64)
    
    source = clip.source_buffer.astype(np.float64)
    source_len = len(source)
    
    total_beats = sum(max(n.duration, 0.125) for n in clip.notes)
    total_samples = clip.beats_to_samples(total_beats)
    
    out = np.zeros(total_samples, dtype=np.float64)
    pos = 0
    
    stutter_divisions = 4  # Repeat 4 times per note
    
    for note in clip.notes:
        dur_beats = max(note.duration, 0.125)
        seg_samples = clip.beats_to_samples(dur_beats)
        
        if pos + seg_samples > total_samples:
            seg_samples = total_samples - pos
        
        if seg_samples <= 0:
            break
        
        if note.note_type == NoteType.REST:
            out[pos:pos + seg_samples] = 0.0
        else:
            stutter_len = seg_samples // stutter_divisions
            if stutter_len < MIN_SEGMENT_SAMPLES:
                stutter_len = seg_samples
                stutter_divisions = 1
            
            src_pos = pos % source_len
            src_seg = np.zeros(stutter_len, dtype=np.float64)
            remaining = stutter_len
            sp = src_pos
            op = 0
            while remaining > 0:
                chunk = min(remaining, source_len - sp)
                src_seg[op:op + chunk] = source[sp:sp + chunk]
                op += chunk
                remaining -= chunk
                sp = 0
            
            shifted = pitch_shift_segment(src_seg, note.semitone, stutter_len)
            shifted = apply_attack_release(shifted, ATTACK_SAMPLES // 2, RELEASE_SAMPLES // 2)
            
            # Repeat stutter
            for i in range(stutter_divisions):
                start = pos + i * stutter_len
                end = min(start + stutter_len, pos + seg_samples, total_samples)
                copy_len = end - start
                if copy_len > 0:
                    out[start:end] = shifted[:copy_len] * (note.velocity / 100.0)
        
        pos += seg_samples
    
    return anti_artifact_process(out)


def algorithm_reverse(clip: PatternClip) -> np.ndarray:
    """Algorithm 6: Reverse - Play pattern notes in reverse order."""
    if len(clip.notes) == 0:
        return algorithm_general(clip)
    
    # Create clip with reversed notes
    reversed_clip = PatternClip(
        name=clip.name,
        source_buffer=clip.source_buffer,
        notes=list(reversed(clip.notes)),
        sample_rate=clip.sample_rate,
        bpm=clip.bpm,
        algorithm=0,  # Use general for the reversed order
        source_frequency=clip.source_frequency,
        mode=clip.mode,
    )
    return algorithm_general(reversed_clip)


def algorithm_bounce(clip: PatternClip) -> np.ndarray:
    """Algorithm 7: Bounce - Forward then backward (ping-pong)."""
    if len(clip.notes) == 0:
        return algorithm_general(clip)
    
    # Forward
    forward = algorithm_general(clip)
    
    # Backward (reverse the result)
    backward = np.flip(forward)
    
    # Crossfade join
    return crossfade_segments(forward, backward, XFADE_SAMPLES)


def algorithm_random(clip: PatternClip) -> np.ndarray:
    """Algorithm 8: Random - Randomize note order (deterministic seed)."""
    if len(clip.notes) == 0:
        return algorithm_general(clip)
    
    # Create deterministic seed from pattern
    seed = sum(int(n.semitone * 100 + n.duration * 10) for n in clip.notes)
    np.random.seed(seed % 2**31)
    
    indices = np.random.permutation(len(clip.notes))
    randomized_notes = [clip.notes[i] for i in indices]
    
    randomized_clip = PatternClip(
        name=clip.name,
        source_buffer=clip.source_buffer,
        notes=randomized_notes,
        sample_rate=clip.sample_rate,
        bpm=clip.bpm,
        algorithm=0,
        source_frequency=clip.source_frequency,
        mode=clip.mode,
    )
    return algorithm_general(randomized_clip)


def algorithm_humanize(clip: PatternClip) -> np.ndarray:
    """Algorithm 9: Humanize - Add timing and velocity variation."""
    if len(clip.notes) == 0:
        return algorithm_general(clip)
    
    # Create deterministic seed
    seed = sum(int(n.semitone * 100 + n.duration * 10) for n in clip.notes)
    np.random.seed(seed % 2**31)
    
    # Humanize notes
    humanized_notes = []
    for note in clip.notes:
        new_note = PatternNote(
            note_type=note.note_type,
            semitone=note.semitone,
            duration=note.duration * (1.0 + (np.random.random() - 0.5) * 0.1),  # ±5% timing
            velocity=note.velocity * (0.9 + np.random.random() * 0.2),  # ±10% velocity
        )
        humanized_notes.append(new_note)
    
    humanized_clip = PatternClip(
        name=clip.name,
        source_buffer=clip.source_buffer,
        notes=humanized_notes,
        sample_rate=clip.sample_rate,
        bpm=clip.bpm,
        algorithm=0,
        source_frequency=clip.source_frequency,
        mode=clip.mode,
    )
    return algorithm_general(humanized_clip)


# Algorithm dispatcher
ALGORITHM_FUNCTIONS: Dict[int, Callable] = {
    0: algorithm_general,
    1: algorithm_staccato,
    2: algorithm_legato,
    3: algorithm_audiorate,
    4: algorithm_glide,
    5: algorithm_stutter,
    6: algorithm_reverse,
    7: algorithm_bounce,
    8: algorithm_random,
    9: algorithm_humanize,
}


# ============================================================================
# MAIN RENDER FUNCTION
# ============================================================================

def render_pattern(
    clip: PatternClip,
    algorithm: Optional[int] = None,
    mix: float = 100.0,
) -> np.ndarray:
    """Render a pattern clip to audio buffer.
    
    Parameters
    ----------
    clip : PatternClip
        Pattern clip to render
    algorithm : int, optional
        Algorithm index (0-9), overrides clip.algorithm if specified
    mix : float
        Wet/dry mix (0-100, default 100 = fully wet)
        
    Returns
    -------
    np.ndarray
        Rendered audio buffer
    """
    if len(clip.source_buffer) == 0 or len(clip.notes) == 0:
        return np.zeros(1, dtype=np.float64)
    
    # Get algorithm
    alg_idx = algorithm if algorithm is not None else clip.algorithm
    alg_func = ALGORITHM_FUNCTIONS.get(alg_idx, algorithm_general)
    
    # Render
    out = alg_func(clip)
    
    # Apply mix
    mix_scaled = scale_wet(mix)
    if mix_scaled < 1.0:
        try:
            import scipy.signal
            dry = clip.source_buffer.astype(np.float64)
            if len(dry) != len(out):
                dry = scipy.signal.resample(dry, len(out))
            out = mix_scaled * out + (1.0 - mix_scaled) * dry
        except ImportError:
            pass
    
    # Final normalization
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak * 0.99
    
    # Cache
    clip._rendered_buffer = out
    
    return out


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_pattern(
    source: np.ndarray,
    pattern: Union[str, List[str], List[PatternNote]],
    bpm: float = DEFAULT_BPM,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    algorithm: int = 0,
    source_frequency: float = 0.0,
    mix: float = 100.0,
) -> np.ndarray:
    """Quick pattern application to audio buffer.
    
    Parameters
    ----------
    source : np.ndarray
        Source audio buffer
    pattern : str, List[str], or List[PatternNote]
        Pattern specification
    bpm : float
        Tempo
    sample_rate : int
        Sample rate
    algorithm : int
        Algorithm index (0-9)
    source_frequency : float
        Known source frequency (0 = auto-detect)
    mix : float
        Wet/dry mix (0-100)
        
    Returns
    -------
    np.ndarray
        Processed audio
    """
    # Parse pattern if needed
    if isinstance(pattern, str):
        adaptive_length = len(source) / sample_rate
        notes = parse_pattern(pattern, adaptive_length, sample_rate, bpm)
    elif isinstance(pattern, list) and len(pattern) > 0:
        if isinstance(pattern[0], str):
            adaptive_length = len(source) / sample_rate
            notes = parse_pattern(pattern, adaptive_length, sample_rate, bpm)
        else:
            notes = pattern
    else:
        notes = []
    
    if not notes:
        return source.copy()
    
    # Create clip
    clip = PatternClip(
        name="quick_pattern",
        source_buffer=source,
        notes=notes,
        sample_rate=sample_rate,
        bpm=bpm,
        algorithm=algorithm,
        source_frequency=source_frequency,
    )
    
    return render_pattern(clip, algorithm, mix)


def arpeggiate(
    source: np.ndarray,
    chord: List[int],
    repeats: int = 2,
    bpm: float = DEFAULT_BPM,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    algorithm: int = 0,
) -> np.ndarray:
    """Create arpeggio from source audio and chord."""
    pattern_tokens = []
    for _ in range(repeats):
        for semitone in chord:
            pattern_tokens.append(str(semitone))
    
    return quick_pattern(source, pattern_tokens, bpm, sample_rate, algorithm)


def list_algorithms() -> List[Dict[str, Any]]:
    """List all available pattern algorithms."""
    return [
        {"index": i, **info}
        for i, info in sorted(ALGORITHM_INFO.items())
    ]


def get_algorithm_info(index: int) -> Optional[Dict[str, str]]:
    """Get info for a specific algorithm."""
    return ALGORITHM_INFO.get(index)


def format_algorithm_list() -> str:
    """Format algorithm list for display (PAG command)."""
    lines = ["Pattern Algorithms:"]
    lines.append("-" * 50)
    for idx in sorted(ALGORITHM_INFO.keys()):
        info = ALGORITHM_INFO[idx]
        lines.append(f"  {idx}: {info['name']:<12} - {info['desc']}")
    lines.append("-" * 50)
    lines.append("Usage: /pat <tokens> /end <algorithm_index>")
    lines.append("       /pat 0 7 12 /end 3   (audio-rate mode)")
    return '\n'.join(lines)


# ============================================================================
# PATTERN PRESETS
# ============================================================================

PATTERN_PRESETS = {
    # Major scale patterns
    'major_up': '0 2 4 5 7 9 11 12',
    'major_down': '12 11 9 7 5 4 2 0',
    'major_arp': '0 4 7 12 7 4',
    
    # Minor scale patterns
    'minor_up': '0 2 3 5 7 8 10 12',
    'minor_down': '12 10 8 7 5 3 2 0',
    'minor_arp': '0 3 7 12 7 3',
    
    # Chord arpeggios
    'maj7_arp': '0 4 7 11 12 11 7 4',
    'min7_arp': '0 3 7 10 12 10 7 3',
    'dom7_arp': '0 4 7 10 12 10 7 4',
    
    # Rhythmic patterns with rests
    'pulse': '0 R 0 R 0 R 0 R',
    'offbeat': 'R 0 R 0 R 0 R 0',
    'dotted': '0 R R 0 R R 0 R',
    
    # Melodic phrases
    'bounce': '0 7 0 7 0 5 0 5',
    'rise': '0 D 2 D 4 D 7 D',
    'fall': '12 D 10 D 7 D 5 D',
    
    # Experimental
    'octaves': '0 12 0 -12 0 12 0 -12',
    'fifths': '0 7 0 7 -5 0 -5 0',
    'chromatic': '0 1 2 3 4 5 6 7',
}


def get_preset_pattern(name: str) -> List[PatternNote]:
    """Get a preset pattern by name."""
    pattern_str = PATTERN_PRESETS.get(name.lower(), '')
    if pattern_str:
        return parse_pattern(pattern_str)
    return []


def list_pattern_presets() -> List[str]:
    """Get list of available preset names."""
    return sorted(PATTERN_PRESETS.keys())


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Classes
    'PatternNote',
    'PatternClip',
    'PlaybackMode',
    'NoteType',
    'PatternAlgorithm',
    # Parsing
    'parse_pattern_token',
    'parse_pattern',
    'pattern_to_string',
    # Rendering
    'render_pattern',
    'pitch_shift_segment',
    # Anti-artifact
    'anti_artifact_process',
    'apply_attack_release',
    'crossfade_segments',
    # Frequency detection
    'detect_fundamental_frequency',
    # Algorithms
    'ALGORITHM_INFO',
    'ALGORITHM_FUNCTIONS',
    'list_algorithms',
    'get_algorithm_info',
    'format_algorithm_list',
    # Presets
    'PATTERN_PRESETS',
    'get_preset_pattern',
    'list_pattern_presets',
    # Convenience
    'quick_pattern',
    'arpeggiate',
]
