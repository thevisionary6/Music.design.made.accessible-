"""Musical transformation engines — Phase 4.5.

Musically-aware transformations that produce usable results:
retrograde, inversion, augmentation, diminution, motivic development.

Operates at two levels:
1. **Note-level** — transforms sequences of MIDI/semitone note data
2. **Audio-level** — transforms rendered audio buffers while preserving
   musical intent (using pitch detection + resynthesis where needed)
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import music_theory as mt


# ======================================================================
# Note-level transforms  (operates on List[int] of MIDI notes)
# ======================================================================

def retrograde(notes: List[int]) -> List[int]:
    """Reverse the note order (musical retrograde)."""
    return list(reversed(notes))


def inversion(notes: List[int], axis: Optional[int] = None) -> List[int]:
    """Mirror notes around *axis* (default: first note).

    Each interval above the axis becomes the same interval below, and
    vice versa.
    """
    if not notes:
        return []
    if axis is None:
        axis = notes[0]
    return [2 * axis - n for n in notes]


def retrograde_inversion(notes: List[int],
                          axis: Optional[int] = None) -> List[int]:
    """Retrograde + inversion combined."""
    return retrograde(inversion(notes, axis))


def augmentation(notes: List[int], factor: float = 2.0) -> List[int]:
    """Stretch intervals by *factor* (augmentation = wider intervals).

    Preserves the first note and scales all subsequent intervals.
    """
    if len(notes) < 2:
        return list(notes)
    root = notes[0]
    return [root] + [root + int(round((n - root) * factor))
                     for n in notes[1:]]


def diminution(notes: List[int], factor: float = 0.5) -> List[int]:
    """Compress intervals by *factor* (diminution = narrower intervals)."""
    return augmentation(notes, factor)


def transpose(notes: List[int], semitones: int) -> List[int]:
    """Transpose all notes by *semitones*."""
    return [n + semitones for n in notes]


def octave_displacement(notes: List[int],
                         probability: float = 0.3,
                         seed: Optional[int] = None) -> List[int]:
    """Randomly displace notes by +/- 1 octave."""
    rng = random.Random(seed)
    result = []
    for n in notes:
        if rng.random() < probability:
            result.append(n + rng.choice([-12, 12]))
        else:
            result.append(n)
    return result


def permute(notes: List[int], mode: str = 'rotate',
            amount: int = 1,
            seed: Optional[int] = None) -> List[int]:
    """Permute note order.

    *mode*: 'rotate' (circular shift), 'shuffle' (random), 'swap'
    (swap adjacent pairs).
    """
    if not notes:
        return []
    result = list(notes)
    if mode == 'rotate':
        amount = amount % len(result)
        result = result[amount:] + result[:amount]
    elif mode == 'shuffle':
        rng = random.Random(seed)
        rng.shuffle(result)
    elif mode == 'swap':
        for i in range(0, len(result) - 1, 2):
            result[i], result[i + 1] = result[i + 1], result[i]
    return result


def motivic_development(notes: List[int],
                         techniques: Optional[List[str]] = None,
                         seed: Optional[int] = None) -> List[int]:
    """Apply a chain of compositional techniques to develop a motif.

    Available techniques: 'retrograde', 'inversion', 'augmentation',
    'diminution', 'transpose_up', 'transpose_down', 'octave_displace',
    'rotate', 'shuffle'.
    """
    rng = random.Random(seed)
    if techniques is None:
        techniques = rng.sample([
            'retrograde', 'inversion', 'augmentation',
            'transpose_up', 'rotate',
        ], k=min(3, rng.randint(1, 3)))

    result = list(notes)
    for tech in techniques:
        if tech == 'retrograde':
            result = retrograde(result)
        elif tech == 'inversion':
            result = inversion(result)
        elif tech == 'augmentation':
            result = augmentation(result, 1.5)
        elif tech == 'diminution':
            result = diminution(result, 0.75)
        elif tech == 'transpose_up':
            result = transpose(result, rng.choice([2, 3, 4, 5, 7]))
        elif tech == 'transpose_down':
            result = transpose(result, -rng.choice([2, 3, 4, 5, 7]))
        elif tech == 'octave_displace':
            result = octave_displacement(result, 0.3, seed)
        elif tech == 'rotate':
            result = permute(result, 'rotate', rng.randint(1, max(1, len(result) - 1)))
        elif tech == 'shuffle':
            result = permute(result, 'shuffle', seed=seed)
    return result


def constrain_to_scale(notes: List[int], root_midi: int,
                        scale_name: str) -> List[int]:
    """Snap all notes to the nearest scale tone."""
    return [mt.snap_to_scale(n, root_midi, scale_name) for n in notes]


# ======================================================================
# Audio-level transforms
# ======================================================================

def audio_retrograde(buf: np.ndarray) -> np.ndarray:
    """Reverse audio buffer."""
    return np.flip(buf).copy()


def audio_inversion(buf: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Spectral inversion — flip frequency content around a center.

    Simple approximation: negate alternate samples (high-pass-like) then
    blend with original.
    """
    inv = buf.copy()
    inv[1::2] *= -1
    # Blend 50/50 for a useful result
    return (buf * 0.5 + inv * 0.5)


def audio_augmentation(buf: np.ndarray, factor: float = 2.0,
                        sr: int = 48000) -> np.ndarray:
    """Time-stretch by *factor* (>1 = slower, <1 = faster)."""
    try:
        from scipy.signal import resample
        new_len = int(len(buf) * factor)
        return resample(buf, new_len)
    except ImportError:
        # Linear interpolation fallback
        old_len = len(buf)
        new_len = int(old_len * factor)
        if new_len <= 0:
            return np.zeros(1)
        indices = np.linspace(0, old_len - 1, new_len)
        return np.interp(indices, np.arange(old_len), buf)


def audio_diminution(buf: np.ndarray, factor: float = 0.5,
                      sr: int = 48000) -> np.ndarray:
    """Time-compress by *factor*."""
    return audio_augmentation(buf, factor, sr)


def audio_pitch_shift(buf: np.ndarray, semitones: float,
                       sr: int = 48000) -> np.ndarray:
    """Pitch-shift audio by *semitones* while preserving duration."""
    ratio = 2.0 ** (semitones / 12.0)
    try:
        from scipy.signal import resample
        # Resample to new pitch, then stretch back to original length
        stretched = resample(buf, int(len(buf) / ratio))
        return resample(stretched, len(buf))
    except ImportError:
        # Simple resample fallback (changes duration)
        new_len = int(len(buf) / ratio)
        indices = np.linspace(0, len(buf) - 1, new_len)
        return np.interp(indices, np.arange(len(buf)), buf)


def audio_stutter(buf: np.ndarray, repeats: int = 4,
                   grain_beats: float = 0.25,
                   bpm: float = 128.0,
                   sr: int = 48000) -> np.ndarray:
    """Stutter effect — repeat a grain *repeats* times."""
    grain_samples = int(sr * 60.0 / bpm * grain_beats)
    grain = buf[:grain_samples].copy()
    # Fade edges
    fade = min(128, len(grain) // 4)
    if fade > 0:
        grain[:fade] *= np.linspace(0, 1, fade)
        grain[-fade:] *= np.linspace(1, 0, fade)
    return np.tile(grain, repeats)


def audio_chop_and_rearrange(buf: np.ndarray, n_slices: int = 8,
                              order: Optional[List[int]] = None,
                              seed: Optional[int] = None) -> np.ndarray:
    """Chop audio into *n_slices* and rearrange."""
    rng = random.Random(seed)
    slice_len = len(buf) // n_slices
    if slice_len <= 0:
        return buf.copy()
    slices = [buf[i * slice_len:(i + 1) * slice_len] for i in range(n_slices)]
    if order is None:
        order = list(range(n_slices))
        rng.shuffle(order)
    result = np.concatenate([slices[i % len(slices)] for i in order])
    # Smooth slice boundaries
    fade = min(64, slice_len // 8)
    if fade > 1:
        for i in range(1, len(order)):
            pos = i * slice_len
            if pos + fade < len(result) and pos - fade >= 0:
                xf = np.linspace(0, 1, fade)
                result[pos:pos + fade] *= xf
    return result


def audio_granular_freeze(buf: np.ndarray, position: float = 0.5,
                           grain_size: int = 2048,
                           n_grains: int = 50,
                           sr: int = 48000,
                           seed: Optional[int] = None) -> np.ndarray:
    """Freeze at a position — extract grains and overlay with scatter."""
    rng = random.Random(seed)
    center = int(position * len(buf))
    spread = grain_size * 4
    out_len = n_grains * grain_size // 2
    out = np.zeros(out_len)
    window = np.hanning(grain_size)
    for i in range(n_grains):
        src = center + rng.randint(-spread, spread)
        src = max(0, min(len(buf) - grain_size, src))
        grain = buf[src:src + grain_size] * window
        dst = int(i * grain_size * 0.5)
        end = min(dst + grain_size, out_len)
        n = end - dst
        if n > 0:
            out[dst:end] += grain[:n]
    peak = np.max(np.abs(out))
    if peak > 1e-10:
        out *= 0.9 / peak
    return out


# ======================================================================
# Compound transforms (apply multiple in sequence)
# ======================================================================

TRANSFORM_PRESETS: Dict[str, List[Tuple[str, dict]]] = {
    'develop': [
        ('retrograde', {}),
        ('constrain', {'scale': 'minor'}),
    ],
    'evolve': [
        ('augmentation', {'factor': 1.5}),
        ('octave_displace', {'probability': 0.2}),
        ('constrain', {'scale': 'major'}),
    ],
    'deconstruct': [
        ('shuffle', {}),
        ('diminution', {'factor': 0.75}),
    ],
    'mirror': [
        ('inversion', {}),
        ('constrain', {'scale': 'minor'}),
    ],
    'expand': [
        ('augmentation', {'factor': 2.0}),
        ('transpose', {'semitones': 7}),
    ],
}

AUDIO_TRANSFORM_PRESETS: Dict[str, List[Tuple[str, dict]]] = {
    'reverse': [('retrograde', {})],
    'halftime': [('augmentation', {'factor': 2.0})],
    'doubletime': [('diminution', {'factor': 0.5})],
    'stutter': [('stutter', {'repeats': 8, 'grain_beats': 0.125})],
    'glitch': [('chop_rearrange', {'n_slices': 16})],
    'freeze': [('granular_freeze', {'position': 0.5})],
    'pitch_up': [('pitch_shift', {'semitones': 7})],
    'pitch_down': [('pitch_shift', {'semitones': -5})],
    'octave_up': [('pitch_shift', {'semitones': 12})],
    'octave_down': [('pitch_shift', {'semitones': -12})],
}


def apply_note_transforms(notes: List[int],
                           transforms: List[Tuple[str, dict]],
                           root_midi: int = 60,
                           scale_name: str = 'major') -> List[int]:
    """Apply a sequence of named transforms to a note list."""
    result = list(notes)
    for name, params in transforms:
        if name == 'retrograde':
            result = retrograde(result)
        elif name == 'inversion':
            result = inversion(result, params.get('axis'))
        elif name == 'augmentation':
            result = augmentation(result, params.get('factor', 2.0))
        elif name == 'diminution':
            result = diminution(result, params.get('factor', 0.5))
        elif name == 'transpose':
            result = transpose(result, params.get('semitones', 0))
        elif name == 'octave_displace':
            result = octave_displacement(result, params.get('probability', 0.3),
                                          params.get('seed'))
        elif name == 'rotate':
            result = permute(result, 'rotate', params.get('amount', 1))
        elif name == 'shuffle':
            result = permute(result, 'shuffle', seed=params.get('seed'))
        elif name == 'constrain':
            result = constrain_to_scale(result, root_midi,
                                         params.get('scale', scale_name))
    return result


def apply_audio_transforms(buf: np.ndarray,
                            transforms: List[Tuple[str, dict]],
                            sr: int = 48000,
                            bpm: float = 128.0) -> np.ndarray:
    """Apply a sequence of named audio transforms to a buffer."""
    result = buf.copy()
    for name, params in transforms:
        if name == 'retrograde':
            result = audio_retrograde(result)
        elif name == 'inversion':
            result = audio_inversion(result, sr)
        elif name == 'augmentation':
            result = audio_augmentation(result, params.get('factor', 2.0), sr)
        elif name == 'diminution':
            result = audio_diminution(result, params.get('factor', 0.5), sr)
        elif name == 'pitch_shift':
            result = audio_pitch_shift(result, params.get('semitones', 0), sr)
        elif name == 'stutter':
            result = audio_stutter(result, params.get('repeats', 4),
                                    params.get('grain_beats', 0.25), bpm, sr)
        elif name == 'chop_rearrange':
            result = audio_chop_and_rearrange(
                result, params.get('n_slices', 8),
                params.get('order'), params.get('seed'))
        elif name == 'granular_freeze':
            result = audio_granular_freeze(
                result, params.get('position', 0.5),
                params.get('grain_size', 2048),
                params.get('n_grains', 50), sr, params.get('seed'))
    return result


def list_note_transforms() -> List[str]:
    return ['retrograde', 'inversion', 'retrograde_inversion',
            'augmentation', 'diminution', 'transpose',
            'octave_displace', 'rotate', 'shuffle', 'constrain',
            'motivic_development']


def list_audio_transforms() -> List[str]:
    return ['retrograde', 'inversion', 'augmentation', 'diminution',
            'pitch_shift', 'stutter', 'chop_rearrange',
            'granular_freeze']


def list_transform_presets() -> Dict[str, List[str]]:
    return {
        'note': sorted(TRANSFORM_PRESETS.keys()),
        'audio': sorted(AUDIO_TRANSFORM_PRESETS.keys()),
    }
