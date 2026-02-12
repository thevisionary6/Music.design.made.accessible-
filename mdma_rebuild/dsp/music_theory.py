"""Music theory utilities — scales, chords, intervals, progressions, voice leading.

Provides the harmonic/melodic intelligence layer that Phase 4 generative
systems build upon.  All pitch representations are **semitone offsets**
relative to an arbitrary root (0 = root) unless stated otherwise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ======================================================================
# Intervals (semitone distances)
# ======================================================================

INTERVALS: Dict[str, int] = {
    'unison': 0, 'min2': 1, 'maj2': 2, 'min3': 3, 'maj3': 4,
    'perf4': 5, 'tritone': 6, 'perf5': 7, 'min6': 8, 'maj6': 9,
    'min7': 10, 'maj7': 11, 'octave': 12,
}

# ======================================================================
# Scales — every entry is a tuple of semitone offsets from root
# ======================================================================

SCALES: Dict[str, Tuple[int, ...]] = {
    # Diatonic modes
    'major':            (0, 2, 4, 5, 7, 9, 11),
    'ionian':           (0, 2, 4, 5, 7, 9, 11),
    'dorian':           (0, 2, 3, 5, 7, 9, 10),
    'phrygian':         (0, 1, 3, 5, 7, 8, 10),
    'lydian':           (0, 2, 4, 6, 7, 9, 11),
    'mixolydian':       (0, 2, 4, 5, 7, 9, 10),
    'minor':            (0, 2, 3, 5, 7, 8, 10),
    'aeolian':          (0, 2, 3, 5, 7, 8, 10),
    'locrian':          (0, 1, 3, 5, 6, 8, 10),
    # Melodic / harmonic minor
    'harmonic_minor':   (0, 2, 3, 5, 7, 8, 11),
    'melodic_minor':    (0, 2, 3, 5, 7, 9, 11),
    # Pentatonic
    'pentatonic_major': (0, 2, 4, 7, 9),
    'pentatonic_minor': (0, 3, 5, 7, 10),
    # Blues
    'blues':            (0, 3, 5, 6, 7, 10),
    'blues_major':      (0, 2, 3, 4, 7, 9),
    # Other
    'whole_tone':       (0, 2, 4, 6, 8, 10),
    'diminished':       (0, 2, 3, 5, 6, 8, 9, 11),
    'chromatic':        tuple(range(12)),
    'japanese':         (0, 1, 5, 7, 8),
    'arabic':           (0, 1, 4, 5, 7, 8, 11),
    'hungarian_minor':  (0, 2, 3, 6, 7, 8, 11),
}

# ======================================================================
# Chords — each is a tuple of semitone offsets from root
# ======================================================================

CHORDS: Dict[str, Tuple[int, ...]] = {
    # Triads
    'maj':   (0, 4, 7),
    'min':   (0, 3, 7),
    'dim':   (0, 3, 6),
    'aug':   (0, 4, 8),
    'sus2':  (0, 2, 7),
    'sus4':  (0, 5, 7),
    # Sevenths
    'maj7':  (0, 4, 7, 11),
    'min7':  (0, 3, 7, 10),
    'dom7':  (0, 4, 7, 10),
    'dim7':  (0, 3, 6, 9),
    'hdim7': (0, 3, 6, 10),
    'm_maj7': (0, 3, 7, 11),
    'aug7':  (0, 4, 8, 10),
    # Extended
    'add9':  (0, 4, 7, 14),
    'min9':  (0, 3, 7, 10, 14),
    'maj9':  (0, 4, 7, 11, 14),
    'dom9':  (0, 4, 7, 10, 14),
    '6':     (0, 4, 7, 9),
    'm6':    (0, 3, 7, 9),
    # Power
    '5':     (0, 7),
    'oct':   (0, 12),
}

# ======================================================================
# Note name ↔ MIDI helpers
# ======================================================================

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

_NOTE_MAP = {
    'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
    'Cb': 11, 'Db': 1, 'Eb': 3, 'Fb': 4, 'Gb': 6, 'Ab': 8, 'Bb': 10,
}


def note_name_to_midi(name: str) -> int:
    """Convert e.g. 'C4', 'A#3', 'Bb5' to MIDI note number."""
    name = name.strip()
    if len(name) < 2:
        raise ValueError(f"Invalid note: {name!r}")
    if name[1] in ('#', 'b') and len(name) > 2:
        letter_part = name[:2]
        octave = int(name[2:])
    else:
        letter_part = name[0]
        octave = int(name[1:])
    base = _NOTE_MAP.get(letter_part)
    if base is None:
        raise ValueError(f"Unknown note letter: {letter_part!r}")
    return base + (octave + 1) * 12


def midi_to_note_name(midi: int) -> str:
    """Convert MIDI note number to name, e.g. 60 → 'C4'."""
    octave = (midi // 12) - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


def midi_to_freq(midi: int) -> float:
    """MIDI note → frequency in Hz (A4 = 440)."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def freq_to_midi(freq: float) -> int:
    """Frequency → nearest MIDI note number."""
    import math
    if freq <= 0:
        return 0
    return round(69 + 12 * math.log2(freq / 440.0))


# ======================================================================
# Scale operations
# ======================================================================

def get_scale(root_midi: int, scale_name: str,
              octaves: int = 1) -> List[int]:
    """Return MIDI notes for *scale_name* starting at *root_midi*.

    >>> get_scale(60, 'major')
    [60, 62, 64, 65, 67, 69, 71]
    """
    intervals = SCALES.get(scale_name.lower())
    if intervals is None:
        raise ValueError(f"Unknown scale: {scale_name!r}. "
                         f"Available: {', '.join(sorted(SCALES))}")
    notes = []
    for oct in range(octaves):
        for iv in intervals:
            notes.append(root_midi + iv + oct * 12)
    return notes


def snap_to_scale(midi: int, root_midi: int, scale_name: str) -> int:
    """Snap a MIDI note to the nearest note in *scale_name*."""
    intervals = SCALES.get(scale_name.lower(), SCALES['chromatic'])
    pc_set = set(intervals)
    pc = (midi - root_midi) % 12
    if pc in pc_set:
        return midi
    # Find nearest pitch class in scale
    best = min(pc_set, key=lambda s: min(abs(s - pc), abs(s - pc + 12),
                                          abs(s - pc - 12)))
    delta = best - pc
    if delta > 6:
        delta -= 12
    elif delta < -6:
        delta += 12
    return midi + delta


def transpose_to_scale(notes: Sequence[int], from_root: int,
                       from_scale: str, to_root: int,
                       to_scale: str) -> List[int]:
    """Transpose a sequence of MIDI notes from one key/scale to another.

    Preserves scale-degree relationships where possible.
    """
    src = SCALES.get(from_scale.lower(), SCALES['major'])
    dst = SCALES.get(to_scale.lower(), SCALES['major'])
    result = []
    for note in notes:
        # Find scale degree in source
        pc = (note - from_root) % 12
        octave_offset = (note - from_root) // 12
        # Find closest scale degree
        best_deg = min(range(len(src)),
                       key=lambda d: min(abs(src[d] - pc),
                                          abs(src[d] - pc + 12)))
        # Map to destination scale degree
        dst_deg = best_deg % len(dst)
        dst_oct = best_deg // len(dst)
        new_note = to_root + dst[dst_deg] + (octave_offset + dst_oct) * 12
        result.append(new_note)
    return result


# ======================================================================
# Chord operations
# ======================================================================

def get_chord(root_midi: int, chord_name: str) -> List[int]:
    """Return MIDI notes for *chord_name* rooted at *root_midi*."""
    intervals = CHORDS.get(chord_name.lower())
    if intervals is None:
        raise ValueError(f"Unknown chord: {chord_name!r}. "
                         f"Available: {', '.join(sorted(CHORDS))}")
    return [root_midi + iv for iv in intervals]


def diatonic_chords(root_midi: int, scale_name: str,
                    seventh: bool = False) -> List[Tuple[int, str, List[int]]]:
    """Return diatonic chords for *scale_name*.

    Returns list of (degree_root_midi, chord_quality, midi_notes).
    """
    intervals = SCALES.get(scale_name.lower(), SCALES['major'])
    results = []
    for i, iv in enumerate(intervals):
        chord_root = root_midi + iv
        # Stack thirds from scale
        notes = [chord_root]
        deg = i
        for _ in range(2 if not seventh else 3):
            deg = (deg + 2) % len(intervals)
            octave_add = 0
            if intervals[deg] <= intervals[i]:
                octave_add = 12
            note = root_midi + intervals[deg] + octave_add
            while note <= notes[-1]:
                note += 12
            notes.append(note)
        # Determine quality from intervals
        ivs = tuple(n - notes[0] for n in notes)
        quality = _identify_chord_quality(ivs)
        results.append((chord_root, quality, notes))
    return results


def _identify_chord_quality(intervals: Tuple[int, ...]) -> str:
    """Identify chord quality from interval tuple."""
    for name, ref in CHORDS.items():
        if intervals[:len(ref)] == ref:
            return name
    # Fallback heuristic
    if len(intervals) >= 2:
        third = intervals[1] if len(intervals) > 1 else 0
        fifth = intervals[2] if len(intervals) > 2 else 0
        if third == 4 and fifth == 7:
            return 'maj'
        if third == 3 and fifth == 7:
            return 'min'
        if third == 3 and fifth == 6:
            return 'dim'
        if third == 4 and fifth == 8:
            return 'aug'
    return 'maj'


# ======================================================================
# Chord progressions
# ======================================================================

# Roman numeral → scale degree index (0-based)
NUMERAL_MAP = {'I': 0, 'II': 1, 'III': 2, 'IV': 3,
               'V': 4, 'VI': 5, 'VII': 6}

# Common progressions as (degree, quality) tuples
PROGRESSIONS: Dict[str, List[Tuple[int, str]]] = {
    'I_IV_V':        [(0, 'maj'), (3, 'maj'), (4, 'maj')],
    'I_V_vi_IV':     [(0, 'maj'), (4, 'maj'), (5, 'min'), (3, 'maj')],
    'ii_V_I':        [(1, 'min7'), (4, 'dom7'), (0, 'maj7')],
    'I_vi_IV_V':     [(0, 'maj'), (5, 'min'), (3, 'maj'), (4, 'maj')],
    'vi_IV_I_V':     [(5, 'min'), (3, 'maj'), (0, 'maj'), (4, 'maj')],
    'I_IV_vi_V':     [(0, 'maj'), (3, 'maj'), (5, 'min'), (4, 'maj')],
    'i_iv_v':        [(0, 'min'), (3, 'min'), (4, 'min')],
    'i_VI_III_VII':  [(0, 'min'), (5, 'maj'), (2, 'maj'), (6, 'maj')],
    'i_iv_VII_III':  [(0, 'min'), (3, 'min'), (6, 'maj'), (2, 'maj')],
    'i_VII_VI_V':    [(0, 'min'), (6, 'maj'), (5, 'maj'), (4, 'maj')],
    '12bar':         [(0, 'dom7'), (0, 'dom7'), (0, 'dom7'), (0, 'dom7'),
                      (3, 'dom7'), (3, 'dom7'), (0, 'dom7'), (0, 'dom7'),
                      (4, 'dom7'), (3, 'dom7'), (0, 'dom7'), (4, 'dom7')],
}


def resolve_progression(root_midi: int, scale_name: str,
                        prog_name: str) -> List[List[int]]:
    """Resolve a named progression to MIDI chord lists."""
    prog = PROGRESSIONS.get(prog_name)
    if prog is None:
        raise ValueError(f"Unknown progression: {prog_name!r}. "
                         f"Available: {', '.join(sorted(PROGRESSIONS))}")
    scale_ivs = SCALES.get(scale_name.lower(), SCALES['major'])
    chords = []
    for deg, quality in prog:
        chord_root = root_midi + scale_ivs[deg % len(scale_ivs)]
        chords.append(get_chord(chord_root, quality))
    return chords


# ======================================================================
# Voice leading
# ======================================================================

def voice_lead(current_chord: List[int],
               next_chord_pitches: List[int]) -> List[int]:
    """Find the voicing of *next_chord_pitches* closest to *current_chord*.

    Minimises total semitone movement across voices.
    """
    if not current_chord or not next_chord_pitches:
        return list(next_chord_pitches)

    # Generate candidate octave positions for each target pitch class
    target_pcs = [p % 12 for p in next_chord_pitches]
    center = sum(current_chord) / len(current_chord)

    candidates_per_voice: List[List[int]] = []
    for pc in target_pcs:
        options = []
        for oct in range(-1, 9):
            note = pc + oct * 12
            if abs(note - center) < 24:
                options.append(note)
        if not options:
            options = [pc + int(center // 12) * 12]
        candidates_per_voice.append(options)

    # Greedy assignment: for each target voice, pick closest to
    # corresponding current voice (or nearest current voice if sizes differ)
    result = []
    for i, options in enumerate(candidates_per_voice):
        ref = current_chord[i] if i < len(current_chord) else center
        best = min(options, key=lambda n: abs(n - ref))
        result.append(best)
    return sorted(result)


# ======================================================================
# Melodic utilities
# ======================================================================

def generate_melody_from_scale(root_midi: int, scale_name: str,
                               length: int = 8,
                               octave_range: int = 1,
                               contour: str = 'arch',
                               seed: Optional[int] = None) -> List[int]:
    """Generate a scale-aware melody with a given contour shape.

    Contours: 'arch', 'descending', 'ascending', 'wave', 'random'.
    """
    rng = random.Random(seed)
    pool = get_scale(root_midi, scale_name, octaves=octave_range + 1)
    if not pool:
        return [root_midi] * length

    mid = len(pool) // 2
    melody: List[int] = []

    for i in range(length):
        t = i / max(1, length - 1)  # 0→1
        if contour == 'arch':
            target_idx = int(mid + mid * (1 - abs(2 * t - 1)))
        elif contour == 'ascending':
            target_idx = int(t * (len(pool) - 1))
        elif contour == 'descending':
            target_idx = int((1 - t) * (len(pool) - 1))
        elif contour == 'wave':
            import math
            target_idx = int(mid + mid * 0.7 * math.sin(2 * math.pi * t))
        else:  # random
            target_idx = rng.randint(0, len(pool) - 1)

        # Add small random variation
        target_idx = max(0, min(len(pool) - 1,
                                target_idx + rng.randint(-1, 1)))
        melody.append(pool[target_idx])

    return melody


def detect_key(midi_notes: Sequence[int]) -> Tuple[int, str, float]:
    """Simple key detection via pitch-class histogram correlation.

    Returns (root_midi, scale_name, confidence 0-1).
    """
    if not midi_notes:
        return (60, 'major', 0.0)

    # Build pitch-class histogram
    histogram = [0] * 12
    for n in midi_notes:
        histogram[n % 12] += 1

    best_root = 0
    best_scale = 'major'
    best_score = -1.0

    for root in range(12):
        for sname, intervals in [('major', SCALES['major']),
                                  ('minor', SCALES['minor']),
                                  ('pentatonic_major', SCALES['pentatonic_major']),
                                  ('pentatonic_minor', SCALES['pentatonic_minor']),
                                  ('blues', SCALES['blues'])]:
            score = 0.0
            total = sum(histogram) or 1
            for iv in intervals:
                score += histogram[(root + iv) % 12]
            score /= total
            if score > best_score:
                best_score = score
                best_root = root
                best_scale = sname

    # Find closest octave to the note centroid
    avg = sum(midi_notes) / len(midi_notes)
    root_midi = best_root + round((avg - best_root) / 12) * 12
    return (root_midi, best_scale, best_score)


# ======================================================================
# Public helpers for listing
# ======================================================================

def list_scales() -> List[str]:
    return sorted(SCALES.keys())

def list_chords() -> List[str]:
    return sorted(CHORDS.keys())

def list_progressions() -> List[str]:
    return sorted(PROGRESSIONS.keys())
