"""Loop generation engine — Phase 4.3.

End-to-end loop generator: specify genre, tempo, key, mood → receive a
rendered, mixable loop.  Combines the beat generator, music theory module,
and synth engine into a single workflow.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import music_theory as mt
from . import beat_gen


# ======================================================================
# Loop spec
# ======================================================================

@dataclass
class LoopSpec:
    """Declarative description of a loop to generate."""
    genre: str = 'house'
    bpm: float = 0.0            # 0 = use genre default
    bars: int = 4
    root_note: int = 60         # MIDI root (60 = C4)
    scale: str = 'minor'
    mood: str = 'neutral'       # neutral, bright, dark, aggressive, chill
    layers: List[str] = field(default_factory=lambda: ['drums'])
    humanize: float = 5.0       # humanization amount 0-100
    seed: Optional[int] = None


# ======================================================================
# Mood → parameter modifiers
# ======================================================================

MOOD_MODIFIERS: Dict[str, dict] = {
    'neutral':    {'velocity_mult': 1.0, 'density': 1.0, 'brightness': 0.0},
    'bright':     {'velocity_mult': 1.1, 'density': 1.1, 'brightness': 3.0},
    'dark':       {'velocity_mult': 0.85, 'density': 0.9, 'brightness': -3.0},
    'aggressive': {'velocity_mult': 1.2, 'density': 1.3, 'brightness': 2.0},
    'chill':      {'velocity_mult': 0.7, 'density': 0.7, 'brightness': -2.0},
}


# ======================================================================
# Bass-line generation
# ======================================================================

def _generate_bassline(spec: LoopSpec, sr: int = 48000) -> np.ndarray:
    """Generate a simple bass-line following a chord progression."""
    rng = random.Random(spec.seed)
    bpm = spec.bpm
    bar_samples = int(sr * 60.0 / bpm * 4)
    total = bar_samples * spec.bars
    out = np.zeros(total)

    # Pick a progression based on scale quality
    if spec.scale in ('minor', 'aeolian', 'dorian', 'phrygian',
                       'harmonic_minor', 'melodic_minor'):
        prog_name = rng.choice(['i_iv_v', 'i_VI_III_VII', 'i_iv_VII_III',
                                 'i_VII_VI_V'])
    else:
        prog_name = rng.choice(['I_IV_V', 'I_V_vi_IV', 'I_vi_IV_V',
                                 'I_IV_vi_V'])

    try:
        chords = mt.resolve_progression(spec.root_note, spec.scale, prog_name)
    except ValueError:
        chords = [[spec.root_note]]

    # Bass plays root of each chord, 1 chord per bar (cycle if needed)
    beat_samples = int(sr * 60.0 / bpm)
    mod = MOOD_MODIFIERS.get(spec.mood, MOOD_MODIFIERS['neutral'])

    for bar in range(spec.bars):
        chord = chords[bar % len(chords)]
        root_midi = chord[0]
        # Move bass to octave 2-3
        bass_midi = (root_midi % 12) + 36
        freq = mt.midi_to_freq(bass_midi)

        # Rhythmic pattern: depends on genre
        genre = spec.genre.lower()
        if genre in ('house', 'techno', 'minimal'):
            # Pumping 8th-note bass
            for beat_idx in range(8):
                t_start = bar * bar_samples + beat_idx * (beat_samples // 2)
                note_len = beat_samples // 2
                t = np.arange(note_len) / sr
                tone = np.sin(2 * np.pi * freq * t)
                env_decay = 0.7 if beat_idx % 2 == 0 else 0.4
                tone *= np.exp(-t / (env_decay * 60.0 / bpm))
                vel = 0.6 if beat_idx % 2 == 0 else 0.35
                vel *= mod['velocity_mult']
                end = min(t_start + note_len, total)
                out[t_start:end] += tone[:end - t_start] * vel
        elif genre in ('hiphop', 'trap', 'lofi'):
            # Sparse 808-style
            hits = [(0, 1.0), (3, 0.6)]
            if rng.random() > 0.4:
                hits.append((2.5, 0.5))
            for beat_pos, vel in hits:
                t_start = bar * bar_samples + int(beat_pos * beat_samples)
                note_len = int(beat_samples * 1.5)
                t = np.arange(note_len) / sr
                # Sub bass with pitch envelope
                p_env = freq + freq * 3 * np.exp(-t / 0.008)
                phase = np.cumsum(2 * np.pi * p_env / sr)
                tone = np.sin(phase) * np.exp(-t / 0.4)
                vel *= mod['velocity_mult']
                end = min(t_start + note_len, total)
                n = end - t_start
                if n > 0:
                    out[t_start:end] += tone[:n] * vel * 0.8
        else:
            # Generic: root on beat 1 and 3
            for beat_pos in [0, 2]:
                t_start = bar * bar_samples + beat_pos * beat_samples
                note_len = beat_samples
                t = np.arange(note_len) / sr
                tone = np.sin(2 * np.pi * freq * t)
                tone *= np.exp(-t / (60.0 / bpm * 0.8))
                vel = 0.55 * mod['velocity_mult']
                end = min(t_start + note_len, total)
                out[t_start:end] += tone[:end - t_start] * vel

    return out


# ======================================================================
# Chord pad generation
# ======================================================================

def _generate_chord_pad(spec: LoopSpec, sr: int = 48000) -> np.ndarray:
    """Generate a chord pad layer following a progression."""
    rng = random.Random(spec.seed)
    bpm = spec.bpm
    bar_samples = int(sr * 60.0 / bpm * 4)
    total = bar_samples * spec.bars
    out = np.zeros(total)

    if spec.scale in ('minor', 'aeolian', 'dorian'):
        prog_name = rng.choice(['i_iv_v', 'i_VI_III_VII'])
    else:
        prog_name = rng.choice(['I_V_vi_IV', 'I_vi_IV_V'])

    try:
        chords = mt.resolve_progression(spec.root_note, spec.scale, prog_name)
    except ValueError:
        chords = [mt.get_chord(spec.root_note, 'min')]

    mod = MOOD_MODIFIERS.get(spec.mood, MOOD_MODIFIERS['neutral'])
    prev_voicing = None

    for bar in range(spec.bars):
        chord_midi = chords[bar % len(chords)]
        # Voice-lead from previous bar
        if prev_voicing is not None:
            chord_midi = mt.voice_lead(prev_voicing, chord_midi)
        prev_voicing = chord_midi

        t_start = bar * bar_samples
        note_len = bar_samples
        t = np.arange(note_len) / sr

        # Render each note as a detuned pad
        for midi_note in chord_midi:
            freq = mt.midi_to_freq(midi_note)
            for detune in [-0.3, 0, 0.3]:
                f = freq * (2.0 ** (detune / 12.0))
                out_chunk = np.sin(2 * np.pi * f * t)
                # Soft attack/release
                atk = min(int(sr * 0.1), note_len // 4)
                rel = min(int(sr * 0.15), note_len // 4)
                env = np.ones(note_len)
                if atk > 0:
                    env[:atk] = np.linspace(0, 1, atk)
                if rel > 0:
                    env[-rel:] = np.linspace(1, 0, rel)
                amp = 0.12 / max(1, len(chord_midi))
                amp *= mod['velocity_mult']
                end = min(t_start + note_len, total)
                n = end - t_start
                if n > 0:
                    out[t_start:end] += out_chunk[:n] * env[:n] * amp

    return out


# ======================================================================
# Melody layer generation
# ======================================================================

def _generate_melody_layer(spec: LoopSpec, sr: int = 48000) -> np.ndarray:
    """Generate a simple melody layer from the scale."""
    rng = random.Random(spec.seed)
    bpm = spec.bpm
    bar_samples = int(sr * 60.0 / bpm * 4)
    total = bar_samples * spec.bars
    out = np.zeros(total)
    mod = MOOD_MODIFIERS.get(spec.mood, MOOD_MODIFIERS['neutral'])

    # Generate melody notes
    n_notes = spec.bars * 4  # quarter-note resolution
    contour = {'bright': 'ascending', 'dark': 'descending',
               'aggressive': 'wave', 'chill': 'arch'}.get(spec.mood, 'arch')
    melody = mt.generate_melody_from_scale(
        spec.root_note + 12, spec.scale, n_notes,
        octave_range=1, contour=contour, seed=spec.seed)

    beat_samples = int(sr * 60.0 / bpm)
    density = mod.get('density', 1.0)

    for i, midi_note in enumerate(melody):
        # Skip some notes based on density
        if rng.random() > density:
            continue
        freq = mt.midi_to_freq(midi_note)
        t_start = i * beat_samples
        note_len = int(beat_samples * rng.uniform(0.4, 0.9))
        if t_start + note_len > total:
            note_len = total - t_start
        if note_len <= 0:
            continue
        t = np.arange(note_len) / sr
        # Simple pluck synthesis
        tone = np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.15)
        tone += np.sin(2 * np.pi * freq * 2 * t) * 0.3 * np.exp(-t / 0.08)
        vel = 0.3 * mod['velocity_mult']
        end = min(t_start + note_len, total)
        n = end - t_start
        if n > 0:
            out[t_start:end] += tone[:n] * vel

    return out


# ======================================================================
# Main loop generation
# ======================================================================

def generate_loop(spec: LoopSpec, sr: int = 48000) -> np.ndarray:
    """Generate a complete loop from a LoopSpec.

    Returns a mono float64 audio buffer.
    """
    # Resolve BPM from genre if not specified
    if spec.bpm <= 0:
        template = beat_gen.GENRE_TEMPLATES.get(spec.genre.lower())
        spec.bpm = template.bpm if template else 128.0

    layers_audio: List[np.ndarray] = []
    bar_samples = int(sr * 60.0 / spec.bpm * 4)
    target_len = bar_samples * spec.bars

    for layer_name in spec.layers:
        layer = layer_name.lower().strip()
        if layer in ('drums', 'beat', 'rhythm'):
            template = beat_gen.GENRE_TEMPLATES.get(spec.genre.lower())
            if template is None:
                template = beat_gen.GENRE_TEMPLATES.get('house')
            import copy
            bp = copy.deepcopy(template)
            bp.bpm = spec.bpm
            if spec.humanize > 0:
                bp = beat_gen.humanize_pattern(bp, spec.humanize,
                                               spec.humanize, spec.seed)
            audio = beat_gen.render_beat(bp, spec.bars, sr)
            layers_audio.append(audio)

        elif layer in ('bass', 'bassline', 'sub'):
            audio = _generate_bassline(spec, sr)
            layers_audio.append(audio)

        elif layer in ('chords', 'pad', 'harmony'):
            audio = _generate_chord_pad(spec, sr)
            layers_audio.append(audio)

        elif layer in ('melody', 'lead', 'top'):
            audio = _generate_melody_layer(spec, sr)
            layers_audio.append(audio)

    if not layers_audio:
        return np.zeros(target_len)

    # Mix all layers to target length
    out = np.zeros(target_len)
    for audio in layers_audio:
        n = min(len(audio), target_len)
        out[:n] += audio[:n]

    # Soft-limit the mix
    peak = np.max(np.abs(out))
    if peak > 0.95:
        out = np.tanh(out / peak * 1.5) * 0.9
    elif peak > 1e-10:
        out *= 0.9 / peak

    return out


def format_loop_info(spec: LoopSpec) -> str:
    """Human-readable description of a LoopSpec."""
    bpm = spec.bpm if spec.bpm > 0 else '(genre default)'
    root = mt.midi_to_note_name(spec.root_note)
    return (
        f"Loop: {spec.genre} | {bpm} BPM | {spec.bars} bars | "
        f"Key: {root} {spec.scale} | Mood: {spec.mood} | "
        f"Layers: {', '.join(spec.layers)}"
    )
