"""Drum beat generation engine — Phase 4.4.

Provides genre-aware beat generation with:
- Algorithmic sound generators for all drum/percussion types
- Genre template library with velocity/swing patterns
- Fill and variation generation
- Humanization engine
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ======================================================================
# Sound generators — all drum voices
# ======================================================================

def _env(length: int, attack: int = 0, decay: int = 0,
         sustain_level: float = 0.0) -> np.ndarray:
    """Simple ADS envelope (no release — drums gate off)."""
    env = np.zeros(length)
    if attack > 0:
        env[:attack] = np.linspace(0, 1, attack, endpoint=False)
    sus_start = attack + decay
    if decay > 0:
        env[attack:sus_start] = np.linspace(1, sustain_level, decay,
                                             endpoint=False)
    if sus_start < length:
        env[sus_start:] = sustain_level
    return env


def generate_kick(sr: int = 48000, variant: str = '808',
                  tune: float = 0.0) -> np.ndarray:
    """Kick drum — variants: 808, acoustic, sub, punch, click."""
    dur = {
        '808': 0.45, 'acoustic': 0.30, 'sub': 0.60,
        'punch': 0.20, 'click': 0.10,
    }.get(variant, 0.35)
    n = int(sr * dur)
    t = np.arange(n) / sr

    base_freq = 55.0 * (2.0 ** (tune / 12.0))
    configs = {
        '808':      (base_freq, 160, 8.0, 0.0),
        'acoustic': (base_freq * 1.2, 120, 12.0, 20.0),
        'sub':      (base_freq * 0.7, 200, 5.0, 0.0),
        'punch':    (base_freq * 1.5, 80, 20.0, 30.0),
        'click':    (base_freq * 2.0, 40, 40.0, 50.0),
    }
    freq, pitch_decay_ms, pitch_range, noise_amt = configs.get(
        variant, configs['808'])
    pitch_env = freq + pitch_range * freq * np.exp(
        -t / (pitch_decay_ms / 1000.0))
    phase = np.cumsum(2 * np.pi * pitch_env / sr)
    body = np.sin(phase)
    amp = np.exp(-t / (dur * 0.5))
    out = body * amp
    if noise_amt > 0:
        click = np.random.randn(min(n, int(sr * 0.005)))
        click *= np.exp(-np.arange(len(click)) / (sr * 0.001))
        out[:len(click)] += click * (noise_amt / 100.0)
    out *= 0.9 / (np.max(np.abs(out)) + 1e-10)
    return out


def generate_snare(sr: int = 48000, variant: str = 'acoustic',
                   tune: float = 0.0) -> np.ndarray:
    """Snare drum — variants: acoustic, 808, tight, fat, clap."""
    dur = 0.25
    n = int(sr * dur)
    t = np.arange(n) / sr
    base = 200.0 * (2.0 ** (tune / 12.0))
    body_freq = {
        'acoustic': base, '808': base * 0.8, 'tight': base * 1.3,
        'fat': base * 0.9, 'clap': base * 1.1,
    }.get(variant, base)
    body = np.sin(2 * np.pi * body_freq * t) * np.exp(-t / 0.06)
    noise = np.random.randn(n)
    noise_env = np.exp(-t / {'acoustic': 0.08, '808': 0.12, 'tight': 0.04,
                               'fat': 0.15, 'clap': 0.10}.get(variant, 0.08))
    noise *= noise_env
    mix = {'acoustic': 0.5, '808': 0.4, 'tight': 0.6,
           'fat': 0.35, 'clap': 0.7}.get(variant, 0.5)
    out = body * (1 - mix) + noise * mix
    if variant == 'clap':
        for offset in [0.008, 0.016, 0.024]:
            s = int(offset * sr)
            burst = np.random.randn(min(n - s, int(sr * 0.01)))
            burst *= np.exp(-np.arange(len(burst)) / (sr * 0.004))
            out[s:s + len(burst)] += burst * 0.3
    out *= 0.9 / (np.max(np.abs(out)) + 1e-10)
    return out


def generate_hihat(sr: int = 48000, variant: str = 'closed',
                   tune: float = 0.0) -> np.ndarray:
    """Hi-hat — variants: closed, open, pedal, sizzle."""
    dur = {'closed': 0.08, 'open': 0.40, 'pedal': 0.06,
           'sizzle': 0.60}.get(variant, 0.10)
    n = int(sr * dur)
    t = np.arange(n) / sr
    noise = np.random.randn(n)
    # High-pass via simple diff
    hp = np.diff(noise, prepend=0)
    hp = np.diff(hp, prepend=0)
    # Metallic resonances
    freqs = [3500 + tune * 100, 5200 + tune * 100, 7800 + tune * 100]
    metallic = sum(np.sin(2 * np.pi * f * t) * 0.3 for f in freqs)
    decay = {'closed': 0.015, 'open': 0.15, 'pedal': 0.01,
             'sizzle': 0.25}.get(variant, 0.03)
    env = np.exp(-t / decay)
    out = (hp * 0.7 + metallic * 0.3) * env
    out *= 0.9 / (np.max(np.abs(out)) + 1e-10)
    return out


def generate_tom(sr: int = 48000, variant: str = 'mid',
                 tune: float = 0.0) -> np.ndarray:
    """Tom — variants: high, mid, low, floor, electronic."""
    base_freqs = {'high': 250, 'mid': 180, 'low': 120,
                  'floor': 80, 'electronic': 150}
    freq = base_freqs.get(variant, 150) * (2.0 ** (tune / 12.0))
    dur = 0.35 if variant != 'electronic' else 0.50
    n = int(sr * dur)
    t = np.arange(n) / sr
    pitch_env = freq + freq * 0.5 * np.exp(-t / 0.01)
    phase = np.cumsum(2 * np.pi * pitch_env / sr)
    body = np.sin(phase)
    env = np.exp(-t / (dur * 0.4))
    out = body * env
    out *= 0.9 / (np.max(np.abs(out)) + 1e-10)
    return out


def generate_cymbal(sr: int = 48000, variant: str = 'crash',
                    tune: float = 0.0) -> np.ndarray:
    """Cymbal — variants: crash, ride, china, splash, bell."""
    durs = {'crash': 1.5, 'ride': 0.8, 'china': 1.2,
            'splash': 0.6, 'bell': 0.4}
    dur = durs.get(variant, 1.0)
    n = int(sr * dur)
    t = np.arange(n) / sr
    noise = np.random.randn(n)
    hp = np.diff(np.diff(noise, prepend=0), prepend=0)
    freqs = [2800 + tune * 80, 4200 + tune * 80, 6100 + tune * 80,
             8500 + tune * 80]
    metallic = sum(np.sin(2 * np.pi * f * t) * 0.15 for f in freqs)
    decays = {'crash': 0.6, 'ride': 0.25, 'china': 0.45,
              'splash': 0.15, 'bell': 0.3}
    env = np.exp(-t / decays.get(variant, 0.4))
    out = (hp * 0.6 + metallic * 0.4) * env
    if variant == 'bell':
        bell_tone = np.sin(2 * np.pi * (3000 + tune * 50) * t) * 0.4
        out += bell_tone * env
    out *= 0.9 / (np.max(np.abs(out)) + 1e-10)
    return out


def generate_clap(sr: int = 48000, variant: str = '808',
                  tune: float = 0.0) -> np.ndarray:
    """Clap — variants: 808, acoustic, layered, big."""
    dur = 0.20
    n = int(sr * dur)
    t = np.arange(n) / sr
    out = np.zeros(n)
    n_layers = {'808': 4, 'acoustic': 3, 'layered': 6, 'big': 8}.get(variant, 4)
    for i in range(n_layers):
        offset = int(sr * random.uniform(0.002, 0.015))
        burst_len = int(sr * 0.008)
        if offset + burst_len < n:
            burst = np.random.randn(burst_len)
            burst *= np.exp(-np.arange(burst_len) / (sr * 0.003))
            out[offset:offset + burst_len] += burst
    tail = np.random.randn(n) * np.exp(-t / 0.06)
    out += tail * 0.3
    out *= 0.9 / (np.max(np.abs(out)) + 1e-10)
    return out


def generate_snap(sr: int = 48000, variant: str = 'dry',
                  tune: float = 0.0) -> np.ndarray:
    """Snap — variants: dry, reverb, electronic."""
    dur = 0.08
    n = int(sr * dur)
    t = np.arange(n) / sr
    click = np.random.randn(int(sr * 0.002))
    click *= np.exp(-np.arange(len(click)) / (sr * 0.0005))
    body = np.sin(2 * np.pi * (1200 + tune * 50) * t) * np.exp(-t / 0.008)
    out = np.zeros(n)
    out[:len(click)] += click * 0.8
    out += body * 0.4
    out *= 0.9 / (np.max(np.abs(out)) + 1e-10)
    return out


def generate_shaker(sr: int = 48000, variant: str = '16th',
                    tune: float = 0.0) -> np.ndarray:
    """Shaker — variants: 16th, 8th, triplet, soft."""
    dur = {'16th': 0.04, '8th': 0.08, 'triplet': 0.06,
           'soft': 0.10}.get(variant, 0.06)
    n = int(sr * dur)
    t = np.arange(n) / sr
    noise = np.random.randn(n)
    # Band-pass via successive high-pass/low-pass
    hp = np.diff(noise, prepend=0)
    env = np.exp(-t / (dur * 0.4))
    out = hp * env * 0.7
    out *= 0.9 / (np.max(np.abs(out)) + 1e-10)
    return out


def generate_stab(sr: int = 48000, variant: str = 'chord',
                  tune: float = 0.0) -> np.ndarray:
    """Stab hit — variants: chord, brass, synth."""
    dur = 0.15
    n = int(sr * dur)
    t = np.arange(n) / sr
    base = 220.0 * (2.0 ** (tune / 12.0))
    notes = {
        'chord': [base, base * 1.26, base * 1.5],
        'brass':  [base, base * 1.5, base * 2.0],
        'synth':  [base, base * 1.26, base * 1.5, base * 2.0],
    }.get(variant, [base])
    out = sum(np.sin(2 * np.pi * f * t) for f in notes) / len(notes)
    env = np.exp(-t / 0.04)
    out = out * env
    out *= 0.9 / (np.max(np.abs(out)) + 1e-10)
    return out


def generate_bass_hit(sr: int = 48000, variant: str = '808',
                      tune: float = 0.0) -> np.ndarray:
    """Bass hit — variants: 808, sub, punch, growl."""
    dur = {'808': 0.8, 'sub': 1.0, 'punch': 0.25, 'growl': 0.5}.get(variant, 0.5)
    n = int(sr * dur)
    t = np.arange(n) / sr
    freq = 45.0 * (2.0 ** (tune / 12.0))
    configs = {
        '808':   (freq, 0.3, 0.0),
        'sub':   (freq * 0.8, 0.5, 0.0),
        'punch': (freq * 1.5, 0.08, 0.1),
        'growl': (freq, 0.2, 0.3),
    }
    f, dec, dist = configs.get(variant, configs['808'])
    phase = np.cumsum(2 * np.pi * (f + f * 2 * np.exp(-t / 0.008)) / sr)
    body = np.sin(phase)
    if dist > 0:
        body = np.tanh(body * (1 + dist * 10))
    env = np.exp(-t / dec)
    out = body * env
    out *= 0.9 / (np.max(np.abs(out)) + 1e-10)
    return out


def generate_bell(sr: int = 48000, variant: str = 'bright',
                  tune: float = 0.0) -> np.ndarray:
    """Bell — variants: bright, dark, tubular, glass."""
    dur = 1.5
    n = int(sr * dur)
    t = np.arange(n) / sr
    base = 880.0 * (2.0 ** (tune / 12.0))
    partials = {
        'bright':  [(1.0, 1.0), (2.0, 0.6), (3.0, 0.3), (4.76, 0.2)],
        'dark':    [(1.0, 1.0), (2.0, 0.4), (3.0, 0.1)],
        'tubular': [(1.0, 1.0), (1.5, 0.5), (2.76, 0.3), (4.07, 0.15)],
        'glass':   [(1.0, 1.0), (2.76, 0.7), (5.4, 0.3), (8.93, 0.1)],
    }
    out = np.zeros(n)
    for ratio, amp in partials.get(variant, partials['bright']):
        decay = 0.6 / ratio
        out += amp * np.sin(2 * np.pi * base * ratio * t) * np.exp(-t / decay)
    out *= 0.9 / (np.max(np.abs(out)) + 1e-10)
    return out


def generate_riser(sr: int = 48000, variant: str = 'noise',
                   tune: float = 0.0) -> np.ndarray:
    """Riser — variants: noise, tonal, sweep, tension."""
    dur = 2.0
    n = int(sr * dur)
    t = np.arange(n) / sr
    env_up = t / dur
    if variant == 'noise':
        out = np.random.randn(n) * env_up
    elif variant == 'tonal':
        freq = np.linspace(200, 2000, n) * (2.0 ** (tune / 12.0))
        phase = np.cumsum(2 * np.pi * freq / sr)
        out = np.sin(phase) * env_up
    elif variant == 'sweep':
        freq = np.logspace(np.log10(100), np.log10(8000), n)
        phase = np.cumsum(2 * np.pi * freq / sr)
        out = np.sin(phase) * env_up
    else:  # tension
        freq = np.linspace(300, 600, n) * (2.0 ** (tune / 12.0))
        phase = np.cumsum(2 * np.pi * freq / sr)
        out = (np.sin(phase) + np.random.randn(n) * 0.3 * env_up) * env_up
    out *= 0.9 / (np.max(np.abs(out)) + 1e-10)
    return out


def generate_downlifter(sr: int = 48000, variant: str = 'drop',
                        tune: float = 0.0) -> np.ndarray:
    """Downlifter — variants: drop, sweep, impact."""
    dur = 1.5
    n = int(sr * dur)
    t = np.arange(n) / sr
    env = 1.0 - t / dur
    freq = np.linspace(4000, 80, n) * (2.0 ** (tune / 12.0))
    phase = np.cumsum(2 * np.pi * freq / sr)
    out = np.sin(phase) * env
    if variant == 'impact':
        click = np.exp(-np.arange(min(n, int(sr * 0.01))) / (sr * 0.002))
        out[:len(click)] += click * 0.4
    out *= 0.9 / (np.max(np.abs(out)) + 1e-10)
    return out


def generate_fx(sr: int = 48000, variant: str = 'zap',
                tune: float = 0.0) -> np.ndarray:
    """FX sounds — variants: zap, laser, whoosh, glitch, vinyl, wind."""
    if variant in ('zap', 'laser'):
        dur = 0.2 if variant == 'zap' else 0.4
        n = int(sr * dur)
        t = np.arange(n) / sr
        freq = np.linspace(4000, 100, n) * (2.0 ** (tune / 12.0))
        phase = np.cumsum(2 * np.pi * freq / sr)
        out = np.sin(phase) * np.exp(-t / (dur * 0.5))
    elif variant == 'whoosh':
        dur = 0.8
        n = int(sr * dur)
        noise = np.random.randn(n)
        env = np.sin(np.linspace(0, np.pi, n))
        out = noise * env
    elif variant == 'glitch':
        dur = 0.3
        n = int(sr * dur)
        out = np.zeros(n)
        chunk = max(1, n // 20)
        for i in range(0, n, chunk):
            end = min(i + chunk, n)
            src = random.randint(0, max(0, n - chunk))
            fill = np.random.randn(end - i)
            out[i:end] = fill * np.exp(-(np.arange(end - i)) / (chunk * 0.3))
    elif variant == 'vinyl':
        dur = 2.0
        n = int(sr * dur)
        crackle = np.zeros(n)
        for _ in range(int(dur * 40)):
            pos = random.randint(0, n - 1)
            crackle[pos] = random.uniform(-0.3, 0.3)
        hiss = np.random.randn(n) * 0.02
        out = crackle + hiss
    elif variant == 'wind':
        dur = 3.0
        n = int(sr * dur)
        noise = np.random.randn(n)
        # Slow modulation
        mod = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(
            2 * np.pi * 0.3 * np.arange(n) / sr))
        out = noise * mod * 0.3
    else:
        dur = 0.2
        n = int(sr * dur)
        out = np.random.randn(n) * np.exp(-np.arange(n) / (n * 0.3))
    out *= 0.9 / (np.max(np.abs(out)) + 1e-10)
    return out


def generate_silence(sr: int = 48000, dur: float = 1.0, **kw) -> np.ndarray:
    return np.zeros(int(sr * dur))


def generate_click_track(sr: int = 48000, bpm: float = 128.0,
                         bars: int = 1, **kw) -> np.ndarray:
    """Metronome click track."""
    beat_samples = int(sr * 60.0 / bpm)
    n_beats = bars * 4
    out = np.zeros(beat_samples * n_beats)
    click = np.sin(2 * np.pi * 1000 * np.arange(int(sr * 0.01)) / sr)
    click *= np.exp(-np.arange(len(click)) / (sr * 0.002))
    accent = np.sin(2 * np.pi * 1500 * np.arange(int(sr * 0.01)) / sr)
    accent *= np.exp(-np.arange(len(accent)) / (sr * 0.002))
    for i in range(n_beats):
        pos = i * beat_samples
        c = accent if i % 4 == 0 else click
        end = min(pos + len(c), len(out))
        out[pos:end] += c[:end - pos]
    return out


def generate_sweep(sr: int = 48000, variant: str = 'up',
                   tune: float = 0.0) -> np.ndarray:
    """Test sweep — variants: up, down."""
    dur = 2.0
    n = int(sr * dur)
    if variant == 'down':
        freq = np.logspace(np.log10(10000), np.log10(20), n)
    else:
        freq = np.logspace(np.log10(20), np.log10(10000), n)
    phase = np.cumsum(2 * np.pi * freq / sr)
    out = np.sin(phase) * 0.5
    return out


# ======================================================================
# Generator registry
# ======================================================================

GENERATORS: Dict[str, dict] = {
    # Drums
    'kick':   {'fn': generate_kick, 'variants': ['808', 'acoustic', 'sub', 'punch', 'click'],
               'desc': 'Kick drum'},
    'snare':  {'fn': generate_snare, 'variants': ['acoustic', '808', 'tight', 'fat', 'clap'],
               'desc': 'Snare drum'},
    'hihat':  {'fn': generate_hihat, 'variants': ['closed', 'open', 'pedal', 'sizzle'],
               'desc': 'Hi-hat'},
    'tom':    {'fn': generate_tom, 'variants': ['high', 'mid', 'low', 'floor', 'electronic'],
               'desc': 'Tom drum'},
    'cymbal': {'fn': generate_cymbal, 'variants': ['crash', 'ride', 'china', 'splash', 'bell'],
               'desc': 'Cymbal'},
    'clap':   {'fn': generate_clap, 'variants': ['808', 'acoustic', 'layered', 'big'],
               'desc': 'Clap'},
    'snap':   {'fn': generate_snap, 'variants': ['dry', 'reverb', 'electronic'],
               'desc': 'Finger snap'},
    'shaker': {'fn': generate_shaker, 'variants': ['16th', '8th', 'triplet', 'soft'],
               'desc': 'Shaker'},
    # Tonal
    'stab':   {'fn': generate_stab, 'variants': ['chord', 'brass', 'synth'],
               'desc': 'Stab / hit'},
    'bass':   {'fn': generate_bass_hit, 'variants': ['808', 'sub', 'punch', 'growl'],
               'desc': 'Bass hit'},
    'bell':   {'fn': generate_bell, 'variants': ['bright', 'dark', 'tubular', 'glass'],
               'desc': 'Bell'},
    # FX / transitions
    'riser':      {'fn': generate_riser, 'variants': ['noise', 'tonal', 'sweep', 'tension'],
                   'desc': 'Riser / build-up'},
    'downlifter': {'fn': generate_downlifter, 'variants': ['drop', 'sweep', 'impact'],
                   'desc': 'Downlifter / drop'},
    'fx':         {'fn': generate_fx,
                   'variants': ['zap', 'laser', 'whoosh', 'glitch', 'vinyl', 'wind'],
                   'desc': 'Sound FX'},
    # Utility
    'silence': {'fn': generate_silence, 'variants': [], 'desc': 'Silence'},
    'click':   {'fn': generate_click_track, 'variants': [], 'desc': 'Click track'},
    'sweep':   {'fn': generate_sweep, 'variants': ['up', 'down'], 'desc': 'Test sweep'},
}

# Short aliases
GENERATOR_ALIASES: Dict[str, str] = {
    'sk': 'kick', 'kk': 'kick', 'bd': 'kick',
    'sn': 'snare', 'sd': 'snare', 'ssn': 'snare',
    'hh': 'hihat', 'ch': 'hihat', 'oh': 'hihat', 'sh': 'hihat',
    'tm': 'tom', 'cy': 'cymbal', 'cp': 'clap',
    'sp': 'snap', 'shk': 'shaker', 'st': 'stab', 'stb': 'stab',
    'bs': 'bass', 'bas': 'bass', 'bl': 'bell', 'bel': 'bell',
    'rs': 'riser', 'rsr': 'riser', 'dl': 'downlifter', 'dwn': 'downlifter',
    'zp': 'fx', 'zap': 'fx', 'lsr': 'fx', 'wsh': 'fx',
    'glt': 'fx', 'vnl': 'fx', 'wnd': 'fx',
    'sil': 'silence', 'clk': 'click', 'swp': 'sweep', 'cal': 'sweep',
}


def resolve_generator(name: str) -> Optional[str]:
    """Resolve alias or exact name to canonical generator name."""
    low = name.lower()
    if low in GENERATORS:
        return low
    return GENERATOR_ALIASES.get(low)


# ======================================================================
# Beat pattern data structures
# ======================================================================

@dataclass
class Hit:
    """Single hit in a beat grid."""
    instrument: str = 'kick'     # canonical generator name
    variant: str = ''            # generator variant ('' = default)
    velocity: float = 100.0      # 0-100
    tune: float = 0.0            # semitone offset
    offset_ms: float = 0.0       # micro-timing offset from grid


@dataclass
class BeatPattern:
    """A grid-based drum pattern.

    *grid* maps step index (0-based, 16th-note resolution) to list of
    simultaneous hits.  *steps* is the grid length.
    """
    name: str = 'pattern'
    steps: int = 16              # grid resolution (16 = 1 bar of 16ths)
    grid: Dict[int, List[Hit]] = field(default_factory=dict)
    swing: float = 0.0           # 0-100, swing amount on even 16ths
    bpm: float = 128.0

    def add_hit(self, step: int, instrument: str, velocity: float = 100.0,
                variant: str = '', tune: float = 0.0):
        if step not in self.grid:
            self.grid[step] = []
        self.grid[step].append(Hit(instrument, variant, velocity, tune))

    @property
    def active_steps(self) -> int:
        return len(self.grid)


# ======================================================================
# Genre templates
# ======================================================================

def _make_pattern(name: str, steps: int, bpm: float, swing: float,
                  hits: List[Tuple[int, str, str, float]]) -> BeatPattern:
    """Shorthand: hits = [(step, instrument, variant, velocity), ...]."""
    bp = BeatPattern(name=name, steps=steps, bpm=bpm, swing=swing)
    for step, inst, var, vel in hits:
        bp.add_hit(step, inst, vel, var)
    return bp


GENRE_TEMPLATES: Dict[str, BeatPattern] = {}

# ---- House (4-on-the-floor) ----
GENRE_TEMPLATES['house'] = _make_pattern('house', 16, 128, 0, [
    (0, 'kick', '808', 100), (4, 'kick', '808', 100),
    (8, 'kick', '808', 100), (12, 'kick', '808', 100),
    (2, 'hihat', 'closed', 60), (6, 'hihat', 'closed', 60),
    (10, 'hihat', 'closed', 60), (14, 'hihat', 'closed', 60),
    (4, 'clap', '808', 85), (12, 'clap', '808', 85),
])

# ---- Techno ----
GENRE_TEMPLATES['techno'] = _make_pattern('techno', 16, 135, 0, [
    (0, 'kick', 'punch', 100), (4, 'kick', 'punch', 100),
    (8, 'kick', 'punch', 100), (12, 'kick', 'punch', 100),
    (2, 'hihat', 'closed', 50), (6, 'hihat', 'closed', 55),
    (10, 'hihat', 'closed', 50), (14, 'hihat', 'closed', 55),
    (4, 'clap', '808', 80), (12, 'clap', '808', 80),
    (7, 'hihat', 'open', 40),
])

# ---- Hip Hop ----
GENRE_TEMPLATES['hiphop'] = _make_pattern('hiphop', 16, 90, 55, [
    (0, 'kick', '808', 100), (6, 'kick', '808', 85),
    (10, 'kick', '808', 70),
    (4, 'snare', '808', 90), (12, 'snare', '808', 90),
    (2, 'hihat', 'closed', 60), (4, 'hihat', 'closed', 50),
    (6, 'hihat', 'closed', 60), (8, 'hihat', 'closed', 50),
    (10, 'hihat', 'closed', 60), (12, 'hihat', 'closed', 50),
    (14, 'hihat', 'closed', 60),
])

# ---- Trap ----
GENRE_TEMPLATES['trap'] = _make_pattern('trap', 32, 140, 0, [
    (0, 'kick', '808', 100), (14, 'kick', '808', 85),
    (8, 'snare', 'tight', 90), (24, 'snare', 'tight', 90),
    # Rapid hi-hat rolls
    (0, 'hihat', 'closed', 50), (2, 'hihat', 'closed', 55),
    (4, 'hihat', 'closed', 50), (5, 'hihat', 'closed', 40),
    (6, 'hihat', 'closed', 55), (7, 'hihat', 'closed', 40),
    (8, 'hihat', 'closed', 50), (10, 'hihat', 'closed', 55),
    (12, 'hihat', 'closed', 50), (13, 'hihat', 'closed', 40),
    (14, 'hihat', 'closed', 55), (15, 'hihat', 'closed', 40),
    (16, 'hihat', 'closed', 50), (18, 'hihat', 'closed', 55),
    (20, 'hihat', 'closed', 50), (21, 'hihat', 'closed', 40),
    (22, 'hihat', 'closed', 55), (23, 'hihat', 'closed', 40),
    (24, 'hihat', 'closed', 50), (26, 'hihat', 'closed', 55),
    (28, 'hihat', 'closed', 50), (30, 'hihat', 'closed', 55),
    (20, 'hihat', 'open', 35),
])

# ---- Drum & Bass ----
GENRE_TEMPLATES['dnb'] = _make_pattern('dnb', 16, 174, 0, [
    (0, 'kick', 'punch', 100), (10, 'kick', 'punch', 85),
    (4, 'snare', 'tight', 95), (12, 'snare', 'tight', 95),
    (0, 'hihat', 'closed', 55), (2, 'hihat', 'closed', 55),
    (4, 'hihat', 'closed', 55), (6, 'hihat', 'closed', 55),
    (8, 'hihat', 'closed', 55), (10, 'hihat', 'closed', 55),
    (12, 'hihat', 'closed', 55), (14, 'hihat', 'closed', 55),
])

# ---- Lo-fi ----
GENRE_TEMPLATES['lofi'] = _make_pattern('lofi', 16, 85, 45, [
    (0, 'kick', 'acoustic', 80), (7, 'kick', 'acoustic', 65),
    (4, 'snare', 'acoustic', 70), (12, 'snare', 'acoustic', 70),
    (2, 'hihat', 'closed', 40), (6, 'hihat', 'closed', 35),
    (10, 'hihat', 'closed', 40), (14, 'hihat', 'closed', 35),
    (8, 'hihat', 'open', 25),
])

# ---- Reggaeton ----
GENRE_TEMPLATES['reggaeton'] = _make_pattern('reggaeton', 16, 95, 0, [
    (0, 'kick', '808', 100), (6, 'kick', '808', 80),
    (8, 'kick', '808', 100), (14, 'kick', '808', 80),
    (3, 'snare', 'tight', 85), (7, 'snare', 'tight', 85),
    (11, 'snare', 'tight', 85), (15, 'snare', 'tight', 85),
    (0, 'hihat', 'closed', 55), (2, 'hihat', 'closed', 55),
    (4, 'hihat', 'closed', 55), (6, 'hihat', 'closed', 55),
    (8, 'hihat', 'closed', 55), (10, 'hihat', 'closed', 55),
    (12, 'hihat', 'closed', 55), (14, 'hihat', 'closed', 55),
])

# ---- Breakbeat ----
GENRE_TEMPLATES['breakbeat'] = _make_pattern('breakbeat', 16, 130, 20, [
    (0, 'kick', 'acoustic', 100), (5, 'kick', 'acoustic', 75),
    (10, 'kick', 'acoustic', 80),
    (4, 'snare', 'acoustic', 90), (12, 'snare', 'acoustic', 90),
    (7, 'snare', 'acoustic', 55),
    (0, 'hihat', 'closed', 55), (2, 'hihat', 'closed', 50),
    (4, 'hihat', 'closed', 55), (6, 'hihat', 'closed', 50),
    (8, 'hihat', 'closed', 55), (10, 'hihat', 'closed', 50),
    (12, 'hihat', 'closed', 55), (14, 'hihat', 'closed', 50),
])

# ---- Dubstep ----
GENRE_TEMPLATES['dubstep'] = _make_pattern('dubstep', 32, 140, 0, [
    (0, 'kick', 'sub', 100),
    (12, 'snare', 'fat', 95), (28, 'snare', 'fat', 95),
    (0, 'hihat', 'closed', 50), (4, 'hihat', 'closed', 50),
    (8, 'hihat', 'closed', 50), (12, 'hihat', 'closed', 50),
    (16, 'hihat', 'closed', 50), (20, 'hihat', 'closed', 50),
    (24, 'hihat', 'closed', 50), (28, 'hihat', 'closed', 50),
])

# ---- Afrobeat ----
GENRE_TEMPLATES['afrobeat'] = _make_pattern('afrobeat', 16, 110, 30, [
    (0, 'kick', 'acoustic', 100), (8, 'kick', 'acoustic', 80),
    (12, 'kick', 'acoustic', 70),
    (4, 'snare', 'acoustic', 85), (12, 'snare', 'acoustic', 80),
    (6, 'shaker', '16th', 50), (10, 'shaker', '16th', 45),
    (14, 'shaker', '16th', 50),
    (0, 'hihat', 'closed', 50), (2, 'hihat', 'closed', 45),
    (4, 'hihat', 'closed', 50), (6, 'hihat', 'closed', 45),
    (8, 'hihat', 'closed', 50), (10, 'hihat', 'closed', 45),
    (12, 'hihat', 'closed', 50), (14, 'hihat', 'closed', 45),
])

# ---- Minimal ----
GENRE_TEMPLATES['minimal'] = _make_pattern('minimal', 16, 125, 0, [
    (0, 'kick', 'click', 90), (4, 'kick', 'click', 90),
    (8, 'kick', 'click', 90), (12, 'kick', 'click', 90),
    (4, 'clap', '808', 65), (12, 'snap', 'dry', 60),
    (2, 'hihat', 'closed', 35), (6, 'hihat', 'closed', 35),
    (10, 'hihat', 'closed', 35), (14, 'hihat', 'closed', 35),
])


def list_genres() -> List[str]:
    return sorted(GENRE_TEMPLATES.keys())


# ======================================================================
# Beat rendering
# ======================================================================

def render_beat(pattern: BeatPattern, bars: int = 1,
                sr: int = 48000) -> np.ndarray:
    """Render a BeatPattern to a mono audio buffer."""
    bpm = pattern.bpm
    step_dur = 60.0 / bpm / 4.0  # 16th note
    total_steps = pattern.steps * bars
    total_samples = int(total_steps * step_dur * sr)
    out = np.zeros(total_samples)

    for bar in range(bars):
        for step, hits in pattern.grid.items():
            abs_step = bar * pattern.steps + step
            # Swing: offset even 16ths
            swing_offset = 0.0
            if pattern.swing > 0 and step % 2 == 1:
                swing_offset = (pattern.swing / 100.0) * step_dur * 0.5
            t_sec = abs_step * step_dur + swing_offset
            pos = int(t_sec * sr)
            if pos >= total_samples:
                continue

            for hit in hits:
                gen_name = resolve_generator(hit.instrument)
                if gen_name is None:
                    continue
                gen_info = GENERATORS.get(gen_name)
                if gen_info is None:
                    continue
                fn = gen_info['fn']
                variant = hit.variant or (gen_info['variants'][0]
                                          if gen_info['variants'] else '')
                # Handle special generators
                kwargs = {'sr': sr, 'tune': hit.tune}
                if gen_name == 'silence':
                    kwargs = {'sr': sr, 'dur': step_dur}
                elif gen_name == 'click':
                    kwargs = {'sr': sr, 'bpm': bpm, 'bars': 1}
                elif gen_name == 'sweep':
                    kwargs = {'sr': sr}
                if variant and gen_name not in ('silence', 'click'):
                    kwargs['variant'] = variant
                try:
                    sound = fn(**kwargs)
                except Exception:
                    continue
                # Apply velocity
                sound = sound * (hit.velocity / 100.0)
                # Apply micro-timing
                offset_samples = int(hit.offset_ms / 1000.0 * sr)
                write_pos = max(0, pos + offset_samples)
                end = min(write_pos + len(sound), total_samples)
                n_copy = end - write_pos
                if n_copy > 0:
                    out[write_pos:end] += sound[:n_copy]

    # Soft-limit
    peak = np.max(np.abs(out))
    if peak > 0.95:
        out = np.tanh(out / peak * 1.5) * 0.9
    elif peak > 0:
        out *= 0.9 / peak
    return out


# ======================================================================
# Humanization
# ======================================================================

def humanize_pattern(pattern: BeatPattern,
                     timing_pct: float = 5.0,
                     velocity_pct: float = 10.0,
                     seed: Optional[int] = None) -> BeatPattern:
    """Return a humanized copy of *pattern*."""
    rng = random.Random(seed)
    import copy
    hp = copy.deepcopy(pattern)
    step_dur_ms = 60000.0 / hp.bpm / 4.0
    for step, hits in hp.grid.items():
        for hit in hits:
            hit.offset_ms += rng.gauss(0, step_dur_ms * timing_pct / 100)
            vel_delta = rng.gauss(0, velocity_pct)
            hit.velocity = max(10, min(100, hit.velocity + vel_delta))
    return hp


# ======================================================================
# Fill generation
# ======================================================================

def generate_fill(pattern: BeatPattern, fill_type: str = 'buildup',
                  seed: Optional[int] = None) -> BeatPattern:
    """Generate a fill variation for the last bar.

    *fill_type*: buildup, breakdown, roll, crash.
    """
    import copy
    rng = random.Random(seed)
    fp = copy.deepcopy(pattern)

    if fill_type == 'buildup':
        # Accelerating snare/tom hits
        for step in range(fp.steps):
            density = step / fp.steps
            if rng.random() < density * 0.6:
                inst = rng.choice(['snare', 'tom'])
                var = rng.choice(GENERATORS[inst]['variants'])
                vel = 50 + density * 50
                fp.add_hit(step, inst, vel, var)
    elif fill_type == 'breakdown':
        # Strip to just kick on 1
        fp.grid = {0: [h for h in fp.grid.get(0, [])
                       if h.instrument == 'kick']}
    elif fill_type == 'roll':
        # Snare roll on last 4 steps
        start = max(0, fp.steps - 4)
        for step in range(start, fp.steps):
            fp.add_hit(step, 'snare', 70 + (step - start) * 8,
                       GENERATORS['snare']['variants'][0])
    elif fill_type == 'crash':
        fp.add_hit(0, 'cymbal', 100, 'crash')

    return fp
