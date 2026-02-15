#!/usr/bin/env python3
"""
MDMA GUI - Phase T: Song-Ready System Audit
==============================================

Full-featured wxPython interface for the MDMA audio engine.

Phase 1 features:
- Hierarchical object tree with live previews
- Inspector panel with full object detail views
- Step grid with playback position and buffer I/O tracking
- Context menus, selection-based FX, accessibility markers

Phase 2 features:
- Monolith Patch Builder panel (operators, routing, inline editing)
- Carrier/Modulator routing visualization (signal flow diagram)
- Oscillator list view with category filter and quick-edit controls
- Full parameter exposure for all Phase 2 wave types
- Extended wave types: supersaw, additive, formant, harmonic,
  waveguide (string/tube/membrane/plate), wavetable, compound

Phase 3 features:
- Impulse-to-LFO waveshape conversion and application
- Impulse-to-envelope amplitude contour extraction
- Advanced convolution reverb with early/late split, stereo width
- Neural-enhanced IR processing (extend, denoise, fill gaps)
- AI-descriptor IR transformation (15 semantic descriptors)
- Granular IR tools (stretch, morph, redesign, freeze)

Phase 4 features:
- Buffer duplication, copy, and import actions
- Text-to-audio generative actions (melody, chords, bass, beats, loops)
- Genetic sample breeding exposure (crossover, mutation, evolution)
- Breeding/mutate dialogs and tree categories
- Step grid click-drag selection highlighting
- Accessible step text field (screen reader character navigation)

Phase T features (Song-Ready):
- Undo/redo for working buffer and tracks
- Parameter snapshot save/restore
- Song section markers (add, list, goto, copy, move)
- Pattern chaining (/pchain) and commit-to-track (/commit)
- Stem export, track export, section export
- Master gain control
- /crossover command wired to genetic breeding engine
- Buffer duplicate (/dup), swap (/swap), metronome (/click)
- Write position display/control (/pos, /seek)
- Auto-save toggle, file FX chain management
- Phase T object tree category with full inspector support

Phase A (Accessibility Audit):
- Context menus accessible via Application key / Shift+F10
- Inspector properties displayed in ListCtrl for screen-reader navigation
- Patch Builder operator & routing lists: context menus, keyboard actions
- Console command input field with history (Up/Down), auto-prefix
- StepGrid: Ctrl+A select all, Up/Down row navigation, Space set write pos
- Oscillator cards: per-operator Wave button, accessible button names
- Accessible names/hints on all interactive controls
- Ctrl+L focuses command input, multi-command newline splitting fixed

Version: 4.2.0
Author: Based on spec by Cyrus
Date: 2026-02-15

Requirements:
    pip install wxPython

Usage:
    python mdma_gui.py

BUILD ID: mdma_gui_v4.2.0_phaseA
"""

import sys
import os

# wxPython import guard — give a clear message instead of a traceback
try:
    import wx
    import wx.adv
    import wx.lib.agw.aui as aui
except ImportError:
    print("MDMA GUI requires wxPython.")
    print("Install with:  pip install wxPython")
    print("\nAlternatives:")
    print("  python run_mdma.py --repl   (terminal, no extra deps)")
    print("  python run_mdma.py --tui    (pip install textual)")
    print("  python run_mdma.py          (auto-detect best interface)")
    sys.exit(1)

import io
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

# Add MDMA to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# THEME / COLORS (Serum-ish muted dark)
# ============================================================================

class Theme:
    """Muted dark theme inspired by Serum."""
    BG_DARK = wx.Colour(30, 30, 35)
    BG_PANEL = wx.Colour(40, 40, 48)
    BG_INPUT = wx.Colour(50, 50, 58)
    FG_TEXT = wx.Colour(220, 220, 225)
    FG_DIM = wx.Colour(180, 180, 190)  # Raised from (140,140,150) for WCAG AA contrast
    ACCENT = wx.Colour(100, 180, 255)
    SUCCESS = wx.Colour(100, 200, 120)
    ERROR = wx.Colour(255, 100, 100)
    WARNING = wx.Colour(255, 200, 100)


# ============================================================================
# ACTION DEFINITIONS
# ============================================================================

@dataclass
class ActionParam:
    """Definition of a parameter for an action."""
    name: str
    label: str
    param_type: str  # 'int', 'float', 'bool', 'enum', 'string', 'file'
    default: Any = None
    choices: List[str] = field(default_factory=list)  # For enum type
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    

@dataclass 
class ActionDef:
    """Definition of an action that can be performed."""
    name: str
    label: str
    command_template: str  # e.g., "/tone {freq} {duration}"
    params: List[ActionParam] = field(default_factory=list)
    description: str = ""


# Actions organized by object type — FULL ENGINE PARITY
ACTIONS: Dict[str, List[ActionDef]] = {
    'engine': [
        ActionDef(
            name='tone',
            label='Generate Tone',
            command_template='/tone {freq} {duration}',
            params=[
                ActionParam('freq', 'Frequency (Hz)', 'float', 440.0, min_val=20, max_val=20000),
                ActionParam('duration', 'Duration (beats)', 'float', 1.0, min_val=0.1, max_val=32),
            ],
            description='Generate a simple tone'
        ),
        ActionDef(
            name='mel',
            label='Generate Melody',
            command_template='/mel {pattern} {root_hz}',
            params=[
                ActionParam('pattern', 'Note Pattern', 'string', '0.4.7'),
                ActionParam('root_hz', 'Root Frequency', 'float', 440.0, min_val=20, max_val=8000),
            ],
            description='Generate a melody from note pattern'
        ),
        ActionDef(
            name='play',
            label='Play Audio',
            command_template='/play',
            params=[],
            description='Play the current audio buffer'
        ),
        ActionDef(
            name='render',
            label='Render to File',
            command_template='/render {filename}',
            params=[
                ActionParam('filename', 'Output File', 'string', 'output.flac'),
            ],
            description='Render audio to file'
        ),
        ActionDef(
            name='bpm',
            label='Set Tempo',
            command_template='/bpm {tempo}',
            params=[
                ActionParam('tempo', 'BPM', 'float', 128.0, min_val=20, max_val=400),
            ],
            description='Set the tempo in beats per minute'
        ),
        ActionDef(
            name='noise',
            label='Generate Noise',
            command_template='/noise {type} {duration}',
            params=[
                ActionParam('type', 'Type', 'enum', 'white', choices=['white', 'pink']),
                ActionParam('duration', 'Duration (beats)', 'float', 1.0, min_val=0.1, max_val=32),
            ],
            description='Generate white or pink noise'
        ),
        ActionDef(
            name='ns',
            label='Note Sequence',
            command_template='/ns {notes}',
            params=[
                ActionParam('notes', 'Notes (e.g. C4 D4 E4)', 'string', 'C4 E4 G4'),
            ],
            description='Generate a note sequence'
        ),
        ActionDef(
            name='version',
            label='Show Version',
            command_template='/version',
            params=[],
            description='Show MDMA version info'
        ),
    ],
    'synth': [
        ActionDef(
            name='wave',
            label='Set Waveform',
            command_template='/wm {waveform}',
            params=[
                ActionParam('waveform', 'Waveform', 'enum', 'sine',
                           choices=['sine', 'triangle', 'saw', 'pulse',
                                    'noise', 'pink',
                                    'physical', 'physical2',
                                    'supersaw', 'additive', 'formant', 'harmonic',
                                    'waveguide_string', 'waveguide_tube',
                                    'waveguide_membrane', 'waveguide_plate',
                                    'wavetable', 'compound']),
            ],
            description='Set the oscillator waveform'
        ),
        ActionDef(
            name='freq',
            label='Set Frequency',
            command_template='/fr {freq}',
            params=[
                ActionParam('freq', 'Frequency (Hz)', 'float', 440.0, min_val=20, max_val=20000),
            ],
            description='Set oscillator frequency'
        ),
        ActionDef(
            name='amp',
            label='Set Amplitude',
            command_template='/amp {level}',
            params=[
                ActionParam('level', 'Amplitude (0-1)', 'float', 0.8, min_val=0, max_val=1),
            ],
            description='Set oscillator amplitude'
        ),
        ActionDef(
            name='pw',
            label='Pulse Width',
            command_template='/pw {width}',
            params=[
                ActionParam('width', 'Pulse Width (0-1)', 'float', 0.5, min_val=0.01, max_val=0.99),
            ],
            description='Set pulse wave width (0.5 = square)'
        ),
        ActionDef(
            name='phys_params',
            label='Physical Model',
            command_template='/phys {even} {odd} {weight} {decay}',
            params=[
                ActionParam('even', 'Even Harmonics', 'int', 8, min_val=0, max_val=32),
                ActionParam('odd', 'Odd Harmonics', 'int', 4, min_val=0, max_val=32),
                ActionParam('weight', 'Even Weight', 'float', 1.0, min_val=0, max_val=5),
                ActionParam('decay', 'Harmonic Decay', 'float', 0.7, min_val=0.1, max_val=1),
            ],
            description='Configure physical model harmonic parameters'
        ),
        ActionDef(
            name='op_select',
            label='Select Operator',
            command_template='/op {index}',
            params=[
                ActionParam('index', 'Operator Index', 'int', 1, min_val=1, max_val=32),
            ],
            description='Select the active operator by index'
        ),
        ActionDef(
            name='tone',
            label='Tone Generator',
            command_template='/tone {freq}',
            params=[
                ActionParam('freq', 'Frequency (Hz)', 'float', 440.0, min_val=20, max_val=20000),
            ],
            description='Play a tone at the specified frequency'
        ),
        ActionDef(
            name='note',
            label='Play Note',
            command_template='/n {note}',
            params=[
                ActionParam('note', 'Note Name', 'enum', 'A4',
                           choices=['C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3',
                                    'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4',
                                    'C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5']),
            ],
            description='Play a note by name (e.g. A4, C#5)'
        ),
        ActionDef(
            name='note_seq',
            label='Note Sequence',
            command_template='/ns {notes}',
            params=[
                ActionParam('notes', 'Notes (comma-sep)', 'str', 'C4,E4,G4'),
            ],
            description='Play a sequence of notes (e.g. C4,E4,G4,C5)'
        ),
        ActionDef(
            name='envelope',
            label='Full Envelope',
            command_template='/env {attack} {decay} {sustain} {release}',
            params=[
                ActionParam('attack', 'Attack (s)', 'float', 0.01, min_val=0, max_val=10),
                ActionParam('decay', 'Decay (s)', 'float', 0.1, min_val=0, max_val=10),
                ActionParam('sustain', 'Sustain (0-1)', 'float', 0.7, min_val=0, max_val=1),
                ActionParam('release', 'Release (s)', 'float', 0.3, min_val=0, max_val=10),
            ],
            description='Set full ADSR envelope in one action'
        ),
        ActionDef(
            name='attack',
            label='Attack',
            command_template='/atk {time}',
            params=[
                ActionParam('time', 'Attack Time (s)', 'float', 0.01, min_val=0, max_val=10),
            ],
            description='Set envelope attack time'
        ),
        ActionDef(
            name='decay',
            label='Decay',
            command_template='/dec {time}',
            params=[
                ActionParam('time', 'Decay Time (s)', 'float', 0.1, min_val=0, max_val=10),
            ],
            description='Set envelope decay time'
        ),
        ActionDef(
            name='sustain',
            label='Sustain',
            command_template='/sus {level}',
            params=[
                ActionParam('level', 'Sustain Level (0-1)', 'float', 0.7, min_val=0, max_val=1),
            ],
            description='Set envelope sustain level'
        ),
        ActionDef(
            name='release',
            label='Release',
            command_template='/rel {time}',
            params=[
                ActionParam('time', 'Release Time (s)', 'float', 0.3, min_val=0, max_val=10),
            ],
            description='Set envelope release time'
        ),
        ActionDef(
            name='phase',
            label='Phase',
            command_template='/ph {phase}',
            params=[
                ActionParam('phase', 'Phase (0-360)', 'float', 0.0, min_val=0, max_val=360),
            ],
            description='Set oscillator phase offset'
        ),
        ActionDef(
            name='carriers',
            label='Carrier Count',
            command_template='/car {count}',
            params=[
                ActionParam('count', 'Carriers', 'int', 4, min_val=1, max_val=32),
            ],
            description='Set the number of carrier operators'
        ),
        ActionDef(
            name='modulators',
            label='Modulator Count',
            command_template='/mod {count}',
            params=[
                ActionParam('count', 'Modulators', 'int', 2, min_val=0, max_val=32),
            ],
            description='Set the number of modulator operators'
        ),
        ActionDef(
            name='detune',
            label='Detune',
            command_template='/dt {cents}',
            params=[
                ActionParam('cents', 'Detune (cents)', 'float', 0.0, min_val=-1200, max_val=1200),
            ],
            description='Detune operator by cents'
        ),
        ActionDef(
            name='stereo',
            label='Stereo Pan',
            command_template='/stereo {pan}',
            params=[
                ActionParam('pan', 'Pan (-1 to 1)', 'float', 0.0, min_val=-1, max_val=1),
            ],
            description='Set operator stereo panning (-1 left, 0 center, 1 right)'
        ),
        ActionDef(
            name='key_track',
            label='Key Tracking',
            command_template='/key {amount}',
            params=[
                ActionParam('amount', 'Amount (0-1)', 'float', 1.0, min_val=0, max_val=4),
            ],
            description='Set keyboard frequency tracking amount'
        ),
        ActionDef(
            name='supersaw',
            label='SuperSaw Params',
            command_template='/ssaw {voices} {spread} {mix}',
            params=[
                ActionParam('voices', 'Voices', 'int', 7, min_val=1, max_val=32),
                ActionParam('spread', 'Spread', 'float', 0.5, min_val=0, max_val=1),
                ActionParam('mix', 'Mix', 'float', 0.75, min_val=0, max_val=1),
            ],
            description='Configure supersaw parameters (voices, detune spread, mix)'
        ),
        ActionDef(
            name='harmonic',
            label='Harmonic Params',
            command_template='/harm {partials} {rolloff}',
            params=[
                ActionParam('partials', 'Partials', 'int', 8, min_val=1, max_val=64),
                ActionParam('rolloff', 'Rolloff', 'float', 1.0, min_val=0.1, max_val=4),
            ],
            description='Configure harmonic oscillator (partials count, rolloff)'
        ),
        ActionDef(
            name='waveguide',
            label='Waveguide Params',
            command_template='/waveguide {damping} {position} {feedback}',
            params=[
                ActionParam('damping', 'Damping', 'float', 0.5, min_val=0, max_val=1),
                ActionParam('position', 'Exciter Pos', 'float', 0.5, min_val=0, max_val=1),
                ActionParam('feedback', 'Feedback', 'float', 0.99, min_val=0, max_val=1),
            ],
            description='Configure waveguide model (damping, exciter position, feedback)'
        ),
        ActionDef(
            name='opinfo',
            label='Operator Info',
            command_template='/opinfo all',
            params=[],
            description='Show detailed info for all operators'
        ),
        ActionDef(
            name='waveinfo',
            label='Wave Type Info',
            command_template='/waveinfo',
            params=[],
            description='List all available wave types and their parameters'
        ),
    ],
    'voice': [
        ActionDef(
            name='voices',
            label='Voice Count',
            command_template='/vc {count}',
            params=[
                ActionParam('count', 'Voices', 'int', 1, min_val=1, max_val=16),
            ],
            description='Set number of voices for unison'
        ),
        ActionDef(
            name='va',
            label='Voice Algorithm',
            command_template='/va {algorithm}',
            params=[
                ActionParam('algorithm', 'Algorithm', 'enum', 'stack',
                           choices=['stack', 'unison', 'wide']),
            ],
            description='Set voice algorithm (stack=classic, unison=phase-random, wide=auto-stereo)'
        ),
        ActionDef(
            name='detune',
            label='Detune',
            command_template='/dt {hz}',
            params=[
                ActionParam('hz', 'Detune (Hz)', 'float', 0, min_val=0, max_val=100),
            ],
            description='Set voice detuning in Hz'
        ),
        ActionDef(
            name='stereo',
            label='Stereo Spread',
            command_template='/stereo {amount}',
            params=[
                ActionParam('amount', 'Spread (0-100)', 'float', 0, min_val=0, max_val=100),
            ],
            description='Set stereo spread width'
        ),
        ActionDef(
            name='vphase',
            label='Phase Spread',
            command_template='/vphase {radians}',
            params=[
                ActionParam('radians', 'Phase (radians)', 'float', 0, min_val=0, max_val=6.28),
            ],
            description='Set per-voice phase offset'
        ),
        ActionDef(
            name='rand',
            label='Random Variation',
            command_template='/rand {amount}',
            params=[
                ActionParam('amount', 'Random (0-100)', 'float', 0, min_val=0, max_val=100),
            ],
            description='Set amplitude/phase randomization'
        ),
        ActionDef(
            name='vmod',
            label='Voice Mod Scale',
            command_template='/vmod {amount}',
            params=[
                ActionParam('amount', 'Mod Scale (0-100)', 'float', 0, min_val=0, max_val=100),
            ],
            description='Set per-voice modulation scaling'
        ),
    ],
    'filter': [
        ActionDef(
            name='filter_type',
            label='Set Filter Type',
            command_template='/ft {filter_type}',
            params=[
                ActionParam('filter_type', 'Filter Type', 'enum', 'lowpass',
                           choices=['lowpass', 'highpass', 'bandpass', 'notch',
                                    'peak', 'ringmod', 'allpass',
                                    'comb_ff', 'comb_fb', 'comb_both',
                                    'analog', 'acid',
                                    'formant_a', 'formant_e', 'formant_i',
                                    'formant_o', 'formant_u',
                                    'lowshelf', 'highshelf',
                                    'moog', 'svf_lp', 'svf_hp', 'svf_bp',
                                    'bitcrush', 'downsample',
                                    'dc_block', 'tilt',
                                    'resonant', 'vocal', 'telephone']),
            ],
            description='Set filter type (30 types available)'
        ),
        ActionDef(
            name='cutoff',
            label='Set Cutoff',
            command_template='/cut {freq}',
            params=[
                ActionParam('freq', 'Cutoff (Hz)', 'float', 4500.0, min_val=20, max_val=20000),
            ],
            description='Set filter cutoff frequency'
        ),
        ActionDef(
            name='resonance',
            label='Set Resonance',
            command_template='/res {amount}',
            params=[
                ActionParam('amount', 'Resonance (0-100)', 'float', 50.0, min_val=0, max_val=100),
            ],
            description='Set filter resonance'
        ),
        ActionDef(
            name='fcount',
            label='Filter Slot Count',
            command_template='/fcount {count}',
            params=[
                ActionParam('count', 'Slots (1-8)', 'int', 1, min_val=1, max_val=8),
            ],
            description='Set number of active filter slots'
        ),
        ActionDef(
            name='fsel',
            label='Select Filter Slot',
            command_template='/fs {slot}',
            params=[
                ActionParam('slot', 'Slot (0-7)', 'int', 0, min_val=0, max_val=7),
            ],
            description='Select active filter slot'
        ),
        ActionDef(
            name='fenable',
            label='Toggle Filter Enable',
            command_template='/fen toggle',
            params=[],
            description='Toggle current filter slot on/off'
        ),
    ],
    'filter_envelope': [
        ActionDef(
            name='fatk',
            label='Filter Attack',
            command_template='/fatk {time}',
            params=[
                ActionParam('time', 'Attack (sec)', 'float', 0.01, min_val=0, max_val=10),
            ],
            description='Set filter envelope attack'
        ),
        ActionDef(
            name='fdec',
            label='Filter Decay',
            command_template='/fdec {time}',
            params=[
                ActionParam('time', 'Decay (sec)', 'float', 0.1, min_val=0, max_val=10),
            ],
            description='Set filter envelope decay'
        ),
        ActionDef(
            name='fsus',
            label='Filter Sustain',
            command_template='/fsus {level}',
            params=[
                ActionParam('level', 'Sustain (0-1)', 'float', 0.8, min_val=0, max_val=1),
            ],
            description='Set filter envelope sustain level'
        ),
        ActionDef(
            name='frel',
            label='Filter Release',
            command_template='/frel {time}',
            params=[
                ActionParam('time', 'Release (sec)', 'float', 0.1, min_val=0, max_val=10),
            ],
            description='Set filter envelope release'
        ),
    ],
    'envelope': [
        ActionDef(
            name='attack',
            label='Set Attack',
            command_template='/atk {time}',
            params=[
                ActionParam('time', 'Attack (sec)', 'float', 0.01, min_val=0, max_val=10),
            ],
            description='Set envelope attack time'
        ),
        ActionDef(
            name='decay',
            label='Set Decay',
            command_template='/dec {time}',
            params=[
                ActionParam('time', 'Decay (sec)', 'float', 0.1, min_val=0, max_val=10),
            ],
            description='Set envelope decay time'
        ),
        ActionDef(
            name='sustain',
            label='Set Sustain',
            command_template='/sus {level}',
            params=[
                ActionParam('level', 'Sustain (0-1)', 'float', 0.8, min_val=0, max_val=1),
            ],
            description='Set envelope sustain level'
        ),
        ActionDef(
            name='release',
            label='Set Release',
            command_template='/rel {time}',
            params=[
                ActionParam('time', 'Release (sec)', 'float', 0.1, min_val=0, max_val=10),
            ],
            description='Set envelope release time'
        ),
        ActionDef(
            name='env_preset',
            label='Envelope Preset',
            command_template='/env {preset}',
            params=[
                ActionParam('preset', 'Preset', 'enum', 'pluck',
                           choices=['pluck', 'pad', 'organ', 'perc', 'slow', 'fast', 'string', 'brass']),
            ],
            description='Load an envelope preset'
        ),
        ActionDef(
            name='venv',
            label='Per-Op Envelope Mode',
            command_template='/venv {level}',
            params=[
                ActionParam('level', 'Level', 'enum', '1',
                           choices=['1', '2']),
            ],
            description='Set envelope editing level (1=global, 2=per-operator)'
        ),
    ],
    'hq': [
        ActionDef(
            name='hq_toggle',
            label='Toggle HQ Mode',
            command_template='/hq {state}',
            params=[
                ActionParam('state', 'State', 'enum', 'on', choices=['on', 'off']),
            ],
            description='Enable/disable high-quality mode'
        ),
        ActionDef(
            name='hq_osc',
            label='HQ Oscillators',
            command_template='/hq osc {state}',
            params=[
                ActionParam('state', 'State', 'enum', 'on', choices=['on', 'off']),
            ],
            description='Toggle band-limited oscillators'
        ),
        ActionDef(
            name='hq_dc',
            label='DC Removal',
            command_template='/hq dc {state}',
            params=[
                ActionParam('state', 'State', 'enum', 'on', choices=['on', 'off']),
            ],
            description='Toggle DC offset removal'
        ),
        ActionDef(
            name='hq_sat',
            label='Saturation',
            command_template='/hq sat {drive}',
            params=[
                ActionParam('drive', 'Drive', 'float', 1.0, min_val=0, max_val=5),
            ],
            description='Set saturation drive amount'
        ),
        ActionDef(
            name='hq_limit',
            label='Limiter',
            command_template='/hq limit {threshold}',
            params=[
                ActionParam('threshold', 'Threshold (dB)', 'float', -1.0, min_val=-20, max_val=0),
            ],
            description='Set limiter threshold'
        ),
        ActionDef(
            name='hq_subsonic',
            label='Subsonic Filter',
            command_template='/hq subsonic {freq}',
            params=[
                ActionParam('freq', 'Cutoff (Hz)', 'float', 20, min_val=1, max_val=100),
            ],
            description='Set subsonic filter cutoff frequency (removes rumble below)'
        ),
        ActionDef(
            name='hq_highend',
            label='High-End Smooth',
            command_template='/hq highend {freq} {db}',
            params=[
                ActionParam('freq', 'Frequency (Hz)', 'float', 16000, min_val=1000, max_val=22000),
                ActionParam('db', 'Reduction (dB)', 'float', -3.0, min_val=-12, max_val=0),
            ],
            description='Set high-end smoothing filter (tames harsh frequencies)'
        ),
        ActionDef(
            name='hq_format',
            label='Export Format',
            command_template='/hq format {format}',
            params=[
                ActionParam('format', 'Format', 'enum', 'wav',
                           choices=['wav', 'flac']),
            ],
            description='Set HQ export format (wav or flac)'
        ),
        ActionDef(
            name='hq_bits',
            label='Bit Depth',
            command_template='/hq bits {bits}',
            params=[
                ActionParam('bits', 'Bits', 'enum', '24',
                           choices=['16', '24', '32']),
            ],
            description='Set HQ export bit depth'
        ),
    ],
    'key': [
        ActionDef(
            name='key_set',
            label='Set Key/Scale',
            command_template='/key {note} {scale}',
            params=[
                ActionParam('note', 'Root Note', 'enum', 'C',
                           choices=['C', 'C#', 'D', 'D#', 'E', 'F',
                                    'F#', 'G', 'G#', 'A', 'A#', 'B']),
                ActionParam('scale', 'Scale', 'enum', 'major',
                           choices=['major', 'minor', 'dorian', 'phrygian',
                                    'lydian', 'mixolydian', 'locrian',
                                    'pentatonic', 'blues', 'harmonic', 'melodic']),
            ],
            description='Set musical key and scale'
        ),
    ],
    'modulation': [
        ActionDef(
            name='fm',
            label='Add FM Routing',
            command_template='/fm {source} {target} {amount}',
            params=[
                ActionParam('source', 'Source Op', 'int', 2, min_val=1, max_val=16),
                ActionParam('target', 'Target Op', 'int', 1, min_val=1, max_val=16),
                ActionParam('amount', 'Amount (0-100)', 'float', 50, min_val=0, max_val=100),
            ],
            description='Add FM modulation routing'
        ),
        ActionDef(
            name='am',
            label='Add AM Routing',
            command_template='/am {source} {target} {amount}',
            params=[
                ActionParam('source', 'Source Op', 'int', 2, min_val=1, max_val=16),
                ActionParam('target', 'Target Op', 'int', 1, min_val=1, max_val=16),
                ActionParam('amount', 'Amount (0-100)', 'float', 50, min_val=0, max_val=100),
            ],
            description='Add AM modulation routing'
        ),
        ActionDef(
            name='rm',
            label='Add RM Routing',
            command_template='/rm {source} {target} {amount}',
            params=[
                ActionParam('source', 'Source Op', 'int', 2, min_val=1, max_val=16),
                ActionParam('target', 'Target Op', 'int', 1, min_val=1, max_val=16),
                ActionParam('amount', 'Amount (0-100)', 'float', 50, min_val=0, max_val=100),
            ],
            description='Add ring modulation routing'
        ),
        ActionDef(
            name='pm',
            label='Add PM Routing',
            command_template='/pm {source} {target} {amount}',
            params=[
                ActionParam('source', 'Source Op', 'int', 2, min_val=1, max_val=16),
                ActionParam('target', 'Target Op', 'int', 1, min_val=1, max_val=16),
                ActionParam('amount', 'Amount (0-100)', 'float', 50, min_val=0, max_val=100),
            ],
            description='Add phase modulation routing'
        ),
        ActionDef(
            name='tfm',
            label='Add TFM Routing',
            command_template='/tfm {source} {target} {amount}',
            params=[
                ActionParam('source', 'Source Op', 'int', 2, min_val=1, max_val=16),
                ActionParam('target', 'Target Op', 'int', 1, min_val=1, max_val=16),
                ActionParam('amount', 'Amount (0-100)', 'float', 50, min_val=0, max_val=100),
            ],
            description='Add through-zero FM routing (harsher, more metallic)'
        ),
        ActionDef(
            name='route_add',
            label='Add Routing (All Types)',
            command_template='/route add {route_type} {source} {target} {amount}',
            params=[
                ActionParam('route_type', 'Type', 'enum', 'fm',
                           choices=['fm', 'tfm', 'am', 'rm', 'pm']),
                ActionParam('source', 'Source Op', 'int', 2, min_val=0, max_val=15),
                ActionParam('target', 'Target Op', 'int', 1, min_val=0, max_val=15),
                ActionParam('amount', 'Amount', 'float', 0.5, min_val=0, max_val=10),
            ],
            description='Add modulation routing (unified — supports all 5 types)'
        ),
        ActionDef(
            name='route_rm',
            label='Remove Routing',
            command_template='/route rm {index}',
            params=[
                ActionParam('index', 'Routing Index', 'int', 0, min_val=0, max_val=32),
            ],
            description='Remove a modulation routing by index'
        ),
        ActionDef(
            name='route_swap',
            label='Swap Routing Order',
            command_template='/route swap {idx1} {idx2}',
            params=[
                ActionParam('idx1', 'Index A', 'int', 0, min_val=0, max_val=32),
                ActionParam('idx2', 'Index B', 'int', 1, min_val=0, max_val=32),
            ],
            description='Swap execution order of two routings'
        ),
        ActionDef(
            name='route_scale',
            label='Scale Routing Amount',
            command_template='/route scale {index} {amount}',
            params=[
                ActionParam('index', 'Routing Index', 'int', 0, min_val=0, max_val=32),
                ActionParam('amount', 'New Amount', 'float', 1.0, min_val=0, max_val=10),
            ],
            description='Change the modulation amount of a routing'
        ),
        ActionDef(
            name='clearalg',
            label='Clear All Routings',
            command_template='/clearalg',
            params=[],
            description='Clear all modulation routings'
        ),
        ActionDef(
            name='ar_interval',
            label='Interval LFO',
            command_template='/imod {op} lfo {rate} {depth} {wave}',
            params=[
                ActionParam('op', 'Operator', 'int', 0, min_val=0, max_val=15),
                ActionParam('rate', 'Rate (Hz)', 'float', 5.0, min_val=0.1, max_val=100),
                ActionParam('depth', 'Depth (semitones)', 'float', 1.0, min_val=0, max_val=24),
                ActionParam('wave', 'LFO Shape', 'enum', 'sine',
                           choices=['sine', 'triangle', 'saw', 'square']),
            ],
            description='Set interval (pitch) LFO on an operator — vibrato / trill'
        ),
        ActionDef(
            name='ar_interval_src',
            label='Interval Mod (Op Source)',
            command_template='/imod {op} src {source_op} {depth}',
            params=[
                ActionParam('op', 'Target Operator', 'int', 0, min_val=0, max_val=15),
                ActionParam('source_op', 'Source Operator', 'int', 1, min_val=0, max_val=15),
                ActionParam('depth', 'Depth (semitones)', 'float', 7.0, min_val=0, max_val=24),
            ],
            description='Use one operator as interval modulation source for another'
        ),
        ActionDef(
            name='ar_interval_off',
            label='Interval Mod Off',
            command_template='/imod {op} off',
            params=[
                ActionParam('op', 'Operator', 'int', 0, min_val=0, max_val=15),
            ],
            description='Disable interval modulation on an operator'
        ),
        ActionDef(
            name='ar_filter',
            label='Filter LFO',
            command_template='/fmod lfo {rate} {depth}',
            params=[
                ActionParam('rate', 'Rate (Hz)', 'float', 2.0, min_val=0.1, max_val=100),
                ActionParam('depth', 'Depth (octaves)', 'float', 1.0, min_val=0, max_val=8),
            ],
            description='Set filter cutoff LFO modulation'
        ),
        ActionDef(
            name='ar_filter_src',
            label='Filter Mod (Op Source)',
            command_template='/fmod src {source_op} {depth}',
            params=[
                ActionParam('source_op', 'Source Operator', 'int', 1, min_val=0, max_val=15),
                ActionParam('depth', 'Depth (octaves)', 'float', 1.5, min_val=0, max_val=8),
            ],
            description='Use an operator as filter modulation source'
        ),
        ActionDef(
            name='ar_filter_op',
            label='Per-Op Filter',
            command_template='/fmod op {op} {cutoff} {resonance}',
            params=[
                ActionParam('op', 'Operator', 'int', 0, min_val=0, max_val=15),
                ActionParam('cutoff', 'Cutoff (Hz)', 'float', 2000, min_val=20, max_val=20000),
                ActionParam('resonance', 'Resonance (0-1)', 'float', 0.5, min_val=0, max_val=1),
            ],
            description='Set per-operator filter with audio-rate modulation'
        ),
        ActionDef(
            name='ar_clear',
            label='Clear Audio-Rate Mod',
            command_template='/imod clear',
            params=[],
            description='Clear all audio-rate interval and filter modulation'
        ),
    ],
    'wavetable': [
        ActionDef(
            name='wt_load',
            label='Load Wavetable',
            command_template='/wt load {name} {path}',
            params=[
                ActionParam('name', 'Name', 'string', 'my_table'),
                ActionParam('path', 'File Path (.wav)', 'string', ''),
            ],
            description='Load a wavetable from .wav file (Serum format)'
        ),
        ActionDef(
            name='wt_use',
            label='Use Wavetable',
            command_template='/wt use {name}',
            params=[
                ActionParam('name', 'Wavetable Name', 'string', ''),
            ],
            description='Set current operator to use a loaded wavetable'
        ),
        ActionDef(
            name='wt_frame',
            label='Set Frame Position',
            command_template='/wt frame {pos}',
            params=[
                ActionParam('pos', 'Position (0-1)', 'float', 0.0, min_val=0, max_val=1),
            ],
            description='Set wavetable frame position'
        ),
        ActionDef(
            name='wt_del',
            label='Delete Wavetable',
            command_template='/wt del {name}',
            params=[
                ActionParam('name', 'Wavetable Name', 'string', ''),
            ],
            description='Delete a loaded wavetable'
        ),
        ActionDef(
            name='wt_list',
            label='List Wavetables',
            command_template='/wt',
            params=[],
            description='List all loaded wavetables'
        ),
    ],
    'compound': [
        ActionDef(
            name='comp_new',
            label='New Compound',
            command_template='/compound new {name}',
            params=[
                ActionParam('name', 'Name', 'string', 'my_compound'),
            ],
            description='Create a new compound wave definition'
        ),
        ActionDef(
            name='comp_add',
            label='Add Layer',
            command_template='/compound add {name} {wave} {detune} {amp}',
            params=[
                ActionParam('name', 'Compound Name', 'string', ''),
                ActionParam('wave', 'Layer Wave', 'enum', 'sine',
                           choices=['sine', 'triangle', 'saw', 'pulse']),
                ActionParam('detune', 'Detune (semitones)', 'float', 0.0, min_val=-24, max_val=24),
                ActionParam('amp', 'Layer Amp', 'float', 1.0, min_val=0, max_val=2),
            ],
            description='Add a layer to a compound wave'
        ),
        ActionDef(
            name='comp_use',
            label='Use Compound',
            command_template='/compound use {name}',
            params=[
                ActionParam('name', 'Compound Name', 'string', ''),
            ],
            description='Set current operator to use a compound wave'
        ),
        ActionDef(
            name='comp_morph',
            label='Set Morph',
            command_template='/compound morph {pos}',
            params=[
                ActionParam('pos', 'Morph (0-1)', 'float', 0.0, min_val=0, max_val=1),
            ],
            description='Set morph position between two layers'
        ),
        ActionDef(
            name='comp_del',
            label='Delete Compound',
            command_template='/compound del {name}',
            params=[
                ActionParam('name', 'Compound Name', 'string', ''),
            ],
            description='Delete a compound wave definition'
        ),
    ],
    'fx': [
        ActionDef(
            name='fx_add',
            label='Add Effect',
            command_template='/fx {effect}',
            params=[
                ActionParam('effect', 'Effect', 'enum', 'reverb',
                           choices=[
                               'reverb', 'r2', 'r3', 'r4', 'r5',
                               'hall', 'room', 'plate', 'spring', 'shimmer', 'reverse',
                               'delay', 'pingpong', 'slapback', 'tape',
                               'saturate', 's2', 'fuzz', 's5',
                               'vamp', 'v1', 'v3', 'od', 'crunch', 'dual',
                               'fold', 'rect',
                               'compress', 'c2', 'limiter', 'expander', 'softclip',
                               'fc_punch', 'fc_glue', 'fc_loud', 'fc_ott',
                               'gate', 'g1', 'g3',
                               'bitcrush', 'chorus', 'flanger', 'phaser', 'halftime',
                               'cloud', 'scatter', 'grstretch', 'grfreeze',
                               'lp', 'hp', 'bp', 'telephone',
                               'pitchup', 'pitchdown', 'octave_up', 'octave_down',
                               'harmonizer', 'h3', 'hchord',
                               'stereo_wide', 'stereo_narrow',
                               'normalize', 'lufs', 'dc', 'fadein', 'fadeout',
                           ]),
            ],
            description='Add an effect to the chain'
        ),
        ActionDef(
            name='fx_params',
            label='Set FX Parameter',
            command_template='/fxp {index} {param} {value}',
            params=[
                ActionParam('index', 'FX Index', 'int', '0'),
                ActionParam('param', 'Parameter', 'string', 'amount'),
                ActionParam('value', 'Value (0-100)', 'float', '50'),
            ],
            description='Set a parameter on an applied effect'
        ),
        ActionDef(
            name='fx_clear',
            label='Clear Effects',
            command_template='/fx clear',
            params=[],
            description='Clear all effects'
        ),
    ],
    'preset': [
        ActionDef(
            name='use_preset',
            label='Use SyDef Preset',
            command_template='/use {name}',
            params=[
                ActionParam('name', 'Preset Name', 'string', 'saw'),
            ],
            description='Load and use a synth definition preset'
        ),
        ActionDef(
            name='list_presets',
            label='List SyDef Presets',
            command_template='/sydef list',
            params=[],
            description='List all available synth definitions'
        ),
        ActionDef(
            name='preset_save',
            label='Save Engine Preset',
            command_template='/preset save {slot} {name}',
            params=[
                ActionParam('slot', 'Slot (0-127)', 'int', 0, min_val=0, max_val=127),
                ActionParam('name', 'Name', 'string', 'my_preset'),
            ],
            description='Save current engine state to preset bank slot'
        ),
        ActionDef(
            name='preset_load',
            label='Load Engine Preset',
            command_template='/preset load {slot}',
            params=[
                ActionParam('slot', 'Slot (0-127)', 'int', 0, min_val=0, max_val=127),
            ],
            description='Load engine state from preset bank slot'
        ),
        ActionDef(
            name='preset_list',
            label='List Engine Presets',
            command_template='/preset',
            params=[],
            description='List all saved engine preset bank slots'
        ),
        ActionDef(
            name='preset_del',
            label='Delete Engine Preset',
            command_template='/preset del {slot}',
            params=[
                ActionParam('slot', 'Slot (0-127)', 'int', 0, min_val=0, max_val=127),
            ],
            description='Delete preset from bank slot'
        ),
    ],
    'bank': [
        ActionDef(
            name='bank_select',
            label='Select Routing Bank',
            command_template='/bk {name}',
            params=[
                ActionParam('name', 'Bank Name', 'string', 'classic_fm'),
            ],
            description='Select a routing/algorithm bank'
        ),
        ActionDef(
            name='bank_list',
            label='List Banks',
            command_template='/bk list',
            params=[],
            description='List all available routing banks'
        ),
        ActionDef(
            name='algo_list',
            label='List Algorithms',
            command_template='/al list',
            params=[],
            description='List algorithms in current bank'
        ),
        ActionDef(
            name='algo_load',
            label='Load Algorithm',
            command_template='/al {index}',
            params=[
                ActionParam('index', 'Algorithm Index', 'int', 0, min_val=0, max_val=31),
            ],
            description='Load an algorithm from current bank'
        ),
    ],

    # ------------------------------------------------------------------
    # Phase 3: Modulation, Impulse & Convolution
    # ------------------------------------------------------------------

    'impulse_lfo': [
        ActionDef(
            name='ilfo_load',
            label='Load Umpulse as LFO',
            command_template='/impulselfo load {name}',
            params=[ActionParam('name', 'Umpulse Name', 'string', '')],
            description='Convert an umpulse into an LFO waveshape'
        ),
        ActionDef(
            name='ilfo_file',
            label='Load File as LFO',
            command_template='/impulselfo file {path}',
            params=[ActionParam('path', 'File Path', 'file', '')],
            description='Load an audio file as LFO waveshape'
        ),
        ActionDef(
            name='ilfo_apply',
            label='Apply Impulse LFO',
            command_template='/impulselfo apply {op} {rate} {depth}',
            params=[
                ActionParam('op', 'Operator', 'int', 0, min_val=0, max_val=15),
                ActionParam('rate', 'Rate (Hz)', 'float', 4.0, min_val=0.01, max_val=100),
                ActionParam('depth', 'Depth (semitones)', 'float', 12.0, min_val=0, max_val=48),
            ],
            description='Apply impulse LFO as interval modulation'
        ),
        ActionDef(
            name='ilfo_filter',
            label='Apply LFO to Filter',
            command_template='/impulselfo filter {rate} {depth}',
            params=[
                ActionParam('rate', 'Rate (Hz)', 'float', 2.0, min_val=0.01, max_val=100),
                ActionParam('depth', 'Depth (octaves)', 'float', 2.0, min_val=0, max_val=8),
            ],
            description='Apply impulse LFO as filter cutoff modulation'
        ),
        ActionDef(
            name='ilfo_clear',
            label='Clear LFO Mod',
            command_template='/impulselfo clear',
            params=[],
            description='Clear all impulse LFO modulation'
        ),
    ],

    'impulse_env': [
        ActionDef(
            name='ienv_load',
            label='Load Umpulse as Envelope',
            command_template='/impenv load {name}',
            params=[ActionParam('name', 'Umpulse Name', 'string', '')],
            description='Extract amplitude envelope from an umpulse'
        ),
        ActionDef(
            name='ienv_file',
            label='Load File as Envelope',
            command_template='/impenv file {path}',
            params=[ActionParam('path', 'File Path', 'file', '')],
            description='Extract amplitude envelope from audio file'
        ),
        ActionDef(
            name='ienv_apply',
            label='Apply to Working Buffer',
            command_template='/impenv apply {duration}',
            params=[
                ActionParam('duration', 'Duration (s)', 'float', 1.0, min_val=0.01, max_val=30),
            ],
            description='Apply impulse envelope to working buffer'
        ),
        ActionDef(
            name='ienv_operator',
            label='Apply to Operator',
            command_template='/impenv operator {op} {duration}',
            params=[
                ActionParam('op', 'Operator', 'int', 0, min_val=0, max_val=15),
                ActionParam('duration', 'Duration (s)', 'float', 1.0, min_val=0.01, max_val=30),
            ],
            description='Set impulse envelope as operator amplitude contour'
        ),
    ],

    'convolution': [
        ActionDef(
            name='conv_load',
            label='Load IR File',
            command_template='/conv load {path}',
            params=[ActionParam('path', 'IR File', 'file', '')],
            description='Load impulse response from WAV file'
        ),
        ActionDef(
            name='conv_preset',
            label='Load IR Preset',
            command_template='/conv preset {preset}',
            params=[
                ActionParam('preset', 'Preset', 'enum', 'hall',
                            choices=['hall', 'hall_long', 'hall_bright', 'hall_dark',
                                     'room', 'room_small', 'room_large',
                                     'plate', 'plate_bright', 'plate_dark',
                                     'spring', 'spring_tight', 'spring_loose',
                                     'shimmer', 'shimmer_fifth',
                                     'reverse', 'reverse_long']),
            ],
            description='Load a built-in IR preset'
        ),
        ActionDef(
            name='conv_apply',
            label='Apply Convolution',
            command_template='/conv apply {wet} {dry}',
            params=[
                ActionParam('wet', 'Wet', 'float', 50, min_val=0, max_val=100),
                ActionParam('dry', 'Dry', 'float', 50, min_val=0, max_val=100),
            ],
            description='Apply convolution reverb to working buffer'
        ),
        ActionDef(
            name='conv_params',
            label='Set Parameters',
            command_template='/conv params wet={wet} dry={dry} pre_delay_ms={pre_delay} decay={decay} stereo_width={width}',
            params=[
                ActionParam('wet', 'Wet', 'float', 50, min_val=0, max_val=100),
                ActionParam('dry', 'Dry', 'float', 50, min_val=0, max_val=100),
                ActionParam('pre_delay', 'Pre-Delay (ms)', 'float', 0, min_val=0, max_val=500),
                ActionParam('decay', 'Decay', 'float', 1.0, min_val=0.1, max_val=5),
                ActionParam('width', 'Stereo Width', 'float', 50, min_val=0, max_val=100),
            ],
            description='Set convolution reverb parameters'
        ),
        ActionDef(
            name='conv_split',
            label='Early/Late Split',
            command_template='/conv split {ms}',
            params=[
                ActionParam('ms', 'Split (ms)', 'float', 80, min_val=10, max_val=500),
            ],
            description='Set early/late reflection split point'
        ),
        ActionDef(
            name='conv_save',
            label='Save IR to Bank',
            command_template='/conv save {name}',
            params=[ActionParam('name', 'Name', 'string', '')],
            description='Save current IR to the bank'
        ),
    ],

    'ir_enhance': [
        ActionDef(
            name='ire_extend',
            label='Extend IR',
            command_template='/irenhance extend {target}',
            params=[
                ActionParam('target', 'Target Duration (s)', 'float', 5.0, min_val=0.1, max_val=30),
            ],
            description='Extend IR tail using neural-inspired decay analysis'
        ),
        ActionDef(
            name='ire_denoise',
            label='Denoise IR',
            command_template='/irenhance denoise {threshold}',
            params=[
                ActionParam('threshold', 'Threshold (dB)', 'float', -60, min_val=-96, max_val=-20),
            ],
            description='Remove noise floor from recorded IR'
        ),
        ActionDef(
            name='ire_fill',
            label='Fill Gaps',
            command_template='/irenhance fill {threshold}',
            params=[
                ActionParam('threshold', 'Gap Threshold (dB)', 'float', -40, min_val=-80, max_val=-10),
            ],
            description='Fill gaps/dropouts in recorded IR'
        ),
    ],

    'ir_transform': [
        ActionDef(
            name='irt_apply',
            label='Transform IR',
            command_template='/irtransform {descriptor} {intensity}',
            params=[
                ActionParam('descriptor', 'Descriptor', 'enum', 'bigger',
                            choices=['bigger', 'smaller', 'brighter', 'darker', 'warmer',
                                     'metallic', 'wooden', 'glass', 'cathedral', 'intimate',
                                     'ethereal', 'haunted', 'telephone', 'underwater', 'vintage']),
                ActionParam('intensity', 'Intensity', 'float', 1.0, min_val=0, max_val=2),
            ],
            description='Transform IR using semantic descriptor'
        ),
        ActionDef(
            name='irt_chain',
            label='Chain Transforms',
            command_template='/irtransform chain {transforms}',
            params=[
                ActionParam('transforms', 'Descriptors (space-separated)', 'string', 'bigger darker'),
            ],
            description='Chain multiple descriptor transformations'
        ),
    ],

    'ir_granular': [
        ActionDef(
            name='irg_stretch',
            label='Granular Stretch',
            command_template='/irgranular stretch {factor} {grain_ms} {density}',
            params=[
                ActionParam('factor', 'Stretch Factor', 'float', 2.0, min_val=0.25, max_val=8),
                ActionParam('grain_ms', 'Grain Size (ms)', 'float', 40, min_val=5, max_val=200),
                ActionParam('density', 'Density', 'float', 8, min_val=1, max_val=32),
            ],
            description='Stretch IR using granular processing'
        ),
        ActionDef(
            name='irg_morph',
            label='Morph IRs',
            command_template='/irgranular morph {ir_a} {ir_b} {position}',
            params=[
                ActionParam('ir_a', 'IR A Name', 'string', ''),
                ActionParam('ir_b', 'IR B Name', 'string', ''),
                ActionParam('position', 'Morph Position', 'float', 0.5, min_val=0, max_val=1),
            ],
            description='Morph between two IRs using granular interleaving'
        ),
        ActionDef(
            name='irg_redesign',
            label='Redesign IR',
            command_template='/irgranular redesign {grain_ms} {density} {scatter} {reverse_prob}',
            params=[
                ActionParam('grain_ms', 'Grain Size (ms)', 'float', 20, min_val=5, max_val=200),
                ActionParam('density', 'Density', 'float', 6, min_val=1, max_val=32),
                ActionParam('scatter', 'Scatter', 'float', 0.3, min_val=0, max_val=1),
                ActionParam('reverse_prob', 'Reverse Probability', 'float', 0.2, min_val=0, max_val=1),
            ],
            description='Redesign IR with granular decomposition and resynthesis'
        ),
        ActionDef(
            name='irg_freeze',
            label='Freeze IR',
            command_template='/irgranular freeze {position}',
            params=[
                ActionParam('position', 'Freeze Position', 'float', 0.5, min_val=0, max_val=1),
            ],
            description='Freeze IR at a specific position'
        ),
    ],

    # ------------------------------------------------------------------
    # Granular Engine (direct /gr access)
    # ------------------------------------------------------------------

    'granular_engine': [
        ActionDef(
            name='gr_process',
            label='Granular Process',
            command_template='/gr process {duration}',
            params=[
                ActionParam('duration', 'Duration (s)', 'float', 2.0, min_val=0.1, max_val=30),
            ],
            description='Process working buffer through granular engine'
        ),
        ActionDef(
            name='gr_freeze',
            label='Granular Freeze',
            command_template='/gr freeze {position} {duration}',
            params=[
                ActionParam('position', 'Position (0-1)', 'float', 0.5, min_val=0, max_val=1),
                ActionParam('duration', 'Duration (s)', 'float', 4.0, min_val=0.1, max_val=30),
            ],
            description='Freeze and sustain grain cloud at a position'
        ),
        ActionDef(
            name='gr_stretch',
            label='Granular Time-Stretch',
            command_template='/gr stretch {factor}',
            params=[
                ActionParam('factor', 'Stretch Factor', 'float', 2.0, min_val=0.25, max_val=8),
            ],
            description='Time-stretch without pitch change (granular)'
        ),
        ActionDef(
            name='gr_shift',
            label='Granular Pitch Shift',
            command_template='/gr shift {semitones}',
            params=[
                ActionParam('semitones', 'Semitones', 'float', 0, min_val=-24, max_val=24),
            ],
            description='Pitch-shift without time change (granular)'
        ),
        ActionDef(
            name='gr_size',
            label='Set Grain Size',
            command_template='/gr size {ms}',
            params=[
                ActionParam('ms', 'Size (ms)', 'float', 50, min_val=1, max_val=500),
            ],
            description='Set grain size in milliseconds'
        ),
        ActionDef(
            name='gr_density',
            label='Set Density',
            command_template='/gr density {value}',
            params=[
                ActionParam('value', 'Density', 'float', 4.0, min_val=0.5, max_val=32),
            ],
            description='Set grain overlap density (higher = denser texture)'
        ),
        ActionDef(
            name='gr_pos',
            label='Set Position',
            command_template='/gr pos {value}',
            params=[
                ActionParam('value', 'Position (0-1)', 'float', 0.5, min_val=0, max_val=1),
            ],
            description='Set read position in source audio'
        ),
        ActionDef(
            name='gr_spread',
            label='Set Spread',
            command_template='/gr spread {value}',
            params=[
                ActionParam('value', 'Spread (0-1)', 'float', 0.1, min_val=0, max_val=1),
            ],
            description='Set random position spread around read position'
        ),
        ActionDef(
            name='gr_pitch',
            label='Set Pitch Ratio',
            command_template='/gr pitch {value}',
            params=[
                ActionParam('value', 'Pitch Ratio', 'float', 1.0, min_val=0.25, max_val=4),
            ],
            description='Set grain pitch playback ratio (1.0 = normal)'
        ),
        ActionDef(
            name='gr_env',
            label='Set Grain Envelope',
            command_template='/gr env {shape}',
            params=[
                ActionParam('shape', 'Shape', 'enum', 'hann',
                           choices=['hann', 'triangle', 'gaussian', 'trapezoid',
                                    'tukey', 'rect']),
            ],
            description='Set grain window envelope shape'
        ),
        ActionDef(
            name='gr_reverse',
            label='Set Reverse Probability',
            command_template='/gr reverse {value}',
            params=[
                ActionParam('value', 'Probability (0-1)', 'float', 0.0, min_val=0, max_val=1),
            ],
            description='Probability of reversed grains (0 = none, 1 = all)'
        ),
        ActionDef(
            name='gr_status',
            label='Granular Status',
            command_template='/gr',
            params=[],
            description='Show current granular engine parameters'
        ),
    ],

    # ------------------------------------------------------------------
    # GPU / AI Settings and Generation
    # ------------------------------------------------------------------

    'gpu': [
        ActionDef(
            name='gpu_steps',
            label='Set Inference Steps',
            command_template='/gpu steps {steps}',
            params=[
                ActionParam('steps', 'Steps', 'int', 150, min_val=1, max_val=500),
            ],
            description='Set AI generation inference steps (more = better quality, slower)'
        ),
        ActionDef(
            name='gpu_cfg',
            label='Set CFG Scale',
            command_template='/gpu cfg {scale}',
            params=[
                ActionParam('scale', 'CFG Scale', 'float', 10, min_val=1, max_val=30),
            ],
            description='Set classifier-free guidance scale (higher = closer to prompt)'
        ),
        ActionDef(
            name='gpu_scheduler',
            label='Set Scheduler',
            command_template='/gpu sk {scheduler}',
            params=[
                ActionParam('scheduler', 'Scheduler', 'enum', '6',
                           choices=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']),
            ],
            description='Set diffusion scheduler (0=DDPM 1=DDIM 4=Euler 6=DPM++ 9=UniPC)'
        ),
        ActionDef(
            name='gpu_model',
            label='Set Model',
            command_template='/gpu model {model}',
            params=[
                ActionParam('model', 'Model', 'enum', '0',
                           choices=['0', '1', '2']),
            ],
            description='Set AI model (0=audioldm2-large 1=music 2=full)'
        ),
        ActionDef(
            name='gpu_device',
            label='Set Device',
            command_template='/gpu device {device}',
            params=[
                ActionParam('device', 'Device', 'enum', 'cuda',
                           choices=['cuda', 'cpu', 'mps']),
            ],
            description='Set compute device (cuda=GPU, cpu=CPU, mps=Apple Silicon)'
        ),
        ActionDef(
            name='gpu_fp16',
            label='Toggle FP16',
            command_template='/gpu fp16 {state}',
            params=[
                ActionParam('state', 'Half Precision', 'enum', 'on',
                           choices=['on', 'off']),
            ],
            description='Enable/disable half-precision for faster GPU inference'
        ),
        ActionDef(
            name='gpu_offload',
            label='CPU Offload',
            command_template='/gpu offload {state}',
            params=[
                ActionParam('state', 'Offload', 'enum', 'off',
                           choices=['on', 'off']),
            ],
            description='Enable CPU offloading to reduce GPU memory usage'
        ),
        ActionDef(
            name='gpu_dur',
            label='Default Duration',
            command_template='/gpu dur {seconds}',
            params=[
                ActionParam('seconds', 'Duration (s)', 'float', 5.0, min_val=0.1, max_val=30),
            ],
            description='Set default generation duration in seconds'
        ),
        ActionDef(
            name='gpu_reset',
            label='Reset GPU Settings',
            command_template='/gpu reset',
            params=[],
            description='Reset all GPU/AI settings to defaults'
        ),
        ActionDef(
            name='gpu_status',
            label='GPU Status',
            command_template='/gpu',
            params=[],
            description='Show current GPU/AI configuration'
        ),
    ],

    'ai_generate': [
        ActionDef(
            name='gen_audio',
            label='Generate from Prompt',
            command_template='/gen {prompt} {duration}',
            params=[
                ActionParam('prompt', 'Text Prompt', 'string', 'warm ambient pad'),
                ActionParam('duration', 'Duration (s)', 'float', 5.0, min_val=0.1, max_val=30),
            ],
            description='Generate audio from text description using AI (requires GPU)'
        ),
        ActionDef(
            name='gen_variations',
            label='Generate Variations',
            command_template='/genv {prompt} {count} {duration}',
            params=[
                ActionParam('prompt', 'Text Prompt', 'string', 'warm ambient pad'),
                ActionParam('count', 'Variations', 'int', 3, min_val=1, max_val=8),
                ActionParam('duration', 'Duration (s)', 'float', 5.0, min_val=0.1, max_val=30),
            ],
            description='Generate multiple variations from text prompt'
        ),
        ActionDef(
            name='ai_analyze',
            label='Analyze Audio',
            command_template='/analyze {mode}',
            params=[
                ActionParam('mode', 'Mode', 'enum', 'detailed',
                           choices=['detailed', 'brief']),
            ],
            description='AI analysis of working buffer attributes'
        ),
        ActionDef(
            name='ai_describe',
            label='Describe Audio',
            command_template='/describe',
            params=[],
            description='Generate semantic description of working buffer'
        ),
        ActionDef(
            name='ai_ask',
            label='Ask AI (Natural Language)',
            command_template='/ask {request}',
            params=[
                ActionParam('request', 'Request', 'string',
                           'make a dark ambient pad in D minor'),
            ],
            description='Natural language request — AI plans and suggests commands'
        ),
    ],

    # ------------------------------------------------------------------
    # Phase 4+: Buffer Operations, Text-to-Audio, Genetic Breeding
    # ------------------------------------------------------------------

    'buffers': [
        ActionDef(
            name='buf_play',
            label='Play Buffer',
            command_template='/pb {index}',
            params=[
                ActionParam('index', 'Buffer Index', 'int', 1, min_val=1, max_val=10),
            ],
            description='Play a numbered buffer'
        ),
        ActionDef(
            name='buf_play_working',
            label='Play Working Buffer',
            command_template='/p',
            params=[],
            description='Play the working buffer'
        ),
        ActionDef(
            name='buf_duplicate',
            label='Duplicate Buffer',
            command_template='/w {source}\n/a',
            params=[
                ActionParam('source', 'Source Buffer', 'int', 1, min_val=1, max_val=10),
            ],
            description='Duplicate a buffer — loads source into working, then appends to next slot'
        ),
        ActionDef(
            name='buf_copy_to_track',
            label='Buffer to Track',
            command_template='/w {source}\n/ta',
            params=[
                ActionParam('source', 'Source Buffer', 'int', 1, min_val=1, max_val=10),
            ],
            description='Copy a buffer to the current track'
        ),
        ActionDef(
            name='buf_bounce_track',
            label='Bounce Track to Working',
            command_template='/btw {track}',
            params=[
                ActionParam('track', 'Track Number', 'int', 1, min_val=1, max_val=16),
            ],
            description='Bounce a track into the working buffer for editing'
        ),
        ActionDef(
            name='buf_bounce_back',
            label='Bounce Working Back',
            command_template='/btw back',
            params=[],
            description='Write working buffer back to the track it was bounced from'
        ),
        ActionDef(
            name='buf_clear',
            label='Clear Buffer',
            command_template='/clr {index}',
            params=[
                ActionParam('index', 'Buffer Index (0=working)', 'int', 0, min_val=0, max_val=10),
            ],
            description='Clear a buffer (0 clears working buffer)'
        ),
        ActionDef(
            name='buf_info',
            label='Buffer Info',
            command_template='/b',
            params=[],
            description='Show buffer overview with durations and sample counts'
        ),
        ActionDef(
            name='buf_import',
            label='Import Audio File',
            command_template='/import {path}',
            params=[
                ActionParam('path', 'File Path', 'file', ''),
            ],
            description='Import a WAV file into the working buffer'
        ),
        ActionDef(
            name='buf_import_to_track',
            label='Import to Track',
            command_template='/import {path} track',
            params=[
                ActionParam('path', 'File Path', 'file', ''),
            ],
            description='Import a WAV file directly to the current track'
        ),
        ActionDef(
            name='buf_stretch',
            label='Time-Stretch',
            command_template='/stretch {factor}',
            params=[
                ActionParam('factor', 'Factor', 'float', 2.0, min_val=0.1, max_val=8),
            ],
            description='Time-stretch working buffer (2.0 = double length, 0.5 = half)'
        ),
        ActionDef(
            name='buf_swap',
            label='Swap Buffers',
            command_template='/swap {buf1} {buf2}',
            params=[
                ActionParam('buf1', 'Buffer A', 'int', 1, min_val=1, max_val=10),
                ActionParam('buf2', 'Buffer B', 'int', 2, min_val=1, max_val=10),
            ],
            description='Swap contents of two buffers'
        ),
        ActionDef(
            name='buf_dup',
            label='Copy Buffer to Next Slot',
            command_template='/dup {index}',
            params=[
                ActionParam('index', 'Source Buffer', 'int', 1, min_val=1, max_val=10),
            ],
            description='Duplicate buffer to next empty slot'
        ),
    ],

    'text_to_audio': [
        ActionDef(
            name='gen2_melody',
            label='Generate Melody',
            command_template='/gen2 melody {scale} {length}',
            params=[
                ActionParam('scale', 'Scale', 'enum', 'minor',
                            choices=['major', 'minor', 'dorian', 'phrygian', 'lydian',
                                     'mixolydian', 'pentatonic_major', 'pentatonic_minor',
                                     'blues', 'harmonic_minor', 'melodic_minor',
                                     'whole_tone', 'japanese', 'arabic', 'hungarian_minor']),
                ActionParam('length', 'Note Count', 'int', 8, min_val=2, max_val=64),
            ],
            description='Generate a melodic sequence from a scale with contour shaping'
        ),
        ActionDef(
            name='gen2_chords',
            label='Generate Chord Progression',
            command_template='/gen2 chord_prog {progression} {bars}',
            params=[
                ActionParam('progression', 'Progression', 'enum', 'I_V_vi_IV',
                            choices=['I_IV_V', 'I_V_vi_IV', 'ii_V_I', 'I_vi_IV_V',
                                     'vi_IV_I_V', 'I_IV_vi_V', 'i_iv_v',
                                     'i_VI_III_VII', 'i_iv_VII_III', 'i_VII_VI_V', '12bar']),
                ActionParam('bars', 'Bars', 'int', 4, min_val=1, max_val=32),
            ],
            description='Render a chord progression with voice-led voicings'
        ),
        ActionDef(
            name='gen2_bassline',
            label='Generate Bassline',
            command_template='/gen2 bassline {scale} {bars}',
            params=[
                ActionParam('scale', 'Scale', 'enum', 'minor',
                            choices=['major', 'minor', 'dorian', 'mixolydian',
                                     'pentatonic_minor', 'blues']),
                ActionParam('bars', 'Bars', 'int', 4, min_val=1, max_val=32),
            ],
            description='Generate a genre-aware bassline pattern'
        ),
        ActionDef(
            name='gen2_arpeggio',
            label='Generate Arpeggio',
            command_template='/gen2 arpeggio {chord} {octaves}',
            params=[
                ActionParam('chord', 'Chord Type', 'enum', 'min7',
                            choices=['maj', 'min', 'dim', 'aug', 'maj7', 'min7',
                                     'dom7', 'sus2', 'sus4', 'add9', 'min9', 'maj9']),
                ActionParam('octaves', 'Octaves', 'int', 2, min_val=1, max_val=4),
            ],
            description='Arpeggiate a chord type across octaves'
        ),
        ActionDef(
            name='gen2_drone',
            label='Generate Drone',
            command_template='/gen2 drone {duration}',
            params=[
                ActionParam('duration', 'Duration (beats)', 'int', 8, min_val=1, max_val=64),
            ],
            description='Generate an ambient drone with detuned oscillators'
        ),
        ActionDef(
            name='gen_beat',
            label='Generate Beat',
            command_template='/beat {genre} {bars}',
            params=[
                ActionParam('genre', 'Genre', 'enum', 'house',
                            choices=['house', 'techno', 'hiphop', 'trap', 'dnb',
                                     'lofi', 'reggaeton', 'breakbeat', 'dubstep',
                                     'afrobeat', 'minimal']),
                ActionParam('bars', 'Bars', 'int', 4, min_val=1, max_val=32),
            ],
            description='Generate a drum beat from genre template'
        ),
        ActionDef(
            name='gen_loop',
            label='Generate Loop',
            command_template='/loop {genre} {bars}',
            params=[
                ActionParam('genre', 'Genre', 'enum', 'house',
                            choices=['house', 'techno', 'hiphop', 'trap', 'dnb',
                                     'lofi', 'reggaeton', 'breakbeat', 'dubstep',
                                     'afrobeat', 'minimal']),
                ActionParam('bars', 'Bars', 'int', 4, min_val=1, max_val=32),
            ],
            description='Generate a multi-layer loop (drums + bass + chords + melody)'
        ),
        ActionDef(
            name='gen_xform',
            label='Apply Transform',
            command_template='/xform {transform}',
            params=[
                ActionParam('transform', 'Transform', 'enum', 'retrograde',
                            choices=['retrograde', 'inversion', 'augmentation',
                                     'diminution', 'permute', 'motivic',
                                     'pitch_shift', 'stutter', 'chop',
                                     'granular_freeze']),
            ],
            description='Apply a musical transformation to the working buffer'
        ),
        ActionDef(
            name='gen_adapt_key',
            label='Adapt Key',
            command_template='/adapt key {target_key} {target_scale}',
            params=[
                ActionParam('target_key', 'Target Key', 'enum', 'C',
                            choices=['C', 'C#', 'D', 'D#', 'E', 'F',
                                     'F#', 'G', 'G#', 'A', 'A#', 'B']),
                ActionParam('target_scale', 'Target Scale', 'enum', 'minor',
                            choices=['major', 'minor', 'dorian', 'phrygian',
                                     'lydian', 'mixolydian', 'pentatonic_major',
                                     'pentatonic_minor', 'blues']),
            ],
            description='Adapt existing pattern to a new key and scale'
        ),
    ],

    'breeding': [
        ActionDef(
            name='breed_buffers',
            label='Breed Two Buffers',
            command_template='/breed {buffer_a} {buffer_b} {children}',
            params=[
                ActionParam('buffer_a', 'Parent A (buffer #)', 'int', 1, min_val=1, max_val=10),
                ActionParam('buffer_b', 'Parent B (buffer #)', 'int', 2, min_val=1, max_val=10),
                ActionParam('children', 'Number of Children', 'int', 4, min_val=1, max_val=16),
            ],
            description='Genetically breed two audio buffers — produces children using crossover and mutation'
        ),
        ActionDef(
            name='breed_targeted',
            label='Targeted Breed',
            command_template='/breed {buffer_a} {buffer_b} targeted {attribute}',
            params=[
                ActionParam('buffer_a', 'Parent A (buffer #)', 'int', 1, min_val=1, max_val=10),
                ActionParam('buffer_b', 'Parent B (buffer #)', 'int', 2, min_val=1, max_val=10),
                ActionParam('attribute', 'Target Attribute', 'enum', 'brightness',
                            choices=['brightness', 'warmth', 'roughness', 'noisiness',
                                     'attack', 'sustain', 'spectral_centroid',
                                     'rms_energy']),
            ],
            description='Breed with fitness targeting — children optimized toward a sonic attribute'
        ),
        ActionDef(
            name='evolve_buffer',
            label='Evolve Population',
            command_template='/evolve {generations} {population}',
            params=[
                ActionParam('generations', 'Generations', 'int', 10, min_val=1, max_val=100),
                ActionParam('population', 'Population Size', 'int', 8, min_val=4, max_val=32),
            ],
            description='Run multi-generation evolution on buffer population with tournament selection'
        ),
        ActionDef(
            name='mutate_buffer',
            label='Mutate Buffer',
            command_template='/mutate {mutation} {amount}',
            params=[
                ActionParam('mutation', 'Mutation Type', 'enum', 'noise',
                            choices=['noise', 'pitch', 'time_stretch', 'freq_shift',
                                     'envelope', 'reverse_segment', 'spectral_smear']),
                ActionParam('amount', 'Amount (1-100)', 'int', 30, min_val=1, max_val=100),
            ],
            description='Apply a mutation operation to the working buffer'
        ),
        ActionDef(
            name='crossover_buffers',
            label='Crossover Two Buffers',
            command_template='/crossover {buffer_a} {buffer_b} {method}',
            params=[
                ActionParam('buffer_a', 'Parent A (buffer #)', 'int', 1, min_val=1, max_val=10),
                ActionParam('buffer_b', 'Parent B (buffer #)', 'int', 2, min_val=1, max_val=10),
                ActionParam('method', 'Crossover Method', 'enum', 'spectral',
                            choices=['temporal', 'spectral', 'blend',
                                     'morphological', 'multi_point']),
            ],
            description='Cross two buffers using a specific crossover method — result goes to working buffer'
        ),
        ActionDef(
            name='breed_config',
            label='Breeding Config',
            command_template='/breed config crossover_prob={crossover_prob} mutation_prob={mutation_prob} elite={elite}',
            params=[
                ActionParam('crossover_prob', 'Crossover Probability', 'float', 0.8, min_val=0, max_val=1),
                ActionParam('mutation_prob', 'Mutation Probability', 'float', 0.15, min_val=0, max_val=1),
                ActionParam('elite', 'Elite Count', 'int', 2, min_val=0, max_val=8),
            ],
            description='Configure genetic breeding parameters'
        ),
    ],

    # -------------------------------------------------------------------
    # Phase T — Song-Ready Tools
    # -------------------------------------------------------------------
    'phase_t_undo': [
        ActionDef(
            name='undo_working',
            label='Undo Working Buffer',
            command_template='/undo',
            params=[],
            description='Undo last operation on working buffer'
        ),
        ActionDef(
            name='redo_working',
            label='Redo Working Buffer',
            command_template='/redo',
            params=[],
            description='Redo previously undone operation'
        ),
        ActionDef(
            name='undo_track',
            label='Undo Track',
            command_template='/undo track {track_n}',
            params=[
                ActionParam('track_n', 'Track Number', 'int', 1, min_val=1, max_val=16),
            ],
            description='Undo last operation on a track'
        ),
        ActionDef(
            name='snapshot_save',
            label='Save Snapshot',
            command_template='/snapshot save',
            params=[],
            description='Save current session parameters as a snapshot'
        ),
        ActionDef(
            name='snapshot_restore',
            label='Restore Snapshot',
            command_template='/snapshot restore {index}',
            params=[
                ActionParam('index', 'Snapshot Index', 'int', -1),
            ],
            description='Restore session parameters from a saved snapshot'
        ),
    ],
    'phase_t_structure': [
        ActionDef(
            name='section_add',
            label='Add Song Section',
            command_template='/section add {name} {start_bar} {end_bar}',
            params=[
                ActionParam('name', 'Section Name', 'text', 'intro'),
                ActionParam('start_bar', 'Start Bar', 'int', 0, min_val=0),
                ActionParam('end_bar', 'End Bar', 'int', 8, min_val=1),
            ],
            description='Define a named song section by bar range'
        ),
        ActionDef(
            name='section_list',
            label='List Sections',
            command_template='/section list',
            params=[],
            description='Show all defined song sections'
        ),
        ActionDef(
            name='section_copy',
            label='Copy Section',
            command_template='/section copy {name} {to_bar}',
            params=[
                ActionParam('name', 'Section Name', 'text', 'intro'),
                ActionParam('to_bar', 'Destination Bar', 'int', 16, min_val=0),
            ],
            description='Copy a section to a new bar position on all tracks'
        ),
        ActionDef(
            name='pchain',
            label='Chain Patterns',
            command_template='/pchain {chain_spec}',
            params=[
                ActionParam('chain_spec', 'Chain Spec (buf repeat pairs)', 'text', '1 4 2 2'),
            ],
            description='Chain buffer patterns into a sequence on the current track'
        ),
        ActionDef(
            name='commit_working',
            label='Commit to Track',
            command_template='/commit',
            params=[],
            description='Write working buffer to current track and advance write position'
        ),
        ActionDef(
            name='set_position',
            label='Set Write Position',
            command_template='/pos {position}',
            params=[
                ActionParam('position', 'Position (seconds or Nb for bars)', 'text', '0'),
            ],
            description='Set the write cursor position on the current track'
        ),
    ],
    'phase_t_export': [
        ActionDef(
            name='export_stems',
            label='Export Stems',
            command_template='/export stems',
            params=[],
            description='Export each track as a separate WAV file'
        ),
        ActionDef(
            name='export_track',
            label='Export Track',
            command_template='/export track {track_n}',
            params=[
                ActionParam('track_n', 'Track Number', 'int', 1, min_val=1, max_val=16),
            ],
            description='Export a single track to a WAV file'
        ),
        ActionDef(
            name='export_section',
            label='Export Section',
            command_template='/export section {name}',
            params=[
                ActionParam('name', 'Section Name', 'text', 'intro'),
            ],
            description='Render and export a named song section'
        ),
        ActionDef(
            name='master_gain',
            label='Master Gain',
            command_template='/master_gain {db}',
            params=[
                ActionParam('db', 'Gain (dB)', 'float', 0.0, min_val=-24, max_val=12),
            ],
            description='Set master output gain in decibels'
        ),
    ],
    'phase_t_tools': [
        ActionDef(
            name='crossover_tool',
            label='Crossover Buffers',
            command_template='/crossover {buffer_a} {buffer_b} {method}',
            params=[
                ActionParam('buffer_a', 'Buffer A', 'int', 1, min_val=1),
                ActionParam('buffer_b', 'Buffer B', 'int', 2, min_val=1),
                ActionParam('method', 'Method', 'choice', 'temporal',
                            choices=['temporal', 'spectral', 'blend', 'morphological', 'multi_point']),
            ],
            description='Genetically crossover two buffers into a new child'
        ),
        ActionDef(
            name='dup_buffer',
            label='Duplicate Buffer',
            command_template='/dup {source}',
            params=[
                ActionParam('source', 'Source Buffer', 'int', 1, min_val=1),
            ],
            description='Duplicate a buffer to the next empty slot'
        ),
        ActionDef(
            name='swap_buffers',
            label='Swap Buffers',
            command_template='/swap {a} {b}',
            params=[
                ActionParam('a', 'Buffer A', 'int', 1, min_val=1),
                ActionParam('b', 'Buffer B', 'int', 2, min_val=1),
            ],
            description='Swap the contents of two buffers'
        ),
        ActionDef(
            name='metronome',
            label='Metronome',
            command_template='/metronome {bars}',
            params=[
                ActionParam('bars', 'Bar Count', 'int', 4, min_val=1, max_val=64),
            ],
            description='Generate a click track in the working buffer'
        ),
    ],
}


# ============================================================================
# COMMAND EXECUTOR
# ============================================================================

class CommandExecutor:
    """Executes MDMA commands and captures output."""

    def __init__(self):
        self.session = None
        self.commands = None
        self._init_engine()

    def _init_engine(self):
        """Initialize the MDMA engine."""
        try:
            from mdma_rebuild.core.session import Session
            import bmdma

            self.session = Session()
            self.commands = bmdma.build_command_table()

            # Load factory presets
            try:
                from mdma_rebuild.commands.sydef_cmds import load_factory_presets
                load_factory_presets(self.session)
            except (ImportError, AttributeError) as e:
                print(f"Note: Factory presets not loaded: {e}")
            except Exception as e:
                print(f"Warning: Error loading factory presets: {e}")

            self.init_error = None

        except ImportError as e:
            self.init_error = str(e)
            print(f"Warning: Could not import MDMA engine: {e}")
            self.session = None
            self.commands = {}

    def execute(self, command: str) -> tuple:
        """Execute a command and return (stdout, stderr, success)."""
        if not command.startswith('/'):
            command = '/' + command

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        success = True

        try:
            # Parse command
            parts = command[1:].split()
            if not parts:
                return "", "Empty command", False

            cmd = parts[0].lower()
            args = parts[1:]

            # Look up and execute
            if self.commands and cmd in self.commands:
                func = self.commands[cmd]
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    result = func(self.session, args)
                if result:
                    stdout_capture.write(str(result) + "\n")
            else:
                stderr_capture.write(f"Unknown command: {cmd}\n")
                success = False

        except Exception as e:
            stderr_capture.write(f"Error: {e}\n")
            success = False

        return stdout_capture.getvalue(), stderr_capture.getvalue(), success

    # ------------------------------------------------------------------
    # Live state queries — read directly from the Session object
    # ------------------------------------------------------------------

    def get_engine_state(self) -> Dict[str, Any]:
        """Return a snapshot of key engine state for GUI display."""
        s = self.session
        if not s:
            return {}

        op_idx = getattr(s, 'current_operator', 0)
        current_op = {}
        if hasattr(s, 'engine') and hasattr(s.engine, 'operators'):
            current_op = s.engine.operators.get(op_idx, {})
        filt_slot = getattr(s, 'selected_filter', 0)
        buf_len = len(s.working_buffer) if s.working_buffer is not None else 0

        # Defensive access for filter attributes
        filter_types = getattr(s, 'filter_types', {})
        filter_names = getattr(s, 'filter_type_names', {})
        filter_cutoffs = getattr(s, 'filter_cutoffs', {})
        filter_resonances = getattr(s, 'filter_resonances', {})

        return {
            'bpm': getattr(s, 'bpm', 128),
            'sample_rate': getattr(s, 'sample_rate', 48000),
            'buffer_samples': buf_len,
            'buffer_seconds': buf_len / s.sample_rate if buf_len else 0,
            'current_operator': op_idx,
            'waveform': current_op.get('wave', '---'),
            'frequency': current_op.get('freq', 0),
            'amplitude': current_op.get('amp', 0),
            'voice_count': getattr(s, 'voice_count', 1),
            'voice_algorithm': getattr(s, 'voice_algorithm', 0),
            'detune': getattr(s, 'dt', 0),
            'filter_type': filter_names.get(
                filter_types.get(filt_slot, 0), 'lowpass'),
            'filter_cutoff': filter_cutoffs.get(filt_slot, 4500.0),
            'filter_resonance': filter_resonances.get(filt_slot, 50.0),
            'attack': getattr(s, 'attack', 0.01),
            'decay': getattr(s, 'decay', 0.1),
            'sustain': getattr(s, 'sustain', 0.8),
            'release': getattr(s, 'release', 0.1),
            'hq_mode': getattr(s, 'hq_mode', False),
            'output_format': getattr(s, 'output_format', 'wav'),
            'output_bit_depth': getattr(s, 'output_bit_depth', 16),
            'track_count': len(getattr(s, 'tracks', [])),
            'effects': list(getattr(s, 'effects', [])),
            'effect_params': list(getattr(s, 'effect_params', [])),
            'selected_effect': getattr(s, 'selected_effect', -1),
            'effects_bypassed': getattr(s, 'effects_bypassed', False),
            # Voice params
            'stereo_spread': getattr(s, 'stereo_spread', 0),
            'phase_spread': getattr(s, 'phase_spread', 0),
            'rand': getattr(s, 'rand', 0),
            'v_mod': getattr(s, 'v_mod', 0),
            # Filter envelope
            'filter_attack': getattr(s, 'filter_attack', 0.01),
            'filter_decay': getattr(s, 'filter_decay', 0.1),
            'filter_sustain': getattr(s, 'filter_sustain', 0.8),
            'filter_release': getattr(s, 'filter_release', 0.1),
            'filter_count': getattr(s, 'filter_count', 1),
            # HQ sub-params
            'hq_oscillators': getattr(s, 'hq_oscillators', False),
            'hq_dc_removal': getattr(s, 'hq_dc_removal', False),
            'hq_saturation': getattr(s, 'hq_saturation', 0),
            'hq_limiter': getattr(s, 'hq_limiter', 0),
            # Key / scale
            'key_note': getattr(s, 'key_note', 'C'),
            'key_scale': getattr(s, 'key_scale', 'major'),
        }

    def _get_sydefs(self) -> dict:
        """Get SyDef definitions from wherever they live.

        SyDefs may be stored in session.sydefs, or in the sydef_cmds
        module-level dict, or on the engine preset_bank.
        """
        s = self.session
        if not s:
            return {}
        # Prefer session attribute
        if hasattr(s, 'sydefs') and s.sydefs:
            return s.sydefs
        # Fall back to sydef_cmds module globals
        try:
            from mdma_rebuild.commands import sydef_cmds
            sydefs = getattr(sydef_cmds, '_SYDEFS', None)
            if sydefs:
                return sydefs
            sydefs = getattr(sydef_cmds, 'SYDEFS', None)
            if sydefs:
                return sydefs
        except (ImportError, AttributeError):
            pass
        # Fall back to engine preset bank
        if hasattr(s, 'engine') and hasattr(s.engine, 'preset_bank'):
            return {str(k): v for k, v in s.engine.preset_bank.items()}
        return {}

    def get_dynamic_tree_data(self) -> Dict[str, List[str]]:
        """Return lists of live objects for the browser tree."""
        s = self.session
        if not s:
            return {}

        # Tracks
        tracks_list = getattr(s, 'tracks', [])
        tracks = [t.get('name', f'track_{i+1}') for i, t in enumerate(tracks_list)]

        # Filled buffers
        buffers_dict = getattr(s, 'buffers', {})
        filled = sorted(k for k, v in buffers_dict.items()
                        if v is not None and len(v) > 0)
        buffers = [f"Buffer {i}" for i in filled] if filled else ["(empty)"]

        # Decks
        decks_dict = getattr(s, 'decks', {})
        decks = [f"Deck {i}" for i in sorted(decks_dict.keys())] if decks_dict else []

        # Effects chain
        effects_list = getattr(s, 'effects', [])
        effects = list(effects_list) if effects_list else ["(none)"]

        # Presets / sydefs
        sydefs = self._get_sydefs()
        presets = sorted(sydefs.keys()) if sydefs else ["(none)"]

        # User functions
        user_funcs = getattr(s, 'user_functions', {})
        funcs = sorted(user_funcs.keys()) if user_funcs else []

        # Chains
        chains_dict = getattr(s, 'chains', {})
        chains = sorted(chains_dict.keys()) if chains_dict else []

        # Operators
        operators = []
        if hasattr(s, 'engine') and hasattr(s.engine, 'operators'):
            for idx in sorted(s.engine.operators.keys()):
                op = s.engine.operators[idx]
                wave = op.get('wave', 'sine')
                freq = op.get('freq', 440)
                operators.append(f"Op {idx}: {wave} @ {freq:.0f}Hz")

        return {
            'tracks': tracks,
            'buffers': buffers,
            'decks': decks,
            'effects': effects,
            'presets': presets,
            'user_functions': funcs,
            'chains': chains,
            'operators': operators,
        }

    # ------------------------------------------------------------------
    # Rich object queries for Phase 1 inspector / tree previews
    # ------------------------------------------------------------------

    def get_track_details(self, index: int) -> Dict[str, Any]:
        """Return detailed info for a single track."""
        import numpy as np
        s = self.session
        tracks = getattr(s, 'tracks', []) if s else []
        if not tracks or index >= len(tracks):
            return {}
        t = tracks[index]
        audio = t.get('audio')
        duration = 0.0
        peak_db = -100.0
        channels = 0
        if audio is not None and len(audio) > 0:
            duration = len(audio) / s.sample_rate
            peak = float(np.max(np.abs(audio)))
            peak_db = 20 * np.log10(peak) if peak > 0 else -100.0
            channels = audio.shape[1] if audio.ndim == 2 else 1
        return {
            'index': index,
            'name': t.get('name', f'track_{index+1}'),
            'duration': duration,
            'peak_db': peak_db,
            'channels': channels,
            'gain': t.get('gain', 1.0),
            'pan': t.get('pan', 0.0),
            'mute': t.get('mute', False),
            'solo': t.get('solo', False),
            'fx_chain': [fx[0] if isinstance(fx, (list, tuple)) else str(fx)
                         for fx in t.get('fx_chain', [])],
            'write_pos': t.get('write_pos', 0),
            'write_pos_sec': t.get('write_pos', 0) / s.sample_rate,
            'has_audio': duration > 0.01,
        }

    def get_buffer_details(self, index: int) -> Dict[str, Any]:
        """Return detailed info for a single buffer."""
        import numpy as np
        s = self.session
        if not s:
            return {}
        audio = getattr(s, 'buffers', {}).get(index)
        if audio is None or len(audio) == 0:
            return {'index': index, 'empty': True, 'duration': 0, 'channels': 0,
                    'peak_db': -100.0}
        duration = len(audio) / s.sample_rate
        peak = float(np.max(np.abs(audio)))
        peak_db = 20 * np.log10(peak) if peak > 0 else -100.0
        channels = audio.shape[1] if audio.ndim == 2 else 1
        return {
            'index': index,
            'empty': False,
            'duration': duration,
            'peak_db': peak_db,
            'channels': channels,
            'samples': len(audio),
        }

    def get_working_buffer_details(self) -> Dict[str, Any]:
        """Return detailed info for the working buffer."""
        import numpy as np
        s = self.session
        if not s:
            return {}
        audio = s.working_buffer
        source = getattr(s, 'working_buffer_source', 'init')
        if audio is None or len(audio) == 0:
            return {'empty': True, 'duration': 0, 'channels': 0,
                    'peak_db': -100.0, 'source': source}
        duration = len(audio) / s.sample_rate
        peak = float(np.max(np.abs(audio)))
        peak_db = 20 * np.log10(peak) if peak > 0 else -100.0
        channels = audio.shape[1] if audio.ndim == 2 else 1
        has_real = hasattr(s, 'has_real_working_audio') and s.has_real_working_audio()
        return {
            'empty': not has_real,
            'duration': duration,
            'peak_db': peak_db,
            'channels': channels,
            'source': source,
            'samples': len(audio),
        }

    def get_deck_details(self, index: int) -> Dict[str, Any]:
        """Return detailed info for a single deck."""
        s = self.session
        decks = getattr(s, 'decks', {}) if s else {}
        if not decks or index not in decks:
            return {'index': index, 'loaded': False}
        dk = decks[index]
        audio = dk.get('buffer')
        duration = 0.0
        if audio is not None and len(audio) > 0:
            duration = len(audio) / s.sample_rate
        return {
            'index': index,
            'loaded': audio is not None and len(audio) > 0 if audio is not None else False,
            'duration': duration,
            'fx_chain': [fx[0] if isinstance(fx, (list, tuple)) else str(fx)
                         for fx in dk.get('fx_chain', [])],
        }

    def get_operator_details(self, index: int) -> Dict[str, Any]:
        """Return detailed info for a single operator."""
        s = self.session
        if not s or not hasattr(s, 'engine') or not hasattr(s.engine, 'operators'):
            return {}
        op = s.engine.operators.get(index, {})
        env = None
        op_envs = getattr(s, 'operator_envelopes', {})
        if op_envs and index in op_envs:
            e = op_envs[index]
            env = {'attack': e.get('attack', getattr(s, 'attack', 0.01)),
                   'decay': e.get('decay', getattr(s, 'decay', 0.1)),
                   'sustain': e.get('sustain', getattr(s, 'sustain', 0.8)),
                   'release': e.get('release', getattr(s, 'release', 0.1))}
        wave = op.get('wave', 'sine')
        # Collect wave-specific params
        wave_params = {}
        if wave == 'pulse':
            wave_params['pw'] = op.get('pw', 0.5)
        elif wave in ('physical', 'physical2'):
            wave_params['even_harmonics'] = op.get('even_harmonics', 8)
            wave_params['odd_harmonics'] = op.get('odd_harmonics', 4)
            wave_params['even_weight'] = op.get('even_weight', 1.0)
            wave_params['decay'] = op.get('decay', 0.7)
            wave_params['inharmonicity'] = op.get('inharmonicity', 0.01)
            wave_params['partials'] = op.get('partials', 12)
        elif wave == 'supersaw':
            wave_params['num_saws'] = op.get('num_saws', 7)
            wave_params['detune_spread'] = op.get('detune_spread', 0.5)
            wave_params['mix'] = op.get('mix', 0.75)
        elif wave == 'additive':
            wave_params['num_harmonics'] = op.get('num_harmonics', 16)
            wave_params['rolloff'] = op.get('rolloff', 1.0)
        elif wave == 'formant':
            wave_params['vowel'] = op.get('vowel', 'a')
        elif wave == 'harmonic':
            wave_params['odd_level'] = op.get('odd_level', 1.0)
            wave_params['even_level'] = op.get('even_level', 1.0)
            wave_params['odd_decay'] = op.get('odd_decay', 0.7)
            wave_params['even_decay'] = op.get('even_decay', 0.7)
            wave_params['num_harmonics'] = op.get('num_harmonics', 16)
        elif wave == 'waveguide_string':
            wave_params['damping'] = op.get('damping', 0.996)
            wave_params['brightness'] = op.get('brightness', 0.5)
            wave_params['position'] = op.get('position', 0.5)
        elif wave == 'waveguide_tube':
            wave_params['damping'] = op.get('damping', 0.996)
            wave_params['reflection'] = op.get('reflection', 0.98)
            wave_params['bore_shape'] = op.get('bore_shape', 'cylindrical')
        elif wave == 'waveguide_membrane':
            wave_params['tension'] = op.get('tension', 0.5)
            wave_params['damping'] = op.get('damping', 0.996)
            wave_params['strike_pos'] = op.get('strike_pos', 0.5)
        elif wave == 'waveguide_plate':
            wave_params['thickness'] = op.get('thickness', 0.5)
            wave_params['damping'] = op.get('damping', 0.996)
            wave_params['material'] = op.get('material', 'steel')
        elif wave == 'wavetable':
            wave_params['wavetable_name'] = op.get('wavetable_name', '')
            wave_params['frame_pos'] = op.get('frame_pos', 0.0)
        elif wave == 'compound':
            wave_params['compound_name'] = op.get('compound_name', '')
            wave_params['morph'] = op.get('morph', 0.0)
        return {
            'index': index,
            'wave': wave,
            'freq': op.get('freq', 440.0),
            'amp': op.get('amp', 0.8),
            'is_current': index == getattr(s, 'current_operator', 0),
            'envelope': env,
            'wave_params': wave_params,
        }

    def get_filter_slot_details(self, slot: int) -> Dict[str, Any]:
        """Return detailed info for a single filter slot."""
        s = self.session
        if not s:
            return {}
        filter_types = getattr(s, 'filter_types', {})
        filter_names = getattr(s, 'filter_type_names', {})
        filter_cutoffs = getattr(s, 'filter_cutoffs', {})
        filter_resonances = getattr(s, 'filter_resonances', {})
        filter_enabled = getattr(s, 'filter_enabled', {})
        type_idx = filter_types.get(slot, 0)
        return {
            'slot': slot,
            'type_name': filter_names.get(type_idx, 'lowpass'),
            'cutoff': filter_cutoffs.get(slot, 4500.0),
            'resonance': filter_resonances.get(slot, 50.0),
            'enabled': filter_enabled.get(slot, slot == 0),
            'is_selected': slot == getattr(s, 'selected_filter', 0),
        }

    def get_all_filter_slots(self) -> List[Dict[str, Any]]:
        """Return info for all active filter slots."""
        s = self.session
        if not s:
            return []
        slots = []
        for slot in range(8):
            info = self.get_filter_slot_details(slot)
            if info.get('enabled', False) or slot == 0:
                slots.append(info)
        return slots

    def get_rich_tree_data(self) -> Dict[str, Any]:
        """Return enriched tree data with inline previews for all objects."""
        s = self.session
        if not s:
            return {}

        # Tracks with previews
        tracks = []
        for i, t in enumerate(getattr(s, 'tracks', [])):
            info = self.get_track_details(i)
            if not info:
                continue
            ch_str = 'stereo' if info.get('channels', 0) == 2 else 'mono'
            if info.get('has_audio'):
                preview = (f"{info['name']}  [{info['duration']:.2f}s, {ch_str}, "
                           f"peak: {info['peak_db']:.1f}dB]")
            else:
                preview = f"{info['name']}  [empty]"
            if info.get('mute'):
                preview += '  [MUTE]'
            if info.get('solo'):
                preview += '  [SOLO]'
            tracks.append({'label': preview, 'index': i, 'type': 'track',
                           'details': info})

        # Buffers with previews
        buffers = []
        buffers_dict = getattr(s, 'buffers', {})
        filled = sorted(k for k, v in buffers_dict.items()
                        if v is not None and len(v) > 0)
        for idx in filled:
            info = self.get_buffer_details(idx)
            ch_str = 'stereo' if info.get('channels', 0) == 2 else 'mono'
            preview = f"Buffer {idx}  [{info['duration']:.2f}s, {ch_str}]"
            buffers.append({'label': preview, 'index': idx, 'type': 'buffer',
                            'details': info})

        # Working buffer
        wb_info = self.get_working_buffer_details()
        if wb_info.get('empty'):
            wb_preview = 'Working Buffer  [empty]'
        else:
            ch_str = 'stereo' if wb_info.get('channels', 0) == 2 else 'mono'
            wb_preview = (f"Working Buffer  [{wb_info['duration']:.2f}s, {ch_str}, "
                          f"src: {wb_info.get('source', '?')}]")

        # Decks with previews
        decks = []
        decks_dict = getattr(s, 'decks', {})
        for dk_id in sorted(decks_dict.keys()):
            info = self.get_deck_details(dk_id)
            if info.get('loaded'):
                preview = f"Deck {dk_id}  [{info['duration']:.2f}s loaded]"
            else:
                preview = f"Deck {dk_id}  [empty]"
            decks.append({'label': preview, 'index': dk_id, 'type': 'deck',
                          'details': info})

        # Operators with previews
        operators = []
        if hasattr(s, 'engine') and hasattr(s.engine, 'operators'):
            for idx in sorted(s.engine.operators.keys()):
                info = self.get_operator_details(idx)
                current = ' *' if info.get('is_current') else ''
                preview = (f"Op {idx}: {info['wave']} @ {info['freq']:.0f}Hz "
                           f"(amp: {info['amp']:.2f}){current}")
                operators.append({'label': preview, 'index': idx,
                                  'type': 'operator', 'details': info})

        # Filter slots
        filters = self.get_all_filter_slots()
        filter_items = []
        for f in filters:
            sel = ' *' if f.get('is_selected') else ''
            en = '' if f.get('enabled') else '  [OFF]'
            preview = (f"Slot {f['slot']}: {f['type_name']} "
                       f"{f['cutoff']:.0f}Hz res:{f['resonance']:.0f}{en}{sel}")
            filter_items.append({'label': preview, 'slot': f['slot'],
                                 'type': 'filter_slot', 'details': f})

        # Effects — include parameters and amount in display
        effects = []
        fx_params_list = getattr(s, 'effect_params', [])
        for i, fx in enumerate(getattr(s, 'effects', [])):
            fx_name = fx[0] if isinstance(fx, (list, tuple)) else str(fx)
            params = fx_params_list[i] if i < len(fx_params_list) else {}
            amt = params.get('amount', 50.0)
            # Build a readable label showing name + amount
            extra_params = {k: v for k, v in params.items() if k != 'amount'}
            if extra_params:
                param_str = ', '.join(f'{k}={v:.1f}' for k, v in extra_params.items())
                label = f"{fx_name}  ({amt:.0f}%)  [{param_str}]"
            else:
                label = f"{fx_name}  ({amt:.0f}%)"
            effects.append({
                'label': label, 'index': i, 'type': 'effect',
                'name': fx_name, 'params': dict(params),
            })

        # Presets (SyDefs from wherever they live)
        presets = []
        sydefs = self._get_sydefs()
        for name in sorted(sydefs.keys()):
            presets.append({'label': name, 'name': name, 'type': 'sydef'})

        # User functions
        funcs = []
        user_funcs = getattr(s, 'user_functions', {})
        if user_funcs:
            for name in sorted(user_funcs.keys()):
                funcs.append({'label': f"fn: {name}", 'name': name,
                              'type': 'user_function'})

        # Chains
        chains = []
        chains_dict = getattr(s, 'chains', {})
        if chains_dict:
            for name in sorted(chains_dict.keys()):
                ch = chains_dict[name]
                n = len(ch) if isinstance(ch, list) else 0
                chains.append({'label': f"chain: {name} ({n} fx)",
                                'name': name, 'type': 'chain'})

        # Deck section markers (accessibility)
        deck_sections = []
        for dk_id in sorted(decks_dict.keys()):
            info = self.get_deck_details(dk_id)
            if info.get('loaded') and info['duration'] > 0:
                dur = info['duration']
                # Create section markers at intro/body/outro boundaries
                sections = []
                if dur > 2:
                    sections.append({'label': f'Deck {dk_id} Intro (0s-{min(dur, 2):.1f}s)',
                                     'deck': dk_id, 'start': 0, 'end': min(dur, 2)})
                if dur > 4:
                    sections.append({'label': f'Deck {dk_id} Body ({2:.1f}s-{dur-2:.1f}s)',
                                     'deck': dk_id, 'start': 2, 'end': dur - 2})
                    sections.append({'label': f'Deck {dk_id} Outro ({dur-2:.1f}s-{dur:.1f}s)',
                                     'deck': dk_id, 'start': dur - 2, 'end': dur})
                deck_sections.extend(sections)

        # Wavetables
        wavetables = []
        if hasattr(s, 'engine') and hasattr(s.engine, 'wavetables'):
            for name in sorted(s.engine.wavetables.keys()):
                wt = s.engine.wavetables[name]
                frames = len(wt) if hasattr(wt, '__len__') else 0
                wavetables.append({'label': f"{name} ({frames} frames)",
                                   'name': name, 'type': 'wavetable'})

        # Compound waves
        compounds = []
        if hasattr(s, 'engine') and hasattr(s.engine, 'compound_waves'):
            for name in sorted(s.engine.compound_waves.keys()):
                cw = s.engine.compound_waves[name]
                layers = len(cw.get('layers', [])) if isinstance(cw, dict) else 0
                compounds.append({'label': f"{name} ({layers} layers)",
                                  'name': name, 'type': 'compound'})

        # Modulation routings
        routings = []
        if hasattr(s, 'engine') and hasattr(s.engine, 'algorithms'):
            for i, entry in enumerate(s.engine.algorithms):
                try:
                    algo_type, src, tgt, amt = entry
                    routings.append({
                        'label': f"{algo_type}: Op{src} → Op{tgt} ({amt:.1f})",
                        'index': i, 'type': 'routing',
                        'algo_type': algo_type, 'src': src, 'tgt': tgt,
                        'amt': amt,
                    })
                except (ValueError, TypeError):
                    routings.append({
                        'label': f"Routing {i}: {entry}",
                        'index': i, 'type': 'routing',
                    })

        return {
            'tracks': tracks,
            'buffers': buffers,
            'working_buffer': {'label': wb_preview, 'type': 'working_buffer',
                               'details': wb_info},
            'decks': decks,
            'operators': operators,
            'filters': filter_items,
            'effects': effects,
            'presets': presets,
            'user_functions': funcs,
            'chains': chains,
            'deck_sections': deck_sections,
            'wavetables': wavetables,
            'compounds': compounds,
            'routings': routings,
        }


# ============================================================================
# GUI COMPONENTS
# ============================================================================

class ObjectBrowser(wx.Panel):
    """Left panel - hierarchical tree of all session objects.

    Provides a rich, navigable tree with:
    - Top-level categories for every object domain
    - Inline preview metadata (duration, peak, channels, state)
    - Context menus for direct actions on objects
    - Accessibility markers for deck sections
    - Dynamic population from live session state
    """

    # Top-level categories — each is its own branch in the tree.
    CATEGORIES = [
        ('engine',          'Engine'),
        ('synth',           'Synthesizer'),
        ('voice',           'Voice'),
        ('filter',          'Filter'),
        ('filter_envelope', 'Filter Envelope'),
        ('envelope',        'Envelope'),
        ('hq',              'HQ Mode'),
        ('key',             'Key / Scale'),
        ('modulation',      'Modulation'),
        ('wavetable',       'Wavetables'),
        ('compound',        'Compound Waves'),
        ('convolution',     'Convolution Reverb'),
        ('impulse_lfo',     'Impulse LFOs'),
        ('impulse_env',     'Impulse Envelopes'),
        ('ir_enhance',      'IR Enhancement'),
        ('ir_transform',    'IR Transform'),
        ('ir_granular',     'IR Granular'),
        ('tracks',          'Tracks'),
        ('buffers',         'Buffers'),
        ('decks',           'Decks'),
        ('fx',              'Effects'),
        ('preset',          'Presets'),
        ('bank',            'Banks'),
        ('generative',      'Generative'),
    ]

    def __init__(self, parent, on_select_callback, executor: CommandExecutor,
                 console_callback=None):
        super().__init__(parent)
        self.on_select = on_select_callback
        self.executor = executor
        self.console_cb = console_callback

        self.SetBackgroundColour(Theme.BG_PANEL)

        sizer = wx.BoxSizer(wx.VERTICAL)

        # Search box
        search_sizer = wx.BoxSizer(wx.HORIZONTAL)
        search_label = wx.StaticText(self, label="Search:")
        search_label.SetForegroundColour(Theme.FG_TEXT)
        self.search_box = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER,
                                       name="ObjectSearch")
        self.search_box.SetName("Search objects — type to filter the tree")
        self.search_box.SetBackgroundColour(Theme.BG_INPUT)
        self.search_box.SetForegroundColour(Theme.FG_TEXT)
        self.search_box.SetHint("Type to search objects...")
        self.search_box.Bind(wx.EVT_TEXT, self.on_search)

        search_sizer.Add(search_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        search_sizer.Add(self.search_box, 1, wx.EXPAND)

        sizer.Add(search_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Tree control
        self.tree = wx.TreeCtrl(self, style=wx.TR_DEFAULT_STYLE | wx.TR_HIDE_ROOT,
                                name="ObjectTree")
        self.tree.SetName("Object Browser — navigate with arrow keys, Application key for context menu")
        self.tree.SetBackgroundColour(Theme.BG_INPUT)
        self.tree.SetForegroundColour(Theme.FG_TEXT)
        self.tree.Bind(wx.EVT_TREE_SEL_CHANGED, self.on_tree_select)
        self.tree.Bind(wx.EVT_TREE_ITEM_RIGHT_CLICK, self.on_context_menu)
        # Allow context menu via Application key / Shift+F10
        self.tree.Bind(wx.EVT_CONTEXT_MENU, self.on_keyboard_context_menu)

        sizer.Add(self.tree, 1, wx.EXPAND | wx.ALL, 5)

        # Refresh button
        refresh_btn = wx.Button(self, label="Refresh (F5)")
        refresh_btn.SetName("Refresh object tree (F5)")
        refresh_btn.Bind(wx.EVT_BUTTON, self.on_refresh)
        sizer.Add(refresh_btn, 0, wx.EXPAND | wx.ALL, 5)

        self.SetSizer(sizer)
        self.category_items: Dict[str, Any] = {}
        self.populate_tree()

    # ------------------------------------------------------------------
    # Tree population
    # ------------------------------------------------------------------

    def populate_tree(self):
        """Build the full object tree from live session data.

        Preserves which categories are expanded and which item is selected
        so the user doesn't lose their place after command execution.
        """
        # ---- Save expansion and selection state ----
        expanded_ids = set()
        selected_id = None

        sel_item = self.tree.GetSelection()
        if sel_item and sel_item.IsOk():
            sel_data = self.tree.GetItemData(sel_item)
            if sel_data:
                selected_id = (sel_data.get('type', ''), sel_data.get('id', ''),
                               sel_data.get('name', ''), sel_data.get('index', ''))

        for cat_key, cat_item in self.category_items.items():
            if cat_item and cat_item.IsOk() and self.tree.IsExpanded(cat_item):
                expanded_ids.add(cat_key)

        # ---- Rebuild tree ----
        self.tree.DeleteAllItems()
        root = self.tree.AddRoot("MDMA Session")
        self.category_items = {}

        state = self.executor.get_engine_state()
        rich = self.executor.get_rich_tree_data()

        # ---- Engine ----
        bpm = state.get('bpm', 128) if state else 128
        sr = state.get('sample_rate', 48000) if state else 48000
        hq = state.get('hq_mode', False) if state else False
        fmt = state.get('output_format', 'wav') if state else 'wav'
        bits = state.get('output_bit_depth', 16) if state else 16
        eng = self.tree.AppendItem(root, f"Engine  ({bpm:.0f} BPM, {sr}Hz)")
        self.tree.SetItemData(eng, {'type': 'category', 'id': 'engine'})
        self.category_items['engine'] = eng
        for lbl in [f"BPM: {bpm:.0f}", f"Sample Rate: {sr}",
                     f"HQ Mode: {'ON' if hq else 'OFF'} ({fmt.upper()} {bits}-bit)"]:
            sub = self.tree.AppendItem(eng, lbl)
            self.tree.SetItemData(sub, {'type': 'engine_prop', 'id': 'engine'})

        # ---- Synthesizer ----
        wave = state.get('waveform', '---') if state else '---'
        syn = self.tree.AppendItem(root, f"Synthesizer  ({wave})")
        self.tree.SetItemData(syn, {'type': 'category', 'id': 'synth'})
        self.category_items['synth'] = syn
        for op_item in rich.get('operators', []):
            sub = self.tree.AppendItem(syn, op_item['label'])
            self.tree.SetItemData(sub, {'type': 'operator',
                                         'index': op_item['index'],
                                         'id': 'synth'})

        # ---- Filter ----
        filt_items = rich.get('filters', [])
        ftype = state.get('filter_type', 'lp') if state else 'lp'
        fcut = state.get('filter_cutoff', 4500) if state else 4500
        flt = self.tree.AppendItem(root, f"Filter  ({ftype} {fcut:.0f}Hz)")
        self.tree.SetItemData(flt, {'type': 'category', 'id': 'filter'})
        self.category_items['filter'] = flt
        for fi in filt_items:
            sub = self.tree.AppendItem(flt, fi['label'])
            self.tree.SetItemData(sub, {'type': 'filter_slot',
                                         'slot': fi['slot'],
                                         'id': 'filter'})

        # ---- Voice ----
        if state:
            vc = state.get('voice_count', 1)
            va = state.get('voice_algorithm', 0)
            va_name = {0: 'stack', 1: 'unison', 2: 'wide'}.get(va, str(va))
            voice_lbl = f"Voice  ({vc} voices, {va_name})"
        else:
            voice_lbl = "Voice"
        voice_cat = self.tree.AppendItem(root, voice_lbl)
        self.tree.SetItemData(voice_cat, {'type': 'category', 'id': 'voice'})
        self.category_items['voice'] = voice_cat
        if state:
            for lbl_v in [
                f"Voices: {state.get('voice_count', 1)}",
                f"Algorithm: {va_name}",
                f"Detune: {state.get('detune', 0):.2f} Hz",
                f"Stereo Spread: {state.get('stereo_spread', 0):.0f}",
                f"Phase Spread: {state.get('phase_spread', 0):.2f} rad",
                f"Random: {state.get('rand', 0):.0f}",
                f"V-Mod: {state.get('v_mod', 0):.0f}",
            ]:
                sub = self.tree.AppendItem(voice_cat, lbl_v)
                self.tree.SetItemData(sub, {'type': 'voice_prop', 'id': 'voice'})

        # ---- Envelope ----
        if state:
            env_lbl = (f"Envelope  (A{state.get('attack',0):.2f} "
                       f"D{state.get('decay',0):.2f} "
                       f"S{state.get('sustain',0):.2f} "
                       f"R{state.get('release',0):.2f})")
        else:
            env_lbl = "Envelope"
        env = self.tree.AppendItem(root, env_lbl)
        self.tree.SetItemData(env, {'type': 'category', 'id': 'envelope'})
        self.category_items['envelope'] = env
        if state:
            for param in ['attack', 'decay', 'sustain', 'release']:
                sub = self.tree.AppendItem(env,
                    f"{param.capitalize()}: {state.get(param, 0):.3f}")
                self.tree.SetItemData(sub, {'type': 'envelope_param',
                                             'param': param, 'id': 'envelope'})

        # ---- Filter Envelope ----
        if state:
            fenv_lbl = (f"Filter Envelope  (A{state.get('filter_attack',0.01):.2f} "
                        f"D{state.get('filter_decay',0.1):.2f} "
                        f"S{state.get('filter_sustain',0.8):.2f} "
                        f"R{state.get('filter_release',0.1):.2f})")
        else:
            fenv_lbl = "Filter Envelope"
        fenv = self.tree.AppendItem(root, fenv_lbl)
        self.tree.SetItemData(fenv, {'type': 'category', 'id': 'filter_envelope'})
        self.category_items['filter_envelope'] = fenv
        if state:
            for p, k in [('Attack', 'filter_attack'), ('Decay', 'filter_decay'),
                         ('Sustain', 'filter_sustain'), ('Release', 'filter_release')]:
                sub = self.tree.AppendItem(fenv, f"{p}: {state.get(k, 0):.3f}")
                self.tree.SetItemData(sub, {'type': 'fenv_param', 'param': k,
                                             'id': 'filter_envelope'})

        # ---- HQ Mode ----
        hq_on = state.get('hq_mode', False) if state else False
        hq_cat = self.tree.AppendItem(root, f"HQ Mode  ({'ON' if hq_on else 'OFF'})")
        self.tree.SetItemData(hq_cat, {'type': 'category', 'id': 'hq'})
        self.category_items['hq'] = hq_cat
        if state:
            for lbl_h in [
                f"HQ Mode: {'ON' if hq_on else 'OFF'}",
                f"Oscillators: {'ON' if state.get('hq_oscillators') else 'OFF'}",
                f"DC Removal: {'ON' if state.get('hq_dc_removal') else 'OFF'}",
                f"Saturation: {state.get('hq_saturation', 0)}",
                f"Limiter: {state.get('hq_limiter', 0)}",
            ]:
                sub = self.tree.AppendItem(hq_cat, lbl_h)
                self.tree.SetItemData(sub, {'type': 'hq_prop', 'id': 'hq'})

        # ---- Key / Scale ----
        key_n = state.get('key_note', 'C') if state else 'C'
        key_s = state.get('key_scale', 'major') if state else 'major'
        key_cat = self.tree.AppendItem(root, f"Key / Scale  ({key_n} {key_s})")
        self.tree.SetItemData(key_cat, {'type': 'category', 'id': 'key'})
        self.category_items['key'] = key_cat

        # ---- Modulation ----
        routing_items = rich.get('routings', [])
        mod_cat = self.tree.AppendItem(root,
            f"Modulation  ({len(routing_items)} routings)")
        self.tree.SetItemData(mod_cat, {'type': 'category', 'id': 'modulation'})
        self.category_items['modulation'] = mod_cat
        for ri in routing_items:
            sub = self.tree.AppendItem(mod_cat, ri['label'])
            self.tree.SetItemData(sub, {'type': 'routing', 'index': ri['index'],
                                         'id': 'modulation'})

        # ---- Wavetables ----
        wt_items = rich.get('wavetables', [])
        wt_cat = self.tree.AppendItem(root, f"Wavetables  ({len(wt_items)})")
        self.tree.SetItemData(wt_cat, {'type': 'category', 'id': 'wavetable'})
        self.category_items['wavetable'] = wt_cat
        for wi in wt_items:
            sub = self.tree.AppendItem(wt_cat, wi['label'])
            self.tree.SetItemData(sub, {'type': 'wavetable', 'name': wi['name'],
                                         'id': 'wavetable'})

        # ---- Compound Waves ----
        cw_items = rich.get('compounds', [])
        cw_cat = self.tree.AppendItem(root, f"Compound Waves  ({len(cw_items)})")
        self.tree.SetItemData(cw_cat, {'type': 'category', 'id': 'compound'})
        self.category_items['compound'] = cw_cat
        for ci_item in cw_items:
            sub = self.tree.AppendItem(cw_cat, ci_item['label'])
            self.tree.SetItemData(sub, {'type': 'compound', 'name': ci_item['name'],
                                         'id': 'compound'})

        # ---- Phase 3: Convolution Reverb ----
        try:
            from mdma_rebuild.dsp.convolution import get_convolution_engine
            ce = get_convolution_engine()
            ce_info = ce.get_info()
            ir_status = ce_info['ir_name'] if ce_info['ir_loaded'] else 'none'
            conv_cat = self.tree.AppendItem(root,
                f"Convolution Reverb  (IR: {ir_status})")
            self.tree.SetItemData(conv_cat, {'type': 'category', 'id': 'convolution'})
            self.category_items['convolution'] = conv_cat
            if ce_info['ir_loaded']:
                for lbl_c in [
                    f"IR: {ce_info['ir_name']} ({ce_info['ir_duration']:.2f}s)",
                    f"Wet/Dry: {ce_info['wet']:.0f}/{ce_info['dry']:.0f}",
                    f"Pre-Delay: {ce_info['pre_delay_ms']:.0f}ms",
                    f"Decay: {ce_info['decay']:.2f}x",
                    f"Stereo Width: {ce_info['stereo_width']:.0f}",
                    f"Early/Late: {ce_info['early_level']:.0f}/{ce_info['late_level']:.0f}",
                ]:
                    sub = self.tree.AppendItem(conv_cat, lbl_c)
                    self.tree.SetItemData(sub, {'type': 'conv_prop', 'id': 'convolution'})
            # Bank entries
            for bn in ce_info.get('bank_names', []):
                sub = self.tree.AppendItem(conv_cat, f"Bank: {bn}")
                self.tree.SetItemData(sub, {'type': 'ir_bank_entry', 'name': bn,
                                             'id': 'convolution'})
        except Exception:
            conv_cat = self.tree.AppendItem(root, "Convolution Reverb")
            self.tree.SetItemData(conv_cat, {'type': 'category', 'id': 'convolution'})
            self.category_items['convolution'] = conv_cat

        # ---- Phase 3: Impulse LFOs ----
        lfo_shapes = {}
        if self.executor.session and hasattr(self.executor.session, '_lfo_waveshapes'):
            lfo_shapes = self.executor.session._lfo_waveshapes
        ilfo_cat = self.tree.AppendItem(root,
            f"Impulse LFOs  ({len(lfo_shapes)})")
        self.tree.SetItemData(ilfo_cat, {'type': 'category', 'id': 'impulse_lfo'})
        self.category_items['impulse_lfo'] = ilfo_cat
        for name in sorted(lfo_shapes.keys()):
            sub = self.tree.AppendItem(ilfo_cat, f"{name} ({len(lfo_shapes[name])} samples)")
            self.tree.SetItemData(sub, {'type': 'lfo_shape', 'name': name,
                                         'id': 'impulse_lfo'})

        # ---- Phase 3: Impulse Envelopes ----
        imp_envs = {}
        if self.executor.session and hasattr(self.executor.session, '_imp_envelopes'):
            imp_envs = self.executor.session._imp_envelopes
        ienv_cat = self.tree.AppendItem(root,
            f"Impulse Envelopes  ({len(imp_envs)})")
        self.tree.SetItemData(ienv_cat, {'type': 'category', 'id': 'impulse_env'})
        self.category_items['impulse_env'] = ienv_cat
        for name in sorted(imp_envs.keys()):
            sub = self.tree.AppendItem(ienv_cat, f"{name} ({len(imp_envs[name])} samples)")
            self.tree.SetItemData(sub, {'type': 'imp_env_shape', 'name': name,
                                         'id': 'impulse_env'})

        # ---- Phase 3: IR Enhancement / Transform / Granular ----
        for cat_id, label in [('ir_enhance', 'IR Enhancement'),
                               ('ir_transform', 'IR Transform'),
                               ('ir_granular', 'IR Granular')]:
            cat = self.tree.AppendItem(root, label)
            self.tree.SetItemData(cat, {'type': 'category', 'id': cat_id})
            self.category_items[cat_id] = cat

        # ---- Tracks ----
        track_items = rich.get('tracks', [])
        trk = self.tree.AppendItem(root, f"Tracks  ({len(track_items)})")
        self.tree.SetItemData(trk, {'type': 'category', 'id': 'tracks'})
        self.category_items['tracks'] = trk
        for ti in track_items:
            sub = self.tree.AppendItem(trk, ti['label'])
            self.tree.SetItemData(sub, {'type': 'track', 'index': ti['index'],
                                         'id': 'tracks',
                                         'details': ti.get('details', {})})

        # ---- Buffers ----
        buf_items = rich.get('buffers', [])
        wb = rich.get('working_buffer', {})
        total_bufs = len(buf_items) + (0 if wb.get('details', {}).get('empty', True) else 1)
        buf_cat = self.tree.AppendItem(root,
            f"Buffers  ({len(buf_items)} filled)")
        self.tree.SetItemData(buf_cat, {'type': 'category', 'id': 'buffers'})
        self.category_items['buffers'] = buf_cat
        # Working buffer first
        wb_sub = self.tree.AppendItem(buf_cat, wb.get('label', 'Working Buffer'))
        self.tree.SetItemData(wb_sub, {'type': 'working_buffer', 'id': 'buffers',
                                        'details': wb.get('details', {})})
        for bi in buf_items:
            sub = self.tree.AppendItem(buf_cat, bi['label'])
            self.tree.SetItemData(sub, {'type': 'buffer', 'index': bi['index'],
                                         'id': 'buffers',
                                         'details': bi.get('details', {})})

        # ---- Decks ----
        deck_items = rich.get('decks', [])
        deck_sections = rich.get('deck_sections', [])
        dk_cat = self.tree.AppendItem(root, f"Decks  ({len(deck_items)})")
        self.tree.SetItemData(dk_cat, {'type': 'category', 'id': 'decks'})
        self.category_items['decks'] = dk_cat
        for di in deck_items:
            dk_sub = self.tree.AppendItem(dk_cat, di['label'])
            self.tree.SetItemData(dk_sub, {'type': 'deck', 'index': di['index'],
                                            'id': 'decks',
                                            'details': di.get('details', {})})
            # Add accessibility section markers as children of each deck
            for sec in deck_sections:
                if sec.get('deck') == di['index']:
                    sec_sub = self.tree.AppendItem(dk_sub, sec['label'])
                    self.tree.SetItemData(sec_sub, {
                        'type': 'deck_section',
                        'deck': sec['deck'],
                        'start': sec['start'],
                        'end': sec['end'],
                        'id': 'decks',
                    })

        # ---- Effects ----
        fx_items = rich.get('effects', [])
        n_fx = len([e for e in fx_items if e.get('label') != '(none)'])
        fx_cat = self.tree.AppendItem(root, f"Effects  ({n_fx} active)")
        self.tree.SetItemData(fx_cat, {'type': 'category', 'id': 'fx'})
        self.category_items['fx'] = fx_cat
        for ei in fx_items:
            sub = self.tree.AppendItem(fx_cat, ei['label'])
            self.tree.SetItemData(sub, {
                'type': 'effect', 'index': ei.get('index', 0),
                'id': 'fx', 'name': ei.get('name', ''),
                'params': ei.get('params', {}),
            })

        # ---- Presets (SyDefs, Functions, Chains) ----
        preset_items = rich.get('presets', [])
        func_items = rich.get('user_functions', [])
        chain_items = rich.get('chains', [])
        total_pre = len(preset_items) + len(func_items) + len(chain_items)
        pre_cat = self.tree.AppendItem(root, f"Presets  ({total_pre})")
        self.tree.SetItemData(pre_cat, {'type': 'category', 'id': 'preset'})
        self.category_items['preset'] = pre_cat
        # SyDefs
        if preset_items:
            sydef_grp = self.tree.AppendItem(pre_cat,
                f"SyDefs ({len(preset_items)})")
            self.tree.SetItemData(sydef_grp, {'type': 'group', 'id': 'preset'})
            for pi in preset_items:
                sub = self.tree.AppendItem(sydef_grp, pi['label'])
                self.tree.SetItemData(sub, {'type': 'sydef',
                                             'name': pi.get('name', ''),
                                             'id': 'preset'})
        # User functions
        if func_items:
            fn_grp = self.tree.AppendItem(pre_cat,
                f"Functions ({len(func_items)})")
            self.tree.SetItemData(fn_grp, {'type': 'group', 'id': 'preset'})
            for fi in func_items:
                sub = self.tree.AppendItem(fn_grp, fi['label'])
                self.tree.SetItemData(sub, {'type': 'user_function',
                                             'name': fi.get('name', ''),
                                             'id': 'preset'})
        # Chains
        if chain_items:
            ch_grp = self.tree.AppendItem(pre_cat,
                f"Chains ({len(chain_items)})")
            self.tree.SetItemData(ch_grp, {'type': 'group', 'id': 'preset'})
            for ci in chain_items:
                sub = self.tree.AppendItem(ch_grp, ci['label'])
                self.tree.SetItemData(sub, {'type': 'chain',
                                             'name': ci.get('name', ''),
                                             'id': 'preset'})

        # ---- Banks ----
        bank_cat = self.tree.AppendItem(root, "Banks")
        self.tree.SetItemData(bank_cat, {'type': 'category', 'id': 'bank'})
        self.category_items['bank'] = bank_cat
        # Preset bank slots
        s = self.executor.session
        if s and hasattr(s, 'engine') and hasattr(s.engine, 'preset_bank'):
            pb = s.engine.preset_bank
            if pb:
                pb_grp = self.tree.AppendItem(bank_cat,
                    f"Preset Bank ({len(pb)} saved)")
                self.tree.SetItemData(pb_grp, {'type': 'group', 'id': 'bank'})
                for slot_num in sorted(pb.keys()):
                    name = pb[slot_num].get('name', f'slot_{slot_num}') if isinstance(pb[slot_num], dict) else f'slot_{slot_num}'
                    sub = self.tree.AppendItem(pb_grp, f"Slot {slot_num}: {name}")
                    self.tree.SetItemData(sub, {'type': 'preset_slot',
                                                 'slot': slot_num, 'id': 'bank'})

        # ---- Granular Engine ----
        gr_cat = self.tree.AppendItem(root, "Granular Engine")
        self.tree.SetItemData(gr_cat, {'type': 'category', 'id': 'granular_engine'})
        self.category_items['granular_engine'] = gr_cat
        # Show granular status as child items
        for gr_lbl in [
            "Grain Size (ms): /gr size",
            "Density: /gr density",
            "Position: /gr pos",
            "Spread: /gr spread",
            "Pitch Ratio: /gr pitch",
            "Envelope: /gr env",
            "Reverse Prob: /gr reverse",
        ]:
            sub = self.tree.AppendItem(gr_cat, gr_lbl)
            self.tree.SetItemData(sub, {'type': 'category',
                                         'id': 'granular_engine'})

        # ---- GPU / AI Settings ----
        gpu_cat = self.tree.AppendItem(root, "GPU / AI Settings")
        self.tree.SetItemData(gpu_cat, {'type': 'category', 'id': 'gpu'})
        self.category_items['gpu'] = gpu_cat
        for gpu_lbl in [
            "Inference Steps: /gpu steps",
            "CFG Scale: /gpu cfg",
            "Scheduler: /gpu sk",
            "Model: /gpu model",
            "Device: /gpu device",
            "FP16 / Offload: /gpu fp16, /gpu offload",
        ]:
            sub = self.tree.AppendItem(gpu_cat, gpu_lbl)
            self.tree.SetItemData(sub, {'type': 'category', 'id': 'gpu'})

        # ---- AI Generation ----
        ai_cat = self.tree.AppendItem(root, "AI Audio Generation")
        self.tree.SetItemData(ai_cat, {'type': 'category', 'id': 'ai_generate'})
        self.category_items['ai_generate'] = ai_cat
        for ai_lbl in [
            "Generate from Prompt: /gen",
            "Generate Variations: /genv",
            "Analyze Audio: /analyze",
            "Describe Audio: /describe",
            "Ask AI (Natural Language): /ask",
        ]:
            sub = self.tree.AppendItem(ai_cat, ai_lbl)
            self.tree.SetItemData(sub, {'type': 'category', 'id': 'ai_generate'})

        # ---- Generative (Phase 4) ----
        gen_cat = self.tree.AppendItem(root, "Generative")
        self.tree.SetItemData(gen_cat, {'type': 'category', 'id': 'generative'})
        self.category_items['generative'] = gen_cat

        # Beat generation subcategory
        gen_beat = self.tree.AppendItem(gen_cat, "Beat Generation")
        self.tree.SetItemData(gen_beat, {'type': 'gen_section', 'section': 'beat',
                                          'id': 'generative'})
        try:
            from mdma_rebuild.dsp import beat_gen
            for gname in sorted(beat_gen.GENRE_TEMPLATES.keys()):
                tmpl = beat_gen.GENRE_TEMPLATES[gname]
                sub = self.tree.AppendItem(gen_beat,
                    f"{gname.title()} ({tmpl.bpm} BPM)")
                self.tree.SetItemData(sub, {'type': 'gen_genre', 'genre': gname,
                                             'bpm': tmpl.bpm, 'id': 'generative'})
        except ImportError:
            pass

        # Loop generation subcategory
        gen_loop = self.tree.AppendItem(gen_cat, "Loop Generation")
        self.tree.SetItemData(gen_loop, {'type': 'gen_section', 'section': 'loop',
                                          'id': 'generative'})
        for layer in ['drums', 'bass', 'chords', 'melody', 'full']:
            sub = self.tree.AppendItem(gen_loop, f"Layer: {layer.title()}")
            self.tree.SetItemData(sub, {'type': 'gen_layer', 'layer': layer,
                                         'id': 'generative'})

        # Transforms subcategory
        gen_xform = self.tree.AppendItem(gen_cat, "Transforms")
        self.tree.SetItemData(gen_xform, {'type': 'gen_section', 'section': 'xform',
                                           'id': 'generative'})
        try:
            from mdma_rebuild.dsp import transforms as tf
            for pname in sorted(tf.AUDIO_TRANSFORM_PRESETS.keys()):
                sub = self.tree.AppendItem(gen_xform, f"Preset: {pname}")
                self.tree.SetItemData(sub, {'type': 'gen_xform_preset',
                                             'preset': pname, 'id': 'generative'})
        except ImportError:
            pass

        # Music Theory subcategory
        gen_theory = self.tree.AppendItem(gen_cat, "Music Theory")
        self.tree.SetItemData(gen_theory, {'type': 'gen_section', 'section': 'theory',
                                            'id': 'generative'})
        for item in ['Scales', 'Chords', 'Progressions']:
            sub = self.tree.AppendItem(gen_theory, item)
            self.tree.SetItemData(sub, {'type': 'gen_theory_item',
                                         'query': item.lower(), 'id': 'generative'})

        # Content Generation subcategory
        gen_content = self.tree.AppendItem(gen_cat, "Content Generation")
        self.tree.SetItemData(gen_content, {'type': 'gen_section', 'section': 'gen2',
                                             'id': 'generative'})
        for item in ['Melody', 'Chord Progression', 'Bassline', 'Arpeggio', 'Drone']:
            sub = self.tree.AppendItem(gen_content, item)
            sub_cmd = item.lower().replace(' ', '_')
            self.tree.SetItemData(sub, {'type': 'gen_content_item',
                                         'content_type': sub_cmd, 'id': 'generative'})

        # Text-to-Audio subcategory
        gen_tta = self.tree.AppendItem(gen_cat, "Text to Audio")
        self.tree.SetItemData(gen_tta, {'type': 'gen_section', 'section': 'text_to_audio',
                                         'id': 'generative'})
        for item, desc in [('Melody from Scale', 'melody'),
                            ('Chord Progression', 'chord_prog'),
                            ('Bassline', 'bassline'),
                            ('Beat from Genre', 'beat'),
                            ('Full Loop', 'loop')]:
            sub = self.tree.AppendItem(gen_tta, item)
            self.tree.SetItemData(sub, {'type': 'gen_tta_item',
                                         'generator': desc, 'id': 'generative'})

        # Genetic Breeding subcategory
        gen_breed = self.tree.AppendItem(gen_cat, "Genetic Breeding")
        self.tree.SetItemData(gen_breed, {'type': 'gen_section', 'section': 'breeding',
                                           'id': 'generative'})
        for item, method in [('Breed (Crossover + Mutate)', 'breed'),
                              ('Temporal Crossover', 'temporal'),
                              ('Spectral Crossover', 'spectral'),
                              ('Blend Crossover', 'blend'),
                              ('Morphological Crossover', 'morphological'),
                              ('Evolve Population', 'evolve'),
                              ('Mutate: Noise', 'mutate_noise'),
                              ('Mutate: Pitch', 'mutate_pitch'),
                              ('Mutate: Time Stretch', 'mutate_time_stretch'),
                              ('Mutate: Spectral Smear', 'mutate_spectral_smear'),
                              ('Mutate: Reverse Segment', 'mutate_reverse_segment')]:
            sub = self.tree.AppendItem(gen_breed, item)
            self.tree.SetItemData(sub, {'type': 'gen_breed_item',
                                         'method': method, 'id': 'generative'})

        # ---- Phase T: Song-Ready Tools ----
        phase_t = self.tree.AppendItem(root, "Song Tools (Phase T)")
        self.tree.SetItemData(phase_t, {'type': 'category', 'id': 'phase_t'})

        # Undo/Redo
        for label, cmd in [('Undo', '/undo'), ('Redo', '/redo'),
                           ('Save Snapshot', '/snapshot save'),
                           ('Restore Snapshot', '/snapshot restore')]:
            sub = self.tree.AppendItem(phase_t, label)
            self.tree.SetItemData(sub, {'type': 'phase_t_cmd', 'command': cmd,
                                         'id': 'phase_t'})

        # Structure
        struct = self.tree.AppendItem(phase_t, "Song Structure")
        self.tree.SetItemData(struct, {'type': 'gen_section', 'section': 'phase_t_structure',
                                        'id': 'phase_t'})
        for label, cmd in [('Add Section', '/section add'),
                           ('List Sections', '/section list'),
                           ('Copy Section', '/section copy'),
                           ('Commit to Track', '/commit'),
                           ('Show Position', '/pos')]:
            sub = self.tree.AppendItem(struct, label)
            self.tree.SetItemData(sub, {'type': 'phase_t_cmd', 'command': cmd,
                                         'id': 'phase_t'})

        # Export
        exp_cat = self.tree.AppendItem(phase_t, "Export")
        self.tree.SetItemData(exp_cat, {'type': 'gen_section', 'section': 'phase_t_export',
                                         'id': 'phase_t'})
        for label, cmd in [('Export Stems', '/export stems'),
                           ('Export Track', '/export track'),
                           ('Export Section', '/export section'),
                           ('Master Gain', '/master_gain')]:
            sub = self.tree.AppendItem(exp_cat, label)
            self.tree.SetItemData(sub, {'type': 'phase_t_cmd', 'command': cmd,
                                         'id': 'phase_t'})

        # Tools
        tools = self.tree.AppendItem(phase_t, "Buffer Tools")
        self.tree.SetItemData(tools, {'type': 'gen_section', 'section': 'phase_t_tools',
                                       'id': 'phase_t'})
        for label, cmd in [('Crossover Buffers', '/crossover'),
                           ('Duplicate Buffer', '/dup'),
                           ('Swap Buffers', '/swap'),
                           ('Metronome', '/metronome')]:
            sub = self.tree.AppendItem(tools, label)
            self.tree.SetItemData(sub, {'type': 'phase_t_cmd', 'command': cmd,
                                         'id': 'phase_t'})

        # ---- Restore expansion and selection state ----
        if expanded_ids:
            # Only expand categories that were previously expanded
            for cat_key, cat_item in self.category_items.items():
                if cat_item and cat_item.IsOk():
                    if cat_key in expanded_ids:
                        self.tree.Expand(cat_item)
                    else:
                        self.tree.Collapse(cat_item)
        else:
            # First load: expand everything
            self.tree.ExpandAll()

        # Restore selection by matching item data
        if selected_id:
            self._restore_selection(root, selected_id)

    def _restore_selection(self, parent, target_id):
        """Walk the tree to find and select an item matching target_id."""
        child, cookie = self.tree.GetFirstChild(parent)
        while child.IsOk():
            data = self.tree.GetItemData(child)
            if data:
                item_id = (data.get('type', ''), data.get('id', ''),
                           data.get('name', ''), data.get('index', ''))
                if item_id == target_id:
                    self.tree.SelectItem(child)
                    self.tree.EnsureVisible(child)
                    return True
            # Recurse into children
            if self.tree.ItemHasChildren(child):
                if self._restore_selection(child, target_id):
                    return True
            child, cookie = self.tree.GetNextChild(parent, cookie)
        return False

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def on_tree_select(self, event):
        """Handle tree selection — sends rich item data to callbacks."""
        item = event.GetItem()
        if not item.IsOk():
            return
        data = self.tree.GetItemData(item)
        if not data:
            return
        # Notify parent with the full data dict
        self.on_select(data)

    # ------------------------------------------------------------------
    # Context menus
    # ------------------------------------------------------------------

    def on_keyboard_context_menu(self, event):
        """Handle context menu triggered by Application key or Shift+F10.

        Retrieves the currently selected tree item and delegates to the
        same handler used by right-click so every object type is supported.
        """
        item = self.tree.GetSelection()
        if not item or not item.IsOk():
            return
        # Build a lightweight wrapper that responds to GetItem()
        class _SyntheticTreeEvent:
            def __init__(self, tree_item):
                self._item = tree_item
            def GetItem(self):
                return self._item
        self.on_context_menu(_SyntheticTreeEvent(item))

    def on_context_menu(self, event):
        """Show a context menu appropriate to the selected tree item."""
        item = event.GetItem()
        if not item.IsOk():
            return
        self.tree.SelectItem(item)
        data = self.tree.GetItemData(item)
        if not data:
            return

        menu = wx.Menu()
        obj_type = data.get('type', '')

        # ==============================================================
        # Track
        # ==============================================================
        if obj_type == 'track':
            idx = data.get('index', 0)
            m_play = wx.NewIdRef()
            m_solo = wx.NewIdRef()
            m_mute = wx.NewIdRef()
            m_fx = wx.NewIdRef()
            m_bounce = wx.NewIdRef()
            m_clear = wx.NewIdRef()

            menu.Append(m_play, f"Play Track {idx+1}")
            menu.Append(m_solo, "Solo")
            menu.Append(m_mute, "Mute")
            menu.AppendSeparator()
            menu.Append(m_fx, "Apply Effect...")
            menu.Append(m_bounce, "Bounce to Working Buffer")
            menu.AppendSeparator()
            menu.Append(m_clear, "Clear Track")

            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/pt {i+1}'), id=m_play)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/tsolo {i+1}'), id=m_solo)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/tmute {i+1}'), id=m_mute)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._show_fx_picker('track', i), id=m_fx)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/btw {i+1}'), id=m_bounce)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/tsel {i+1}\n/rc'), id=m_clear)

        # ==============================================================
        # Buffer
        # ==============================================================
        elif obj_type == 'buffer':
            idx = data.get('index', 1)
            m_play = wx.NewIdRef()
            m_to_work = wx.NewIdRef()
            m_to_track = wx.NewIdRef()
            m_dup = wx.NewIdRef()
            m_fx = wx.NewIdRef()
            m_breed = wx.NewIdRef()
            m_mutate = wx.NewIdRef()
            m_clear = wx.NewIdRef()
            menu.Append(m_play, f"Play Buffer {idx}")
            menu.Append(m_to_work, "Load to Working Buffer")
            menu.Append(m_to_track, "Write to Current Track")
            menu.Append(m_dup, "Duplicate Buffer")
            menu.AppendSeparator()
            menu.Append(m_fx, "Apply Effect...")
            menu.AppendSeparator()
            menu.Append(m_breed, "Breed with Another Buffer...")
            menu.Append(m_mutate, "Mutate Buffer")
            menu.AppendSeparator()
            menu.Append(m_clear, "Clear Buffer")

            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/pb {i}'), id=m_play)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/w {i}'), id=m_to_work)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/w {i}\n/ta'), id=m_to_track)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/w {i}\n/a'), id=m_dup)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._show_fx_picker('buffer', i), id=m_fx)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._show_breed_picker(i), id=m_breed)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/w {i}\n/mutate noise 30'), id=m_mutate)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/clr {i}'), id=m_clear)

        # ==============================================================
        # Working Buffer
        # ==============================================================
        elif obj_type == 'working_buffer':
            m_play = wx.NewIdRef()
            m_to_track = wx.NewIdRef()
            m_to_buf = wx.NewIdRef()
            m_dup = wx.NewIdRef()
            m_fx = wx.NewIdRef()
            m_mutate = wx.NewIdRef()
            m_gen_mel = wx.NewIdRef()
            m_gen_beat = wx.NewIdRef()
            m_gen_drone = wx.NewIdRef()
            m_clear = wx.NewIdRef()
            menu.Append(m_play, "Play Working Buffer")
            menu.AppendSeparator()
            menu.Append(m_to_track, "Commit to Current Track")
            menu.Append(m_to_buf, "Commit to Buffer")
            menu.Append(m_dup, "Duplicate to Buffer")
            menu.AppendSeparator()
            menu.Append(m_fx, "Apply Effect...")
            menu.Append(m_mutate, "Mutate...")
            menu.AppendSeparator()
            menu.Append(m_gen_mel, "Generate Melody Here")
            menu.Append(m_gen_beat, "Generate Beat Here")
            menu.Append(m_gen_drone, "Generate Drone Here")
            menu.AppendSeparator()
            menu.Append(m_clear, "Clear Working Buffer")

            self.Bind(wx.EVT_MENU, lambda e: self._exec('/p'), id=m_play)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/ta'), id=m_to_track)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/a'), id=m_to_buf)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/a'), id=m_dup)
            self.Bind(wx.EVT_MENU,
                lambda e: self._show_fx_picker('working', 0), id=m_fx)
            self.Bind(wx.EVT_MENU,
                lambda e: self._show_mutate_picker(), id=m_mutate)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/gen2 melody'), id=m_gen_mel)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/beat house 4'), id=m_gen_beat)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/gen2 drone'), id=m_gen_drone)
            self.Bind(wx.EVT_MENU, lambda e: self._exec('/wbc'), id=m_clear)

        # ==============================================================
        # Deck
        # ==============================================================
        elif obj_type == 'deck':
            idx = data.get('index', 1)
            m_play = wx.NewIdRef()
            m_fx = wx.NewIdRef()
            m_clear = wx.NewIdRef()
            menu.Append(m_play, f"Play Deck {idx}")
            menu.AppendSeparator()
            menu.Append(m_fx, "Apply Effect...")
            menu.AppendSeparator()
            menu.Append(m_clear, "Clear Deck")

            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/pd {i}'), id=m_play)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._show_fx_picker('deck', i), id=m_fx)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._show_placeholder(
                    "Clear Deck",
                    f"Use /deck {i} load to reload, or unload audio manually."),
                id=m_clear)

        # ==============================================================
        # Deck Section
        # ==============================================================
        elif obj_type == 'deck_section':
            dk = data.get('deck', 1)
            start = data.get('start', 0)
            end = data.get('end', 0)
            m_select = wx.NewIdRef()
            m_play = wx.NewIdRef()
            menu.Append(m_select, f"Select Deck {dk}")
            menu.Append(m_play, f"Play Deck {dk}")

            self.Bind(wx.EVT_MENU,
                lambda e, d=dk: self._exec(f'/deck {d}'), id=m_select)
            self.Bind(wx.EVT_MENU,
                lambda e, d=dk: self._exec(f'/pd {d}'), id=m_play)

        # ==============================================================
        # Effect
        # ==============================================================
        elif obj_type == 'effect':
            idx = data.get('index', 0)
            fx_name = data.get('name', f'effect_{idx}')
            fx_params = data.get('params', {})
            m_select = wx.NewIdRef()
            m_amount = wx.NewIdRef()
            m_params = wx.NewIdRef()
            m_bypass = wx.NewIdRef()
            m_remove = wx.NewIdRef()
            m_clear_all = wx.NewIdRef()
            menu.Append(m_select, f"Select: {fx_name}")
            menu.AppendSeparator()
            menu.Append(m_amount, "Set Amount (Wet/Dry)...")
            menu.Append(m_params, "Edit Parameters...")
            menu.AppendSeparator()
            menu.Append(m_bypass, "Toggle Bypass")
            menu.AppendSeparator()
            menu.Append(m_remove, "Remove This Effect")
            menu.Append(m_clear_all, "Clear All Effects")

            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/fxs {i}'), id=m_select)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._show_value_editor(
                    "Effect Amount", "Wet/dry amount (0-100):",
                    str(int(fx_params.get('amount', 50))),
                    f'/fxp {i} {{}}'), id=m_amount)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx, n=fx_name, p=fx_params:
                    self._show_fx_param_editor(i, n, p), id=m_params)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/bypass'), id=m_bypass)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/fxr {i}'), id=m_remove)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/fx clear'), id=m_clear_all)

        # ==============================================================
        # Operator
        # ==============================================================
        elif obj_type == 'operator':
            idx = data.get('index', 0)
            m_select = wx.NewIdRef()
            m_gen = wx.NewIdRef()
            m_wave = wx.NewIdRef()
            m_freq = wx.NewIdRef()
            m_amp = wx.NewIdRef()
            m_fm = wx.NewIdRef()
            m_tfm = wx.NewIdRef()
            m_am = wx.NewIdRef()
            m_rm = wx.NewIdRef()
            m_pm = wx.NewIdRef()
            menu.Append(m_select, f"Select Operator {idx}")
            menu.Append(m_gen, "Generate Tone (440Hz)")
            menu.AppendSeparator()
            menu.Append(m_wave, "Set Waveform...")
            menu.Append(m_freq, "Set Frequency...")
            menu.Append(m_amp, "Set Amplitude...")
            menu.AppendSeparator()
            mod_sub = wx.Menu()
            mod_sub.Append(m_fm, "FM (Frequency Mod)...")
            mod_sub.Append(m_tfm, "TFM (Through-Zero FM)...")
            mod_sub.Append(m_am, "AM (Amplitude Mod)...")
            mod_sub.Append(m_rm, "RM (Ring Mod)...")
            mod_sub.Append(m_pm, "PM (Phase Mod)...")
            menu.AppendSubMenu(mod_sub, f"Add Routing from Op {idx}")

            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/op {i}'), id=m_select)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/op {i}\n/tone 440 1'),
                id=m_gen)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._show_waveform_picker(i), id=m_wave)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._show_value_editor(
                    "Frequency", "Enter frequency (Hz):", "440",
                    f'/op {i}\n/fr {{}}'), id=m_freq)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._show_value_editor(
                    "Amplitude", "Enter amplitude (0-1):", "0.8",
                    f'/op {i}\n/amp {{}}'), id=m_amp)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._show_routing_picker(i, 'fm'), id=m_fm)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._show_routing_picker(i, 'tfm'), id=m_tfm)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._show_routing_picker(i, 'am'), id=m_am)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._show_routing_picker(i, 'rm'), id=m_rm)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._show_routing_picker(i, 'pm'), id=m_pm)

        # ==============================================================
        # Filter Slot
        # ==============================================================
        elif obj_type == 'filter_slot':
            slot = data.get('slot', 0)
            m_select = wx.NewIdRef()
            m_type = wx.NewIdRef()
            m_cutoff = wx.NewIdRef()
            m_res = wx.NewIdRef()
            m_clear = wx.NewIdRef()
            menu.Append(m_select, f"Select Filter {slot}")
            menu.AppendSeparator()
            menu.Append(m_type, "Set Filter Type...")
            menu.Append(m_cutoff, "Set Cutoff...")
            menu.Append(m_res, "Set Resonance...")
            menu.AppendSeparator()
            menu.Append(m_clear, "Clear Filter")

            self.Bind(wx.EVT_MENU,
                lambda e, s=slot: self._exec(f'/fs {s}'), id=m_select)
            self.Bind(wx.EVT_MENU,
                lambda e, s=slot: self._show_filter_type_picker(s), id=m_type)
            self.Bind(wx.EVT_MENU,
                lambda e, s=slot: self._show_value_editor(
                    "Cutoff Frequency", "Enter cutoff (Hz):", "4500",
                    f'/fs {s}\n/cut {{}}'), id=m_cutoff)
            self.Bind(wx.EVT_MENU,
                lambda e, s=slot: self._show_value_editor(
                    "Resonance", "Enter resonance (0-1):", "0.5",
                    f'/fs {s}\n/res {{}}'), id=m_res)
            self.Bind(wx.EVT_MENU,
                lambda e, s=slot: self._show_placeholder(
                    "Clear Filter",
                    f"Filter slot {s} — use /fs {s} then /ft lp to reset type."),
                id=m_clear)

        # ==============================================================
        # Routing (modulation)
        # ==============================================================
        elif obj_type == 'routing':
            idx = data.get('index', 0)
            m_edit = wx.NewIdRef()
            m_clear = wx.NewIdRef()
            menu.Append(m_edit, f"Edit Routing {idx}...")
            menu.AppendSeparator()
            menu.Append(m_clear, "Clear All Routings")

            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._show_placeholder(
                    "Edit Routing",
                    f"Routing {i} editing — use /route command for full control."),
                id=m_edit)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/rt clear'), id=m_clear)

        # ==============================================================
        # Wavetable
        # ==============================================================
        elif obj_type == 'wavetable':
            name = data.get('name', '')
            m_use = wx.NewIdRef()
            m_del = wx.NewIdRef()
            menu.Append(m_use, f"Use Wavetable: {name}")
            menu.AppendSeparator()
            menu.Append(m_del, "Delete Wavetable")

            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._exec(f'/wm wavetable\n/wt use {n}'),
                id=m_use)
            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._exec(f'/wt del {n}'), id=m_del)

        # ==============================================================
        # Compound Wave
        # ==============================================================
        elif obj_type == 'compound':
            name = data.get('name', '')
            m_use = wx.NewIdRef()
            m_del = wx.NewIdRef()
            menu.Append(m_use, f"Use Compound: {name}")
            menu.AppendSeparator()
            menu.Append(m_del, "Delete Compound")

            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._exec(f'/wm compound\n/compound use {n}'),
                id=m_use)
            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._exec(f'/compound del {n}'),
                id=m_del)

        # ==============================================================
        # SyDef (preset)
        # ==============================================================
        elif obj_type == 'sydef':
            name = data.get('name', '')
            m_use = wx.NewIdRef()
            m_use_args = wx.NewIdRef()
            m_show = wx.NewIdRef()
            m_copy = wx.NewIdRef()
            m_del = wx.NewIdRef()

            menu.Append(m_use, f"Use Preset: {name}")
            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._exec(f'/use {n}'), id=m_use)

            menu.Append(m_use_args, "Use with Arguments...")
            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._sydef_use_with_args(n), id=m_use_args)

            menu.AppendSeparator()
            menu.Append(m_show, "Show Contents")
            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._exec(f'/sydef show {n}'), id=m_show)

            menu.Append(m_copy, "Copy...")
            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._sydef_copy_dialog(n), id=m_copy)

            menu.AppendSeparator()
            menu.Append(m_del, "Delete")
            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._exec(f'/sydef del {n}'), id=m_del)

        # ==============================================================
        # Chain
        # ==============================================================
        elif obj_type == 'chain':
            name = data.get('name', '')
            m_apply = wx.NewIdRef()
            m_del = wx.NewIdRef()
            menu.Append(m_apply, f"Apply Chain: {name}")
            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._exec(f'/chain load {n}'), id=m_apply)

        # ==============================================================
        # User Function
        # ==============================================================
        elif obj_type == 'user_function':
            name = data.get('name', '')
            m_run = wx.NewIdRef()
            m_edit = wx.NewIdRef()
            m_del = wx.NewIdRef()
            menu.Append(m_run, f"Run: {name}")
            menu.Append(m_edit, "Edit Definition...")
            menu.AppendSeparator()
            menu.Append(m_del, "Delete Function")

            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._exec(f'/run {n}'), id=m_run)
            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._show_placeholder(
                    "Edit Function",
                    f"Function '{n}' — use /fn to redefine."), id=m_edit)
            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._show_placeholder(
                    "Delete Function",
                    f"Function '{n}' — use /fn to redefine or overwrite."),
                id=m_del)

        # ==============================================================
        # Preset Slot (bank)
        # ==============================================================
        elif obj_type == 'preset_slot':
            slot = data.get('slot', 0)
            m_load = wx.NewIdRef()
            m_info = wx.NewIdRef()
            menu.Append(m_load, f"Load Slot {slot}")
            menu.Append(m_info, "Show Slot Info")

            self.Bind(wx.EVT_MENU,
                lambda e, s=slot: self._show_placeholder(
                    "Load Preset Slot",
                    f"Slot {s} — use /preset <name> to load a saved preset."),
                id=m_load)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/bk'), id=m_info)

        # ==============================================================
        # IR Bank Entry (convolution)
        # ==============================================================
        elif obj_type == 'ir_bank_entry':
            name = data.get('name', '')
            m_use = wx.NewIdRef()
            m_info = wx.NewIdRef()
            menu.Append(m_use, f"Use IR: {name}")
            menu.Append(m_info, "IR Bank Info")

            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._exec(f'/conv use {n}'), id=m_use)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/conv bank'), id=m_info)

        # ==============================================================
        # LFO Shape (impulse LFO)
        # ==============================================================
        elif obj_type == 'lfo_shape':
            name = data.get('name', '')
            m_apply_op = wx.NewIdRef()
            m_apply_filt = wx.NewIdRef()
            m_info = wx.NewIdRef()
            m_clear = wx.NewIdRef()
            menu.Append(m_apply_op, "Apply to Operator...")
            menu.Append(m_apply_filt, "Apply to Filter...")
            menu.AppendSeparator()
            menu.Append(m_info, "Info")
            menu.Append(m_clear, "Clear LFO Modulation")

            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._show_value_editor(
                    "Apply LFO", "Operator index:", "0",
                    f'/impulselfo load {n}\n/impulselfo apply {{}} 4.0'),
                id=m_apply_op)
            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._show_value_editor(
                    "Apply to Filter", "Filter rate (Hz):", "2.0",
                    f'/impulselfo load {n}\n/impulselfo filter {{}}'),
                id=m_apply_filt)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/impulselfo info'), id=m_info)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/impulselfo clear'), id=m_clear)

        # ==============================================================
        # Impulse Envelope Shape
        # ==============================================================
        elif obj_type == 'imp_env_shape':
            name = data.get('name', '')
            m_apply_buf = wx.NewIdRef()
            m_apply_op = wx.NewIdRef()
            m_info = wx.NewIdRef()
            menu.Append(m_apply_buf, "Apply to Working Buffer")
            menu.Append(m_apply_op, "Apply to Operator...")
            menu.AppendSeparator()
            menu.Append(m_info, "Info")

            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._exec(
                    f'/impenv load {n}\n/impenv apply'), id=m_apply_buf)
            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._show_value_editor(
                    "Apply Envelope", "Operator index:", "0",
                    f'/impenv load {n}\n/impenv operator {{}}'),
                id=m_apply_op)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/impenv info'), id=m_info)

        # ==============================================================
        # Convolution Property
        # ==============================================================
        elif obj_type == 'conv_prop':
            m_edit = wx.NewIdRef()
            m_presets = wx.NewIdRef()
            m_clear = wx.NewIdRef()
            menu.Append(m_edit, "Edit Parameters...")
            menu.Append(m_presets, "Load Preset...")
            menu.AppendSeparator()
            menu.Append(m_clear, "Clear Convolution")

            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/conv info'), id=m_edit)
            self.Bind(wx.EVT_MENU,
                lambda e: self._show_conv_preset_picker(), id=m_presets)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/conv clear'), id=m_clear)

        # ==============================================================
        # Envelope Parameter (quick-edit)
        # ==============================================================
        elif obj_type == 'envelope_param':
            param = data.get('param', 'attack')
            cmd_map = {'attack': '/atk', 'decay': '/dec',
                       'sustain': '/sus', 'release': '/rel'}
            cmd = cmd_map.get(param, '/env')
            m_edit = wx.NewIdRef()
            m_reset = wx.NewIdRef()
            menu.Append(m_edit, f"Set {param.capitalize()}...")
            menu.Append(m_reset, f"Reset {param.capitalize()}")

            defaults = {'attack': '0.01', 'decay': '0.1',
                        'sustain': '0.8', 'release': '0.1'}
            self.Bind(wx.EVT_MENU,
                lambda e, c=cmd, p=param: self._show_value_editor(
                    f"Set {p.capitalize()}", f"Enter {p} value:", "0.1",
                    c + ' {}'), id=m_edit)
            self.Bind(wx.EVT_MENU,
                lambda e, c=cmd, d=defaults.get(param, '0.1'):
                    self._exec(f'{c} {d}'), id=m_reset)

        # ==============================================================
        # Filter Envelope Parameter (quick-edit)
        # ==============================================================
        elif obj_type == 'fenv_param':
            param = data.get('param', 'filter_attack')
            short = param.replace('filter_', '')
            cmd_map = {'filter_attack': '/fatk', 'filter_decay': '/fdec',
                       'filter_sustain': '/fsus', 'filter_release': '/frel'}
            cmd = cmd_map.get(param, '/fenv')
            m_edit = wx.NewIdRef()
            m_reset = wx.NewIdRef()
            menu.Append(m_edit, f"Set Filter {short.capitalize()}...")
            menu.Append(m_reset, f"Reset Filter {short.capitalize()}")

            defaults = {'filter_attack': '0.01', 'filter_decay': '0.1',
                        'filter_sustain': '0.8', 'filter_release': '0.1'}
            self.Bind(wx.EVT_MENU,
                lambda e, c=cmd, s=short: self._show_value_editor(
                    f"Filter {s.capitalize()}", f"Enter {s} value:", "0.1",
                    c + ' {}'), id=m_edit)
            self.Bind(wx.EVT_MENU,
                lambda e, c=cmd, d=defaults.get(param, '0.1'):
                    self._exec(f'{c} {d}'), id=m_reset)

        # ==============================================================
        # Voice Property
        # ==============================================================
        elif obj_type == 'voice_prop':
            m_voices = wx.NewIdRef()
            m_algo = wx.NewIdRef()
            m_detune = wx.NewIdRef()
            menu.Append(m_voices, "Set Voice Count...")
            menu.Append(m_algo, "Set Algorithm...")
            menu.Append(m_detune, "Set Detune...")

            self.Bind(wx.EVT_MENU,
                lambda e: self._show_value_editor(
                    "Voices", "Number of voices:", "4",
                    '/v {}'), id=m_voices)
            self.Bind(wx.EVT_MENU,
                lambda e: self._show_voice_algo_picker(), id=m_algo)
            self.Bind(wx.EVT_MENU,
                lambda e: self._show_value_editor(
                    "Detune", "Detune amount (Hz):", "0.5",
                    '/dt {}'), id=m_detune)

        # ==============================================================
        # HQ Property
        # ==============================================================
        elif obj_type == 'hq_prop':
            m_on = wx.NewIdRef()
            m_off = wx.NewIdRef()
            m_dc = wx.NewIdRef()
            m_osc_smooth = wx.NewIdRef()
            m_osc_fast = wx.NewIdRef()
            m_info = wx.NewIdRef()
            menu.Append(m_on, "Enable HQ Mode")
            menu.Append(m_off, "Disable HQ Mode")
            menu.AppendSeparator()
            menu.Append(m_dc, "Toggle DC Removal")
            menu.Append(m_osc_smooth, "HQ Oscillators: Smooth")
            menu.Append(m_osc_fast, "HQ Oscillators: Fast")
            menu.AppendSeparator()
            menu.Append(m_info, "HQ Info")

            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/hq on'), id=m_on)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/hq off'), id=m_off)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/hq dc'), id=m_dc)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/hq osc smooth'), id=m_osc_smooth)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/hq osc fast'), id=m_osc_fast)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/hq'), id=m_info)

        # ==============================================================
        # Engine Property
        # ==============================================================
        elif obj_type == 'engine_prop':
            m_bpm = wx.NewIdRef()
            m_sr = wx.NewIdRef()
            menu.Append(m_bpm, "Set BPM...")
            menu.Append(m_sr, "Set Sample Rate...")

            self.Bind(wx.EVT_MENU,
                lambda e: self._show_value_editor(
                    "BPM", "Enter BPM:", "128", '/bpm {}'), id=m_bpm)
            self.Bind(wx.EVT_MENU,
                lambda e: self._show_sr_picker(), id=m_sr)

        # ==============================================================
        # Generative Items
        # ==============================================================
        elif obj_type == 'gen_genre':
            genre = data.get('genre', 'house')
            m_gen4 = wx.NewIdRef()
            m_gen8 = wx.NewIdRef()
            m_fill = wx.NewIdRef()
            menu.Append(m_gen4, f"Generate 4-bar {genre.title()} Beat")
            menu.Append(m_gen8, f"Generate 8-bar {genre.title()} Beat")
            menu.AppendSeparator()
            menu.Append(m_fill, f"Generate {genre.title()} Fill")
            self.Bind(wx.EVT_MENU,
                lambda e, g=genre: self._exec(f'/beat {g} 4'), id=m_gen4)
            self.Bind(wx.EVT_MENU,
                lambda e, g=genre: self._exec(f'/beat {g} 8'), id=m_gen8)
            self.Bind(wx.EVT_MENU,
                lambda e, g=genre: self._exec(f'/beat fill buildup'), id=m_fill)

        elif obj_type == 'gen_layer':
            layer = data.get('layer', 'drums')
            m_gen = wx.NewIdRef()
            menu.Append(m_gen, f"Generate Loop with {layer.title()}")
            self.Bind(wx.EVT_MENU,
                lambda e, l=layer: self._exec(f'/loop house {l}'), id=m_gen)

        elif obj_type == 'gen_xform_preset':
            preset = data.get('preset', 'reverse')
            m_apply = wx.NewIdRef()
            menu.Append(m_apply, f"Apply '{preset}' Transform")
            self.Bind(wx.EVT_MENU,
                lambda e, p=preset: self._exec(f'/xform preset {p}'), id=m_apply)

        elif obj_type == 'gen_theory_item':
            query = data.get('query', 'scales')
            m_show = wx.NewIdRef()
            menu.Append(m_show, f"Show {query.title()}")
            self.Bind(wx.EVT_MENU,
                lambda e, q=query: self._exec(f'/theory {q}'), id=m_show)

        elif obj_type == 'gen_content_item':
            ctype = data.get('content_type', 'melody')
            m_gen = wx.NewIdRef()
            menu.Append(m_gen, f"Generate {ctype.replace('_', ' ').title()}")
            self.Bind(wx.EVT_MENU,
                lambda e, c=ctype: self._exec(f'/gen2 {c}'), id=m_gen)

        elif obj_type == 'gen_tta_item':
            gen = data.get('generator', 'melody')
            m_gen = wx.NewIdRef()
            if gen in ('beat',):
                menu.Append(m_gen, f"Generate {gen.title()}")
                self.Bind(wx.EVT_MENU,
                    lambda e, g=gen: self._exec(f'/beat house 4'), id=m_gen)
            elif gen in ('loop',):
                menu.Append(m_gen, f"Generate Full Loop")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec(f'/loop house 4'), id=m_gen)
            else:
                menu.Append(m_gen, f"Generate {gen.replace('_', ' ').title()}")
                self.Bind(wx.EVT_MENU,
                    lambda e, g=gen: self._exec(f'/gen2 {g}'), id=m_gen)

        elif obj_type == 'gen_breed_item':
            method = data.get('method', 'breed')
            if method == 'breed':
                m_gen = wx.NewIdRef()
                menu.Append(m_gen, "Breed Buffers 1 + 2")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/breed 1 2 4'), id=m_gen)
            elif method == 'evolve':
                m_gen = wx.NewIdRef()
                menu.Append(m_gen, "Evolve (10 Generations)")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/evolve 10 8'), id=m_gen)
            elif method.startswith('mutate_'):
                mut_type = method.replace('mutate_', '')
                m_gen = wx.NewIdRef()
                menu.Append(m_gen, f"Apply {mut_type.replace('_', ' ').title()} Mutation")
                self.Bind(wx.EVT_MENU,
                    lambda e, m=mut_type: self._exec(f'/mutate {m} 30'), id=m_gen)
            else:
                m_gen = wx.NewIdRef()
                menu.Append(m_gen, f"Crossover ({method.title()})")
                self.Bind(wx.EVT_MENU,
                    lambda e, m=method: self._exec(f'/crossover 1 2 {m}'), id=m_gen)

        elif obj_type == 'gen_section':
            section = data.get('section', '')
            if section == 'beat':
                m_gen = wx.NewIdRef()
                menu.Append(m_gen, "Generate Beat...")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_gen_beat_dialog(), id=m_gen)
            elif section == 'loop':
                m_gen = wx.NewIdRef()
                menu.Append(m_gen, "Generate Loop...")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_gen_loop_dialog(), id=m_gen)
            elif section == 'xform':
                m_list = wx.NewIdRef()
                menu.Append(m_list, "List Transforms")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/xform'), id=m_list)
            elif section == 'theory':
                m_scales = wx.NewIdRef()
                m_chords = wx.NewIdRef()
                menu.Append(m_scales, "Show Scales")
                menu.Append(m_chords, "Show Chords")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/theory scales'), id=m_scales)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/theory chords'), id=m_chords)
            elif section == 'gen2':
                m_mel = wx.NewIdRef()
                m_chd = wx.NewIdRef()
                m_bas = wx.NewIdRef()
                menu.Append(m_mel, "Generate Melody")
                menu.Append(m_chd, "Generate Chord Progression")
                menu.Append(m_bas, "Generate Bassline")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/gen2 melody'), id=m_mel)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/gen2 chord_prog'), id=m_chd)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/gen2 bassline'), id=m_bas)
            elif section == 'text_to_audio':
                m_mel = wx.NewIdRef()
                m_beat = wx.NewIdRef()
                m_loop = wx.NewIdRef()
                m_drone = wx.NewIdRef()
                menu.Append(m_mel, "Generate Melody from Scale")
                menu.Append(m_beat, "Generate Beat from Genre")
                menu.Append(m_loop, "Generate Full Loop")
                menu.Append(m_drone, "Generate Drone")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/gen2 melody'), id=m_mel)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/beat house 4'), id=m_beat)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/loop house 4'), id=m_loop)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/gen2 drone'), id=m_drone)
            elif section == 'breeding':
                m_breed = wx.NewIdRef()
                m_evolve = wx.NewIdRef()
                m_mutate = wx.NewIdRef()
                menu.Append(m_breed, "Breed Buffers 1 + 2")
                menu.Append(m_evolve, "Evolve Population")
                menu.Append(m_mutate, "Mutate Working Buffer")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/breed 1 2 4'), id=m_breed)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/evolve 10 8'), id=m_evolve)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/mutate noise 30'), id=m_mutate)
            elif section == 'phase_t_structure':
                for label, cmd in [('Add Section', '/section add intro 0 8'),
                                   ('List Sections', '/section list'),
                                   ('Commit to Track', '/commit'),
                                   ('Show Position', '/pos')]:
                    m_id = wx.NewIdRef()
                    menu.Append(m_id, label)
                    self.Bind(wx.EVT_MENU,
                        lambda e, c=cmd: self._exec(c), id=m_id)
            elif section == 'phase_t_export':
                for label, cmd in [('Export Stems', '/export stems'),
                                   ('Export Current Track', '/export track 1'),
                                   ('Master Gain +0dB', '/master_gain 0')]:
                    m_id = wx.NewIdRef()
                    menu.Append(m_id, label)
                    self.Bind(wx.EVT_MENU,
                        lambda e, c=cmd: self._exec(c), id=m_id)
            elif section == 'phase_t_tools':
                for label, cmd in [('Crossover Buffers 1 + 2', '/crossover 1 2 temporal'),
                                   ('Duplicate Buffer 1', '/dup 1'),
                                   ('Swap Buffers 1 + 2', '/swap 1 2'),
                                   ('Metronome 4 Bars', '/metronome 4')]:
                    m_id = wx.NewIdRef()
                    menu.Append(m_id, label)
                    self.Bind(wx.EVT_MENU,
                        lambda e, c=cmd: self._exec(c), id=m_id)

        elif obj_type == 'phase_t_cmd':
            cmd = data.get('command', '')
            m_run = wx.NewIdRef()
            menu.Append(m_run, f"Run: {cmd}")
            self.Bind(wx.EVT_MENU,
                lambda e, c=cmd: self._exec(c), id=m_run)

        # ==============================================================
        # Category-level context menus
        # ==============================================================
        elif obj_type == 'category':
            cat_id = data.get('id', '')

            if cat_id == 'tracks':
                m_new = wx.NewIdRef()
                menu.Append(m_new, "Add New Track")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/tn'), id=m_new)

            elif cat_id == 'fx':
                m_add = wx.NewIdRef()
                m_bypass = wx.NewIdRef()
                m_wet = wx.NewIdRef()
                m_half = wx.NewIdRef()
                m_dry = wx.NewIdRef()
                m_apply = wx.NewIdRef()
                m_clear = wx.NewIdRef()
                menu.Append(m_add, "Add Effect...")
                menu.AppendSeparator()
                menu.Append(m_bypass, "Toggle Bypass")
                menu.Append(m_wet, "All Wet (100%)")
                menu.Append(m_half, "All Half (50%)")
                menu.Append(m_dry, "All Dry (0%)")
                menu.AppendSeparator()
                menu.Append(m_apply, "Apply Chain to Working Buffer")
                menu.AppendSeparator()
                menu.Append(m_clear, "Clear All Effects")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_fx_picker('global', 0), id=m_add)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/bypass'), id=m_bypass)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/wet'), id=m_wet)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/half'), id=m_half)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/dry'), id=m_dry)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/fxa'), id=m_apply)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/fx clear'), id=m_clear)

            elif cat_id == 'synth':
                m_add_op = wx.NewIdRef()
                m_wave = wx.NewIdRef()
                menu.Append(m_add_op, "Set Operator Count...")
                menu.Append(m_wave, "Set Waveform...")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_value_editor(
                        "Operators", "Number of operators:", "4",
                        '/mod {}'), id=m_add_op)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_waveform_picker(None), id=m_wave)

            elif cat_id == 'filter':
                m_add = wx.NewIdRef()
                m_type = wx.NewIdRef()
                menu.Append(m_add, "Add Filter Slot")
                menu.Append(m_type, "Set Filter Type...")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_placeholder(
                        "Add Filter",
                        "Use /fs <slot> to select a filter slot."), id=m_add)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_filter_type_picker(None), id=m_type)

            elif cat_id == 'envelope':
                m_preset = wx.NewIdRef()
                m_reset = wx.NewIdRef()
                menu.Append(m_preset, "Apply Envelope Preset...")
                menu.Append(m_reset, "Reset Envelope")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_envelope_preset_picker(), id=m_preset)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/env 0.01 0.1 0.8 0.1'), id=m_reset)

            elif cat_id == 'filter_envelope':
                m_reset = wx.NewIdRef()
                menu.Append(m_reset, "Reset Filter Envelope")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/fatk 0.01\n/fdec 0.1\n/fsus 0.8\n/frel 0.1'),
                    id=m_reset)

            elif cat_id == 'voice':
                m_reset = wx.NewIdRef()
                m_voices = wx.NewIdRef()
                menu.Append(m_voices, "Set Voice Count...")
                menu.Append(m_reset, "Reset Voice Settings")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_value_editor(
                        "Voices", "Number of voices:", "1", '/v {}'),
                    id=m_voices)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/v 1\n/va stack\n/dt 0'),
                    id=m_reset)

            elif cat_id == 'hq':
                m_on = wx.NewIdRef()
                m_off = wx.NewIdRef()
                m_info = wx.NewIdRef()
                menu.Append(m_on, "Enable HQ Mode")
                menu.Append(m_off, "Disable HQ Mode")
                menu.AppendSeparator()
                menu.Append(m_info, "HQ Info")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/hq on'), id=m_on)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/hq off'), id=m_off)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/hq'), id=m_info)

            elif cat_id == 'key':
                m_set = wx.NewIdRef()
                menu.Append(m_set, "Set Key / Scale...")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_key_scale_picker(), id=m_set)

            elif cat_id == 'modulation':
                m_add_fm = wx.NewIdRef()
                m_add_tfm = wx.NewIdRef()
                m_add_am = wx.NewIdRef()
                m_add_rm = wx.NewIdRef()
                m_add_pm = wx.NewIdRef()
                m_show = wx.NewIdRef()
                m_clear = wx.NewIdRef()
                menu.Append(m_add_fm, "Add FM Routing (Frequency Mod)...")
                menu.Append(m_add_tfm, "Add TFM Routing (Through-Zero FM)...")
                menu.Append(m_add_am, "Add AM Routing (Amplitude Mod)...")
                menu.Append(m_add_rm, "Add RM Routing (Ring Mod)...")
                menu.Append(m_add_pm, "Add PM Routing (Phase Mod)...")
                menu.AppendSeparator()
                menu.Append(m_show, "Show All Routings")
                menu.Append(m_clear, "Clear All Routings")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_routing_picker(None, 'fm'),
                    id=m_add_fm)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_routing_picker(None, 'tfm'),
                    id=m_add_tfm)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_routing_picker(None, 'am'),
                    id=m_add_am)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_routing_picker(None, 'rm'),
                    id=m_add_rm)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_routing_picker(None, 'pm'),
                    id=m_add_pm)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/rt'), id=m_show)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/rt clear'), id=m_clear)

            elif cat_id == 'wavetable':
                m_load = wx.NewIdRef()
                menu.Append(m_load, "Generate Wavetable...")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_placeholder(
                        "Wavetable",
                        "Use /wt <file> to generate a wavetable from audio."),
                    id=m_load)

            elif cat_id == 'compound':
                m_create = wx.NewIdRef()
                menu.Append(m_create, "Create Compound Wave...")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_placeholder(
                        "Compound Wave",
                        "Use /compound <name> <wave1> <wave2> ... to define."),
                    id=m_create)

            elif cat_id == 'convolution':
                m_load = wx.NewIdRef()
                m_preset = wx.NewIdRef()
                m_clear = wx.NewIdRef()
                menu.Append(m_load, "Load IR File...")
                menu.Append(m_preset, "Load Preset...")
                menu.AppendSeparator()
                menu.Append(m_clear, "Clear Convolution")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_ir_file_picker(), id=m_load)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_conv_preset_picker(), id=m_preset)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/conv clear'), id=m_clear)

            elif cat_id == 'impulse_lfo':
                m_load = wx.NewIdRef()
                m_list = wx.NewIdRef()
                menu.Append(m_load, "Load LFO from File...")
                menu.Append(m_list, "List LFO Shapes")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_placeholder(
                        "Load LFO",
                        "Use /impulselfo file <path> <name> to import."),
                    id=m_load)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/impulselfo list'), id=m_list)

            elif cat_id == 'impulse_env':
                m_load = wx.NewIdRef()
                m_list = wx.NewIdRef()
                menu.Append(m_load, "Load Envelope from File...")
                menu.Append(m_list, "List Envelopes")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_placeholder(
                        "Load Envelope",
                        "Use /impenv file <path> <name> to import."),
                    id=m_load)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/impenv list'), id=m_list)

            elif cat_id == 'ir_enhance':
                m_extend = wx.NewIdRef()
                m_denoise = wx.NewIdRef()
                m_fill = wx.NewIdRef()
                menu.Append(m_extend, "Extend IR Tail...")
                menu.Append(m_denoise, "Denoise IR")
                menu.Append(m_fill, "Fill IR Gaps")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_value_editor(
                        "Extend IR", "Target duration (seconds):", "3.0",
                        '/irenhance extend {}'), id=m_extend)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/irenhance denoise'), id=m_denoise)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/irenhance fill'), id=m_fill)

            elif cat_id == 'ir_transform':
                m_desc = wx.NewIdRef()
                m_list = wx.NewIdRef()
                menu.Append(m_desc, "Transform by Descriptor...")
                menu.Append(m_list, "List Descriptors")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_ir_descriptor_picker(), id=m_desc)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/irtransform list'), id=m_list)

            elif cat_id == 'ir_granular':
                m_stretch = wx.NewIdRef()
                m_morph = wx.NewIdRef()
                m_redesign = wx.NewIdRef()
                menu.Append(m_stretch, "Stretch IR...")
                menu.Append(m_morph, "Morph IR...")
                menu.Append(m_redesign, "Redesign IR")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_value_editor(
                        "Stretch", "Stretch factor:", "2.0",
                        '/irgranular stretch {}'), id=m_stretch)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_value_editor(
                        "Morph", "Morph amount (0-1):", "0.5",
                        '/irgranular morph {}'), id=m_morph)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/irgranular redesign'), id=m_redesign)

            elif cat_id == 'granular_engine':
                m_proc = wx.NewIdRef()
                m_freeze = wx.NewIdRef()
                m_stretch = wx.NewIdRef()
                m_shift = wx.NewIdRef()
                m_status = wx.NewIdRef()
                menu.Append(m_proc, "Process Buffer (Granular)...")
                menu.Append(m_freeze, "Freeze at Position...")
                menu.Append(m_stretch, "Time-Stretch...")
                menu.Append(m_shift, "Pitch Shift...")
                menu.AppendSeparator()
                menu.Append(m_status, "Show Granular Status")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_value_editor(
                        "Granular Process", "Duration (s):", "2.0",
                        '/gr process {}'), id=m_proc)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_value_editor(
                        "Granular Freeze", "Position (0-1):", "0.5",
                        '/gr freeze {} 4'), id=m_freeze)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_value_editor(
                        "Granular Stretch", "Factor:", "2.0",
                        '/gr stretch {}'), id=m_stretch)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_value_editor(
                        "Granular Pitch Shift", "Semitones:", "0",
                        '/gr shift {}'), id=m_shift)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/gr'), id=m_status)

            elif cat_id == 'gpu':
                m_status = wx.NewIdRef()
                m_steps = wx.NewIdRef()
                m_cfg = wx.NewIdRef()
                m_device = wx.NewIdRef()
                m_reset = wx.NewIdRef()
                menu.Append(m_status, "Show GPU Settings")
                menu.Append(m_steps, "Set Inference Steps...")
                menu.Append(m_cfg, "Set CFG Scale...")
                menu.Append(m_device, "Set Device...")
                menu.AppendSeparator()
                menu.Append(m_reset, "Reset GPU to Defaults")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/gpu'), id=m_status)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_value_editor(
                        "Steps", "Inference steps (1-500):", "150",
                        '/gpu steps {}'), id=m_steps)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_value_editor(
                        "CFG Scale", "CFG (1-30):", "10",
                        '/gpu cfg {}'), id=m_cfg)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_device_picker(), id=m_device)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/gpu reset'), id=m_reset)

            elif cat_id == 'ai_generate':
                m_gen = wx.NewIdRef()
                m_genv = wx.NewIdRef()
                m_analyze = wx.NewIdRef()
                m_describe = wx.NewIdRef()
                m_ask = wx.NewIdRef()
                menu.Append(m_gen, "Generate Audio from Prompt...")
                menu.Append(m_genv, "Generate Variations...")
                menu.AppendSeparator()
                menu.Append(m_analyze, "Analyze Working Buffer")
                menu.Append(m_describe, "Describe Working Buffer")
                menu.AppendSeparator()
                menu.Append(m_ask, "Ask AI (Natural Language)...")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_value_editor(
                        "Generate Audio", "Text prompt:", "warm ambient pad",
                        '/gen {}'), id=m_gen)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_value_editor(
                        "Generate Variations", "Text prompt:", "warm ambient pad",
                        '/genv {} 3'), id=m_genv)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/analyze detailed'), id=m_analyze)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/describe'), id=m_describe)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_value_editor(
                        "Ask AI", "What would you like?:",
                        "make a dark ambient pad in D minor",
                        '/ask {}'), id=m_ask)

            elif cat_id == 'engine':
                m_bpm = wx.NewIdRef()
                m_sr = wx.NewIdRef()
                menu.Append(m_bpm, "Set BPM...")
                menu.Append(m_sr, "Set Sample Rate...")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_value_editor(
                        "BPM", "Enter BPM:", "128", '/bpm {}'), id=m_bpm)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_sr_picker(), id=m_sr)

            elif cat_id == 'buffers':
                m_commit = wx.NewIdRef()
                m_info = wx.NewIdRef()
                m_import = wx.NewIdRef()
                m_clear_all = wx.NewIdRef()
                menu.Append(m_commit, "Commit Working to Buffer")
                menu.Append(m_info, "Buffer Overview")
                menu.AppendSeparator()
                menu.Append(m_import, "Import Audio File...")
                menu.AppendSeparator()
                menu.Append(m_clear_all, "Clear All Buffers")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/a'), id=m_commit)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/b'), id=m_info)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_placeholder(
                        "Import Audio",
                        "Use /import <path.wav> to load audio into working buffer."),
                    id=m_import)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/clr all'), id=m_clear_all)

            elif cat_id == 'decks':
                m_new = wx.NewIdRef()
                menu.Append(m_new, "Add Deck")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/deck+'), id=m_new)

            elif cat_id == 'preset':
                m_save = wx.NewIdRef()
                m_load = wx.NewIdRef()
                menu.Append(m_save, "Save Current as Preset...")
                menu.Append(m_load, "Load Preset...")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_value_editor(
                        "Save Preset", "Enter preset name:", "my_preset",
                        '/preset save {}'), id=m_save)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_placeholder(
                        "Load Preset",
                        "Use /preset load <name> or /use <name>."),
                    id=m_load)

            elif cat_id == 'bank':
                m_save = wx.NewIdRef()
                m_info = wx.NewIdRef()
                menu.Append(m_save, "Save Routing Bank...")
                menu.Append(m_info, "Bank Info")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_value_editor(
                        "Save Bank", "Enter bank name:", "my_bank",
                        '/bk save {}'), id=m_save)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/bk list'), id=m_info)

            elif cat_id == 'generative':
                m_beat = wx.NewIdRef()
                m_loop = wx.NewIdRef()
                m_melody = wx.NewIdRef()
                m_theory = wx.NewIdRef()
                menu.Append(m_beat, "Generate Beat...")
                menu.Append(m_melody, "Generate Melody...")
                menu.Append(m_loop, "Generate Full Loop...")
                menu.AppendSeparator()
                menu.Append(m_theory, "Music Theory Info")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_gen_beat_dialog(), id=m_beat)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/gen2 melody'), id=m_melody)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_gen_loop_dialog(), id=m_loop)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/theory scales'), id=m_theory)

        if menu.GetMenuItemCount() > 0:
            self.PopupMenu(menu)
        menu.Destroy()

    def _exec(self, command: str):
        """Execute one or more newline-separated commands via the executor."""
        commands = [c.strip() for c in command.split('\n') if c.strip()]
        last_ok = True
        for cmd in commands:
            if self.console_cb:
                self.console_cb(f">>> {cmd}\n", 'command')
            stdout, stderr, ok = self.executor.execute(cmd)
            last_ok = ok
            if self.console_cb:
                if stdout:
                    self.console_cb(stdout, 'stdout')
                if stderr:
                    self.console_cb(stderr, 'stderr')
                status = "OK" if ok else "ERROR"
                self.console_cb(f"[{status}]\n", 'success' if ok else 'error')
            if not ok:
                break
        if self.console_cb:
            self.console_cb("\n", 'stdout')
        self.populate_tree()

    def _sydef_use_with_args(self, name):
        """Show a dialog to supply arguments when using a SyDef preset."""
        s = self.executor.session
        sydef = None
        if s and hasattr(s, 'sydefs') and name in s.sydefs:
            sydef = s.sydefs[name]

        params = getattr(sydef, 'params', []) if sydef else []
        if not params:
            # No parameters — just use directly
            self._exec(f'/use {name}')
            return

        dlg = wx.Dialog(self, title=f"Use SyDef: {name}", size=(400, 300))
        panel = wx.Panel(dlg)
        sizer = wx.BoxSizer(wx.VERTICAL)

        sizer.Add(wx.StaticText(panel, label=f"Parameters for {name}:"),
                   0, wx.ALL, 5)

        entries = []
        for p in params:
            p_name = p.name if hasattr(p, 'name') else str(p)
            p_default = str(p.default) if hasattr(p, 'default') else ''
            row = wx.BoxSizer(wx.HORIZONTAL)
            lbl = wx.StaticText(panel, label=f"${p_name}:")
            txt = wx.TextCtrl(panel, value=p_default, name=f"Param {p_name}")
            txt.SetName(f"Parameter {p_name}, default {p_default}")
            row.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
            row.Add(txt, 1, wx.EXPAND)
            sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)
            entries.append((p_name, txt))

        btn_sizer = dlg.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 5)

        panel.SetSizer(sizer)
        dlg_sizer = wx.BoxSizer(wx.VERTICAL)
        dlg_sizer.Add(panel, 1, wx.EXPAND)
        dlg.SetSizer(dlg_sizer)
        dlg.Fit()

        if dlg.ShowModal() == wx.ID_OK:
            arg_parts = []
            for p_name, txt in entries:
                val = txt.GetValue().strip()
                if val:
                    arg_parts.append(f'{p_name}={val}')
            arg_str = ' '.join(arg_parts)
            self._exec(f'/use {name} {arg_str}')
        dlg.Destroy()

    def _sydef_copy_dialog(self, name):
        """Show a dialog to copy a SyDef preset under a new name."""
        dlg = wx.TextEntryDialog(self, f"Copy '{name}' — enter new name:",
                                  "Copy SyDef", f"{name}_copy")
        if dlg.ShowModal() == wx.ID_OK:
            new_name = dlg.GetValue().strip()
            if new_name:
                self._exec(f'/sydef copy {name} {new_name}')
        dlg.Destroy()

    # ------------------------------------------------------------------
    # Categorised effect catalogue (aliases → canonical names)
    # ------------------------------------------------------------------
    _FX_CATALOGUE = {
        'Reverb': [
            ('reverb', 'reverb_small', 'Small Room Reverb'),
            ('r2', 'reverb_large', 'Large Hall Reverb'),
            ('r3', 'reverb_plate', 'Plate Reverb'),
            ('r4', 'reverb_spring', 'Spring Reverb'),
            ('r5', 'reverb_cathedral', 'Cathedral Reverb'),
        ],
        'Convolution Reverb': [
            ('hall', 'conv_hall', 'Hall IR'),
            ('cr2', 'conv_hall_long', 'Long Hall IR'),
            ('room', 'conv_room', 'Room IR'),
            ('plate', 'conv_plate', 'Plate IR'),
            ('spring', 'conv_spring', 'Spring IR'),
            ('shimmer', 'conv_shimmer', 'Shimmer IR'),
            ('reverse', 'conv_reverse', 'Reverse IR'),
        ],
        'Delay': [
            ('delay', 'delay_simple', 'Simple Delay'),
            ('pingpong', 'delay_pingpong', 'Ping Pong Delay'),
            ('d3', 'delay_multitap', 'Multi-Tap Delay'),
            ('slapback', 'delay_slapback', 'Slapback Delay'),
            ('tape', 'delay_tape', 'Tape Delay'),
        ],
        'Saturation': [
            ('saturate', 'saturate_soft', 'Soft Saturation'),
            ('s2', 'saturate_hard', 'Hard Saturation'),
            ('s3', 'saturate_overdrive', 'Overdrive Saturation'),
            ('fuzz', 'saturate_fuzz', 'Fuzz Saturation'),
            ('s5', 'saturate_tube', 'Tube Saturation'),
        ],
        'Vamp / Overdrive': [
            ('v1', 'vamp_light', 'Light Vamp'),
            ('vamp', 'vamp_medium', 'Medium Vamp'),
            ('v3', 'vamp_heavy', 'Heavy Vamp'),
            ('v4', 'vamp_fuzz', 'Fuzz Vamp'),
            ('od', 'overdrive_classic', 'Classic Overdrive'),
            ('o1', 'overdrive_soft', 'Soft Overdrive'),
            ('crunch', 'overdrive_crunch', 'Crunch Overdrive'),
            ('dual', 'dual_od_warm', 'Dual OD Warm'),
            ('do2', 'dual_od_bright', 'Dual OD Bright'),
            ('do3', 'dual_od_heavy', 'Dual OD Heavy'),
        ],
        'Waveshaping': [
            ('fold', 'waveshape_fold', 'Wave Fold'),
            ('rect', 'waveshape_rectify', 'Wave Rectify'),
            ('ws3', 'waveshape_sine', 'Sine Shaper'),
        ],
        'Dynamics': [
            ('compress', 'compress_mild', 'Mild Compressor'),
            ('c2', 'compress_hard', 'Hard Compressor'),
            ('limiter', 'compress_limiter', 'Limiter'),
            ('expander', 'compress_expander', 'Expander'),
            ('softclip', 'compress_softclipper', 'Soft Clipper'),
        ],
        'Forever Compression': [
            ('fc_punch', 'fc_punch', 'FC Punch'),
            ('fc_glue', 'fc_glue', 'FC Glue'),
            ('fc_loud', 'fc_loud', 'FC Loud'),
            ('fc_soft', 'fc_soft', 'FC Soft'),
            ('fc_ott', 'fc_ott', 'FC OTT'),
        ],
        'Gate': [
            ('g1', 'gate1', 'Gate — Tight'),
            ('gate', 'gate2', 'Gate — Standard'),
            ('g3', 'gate3', 'Gate — Medium'),
            ('g4', 'gate4', 'Gate — Loose'),
            ('g5', 'gate5', 'Gate — Open'),
        ],
        'Giga Gate': [
            ('gg_half', 'gg_half', 'GG Half'),
            ('gg_quarter', 'gg_quarter', 'GG Quarter'),
            ('gg_tresillo', 'gg_tresillo', 'GG Tresillo'),
            ('gg_glitch', 'gg_glitch', 'GG Glitch'),
            ('gg_stutter', 'gg_stutter', 'GG Stutter'),
            ('gg_halftime', 'gg_halftime', 'GG Halftime'),
            ('gg_tape_stop', 'gg_tape_stop', 'GG Tape Stop'),
        ],
        'Lo-fi': [
            ('bitcrush', 'lofi_bitcrush', 'Bitcrush'),
            ('chorus', 'lofi_chorus', 'LoFi Chorus'),
            ('flanger', 'lofi_flanger', 'LoFi Flanger'),
            ('phaser', 'lofi_phaser', 'LoFi Phaser'),
            ('l5', 'lofi_filter', 'LoFi Filter'),
            ('halftime', 'lofi_halftime', 'LoFi Halftime'),
        ],
        'Vocoder': [
            ('vocoder_synth', 'vocoder_synth', 'Vocoder Synth'),
            ('vocoder_noise', 'vocoder_noise', 'Vocoder Noise'),
            ('vocoder_chord', 'vocoder_chord', 'Vocoder Chord'),
        ],
        'Spectral': [
            ('spc_freeze', 'spc_freeze', 'Spectral Freeze'),
            ('spc_blur', 'spc_blur', 'Spectral Blur'),
            ('spc_shift_up', 'spc_shift_up', 'Spectral Shift Up'),
            ('spc_shift_down', 'spc_shift_down', 'Spectral Shift Down'),
        ],
        'Filter': [
            ('lp', 'filter_lowpass', 'Low Pass'),
            ('lp1', 'filter_lowpass_soft', 'Low Pass Soft'),
            ('lp3', 'filter_lowpass_hard', 'Low Pass Hard'),
            ('hp', 'filter_highpass', 'High Pass'),
            ('hp1', 'filter_highpass_soft', 'High Pass Soft'),
            ('hp3', 'filter_highpass_hard', 'High Pass Hard'),
            ('bp', 'filter_bandpass', 'Band Pass'),
            ('telephone', 'filter_bandpass_narrow', 'Telephone (Narrow BP)'),
        ],
        'Pitch Shift': [
            ('pu2', 'pitch_up_2', 'Pitch +2 semitones'),
            ('pu5', 'pitch_up_5', 'Pitch +5 semitones'),
            ('pu7', 'pitch_up_7', 'Pitch +7 semitones'),
            ('octave_up', 'pitch_up_12', 'Octave Up'),
            ('pd2', 'pitch_down_2', 'Pitch -2 semitones'),
            ('pd5', 'pitch_down_5', 'Pitch -5 semitones'),
            ('pd7', 'pitch_down_7', 'Pitch -7 semitones'),
            ('octave_down', 'pitch_down_12', 'Octave Down'),
        ],
        'Harmonizer': [
            ('h3', 'harmonizer_3rd', 'Harmonize 3rd'),
            ('h5', 'harmonizer_5th', 'Harmonize 5th'),
            ('h8', 'harmonizer_octave', 'Harmonize Octave'),
            ('hchord', 'harmonizer_chord', 'Harmonize Chord'),
        ],
        'Granular': [
            ('cloud', 'granular_cloud', 'Granular Cloud'),
            ('scatter', 'granular_scatter', 'Granular Scatter'),
            ('grstretch', 'granular_stretch', 'Granular Stretch'),
            ('grfreeze', 'granular_freeze', 'Granular Freeze'),
            ('grshimmer', 'granular_shimmer', 'Granular Shimmer'),
            ('grrev', 'granular_reverse', 'Granular Reverse'),
            ('grstutter', 'granular_stutter', 'Granular Stutter'),
        ],
        'LFO': [
            ('lfo_filter_slow', 'lfo_filter_slow', 'LFO Filter Slow'),
            ('lfo_filter_fast', 'lfo_filter_fast', 'LFO Filter Fast'),
            ('lfo_tremolo', 'lfo_tremolo', 'LFO Tremolo'),
            ('lfo_vibrato', 'lfo_vibrato', 'LFO Vibrato'),
        ],
        'Stereo': [
            ('stereo_wide', 'stereo_wide', 'Stereo Widen'),
            ('stereo_narrow', 'stereo_narrow', 'Stereo Narrow'),
        ],
        'Utility': [
            ('normalize', 'util_normalize', 'Normalize (Peak)'),
            ('lufs', 'util_normalize_rms', 'Normalize (RMS/LUFS)'),
            ('declip', 'util_declip', 'Declip'),
            ('declick', 'util_declick', 'Declick'),
            ('smooth', 'util_smooth', 'Smooth / Warmth'),
            ('muffle', 'util_smooth_heavy', 'Heavy Smooth / Muffle'),
            ('dc', 'util_dc_remove', 'DC Offset Remove'),
            ('fadein', 'util_fade_in', 'Fade In'),
            ('fadeout', 'util_fade_out', 'Fade Out'),
            ('fades', 'util_fade_both', 'Fade In + Out'),
        ],
    }

    def _show_fx_picker(self, target_type: str, target_id: int):
        """Show a categorised dialog to pick an effect and apply it."""
        # Build flat list for display: "Category: Description  (alias)"
        choices = []
        alias_map = {}   # display string → alias
        for cat, entries in self._FX_CATALOGUE.items():
            for alias, canonical, desc in entries:
                label = f"{cat}:  {desc}  ({alias})"
                choices.append(label)
                alias_map[label] = alias

        dlg = wx.SingleChoiceDialog(
            self, "Select an effect to apply:\n"
            "Effects are listed as  Category: Description (shortcut code)",
            "Apply Effect — Full Catalogue", choices)
        dlg.SetSize((520, 540))

        if dlg.ShowModal() == wx.ID_OK:
            fx_alias = alias_map[dlg.GetStringSelection()]
            if target_type == 'track':
                self._exec(f'/tsel {target_id+1}\n/fx {fx_alias}')
            elif target_type == 'buffer':
                self._exec(f'/bu {target_id}\n/fx {fx_alias}')
            elif target_type == 'deck':
                self._exec(f'/deck {target_id}\n/fx {fx_alias}')
            else:
                self._exec(f'/fx {fx_alias}')
        dlg.Destroy()

    # ------------------------------------------------------------------
    # Context-menu helper dialogs
    # ------------------------------------------------------------------

    def _show_placeholder(self, title: str, message: str):
        """Show an informational placeholder dialog for not-yet-wired features."""
        wx.MessageBox(message, title, wx.OK | wx.ICON_INFORMATION, self)

    def _show_value_editor(self, title: str, prompt: str, default: str,
                           cmd_template: str):
        """Show a text entry dialog and execute a command with the value.

        *cmd_template* should contain ``{}`` where the user value goes.
        """
        dlg = wx.TextEntryDialog(self, prompt, title, default)
        if dlg.ShowModal() == wx.ID_OK:
            val = dlg.GetValue().strip()
            if val:
                self._exec(cmd_template.format(val))
        dlg.Destroy()

    def _show_gen_beat_dialog(self):
        """Show dialog for generating a drum beat."""
        try:
            from mdma_rebuild.dsp import beat_gen
            genres = sorted(beat_gen.GENRE_TEMPLATES.keys())
        except ImportError:
            genres = ['house', 'techno', 'hiphop', 'trap', 'dnb', 'lofi']
        dlg = wx.SingleChoiceDialog(self, "Select genre:", "Generate Beat",
                                     [g.title() for g in genres])
        if dlg.ShowModal() == wx.ID_OK:
            genre = genres[dlg.GetSelection()]
            bars_dlg = wx.TextEntryDialog(self, "Number of bars:", "Bars", "4")
            if bars_dlg.ShowModal() == wx.ID_OK:
                bars = bars_dlg.GetValue().strip() or '4'
                self._exec(f'/beat {genre} {bars}')
            bars_dlg.Destroy()
        dlg.Destroy()

    def _show_gen_loop_dialog(self):
        """Show dialog for generating a full loop."""
        try:
            from mdma_rebuild.dsp import beat_gen
            genres = sorted(beat_gen.GENRE_TEMPLATES.keys())
        except ImportError:
            genres = ['house', 'techno', 'hiphop', 'trap', 'dnb', 'lofi']
        dlg = wx.SingleChoiceDialog(self, "Select genre:", "Generate Loop",
                                     [g.title() for g in genres])
        if dlg.ShowModal() == wx.ID_OK:
            genre = genres[dlg.GetSelection()]
            layers = ['full', 'drums', 'drums bass', 'drums bass chords',
                      'drums melody', 'bass chords melody']
            layer_dlg = wx.SingleChoiceDialog(self, "Select layers:",
                                               "Loop Layers", layers)
            if layer_dlg.ShowModal() == wx.ID_OK:
                layer_choice = layers[layer_dlg.GetSelection()]
                self._exec(f'/loop {genre} {layer_choice}')
            layer_dlg.Destroy()
        dlg.Destroy()

    def _show_breed_picker(self, source_buf: int):
        """Show dialog to pick a second buffer for breeding."""
        other_bufs = [str(i) for i in range(1, 11) if i != source_buf]
        dlg = wx.SingleChoiceDialog(
            self,
            f"Breed buffer {source_buf} with:",
            "Genetic Breeding",
            other_bufs)
        if dlg.ShowModal() == wx.ID_OK:
            other = other_bufs[dlg.GetSelection()]
            methods = ['breed (auto)', 'temporal', 'spectral', 'blend',
                       'morphological', 'multi_point']
            m_dlg = wx.SingleChoiceDialog(
                self, "Crossover method:", "Breeding Method", methods)
            if m_dlg.ShowModal() == wx.ID_OK:
                method = methods[m_dlg.GetSelection()]
                if method == 'breed (auto)':
                    self._exec(f'/breed {source_buf} {other} 4')
                else:
                    self._exec(f'/crossover {source_buf} {other} {method}')
            m_dlg.Destroy()
        dlg.Destroy()

    def _show_mutate_picker(self):
        """Show dialog to pick a mutation type for the working buffer."""
        mutations = ['noise', 'pitch', 'time_stretch', 'freq_shift',
                     'envelope', 'reverse_segment', 'spectral_smear']
        labels = ['Noise (random perturbation)',
                  'Pitch (resampling shift)',
                  'Time Stretch (segment warping)',
                  'Frequency Shift (band displacement)',
                  'Envelope (amplitude modulation)',
                  'Reverse Segment (flip chunks)',
                  'Spectral Smear (blur frequencies)']
        dlg = wx.SingleChoiceDialog(
            self, "Select mutation type:", "Mutate Working Buffer", labels)
        if dlg.ShowModal() == wx.ID_OK:
            mutation = mutations[dlg.GetSelection()]
            amt_dlg = wx.TextEntryDialog(
                self, "Mutation amount (1-100):", "Amount", "30")
            if amt_dlg.ShowModal() == wx.ID_OK:
                amt = amt_dlg.GetValue().strip() or '30'
                self._exec(f'/mutate {mutation} {amt}')
            amt_dlg.Destroy()
        dlg.Destroy()

    # ------------------------------------------------------------------
    # Known DSP parameter ranges (mirrors effects.py _get_param_range)
    # ------------------------------------------------------------------
    _PARAM_RANGES = {
        'amount': (0, 100, 'Wet/dry mix %'),
        'wet': (0, 100, 'Wet level %'),
        'dry': (0, 100, 'Dry level %'),
        'drive': (0, 100, 'Drive amount'),
        'gain': (0, 100, 'Output gain %'),
        'mix': (0, 100, 'Mix %'),
        'blend': (0, 100, 'Blend %'),
        'output': (0, 100, 'Output level %'),
        'threshold': (0, 100, 'Threshold'),
        'ratio': (0, 100, 'Ratio'),
        'makeup': (0, 100, 'Makeup gain'),
        'depth': (0, 100, 'Depth'),
        'low_amount': (0, 100, 'Low band amount'),
        'mid_amount': (0, 100, 'Mid band amount'),
        'high_amount': (0, 100, 'High band amount'),
        'upward': (0, 100, 'Upward compression'),
        'downward': (0, 100, 'Downward compression'),
        'feedback': (0, 100, 'Feedback %'),
        'decay': (0, 100, 'Decay'),
        'attack': (0.1, 100, 'Attack ms'),
        'release': (0.1, 100, 'Release ms'),
        'post_filter': (20, 20000, 'Post-filter cutoff Hz'),
        'pre_filter': (20, 20000, 'Pre-filter cutoff Hz'),
        'high_cut': (20, 20000, 'High cut Hz'),
        'low_cut': (20, 20000, 'Low cut Hz'),
        'crossover': (20, 20000, 'Crossover frequency Hz'),
        'delay_time': (0, 2000, 'Delay time ms'),
        'pre_delay': (0, 200, 'Pre-delay ms'),
        'stretch': (0.1, 4.0, 'Stretch factor'),
        'semitones': (-24, 24, 'Semitones'),
        'position': (0, 1.0, 'Position 0-1'),
        'steps': (1, 64, 'Steps'),
        'lfo_rate': (0.01, 20, 'LFO rate Hz'),
    }

    # Per-effect editable parameters (canonical_name → list of param keys)
    _FX_DSP_PARAMS = {
        # Convolution reverbs
        'conv_hall': ['amount', 'wet', 'dry', 'stretch', 'pre_delay', 'high_cut', 'low_cut'],
        'conv_hall_long': ['amount', 'wet', 'dry', 'stretch', 'pre_delay', 'high_cut', 'low_cut'],
        'conv_room': ['amount', 'wet', 'dry', 'stretch', 'pre_delay', 'high_cut', 'low_cut'],
        'conv_plate': ['amount', 'wet', 'dry', 'stretch', 'pre_delay', 'high_cut', 'low_cut'],
        'conv_spring': ['amount', 'wet', 'dry', 'stretch', 'pre_delay', 'high_cut', 'low_cut'],
        'conv_shimmer': ['amount', 'wet', 'dry', 'stretch', 'pre_delay', 'high_cut', 'low_cut'],
        'conv_reverse': ['amount', 'wet', 'dry', 'stretch', 'pre_delay', 'high_cut', 'low_cut'],
        # Vamp / overdrive
        'vamp_light': ['amount', 'drive', 'post_filter', 'gain', 'mix'],
        'vamp_medium': ['amount', 'drive', 'post_filter', 'gain', 'mix'],
        'vamp_heavy': ['amount', 'drive', 'post_filter', 'pre_filter', 'gain', 'mix'],
        'vamp_fuzz': ['amount', 'drive', 'post_filter', 'gain', 'mix'],
        'overdrive_soft': ['amount', 'drive', 'post_filter', 'gain', 'mix'],
        'overdrive_classic': ['amount', 'drive', 'post_filter', 'pre_filter', 'gain', 'mix'],
        'overdrive_crunch': ['amount', 'drive', 'post_filter', 'gain', 'mix'],
        'waveshape_fold': ['amount', 'drive', 'gain', 'mix'],
        'waveshape_rectify': ['amount', 'drive', 'post_filter', 'mix'],
        'waveshape_sine': ['amount', 'drive', 'post_filter', 'mix'],
        # Dual overdrive
        'dual_od_warm': ['amount', 'drive_low', 'drive_high', 'crossover', 'blend', 'gain'],
        'dual_od_bright': ['amount', 'drive_low', 'drive_high', 'crossover', 'blend', 'gain'],
        'dual_od_heavy': ['amount', 'drive_low', 'drive_high', 'crossover', 'blend', 'gain'],
        # Dynamics
        'compress_mild': ['amount', 'threshold', 'ratio', 'makeup'],
        'compress_hard': ['amount', 'threshold', 'ratio', 'makeup'],
        'compress_limiter': ['amount', 'threshold', 'ratio', 'makeup'],
        # Forever compression
        'fc_punch': ['amount', 'depth', 'low_amount', 'mid_amount', 'high_amount', 'upward', 'downward', 'mix', 'output'],
        'fc_glue': ['amount', 'depth', 'low_amount', 'mid_amount', 'high_amount', 'upward', 'downward', 'mix', 'output'],
        'fc_loud': ['amount', 'depth', 'low_amount', 'mid_amount', 'high_amount', 'upward', 'downward', 'mix', 'output'],
        'fc_soft': ['amount', 'depth', 'low_amount', 'mid_amount', 'high_amount', 'upward', 'downward', 'mix', 'output'],
        'fc_ott': ['amount', 'depth', 'low_amount', 'mid_amount', 'high_amount', 'upward', 'downward', 'mix', 'output'],
        # Gate
        'gate1': ['amount', 'threshold', 'attack', 'release'],
        'gate2': ['amount', 'threshold', 'attack', 'release'],
        'gate3': ['amount', 'threshold', 'attack', 'release'],
        'gate4': ['amount', 'threshold', 'attack', 'release'],
        'gate5': ['amount', 'threshold', 'attack', 'release'],
        # Giga gate
        'gg_half': ['amount', 'steps', 'attack', 'release', 'mix'],
        'gg_quarter': ['amount', 'steps', 'attack', 'release', 'mix'],
        'gg_tresillo': ['amount', 'steps', 'attack', 'release', 'mix'],
        'gg_glitch': ['amount', 'steps', 'attack', 'release', 'mix'],
        'gg_stutter': ['amount', 'steps', 'attack', 'release', 'mix'],
        'gg_halftime': ['amount', 'steps', 'attack', 'release', 'mix'],
        'gg_tape_stop': ['amount', 'steps', 'attack', 'release', 'mix'],
        # Spectral
        'spc_freeze': ['amount', 'position'],
        'spc_blur': ['amount'],
        'spc_shift_up': ['amount', 'semitones'],
        'spc_shift_down': ['amount', 'semitones'],
        # Basic reverbs
        'reverb_small': ['amount', 'decay', 'wet'],
        'reverb_large': ['amount', 'decay', 'wet'],
        'reverb_plate': ['amount', 'decay', 'wet'],
        'reverb_spring': ['amount', 'decay', 'wet'],
        'reverb_cathedral': ['amount', 'decay', 'wet'],
        # Delays
        'delay_simple': ['amount', 'delay_time', 'feedback', 'wet'],
        'delay_pingpong': ['amount', 'delay_time', 'feedback', 'wet'],
        'delay_multitap': ['amount', 'delay_time', 'feedback', 'wet'],
        'delay_slapback': ['amount', 'delay_time', 'feedback', 'wet'],
        'delay_tape': ['amount', 'delay_time', 'feedback', 'wet'],
        # Saturations
        'saturate_soft': ['amount', 'drive'],
        'saturate_hard': ['amount', 'drive'],
        'saturate_overdrive': ['amount', 'drive'],
        'saturate_fuzz': ['amount', 'drive'],
        'saturate_tube': ['amount', 'drive'],
    }

    def _show_fx_param_editor(self, fx_index: int, fx_name: str,
                               current_params: dict):
        """Show a multi-parameter editor dialog for an effect's DSP params."""
        known_params = self._FX_DSP_PARAMS.get(fx_name, ['amount'])
        if isinstance(known_params, list) and not known_params:
            known_params = ['amount']

        dlg = wx.Dialog(self, title=f"Parameters — {fx_name}",
                        style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        panel = wx.Panel(dlg)
        sizer = wx.FlexGridSizer(cols=3, hgap=8, vgap=6)
        sizer.AddGrowableCol(1, 1)

        controls = {}
        for param_key in known_params:
            lo, hi, desc = self._PARAM_RANGES.get(param_key, (0, 100, param_key))
            current_val = current_params.get(param_key, (lo + hi) / 2)

            label = wx.StaticText(panel, label=f"{param_key}:")
            label.SetToolTip(desc)

            # Use slider + spin for numeric ranges
            slider = wx.Slider(panel, value=int(current_val), minValue=int(lo),
                               maxValue=int(hi),
                               style=wx.SL_HORIZONTAL)
            slider.SetName(f"{param_key} slider")
            spin = wx.SpinCtrlDouble(panel, value=str(current_val),
                                      min=float(lo), max=float(hi),
                                      inc=0.1 if hi <= 4 else 1.0)
            spin.SetValue(current_val)
            spin.SetMinSize(wx.Size(80, -1))
            spin.SetName(f"{param_key} value")

            # Sync slider ↔ spin
            def _on_slider(evt, sp=spin):
                sp.SetValue(evt.GetInt())
            def _on_spin(evt, sl=slider):
                sl.SetValue(int(evt.GetValue()))
            slider.Bind(wx.EVT_SLIDER, _on_slider)
            spin.Bind(wx.EVT_SPINCTRLDOUBLE, _on_spin)

            sizer.Add(label, 0, wx.ALIGN_CENTER_VERTICAL)
            sizer.Add(slider, 1, wx.EXPAND)
            sizer.Add(spin, 0, wx.ALIGN_CENTER_VERTICAL)
            controls[param_key] = spin

        panel.SetSizer(sizer)

        btn_sizer = dlg.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(panel, 1, wx.ALL | wx.EXPAND, 10)
        main_sizer.Add(btn_sizer, 0, wx.ALL | wx.EXPAND, 5)
        dlg.SetSizer(main_sizer)
        dlg.SetSize((440, min(80 + len(known_params) * 45, 600)))
        dlg.CenterOnParent()
        # Set initial focus to first spin control for keyboard accessibility
        if controls:
            first_ctrl = next(iter(controls.values()))
            first_ctrl.SetFocus()

        if dlg.ShowModal() == wx.ID_OK:
            cmds = []
            for param_key, spin_ctrl in controls.items():
                val = spin_ctrl.GetValue()
                if param_key == 'amount':
                    cmds.append(f'/fxp {fx_index} {val:.1f}')
                else:
                    cmds.append(f'/fxp {fx_index} {param_key} {val:.1f}')
            self._exec('\n'.join(cmds))
        dlg.Destroy()

    def _show_waveform_picker(self, op_index):
        """Show a picker for waveform types, optionally targeting an operator."""
        waveforms = ['sine', 'saw', 'square', 'triangle', 'noise',
                     'supersaw', 'additive', 'formant', 'harmonic',
                     'physical', 'physical2', 'wavetable', 'compound',
                     'pluck', 'string', 'reed', 'flute', 'drum', 'bell']
        dlg = wx.SingleChoiceDialog(self, "Select waveform:",
                                     "Set Waveform", waveforms)
        if dlg.ShowModal() == wx.ID_OK:
            wf = dlg.GetStringSelection()
            if op_index is not None:
                self._exec(f'/op {op_index}\n/wm {wf}')
            else:
                self._exec(f'/wm {wf}')
        dlg.Destroy()

    def _show_filter_type_picker(self, slot):
        """Show a picker for filter types, optionally targeting a slot."""
        ftypes = ['lp', 'hp', 'bp', 'notch', 'peak', 'lowshelf',
                  'highshelf', 'allpass', 'acid', 'moog', 'comb',
                  'formant', 'vowel']
        dlg = wx.SingleChoiceDialog(self, "Select filter type:",
                                     "Set Filter Type", ftypes)
        if dlg.ShowModal() == wx.ID_OK:
            ft = dlg.GetStringSelection()
            if slot is not None:
                self._exec(f'/fs {slot}\n/ft {ft}')
            else:
                self._exec(f'/ft {ft}')
        dlg.Destroy()

    def _show_routing_picker(self, op_index, route_type='fm'):
        """Show a dialog to add a modulation routing.

        If *op_index* is not None, it's pre-filled as the source operator
        and the user only needs to enter the target (and optional amount).
        """
        if op_index is not None:
            dlg = wx.TextEntryDialog(
                self,
                f"Enter target operator for {route_type.upper()} from Op {op_index}:\n"
                f"(e.g. '0' or '0 2.5' for target with amount)",
                f"Add {route_type.upper()} Routing from Op {op_index}",
                "0")
            if dlg.ShowModal() == wx.ID_OK:
                val = dlg.GetValue().strip()
                if val:
                    self._exec(f'/{route_type} {op_index} {val}')
            dlg.Destroy()
        else:
            dlg = wx.TextEntryDialog(
                self,
                f"Enter {route_type.upper()} routing  source target [amount]:\n"
                f"e.g. '1 0' or '1 0 2.5'",
                f"Add {route_type.upper()} Routing",
                "1 0")
            if dlg.ShowModal() == wx.ID_OK:
                val = dlg.GetValue().strip()
                if val:
                    self._exec(f'/{route_type} {val}')
            dlg.Destroy()

    def _show_device_picker(self):
        """Show picker for GPU compute device."""
        devices = ['cuda (NVIDIA GPU)', 'cpu (CPU fallback)', 'mps (Apple Silicon)']
        dev_cmds = ['cuda', 'cpu', 'mps']
        dlg = wx.SingleChoiceDialog(self, "Select compute device:",
                                     "GPU Device", devices)
        if dlg.ShowModal() == wx.ID_OK:
            device = dev_cmds[dlg.GetSelection()]
            self._exec(f'/gpu device {device}')
        dlg.Destroy()

    def _show_voice_algo_picker(self):
        """Show a picker for voice algorithms."""
        algos = ['stack', 'unison', 'wide']
        dlg = wx.SingleChoiceDialog(self, "Select voice algorithm:",
                                     "Voice Algorithm", algos)
        if dlg.ShowModal() == wx.ID_OK:
            algo = dlg.GetStringSelection()
            self._exec(f'/va {algo}')
        dlg.Destroy()

    def _show_key_scale_picker(self):
        """Show a two-step picker for key and scale."""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F',
                 'F#', 'G', 'G#', 'A', 'A#', 'B']
        scales = ['major', 'minor', 'dorian', 'phrygian', 'lydian',
                  'mixolydian', 'aeolian', 'locrian', 'pentatonic',
                  'blues', 'harmonic_minor', 'melodic_minor', 'chromatic']
        dlg1 = wx.SingleChoiceDialog(self, "Select root note:", "Key", notes)
        if dlg1.ShowModal() == wx.ID_OK:
            note = dlg1.GetStringSelection()
            dlg1.Destroy()
            dlg2 = wx.SingleChoiceDialog(self, "Select scale:", "Scale", scales)
            if dlg2.ShowModal() == wx.ID_OK:
                scale = dlg2.GetStringSelection()
                self._exec(f'/key {note} {scale}')
            dlg2.Destroy()
        else:
            dlg1.Destroy()

    def _show_sr_picker(self):
        """Show a placeholder for sample rate setting."""
        self._show_placeholder(
            "Sample Rate",
            "Sample rate is set at session creation.\n"
            "Use the console: set sample_rate <rate>")

    def _show_envelope_preset_picker(self):
        """Show presets for common envelope shapes."""
        presets = {
            'Pluck': '0.001 0.2 0.0 0.1',
            'Pad': '0.5 0.3 0.7 1.0',
            'Organ': '0.01 0.01 1.0 0.05',
            'String': '0.3 0.2 0.6 0.5',
            'Percussive': '0.001 0.1 0.0 0.05',
            'Swell': '1.0 0.0 1.0 0.5',
        }
        dlg = wx.SingleChoiceDialog(self, "Select envelope preset:",
                                     "Envelope Preset", list(presets.keys()))
        if dlg.ShowModal() == wx.ID_OK:
            name = dlg.GetStringSelection()
            self._exec(f'/env {presets[name]}')
        dlg.Destroy()

    def _show_conv_preset_picker(self):
        """Show a picker for convolution reverb presets."""
        presets = ['hall', 'room', 'plate', 'spring', 'chamber',
                   'cathedral', 'arena', 'tunnel', 'parking',
                   'bathroom', 'stairwell', 'studio', 'cave',
                   'forest', 'canyon', 'underwater', 'telephone']
        dlg = wx.SingleChoiceDialog(self, "Select convolution preset:",
                                     "Convolution Preset", presets)
        if dlg.ShowModal() == wx.ID_OK:
            p = dlg.GetStringSelection()
            self._exec(f'/conv preset {p}')
        dlg.Destroy()

    def _show_ir_file_picker(self):
        """Show a file dialog to load an IR wav file."""
        dlg = wx.FileDialog(self, "Load Impulse Response",
                            wildcard="WAV files (*.wav)|*.wav|All files (*.*)|*.*",
                            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self._exec(f'/conv load {path}')
        dlg.Destroy()

    def _show_ir_descriptor_picker(self):
        """Show a picker for IR transform descriptors."""
        descriptors = ['bigger', 'smaller', 'brighter', 'darker',
                       'warmer', 'colder', 'longer', 'shorter',
                       'wider', 'narrower', 'metallic', 'wooden',
                       'smooth', 'rough', 'airy']
        dlg = wx.SingleChoiceDialog(self, "Select IR descriptor:",
                                     "Transform Descriptor", descriptors)
        if dlg.ShowModal() == wx.ID_OK:
            desc = dlg.GetStringSelection()
            self._exec(f'/irtransform {desc}')
        dlg.Destroy()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def on_search(self, event):
        """Filter tree by expanding matching categories and collapsing others."""
        query = self.search_box.GetValue().lower().strip()
        if not query:
            self.tree.ExpandAll()
            return

        root = self.tree.GetRootItem()
        if not root.IsOk():
            return

        item, cookie = self.tree.GetFirstChild(root)
        while item.IsOk():
            cat_text = self.tree.GetItemText(item).lower()
            cat_match = query in cat_text
            child_match = self._search_children(item, query)

            if cat_match or child_match:
                self.tree.Expand(item)
            else:
                self.tree.Collapse(item)

            item, cookie = self.tree.GetNextChild(root, cookie)

    def _search_children(self, parent, query: str) -> bool:
        """Recursively check if any child matches the query."""
        child, cookie = self.tree.GetFirstChild(parent)
        while child.IsOk():
            if query in self.tree.GetItemText(child).lower():
                return True
            if self._search_children(child, query):
                return True
            child, cookie = self.tree.GetNextChild(parent, cookie)
        return False

    def on_refresh(self, event):
        """Re-read session state and rebuild the tree."""
        self.populate_tree()


class ActionPanel(wx.Panel):
    """Right panel - action selection and parameter editing."""

    def __init__(self, parent, executor: CommandExecutor, console_callback,
                 state_sync_callback: Optional[Callable] = None):
        super().__init__(parent, style=wx.TAB_TRAVERSAL)
        self.executor = executor
        self.console_callback = console_callback
        self.state_sync_callback = state_sync_callback
        self.current_category = 'engine'
        self.current_action: Optional[ActionDef] = None
        self.param_controls: Dict[str, wx.Control] = {}
        
        self.SetBackgroundColour(Theme.BG_PANEL)
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Header
        self.header = wx.StaticText(self, label="Select an object from the browser")
        self.header.SetForegroundColour(Theme.ACCENT)
        font = self.header.GetFont()
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.header.SetFont(font)
        self.sizer.Add(self.header, 0, wx.ALL, 10)
        
        # Action dropdown
        action_sizer = wx.BoxSizer(wx.HORIZONTAL)
        action_label = wx.StaticText(self, label="Action:")
        action_label.SetForegroundColour(Theme.FG_TEXT)
        self.action_choice = wx.Choice(self)
        self.action_choice.SetName("Action Selection")
        self.action_choice.SetBackgroundColour(Theme.BG_INPUT)
        self.action_choice.Bind(wx.EVT_CHOICE, self.on_action_select)
        
        action_sizer.Add(action_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        action_sizer.Add(self.action_choice, 1, wx.EXPAND)
        self.sizer.Add(action_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Description
        self.description = wx.StaticText(self, label="")
        self.description.SetForegroundColour(Theme.FG_DIM)
        self.sizer.Add(self.description, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        
        # Parameters panel (scrollable)
        self.params_panel = wx.ScrolledWindow(self)
        self.params_panel.SetName("Action Parameters Panel")
        self.params_panel.SetBackgroundColour(Theme.BG_PANEL)
        self.params_panel.SetScrollRate(0, 20)
        self.params_sizer = wx.BoxSizer(wx.VERTICAL)
        self.params_panel.SetSizer(self.params_sizer)
        self.sizer.Add(self.params_panel, 1, wx.EXPAND | wx.ALL, 5)
        
        # Command preview
        preview_label = wx.StaticText(self, label="Command:")
        preview_label.SetForegroundColour(Theme.FG_DIM)
        self.sizer.Add(preview_label, 0, wx.LEFT | wx.TOP, 10)
        
        self.command_preview = wx.TextCtrl(self, style=wx.TE_READONLY)
        self.command_preview.SetName("Generated Command Preview")
        self.command_preview.SetBackgroundColour(Theme.BG_INPUT)
        self.command_preview.SetForegroundColour(Theme.ACCENT)
        self.sizer.Add(self.command_preview, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        
        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.run_btn = wx.Button(self, label="Run (Ctrl+R)")
        self.run_btn.SetName("Run current action (Ctrl+R)")
        self.run_btn.SetBackgroundColour(Theme.ACCENT)
        self.run_btn.Bind(wx.EVT_BUTTON, self.on_run)

        self.copy_btn = wx.Button(self, label="Copy Command")
        self.copy_btn.SetName("Copy generated command to clipboard")
        self.copy_btn.Bind(wx.EVT_BUTTON, self.on_copy)

        self.reset_btn = wx.Button(self, label="Reset Params")
        self.reset_btn.SetName("Reset parameters to default values")
        self.reset_btn.Bind(wx.EVT_BUTTON, self.on_reset)
        
        btn_sizer.Add(self.run_btn, 1, wx.RIGHT, 5)
        btn_sizer.Add(self.copy_btn, 1, wx.RIGHT, 5)
        btn_sizer.Add(self.reset_btn, 1)
        
        self.sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        self.SetSizer(self.sizer)
        
        # Initialize with engine actions
        self.set_category('engine')
    
    def set_category(self, category: str):
        """Set the current category and update action list."""
        self.current_category = category
        self.header.SetLabel(f"{category.upper()} Actions")
        
        # Populate action dropdown
        self.action_choice.Clear()
        if category in ACTIONS:
            for action in ACTIONS[category]:
                self.action_choice.Append(action.label)
            if ACTIONS[category]:
                self.action_choice.SetSelection(0)
                self.on_action_select(None)
    
    def on_action_select(self, event):
        """Handle action selection."""
        idx = self.action_choice.GetSelection()
        if idx >= 0 and self.current_category in ACTIONS:
            actions = ACTIONS[self.current_category]
            if idx < len(actions):
                self.current_action = actions[idx]
                self.description.SetLabel(self.current_action.description)
                self.build_param_controls()
                self.update_command_preview()
    
    def build_param_controls(self):
        """Build parameter controls for current action."""
        # Clear existing
        self.params_sizer.Clear(True)
        self.param_controls.clear()
        
        if not self.current_action:
            return
        
        for param in self.current_action.params:
            row_sizer = wx.BoxSizer(wx.HORIZONTAL)
            
            # Label
            label = wx.StaticText(self.params_panel, label=f"{param.label}:")
            label.SetForegroundColour(Theme.FG_TEXT)
            label.SetMinSize((120, -1))
            row_sizer.Add(label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
            
            # Control based on type
            ctrl = None
            if param.param_type == 'int':
                ctrl = wx.SpinCtrl(self.params_panel, min=int(param.min_val or 0), 
                                   max=int(param.max_val or 100))
                ctrl.SetValue(int(param.default or 0))
                ctrl.Bind(wx.EVT_SPINCTRL, lambda e: self.update_command_preview())
            
            elif param.param_type == 'float':
                ctrl = wx.SpinCtrlDouble(self.params_panel, min=param.min_val or 0,
                                         max=param.max_val or 100, inc=0.1)
                ctrl.SetValue(float(param.default or 0))
                ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, lambda e: self.update_command_preview())
            
            elif param.param_type == 'bool':
                ctrl = wx.CheckBox(self.params_panel)
                ctrl.SetValue(bool(param.default))
                ctrl.Bind(wx.EVT_CHECKBOX, lambda e: self.update_command_preview())
            
            elif param.param_type == 'enum':
                ctrl = wx.Choice(self.params_panel, choices=param.choices)
                if param.default in param.choices:
                    ctrl.SetSelection(param.choices.index(param.default))
                else:
                    ctrl.SetSelection(0)
                ctrl.Bind(wx.EVT_CHOICE, lambda e: self.update_command_preview())
            
            elif param.param_type == 'file':
                file_sizer = wx.BoxSizer(wx.HORIZONTAL)
                ctrl = wx.TextCtrl(self.params_panel, value=str(param.default or ''))
                ctrl.Bind(wx.EVT_TEXT, lambda e: self.update_command_preview())
                browse_btn = wx.Button(self.params_panel, label="...", size=(30, -1))
                browse_btn.Bind(wx.EVT_BUTTON, lambda e, c=ctrl: self.on_browse_file(c))
                file_sizer.Add(ctrl, 1, wx.EXPAND | wx.RIGHT, 5)
                file_sizer.Add(browse_btn, 0)
                row_sizer.Add(file_sizer, 1, wx.EXPAND)
                self.param_controls[param.name] = ctrl
                self.params_sizer.Add(row_sizer, 0, wx.EXPAND | wx.ALL, 5)
                continue
            
            else:  # string
                ctrl = wx.TextCtrl(self.params_panel, value=str(param.default or ''))
                ctrl.Bind(wx.EVT_TEXT, lambda e: self.update_command_preview())
            
            if ctrl:
                ctrl.SetBackgroundColour(Theme.BG_INPUT)
                ctrl.SetForegroundColour(Theme.FG_TEXT)
                row_sizer.Add(ctrl, 1, wx.EXPAND)
                self.param_controls[param.name] = ctrl
            
            self.params_sizer.Add(row_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        self.params_panel.FitInside()
        self.Layout()
    
    def on_browse_file(self, text_ctrl):
        """Handle file browse button."""
        dlg = wx.FileDialog(self, "Select file", wildcard="All files (*.*)|*.*")
        if dlg.ShowModal() == wx.ID_OK:
            text_ctrl.SetValue(dlg.GetPath())
        dlg.Destroy()
    
    def get_param_values(self) -> Dict[str, Any]:
        """Get current parameter values."""
        values = {}
        if not self.current_action:
            return values
        
        for param in self.current_action.params:
            ctrl = self.param_controls.get(param.name)
            if ctrl:
                if param.param_type == 'int':
                    values[param.name] = ctrl.GetValue()
                elif param.param_type == 'float':
                    values[param.name] = ctrl.GetValue()
                elif param.param_type == 'bool':
                    values[param.name] = ctrl.GetValue()
                elif param.param_type == 'enum':
                    idx = ctrl.GetSelection()
                    if 0 <= idx < len(param.choices):
                        values[param.name] = param.choices[idx]
                    else:
                        values[param.name] = param.default
                else:
                    values[param.name] = ctrl.GetValue()
        
        return values
    
    def build_command(self) -> str:
        """Build the command string from current action and params."""
        if not self.current_action:
            return ""
        
        values = self.get_param_values()
        try:
            return self.current_action.command_template.format(**values)
        except KeyError as e:
            self.console_callback(
                f"Error: missing parameter {e} in command template\n",
                'error')
            return ""
    
    def update_command_preview(self):
        """Update the command preview text."""
        cmd = self.build_command()
        self.command_preview.SetValue(cmd)
    
    def on_run(self, event):
        """Execute the current command and sync GUI state."""
        cmd = self.build_command()
        if cmd:
            self.console_callback(f">>> {cmd}\n", 'command')
            stdout, stderr, success = self.executor.execute(cmd)
            if stdout:
                self.console_callback(stdout, 'stdout')
            if stderr:
                self.console_callback(stderr, 'stderr')
            status = "OK" if success else "ERROR"
            color = 'success' if success else 'error'
            self.console_callback(f"[{status}]\n\n", color)

            # Tell the frame to refresh status bar + browser
            if self.state_sync_callback:
                self.state_sync_callback()
    
    def on_copy(self, event):
        """Copy command to clipboard."""
        cmd = self.build_command()
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(cmd))
            wx.TheClipboard.Close()
            self.console_callback(f"Copied: {cmd}\n", 'info')
    
    def on_reset(self, event):
        """Reset parameters to defaults."""
        if self.current_action:
            for param in self.current_action.params:
                ctrl = self.param_controls.get(param.name)
                if ctrl:
                    if param.param_type in ('int', 'float'):
                        ctrl.SetValue(param.default or 0)
                    elif param.param_type == 'bool':
                        ctrl.SetValue(bool(param.default))
                    elif param.param_type == 'enum':
                        if param.default in param.choices:
                            ctrl.SetSelection(param.choices.index(param.default))
                    else:
                        ctrl.SetValue(str(param.default or ''))
            self.update_command_preview()


class InspectorPanel(wx.Panel):
    """Detail inspector panel — shows full properties of the selected object.

    When a track, buffer, deck, operator, filter slot, or other object
    is selected in the ObjectBrowser, this panel displays all its
    properties in a readable, keyboard-navigable layout.
    """

    def __init__(self, parent, executor: CommandExecutor,
                 console_callback=None, state_sync_callback=None):
        super().__init__(parent, style=wx.TAB_TRAVERSAL)
        self.executor = executor
        self.console_cb = console_callback
        self.state_sync_cb = state_sync_callback
        self.current_data = None

        self.SetBackgroundColour(Theme.BG_PANEL)

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        # Title
        self.title = wx.StaticText(self, label="Inspector")
        self.title.SetForegroundColour(Theme.ACCENT)
        font = self.title.GetFont()
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.title.SetFont(font)
        self.sizer.Add(self.title, 0, wx.ALL, 10)

        # Subtitle (object type)
        self.subtitle = wx.StaticText(self, label="Select an object to inspect")
        self.subtitle.SetForegroundColour(Theme.FG_DIM)
        self.sizer.Add(self.subtitle, 0, wx.LEFT | wx.BOTTOM, 10)

        # Properties list (ListCtrl for screen-reader navigation)
        self.props_list = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL
                                      | wx.LC_NO_HEADER)
        self.props_list.SetName("Object Properties")
        self.props_list.SetBackgroundColour(Theme.BG_PANEL)
        self.props_list.SetForegroundColour(Theme.FG_TEXT)
        self.props_list.InsertColumn(0, "Property", width=130)
        self.props_list.InsertColumn(1, "Value", width=280)
        self.sizer.Add(self.props_list, 1, wx.EXPAND | wx.ALL, 5)

        # Quick action buttons (contextual)
        self.action_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self.action_sizer, 0, wx.EXPAND | wx.ALL, 10)

        self.SetSizer(self.sizer)

    def inspect(self, data: dict):
        """Display detail view for the given object data dict."""
        self.current_data = data
        obj_type = data.get('type', '')

        # Clear previous
        self.props_list.DeleteAllItems()
        self.action_sizer.Clear(True)

        if obj_type == 'track':
            self._inspect_track(data)
        elif obj_type == 'buffer':
            self._inspect_buffer(data)
        elif obj_type == 'working_buffer':
            self._inspect_working_buffer(data)
        elif obj_type == 'deck':
            self._inspect_deck(data)
        elif obj_type == 'operator':
            self._inspect_operator(data)
        elif obj_type == 'filter_slot':
            self._inspect_filter_slot(data)
        elif obj_type == 'effect':
            self._inspect_effect(data)
        elif obj_type == 'sydef':
            self._inspect_sydef(data)
        elif obj_type == 'chain':
            self._inspect_chain(data)
        elif obj_type == 'deck_section':
            self._inspect_deck_section(data)
        elif obj_type == 'routing':
            self._inspect_routing(data)
        elif obj_type == 'wavetable':
            self._inspect_wavetable(data)
        elif obj_type == 'compound':
            self._inspect_compound(data)
        elif obj_type == 'conv_prop':
            self._inspect_convolution(data)
        elif obj_type == 'ir_bank_entry':
            self._inspect_ir_bank(data)
        elif obj_type == 'lfo_shape':
            self._inspect_lfo_shape(data)
        elif obj_type == 'imp_env_shape':
            self._inspect_imp_env_shape(data)
        elif obj_type == 'category':
            self._inspect_category(data)
        elif obj_type in ('gen_genre', 'gen_layer', 'gen_xform_preset',
                          'gen_theory_item', 'gen_content_item', 'gen_section'):
            self._inspect_generative(data)
        else:
            self.title.SetLabel("Inspector")
            self.subtitle.SetLabel(f"Object type: {obj_type}")

        self.Layout()

    def _add_prop(self, label: str, value: str):
        """Add a property row to the inspector (ListCtrl for screen readers)."""
        idx = self.props_list.GetItemCount()
        self.props_list.InsertItem(idx, label)
        self.props_list.SetItem(idx, 1, str(value))

    def _add_separator(self, label: str = ''):
        """Add a section separator row to the inspector."""
        idx = self.props_list.GetItemCount()
        if label:
            self.props_list.InsertItem(idx, f"--- {label} ---")
        else:
            self.props_list.InsertItem(idx, "---")
        self.props_list.SetItem(idx, 1, "")

    def _add_action_btn(self, label: str, command: str):
        """Add a quick-action button with accessible name."""
        btn = wx.Button(self, label=label)
        btn.SetName(f"{label} ({command})")
        btn.Bind(wx.EVT_BUTTON,
                 lambda e, cmd=command: self._exec_action(cmd))
        self.action_sizer.Add(btn, 0, wx.RIGHT, 5)

    def _exec_action(self, command: str):
        """Execute a command and sync state."""
        if self.console_cb:
            self.console_cb(f">>> {command}\n", 'command')
        stdout, stderr, ok = self.executor.execute(command)
        if self.console_cb:
            if stdout:
                self.console_cb(stdout, 'stdout')
            if stderr:
                self.console_cb(stderr, 'stderr')
            self.console_cb(
                f"[{'OK' if ok else 'ERROR'}]\n\n",
                'success' if ok else 'error')
        if self.state_sync_cb:
            self.state_sync_cb()

    def _inspect_track(self, data):
        details = data.get('details', {})
        idx = data.get('index', 0)
        name = details.get('name', f'track_{idx+1}')
        self.title.SetLabel(f"Track {idx+1}: {name}")
        self.subtitle.SetLabel("Track Properties")

        self._add_prop("Name:", name)
        self._add_prop("Index:", str(idx))
        dur = details.get('duration', 0)
        self._add_prop("Duration:", f"{dur:.2f}s")
        ch = details.get('channels', 0)
        self._add_prop("Channels:", 'Stereo' if ch == 2 else 'Mono' if ch == 1 else str(ch))
        self._add_prop("Peak:", f"{details.get('peak_db', -100):.1f} dB")
        self._add_prop("Gain:", f"{details.get('gain', 1.0):.2f}")
        self._add_prop("Pan:", f"{details.get('pan', 0.0):.2f}")
        self._add_prop("Mute:", "Yes" if details.get('mute') else "No")
        self._add_prop("Solo:", "Yes" if details.get('solo') else "No")
        fx = details.get('fx_chain', [])
        self._add_prop("FX Chain:", ', '.join(fx) if fx else "(none)")
        wp = details.get('write_pos_sec', 0)
        self._add_prop("Write Pos:", f"{wp:.2f}s")

        self._add_action_btn("Play", f"/pt {idx+1}")
        self._add_action_btn("Solo", f"/tsolo {idx+1}")
        self._add_action_btn("Mute", f"/tmute {idx+1}")
        self._add_action_btn("Bounce", f"/btw {idx+1}")

    def _inspect_buffer(self, data):
        details = data.get('details', {})
        idx = data.get('index', 1)
        self.title.SetLabel(f"Buffer {idx}")
        self.subtitle.SetLabel("Buffer Properties")

        if details.get('empty'):
            self._add_prop("Status:", "Empty")
        else:
            self._add_prop("Duration:", f"{details.get('duration', 0):.2f}s")
            ch = details.get('channels', 0)
            self._add_prop("Channels:", 'Stereo' if ch == 2 else 'Mono')
            self._add_prop("Peak:", f"{details.get('peak_db', -100):.1f} dB")
            self._add_prop("Samples:", str(details.get('samples', 0)))

        self._add_action_btn("Play", f"/pb {idx}")
        self._add_action_btn("To Working", f"/w {idx}")
        self._add_action_btn("Clear", f"/clr {idx}")

    def _inspect_working_buffer(self, data):
        details = data.get('details', {})
        self.title.SetLabel("Working Buffer")
        self.subtitle.SetLabel("Working Buffer Properties")

        if details.get('empty'):
            self._add_prop("Status:", "Empty")
        else:
            self._add_prop("Duration:", f"{details.get('duration', 0):.2f}s")
            ch = details.get('channels', 0)
            self._add_prop("Channels:", 'Stereo' if ch == 2 else 'Mono')
            self._add_prop("Peak:", f"{details.get('peak_db', -100):.1f} dB")
            self._add_prop("Source:", details.get('source', 'unknown'))

        self._add_action_btn("Play", "/p")
        self._add_action_btn("Clear", "/wbc")

    def _inspect_deck(self, data):
        details = data.get('details', {})
        idx = data.get('index', 1)
        self.title.SetLabel(f"Deck {idx}")
        self.subtitle.SetLabel("Deck Properties")

        loaded = details.get('loaded', False)
        self._add_prop("Status:", "Loaded" if loaded else "Empty")
        if loaded:
            self._add_prop("Duration:", f"{details.get('duration', 0):.2f}s")
            fx = details.get('fx_chain', [])
            self._add_prop("FX Chain:", ', '.join(fx) if fx else "(none)")

        self._add_action_btn("Play", f"/pd {idx}")
        self._add_action_btn("Select", f"/deck {idx}")

    def _inspect_operator(self, data):
        idx = data.get('index', 0)
        details = self.executor.get_operator_details(idx)
        wave = details.get('wave', 'sine')
        self.title.SetLabel(f"Operator {idx}: {wave}")
        self.subtitle.SetLabel("Operator Properties")

        self._add_prop("Waveform:", wave)
        self._add_prop("Frequency:", f"{details.get('freq', 440):.1f} Hz")
        self._add_prop("Amplitude:", f"{details.get('amp', 0.8):.2f}")
        self._add_prop("Current:", "Yes" if details.get('is_current') else "No")
        env = details.get('envelope')
        if env:
            self._add_prop("Envelope:",
                f"A{env['attack']:.3f} D{env['decay']:.3f} "
                f"S{env['sustain']:.3f} R{env['release']:.3f}")

        # Wave-specific parameters
        wp = details.get('wave_params', {})
        if wp:
            self._add_prop("", "--- Wave Parameters ---")
            for k, v in wp.items():
                if isinstance(v, float):
                    self._add_prop(f"{k}:", f"{v:.4f}")
                else:
                    self._add_prop(f"{k}:", str(v))

        self._add_action_btn("Select", f"/op {idx}")
        self._add_action_btn("Generate", f"/tone 440 1")

    def _inspect_filter_slot(self, data):
        slot = data.get('slot', 0)
        details = self.executor.get_filter_slot_details(slot)
        self.title.SetLabel(f"Filter Slot {slot}")
        self.subtitle.SetLabel("Filter Properties")

        self._add_prop("Type:", details.get('type_name', 'lowpass'))
        self._add_prop("Cutoff:", f"{details.get('cutoff', 4500):.0f} Hz")
        self._add_prop("Resonance:", f"{details.get('resonance', 50):.0f}")
        self._add_prop("Enabled:", "Yes" if details.get('enabled') else "No")
        self._add_prop("Selected:", "Yes" if details.get('is_selected') else "No")

        # Filter envelope (from engine state)
        state = self.executor.get_engine_state()
        if state:
            self._add_prop("", "--- Filter Envelope ---")
            self._add_prop("F.Attack:", f"{state.get('filter_attack', 0.01):.3f}s")
            self._add_prop("F.Decay:", f"{state.get('filter_decay', 0.1):.3f}s")
            self._add_prop("F.Sustain:", f"{state.get('filter_sustain', 0.8):.3f}")
            self._add_prop("F.Release:", f"{state.get('filter_release', 0.1):.3f}s")

        self._add_action_btn("Select", f"/fs {slot}")
        self._add_action_btn("Toggle", "/fen toggle")

    def _inspect_effect(self, data):
        idx = data.get('index', 0)
        fx_name = data.get('name', f'effect_{idx}')
        fx_params = data.get('params', {})

        self.title.SetLabel(f"FX #{idx}: {fx_name}")
        self.subtitle.SetLabel("Effect in Chain")

        # Core info
        self._add_prop("Effect:", fx_name)
        self._add_prop("Position:", str(idx))
        amt = fx_params.get('amount', 50.0)
        self._add_prop("Amount:", f"{amt:.0f}%")

        # Show all DSP parameters
        extra = {k: v for k, v in fx_params.items() if k != 'amount'}
        if extra:
            self._add_separator("DSP Parameters")
            for k, v in sorted(extra.items()):
                self._add_prop(f"  {k}:", f"{v:.2f}")

        # Indicate what DSP params are available (even if not yet set)
        known = ObjectBrowser._FX_DSP_PARAMS.get(fx_name, [])
        unset = [p for p in known if p not in fx_params and p != 'amount']
        if unset:
            self._add_separator("Available Parameters")
            self._add_prop("  (editable):", ', '.join(unset))

        # Actions
        self._add_separator("Actions")
        self._add_action_btn("Set Amount", f"/fxp {idx} 50")
        self._add_action_btn("Select", f"/fxs {idx}")
        self._add_action_btn("Bypass", "/bypass")
        self._add_action_btn("Remove", f"/fxr {idx}")
        self._add_action_btn("Clear All", "/fx clear")

    def _inspect_sydef(self, data):
        name = data.get('name', '')
        self.title.SetLabel(f"SyDef: {name}")
        self.subtitle.SetLabel("Synth Definition Preset")

        self._add_prop("Name:", name)

        # Look up full SyDef details from session
        sydef = None
        s = self.executor.session
        if s and hasattr(s, 'sydefs') and name in s.sydefs:
            sydef = s.sydefs[name]

        if sydef:
            # Description
            desc = getattr(sydef, 'description', '') or ''
            if desc:
                self._add_prop("Description:", desc)

            # Parameters
            params = getattr(sydef, 'params', [])
            if params:
                self._add_separator("Parameters")
                for p in params:
                    p_name = p.name if hasattr(p, 'name') else str(p)
                    p_default = p.default if hasattr(p, 'default') else '?'
                    self._add_prop(f"  ${p_name}:", str(p_default))

            # Commands
            commands = getattr(sydef, 'commands', [])
            if commands:
                self._add_separator(f"Commands ({len(commands)})")
                for i, cmd in enumerate(commands[:12]):
                    self._add_prop(f"  {i+1}:", cmd)
                if len(commands) > 12:
                    self._add_prop("  ...:", f"+{len(commands) - 12} more")

            # Factory preset indicator
            factory_names = getattr(s, '_factory_sydef_names', set())
            if name in factory_names:
                self._add_prop("Origin:", "Factory Preset")
            else:
                self._add_prop("Origin:", "User-Defined")
        else:
            self._add_prop("Status:", "Definition not found in session")

        self._add_separator("Actions")
        self._add_action_btn("Use (defaults)", f"/use {name}")
        self._add_action_btn("Show Contents", f"/sydef show {name}")
        self._add_action_btn("Copy...", f"/sydef copy {name} {name}_copy")
        self._add_action_btn("Delete", f"/sydef del {name}")

    def _inspect_chain(self, data):
        name = data.get('name', '')
        self.title.SetLabel(f"Chain: {name}")
        self.subtitle.SetLabel("Effect Chain")

        self._add_prop("Name:", name)
        self._add_action_btn("Apply", f"/chain load {name}")

    def _inspect_deck_section(self, data):
        deck = data.get('deck', 0)
        start = data.get('start', 0)
        end = data.get('end', 0)
        self.title.SetLabel(f"Deck {deck} Section")
        self.subtitle.SetLabel("Accessibility Section Marker")

        self._add_prop("Deck:", str(deck))
        self._add_prop("Start:", f"{start:.1f}s")
        self._add_prop("End:", f"{end:.1f}s")
        self._add_prop("Duration:", f"{end - start:.1f}s")

    def _inspect_routing(self, data):
        idx = data.get('index', 0)
        self.title.SetLabel(f"Routing #{idx}")
        self.subtitle.SetLabel("Modulation Routing")

        self._add_prop("Type:", data.get('algo_type', ''))
        self._add_prop("Source:", f"Op {data.get('src', 0)}")
        self._add_prop("Target:", f"Op {data.get('tgt', 0)}")
        self._add_prop("Amount:", f"{data.get('amt', 0):.3f}")

        self._add_action_btn("Clear All", "/clearalg")

    def _inspect_wavetable(self, data):
        name = data.get('name', '')
        self.title.SetLabel(f"Wavetable: {name}")
        self.subtitle.SetLabel("Loaded Wavetable")

        self._add_prop("Name:", name)
        s = self.executor.session
        if s and hasattr(s, 'engine') and hasattr(s.engine, 'wavetables'):
            wt = s.engine.wavetables.get(name)
            if wt is not None and hasattr(wt, '__len__'):
                self._add_prop("Frames:", str(len(wt)))

        self._add_action_btn("Use", f"/wt use {name}")
        self._add_action_btn("Delete", f"/wt del {name}")

    def _inspect_compound(self, data):
        name = data.get('name', '')
        self.title.SetLabel(f"Compound: {name}")
        self.subtitle.SetLabel("Compound Wave")

        self._add_prop("Name:", name)
        s = self.executor.session
        if s and hasattr(s, 'engine') and hasattr(s.engine, 'compound_waves'):
            cw = s.engine.compound_waves.get(name)
            if isinstance(cw, dict):
                layers = cw.get('layers', [])
                self._add_prop("Layers:", str(len(layers)))
                for i, layer in enumerate(layers):
                    wave_l = layer.get('wave', 'sine') if isinstance(layer, dict) else str(layer)
                    self._add_prop(f"  Layer {i}:", wave_l)

        self._add_action_btn("Use", f"/compound use {name}")
        self._add_action_btn("Delete", f"/compound del {name}")

    def _inspect_convolution(self, data):
        self.title.SetLabel("Convolution Reverb")
        self.subtitle.SetLabel("Convolution Engine State")
        try:
            from mdma_rebuild.dsp.convolution import get_convolution_engine
            ce = get_convolution_engine()
            info = ce.get_info()
            if info['ir_loaded']:
                self._add_prop("IR:", info['ir_name'])
                self._add_prop("Duration:", f"{info['ir_duration']:.2f}s")
            self._add_prop("Wet/Dry:", f"{info['wet']:.0f}/{info['dry']:.0f}")
            self._add_prop("Pre-Delay:", f"{info['pre_delay_ms']:.0f}ms")
            self._add_prop("Decay:", f"{info['decay']:.2f}x")
            self._add_prop("Width:", f"{info['stereo_width']:.0f}")
            self._add_prop("Early:", f"{info['early_level']:.0f}")
            self._add_prop("Late:", f"{info['late_level']:.0f}")
            self._add_prop("Split:", f"{info['early_late_split_ms']:.0f}ms")
        except Exception:
            self._add_prop("Status:", "Engine not available")

    def _inspect_ir_bank(self, data):
        name = data.get('name', '')
        self.title.SetLabel(f"IR: {name}")
        self.subtitle.SetLabel("Impulse Response in Bank")
        self._add_prop("Name:", name)
        self._add_action_btn("Use", f"/conv use {name}")

    def _inspect_lfo_shape(self, data):
        name = data.get('name', '')
        self.title.SetLabel(f"LFO Shape: {name}")
        self.subtitle.SetLabel("Impulse-derived LFO Waveshape")
        self._add_prop("Name:", name)
        s = self.executor.session
        if s and hasattr(s, '_lfo_waveshapes') and name in s._lfo_waveshapes:
            shape = s._lfo_waveshapes[name]
            self._add_prop("Samples:", str(len(shape)))
            self._add_prop("Peak:", f"{np.max(np.abs(shape)):.3f}")
        self._add_action_btn("Apply Op0", f"/impulselfo apply 0 4 12")

    def _inspect_imp_env_shape(self, data):
        name = data.get('name', '')
        self.title.SetLabel(f"Envelope: {name}")
        self.subtitle.SetLabel("Impulse-derived Amplitude Envelope")
        self._add_prop("Name:", name)
        s = self.executor.session
        if s and hasattr(s, '_imp_envelopes') and name in s._imp_envelopes:
            env = s._imp_envelopes[name]
            self._add_prop("Samples:", str(len(env)))
            sr = getattr(s, 'sample_rate', 48000)
            self._add_prop("Duration:", f"{len(env)/sr:.3f}s")
        self._add_action_btn("Apply Buffer", f"/impenv apply 1.0")

    def _inspect_generative(self, data):
        """Display detail view for generative items."""
        obj_type = data.get('type', '')

        if obj_type == 'gen_genre':
            genre = data.get('genre', 'house')
            bpm = data.get('bpm', 128)
            self.title.SetLabel(f"{genre.title()} Beat")
            self.subtitle.SetLabel("Genre Template")
            self._add_prop("Genre:", genre.title())
            self._add_prop("Default BPM:", str(bpm))
            try:
                from mdma_rebuild.dsp import beat_gen
                tmpl = beat_gen.GENRE_TEMPLATES.get(genre)
                if tmpl:
                    self._add_prop("Steps:", str(tmpl.steps))
                    self._add_prop("Swing:", f"{tmpl.swing:.0f}%")
                    n_instruments = len(tmpl.hits)
                    self._add_prop("Instruments:", str(n_instruments))
            except ImportError:
                pass
            self._add_separator("Actions")
            self._add_action_btn("Generate 4 Bars", f"/beat {genre} 4")
            self._add_action_btn("Generate 8 Bars", f"/beat {genre} 8")
            self._add_action_btn("Generate Fill", "/beat fill buildup")

        elif obj_type == 'gen_layer':
            layer = data.get('layer', 'drums')
            self.title.SetLabel(f"{layer.title()} Layer")
            self.subtitle.SetLabel("Loop Layer")
            self._add_prop("Layer:", layer.title())
            layer_desc = {
                'drums': 'Rhythmic drum pattern from genre template',
                'bass': 'Bass line following chord progression',
                'chords': 'Chord pad with voice-led progressions',
                'melody': 'Melodic line from scale with contour',
                'full': 'All layers combined (drums + bass + chords + melody)',
            }
            self._add_prop("Description:", layer_desc.get(layer, 'Audio layer'))
            self._add_separator("Actions")
            self._add_action_btn("Generate Loop", f"/loop house {layer}")

        elif obj_type == 'gen_xform_preset':
            preset = data.get('preset', 'reverse')
            self.title.SetLabel(f"Transform: {preset}")
            self.subtitle.SetLabel("Audio Transform Preset")
            try:
                from mdma_rebuild.dsp import transforms as tf
                steps = tf.AUDIO_TRANSFORM_PRESETS.get(preset, [])
                self._add_prop("Steps:", str(len(steps)))
                for i, (name, params) in enumerate(steps):
                    p_str = ', '.join(f'{k}={v}' for k, v in params.items())
                    self._add_prop(f"  Step {i+1}:", f"{name}({p_str})" if p_str else name)
            except ImportError:
                pass
            self._add_separator("Actions")
            self._add_action_btn("Apply Transform", f"/xform preset {preset}")

        elif obj_type == 'gen_theory_item':
            query = data.get('query', 'scales')
            self.title.SetLabel(query.title())
            self.subtitle.SetLabel("Music Theory")
            try:
                from mdma_rebuild.dsp import music_theory as mt
                if query == 'scales':
                    self._add_prop("Available:", str(len(mt.SCALES)))
                    for name in sorted(mt.SCALES.keys())[:12]:
                        intervals = mt.SCALES[name]
                        self._add_prop(f"  {name}:", str(intervals))
                elif query == 'chords':
                    self._add_prop("Available:", str(len(mt.CHORDS)))
                    for name in sorted(mt.CHORDS.keys())[:12]:
                        intervals = mt.CHORDS[name]
                        self._add_prop(f"  {name}:", str(intervals))
                elif query == 'progressions':
                    self._add_prop("Available:", str(len(mt.PROGRESSIONS)))
                    for name in sorted(mt.PROGRESSIONS.keys())[:8]:
                        self._add_prop(f"  {name}:", str(mt.PROGRESSIONS[name]))
            except ImportError:
                self._add_prop("Status:", "Module not available")
            self._add_separator("Actions")
            self._add_action_btn("Show All", f"/theory {query}")

        elif obj_type == 'gen_content_item':
            ctype = data.get('content_type', 'melody')
            self.title.SetLabel(ctype.replace('_', ' ').title())
            self.subtitle.SetLabel("Content Generator")
            desc = {
                'melody': 'Generate melodic sequences from scale with contour shaping',
                'chord_prog': 'Render chord progressions with voice-leading',
                'bassline': 'Genre-aware bassline generation',
                'arpeggio': 'Arpeggiate chords in patterns',
                'drone': 'Ambient drone with detuned oscillators',
            }
            self._add_prop("Description:", desc.get(ctype, 'Generate audio content'))
            self._add_separator("Actions")
            self._add_action_btn("Generate", f"/gen2 {ctype}")

        elif obj_type == 'gen_tta_item':
            gen = data.get('generator', 'melody')
            self.title.SetLabel(gen.replace('_', ' ').title())
            self.subtitle.SetLabel("Text to Audio Generator")
            desc = {
                'melody': 'Describe a scale and length to generate a melodic phrase',
                'chord_prog': 'Pick a progression style to render voice-led chords',
                'bassline': 'Generate a rhythmic bassline from genre and scale',
                'beat': 'Generate a full drum pattern from a genre template',
                'loop': 'Generate a multi-layer loop (drums, bass, chords, melody)',
            }
            self._add_prop("Description:", desc.get(gen, 'Generate audio from parameters'))
            self._add_separator("Actions")
            if gen == 'beat':
                self._add_action_btn("Generate House 4 Bars", "/beat house 4")
                self._add_action_btn("Generate Techno 4 Bars", "/beat techno 4")
            elif gen == 'loop':
                self._add_action_btn("Generate House Loop", "/loop house full")
                self._add_action_btn("Generate Lo-fi Loop", "/loop lofi full")
            else:
                self._add_action_btn("Generate", f"/gen2 {gen}")

        elif obj_type == 'gen_breed_item':
            method = data.get('method', 'breed')
            self.title.SetLabel(method.replace('_', ' ').title())
            self.subtitle.SetLabel("Genetic Breeding")
            desc = {
                'breed': 'Full breed: 5 crossover methods + mutation, produces multiple children',
                'temporal': 'Split parents at a time point, swap halves with crossfade',
                'spectral': 'Frequency-domain crossover — take lows from A, highs from B',
                'blend': 'Linear amplitude blend between parents (morph)',
                'morphological': 'Separate envelope and spectral content, blend independently',
                'evolve': 'Multi-generation tournament evolution with elitism',
                'mutate_noise': 'Add random noise perturbation to audio',
                'mutate_pitch': 'Resampling-based pitch shift mutation',
                'mutate_time_stretch': 'Segment-based time stretch mutation',
                'mutate_spectral_smear': 'Blur spectral content across frequency bins',
                'mutate_reverse_segment': 'Reverse random chunks within the audio',
            }
            self._add_prop("Description:", desc.get(method, 'Genetic audio operation'))
            self._add_separator("Actions")
            if method == 'breed':
                self._add_action_btn("Breed Buffers 1+2", "/breed 1 2 4")
            elif method == 'evolve':
                self._add_action_btn("Evolve 10 Generations", "/evolve 10 8")
            elif method.startswith('mutate_'):
                mut = method.replace('mutate_', '')
                self._add_action_btn(f"Apply (30%)", f"/mutate {mut} 30")
                self._add_action_btn(f"Apply (60%)", f"/mutate {mut} 60")
            else:
                self._add_action_btn(f"Crossover Buf 1+2", f"/crossover 1 2 {method}")

        elif obj_type == 'phase_t_cmd':
            cmd = data.get('command', '')
            label = cmd.lstrip('/').replace('_', ' ').title()
            self.title.SetLabel(label)
            self.subtitle.SetLabel("Song-Ready Tool (Phase T)")
            cmd_desc = {
                '/undo': 'Undo the last destructive operation on working buffer or track',
                '/redo': 'Re-apply previously undone operation',
                '/snapshot save': 'Save all session parameters (BPM, FX, gains) as a snapshot',
                '/snapshot restore': 'Restore session parameters from a saved snapshot',
                '/section add': 'Define a named section by bar range for song arrangement',
                '/section list': 'Show all defined song sections with bar positions',
                '/section copy': 'Copy a section to a new bar position across all tracks',
                '/commit': 'Write working buffer to current track, advance write position',
                '/pos': 'Show or set the track write cursor position',
                '/export stems': 'Export each track as a separate WAV file',
                '/export track': 'Export a single track to a WAV file',
                '/export section': 'Render and export a named song section',
                '/master_gain': 'Set the master output gain in dB before limiting',
                '/crossover': 'Genetically crossover two buffers using breeding algorithms',
                '/dup': 'Duplicate a buffer to the next empty slot',
                '/swap': 'Swap the contents of two buffers',
                '/metronome': 'Generate a click track in the working buffer',
            }
            self._add_prop("Description:", cmd_desc.get(cmd, 'Phase T command'))
            self._add_prop("Command:", cmd)
            self._add_separator("Actions")
            self._add_action_btn("Execute", cmd)

        elif obj_type == 'gen_section':
            section = data.get('section', '')
            titles = {'beat': 'Beat Generation', 'loop': 'Loop Generation',
                      'xform': 'Transforms', 'theory': 'Music Theory',
                      'gen2': 'Content Generation',
                      'text_to_audio': 'Text to Audio',
                      'breeding': 'Genetic Breeding',
                      'phase_t_structure': 'Song Structure',
                      'phase_t_export': 'Export & Render',
                      'phase_t_tools': 'Buffer Tools'}
            self.title.SetLabel(titles.get(section, section.title()))
            self.subtitle.SetLabel("Generative Section")
            if section == 'breeding':
                self._add_prop("Engine:", "ai/breeding.py")
                self._add_prop("Crossover Methods:", "temporal, spectral, blend, morphological, multi-point")
                self._add_prop("Mutation Types:", "noise, pitch, time stretch, freq shift, envelope, reverse, smear")
                self._add_prop("Evolution:", "Tournament selection with elitism")
            elif section == 'text_to_audio':
                self._add_prop("Generators:", "melody, chords, bassline, arpeggio, drone, beat, loop")
                self._add_prop("Scales:", "21 scales available")
                self._add_prop("Genres:", "11 beat/loop genre templates")
            elif section == 'phase_t_structure':
                self._add_prop("Commands:", "/section, /pchain, /commit, /pos, /seek")
                self._add_prop("Sections:", "Named bar ranges for arrangement")
                self._add_prop("Chaining:", "Chain buffer patterns into track sequences")
            elif section == 'phase_t_export':
                self._add_prop("Commands:", "/export, /master_gain")
                self._add_prop("Stems:", "Export each track as separate WAV")
                self._add_prop("Sections:", "Render named sections to file")
            elif section == 'phase_t_tools':
                self._add_prop("Commands:", "/crossover, /dup, /swap, /metronome")
                self._add_prop("Crossover:", "5 genetic crossover methods")
                self._add_prop("Metronome:", "Click track generator")

    def _inspect_category(self, data):
        cat_id = data.get('id', '')
        names = {'engine': 'Engine', 'synth': 'Synthesizer',
                 'voice': 'Voice', 'filter': 'Filter',
                 'filter_envelope': 'Filter Envelope',
                 'envelope': 'Envelope', 'hq': 'HQ Mode',
                 'key': 'Key / Scale', 'modulation': 'Modulation',
                 'wavetable': 'Wavetables', 'compound': 'Compound Waves',
                 'convolution': 'Convolution Reverb',
                 'impulse_lfo': 'Impulse LFOs',
                 'impulse_env': 'Impulse Envelopes',
                 'ir_enhance': 'IR Enhancement',
                 'ir_transform': 'IR Transform',
                 'ir_granular': 'IR Granular',
                 'tracks': 'Tracks', 'buffers': 'Buffers',
                 'decks': 'Decks', 'fx': 'Effects',
                 'preset': 'Presets', 'bank': 'Banks',
                 'generative': 'Generative',
                 'text_to_audio': 'Text to Audio',
                 'breeding': 'Genetic Breeding',
                 'phase_t': 'Song Tools (Phase T)'}
        self.title.SetLabel(names.get(cat_id, cat_id.title()))
        self.subtitle.SetLabel("Category")

        # Show summary for new categories
        state = self.executor.get_engine_state()
        if not state:
            return

        if cat_id == 'voice':
            vc = state.get('voice_count', 1)
            va = state.get('voice_algorithm', 0)
            va_name = {0: 'stack', 1: 'unison', 2: 'wide'}.get(va, str(va))
            self._add_prop("Voices:", str(vc))
            self._add_prop("Algorithm:", va_name)
            self._add_prop("Detune:", f"{state.get('detune', 0):.2f} Hz")
            self._add_prop("Stereo:", f"{state.get('stereo_spread', 0):.0f}")
            self._add_prop("Phase:", f"{state.get('phase_spread', 0):.2f} rad")
            self._add_prop("Random:", f"{state.get('rand', 0):.0f}")
            self._add_prop("V-Mod:", f"{state.get('v_mod', 0):.0f}")
        elif cat_id == 'hq':
            self._add_prop("HQ Mode:", "ON" if state.get('hq_mode') else "OFF")
            self._add_prop("Oscillators:", "ON" if state.get('hq_oscillators') else "OFF")
            self._add_prop("DC Removal:", "ON" if state.get('hq_dc_removal') else "OFF")
            self._add_prop("Saturation:", str(state.get('hq_saturation', 0)))
            self._add_prop("Limiter:", str(state.get('hq_limiter', 0)))
            self._add_action_btn("Toggle HQ", "/hq on")
        elif cat_id == 'fx':
            fx_list = state.get('effects', [])
            n = len(fx_list)
            self._add_prop("Active Effects:", str(n))
            s = self.executor.session
            if s:
                bypassed = getattr(s, 'effects_bypassed', False)
                self._add_prop("Bypass:", "ON" if bypassed else "OFF")
                sel = getattr(s, 'selected_effect', -1)
                if sel >= 0 and sel < n:
                    sel_name = fx_list[sel] if isinstance(fx_list[sel], str) else fx_list[sel][0]
                    self._add_prop("Selected:", f"#{sel} — {sel_name}")
                # List all effects in chain
                params_list = getattr(s, 'effect_params', [])
                if n > 0:
                    self._add_separator("Effect Chain")
                    for i, fx in enumerate(fx_list):
                        fx_name = fx if isinstance(fx, str) else fx[0]
                        p = params_list[i] if i < len(params_list) else {}
                        amt = p.get('amount', 50.0)
                        self._add_prop(f"  #{i}:", f"{fx_name} ({amt:.0f}%)")
            self._add_separator("Actions")
            self._add_action_btn("Bypass Toggle", "/bypass")
            self._add_action_btn("All Wet (100%)", "/wet")
            self._add_action_btn("All Half (50%)", "/half")
            self._add_action_btn("All Dry (0%)", "/dry")
            self._add_action_btn("Clear All", "/fx clear")
        elif cat_id == 'key':
            self._add_prop("Note:", state.get('key_note', 'C'))
            self._add_prop("Scale:", state.get('key_scale', 'major'))
        elif cat_id == 'filter_envelope':
            self._add_prop("Attack:", f"{state.get('filter_attack', 0.01):.3f}s")
            self._add_prop("Decay:", f"{state.get('filter_decay', 0.1):.3f}s")
            self._add_prop("Sustain:", f"{state.get('filter_sustain', 0.8):.3f}")
            self._add_prop("Release:", f"{state.get('filter_release', 0.1):.3f}s")
        elif cat_id == 'convolution':
            try:
                from mdma_rebuild.dsp.convolution import get_convolution_engine
                info = get_convolution_engine().get_info()
                self._add_prop("IR:", info['ir_name'] or 'none')
                self._add_prop("Wet/Dry:", f"{info['wet']:.0f}/{info['dry']:.0f}")
                self._add_prop("Bank:", f"{info['bank_count']} IRs")
            except Exception:
                self._add_prop("Status:", "Not available")
            self._add_action_btn("Preset: Hall", "/conv preset hall")
        elif cat_id == 'impulse_lfo':
            s = self.executor.session
            if s and hasattr(s, '_lfo_waveshapes'):
                n = len(s._lfo_waveshapes)
                self._add_prop("Shapes:", str(n))
                cur = getattr(s, '_current_lfo_name', '')
                if cur:
                    self._add_prop("Current:", cur)
        elif cat_id == 'impulse_env':
            s = self.executor.session
            if s and hasattr(s, '_imp_envelopes'):
                n = len(s._imp_envelopes)
                self._add_prop("Envelopes:", str(n))
                cur = getattr(s, '_current_imp_env_name', '')
                if cur:
                    self._add_prop("Current:", cur)
        elif cat_id in ('ir_enhance', 'ir_transform', 'ir_granular'):
            try:
                from mdma_rebuild.dsp.convolution import get_convolution_engine
                ce = get_convolution_engine()
                if ce.ir is not None:
                    self._add_prop("Active IR:", ce.ir_name)
                    self._add_prop("Duration:", f"{len(ce.ir)/ce.sr:.2f}s")
                else:
                    self._add_prop("Status:", "Load an IR first (/conv)")
            except Exception:
                self._add_prop("Status:", "Not available")
            if cat_id == 'ir_transform':
                try:
                    from mdma_rebuild.dsp.convolution import list_descriptors
                    descs = list_descriptors()
                    self._add_prop("Descriptors:", f"{len(descs)} available")
                except Exception:
                    self._add_prop("Descriptors:", "Module not available")
        elif cat_id == 'granular_engine':
            self._add_prop("Engine:", "GranularEngine (DSP)")
            self._add_prop("Envelopes:", "hann, triangle, gaussian, trapezoid, tukey, rect")
            self._add_prop("Parameters:", "size, density, position, spread, pitch, reverse")
            self._add_prop("Modulation:", "Audio-rate mod for position, pitch, density, size")
            self._add_separator("Operations")
            self._add_prop("Process:", "/gr process <dur> — full granular cloud")
            self._add_prop("Freeze:", "/gr freeze <pos> <dur> — sustain texture")
            self._add_prop("Stretch:", "/gr stretch <factor> — time-stretch")
            self._add_prop("Shift:", "/gr shift <semi> — pitch shift")
            self._add_separator("Quick Actions")
            self._add_action_btn("Status", "/gr")
            self._add_action_btn("Process 2s", "/gr process 2")
            self._add_action_btn("Freeze Center", "/gr freeze 0.5 4")
        elif cat_id == 'gpu':
            self._add_prop("Engine:", "AudioLDM2 (AI Generation)")
            self._add_prop("Models:", "audioldm2-large, audioldm2-music, audioldm2-full")
            self._add_prop("Schedulers:", "DDPM, DDIM, PNDM, LMS, Euler, DPM++, UniPC")
            self._add_prop("Devices:", "cuda (GPU), cpu, mps (Apple)")
            self._add_separator("Quick Actions")
            self._add_action_btn("Show Status", "/gpu")
            self._add_action_btn("Reset Defaults", "/gpu reset")
        elif cat_id == 'ai_generate':
            self._add_prop("Text-to-Audio:", "/gen <prompt> [dur] — generate from description")
            self._add_prop("Variations:", "/genv <prompt> <count> — multiple versions")
            self._add_prop("Analysis:", "/analyze — attribute vector of audio")
            self._add_prop("Describe:", "/describe — semantic descriptor profile")
            self._add_prop("Ask AI:", "/ask <request> — natural language commands")
            self._add_separator("Quick Actions")
            self._add_action_btn("Analyze", "/analyze detailed")
            self._add_action_btn("Describe", "/describe")
        elif cat_id == 'generative':
            try:
                from mdma_rebuild.dsp import beat_gen
                n_genres = len(beat_gen.GENRE_TEMPLATES)
                n_gens = len(beat_gen.GENERATORS)
                self._add_prop("Genre Templates:", str(n_genres))
                self._add_prop("Sound Generators:", str(n_gens))
            except ImportError:
                self._add_prop("Beat Gen:", "Not available")
            try:
                from mdma_rebuild.dsp import transforms as tf
                n_note = len(tf.TRANSFORM_PRESETS)
                n_audio = len(tf.AUDIO_TRANSFORM_PRESETS)
                self._add_prop("Note Presets:", str(n_note))
                self._add_prop("Audio Presets:", str(n_audio))
            except ImportError:
                self._add_prop("Transforms:", "Not available")
            try:
                from mdma_rebuild.dsp import music_theory as mt
                self._add_prop("Scales:", str(len(mt.SCALES)))
                self._add_prop("Chord Types:", str(len(mt.CHORDS)))
            except ImportError:
                self._add_prop("Music Theory:", "Not available")
            self._add_separator("Quick Actions")
            self._add_action_btn("Generate Beat", "/beat house 4")
            self._add_action_btn("Generate Loop", "/loop house full")
            self._add_action_btn("Generate Melody", "/gen2 melody")
            self._add_action_btn("List Scales", "/theory scales")


class StepGridPanel(wx.Panel):
    """Step grid with playback position highlighting and buffer I/O tracking.

    Shows a visual grid representing beats/steps in the current session,
    highlights the active playback position, and indicates buffer write
    positions. Supports click-drag selection highlighting.
    Includes an accessible text field that represents steps as characters
    so screen readers can navigate and select step ranges using standard
    text-selection shortcuts.
    """

    STEPS_PER_ROW = 16
    CELL_SIZE = 28
    CELL_PAD = 2

    # Colors
    COL_EMPTY = wx.Colour(50, 50, 58)
    COL_FILLED = wx.Colour(80, 130, 200)
    COL_PLAYHEAD = wx.Colour(100, 200, 120)
    COL_WRITE_POS = wx.Colour(255, 200, 100)
    COL_GRID_BG = wx.Colour(35, 35, 42)
    COL_BORDER = wx.Colour(60, 60, 70)
    COL_BEAT_MARKER = wx.Colour(70, 70, 80)
    COL_SELECTED = wx.Colour(140, 100, 220)  # Selection highlight

    # Step text characters for accessible representation
    CHAR_EMPTY = '-'
    CHAR_FILLED = '#'
    CHAR_PLAYHEAD = '>'
    CHAR_WRITE = 'W'
    CHAR_SELECTED = '*'

    def __init__(self, parent, executor: CommandExecutor):
        super().__init__(parent, name="StepGrid", style=wx.TAB_TRAVERSAL)
        self.executor = executor
        self.SetBackgroundColour(self.COL_GRID_BG)

        self.total_steps = 32
        self.playhead_pos = -1  # -1 = not playing
        self.write_pos = 0
        self.filled_steps: set = set()  # Steps that contain audio
        self.track_fills: Dict[int, set] = {}  # Per-track fill data

        # Click-drag selection state
        self.selected_steps: set = set()
        self._drag_active = False
        self._drag_start_step = -1

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        # Header
        hdr_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.header = wx.StaticText(self, label="Step Grid")
        self.header.SetForegroundColour(Theme.FG_DIM)
        hdr_sizer.Add(self.header, 1, wx.ALIGN_CENTER_VERTICAL)

        # Step count selector
        self.step_choice = wx.Choice(self, choices=['16', '32', '64', '128'])
        self.step_choice.SetName("Step Count Selector")
        self.step_choice.SetSelection(1)  # 32 default
        self.step_choice.Bind(wx.EVT_CHOICE, self.on_step_count_change)
        hdr_sizer.Add(wx.StaticText(self, label="Steps:"), 0,
                       wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        hdr_sizer.Add(self.step_choice, 0, wx.LEFT, 4)

        self.sizer.Add(hdr_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Grid canvas
        self.canvas = wx.Panel(self, name="StepGridCanvas")
        self.canvas.SetName("Step grid — use arrow keys to navigate, Shift+Arrow to select, Ctrl+A for all")
        self.canvas.SetBackgroundColour(self.COL_GRID_BG)
        self.canvas.Bind(wx.EVT_PAINT, self.on_paint)
        self.canvas.Bind(wx.EVT_LEFT_DOWN, self.on_grid_mouse_down)
        self.canvas.Bind(wx.EVT_LEFT_UP, self.on_grid_mouse_up)
        self.canvas.Bind(wx.EVT_MOTION, self.on_grid_mouse_drag)
        self.canvas.Bind(wx.EVT_LEFT_DCLICK, self.on_grid_dclick)
        self.canvas.Bind(wx.EVT_KEY_DOWN, self.on_grid_key)
        self.canvas.SetFocus()
        self._kb_cursor = 0  # Keyboard cursor position for arrow-key navigation
        self.canvas.SetMinSize((self.STEPS_PER_ROW * (self.CELL_SIZE + self.CELL_PAD) + 40, 80))
        self.sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)

        # Legend (includes selection color)
        legend_sizer = wx.BoxSizer(wx.HORIZONTAL)
        for color, label in [
            (self.COL_FILLED, "Audio"),
            (self.COL_PLAYHEAD, "Playhead"),
            (self.COL_WRITE_POS, "Write Pos"),
            (self.COL_SELECTED, "Selected"),
            (self.COL_EMPTY, "Empty"),
        ]:
            dot = wx.Panel(self, size=(10, 10))
            dot.SetBackgroundColour(color)
            dot.SetName("")  # Decorative; skip in screen readers
            legend_sizer.Add(dot, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            lbl = wx.StaticText(self, label=label)
            lbl.SetForegroundColour(Theme.FG_DIM)
            legend_sizer.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 3)
        self.sizer.Add(legend_sizer, 0, wx.ALL, 5)

        # ------------------------------------------------------------------
        # Accessible text field: step characters for screen readers
        # Each step maps to a single character. Screen readers can read,
        # arrow-navigate, and select ranges using standard text shortcuts
        # (Shift+Arrow, Ctrl+Shift+Arrow, etc.).  Characters:
        #   -  = empty step       #  = filled (has audio)
        #   >  = playhead         W  = write position
        #   *  = selected
        # ------------------------------------------------------------------
        acc_label = wx.StaticText(self, label="Steps (text, for screen reader selection):")
        acc_label.SetForegroundColour(Theme.FG_DIM)
        self.sizer.Add(acc_label, 0, wx.LEFT | wx.TOP, 5)

        self.step_text = wx.TextCtrl(
            self,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
            name="StepGridText")
        self.step_text.SetBackgroundColour(Theme.BG_INPUT)
        self.step_text.SetForegroundColour(Theme.FG_TEXT)
        font = wx.Font(11, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL,
                        wx.FONTWEIGHT_NORMAL)
        self.step_text.SetFont(font)
        self.step_text.SetToolTip(
            "Step grid as text. Each character is one step: "
            "- empty, # audio, > playhead, W write-pos, * selected. "
            "Use arrow keys and Shift+Arrow to select ranges.")
        # Accessible name/description
        self.step_text.SetName("Step grid text view")
        self.sizer.Add(self.step_text, 0, wx.EXPAND | wx.ALL, 5)

        self.SetSizer(self.sizer)

        # Initial text sync
        self._update_step_text()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _step_from_position(self, x: int, y: int) -> int:
        """Convert pixel position on canvas to step index, or -1."""
        margin_left = 30
        cols = self.STEPS_PER_ROW
        cs = self.CELL_SIZE
        pad = self.CELL_PAD
        col = (x - margin_left) // (cs + pad)
        row = y // (cs + pad)
        if col < 0 or col >= cols:
            return -1
        step = row * cols + col
        if 0 <= step < self.total_steps:
            return step
        return -1

    # ------------------------------------------------------------------
    # Mouse handlers for click-drag selection
    # ------------------------------------------------------------------

    def on_grid_mouse_down(self, event):
        """Start drag selection or single-click write position."""
        step = self._step_from_position(*event.GetPosition())
        if step < 0:
            return
        self._drag_active = True
        self._drag_start_step = step
        # Clear previous selection unless Shift held
        if not event.ShiftDown():
            self.selected_steps = set()
        self.selected_steps.add(step)
        self.canvas.CaptureMouse()
        self.canvas.Refresh()

    def on_grid_mouse_drag(self, event):
        """Extend selection as mouse moves while button is held."""
        if not self._drag_active:
            return
        step = self._step_from_position(*event.GetPosition())
        if step < 0:
            return
        # Select the full range between drag start and current position
        lo = min(self._drag_start_step, step)
        hi = max(self._drag_start_step, step)
        self.selected_steps = set(range(lo, hi + 1))
        self.canvas.Refresh()
        self._update_step_text()

    def on_grid_mouse_up(self, event):
        """Finish drag selection."""
        if self._drag_active:
            self._drag_active = False
            if self.canvas.HasCapture():
                self.canvas.ReleaseMouse()
            step = self._step_from_position(*event.GetPosition())
            if step >= 0:
                lo = min(self._drag_start_step, step)
                hi = max(self._drag_start_step, step)
                self.selected_steps = set(range(lo, hi + 1))
            # If single click (no drag), also set write position
            if len(self.selected_steps) == 1:
                self.write_pos = self._drag_start_step
            self.canvas.Refresh()
            self._update_step_text()

    def on_grid_dclick(self, event):
        """Double-click clears selection."""
        self.selected_steps = set()
        self.canvas.Refresh()
        self._update_step_text()

    def on_grid_key(self, event):
        """Keyboard navigation for step grid (accessibility).

        Arrow Left/Right moves cursor. Shift+Arrow extends selection.
        Up/Down moves by row (16 steps). Home/End jumps to start/end.
        Ctrl+A selects all steps. Space toggles write-position at cursor.
        Escape clears selection.
        """
        key = event.GetKeyCode()
        shift = event.ShiftDown()
        ctrl = event.ControlDown()
        old_cursor = self._kb_cursor

        # Ctrl+A: select all steps
        if ctrl and key == ord('A'):
            self.selected_steps = set(range(self.total_steps))
            self.canvas.Refresh()
            self._update_step_text()
            return

        if key == wx.WXK_RIGHT:
            self._kb_cursor = min(self._kb_cursor + 1, self.total_steps - 1)
        elif key == wx.WXK_LEFT:
            self._kb_cursor = max(self._kb_cursor - 1, 0)
        elif key == wx.WXK_DOWN:
            self._kb_cursor = min(self._kb_cursor + self.STEPS_PER_ROW,
                                  self.total_steps - 1)
        elif key == wx.WXK_UP:
            self._kb_cursor = max(self._kb_cursor - self.STEPS_PER_ROW, 0)
        elif key == wx.WXK_HOME:
            self._kb_cursor = 0
        elif key == wx.WXK_END:
            self._kb_cursor = self.total_steps - 1
        elif key == wx.WXK_SPACE:
            # Space sets write position at cursor
            self.write_pos = self._kb_cursor
            self.canvas.Refresh()
            self._update_step_text()
            return
        elif key == wx.WXK_ESCAPE:
            self.selected_steps = set()
            self.canvas.Refresh()
            self._update_step_text()
            return
        else:
            event.Skip()
            return

        if shift:
            # Extend selection from old cursor to new
            lo = min(old_cursor, self._kb_cursor)
            hi = max(old_cursor, self._kb_cursor)
            for s in range(lo, hi + 1):
                self.selected_steps.add(s)
        else:
            self.selected_steps = {self._kb_cursor}

        self.canvas.Refresh()
        self._update_step_text()

    # ------------------------------------------------------------------
    # Session update
    # ------------------------------------------------------------------

    def update_from_session(self):
        """Pull step data from the session and refresh the grid."""
        state = self.executor.get_engine_state()
        if not state:
            return

        sr = state.get('sample_rate', 48000)
        bpm = state.get('bpm', 128)
        beat_samples = int(60.0 / bpm * sr)

        # Determine filled steps from working buffer
        buf_samples = state.get('buffer_samples', 0)
        if beat_samples > 0 and buf_samples > 0:
            n_filled = min(buf_samples // beat_samples + 1, self.total_steps)
            self.filled_steps = set(range(n_filled))
        else:
            self.filled_steps = set()

        # Track fill data
        s = self.executor.session
        self.track_fills = {}
        if s:
            tracks = getattr(s, 'tracks', [])
            for i, t in enumerate(tracks):
                audio = t.get('audio')
                if audio is not None and len(audio) > 0:
                    wp = t.get('write_pos', 0)
                    if beat_samples > 0:
                        n = min(wp // beat_samples + 1, self.total_steps)
                        self.track_fills[i] = set(range(n))

            # Write position of current track
            if tracks:
                ct_idx = getattr(s, 'current_track_index',
                         getattr(s, 'current_track', 0))
                ct = min(ct_idx, len(tracks) - 1)
                ct = max(ct, 0)
                wp = tracks[ct].get('write_pos', 0)
                self.write_pos = wp // beat_samples if beat_samples > 0 else 0

        self.canvas.Refresh()
        self._update_step_text()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def on_paint(self, event):
        """Draw the step grid with selection highlighting."""
        dc = wx.PaintDC(self.canvas)
        dc.SetBackground(wx.Brush(self.COL_GRID_BG))
        dc.Clear()

        w, h = self.canvas.GetSize()
        cols = self.STEPS_PER_ROW
        rows = max(1, (self.total_steps + cols - 1) // cols)
        cs = self.CELL_SIZE
        pad = self.CELL_PAD

        # Left margin for row labels
        margin_left = 30

        dc.SetFont(wx.Font(8, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL,
                            wx.FONTWEIGHT_NORMAL))

        for step in range(self.total_steps):
            row = step // cols
            col = step % cols
            x = margin_left + col * (cs + pad)
            y = row * (cs + pad)

            # Determine cell color — selection takes visual priority
            if step in self.selected_steps:
                color = self.COL_SELECTED
            elif step == self.playhead_pos:
                color = self.COL_PLAYHEAD
            elif step == self.write_pos:
                color = self.COL_WRITE_POS
            elif step in self.filled_steps:
                color = self.COL_FILLED
            else:
                color = self.COL_EMPTY

            dc.SetBrush(wx.Brush(color))
            dc.SetPen(wx.Pen(self.COL_BORDER, 1))
            dc.DrawRoundedRectangle(x, y, cs, cs, 3)

            # Beat markers (every 4 steps)
            if col % 4 == 0:
                dc.SetTextForeground(Theme.FG_DIM)
                dc.DrawText(str(step + 1), x + 2, y + 2)

            # Row labels
            if col == 0:
                dc.SetTextForeground(Theme.FG_DIM)
                dc.DrawText(f"{row+1}", 2, y + (cs // 2) - 5)

    # ------------------------------------------------------------------
    # Accessible text field
    # ------------------------------------------------------------------

    def _update_step_text(self):
        """Rebuild the text representation of the step grid.

        Each step is one character.  Rows of STEPS_PER_ROW characters are
        separated by newlines so the text wraps identically to the visual
        grid.  Beat boundaries (every 4 steps) are separated by a space
        for easier screen-reader word-navigation (Ctrl+Arrow).
        """
        cols = self.STEPS_PER_ROW
        lines: List[str] = []
        for row_start in range(0, self.total_steps, cols):
            row_chars: List[str] = []
            for step in range(row_start, min(row_start + cols, self.total_steps)):
                if step in self.selected_steps:
                    ch = self.CHAR_SELECTED
                elif step == self.playhead_pos:
                    ch = self.CHAR_PLAYHEAD
                elif step == self.write_pos:
                    ch = self.CHAR_WRITE
                elif step in self.filled_steps:
                    ch = self.CHAR_FILLED
                else:
                    ch = self.CHAR_EMPTY
                row_chars.append(ch)
                # Space after every 4 characters for word-navigation
                if (step - row_start + 1) % 4 == 0 and step != row_start + cols - 1:
                    row_chars.append(' ')
            lines.append(''.join(row_chars))

        text = '\n'.join(lines)
        # Preserve insertion point across updates
        pos = self.step_text.GetInsertionPoint()
        self.step_text.SetValue(text)
        if pos <= len(text):
            self.step_text.SetInsertionPoint(pos)

    # ------------------------------------------------------------------
    # Step count / playhead
    # ------------------------------------------------------------------

    def on_step_count_change(self, event):
        """Handle step count selection change."""
        choices = [16, 32, 64, 128]
        idx = self.step_choice.GetSelection()
        if 0 <= idx < len(choices):
            self.total_steps = choices[idx]
            self.selected_steps = set()
            self.canvas.Refresh()
            self._update_step_text()

    def set_playhead(self, position: int):
        """Set the playhead position (step index, -1 = stopped)."""
        self.playhead_pos = position
        self.canvas.Refresh()
        self._update_step_text()

    def get_selection_range(self) -> Optional[tuple]:
        """Return (start, end) of selected step range, or None."""
        if not self.selected_steps:
            return None
        return (min(self.selected_steps), max(self.selected_steps))


class ConsolePanel(wx.Panel):
    """Bottom panel - output console with command input field."""

    def __init__(self, parent, executor=None, state_sync_callback=None):
        super().__init__(parent, style=wx.TAB_TRAVERSAL)
        self.executor = executor
        self.state_sync_cb = state_sync_callback
        self._cmd_history: List[str] = []
        self._history_idx = -1

        self.SetBackgroundColour(Theme.BG_DARK)

        sizer = wx.BoxSizer(wx.VERTICAL)

        # Header
        header_sizer = wx.BoxSizer(wx.HORIZONTAL)
        header = wx.StaticText(self, label="Console Output")
        header.SetForegroundColour(Theme.FG_DIM)

        clear_btn = wx.Button(self, label="Clear", size=(60, -1))
        clear_btn.Bind(wx.EVT_BUTTON, self.on_clear)

        header_sizer.Add(header, 1, wx.ALIGN_CENTER_VERTICAL)
        header_sizer.Add(clear_btn, 0)

        sizer.Add(header_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Console text (output)
        self.console = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY |
                                   wx.TE_RICH2 | wx.HSCROLL)
        self.console.SetName("Command Output Console")
        self.console.SetBackgroundColour(Theme.BG_DARK)
        self.console.SetForegroundColour(Theme.FG_TEXT)

        # Use monospace font
        font = wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.console.SetFont(font)

        sizer.Add(self.console, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)

        # Command input field
        input_sizer = wx.BoxSizer(wx.HORIZONTAL)
        input_label = wx.StaticText(self, label="Command:")
        input_label.SetForegroundColour(Theme.ACCENT)
        input_sizer.Add(input_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)

        self.cmd_input = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER,
                                      name="CommandInput")
        self.cmd_input.SetName("Command Input — type a slash command and press Enter")
        self.cmd_input.SetBackgroundColour(Theme.BG_INPUT)
        self.cmd_input.SetForegroundColour(Theme.FG_TEXT)
        self.cmd_input.SetFont(font)
        self.cmd_input.SetHint("Type a command (e.g. /tone 440 1) and press Enter")
        self.cmd_input.Bind(wx.EVT_TEXT_ENTER, self._on_cmd_enter)
        self.cmd_input.Bind(wx.EVT_KEY_DOWN, self._on_cmd_key)
        input_sizer.Add(self.cmd_input, 1, wx.EXPAND)

        run_btn = wx.Button(self, label="Run", size=(50, -1))
        run_btn.SetName("Run command")
        run_btn.Bind(wx.EVT_BUTTON, self._on_cmd_enter)
        input_sizer.Add(run_btn, 0, wx.LEFT, 4)

        sizer.Add(input_sizer, 0, wx.EXPAND | wx.ALL, 5)

        self.SetSizer(sizer)

        # Color mapping
        self.colors = {
            'command': Theme.ACCENT,
            'stdout': Theme.FG_TEXT,
            'stderr': Theme.ERROR,
            'success': Theme.SUCCESS,
            'error': Theme.ERROR,
            'info': Theme.FG_DIM,
            'warning': Theme.WARNING,
        }

    def set_executor(self, executor, state_sync_callback=None):
        """Set the executor after construction (for deferred wiring)."""
        self.executor = executor
        self.state_sync_cb = state_sync_callback

    def append(self, text: str, style: str = 'stdout'):
        """Append text to console with color."""
        color = self.colors.get(style, Theme.FG_TEXT)

        # Get current position
        pos = self.console.GetLastPosition()

        # Append text
        self.console.AppendText(text)

        # Apply color to new text
        new_pos = self.console.GetLastPosition()
        self.console.SetStyle(pos, new_pos, wx.TextAttr(color))

        # Scroll to end
        self.console.ShowPosition(new_pos)

    def on_clear(self, event):
        """Clear the console."""
        self.console.Clear()

    def _on_cmd_enter(self, event):
        """Execute the typed command."""
        cmd = self.cmd_input.GetValue().strip()
        if not cmd:
            return
        # Auto-prefix with / if missing
        if not cmd.startswith('/'):
            cmd = '/' + cmd
        # Add to history
        self._cmd_history.append(cmd)
        self._history_idx = -1

        self.append(f">>> {cmd}\n", 'command')

        if self.executor:
            stdout, stderr, ok = self.executor.execute(cmd)
            if stdout:
                self.append(stdout, 'stdout')
            if stderr:
                self.append(stderr, 'stderr')
            self.append(
                f"[{'OK' if ok else 'ERROR'}]\n\n",
                'success' if ok else 'error')
            if self.state_sync_cb:
                self.state_sync_cb()
        else:
            self.append("No engine loaded — cannot execute commands.\n", 'error')

        self.cmd_input.Clear()

    def _on_cmd_key(self, event):
        """Handle Up/Down for command history navigation."""
        key = event.GetKeyCode()
        if key == wx.WXK_UP and self._cmd_history:
            if self._history_idx == -1:
                self._history_idx = len(self._cmd_history) - 1
            elif self._history_idx > 0:
                self._history_idx -= 1
            self.cmd_input.SetValue(self._cmd_history[self._history_idx])
            self.cmd_input.SetInsertionPointEnd()
        elif key == wx.WXK_DOWN and self._cmd_history:
            if self._history_idx >= 0 and self._history_idx < len(self._cmd_history) - 1:
                self._history_idx += 1
                self.cmd_input.SetValue(self._cmd_history[self._history_idx])
                self.cmd_input.SetInsertionPointEnd()
            else:
                self._history_idx = -1
                self.cmd_input.Clear()
        else:
            event.Skip()


# ============================================================================
# MAIN FRAME
# ============================================================================

# ============================================================================
# PHASE 2: PATCH BUILDER PANEL (Feature 2.1)
# ============================================================================

class PatchBuilderPanel(wx.Panel):
    """Dedicated panel for building and editing Monolith patches.

    Displays all operators, their wave types, frequencies, amplitudes,
    and modulation routings in a unified view. Supports inline editing
    of all parameters.
    """

    def __init__(self, parent, executor, console_callback=None, state_sync_callback=None):
        super().__init__(parent, style=wx.TAB_TRAVERSAL)
        self.executor = executor
        self.console_cb = console_callback or (lambda *a: None)
        self.sync_cb = state_sync_callback or (lambda: None)
        self.SetBackgroundColour(Theme.BG_PANEL)
        self._build_ui()

    def _build_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Header
        header = wx.StaticText(self, label="Monolith Patch Builder")
        header.SetForegroundColour(Theme.ACCENT)
        header.SetFont(header.GetFont().Bold())
        sizer.Add(header, 0, wx.ALL, 8)

        # Operator list
        self.op_list = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.op_list.SetName("Operator List — use Enter to edit, Application key for context menu")
        self.op_list.SetBackgroundColour(Theme.BG_INPUT)
        self.op_list.SetForegroundColour(Theme.FG_TEXT)
        self.op_list.InsertColumn(0, "Op", width=40)
        self.op_list.InsertColumn(1, "Wave", width=120)
        self.op_list.InsertColumn(2, "Freq (Hz)", width=80)
        self.op_list.InsertColumn(3, "Amp", width=60)
        self.op_list.InsertColumn(4, "Parameters", width=250)
        self.op_list.Bind(wx.EVT_CONTEXT_MENU, self._on_op_context_menu)
        self.op_list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._on_op_activated)
        sizer.Add(self.op_list, 1, wx.EXPAND | wx.ALL, 4)

        # Routing section
        route_label = wx.StaticText(self, label="Modulation Routing")
        route_label.SetForegroundColour(Theme.ACCENT)
        sizer.Add(route_label, 0, wx.LEFT | wx.TOP, 8)

        self.route_list = wx.ListCtrl(self, style=wx.LC_REPORT)
        self.route_list.SetName("Modulation Routing Table — use Application key for context menu")
        self.route_list.SetBackgroundColour(Theme.BG_INPUT)
        self.route_list.SetForegroundColour(Theme.FG_TEXT)
        self.route_list.InsertColumn(0, "#", width=30)
        self.route_list.InsertColumn(1, "Type", width=60)
        self.route_list.InsertColumn(2, "Source", width=60)
        self.route_list.InsertColumn(3, "Target", width=60)
        self.route_list.InsertColumn(4, "Amount", width=70)
        self.route_list.Bind(wx.EVT_CONTEXT_MENU, self._on_route_context_menu)
        sizer.Add(self.route_list, 0, wx.EXPAND | wx.ALL, 4)
        self.route_list.SetMinSize((-1, 100))

        # Quick action buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        for label, cmd in [
            ("Add Op", "/op"),
            ("Set Wave", None),
            ("Add FM", None),
            ("Clear Routes", "/clearalg"),
            ("Preview", "/tone 440 1"),
        ]:
            btn = wx.Button(self, label=label)
            btn.SetBackgroundColour(Theme.BG_INPUT)
            btn.SetForegroundColour(Theme.FG_TEXT)
            if cmd:
                btn.Bind(wx.EVT_BUTTON, lambda e, c=cmd: self._exec(c))
            elif label == "Set Wave":
                btn.Bind(wx.EVT_BUTTON, self._on_set_wave)
            elif label == "Add FM":
                btn.Bind(wx.EVT_BUTTON, self._on_add_routing)
            btn_sizer.Add(btn, 0, wx.ALL, 2)
        sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 4)

        # Parameter quick-edit section
        param_sizer = wx.BoxSizer(wx.HORIZONTAL)
        param_sizer.Add(wx.StaticText(self, label="Param:"), 0,
                        wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.param_input = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        self.param_input.SetName("Patch parameter command — type a command and press Enter")
        self.param_input.SetBackgroundColour(Theme.BG_INPUT)
        self.param_input.SetForegroundColour(Theme.FG_TEXT)
        self.param_input.SetHint("e.g. /wm supersaw saws=7 spread=0.5")
        self.param_input.Bind(wx.EVT_TEXT_ENTER, self._on_param_enter)
        param_sizer.Add(self.param_input, 1, wx.EXPAND)
        sizer.Add(param_sizer, 0, wx.EXPAND | wx.ALL, 4)

        self.SetSizer(sizer)

    def _exec(self, cmd):
        """Execute one or more newline-separated commands."""
        commands = [c.strip() for c in cmd.split('\n') if c.strip()]
        try:
            for single_cmd in commands:
                stdout, stderr, ok = self.executor.execute(single_cmd)
                if stdout and stdout.strip():
                    self.console_cb(stdout, 'info')
                if stderr and stderr.strip():
                    self.console_cb(stderr, 'error')
                if not ok:
                    break
            self.sync_cb()
            self.refresh()
        except Exception as e:
            self.console_cb(f"ERROR: {e}\n", 'error')

    def _on_op_context_menu(self, event):
        """Context menu for operator list (keyboard and mouse)."""
        sel = self.op_list.GetFirstSelected()
        if sel < 0:
            return
        op_text = self.op_list.GetItemText(sel, 0)
        try:
            op_idx = int(op_text)
        except ValueError:
            return
        wave = self.op_list.GetItemText(sel, 1)

        menu = wx.Menu()
        m_select = wx.NewIdRef()
        m_wave = wx.NewIdRef()
        m_freq = wx.NewIdRef()
        m_amp = wx.NewIdRef()
        m_gen = wx.NewIdRef()
        m_fm = wx.NewIdRef()
        m_tfm = wx.NewIdRef()
        m_am = wx.NewIdRef()
        m_rm = wx.NewIdRef()
        m_pm = wx.NewIdRef()

        menu.Append(m_select, f"Select Operator {op_idx}")
        menu.Append(m_gen, "Generate Tone (440Hz)")
        menu.AppendSeparator()
        menu.Append(m_wave, f"Set Waveform (current: {wave})...")
        menu.Append(m_freq, "Set Frequency...")
        menu.Append(m_amp, "Set Amplitude...")
        menu.AppendSeparator()
        mod_sub = wx.Menu()
        mod_sub.Append(m_fm, "FM (Frequency Mod)...")
        mod_sub.Append(m_tfm, "TFM (Through-Zero FM)...")
        mod_sub.Append(m_am, "AM (Amplitude Mod)...")
        mod_sub.Append(m_rm, "RM (Ring Mod)...")
        mod_sub.Append(m_pm, "PM (Phase Mod)...")
        menu.AppendSubMenu(mod_sub, f"Add Routing from Op {op_idx}")

        self.Bind(wx.EVT_MENU,
            lambda e, i=op_idx: self._exec(f'/op {i}'), id=m_select)
        self.Bind(wx.EVT_MENU,
            lambda e, i=op_idx: self._exec(f'/op {i}\n/tone 440 1'), id=m_gen)
        self.Bind(wx.EVT_MENU,
            lambda e, i=op_idx: self._on_set_wave_for_op(i), id=m_wave)
        self.Bind(wx.EVT_MENU,
            lambda e, i=op_idx: self._on_set_freq(i), id=m_freq)
        self.Bind(wx.EVT_MENU,
            lambda e, i=op_idx: self._on_set_amp(i), id=m_amp)
        self.Bind(wx.EVT_MENU,
            lambda e, i=op_idx: self._on_add_routing_typed(i, 'fm'), id=m_fm)
        self.Bind(wx.EVT_MENU,
            lambda e, i=op_idx: self._on_add_routing_typed(i, 'tfm'), id=m_tfm)
        self.Bind(wx.EVT_MENU,
            lambda e, i=op_idx: self._on_add_routing_typed(i, 'am'), id=m_am)
        self.Bind(wx.EVT_MENU,
            lambda e, i=op_idx: self._on_add_routing_typed(i, 'rm'), id=m_rm)
        self.Bind(wx.EVT_MENU,
            lambda e, i=op_idx: self._on_add_routing_typed(i, 'pm'), id=m_pm)

        self.PopupMenu(menu)
        menu.Destroy()

    def _on_op_activated(self, event):
        """Handle Enter / double-click on operator list — select operator."""
        sel = self.op_list.GetFirstSelected()
        if sel < 0:
            return
        op_text = self.op_list.GetItemText(sel, 0)
        try:
            op_idx = int(op_text)
        except ValueError:
            return
        self._exec(f'/op {op_idx}')

    def _on_route_context_menu(self, event):
        """Context menu for routing list (keyboard and mouse)."""
        sel = self.route_list.GetFirstSelected()
        menu = wx.Menu()
        m_add = wx.NewIdRef()
        m_clear = wx.NewIdRef()
        menu.Append(m_add, "Add New Routing...")
        menu.AppendSeparator()
        menu.Append(m_clear, "Clear All Routings")

        self.Bind(wx.EVT_MENU,
            lambda e: self._on_add_routing(None), id=m_add)
        self.Bind(wx.EVT_MENU,
            lambda e: self._exec('/clearalg'), id=m_clear)

        self.PopupMenu(menu)
        menu.Destroy()

    def _on_add_routing_typed(self, source_op, route_type='fm'):
        """Add a modulation routing with source pre-filled from operator."""
        dlg = wx.TextEntryDialog(
            self,
            f"Enter target operator for {route_type.upper()} from Op {source_op}:\n"
            f"(e.g. '0' or '0 2.5' for target with amount)",
            f"Add {route_type.upper()} Routing from Op {source_op}",
            "0")
        if dlg.ShowModal() == wx.ID_OK:
            val = dlg.GetValue().strip()
            if val:
                self._exec(f'/{route_type} {source_op} {val}')
        dlg.Destroy()

    def _on_set_wave_for_op(self, op_idx):
        """Show waveform picker targeting a specific operator."""
        wave_types = [
            'sine', 'triangle', 'saw', 'pulse', 'noise', 'pink',
            'physical', 'physical2',
            'supersaw', 'additive', 'formant', 'harmonic',
            'waveguide_string', 'waveguide_tube', 'waveguide_membrane', 'waveguide_plate',
            'wavetable', 'compound',
        ]
        dlg = wx.SingleChoiceDialog(self, "Select waveform:", "Set Waveform", wave_types)
        if dlg.ShowModal() == wx.ID_OK:
            wave = dlg.GetStringSelection()
            self._exec(f"/op {op_idx}\n/wm {wave}")
        dlg.Destroy()

    def _on_set_freq(self, op_idx):
        """Show frequency editor for an operator."""
        dlg = wx.TextEntryDialog(self, "Enter frequency (Hz):",
                                  "Set Frequency", "440")
        if dlg.ShowModal() == wx.ID_OK:
            val = dlg.GetValue().strip()
            if val:
                self._exec(f"/op {op_idx}\n/freq {val}")
        dlg.Destroy()

    def _on_set_amp(self, op_idx):
        """Show amplitude editor for an operator."""
        dlg = wx.TextEntryDialog(self, "Enter amplitude (0-1):",
                                  "Set Amplitude", "0.8")
        if dlg.ShowModal() == wx.ID_OK:
            val = dlg.GetValue().strip()
            if val:
                self._exec(f"/op {op_idx}\n/amp {val}")
        dlg.Destroy()

    def _on_param_enter(self, event):
        cmd = self.param_input.GetValue().strip()
        if cmd:
            if not cmd.startswith('/'):
                cmd = '/' + cmd
            self._exec(cmd)
            self.param_input.Clear()

    def _on_set_wave(self, event):
        wave_types = [
            'sine', 'triangle', 'saw', 'pulse', 'noise', 'pink',
            'physical', 'physical2',
            'supersaw', 'additive', 'formant', 'harmonic',
            'waveguide_string', 'waveguide_tube', 'waveguide_membrane', 'waveguide_plate',
            'wavetable', 'compound',
        ]
        dlg = wx.SingleChoiceDialog(self, "Select waveform:", "Set Waveform", wave_types)
        if dlg.ShowModal() == wx.ID_OK:
            wave = dlg.GetStringSelection()
            self._exec(f"/wm {wave}")
        dlg.Destroy()

    def _on_add_routing(self, event):
        dlg = wx.TextEntryDialog(self, "Enter routing (e.g. FM 2 1 50):",
                                  "Add Modulation Routing")
        if dlg.ShowModal() == wx.ID_OK:
            parts = dlg.GetValue().strip().split()
            if len(parts) >= 3:
                algo = parts[0].lower()
                self._exec(f"/{algo} {' '.join(parts[1:])}")
        dlg.Destroy()

    def refresh(self):
        """Refresh the operator and routing displays."""
        self.op_list.DeleteAllItems()
        self.route_list.DeleteAllItems()

        if not self.executor.session or not hasattr(self.executor.session, 'engine'):
            return

        engine = self.executor.session.engine
        if not hasattr(engine, 'operators') or not hasattr(engine, 'algorithms'):
            return

        # Populate operators
        for idx in sorted(engine.operators.keys()):
            op = engine.operators[idx]
            row = self.op_list.GetItemCount()
            self.op_list.InsertItem(row, str(idx))
            self.op_list.SetItem(row, 1, op.get('wave', 'sine'))
            self.op_list.SetItem(row, 2, f"{op.get('freq', 440.0):.1f}")
            self.op_list.SetItem(row, 3, f"{op.get('amp', 1.0):.3f}")

            # Build params string based on wave type
            wave = op.get('wave', 'sine')
            params = []
            if wave == 'pulse':
                params.append(f"pw={op.get('pw', 0.5):.2f}")
            elif wave in ('physical', 'physical2'):
                params.append(f"even={op.get('even_harmonics', 8)}")
                params.append(f"odd={op.get('odd_harmonics', 4)}")
                params.append(f"wt={op.get('even_weight', 1.0):.1f}")
                params.append(f"dec={op.get('decay', 0.7):.2f}")
                if wave == 'physical2':
                    params.append(f"inh={op.get('inharmonicity', 0.01):.3f}")
                    params.append(f"parts={op.get('partials', 12)}")
            elif wave == 'supersaw':
                params.append(f"saws={op.get('num_saws', 7)}")
                params.append(f"spread={op.get('detune_spread', 0.5):.2f}")
                params.append(f"mix={op.get('mix', 0.75):.2f}")
            elif wave == 'additive':
                params.append(f"nharm={op.get('num_harmonics', 16)}")
                params.append(f"rolloff={op.get('rolloff', 1.0):.1f}")
            elif wave == 'formant':
                params.append(f"vowel={op.get('vowel', 'a')}")
            elif wave == 'harmonic':
                params.append(f"odd={op.get('odd_level', 1.0):.1f}")
                params.append(f"even={op.get('even_level', 1.0):.1f}")
                params.append(f"nharm={op.get('num_harmonics', 16)}")
            elif wave == 'waveguide_string':
                params.append(f"damp={op.get('damping', 0.996):.3f}")
                params.append(f"bright={op.get('brightness', 0.5):.2f}")
                params.append(f"pos={op.get('position', 0.5):.2f}")
            elif wave == 'waveguide_tube':
                params.append(f"damp={op.get('damping', 0.996):.3f}")
                params.append(f"refl={op.get('reflection', 0.98):.2f}")
                params.append(f"bore={op.get('bore_shape', 'cylindrical')}")
            elif wave == 'waveguide_membrane':
                params.append(f"tens={op.get('tension', 0.5):.2f}")
                params.append(f"damp={op.get('damping', 0.996):.3f}")
                params.append(f"strike={op.get('strike_pos', 0.5):.2f}")
            elif wave == 'waveguide_plate':
                params.append(f"thick={op.get('thickness', 0.5):.2f}")
                params.append(f"damp={op.get('damping', 0.996):.3f}")
                params.append(f"mat={op.get('material', 'steel')}")
            elif wave == 'wavetable':
                params.append(f"table={op.get('wavetable_name', '')}")
                params.append(f"frame={op.get('frame_pos', 0.0):.2f}")
            elif wave == 'compound':
                params.append(f"name={op.get('compound_name', '')}")
                params.append(f"morph={op.get('morph', 0.0):.2f}")
            self.op_list.SetItem(row, 4, ', '.join(params))

        # Populate routings
        for i, (algo_type, src, tgt, amt) in enumerate(engine.algorithms):
            row = self.route_list.GetItemCount()
            self.route_list.InsertItem(row, str(i))
            self.route_list.SetItem(row, 1, algo_type)
            self.route_list.SetItem(row, 2, f"op{src}")
            self.route_list.SetItem(row, 3, f"op{tgt}")
            self.route_list.SetItem(row, 4, f"{amt:.3f}")


# ============================================================================
# PHASE 2: CARRIER/MODULATOR ROUTING VIEW (Feature 2.2)
# ============================================================================

class RoutingPanel(wx.Panel):
    """Visual representation of carrier/modulator signal flow.

    Shows operators as boxes connected by arrows representing
    modulation routings, with signal type and amount labels.
    """

    def __init__(self, parent, executor):
        super().__init__(parent, style=wx.TAB_TRAVERSAL)
        self.executor = executor
        self.SetBackgroundColour(Theme.BG_DARK)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.SetMinSize((300, 200))

    def on_paint(self, event):
        dc = wx.BufferedPaintDC(self)
        gc = wx.GraphicsContext.Create(dc)
        if not gc:
            return

        w, h = self.GetSize()
        gc.SetBrush(wx.Brush(Theme.BG_DARK))
        gc.SetPen(wx.TRANSPARENT_PEN)
        gc.DrawRectangle(0, 0, w, h)

        if not self.executor.session or not hasattr(self.executor.session, 'engine'):
            gc.SetFont(gc.CreateFont(wx.Font(12, wx.FONTFAMILY_DEFAULT,
                        wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL), Theme.FG_DIM))
            gc.DrawText("No engine loaded", 20, h // 2)
            return

        engine = self.executor.session.engine
        if not hasattr(engine, 'operators') or not hasattr(engine, 'algorithms'):
            gc.SetFont(gc.CreateFont(wx.Font(12, wx.FONTFAMILY_DEFAULT,
                        wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL), Theme.FG_DIM))
            gc.DrawText("Engine not initialized", 20, h // 2)
            return

        ops = sorted(engine.operators.keys())

        if not ops:
            gc.SetFont(gc.CreateFont(wx.Font(12, wx.FONTFAMILY_DEFAULT,
                        wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL), Theme.FG_DIM))
            gc.DrawText("No operators defined", 20, h // 2)
            return

        # Layout operators as boxes
        num_ops = len(ops)
        box_w, box_h = 100, 60
        margin = 20
        spacing = max(30, (w - margin * 2 - box_w * num_ops) / max(1, num_ops - 1) + box_w)

        op_positions = {}
        for i, idx in enumerate(ops):
            x = margin + i * spacing
            y = h // 2 - box_h // 2
            op_positions[idx] = (x, y)

            # Draw operator box
            op = engine.operators[idx]
            wave = op.get('wave', 'sine')
            freq = op.get('freq', 440.0)

            gc.SetBrush(wx.Brush(Theme.BG_PANEL))
            gc.SetPen(wx.Pen(Theme.ACCENT, 2))
            gc.DrawRoundedRectangle(x, y, box_w, box_h, 6)

            # Label
            gc.SetFont(gc.CreateFont(wx.Font(9, wx.FONTFAMILY_DEFAULT,
                        wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD), Theme.FG_TEXT))
            gc.DrawText(f"Op {idx}", x + 8, y + 4)
            gc.SetFont(gc.CreateFont(wx.Font(8, wx.FONTFAMILY_DEFAULT,
                        wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL), Theme.FG_DIM))
            gc.DrawText(wave, x + 8, y + 22)
            gc.DrawText(f"{freq:.0f} Hz", x + 8, y + 38)

        # Draw routing arrows
        algo_colors = {
            'FM': Theme.ACCENT,
            'TFM': wx.Colour(180, 100, 255),
            'AM': Theme.SUCCESS,
            'RM': Theme.WARNING,
            'PM': Theme.ERROR,
        }

        for algo_type, src, tgt, amt in engine.algorithms:
            if src not in op_positions or tgt not in op_positions:
                continue
            sx, sy = op_positions[src]
            tx, ty = op_positions[tgt]

            color = algo_colors.get(algo_type, Theme.FG_DIM)
            gc.SetPen(wx.Pen(color, 2))

            # Draw arrow from source bottom to target top
            start_x = sx + box_w // 2
            start_y = sy + box_h
            end_x = tx + box_w // 2
            end_y = ty

            path = gc.CreatePath()
            path.MoveToPoint(start_x, start_y)
            # Bezier curve for nice routing lines
            mid_y = start_y + 30
            path.AddCurveToPoint(start_x, mid_y, end_x, mid_y, end_x, end_y)
            gc.StrokePath(path)

            # Label
            label_x = (start_x + end_x) / 2
            label_y = mid_y - 8
            gc.SetFont(gc.CreateFont(wx.Font(7, wx.FONTFAMILY_DEFAULT,
                        wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL), color))
            gc.DrawText(f"{algo_type} {amt:.1f}", label_x - 15, label_y)

    def refresh(self):
        self.Refresh()


# ============================================================================
# PHASE 2: OSCILLATOR LIST VIEW (Feature 2.4)
# ============================================================================

class OscillatorListPanel(wx.Panel):
    """Scrollable list-based oscillator browser.

    Each entry shows waveform name, frequency, amplitude, and
    quick-edit controls for common parameters.
    """

    def __init__(self, parent, executor, console_callback=None, state_sync_callback=None):
        super().__init__(parent, style=wx.TAB_TRAVERSAL)
        self.executor = executor
        self.console_cb = console_callback or (lambda *a: None)
        self.sync_cb = state_sync_callback or (lambda: None)
        self.SetBackgroundColour(Theme.BG_PANEL)
        self._build_ui()

    def _build_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)

        header = wx.StaticText(self, label="Oscillator Browser")
        header.SetForegroundColour(Theme.ACCENT)
        header.SetFont(header.GetFont().Bold())
        sizer.Add(header, 0, wx.ALL, 8)

        # Category filter
        filter_sizer = wx.BoxSizer(wx.HORIZONTAL)
        filter_sizer.Add(wx.StaticText(self, label="Filter:"), 0,
                         wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.wave_filter = wx.Choice(self, choices=[
            "All", "Basic", "Noise", "Physical", "Extended",
            "Waveguide", "Wavetable", "Compound"
        ])
        self.wave_filter.SetName("Oscillator Wave Type Filter")
        self.wave_filter.SetSelection(0)
        self.wave_filter.Bind(wx.EVT_CHOICE, lambda e: self.refresh())
        filter_sizer.Add(self.wave_filter, 0, wx.RIGHT, 8)

        # Add operator button
        add_btn = wx.Button(self, label="+ Add Operator")
        add_btn.SetName("Add New Oscillator Operator")
        add_btn.SetBackgroundColour(Theme.BG_INPUT)
        add_btn.SetForegroundColour(Theme.SUCCESS)
        add_btn.Bind(wx.EVT_BUTTON, self._on_add_op)
        filter_sizer.Add(add_btn, 0)

        sizer.Add(filter_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        # Scrolled operator cards
        self.scroll = wx.ScrolledWindow(self)
        self.scroll.SetName("Oscillator Cards")
        self.scroll.SetScrollRate(0, 20)
        self.scroll.SetBackgroundColour(Theme.BG_DARK)
        self.scroll_sizer = wx.BoxSizer(wx.VERTICAL)
        self.scroll.SetSizer(self.scroll_sizer)
        sizer.Add(self.scroll, 1, wx.EXPAND | wx.ALL, 4)

        self.SetSizer(sizer)

    def _on_add_op(self, event):
        """Add a new operator."""
        if not self.executor.session or not hasattr(self.executor.session, 'engine'):
            return
        engine = self.executor.session.engine
        if not hasattr(engine, 'operators'):
            return
        # Find next available index
        existing = set(engine.operators.keys())
        new_idx = 1
        while new_idx in existing:
            new_idx += 1
        try:
            stdout, stderr, ok = self.executor.execute(f"/op {new_idx}")
            if stdout and stdout.strip():
                self.console_cb(stdout, 'info')
            if stderr and stderr.strip():
                self.console_cb(stderr, 'error')
            self.sync_cb()
            self.refresh()
        except Exception as e:
            self.console_cb(f"ERROR: {e}\n", 'error')

    def refresh(self):
        """Rebuild the oscillator cards."""
        self.scroll_sizer.Clear(True)

        if not self.executor.session or not hasattr(self.executor.session, 'engine'):
            return

        engine = self.executor.session.engine
        if not hasattr(engine, 'operators'):
            return

        filter_sel = self.wave_filter.GetStringSelection()

        # Category mapping
        categories = {
            'Basic': {'sine', 'triangle', 'saw', 'pulse'},
            'Noise': {'noise', 'pink'},
            'Physical': {'physical', 'physical2'},
            'Extended': {'supersaw', 'additive', 'formant', 'harmonic'},
            'Waveguide': {'waveguide_string', 'waveguide_tube', 'waveguide_membrane', 'waveguide_plate'},
            'Wavetable': {'wavetable'},
            'Compound': {'compound'},
        }

        for idx in sorted(engine.operators.keys()):
            op = engine.operators[idx]
            wave = op.get('wave', 'sine')

            # Apply filter
            if filter_sel != "All":
                allowed = categories.get(filter_sel, set())
                if wave not in allowed:
                    continue

            card = self._create_op_card(idx, op)
            self.scroll_sizer.Add(card, 0, wx.EXPAND | wx.ALL, 4)

        self.scroll.FitInside()
        self.scroll.Layout()

    def _create_op_card(self, idx, op):
        """Create a card panel for one operator with keyboard access."""
        card = wx.Panel(self.scroll, style=wx.TAB_TRAVERSAL)
        card.SetBackgroundColour(Theme.BG_PANEL)
        card_sizer = wx.BoxSizer(wx.HORIZONTAL)

        wave = op.get('wave', 'sine')
        freq = op.get('freq', 440.0)
        amp_val = op.get('amp', 1.0)

        # Op index badge
        badge = wx.StaticText(card, label=f" Op{idx} ")
        badge.SetForegroundColour(Theme.BG_DARK)
        badge.SetBackgroundColour(Theme.ACCENT)
        badge.SetFont(badge.GetFont().Bold())
        card_sizer.Add(badge, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)

        # Info
        info_sizer = wx.BoxSizer(wx.VERTICAL)
        wave_text = wx.StaticText(card, label=wave)
        wave_text.SetForegroundColour(Theme.FG_TEXT)
        wave_text.SetFont(wave_text.GetFont().Bold())
        info_sizer.Add(wave_text, 0)

        detail = f"{freq:.1f} Hz  |  amp: {amp_val:.3f}"
        detail_text = wx.StaticText(card, label=detail)
        detail_text.SetForegroundColour(Theme.FG_DIM)
        info_sizer.Add(detail_text, 0)

        card_sizer.Add(info_sizer, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)

        # Quick-edit buttons
        select_btn = wx.Button(card, label="Select", size=(55, -1))
        select_btn.SetName(f"Select Operator {idx} ({wave} {freq:.0f}Hz)")
        select_btn.Bind(wx.EVT_BUTTON, lambda e, i=idx: self._select_op(i))
        card_sizer.Add(select_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)

        wave_btn = wx.Button(card, label="Wave", size=(55, -1))
        wave_btn.SetName(f"Change waveform for Operator {idx}")
        wave_btn.Bind(wx.EVT_BUTTON, lambda e, i=idx: self._change_wave(i))
        card_sizer.Add(wave_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)

        card.SetSizer(card_sizer)
        return card

    def _change_wave(self, idx):
        """Show waveform picker for an oscillator."""
        wave_types = [
            'sine', 'triangle', 'saw', 'pulse', 'noise', 'pink',
            'physical', 'physical2',
            'supersaw', 'additive', 'formant', 'harmonic',
            'waveguide_string', 'waveguide_tube', 'waveguide_membrane', 'waveguide_plate',
            'wavetable', 'compound',
        ]
        dlg = wx.SingleChoiceDialog(self, "Select waveform:", "Set Waveform", wave_types)
        if dlg.ShowModal() == wx.ID_OK:
            wave = dlg.GetStringSelection()
            try:
                for cmd in [f"/op {idx}", f"/wm {wave}"]:
                    stdout, stderr, ok = self.executor.execute(cmd)
                    if stdout and stdout.strip():
                        self.console_cb(stdout, 'info')
                    if stderr and stderr.strip():
                        self.console_cb(stderr, 'error')
                    if not ok:
                        break
                self.sync_cb()
                self.refresh()
            except Exception as e:
                self.console_cb(f"ERROR: {e}\n", 'error')
        dlg.Destroy()

    def _select_op(self, idx):
        try:
            stdout, stderr, ok = self.executor.execute(f"/op {idx}")
            if stdout and stdout.strip():
                self.console_cb(stdout, 'info')
            if stderr and stderr.strip():
                self.console_cb(stderr, 'error')
            self.sync_cb()
        except Exception as e:
            self.console_cb(f"ERROR: {e}\n", 'error')


class MDMAFrame(wx.Frame):
    """Main application window.

    Layout (Phase 2):
    +-----------+----------------------------------+
    |  Object   |  Inspector / Actions / Patch /   |
    |  Browser  |  Oscillators / Routing           |
    |  (tree)   |  (notebook tabs)                 |
    +-----------+----------------------------------+
    |  Step Grid  |  Console Output                |
    +-------------+--------------------------------+
    """

    VERSION = "4.2.0"

    def __init__(self):
        super().__init__(None, title="MDMA - Music Design Made Accessible",
                         size=(1400, 900))

        self.SetBackgroundColour(Theme.BG_DARK)

        # Initialize executor
        self.executor = CommandExecutor()

        # Auto-refresh timer (500ms interval for live state updates)
        self.refresh_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_auto_refresh, self.refresh_timer)

        # Create menu bar
        self.create_menu_bar()

        # Create main layout with splitters
        self.create_layout()

        # Set up keyboard shortcuts
        self.setup_shortcuts()

        # Status bar with multiple fields
        self.CreateStatusBar(3)
        self.SetStatusText(f"MDMA GUI v{self.VERSION} - Ready", 0)

        # Center on screen
        self.Centre()

        # Welcome message
        self.console.append(f"MDMA GUI v{self.VERSION} - Phase A: Accessibility Audit\n", 'info')
        self.console.append("=" * 55 + "\n", 'info')

        # Warn if engine failed to load
        if self.executor.init_error:
            self.console.append(
                f"WARNING: MDMA engine failed to load: "
                f"{self.executor.init_error}\n", 'error')
            self.console.append(
                "Commands will not execute. Check your installation.\n\n",
                'error')
            self.SetStatusText(
                f"MDMA GUI v{self.VERSION} - ENGINE NOT LOADED", 0)
        else:
            self.console.append(
                "Browse objects on the left. Select to inspect.\n", 'info')
            self.console.append(
                "Right-click or Application key for context actions.\n", 'info')
            self.console.append(
                "Ctrl+L to focus command input. Ctrl+R to run action.\n\n", 'info')
            # Show live engine state on startup
            self.sync_state()
            # Start auto-refresh timer
            self.refresh_timer.Start(500)

        self.Bind(wx.EVT_CLOSE, self.on_close)

    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = wx.MenuBar()

        # File menu
        file_menu = wx.Menu()
        file_menu.Append(wx.ID_NEW, "&New Session\tCtrl+N")
        file_menu.Append(wx.ID_OPEN, "&Open Project\tCtrl+O")
        file_menu.Append(wx.ID_SAVE, "&Save Project\tCtrl+S")
        file_menu.AppendSeparator()
        file_menu.Append(wx.ID_EXIT, "E&xit\tAlt+F4")
        menubar.Append(file_menu, "&File")

        # View menu
        view_menu = wx.Menu()
        self.view_refresh_id = wx.NewIdRef()
        self.view_console_id = wx.NewIdRef()
        self.view_search_id = wx.NewIdRef()
        self.view_inspector_id = wx.NewIdRef()
        self.view_grid_id = wx.NewIdRef()
        view_menu.Append(self.view_refresh_id, "&Refresh\tF5")
        view_menu.Append(self.view_console_id, "Focus &Console\tCtrl+L")
        view_menu.Append(self.view_search_id, "Focus &Search\tCtrl+F")
        view_menu.AppendSeparator()
        view_menu.Append(self.view_inspector_id, "Show &Inspector\tCtrl+I")
        view_menu.Append(self.view_grid_id, "Show Step &Grid\tCtrl+G")
        menubar.Append(view_menu, "&View")

        # Engine menu
        engine_menu = wx.Menu()
        self.engine_version_id = wx.NewIdRef()
        self.engine_help_id = wx.NewIdRef()
        self.engine_output_id = wx.NewIdRef()
        engine_menu.Append(self.engine_version_id, "&Version Info")
        engine_menu.Append(self.engine_help_id, "&Help / Quick Reference")
        engine_menu.AppendSeparator()
        engine_menu.Append(self.engine_output_id, "Open &Output Folder")
        menubar.Append(engine_menu, "&Engine")

        # Help menu
        help_menu = wx.Menu()
        help_menu.Append(wx.ID_ABOUT, "&About")
        menubar.Append(help_menu, "&Help")

        self.SetMenuBar(menubar)

        # Bind events
        self.Bind(wx.EVT_MENU, self.on_new_session, id=wx.ID_NEW)
        self.Bind(wx.EVT_MENU, self.on_open_project, id=wx.ID_OPEN)
        self.Bind(wx.EVT_MENU, self.on_save_project, id=wx.ID_SAVE)
        self.Bind(wx.EVT_MENU, self.on_exit, id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU, self.on_about, id=wx.ID_ABOUT)
        self.Bind(wx.EVT_MENU,
            lambda e: self._exec_menu_cmd('/version'), id=self.engine_version_id)
        self.Bind(wx.EVT_MENU,
            lambda e: self._exec_menu_cmd('/help'), id=self.engine_help_id)

    def create_layout(self):
        """Create the main layout with splitters.

        Layout:
        main_splitter (horizontal) splits top/bottom
          top_splitter (vertical) splits left/right
            left: ObjectBrowser
            right: notebook with Inspector + ActionPanel tabs
          bottom_splitter (vertical) splits grid/console
            left: StepGridPanel
            right: ConsolePanel
        """
        # Main horizontal splitter (top panels / bottom panels)
        self.main_splitter = wx.SplitterWindow(self)

        # Top vertical splitter (browser / right notebook)
        self.top_splitter = wx.SplitterWindow(self.main_splitter)

        # Bottom vertical splitter (step grid / console)
        self.bottom_splitter = wx.SplitterWindow(self.main_splitter)

        # --- Left panel: Object Browser ---
        self.browser = ObjectBrowser(
            self.top_splitter, self.on_object_select, self.executor,
            console_callback=self.console_append)

        # --- Right panel: Notebook with Inspector + Action Panel ---
        self.right_notebook = wx.Notebook(self.top_splitter,
                                           name="RightNotebook")
        self.right_notebook.SetName("Inspector and tools — use Ctrl+Tab to switch tabs")
        self.right_notebook.SetBackgroundColour(Theme.BG_PANEL)

        self.inspector = InspectorPanel(
            self.right_notebook, self.executor,
            console_callback=self.console_append,
            state_sync_callback=self.sync_state)
        self.right_notebook.AddPage(self.inspector, "Inspector")

        self.action_panel = ActionPanel(
            self.right_notebook, self.executor, self.console_append,
            state_sync_callback=self.sync_state)
        self.right_notebook.AddPage(self.action_panel, "Actions")

        # Phase 2 panels
        self.patch_builder = PatchBuilderPanel(
            self.right_notebook, self.executor,
            console_callback=self.console_append,
            state_sync_callback=self.sync_state)
        self.right_notebook.AddPage(self.patch_builder, "Patch")

        self.osc_list = OscillatorListPanel(
            self.right_notebook, self.executor,
            console_callback=self.console_append,
            state_sync_callback=self.sync_state)
        self.right_notebook.AddPage(self.osc_list, "Oscillators")

        self.routing_panel = RoutingPanel(
            self.right_notebook, self.executor)
        self.right_notebook.AddPage(self.routing_panel, "Routing")

        # --- Bottom left: Step Grid ---
        self.step_grid = StepGridPanel(self.bottom_splitter, self.executor)

        # --- Bottom right: Console (with command input) ---
        self.console = ConsolePanel(self.bottom_splitter,
                                     executor=self.executor,
                                     state_sync_callback=self.sync_state)

        # Configure splitters
        self.top_splitter.SplitVertically(
            self.browser, self.right_notebook, 300)
        self.top_splitter.SetMinimumPaneSize(220)

        self.bottom_splitter.SplitVertically(
            self.step_grid, self.console, 500)
        self.bottom_splitter.SetMinimumPaneSize(200)

        self.main_splitter.SplitHorizontally(
            self.top_splitter, self.bottom_splitter, -250)
        self.main_splitter.SetMinimumPaneSize(150)

    def setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        self.run_id = wx.NewIdRef()
        self.console_focus_id = wx.NewIdRef()
        self.search_focus_id = wx.NewIdRef()
        self.refresh_shortcut_id = wx.NewIdRef()
        self.inspector_focus_id = wx.NewIdRef()

        accel_table = wx.AcceleratorTable([
            (wx.ACCEL_CTRL, ord('R'), self.run_id),
            (wx.ACCEL_CTRL, ord('L'), self.console_focus_id),
            (wx.ACCEL_CTRL, ord('F'), self.search_focus_id),
            (wx.ACCEL_NORMAL, wx.WXK_F5, self.refresh_shortcut_id),
            (wx.ACCEL_CTRL, ord('I'), self.inspector_focus_id),
        ])
        self.SetAcceleratorTable(accel_table)

        self.Bind(wx.EVT_MENU,
            lambda e: self.action_panel.on_run(e), id=self.run_id)
        self.Bind(wx.EVT_MENU,
            lambda e: self.console.cmd_input.SetFocus(),
            id=self.console_focus_id)
        self.Bind(wx.EVT_MENU,
            lambda e: self.browser.search_box.SetFocus(),
            id=self.search_focus_id)
        self.Bind(wx.EVT_MENU,
            lambda e: self.on_manual_refresh(),
            id=self.refresh_shortcut_id)
        self.Bind(wx.EVT_MENU,
            lambda e: self.right_notebook.SetSelection(0),
            id=self.inspector_focus_id)

    # ------------------------------------------------------------------
    # State synchronisation — called after every command execution
    # ------------------------------------------------------------------

    def sync_state(self):
        """Read engine state and update all GUI panels."""
        state = self.executor.get_engine_state()
        if not state:
            return

        # Status bar field 0: core engine state
        try:
            buf_sec = state.get('buffer_seconds', 0)
            status = (
                f"BPM: {state.get('bpm', 128):.0f}  |  "
                f"Buffer: {buf_sec:.2f}s  |  "
                f"{state.get('waveform', '---')} @ "
                f"{state.get('frequency', 440):.0f}Hz  |  "
                f"Voices: {state.get('voice_count', 1)} "
                f"({state.get('voice_algorithm', 0)})"
            )
            self.SetStatusText(status, 0)

            # Status bar field 1: filter + FX
            fx_list = state.get('effects', [])
            filter_status = (
                f"Filter: {state.get('filter_type', 'lp')} "
                f"{state.get('filter_cutoff', 4500):.0f}Hz  |  "
                f"FX: {len(fx_list)}"
            )
            self.SetStatusText(filter_status, 1)

            # Status bar field 2: HQ + tracks
            hq_status = (
                f"HQ: {'ON' if state.get('hq_mode') else 'OFF'} "
                f"{state.get('output_format', 'wav').upper()} "
                f"{state.get('output_bit_depth', 16)}bit  |  "
                f"Tracks: {state.get('track_count', 0)}"
            )
            self.SetStatusText(hq_status, 2)
        except Exception:
            pass  # Silently handle missing state keys

        # Rebuild browser tree with fresh data
        self.browser.populate_tree()

        # Update step grid
        self.step_grid.update_from_session()

        # Phase 2: refresh patch builder, oscillator list, routing view
        self.patch_builder.refresh()
        self.osc_list.refresh()
        self.routing_panel.refresh()

    # ------------------------------------------------------------------
    # Auto-refresh timer
    # ------------------------------------------------------------------

    def on_auto_refresh(self, event):
        """Lightweight periodic refresh — updates step grid and status bar
        without rebuilding the entire tree (that happens on command execution).
        """
        state = self.executor.get_engine_state()
        if not state:
            return

        # Update status bar (lightweight — fields 0 and 1)
        try:
            buf_sec = state.get('buffer_seconds', 0)
            self.SetStatusText(
                f"BPM: {state.get('bpm', 128):.0f}  |  "
                f"Buffer: {buf_sec:.2f}s  |  "
                f"{state.get('waveform', '---')} @ "
                f"{state.get('frequency', 440):.0f}Hz  |  "
                f"Voices: {state.get('voice_count', 1)} "
                f"({state.get('voice_algorithm', 0)})", 0)
            fx_list = state.get('effects', [])
            self.SetStatusText(
                f"Filter: {state.get('filter_type', 'lp')} "
                f"{state.get('filter_cutoff', 4500):.0f}Hz  |  "
                f"FX: {len(fx_list)}", 1)
        except Exception:
            pass

    def on_manual_refresh(self):
        """Full refresh triggered by F5."""
        self.sync_state()
        self.console.append("Refreshed.\n", 'info')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_object_select(self, data: dict):
        """Handle object selection from browser.

        Routes selection to both the ActionPanel (for category context)
        and the InspectorPanel (for detailed view).
        """
        obj_type = data.get('type', '')
        obj_id = data.get('id', '')

        # Update action panel category
        if obj_type == 'category':
            # Map tree category IDs to action panel categories
            cat_map = {'tracks': 'engine', 'buffers': 'buffers',
                       'decks': 'engine',
                       'granular_engine': 'granular_engine',
                       'gpu': 'gpu', 'ai_generate': 'ai_generate'}
            action_cat = cat_map.get(obj_id, obj_id)
            if action_cat in ACTIONS:
                self.action_panel.set_category(action_cat)
        elif obj_type in ('track', 'deck'):
            self.action_panel.set_category('engine')
        elif obj_type in ('buffer', 'working_buffer'):
            self.action_panel.set_category('buffers')
        elif obj_type in ('operator', 'envelope_param'):
            self.action_panel.set_category('synth')
        elif obj_type == 'filter_slot':
            self.action_panel.set_category('filter')
        elif obj_type in ('fenv_param',):
            self.action_panel.set_category('filter_envelope')
        elif obj_type in ('voice_prop',):
            self.action_panel.set_category('voice')
        elif obj_type in ('hq_prop',):
            self.action_panel.set_category('hq')
        elif obj_type in ('routing',):
            self.action_panel.set_category('modulation')
        elif obj_type == 'wavetable':
            self.action_panel.set_category('wavetable')
        elif obj_type == 'compound':
            self.action_panel.set_category('compound')
        elif obj_type in ('conv_prop', 'ir_bank_entry'):
            self.action_panel.set_category('convolution')
        elif obj_type == 'lfo_shape':
            self.action_panel.set_category('impulse_lfo')
        elif obj_type == 'imp_env_shape':
            self.action_panel.set_category('impulse_env')
        elif obj_type in ('effect',):
            self.action_panel.set_category('fx')
        elif obj_type in ('sydef', 'user_function', 'chain'):
            self.action_panel.set_category('preset')
        elif obj_type in ('preset_slot',):
            self.action_panel.set_category('preset')
        elif obj_type in ('gen_genre', 'gen_layer', 'gen_xform_preset',
                          'gen_theory_item', 'gen_content_item'):
            self.action_panel.set_category('text_to_audio')
        elif obj_type == 'gen_tta_item':
            self.action_panel.set_category('text_to_audio')
        elif obj_type == 'gen_breed_item':
            self.action_panel.set_category('breeding')
        elif obj_type == 'gen_section':
            section = data.get('section', '')
            sec_map = {'beat': 'text_to_audio', 'loop': 'text_to_audio',
                       'xform': 'text_to_audio', 'theory': 'text_to_audio',
                       'gen2': 'text_to_audio', 'text_to_audio': 'text_to_audio',
                       'breeding': 'breeding'}
            cat = sec_map.get(section, 'text_to_audio')
            if cat in ACTIONS:
                self.action_panel.set_category(cat)

        # Update inspector with full object details
        self.inspector.inspect(data)

        # Switch to inspector tab when an object is selected
        if obj_type not in ('category', 'group', 'engine_prop'):
            self.right_notebook.SetSelection(0)

    def console_append(self, text: str, style: str = 'stdout'):
        """Append text to console (callback for child panels)."""
        self.console.append(text, style)

    def _exec_menu_cmd(self, command: str):
        """Execute a command from a menu item."""
        self.console_append(f">>> {command}\n", 'command')
        stdout, stderr, ok = self.executor.execute(command)
        if stdout:
            self.console_append(stdout, 'stdout')
        if stderr:
            self.console_append(stderr, 'stderr')
        self.console_append(
            f"[{'OK' if ok else 'ERROR'}]\n\n",
            'success' if ok else 'error')
        self.sync_state()

    # ------------------------------------------------------------------
    # Menu handlers
    # ------------------------------------------------------------------

    def on_new_session(self, event):
        """Reset the engine to a fresh session."""
        self.executor._init_engine()
        self.console.append("Session reset.\n", 'info')
        self.sync_state()

    def on_open_project(self, event):
        """Open a project via file dialog."""
        dlg = wx.FileDialog(self, "Open MDMA Project",
                            wildcard="MDMA files (*.mdma)|*.mdma|"
                                     "All files (*.*)|*.*",
                            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            name = os.path.splitext(os.path.basename(path))[0]
            self._exec_menu_cmd(f"/load {name}")
        dlg.Destroy()

    def on_save_project(self, event):
        """Save the current project via dialog."""
        dlg = wx.TextEntryDialog(self, "Project name:",
                                  "Save Project", "myproject")
        if dlg.ShowModal() == wx.ID_OK:
            name = dlg.GetValue().strip()
            if name:
                self._exec_menu_cmd(f"/save {name}")
        dlg.Destroy()

    def on_exit(self, event):
        """Handle exit."""
        self.Close()

    def on_close(self, event):
        """Clean up timers on close."""
        self.refresh_timer.Stop()
        event.Skip()

    def on_about(self, event):
        """Show about dialog."""
        info = wx.adv.AboutDialogInfo()
        info.SetName("MDMA GUI")
        info.SetVersion(self.VERSION)
        info.SetDescription(
            "Music Design Made Accessible\n\n"
            "Phase 1: Core Interface & Workflow\n"
            "Phase 2: Monolith Engine & Synthesis Expansion\n"
            "Phase 3: Modulation, Impulse & Convolution\n"
            "Phase A: Accessibility Audit & Engine Parity\n\n"
            "Full 5-type modulation (FM/TFM/AM/RM/PM),\n"
            "granular engine, GPU/AI generation,\n"
            "command input, keyboard context menus.")
        info.SetCopyright("(C) 2026")
        wx.adv.AboutBox(info)


# ============================================================================
# APPLICATION
# ============================================================================

class MDMAApp(wx.App):
    """Main application."""
    
    def OnInit(self):
        frame = MDMAFrame()
        frame.Show()
        return True


def main():
    """Entry point."""
    app = MDMAApp()
    app.MainLoop()


if __name__ == '__main__':
    main()
