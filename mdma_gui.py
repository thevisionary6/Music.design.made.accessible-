#!/usr/bin/env python3
"""
MDMA GUI - Phase 3: Modulation, Impulse & Convolution
==========================================================

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

Version: 3.0.0
Author: Based on spec by Cyrus
Date: 2026-02-11

Requirements:
    pip install wxPython

Usage:
    python mdma_gui.py

BUILD ID: mdma_gui_v2.0.0_phase2
"""

import sys
import os

# wxPython import guard — give a clear message instead of a traceback
try:
    import wx
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
            name='clearalg',
            label='Clear All Routings',
            command_template='/clearalg',
            params=[],
            description='Clear all modulation routings'
        ),
        ActionDef(
            name='ar_interval',
            label='Interval LFO',
            command_template='/audiorate interval {op} lfo {rate} {depth} {wave}',
            params=[
                ActionParam('op', 'Operator', 'int', 1, min_val=1, max_val=16),
                ActionParam('rate', 'Rate (Hz)', 'float', 5.0, min_val=0.1, max_val=100),
                ActionParam('depth', 'Depth (semitones)', 'float', 1.0, min_val=0, max_val=24),
                ActionParam('wave', 'LFO Shape', 'enum', 'sine',
                           choices=['sine', 'triangle', 'saw', 'square']),
            ],
            description='Set audio-rate interval LFO modulation'
        ),
        ActionDef(
            name='ar_filter',
            label='Filter LFO',
            command_template='/audiorate filter lfo {rate} {depth}',
            params=[
                ActionParam('rate', 'Rate (Hz)', 'float', 2.0, min_val=0.1, max_val=100),
                ActionParam('depth', 'Depth (octaves)', 'float', 1.0, min_val=0, max_val=8),
            ],
            description='Set audio-rate filter cutoff LFO'
        ),
        ActionDef(
            name='ar_clear',
            label='Clear Audio-Rate Mod',
            command_template='/audiorate clear',
            params=[],
            description='Clear all audio-rate modulation sources'
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
                           choices=['reverb', 'delay', 'chorus', 'distortion', 'phaser',
                                   'flanger', 'compressor', 'eq', 'bitcrush']),
            ],
            description='Add an effect to the chain'
        ),
        ActionDef(
            name='fx_clear',
            label='Clear Effects',
            command_template='/fxc',
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

        # Effects
        effects = []
        for i, fx in enumerate(getattr(s, 'effects', [])):
            fx_name = fx[0] if isinstance(fx, (list, tuple)) else str(fx)
            effects.append({'label': fx_name, 'index': i, 'type': 'effect'})

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
            for i, (algo_type, src, tgt, amt) in enumerate(s.engine.algorithms):
                routings.append({
                    'label': f"{algo_type}: Op{src} → Op{tgt} ({amt:.1f})",
                    'index': i, 'type': 'routing',
                    'algo_type': algo_type, 'src': src, 'tgt': tgt, 'amt': amt,
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
        self.search_box.SetBackgroundColour(Theme.BG_INPUT)
        self.search_box.SetForegroundColour(Theme.FG_TEXT)
        self.search_box.Bind(wx.EVT_TEXT, self.on_search)

        search_sizer.Add(search_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        search_sizer.Add(self.search_box, 1, wx.EXPAND)

        sizer.Add(search_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Tree control
        self.tree = wx.TreeCtrl(self, style=wx.TR_DEFAULT_STYLE | wx.TR_HIDE_ROOT,
                                name="ObjectTree")
        self.tree.SetBackgroundColour(Theme.BG_INPUT)
        self.tree.SetForegroundColour(Theme.FG_TEXT)
        self.tree.Bind(wx.EVT_TREE_SEL_CHANGED, self.on_tree_select)
        self.tree.Bind(wx.EVT_TREE_ITEM_RIGHT_CLICK, self.on_context_menu)

        sizer.Add(self.tree, 1, wx.EXPAND | wx.ALL, 5)

        # Refresh button
        refresh_btn = wx.Button(self, label="Refresh (F5)")
        refresh_btn.Bind(wx.EVT_BUTTON, self.on_refresh)
        sizer.Add(refresh_btn, 0, wx.EXPAND | wx.ALL, 5)

        self.SetSizer(sizer)
        self.category_items: Dict[str, Any] = {}
        self.populate_tree()

    # ------------------------------------------------------------------
    # Tree population
    # ------------------------------------------------------------------

    def populate_tree(self):
        """Build the full object tree from live session data."""
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
            self.tree.SetItemData(sub, {'type': 'effect', 'index': ei.get('index', 0),
                                         'id': 'fx'})

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

        self.tree.ExpandAll()

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
                lambda e, i=idx: self._exec(f'/tclr {i+1}'), id=m_clear)

        elif obj_type == 'buffer':
            idx = data.get('index', 1)
            m_play = wx.NewIdRef()
            m_to_work = wx.NewIdRef()
            m_fx = wx.NewIdRef()
            m_clear = wx.NewIdRef()
            menu.Append(m_play, f"Play Buffer {idx}")
            menu.Append(m_to_work, "Copy to Working Buffer")
            menu.AppendSeparator()
            menu.Append(m_fx, "Apply Effect...")
            menu.AppendSeparator()
            menu.Append(m_clear, "Clear Buffer")

            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/pb {i}'), id=m_play)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/bu {i}'), id=m_to_work)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._show_fx_picker('buffer', i), id=m_fx)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/clr {i}'), id=m_clear)

        elif obj_type == 'working_buffer':
            m_play = wx.NewIdRef()
            m_fx = wx.NewIdRef()
            m_clear = wx.NewIdRef()
            menu.Append(m_play, "Play Working Buffer")
            menu.AppendSeparator()
            menu.Append(m_fx, "Apply Effect...")
            menu.AppendSeparator()
            menu.Append(m_clear, "Clear Working Buffer")

            self.Bind(wx.EVT_MENU, lambda e: self._exec('/p'), id=m_play)
            self.Bind(wx.EVT_MENU,
                lambda e: self._show_fx_picker('working', 0), id=m_fx)
            self.Bind(wx.EVT_MENU, lambda e: self._exec('/wbc'), id=m_clear)

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
                lambda e, i=idx: self._exec(f'/deck {i} clear'), id=m_clear)

        elif obj_type == 'effect':
            idx = data.get('index', 0)
            m_remove = wx.NewIdRef()
            m_clear_all = wx.NewIdRef()
            menu.Append(m_remove, f"Remove This Effect")
            menu.AppendSeparator()
            menu.Append(m_clear_all, "Clear All Effects")

            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/fxr {i}'), id=m_remove)
            self.Bind(wx.EVT_MENU,
                lambda e: self._exec('/fxc'), id=m_clear_all)

        elif obj_type == 'operator':
            idx = data.get('index', 0)
            m_select = wx.NewIdRef()
            m_gen = wx.NewIdRef()
            menu.Append(m_select, f"Select Operator {idx}")
            menu.Append(m_gen, "Generate Tone on This Operator")

            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/op {i}'), id=m_select)
            self.Bind(wx.EVT_MENU,
                lambda e, i=idx: self._exec(f'/op {i}\n/tone 440 1'),
                id=m_gen)

        elif obj_type == 'sydef':
            name = data.get('name', '')
            m_use = wx.NewIdRef()
            menu.Append(m_use, f"Use Preset: {name}")
            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._exec(f'/use {n}'), id=m_use)

        elif obj_type == 'chain':
            name = data.get('name', '')
            m_apply = wx.NewIdRef()
            menu.Append(m_apply, f"Apply Chain: {name}")
            self.Bind(wx.EVT_MENU,
                lambda e, n=name: self._exec(f'/chain {n} apply'), id=m_apply)

        elif obj_type == 'category':
            cat_id = data.get('id', '')
            if cat_id == 'tracks':
                m_new = wx.NewIdRef()
                menu.Append(m_new, "Add New Track")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/new track'), id=m_new)
            elif cat_id == 'fx':
                m_add = wx.NewIdRef()
                m_clear = wx.NewIdRef()
                menu.Append(m_add, "Add Effect...")
                menu.Append(m_clear, "Clear All Effects")
                self.Bind(wx.EVT_MENU,
                    lambda e: self._show_fx_picker('global', 0), id=m_add)
                self.Bind(wx.EVT_MENU,
                    lambda e: self._exec('/fxc'), id=m_clear)

        if menu.GetMenuItemCount() > 0:
            self.PopupMenu(menu)
        menu.Destroy()

    def _exec(self, command: str):
        """Execute a command via the executor and log to console."""
        if self.console_cb:
            self.console_cb(f">>> {command}\n", 'command')
        stdout, stderr, ok = self.executor.execute(command)
        if self.console_cb:
            if stdout:
                self.console_cb(stdout, 'stdout')
            if stderr:
                self.console_cb(stderr, 'stderr')
            status = "OK" if ok else "ERROR"
            self.console_cb(f"[{status}]\n\n", 'success' if ok else 'error')
        self.populate_tree()

    def _show_fx_picker(self, target_type: str, target_id: int):
        """Show a dialog to pick an effect and apply it to a target."""
        effects = ['reverb', 'delay', 'chorus', 'distortion', 'phaser',
                    'flanger', 'compressor', 'eq', 'bitcrush', 'normalize',
                    'compress', 'saturate', 'stereo_wide', 'granular']
        dlg = wx.SingleChoiceDialog(self, "Select an effect to apply:",
                                     "Apply Effect", effects)
        if dlg.ShowModal() == wx.ID_OK:
            fx_name = dlg.GetStringSelection()
            if target_type == 'track':
                self._exec(f'/tsel {target_id+1}\n/fx {fx_name}')
            elif target_type == 'buffer':
                self._exec(f'/bu {target_id}\n/fx {fx_name}')
            elif target_type == 'deck':
                self._exec(f'/deck {target_id}\n/fx {fx_name}')
            else:
                self._exec(f'/fx {fx_name}')
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
        super().__init__(parent)
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
        self.command_preview.SetBackgroundColour(Theme.BG_INPUT)
        self.command_preview.SetForegroundColour(Theme.ACCENT)
        self.sizer.Add(self.command_preview, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        
        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.run_btn = wx.Button(self, label="Run (Ctrl+R)")
        self.run_btn.SetBackgroundColour(Theme.ACCENT)
        self.run_btn.Bind(wx.EVT_BUTTON, self.on_run)
        
        self.copy_btn = wx.Button(self, label="Copy Command")
        self.copy_btn.Bind(wx.EVT_BUTTON, self.on_copy)
        
        self.reset_btn = wx.Button(self, label="Reset Params")
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
        super().__init__(parent)
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

        # Properties list (scrollable)
        self.props_panel = wx.ScrolledWindow(self)
        self.props_panel.SetBackgroundColour(Theme.BG_PANEL)
        self.props_panel.SetScrollRate(0, 20)
        self.props_sizer = wx.FlexGridSizer(cols=2, hgap=12, vgap=6)
        self.props_sizer.AddGrowableCol(1, 1)
        self.props_panel.SetSizer(self.props_sizer)
        self.sizer.Add(self.props_panel, 1, wx.EXPAND | wx.ALL, 5)

        # Quick action buttons (contextual)
        self.action_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self.action_sizer, 0, wx.EXPAND | wx.ALL, 10)

        self.SetSizer(self.sizer)

    def inspect(self, data: dict):
        """Display detail view for the given object data dict."""
        self.current_data = data
        obj_type = data.get('type', '')

        # Clear previous
        self.props_sizer.Clear(True)
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
        else:
            self.title.SetLabel("Inspector")
            self.subtitle.SetLabel(f"Object type: {obj_type}")

        self.props_panel.FitInside()
        self.Layout()

    def _add_prop(self, label: str, value: str):
        """Add a property row to the inspector."""
        lbl = wx.StaticText(self.props_panel, label=label)
        lbl.SetForegroundColour(Theme.FG_DIM)
        lbl.SetMinSize((120, -1))
        val = wx.StaticText(self.props_panel, label=str(value))
        val.SetForegroundColour(Theme.FG_TEXT)
        self.props_sizer.Add(lbl, 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        self.props_sizer.Add(val, 1, wx.EXPAND)

    def _add_action_btn(self, label: str, command: str):
        """Add a quick-action button."""
        btn = wx.Button(self, label=label)
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
        self._add_action_btn("To Working", f"/bu {idx}")
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
        self.title.SetLabel(f"Effect #{idx}")
        self.subtitle.SetLabel("Effect in Chain")

        self._add_prop("Position:", str(idx))

        self._add_action_btn("Remove", f"/fxr {idx}")
        self._add_action_btn("Clear All", "/fxc")

    def _inspect_sydef(self, data):
        name = data.get('name', '')
        self.title.SetLabel(f"SyDef: {name}")
        self.subtitle.SetLabel("Synth Definition Preset")

        self._add_prop("Name:", name)
        self._add_action_btn("Use", f"/use {name}")

    def _inspect_chain(self, data):
        name = data.get('name', '')
        self.title.SetLabel(f"Chain: {name}")
        self.subtitle.SetLabel("Effect Chain")

        self._add_prop("Name:", name)
        self._add_action_btn("Apply", f"/chain {name} apply")

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
                 'preset': 'Presets', 'bank': 'Banks'}
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
                from mdma_rebuild.dsp.convolution import list_descriptors
                descs = list_descriptors()
                self._add_prop("Descriptors:", f"{len(descs)} available")


class StepGridPanel(wx.Panel):
    """Step grid with playback position highlighting and buffer I/O tracking.

    Shows a visual grid representing beats/steps in the current session,
    highlights the active playback position, and indicates buffer write
    positions. Fully keyboard-navigable.
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

    def __init__(self, parent, executor: CommandExecutor):
        super().__init__(parent, name="StepGrid")
        self.executor = executor
        self.SetBackgroundColour(self.COL_GRID_BG)

        self.total_steps = 32
        self.playhead_pos = -1  # -1 = not playing
        self.write_pos = 0
        self.filled_steps: set = set()  # Steps that contain audio
        self.track_fills: Dict[int, set] = {}  # Per-track fill data

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        # Header
        hdr_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.header = wx.StaticText(self, label="Step Grid")
        self.header.SetForegroundColour(Theme.FG_DIM)
        hdr_sizer.Add(self.header, 1, wx.ALIGN_CENTER_VERTICAL)

        # Step count selector
        self.step_choice = wx.Choice(self, choices=['16', '32', '64', '128'])
        self.step_choice.SetSelection(1)  # 32 default
        self.step_choice.Bind(wx.EVT_CHOICE, self.on_step_count_change)
        hdr_sizer.Add(wx.StaticText(self, label="Steps:"), 0,
                       wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        hdr_sizer.Add(self.step_choice, 0, wx.LEFT, 4)

        self.sizer.Add(hdr_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Grid canvas
        self.canvas = wx.Panel(self, name="StepGridCanvas")
        self.canvas.SetBackgroundColour(self.COL_GRID_BG)
        self.canvas.Bind(wx.EVT_PAINT, self.on_paint)
        self.canvas.Bind(wx.EVT_LEFT_DOWN, self.on_grid_click)
        self.canvas.SetMinSize((self.STEPS_PER_ROW * (self.CELL_SIZE + self.CELL_PAD) + 40, 80))
        self.sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)

        # Legend
        legend_sizer = wx.BoxSizer(wx.HORIZONTAL)
        for color, label in [
            (self.COL_FILLED, "Audio"),
            (self.COL_PLAYHEAD, "Playhead"),
            (self.COL_WRITE_POS, "Write Pos"),
            (self.COL_EMPTY, "Empty"),
        ]:
            dot = wx.Panel(self, size=(10, 10))
            dot.SetBackgroundColour(color)
            legend_sizer.Add(dot, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            lbl = wx.StaticText(self, label=label)
            lbl.SetForegroundColour(Theme.FG_DIM)
            legend_sizer.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 3)
        self.sizer.Add(legend_sizer, 0, wx.ALL, 5)

        self.SetSizer(self.sizer)

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

    def on_paint(self, event):
        """Draw the step grid."""
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

            # Determine cell color
            if step == self.playhead_pos:
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

    def on_grid_click(self, event):
        """Handle click on grid cell — set write position."""
        x, y = event.GetPosition()
        margin_left = 30
        cols = self.STEPS_PER_ROW
        cs = self.CELL_SIZE
        pad = self.CELL_PAD

        col = (x - margin_left) // (cs + pad)
        row = y // (cs + pad)
        step = row * cols + col
        if 0 <= step < self.total_steps:
            self.write_pos = step
            self.canvas.Refresh()

    def on_step_count_change(self, event):
        """Handle step count selection change."""
        choices = [16, 32, 64, 128]
        idx = self.step_choice.GetSelection()
        if 0 <= idx < len(choices):
            self.total_steps = choices[idx]
            self.canvas.Refresh()

    def set_playhead(self, position: int):
        """Set the playhead position (step index, -1 = stopped)."""
        self.playhead_pos = position
        self.canvas.Refresh()


class ConsolePanel(wx.Panel):
    """Bottom panel - output console."""
    
    def __init__(self, parent):
        super().__init__(parent)
        
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
        
        # Console text
        self.console = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY | 
                                   wx.TE_RICH2 | wx.HSCROLL)
        self.console.SetBackgroundColour(Theme.BG_DARK)
        self.console.SetForegroundColour(Theme.FG_TEXT)
        
        # Use monospace font
        font = wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.console.SetFont(font)
        
        sizer.Add(self.console, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        
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
        super().__init__(parent)
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
        self.op_list.SetBackgroundColour(Theme.BG_INPUT)
        self.op_list.SetForegroundColour(Theme.FG_TEXT)
        self.op_list.InsertColumn(0, "Op", width=40)
        self.op_list.InsertColumn(1, "Wave", width=120)
        self.op_list.InsertColumn(2, "Freq (Hz)", width=80)
        self.op_list.InsertColumn(3, "Amp", width=60)
        self.op_list.InsertColumn(4, "Parameters", width=250)
        sizer.Add(self.op_list, 1, wx.EXPAND | wx.ALL, 4)

        # Routing section
        route_label = wx.StaticText(self, label="Modulation Routing")
        route_label.SetForegroundColour(Theme.ACCENT)
        sizer.Add(route_label, 0, wx.LEFT | wx.TOP, 8)

        self.route_list = wx.ListCtrl(self, style=wx.LC_REPORT)
        self.route_list.SetBackgroundColour(Theme.BG_INPUT)
        self.route_list.SetForegroundColour(Theme.FG_TEXT)
        self.route_list.InsertColumn(0, "#", width=30)
        self.route_list.InsertColumn(1, "Type", width=60)
        self.route_list.InsertColumn(2, "Source", width=60)
        self.route_list.InsertColumn(3, "Target", width=60)
        self.route_list.InsertColumn(4, "Amount", width=70)
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
        self.param_input.SetBackgroundColour(Theme.BG_INPUT)
        self.param_input.SetForegroundColour(Theme.FG_TEXT)
        self.param_input.SetHint("e.g. /wm supersaw saws=7 spread=0.5")
        self.param_input.Bind(wx.EVT_TEXT_ENTER, self._on_param_enter)
        param_sizer.Add(self.param_input, 1, wx.EXPAND)
        sizer.Add(param_sizer, 0, wx.EXPAND | wx.ALL, 4)

        self.SetSizer(sizer)

    def _exec(self, cmd):
        try:
            stdout, stderr, ok = self.executor.execute(cmd)
            if stdout and stdout.strip():
                self.console_cb(stdout, 'info')
            if stderr and stderr.strip():
                self.console_cb(stderr, 'error')
            self.sync_cb()
            self.refresh()
        except Exception as e:
            self.console_cb(f"ERROR: {e}\n", 'error')

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
        super().__init__(parent)
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
        super().__init__(parent)
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
        self.wave_filter.SetSelection(0)
        self.wave_filter.Bind(wx.EVT_CHOICE, lambda e: self.refresh())
        filter_sizer.Add(self.wave_filter, 0, wx.RIGHT, 8)

        # Add operator button
        add_btn = wx.Button(self, label="+ Add Operator")
        add_btn.SetBackgroundColour(Theme.BG_INPUT)
        add_btn.SetForegroundColour(Theme.SUCCESS)
        add_btn.Bind(wx.EVT_BUTTON, self._on_add_op)
        filter_sizer.Add(add_btn, 0)

        sizer.Add(filter_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        # Scrolled operator cards
        self.scroll = wx.ScrolledWindow(self)
        self.scroll.SetScrollRate(0, 20)
        self.scroll.SetBackgroundColour(Theme.BG_DARK)
        self.scroll_sizer = wx.BoxSizer(wx.VERTICAL)
        self.scroll.SetSizer(self.scroll_sizer)
        sizer.Add(self.scroll, 1, wx.EXPAND | wx.ALL, 4)

        self.SetSizer(sizer)

    def _on_add_op(self, event):
        """Add a new operator."""
        if not self.executor.session:
            return
        engine = self.executor.session.engine
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
        """Create a card panel for one operator."""
        card = wx.Panel(self.scroll)
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
        select_btn.Bind(wx.EVT_BUTTON, lambda e, i=idx: self._select_op(i))
        card_sizer.Add(select_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)

        card.SetSizer(card_sizer)
        return card

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

    VERSION = "3.0.0"

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
        self.console.append(f"MDMA GUI v{self.VERSION} - Phase 3: Modulation & Convolution\n", 'info')
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
                "Right-click for context actions. Ctrl+R to run.\n\n", 'info')
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

        # --- Bottom right: Console ---
        self.console = ConsolePanel(self.bottom_splitter)

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
            lambda e: self.console.console.SetFocus(),
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
        buf_sec = state.get('buffer_seconds', 0)
        status = (
            f"BPM: {state['bpm']:.0f}  |  "
            f"Buffer: {buf_sec:.2f}s  |  "
            f"{state['waveform']} @ {state['frequency']:.0f}Hz  |  "
            f"Voices: {state['voice_count']} ({state['voice_algorithm']})"
        )
        self.SetStatusText(status, 0)

        # Status bar field 1: filter + FX
        filter_status = (
            f"Filter: {state['filter_type']} {state['filter_cutoff']:.0f}Hz  |  "
            f"FX: {len(state['effects'])}"
        )
        self.SetStatusText(filter_status, 1)

        # Status bar field 2: HQ + tracks
        hq_status = (
            f"HQ: {'ON' if state['hq_mode'] else 'OFF'} "
            f"{state['output_format'].upper()} {state['output_bit_depth']}bit  |  "
            f"Tracks: {state['track_count']}"
        )
        self.SetStatusText(hq_status, 2)

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

        # Update status bar only (lightweight)
        buf_sec = state.get('buffer_seconds', 0)
        self.SetStatusText(
            f"BPM: {state['bpm']:.0f}  |  "
            f"Buffer: {buf_sec:.2f}s  |  "
            f"{state['waveform']} @ {state['frequency']:.0f}Hz  |  "
            f"Voices: {state['voice_count']} ({state['voice_algorithm']})",
            0)

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
            # Map new category IDs to action panel categories
            cat_map = {'tracks': 'engine', 'buffers': 'engine',
                       'decks': 'engine'}
            action_cat = cat_map.get(obj_id, obj_id)
            if action_cat in ACTIONS:
                self.action_panel.set_category(action_cat)
        elif obj_type in ('track', 'buffer', 'working_buffer', 'deck'):
            self.action_panel.set_category('engine')
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
            "Phase 3: Modulation, Impulse & Convolution\n\n"
            "Advanced convolution reverb, impulse-to-LFO/envelope,\n"
            "neural IR enhancement, AI descriptor transforms,\n"
            "granular IR tools, 15 semantic descriptors.")
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
