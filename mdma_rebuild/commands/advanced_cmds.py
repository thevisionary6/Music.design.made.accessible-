"""Advanced commands for MDMA rebuild.

This module implements:
- Note/temperament system (/nt, /nts)
- Duration and rest flags (/d, /r)
- Chord mode (/c, /co)
- Repeat blocks (/t, /end)
- User stack system (/=, /0, /<n>, /app)
- Random functions (/rn, /rsd, /rm)
- Function definitions (/define, /end)
- Math operators (/plus, /minus, /tms, /power)
- Random choice (/or)
- Custom wave (/w, /wp)
- Conditionals (/if, /elif, /else, /end)
- Constants system (dot flags, /u.constant)
- User preferences (/up, /upl)
- Live loop mode (/lbm, /tll, /start, /stop, /m)
- Algorithm generation (/g)
- File processing (/norm, /st, /rs, /grain)
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import List, Optional, Any, Callable, Tuple

import numpy as np

from ..core.session import Session


# ============================================================================
# CONSTANTS AND GLOBALS
# ============================================================================

# Equal temperament at 432Hz, MIDI note 60 = middle C
A4_FREQ = 432.0
A4_MIDI = 69  # A4 is MIDI note 69

# User constants directory (Windows: C:\Users\<user>\Documents\MDMAConstants)
def _get_constants_dir() -> Path:
    """Get the user constants directory, creating if needed."""
    if os.name == 'nt':
        docs = Path(os.environ.get('USERPROFILE', '~')) / 'Documents'
    else:
        docs = Path.home() / 'Documents'
    const_dir = docs / 'MDMAConstants'
    const_dir.mkdir(parents=True, exist_ok=True)
    return const_dir

def _get_prefs_path() -> Path:
    """Get the user preferences file path."""
    return _get_constants_dir() / 'preferences.json'


# Check for audio playback library
PLAYBACK_AVAILABLE = False
PLAYBACK_ERROR = None
PLAYBACK_LIB = None  # 'simpleaudio' or 'sounddevice'

try:
    import simpleaudio as sa
    PLAYBACK_AVAILABLE = True
    PLAYBACK_LIB = 'simpleaudio'
except (ImportError, OSError):
    PLAYBACK_ERROR = "simpleaudio not installed. Install with: pip install simpleaudio"

# Fallback to sounddevice if simpleaudio not available
if not PLAYBACK_AVAILABLE:
    try:
        import sounddevice as sd
        PLAYBACK_AVAILABLE = True
        PLAYBACK_LIB = 'sounddevice'
        PLAYBACK_ERROR = None
    except (ImportError, OSError) as e:
        if PLAYBACK_ERROR:
            PLAYBACK_ERROR += f"\nOr install sounddevice: pip install sounddevice\n(Error: {e})"
        else:
            PLAYBACK_ERROR = f"No audio playback library found.\nInstall: pip install simpleaudio\nError: {e}"


# Built-in constants (dot flags)
_sr = 48000
_t = np.linspace(0, 1, _sr, dtype=np.float64)

BUILTIN_CONSTANTS = {
    # Mathematical constants
    'pi': math.pi,
    'tau': math.tau,
    'e': math.e,
    'sqrt2': math.sqrt(2),
    'phi': (1 + math.sqrt(5)) / 2,  # Golden ratio
    
    # Audio constants
    'sr': 48000,  # Default sample rate
    'a4': 432.0,  # A4 frequency
    'c4': 432.0 * (2 ** (-9/12)),  # Middle C frequency (~256.87 Hz at 432)
    
    # Basic waveforms (1 second at sample rate)
    'sin1s': np.sin(2 * np.pi * _t),  # 1Hz sine
    'cos1s': np.cos(2 * np.pi * _t),  # 1Hz cosine
    'saw1s': 2.0 * (_t - np.floor(_t + 0.5)),  # Sawtooth
    'tri1s': 2.0 * np.abs(2.0 * (_t - np.floor(_t + 0.5))) - 1.0,  # Triangle
    'sqr1s': np.sign(np.sin(2 * np.pi * _t)),  # Square
    'noise1s': np.random.randn(_sr).astype(np.float64),  # White noise
    
    # Envelope shapes (normalized 0-1 over 1 second)
    'env_lin': _t,  # Linear ramp up
    'env_exp': 1.0 - np.exp(-5.0 * _t),  # Exponential attack
    'env_log': np.log1p(_t * (math.e - 1)) / math.log(math.e),  # Logarithmic
    'env_cos': 0.5 * (1.0 - np.cos(np.pi * _t)),  # Cosine fade
    'env_perc': np.exp(-8.0 * _t),  # Percussive decay
    
    # Window functions
    'win_hann': 0.5 * (1.0 - np.cos(2 * np.pi * _t)),  # Hann window
    'win_hamm': 0.54 - 0.46 * np.cos(2 * np.pi * _t),  # Hamming window
    'win_black': (0.42 - 0.5 * np.cos(2 * np.pi * _t) + 
                  0.08 * np.cos(4 * np.pi * _t)),  # Blackman window
    
    # LFO shapes (10 cycles in 1 second for modulation)
    'lfo_sin': np.sin(2 * np.pi * 10 * _t),
    'lfo_tri': 2.0 * np.abs(2.0 * (10 * _t - np.floor(10 * _t + 0.5))) - 1.0,
    'lfo_saw': 2.0 * (10 * _t - np.floor(10 * _t + 0.5)),
    
    # Useful short arrays
    'zero': np.zeros(1024, dtype=np.float64),
    'one': np.ones(1024, dtype=np.float64),
    'impulse': np.concatenate([[1.0], np.zeros(1023)]),
}

# Wave preset functions (callable to generate specific lengths)
def _make_sine(length: int, freq: float = 1.0, sr: int = 48000) -> np.ndarray:
    t = np.arange(length) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float64)

def _make_saw(length: int, freq: float = 1.0, sr: int = 48000) -> np.ndarray:
    t = np.arange(length) / sr
    phase = freq * t
    return (2.0 * (phase - np.floor(phase + 0.5))).astype(np.float64)

def _make_square(length: int, freq: float = 1.0, sr: int = 48000) -> np.ndarray:
    t = np.arange(length) / sr
    return np.sign(np.sin(2 * np.pi * freq * t)).astype(np.float64)

def _make_triangle(length: int, freq: float = 1.0, sr: int = 48000) -> np.ndarray:
    t = np.arange(length) / sr
    phase = freq * t
    return (2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0).astype(np.float64)

def _make_noise(length: int) -> np.ndarray:
    return np.random.randn(length).astype(np.float64)

WAVE_PRESETS = {
    'sine': _make_sine,
    'sin': _make_sine,
    'saw': _make_saw,
    'sawtooth': _make_saw,
    'square': _make_square,
    'sqr': _make_square,
    'triangle': _make_triangle,
    'tri': _make_triangle,
    'noise': _make_noise,
    'white': _make_noise,
}


# ============================================================================
# SESSION EXTENSIONS - Add these attributes to Session
# ============================================================================

def _ensure_advanced_attrs(session: Session) -> None:
    """Ensure session has all advanced attributes initialized."""
    # Note/duration state
    if not hasattr(session, 'note_duration'):
        session.note_duration = 1.0  # beats
    if not hasattr(session, 'rest_duration'):
        session.rest_duration = 0.0  # beats
    if not hasattr(session, 'chord_mode'):
        session.chord_mode = False
    if not hasattr(session, 'chord_notes'):
        session.chord_notes = []

    # Note buffer / mode attributes
    # When note_mode is True, note commands accumulate into a buffer rather than rendering immediately.
    if not hasattr(session, 'note_mode'):
        session.note_mode = False
    if not hasattr(session, 'note_buffer'):
        session.note_buffer = []
    # Default root for interval interpretation (MIDI note number). Defaults to C5 (midi 72)
    if not hasattr(session, 'note_root_midi'):
        try:
            session.note_root_midi = parse_note('C5')
        except Exception:
            # Fallback to 72 if parse_note isn't available yet
            session.note_root_midi = 72
    
    # User stack
    if not hasattr(session, 'user_stack'):
        session.user_stack = []
    
    # Random state
    if not hasattr(session, 'random_mode'):
        session.random_mode = 'off'  # 'off', 'incremental', 'fixed'
    if not hasattr(session, 'random_seed'):
        session.random_seed = 42
    if not hasattr(session, 'random_counter'):
        session.random_counter = 0
    
    # Repeat/block state
    if not hasattr(session, 'block_stack'):
        session.block_stack = []  # Stack of (type, data) for nested blocks
    if not hasattr(session, 'recording_block'):
        session.recording_block = False
    if not hasattr(session, 'block_commands'):
        session.block_commands = []
    
    # Function definitions
    if not hasattr(session, 'user_functions'):
        session.user_functions = {}
    if not hasattr(session, 'defining_function'):
        session.defining_function = None
    if not hasattr(session, 'function_commands'):
        session.function_commands = []
    
    # Command executor for repeat blocks and other features
    # This gets set by the main REPL loop but we initialize it to None here
    if not hasattr(session, 'command_executor'):
        session.command_executor = None
    
    # User constants (loaded from disk)
    if not hasattr(session, 'user_constants'):
        session.user_constants = _load_user_constants()
    
    # Custom waves
    if not hasattr(session, 'custom_waves'):
        session.custom_waves = {}
    if not hasattr(session, 'current_wave_name'):
        session.current_wave_name = None
    
    # Live loop state
    if not hasattr(session, 'live_loop_mode'):
        session.live_loop_mode = False
    if not hasattr(session, 'live_loops'):
        session.live_loops = {}  # index -> {'buffer': np.array, 'playing': bool, 'mutations': []}
    
    # Random choice queue (for /or)
    if not hasattr(session, 'or_choices'):
        session.or_choices = []


def _load_user_constants() -> dict:
    """Load user constants from disk."""
    const_file = _get_constants_dir() / 'constants.json'
    if const_file.exists():
        try:
            with open(const_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_user_constants(constants: dict) -> None:
    """Save user constants to disk."""
    const_file = _get_constants_dir() / 'constants.json'
    try:
        with open(const_file, 'w') as f:
            json.dump(constants, f, indent=2)
    except Exception:
        pass


def _load_user_prefs() -> dict:
    """Load user preferences from disk."""
    prefs_file = _get_prefs_path()
    if prefs_file.exists():
        try:
            with open(prefs_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_user_prefs(prefs: dict) -> None:
    """Save user preferences to disk."""
    prefs_file = _get_prefs_path()
    try:
        with open(prefs_file, 'w') as f:
            json.dump(prefs, f, indent=2)
    except Exception:
        pass


# ============================================================================
# NOTE/TEMPERAMENT SYSTEM
# ============================================================================

def midi_to_freq(midi_note: int, root_freq: float = 432.0) -> float:
    """Convert MIDI note number to frequency using equal temperament.
    
    Parameters
    ----------
    midi_note : int
        MIDI note number (60 = middle C)
    root_freq : float
        Frequency of A4 (default 432Hz)
    
    Returns
    -------
    float
        Frequency in Hz
    """
    # MIDI note 69 = A4
    return root_freq * (2 ** ((midi_note - A4_MIDI) / 12.0))


def parse_note(note_str: str) -> int:
    """Parse note string to MIDI number.
    
    Accepts:
    - Integer MIDI number: "60", "72"
    - Note name: "C4", "A#3", "Bb5", "C", "D#"
    
    Returns MIDI note number.
    """
    note_str = note_str.strip()
    
    # Try as integer first
    try:
        return int(note_str)
    except ValueError:
        pass
    
    # Parse note name
    note_names = {
        'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11
    }
    
    if not note_str:
        raise ValueError("empty note string")
    
    # Extract note letter
    note_char = note_str[0].upper()
    if note_char not in note_names:
        raise ValueError(f"invalid note: {note_str}")
    
    semitone = note_names[note_char]
    rest = note_str[1:]
    
    # Handle accidentals
    while rest and rest[0] in '#b':
        if rest[0] == '#':
            semitone += 1
        else:
            semitone -= 1
        rest = rest[1:]
    
    # Handle octave (default to 4 if not specified)
    if rest:
        try:
            octave = int(rest)
        except ValueError:
            raise ValueError(f"invalid octave in: {note_str}")
    else:
        octave = 4
    
    # MIDI note: C4 = 60
    return 12 * (octave + 1) + semitone


def cmd_nt(session: Session, args: List[str]) -> str:
    """Play a note in equal temperament rooted at 432Hz.
    
    Usage:
      /nt <note>           -> Play note (MIDI number or name like C4, A#3)
      /nt <note> <beats>   -> Play note for specified beats
      /nt <note> <beats> <amp> -> Play with amplitude
    
    MIDI 60 = Middle C (~256.87 Hz at 432Hz tuning)
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        return "ERROR: /nt requires a note (e.g., /nt 60, /nt C4, /nt A#3)"
    
    try:
        # In note mode we accumulate notes into buffer instead of rendering immediately
        if getattr(session, 'note_mode', False):
            midi_note = parse_note(args[0])
            beats = session.note_duration
            if len(args) > 1:
                beats = float(args[1])
            amp = 1.0
            if len(args) > 2:
                amp = float(args[2])
            # Append to note buffer; rest uses session.rest_duration for now
            session.note_buffer.append((midi_note, beats, session.rest_duration, amp))
            return f"NOTE: added {args[0]} to buffer ({beats} beats)"
        else:
            midi_note = parse_note(args[0])
            freq = midi_to_freq(midi_note)
            beats = session.note_duration
            if len(args) > 1:
                beats = float(args[1])
            amp = 1.0
            if len(args) > 2:
                amp = float(args[2])
            # Handle chord mode
            if session.chord_mode:
                session.chord_notes.append((freq, beats, amp))
                return f"CHORD: added {args[0]} ({freq:.2f}Hz) - {len(session.chord_notes)} notes"
            session.generate_tone(freq, beats, amp)
            return f"OK: {args[0]} = {freq:.2f}Hz for {beats} beats"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_nts(session: Session, args: List[str]) -> str:
    """Play note sequence and insert as clip with pattern.
    
    Usage:
      /nts <notes> [d:<durations>] [r:<rests>]
      
    Examples:
      /nts C4,E4,G4                    -> Play C major arpeggio with default duration
      /nts C4,E4,G4 d:0.5,0.5,1.0      -> With specific durations per note
      /nts C4,E4,G4 d:0.5 r:0.25       -> Uniform duration and rest
      /nts 60,64,67 d:0.25,0.5,0.25    -> Using MIDI numbers
    
    If a clip is selected, its audio is used as the source for pitch shifting.
    Otherwise, generates tones using current synth settings.
    
    Flags:
      d:<val> or d:<val1,val2,...>  - Duration(s) in beats
      r:<val> or r:<val1,val2,...>  - Rest(s) between notes in beats
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        return ("ERROR: /nts requires notes\n"
                "Usage: /nts <notes> [d:<durations>] [r:<rests>]\n"
                "Example: /nts C4,E4,G4 d:0.5 r:0.1")
    
    try:
        # Parse notes (comma-separated)
        notes_str = args[0]
        note_strs = [n.strip() for n in notes_str.split(',')]
        midi_notes = [parse_note(n) for n in note_strs]
        
        # Parse flags
        durations = [session.note_duration] * len(midi_notes)
        rests = [session.rest_duration] * len(midi_notes)
        
        for arg in args[1:]:
            if arg.lower().startswith('d:'):
                dur_str = arg[2:]
                if ',' in dur_str:
                    dur_vals = [float(d) for d in dur_str.split(',')]
                    # Extend or truncate to match note count
                    while len(dur_vals) < len(midi_notes):
                        dur_vals.append(dur_vals[-1])
                    durations = dur_vals[:len(midi_notes)]
                else:
                    durations = [float(dur_str)] * len(midi_notes)
            elif arg.lower().startswith('r:'):
                rest_str = arg[2:]
                if ',' in rest_str:
                    rest_vals = [float(r) for r in rest_str.split(',')]
                    while len(rest_vals) < len(midi_notes):
                        rest_vals.append(rest_vals[-1])
                    rests = rest_vals[:len(midi_notes)]
                else:
                    rests = [float(rest_str)] * len(midi_notes)
        
        # Check if we have a source clip to pitch-shift
        use_clip_audio = False
        source_buffer = None
        source_freq = 440.0  # Reference frequency for pitch shifting
        
        if session.current_clip and session.current_clip in session.clips:
            source_buffer = session.clips[session.current_clip].copy()
            use_clip_audio = True
            # Assume source is at A4 unless we know otherwise
            source_freq = midi_to_freq(69)  # A4
        
        # Calculate total output length
        total_beats = sum(durations) + sum(rests[:-1])  # No rest after last note
        total_sec = total_beats * 60.0 / session.bpm
        total_samples = int(total_sec * session.sample_rate)
        
        # Build output buffer
        output = np.zeros(total_samples, dtype=np.float64)
        position = 0
        
        for i, midi_note in enumerate(midi_notes):
            freq = midi_to_freq(midi_note)
            dur = durations[i]
            rest = rests[i] if i < len(midi_notes) - 1 else 0
            
            dur_samples = int(dur * 60.0 / session.bpm * session.sample_rate)
            
            if use_clip_audio and source_buffer is not None:
                # Pitch shift the source audio
                ratio = source_freq / freq
                new_len = int(len(source_buffer) * ratio)
                if new_len > 0:
                    x_old = np.linspace(0, 1, len(source_buffer))
                    x_new = np.linspace(0, 1, new_len)
                    shifted = np.interp(x_new, x_old, source_buffer)
                    
                    # Truncate or tile to fit duration
                    if len(shifted) >= dur_samples:
                        note_buf = shifted[:dur_samples]
                    else:
                        reps = (dur_samples // len(shifted)) + 1
                        note_buf = np.tile(shifted, reps)[:dur_samples]
                else:
                    note_buf = np.zeros(dur_samples)
            else:
                # Generate tone using synth
                session.generate_tone(freq, dur, 1.0)
                note_buf = session.last_buffer[:dur_samples] if len(session.last_buffer) >= dur_samples else session.last_buffer
            
            # Apply micro fade to prevent clicks
            fade_samples = min(int(0.005 * session.sample_rate), len(note_buf) // 4)
            if fade_samples > 0:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                note_buf[:fade_samples] *= fade_in
                note_buf[-fade_samples:] *= fade_out
            
            # Insert into output
            end_pos = min(position + len(note_buf), len(output))
            output[position:end_pos] += note_buf[:end_pos - position]
            
            # Advance position (note + rest)
            rest_samples = int(rest * 60.0 / session.bpm * session.sample_rate)
            position += dur_samples + rest_samples
        
        # Normalize if needed
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val
        
        session.last_buffer = output
        
        # Create clip and insert to track
        clip_name = f"seq_{session.clip_count}"
        session.clip_count += 1
        session.clips[clip_name] = output.copy()
        session.current_clip = clip_name
        
        # Write into current track at its cursor (continuous track workflow)
        if getattr(session, 'tracks', None):
            try:
                session.write_to_track(output, mode='overwrite')
            except Exception:
                pass
        
        source_info = " (using clip audio)" if use_clip_audio else ""
        return f"OK: sequence '{clip_name}' with {len(midi_notes)} notes{source_info}"
    except Exception as e:
        return (f"ERROR: {e}\n"
                "Usage: /nts <notes> [d:<durations>] [r:<rests>]\n"
                "Example: /nts C4,E4,G4 d:0.5,0.5,1.0 r:0.1")


# ==========================================================================
# NOTE BUFFER SEQUENCE HELPERS AND COMMANDS
# ==========================================================================

def _parse_sequence_tokens(args: List[str]) -> Tuple[List[str], List[str], List[bool]]:
    """
    Helper to split arguments into note tokens, flag tokens and interval flags.

    Notes may be provided as comma-separated values or separate args. Flags start with d: or r:.
    A note token consisting solely of digits that starts with '0' and has length > 1 (e.g. '037')
    is interpreted as a sequence of interval digits. Each digit is split into its own token and
    marked as an interval (True in the interval_flags list). All other tokens are marked False.

    Returns:
        note_tokens (List[str]): flattened list of note/interval tokens
        flag_tokens (List[str]): list of flag strings (duration/rest)
        interval_flags (List[bool]): boolean list indicating whether each note token is an interval
    """
    note_parts: List[str] = []
    flags: List[str] = []
    interval_flags: List[bool] = []
    for arg in args:
        low = arg.lower()
        if low.startswith('d:') or low.startswith('r:'):
            flags.append(arg)
        else:
            # may contain comma-separated notes
            for part in arg.split(','):
                part = part.strip()
                if not part:
                    continue
                # If the part is a string of digits starting with '0' and length>1, treat each digit as an interval
                if part.isdigit() and len(part) > 1 and part.startswith('0'):
                    for ch in part:
                        note_parts.append(ch)
                        interval_flags.append(True)
                else:
                    note_parts.append(part)
                    interval_flags.append(False)
    return note_parts, flags, interval_flags


def _expand_interval_token(token: str) -> List[str]:
    """
    Expand a token like '037' into ['0','3','7'] if it is all digits and length>1
    and starts with '0'. Otherwise return [token].
    """
    if token.isdigit() and len(token) > 1 and token.startswith('0'):
        return list(token)
    return [token]


def _freq_to_midi(freq: float) -> int:
    """
    Convert a frequency in Hz to a MIDI note number using the same reference
    frequency as midi_to_freq (A4_FREQ). Returns an integer MIDI note.
    """
    try:
        # Avoid math domain errors
        if freq <= 0:
            return int(parse_note('C5'))
        midi_val = A4_MIDI + 12 * math.log(freq / A4_FREQ, 2)
        return int(round(midi_val))
    except Exception:
        # Fallback to C5
        try:
            return parse_note('C5')
        except Exception:
            return 72


def cmd_nbr(session: Session, args: List[str]) -> str:
    """
    Show or set the interval root for digit‐based note sequences.

    When interpreting continuous digit strings like ``037`` as interval offsets, a
    "root" MIDI note is used as the base. By default this root is C5 (MIDI 72).
    You can change the root using this command.

    Usage:
        /nbr                     -> display current root and its frequency
        /nbr <value>             -> set root based on a note name or frequency
        /nbr root <value>        -> same as above; explicit keyword for clarity
        /nbr note <note_name>    -> set root using a note name (e.g., C4, A#3)
        /nbr hz <frequency>      -> set root using a frequency in Hz (e.g., 440)

    Examples:
        /nbr C4           # set root to note C4
        /nbr 440          # set root to A4 (from frequency in Hz)
        /nbr note D#5     # set root explicitly using note keyword
        /nbr hz 261.63    # set root using frequency (middle C)
    """
    _ensure_advanced_attrs(session)
    # Show current root if no arguments provided
    if not args:
        try:
            root_freq = midi_to_freq(int(session.note_root_midi))
        except Exception:
            root_freq = 0.0
        return f"ROOT: MIDI {int(session.note_root_midi)} ({root_freq:.2f} Hz)"

    # Support optional keywords "root", "note", "hz" for clarity
    keyword = None
    value_index = 0
    if args:
        first = args[0].lower()
        # If the first token is a known keyword, pop it and treat the next as the value
        if first in ("root", "note", "hz"):
            keyword = first
            if len(args) < 2:
                return "ERROR: /nbr requires a value after the keyword"
            value_index = 1
    # Determine value to parse
    val = args[value_index]
    # If keyword explicitly states note or hz, dispatch accordingly
    if keyword == "hz":
        try:
            freq_val = float(val)
        except Exception:
            return f"ERROR: invalid frequency value '{val}'"
        midi_val = _freq_to_midi(freq_val)
        session.note_root_midi = midi_val
        return f"OK: root set to MIDI {midi_val} ({freq_val:.2f} Hz)"
    elif keyword == "note":
        try:
            midi_val = parse_note(val)
        except Exception as e:
            return f"ERROR: invalid note '{val}': {e}"
        session.note_root_midi = midi_val
        freq_val = midi_to_freq(midi_val)
        return f"OK: root set to {val} (MIDI {midi_val}, {freq_val:.2f} Hz)"
    # Otherwise keyword is None or "root"; attempt to parse as frequency first, then note
    try:
        freq_val = float(val)
        midi_val = _freq_to_midi(freq_val)
        session.note_root_midi = midi_val
        return f"OK: root set to MIDI {midi_val} ({freq_val:.2f} Hz)"
    except Exception:
        pass
    try:
        midi_val = parse_note(val)
    except Exception as e:
        return f"ERROR: invalid root '{val}': {e}"
    session.note_root_midi = midi_val
    freq_val = midi_to_freq(midi_val)
    return f"OK: root set to {val} (MIDI {midi_val}, {freq_val:.2f} Hz)"


def cmd_ntm(session: Session, args: List[str]) -> str:
    """
    Toggle or show Note Track Mode.

    When Note Track Mode is on, /nt and /nbs accumulate notes into a buffer instead of rendering immediately.

    Usage:
      /ntm          -> show status (ON/OFF)
      /ntm on       -> enable note mode (clears buffer)
      /ntm off      -> disable note mode
    """
    _ensure_advanced_attrs(session)
    if not hasattr(session, 'note_mode'):
        session.note_mode = False
    if not hasattr(session, 'note_buffer'):
        session.note_buffer = []
    if not args:
        return f"NOTE MODE: {'ON' if session.note_mode else 'OFF'} ({len(session.note_buffer)} notes buffered)"
    sub = args[0].lower()
    if sub in ('on', '1', 'enable', 'start'):
        session.note_mode = True
        session.note_buffer = []
        return "OK: Note mode enabled. Buffer cleared."
    elif sub in ('off', '0', 'disable', 'stop'):
        session.note_mode = False
        return "OK: Note mode disabled."
    return "Usage: /ntm [on|off]"


def cmd_nbs(session: Session, args: List[str]) -> str:
    """
    Add a sequence of notes to the note buffer.

    Usage:
      /nbs <notes> [d:<durations>] [r:<rests>]

    Notes can be comma-separated or space-separated. You can also specify a continuous
    digit string starting with 0 (e.g. 037) which will be expanded to separate note tokens.
    Durations (d:) and rests (r:) accept comma-separated lists or single values.
    Examples:
      /nbs C4 E4 G4 d:0.25 r:0.05
      /nbs C4,E4,G4 d:0.25,0.25,0.5
      /nbs 037 d:0.5  -> expands to 0,3,7

    In note mode (see /ntm on), notes are accumulated and rendered only when /ntrn is called.
    """
    _ensure_advanced_attrs(session)
    if not hasattr(session, 'note_buffer'):
        session.note_buffer = []
    if not args:
        if session.note_buffer:
            return f"BUFFERED NOTES: {len(session.note_buffer)} events"
        return ("ERROR: /nbs requires notes\n"
                "Usage: /nbs <notes> [d:<durations>] [r:<rests>]")
    # Split arguments into note tokens, flags and interval indicators
    note_tokens_raw, flags, interval_flags = _parse_sequence_tokens(args)
    # Convert tokens to MIDI notes, handling intervals relative to root
    midi_notes: List[int] = []
    try:
        for tok, is_interval in zip(note_tokens_raw, interval_flags):
            if is_interval:
                # Interpret token as interval (semitone offset) relative to the session root
                offset = int(tok)
                midi_notes.append(int(session.note_root_midi + offset))
            else:
                midi_notes.append(parse_note(tok))
    except Exception as e:
        return f"ERROR: invalid note '{tok}': {e}"
    # Parse durations and rests
    durations: List[float] = [session.note_duration] * len(midi_notes)
    rests: List[float] = [session.rest_duration] * len(midi_notes)
    for flag in flags:
        low = flag.lower()
        if low.startswith('d:'):
            vals = low[2:]
            if ',' in vals:
                try:
                    vlist = [float(x) for x in vals.split(',') if x]
                except Exception:
                    return f"ERROR: invalid duration values: {vals}"
                while len(vlist) < len(midi_notes):
                    vlist.append(vlist[-1])
                durations = vlist[:len(midi_notes)]
            else:
                try:
                    val = float(vals)
                except Exception:
                    return f"ERROR: invalid duration value: {vals}"
                durations = [val] * len(midi_notes)
        elif low.startswith('r:'):
            vals = low[2:]
            if ',' in vals:
                try:
                    vlist = [float(x) for x in vals.split(',') if x]
                except Exception:
                    return f"ERROR: invalid rest values: {vals}"
                while len(vlist) < len(midi_notes):
                    vlist.append(vlist[-1])
                rests = vlist[:len(midi_notes)]
            else:
                try:
                    val = float(vals)
                except Exception:
                    return f"ERROR: invalid rest value: {vals}"
                rests = [val] * len(midi_notes)
    # Append each note to buffer
    for i, midi_note in enumerate(midi_notes):
        dur = durations[i]
        rest = rests[i] if i < len(midi_notes) - 1 else 0.0
        session.note_buffer.append((midi_note, dur, rest, 1.0))
    return f"OK: added {len(midi_notes)} notes to buffer (total {len(session.note_buffer)})"


def cmd_ntrn(session: Session, args: List[str]) -> str:
    """
    Render the buffered note sequence to audio and insert as a clip.

    Usage:
      /ntrn           -> render current note buffer to audio and clear buffer
      /ntrn keep      -> render but keep buffer contents (for layering)
      /ntrn clear     -> clear buffer without rendering

    Renders using the current synth voice and inserts the resulting clip to the current track.
    """
    _ensure_advanced_attrs(session)
    if not hasattr(session, 'note_buffer'):
        session.note_buffer = []
    if args:
        sub = args[0].lower()
        if sub == 'clear':
            session.note_buffer = []
            return "OK: note buffer cleared"
        elif sub == 'keep':
            keep = True
        else:
            keep = False
    else:
        keep = False
    if not session.note_buffer:
        return "ERROR: note buffer is empty"
    midi_notes = [ev[0] for ev in session.note_buffer]
    durations = [ev[1] for ev in session.note_buffer]
    rests = [ev[2] for ev in session.note_buffer]
    total_beats = sum(durations) + sum(rests[:-1]) if len(rests) > 1 else sum(durations)
    total_sec = total_beats * 60.0 / session.bpm
    total_samples = int(total_sec * session.sample_rate)
    output = np.zeros(total_samples, dtype=np.float64)
    position = 0
    for i, midi_note in enumerate(midi_notes):
        freq = midi_to_freq(midi_note)
        dur = durations[i]
        rest_dur = rests[i] if i < len(midi_notes) - 1 else 0.0
        dur_samples = int(dur * 60.0 / session.bpm * session.sample_rate)
        try:
            session.generate_tone(freq, dur, 1.0)
        except Exception:
            t = np.linspace(0, dur, dur_samples, False)
            tone = np.sin(2 * np.pi * freq * t)
            session.last_buffer = tone
        note_buf = session.last_buffer[:dur_samples] if len(session.last_buffer) >= dur_samples else session.last_buffer
        fade_samples = min(int(0.005 * session.sample_rate), len(note_buf) // 4)
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            note_buf[:fade_samples] *= fade_in
            note_buf[-fade_samples:] *= fade_out
        end_pos = min(position + len(note_buf), len(output))
        output[position:end_pos] += note_buf[:end_pos - position]
        rest_samples = int(rest_dur * 60.0 / session.bpm * session.sample_rate)
        position += dur_samples + rest_samples
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val
    session.last_buffer = output
    clip_name = f"seq_{session.clip_count}"
    session.clip_count += 1
    session.clips[clip_name] = output.copy()
    session.current_clip = clip_name
    if getattr(session, 'tracks', None):
        try:
            session.write_to_track(output, mode='overwrite')
        except Exception:
            pass
    if not keep:
        session.note_buffer = []
    return f"OK: rendered {len(midi_notes)} notes to '{clip_name}'"


# ============================================================================
# DURATION AND REST FLAGS
# ============================================================================

def cmd_d(session: Session, args: List[str]) -> str:
    """Set current note duration in beats.
    
    Usage:
      /d          -> Show current duration
      /d <beats>  -> Set duration for following notes
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        return f"DURATION: {session.note_duration} beats"
    
    try:
        session.note_duration = float(args[0])
        return f"OK: note duration set to {session.note_duration} beats"
    except Exception:
        return "ERROR: invalid duration value"


def cmd_r(session: Session, args: List[str]) -> str:
    """Set current rest duration in beats (silence between notes).
    
    Usage:
      /r          -> Show current rest duration
      /r <beats>  -> Set rest duration
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        return f"REST: {session.rest_duration} beats"
    
    try:
        session.rest_duration = float(args[0])
        return f"OK: rest duration set to {session.rest_duration} beats"
    except Exception:
        return "ERROR: invalid rest value"


# ============================================================================
# CHORD MODE
# ============================================================================

def cmd_c(session: Session, args: List[str]) -> str:
    """Enable chord mode. Notes will be collected until /co."""
    _ensure_advanced_attrs(session)
    session.chord_mode = True
    session.chord_notes = []
    return "CHORD MODE: ON - use /nt to add notes, /co to play chord"


def cmd_co(session: Session, args: List[str]) -> str:
    """Disable chord mode and play collected notes as chord."""
    _ensure_advanced_attrs(session)
    
    if not session.chord_mode:
        return "CHORD MODE: already off"
    
    session.chord_mode = False
    
    if not session.chord_notes:
        return "CHORD MODE: OFF (no notes collected)"
    
    # Mix all notes together
    try:
        # Find max duration
        max_beats = max(n[1] for n in session.chord_notes)
        duration_sec = max_beats * 60.0 / session.bpm
        max_samples = int(duration_sec * session.sample_rate)
        
        mix = np.zeros(max_samples, dtype=np.float64)
        
        for freq, beats, amp in session.chord_notes:
            session.generate_tone(freq, beats, amp)
            buf = session.last_buffer
            mix[:len(buf)] += buf
        
        # Normalize
        max_val = np.max(np.abs(mix))
        if max_val > 1.0:
            mix = mix / max_val
        
        session.last_buffer = mix
        note_count = len(session.chord_notes)
        session.chord_notes = []
        
        return f"CHORD MODE: OFF - played chord with {note_count} notes"
    except Exception as e:
        session.chord_notes = []
        return f"ERROR: chord playback failed - {e}"


# ============================================================================
# USER STACK SYSTEM
# ============================================================================

def cmd_eq(session: Session, args: List[str]) -> str:
    """Push value to user stack (/=).
    
    Usage:
      /= <value>  -> Push value onto stack
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        return f"STACK: {len(session.user_stack)} items - {session.user_stack[-5:]}"
    
    try:
        # Try to parse as number first
        val = args[0]
        try:
            val = float(val)
            if val == int(val):
                val = int(val)
        except ValueError:
            pass  # Keep as string
        
        session.user_stack.append(val)
        return f"PUSH: {val} -> stack[{len(session.user_stack)-1}]"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_0(session: Session, args: List[str]) -> str:
    """Pop top of user stack (/0).
    
    Returns the value that was on top.
    """
    _ensure_advanced_attrs(session)
    
    if not session.user_stack:
        return "ERROR: stack is empty"
    
    val = session.user_stack.pop()
    return f"POP: {val} (stack now has {len(session.user_stack)} items)"


def cmd_stack_get(session: Session, args: List[str], index: int) -> str:
    """Get value at stack index (/<n>).
    
    Index 0 = top, 1 = second from top, etc.
    """
    _ensure_advanced_attrs(session)
    
    if not session.user_stack:
        return "ERROR: stack is empty"
    
    # Convert to actual index (0 = top = last item)
    actual_idx = len(session.user_stack) - 1 - index
    
    if actual_idx < 0 or actual_idx >= len(session.user_stack):
        return f"ERROR: stack index {index} out of range (stack has {len(session.user_stack)} items)"
    
    val = session.user_stack[actual_idx]
    return f"STACK[{index}]: {val}"


def cmd_app(session: Session, args: List[str]) -> str:
    """Append multiple values to stack quickly.
    
    Usage:
      /app <val1> <val2> ...  -> Push all values
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        return "ERROR: /app requires at least one value"
    
    for arg in args:
        try:
            val = float(arg)
            if val == int(val):
                val = int(val)
        except ValueError:
            val = arg
        session.user_stack.append(val)
    
    return f"APPEND: pushed {len(args)} values, stack has {len(session.user_stack)} items"


# ============================================================================
# RANDOM FUNCTIONS
# ============================================================================

def cmd_rn(session: Session, args: List[str]) -> str:
    """Get random integer between min and max inclusive.
    
    Usage:
      /rn <min> <max>  -> Random int in [min, max]
    """
    _ensure_advanced_attrs(session)
    
    if len(args) < 2:
        return "ERROR: /rn requires min and max arguments"
    
    try:
        min_val = int(args[0])
        max_val = int(args[1])
        
        # Handle random mode
        if session.random_mode == 'fixed':
            random.seed(session.random_seed)
        elif session.random_mode == 'incremental':
            random.seed(session.random_seed + session.random_counter)
            session.random_counter += 1
        # 'off' = truly random each time
        
        result = random.randint(min_val, max_val)
        session.user_stack.append(result)
        return f"RANDOM: {result} (pushed to stack)"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_rsd(session: Session, args: List[str]) -> str:
    """Get random seed between 0 and 1.
    
    Usage:
      /rsd  -> Random float in [0, 1), pushed to stack
    """
    _ensure_advanced_attrs(session)
    
    if session.random_mode == 'fixed':
        random.seed(session.random_seed)
    elif session.random_mode == 'incremental':
        random.seed(session.random_seed + session.random_counter)
        session.random_counter += 1
    
    result = random.random()
    session.user_stack.append(result)
    return f"RANDOM SEED: {result:.6f} (pushed to stack)"


def cmd_rmode(session: Session, args: List[str]) -> str:
    """Set random mode for deterministic/non-deterministic generation.
    
    Usage:
      /rmode              -> Show current mode
      /rmode off          -> Truly random each time
      /rmode fixed        -> Same sequence from seed
      /rmode inc          -> Incrementing seed
      /rmode seed <n>     -> Set the seed value
    
    Aliases: /rmode, /randmode
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        return f"RANDOM MODE: {session.random_mode}, seed={session.random_seed}"
    
    mode = args[0].lower()
    
    if mode == 'seed' and len(args) > 1:
        try:
            session.random_seed = int(args[1])
            session.random_counter = 0
            return f"OK: random seed set to {session.random_seed}"
        except ValueError:
            return "ERROR: seed must be an integer"
    
    if mode in ('off', 'fixed', 'inc', 'incremental'):
        session.random_mode = 'incremental' if mode == 'inc' else mode
        session.random_counter = 0
        return f"OK: random mode set to {session.random_mode}"
    
    return "ERROR: mode must be 'off', 'fixed', 'inc', or 'seed <n>'"


def cmd_sm(session: Session, args: List[str]) -> str:
    """Set seed mode by index for quick switching.
    
    Usage:
      /sm         -> Show current seed mode
      /sm 0       -> off (truly random)
      /sm 1       -> fixed (same sequence)
      /sm 2       -> incremental (auto-increment)
      /sm <seed>  -> Set seed directly if > 10
    
    This is a quick version of /rmode for rapid mode switching.
    """
    _ensure_advanced_attrs(session)
    
    modes = {
        0: 'off',
        1: 'fixed',
        2: 'incremental',
    }
    
    if not args:
        mode_num = {v: k for k, v in modes.items()}.get(session.random_mode, '?')
        return f"SEED MODE: {mode_num} ({session.random_mode}), seed={session.random_seed}"
    
    try:
        val = int(args[0])
        if val in modes:
            session.random_mode = modes[val]
            session.random_counter = 0
            return f"OK: seed mode {val} ({modes[val]})"
        elif val > 10:
            # Treat as seed value
            session.random_seed = val
            session.random_counter = 0
            return f"OK: seed set to {val}"
        else:
            return f"ERROR: mode must be 0-2 or seed value > 10"
    except ValueError:
        return "ERROR: argument must be a number (0-2 for mode, >10 for seed)"


# ============================================================================
# MATH OPERATORS
# ============================================================================

def cmd_plus(session: Session, args: List[str]) -> str:
    """Add two values. Uses stack or arguments.
    
    Usage:
      /plus <a> <b>  -> Push a + b to stack
      /plus          -> Pop two from stack, push sum
    """
    _ensure_advanced_attrs(session)
    
    try:
        if len(args) >= 2:
            a, b = float(args[0]), float(args[1])
        elif len(session.user_stack) >= 2:
            b = session.user_stack.pop()
            a = session.user_stack.pop()
            a, b = float(a), float(b)
        else:
            return "ERROR: /plus requires two values (args or stack)"
        
        result = a + b
        session.user_stack.append(result)
        return f"PLUS: {a} + {b} = {result}"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_minus(session: Session, args: List[str]) -> str:
    """Subtract two values.
    
    Usage:
      /minus <a> <b>  -> Push a - b to stack
      /minus          -> Pop b, pop a, push a - b
    """
    _ensure_advanced_attrs(session)
    
    try:
        if len(args) >= 2:
            a, b = float(args[0]), float(args[1])
        elif len(session.user_stack) >= 2:
            b = session.user_stack.pop()
            a = session.user_stack.pop()
            a, b = float(a), float(b)
        else:
            return "ERROR: /minus requires two values"
        
        result = a - b
        session.user_stack.append(result)
        return f"MINUS: {a} - {b} = {result}"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_tms(session: Session, args: List[str]) -> str:
    """Multiply two values.
    
    Usage:
      /tms <a> <b>  -> Push a * b to stack
      /tms          -> Pop two, push product
    """
    _ensure_advanced_attrs(session)
    
    try:
        if len(args) >= 2:
            a, b = float(args[0]), float(args[1])
        elif len(session.user_stack) >= 2:
            b = session.user_stack.pop()
            a = session.user_stack.pop()
            a, b = float(a), float(b)
        else:
            return "ERROR: /tms requires two values"
        
        result = a * b
        session.user_stack.append(result)
        return f"TIMES: {a} * {b} = {result}"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_power(session: Session, args: List[str]) -> str:
    """Raise base to exponent.
    
    Usage:
      /power <base> <exp>  -> Push base^exp to stack
      /power               -> Pop exp, pop base, push result
    """
    _ensure_advanced_attrs(session)
    
    try:
        if len(args) >= 2:
            base, exp = float(args[0]), float(args[1])
        elif len(session.user_stack) >= 2:
            exp = session.user_stack.pop()
            base = session.user_stack.pop()
            base, exp = float(base), float(exp)
        else:
            return "ERROR: /power requires base and exponent"
        
        result = base ** exp
        session.user_stack.append(result)
        return f"POWER: {base}^{exp} = {result}"
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================================
# CONSTANTS SYSTEM
# ============================================================================

def cmd_dot(session: Session, args: List[str]) -> str:
    """Get a built-in or user constant.
    
    Usage:
      /. <name>     -> Push constant value to stack
      /. list       -> List available constants
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        return "ERROR: /. requires constant name"
    
    name = args[0].lower()
    
    if name == 'list':
        builtin_names = list(BUILTIN_CONSTANTS.keys())
        user_names = list(session.user_constants.keys())
        return f"CONSTANTS:\n  Built-in: {', '.join(builtin_names)}\n  User: {', '.join(user_names) or '(none)'}"
    
    # Check built-in first, then user
    if name in BUILTIN_CONSTANTS:
        val = BUILTIN_CONSTANTS[name]
        if isinstance(val, np.ndarray):
            session.user_stack.append(val)
            return f"CONST .{name}: <array len={len(val)}> (pushed to stack)"
        session.user_stack.append(val)
        return f"CONST .{name}: {val}"
    
    if name in session.user_constants:
        val = session.user_constants[name]
        session.user_stack.append(val)
        return f"CONST .{name}: {val}"
    
    return f"ERROR: unknown constant '{name}'"


def cmd_u_constant(session: Session, args: List[str]) -> str:
    """Define a user constant (saved to disk, immutable).
    
    Usage:
      /u.constant <name> <value>  -> Define constant
    
    Constants are saved to ~/Documents/MDMAConstants/constants.json
    """
    _ensure_advanced_attrs(session)
    
    if len(args) < 2:
        return "ERROR: /u.constant requires name and value"
    
    name = args[0].lower()
    
    # Check if already exists (immutable)
    if name in session.user_constants:
        return f"ERROR: constant '{name}' already exists and is immutable"
    
    if name in BUILTIN_CONSTANTS:
        return f"ERROR: '{name}' is a built-in constant"
    
    # Parse value
    try:
        val = ' '.join(args[1:])
        try:
            val = float(val)
            if val == int(val):
                val = int(val)
        except ValueError:
            pass  # Keep as string
        
        session.user_constants[name] = val
        _save_user_constants(session.user_constants)
        return f"OK: constant '{name}' = {val} (saved to disk)"
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================================
# USER PREFERENCES
# ============================================================================

def cmd_up(session: Session, args: List[str]) -> str:
    """Update default settings to user preferences.
    
    Usage:
      /up              -> Save current session settings as defaults
      /up load         -> Load saved preferences
      /up show         -> Show current preferences
      /up reset        -> Reset to factory defaults
    """
    _ensure_advanced_attrs(session)
    
    if not args or args[0] == 'save':
        # Save current settings
        prefs = {
            'bpm': session.bpm,
            'step': session.step,
            'attack': session.attack,
            'decay': session.decay,
            'sustain': session.sustain,
            'release': session.release,
            'filter_count': session.filter_count,
            'voice_count': session.voice_count,
            'carrier_count': session.carrier_count,
            'autoplay': session.autoplay,
            'note_duration': getattr(session, 'note_duration', 1.0),
            'rest_duration': getattr(session, 'rest_duration', 0.0),
            'random_mode': getattr(session, 'random_mode', 'off'),
            'random_seed': getattr(session, 'random_seed', 42),
        }
        _save_user_prefs(prefs)
        return f"OK: preferences saved to {_get_prefs_path()}"
    
    if args[0] == 'load':
        prefs = _load_user_prefs()
        if not prefs:
            return "No saved preferences found"
        
        for key, val in prefs.items():
            if hasattr(session, key):
                setattr(session, key, val)
        return f"OK: loaded preferences ({len(prefs)} settings)"
    
    if args[0] == 'show':
        prefs = _load_user_prefs()
        if not prefs:
            return "No saved preferences"
        lines = ["Saved preferences:"]
        for k, v in prefs.items():
            lines.append(f"  {k}: {v}")
        return '\n'.join(lines)
    
    if args[0] == 'reset':
        prefs_file = _get_prefs_path()
        if prefs_file.exists():
            prefs_file.unlink()
        return "OK: preferences reset to factory defaults"
    
    return "ERROR: unknown subcommand (use: save, load, show, reset)"


# ============================================================================
# REPEAT BLOCKS (STUB - partial implementation)
# ============================================================================

def cmd_t(session: Session, args: List[str]) -> str:
    """Start a repeat block.
    
    Usage:
      /t <repeats>  -> Start block that repeats N times
    
    Must be closed with /end
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        return "ERROR: /t requires repeat count"
    
    try:
        repeats = int(args[0])
        session.block_stack.append(('repeat', repeats, []))
        session.recording_block = True
        return f"REPEAT BLOCK: started (will repeat {repeats} times)"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_end(session: Session, args: List[str]) -> str:
    """End a block (repeat, define, if).
    
    Usage:
      /end  -> Close current block
    """
    _ensure_advanced_attrs(session)
    
    # Handle function definition
    if session.defining_function:
        fname = session.defining_function
        session.user_functions[fname] = session.function_commands.copy()
        session.defining_function = None
        session.function_commands = []
        return f"DEFINE: function '{fname}' saved with {len(session.user_functions[fname])} commands"
    
    # Handle repeat block
    if session.block_stack:
        block_type, data, commands = session.block_stack.pop()
        
        if block_type == 'repeat':
            # Execute commands 'data' times
            # Note: This is a stub - full implementation would need command dispatcher
            return f"REPEAT END: would execute {len(commands)} commands {data} times (stub)"
        
        if block_type == 'if':
            return "IF END: conditional block closed (stub)"
    
    if not session.block_stack:
        session.recording_block = False
    
    return "END: no active block"


# ============================================================================
# FUNCTION DEFINITIONS - /fn, /run, /def
# ============================================================================

def cmd_fn(session: Session, args: List[str]) -> str:
    """Define a user function with optional named arguments.
    
    Usage:
      /fn                        List all functions
      /fn <name>                 Start recording function (no args)
      /fn <name> <arg1> <arg2>   Start recording with named arguments
      /fn <name> show            Show function contents
      /fn <name> del             Delete function
    
    Arguments can be referenced in commands as $arg_name or $1, $2, etc.
    
    Examples:
      /fn chord root third fifth
      /tone $root 1
      /tone $third 1
      /tone $fifth 1
      /end
      
      /run chord C4 E4 G4        Execute with positional args
      /run chord root=C4 third=E4 fifth=G4  Named args
    
    Short aliases: /fn, /func
    """
    _ensure_advanced_attrs(session)
    
    # Initialize function argument storage
    if not hasattr(session, 'function_args'):
        session.function_args = {}  # name -> list of arg names
    
    if not args:
        # List all functions
        if not session.user_functions:
            return "No functions defined. Use: /fn <name> [args...] to define"
        
        lines = ["=== DEFINED FUNCTIONS ==="]
        for name, cmds in session.user_functions.items():
            arg_list = session.function_args.get(name, [])
            arg_str = f"({', '.join(arg_list)})" if arg_list else ""
            lines.append(f"  {name}{arg_str}: {len(cmds)} commands")
        return '\n'.join(lines)
    
    fname = args[0].lower()
    
    # Check for subcommands
    if len(args) >= 2:
        sub = args[1].lower()
        
        if sub == 'show':
            if fname not in session.user_functions:
                return f"ERROR: function '{fname}' not found"
            cmds = session.user_functions[fname]
            arg_list = session.function_args.get(fname, [])
            arg_str = f"({', '.join(arg_list)})" if arg_list else ""
            lines = [f"=== FUNCTION: {fname}{arg_str} ==="]
            for i, cmd in enumerate(cmds, 1):
                lines.append(f"  {i}. {cmd}")
            return '\n'.join(lines)
        
        if sub == 'del' or sub == 'delete':
            if fname in session.user_functions:
                del session.user_functions[fname]
                if fname in session.function_args:
                    del session.function_args[fname]
                return f"OK: function '{fname}' deleted"
            return f"ERROR: function '{fname}' not found"
    
    # Start recording new function
    # args[1:] are the argument names
    arg_names = [a.lower() for a in args[1:] if not a.startswith('/')]
    
    # Validate arg names (no special characters)
    for arg in arg_names:
        if not arg.replace('_', '').isalnum():
            return f"ERROR: invalid argument name '{arg}'. Use alphanumeric and underscore only."
    
    session.defining_function = fname
    session.function_commands = []
    session.function_args[fname] = arg_names
    
    if arg_names:
        return f"FN: Recording '{fname}' with args ({', '.join(arg_names)}). Enter commands, /end to finish."
    else:
        return f"FN: Recording '{fname}'. Enter commands, /end to finish."


def cmd_run(session: Session, args: List[str]) -> str:
    """Execute a defined function with arguments.
    
    Usage:
      /run <name>                    Execute function (no args)
      /run <name> <val1> <val2>      Execute with positional args
      /run <name> arg=val arg2=val2  Execute with named args
      /run <name> <n>x               Execute N times (loop)
    
    Arguments substitute $arg_name or $1, $2, etc. in function body.
    
    Examples:
      /run myseq                     Execute myseq
      /run chord C4 E4 G4            Pass positional args
      /run chord root=D4 third=F#4   Pass named args
      /run beat 4x                   Execute 4 times
    
    Short aliases: /run, /exec
    """
    _ensure_advanced_attrs(session)
    
    if not hasattr(session, 'function_args'):
        session.function_args = {}
    
    if not args:
        if session.user_functions:
            names = []
            for name in session.user_functions.keys():
                arg_list = session.function_args.get(name, [])
                arg_str = f"({', '.join(arg_list)})" if arg_list else ""
                names.append(f"{name}{arg_str}")
            return f"Functions: {', '.join(names)}"
        return "ERROR: no function name. Define with /fn first."
    
    fname = args[0].lower()
    
    if fname not in session.user_functions:
        similar = [n for n in session.user_functions.keys() if fname in n or n in fname]
        if similar:
            return f"ERROR: '{fname}' not found. Did you mean: {', '.join(similar)}?"
        return f"ERROR: '{fname}' not found. Defined: {', '.join(session.user_functions.keys()) or '(none)'}"
    
    # Parse arguments and loop count
    loop_count = 1
    positional_args = []
    named_args = {}
    
    for arg in args[1:]:
        # Check for loop modifier (e.g., "4x")
        if arg.endswith('x') and arg[:-1].isdigit():
            loop_count = int(arg[:-1])
            continue
        
        # Check for named argument (e.g., "root=C4")
        if '=' in arg:
            key, val = arg.split('=', 1)
            named_args[key.lower()] = val
        else:
            positional_args.append(arg)
    
    # Get function's expected args
    expected_args = session.function_args.get(fname, [])
    
    # Build substitution map
    subs = {}
    
    # Map positional args to expected arg names
    for i, val in enumerate(positional_args):
        # $1, $2, etc.
        subs[f'${i+1}'] = val
        # $arg_name if we have expected args
        if i < len(expected_args):
            subs[f'${expected_args[i]}'] = val
    
    # Add named args
    for key, val in named_args.items():
        subs[f'${key}'] = val
    
    # Get command executor
    if not hasattr(session, 'command_executor'):
        return f"ERROR: command executor not available"
    
    # Execute function
    all_results = []
    
    for loop_i in range(loop_count):
        results = []
        for cmd_line in session.user_functions[fname]:
            # Substitute arguments
            expanded_cmd = cmd_line
            for var, val in subs.items():
                expanded_cmd = expanded_cmd.replace(var, str(val))
            
            # Check for unsubstituted variables
            if '$' in expanded_cmd:
                import re
                missing = re.findall(r'\$\w+', expanded_cmd)
                if missing:
                    results.append(f"WARNING: unbound variables in '{expanded_cmd}': {missing}")
            
            try:
                result = session.command_executor(expanded_cmd)
                if result and not result.startswith('OK'):
                    results.append(result)
            except Exception as e:
                results.append(f"ERROR in '{expanded_cmd}': {e}")
        
        if results:
            if loop_count > 1:
                all_results.append(f"[{loop_i+1}/{loop_count}] " + '; '.join(results[:3]))
            else:
                all_results.extend(results)
    
    if all_results:
        return f"RUN '{fname}':\n" + '\n'.join(all_results[:10])
    return f"OK: executed '{fname}'" + (f" x{loop_count}" if loop_count > 1 else "")


def cmd_def(session: Session, args: List[str]) -> str:
    """Define a user function with optional arguments.
    
    Usage:
      /def                        List all functions
      /def <name>                 Start recording function (no args)
      /def <name> <arg1> <arg2>   Start recording with named arguments
      /def <name> show            Show function contents
      /def <name> del             Delete function
    
    Arguments can be referenced in commands as $arg_name or $1, $2, etc.
    
    Examples:
      /def chord root third fifth
      /tone $root 1
      /tone $third 1  
      /tone $fifth 1
      /end
      
      /run chord C4 E4 G4        Execute with positional args
      /run chord root=C4         Execute with named args
    """
    _ensure_advanced_attrs(session)
    
    # Initialize function argument storage
    if not hasattr(session, 'function_args'):
        session.function_args = {}
    
    if not args:
        # List all functions
        if not session.user_functions:
            return "No functions defined. Use: /def <name> [args...] to define"
        
        lines = ["=== DEFINED FUNCTIONS ==="]
        for name, cmds in session.user_functions.items():
            arg_list = session.function_args.get(name, [])
            arg_str = f"({', '.join(arg_list)})" if arg_list else ""
            lines.append(f"  {name}{arg_str}: {len(cmds)} commands")
        return '\n'.join(lines)
    
    fname = args[0].lower()
    
    # Check for subcommands
    if len(args) >= 2:
        sub = args[1].lower()
        
        if sub == 'show':
            if fname not in session.user_functions:
                return f"ERROR: function '{fname}' not found"
            cmds = session.user_functions[fname]
            arg_list = session.function_args.get(fname, [])
            arg_str = f"({', '.join(arg_list)})" if arg_list else ""
            lines = [f"=== FUNCTION: {fname}{arg_str} ==="]
            for i, cmd in enumerate(cmds, 1):
                lines.append(f"  {i}. {cmd}")
            return '\n'.join(lines)
        
        if sub == 'del' or sub == 'delete':
            if fname in session.user_functions:
                del session.user_functions[fname]
                if fname in session.function_args:
                    del session.function_args[fname]
                return f"OK: function '{fname}' deleted"
            return f"ERROR: function '{fname}' not found"
    
    # Start recording new function
    # args[1:] are the argument names (filter out subcommands)
    arg_names = []
    for a in args[1:]:
        a_lower = a.lower()
        if a_lower in ('show', 'del', 'delete'):
            continue
        if not a.replace('_', '').isalnum():
            return f"ERROR: invalid argument name '{a}'. Use alphanumeric and underscore only."
        arg_names.append(a_lower)
    
    session.defining_function = fname
    session.function_commands = []
    session.function_args[fname] = arg_names
    
    if arg_names:
        return f"DEF: Recording '{fname}' with args ({', '.join(arg_names)}). Enter commands, /end to finish."
    else:
        return f"DEF: Recording '{fname}'. Enter commands, /end to finish."


def cmd_define(session: Session, args: List[str]) -> str:
    """Alias for /def."""
    return cmd_def(session, args)


def cmd_e(session: Session, args: List[str]) -> str:
    """Execute a defined function (legacy - use /run instead).
    
    Usage:
      /e <name>        Execute function
      /e <name> <args> Execute with args pushed to stack
    
    Note: /run is preferred as it supports named arguments.
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        if session.user_functions:
            return f"Functions: {', '.join(session.user_functions.keys())}"
        return "ERROR: no functions defined"
    
    fname = args[0].lower()
    
    if fname not in session.user_functions:
        return f"ERROR: function '{fname}' not found"
    
    # Push args to stack for legacy support
    if len(args) > 1:
        for arg in args[1:]:
            try:
                val = float(arg)
                if val == int(val):
                    val = int(val)
            except ValueError:
                val = arg
            session.user_stack.append(val)
    
    # Get command executor
    if not hasattr(session, 'command_executor'):
        return f"ERROR: command executor not available"
    
    results = []
    for cmd_line in session.user_functions[fname]:
        try:
            result = session.command_executor(cmd_line)
            if result:
                results.append(result)
        except Exception as e:
            results.append(f"ERROR: {e}")
    
    if results:
        return f"EXEC '{fname}':\n" + '\n'.join(results[:5])
    return f"OK: executed '{fname}' ({len(session.user_functions[fname])} commands)"


# ============================================================================
# RANDOM CHOICE (/or) - STUB
# ============================================================================

def cmd_or(session: Session, args: List[str]) -> str:
    """Append equal-weighted random choice to argument.
    
    Usage:
      /or <choice>  -> Add choice to current selection pool
    
    Choice is made at render/preview time.
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        if session.or_choices:
            return f"OR CHOICES: {session.or_choices}"
        return "OR: no choices queued"
    
    session.or_choices.append(' '.join(args))
    return f"OR: added choice '{args[0]}' ({len(session.or_choices)} total)"


# ============================================================================
# CUSTOM WAVE (/w, /wp) - STUB
# ============================================================================

def cmd_w(session: Session, args: List[str]) -> str:
    """Define a custom wave array.
    
    Usage:
      /w <name> <values...>  -> Define wave from values
      /w <name> from stack   -> Use array from stack
    
    Stub - needs full implementation.
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        names = list(session.custom_waves.keys())
        return f"CUSTOM WAVES: {names or '(none)'}"
    
    name = args[0]
    
    if len(args) > 1 and args[1] == 'from' and args[2] == 'stack':
        if session.user_stack and isinstance(session.user_stack[-1], np.ndarray):
            session.custom_waves[name] = session.user_stack.pop()
            return f"OK: wave '{name}' defined from stack"
        return "ERROR: no array on stack"
    
    # Parse values
    try:
        values = [float(x) for x in args[1:]]
        session.custom_waves[name] = np.array(values, dtype=np.float64)
        return f"OK: wave '{name}' defined with {len(values)} samples"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_wp(session: Session, args: List[str]) -> str:
    """Play custom wave for duration.
    
    Usage:
      /wp <name> <duration>  -> Play wave for duration in beats
    """
    _ensure_advanced_attrs(session)
    
    if len(args) < 2:
        return "ERROR: /wp requires wave name and duration"
    
    name = args[0]
    if name not in session.custom_waves:
        return f"ERROR: unknown wave '{name}'"
    
    try:
        duration = float(args[1])
        wave = session.custom_waves[name]
        
        # Calculate output length
        duration_sec = duration * 60.0 / session.bpm
        out_samples = int(duration_sec * session.sample_rate)
        
        # Tile or truncate wave to fit duration
        if len(wave) >= out_samples:
            buf = wave[:out_samples]
        else:
            reps = (out_samples // len(wave)) + 1
            buf = np.tile(wave, reps)[:out_samples]
        
        session.last_buffer = buf.astype(np.float64)
        return f"OK: playing wave '{name}' for {duration} beats"
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================================
# CONDITIONALS (/if, /elif, /else) - STUB
# ============================================================================

def cmd_if(session: Session, args: List[str]) -> str:
    """Start conditional block.
    
    Usage:
      /if <condition>  -> Start if block
    
    Stub - needs expression parser.
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        return "ERROR: /if requires condition"
    
    # Stub: just push to block stack
    condition = ' '.join(args)
    session.block_stack.append(('if', condition, []))
    session.recording_block = True
    return f"IF: started conditional block (condition: {condition}) [STUB]"


def cmd_elif(session: Session, args: List[str]) -> str:
    """Else-if branch in conditional."""
    _ensure_advanced_attrs(session)
    return "ELIF: [STUB] needs implementation"


def cmd_else(session: Session, args: List[str]) -> str:
    """Else branch in conditional."""
    _ensure_advanced_attrs(session)
    return "ELSE: [STUB] needs implementation"


# ============================================================================
# FILE UPLOAD (/upl) - STUB
# ============================================================================

def cmd_upl(session: Session, args: List[str]) -> str:
    """Open file picker to upload audio file.
    
    Files are converted to float64 format.
    
    Usage:
      /upl          -> Open Windows file picker dialog
      /upl <path>   -> Load specific file
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        # Try to open Windows file dialog
        path = None
        if os.name == 'nt':
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()  # Hide the main window
                root.attributes('-topmost', True)  # Bring dialog to front
                path = filedialog.askopenfilename(
                    title="Select Audio File",
                    filetypes=[
                        ("Audio files", "*.wav *.mp3 *.flac *.ogg *.aiff"),
                        ("WAV files", "*.wav"),
                        ("All files", "*.*")
                    ]
                )
                root.destroy()
                if not path:
                    return "UPLOAD: cancelled by user"
            except ImportError:
                return "ERROR: tkinter not available. Use /upl <path> instead."
            except Exception as e:
                return f"ERROR: could not open file picker - {e}\nUse /upl <path> instead."
        else:
            return "UPLOAD: file picker only available on Windows. Use /upl <path>"
        
        if path:
            args = [path]
        else:
            return "UPLOAD: no file selected"
    
    path = ' '.join(args)
    
    if not os.path.exists(path):
        return f"ERROR: file not found: {path}"
    
    try:
        import wave
        
        with wave.open(path, 'rb') as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        
        # Convert to float64
        if sampwidth == 1:
            data = np.frombuffer(frames, dtype=np.uint8).astype(np.float64)
            data = (data - 128) / 128.0
        elif sampwidth == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
            data = data / 32768.0
        elif sampwidth == 4:
            data = np.frombuffer(frames, dtype=np.int32).astype(np.float64)
            data = data / 2147483648.0
        else:
            return f"ERROR: unsupported sample width {sampwidth}"
        
        # Convert stereo to mono if needed
        if channels == 2:
            data = (data[::2] + data[1::2]) / 2.0
        
        # Resample if needed
        if framerate != session.sample_rate:
            ratio = session.sample_rate / framerate
            new_len = int(len(data) * ratio)
            x_old = np.linspace(0, 1, len(data))
            x_new = np.linspace(0, 1, new_len)
            data = np.interp(x_new, x_old, data)
        
        session.last_buffer = data
        return f"OK: loaded {path} ({len(data)} samples, converted to float64)"
    except Exception as e:
        return f"ERROR: could not load file - {e}\nTry again? (y/n)"


# ============================================================================
# LIVE LOOP MODE (/lbm, /tll, /start, /stop, /m) - STUB
# ============================================================================

def cmd_lbm(session: Session, args: List[str]) -> str:
    """Enable/disable loop back mode.
    
    Usage:
      /lbm       -> Toggle mode
      /lbm on    -> Enable
      /lbm off   -> Disable
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        session.live_loop_mode = not session.live_loop_mode
    elif args[0].lower() in ('on', '1', 'true'):
        session.live_loop_mode = True
    elif args[0].lower() in ('off', '0', 'false'):
        session.live_loop_mode = False
    
    if session.live_loop_mode:
        return "LIVE LOOP MODE: ON [STUB - uses offline render with loop playback]"
    return "LIVE LOOP MODE: OFF"


def cmd_tll(session: Session, args: List[str]) -> str:
    """Turn current clip/file into a live loop.
    
    Usage:
      /tll        -> Convert current clip to live loop
      /tll <idx>  -> Assign to specific loop index
    """
    _ensure_advanced_attrs(session)
    
    if session.last_buffer is None:
        return "ERROR: no buffer to convert"
    
    idx = len(session.live_loops)
    if args:
        try:
            idx = int(args[0])
        except ValueError:
            pass
    
    session.live_loops[idx] = {
        'buffer': session.last_buffer.copy(),
        'playing': False,
        'mutations': []
    }
    return f"OK: live loop {idx} created ({len(session.last_buffer)} samples)"


def cmd_start(session: Session, args: List[str]) -> str:
    """Start a live loop by index.
    
    Usage:
      /start <idx>  -> Start loop
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        return "ERROR: /start requires loop index"
    
    try:
        idx = int(args[0])
        if idx not in session.live_loops:
            return f"ERROR: loop {idx} does not exist"
        
        session.live_loops[idx]['playing'] = True
        return f"LOOP {idx}: started [STUB - would begin playback]"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_stop(session: Session, args: List[str]) -> str:
    """Stop a live loop by index.
    
    Usage:
      /stop <idx>  -> Stop loop
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        return "ERROR: /stop requires loop index"
    
    try:
        idx = int(args[0])
        if idx not in session.live_loops:
            return f"ERROR: loop {idx} does not exist"
        
        session.live_loops[idx]['playing'] = False
        return f"LOOP {idx}: stopped"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_m(session: Session, args: List[str]) -> str:
    """Mutate a live loop.
    
    Usage:
      /m <idx> <command>  -> Queue mutation for loop
    
    Mutation applies on next loop cycle.
    """
    _ensure_advanced_attrs(session)
    
    if len(args) < 2:
        return "ERROR: /m requires loop index and command"
    
    try:
        idx = int(args[0])
        if idx not in session.live_loops:
            return f"ERROR: loop {idx} does not exist"
        
        cmd = ' '.join(args[1:])
        session.live_loops[idx]['mutations'].append(cmd)
        return f"MUTATION queued for loop {idx}: {cmd}"
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================================
# ALGORITHM GENERATION (/g) - PARTIAL IMPLEMENTATION
# ============================================================================

def _generate_simple_kick(session: Session, variant: str = '1') -> np.ndarray:
    """Generate a simple kick drum sound."""
    sr = session.sample_rate
    duration = 0.3  # 300ms
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, dtype=np.float64)
    
    # Pitch envelope: starts high, drops quickly
    freq_start = 150.0
    freq_end = 50.0
    freq_decay = 20.0
    freq = freq_end + (freq_start - freq_end) * np.exp(-freq_decay * t)
    
    # Phase accumulation
    phase = np.cumsum(freq) / sr * 2 * np.pi
    osc = np.sin(phase)
    
    # Amplitude envelope
    amp_env = np.exp(-8.0 * t)
    
    return (osc * amp_env).astype(np.float64)


def _generate_simple_snare(session: Session, variant: str = '1') -> np.ndarray:
    """Generate a simple snare drum sound."""
    sr = session.sample_rate
    duration = 0.25
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, dtype=np.float64)
    
    # Body: pitched component
    body_freq = 200.0
    body = np.sin(2 * np.pi * body_freq * t) * np.exp(-15 * t)
    
    # Noise: snare wires
    noise = np.random.randn(samples) * np.exp(-10 * t)
    
    # Mix
    mix = 0.5 * body + 0.5 * noise
    return (mix / np.max(np.abs(mix))).astype(np.float64)


def _generate_simple_hihat(session: Session, variant: str = '1') -> np.ndarray:
    """Generate a simple hihat sound."""
    sr = session.sample_rate
    duration = 0.1 if variant == '1' else 0.3  # closed vs open
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, dtype=np.float64)
    
    # High-passed noise
    noise = np.random.randn(samples)
    
    # Simple HP approximation: differentiate
    hp_noise = np.diff(noise, prepend=noise[0])
    
    # Envelope
    decay = 30.0 if variant == '1' else 10.0
    env = np.exp(-decay * t)
    
    result = hp_noise * env
    return (result / np.max(np.abs(result))).astype(np.float64)


def _generate_simple_impact(session: Session, variant: str = '1') -> np.ndarray:
    """Generate a simple noise impact."""
    sr = session.sample_rate
    duration = 0.15
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, dtype=np.float64)
    
    noise = np.random.randn(samples)
    env = np.exp(-25 * t)
    
    result = noise * env
    return (result / np.max(np.abs(result))).astype(np.float64)


def _generate_pluck(session: Session, variant: str = '1') -> np.ndarray:
    """Generate a plucked string sound using Karplus-Strong synthesis."""
    sr = session.sample_rate
    
    # Variant controls pitch: 1=low, 2=mid, 3=high
    freqs = {'1': 110.0, '2': 220.0, '3': 440.0, '4': 880.0}
    freq = freqs.get(variant, 220.0)
    
    duration = 1.5  # Longer decay for pluck
    samples = int(sr * duration)
    
    # Karplus-Strong: delay line with low-pass filter
    delay_samples = int(sr / freq)
    
    # Initialize with noise burst
    buf = np.zeros(samples, dtype=np.float64)
    buf[:delay_samples] = np.random.uniform(-1, 1, delay_samples)
    
    # Simple low-pass filter in feedback loop
    for i in range(delay_samples, samples):
        # Average of two adjacent samples (simple LP)
        buf[i] = 0.5 * (buf[i - delay_samples] + buf[i - delay_samples - 1]) * 0.996
    
    # Normalize
    max_val = np.max(np.abs(buf))
    if max_val > 0:
        buf = buf / max_val
    
    return buf.astype(np.float64)


def _generate_pad(session: Session, variant: str = '1') -> np.ndarray:
    """Generate a soft pad sound with multiple detuned oscillators."""
    sr = session.sample_rate
    
    # Variant: 1=warm, 2=bright, 3=dark
    duration = 2.0
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, dtype=np.float64)
    
    # Base frequency
    base_freq = 220.0
    
    # Multiple detuned oscillators for richness
    detune_cents = [0, 7, -7, 12, -5]  # Slight detuning in cents
    
    pad = np.zeros(samples, dtype=np.float64)
    for cents in detune_cents:
        freq = base_freq * (2 ** (cents / 1200.0))
        osc = np.sin(2 * np.pi * freq * t)
        pad += osc
    
    # Add some harmonics based on variant
    if variant == '2':  # Bright
        pad += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
        pad += 0.1 * np.sin(2 * np.pi * base_freq * 3 * t)
    elif variant == '3':  # Dark - filter effect via fewer harmonics
        pass  # Just use fundamentals
    else:  # Warm (default)
        pad += 0.2 * np.sin(2 * np.pi * base_freq * 2 * t)
    
    # Soft attack/release envelope
    attack = int(0.3 * sr)
    release = int(0.5 * sr)
    
    env = np.ones(samples, dtype=np.float64)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    
    pad = pad * env
    
    # Normalize
    max_val = np.max(np.abs(pad))
    if max_val > 0:
        pad = pad / max_val
    
    return pad.astype(np.float64)


ALGO_GENERATORS = {
    'sk': _generate_simple_kick,
    'ssn': _generate_simple_snare,
    'sh': _generate_simple_hihat,
    'si': _generate_simple_impact,
    'pluck': _generate_pluck,
    'pl': _generate_pluck,
    'pad': _generate_pad,
}

# Generator types that are stubbed (not yet implemented)
STUB_GENERATORS = {
    'tom': 'Toms (tuned/electronic)',
    'cym': 'Cymbals (crash/ride/china/splash)',
    'clp': 'Claps (808/acoustic/layered)',
    'snp': 'Snaps (dry/reverb)',
    'shk': 'Shakers (16th/8th/triplet)',
    'blp': 'Bleep hits (short/pitched/glitch)',
    'stb': 'Stabs (chord/brass/synth)',
    'zap': 'Zaps (short/long/sweep)',
    'lsr': 'Laser FX (pew/beam/charge)',
    'bel': 'Bells (bright/dark/tubular/glass)',
    'bas': 'Bass hits (808/sub/punch/growl)',
    'rsr': 'Risers (noise/tonal/sweep/tension)',
    'dwn': 'Downlifters (drop/sweep/impact)',
    'wsh': 'Whooshes (fast/slow/textured)',
    'glt': 'Glitches (stutter/buffer/crush)',
    'vnl': 'Vinyl texture (crackle/hiss)',
    'wnd': 'Wind/noise (gentle/harsh/filtered)',
    'sil': 'Silence/spacer',
    'clk': 'Click track',
    'cal': 'Calibration tone',
    'swp': 'Test sweep',
}


def cmd_g(session: Session, args: List[str]) -> str:
    """Generate audio using algorithm.
    
    Usage:
      /g <algo> [variant] [c]  -> Generate and optionally create clip
      /g                       -> List all available algorithms
      /g all                   -> List all algorithms including stubs
    
    Implemented:
      sk     - Simple kick
      ssn    - Simple snare
      sh     - Simple hihat (variant: 1=closed, 2=open)
      si     - Simple impact
      pluck  - Karplus-Strong pluck (variant: 1=low, 2=mid, 3=high, 4=very high)
      pl     - Alias for pluck
      pad    - Soft pad sound (variant: 1=warm, 2=bright, 3=dark)
    
    Stubbed (coming soon):
      tom, cym, clp, snp, shk, blp, stb, zap, lsr, bel, bas,
      rsr, dwn, wsh, glt, vnl, wnd, sil, clk, cal, swp
    
    Add 'c' to create a clip: /g sk 1 c
    """
    _ensure_advanced_attrs(session)
    
    if not args:
        algos = list(ALGO_GENERATORS.keys())
        return f"ALGORITHMS (implemented): {', '.join(algos)}\nUse /g all to see stubs"
    
    if args[0].lower() == 'all':
        impl = list(ALGO_GENERATORS.keys())
        stub = list(STUB_GENERATORS.keys())
        lines = [
            "=== IMPLEMENTED ===",
            ', '.join(impl),
            "",
            "=== STUBBED (Section J2) ==="
        ]
        for key, desc in STUB_GENERATORS.items():
            lines.append(f"  {key:4s} - {desc}")
        return '\n'.join(lines)
    
    algo = args[0].lower()
    variant = args[1] if len(args) > 1 else '1'
    create_clip = 'c' in [a.lower() for a in args]
    
    # Check if it's a stubbed generator
    if algo in STUB_GENERATORS:
        return (f"STUB: /g {algo} - {STUB_GENERATORS[algo]}\n"
                f"  Variants: 1-4 (depending on type)\n"
                f"  Section J2: Not yet implemented\n"
                f"  Add 'c' to create clip: /g {algo} [var] c")
    
    if algo not in ALGO_GENERATORS:
        # Suggest similar
        all_algos = list(ALGO_GENERATORS.keys()) + list(STUB_GENERATORS.keys())
        similar = [a for a in all_algos if algo in a or a.startswith(algo[:2])][:5]
        if similar:
            return f"ERROR: unknown algorithm '{algo}'. Did you mean: {', '.join(similar)}?"
        return f"ERROR: unknown algorithm '{algo}'"
    
    try:
        import numpy as np
        buf = ALGO_GENERATORS[algo](session, variant)
        session.last_buffer = buf
        
        # Calculate metrics
        peak = np.max(np.abs(buf))
        rms = np.sqrt(np.mean(buf ** 2))
        dur = len(buf) / session.sample_rate
        
        lines = [f"OK: {algo} generated"]
        lines.append(f"  {dur:.3f}s, peak={peak:.3f}, rms={rms:.3f}")
        
        if create_clip:
            clip_name = f"gen_{algo}_{session.clip_count}"
            session.clip_count += 1
            session.clips[clip_name] = buf.copy()
            session.current_clip = clip_name
            lines.append(f"  -> clip '{clip_name}'")
        
        return '\n'.join(lines)
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================================
# FILE PROCESSING (/norm, /st, /rs, /grain) - STUBS
# ============================================================================

def cmd_norm(session: Session, args: List[str]) -> str:
    """Normalize current buffer manually.
    
    Usage:
      /norm        -> Normalize to -3dB peak
      /norm <dB>   -> Normalize to specified dB
    """
    _ensure_advanced_attrs(session)
    
    if session.last_buffer is None:
        return "ERROR: no buffer to normalize"
    
    target_db = -3.0
    if args:
        try:
            target_db = float(args[0])
        except ValueError:
            pass
    
    buf = session.last_buffer
    max_val = np.max(np.abs(buf))
    if max_val < 1e-10:
        return "ERROR: buffer is silent"
    
    target_linear = 10.0 ** (target_db / 20.0)
    session.last_buffer = buf * (target_linear / max_val)
    
    return f"OK: normalized to {target_db}dB"


def cmd_st(session: Session, args: List[str]) -> str:
    """Stretch buffer by factor (time stretch).
    
    Usage:
      /st <factor>  -> Stretch by factor (>1 = longer, <1 = shorter)
    """
    _ensure_advanced_attrs(session)
    
    if session.last_buffer is None:
        return "ERROR: no buffer to stretch"
    
    if not args:
        return "ERROR: /st requires stretch factor"
    
    try:
        factor = float(args[0])
        buf = session.last_buffer
        new_len = int(len(buf) * factor)
        
        # Simple linear interpolation stretch
        x_old = np.linspace(0, 1, len(buf))
        x_new = np.linspace(0, 1, new_len)
        session.last_buffer = np.interp(x_new, x_old, buf).astype(np.float64)
        
        return f"OK: stretched by {factor}x ({len(buf)} -> {new_len} samples)"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_rs(session: Session, args: List[str]) -> str:
    """Resample to different pitch.
    
    Usage:
      /rs <semitones>  -> Shift pitch by semitones
      /rs <ratio>      -> Resample by ratio (if decimal)
    """
    _ensure_advanced_attrs(session)
    
    if session.last_buffer is None:
        return "ERROR: no buffer to resample"
    
    if not args:
        return "ERROR: /rs requires pitch shift or ratio"
    
    try:
        val = float(args[0])
        
        # If integer-like, treat as semitones
        if val == int(val) and abs(val) <= 48:
            ratio = 2 ** (val / 12.0)
        else:
            ratio = val
        
        buf = session.last_buffer
        new_len = int(len(buf) / ratio)
        
        x_old = np.linspace(0, 1, len(buf))
        x_new = np.linspace(0, 1, new_len)
        session.last_buffer = np.interp(x_new, x_old, buf).astype(np.float64)
        
        return f"OK: resampled by {ratio:.4f}x ({len(buf)} -> {new_len} samples)"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_grain(session: Session, args: List[str]) -> str:
    """Chop buffer into grains for granular processing.
    
    Usage:
      /grain <size_ms>  -> Chop into grains of specified size
    
    Stub - opens granular functions.
    """
    _ensure_advanced_attrs(session)
    
    if session.last_buffer is None:
        return "ERROR: no buffer to grain"
    
    size_ms = 50.0
    if args:
        try:
            size_ms = float(args[0])
        except ValueError:
            pass
    
    grain_samples = int(size_ms / 1000.0 * session.sample_rate)
    num_grains = len(session.last_buffer) // grain_samples
    
    # Store grains metadata (stub)
    if not hasattr(session, 'grains'):
        session.grains = {}
    
    session.grains['current'] = {
        'buffer': session.last_buffer.copy(),
        'grain_size': grain_samples,
        'num_grains': num_grains
    }
    
    return f"GRAIN: chopped into {num_grains} grains of {size_ms}ms [STUB - use /g grain <function> for processing]"




# ============================================================================
# IN-HOUSE PLAYBACK (/play, /stop_play)
# ============================================================================

_current_playback = None  # Global to track current playback


def cmd_play(session: Session, args: List[str]) -> str:
    """Play current buffer using in-house audio playback.
    
    Usage:
      /play         -> Play last_buffer
      /play loop    -> Play in loop mode
      /play stop    -> Stop current playback
    
    Requires: simpleaudio or sounddevice
    Install with: pip install simpleaudio
    """
    global _current_playback
    _ensure_advanced_attrs(session)
    
    # Check for stop command
    if args and args[0].lower() in ('stop', 's'):
        if _current_playback is not None:
            try:
                _current_playback.stop()
                _current_playback = None
                return "PLAYBACK: stopped"
            except Exception:
                pass
        return "PLAYBACK: nothing playing"
    
    # Check if playback library is available
    if not PLAYBACK_AVAILABLE:
        return f"ERROR: Playback not available.\n{PLAYBACK_ERROR}\n\nInstall with: pip install simpleaudio"
    
    if session.last_buffer is None:
        return "ERROR: no buffer to play. Generate audio first."
    
    # Convert to int16 for playback
    buf = session.last_buffer.copy()
    
    # Normalize if needed
    max_val = np.max(np.abs(buf))
    if max_val > 1.0:
        buf = buf / max_val
    
    # Convert to int16
    audio_int16 = (buf * 32767).astype(np.int16)
    
    loop_mode = args and args[0].lower() in ('loop', 'l')
    
    try:
        # Try simpleaudio first
        try:
            import simpleaudio as sa
            
            # Stop any current playback
            if _current_playback is not None:
                try:
                    _current_playback.stop()
                except Exception:
                    pass
            
            play_obj = sa.play_buffer(
                audio_int16.tobytes(),
                num_channels=1,
                bytes_per_sample=2,
                sample_rate=session.sample_rate
            )
            _current_playback = play_obj
            
            if loop_mode:
                return f"PLAYBACK: started (loop mode) - {len(buf)} samples @ {session.sample_rate}Hz\nUse /play stop to stop"
            return f"PLAYBACK: started - {len(buf)} samples @ {session.sample_rate}Hz ({len(buf)/session.sample_rate:.2f}s)"
            
        except ImportError:
            # Fall back to sounddevice
            import sounddevice as sd
            
            sd.stop()  # Stop any current playback
            
            if loop_mode:
                # For loop mode, we'd need a callback - just play once for now
                sd.play(buf.astype(np.float32), session.sample_rate, loop=True)
                return f"PLAYBACK: started (loop mode) - use /play stop to stop"
            else:
                sd.play(buf.astype(np.float32), session.sample_rate)
                return f"PLAYBACK: started - {len(buf)} samples @ {session.sample_rate}Hz ({len(buf)/session.sample_rate:.2f}s)"
                
    except Exception as e:
        return f"ERROR: playback failed - {e}\n\nMake sure audio device is available."


def cmd_stop_play(session: Session, args: List[str]) -> str:
    """Stop current audio playback.
    
    Usage:
      /stop_play  -> Stop any playing audio
    """
    global _current_playback
    
    try:
        if _current_playback is not None:
            _current_playback.stop()
            _current_playback = None
        
        # Also try sounddevice stop
        try:
            import sounddevice as sd
            sd.stop()
        except ImportError:
            pass
        
        return "PLAYBACK: stopped"
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================================
# HELP SYSTEM
# ============================================================================

# Command documentation for help system
COMMAND_HELP = {
    'nt': {
        'desc': 'Play a note in 432Hz equal temperament',
        'usage': '/nt <note> [beats] [amp]',
        'params': {
            'note': 'MIDI number (60=C4) or note name (C4, A#3, Bb5)',
            'beats': 'Duration in beats (default: current /d setting)',
            'amp': 'Amplitude 0-1 (default: 1.0)'
        },
        'examples': ['/nt 60', '/nt C4 2', '/nt A#3 1 0.5']
    },
    'nts': {
        'desc': 'Play note sequence and insert as clip',
        'usage': '/nts <notes> [d:<durations>] [r:<rests>]',
        'params': {
            'notes': 'Comma-separated notes (C4,E4,G4 or 60,64,67)',
            'd:': 'Duration(s) - single value or comma-separated',
            'r:': 'Rest(s) between notes'
        },
        'examples': ['/nts C4,E4,G4', '/nts C4,E4,G4 d:0.5', '/nts 60,64,67 d:0.25,0.5,0.25 r:0.1']
    },
    'g': {
        'desc': 'Generate audio using built-in algorithms',
        'usage': '/g <algo> [variant] [c]',
        'params': {
            'algo': 'sk(kick), ssn(snare), sh(hihat), si(impact), pluck/pl, pad',
            'variant': 'Algorithm-specific variant number',
            'c': 'Add to create clip'
        },
        'examples': ['/g sk', '/g sh 2', '/g pluck 3 c', '/g pad 1 c']
    },
    'def': {
        'desc': 'Define a function with optional named arguments',
        'usage': '/def <n> [arg1 arg2 ...]',
        'params': {
            'name': 'Function name (case-insensitive)',
            'args': 'Optional argument names (use $arg in commands)'
        },
        'examples': ['/def myseq', '/def chord root third fifth', '/def mychord show', '/def mychord del']
    },
    'run': {
        'desc': 'Execute a defined function with arguments',
        'usage': '/run <n> [val1 val2 ...] [Nx]',
        'params': {
            'name': 'Function name',
            'values': 'Positional args or name=value pairs',
            'Nx': 'Loop count (e.g., 4x runs 4 times)'
        },
        'examples': ['/run myseq', '/run chord C4 E4 G4', '/run chord root=D4', '/run beat 4x']
    },
    'play': {
        'desc': 'Play current buffer through speakers',
        'usage': '/play [loop|stop]',
        'params': {
            'loop': 'Play in loop mode',
            'stop': 'Stop current playback'
        },
        'examples': ['/play', '/play loop', '/play stop']
    },
    'upl': {
        'desc': 'Upload/load an audio file',
        'usage': '/upl [path]',
        'params': {
            'path': 'File path (optional - opens file picker on Windows)'
        },
        'examples': ['/upl', '/upl C:\\music\\sample.wav']
    },
    '.': {
        'desc': 'Get a built-in or user constant',
        'usage': '/. <name>',
        'params': {
            'name': 'Constant name or "list" to show all'
        },
        'examples': ['/. pi', '/. list', '/. sin1s', '/. env_perc']
    }
}


def get_command_help(cmd_name: str) -> str:
    """Get formatted help for a command."""
    if cmd_name not in COMMAND_HELP:
        return None
    
    info = COMMAND_HELP[cmd_name]
    lines = [
        f"/{cmd_name}: {info['desc']}",
        f"",
        f"Usage: {info['usage']}",
        f"",
        "Parameters:"
    ]
    
    for param, desc in info['params'].items():
        lines.append(f"  {param}: {desc}")
    
    lines.append("")
    lines.append("Examples:")
    for ex in info['examples']:
        lines.append(f"  {ex}")
    
    return '\n'.join(lines)


def suggest_on_error(cmd_name: str, error_msg: str) -> str:
    """Add helpful suggestions to error messages."""
    help_text = get_command_help(cmd_name)
    if help_text:
        return f"{error_msg}\n\n--- Help for /{cmd_name} ---\n{help_text}"
    return error_msg


# ============================================================================
# COMMAND REGISTRY
# ============================================================================

ADVANCED_COMMANDS = {
    # Note system
    'nt': cmd_nt,
    'nts': cmd_nts,
    # Note buffer system
    'ntm': cmd_ntm,
    'nbs': cmd_nbs,
    'ntrn': cmd_ntrn,

    # Interval root control
    'nbr': cmd_nbr,
    
    # Duration/rest
    'd': cmd_d,
    'r': cmd_r,
    
    # Chord mode
    'c': cmd_c,
    'co': cmd_co,
    
    # Stack
    '=': cmd_eq,
    '0': cmd_0,
    'app': cmd_app,
    
    # Random
    'rn': cmd_rn,
    'rsd': cmd_rsd,
    'rmode': cmd_rmode,
    'randmode': cmd_rmode,
    'sm': cmd_sm,
    
    # Math
    'plus': cmd_plus,
    'minus': cmd_minus,
    'tms': cmd_tms,
    'power': cmd_power,
    
    # Constants
    '.': cmd_dot,
    'u.constant': cmd_u_constant,
    
    # Preferences
    'up': cmd_up,
    'upl': cmd_upl,
    
    # Blocks
    't': cmd_t,
    'end': cmd_end,
    'define': cmd_define,
    'def': cmd_def,
    
    # Function system with arguments
    'fn': cmd_fn,
    'func': cmd_fn,

    # Function execution (legacy alias for /run)
    'e': cmd_run,
    'run': cmd_run,
    'exec': cmd_run,
    
    # Conditionals
    'if': cmd_if,
    'elif': cmd_elif,
    'else': cmd_else,
    
    # Random choice
    'or': cmd_or,
    
    # Custom wave
    'w': cmd_w,
    'wp': cmd_wp,
    
    # Live loops
    'lbm': cmd_lbm,
    'tll': cmd_tll,
    'start': cmd_start,
    'stop': cmd_stop,
    'm': cmd_m,
    
    # Algorithm generation
    'g': cmd_g,
    
    # File processing
    'norm': cmd_norm,
    'st': cmd_st,
    'rs': cmd_rs,
    'grain': cmd_grain,
    
    # Playback
    'play': cmd_play,
    'stop_play': cmd_stop_play,
}


def get_stack_value(session: Session, index: int) -> Any:
    """Helper to get stack value by index for command substitution."""
    _ensure_advanced_attrs(session)
    if not session.user_stack:
        return None
    actual_idx = len(session.user_stack) - 1 - index
    if 0 <= actual_idx < len(session.user_stack):
        return session.user_stack[actual_idx]
    return None
