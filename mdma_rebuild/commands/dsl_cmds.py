"""
MAD DSL (MDMA Advanced DSL) Command Module - v2

Unified audio buffer system. All audio flows to session.working_buffer.
Note notation is consistent everywhere (semitone intervals −24…24, MIDI outside that).
/play finds audio automatically. /wa always appends. No lost audio.

DSL Quick Reference:
  /mel <pattern> [hz]     Melody to working buffer
  /out ... /end           Note block to working buffer
  /wa tone 440 0.5        Append tone to working
  /wa mel 0.4.7           Append melody to working
  /wa <n>                 Commit working to buffer N
  /chain <id>             Create FX chain
  /<chain> add <fx>       Add effect to chain
  /<chain>                Start effect block
  /apply <chain>          One-shot apply chain
  /play                   Play (finds audio automatically)
  /render [path]          Render working to WAV file
  /loop [times]           Loop working buffer
  /b                      Show all buffers

Live Loops:
  /live <name>            Start recording a live loop
    (commands recorded, not executed)
  /end                    Starts the loop playing in background
  /live                   List all active loops
  /kill <name>            Stop a specific loop
  /ka                     Kill ALL loops
"""

from __future__ import annotations

import re
import os
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field

from ..core.session import Session
from ..dsp.advanced_ops import get_user_stack


# ============================================================================
# DSL DEBUGGING
# ============================================================================

# Debug logging flag
_DSL_DEBUG = False

def dsl_debug_log(msg: str) -> None:
    """Log a debug message if debugging is enabled."""
    if _DSL_DEBUG:
        print(f"[DSL] {msg}")

def set_dsl_debug(enabled: bool) -> None:
    """Enable or disable DSL debug logging."""
    global _DSL_DEBUG
    _DSL_DEBUG = enabled

def is_dsl_debug() -> bool:
    """Check if DSL debugging is enabled."""
    return _DSL_DEBUG


# ============================================================================
# DSL STATE
# ============================================================================

@dataclass
class DSLState:
    """State for MAD DSL. Audio always goes to session buffers."""
    # Mode flags
    dsl_mode: bool = False
    out_block: bool = False
    fx_block: bool = False
    live_block: bool = False
    
    # Synth instances
    synth_instances: Dict[str, dict] = field(default_factory=dict)
    current_synth: int = 0
    synth_names: List[str] = field(default_factory=list)
    
    # FX chains
    fx_chains: Dict[str, List[str]] = field(default_factory=dict)
    current_chain: Optional[str] = None
    
    # Out block state
    out_commands: List[str] = field(default_factory=list)
    out_chain: Optional[str] = None
    out_root_hz: float = 440.0
    
    # Sticky params for /out blocks
    sticky_amp: float = 0.5
    sticky_atk: float = 0.01
    sticky_dec: float = 0.1
    sticky_sus: float = 0.8
    sticky_rel: float = 0.1
    sticky_dur: float = 0.25
    
    last_chord: List[int] = field(default_factory=list)
    dsl_commands: List[str] = field(default_factory=list)
    fx_block_chain: Optional[str] = None
    
    # Loop state
    loop_active: bool = False
    loop_count: int = 0
    
    # Live loop recording
    live_block_name: Optional[str] = None
    live_block_commands: List[str] = field(default_factory=list)
    
    def get_status(self) -> str:
        """Return a string representation of current DSL state."""
        lines = ["=== DSL STATE ==="]
        lines.append(f"  dsl_mode: {self.dsl_mode}")
        lines.append(f"  out_block: {self.out_block}")
        lines.append(f"  fx_block: {self.fx_block} (chain: {self.fx_block_chain})")
        lines.append(f"  live_block: {self.live_block} (name: {self.live_block_name})")
        lines.append(f"  synth_instances: {list(self.synth_instances.keys())}")
        lines.append(f"  fx_chains: {list(self.fx_chains.keys())}")
        lines.append(f"  current_chain: {self.current_chain}")
        lines.append(f"  dsl_commands queued: {len(self.dsl_commands)}")
        return "\n".join(lines)


# Global state
_dsl_state = DSLState()

def get_dsl_state() -> DSLState:
    return _dsl_state

def reset_dsl_state() -> None:
    global _dsl_state
    _dsl_state = DSLState()
    dsl_debug_log("DSL state reset to defaults")


# ============================================================================
# COMMENT AND PREPROCESSING
# ============================================================================

def strip_comments(line: str) -> str:
    """Remove /// comments /// from a line."""
    result = re.sub(r'///.*?///', '', line)
    return result.strip()

def preprocess_dsl_line(line: str) -> Optional[str]:
    """Preprocess a DSL line. Returns None to skip."""
    line = strip_comments(line)
    if not line:
        return None
    state = get_dsl_state()
    if state.dsl_mode:
        lower = line.lower()
        # Handle commands that can be typed without / in DSL mode
        if lower.startswith('bpm '):
            return '/bpm ' + line[4:]
        if lower == 'aon':
            return '/autoplay on'
        if lower == 'aoff':
            return '/autoplay off'
        # Parameter commands (v47)
        if lower.startswith('parm'):
            return '/' + line
        if lower.startswith('cha ') or lower == 'cha':
            return '/' + line
        if lower.startswith('plist'):
            return '/' + line
        if lower.startswith('hq ') or lower == 'hq':
            return '/' + line
    return line


# ============================================================================
# NOTE CONVERSION - unified system
# ============================================================================

def note_to_hz(note_val: int, root_hz: float = 440.0) -> float:
    """Convert a note value to frequency.

    Interval mode (−24 to 24 inclusive):
        Semitone offset from *root_hz*.
        0 = root, 7 = perfect 5th, 12 = octave, 24 = 2 octaves,
        −12 = octave below, etc.

    MIDI mode (outside −24…24):
        Absolute MIDI note number.  60 = middle C (261.63 Hz),
        69 = A4 (440 Hz).

    This is the SINGLE source of truth for note→frequency conversion.
    Used by /mel, /cor, and /out.
    """
    if -24 <= note_val <= 24:
        # Semitone offset from root
        return root_hz * (2.0 ** (note_val / 12.0))
    else:
        # MIDI note: A4 (69) = 440Hz
        return 440.0 * (2.0 ** ((note_val - 69) / 12.0))


# ============================================================================
# AUDIO RENDERING HELPERS
# ============================================================================

def render_tone(freq: float, duration: float, amp: float, sr: int,
                atk: float = 0.01, rel: float = 0.05):
    """Render a single tone with ADSR envelope. Used everywhere."""
    import numpy as np
    
    samples = int(duration * sr)
    if samples == 0:
        return np.zeros(0, dtype=np.float64)
    
    t = np.arange(samples) / sr
    tone = np.sin(2 * np.pi * freq * t).astype(np.float64) * amp
    
    # Envelope
    atk_s = min(int(atk * sr), samples // 3)
    rel_s = min(int(rel * sr), samples // 3)
    
    if atk_s > 0:
        tone[:atk_s] *= np.linspace(0, 1, atk_s)
    if rel_s > 0:
        tone[-rel_s:] *= np.linspace(1, 0, rel_s)
    
    return tone


def render_events(events, root_hz: float, sr: int,
                  unit_dur: float = 0.25, amp: float = 0.5,
                  session=None, sydef_name: str = None):
    """Render note events to audio.

    Timing
    ------
    When *session* is provided the timing grid is locked to BPM:
      1 pattern unit = 1 beat (quarter note)
      duration (sec) = units × 60 / BPM

    Without *session* the legacy ``unit_dur`` (seconds) fallback is used.

    If *session* is provided, each note is rendered through the full synth
    engine (``session.generate_tone``) which applies operators, filters,
    ADSR, voice unison, and the effects chain.  When *sydef_name* is also
    given the named SyDef is instantiated (via ``/use``) before the first
    note so the engine is configured to the patch.

    Without *session* the legacy ``render_tone()`` sine fallback is used.
    """
    import numpy as np

    # --- Compute timing grid ---
    # 1 pattern unit = 1 beat when session available
    if session:
        bpm = getattr(session, 'bpm', 120.0)
        sec_per_unit = 60.0 / bpm
    else:
        sec_per_unit = unit_dur

    total_dur = sum(e['duration'] for e in events)
    total_seconds = total_dur * sec_per_unit
    total_samples = int(total_seconds * sr)

    if total_samples == 0:
        return np.zeros(0, dtype=np.float64)

    # Apply SyDef if requested (configures synth engine state)
    if session and sydef_name:
        _apply_sydef_setup(session, sydef_name)

    audio = np.zeros(total_samples, dtype=np.float64)
    pos = 0

    for event in events:
        event_samples = int(event['duration'] * sec_per_unit * sr)

        if event['type'] == 'note':
            freq = note_to_hz(event['pitch'], root_hz)

            if session:
                # Full synth engine render — beats directly
                note_beats = float(event['duration'])
                try:
                    tone = session.generate_tone(freq, note_beats, amp)
                    if tone is not None and len(tone) > 0:
                        tone = tone.astype(np.float64)
                        # Handle stereo -> mono for melody mix
                        if tone.ndim == 2:
                            tone = np.mean(tone, axis=1).astype(np.float64)
                        end = min(pos + len(tone), total_samples)
                        audio[pos:end] += tone[:end - pos]
                except Exception:
                    # Fallback to simple sine
                    note_dur_sec = event['duration'] * sec_per_unit
                    tone = render_tone(freq, note_dur_sec, amp, sr)
                    end = min(pos + len(tone), total_samples)
                    audio[pos:end] += tone[:end - pos]
            else:
                # Simple sine fallback (seconds-based)
                note_dur_sec = event['duration'] * sec_per_unit
                tone = render_tone(freq, note_dur_sec, amp, sr)
                end = min(pos + len(tone), total_samples)
                audio[pos:end] += tone[:end - pos]

        elif event['type'] == 'chord':
            # Inline chord from (N,N,N) grouping — render all pitches
            pitches = event['pitches']
            n_notes = max(len(pitches), 1)
            chord_amp = amp / (n_notes ** 0.5)  # equal-power scaling

            for pitch in pitches:
                freq = note_to_hz(pitch, root_hz)
                if session:
                    note_beats = float(event['duration'])
                    try:
                        tone = session.generate_tone(freq, note_beats, chord_amp)
                        if tone is not None and len(tone) > 0:
                            tone = tone.astype(np.float64)
                            if tone.ndim == 2:
                                tone = np.mean(tone, axis=1).astype(np.float64)
                            end = min(pos + len(tone), total_samples)
                            audio[pos:end] += tone[:end - pos]
                    except Exception:
                        note_dur_sec = event['duration'] * sec_per_unit
                        tone = render_tone(freq, note_dur_sec, chord_amp, sr)
                        end = min(pos + len(tone), total_samples)
                        audio[pos:end] += tone[:end - pos]
                else:
                    note_dur_sec = event['duration'] * sec_per_unit
                    tone = render_tone(freq, note_dur_sec, chord_amp, sr)
                    end = min(pos + len(tone), total_samples)
                    audio[pos:end] += tone[:end - pos]

        pos += event_samples

    return audio


def render_chord_events(events, root_hz: float, sr: int,
                        unit_dur: float = 0.25, amp: float = 0.5,
                        session=None, sydef_name: str = None):
    """Render chord events to audio.

    Timing is BPM-locked when *session* is available:
    1 pattern unit = 1 beat (quarter note).
    Each event has type 'chord' with a list of pitches played simultaneously,
    or type 'rest'.  Uses the synth engine when *session* is available.
    """
    import numpy as np

    # --- Compute timing grid ---
    if session:
        bpm = getattr(session, 'bpm', 120.0)
        sec_per_unit = 60.0 / bpm
    else:
        sec_per_unit = unit_dur

    total_dur = sum(e['duration'] for e in events)
    total_seconds = total_dur * sec_per_unit
    total_samples = int(total_seconds * sr)

    if total_samples == 0:
        return np.zeros(0, dtype=np.float64)

    # Apply SyDef if requested
    if session and sydef_name:
        _apply_sydef_setup(session, sydef_name)

    audio = np.zeros(total_samples, dtype=np.float64)
    pos = 0

    for event in events:
        event_samples = int(event['duration'] * sec_per_unit * sr)

        if event['type'] == 'chord':
            pitches = event['pitches']
            n_notes = max(len(pitches), 1)
            chord_amp = amp / (n_notes ** 0.5)  # equal-power scaling

            for pitch in pitches:
                freq = note_to_hz(pitch, root_hz)

                if session:
                    note_beats = float(event['duration'])
                    try:
                        tone = session.generate_tone(freq, note_beats, chord_amp)
                        if tone is not None and len(tone) > 0:
                            tone = tone.astype(np.float64)
                            if tone.ndim == 2:
                                tone = np.mean(tone, axis=1).astype(np.float64)
                            end = min(pos + len(tone), total_samples)
                            audio[pos:end] += tone[:end - pos]
                    except Exception:
                        note_dur_sec = event['duration'] * sec_per_unit
                        tone = render_tone(freq, note_dur_sec, chord_amp, sr)
                        end = min(pos + len(tone), total_samples)
                        audio[pos:end] += tone[:end - pos]
                else:
                    note_dur_sec = event['duration'] * sec_per_unit
                    tone = render_tone(freq, note_dur_sec, chord_amp, sr)
                    end = min(pos + len(tone), total_samples)
                    audio[pos:end] += tone[:end - pos]

        pos += event_samples

    # Soft clip
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio /= peak

    return audio


def _apply_sydef_setup(session, sydef_name: str):
    """Apply a SyDef's setup commands (everything except /tone) to configure
    the synth engine before rendering notes."""
    try:
        from .sydef_cmds import _ensure_sydef_attrs
        _ensure_sydef_attrs(session)
        if sydef_name not in session.sydefs:
            return
        sd = session.sydefs[sydef_name]
        executor = getattr(session, 'command_executor', None)
        if not executor:
            return
        subs = sd.default_map()
        # Execute only setup commands (skip /tone — we render notes ourselves)
        for cmd_line in sd.commands:
            expanded = cmd_line
            for key in sorted(subs.keys(), key=len, reverse=True):
                expanded = expanded.replace(f'${key}', str(subs[key]))
            # Skip tone/generate commands — we drive note rendering
            parts = expanded.lstrip('/').split()
            if parts and parts[0].lower() in ('tone', 'n', 'g', 'gen'):
                continue
            try:
                executor(expanded)
            except Exception:
                pass
    except ImportError:
        pass


# ============================================================================
# MELODY PATTERN PARSER
# ============================================================================

def parse_melody_pattern(pattern: str) -> List[dict]:
    """Parse a melody pattern string into note events.

    Syntax
    ------
    -24 to 24   Semitone interval from root (0=root, 7=5th, 12=octave,
                24=two octaves, -12=octave below)
    25+         MIDI note number (60=C4, 69=A4)
    .           Note separator (single dot between notes)
    ..          Extra dots EXTEND the preceding note by 1 beat each
    r           Rest (1 beat)
    _           Rest (1 beat) — explicit rest token (Phase 6)
    bN          Flat (subtract 1 semitone from N)
    #N          Sharp (add 1 semitone to N)
    (N,N,N)     Inline chord grouping (Phase 6) — e.g. (0,4,7)

    Duration rules
    ~~~~~~~~~~~~~~
    Each note/rest starts with 1 beat.  The FIRST dot after any
    element is a separator.  Additional consecutive dots extend
    the preceding element by 1 beat each.

    Examples:
      "0.4.7"         3 notes, 1 beat each  = 3 beats total
      "0.12.7"        root, octave, 5th (3 beats)
      "0..4.7"        note 0 for 2 beats, 4 for 1, 7 for 1  = 4 beats
      "-12.0.12"      octave below, root, octave above
      "0.r.4"         note 0 (1), rest (1), note 4 (1)       = 3 beats
      "0._.4"         same as above using explicit rest token
      "60.64.67"      MIDI: C4, E4, G4
      "(0,4,7).0.4"   inline chord then two single notes
    """
    events = []
    i = 0
    current_note = None
    current_duration = 1
    # How many consecutive dots since the last content (note/rest).
    # dot 1 = separator, dot 2+ = extend.
    dot_run = 0

    def flush():
        nonlocal current_note, current_duration
        if current_note is not None:
            events.append({'type': 'note', 'pitch': current_note, 'duration': current_duration})
        current_note = None
        current_duration = 1

    while i < len(pattern):
        ch = pattern[i]

        if ch == '.':
            dot_run += 1
            if dot_run == 1:
                # First dot = separator — flush pending note
                flush()
            else:
                # dot_run >= 2 → extend last event
                if events:
                    events[-1]['duration'] += 1
            i += 1
            continue

        # Non-dot character resets run counter
        dot_run = 0

        if ch == 'r' or ch == '_':
            flush()
            events.append({'type': 'rest', 'duration': 1})
            i += 1
            continue

        # Inline chord grouping: (N,N,N)
        if ch == '(':
            flush()
            i += 1  # skip '('
            group_str = ''
            while i < len(pattern) and pattern[i] != ')':
                group_str += pattern[i]
                i += 1
            if i < len(pattern):
                i += 1  # skip ')'
            pitches = []
            for tok in group_str.split(','):
                tok = tok.strip()
                if not tok:
                    continue
                offset = 0
                if tok.startswith('#'):
                    offset = 1
                    tok = tok[1:]
                elif tok.startswith('b') and len(tok) > 1 and tok[1:].lstrip('-').isdigit():
                    offset = -1
                    tok = tok[1:]
                try:
                    pitches.append(int(tok) + offset)
                except ValueError:
                    pass
            if pitches:
                events.append({'type': 'chord', 'pitches': pitches, 'duration': 1})
            continue

        if ch == 'b' and i + 1 < len(pattern) and pattern[i + 1].isdigit():
            flush()
            i += 1
            num = ''
            while i < len(pattern) and pattern[i].isdigit():
                num += pattern[i]
                i += 1
            if num:
                current_note = int(num) - 1
                current_duration = 1
            continue

        if ch == '#':
            flush()
            i += 1
            num = ''
            while i < len(pattern) and pattern[i].isdigit():
                num += pattern[i]
                i += 1
            if num:
                current_note = int(num) + 1
                current_duration = 1
            continue

        if ch.isdigit() or (ch == '-' and i + 1 < len(pattern) and pattern[i + 1].isdigit()):
            flush()
            num = ''
            if ch == '-':
                num = '-'
                i += 1
            while i < len(pattern) and pattern[i].isdigit():
                num += pattern[i]
                i += 1
            current_note = int(num)
            current_duration = 1
            continue

        i += 1

    flush()
    return events


# ============================================================================
# BUFFER COMMANDS
# ============================================================================

def cmd_b(session: Session, args: List[str]) -> str:
    """Show or select buffers.
    
    Usage:
      /b              List all buffers with audio
      /b <n>          Select buffer N
      /b w            Show working buffer info
    """
    sr = getattr(session, 'sample_rate', 44100)
    
    if not args:
        filled = session.get_filled_buffers()
        lines = ["BUFFERS:"]
        
        if session.has_real_working_audio():
            wb = session.working_buffer
            lines.append(f"  W (working): {len(wb)/sr:.2f}s [{session.working_buffer_source}]")
        else:
            lines.append(f"  W (working): empty")
        
        for idx in filled:
            buf = session.buffers[idx]
            marker = " *" if idx == session.current_buffer_index else ""
            lines.append(f"  {idx}: {len(buf)/sr:.2f}s{marker}")
        
        if not filled:
            lines.append("  (no numbered buffers)")
        return "\n".join(lines)
    
    target = args[0].lower()
    if target == 'w':
        if session.has_real_working_audio():
            wb = session.working_buffer
            return f"WORKING: {len(wb)/sr:.2f}s ({len(wb)} samples) [{session.working_buffer_source}]"
        return "WORKING: empty"
    
    try:
        idx = int(target)
        session.current_buffer_index = idx
        session.ensure_buffer_count(idx)
        buf = session.buffers.get(idx)
        if buf is not None and len(buf) > 0:
            return f"BUFFER {idx}: selected ({len(buf)/sr:.2f}s)"
        return f"BUFFER {idx}: selected (empty)"
    except ValueError:
        return f"ERROR: /b needs a number or 'w'. Got '{target}'"


# ============================================================================
# MELODY COMMAND
# ============================================================================

def cmd_mel(session: Session, args: List[str]) -> str:
    """Render a melody pattern to working buffer using the synth engine.
    
    Usage:
      /mel                          Default triad at 440Hz
      /mel <pattern>                Pattern at 440Hz
      /mel <pattern> <hz>           Pattern at custom root
      /mel <pattern> <hz> sydef=<n> Pattern using a SyDef patch
      /mel <pattern> sydef=<n>      Pattern at 440Hz with SyDef
    
    Pattern syntax:
      -24 to 24 = semitone intervals from root (12=octave, 7=5th).
      25+ = MIDI note numbers (60=C4, 69=A4).
      Dots (.) separate notes (1 beat each).
      Extra dots (..) extend the preceding note by 1 beat.
      r = rest.

    Timing:
      1 unit = 1 beat at current BPM.
      At 120 BPM each unit = 0.5s, at 60 BPM = 1.0s.

    Examples:
      /mel 0.4.7                    Major triad (3 beats)
      /mel 0.12.7                   Root, octave, 5th
      /mel -12.0.12.24              Two-octave sweep
      /mel 0..4.7                   Root held 2 beats, then 4, 7
      /mel 0.4.7.0.3.7 sydef=acid  6 notes with acid patch
      /mel 60.64.67 440             MIDI notes C4-E4-G4

    Audio REPLACES working buffer. Use /wa mel to APPEND.
    """
    import numpy as np
    
    pattern = args[0] if args else "0.2.4"
    root_hz = 440.0
    sydef_name = None
    
    for arg in args[1:]:
        clean = arg.lstrip('/')
        # Check for sydef= parameter
        if clean.lower().startswith('sydef='):
            sydef_name = clean.split('=', 1)[1].lower()
            continue
        try:
            root_hz = float(clean)
        except ValueError:
            # Check if the first non-pattern arg is a sydef name
            if '=' not in clean:
                try:
                    from .sydef_cmds import _ensure_sydef_attrs
                    _ensure_sydef_attrs(session)
                    if clean.lower() in getattr(session, 'sydefs', {}):
                        sydef_name = clean.lower()
                        continue
                except ImportError:
                    pass
    
    # Also check if first arg itself is sydef=
    if args and args[0].lower().startswith('sydef='):
        sydef_name = args[0].split('=', 1)[1].lower()
        pattern = args[1] if len(args) > 1 else "0.2.4"
    
    events = parse_melody_pattern(pattern)
    if not events:
        return "ERROR: No notes in pattern. Example: /mel 0.4.7"
    
    sr = getattr(session, 'sample_rate', 44100)

    # Render through synth engine (with optional SyDef)
    try:
        audio = render_events(events, root_hz, sr,
                              session=session, sydef_name=sydef_name)
    except Exception as e:
        return f"ERROR: melody render failed: {e}"
    
    if audio is None or len(audio) == 0:
        return "ERROR: Pattern produces no audio"
    
    # Ensure valid audio (no NaN/Inf)
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    
    session.set_working_buffer(audio, 'melody')
    # Also set last_buffer so /play always finds it
    session.last_buffer = audio.astype(np.float64).copy()
    
    note_count = sum(1 for e in events if e['type'] == 'note')
    dur = len(audio) / sr
    # 1 unit = 1 beat
    total_beats = sum(e['duration'] for e in events)
    sd_info = f" sydef={sydef_name}" if sydef_name else " (engine)"
    return f"MEL: {note_count} notes ({total_beats:.1f} beats = {dur:.2f}s @ {session.bpm:.0f}bpm){sd_info} -> working"


# ============================================================================
# CHORD PATTERN PARSER AND /COR COMMAND
# ============================================================================

def parse_chord_pattern(pattern: str) -> List[dict]:
    """Parse a chord pattern string into chord events.

    Syntax
    ------
    Chords are groups of comma-separated interval/MIDI values.
    Dots (.) separate chords.  Extra consecutive dots extend the
    preceding chord by 1 beat each (same rule as melodies).
    Use ``r`` for rests.

    Duration rules
    ~~~~~~~~~~~~~~
    Each chord starts with 1 beat.  The first dot after a chord is
    a separator.  Additional consecutive dots add 1 beat each.

    Examples:
      "0,4,7.0,3,7"         Major then minor, 1 beat each
      "0,4,7..0,3,7"        Major for 2 beats, minor for 1
      "0,4,7...0,3,7..."    Major for 3 beats, minor for 3
      "0,4,7.r.0,3,7"       Major, rest, minor (1 beat each)
      "0,4,7._.0,3,7"       Same using explicit rest token
    """
    events = []
    current_chord = None   # list of ints or None
    current_duration = 1

    def flush():
        nonlocal current_chord, current_duration
        if current_chord is not None:
            events.append({
                'type': 'chord',
                'pitches': current_chord,
                'duration': current_duration,
            })
        current_chord = None
        current_duration = 1

    i = 0
    dot_run = 0
    while i < len(pattern):
        ch = pattern[i]

        if ch == '.':
            dot_run += 1
            if dot_run == 1:
                # First dot = separator — flush pending chord
                flush()
            else:
                # dot_run >= 2 → extend last event
                if events:
                    events[-1]['duration'] += 1
            i += 1
            continue

        # Non-dot resets counter
        dot_run = 0

        # Rest (r or _ token)
        if ch == 'r' or ch == '_':
            flush()
            events.append({'type': 'rest', 'duration': 1})
            i += 1
            continue

        # Start of a chord group: read comma-separated numbers
        if ch.isdigit() or ch == '-' or ch == '#' or ch == 'b':
            flush()
            # Read until we hit a dot, 'r', or end
            group_str = ''
            while i < len(pattern) and pattern[i] not in ('.', 'r'):
                group_str += pattern[i]
                i += 1

            # Parse the comma-separated values
            pitches = []
            for token in group_str.split(','):
                token = token.strip()
                if not token:
                    continue
                offset = 0
                # Handle sharps/flats
                if token.startswith('#'):
                    offset = 1
                    token = token[1:]
                elif token.startswith('b'):
                    offset = -1
                    token = token[1:]
                try:
                    pitches.append(int(token) + offset)
                except ValueError:
                    pass

            if pitches:
                current_chord = pitches
                current_duration = 1
            continue

        # Skip unknown chars
        i += 1

    flush()
    return events


def cmd_cor(session: Session, args: List[str]) -> str:
    """Render a chord sequence to working buffer using the synth engine.

    Usage:
      /cor <pattern>                   Chords at 440Hz root
      /cor <pattern> <hz>              Custom root frequency
      /cor <pattern> <hz> sydef=<n> Use a SyDef patch
      /cor <pattern> sydef=<n>      Chords with SyDef at 440Hz

    Pattern syntax:
      Commas separate notes within a chord.
      Dots (.) separate chords (1 beat each).
      Extra dots (..) extend the preceding chord by 1 beat.
      Values -24 to 24 = semitone intervals from root.
      Values outside that range = MIDI note numbers.
      r = rest.

    Timing:
      1 dot-unit = 1 beat at current BPM.
      At 120 BPM each unit = 0.5s, at 60 BPM = 1.0s.

    Examples:
      /cor 0,4,7                    C major triad (1 beat)
      /cor 0,4,7..0,3,7             Cmaj 2 beats, Cmin 1 beat
      /cor 0,4,7.0,3,7.0,3,6,10    Maj, min, dim7 (1 beat each)
      /cor 60,64,67..65,69,72       MIDI: Cmaj 2 beats, Fmaj 1 beat
      /cor 0,4,7 220 sydef=pad      Pad chord at 220Hz
    """
    import numpy as np

    if not args:
        return ("ERROR: /cor needs a chord pattern.\n"
                "  Example: /cor 0,4,7...0,3,7  (major then minor)\n"
                "  Notes: commas = chord, dots = extend/separate")

    pattern = args[0]
    root_hz = 440.0
    sydef_name = None

    for arg in args[1:]:
        clean = arg.lstrip('/')
        if clean.lower().startswith('sydef='):
            sydef_name = clean.split('=', 1)[1].lower()
            continue
        try:
            root_hz = float(clean)
        except ValueError:
            # Check if it's a sydef name
            try:
                from .sydef_cmds import _ensure_sydef_attrs
                _ensure_sydef_attrs(session)
                if clean.lower() in getattr(session, 'sydefs', {}):
                    sydef_name = clean.lower()
                    continue
            except ImportError:
                pass

    events = parse_chord_pattern(pattern)
    if not events:
        return ("ERROR: No chords in pattern.\n"
                "  Example: /cor 0,4,7...2,5,9  (commas = chord notes, dots = duration)")

    sr = getattr(session, 'sample_rate', 44100)

    try:
        audio = render_chord_events(events, root_hz, sr,
                                    session=session, sydef_name=sydef_name)
    except Exception as e:
        return f"ERROR: chord render failed: {e}"

    if audio is None or len(audio) == 0:
        return "ERROR: Chord pattern produces no audio"

    # Ensure valid audio (no NaN/Inf)
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

    session.set_working_buffer(audio, 'chord')
    session.last_buffer = audio.astype(np.float64).copy()

    chord_count = sum(1 for e in events if e['type'] == 'chord')
    rest_count = sum(1 for e in events if e['type'] == 'rest')
    dur = len(audio) / sr
    # 1 unit = 1 beat
    total_beats = sum(e['duration'] for e in events)
    sd_info = f" sydef={sydef_name}" if sydef_name else " (engine)"
    rest_info = f", {rest_count} rests" if rest_count else ""
    return f"COR: {chord_count} chords{rest_info} ({total_beats:.1f} beats = {dur:.2f}s @ {session.bpm:.0f}bpm){sd_info} -> working"


# ============================================================================
# OUTPUT BLOCKS - now uses unified note_to_hz
# ============================================================================

def cmd_out_block(session: Session, args: List[str]) -> str:
    """Start an output block.
    
    Usage: /out [chain_id]
    
    Inside /out block:
      /60             MIDI note 60 (middle C)
      /0 /4 /7        Semitone intervals (-24…24 = relative, outside = MIDI)
      /amp 0.5        Set amplitude (sticky)
      /atk 0.1        Set attack (sticky)
      /c 60 64 67     Chord (MIDI notes)
      /.              Rest
      /end            Render to working buffer
    """
    state = get_dsl_state()
    if state.out_block:
        return "ERROR: Already in /out block. Type /end first."
    
    state.out_block = True
    state.out_commands = []
    state.out_chain = args[0].lower() if args else None
    
    chain_info = f" (chain: {state.out_chain})" if state.out_chain else ""
    return f"OUT: started{chain_info}. Type notes, then /end."


def process_out_line(session: Session, line: str) -> str:
    """Process a line inside /out block."""
    state = get_dsl_state()
    if not state.out_block:
        return None
    
    parts = line.split()
    results = []
    
    i = 0
    while i < len(parts):
        part = parts[i].lower().lstrip('/')
        
        if part == 'end':
            return None  # Let /end handler deal with it
        
        if part == '.':
            state.out_commands.append({'type': 'rest', 'duration': 1})
            results.append("  rest")
            i += 1
            continue
        
        if part == 'c':
            chord_notes = []
            i += 1
            while i < len(parts):
                ns = parts[i].lstrip('/')
                if ns.isdigit():
                    chord_notes.append(int(ns))
                    i += 1
                else:
                    break
            if not chord_notes and state.last_chord:
                chord_notes = state.last_chord.copy()
            if chord_notes:
                state.last_chord = chord_notes
                state.out_commands.append({
                    'type': 'chord', 'notes': chord_notes,
                    'amp': state.sticky_amp, 'atk': state.sticky_atk
                })
                results.append(f"  chord: {chord_notes}")
            continue
        
        # Sticky params
        param_map = {
            'amp': 'sticky_amp', 'atk': 'sticky_atk', 'dec': 'sticky_dec',
            'sus': 'sticky_sus', 'rel': 'sticky_rel', 'dur': 'sticky_dur',
        }
        if part in param_map and i + 1 < len(parts):
            try:
                val = float(parts[i+1].lstrip('/'))
                setattr(state, param_map[part], val)
                results.append(f"  {part}: {val}")
            except ValueError:
                pass
            i += 2
            continue
        
        # Root frequency
        if part == 'root' and i + 1 < len(parts):
            try:
                state.out_root_hz = float(parts[i+1].lstrip('/'))
                results.append(f"  root: {state.out_root_hz}Hz")
            except ValueError:
                pass
            i += 2
            continue
        
        # Note number (unified: -24…24 = semitone interval, outside = MIDI)
        if part.lstrip('-').isdigit():
            note = int(part)
            state.out_commands.append({
                'type': 'note', 'note': note,
                'amp': state.sticky_amp, 'atk': state.sticky_atk
            })
            results.append(f"  note: {note}")
            i += 1
            continue
        
        i += 1
    
    return "\n".join(results) if results else "  (ok)"


def end_out_block(session: Session) -> str:
    """End /out block, render to working buffer."""
    import numpy as np
    
    state = get_dsl_state()
    if not state.out_block:
        return None
    
    state.out_block = False
    
    sr = getattr(session, 'sample_rate', 44100)
    
    # Compute timing grid from BPM: 1 unit = 1 beat
    bpm = getattr(session, 'bpm', 120.0) if session else 120.0
    sec_per_unit = 60.0 / bpm
    root_hz = state.out_root_hz
    
    # Count
    notes = sum(1 for c in state.out_commands if c['type'] == 'note')
    chords = sum(1 for c in state.out_commands if c['type'] == 'chord')
    rests = sum(1 for c in state.out_commands if c['type'] == 'rest')
    
    total_units = sum(c.get('duration', 1) for c in state.out_commands)
    total_secs = total_units * sec_per_unit
    total_samples = int(total_secs * sr)
    
    if total_samples == 0:
        state.out_commands = []
        return "OUT END: nothing to render"
    
    audio = np.zeros(total_samples, dtype=np.float64)
    pos = 0
    
    for cmd in state.out_commands:
        event_samples = int(cmd.get('duration', 1) * sec_per_unit * sr)
        amp = cmd.get('amp', state.sticky_amp)
        atk = cmd.get('atk', state.sticky_atk)
        
        if cmd['type'] == 'note':
            freq = note_to_hz(cmd['note'], root_hz)
            if session:
                note_beats = float(cmd.get('duration', 1))
                try:
                    tone = session.generate_tone(freq, note_beats, amp)
                    if tone is not None and len(tone) > 0:
                        tone = tone.astype(np.float64)
                        if tone.ndim == 2:
                            tone = np.mean(tone, axis=1).astype(np.float64)
                        end = min(pos + len(tone), total_samples)
                        audio[pos:end] += tone[:end - pos]
                except Exception:
                    tone = render_tone(freq, cmd.get('duration', 1) * sec_per_unit, amp, sr, atk)
                    end = min(pos + len(tone), total_samples)
                    audio[pos:end] += tone[:end - pos]
            else:
                tone = render_tone(freq, cmd.get('duration', 1) * sec_per_unit, amp, sr, atk)
                end = min(pos + len(tone), total_samples)
                audio[pos:end] += tone[:end - pos]
        
        elif cmd['type'] == 'chord':
            n_notes = len(cmd['notes'])
            chord_amp = amp / max(n_notes, 1)
            for midi_note in cmd['notes']:
                freq = note_to_hz(midi_note, root_hz)
                if session:
                    note_beats = float(cmd.get('duration', 1))
                    try:
                        tone = session.generate_tone(freq, note_beats, chord_amp)
                        if tone is not None and len(tone) > 0:
                            tone = tone.astype(np.float64)
                            if tone.ndim == 2:
                                tone = np.mean(tone, axis=1).astype(np.float64)
                            end = min(pos + len(tone), total_samples)
                            audio[pos:end] += tone[:end - pos]
                    except Exception:
                        tone = render_tone(freq, cmd.get('duration', 1) * sec_per_unit, chord_amp, sr, atk)
                        end = min(pos + len(tone), total_samples)
                        audio[pos:end] += tone[:end - pos]
                else:
                    tone = render_tone(freq, cmd.get('duration', 1) * sec_per_unit, chord_amp, sr, atk)
                    end = min(pos + len(tone), total_samples)
                    audio[pos:end] += tone[:end - pos]
        
        pos += event_samples
    
    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio /= peak
    
    # Apply chain if specified
    if state.out_chain and state.out_chain in state.fx_chains:
        session.set_working_buffer(audio, 'out_block')
        _apply_chain_to_working(session, state.out_chain)
        audio = session.working_buffer
    
    session.set_working_buffer(audio, 'out_block')
    session.last_buffer = audio.astype(np.float64).copy()
    
    state.out_commands = []
    chain_info = f" via '{state.out_chain}'" if state.out_chain else ""
    state.out_chain = None
    
    return f"OUT: {notes} notes, {chords} chords, {rests} rests ({total_secs:.2f}s) -> working{chain_info}"


# ============================================================================
# SYNTH INSTANCES
# ============================================================================

def cmd_synth(session: Session, args: List[str]) -> str:
    """Create or select a synth instance.
    
    Usage:
      /synth              List synths
      /synth <n>       Create/select synth
    
    Then set params: /<n> wave saw  |  /<n> fc 2  |  /<n> detune 5
    
    Available params: wave (sin/saw/sq/tri), fc (filter count),
    vc (voice count), op (operators), detune (cents), pan (-1 to 1)
    """
    state = get_dsl_state()
    
    if not args:
        if not state.synth_instances:
            return "SYNTHS: 0: global (default) - use /synth <n> to create"
        lines = ["SYNTHS:", "  0: global (default)"]
        for i, name in enumerate(state.synth_names, 1):
            synth = state.synth_instances[name]
            marker = " *" if i == state.current_synth else ""
            wave = synth.get('params', {}).get('wave', 'sin')
            lines.append(f"  {i}: {name} [wave={wave}]{marker}")
        return "\n".join(lines)
    
    name = args[0].lower()
    
    if name in state.synth_instances:
        state.current_synth = state.synth_names.index(name) + 1
        return f"SYNTH: selected '{name}'"
    
    # Create with sensible defaults
    state.synth_instances[name] = {
        'name': name,
        'filter_count': 1,
        'voice_count': 1,
        'operators': 1,
        'params': {
            'wave': 'sin',
            'detune': 0,
            'pan': 0.0,
        },
    }
    state.synth_names.append(name)
    state.current_synth = len(state.synth_names)
    
    return f"SYNTH: created '{name}' [sin, 1 voice, 1 filter]"



# ============================================================================
# FX CHAINS
# ============================================================================

def cmd_chain(session: Session, args: List[str]) -> str:
    """Create or list FX chains.
    
    Usage:
      /chain              List all chains
      /chain <id>         Create/select chain
    
    Then: /<id> add reverb  |  /<id> add compress  |  /<id> show
    """
    state = get_dsl_state()
    
    if not args:
        if not state.fx_chains:
            return "CHAINS: (none) - use /chain <n> to create"
        lines = ["CHAINS:"]
        for cid, effects in state.fx_chains.items():
            marker = " *" if cid == state.current_chain else ""
            fx_str = " -> ".join(effects) if effects else "(empty)"
            lines.append(f"  {cid}: {fx_str}{marker}")
        return "\n".join(lines)
    
    chain_id = args[0].lower()
    
    if chain_id in state.fx_chains:
        state.current_chain = chain_id
        return f"CHAIN: selected '{chain_id}' ({len(state.fx_chains[chain_id])} effects)"
    
    state.fx_chains[chain_id] = []
    state.current_chain = chain_id
    return f"CHAIN: created '{chain_id}'"


def handle_chain_command(session: Session, chain_id: str, args: List[str]) -> str:
    """Handle /<chain_id> [subcmd].
    
    No args = start effect block.
    add <fx> = add effect.
    rm <n> = remove. clear = clear. show = show.
    """
    state = get_dsl_state()
    
    if chain_id not in state.fx_chains:
        return None
    
    if not args:
        # Start effect block
        if state.fx_block:
            return f"ERROR: Already in effect block '{state.fx_block_chain}'. /end first."
        state.fx_block = True
        state.fx_block_chain = chain_id
        effects = state.fx_chains[chain_id]
        fx_str = " -> ".join(effects) if effects else "(empty)"
        return f"FX BLOCK: [{chain_id}: {fx_str}] - build audio, then /end to apply."
    
    sub = args[0].lower()
    
    if sub == 'show':
        effects = state.fx_chains[chain_id]
        if not effects:
            return f"CHAIN '{chain_id}': (empty)"
        return f"CHAIN '{chain_id}': {' -> '.join(effects)}"
    
    if sub == 'add' and len(args) > 1:
        fx_input = args[1].lower()
        # Import resolve helper from fx_cmds
        try:
            from .fx_cmds import resolve_effect_name
            fx_name, err = resolve_effect_name(fx_input)
            if err:
                return err
        except ImportError:
            fx_name = fx_input  # Fallback: accept raw name
        state.fx_chains[chain_id].append(fx_name)
        result = f"CHAIN '{chain_id}': +{fx_name}"
        
        # Handle /and chaining
        if len(args) > 2 and args[2].lower().lstrip('/') == 'and':
            remaining = args[3:]
            if remaining and remaining[0].lower() == 'add':
                more = handle_chain_command(session, chain_id, remaining)
                if more:
                    result += "\n" + more
        return result
    
    if sub in ('rm', 'remove') and len(args) > 1:
        try:
            idx = int(args[1])
            if 0 <= idx < len(state.fx_chains[chain_id]):
                removed = state.fx_chains[chain_id].pop(idx)
                return f"CHAIN '{chain_id}': removed {removed}"
            return f"ERROR: Index {idx} out of range (chain has {len(state.fx_chains[chain_id])} effects)"
        except ValueError:
            return f"ERROR: /rm needs a number, got '{args[1]}'"
    
    if sub == 'clear':
        state.fx_chains[chain_id] = []
        return f"CHAIN '{chain_id}': cleared"
    
    return f"CHAIN '{chain_id}': unknown '{sub}'. Use: add, rm, clear, show"


def _apply_chain_to_working(session: Session, chain_id: str) -> List[str]:
    """Apply chain effects to working buffer. Returns list of applied effects."""
    state = get_dsl_state()
    
    if chain_id not in state.fx_chains:
        return []
    
    effects = state.fx_chains[chain_id]
    if not effects:
        return []
    
    executor = getattr(session, 'command_executor', None)
    if not executor:
        return []
    
    applied = []
    for fx in effects:
        try:
            result = executor(f'/fx {fx}')
            if result and 'ERROR' not in result.upper():
                applied.append(fx)
        except Exception:
            pass
    
    return applied


# ============================================================================
# OBJECT REFERENCE COMMANDS
# ============================================================================

def handle_object_command(session: Session, obj_name: str, args: List[str]) -> Optional[str]:
    """Handle /<object_name> <param> <value>."""
    state = get_dsl_state()
    obj_name = obj_name.lower()
    
    # Chain?
    if obj_name in state.fx_chains:
        return handle_chain_command(session, obj_name, args)
    
    # Synth?
    if obj_name in state.synth_instances:
        synth = state.synth_instances[obj_name]
        
        if not args:
            wave = synth.get('params', {}).get('wave', 'sin')
            return f"SYNTH '{obj_name}': wave={wave}, fc={synth['filter_count']}, vc={synth['voice_count']}"
        
        param = args[0].lower()
        
        # Known params with validation
        int_params = {'fc': 'filter_count', 'vc': 'voice_count', 'op': 'operators'}
        if param in int_params and len(args) > 1:
            try:
                val = int(args[1])
                if val < 1:
                    return f"ERROR: {param} must be >= 1"
                synth[int_params[param]] = val
                return f"SYNTH '{obj_name}': {int_params[param]} = {val}"
            except ValueError:
                return f"ERROR: {param} needs a number, got '{args[1]}'"
        
        # Wave type
        if param == 'wave' and len(args) > 1:
            wave = args[1].lower()
            valid = ('sin', 'sine', 'saw', 'sawtooth', 'sq', 'square', 'tri', 'triangle')
            if wave not in valid:
                return f"ERROR: wave must be one of: sin, saw, sq, tri. Got '{wave}'"
            synth['params']['wave'] = wave
            return f"SYNTH '{obj_name}': wave = {wave}"
        
        # Generic param
        if len(args) > 1:
            synth['params'][param] = args[1]
            return f"SYNTH '{obj_name}': {param} = {args[1]}"
        
        return f"SYNTH '{obj_name}': missing value for '{param}'"
    
    return None


# ============================================================================
# CONTROL FLOW (DEPRECATED - v52)
# ============================================================================
# The /start.../final DSL mode is deprecated as a primary interface.
# Use direct commands instead. DSL is preserved for batch/macro use cases.

def cmd_start(session: Session, args: List[str]) -> str:
    """[DEPRECATED] Start a DSL block.
    
    NOTE: DSL mode is deprecated as of v52. Commands now execute immediately
    in the main REPL for simpler, more debuggable workflow.
    
    If you need batch execution, consider:
    - User functions (/fn name ... /end)
    - SyDef definitions (/sydef name ... /end)
    - Script files
    """
    return ("DEPRECATED: /start.../final DSL mode is no longer the primary interface.\n"
            "Commands now execute immediately. Use /fn for reusable command sequences.")


def cmd_final(session: Session, args: List[str]) -> str:
    """[DEPRECATED] End DSL block.
    
    NOTE: DSL mode is deprecated as of v52.
    """
    state = get_dsl_state()
    if not state.dsl_mode:
        return "DEPRECATED: DSL mode (/start.../final) is no longer active."
    
    state.dsl_mode = False
    state.dsl_commands = []
    return "DSL mode exited. (Note: DSL is deprecated - use direct commands)"


def cmd_end_dsl(session: Session, args: List[str]) -> str:
    """End current block (/out, effect block, etc.)."""
    state = get_dsl_state()
    if state.out_block:
        return end_out_block(session)
    if state.fx_block:
        return end_fx_block(session)
    # Return None to let REPL handle (function defs, repeat blocks)
    return None


# ============================================================================
# WORKING BUFFER APPEND
# ============================================================================

def cmd_wa(session: Session, args: List[str]) -> str:
    """Append to working buffer, or commit working to a numbered buffer.
    
    Usage:
      /wa                   Commit working to lowest empty buffer
      /wa <n>               Commit working to buffer N
      /wa tone <hz> <dur>   Append tone to working (shortcut: /wa t)
      /wa silence <dur>     Append silence (shortcut: /wa s)
      /wa mel <pat> [hz]    Append melody (shortcut: /wa m)
      /wa info              Show working buffer info
    """
    import numpy as np
    
    sr = getattr(session, 'sample_rate', 44100)
    
    if not args:
        # Commit working to lowest empty buffer
        if not session.has_real_working_audio():
            return "WORKING: empty - nothing to commit. Use /mel or /wa tone first."
        
        target = session.get_lowest_empty_buffer()
        session.append_to_buffer(target, session.working_buffer.astype(np.float64))
        dur = len(session.working_buffer) / sr
        
        # Also keep in last_buffer so /play works
        session.last_buffer = session.working_buffer.astype(np.float64).copy()
        
        # Reset working
        session.working_buffer = np.zeros(sr, dtype=np.float64)
        session.working_buffer_source = 'init'
        
        return f"WA: committed {dur:.2f}s to buffer {target}"
    
    cmd = args[0].lower()
    
    # Commit to specific buffer
    try:
        target = int(cmd)
        if not session.has_real_working_audio():
            return "ERROR: Working buffer is empty - nothing to commit."
        
        session.append_to_buffer(target, session.working_buffer.astype(np.float64))
        dur = len(session.working_buffer) / sr
        buf_dur = len(session.buffers[target]) / sr
        
        session.last_buffer = session.working_buffer.astype(np.float64).copy()
        session.working_buffer = np.zeros(sr, dtype=np.float64)
        session.working_buffer_source = 'init'
        
        return f"WA: committed {dur:.2f}s to buffer {target} (now {buf_dur:.2f}s)"
    except ValueError:
        pass
    
    # Append silence (dur is in beats)
    if cmd in ('silence', 's'):
        beats = float(args[1]) if len(args) > 1 else 1.0
        dur_sec = beats * 60.0 / getattr(session, 'bpm', 120.0) if session else beats * 0.5
        silence = np.zeros(int(dur_sec * sr), dtype=np.float64)
        session.append_to_working(silence)
        total = len(session.working_buffer) / sr
        return f"WA: +{beats} beats silence ({dur_sec:.2f}s) -> working {total:.2f}s"
    
    # Append tone (dur is in beats, quantized to BPM)
    if cmd in ('tone', 't'):
        freq = float(args[1]) if len(args) > 1 else 440.0
        beats = float(args[2]) if len(args) > 2 else 1.0
        amp = float(args[3]) if len(args) > 3 else 0.5
        
        if session:
            try:
                tone = session.generate_tone(freq, beats, amp)
                if tone is not None and len(tone) > 0:
                    tone = tone.astype(np.float64)
                    if tone.ndim == 2:
                        tone = np.mean(tone, axis=1).astype(np.float64)
                    session.append_to_working(tone)
                    total = len(session.working_buffer) / sr
                    dur_sec = beats * 60.0 / session.bpm
                    return f"WA: +{freq}Hz ({beats} beats = {dur_sec:.2f}s) -> working {total:.2f}s"
            except Exception:
                pass
        # Fallback: beats -> seconds
        dur_sec = beats * 60.0 / getattr(session, 'bpm', 120.0) if session else beats * 0.5
        tone = render_tone(freq, dur_sec, amp, sr)
        session.append_to_working(tone)
        total = len(session.working_buffer) / sr
        return f"WA: +{freq}Hz ({beats} beats = {dur_sec:.2f}s) -> working {total:.2f}s"
    
    # Append melody
    if cmd in ('mel', 'm'):
        pattern = args[1] if len(args) > 1 else '0.4.7'
        root_hz = 440.0
        sydef_name = None
        for arg in args[2:]:
            if arg.lower().startswith('sydef='):
                sydef_name = arg.split('=', 1)[1].lower()
                continue
            try:
                root_hz = float(arg)
            except ValueError:
                pass
        
        events = parse_melody_pattern(pattern)
        if not events:
            return "ERROR: No notes in pattern"
        
        audio = render_events(events, root_hz, sr,
                              session=session, sydef_name=sydef_name)
        session.append_to_working(audio)
        
        note_count = sum(1 for e in events if e['type'] == 'note')
        total = len(session.working_buffer) / sr
        sd_info = f" sydef={sydef_name}" if sydef_name else ""
        return f"WA: +{note_count} notes{sd_info} -> working {total:.2f}s"
    
    # Append chord sequence
    if cmd in ('cor', 'chord', 'c'):
        pattern = args[1] if len(args) > 1 else '0,4,7'
        root_hz = 440.0
        sydef_name = None
        for arg in args[2:]:
            if arg.lower().startswith('sydef='):
                sydef_name = arg.split('=', 1)[1].lower()
                continue
            try:
                root_hz = float(arg)
            except ValueError:
                pass
        
        events = parse_chord_pattern(pattern)
        if not events:
            return "ERROR: No chords in pattern"
        
        audio = render_chord_events(events, root_hz, sr,
                                    session=session, sydef_name=sydef_name)
        session.append_to_working(audio)
        
        chord_count = sum(1 for e in events if e['type'] == 'chord')
        total = len(session.working_buffer) / sr
        sd_info = f" sydef={sydef_name}" if sydef_name else ""
        return f"WA: +{chord_count} chords{sd_info} -> working {total:.2f}s"
    
    # Info
    if cmd in ('info', 'i'):
        if session.has_real_working_audio():
            return f"WORKING: {len(session.working_buffer)/sr:.2f}s [{session.working_buffer_source}]"
        return "WORKING: empty"
    
    return f"ERROR: /wa unknown '{cmd}'. Use: tone/t, silence/s, mel/m, cor/c, info, or a buffer number."


# ============================================================================
# FX BLOCK END AND APPLY
# ============================================================================

def end_fx_block(session: Session) -> str:
    """End effect block, apply chain to working buffer."""
    state = get_dsl_state()
    if not state.fx_block:
        return None
    
    state.fx_block = False
    chain_id = state.fx_block_chain
    state.fx_block_chain = None
    
    if not chain_id or chain_id not in state.fx_chains:
        return "FX BLOCK END: no chain to apply"
    
    if not session.has_real_working_audio():
        return "FX BLOCK END: working buffer is empty - nothing to process"
    
    applied = _apply_chain_to_working(session, chain_id)
    
    sr = getattr(session, 'sample_rate', 44100)
    dur = len(session.working_buffer) / sr
    
    if applied:
        return f"FX BLOCK END: applied {' -> '.join(applied)} to {dur:.2f}s"
    return f"FX BLOCK END: chain '{chain_id}' had no applicable effects"


def cmd_apply(session: Session, args: List[str]) -> str:
    """Apply a chain's effects to working buffer (one-shot, no block needed).
    
    Usage:
      /apply <chain_id>     Apply chain to working buffer
      /apply                Apply current chain
    """
    state = get_dsl_state()
    chain_id = args[0].lower() if args else state.current_chain
    
    if not chain_id:
        return "ERROR: /apply needs a chain name. Example: /apply cook"
    
    if chain_id not in state.fx_chains:
        return f"ERROR: Chain '{chain_id}' doesn't exist. /chain {chain_id} to create."
    
    if not session.has_real_working_audio():
        return "ERROR: Working buffer is empty. Use /mel or /wa tone first."
    
    applied = _apply_chain_to_working(session, chain_id)
    
    sr = getattr(session, 'sample_rate', 44100)
    dur = len(session.working_buffer) / sr
    
    if applied:
        return f"APPLY '{chain_id}': {' -> '.join(applied)} on {dur:.2f}s"
    return f"APPLY '{chain_id}': no effects applied (chain may be empty)"


# ============================================================================
# SMART PLAY - finds audio from ANY source
# ============================================================================

def cmd_play_dsl(session: Session, args: List[str]) -> str:
    """Play audio - finds it automatically.
    
    Usage:
      /play               Auto-find audio (working > last > buffers)
      /play w             Play working buffer
      /play <n>           Play buffer N
    
    If nothing specified, plays the first thing that has audio.
    """
    import numpy as np
    
    sr = getattr(session, 'sample_rate', 44100)
    
    if args:
        target = args[0].lower()
        if target == 'w':
            if not session.has_real_working_audio():
                return "ERROR: Working buffer empty. Use /mel first."
            audio = session.working_buffer
            source = 'working'
        else:
            try:
                idx = int(target)
                if idx in session.buffers and len(session.buffers[idx]) > 0:
                    audio = session.buffers[idx]
                    source = f'buffer {idx}'
                else:
                    return f"ERROR: Buffer {idx} is empty"
            except ValueError:
                # Pass through to session.play() for volume args
                return session.play(0.8)
    else:
        # Auto-find: working > last > any buffer
        audio, source = session.get_any_audio()
        if audio is None:
            return "ERROR: No audio anywhere. Use /mel, /out, or /wa tone first."
    
    # Copy to last_buffer so session.play() and other tools can find it
    session.last_buffer = audio.astype(np.float64).copy()
    
    dur = len(audio) / sr
    return session.play(0.8)


# ============================================================================
# RENDER - output to WAV file
# ============================================================================

def cmd_render_dsl(session: Session, args: List[str]) -> str:
    """Render working buffer (or any audio) to WAV file.
    
    Usage:
      /render                   Render to auto-named file
      /render <path>            Render to specific path
      /render <n>               Render buffer N
    """
    import numpy as np
    import wave
    
    sr = getattr(session, 'sample_rate', 44100)
    
    # Determine what to render
    audio = None
    source = ''
    
    if args:
        # Check for buffer number
        try:
            idx = int(args[0])
            if idx in session.buffers and len(session.buffers[idx]) > 0:
                audio = session.buffers[idx]
                source = f'buffer_{idx}'
            else:
                return f"ERROR: Buffer {idx} is empty"
        except ValueError:
            # It's a filename
            pass
    
    if audio is None:
        audio, source = session.get_any_audio()
        if audio is None:
            return "ERROR: No audio to render. Use /mel or /wa first."
    
    # Determine output path
    if args and not args[0].isdigit():
        path = args[0]
        if not path.endswith('.wav'):
            path += '.wav'
    else:
        # Auto-name
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        temp_dir = getattr(session, 'temp_dir', '/tmp')
        path = os.path.join(temp_dir, f'mdma_{timestamp}.wav')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Normalize and write
    audio_f = audio.astype(np.float64)
    peak = np.max(np.abs(audio_f))
    if peak > 0:
        audio_f = audio_f / peak * 0.9  # -1dB headroom
    
    data_int16 = np.int16(np.clip(audio_f * 32767, -32767, 32767))
    
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data_int16.tobytes())
    
    dur = len(audio) / sr
    return f"RENDER: {source} ({dur:.2f}s) -> {path}"


# ============================================================================
# LOOP - repeat working buffer
# ============================================================================

def cmd_loop(session: Session, args: List[str]) -> str:
    """Loop the working buffer content.
    
    Usage:
      /loop              Loop 4 times (default)
      /loop <n>          Loop N times
    
    Result replaces working buffer.
    """
    import numpy as np
    
    if not session.has_real_working_audio():
        return "ERROR: Working buffer empty. Use /mel or /wa first."
    
    times = 4
    if args:
        try:
            times = int(args[0])
        except ValueError:
            return f"ERROR: /loop needs a number, got '{args[0]}'"
    
    if times < 1:
        return "ERROR: /loop needs at least 1 repetition"
    
    sr = getattr(session, 'sample_rate', 44100)
    audio = session.working_buffer
    
    looped = np.tile(audio, times)
    session.set_working_buffer(looped, 'looped')
    session.last_buffer = looped.astype(np.float64).copy()
    
    dur = len(looped) / sr
    return f"LOOP: {times}x -> working {dur:.2f}s"


# ============================================================================
# MUTATE - modify working buffer
# ============================================================================

def cmd_mutate(session: Session, args: List[str]) -> str:
    """Mutate working buffer with simple transformations.
    
    Usage:
      /mutate reverse       Reverse audio
      /mutate half          Half speed (octave down)
      /mutate double        Double speed (octave up)
      /mutate chop <n>      Chop into N pieces, shuffle
      /mutate fade          Apply fade in/out
    """
    import numpy as np
    
    if not session.has_real_working_audio():
        return "ERROR: Working buffer empty. Use /mel first."
    
    if not args:
        return "ERROR: /mutate needs a type: reverse, half, double, chop, fade"
    
    sr = getattr(session, 'sample_rate', 44100)
    audio = session.working_buffer.copy()
    mutation = args[0].lower()
    
    if mutation in ('reverse', 'rev'):
        audio = audio[::-1].copy()
        session.set_working_buffer(audio, 'mutated:reverse')
        return f"MUTATE: reversed ({len(audio)/sr:.2f}s)"
    
    if mutation == 'half':
        indices = np.linspace(0, len(audio) - 1, len(audio) * 2).astype(int)
        indices = np.clip(indices, 0, len(audio) - 1)
        audio = audio[indices]
        session.set_working_buffer(audio, 'mutated:half')
        return f"MUTATE: half speed ({len(audio)/sr:.2f}s)"
    
    if mutation == 'double':
        audio = audio[::2].copy()
        session.set_working_buffer(audio, 'mutated:double')
        return f"MUTATE: double speed ({len(audio)/sr:.2f}s)"
    
    if mutation == 'chop':
        n = int(args[1]) if len(args) > 1 else 4
        chunk_size = len(audio) // n
        if chunk_size == 0:
            return "ERROR: Audio too short to chop that many times"
        chunks = [audio[i*chunk_size:(i+1)*chunk_size] for i in range(n)]
        import random
        random.shuffle(chunks)
        audio = np.concatenate(chunks)
        session.set_working_buffer(audio, f'mutated:chop{n}')
        return f"MUTATE: chopped into {n} pieces, shuffled ({len(audio)/sr:.2f}s)"
    
    if mutation == 'fade':
        fade_len = min(int(0.1 * sr), len(audio) // 4)
        if fade_len > 0:
            audio[:fade_len] *= np.linspace(0, 1, fade_len)
            audio[-fade_len:] *= np.linspace(1, 0, fade_len)
        session.set_working_buffer(audio, 'mutated:fade')
        return f"MUTATE: faded ({len(audio)/sr:.2f}s)"
    
    return f"ERROR: Unknown mutation '{mutation}'. Use: reverse, half, double, chop, fade"


# ============================================================================
# VARIABLE EXPRESSIONS
# ============================================================================

def cmd_eq_expr(session: Session, args: List[str]) -> str:
    """Set a variable. /= name value  |  /= name a /add b"""
    if len(args) < 2:
        return "ERROR: /= needs name and value. Example: /= tempo 120"
    
    name = args[0].lower()
    stack = get_user_stack()
    
    if len(args) >= 4 and args[2].lower().lstrip('/') in ('add', 'sub', 'times', 'over', 'mul', 'div'):
        a_tok, op, b_tok = args[1], args[2].lower().lstrip('/'), args[3]
        
        def resolve(t):
            if stack.exists(t.lower()):
                return stack.get(t.lower())
            try: return int(t)
            except ValueError:
                try: return float(t)
                except ValueError: return t
        
        a, b = resolve(a_tok), resolve(b_tok)
        
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            ops = {'add': a+b, 'sub': a-b, 'times': a*b, 'mul': a*b,
                   'over': a/b if b else 0, 'div': a/b if b else 0}
            result = ops.get(op, a)
            stack.set(name, result)
            return f"SET: {name} = {result}"
        
        stack.set(name, f"{a} {op} {b}")
        return f"SET: {name} = '{a} {op} {b}'"
    
    value = args[1]
    if stack.exists(value.lower()):
        resolved = stack.get(value.lower())
        stack.set(name, resolved)
        return f"SET: {name} = {resolved}"
    
    try: value = int(args[1])
    except ValueError:
        try: value = float(args[1])
        except ValueError: pass
    
    stack.set(name, value)
    return f"SET: {name} = {value}"


def cmd_and(session: Session, args: List[str]) -> str:
    """Chain sub-commands. Used in chain building."""
    if not args:
        return "ERROR: /and needs a command after it"
    cmd_line = '/' + ' '.join(args)
    executor = getattr(session, 'command_executor', None)
    if executor:
        return executor(cmd_line)
    return f"AND: {cmd_line}"


def cmd_update(session: Session, args: List[str]) -> str:
    """Load preferences and check registry cache."""
    try:
        from ..core.user_data import load_user_preferences
        prefs = load_user_preferences()
        pref_count = len(prefs) if prefs else 0
    except Exception:
        pref_count = 0
    try:
        from ..core.song_registry import get_registry_stats
        reg_stats = get_registry_stats()
        reg_count = reg_stats.get('total', 0) if reg_stats else 0
    except Exception:
        reg_count = 0
    return f"UPDATE: {pref_count} preferences, {reg_count} registry entries"


# ============================================================================
# LIVE LOOPS
# ============================================================================

import threading

@dataclass
class LiveLoop:
    """A continuously-looping audio block."""
    name: str
    commands: List[str]
    audio: Optional[Any] = None          # np.ndarray once rendered
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    iterations: int = 0
    volume: float = 0.8
    active: bool = False

# Global live loop registry
_live_loops: Dict[str, LiveLoop] = {}

def get_live_loops() -> Dict[str, LiveLoop]:
    return _live_loops


def _render_live_commands(session: Session, commands: List[str]):
    """Execute a list of commands and return whatever ends up in working buffer.
    
    DSL block state is saved/restored so live loop rendering can't
    accidentally start new /out, /live, or /fx blocks.
    """
    import numpy as np
    
    executor = getattr(session, 'command_executor', None)
    if not executor:
        return np.zeros(0, dtype=np.float64)
    
    # Save current working buffer
    saved_wb = session.working_buffer.copy() if session.working_buffer is not None else None
    saved_src = session.working_buffer_source
    
    # Save DSL block state so renders can't leak side effects
    state = get_dsl_state()
    saved_out = state.out_block
    saved_fx = state.fx_block
    saved_live = state.live_block
    
    # Reset working for this render pass
    session.working_buffer = np.zeros(session.sample_rate, dtype=np.float64)
    session.working_buffer_source = 'init'
    
    for cmd in commands:
        try:
            executor(cmd)
        except Exception:
            pass
    
    # Grab the result
    if session.has_real_working_audio():
        result = session.working_buffer.copy()
    else:
        result = np.zeros(0, dtype=np.float64)
    
    # Restore previous working buffer
    session.working_buffer = saved_wb if saved_wb is not None else np.zeros(session.sample_rate, dtype=np.float64)
    session.working_buffer_source = saved_src if saved_src else 'init'
    
    # Restore DSL block state
    state.out_block = saved_out
    state.fx_block = saved_fx
    state.live_block = saved_live
    
    return result


def _live_loop_thread(session: Session, loop: LiveLoop):
    """Background thread that continuously plays a live loop."""
    import numpy as np
    
    sr = getattr(session, 'sample_rate', 44100)
    
    while not loop.stop_event.is_set():
        # Render the commands each cycle (allows mutations to propagate)
        audio = _render_live_commands(session, loop.commands)
        
        if audio is None or len(audio) == 0:
            # Nothing rendered, sleep briefly and retry
            loop.stop_event.wait(0.5)
            continue
        
        loop.audio = audio
        loop.iterations += 1
        
        # Play through the playback engine (non-blocking, we manage timing)
        try:
            from ..dsp.playback import PlaybackEngine
            engine = PlaybackEngine()
            engine.set_volume(loop.volume)
            
            # Normalize
            data = audio.astype(np.float64)
            peak = np.max(np.abs(data))
            if peak > 0:
                data = data / peak * 0.85
            
            engine.play(data, sr, blocking=False)
            
            # Wait for the duration of the audio (or until killed)
            duration = len(audio) / sr
            loop.stop_event.wait(duration)
            
            engine.stop()
        except Exception:
            # No audio backend - just wait the duration and count iterations
            duration = len(audio) / sr if len(audio) > 0 else 1.0
            loop.stop_event.wait(duration)
    
    loop.active = False


def cmd_live(session: Session, args: List[str]) -> str:
    """Start a live loop block.
    
    Usage:
      /live <name>          Start recording live loop
      /live                 List active live loops
    
    Inside the block, use normal DSL commands (/mel, /wa, /out, etc.)
    Then /end to start the loop playing continuously.
    
    Examples:
      /live bass
        /mel 0...0...b3...0... 110
      /end
      
      /live drums
        /wa tone 80 0.05
        /wa silence 0.2
        /wa tone 80 0.05
        /wa silence 0.45
      /end
    """
    state = get_dsl_state()
    
    if not args:
        # List active loops
        loops = get_live_loops()
        if not loops:
            return "LIVE: no active loops"
        
        sr = getattr(session, 'sample_rate', 44100)
        lines = ["LIVE LOOPS:"]
        for name, loop in loops.items():
            status = "playing" if loop.active else "stopped"
            dur = f"{len(loop.audio)/sr:.2f}s" if loop.audio is not None and len(loop.audio) > 0 else "empty"
            cmds = len(loop.commands)
            lines.append(f"  {name}: {status} ({dur}, {cmds} cmds, iter {loop.iterations})")
        return "\n".join(lines)
    
    name = args[0].lower()
    
    # Check for conflicts
    if state.out_block:
        return "ERROR: Already in /out block. /end first."
    if state.fx_block:
        return "ERROR: Already in effect block. /end first."
    if state.live_block:
        return f"ERROR: Already recording live loop '{state.live_block_name}'. /end first."
    
    # Warn if loop already exists (will be replaced)
    replaced = ""
    if name in _live_loops and _live_loops[name].active:
        _live_loops[name].stop_event.set()
        replaced = " (replacing existing)"
    
    state.live_block = True
    state.live_block_name = name
    state.live_block_commands = []
    
    return f"LIVE: recording '{name}'{replaced}. Add commands, then /end to start."


def process_live_line(session: Session, line: str) -> str:
    """Record a command inside a /live block."""
    state = get_dsl_state()
    if not state.live_block:
        return None
    
    # Store the raw command line
    state.live_block_commands.append(line)
    return f"  live[{state.live_block_name}]: {line}"


def end_live_block(session: Session) -> str:
    """End a /live block: render once to verify, then start the loop thread."""
    import numpy as np
    
    state = get_dsl_state()
    if not state.live_block:
        return None
    
    name = state.live_block_name
    commands = state.live_block_commands.copy()
    
    state.live_block = False
    state.live_block_name = None
    state.live_block_commands = []
    
    if not commands:
        return f"LIVE '{name}': no commands - loop not started"
    
    # Test-render once to verify it produces audio
    audio = _render_live_commands(session, commands)
    sr = getattr(session, 'sample_rate', 44100)
    
    if audio is None or len(audio) == 0:
        return f"LIVE '{name}': commands produced no audio - loop not started"
    
    dur = len(audio) / sr
    
    # Create the loop object
    loop = LiveLoop(
        name=name,
        commands=commands,
        audio=audio,
        active=True,
    )
    
    # Store in registry (kills any previous loop with same name)
    if name in _live_loops:
        _live_loops[name].stop_event.set()
        old_thread = _live_loops[name].thread
        if old_thread and old_thread.is_alive():
            old_thread.join(timeout=2.0)
    
    _live_loops[name] = loop
    
    # Start the loop thread
    thread = threading.Thread(
        target=_live_loop_thread,
        args=(session, loop),
        daemon=True,
        name=f"live_{name}",
    )
    loop.thread = thread
    thread.start()
    
    return f"LIVE '{name}': started ({dur:.2f}s, {len(commands)} cmds)"


def cmd_kill(session: Session, args: List[str]) -> str:
    """Kill (stop) a live loop.
    
    Usage:
      /kill <name>          Stop a specific live loop
      /kill                 List active loops (same as /live)
    
    The loop stops at the end of its current cycle.
    """
    if not args:
        loops = get_live_loops()
        if not loops:
            return "KILL: no active loops to kill"
        active = [n for n, l in loops.items() if l.active]
        if not active:
            return "KILL: no active loops"
        return f"KILL: active loops: {', '.join(active)}. Use /kill <name> or /ka"
    
    name = args[0].lower()
    
    if name not in _live_loops:
        available = list(_live_loops.keys())
        if available:
            return f"ERROR: No loop '{name}'. Active: {', '.join(available)}"
        return f"ERROR: No loop '{name}'. No loops are running."
    
    loop = _live_loops[name]
    
    if not loop.active:
        del _live_loops[name]
        return f"KILL '{name}': was already stopped (removed)"
    
    loop.stop_event.set()
    
    # Wait briefly for clean shutdown
    if loop.thread and loop.thread.is_alive():
        loop.thread.join(timeout=2.0)
    
    iters = loop.iterations
    del _live_loops[name]
    
    return f"KILL '{name}': stopped after {iters} iterations"


def cmd_ka(session: Session, args: List[str]) -> str:
    """Kill ALL live loops.
    
    Usage:
      /ka                   Stop every running live loop
    
    All loops stop at the end of their current cycle.
    """
    loops = get_live_loops()
    
    if not loops:
        return "KA: no loops running"
    
    names = list(loops.keys())
    count = 0
    
    # Signal all to stop
    for name in names:
        loops[name].stop_event.set()
    
    # Wait for clean shutdown
    for name in names:
        loop = loops[name]
        if loop.thread and loop.thread.is_alive():
            loop.thread.join(timeout=1.0)
        count += 1
    
    _live_loops.clear()
    
    return f"KA: killed {count} loop{'s' if count != 1 else ''} ({', '.join(names)})"


# ============================================================================
# COMMAND DISPATCHER
# ============================================================================

def is_dsl_command(cmd: str) -> bool:
    return cmd.lower() in DSL_COMMANDS

def dispatch_dsl_command(session: Session, cmd: str, args: List[str]) -> Optional[str]:
    """Dispatch a DSL command. Called BEFORE the main command table.
    
    UNIFIED DISPATCH RULES (v51):
    1. Block handling (live, out, fx) ALWAYS takes priority
    2. /end ALWAYS handled here for block closing
    3. Object references (chains, synths) ALWAYS handled here
    4. Core DSL commands (mel, out, wa, chain, etc.) handled here
    5. Common commands (play, render) pass through to main command table
       for consistency between DSL and regular mode
    """
    state = get_dsl_state()
    cmd_lower = cmd.lower()
    
    # Live block captures commands EXCEPT control escapes
    _LIVE_ESCAPE_CMDS = {'end', 'live', 'kill', 'ka'}
    if state.live_block and cmd_lower not in _LIVE_ESCAPE_CMDS:
        line = '/' + cmd + (' ' + ' '.join(args) if args else '')
        return process_live_line(session, line)
    
    # /live inside a live block = error
    if state.live_block and cmd_lower == 'live':
        return f"ERROR: Already recording live loop '{state.live_block_name}'. /end first."
    
    # Out block captures everything except /end
    if state.out_block and cmd_lower != 'end':
        line = '/' + cmd + (' ' + ' '.join(args) if args else '')
        return process_out_line(session, line)
    
    # /end closes active block (priority: live > out > fx > fallthrough)
    if cmd_lower == 'end':
        if state.live_block:
            return end_live_block(session)
        if state.out_block:
            return end_out_block(session)
        if state.fx_block:
            return end_fx_block(session)
        return None  # Fall through for function/repeat /end
    
    # Object reference (chain or synth name)
    result = handle_object_command(session, cmd, args)
    if result is not None:
        return result
    
    # Commands that should ALWAYS pass through to main command table
    # for unified behavior between DSL mode and regular mode
    _PASSTHROUGH_CMDS = {'play', 'render', 'rn', 'p', 'pb', 'pt', 'pts', 'pall'}
    if cmd_lower in _PASSTHROUGH_CMDS:
        dsl_debug_log(f"Command '{cmd}' passed through to main command table")
        return None
    
    # DSL command table - only for DSL-specific commands
    if cmd_lower in DSL_COMMANDS:
        func = DSL_COMMANDS[cmd_lower]
        dsl_debug_log(f"Dispatching to DSL_COMMANDS['{cmd_lower}']")
        return func(session, args)
    
    dsl_debug_log(f"Command '{cmd}' not handled by DSL, passing through")
    return None


# ============================================================================
# DSL MANAGEMENT COMMAND
# ============================================================================

def cmd_dsl(session: Session, args: List[str]) -> str:
    """DSL system management and debugging.
    
    Usage:
      /dsl                Show DSL state
      /dsl state          Show DSL state (same as above)
      /dsl debug on       Enable debug logging
      /dsl debug off      Disable debug logging
      /dsl reset          Reset DSL state to defaults
      /dsl test           Run DSL self-test
    
    Examples:
      /dsl debug on       See what DSL is doing
      /dsl state          Check current mode and blocks
      /dsl reset          Clear all DSL state
    """
    state = get_dsl_state()
    
    if not args:
        return state.get_status()
    
    subcmd = args[0].lower()
    
    if subcmd in ('state', 'status', 'st', '?'):
        return state.get_status()
    
    if subcmd == 'debug':
        if len(args) < 2:
            return f"DSL debug: {'ON' if is_dsl_debug() else 'OFF'}"
        if args[1].lower() in ('on', 'true', '1', 'yes'):
            set_dsl_debug(True)
            return "OK: DSL debug logging ENABLED"
        elif args[1].lower() in ('off', 'false', '0', 'no'):
            set_dsl_debug(False)
            return "OK: DSL debug logging DISABLED"
        return f"ERROR: Use /dsl debug on|off"
    
    if subcmd == 'reset':
        reset_dsl_state()
        return "OK: DSL state reset to defaults"
    
    if subcmd == 'test':
        return _run_dsl_self_test(session)
    
    return f"ERROR: Unknown /dsl subcommand '{subcmd}'. Use: state, debug, reset, test"


def _run_dsl_self_test(session: Session) -> str:
    """Run DSL self-test to verify functionality."""
    lines = ["=== DSL SELF-TEST ==="]
    passed = 0
    failed = 0
    
    # Test 1: State management
    try:
        state = get_dsl_state()
        reset_dsl_state()
        state = get_dsl_state()
        assert state.dsl_mode == False
        assert state.out_block == False
        lines.append("  ✓ State management")
        passed += 1
    except Exception as e:
        lines.append(f"  ✗ State management: {e}")
        failed += 1
    
    # Test 2: DSL mode toggle
    try:
        result = cmd_start(session, [])
        state = get_dsl_state()
        assert state.dsl_mode == True, f"Expected dsl_mode=True, got {state.dsl_mode}"
        result = cmd_final(session, [])
        state = get_dsl_state()
        assert state.dsl_mode == False, f"Expected dsl_mode=False, got {state.dsl_mode}"
        lines.append("  ✓ DSL mode toggle (/start, /final)")
        passed += 1
    except Exception as e:
        lines.append(f"  ✗ DSL mode toggle: {e}")
        failed += 1
    
    # Test 3: Preprocess DSL line
    try:
        # Enable DSL mode for preprocessing test
        cmd_start(session, [])
        
        result = preprocess_dsl_line('parm')
        assert result == '/parm', f"Expected '/parm', got '{result}'"
        
        result = preprocess_dsl_line('cha 0.5')
        assert result == '/cha 0.5', f"Expected '/cha 0.5', got '{result}'"
        
        result = preprocess_dsl_line('hq')
        assert result == '/hq', f"Expected '/hq', got '{result}'"
        
        result = preprocess_dsl_line('bpm 140')
        assert result == '/bpm 140', f"Expected '/bpm 140', got '{result}'"
        
        cmd_final(session, [])
        lines.append("  ✓ DSL preprocessing (parm, cha, hq, bpm)")
        passed += 1
    except Exception as e:
        lines.append(f"  ✗ DSL preprocessing: {e}")
        failed += 1
        # Make sure we exit DSL mode
        try:
            cmd_final(session, [])
        except:
            pass
    
    # Test 4: Chain creation
    try:
        result = cmd_chain(session, ['testchain'])
        state = get_dsl_state()
        assert 'testchain' in state.fx_chains, f"Chain not created"
        lines.append("  ✓ Chain creation")
        passed += 1
    except Exception as e:
        lines.append(f"  ✗ Chain creation: {e}")
        failed += 1
    
    # Test 5: dispatch_dsl_command passthrough
    try:
        # Commands not in DSL should return None
        result = dispatch_dsl_command(session, 'parm', [])
        assert result is None, f"Expected None for 'parm', got {result}"
        
        result = dispatch_dsl_command(session, 'cha', ['0.5'])
        assert result is None, f"Expected None for 'cha', got {result}"
        
        result = dispatch_dsl_command(session, 'hq', [])
        assert result is None, f"Expected None for 'hq', got {result}"
        
        lines.append("  ✓ Command passthrough (parm, cha, hq -> None)")
        passed += 1
    except Exception as e:
        lines.append(f"  ✗ Command passthrough: {e}")
        failed += 1
    
    # Test 6: Unified dispatch - play/render pass through
    try:
        # play, render should pass through to main command table
        result = dispatch_dsl_command(session, 'play', [])
        assert result is None, f"Expected None for 'play', got {result}"
        
        result = dispatch_dsl_command(session, 'render', [])
        assert result is None, f"Expected None for 'render', got {result}"
        
        result = dispatch_dsl_command(session, 'p', [])
        assert result is None, f"Expected None for 'p', got {result}"
        
        lines.append("  ✓ Unified dispatch (play, render, p -> passthrough)")
        passed += 1
    except Exception as e:
        lines.append(f"  ✗ Unified dispatch: {e}")
        failed += 1
    
    # Clean up
    reset_dsl_state()
    
    lines.append("")
    lines.append(f"Results: {passed} passed, {failed} failed")
    
    return "\n".join(lines)


# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

DSL_COMMANDS = {
    'start': cmd_start,
    'final': cmd_final,
    'end': cmd_end_dsl,
    'dsl': cmd_dsl,  # DSL management command
    'b': cmd_b,
    'mel': cmd_mel,
    'cor': cmd_cor,
    'chord': cmd_cor,
    'out': cmd_out_block,
    'wa': cmd_wa,
    'play': cmd_play_dsl,
    'render': cmd_render_dsl,
    'rn': cmd_render_dsl,
    'loop': cmd_loop,
    'mutate': cmd_mutate,
    'synth': cmd_synth,
    'chain': cmd_chain,
    'apply': cmd_apply,
    '=': cmd_eq_expr,
    'update': cmd_update,
    'and': cmd_and,
    # Live loops
    'live': cmd_live,
    'kill': cmd_kill,
    'ka': cmd_ka,
}

def get_dsl_commands() -> dict:
    return DSL_COMMANDS.copy()
