"""Pattern Commands for MDMA - Enhanced.

Commands for audio-rate pattern modulation including:
- Pattern application with multiple algorithms
- PAG command to list algorithms
- Adaptive duration (equal division based on audio length)
- Beat-based duration notation (D1, D0.5, R1, R0.5)
- Pattern block mode (/pat ... /end [algorithm])
- Arpeggio generation
- Pattern presets

DURATION NOTATION:
-----------------
D1   = hold for 1 beat
D0.5 = hold for half beat
R1   = rest for 1 beat
R0.5 = rest for half beat

ADAPTIVE MODE:
-------------
If no durations specified, notes divide audio equally.
"0 7 12" on 3-second audio = 1 second per note.

BUILD ID: pattern_cmds_v14.2_enhanced
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..core.session import Session


# ============================================================================
# BUFFER-AWARE PATTERN HELPER
# ============================================================================

def _apply_pattern_to_last_append(
    session: "Session", 
    args: List[str], 
    idx: int, 
    buf: np.ndarray, 
    append_pos: int
) -> str:
    """Apply pattern only to the last appended section of buffer.
    
    This is called when the buffer system is active and there's a
    tracked append position.
    """
    # Extract last appended section
    last_section = buf[append_pos:].copy()
    
    if len(last_section) == 0:
        return "ERROR: no appended section to process"
    
    # Temporarily set last_buffer to the section
    original_buffer = session.last_buffer
    session.last_buffer = last_section
    
    # Apply pattern using the main logic
    result = _apply_pattern_core(session, args)
    
    if result.startswith("ERROR"):
        session.last_buffer = original_buffer
        return result
    
    # Get processed section
    processed = session.last_buffer
    
    # Combine: original up to append_pos + processed section
    original_part = buf[:append_pos]
    
    if len(original_part) > 0:
        # Crossfade
        xfade = min(512, len(original_part) // 4, len(processed) // 4)
        if xfade >= 16:
            a_end = original_part[-xfade:].astype(np.float64)
            b_start = processed[:xfade].astype(np.float64)
            fade_out = np.linspace(1, 0, xfade)
            fade_in = np.linspace(0, 1, xfade)
            crossfaded = a_end * fade_out + b_start * fade_in
            combined = np.concatenate([
                original_part[:-xfade],
                crossfaded,
                processed[xfade:]
            ])
        else:
            combined = np.concatenate([original_part, processed])
    else:
        combined = processed
    
    # Update buffer system
    session.buffers[idx] = combined
    session.last_buffer = combined
    
    section_dur = len(last_section) / session.sample_rate
    processed_dur = len(processed) / session.sample_rate
    
    return f"OK: pattern applied to last appended section ({section_dur:.3f}s -> {processed_dur:.3f}s)"


def _apply_pattern_core(session: "Session", args: List[str]) -> str:
    """Core pattern application logic (used by both cmd_pat and buffer-aware version)."""
    if not args:
        return "ERROR: no pattern tokens provided"
    
    # Parse /end and algorithm if present
    algorithm = 0
    source_freq = 0.0
    pattern_tokens = []
    
    i = 0
    while i < len(args):
        arg = args[i].lower()
        if arg == '/end' or arg == 'end':
            # Get algorithm index
            if i + 1 < len(args):
                try:
                    algorithm = int(args[i + 1])
                    i += 1
                except ValueError:
                    pass
            # Get optional frequency for audio-rate mode
            if i + 1 < len(args) and algorithm == 3:
                try:
                    source_freq = float(args[i + 1])
                    i += 1
                except ValueError:
                    pass
            i += 1
            continue
        
        pattern_tokens.append(args[i])
        i += 1
    
    if not pattern_tokens:
        return "ERROR: no pattern tokens provided"
    
    # Check for preset
    if len(pattern_tokens) == 1 and pattern_tokens[0].lower().startswith('preset:'):
        preset_name = pattern_tokens[0].split(':', 1)[1]
        from ..dsp.pattern import get_preset_pattern, pattern_to_string
        notes = get_preset_pattern(preset_name)
        if not notes:
            from ..dsp.pattern import list_pattern_presets
            available = ', '.join(list_pattern_presets()[:10])
            return f"ERROR: unknown preset '{preset_name}'\n  Available: {available}..."
        tokens_str = pattern_to_string(notes)
    else:
        tokens_str = ' '.join(pattern_tokens)
    
    try:
        from ..dsp.pattern import (
            quick_pattern, ALGORITHM_INFO, 
            detect_fundamental_frequency
        )
        
        # Auto-detect frequency for audio-rate if not specified
        if algorithm == 3 and source_freq <= 0:
            source_freq = detect_fundamental_frequency(
                session.last_buffer, 
                session.sample_rate
            )
            if source_freq <= 0:
                source_freq = 100.0  # Default
        
        result = quick_pattern(
            session.last_buffer,
            tokens_str,
            bpm=session.bpm,
            sample_rate=session.sample_rate,
            algorithm=algorithm,
            source_frequency=source_freq,
        )
        session.last_buffer = result
        
        alg_name = ALGORITHM_INFO.get(algorithm, {}).get('name', 'unknown')
        freq_info = f", freq={source_freq:.1f}Hz" if algorithm == 3 else ""
        return f"OK: pattern applied ({len(pattern_tokens)} notes, alg={algorithm}:{alg_name}{freq_info}) -> {len(result)} samples"
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================================
# PAG - PATTERN ALGORITHM LIST
# ============================================================================

def cmd_pag(session: "Session", args: List[str]) -> str:
    """List all pattern algorithms with descriptions.
    
    Usage:
      /pag                   -> List all algorithms
      /pag <index>           -> Show details for specific algorithm
    
    Algorithms:
      0: general     - Equal division with smooth crossfades
      1: staccato    - Short notes (70%) with silent gaps
      2: legato      - Overlapping notes for smooth flow
      3: audiorate   - Granular repetition at detected/known frequency
      4: glide       - Smooth pitch transitions between notes
      5: stutter     - Rhythmic repetition of each note
      6: reverse     - Play pattern notes in reverse order
      7: bounce      - Forward then backward (ping-pong)
      8: random      - Randomize note order (deterministic seed)
      9: humanize    - Add timing and velocity variation
    
    Example:
      /pat 0 7 12 /end 3     -> Apply pattern with audio-rate algorithm
    """
    from ..dsp.pattern import format_algorithm_list, get_algorithm_info
    
    if args:
        try:
            idx = int(args[0])
            info = get_algorithm_info(idx)
            if info:
                lines = [f"Algorithm {idx}: {info['name']}"]
                lines.append(f"  Description: {info['desc']}")
                
                # Add algorithm-specific tips
                if idx == 0:
                    lines.append("  Usage: Default algorithm, smooth playback")
                    lines.append("  Best for: General purpose, melodic patterns")
                elif idx == 3:
                    lines.append("  Usage: /pat ... /end 3 [frequency]")
                    lines.append("  Best for: Drone effects, granular synthesis")
                    lines.append("  Tip: Specify source frequency for best results")
                elif idx == 4:
                    lines.append("  Usage: Smooth pitch glides between all notes")
                    lines.append("  Best for: Portamento, continuous pitch movement")
                elif idx == 5:
                    lines.append("  Usage: Repeats each note 4x within duration")
                    lines.append("  Best for: Stutter effects, rhythmic emphasis")
                
                return '\n'.join(lines)
            else:
                return f"ERROR: unknown algorithm index {idx}"
        except ValueError:
            pass
    
    return format_algorithm_list()


# ============================================================================
# PATTERN APPLICATION COMMANDS
# ============================================================================

def cmd_pat(session: "Session", args: List[str]) -> str:
    """Apply pattern to current buffer (or last append when using buffer system).
    
    Usage:
      /pat <tokens...>              -> Apply pattern with default algorithm (0)
      /pat <tokens...> /end <alg>   -> Apply with specific algorithm
      /pat 0 7 12 5                 -> Major chord arpeggio (adaptive timing)
      /pat 0/1 7/1 12/2             -> With explicit beat durations
      /pat 0 R 0 R                  -> Gated pattern with rests
      /pat preset:major_up          -> Use preset pattern
    
    BUFFER SYSTEM BEHAVIOR:
      When using /a to append audio, /pat applies ONLY to the last
      appended section. Use /apat to apply to the entire buffer.
    
    Pattern tokens:
      <number>      Pitch offset in semitones (0 = original pitch)
      <number>/D    With explicit beat duration (7/1 = 7 semitones, 1 beat)
      R or R<D>     Rest (R = adaptive, R1 = 1 beat, R0.5 = half beat)
      D or D<D>     Hold/extend previous note (D = adaptive, D1 = 1 beat)
      G<n>          Glide to semitone n
      V<n>:<p>      Velocity n (0-100), pitch p
    
    Duration notation:
      No duration = adaptive (divides audio equally among notes)
      D1 or /1     = 1 beat
      D0.5 or /0.5 = half beat (1/8 note at standard tempo)
      D2 or /2     = 2 beats
      D4 or /4     = 4 beats (1 bar at 4/4)
    
    Algorithms (use /pag to see all):
      0: general   - Equal division, smooth crossfades (default)
      3: audiorate - Granular repetition at source frequency
      4: glide     - Smooth pitch transitions
    
    Examples:
      /pat 0 4 7 12             -> Major arpeggio, adaptive timing
      /pat 0 4 7 12 /end 3      -> Same with audio-rate algorithm
      /pat 0/1 R/0.5 7/1.5      -> Explicit beat durations
      /pat preset:minor_arp     -> Use preset
    """
    # Check if buffer system is active with append tracking
    _use_buffer_system = (
        hasattr(session, 'buffers') and 
        hasattr(session, 'current_buffer_index') and
        hasattr(session, 'buffer_append_positions')
    )
    
    if _use_buffer_system:
        idx = session.current_buffer_index
        buf = session.buffers.get(idx)
        append_pos = session.buffer_append_positions.get(idx, 0)
        
        # If we have an append position that's not at the start,
        # apply pattern only to the last appended section
        if buf is not None and len(buf) > 0 and append_pos > 0 and append_pos < len(buf):
            return _apply_pattern_to_last_append(session, args, idx, buf, append_pos)
    
    if session.last_buffer is None:
        return "ERROR: no buffer. Generate audio first with /tone or load a file."
    
    if not args:
        return "ERROR: usage: /pat <pattern tokens...> [/end <algorithm>]\n  Example: /pat 0 7 12 /end 0"
    
    # Parse /end and algorithm if present
    algorithm = 0
    source_freq = 0.0
    pattern_tokens = []
    
    found_end = False
    i = 0
    while i < len(args):
        arg = args[i].lower()
        if arg == '/end' or arg == 'end':
            found_end = True
            # Get algorithm index
            if i + 1 < len(args):
                try:
                    algorithm = int(args[i + 1])
                    i += 1
                except ValueError:
                    pass
            # Get optional frequency for audio-rate mode
            if i + 1 < len(args) and algorithm == 3:
                try:
                    source_freq = float(args[i + 1])
                    i += 1
                except ValueError:
                    pass
            i += 1
            continue
        
        pattern_tokens.append(args[i])
        i += 1
    
    if not pattern_tokens:
        return "ERROR: no pattern tokens provided"
    
    # Check for preset
    if len(pattern_tokens) == 1 and pattern_tokens[0].lower().startswith('preset:'):
        preset_name = pattern_tokens[0].split(':', 1)[1]
        from ..dsp.pattern import get_preset_pattern, pattern_to_string
        notes = get_preset_pattern(preset_name)
        if not notes:
            from ..dsp.pattern import list_pattern_presets
            available = ', '.join(list_pattern_presets()[:10])
            return f"ERROR: unknown preset '{preset_name}'\n  Available: {available}..."
        tokens_str = pattern_to_string(notes)
    else:
        tokens_str = ' '.join(pattern_tokens)
    
    # Use core pattern application
    result = _apply_pattern_core(session, args)
    
    # Update file/clip if applicable (only if successful)
    if not result.startswith("ERROR"):
        if session.current_file and session.current_file in session.files:
            session.files[session.current_file] = session.last_buffer.copy()
        if session.current_clip and session.current_clip in session.clips:
            session.clips[session.current_clip] = session.last_buffer.copy()
        
        # Update buffer system if active
        if (hasattr(session, 'buffers') and hasattr(session, 'current_buffer_index')):
            idx = session.current_buffer_index
            if idx in session.buffers:
                session.buffers[idx] = session.last_buffer.copy()
    
    return result


def cmd_arp(session: "Session", args: List[str]) -> str:
    """Quick arpeggio from chord.
    
    Usage:
      /arp <semitones...> [repeats] [/end <alg>]
      /arp 0 4 7                    -> Major triad arp
      /arp 0 3 7 10 2               -> Minor 7th, 2 repeats
      /arp maj                      -> Major chord preset
      /arp min7 4 /end 3            -> Minor 7th, 4 repeats, audio-rate
    
    Chord presets:
      maj     -> 0 4 7 (major triad)
      min     -> 0 3 7 (minor triad)
      dim     -> 0 3 6 (diminished)
      aug     -> 0 4 8 (augmented)
      maj7    -> 0 4 7 11
      min7    -> 0 3 7 10
      dom7    -> 0 4 7 10
      sus4    -> 0 5 7
      sus2    -> 0 2 7
      add9    -> 0 4 7 14
      6       -> 0 4 7 9
      m6      -> 0 3 7 9
    """
    if session.last_buffer is None:
        return "ERROR: no buffer. Generate audio first."
    
    if not args:
        return "ERROR: usage: /arp <semitones...> [repeats] [/end <alg>]\n  Example: /arp 0 4 7"
    
    # Chord presets
    chord_presets = {
        'maj': [0, 4, 7],
        'major': [0, 4, 7],
        'min': [0, 3, 7],
        'minor': [0, 3, 7],
        'dim': [0, 3, 6],
        'diminished': [0, 3, 6],
        'aug': [0, 4, 8],
        'augmented': [0, 4, 8],
        'maj7': [0, 4, 7, 11],
        'min7': [0, 3, 7, 10],
        'dom7': [0, 4, 7, 10],
        '7': [0, 4, 7, 10],
        'sus4': [0, 5, 7],
        'sus2': [0, 2, 7],
        'add9': [0, 4, 7, 14],
        '6': [0, 4, 7, 9],
        'm6': [0, 3, 7, 9],
    }
    
    # Parse algorithm
    algorithm = 0
    end_idx = -1
    for i, arg in enumerate(args):
        if arg.lower() in ('/end', 'end'):
            end_idx = i
            if i + 1 < len(args):
                try:
                    algorithm = int(args[i + 1])
                except ValueError:
                    pass
            break
    
    # Remove /end and algorithm from args
    if end_idx >= 0:
        args = args[:end_idx]
    
    # Check for preset
    if len(args) >= 1 and args[0].lower() in chord_presets:
        chord = chord_presets[args[0].lower()]
        repeats = int(args[1]) if len(args) > 1 else 2
    else:
        # Parse semitones
        chord = []
        repeats = 2
        for i, arg in enumerate(args):
            try:
                val = int(arg)
                # If last arg and chord already has notes, might be repeats
                if i == len(args) - 1 and len(chord) >= 2 and 1 <= val <= 16:
                    repeats = val
                else:
                    chord.append(val)
            except ValueError:
                continue
        
        if not chord:
            return "ERROR: no valid semitones found"
    
    try:
        from ..dsp.pattern import arpeggiate, ALGORITHM_INFO
        result = arpeggiate(
            session.last_buffer,
            chord,
            repeats=repeats,
            bpm=session.bpm,
            sample_rate=session.sample_rate,
            algorithm=algorithm,
        )
        session.last_buffer = result
        
        chord_str = ' '.join(str(s) for s in chord)
        alg_name = ALGORITHM_INFO.get(algorithm, {}).get('name', 'general')
        return f"OK: arpeggio [{chord_str}] x{repeats} (alg={alg_name}) -> {len(result)} samples"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_patlist(session: "Session", args: List[str]) -> str:
    """List available pattern presets.
    
    Usage:
      /patlist              -> Show all presets
      /patlist <category>   -> Show presets by category
    
    Categories: scale, arp, rhythm, phrase
    """
    from ..dsp.pattern import PATTERN_PRESETS, list_pattern_presets
    
    lines = ["Pattern presets:"]
    
    # Group by prefix
    scales = []
    arps = []
    rhythms = []
    phrases = []
    other = []
    
    for name in sorted(PATTERN_PRESETS.keys()):
        pattern = PATTERN_PRESETS[name]
        if 'arp' in name:
            arps.append((name, pattern))
        elif 'up' in name or 'down' in name or 'chromatic' in name:
            scales.append((name, pattern))
        elif 'pulse' in name or 'beat' in name or 'dotted' in name:
            rhythms.append((name, pattern))
        elif 'bounce' in name or 'rise' in name or 'fall' in name:
            phrases.append((name, pattern))
        else:
            other.append((name, pattern))
    
    def format_group(title, items):
        if not items:
            return []
        result = [f"\n  {title}:"]
        for name, pattern in items:
            result.append(f"    {name:15} -> {pattern}")
        return result
    
    lines.extend(format_group("Scales", scales))
    lines.extend(format_group("Arpeggios", arps))
    lines.extend(format_group("Rhythms", rhythms))
    lines.extend(format_group("Phrases", phrases))
    lines.extend(format_group("Other", other))
    
    lines.append(f"\nTotal: {len(PATTERN_PRESETS)} presets")
    lines.append("Usage: /pat preset:<name>")
    
    return '\n'.join(lines)


def cmd_patinfo(session: "Session", args: List[str]) -> str:
    """Show pattern module info and current settings.
    
    Usage:
      /patinfo              -> Show current pattern settings
    """
    lines = ["Pattern settings:"]
    lines.append(f"  BPM: {session.bpm}")
    beat_ms = 60.0 / session.bpm * 1000
    lines.append(f"  Beat length: {beat_ms:.1f}ms")
    lines.append(f"  Sample rate: {session.sample_rate}")
    
    if session.last_buffer is not None:
        import numpy as np
        dur_sec = len(session.last_buffer) / session.sample_rate
        dur_beats = dur_sec * session.bpm / 60.0
        lines.append(f"\nCurrent buffer:")
        lines.append(f"  Length: {len(session.last_buffer)} samples ({dur_sec:.3f}s)")
        lines.append(f"  Beats: {dur_beats:.2f}")
        lines.append(f"  Peak: {np.max(np.abs(session.last_buffer)):.3f}")
        
        # Try to detect frequency
        from ..dsp.pattern import detect_fundamental_frequency
        freq = detect_fundamental_frequency(session.last_buffer, session.sample_rate)
        if freq > 0:
            lines.append(f"  Detected frequency: {freq:.1f}Hz")
    else:
        lines.append("\nNo buffer loaded. Use /tone to generate.")
    
    lines.append("\nUse /pag to see available algorithms.")
    
    return '\n'.join(lines)


# ============================================================================
# BUFFER MANIPULATION
# ============================================================================

def cmd_reverse(session: "Session", args: List[str]) -> str:
    """Reverse the current buffer.
    
    Usage:
      /reverse              -> Reverse entire buffer
      /rev                  -> Same as /reverse
    """
    if session.last_buffer is None:
        return "ERROR: no buffer to reverse"
    
    import numpy as np
    session.last_buffer = np.flip(session.last_buffer).copy()
    
    return f"OK: buffer reversed ({len(session.last_buffer)} samples)"


def cmd_chop(session: "Session", args: List[str]) -> str:
    """Chop and rearrange buffer by step indices.
    
    Usage:
      /chop <indices...>    -> Rearrange steps
      /chop 1 3 2 4         -> Reorder steps 1,3,2,4
      /chop 1 1 2 2 3 3     -> Repeat steps
      /chop 4 3 2 1         -> Reverse step order
    
    Steps are 1-indexed and based on buffer divided equally.
    """
    if session.last_buffer is None:
        return "ERROR: no buffer to chop"
    
    if not args:
        return "ERROR: usage: /chop <indices...>\n  Example: /chop 1 3 2 4"
    
    try:
        indices = [int(x) for x in args]
    except ValueError:
        return "ERROR: indices must be integers"
    
    if not indices:
        return "ERROR: no valid indices"
    
    # Calculate number of steps (max index)
    max_idx = max(indices)
    step_count = max(max_idx, len(indices))
    
    import numpy as np
    buf = session.last_buffer
    step_samples = len(buf) // step_count
    
    if step_samples < 64:
        return f"ERROR: buffer too short for {step_count} steps"
    
    # Build output
    out_parts = []
    for idx in indices:
        if 1 <= idx <= step_count:
            start = (idx - 1) * step_samples
            end = start + step_samples
            if end <= len(buf):
                out_parts.append(buf[start:end])
    
    if not out_parts:
        return "ERROR: no valid steps found"
    
    session.last_buffer = np.concatenate(out_parts)
    
    return f"OK: chopped to {len(indices)} steps ({len(session.last_buffer)} samples)"


def cmd_stretch(session: "Session", args: List[str]) -> str:
    """Time-stretch buffer without pitch change.
    
    Usage:
      /stretch <factor>     -> Stretch by factor (1.0 = no change)
      /stretch 2.0          -> Double length (slower)
      /stretch 0.5          -> Half length (faster)
    """
    if session.last_buffer is None:
        return "ERROR: no buffer to stretch"
    
    if not args:
        return "ERROR: usage: /stretch <factor>\n  Example: /stretch 1.5"
    
    try:
        factor = float(args[0])
    except ValueError:
        return f"ERROR: invalid factor '{args[0]}'"
    
    if factor <= 0:
        return "ERROR: factor must be positive"
    
    try:
        import scipy.signal
        target_len = int(len(session.last_buffer) * factor)
        if target_len < 64:
            return "ERROR: result would be too short"
        
        result = scipy.signal.resample(session.last_buffer.astype(float), target_len)
        session.last_buffer = result.astype(session.last_buffer.dtype)
        
        return f"OK: stretched x{factor:.2f} ({target_len} samples)"
    except ImportError:
        return "ERROR: scipy required for time-stretch"


def cmd_pitch(session: "Session", args: List[str]) -> str:
    """Pitch-shift entire buffer.
    
    Usage:
      /pitch <semitones>    -> Shift pitch
      /pitch 7              -> Up a fifth
      /pitch -12            -> Down an octave
    """
    if session.last_buffer is None:
        return "ERROR: no buffer to pitch-shift"
    
    if not args:
        return "ERROR: usage: /pitch <semitones>\n  Example: /pitch 7"
    
    try:
        semitones = float(args[0])
    except ValueError:
        return f"ERROR: invalid semitones '{args[0]}'"
    
    try:
        from ..dsp.pattern import pitch_shift_segment
        result = pitch_shift_segment(
            session.last_buffer,
            semitones,
            len(session.last_buffer),
        )
        session.last_buffer = result
        
        direction = "up" if semitones > 0 else "down" if semitones < 0 else "unchanged"
        return f"OK: pitch shifted {semitones:+g} semitones ({direction})"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_freq(session: "Session", args: List[str]) -> str:
    """Set or detect source frequency for audio-rate patterns.
    
    Usage:
      /freq                 -> Auto-detect frequency
      /freq <hz>            -> Set explicit frequency
      /freq 440             -> Set to 440Hz
    
    This frequency is used by algorithm 3 (audiorate) for
    determining the grain size in granular repetition.
    """
    if session.last_buffer is None:
        return "ERROR: no buffer loaded"
    
    if args:
        try:
            freq = float(args[0])
            if freq <= 0:
                return "ERROR: frequency must be positive"
            # Store in session for future pattern operations
            session._pattern_source_freq = freq
            return f"OK: source frequency set to {freq:.1f}Hz"
        except ValueError:
            return f"ERROR: invalid frequency '{args[0]}'"
    else:
        from ..dsp.pattern import detect_fundamental_frequency
        freq = detect_fundamental_frequency(session.last_buffer, session.sample_rate)
        if freq > 0:
            session._pattern_source_freq = freq
            return f"OK: detected frequency {freq:.1f}Hz"
        else:
            return "WARNING: could not detect frequency. Use /freq <hz> to set manually."


# ============================================================================
# ALIASES
# ============================================================================

cmd_rev = cmd_reverse
cmd_alg = cmd_pag


# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

def get_pattern_commands() -> dict:
    """Return pattern commands for registration."""
    return {
        # Main pattern commands
        'pat': cmd_pat,
        'pattern': cmd_pat,
        'arp': cmd_arp,
        'arpeggiate': cmd_arp,
        
        # Algorithm info
        'pag': cmd_pag,
        'alg': cmd_alg,
        
        # Pattern info
        'patlist': cmd_patlist,
        'patinfo': cmd_patinfo,
        
        # Buffer manipulation
        'reverse': cmd_reverse,
        'rev': cmd_rev,
        'chop': cmd_chop,
        'stretch': cmd_stretch,
        'pitch': cmd_pitch,
        'freq': cmd_freq,
    }
