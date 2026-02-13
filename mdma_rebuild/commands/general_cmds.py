"""General commands for the MDMA rebuild REPL.

These commands control tempo, step length and other miscellaneous
settings.  They operate on the provided Session instance and are
intended primarily for demonstration of the interface.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..core.session import Session


def cmd_bpm(session: Session, args: List[str]) -> str:
    """Get or set the session's tempo in beats per minute.

    Usage:
      /bpm            -> show current BPM
      /bpm <number>   -> set BPM to number (float)
    """
    if not args:
        return f"BPM: {session.bpm:.2f}"
    try:
        bpm = float(args[0])
        if bpm <= 0:
            return "ERROR: BPM must be positive"
        session.bpm = bpm
        return f"OK: BPM set to {session.bpm:.2f}"
    except ValueError:
        return "ERROR: invalid BPM value"


def cmd_step(session: Session, args: List[str]) -> str:
    """Get or set the session's step length in beats.

    The step length controls the granularity used by pattern commands.

    Usage:
      /step            -> show current step in beats
      /step <float>    -> set step length (beats)
    """
    if not args:
        return f"STEP: {session.step:.3f} beats"
    try:
        value = float(args[0])
        if value <= 0:
            return "ERROR: step must be positive"
        session.step = value
        return f"OK: step set to {session.step:.3f} beats"
    except ValueError:
        return "ERROR: invalid step value"


def cmd_pat(session: Session, args: List[str]) -> str:
    """Apply a pattern to the last rendered buffer or current clip.

    Usage:
      /pat <pattern tokens...> [algorithm]

    The pattern tokens are passed directly to ``Session.apply_pattern``.
    If the final token is 'general' or 'chaotic' it is treated as
    the algorithm name; otherwise 'general' is assumed.  See
    ``Session.apply_pattern`` for details.
    """
    if not args:
        return "ERROR: missing pattern arguments"
    # Detect algorithm in last token
    alg = 'general'
    tokens = list(args)
    if tokens[-1].lower() in ('general', 'chaotic'):
        alg = tokens.pop().lower()
    # Apply pattern
    msg = session.apply_pattern(tokens, algorithm=alg)
    return f"OK: {msg}"

# Alias for pattern application (/apat)
def cmd_apat(session: Session, args: List[str]) -> str:
    """Alias for /pat.  Calls the pattern command handler."""
    return cmd_pat(session, args)


# Level management
def cmd_lv(session: Session, args: List[str]) -> str:
    """Get or set the current hierarchy level.

    Valid levels: global, project, sketch, file, clip.

    Usage:
      /lv             -> show current level
      /lv <level>     -> set current level (resets deeper context)
    """
    if not args:
        return f"LEVEL: {session.current_level}"
    # Map short codes and numeric indices to hierarchy levels.  The numeric
    # indices correspond to ascending depth: 0=global, 1=project,
    # 2=sketch, 3=file and 4=clip.  Previous versions erroneously
    # mapped "1" to the global level.  Support both numbers and
    # mnemonic letters for convenience.
    level_map = {
        'g': 'global', 'global': 'global', '0': 'global',
        'p': 'project', 'project': 'project', '1': 'project',
        's': 'sketch', 'sketch': 'sketch', '2': 'sketch',
        'f': 'file', 'file': 'file', '3': 'file',
        'c': 'clip', 'clip': 'clip', '4': 'clip',
    }
    key = args[0].lower()
    target = level_map.get(key)
    if target is None:
        return "ERROR: invalid level (use global/project/sketch/file/clip or 0-5)"
    try:
        session.set_level(target)
        return f"OK: level set to {session.current_level}"
    except Exception as exc:
        return f"ERROR: {exc}"


# Synth level management
def cmd_slv(session: Session, args: List[str]) -> str:
    """Get or set the synth editing level.

    Synth levels: 1=global parameters, 2=operator parameters, 3=voice parameters.

    Usage:
      /slv         -> show current synth level
      /slv <1-3>   -> set synth level
    """
    if not args:
        return f"SYNTH LEVEL: {session.synth_level}"
    try:
        val = int(float(args[0]))
        session.set_synth_level(val)
        return f"OK: synth level set to {session.synth_level}"
    except Exception as exc:
        return f"ERROR: {exc}"


# Create new entity
def cmd_new(session: Session, args: List[str]) -> str:
    """Create a new project, sketch, file or clip based on current level.

    Usage:
      /new              -> Create with auto-generated name
      /new <name>       -> Create with specified name

    At global level creates a project and enters project level.  At
    project level creates a sketch and enters sketch level, etc.  At
    clip level this command returns an error.
    
    Examples:
      /new              -> Create "project_1" or "sketch_1" etc.
      /new my_song      -> Create "my_song" at current level
      /new bass_loop    -> Create "bass_loop" sketch/file/etc.
    """
    # Get optional name argument
    name = args[0] if args else None
    
    try:
        created = session.new_item(name=name)
        return f"OK: {created} (level is now {session.current_level})"
    except Exception as exc:
        return f"ERROR: {exc}"

# File listing and selection
def cmd_i(session: Session, args: List[str]) -> str:
    """List or select files in the current sketch.

    Usage:
      /i          -> list all files created in the current sketch
      /i <index>  -> select file by 1‑based index and set current level

    Files are named sequentially (file_1, file_2, ...).  Selecting a
    file also enters file level.  An error is returned if the index
    is out of range or no files exist.
    """
    # Listing: show files
    if not args:
        if session.file_count == 0:
            return "FILES: (none)"
        names = [f"file_{i+1}" for i in range(session.file_count)]
        return "FILES: " + ", ".join(names)
    # Selection
    try:
        idx = int(float(args[0])) - 1  # 1‑based to 0‑based index
        if idx < 0 or idx >= session.file_count:
            return "ERROR: invalid file index"
        session.current_file = f"file_{idx+1}"
        session.current_level = 'file'
        return f"OK: current file set to {session.current_file}"
    except Exception:
        return "ERROR: invalid file index"

# Insert file as clip
def cmd_ac(session: Session, args: List[str]) -> str:
    """Insert the current file (or a specified file) as a new clip.

    Usage:
      /ac         -> insert the current file as a clip
      /ac <index> -> insert file by 1‑based index as a clip

    This command increments the clip counter and sets the current
    level to 'clip'.  If no file is selected or no files exist, an
    error is returned.
    """
    # Determine which file to use
    if args:
        # Insert specified file index
        try:
            idx = int(float(args[0])) - 1
            if idx < 0 or idx >= session.file_count:
                return "ERROR: invalid file index"
            file_name = f"file_{idx+1}"
        except Exception:
            return "ERROR: invalid file index"
    else:
        # Use current file if set
        file_name = session.current_file
        if not file_name:
            return "ERROR: no current file selected"
    # Create new clip entry.  Persist the selected file's buffer as
    # the clip audio and update the current clip and level.  If the
    # file does not exist in the session.files dictionary an error
    # should be returned.  This ensures clips are stored and can be
    # referenced by /ic and other commands.
    if file_name not in session.files:
        return "ERROR: no such file to insert"
    session.clip_count += 1
    clip_name = f"clip_{session.clip_count}"
    # Copy the file buffer into the clips dict for persistence
    session.clips[clip_name] = session.files[file_name].copy()
    session.current_clip = clip_name
    session.current_level = 'clip'
    return f"OK: {file_name} inserted as {session.current_clip}"

# Clip listing and selection
def cmd_ci(session: Session, args: List[str]) -> str:
    """List or select clips in the current file.

    Usage:
      /ci          -> list all clips created
      /ci <index>  -> select clip by 1‑based index and set current level to clip
    """
    if not args:
        if session.clip_count == 0:
            return "CLIPS: (none)"
        names = [f"clip_{i+1}" for i in range(session.clip_count)]
        return "CLIPS: " + ", ".join(names)
    try:
        idx = int(float(args[0])) - 1
        if idx < 0 or idx >= session.clip_count:
            return "ERROR: invalid clip index"
        session.current_clip = f"clip_{idx+1}"
        session.current_level = 'clip'
        return f"OK: current clip set to {session.current_clip}"
    except Exception:
        return "ERROR: invalid clip index"

# Track commands
def cmd_tn(session: Session, args: List[str]) -> str:
    """Create a new track in the current project.

    Usage:
      /tn  -> create a new track and select it
    """
    if args:
        return "ERROR: /tn does not take arguments"
    msg = session.new_track()
    return f"OK: {msg} (selected {session.tracks[session.current_track_index]['name']})"


def cmd_ti(session: Session, args: List[str]) -> str:
    """List or select tracks.

    Usage:
      /ti         -> list all tracks
      /ti <index> -> select track by 1‑based index
    """
    if not args:
        names = session.list_tracks()
        return "TRACKS: " + ", ".join(f"{i+1}:{name}" for i, name in enumerate(names))
    try:
        idx = int(float(args[0])) - 1
        name = session.select_track(idx)
        return f"OK: selected track {idx+1}:{name}"
    except Exception:
        return "ERROR: invalid track index"


def cmd_ic(session: Session, args: List[str]) -> str:
    """Insert a clip or file into the current track at a beat position.

    Usage:
      /ic <beats>              -> insert current file as clip at beat position
      /ic <name> <beats>       -> insert specified clip/file at beat position

    Names refer to either a clip (clip_X) or file (file_X).
    """
    if not args:
        return "ERROR: usage /ic [<clip>|<index>|<file>] <beats>"
    # If only beat provided, use current clip, otherwise current file
    name: str
    beats_str: str
    if len(args) == 1:
        # Single argument: beat position; choose current clip first, then file
        beats_str = args[0]
        if session.current_clip:
            name = session.current_clip
        elif session.current_file:
            name = session.current_file
        else:
            return "ERROR: no current file or clip selected"
    else:
        name = args[0]
        beats_str = args[1]
    # Parse beat position
    try:
        beats = float(beats_str)
    except Exception:
        return "ERROR: invalid beat position"
    # Determine clip/file name: numeric index (1‑based) or name
    # Normalize name to lowercase for case‑insensitivity
    name_lower = name.lower()
    # Check if numeric index
    idx = None
    # Try to interpret the provided name as a numeric index.  Treat any
    # string consisting solely of digits as an index.  Do not treat
    # "0" as False; instead check for None explicitly.
    if name_lower.isdigit():
        try:
            idx = int(name_lower)
        except Exception:
            idx = None
    # If numeric index provided, map it to a clip or file name
    if idx is not None:
        if 1 <= idx <= session.clip_count:
            name_lower = f"clip_{idx}"
        elif 1 <= idx <= session.file_count:
            name_lower = f"file_{idx}"
        else:
            return "ERROR: invalid clip or file index"
    # Insert the named clip or file into the current track.  Keys are
    # stored in lower‑case internally so case sensitivity is not
    # important.  Let the session handle unknown names.
    target_name = name_lower
    try:
        msg = session.insert_clip_to_track(target_name, beats)
        return f"OK: {msg}"
    except Exception as exc:
        return f"ERROR: {exc}"


def cmd_rc(session: Session, args: List[str]) -> str:
    """Clear (reset) the current track audio.

    Legacy note: /rc used to remove clips. In the simplified continuous-track
    workflow, tracks no longer store clips, so /rc now clears the whole track.

    Usage:
      /rc            -> clear current track (silence, cursor=0)
    """
    if not session.tracks:
        return "ERROR: no tracks exist"
    track = session.tracks[session.current_track_index]
    try:
        track['audio'][:] = 0.0
    except Exception:
        track['audio'] = np.zeros(session.project_length_samples, dtype=np.float64)
    track['write_pos'] = 0
    return f"OK: cleared {track.get('name','track')}"


# ============================================================================
# PATTERN MANIPULATION COMMANDS
# ============================================================================

def cmd_dup(session: Session, args: List[str]) -> str:
    """Double the length of current buffer by looping it.
    
    Usage:
      /dup           -> Loop buffer twice (2x length)
      /dup <n>       -> Loop buffer n times
      /dup fade      -> Loop with crossfade at join points
    """
    import numpy as np
    
    if session.last_buffer is None:
        return "ERROR: no buffer to duplicate. Generate audio first."
    
    times = 2
    crossfade = False
    
    for arg in args:
        if arg.lower() in ('fade', 'xfade', 'crossfade'):
            crossfade = True
        else:
            try:
                times = int(arg)
                if times < 1:
                    times = 2
            except ValueError:
                pass
    
    buf = session.last_buffer
    
    if crossfade:
        # Crossfade at join points for smoother looping
        fade_samples = min(int(0.01 * 48000), len(buf) // 10)  # 10ms or 10% of buffer
        
        # Create output buffer
        result_len = len(buf) * times - (times - 1) * fade_samples
        result = np.zeros(result_len, dtype=np.float64)
        
        pos = 0
        for i in range(times):
            if i == 0:
                # First copy - full length
                result[:len(buf)] = buf
                pos = len(buf) - fade_samples
            else:
                # Subsequent copies - crossfade at start
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                
                # Crossfade region
                result[pos:pos + fade_samples] = (
                    result[pos:pos + fade_samples] * fade_out + 
                    buf[:fade_samples] * fade_in
                )
                
                # Rest of the buffer
                end_pos = pos + len(buf)
                if end_pos <= len(result):
                    result[pos + fade_samples:end_pos] = buf[fade_samples:]
                    pos = end_pos - fade_samples
        
        session.last_buffer = result[:pos + fade_samples]
    else:
        # Simple concatenation
        session.last_buffer = np.tile(buf, times)
    
    return f"OK: buffer duplicated {times}x ({len(session.last_buffer)} samples)"


def cmd_lk(session: Session, args: List[str]) -> str:
    """Lock pitch when stretching (preserve pitch during time stretch).
    
    Usage:
      /lk            -> Enable pitch lock (default: on)
      /lk on         -> Enable pitch lock
      /lk off        -> Disable pitch lock (pitch follows length)
    
    When pitch is locked, /st will change duration without affecting pitch.
    When unlocked, /st will change both duration and pitch (like tape speed).
    """
    if not hasattr(session, 'pitch_locked'):
        session.pitch_locked = True
    
    if not args:
        session.pitch_locked = True
        return f"PITCH LOCK: ON (pitch preserved during stretch)"
    
    arg = args[0].lower()
    if arg in ('on', '1', 'yes', 'true', 'enable'):
        session.pitch_locked = True
        return f"PITCH LOCK: ON (pitch preserved during stretch)"
    elif arg in ('off', '0', 'no', 'false', 'disable'):
        session.pitch_locked = False
        return f"PITCH LOCK: OFF (pitch follows stretch like tape)"
    else:
        return f"PITCH LOCK: {'ON' if session.pitch_locked else 'OFF'}"


def cmd_lko(session: Session, args: List[str]) -> str:
    """Unlock pitch for stretch operations (pitch follows length).
    
    Usage:
      /lko           -> Disable pitch lock
    
    This is the opposite of /lk - when unlocked, stretching changes pitch
    like changing tape speed.
    """
    if not hasattr(session, 'pitch_locked'):
        session.pitch_locked = True
    
    session.pitch_locked = False
    return f"PITCH LOCK: OFF (pitch follows stretch like tape)"


def cmd_clip(session: Session, args: List[str]) -> str:
    """Manage clips in the session.
    
    Usage:
      /clip                -> List all clips
      /clip <n>         -> Select clip by name or index
      /clip save <n>    -> Save current buffer as clip
      /clip load <n>    -> Load clip to current buffer
      /clip del <n>     -> Delete clip
      /clip dup <n>     -> Duplicate clip
      /clip info          -> Show info about current clip
    """
    import numpy as np
    
    if not args:
        # List clips
        if not session.clips:
            return "CLIPS: (none)"
        lines = ["CLIPS:"]
        for i, (name, buf) in enumerate(session.clips.items()):
            current = "*" if name == session.current_clip else " "
            dur_ms = len(buf) / session.sample_rate * 1000
            lines.append(f"  [{i}]{current} {name}: {len(buf)} samples ({dur_ms:.1f}ms)")
        return '\n'.join(lines)
    
    sub = args[0].lower()
    
    if sub == 'save':
        if len(args) < 2:
            name = f"clip_{session.clip_count}"
        else:
            name = args[1]
        if session.last_buffer is None:
            return "ERROR: no buffer to save"
        session.clips[name] = session.last_buffer.copy()
        session.current_clip = name
        session.clip_count += 1
        return f"OK: saved clip '{name}' ({len(session.last_buffer)} samples)"
    
    elif sub == 'load':
        if len(args) < 2:
            return "ERROR: specify clip name or index"
        key = args[1]
        # Try as index first
        try:
            idx = int(key)
            names = list(session.clips.keys())
            if 0 <= idx < len(names):
                key = names[idx]
        except ValueError:
            pass
        if key not in session.clips:
            return f"ERROR: clip '{key}' not found"
        session.last_buffer = session.clips[key].copy()
        session.current_clip = key
        return f"OK: loaded clip '{key}' ({len(session.last_buffer)} samples)"
    
    elif sub == 'del':
        if len(args) < 2:
            return "ERROR: specify clip name or index"
        key = args[1]
        try:
            idx = int(key)
            names = list(session.clips.keys())
            if 0 <= idx < len(names):
                key = names[idx]
        except ValueError:
            pass
        if key not in session.clips:
            return f"ERROR: clip '{key}' not found"
        del session.clips[key]
        if session.current_clip == key:
            session.current_clip = None
        return f"OK: deleted clip '{key}'"
    
    elif sub == 'dup':
        if len(args) < 2:
            if session.current_clip and session.current_clip in session.clips:
                key = session.current_clip
            else:
                return "ERROR: specify clip name or select a clip"
        else:
            key = args[1]
        try:
            idx = int(key)
            names = list(session.clips.keys())
            if 0 <= idx < len(names):
                key = names[idx]
        except ValueError:
            pass
        if key not in session.clips:
            return f"ERROR: clip '{key}' not found"
        new_name = f"{key}_dup"
        i = 1
        while new_name in session.clips:
            new_name = f"{key}_dup{i}"
            i += 1
        session.clips[new_name] = session.clips[key].copy()
        return f"OK: duplicated '{key}' as '{new_name}'"
    
    elif sub == 'info':
        if session.current_clip and session.current_clip in session.clips:
            buf = session.clips[session.current_clip]
            dur_sec = len(buf) / session.sample_rate
            dur_beats = dur_sec * session.bpm / 60.0
            max_amp = float(np.max(np.abs(buf)))
            return (f"CLIP: {session.current_clip}\n"
                   f"  Samples: {len(buf)}\n"
                   f"  Duration: {dur_sec:.3f}s ({dur_beats:.2f} beats @ {session.bpm} BPM)\n"
                   f"  Peak: {max_amp:.4f} ({20*np.log10(max_amp+1e-10):.1f} dB)")
        return "ERROR: no clip selected"
    
    else:
        # Try to select clip by name or index
        key = sub
        try:
            idx = int(key)
            names = list(session.clips.keys())
            if 0 <= idx < len(names):
                key = names[idx]
        except ValueError:
            pass
        if key in session.clips:
            session.current_clip = key
            session.last_buffer = session.clips[key].copy()
            return f"OK: selected clip '{key}'"
        return f"ERROR: clip '{key}' not found. Use /clip to list clips."


def cmd_file(session: Session, args: List[str]) -> str:
    """Manage files in the session.
    
    Usage:
      /file                -> List all files
      /file <n>         -> Select file by name or index
      /file save <n>    -> Save current buffer as file
      /file load <n>    -> Load file to current buffer
      /file export <path> -> Export current buffer to WAV file
    """
    import numpy as np
    import wave
    
    if not args:
        # List files
        if not session.files:
            return "FILES: (none)"
        lines = ["FILES:"]
        for i, (name, buf) in enumerate(session.files.items()):
            current = "*" if hasattr(session, 'current_file_buffer') and name == session.current_file_buffer else " "
            dur_ms = len(buf) / session.sample_rate * 1000
            lines.append(f"  [{i}]{current} {name}: {len(buf)} samples ({dur_ms:.1f}ms)")
        return '\n'.join(lines)
    
    sub = args[0].lower()
    
    if sub == 'save':
        if len(args) < 2:
            name = f"file_{len(session.files)}"
        else:
            name = args[1]
        if session.last_buffer is None:
            return "ERROR: no buffer to save"
        session.files[name] = session.last_buffer.copy()
        return f"OK: saved file '{name}' ({len(session.last_buffer)} samples)"
    
    elif sub == 'load':
        if len(args) < 2:
            return "ERROR: specify file name or index"
        key = args[1]
        try:
            idx = int(key)
            names = list(session.files.keys())
            if 0 <= idx < len(names):
                key = names[idx]
        except ValueError:
            pass
        if key not in session.files:
            return f"ERROR: file '{key}' not found"
        session.last_buffer = session.files[key].copy()
        return f"OK: loaded file '{key}' ({len(session.last_buffer)} samples)"
    
    elif sub == 'export':
        if len(args) < 2:
            return "ERROR: specify export path"
        path = ' '.join(args[1:])
        if session.last_buffer is None:
            return "ERROR: no buffer to export"
        try:
            buf = session.last_buffer
            # Normalize if needed
            max_val = np.max(np.abs(buf))
            if max_val > 1.0:
                buf = buf / max_val
            # Convert to int16
            audio_int16 = (buf * 32767).astype(np.int16)
            with wave.open(path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(session.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            return f"OK: exported to {path}"
        except Exception as e:
            return f"ERROR: {e}"
    
    else:
        # Try to select file by name or index
        key = sub
        try:
            idx = int(key)
            names = list(session.files.keys())
            if 0 <= idx < len(names):
                key = names[idx]
        except ValueError:
            pass
        if key in session.files:
            session.last_buffer = session.files[key].copy()
            return f"OK: loaded file '{key}'"
        return f"ERROR: file '{key}' not found. Use /file to list files."


# ============================================================================
# TIMELINE COMMANDS
# ============================================================================

def cmd_tl(session: Session, args: List[str]) -> str:
    """View timeline - show all tracks and clips.
    
    Usage:
      /tl               -> Show timeline overview
      /tl <track>       -> Show clips on specific track
      /tl beats         -> Show with beat positions
      /tl samples       -> Show with sample positions
    """
    if not session.tracks:
        return "TIMELINE: (empty). Use /track to create tracks."
    
    show_beats = True
    track_filter = None
    
    for arg in args:
        if arg.lower() == 'samples':
            show_beats = False
        elif arg.lower() == 'beats':
            show_beats = True
        else:
            try:
                track_filter = int(arg)
            except ValueError:
                pass
    
    lines = ["TIMELINE:"]
    
    for i, track in enumerate(session.tracks):
        if track_filter is not None and i != track_filter:
            continue
        
        mark = "*" if i == session.current_track_index else " "
        name = track.get('name', f'track_{i}')
        clips = track.get('clips', [])
        
        lines.append(f"  [{i}]{mark} {name}: {len(clips)} clip(s)")
        
        for j, clip_data in enumerate(clips):
            if isinstance(clip_data, dict):
                start = clip_data.get('start', 0)
                buf = clip_data.get('buffer', np.zeros(0))
                stretch = clip_data.get('stretch', 1.0)
                clip_name = clip_data.get('name', f'clip_{j}')
            else:
                start, buf = clip_data[0], clip_data[1]
                stretch = 1.0
                clip_name = f'clip_{j}'
            
            samples = len(buf)
            duration_sec = samples / session.sample_rate
            
            if show_beats:
                start_beats = start * session.bpm / 60.0 / session.sample_rate
                duration_beats = duration_sec * session.bpm / 60.0
                pos_str = f"{start_beats:.2f}b"
                dur_str = f"{duration_beats:.2f}b"
            else:
                pos_str = f"{start}"
                dur_str = f"{samples}"
            
            stretch_str = f" x{stretch:.2f}" if stretch != 1.0 else ""
            lines.append(f"      {j}: {clip_name} @ {pos_str} ({dur_str}){stretch_str}")
    
    return '\n'.join(lines)


def cmd_tlc(session: Session, args: List[str]) -> str:
    """Timeline clip operations.
    
    Usage:
      /tlc <track> <clip> del    -> Delete clip from timeline
      /tlc <track> <clip> move <beats> -> Move clip to new position
      /tlc <track> <clip> dup    -> Duplicate clip
      /tlc <track> <clip> st <factor> -> Set clip stretch
    """
    if len(args) < 3:
        return "ERROR: usage: /tlc <track> <clip> <operation> [args]"
    
    try:
        track_idx = int(args[0])
        clip_idx = int(args[1])
        operation = args[2].lower()
    except (ValueError, IndexError):
        return "ERROR: invalid track or clip index"
    
    if track_idx < 0 or track_idx >= len(session.tracks):
        return f"ERROR: track {track_idx} out of range"
    
    track = session.tracks[track_idx]
    clips = track.get('clips', [])
    
    if clip_idx < 0 or clip_idx >= len(clips):
        return f"ERROR: clip {clip_idx} out of range on track {track_idx}"
    
    if operation == 'del':
        removed = clips.pop(clip_idx)
        return f"OK: removed clip {clip_idx} from track {track_idx}"
    
    elif operation == 'move':
        if len(args) < 4:
            return "ERROR: usage: /tlc <track> <clip> move <beats>"
        try:
            new_beats = float(args[3])
            new_start = int(new_beats * 60.0 / session.bpm * session.sample_rate)
            
            clip_data = clips[clip_idx]
            if isinstance(clip_data, dict):
                clip_data['start'] = new_start
            else:
                # Convert tuple to dict
                clips[clip_idx] = {
                    'start': new_start,
                    'buffer': clip_data[1],
                    'stretch': 1.0,
                    'pitch_locked': True,
                    'effects': [],
                    'gain': 1.0,
                }
            return f"OK: moved clip {clip_idx} to {new_beats:.2f} beats"
        except ValueError:
            return f"ERROR: invalid beat position"
    
    elif operation == 'dup':
        clip_data = clips[clip_idx]
        if isinstance(clip_data, dict):
            new_clip = clip_data.copy()
            new_clip['buffer'] = clip_data['buffer'].copy()
        else:
            new_clip = {
                'start': clip_data[0],
                'buffer': clip_data[1].copy(),
                'stretch': 1.0,
                'pitch_locked': True,
                'effects': [],
                'gain': 1.0,
            }
        clips.append(new_clip)
        return f"OK: duplicated clip {clip_idx} to clip {len(clips)-1}"
    
    elif operation == 'st':
        if len(args) < 4:
            return "ERROR: usage: /tlc <track> <clip> st <factor>"
        try:
            stretch = float(args[3])
            stretch = max(0.1, min(10.0, stretch))
            
            clip_data = clips[clip_idx]
            if isinstance(clip_data, dict):
                clip_data['stretch'] = stretch
            else:
                clips[clip_idx] = {
                    'start': clip_data[0],
                    'buffer': clip_data[1],
                    'stretch': stretch,
                    'pitch_locked': getattr(session, 'pitch_locked', True),
                    'effects': [],
                    'gain': 1.0,
                }
            return f"OK: clip {clip_idx} stretch set to {stretch:.2f}x"
        except ValueError:
            return f"ERROR: invalid stretch factor"
    
    else:
        return f"ERROR: unknown operation '{operation}'. Use: del, move, dup, st"


def cmd_ins(session: Session, args: List[str]) -> str:
    """Quick insert clip to timeline.
    
    Usage:
      /ins <clip> [beats]       -> Insert clip at position (default: 0)
      /ins <clip> <beats> <stretch> -> Insert with stretch factor
      /ins buf [beats]          -> Insert current buffer as clip
    
    Examples:
      /ins kick 0               -> Insert 'kick' at beat 0
      /ins snare 2 1.5          -> Insert 'snare' at beat 2, 1.5x stretch
      /ins buf 4                -> Insert buffer at beat 4
    """
    if not args:
        return "ERROR: usage: /ins <clip> [beats] [stretch]"
    
    clip_name = args[0].lower()
    start_beats = 0.0
    stretch = getattr(session, 'clip_stretch', 1.0)
    
    if len(args) > 1:
        try:
            start_beats = float(args[1])
        except ValueError:
            pass
    
    if len(args) > 2:
        try:
            stretch = float(args[2])
            stretch = max(0.1, min(10.0, stretch))
        except ValueError:
            pass
    
    # Handle 'buf' to insert current buffer
    if clip_name == 'buf':
        if session.last_buffer is None or len(session.last_buffer) == 0:
            return "ERROR: no buffer to insert"
        # Save buffer as temporary clip
        temp_name = f"_buf_{len(session.clips)}"
        session.clips[temp_name] = session.last_buffer.copy()
        clip_name = temp_name
    
    # Check clip exists
    if clip_name not in session.clips and clip_name not in session.files:
        return f"ERROR: clip '{clip_name}' not found"
    
    # Ensure track exists
    if not session.tracks:
        session.new_track()
    
    try:
        result = session.insert_clip_to_track(
            clip_name, 
            start_beats,
            stretch=stretch,
            pitch_locked=getattr(session, 'pitch_locked', True)
        )
        return f"OK: {result}"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_tracks(session: Session, args: List[str]) -> str:
    """List or manage tracks.
    
    Usage:
      /tracks           -> List all tracks with status
      /tracks new [n]   -> Create new track(s)
      /tracks del <n>   -> Delete track
      /tracks sel <n>   -> Select track
      /tracks clear     -> Reset to one empty track
    """
    if not args:
        if not session.tracks:
            return "TRACKS: (none). Use /tracks new to create."
        parts = []
        for i, track in enumerate(session.tracks):
            mark = "*" if i == session.current_track_index else " "
            name = track.get('name', f'track_{i}')
            # Show meaningful info from new schema
            audio = track.get('audio')
            if audio is not None and hasattr(audio, 'shape'):
                if audio.ndim >= 1:
                    dur = audio.shape[0] / session.sample_rate
                    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
                else:
                    dur = 0.0
                    peak = 0.0
            else:
                dur = 0.0
                peak = 0.0
            mute_str = "M" if track.get('mute') else ""
            solo_str = "S" if track.get('solo') else ""
            flags = mute_str + solo_str
            flags_str = f"[{flags}]" if flags else ""
            pan = track.get('pan', 0.0)
            pan_str = f"L{abs(pan):.1f}" if pan < -0.05 else f"R{pan:.1f}" if pan > 0.05 else "C"
            parts.append(f"[{i+1}]{mark}{name} {dur:.1f}s pk={peak:.2f} {pan_str} {flags_str}")
        return "TRACKS:\n  " + "\n  ".join(parts)
    
    sub = args[0].lower()
    
    if sub == 'new':
        count = 1
        if len(args) > 1:
            try:
                count = int(args[1])
                count = max(1, min(16, count))
            except ValueError:
                pass
        for _ in range(count):
            session.new_track()
        return f"OK: created {count} track(s). Total: {len(session.tracks)}"
    
    elif sub == 'del':
        if len(args) < 2:
            return "ERROR: usage: /tracks del <n>"
        try:
            idx = int(args[1])
            if idx < 1 or idx > len(session.tracks):
                return f"ERROR: track {idx} out of range (1-{len(session.tracks)})"
            removed = session.tracks.pop(idx - 1)
            if not session.tracks:
                session._init_default_tracks(1)
            if session.current_track_index >= len(session.tracks):
                session.current_track_index = max(0, len(session.tracks) - 1)
            return f"OK: deleted track {idx}"
        except ValueError:
            return "ERROR: invalid track index"
    
    elif sub == 'sel':
        if len(args) < 2:
            return "ERROR: usage: /tracks sel <n>"
        try:
            idx = int(args[1])
            if idx < 1 or idx > len(session.tracks):
                return f"ERROR: track {idx} out of range (1-{len(session.tracks)})"
            session.current_track_index = idx - 1
            return f"OK: selected track {idx}"
        except ValueError:
            return "ERROR: invalid track index"
    
    elif sub == 'clear':
        session._init_default_tracks(1)
        return "OK: tracks reset to 1 empty stereo track"
    
    else:
        return f"ERROR: unknown subcommand '{sub}'. Use: new, del, sel, clear"


# ============================================================================
# PROJECT SAVE/LOAD
# ============================================================================

def cmd_save(session: Session, args: List[str]) -> str:
    """Save current project to file.
    
    Usage:
      /save              -> Save to default location (project_name.mdma)
      /save <path>       -> Save to specified path
    
    Saves all project state including:
    - Buffers and clips
    - Effects chains
    - Operators and modulation routing
    - Track structure
    - Session parameters
    
    Examples:
      /save                   -> Save as "my_project.mdma"
      /save ~/music/song.mdma -> Save to specific path
    """
    import json
    import base64
    
    # Determine save path
    if args:
        save_path = args[0]
        if not save_path.endswith('.mdma'):
            save_path += '.mdma'
    else:
        # Use project name or default
        proj_name = session.current_project or 'project'
        save_path = f"{proj_name}.mdma"
    
    # Build project data
    project_data = {
        'version': '44.0',
        'bpm': session.bpm,
        'sample_rate': session.sample_rate,
        'current_level': session.current_level,
        'project_name': session.current_project,
        'sketch_name': session.current_sketch,
        
        # Counts
        'project_count': session.project_count,
        'sketch_count': session.sketch_count,
        'file_count': session.file_count,
        'clip_count': session.clip_count,
        
        # Synthesis parameters
        'current_operator': session.current_operator,
        'voice_count': session.voice_count,
        'carrier_count': session.carrier_count,
        'dt': session.dt,
        'rand': session.rand,
        'v_mod': session.v_mod,
        
        # Envelope
        'envelope': {
            'attack': session.attack,
            'decay': session.decay,
            'sustain': session.sustain,
            'release': session.release,
        },
        
        # Per-operator envelopes
        'operator_envelopes': getattr(session, 'operator_envelopes', {}),

        # Filter state
        'filter_count': getattr(session, 'filter_count', 1),
        'selected_filter': getattr(session, 'selected_filter', 0),
        'filter_types': {str(k): v for k, v in getattr(session, 'filter_types', {}).items()},
        'filter_cutoffs': {str(k): v for k, v in getattr(session, 'filter_cutoffs', {}).items()},
        'filter_resonances': {str(k): v for k, v in getattr(session, 'filter_resonances', {}).items()},
        'filter_enabled': {str(k): v for k, v in getattr(session, 'filter_enabled', {}).items()},
        'selected_filter_envelope': getattr(session, 'selected_filter_envelope', 0),
        'filter_envelopes': getattr(session, 'filter_envelopes', {}),

        # Effects chain
        'effects': session.effects if hasattr(session, 'effects') else [],

        # FX chains
        'buffer_fx_chain': getattr(session, 'buffer_fx_chain', []),
        'track_fx_chain': getattr(session, 'track_fx_chain', []),
        'master_fx_chain': getattr(session, 'master_fx_chain', []),
        'file_fx_chain': getattr(session, 'file_fx_chain', []),
        
        # Tracks (serialize audio as base64)
        'tracks': [],
        'current_track_index': session.current_track_index,
        
        # Operators (convert for JSON)
        'operators': {},
        
        # Modulation algorithms
        'algorithms': [],
    }
    
    # Serialize operator data
    if hasattr(session, 'engine') and hasattr(session.engine, 'operators'):
        for idx, op in session.engine.operators.items():
            project_data['operators'][str(idx)] = {
                'wave': op.get('wave', 'sine'),
                'freq': op.get('freq', 440.0),
                'amp': op.get('amp', 1.0),
                'phase': op.get('phase', 0.0),
            }
    
    # Serialize algorithms
    if hasattr(session, 'engine') and hasattr(session.engine, 'algorithms'):
        project_data['algorithms'] = [
            {'type': t, 'source': s, 'target': tg, 'amount': a}
            for t, s, tg, a in session.engine.algorithms
        ]
    
    # Serialize tracks (audio as base64, metadata as plain JSON)
    tracks_serialized = []
    for t in session.tracks:
        td = {
            'name': t.get('name', ''),
            'fx_chain': t.get('fx_chain', []),
            'write_pos': t.get('write_pos', 0),
            'gain': float(t.get('gain', 1.0)),
            'pan': float(t.get('pan', 0.0)),
            'mute': bool(t.get('mute', False)),
            'solo': bool(t.get('solo', False)),
        }
        audio = t.get('audio')
        if audio is not None and hasattr(audio, '__len__') and len(audio) > 0:
            # Check if there's actual non-silent audio
            peak = float(np.max(np.abs(audio)))
            if peak > 1e-7:
                td['audio_b64'] = base64.b64encode(
                    audio.astype('float32').tobytes()).decode('ascii')
                td['audio_shape'] = list(audio.shape)
        tracks_serialized.append(td)
    project_data['tracks'] = tracks_serialized

    # Encode buffers as base64
    buffers_data = {}
    if hasattr(session, 'buffers'):
        for idx, buf in session.buffers.items():
            if buf is not None and len(buf) > 0:
                # Convert to bytes and base64 encode
                buf_bytes = buf.astype('float32').tobytes()
                buffers_data[str(idx)] = base64.b64encode(buf_bytes).decode('ascii')
    project_data['buffers'] = buffers_data
    
    # Encode clips
    clips_data = {}
    if hasattr(session, 'clips'):
        for name, buf in session.clips.items():
            if buf is not None and len(buf) > 0:
                buf_bytes = buf.astype('float32').tobytes()
                clips_data[name] = base64.b64encode(buf_bytes).decode('ascii')
    project_data['clips'] = clips_data
    
    # Serialize SyDefs (user-created only, factory excluded)
    sydefs_data = {}
    if hasattr(session, 'sydefs'):
        factory_names = set(getattr(session, '_factory_sydef_names', []))
        for name, sd in session.sydefs.items():
            if name not in factory_names and hasattr(sd, 'to_dict'):
                sydefs_data[name] = sd.to_dict()
    project_data['sydefs'] = sydefs_data

    # Serialize user functions (/fn blocks)
    if hasattr(session, 'user_functions') and session.user_functions:
        project_data['user_functions'] = dict(session.user_functions)

    # Serialize named chains
    chains_data = {}
    if hasattr(session, 'chains'):
        for cname, clist in session.chains.items():
            chains_data[cname] = list(clist)
    project_data['chains'] = chains_data

    # Phase T state
    project_data['sections'] = getattr(session, 'sections', [])
    project_data['master_gain'] = getattr(session, 'master_gain', 0.0)
    project_data['autosave_enabled'] = getattr(session, '_autosave_enabled', False)
    project_data['autosave_interval'] = getattr(session, '_autosave_interval', 5)
    # Snapshots (parameter-only, lightweight)
    project_data['snapshots'] = getattr(session, '_snapshots', [])

    # Serialize imported data / working buffer source info
    project_data['working_buffer_source'] = getattr(
        session, 'working_buffer_source', 'init')

    # Encode working buffer if non-trivial
    if hasattr(session, 'working_buffer') and session.working_buffer is not None:
        wb = session.working_buffer
        if hasattr(wb, '__len__') and len(wb) > 0:
            rms = float(np.sqrt(np.mean(wb.astype(float) ** 2)))
            if rms > 1e-7:  # only save if not silence
                wb_bytes = wb.astype('float32').tobytes()
                project_data['working_buffer'] = base64.b64encode(
                    wb_bytes).decode('ascii')

    # Write to file
    try:
        def _json_default(obj):
            """JSON fallback for numpy and other non-standard types."""
            if hasattr(obj, 'tobytes'):  # numpy array
                import base64 as _b64
                return {'__ndarray__': True,
                        'dtype': str(obj.dtype),
                        'shape': list(obj.shape),
                        'data': _b64.b64encode(obj.astype('float32').tobytes()).decode('ascii')}
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            return str(obj)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2, default=_json_default)
        
        # Calculate file size
        import os
        size = os.path.getsize(save_path)
        size_str = f"{size/1024:.1f}KB" if size > 1024 else f"{size}B"
        
        return f"OK: saved project to '{save_path}' ({size_str})"
    except Exception as e:
        return f"ERROR: failed to save - {e}"


def cmd_load(session: Session, args: List[str]) -> str:
    """Load project from file.
    
    Usage:
      /load <path>       -> Load project from file
    
    Restores all project state from saved .mdma file.
    
    Examples:
      /load my_project.mdma
      /load ~/music/song.mdma
    """
    import json
    import base64
    import numpy as np
    
    if not args:
        return "ERROR: usage: /load <path>"
    
    load_path = args[0]
    if not load_path.endswith('.mdma'):
        load_path += '.mdma'
    
    try:
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return f"ERROR: file not found: {load_path}"
    except json.JSONDecodeError as e:
        return f"ERROR: invalid project file - {e}"
    
    # Restore basic parameters
    session.bpm = data.get('bpm', 120.0)
    session.current_level = data.get('current_level', 'global')
    session.current_project = data.get('project_name')
    session.current_sketch = data.get('sketch_name')
    
    # Restore counts
    session.project_count = data.get('project_count', 0)
    session.sketch_count = data.get('sketch_count', 0)
    session.file_count = data.get('file_count', 0)
    session.clip_count = data.get('clip_count', 0)
    
    # Restore synthesis
    session.current_operator = data.get('current_operator', 0)
    session.voice_count = data.get('voice_count', 1)
    session.carrier_count = data.get('carrier_count', 1)
    session.dt = data.get('dt', 0.0)
    session.rand = data.get('rand', 0.0)
    session.v_mod = data.get('v_mod', 0.0)
    
    # Restore envelope
    env = data.get('envelope', {})
    session.attack = env.get('attack', 0.01)
    session.decay = env.get('decay', 0.1)
    session.sustain = env.get('sustain', 0.8)
    session.release = env.get('release', 0.2)
    
    # Restore per-operator envelopes
    session.operator_envelopes = data.get('operator_envelopes', {})
    # Convert string keys back to int (JSON serialization may stringify them)
    if session.operator_envelopes:
        session.operator_envelopes = {
            int(k): v for k, v in session.operator_envelopes.items()
        }

    # Restore filter state
    session.filter_count = data.get('filter_count', 1)
    session.selected_filter = data.get('selected_filter', 0)
    ft = data.get('filter_types')
    if ft:
        session.filter_types = {int(k): v for k, v in ft.items()}
    fc = data.get('filter_cutoffs')
    if fc:
        session.filter_cutoffs = {int(k): v for k, v in fc.items()}
    fr = data.get('filter_resonances')
    if fr:
        session.filter_resonances = {int(k): v for k, v in fr.items()}
    fe = data.get('filter_enabled')
    if fe:
        session.filter_enabled = {int(k): v for k, v in fe.items()}
    session.selected_filter_envelope = data.get('selected_filter_envelope', 0)
    fenv = data.get('filter_envelopes', {})
    if fenv:
        session.filter_envelopes = {int(k): v for k, v in fenv.items()}

    # Restore effects
    session.effects = data.get('effects', [])

    # Restore FX chains
    session.buffer_fx_chain = data.get('buffer_fx_chain', [])
    session.track_fx_chain = data.get('track_fx_chain', [])
    session.master_fx_chain = data.get('master_fx_chain', [])
    session.file_fx_chain = data.get('file_fx_chain', [])
    
    # Restore tracks (deserialize audio from base64)
    raw_tracks = data.get('tracks', [])
    restored_tracks = []
    for td in raw_tracks:
        track = {
            'name': td.get('name', ''),
            'fx_chain': td.get('fx_chain', []),
            'write_pos': td.get('write_pos', 0),
            'gain': float(td.get('gain', 1.0)),
            'pan': float(td.get('pan', 0.0)),
            'mute': bool(td.get('mute', False)),
            'solo': bool(td.get('solo', False)),
        }
        if 'audio_b64' in td:
            buf_bytes = base64.b64decode(td['audio_b64'])
            shape = td.get('audio_shape', None)
            arr = np.frombuffer(buf_bytes, dtype='float32').astype('float64')
            if shape and len(shape) == 2:
                arr = arr.reshape(shape)
            track['audio'] = arr
        elif 'audio' in td and td['audio'] is not None:
            audio_val = td['audio']
            if isinstance(audio_val, dict) and audio_val.get('__ndarray__'):
                # v43 format: base64-encoded ndarray
                buf_bytes = base64.b64decode(audio_val['data'])
                arr = np.frombuffer(buf_bytes, dtype='float32').astype('float64')
                shape = audio_val.get('shape')
                if shape and len(shape) == 2:
                    arr = arr.reshape(shape)
                track['audio'] = arr
            else:
                # Legacy: audio stored as list
                track['audio'] = np.array(audio_val, dtype='float64')
        else:
            # Allocate empty track (30s default)
            track['audio'] = np.zeros(
                (int(session.sample_rate * 30), 2), dtype=np.float64)
        restored_tracks.append(track)
    if restored_tracks:
        session.tracks = restored_tracks
    session.current_track_index = data.get('current_track_index', 0)
    
    # Restore operators
    if hasattr(session, 'engine'):
        session.engine.operators = {}
        for idx_str, op_data in data.get('operators', {}).items():
            idx = int(idx_str)
            session.engine.set_operator(
                idx,
                op_data.get('wave', 'sine'),
                op_data.get('freq', 440.0),
                op_data.get('amp', 1.0),
                op_data.get('phase', 0.0)
            )
        
        # Restore algorithms
        session.engine.algorithms = []
        for alg in data.get('algorithms', []):
            session.engine.algorithms.append((
                alg['type'], alg['source'], alg['target'], alg['amount']
            ))
    
    # Restore buffers
    if hasattr(session, 'buffers'):
        session.buffers = {}
        for idx_str, b64_data in data.get('buffers', {}).items():
            idx = int(idx_str)
            buf_bytes = base64.b64decode(b64_data)
            session.buffers[idx] = np.frombuffer(buf_bytes, dtype='float32').astype('float64')
        
        # Set last_buffer to first buffer if available
        if session.buffers:
            first_idx = min(session.buffers.keys())
            session.last_buffer = session.buffers[first_idx].copy()
    
    # Restore clips
    session.clips = {}
    for name, b64_data in data.get('clips', {}).items():
        buf_bytes = base64.b64decode(b64_data)
        session.clips[name] = np.frombuffer(buf_bytes, dtype='float32').astype('float64')
    
    # Restore SyDefs
    restored_parts = []
    sydefs_data = data.get('sydefs', {})
    if sydefs_data:
        try:
            from mdma_rebuild.commands.sydef_cmds import SyDef as _SD
            for name, d in sydefs_data.items():
                session.sydefs[name.lower()] = _SD.from_dict(d)
            restored_parts.append(f"{len(sydefs_data)} sydefs")
        except Exception:
            pass

    # Restore user functions
    ufn = data.get('user_functions', {})
    if ufn:
        if not hasattr(session, 'user_functions'):
            session.user_functions = {}
        session.user_functions.update(ufn)
        restored_parts.append(f"{len(ufn)} functions")

    # Restore chains
    chains = data.get('chains', {})
    if chains:
        if not hasattr(session, 'chains'):
            session.chains = {}
        session.chains.update(chains)
        restored_parts.append(f"{len(chains)} chains")

    # Restore Phase T state
    session.sections = data.get('sections', [])
    session.master_gain = data.get('master_gain', 0.0)
    session._autosave_enabled = data.get('autosave_enabled', False)
    session._autosave_interval = data.get('autosave_interval', 5)
    session._snapshots = data.get('snapshots', [])
    # Clear undo/redo stacks on load (fresh session)
    session._undo_stack = []
    session._redo_stack = []
    session._track_undo_stacks = {}
    session._track_redo_stacks = {}

    # Cancel any existing autosave timer and restart if enabled
    old_timer = getattr(session, '_autosave_timer', None)
    if old_timer is not None:
        old_timer.cancel()
        session._autosave_timer = None
    if session._autosave_enabled:
        try:
            from .phase_t_cmds import _autosave_tick
            import threading
            t = threading.Timer(session._autosave_interval * 60,
                                _autosave_tick, args=(session,))
            t.daemon = True
            session._autosave_timer = t
            t.start()
        except ImportError:
            pass

    # Restore working buffer
    wb_b64 = data.get('working_buffer')
    if wb_b64:
        try:
            wb_bytes = base64.b64decode(wb_b64)
            session.working_buffer = np.frombuffer(
                wb_bytes, dtype='float32').astype(np.float64)
            session.working_buffer_source = data.get(
                'working_buffer_source', 'loaded')
            restored_parts.append("working buffer")
        except Exception:
            pass

    version = data.get('version', 'unknown')
    extra = f" + {', '.join(restored_parts)}" if restored_parts else ""
    return f"OK: loaded '{load_path}' (v{version}){extra}"


def cmd_import(session: Session, args: List[str]) -> str:
    """Import an audio or data file into the session.

    Usage:
      /import <path>           Import audio file (wav/mp3/flac) into working buffer
      /import <path> track     Import directly to current track
      /import <path> sydef     Import SyDef definitions from JSON
      /import <path> project   Merge another .mdma project into this one

    Supported audio formats: .wav, .mp3, .flac, .ogg, .aiff
    Supported data formats: .json (sydefs, chains, functions), .mdma (project merge)

    Examples:
      /import kick.wav              Load kick into working buffer
      /import synth_patches.json    Import SyDef definitions
      /import other_project.mdma    Merge project data
    """
    import os

    if not args:
        return "ERROR: usage: /import <path> [track|sydef|project]"

    file_path = args[0]
    target = args[1].lower() if len(args) > 1 else 'auto'

    if not os.path.isfile(file_path):
        return f"ERROR: file not found: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()

    # --- Audio import ---
    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif'}
    if ext in audio_exts or target == 'track':
        return _import_audio(session, file_path, target)

    # --- JSON data import (sydefs, chains, functions) ---
    if ext == '.json' or target == 'sydef':
        return _import_json_data(session, file_path)

    # --- Project merge ---
    if ext == '.mdma' or target == 'project':
        return _import_project_merge(session, file_path)

    return f"ERROR: unsupported file type '{ext}'. Use .wav/.mp3/.flac for audio, .json for data, .mdma for project merge."


def _import_audio(session, file_path: str, target: str) -> str:
    """Import audio file into working buffer or track."""
    import os
    try:
        import wave
        import struct

        with wave.open(file_path, 'rb') as wf:
            n_ch = wf.getnchannels()
            swidth = wf.getsampwidth()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if swidth == 2:
            samples = np.array(struct.unpack(f'<{n_frames * n_ch}h', raw),
                               dtype=np.float64) / 32768.0
        elif swidth == 3:
            # 24-bit
            samples = np.zeros(n_frames * n_ch, dtype=np.float64)
            for i in range(n_frames * n_ch):
                b = raw[i*3:(i+1)*3]
                val = int.from_bytes(b, 'little', signed=True)
                samples[i] = val / 8388608.0
        elif swidth == 4:
            samples = np.array(struct.unpack(f'<{n_frames * n_ch}i', raw),
                               dtype=np.float64) / 2147483648.0
        else:
            samples = np.array(struct.unpack(f'<{n_frames * n_ch}B', raw),
                               dtype=np.float64) / 128.0 - 1.0

        # Convert to stereo if needed
        if n_ch == 1:
            audio = np.column_stack([samples, samples])
        elif n_ch == 2:
            audio = samples.reshape(-1, 2)
        else:
            audio = samples.reshape(-1, n_ch)[:, :2]

        # Resample if needed (T.8 — improved sample rate conversion)
        session_sr = getattr(session, 'sample_rate', 48000)
        if sr != session_sr:
            try:
                from .phase_t_cmds import resample_audio
                audio = resample_audio(audio, sr, session_sr)
            except ImportError:
                # Fallback to linear interpolation
                ratio = session_sr / sr
                new_len = int(len(audio) * ratio)
                x_old = np.linspace(0, 1, len(audio))
                x_new = np.linspace(0, 1, new_len)
                resampled = np.column_stack([
                    np.interp(x_new, x_old, audio[:, 0]),
                    np.interp(x_new, x_old, audio[:, 1]),
                ])
                audio = resampled

        dur = len(audio) / session_sr

        if target == 'track':
            # Write directly to track
            try:
                session.write_to_track(audio, mode='overwrite')
                return f"IMPORT: {dur:.2f}s audio -> track {session.current_track_index + 1}"
            except Exception as e:
                return f"ERROR: failed to write to track: {e}"

        # Default: working buffer (mono)
        mono = np.mean(audio, axis=1).astype(np.float64)
        session.working_buffer = mono
        session.working_buffer_source = f'import:{os.path.basename(file_path)}'
        session.last_buffer = mono.copy()

        return f"IMPORT: {dur:.2f}s audio -> working buffer ({os.path.basename(file_path)})"

    except Exception as e:
        return f"ERROR: failed to import audio: {e}"


def _import_json_data(session, file_path: str) -> str:
    """Import SyDef/chain/function definitions from JSON."""
    import json as _json
    import os

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = _json.load(f)
    except Exception as e:
        return f"ERROR: failed to read JSON: {e}"

    parts = []

    # Detect format: could be sydefs, chains, functions, or a mixed export
    if 'sydefs' in data:
        # Mixed format
        sd_data = data['sydefs']
    elif all(isinstance(v, dict) and ('commands' in v or 'params' in v) for v in data.values()):
        # Looks like pure SyDef data
        sd_data = data
    else:
        sd_data = {}

    if sd_data:
        try:
            from mdma_rebuild.commands.sydef_cmds import SyDef as _SD
            count = 0
            for name, d in sd_data.items():
                # Ensure 'name' key is present (key may be the name)
                if 'name' not in d:
                    d = dict(d)
                    d['name'] = name
                session.sydefs[name.lower()] = _SD.from_dict(d)
                count += 1
            parts.append(f"{count} sydefs")
        except Exception:
            pass

    # Chains
    ch_data = data.get('chains', {})
    if ch_data:
        if not hasattr(session, 'chains'):
            session.chains = {}
        session.chains.update(ch_data)
        parts.append(f"{len(ch_data)} chains")

    # Functions
    fn_data = data.get('user_functions', data.get('functions', {}))
    if fn_data and isinstance(fn_data, dict):
        if not hasattr(session, 'user_functions'):
            session.user_functions = {}
        session.user_functions.update(fn_data)
        parts.append(f"{len(fn_data)} functions")

    if not parts:
        return f"ERROR: no recognised data in {file_path}"

    return f"IMPORT: {', '.join(parts)} from {os.path.basename(file_path)}"


def _import_project_merge(session, file_path: str) -> str:
    """Merge another .mdma project's definitions (no audio replacement)."""
    import json as _json
    import os

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = _json.load(f)
    except Exception as e:
        return f"ERROR: {e}"

    parts = []

    # Merge SyDefs
    sd = data.get('sydefs', {})
    if sd:
        try:
            from mdma_rebuild.commands.sydef_cmds import SyDef as _SD
            for name, d in sd.items():
                if 'name' not in d:
                    d = dict(d)
                    d['name'] = name
                session.sydefs[name.lower()] = _SD.from_dict(d)
            parts.append(f"{len(sd)} sydefs")
        except Exception:
            pass

    # Merge chains
    ch = data.get('chains', {})
    if ch:
        if not hasattr(session, 'chains'):
            session.chains = {}
        session.chains.update(ch)
        parts.append(f"{len(ch)} chains")

    # Merge functions
    fn = data.get('user_functions', {})
    if fn:
        if not hasattr(session, 'user_functions'):
            session.user_functions = {}
        session.user_functions.update(fn)
        parts.append(f"{len(fn)} functions")

    if not parts:
        return f"IMPORT: no mergeable data in {os.path.basename(file_path)}"

    return f"IMPORT: merged {', '.join(parts)} from {os.path.basename(file_path)}"


# ============================================================================
# CHUNK SYSTEM - Building longer audio
# ============================================================================

def _ensure_chunk_system(session: Session) -> None:
    """Ensure chunk system attributes exist on session."""
    if not hasattr(session, 'chunks'):
        session.chunks = []  # List of numpy arrays
    if not hasattr(session, 'chunk_xfade_ms'):
        session.chunk_xfade_ms = 200


def cmd_chunk(session: Session, args: List[str]) -> str:
    """Chunk system for building longer audio pieces.
    
    Usage:
      /chunk                    -> Show chunk list and status
      /chunk add                -> Add current buffer as chunk
      /chunk add <n>            -> Add buffer <n> as chunk
      /chunk rm <n>             -> Remove chunk at position
      /chunk clear              -> Clear all chunks
      /chunk build [path]       -> Stitch chunks with crossfade
      /chunk xfade <ms>         -> Set crossfade duration
      /chunk play               -> Play stitched result
    
    Chunks are stitched together with crossfade to create
    longer continuous audio pieces.
    
    Examples:
      /tone 440 2 0.8
      /chunk add          -> Add 2s tone as chunk 1
      /tone 880 2 0.6
      /chunk add          -> Add 2s tone as chunk 2
      /chunk build        -> Create 4s piece with crossfade
    """
    import numpy as np
    
    _ensure_chunk_system(session)
    
    if not args:
        # Show status
        lines = [f"CHUNKS: {len(session.chunks)} pieces"]
        lines.append(f"  Crossfade: {session.chunk_xfade_ms}ms")
        
        total_samples = 0
        for i, chunk in enumerate(session.chunks, 1):
            dur = len(chunk) / session.sample_rate
            total_samples += len(chunk)
            lines.append(f"  [{i}] {dur:.2f}s ({len(chunk)} samples)")
        
        if session.chunks:
            # Estimate final duration (with crossfade reduction)
            xfade_samples = int(session.sample_rate * session.chunk_xfade_ms / 1000)
            overlap = xfade_samples * (len(session.chunks) - 1) if len(session.chunks) > 1 else 0
            final_samples = total_samples - overlap
            final_dur = final_samples / session.sample_rate
            lines.append(f"  Estimated output: {final_dur:.2f}s")
        
        return '\n'.join(lines)
    
    sub = args[0].lower()
    
    if sub == 'add':
        # Add buffer as chunk
        if len(args) > 1:
            # Add specific buffer
            try:
                buf_idx = int(args[1])
                if hasattr(session, 'buffers') and buf_idx in session.buffers:
                    buf = session.buffers[buf_idx]
                else:
                    return f"ERROR: buffer {buf_idx} not found"
            except ValueError:
                return "ERROR: invalid buffer index"
        else:
            # Add current buffer
            if session.last_buffer is None or len(session.last_buffer) == 0:
                return "ERROR: no buffer to add as chunk"
            buf = session.last_buffer
        
        session.chunks.append(buf.copy())
        dur = len(buf) / session.sample_rate
        return f"OK: added chunk {len(session.chunks)} ({dur:.2f}s)"
    
    elif sub == 'rm':
        if len(args) < 2:
            return "ERROR: usage: /chunk rm <n>"
        try:
            idx = int(args[1]) - 1  # 1-indexed
            if idx < 0 or idx >= len(session.chunks):
                return f"ERROR: chunk {idx+1} not found"
            removed = session.chunks.pop(idx)
            return f"OK: removed chunk {idx+1}"
        except ValueError:
            return "ERROR: invalid chunk number"
    
    elif sub == 'clear':
        count = len(session.chunks)
        session.chunks = []
        return f"OK: cleared {count} chunks"
    
    elif sub == 'xfade':
        if len(args) < 2:
            return f"CROSSFADE: {session.chunk_xfade_ms}ms"
        try:
            ms = int(args[1])
            session.chunk_xfade_ms = max(0, min(2000, ms))
            return f"OK: crossfade set to {session.chunk_xfade_ms}ms"
        except ValueError:
            return "ERROR: invalid crossfade value"
    
    elif sub == 'build':
        if not session.chunks:
            return "ERROR: no chunks to build"
        
        # Stitch with crossfade
        xfade_samples = int(session.sample_rate * session.chunk_xfade_ms / 1000)
        
        result = session.chunks[0].copy()
        
        for next_chunk in session.chunks[1:]:
            if xfade_samples > 0 and len(result) >= xfade_samples and len(next_chunk) >= xfade_samples:
                # Crossfade
                tail = result[-xfade_samples:]
                head = next_chunk[:xfade_samples]
                fade = np.linspace(0.0, 1.0, xfade_samples, dtype=np.float64)
                mixed = tail * (1.0 - fade) + head * fade
                result = np.concatenate([result[:-xfade_samples], mixed, next_chunk[xfade_samples:]])
            else:
                result = np.concatenate([result, next_chunk])
        
        # Normalize if needed
        peak = np.max(np.abs(result))
        if peak > 0.99:
            result = result / peak * 0.99
        
        # Store as buffer
        session.last_buffer = result
        if hasattr(session, 'buffers'):
            idx = session.current_buffer_index if hasattr(session, 'current_buffer_index') else 1
            session.buffers[idx] = result
        
        # Optionally save to file
        if len(args) > 1:
            path = args[1]
            if not path.endswith('.wav'):
                path += '.wav'
            try:
                import wave
                data_int16 = np.int16(np.clip(result * 32767, -32767, 32767))
                with wave.open(path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(session.sample_rate)
                    wf.writeframes(data_int16.tobytes())
                dur = len(result) / session.sample_rate
                return f"OK: built {dur:.2f}s -> {path}"
            except Exception as e:
                return f"ERROR: build ok but save failed - {e}"
        
        dur = len(result) / session.sample_rate
        return f"OK: built {len(session.chunks)} chunks -> {dur:.2f}s in buffer"
    
    elif sub == 'play':
        if not session.chunks:
            return "ERROR: no chunks to play"
        # Build temporarily and play
        result = cmd_chunk(session, ['build'])
        if result.startswith("OK"):
            return session.play(0.8)
        return result
    
    else:
        return f"ERROR: unknown subcommand '{sub}'"


# ============================================================================
# PREVIEW WITH EFFECTS
# ============================================================================

def cmd_pn(session: Session, args: List[str]) -> str:
    """Preview current buffer with effects applied.
    
    Usage:
      /pn                -> Preview with current effect chain
      /pn dry            -> Preview without effects (dry)
      /pn <vol>          -> Preview at specified volume
    
    This applies the current buffer FX chain (/bfx) to a copy
    of the buffer and plays the result without modifying the
    original buffer.
    
    Examples:
      /bfx add reverb amount=50
      /pn              -> Hear buffer with reverb
      /pn dry          -> Hear original without reverb
    """
    import numpy as np
    
    if session.last_buffer is None or len(session.last_buffer) == 0:
        return "ERROR: no buffer to preview"
    
    volume = 0.8
    dry = False
    
    if args:
        if args[0].lower() == 'dry':
            dry = True
        else:
            try:
                vol = float(args[0])
                if vol > 1.0:
                    vol = vol / 100.0
                volume = max(0.0, min(1.0, vol))
            except ValueError:
                pass
    
    # Make a copy for preview
    preview_buf = session.last_buffer.copy()
    
    # Apply buffer FX chain if not dry
    if not dry and hasattr(session, 'buffer_fx_chain') and session.buffer_fx_chain:
        try:
            from ..dsp.effects import apply_effects_with_params
            for effect_name, params in session.buffer_fx_chain:
                preview_buf = apply_effects_with_params(
                    preview_buf, effect_name, params, session.sample_rate
                )
        except Exception as e:
            return f"ERROR: effect preview failed - {e}"
    
    # Also apply main effects chain if present
    if not dry and hasattr(session, 'effects') and session.effects:
        try:
            from ..dsp.effects import apply_effect
            for effect_name in session.effects:
                preview_buf = apply_effect(preview_buf, effect_name, session.sample_rate)
        except Exception:
            pass  # Continue even if old effects fail
    
    # Play the preview
    original_buf = session.last_buffer
    session.last_buffer = preview_buf
    result = session.play(volume)
    session.last_buffer = original_buf  # Restore original
    
    fx_count = len(getattr(session, 'buffer_fx_chain', [])) + len(getattr(session, 'effects', []))
    if dry:
        return f"PREVIEW (dry): {result}"
    else:
        return f"PREVIEW ({fx_count} fx): {result}"



# ============================================================================
# FULL ROUTING COMMAND
# ============================================================================

def cmd_route(session: Session, args: List[str]) -> str:
    """Full modulation routing management.
    
    Usage:
      /route                      -> Show all active routings
      /route add <type> <s> <t> <amt>  -> Add routing
      /route rm <idx>             -> Remove routing by index
      /route clear                -> Clear all routings
      /route swap <idx1> <idx2>   -> Swap routing order
      /route scale <idx> <amt>    -> Scale routing amount
    
    Types:
      fm   - Frequency Modulation (classic DX-style)
      tfm  - Through-zero FM (linear FM)
      am   - Amplitude Modulation (tremolo)
      rm   - Ring Modulation (carrier × modulator)
      pm   - Phase Modulation (similar to FM)
    
    Examples:
      /route add fm 1 0 0.5       -> FM: op1 modulates op0 at 50%
      /route add rm 2 1 1.0       -> Ring mod: op2 × op1
      /route rm 0                 -> Remove first routing
      /route scale 0 0.7          -> Scale first routing to 70%
    
    Routing flows modulator -> carrier (source -> target).
    Amounts are 0.0-1.0 (scaled internally per type).
    """
    if not hasattr(session, 'engine'):
        return "ERROR: synth engine not initialized"
    
    if not args:
        # Show all routings
        if not session.engine.algorithms:
            return "ROUTINGS: (none)\n  Use /route add <type> <src> <tgt> <amt>"
        
        lines = ["=== MODULATION ROUTING ==="]
        for i, (algo_type, src, tgt, amt) in enumerate(session.engine.algorithms):
            lines.append(f"  [{i}] {algo_type.upper()}: op{src} -> op{tgt} ({amt:.3f})")
        
        # Show interval modulation
        if session.engine.interval_mod:
            lines.append("")
            lines.append("=== INTERVAL MODULATION ===")
            for op_idx, mod in session.engine.interval_mod.items():
                if mod is not None:
                    depth = session.engine.operators.get(op_idx, {}).get('interval_mod_depth', 12)
                    lines.append(f"  op{op_idx}: ±{depth}st (audio-rate)")
        
        # Show LFO-based interval mod
        for op_idx, op in session.engine.operators.items():
            lfo_rate = op.get('interval_lfo_rate', 0)
            if lfo_rate > 0:
                lfo_depth = op.get('interval_lfo_depth', 0)
                lfo_wave = op.get('interval_lfo_wave', 'sine')
                lines.append(f"  op{op_idx}: LFO {lfo_rate:.1f}Hz ±{lfo_depth:.1f}st ({lfo_wave})")
        
        # Show filter modulation
        if session.engine.filter_mod is not None:
            lines.append("")
            lines.append("=== FILTER MODULATION ===")
            lines.append(f"  Global: ±{session.engine.filter_mod_depth:.1f} octaves (audio-rate)")
        
        return '\n'.join(lines)
    
    sub = args[0].lower()
    
    if sub == 'add':
        if len(args) < 5:
            return "ERROR: usage: /route add <type> <src> <tgt> <amount>"
        try:
            algo_type = args[1].lower()
            src = int(args[2])
            tgt = int(args[3])
            amt = float(args[4])
            
            valid_types = ['fm', 'tfm', 'am', 'rm', 'pm']
            if algo_type not in valid_types:
                return f"ERROR: type must be one of: {', '.join(valid_types)}"
            
            session.engine.add_algorithm(algo_type, src, tgt, amt)
            return f"OK: added {algo_type.upper()} op{src} -> op{tgt} ({amt:.3f})"
        except ValueError:
            return "ERROR: invalid parameters"
    
    elif sub == 'rm':
        if len(args) < 2:
            return "ERROR: usage: /route rm <idx>"
        try:
            idx = int(args[1])
            if idx < 0 or idx >= len(session.engine.algorithms):
                return f"ERROR: routing {idx} not found"
            removed = session.engine.algorithms.pop(idx)
            return f"OK: removed routing {idx} ({removed[0]})"
        except ValueError:
            return "ERROR: invalid index"
    
    elif sub == 'clear':
        count = len(session.engine.algorithms)
        session.engine.algorithms.clear()
        return f"OK: cleared {count} routings"
    
    elif sub == 'swap':
        if len(args) < 3:
            return "ERROR: usage: /route swap <idx1> <idx2>"
        try:
            idx1 = int(args[1])
            idx2 = int(args[2])
            n = len(session.engine.algorithms)
            if idx1 < 0 or idx1 >= n or idx2 < 0 or idx2 >= n:
                return "ERROR: invalid routing indices"
            session.engine.algorithms[idx1], session.engine.algorithms[idx2] = \
                session.engine.algorithms[idx2], session.engine.algorithms[idx1]
            return f"OK: swapped routings {idx1} and {idx2}"
        except ValueError:
            return "ERROR: invalid indices"
    
    elif sub == 'scale':
        if len(args) < 3:
            return "ERROR: usage: /route scale <idx> <amount>"
        try:
            idx = int(args[1])
            amt = float(args[2])
            if idx < 0 or idx >= len(session.engine.algorithms):
                return f"ERROR: routing {idx} not found"
            algo_type, src, tgt, _ = session.engine.algorithms[idx]
            session.engine.algorithms[idx] = (algo_type, src, tgt, amt)
            return f"OK: routing {idx} amount = {amt:.3f}"
        except ValueError:
            return "ERROR: invalid parameters"
    
    else:
        return f"ERROR: unknown subcommand '{sub}'"


# ============================================================================
# /ch - CHUNK ALIAS
# ============================================================================

def cmd_ch(session: Session, args: List[str]) -> str:
    """Alias for /chunk command."""
    return cmd_chunk(session, args)


# ============================================================================
# GRANULAR ENGINE COMMANDS
# ============================================================================

def _ensure_granular(session: Session) -> None:
    """Ensure granular engine exists on session."""
    if not hasattr(session, 'granular_engine'):
        from ..dsp.granular import GranularEngine
        session.granular_engine = GranularEngine(session.sample_rate)


def cmd_gr(session: Session, args: List[str]) -> str:
    """Granular processing engine.
    
    Usage:
      /gr                       -> Show granular engine status
      /gr size <ms>             -> Set grain size (1-500ms)
      /gr density <n>           -> Set grain density (0.5-32)
      /gr pos <0-1>             -> Set position in source
      /gr spread <0-1>          -> Set position spread
      /gr pitch <ratio>         -> Set pitch ratio (0.25-4)
      /gr env <type>            -> Set envelope (hann,tri,gauss,trap,tukey,rect)
      /gr reverse <0-1>         -> Set reverse probability
      /gr seed <n>              -> Set random seed
      /gr process [dur]         -> Process buffer through granular
      /gr freeze <pos> [dur]    -> Freeze at position
      /gr stretch <factor>      -> Time stretch
      /gr shift <semitones>     -> Pitch shift
      /gr save <slot>           -> Save granular preset
      /gr load <slot>           -> Load granular preset
    
    Examples:
      /tone 440 2 0.8
      /gr size 50
      /gr density 8
      /gr process 4            -> 4 second granular processing
      /gr freeze 0.5 3         -> Freeze at center for 3s
    """
    import numpy as np
    
    _ensure_granular(session)
    engine = session.granular_engine
    
    if not args:
        return engine.get_status()
    
    sub = args[0].lower()
    
    if sub == 'size' and len(args) > 1:
        try:
            engine.set_params(grain_size=float(args[1]))
            return f"OK: grain size = {engine.grain_size_ms:.1f}ms"
        except ValueError:
            return "ERROR: invalid grain size"
    
    elif sub == 'density' and len(args) > 1:
        try:
            engine.set_params(density=float(args[1]))
            return f"OK: density = {engine.density:.1f}"
        except ValueError:
            return "ERROR: invalid density"
    
    elif sub == 'pos' and len(args) > 1:
        try:
            engine.set_params(position=float(args[1]))
            return f"OK: position = {engine.position:.2f}"
        except ValueError:
            return "ERROR: invalid position"
    
    elif sub == 'spread' and len(args) > 1:
        try:
            engine.set_params(position_spread=float(args[1]))
            return f"OK: spread = {engine.position_spread:.2f}"
        except ValueError:
            return "ERROR: invalid spread"
    
    elif sub == 'pitch' and len(args) > 1:
        try:
            engine.set_params(pitch=float(args[1]))
            return f"OK: pitch = {engine.pitch_ratio:.2f}x"
        except ValueError:
            return "ERROR: invalid pitch"
    
    elif sub == 'env' and len(args) > 1:
        env_type = args[1].lower()
        engine.set_params(envelope=env_type)
        return f"OK: envelope = {engine.envelope}"
    
    elif sub == 'reverse' and len(args) > 1:
        try:
            engine.set_params(reverse_prob=float(args[1]))
            return f"OK: reverse prob = {engine.reverse_prob:.2f}"
        except ValueError:
            return "ERROR: invalid reverse probability"
    
    elif sub == 'seed' and len(args) > 1:
        try:
            engine.set_params(seed=int(args[1]) if args[1].lower() != 'off' else None)
            return f"OK: seed = {engine.seed}"
        except ValueError:
            return "ERROR: invalid seed"
    
    elif sub == 'process':
        if session.last_buffer is None or len(session.last_buffer) == 0:
            return "ERROR: no buffer to process"
        
        duration = float(args[1]) if len(args) > 1 else len(session.last_buffer) / session.sample_rate
        result = engine.process(session.last_buffer, duration)
        session.last_buffer = result
        
        if hasattr(session, 'buffers') and hasattr(session, 'current_buffer_index'):
            session.buffers[session.current_buffer_index] = result
        
        return f"OK: granular processed -> {duration:.2f}s ({len(result)} samples)"
    
    elif sub == 'freeze' and len(args) > 1:
        if session.last_buffer is None or len(session.last_buffer) == 0:
            return "ERROR: no buffer to freeze"
        
        try:
            pos = float(args[1])
            duration = float(args[2]) if len(args) > 2 else 2.0
            result = engine.freeze(session.last_buffer, duration, pos)
            session.last_buffer = result
            
            if hasattr(session, 'buffers') and hasattr(session, 'current_buffer_index'):
                session.buffers[session.current_buffer_index] = result
            
            return f"OK: frozen at {pos:.2f} for {duration:.2f}s"
        except (ValueError, IndexError):
            return "ERROR: usage: /gr freeze <pos> [duration]"
    
    elif sub == 'stretch' and len(args) > 1:
        if session.last_buffer is None or len(session.last_buffer) == 0:
            return "ERROR: no buffer to stretch"
        
        try:
            factor = float(args[1])
            result = engine.time_stretch(session.last_buffer, factor)
            session.last_buffer = result
            
            if hasattr(session, 'buffers') and hasattr(session, 'current_buffer_index'):
                session.buffers[session.current_buffer_index] = result
            
            return f"OK: stretched {factor:.2f}x -> {len(result)/session.sample_rate:.2f}s"
        except ValueError:
            return "ERROR: invalid stretch factor"
    
    elif sub == 'shift' and len(args) > 1:
        if session.last_buffer is None or len(session.last_buffer) == 0:
            return "ERROR: no buffer to shift"
        
        try:
            semitones = float(args[1])
            result = engine.pitch_shift(session.last_buffer, semitones)
            session.last_buffer = result
            
            if hasattr(session, 'buffers') and hasattr(session, 'current_buffer_index'):
                session.buffers[session.current_buffer_index] = result
            
            return f"OK: pitch shifted {semitones:+.1f} semitones"
        except ValueError:
            return "ERROR: invalid semitone value"
    
    elif sub == 'save' and len(args) > 1:
        try:
            slot = int(args[1])
            engine.save_preset(slot)
            return f"OK: saved granular preset to slot {slot}"
        except ValueError:
            return "ERROR: invalid slot number"
    
    elif sub == 'load' and len(args) > 1:
        try:
            slot = int(args[1])
            if engine.load_preset(slot):
                return f"OK: loaded granular preset from slot {slot}"
            return f"ERROR: no preset in slot {slot}"
        except ValueError:
            return "ERROR: invalid slot number"
    
    else:
        return f"ERROR: unknown granular subcommand '{sub}'"


# ============================================================================
# AUDIO-RATE INTERVAL MODULATION
# ============================================================================

def cmd_imod(session: Session, args: List[str]) -> str:
    """Audio-rate interval modulation per operator.
    
    Usage:
      /imod                     -> Show interval modulation status
      /imod <op> lfo <rate> <depth> [wave]
                                -> Set LFO interval mod on operator
      /imod <op> src <source_op> <depth>
                                -> Use operator as interval mod source
      /imod <op> off            -> Disable interval modulation
      /imod clear               -> Clear all interval modulation
    
    Parameters:
      op: Operator index (0-15)
      rate: LFO rate in Hz
      depth: Modulation depth in semitones
      wave: LFO wave (sine, tri, saw, square)
      source_op: Operator to use as modulation source
    
    Examples:
      /imod 0 lfo 5 12 sine     -> Op 0: 5Hz LFO, ±12 semitones
      /imod 1 src 2 7           -> Op 1: modulated by op 2, ±7 semitones
      /imod 0 off               -> Disable interval mod on op 0
    
    Audio-rate interval modulation allows for vibrato, trills, and
    complex pitch effects at sample-accurate resolution.
    """
    if not hasattr(session, 'engine'):
        return "ERROR: synth engine not initialized"
    
    if not args:
        # Show status
        lines = ["INTERVAL MODULATION:"]
        for idx, op in session.engine.operators.items():
            lfo_rate = op.get('interval_lfo_rate', 0)
            lfo_depth = op.get('interval_lfo_depth', 0)
            lfo_wave = op.get('interval_lfo_wave', 'sine')
            has_mod = op.get('interval_mod') is not None
            
            if lfo_rate > 0 or has_mod:
                if lfo_rate > 0:
                    lines.append(f"  op{idx}: LFO {lfo_rate:.1f}Hz ±{lfo_depth:.1f}st ({lfo_wave})")
                elif has_mod:
                    depth = op.get('interval_mod_depth', 12)
                    lines.append(f"  op{idx}: External mod ±{depth:.1f}st")
            else:
                lines.append(f"  op{idx}: (none)")
        
        if len(lines) == 1:
            return "INTERVAL MODULATION: (no operators)"
        return '\n'.join(lines)
    
    # Parse operator index
    try:
        if args[0].lower() == 'clear':
            session.engine.clear_modulation()
            return "OK: cleared all interval modulation"
        
        op_idx = int(args[0])
        if op_idx not in session.engine.operators:
            session.engine.set_operator(op_idx)
        
        if len(args) < 2:
            return f"ERROR: usage: /imod {op_idx} lfo|src|off ..."
        
        sub = args[1].lower()
        
        if sub == 'lfo':
            if len(args) < 4:
                return f"ERROR: usage: /imod {op_idx} lfo <rate> <depth> [wave]"
            rate = float(args[2])
            depth = float(args[3])
            wave = args[4] if len(args) > 4 else 'sine'
            session.engine.set_interval_lfo(op_idx, rate, depth, wave)
            return f"OK: op{op_idx} interval LFO: {rate:.1f}Hz ±{depth:.1f}st ({wave})"
        
        elif sub == 'src':
            if len(args) < 4:
                return f"ERROR: usage: /imod {op_idx} src <source_op> <depth>"
            source_op = int(args[2])
            depth = float(args[3])
            
            # Generate buffer from source op and use as modulation
            import numpy as np
            duration = 1.0  # 1 second of mod signal
            t = np.arange(int(session.sample_rate * duration)) / session.sample_rate
            
            # Get source operator's output as modulation signal
            if source_op in session.engine.operators:
                src_op = session.engine.operators[source_op]
                freq = src_op.get('freq', 440.0)
                mod_signal = np.sin(2 * np.pi * freq * t)  # Normalize -1 to 1
                session.engine.set_interval_mod(op_idx, mod_signal, depth)
                return f"OK: op{op_idx} interval mod from op{source_op} ±{depth:.1f}st"
            return f"ERROR: source operator {source_op} not found"
        
        elif sub == 'off':
            session.engine.set_interval_mod(op_idx, None)
            if op_idx in session.engine.operators:
                session.engine.operators[op_idx].pop('interval_lfo_rate', None)
                session.engine.operators[op_idx].pop('interval_lfo_depth', None)
            return f"OK: disabled interval mod on op{op_idx}"
        
        else:
            return f"ERROR: unknown subcommand '{sub}'"
    
    except (ValueError, IndexError) as e:
        return f"ERROR: {e}"


def cmd_fmod(session: Session, args: List[str]) -> str:
    """Audio-rate filter modulation.
    
    Usage:
      /fmod                     -> Show filter modulation status
      /fmod lfo <rate> <depth>  -> Set LFO filter mod (depth in octaves)
      /fmod src <op> <depth>    -> Use operator as filter mod source
      /fmod off                 -> Disable filter modulation
      /fmod op <n> <cutoff> <res> -> Per-operator filter
      /fmod op <n> off          -> Disable per-op filter
    
    Examples:
      /fmod lfo 2 2             -> 2Hz LFO, ±2 octaves
      /fmod src 3 1.5           -> Op 3 modulates filter ±1.5 octaves
      /fmod op 0 2000 0.7       -> Op 0 filter: 2kHz, res 0.7
    
    Audio-rate filter modulation creates dynamic filter sweeps
    and complex timbral changes at sample rate.
    """
    import numpy as np
    
    if not hasattr(session, 'engine'):
        return "ERROR: synth engine not initialized"
    
    if not args:
        lines = ["FILTER MODULATION:"]
        if session.engine.filter_mod is not None:
            lines.append(f"  Global: depth={session.engine.filter_mod_depth:.1f} octaves")
        else:
            lines.append("  Global: (none)")
        
        for idx, op in session.engine.operators.items():
            if op.get('filter_enabled'):
                cutoff = op.get('filter_cutoff', 2000)
                res = op.get('filter_resonance', 0.5)
                lines.append(f"  op{idx}: {cutoff:.0f}Hz res={res:.2f}")
        
        return '\n'.join(lines)
    
    sub = args[0].lower()
    
    if sub == 'lfo':
        if len(args) < 3:
            return "ERROR: usage: /fmod lfo <rate> <depth_octaves>"
        rate = float(args[1])
        depth = float(args[2])
        
        # Generate LFO signal
        duration = 1.0
        t = np.arange(int(session.sample_rate * duration)) / session.sample_rate
        mod_signal = np.sin(2 * np.pi * rate * t)
        session.engine.set_filter_mod(mod_signal, depth)
        return f"OK: filter LFO: {rate:.1f}Hz ±{depth:.1f} octaves"
    
    elif sub == 'src':
        if len(args) < 3:
            return "ERROR: usage: /fmod src <op> <depth_octaves>"
        source_op = int(args[1])
        depth = float(args[2])
        
        if source_op in session.engine.operators:
            src_op = session.engine.operators[source_op]
            freq = src_op.get('freq', 440.0)
            
            duration = 1.0
            t = np.arange(int(session.sample_rate * duration)) / session.sample_rate
            mod_signal = np.sin(2 * np.pi * freq * t)
            session.engine.set_filter_mod(mod_signal, depth)
            return f"OK: filter mod from op{source_op} ±{depth:.1f} octaves"
        return f"ERROR: operator {source_op} not found"
    
    elif sub == 'off':
        session.engine.set_filter_mod(None)
        return "OK: disabled filter modulation"
    
    elif sub == 'op':
        if len(args) < 3:
            return "ERROR: usage: /fmod op <n> <cutoff> <res> OR /fmod op <n> off"
        
        op_idx = int(args[1])
        
        if args[2].lower() == 'off':
            session.engine.set_op_filter_mod(op_idx, None)
            return f"OK: disabled filter on op{op_idx}"
        
        cutoff = float(args[2])
        res = float(args[3]) if len(args) > 3 else 0.5
        
        # Simple static filter for now (could add modulation signal)
        session.engine.set_op_filter_mod(op_idx, None, cutoff, res)
        return f"OK: op{op_idx} filter: {cutoff:.0f}Hz res={res:.2f}"
    
    else:
        return f"ERROR: unknown subcommand '{sub}'"


# ============================================================================
# PRESET BANK
# ============================================================================

def cmd_preset(session: Session, args: List[str]) -> str:
    """Synthesis preset bank management.
    
    Usage:
      /preset                   -> List saved presets
      /preset save <slot> [name] -> Save current state to slot
      /preset load <slot>       -> Load state from slot
      /preset del <slot>        -> Delete preset
      /preset copy <from> <to>  -> Copy preset
    
    Presets store:
    - All operator configurations
    - Modulation routing
    - Interval and filter modulation settings
    
    Examples:
      /preset save 1 bass       -> Save as "bass" in slot 1
      /preset load 1            -> Restore slot 1
      /preset                   -> List all presets
    """
    if not hasattr(session, 'engine'):
        return "ERROR: synth engine not initialized"
    
    if not args:
        presets = session.engine.list_presets()
        if not presets:
            return "PRESETS: (none)\n  Use /preset save <slot> [name] to create"
        
        lines = ["PRESETS:"]
        for slot, name in presets:
            lines.append(f"  [{slot}] {name}")
        return '\n'.join(lines)
    
    sub = args[0].lower()
    
    if sub == 'save':
        if len(args) < 2:
            return "ERROR: usage: /preset save <slot> [name]"
        try:
            slot = int(args[1])
            name = args[2] if len(args) > 2 else f'preset_{slot}'
            session.engine.save_preset(slot, name)
            return f"OK: saved preset '{name}' to slot {slot}"
        except ValueError:
            return "ERROR: invalid slot number"
    
    elif sub == 'load':
        if len(args) < 2:
            return "ERROR: usage: /preset load <slot>"
        try:
            slot = int(args[1])
            if session.engine.load_preset(slot):
                presets = session.engine.list_presets()
                name = dict(presets).get(slot, f'preset_{slot}')
                return f"OK: loaded preset '{name}' from slot {slot}"
            return f"ERROR: no preset in slot {slot}"
        except ValueError:
            return "ERROR: invalid slot number"
    
    elif sub == 'del':
        if len(args) < 2:
            return "ERROR: usage: /preset del <slot>"
        try:
            slot = int(args[1])
            if session.engine.delete_preset(slot):
                return f"OK: deleted preset from slot {slot}"
            return f"ERROR: no preset in slot {slot}"
        except ValueError:
            return "ERROR: invalid slot number"
    
    elif sub == 'copy':
        if len(args) < 3:
            return "ERROR: usage: /preset copy <from> <to>"
        try:
            from_slot = int(args[1])
            to_slot = int(args[2])
            
            if from_slot not in session.engine.preset_bank:
                return f"ERROR: no preset in slot {from_slot}"
            
            import copy
            session.engine.preset_bank[to_slot] = copy.deepcopy(
                session.engine.preset_bank[from_slot]
            )
            return f"OK: copied preset from slot {from_slot} to {to_slot}"
        except ValueError:
            return "ERROR: invalid slot numbers"
    
    else:
        return f"ERROR: unknown subcommand '{sub}'"


# ============================================================================
# USER DATA MANAGEMENT
# ============================================================================

def cmd_userdata(session: Session, args: List[str]) -> str:
    """User data directory management.
    
    Usage:
      /userdata              -> Show user data info and paths
      /userdata init         -> Initialize all directories
      /userdata open         -> Open user data folder (if supported)
      /userdata reset        -> Reset preferences to defaults
    
    User Data Location:
      Windows: C:\\Users\\<you>\\Documents\\MDMA\\
      Linux/Mac: ~/Documents/MDMA/
    
    Directory Structure:
      MDMA/
      ├── constants.json     User-defined constants
      ├── preferences.json   Session defaults
      ├── banks/             Routing algorithm banks
      │   ├── factory/       Built-in banks
      │   └── user/          Your banks
      ├── presets/           Synth presets
      │   ├── factory/       Built-in presets
      │   └── user/          Your presets
      ├── packs/             Sound/sample packs
      ├── songs/             DJ mode song library
      │   ├── library.json   Song metadata
      │   └── playlists/     Your playlists
      └── projects/          Project files
    """
    from ..core.user_data import (
        initialize_user_data, get_user_data_info, get_mdma_root,
        save_preferences, DEFAULT_PREFERENCES
    )
    
    if not args:
        return get_user_data_info()
    
    sub = args[0].lower()
    
    if sub == 'init':
        result = initialize_user_data()
        return f"OK: {result}"
    
    elif sub == 'open':
        import subprocess
        import sys
        root = get_mdma_root()
        try:
            if sys.platform == 'win32':
                subprocess.Popen(['explorer', str(root)])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', str(root)])
            else:
                subprocess.Popen(['xdg-open', str(root)])
            return f"OK: opened {root}"
        except Exception as e:
            return f"ERROR: could not open folder: {e}\nPath: {root}"
    
    elif sub == 'reset':
        if save_preferences(DEFAULT_PREFERENCES):
            return "OK: preferences reset to defaults"
        else:
            return "ERROR: failed to reset preferences"
    
    else:
        return f"ERROR: unknown subcommand '{sub}'. Use: init, open, reset"


def cmd_const(session: Session, args: List[str]) -> str:
    """User constants management.
    
    Usage:
      /const                  -> List all constants
      /const <n>           -> Show constant value
      /const <n> <value>   -> Set new constant (immutable once set!)
      /const del <n>       -> Delete constant (admin only)
      /const export           -> Export constants to file
    
    Constants are saved to: ~/Documents/MDMA/constants.json
    
    Note: Constants are immutable by default - once set, they cannot
    be changed. This ensures consistency in your projects.
    """
    from ..core.user_data import load_constants, save_constants, add_constant
    
    # Load current constants
    constants = load_constants()
    
    if not args:
        if not constants:
            return "No user constants defined. Use /const <n> <value> to create one."
        
        lines = ["=== USER CONSTANTS ===", ""]
        for name, val in sorted(constants.items()):
            lines.append(f"  .{name} = {val}")
        lines.append("")
        lines.append(f"Total: {len(constants)} constants")
        lines.append("Access with /.name (e.g., /.mybpm)")
        return '\n'.join(lines)
    
    name = args[0].lower()
    
    if name == 'export':
        from ..core.user_data import get_constants_path
        return f"Constants file: {get_constants_path()}"
    
    if name == 'del' and len(args) > 1:
        const_name = args[1].lower()
        if const_name in constants:
            del constants[const_name]
            save_constants(constants)
            return f"OK: deleted constant '{const_name}'"
        else:
            return f"ERROR: constant '{const_name}' not found"
    
    if len(args) == 1:
        # Show single constant
        if name in constants:
            return f".{name} = {constants[name]}"
        else:
            return f"ERROR: constant '{name}' not found"
    
    # Set new constant
    value_str = ' '.join(args[1:])
    
    # Parse value
    try:
        value = float(value_str)
        if value == int(value):
            value = int(value)
    except ValueError:
        value = value_str  # Keep as string
    
    success, msg = add_constant(name, value, constants)
    if success:
        return f"OK: {msg}"
    else:
        return f"ERROR: {msg}"


# ============================================================================
# HELP SYSTEM
# ============================================================================

def cmd_help(session: Session, args: List[str]) -> str:
    """Show help for MDMA commands.
    
    Usage:
      /help              Show command categories
      /help <category>   Show commands in category
      /help <command>    Show help for specific command
      /help all          Show all commands
    
    Categories: dj, fx, synth, ai, buffer, pattern, playback, perf
    
    Short aliases: /h, /?
    """
    
    # Command reference organized by category
    HELP_CATEGORIES = {
        'dj': {
            'name': 'DJ Mode & Mixing',
            'commands': {
                # Mode control
                'djm, dj': 'Toggle/control DJ mode',
                
                # Playback
                'play, p, start': 'Start deck playback',
                'stop, pause': 'Stop deck playback',
                
                # Deck selection
                'deck, dk, d': 'Select active deck',
                'deck+, dk+': 'Add new deck',
                'deck-, dk-': 'Remove deck',
                
                # Tempo
                'tempo, bpm, t': 'Get/set deck tempo (20-300 BPM)',
                
                # Volume
                'vol, volume, v': 'Get/set deck volume (0-100)',
                
                # Crossfader
                'cf, crossfader, xfader': 'Get/set crossfader (0-100)',
                'xfade, xf, cross': 'Crossfade between decks',
                
                # Filter
                'fl, flt, filter': 'Filter cutoff (1-100)',
                'flr, res, q': 'Filter resonance (1-100)',
                
                # Navigation
                'j, jump, goto': 'Jump to position (start/drop/beat#/time)',
                
                # Effects
                'dfx, deckfx': 'Apply deck effect (echo/vamp/reverb...)',
                'scr, scratch': 'Trigger scratch preset (1-5)',
                'stud, stutter': 'Trigger stutter effect',
                
                # Loops
                'lpc, loopcount': 'Set loop repeat count',
                'lpg, lg, loopgo': 'Jump to loop start',
                'dur, duration': 'Set effect duration',
                
                # Transitions
                'tran, tr, trans, x': 'Quick transition',
                'transition': 'Full transition control',
                'drop, drp, !': 'Instant drop',
                
                # Sync
                'sync, sy': 'Sync deck tempos',
                'cue, c': 'Set/trigger cue point',
                
                # Stems
                'stem, st': 'Stem separation/control',
                
                # Sections
                'section, sec': 'Navigate to section',
                'chop, slice': 'Chop audio into sections',
                
                # Streaming
                'stream, str, sc': 'Stream audio source',
                
                # AI Enhancement
                'ai': 'Toggle AI audio enhancement',
                'enhance, enh': 'Enhancement settings/presets',
                
                # Devices
                'do, devices, dev': 'List audio devices',
                'doc, master': 'Set master output device',
                'hep, headphones': 'Headphone output',
                
                # Screen Reader
                'sr, nvda, reader': 'Screen reader settings',
                
                # Safety
                'fallback, fb, safe': 'Enable fallback mode',
                
                # Library
                'library, lib': 'Track library',
                'playlist, pl': 'Playlist management',
                
                # Song Registry
                'reg, registry, songs': 'Song registry system',
            }
        },
        
        'registry': {
            'name': 'Song Registry & Quality Assurance',
            'commands': {
                # Scanning
                'reg scan <folder>': 'Scan folder for songs (recursive)',
                'reg rescan': 'Re-check all registered songs',
                
                # Browsing
                'reg list [query]': 'List/search songs',
                'reg info <id>': 'Show song details',
                'reg high': 'List high-quality songs only',
                'reg recent': 'Show recently added',
                'reg played': 'Show most played',
                'reg favs': 'Show favorites',
                
                # Filtering
                'reg bpm <range>': 'Filter by BPM (e.g., 120-140)',
                'reg genre <genre>': 'Filter by genre hint',
                'reg quality': 'Show quality breakdown',
                
                # Loading
                'reg load <id|name>': 'Load song to active deck',
                'reg load <id> <deck>': 'Load to specific deck',
                
                # Metadata
                'reg tag <id> <tags>': 'Add tags to song',
                'reg rate <id> <1-5>': 'Rate song (1-5 stars)',
                'reg fav <id>': 'Toggle favorite status',
                
                # Management
                'reg fix <id>': 'Re-analyze and fix quality',
                'reg remove <id>': 'Remove from registry',
            }
        },
        
        'fx': {
            'name': 'Effects & Processing',
            'commands': {
                # Quick effects
                'r1-r5': 'Reverb (small/large/plate/spring/cathedral)',
                'd1-d5': 'Delay (simple/pingpong/multitap/slapback/tape)',
                's1-s5': 'Saturation (soft/hard/overdrive/fuzz/tube)',
                'v1-v4': 'Vamp (light/medium/heavy/fuzz)',
                'l1-l6': 'Lo-fi (bitcrush/chorus/flanger/phaser/filter/halftime)',
                'c1-c5': 'Compression (mild/hard/limiter/expander/softclip)',
                'g1-g5': 'Gates (1-5 patterns)',
                
                # Effect control
                'fx': 'Apply effect to buffer',
                'fxl': 'List all effects',
                'fxa': 'Apply effect with amount',
                'amt': 'Set effect wet/dry amount',
                
                # Filter slots
                'sfc': 'Set filter slot count',
                'sf': 'Select active filter slot',
                'e0-e9': 'Quick filter slot select',
                
                # Specific effects
                'vamp': 'Vamp/overdrive effect',
                'reverb, verb': 'Reverb effect',
                'delay': 'Delay effect',
                'saturate, sat': 'Saturation effect',
                'compress, comp': 'Compression effect',
                'lofi': 'Lo-fi effect',
                'gate': 'Gate effect',
                'shimmer': 'Shimmer reverb',
                'conv': 'Convolution reverb',
                
                # EQ/Filter
                'lp, lpf': 'Low-pass filter',
                'hp, hpf': 'High-pass filter',
                'bp': 'Band-pass filter',
                
                # Modulation
                'chorus': 'Chorus effect',
                'flanger': 'Flanger effect',
                'phaser': 'Phaser effect',
                
                # Distortion
                'od, overdrive': 'Overdrive',
                'dist, distort': 'Distortion',
                'fuzz': 'Fuzz effect',
                'crush, bitcrush': 'Bitcrusher',
                
                # Time
                'half': 'Half-speed effect',
                'double': 'Double-speed effect',
                'reverse, rev': 'Reverse audio',
                
                # Chain
                'chain': 'Effect chain management',
                'bypass': 'Bypass effects',
                'dry': 'Set dry level',
            }
        },
        
        'synth': {
            'name': 'Synthesis & Sound Design',
            'commands': {
                # Oscillator
                'wave': 'Set waveform (sine/saw/square/tri/noise)',
                'osc': 'Oscillator settings',
                'freq': 'Set frequency (Hz)',
                'note': 'Set note (c4, a#3, etc.)',
                'det, detune': 'Detune amount',
                
                # Envelope
                'att, atk': 'Attack time',
                'dec': 'Decay time',
                'sus': 'Sustain level',
                'rel': 'Release time',
                
                # Filter
                'cut, cutoff': 'Filter cutoff',
                'res, reso': 'Filter resonance',
                'fenv': 'Filter envelope',
                'fatk, fdec, frel': 'Filter envelope times',
                
                # Modulation
                'lfo': 'LFO settings',
                'mod': 'Modulation amount',
                'fm': 'FM synthesis',
                'pm': 'Phase modulation',
                
                # FM/Operators
                'op': 'Select operator',
                'car': 'Carrier settings',
                'alg': 'FM algorithm',
                'ratio': 'Operator ratio',
                'fb, feedback': 'FM feedback',
                
                # Generation
                'gen': 'Generate sound',
                'render, rn': 'Render to buffer',
                
                # Presets
                'preset, pre': 'Load/save preset',
                'bank, bk': 'Preset bank',
            }
        },
        
        'ai': {
            'name': 'AI & Generation',
            'commands': {
                # Enhancement
                'ai': 'Toggle AI enhancement',
                'enhance, enh': 'Enhancement settings',
                
                # Generation
                'gen': 'AI audio generation',
                'prompt': 'Set generation prompt',
                'style': 'Set generation style',
                'seed': 'Set random seed',
                
                # Analysis
                'analyze': 'Analyze audio',
                'describe': 'AI describe sound',
                'classify': 'Classify audio type',
                
                # Breeding
                'breed': 'Breed two sounds',
                'mutate': 'Mutate sound',
                'evolve': 'Evolve sound population',
                
                # Routing
                'router': 'AI parameter routing',
                'map': 'Map parameters',
            }
        },
        
        'buffer': {
            'name': 'Buffer & Audio Management',
            'commands': {
                # Buffer control
                'buf, b': 'Select buffer',
                'new': 'Create new buffer',
                'clear, clr': 'Clear buffer',
                'copy, cp': 'Copy buffer',
                'len': 'Get/set buffer length',
                
                # Operations
                'norm, normalize': 'Normalize audio',
                'gain, g': 'Apply gain',
                'fade': 'Apply fade in/out',
                'trim': 'Trim silence',
                'crop': 'Crop to selection',
                
                # Selection
                'sel': 'Set selection range',
                'all': 'Select all',
                
                # Analysis
                'peak': 'Show peak level',
                'rms': 'Show RMS level',
                'dc': 'Remove DC offset',
            }
        },
        
        'pattern': {
            'name': 'Pattern & Sequencing',
            'commands': {
                # Patterns
                'pat': 'Apply pattern',
                'apat': 'Advanced pattern',
                'seq': 'Sequence editor',
                
                # Rhythm
                'euclid': 'Euclidean rhythm',
                'poly': 'Polyrhythm',
                'swing': 'Apply swing',
                
                # Grid
                'grid': 'Set grid resolution',
                'quant': 'Quantize',
                'step': 'Step sequencer',
            }
        },
        
        'playback': {
            'name': 'Playback & Recording',
            'commands': {
                # Playback
                'play, p': 'Play audio',
                'stop': 'Stop playback',
                'pause': 'Pause playback',
                'loop': 'Toggle loop mode',
                
                # Position
                'pos': 'Get/set position',
                'seek': 'Seek to time',
                'rewind, rw': 'Rewind',
                
                # Recording
                'rec': 'Start recording',
                'arm': 'Arm for recording',
                
                # File
                'load': 'Load audio file',
                'save': 'Save audio file',
                'export': 'Export audio',
                'render': 'Render to file',
            }
        },
        
        'perf': {
            'name': 'Performance & Live',
            'commands': {
                # Performance
                'perf': 'Performance mode',
                'snap': 'Snapshot state',
                'recall': 'Recall snapshot',
                
                # Macros
                'macro': 'Create macro',
                'bind': 'Bind control',
                
                # Monitoring
                'meter': 'Level meter',
                'scope': 'Oscilloscope',
                'spec': 'Spectrum analyzer',
                
                # Safety
                'panic': 'Audio panic (stop all)',
                'mute': 'Mute output',
                'solo': 'Solo channel',
            }
        },
    }
    
    if not args:
        # Show category overview
        lines = ["=== MDMA COMMAND HELP ===", ""]
        lines.append("Categories:")
        for cat_id, cat_info in HELP_CATEGORIES.items():
            lines.append(f"  /{cat_id:10} - {cat_info['name']}")
        lines.append("")
        lines.append("Use /help <category> for commands")
        lines.append("Use /help <command> for specific help")
        lines.append("Use /help all for complete list")
        return "\n".join(lines)
    
    query = args[0].lower()
    
    # Check if it's a category
    if query in HELP_CATEGORIES:
        cat = HELP_CATEGORIES[query]
        lines = [f"=== {cat['name'].upper()} ===", ""]
        for cmd, desc in cat['commands'].items():
            lines.append(f"  /{cmd:20} {desc}")
        return "\n".join(lines)
    
    # Check for 'all'
    if query == 'all':
        lines = ["=== ALL MDMA COMMANDS ===", ""]
        for cat_id, cat_info in HELP_CATEGORIES.items():
            lines.append(f"--- {cat_info['name']} ---")
            for cmd, desc in cat_info['commands'].items():
                lines.append(f"  /{cmd}: {desc}")
            lines.append("")
        return "\n".join(lines)
    
    # Search for specific command
    query = query.lstrip('/')
    for cat_id, cat_info in HELP_CATEGORIES.items():
        for cmd_str, desc in cat_info['commands'].items():
            cmds = [c.strip() for c in cmd_str.split(',')]
            if query in cmds:
                return f"/{cmd_str}: {desc}\nCategory: {cat_info['name']}"
    
    return f"Unknown command or category: {query}\nUse /help to see categories"


# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

def get_general_commands() -> dict:
    """Return general commands for registration."""
    return {
        # Help
        'help': cmd_help,
        'h': cmd_help,
        '?': cmd_help,
        
        # Tempo/timing
        'bpm': cmd_bpm,
        'step': cmd_step,
        
        # Pattern
        'pat': cmd_pat,
        'apat': cmd_apat,
        
        # Level
        'lv': cmd_lv,
        'slv': cmd_slv,
        
        # File
        'file': cmd_file,
        
        # Project save/load
        'save': cmd_save,
        'load': cmd_load,
        
        # Clip
        'clip': cmd_clip,
        'ci': cmd_ci,
        
        # Chunk
        'chunk': cmd_chunk,
        'ch': cmd_ch,
        
        # Granular
        'gr': cmd_gr,
        
        # Modulation
        'imod': cmd_imod,
        'fmod': cmd_fmod,
        
        # Routing
        'route': cmd_route,
        'pn': cmd_pn,
        
        # Preset
        'preset': cmd_preset,
        
        # Userdata
        'userdata': cmd_userdata,
        
        # Duplicate
        'dup': cmd_dup,
        
        # Tracks
        'tracks': cmd_tracks,
        
        # Constant
        'const': cmd_const,
        '.': cmd_const,
        
        # Creation
        'new': cmd_new,
    }



# ============================================================================
# SIMPLIFIED TRACK WORKFLOW COMMANDS (continuous track audio)
# ============================================================================

def cmd_tlen(session: Session, args: List[str]) -> str:
    """Set project/track length in seconds (resets all track audio).

    Usage:
      /tlen <seconds>

    Example:
      /tlen 60    -> 60 second project, re-init tracks to silence
    """
    if not args:
        return f"OK: project length = {getattr(session,'project_length_seconds',0):.2f}s"
    try:
        seconds = float(args[0])
        session.set_project_length_seconds(seconds, reset_audio=True)
        return f"OK: project length set to {session.project_length_seconds:.2f}s (tracks reset)"
    except Exception as exc:
        return f"ERROR: {exc}"


def cmd_tsel(session: Session, args: List[str]) -> str:
    """Select a track (1-based).

    Usage:
      /tsel <n>
    """
    if not args:
        return f"OK: current track = {session.current_track_index+1}"
    try:
        n = max(1, int(float(args[0])))
        session.ensure_track_index(n)
        session.current_track_index = n - 1
        t = session.get_current_track()
        return f"OK: selected track {n} ({t.get('name','track')})"
    except Exception as exc:
        return f"ERROR: {exc}"


def cmd_tpos(session: Session, args: List[str]) -> str:
    """Set the current track write cursor position in seconds.

    Usage:
      /tpos <seconds>
    """
    if not args:
        t = session.get_current_track()
        return f"OK: cursor = {t.get('write_pos',0)/session.sample_rate:.3f}s"
    try:
        sec = float(args[0])
        session.set_track_cursor_seconds(sec)
        t = session.get_current_track()
        return f"OK: cursor set to {t.get('write_pos',0)/session.sample_rate:.3f}s"
    except Exception as exc:
        return f"ERROR: {exc}"


def cmd_twrite(session: Session, args: List[str]) -> str:
    """Write last_buffer into the current track at the cursor.

    Usage:
      /twrite            -> overwrite write
      /twrite add        -> additive write (sum into track)

    Notes:
      - This is the main 'commit audio to timeline' action in the simplified model.
    """
    if session.last_buffer is None or len(session.last_buffer) == 0:
        return "ERROR: no last_buffer to write. Generate audio first."
    mode = 'overwrite'
    if args and args[0].lower() in ('add','sum','plus'):
        mode = 'add'
    try:
        start, end = session.write_to_track(session.last_buffer, mode=mode)
        return f"OK: wrote {mode} to track {session.current_track_index+1} at {start/session.sample_rate:.3f}s → {end/session.sample_rate:.3f}s"
    except Exception as exc:
        return f"ERROR: {exc}"


def cmd_ta(session: Session, args: List[str]) -> str:
    """Append audio to current track at cursor (like /wa but writes to track).

    Usage:
      /ta                      Append working buffer to track
      /ta tone <hz> [beats]    Append tone to track
      /ta t <hz> [beats]       Short for /ta tone
      /ta silence <beats>      Append silence to track
      /ta s <beats>            Short for /ta silence
      /ta mel <pat> [hz]       Append melody to track
      /ta m <pat> [hz]         Short for /ta mel
      /ta cor <pat> [hz]       Append chord sequence to track
      /ta c <pat> [hz]         Short for /ta cor

    Mode modifiers (add before subcommand):
      /ta add tone ...         Additive (sum) instead of overwrite
      /ta add mel ...          Additive melody

    All timing is BPM-locked (1 unit = 1 beat).
    Cursor advances after each write.
    """
    import numpy as np
    from .dsl_cmds import (
        render_tone, render_events, render_chord_events,
        parse_melody_pattern, parse_chord_pattern, note_to_hz,
    )

    sr = getattr(session, 'sample_rate', 44100)
    bpm = getattr(session, 'bpm', 120.0)
    mode = 'overwrite'
    track_n = session.current_track_index + 1

    if not args:
        # Append working buffer to track
        if not session.has_real_working_audio():
            return "ERROR: working buffer is empty. Generate audio first."
        buf = session.working_buffer
        try:
            start, end = session.write_to_track(buf, mode=mode)
            dur = (end - start) / sr
            return f"TA: {dur:.2f}s -> track {track_n} at {start/sr:.3f}s"
        except Exception as exc:
            return f"ERROR: {exc}"

    # Check for mode modifier
    offset = 0
    if args[0].lower() in ('add', 'sum', 'plus'):
        mode = 'add'
        offset = 1

    if offset >= len(args):
        # Just /ta add with no subcommand — append working in add mode
        if not session.has_real_working_audio():
            return "ERROR: working buffer is empty."
        buf = session.working_buffer
        start, end = session.write_to_track(buf, mode=mode)
        dur = (end - start) / sr
        return f"TA: {dur:.2f}s ({mode}) -> track {track_n} at {start/sr:.3f}s"

    cmd = args[offset].lower()
    sub_args = args[offset + 1:]

    # Silence
    if cmd in ('silence', 's'):
        beats = float(sub_args[0]) if sub_args else 1.0
        dur_sec = beats * 60.0 / bpm
        silence = np.zeros(int(dur_sec * sr), dtype=np.float64)
        start, end = session.write_to_track(silence, mode=mode)
        return f"TA: +{beats} beats silence ({dur_sec:.2f}s) -> track {track_n} at {start/sr:.3f}s"

    # Tone
    if cmd in ('tone', 't'):
        freq = float(sub_args[0]) if len(sub_args) > 0 else 440.0
        beats = float(sub_args[1]) if len(sub_args) > 1 else 1.0
        amp = float(sub_args[2]) if len(sub_args) > 2 else 0.5
        try:
            tone = session.generate_tone(freq, beats, amp)
            if tone is not None and len(tone) > 0:
                start, end = session.write_to_track(tone, mode=mode)
                dur_sec = (end - start) / sr
                return f"TA: +{freq}Hz ({beats} beats = {dur_sec:.2f}s) ({mode}) -> track {track_n} at {start/sr:.3f}s"
        except Exception:
            pass
        dur_sec = beats * 60.0 / bpm
        tone = render_tone(freq, dur_sec, amp, sr)
        start, end = session.write_to_track(tone, mode=mode)
        return f"TA: +{freq}Hz ({beats} beats) -> track {track_n} at {start/sr:.3f}s"

    # Melody
    if cmd in ('mel', 'm'):
        pattern = sub_args[0] if sub_args else '0.4.7'
        root_hz = 440.0
        sydef_name = None
        for arg in sub_args[1:]:
            if arg.lower().startswith('sydef='):
                sydef_name = arg.split('=', 1)[1].lower()
                continue
            try:
                root_hz = float(arg)
            except ValueError:
                pass
        events = parse_melody_pattern(pattern)
        if not events:
            return "ERROR: no notes in pattern"
        audio = render_events(events, root_hz, sr,
                              session=session, sydef_name=sydef_name)
        if len(audio) == 0:
            return "ERROR: pattern produces no audio"
        start, end = session.write_to_track(audio, mode=mode)
        note_count = sum(1 for e in events if e['type'] == 'note')
        dur = (end - start) / sr
        sd_info = f" sydef={sydef_name}" if sydef_name else ""
        return f"TA: +{note_count} notes{sd_info} ({dur:.2f}s) ({mode}) -> track {track_n} at {start/sr:.3f}s"

    # Chord
    if cmd in ('cor', 'chord', 'c'):
        pattern = sub_args[0] if sub_args else '0,4,7'
        root_hz = 440.0
        sydef_name = None
        for arg in sub_args[1:]:
            if arg.lower().startswith('sydef='):
                sydef_name = arg.split('=', 1)[1].lower()
                continue
            try:
                root_hz = float(arg)
            except ValueError:
                pass
        events = parse_chord_pattern(pattern)
        if not events:
            return "ERROR: no chords in pattern"
        audio = render_chord_events(events, root_hz, sr,
                                    session=session, sydef_name=sydef_name)
        if len(audio) == 0:
            return "ERROR: pattern produces no audio"
        start, end = session.write_to_track(audio, mode=mode)
        chord_count = sum(1 for e in events if e['type'] == 'chord')
        dur = (end - start) / sr
        sd_info = f" sydef={sydef_name}" if sydef_name else ""
        return f"TA: +{chord_count} chords{sd_info} ({dur:.2f}s) ({mode}) -> track {track_n} at {start/sr:.3f}s"

    return f"ERROR: /ta unknown '{cmd}'. Use: tone/t, silence/s, mel/m, cor/c, or no args for working buffer."


def cmd_wta(session: Session, args: List[str]) -> str:
    """Commit working buffer to current track and clear working.

    This is the track equivalent of /wa (which commits working to
    a numbered buffer).  After commit, working buffer is reset.

    Usage:
      /wta               Commit working to track (overwrite at cursor)
      /wta add           Commit working to track (additive)
      /wta <track_n>     Commit working to specific track (1-based)
      /wta <track_n> add Commit to specific track, additive

    Cursor advances past the written region.
    """
    import numpy as np

    sr = getattr(session, 'sample_rate', 44100)

    if not session.has_real_working_audio():
        return "ERROR: working buffer is empty. Use /mel, /wa tone, etc. first."

    mode = 'overwrite'
    track_idx = None

    for arg in args:
        low = arg.lower()
        if low in ('add', 'sum', 'plus'):
            mode = 'add'
        else:
            try:
                track_idx = int(arg)
            except ValueError:
                pass

    buf = session.working_buffer
    dur = len(buf) / sr

    try:
        if track_idx is not None:
            start, end = session.write_to_track(buf, mode=mode, track_idx=track_idx)
            target = track_idx
        else:
            start, end = session.write_to_track(buf, mode=mode)
            target = session.current_track_index + 1
    except Exception as exc:
        return f"ERROR: {exc}"

    # Keep in last_buffer
    session.last_buffer = buf.astype(np.float64).copy()
    # Reset working
    session.working_buffer = np.zeros(sr, dtype=np.float64)
    session.working_buffer_source = 'init'

    return f"WTA: committed {dur:.2f}s ({mode}) -> track {target} at {start/sr:.3f}s"


def cmd_btw(session: Session, args: List[str]) -> str:
    """Bounce track to working buffer (and back).

    Copies the ENTIRE current track audio into the working buffer so
    you can process it with working-buffer commands (/fx, /wa mel, etc.),
    then use /btw back (or /wta) to write it back to the track.

    Usage:
      /btw               Bounce current track -> working buffer
      /btw <n>           Bounce track n -> working buffer
      /btw back          Write working buffer back to track (overwrite whole track)
      /btw back add      Write back additively

    Typical workflow:
      /btw               Pull track into working
      /fx reverb         Process it
      /fx normalize      Clean up levels
      /btw back          Put it back

    The track cursor is reset to 0 for the write-back so the audio
    replaces the entire track contents.
    """
    import numpy as np

    sr = getattr(session, 'sample_rate', 44100)

    if args and args[0].lower() in ('back', 'wb', 'return', 'put'):
        # Write working buffer back to track
        if not session.has_real_working_audio():
            return "ERROR: working buffer is empty. Nothing to write back."
        mode = 'overwrite'
        if len(args) > 1 and args[1].lower() in ('add', 'sum', 'plus'):
            mode = 'add'
        # Reset cursor to 0 so we overwrite from the start
        track = session.get_current_track()
        track['write_pos'] = 0
        buf = session.working_buffer
        try:
            start, end = session.write_to_track(buf, mode=mode)
        except Exception as exc:
            return f"ERROR: {exc}"
        dur = (end - start) / sr
        track_n = session.current_track_index + 1
        # Also keep in last_buffer
        session.last_buffer = buf.astype(np.float64).copy()
        # Clear working
        session.working_buffer = np.zeros(sr, dtype=np.float64)
        session.working_buffer_source = 'init'
        return f"BTW: wrote back {dur:.2f}s ({mode}) -> track {track_n}"

    # Bounce track -> working
    track_idx = None
    for arg in args:
        try:
            track_idx = int(arg)
        except ValueError:
            pass

    if track_idx is not None:
        session.ensure_track_index(track_idx)
        track = session.tracks[track_idx - 1]
        track_n = track_idx
    else:
        track = session.get_current_track()
        track_n = session.current_track_index + 1

    t_audio = track.get('audio')
    if t_audio is None or (hasattr(t_audio, '__len__') and len(t_audio) == 0):
        return f"ERROR: track {track_n} is empty."

    # Convert to mono float64 for working buffer
    if t_audio.ndim == 2:
        mono = np.mean(t_audio, axis=1).astype(np.float64)
    else:
        mono = t_audio.astype(np.float64)

    # Trim trailing silence for efficiency
    abs_mono = np.abs(mono)
    nonzero = np.where(abs_mono > 1e-7)[0]
    if len(nonzero) == 0:
        return f"ERROR: track {track_n} contains only silence."
    last_nonzero = nonzero[-1]
    # Keep a small tail (100ms)
    end_sample = min(len(mono), last_nonzero + int(0.1 * sr))
    trimmed = mono[:end_sample]

    session.working_buffer = trimmed
    session.working_buffer_source = f'btw:track{track_n}'
    session.last_buffer = trimmed.astype(np.float64).copy()
    dur = len(trimmed) / sr

    return f"BTW: bounced track {track_n} -> working ({dur:.2f}s). Process with /fx etc., then /btw back."


def cmd_tgain(session: Session, args: List[str]) -> str:
    """Set current track gain in dB.

    Usage:
      /tgain           -> Show current track gain
      /tgain <dB>      -> Set gain (0 = unity, -6 = half, +6 = double)
      /tgain <n> <dB>  -> Set gain for track n
    """
    if not args:
        t = session.get_current_track()
        gain = t.get('gain', 1.0)
        db = 20 * np.log10(max(gain, 1e-10))
        return f"OK: track {session.current_track_index+1} gain = {gain:.3f} ({db:.1f} dB)"
    try:
        if len(args) >= 2:
            n = int(args[0])
            db = float(args[1])
            session.ensure_track_index(n)
            t = session.tracks[n - 1]
        else:
            db = float(args[0])
            t = session.get_current_track()
            n = session.current_track_index + 1
        gain = 10 ** (db / 20.0)
        t['gain'] = gain
        return f"OK: track {n} gain = {gain:.3f} ({db:.1f} dB)"
    except Exception as exc:
        return f"ERROR: {exc}"


def cmd_tpan(session: Session, args: List[str]) -> str:
    """Set current track pan position.

    Usage:
      /tpan             -> Show current pan
      /tpan <value>     -> Set pan (-100=full left, 0=center, 100=full right)
      /tpan <n> <value> -> Set pan for track n

    Values: -100 to 100 (mapped to -1.0 to 1.0 internally)
    Shortcuts: L = -100, C = 0, R = 100
    """
    if not args:
        t = session.get_current_track()
        pan = t.get('pan', 0.0)
        label = "L" if pan < -0.05 else "R" if pan > 0.05 else "C"
        return f"OK: track {session.current_track_index+1} pan = {pan:.2f} ({label})"
    try:
        if len(args) >= 2:
            n = int(args[0])
            val_str = args[1]
            session.ensure_track_index(n)
            t = session.tracks[n - 1]
        else:
            val_str = args[0]
            t = session.get_current_track()
            n = session.current_track_index + 1
        # Parse shortcuts
        val_str = val_str.upper()
        if val_str == 'L':
            pan = -1.0
        elif val_str == 'C':
            pan = 0.0
        elif val_str == 'R':
            pan = 1.0
        else:
            pan = float(val_str)
            if abs(pan) > 1.0:
                pan = pan / 100.0  # Convert 0-100 scale to 0-1
            pan = max(-1.0, min(1.0, pan))
        t['pan'] = pan
        label = "L" if pan < -0.05 else "R" if pan > 0.05 else "C"
        return f"OK: track {n} pan = {pan:.2f} ({label})"
    except Exception as exc:
        return f"ERROR: {exc}"


def cmd_tmute(session: Session, args: List[str]) -> str:
    """Toggle mute on a track.

    Usage:
      /tmute          -> Toggle mute on current track
      /tmute <n>      -> Toggle mute on track n
    """
    try:
        if args:
            n = int(args[0])
            session.ensure_track_index(n)
            t = session.tracks[n - 1]
        else:
            t = session.get_current_track()
            n = session.current_track_index + 1
        t['mute'] = not t.get('mute', False)
        state = "MUTED" if t['mute'] else "UNMUTED"
        return f"OK: track {n} {state}"
    except Exception as exc:
        return f"ERROR: {exc}"


def cmd_tsolo(session: Session, args: List[str]) -> str:
    """Toggle solo on a track.

    Usage:
      /tsolo          -> Toggle solo on current track
      /tsolo <n>      -> Toggle solo on track n
    """
    try:
        if args:
            n = int(args[0])
            session.ensure_track_index(n)
            t = session.tracks[n - 1]
        else:
            t = session.get_current_track()
            n = session.current_track_index + 1
        t['solo'] = not t.get('solo', False)
        state = "SOLO" if t['solo'] else "UNSOLO"
        return f"OK: track {n} {state}"
    except Exception as exc:
        return f"ERROR: {exc}"


def cmd_tinfo(session: Session, args: List[str]) -> str:
    """Show detailed info for a track.

    Usage:
      /tinfo          -> Info for current track
      /tinfo <n>      -> Info for track n
    """
    try:
        if args:
            n = int(args[0])
            session.ensure_track_index(n)
            t = session.tracks[n - 1]
        else:
            t = session.get_current_track()
            n = session.current_track_index + 1
        name = t.get('name', f'track_{n}')
        audio = t.get('audio')
        if audio is not None and hasattr(audio, 'shape') and audio.size > 0:
            if audio.ndim == 2:
                dur = audio.shape[0] / session.sample_rate
                peak = float(np.max(np.abs(audio)))
                ch = f"stereo ({audio.shape[1]}ch)"
            elif audio.ndim == 1:
                dur = len(audio) / session.sample_rate
                peak = float(np.max(np.abs(audio)))
                ch = "mono"
            else:
                dur = 0.0
                peak = 0.0
                ch = "unknown"
        else:
            dur = 0.0
            peak = 0.0
            ch = "empty"
        cursor = t.get('write_pos', 0) / session.sample_rate
        gain = t.get('gain', 1.0)
        gain_db = 20 * np.log10(max(gain, 1e-10))
        pan = t.get('pan', 0.0)
        mute = "YES" if t.get('mute') else "no"
        solo = "YES" if t.get('solo') else "no"
        fx_count = len(t.get('fx_chain', []))
        lines = [
            f"TRACK {n}: {name}",
            f"  Format: {ch}, {dur:.2f}s",
            f"  Peak: {peak:.4f}",
            f"  Cursor: {cursor:.3f}s",
            f"  Gain: {gain:.3f} ({gain_db:.1f} dB)",
            f"  Pan: {pan:.2f}",
            f"  Mute: {mute}, Solo: {solo}",
            f"  FX chain: {fx_count} effect(s)",
        ]
        return '\n'.join(lines)
    except Exception as exc:
        return f"ERROR: {exc}"
