"""Buffer Management Commands for MDMA.

Multi-buffer system for building complex sounds without touching clips.

BUFFER SYSTEM:
-------------
Buffers are indexed working areas (1-indexed for accessibility).
Each buffer tracks its content and last append position.
Pattern commands can target just the last appended section.

COMMANDS:
--------
/buf              - Show current buffer index and info
/bu <n>           - Select buffer by index (1-indexed)
/bu+ <n>          - Set total buffer count
/a <cmd>          - Append command output to buffer (not overwrite)
/ex <val|cmd>     - Extend buffer with silence or command output

PLAYBACK:
--------
/p or /play       - Play current buffer
/p <n>            - Play buffer at index n
/pa               - Play all buffers + sketch (pre-render)

RENDER:
------
/b                - Render current buffer + sketch as separate files
/ba               - Render all buffers + sketch as separate files
/bo               - Render omni (everything combined into one file)

PATTERN:
-------
/pat              - Apply pattern to LAST APPENDED section only
/apat             - Apply pattern to ENTIRE buffer

CLIPS:
-----
/bc <beat>        - Insert current buffer as clip at beat position

BUILD ID: buffer_cmds_v14.3
"""

from __future__ import annotations

import os
import time
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..core.session import Session


# ============================================================================
# BUFFER INFO AND SELECTION
# ============================================================================

def cmd_buf(session: "Session", args: List[str]) -> str:
    """Show current buffer index and info.
    
    Usage:
      /buf              -> Show current buffer info
      /buf all          -> Show all buffers info
    
    Shows:
      - Current buffer index
      - Buffer length in samples and seconds
      - Last append position
      - Peak amplitude
    """
    # Initialize buffer system if needed
    _ensure_buffer_system(session)
    
    if args and args[0].lower() == 'all':
        lines = ["All Buffers:"]
        lines.append("-" * 50)
        for idx in sorted(session.buffers.keys()):
            buf = session.buffers[idx]
            info = _format_buffer_info(session, idx, buf)
            marker = " <-- current" if idx == session.current_buffer_index else ""
            lines.append(f"  {info}{marker}")
        lines.append("-" * 50)
        lines.append(f"Total: {len(session.buffers)} buffers")
        return '\n'.join(lines)
    
    idx = session.current_buffer_index
    buf = session.buffers.get(idx)
    
    if buf is None or len(buf) == 0:
        return f"Buffer {idx}: (empty)"
    
    return _format_buffer_info_detailed(session, idx, buf)


def cmd_bu(session: "Session", args: List[str]) -> str:
    """Select buffer by index or set buffer count.
    
    Usage:
      /bu <n>           -> Select buffer n (1-indexed)
      /bu               -> Show current buffer index
    
    Examples:
      /bu 1             -> Select buffer 1
      /bu 3             -> Select buffer 3 (creates if needed)
    """
    _ensure_buffer_system(session)
    
    if not args:
        return f"Current buffer: {session.current_buffer_index}"
    
    try:
        idx = int(args[0])
        if idx < 1:
            return "ERROR: buffer index must be >= 1"
        
        # Create buffer if doesn't exist
        if idx not in session.buffers:
            session.buffers[idx] = np.zeros(0, dtype=np.float64)
            session.buffer_append_positions[idx] = 0
        
        session.current_buffer_index = idx
        
        # Update playback context to buffer mode
        try:
            from .playback_cmds import get_playback_context
            ctx = get_playback_context()
            ctx.set_context('buffer', idx)
        except ImportError:
            pass
        
        # Update last_buffer to point to current buffer
        if len(session.buffers[idx]) > 0:
            session.last_buffer = session.buffers[idx]
        
        buf = session.buffers[idx]
        if len(buf) > 0:
            dur = len(buf) / session.sample_rate
            return f"OK: selected buffer {idx} ({len(buf)} samples, {dur:.3f}s)"
        else:
            return f"OK: selected buffer {idx} (empty)"
    except ValueError:
        return f"ERROR: invalid buffer index '{args[0]}'"


def cmd_bu_plus(session: "Session", args: List[str]) -> str:
    """Set total buffer count (create multiple empty buffers).
    
    Usage:
      /bu+ <n>          -> Ensure at least n buffers exist
    
    Example:
      /bu+ 8            -> Create buffers 1-8
    """
    _ensure_buffer_system(session)
    
    if not args:
        return f"Current buffer count: {len(session.buffers)}"
    
    try:
        count = int(args[0])
        if count < 1:
            return "ERROR: buffer count must be >= 1"
        
        created = 0
        for i in range(1, count + 1):
            if i not in session.buffers:
                session.buffers[i] = np.zeros(0, dtype=np.float64)
                session.buffer_append_positions[i] = 0
                created += 1
        
        return f"OK: {len(session.buffers)} buffers available ({created} created)"
    except ValueError:
        return f"ERROR: invalid count '{args[0]}'"


def cmd_clr(session: "Session", args: List[str]) -> str:
    """Clear buffer(s) and/or last_buffer.
    
    Usage:
      /clr              Clear current buffer AND last_buffer
      /clr <n>          Clear specific buffer
      /clr all          Clear ALL buffers
      /clr last         Clear only last_buffer (pending audio)
    
    Examples:
      /clr              Clear current context
      /clr 2            Clear buffer 2
      /clr all          Start fresh
    """
    _ensure_buffer_system(session)
    
    if not args:
        # Clear current buffer and last_buffer
        idx = session.current_buffer_index
        if idx in session.buffers:
            session.buffers[idx] = np.zeros(0, dtype=np.float64)
            session.buffer_append_positions[idx] = 0
        session.last_buffer = None
        return f"OK: Cleared buffer {idx} and last_buffer"
    
    arg = args[0].lower()
    
    if arg == 'all':
        count = len(session.buffers)
        for idx in list(session.buffers.keys()):
            session.buffers[idx] = np.zeros(0, dtype=np.float64)
            session.buffer_append_positions[idx] = 0
        session.last_buffer = None
        return f"OK: Cleared all {count} buffers"
    
    if arg == 'last':
        session.last_buffer = None
        return "OK: Cleared last_buffer (pending audio)"
    
    try:
        idx = int(arg)
        if idx in session.buffers:
            session.buffers[idx] = np.zeros(0, dtype=np.float64)
            session.buffer_append_positions[idx] = 0
            return f"OK: Cleared buffer {idx}"
        return f"ERROR: buffer {idx} doesn't exist"
    except ValueError:
        return f"ERROR: invalid argument '{arg}'"


# ============================================================================
# APPEND AND EXTEND
# ============================================================================

def cmd_a(session: "Session", args: List[str]) -> str:
    """Append/insert pending audio into current buffer at position.
    
    Usage:
      /a              -> Append at END of buffer (default)
      /a 0            -> Insert at BEGINNING
      /a <position>   -> Insert at specified sample position
      /a <seconds>s   -> Insert at time position (e.g., /a 1.5s)
    
    The pending audio is whatever is in last_buffer (from /tone, /g, etc).
    Position 0 = very beginning of buffer.
    No argument = append at end.
    
    The append position is tracked so /pat can apply patterns to
    just the newly appended section.
    
    Examples:
      /tone 440 0.5 0.8    -> Generate tone (stored in last_buffer)
      /a                   -> Append at end
      
      /tone 880 0.5 0.6    -> Generate another tone
      /a 0                 -> Insert at beginning
      
      /tone 660 0.3 0.5    -> Generate third tone
      /a 1.0s              -> Insert at 1 second mark
    """
    _ensure_buffer_system(session)
    
    # Check for pending audio
    if session.last_buffer is None or len(session.last_buffer) == 0:
        return "ERROR: no pending audio. Generate with /tone, /g, etc. first."
    
    new_audio = session.last_buffer.copy()
    idx = session.current_buffer_index
    current_buf = session.buffers.get(idx, np.zeros(0, dtype=np.float64))
    
    # Parse position argument
    if not args:
        # No args = append at end
        insert_pos = len(current_buf)
    else:
        pos_str = args[0].lower()
        
        # Check for time notation (e.g., "1.5s")
        if pos_str.endswith('s'):
            try:
                seconds = float(pos_str[:-1])
                insert_pos = int(seconds * session.sample_rate)
            except ValueError:
                return f"ERROR: invalid time position '{pos_str}'"
        else:
            # Numeric position (samples)
            try:
                insert_pos = int(float(args[0]))
            except ValueError:
                return f"ERROR: invalid position '{args[0]}'. Use number or time (e.g., 1.5s)"
    
    # Clamp position to valid range
    insert_pos = max(0, min(insert_pos, len(current_buf)))
    
    # Track append position (start of new content)
    append_start = insert_pos
    
    # Insert audio at position
    if len(current_buf) == 0:
        # Empty buffer - just use new audio
        combined = new_audio.copy()
    elif insert_pos == 0:
        # Insert at beginning with crossfade
        combined = _crossfade_append(new_audio, current_buf, session.sample_rate)
    elif insert_pos >= len(current_buf):
        # Append at end with crossfade
        combined = _crossfade_append(current_buf, new_audio, session.sample_rate)
    else:
        # Insert in middle
        before = current_buf[:insert_pos]
        after = current_buf[insert_pos:]
        
        # Crossfade before->new and new->after
        first_part = _crossfade_append(before, new_audio, session.sample_rate)
        combined = _crossfade_append(first_part, after, session.sample_rate)
    
    # Store in buffer system
    session.buffers[idx] = combined
    session.buffer_append_positions[idx] = append_start
    
    # Also update last_buffer to the combined result
    session.last_buffer = combined
    
    new_dur = len(new_audio) / session.sample_rate
    total_dur = len(combined) / session.sample_rate
    pos_sec = insert_pos / session.sample_rate
    
    if insert_pos == 0:
        pos_desc = "beginning"
    elif insert_pos >= len(current_buf) - 1:
        pos_desc = "end"
    else:
        pos_desc = f"{pos_sec:.3f}s"
    
    return f"OK: inserted {new_dur:.3f}s at {pos_desc} -> buffer {idx} total {total_dur:.3f}s"


def cmd_ex(session: "Session", args: List[str]) -> str:
    """Extend buffer with silence or command output.
    
    Usage:
      /ex <seconds>            -> Extend with silence
      /ex <cmd> [args...]      -> Extend with command output
    
    Examples:
      /ex 0.5                   -> Extend with 0.5s silence
      /ex 1.0                   -> Extend with 1s silence
      /ex tone 440 0.5 0.5      -> Extend with tone
    """
    _ensure_buffer_system(session)
    
    if not args:
        return "ERROR: usage: /ex <seconds> OR /ex <command> [args...]"
    
    idx = session.current_buffer_index
    current_buf = session.buffers.get(idx, np.zeros(0, dtype=np.float64))
    
    # Check if first arg is a number (silence) or command
    try:
        seconds = float(args[0])
        # It's a number - extend with silence
        silence_samples = int(seconds * session.sample_rate)
        silence = np.zeros(silence_samples, dtype=np.float64)
        
        if len(current_buf) > 0:
            combined = np.concatenate([current_buf, silence])
        else:
            combined = silence
        
        session.buffers[idx] = combined
        session.buffer_append_positions[idx] = len(current_buf)
        session.last_buffer = combined
        
        return f"OK: extended buffer {idx} with {seconds:.3f}s silence -> {len(combined)} samples"
    except ValueError:
        # It's a command
        cmd_name = args[0].lower()
        cmd_args = args[1:] if len(args) > 1 else []
        
        result = _execute_generation_command(session, cmd_name, cmd_args)
        if result.startswith("ERROR"):
            return result
        
        new_audio = session.last_buffer
        if new_audio is None or len(new_audio) == 0:
            return "ERROR: command produced no audio"
        
        append_start = len(current_buf)
        
        if len(current_buf) > 0:
            combined = np.concatenate([current_buf, new_audio])
        else:
            combined = new_audio.copy()
        
        session.buffers[idx] = combined
        session.buffer_append_positions[idx] = append_start
        session.last_buffer = combined
        
        new_dur = len(new_audio) / session.sample_rate
        return f"OK: extended buffer {idx} with command output ({new_dur:.3f}s)"


# ============================================================================
# PLAYBACK COMMANDS
# ============================================================================

def cmd_p(session: "Session", args: List[str]) -> str:
    """Play working buffer or buffer by index.
    
    Usage:
      /p                -> Play working buffer
      /p <n>            -> Play buffer n (shortcut for /pb <n>)
    
    Note: /p without arguments plays the working buffer.
    For explicit buffer playback, use /pb <n>.
    For track playback, use /pt <n>.
    For song (all tracks), use /pts.
    """
    _ensure_buffer_system(session)
    
    # If argument given, treat as buffer index shortcut
    if args:
        try:
            from .playback_cmds import cmd_pb
            return cmd_pb(session, args)
        except ImportError:
            pass
        
        # Fallback: simple buffer playback
        try:
            idx = int(args[0])
            if idx not in session.buffers or len(session.buffers[idx]) == 0:
                return f"ERROR: buffer {idx} is empty or doesn't exist"
            buf = session.buffers[idx]
            session.last_buffer = buf
            dur = len(buf) / session.sample_rate
            if session._play_buffer(buf, 0.8):
                return f"▶ Playing: buffer {idx} ({dur:.2f}s) @ {session.sample_rate}Hz"
            return f"ERROR: Playback failed"
        except ValueError:
            return f"ERROR: invalid buffer index '{args[0]}'"
    
    # No args: delegate to playback_cmds for working buffer
    try:
        from .playback_cmds import cmd_p as play_working
        return play_working(session, [])
    except ImportError:
        pass
    
    # Fallback: play current buffer
    idx = session.current_buffer_index
    buf = session.buffers.get(idx)
    if buf is None or len(buf) == 0:
        if session.last_buffer is not None and len(session.last_buffer) > 0:
            buf = session.last_buffer
            idx = "last"
        else:
            return "ERROR: No audio to play. Generate with /tone, /mel, etc."
    
    session.last_buffer = buf
    dur = len(buf) / session.sample_rate
    if session._play_buffer(buf, 0.8):
        return f"▶ Playing: buffer {idx} ({dur:.2f}s) @ {session.sample_rate}Hz"
    return f"ERROR: Playback failed"


def cmd_pa(session: "Session", args: List[str]) -> str:
    """Play all buffers combined with sketch (pre-render).
    
    Usage:
      /pa               -> Render and play all buffers + sketch
    
    Combines all non-empty buffers and any sketch content,
    renders to a temp file, and plays.
    """
    _ensure_buffer_system(session)
    
    # Collect all non-empty buffers
    all_buffers = []
    for idx in sorted(session.buffers.keys()):
        buf = session.buffers[idx]
        if buf is not None and len(buf) > 0:
            all_buffers.append((idx, buf))
    
    if not all_buffers:
        return "ERROR: no buffers with content"
    
    # Find max length
    max_len = max(len(buf) for _, buf in all_buffers)
    
    # Mix all buffers (simple sum with scaling)
    combined = np.zeros(max_len, dtype=np.float64)
    for idx, buf in all_buffers:
        combined[:len(buf)] += buf
    
    # Normalize
    peak = np.max(np.abs(combined))
    if peak > 1.0:
        combined = combined / peak * 0.95
    
    # Play
    session.last_buffer = combined
    session.play()
    
    dur = len(combined) / session.sample_rate
    return f"Playing all {len(all_buffers)} buffers combined ({dur:.3f}s)"


# ============================================================================
# RENDER COMMANDS
# ============================================================================

def cmd_b(session: "Session", args: List[str]) -> str:
    """Render current buffer (and sketch) as separate files.
    
    Usage:
      /b                -> Render current buffer to file
      /b <filename>     -> Render with custom filename
    
    Creates WAV file in outputs directory.
    """
    _ensure_buffer_system(session)
    
    idx = session.current_buffer_index
    buf = session.buffers.get(idx)
    
    if buf is None or len(buf) == 0:
        return f"ERROR: buffer {idx} is empty"
    
    # Parse format arguments
    # /b [filename] [format] [bits]
    filename = None
    out_format = session.output_format
    bit_depth = session.output_bit_depth
    
    for arg in args:
        arg_lower = arg.lower()
        if arg_lower in ('wav', 'wave'):
            out_format = 'wav'
        elif arg_lower == 'flac':
            out_format = 'flac'
        elif arg_lower in ('16', '24', '32'):
            bit_depth = int(arg_lower)
        elif not filename:
            filename = arg
    
    # Generate filename
    ext = f'.{out_format}'
    if filename:
        if not filename.endswith(ext):
            # Remove any existing extension and add correct one
            if '.' in filename:
                filename = filename.rsplit('.', 1)[0]
            filename += ext
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"buffer_{idx}_{timestamp}{ext}"
    
    # Ensure outputs directory
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, filename)
    
    # Apply HQ render chain if enabled
    audio_out = buf.copy().astype(np.float64)
    if session.hq_mode:
        try:
            from .hq_cmds import apply_hq_chain
            audio_out = apply_hq_chain(session, audio_out)
        except ImportError:
            pass
    
    # Save with format selection
    try:
        from .hq_cmds import save_audio_hq
        actual_path = save_audio_hq(audio_out, out_path, session.sample_rate,
                                    format=out_format, bit_depth=bit_depth)
        dur = len(buf) / session.sample_rate
        hq_str = " [HQ]" if session.hq_mode else ""
        return f"OK: rendered buffer {idx} to {os.path.basename(actual_path)} ({dur:.3f}s, {out_format.upper()} {bit_depth}-bit){hq_str}"
    except ImportError:
        # Fallback to basic soundfile
        try:
            import soundfile as sf
            sf.write(out_path, audio_out, session.sample_rate)
            dur = len(buf) / session.sample_rate
            return f"OK: rendered buffer {idx} to {filename} ({dur:.3f}s)"
        except ImportError:
            # Fallback to scipy
            try:
                from scipy.io import wavfile
                wavfile.write(out_path, session.sample_rate, (audio_out * 32767).astype(np.int16))
                return f"OK: rendered buffer {idx} to {filename}"
            except Exception as e:
                return f"ERROR: could not write file: {e}"


def cmd_ba(session: "Session", args: List[str]) -> str:
    """Render all buffers as separate files.
    
    Usage:
      /ba               -> Render all buffers to separate files
      /ba <prefix>      -> Use custom prefix for filenames
      /ba flac          -> Export as FLAC
      /ba 24            -> Export as 24-bit
    
    Creates one file per non-empty buffer in session's output format.
    """
    _ensure_buffer_system(session)
    
    # Parse arguments
    prefix = "buffer"
    out_format = session.output_format
    bit_depth = session.output_bit_depth
    
    for arg in args:
        arg_lower = arg.lower()
        if arg_lower in ('wav', 'wave'):
            out_format = 'wav'
        elif arg_lower == 'flac':
            out_format = 'flac'
        elif arg_lower in ('16', '24', '32'):
            bit_depth = int(arg_lower)
        else:
            prefix = arg
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ext = f'.{out_format}'
    
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    rendered = []
    for idx in sorted(session.buffers.keys()):
        buf = session.buffers[idx]
        if buf is None or len(buf) == 0:
            continue
        
        filename = f"{prefix}_{idx}_{timestamp}{ext}"
        out_path = os.path.join(out_dir, filename)
        
        # Apply HQ chain
        audio_out = buf.copy().astype(np.float64)
        if session.hq_mode:
            try:
                from .hq_cmds import apply_hq_chain
                audio_out = apply_hq_chain(session, audio_out)
            except ImportError:
                pass
        
        try:
            from .hq_cmds import save_audio_hq
            save_audio_hq(audio_out, out_path, session.sample_rate,
                         format=out_format, bit_depth=bit_depth)
            rendered.append(f"  {idx}: {filename}")
        except ImportError:
            try:
                import soundfile as sf
                sf.write(out_path, audio_out, session.sample_rate)
                rendered.append(f"  {idx}: {filename}")
            except ImportError:
                try:
                    from scipy.io import wavfile
                    wavfile.write(out_path, session.sample_rate, (audio_out * 32767).astype(np.int16))
                    rendered.append(f"  {idx}: {filename}")
                except Exception as e:
                    rendered.append(f"  {idx}: ERROR - {e}")
    
    if not rendered:
        return "ERROR: no non-empty buffers to render"
    
    hq_str = " [HQ]" if session.hq_mode else ""
    return f"OK: rendered {len(rendered)} buffers ({out_format.upper()} {bit_depth}-bit){hq_str}:\n" + '\n'.join(rendered)


def cmd_bo(session: "Session", args: List[str]) -> str:
    """Render omni - combine everything into one file.
    
    Usage:
      /bo               -> Render all buffers combined to one file
      /bo <filename>    -> Use custom filename
      /bo flac          -> Export as FLAC
      /bo 24            -> Export as 24-bit
    
    Mixes all buffers together with normalization and HQ processing.
    """
    _ensure_buffer_system(session)
    
    # Parse arguments
    filename = None
    out_format = session.output_format
    bit_depth = session.output_bit_depth
    
    for arg in args:
        arg_lower = arg.lower()
        if arg_lower in ('wav', 'wave'):
            out_format = 'wav'
        elif arg_lower == 'flac':
            out_format = 'flac'
        elif arg_lower in ('16', '24', '32'):
            bit_depth = int(arg_lower)
        elif not filename:
            filename = arg
    
    # Collect all non-empty buffers
    all_buffers = []
    for idx in sorted(session.buffers.keys()):
        buf = session.buffers[idx]
        if buf is not None and len(buf) > 0:
            all_buffers.append(buf)
    
    if not all_buffers:
        return "ERROR: no buffers with content"
    
    # Find max length
    max_len = max(len(buf) for buf in all_buffers)
    
    # Mix all
    combined = np.zeros(max_len, dtype=np.float64)
    for buf in all_buffers:
        combined[:len(buf)] += buf.astype(np.float64)
    
    # Normalize
    peak = np.max(np.abs(combined))
    if peak > 1.0:
        combined = combined / peak * 0.95
    
    # Apply HQ render chain if enabled
    if session.hq_mode:
        try:
            from .hq_cmds import apply_hq_chain
            combined = apply_hq_chain(session, combined)
        except ImportError:
            pass
    
    # Generate filename
    ext = f'.{out_format}'
    if filename:
        if not filename.endswith(ext):
            if '.' in filename:
                filename = filename.rsplit('.', 1)[0]
            filename += ext
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"omni_{timestamp}{ext}"
    
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    
    try:
        from .hq_cmds import save_audio_hq
        actual_path = save_audio_hq(combined, out_path, session.sample_rate,
                                    format=out_format, bit_depth=bit_depth)
        dur = len(combined) / session.sample_rate
        hq_str = " [HQ]" if session.hq_mode else ""
        return f"OK: rendered omni ({len(all_buffers)} buffers) to {os.path.basename(actual_path)} ({dur:.3f}s, {out_format.upper()} {bit_depth}-bit){hq_str}"
    except ImportError:
        try:
            import soundfile as sf
            sf.write(out_path, combined, session.sample_rate)
            dur = len(combined) / session.sample_rate
            return f"OK: rendered omni ({len(all_buffers)} buffers) to {filename} ({dur:.3f}s)"
        except ImportError:
            try:
                from scipy.io import wavfile
                wavfile.write(out_path, session.sample_rate, (combined * 32767).astype(np.int16))
                return f"OK: rendered omni to {filename}"
            except Exception as e:
                return f"ERROR: could not write file: {e}"


# ============================================================================
# CLIP INSERTION
# ============================================================================

def cmd_bc(session: "Session", args: List[str]) -> str:
    """Insert current buffer as clip at specified beat position.
    
    Usage:
      /bc <beat>        -> Insert buffer as clip at beat position
      /bc <beat> <name> -> Insert with custom clip name
    
    Examples:
      /bc 0             -> Insert at beat 0 (start)
      /bc 4             -> Insert at beat 4
      /bc 8 my_clip     -> Insert at beat 8 with name "my_clip"
    """
    _ensure_buffer_system(session)
    
    if not args:
        return "ERROR: usage: /bc <beat> [name]"
    
    try:
        beat = float(args[0])
    except ValueError:
        return f"ERROR: invalid beat position '{args[0]}'"
    
    idx = session.current_buffer_index
    buf = session.buffers.get(idx)
    
    if buf is None or len(buf) == 0:
        return f"ERROR: buffer {idx} is empty"
    
    # Generate clip name
    if len(args) > 1:
        clip_name = args[1]
    else:
        session.clip_count += 1
        clip_name = f"clip_{session.clip_count}"
    
    # Store clip
    session.clips[clip_name] = buf.copy()
    
    # Calculate sample position
    beat_samples = int(beat * 60.0 / session.bpm * session.sample_rate)
    
    # Track clip placement (for sketch rendering)
    if not hasattr(session, 'clip_placements'):
        session.clip_placements = {}
    session.clip_placements[clip_name] = beat_samples
    
    dur = len(buf) / session.sample_rate
    return f"OK: inserted buffer {idx} as clip '{clip_name}' at beat {beat} ({dur:.3f}s)"


# ============================================================================
# PATTERN COMMANDS (APPEND-AWARE)
# ============================================================================

def cmd_pat_buffer(session: "Session", args: List[str]) -> str:
    """Apply pattern to LAST APPENDED section of buffer only.
    
    This is the default /pat behavior when using the buffer system.
    It only affects the most recently appended audio, allowing
    you to build up complex sounds incrementally.
    
    Usage:
      /pat <tokens...>           -> Pattern on last append only
      /pat 0 7 12 /end 0         -> With algorithm
    """
    _ensure_buffer_system(session)
    
    idx = session.current_buffer_index
    buf = session.buffers.get(idx)
    
    if buf is None or len(buf) == 0:
        return "ERROR: buffer is empty"
    
    append_pos = session.buffer_append_positions.get(idx, 0)
    
    if append_pos >= len(buf):
        # No append section, apply to whole buffer
        return cmd_apat(session, args)
    
    # Extract last appended section
    last_section = buf[append_pos:].copy()
    
    if len(last_section) == 0:
        return "ERROR: no appended section to process"
    
    # Apply pattern to last section
    session.last_buffer = last_section
    
    # Use pattern command
    from .pattern_cmds import cmd_pat
    result = cmd_pat(session, args)
    
    if result.startswith("ERROR"):
        # Restore original last_buffer
        session.last_buffer = buf
        return result
    
    # Get processed section
    processed = session.last_buffer
    
    # Combine: original up to append_pos + processed section
    original_part = buf[:append_pos]
    if len(original_part) > 0:
        combined = _crossfade_append(original_part, processed, session.sample_rate)
    else:
        combined = processed
    
    # Update buffer
    session.buffers[idx] = combined
    session.last_buffer = combined
    
    return f"OK: pattern applied to last appended section ({len(last_section)} -> {len(processed)} samples)"


def cmd_apat(session: "Session", args: List[str]) -> str:
    """Apply pattern to ENTIRE buffer.
    
    Unlike /pat which only affects the last appended section,
    /apat applies the pattern to the whole buffer.
    
    Usage:
      /apat <tokens...>          -> Pattern on entire buffer
      /apat 0 7 12 /end 3        -> With algorithm
    """
    _ensure_buffer_system(session)
    
    idx = session.current_buffer_index
    buf = session.buffers.get(idx)
    
    if buf is None or len(buf) == 0:
        return "ERROR: buffer is empty"
    
    # Apply to entire buffer
    session.last_buffer = buf.copy()
    
    from .pattern_cmds import cmd_pat
    result = cmd_pat(session, args)
    
    if result.startswith("ERROR"):
        session.last_buffer = buf
        return result
    
    # Update buffer with processed result
    session.buffers[idx] = session.last_buffer.copy()
    
    return f"OK: pattern applied to entire buffer {idx} ({len(buf)} -> {len(session.last_buffer)} samples)"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _ensure_buffer_system(session: "Session") -> None:
    """Initialize buffer system on session if not present."""
    if not hasattr(session, 'buffers'):
        session.buffers: Dict[int, np.ndarray] = {1: np.zeros(0, dtype=np.float64)}
    if not hasattr(session, 'current_buffer_index'):
        session.current_buffer_index = 1
    if not hasattr(session, 'buffer_append_positions'):
        session.buffer_append_positions: Dict[int, int] = {1: 0}


def _format_buffer_info(session: "Session", idx: int, buf: np.ndarray) -> str:
    """Format short buffer info line."""
    if len(buf) == 0:
        return f"Buffer {idx}: (empty)"
    dur = len(buf) / session.sample_rate
    peak = np.max(np.abs(buf))
    return f"Buffer {idx}: {len(buf)} samples ({dur:.3f}s), peak {peak:.3f}"


def _format_buffer_info_detailed(session: "Session", idx: int, buf: np.ndarray) -> str:
    """Format detailed buffer info with deviation metrics."""
    dur = len(buf) / session.sample_rate
    peak = np.max(np.abs(buf))
    rms = np.sqrt(np.mean(buf ** 2))
    
    # Get append position if available
    append_pos = 0
    if hasattr(session, 'buffer_append_positions'):
        append_pos = session.buffer_append_positions.get(idx, 0)
    last_section_len = len(buf) - append_pos
    
    # Calculate deviation metrics
    mean_val = np.mean(buf)
    std_dev = np.std(buf)
    zero_crossings = np.sum(np.diff(np.sign(buf)) != 0)
    zc_rate = zero_crossings / dur if dur > 0 else 0
    
    # Crest factor (peak/RMS ratio) - indicates dynamic range
    crest_factor = peak / rms if rms > 0 else 0
    crest_db = 20 * np.log10(crest_factor) if crest_factor > 0 else 0
    
    lines = [f"Buffer {idx}:"]
    lines.append(f"  Samples: {len(buf)}")
    lines.append(f"  Duration: {dur:.3f}s")
    lines.append(f"  Peak: {peak:.4f} ({20*np.log10(max(peak, 1e-10)):.1f} dB)")
    lines.append(f"  RMS: {rms:.4f} ({20*np.log10(max(rms, 1e-10)):.1f} dB)")
    lines.append(f"  Crest: {crest_db:.1f} dB (peak/RMS)")
    lines.append(f"  StdDev: {std_dev:.4f}")
    lines.append(f"  DC Offset: {mean_val:.6f}")
    lines.append(f"  Zero-crossings: {zero_crossings} ({zc_rate:.0f}/s)")
    if append_pos > 0:
        lines.append(f"  Last append at: {append_pos} samples")
        lines.append(f"  Last section: {last_section_len} samples ({last_section_len/session.sample_rate:.3f}s)")
    
    return '\n'.join(lines)


def _crossfade_append(buf_a: np.ndarray, buf_b: np.ndarray, sample_rate: int) -> np.ndarray:
    """Append buf_b to buf_a with short crossfade."""
    xfade_samples = min(512, len(buf_a) // 4, len(buf_b) // 4)
    
    if xfade_samples < 16:
        return np.concatenate([buf_a, buf_b])
    
    # Create crossfade
    a_end = buf_a[-xfade_samples:].astype(np.float64)
    b_start = buf_b[:xfade_samples].astype(np.float64)
    
    fade_out = np.linspace(1, 0, xfade_samples)
    fade_in = np.linspace(0, 1, xfade_samples)
    
    crossfaded = a_end * fade_out + b_start * fade_in
    
    return np.concatenate([
        buf_a[:-xfade_samples],
        crossfaded,
        buf_b[xfade_samples:]
    ])


def _execute_generation_command(session: "Session", cmd_name: str, args: List[str]) -> str:
    """Execute a generation command and return result string."""
    # Map of generation commands
    if cmd_name in ('tone', 't'):
        # Parse tone args: freq, duration, amplitude
        try:
            freq = float(args[0]) if len(args) > 0 else 440.0
            dur = float(args[1]) if len(args) > 1 else 1.0
            amp = float(args[2]) if len(args) > 2 else 0.8
            session.generate_tone(freq, dur, amp)
            return f"OK: generated {freq}Hz tone"
        except (ValueError, IndexError) as e:
            return f"ERROR: invalid tone args: {e}"
    
    elif cmd_name in ('noise', 'n'):
        try:
            dur = float(args[0]) if len(args) > 0 else 1.0
            amp = float(args[1]) if len(args) > 1 else 0.5
            noise_type = args[2] if len(args) > 2 else 'white'
            # Generate noise
            samples = int(dur * session.sample_rate)
            if noise_type == 'pink':
                # Simple pink noise approximation
                white = np.random.randn(samples)
                b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
                a = [1, -2.494956002, 2.017265875, -0.522189400]
                from scipy.signal import lfilter
                session.last_buffer = lfilter(b, a, white) * amp
            else:
                session.last_buffer = np.random.randn(samples) * amp
            return f"OK: generated {noise_type} noise"
        except Exception as e:
            return f"ERROR: noise generation failed: {e}"
    
    elif cmd_name in ('silence', 'sil'):
        try:
            dur = float(args[0]) if len(args) > 0 else 1.0
            samples = int(dur * session.sample_rate)
            session.last_buffer = np.zeros(samples, dtype=np.float64)
            return f"OK: generated {dur}s silence"
        except Exception as e:
            return f"ERROR: silence generation failed: {e}"
    
    else:
        return f"ERROR: unknown generation command '{cmd_name}'"


# ============================================================================
# BUFFER OVERLAY AND SKETCH COMMANDS
# ============================================================================

def cmd_bov(session: "Session", args: List[str]) -> str:
    """Buffer overlay - mix audio ON TOP of existing content at position.
    
    Unlike /a which INSERTS audio (making buffer longer), /bov OVERLAYS
    audio by mixing it with existing content at the specified position.
    
    Usage:
      /bov              -> Overlay at current append position
      /bov 0            -> Overlay at beginning
      /bov <position>   -> Overlay at sample position
      /bov <seconds>s   -> Overlay at time position (e.g., /bov 1.5s)
      /bov end          -> Overlay at end (extends if needed)
    
    The pending audio (from /tone, /g, etc) is MIXED with existing content.
    If the new audio extends past buffer end, the buffer is extended.
    
    Examples:
      /tone 440 1.0 0.5    -> Generate 1s tone
      /a                   -> Append to buffer (buffer is now 1s)
      /tone 660 0.5 0.3    -> Generate 0.5s tone
      /bov 0.5s            -> Overlay at 0.5s - mixes with existing
    """
    _ensure_buffer_system(session)
    
    # Check for pending audio
    if session.last_buffer is None or len(session.last_buffer) == 0:
        return "ERROR: no pending audio. Generate with /tone, /g, etc. first."
    
    new_audio = session.last_buffer.copy()
    idx = session.current_buffer_index
    current_buf = session.buffers.get(idx, np.zeros(0, dtype=np.float64))
    
    # Parse position argument
    if not args:
        # Use last append position
        overlay_pos = session.buffer_append_positions.get(idx, 0)
    else:
        pos_str = args[0].lower()
        
        if pos_str == 'end':
            overlay_pos = len(current_buf)
        elif pos_str.endswith('s'):
            try:
                seconds = float(pos_str[:-1])
                overlay_pos = int(seconds * session.sample_rate)
            except ValueError:
                return f"ERROR: invalid time position '{pos_str}'"
        else:
            try:
                overlay_pos = int(float(args[0]))
            except ValueError:
                return f"ERROR: invalid position '{args[0]}'. Use number, time (1.5s), or 'end'"
    
    overlay_pos = max(0, overlay_pos)
    
    # Calculate required buffer length
    required_len = overlay_pos + len(new_audio)
    
    # Extend buffer if needed
    if len(current_buf) < required_len:
        extended = np.zeros(required_len, dtype=np.float64)
        if len(current_buf) > 0:
            extended[:len(current_buf)] = current_buf
        current_buf = extended
    
    # Mix (overlay) the new audio at position
    end_pos = overlay_pos + len(new_audio)
    current_buf[overlay_pos:end_pos] += new_audio
    
    # Normalize if needed
    peak = np.max(np.abs(current_buf))
    if peak > 1.0:
        current_buf = current_buf / peak * 0.95
    
    # Store result
    session.buffers[idx] = current_buf
    session.buffer_append_positions[idx] = overlay_pos
    session.last_buffer = current_buf
    
    new_dur = len(new_audio) / session.sample_rate
    total_dur = len(current_buf) / session.sample_rate
    pos_sec = overlay_pos / session.sample_rate
    
    return f"OK: overlaid {new_dur:.3f}s at {pos_sec:.3f}s -> buffer {idx} total {total_dur:.3f}s"


def cmd_bs(session: "Session", args: List[str]) -> str:
    """Buffer sketch - render sketch/arrangement as buffer.
    
    Usage:
      /bs               -> Render sketch to current buffer
      /bs <n>           -> Render sketch to buffer n
      /bs new           -> Render sketch to new buffer
      /bs file [name]   -> Render sketch directly to file
    
    The sketch contains clips arranged on the timeline.
    This command renders the entire sketch to a buffer for
    further processing or export.
    """
    _ensure_buffer_system(session)
    
    # Get the sketch/arrangement from session
    if not hasattr(session, 'clips') or not session.clips:
        return "ERROR: no clips in sketch. Use /bc to add clips."
    
    if not args:
        target_idx = session.current_buffer_index
    elif args[0].lower() == 'new':
        target_idx = max(session.buffers.keys()) + 1 if session.buffers else 1
        session.buffers[target_idx] = np.zeros(0, dtype=np.float64)
        session.buffer_append_positions[target_idx] = 0
    elif args[0].lower() == 'file':
        # Render directly to file
        filename = args[1] if len(args) > 1 else None
        return _render_sketch_to_file(session, filename)
    else:
        try:
            target_idx = int(args[0])
        except ValueError:
            return f"ERROR: invalid buffer index '{args[0]}'"
    
    # Calculate total sketch length
    max_end = 0
    for clip in session.clips:
        beat = clip.get('beat', 0)
        audio = clip.get('audio', np.zeros(0))
        beat_samples = int(beat * session.sample_rate * 60 / session.bpm)
        end_samples = beat_samples + len(audio)
        max_end = max(max_end, end_samples)
    
    if max_end == 0:
        return "ERROR: sketch has no audio content"
    
    # Render sketch
    sketch_buf = np.zeros(max_end, dtype=np.float64)
    for clip in session.clips:
        beat = clip.get('beat', 0)
        audio = clip.get('audio', np.zeros(0))
        beat_samples = int(beat * session.sample_rate * 60 / session.bpm)
        end_pos = min(beat_samples + len(audio), max_end)
        sketch_buf[beat_samples:end_pos] += audio[:end_pos - beat_samples]
    
    # Normalize
    peak = np.max(np.abs(sketch_buf))
    if peak > 1.0:
        sketch_buf = sketch_buf / peak * 0.95
    
    session.buffers[target_idx] = sketch_buf
    session.buffer_append_positions[target_idx] = 0
    session.last_buffer = sketch_buf
    
    dur = len(sketch_buf) / session.sample_rate
    return f"OK: rendered sketch ({len(session.clips)} clips) to buffer {target_idx} ({dur:.3f}s)"


def _render_sketch_to_file(session: "Session", filename: Optional[str] = None) -> str:
    """Render sketch directly to file."""
    if not session.clips:
        return "ERROR: no clips in sketch"
    
    # Calculate and render
    max_end = 0
    for clip in session.clips:
        beat = clip.get('beat', 0)
        audio = clip.get('audio', np.zeros(0))
        beat_samples = int(beat * session.sample_rate * 60 / session.bpm)
        max_end = max(max_end, beat_samples + len(audio))
    
    sketch_buf = np.zeros(max_end, dtype=np.float64)
    for clip in session.clips:
        beat = clip.get('beat', 0)
        audio = clip.get('audio', np.zeros(0))
        beat_samples = int(beat * session.sample_rate * 60 / session.bpm)
        end_pos = min(beat_samples + len(audio), max_end)
        sketch_buf[beat_samples:end_pos] += audio[:end_pos - beat_samples]
    
    peak = np.max(np.abs(sketch_buf))
    if peak > 1.0:
        sketch_buf = sketch_buf / peak * 0.95
    
    # Generate filename
    if not filename:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"sketch_{timestamp}.wav"
    elif not filename.endswith('.wav'):
        filename += '.wav'
    
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    
    try:
        import soundfile as sf
        sf.write(out_path, sketch_buf, session.sample_rate)
        dur = len(sketch_buf) / session.sample_rate
        return f"OK: rendered sketch to {filename} ({dur:.3f}s)"
    except ImportError:
        try:
            from scipy.io import wavfile
            wavfile.write(out_path, session.sample_rate, (sketch_buf * 32767).astype(np.int16))
            return f"OK: rendered sketch to {filename}"
        except Exception as e:
            return f"ERROR: could not write file: {e}"


def cmd_bpa(session: "Session", args: List[str]) -> str:
    """Buffer pack - export buffers as a sample pack.
    
    Usage:
      /bpa <pack_name>              -> Export all buffers as pack
      /bpa <pack_name> <1,2,3>      -> Export specific buffers
      /bpa <pack_name> normalize    -> Normalize all samples
    
    Creates a pack folder in ~/Documents/MDMA/packs/ with:
      - Individual WAV files for each buffer
      - pack.json manifest file
    
    Examples:
      /bpa my_drums                 -> Export all buffers as "my_drums" pack
      /bpa bass_hits 1,2,3          -> Export buffers 1,2,3
      /bpa leads normalize          -> Export normalized
    """
    _ensure_buffer_system(session)
    
    if not args:
        return ("Usage: /bpa <pack_name> [buffer_list] [normalize]\n"
                "  /bpa my_pack         -> Export all buffers\n"
                "  /bpa my_pack 1,2,3   -> Export buffers 1,2,3\n"
                "  /bpa my_pack normalize -> Normalize samples")
    
    pack_name = args[0]
    buffer_list = None
    normalize = False
    
    # Parse additional args
    for arg in args[1:]:
        if arg.lower() == 'normalize':
            normalize = True
        elif ',' in arg or arg.isdigit():
            # Buffer list
            try:
                buffer_list = [int(x.strip()) for x in arg.split(',')]
            except ValueError:
                return f"ERROR: invalid buffer list '{arg}'"
    
    # Get target buffers
    if buffer_list:
        target_buffers = [(i, session.buffers.get(i)) for i in buffer_list if i in session.buffers]
    else:
        target_buffers = [(i, buf) for i, buf in session.buffers.items() if buf is not None and len(buf) > 0]
    
    if not target_buffers:
        return "ERROR: no non-empty buffers to export"
    
    # Create pack directory
    try:
        from ..core.user_data import get_packs_dir, create_pack_manifest
        pack_dir = get_packs_dir() / pack_name
        pack_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return f"ERROR: could not create pack directory: {e}"
    
    # Export samples
    exported = []
    for idx, buf in target_buffers:
        if buf is None or len(buf) == 0:
            continue
        
        # Normalize if requested
        if normalize:
            peak = np.max(np.abs(buf))
            if peak > 0:
                buf = buf / peak * 0.95
        
        filename = f"sample_{idx:02d}.wav"
        out_path = pack_dir / filename
        
        try:
            import soundfile as sf
            sf.write(str(out_path), buf, session.sample_rate)
            exported.append(filename)
        except ImportError:
            try:
                from scipy.io import wavfile
                wavfile.write(str(out_path), session.sample_rate, (buf * 32767).astype(np.int16))
                exported.append(filename)
            except Exception as e:
                exported.append(f"{filename} (ERROR: {e})")
    
    # Create manifest
    try:
        create_pack_manifest(pack_name, author='MDMA User', 
                           description=f'Exported from MDMA buffers')
    except Exception:
        pass  # Non-fatal
    
    return f"OK: exported {len(exported)} samples to pack '{pack_name}':\n  " + '\n  '.join(exported)


# ============================================================================
# ALIASES
# ============================================================================

cmd_play = cmd_p
cmd_buffer = cmd_buf
cmd_overlay = cmd_bov


# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

def get_buffer_commands() -> dict:
    """Return buffer commands for registration."""
    return {
        # Buffer info and selection
        'buf': cmd_buf,
        'buffer': cmd_buf,
        'bu': cmd_bu,
        'bu+': cmd_bu_plus,
        'clr': cmd_clr,
        'clear': cmd_clr,
        
        # Append and extend
        'a': cmd_a,
        'append': cmd_a,
        'ex': cmd_ex,
        'extend': cmd_ex,
        
        # Overlay (mix on top)
        'bov': cmd_bov,
        'overlay': cmd_overlay,
        
        # Playback
        'p': cmd_p,
        'play': cmd_play,
        'pa': cmd_pa,
        
        # Render
        'b': cmd_b,
        'ba': cmd_ba,
        'bo': cmd_bo,
        
        # Sketch
        'bs': cmd_bs,
        
        # Pack export
        'bpa': cmd_bpa,
        'bpack': cmd_bpa,
        
        # Clip insertion
        'bc': cmd_bc,
        
        # Pattern (append-aware)
        'apat': cmd_apat,
    }
