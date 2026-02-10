"""MDMA Advanced Commands.

Chunking, remix, rhythmic patterns, user variables, and system bridges.

Commands:
    /CHK <algo>        Auto-chunk current buffer/deck
    /remix <algo>      Remix with algorithm
    /RPAT <pattern>    Apply rhythmic pattern
    /CBI <idx> ...     Combine buffer indexes
    /= <name> <val>    Set user variable
    /GET <name>        Get user variable
    /PR <deck>         Print buffer to deck
    /YT <url>          Pull YouTube to buffer

BUILD ID: adv_cmds_v1.0
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.session import Session


# ============================================================================
# USER VARIABLE COMMANDS
# ============================================================================

def cmd_set(session: "Session", args: List[str]) -> str:
    """Set user variable.
    
    Usage:
      /= name value          Set string value
      /= name 123            Set numeric value
      /= name [1,2,3]        Set list value
      /= name {a:1,b:2}      Set dict value
    
    Examples:
      /= mybpm 128
      /= kick_vol 0.8
      /= pattern x.x.x.x.
    """
    from ..dsp.advanced_ops import get_user_stack
    
    if len(args) < 2:
        return "Usage: /= name value"
    
    name = args[0]
    value_str = ' '.join(args[1:])
    
    # Try to parse as different types
    value = _parse_value(value_str)
    
    stack = get_user_stack()
    stack.set(name, value)
    
    return f"OK: {name} = {repr(value)}"


def cmd_get(session: "Session", args: List[str]) -> str:
    """Get user variable.
    
    Usage:
      /GET name              Get variable value
      /GET name.key          Get dict key or list index
      /GET                   List all variables
    
    Examples:
      /GET mybpm
      /GET mydict.key
      /GET mylist.0
    """
    from ..dsp.advanced_ops import get_user_stack
    
    stack = get_user_stack()
    
    if not args:
        # List all variables
        vars_dict = stack.list_vars()
        if not vars_dict:
            return "No variables set. Use /= name value to set."
        
        lines = ["=== USER VARIABLES ==="]
        for name, value in sorted(vars_dict.items()):
            val_str = repr(value)
            if len(val_str) > 50:
                val_str = val_str[:47] + "..."
            lines.append(f"  {name} = {val_str}")
        return "\n".join(lines)
    
    name = args[0]
    
    if not stack.exists(name.split('.')[0]):
        return f"ERROR: Variable '{name.split('.')[0]}' not defined. Use /= to set."
    
    value = stack.get(name)
    return f"{name} = {repr(value)}"


def cmd_del_var(session: "Session", args: List[str]) -> str:
    """Delete user variable.
    
    Usage:
      /DEL name              Delete variable
      /DEL *                 Clear all variables
    """
    from ..dsp.advanced_ops import get_user_stack
    
    stack = get_user_stack()
    
    if not args:
        return "Usage: /DEL name or /DEL *"
    
    name = args[0]
    
    if name == '*':
        stack.clear()
        return "OK: Cleared all variables"
    
    if stack.delete(name):
        return f"OK: Deleted {name}"
    else:
        return f"Variable '{name}' not found"


def _parse_value(value_str: str):
    """Parse value string to appropriate type."""
    value_str = value_str.strip()
    
    # Try numeric
    try:
        if '.' in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass
    
    # Try boolean
    if value_str.lower() in ('true', 'yes', 'on'):
        return True
    if value_str.lower() in ('false', 'no', 'off'):
        return False
    
    # Try list [1,2,3]
    if value_str.startswith('[') and value_str.endswith(']'):
        try:
            import ast
            return ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            pass
    
    # Try dict {a:1}
    if value_str.startswith('{') and value_str.endswith('}'):
        try:
            import ast
            return ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            pass
    
    # Return as string
    return value_str


# ============================================================================
# CHUNKING COMMANDS
# ============================================================================

def cmd_chk(session: "Session", args: List[str]) -> str:
    """Auto-chunk current buffer or deck.
    
    Usage:
      /CHK [algo] [num]      Chunk with algorithm
      /CHK use <idx>         Load chunk by index
      /CHK use all           Concatenate all chunks
      
    Algorithms:
      auto       Automatic selection (default)
      transient  Split at transients
      beat       Split at detected beats
      zero       Split at zero crossings
      equal      Equal-sized chunks
      wavetable  Single-cycle extraction
      energy     Energy-based segmentation
      spectral   Spectral similarity
    
    Examples:
      /CHK                   Auto-chunk current buffer
      /CHK beat              Chunk at beats
      /CHK equal 16          Split into 16 equal chunks
      /CHK use 0             Load first chunk
    """
    from ..dsp.advanced_ops import auto_chunk, AudioChunk
    
    # Handle 'use' subcommand
    if args and args[0].lower() == 'use':
        return cmd_chk_use(session, args[1:])
    
    # Get current audio from working buffer or buffer
    audio = _get_current_audio(session)
    if audio is None or len(audio) == 0:
        return "ERROR: No audio in buffer. Generate or load audio first."
    
    # Parse args
    algorithm = "auto"
    num_chunks = 0
    
    if args:
        arg0 = args[0].lower()
        # Check if first arg is a number (for equal chunks shorthand)
        try:
            num_chunks = int(args[0])
            algorithm = "equal"
        except ValueError:
            algorithm = arg0
    
    if len(args) > 1:
        try:
            num_chunks = int(args[1])
        except ValueError:
            pass
    
    # Perform chunking
    chunks = auto_chunk(
        audio, 
        session.sample_rate, 
        algorithm=algorithm,
        num_chunks=num_chunks
    )
    
    if not chunks:
        return "ERROR: Could not chunk audio"
    
    # Store chunks in session
    if not hasattr(session, 'chunks'):
        session.chunks = {}
    
    session.chunks['last'] = chunks
    
    # Also store individual chunks in user stack for access
    from ..dsp.advanced_ops import get_user_stack
    stack = get_user_stack()
    stack.set('chunks', [c.audio for c in chunks])
    stack.set('chunk_count', len(chunks))
    
    # Summary
    lines = [f"=== CHUNKED: {len(chunks)} chunks ==="]
    for i, chunk in enumerate(chunks[:10]):  # Show first 10
        dur_ms = len(chunk.audio) / session.sample_rate * 1000
        lines.append(f"  {i:2d}: {dur_ms:6.1f}ms  E={chunk.energy:.3f}  P={chunk.peak:.3f}")
    
    if len(chunks) > 10:
        lines.append(f"  ... and {len(chunks) - 10} more")
    
    lines.append(f"\nUse /GET chunks.N to access chunk N")
    lines.append(f"Use /CHK use N to load chunk to buffer")
    
    return "\n".join(lines)


def cmd_chk_use(session: "Session", args: List[str]) -> str:
    """Load a chunk to buffer.
    
    Usage:
      /CHK use <idx>         Load chunk by index
      /CHK use all           Concatenate all chunks
      /CHK use <i> <j> ...   Load specific chunks
    """
    if not hasattr(session, 'chunks') or 'last' not in session.chunks:
        return "ERROR: No chunks available. Run /CHK first."
    
    chunks = session.chunks['last']
    
    if not args:
        return f"Usage: /CHK use <idx>\n{len(chunks)} chunks available (0-{len(chunks)-1})"
    
    if args[0].lower() == 'all':
        # Concatenate all
        audio = np.concatenate([c.audio for c in chunks])
        _set_current_audio(session, audio)
        return f"OK: Loaded all {len(chunks)} chunks ({len(audio)/session.sample_rate:.2f}s)"
    
    # Load specific indices
    indices = []
    for a in args:
        try:
            idx = int(a)
            if 0 <= idx < len(chunks):
                indices.append(idx)
        except ValueError:
            pass
    
    if not indices:
        return "ERROR: No valid chunk indices provided"
    
    if len(indices) == 1:
        audio = chunks[indices[0]].audio
    else:
        audio = np.concatenate([chunks[i].audio for i in indices])
    
    _set_current_audio(session, audio)
    return f"OK: Loaded chunk(s) {indices} ({len(audio)/session.sample_rate:.2f}s)"


# ============================================================================
# REMIX COMMANDS
# ============================================================================

def cmd_remix(session: "Session", args: List[str]) -> str:
    """Remix current audio with algorithms.
    
    Usage:
      /remix [algo] [intensity]
      
    Algorithms:
      shuffle    Random chunk shuffle (default)
      reverse    Reverse random chunks
      stutter    Add stutter effects
      glitch     Glitch/noise effects
      chop       Beat chop/rearrange
      layer      Layer chunks
      evolve     AI-style evolution
      granular   Granular reconstruction
    
    Intensity: 0.0 - 1.0 (default 0.5)
    
    Examples:
      /remix                 Shuffle at 50% intensity
      /remix glitch 0.8      Heavy glitch
      /remix evolve 0.3      Subtle evolution
    """
    from ..dsp.advanced_ops import remix_audio
    
    # Get current audio
    audio = _get_current_audio(session)
    if audio is None or len(audio) == 0:
        return "ERROR: No audio in buffer"
    
    # Parse args
    algorithm = "shuffle"
    intensity = 0.5
    seed = None
    
    if args:
        algorithm = args[0].lower()
    if len(args) > 1:
        try:
            intensity = float(args[1])
            intensity = max(0.0, min(1.0, intensity))
        except ValueError:
            pass
    if len(args) > 2:
        try:
            seed = int(args[2])
        except ValueError:
            pass
    
    # Remix
    result = remix_audio(audio, session.sample_rate, algorithm, intensity, seed)
    
    # Store result
    _set_current_audio(session, result)
    
    return f"OK: Remixed with '{algorithm}' @ {intensity:.0%} intensity ({len(result)/session.sample_rate:.2f}s)"


# ============================================================================
# RHYTHMIC PATTERN COMMANDS
# ============================================================================

def cmd_rpat(session: "Session", args: List[str]) -> str:
    """Apply rhythmic pattern to current audio.
    
    Usage:
      /RPAT <pattern> [beats]
      
    Pattern formats:
      Binary:     x.x.x.x.    (x=hit, .=rest)
      Numeric:    1010101     (1=hit, 0=rest)
      Velocity:   1 0.5 0 0.8 (space-separated 0-1 values)
      Letters:    h.m.l.h.    (h=high, m=medium, l=low)
    
    The pattern triggers the current audio at each position.
    Velocity values control amplitude (0 = silent, 1 = full).
    
    Examples:
      /RPAT x.x.x.x.          Basic 8th notes
      /RPAT x..x..x.          Syncopated
      /RPAT 1 0.5 0 0.8       Velocity pattern
      /RPAT x.x.x.x. 4        4 beats of pattern
    """
    from ..dsp.advanced_ops import apply_rhythmic_pattern, parse_pattern_string
    
    if not args:
        return ("Usage: /RPAT <pattern> [beats]\n"
                "Patterns: x.x.x.x. (binary) or 1 0.5 0 0.8 (velocity)")
    
    # Get current audio
    audio = _get_current_audio(session)
    if audio is None or len(audio) == 0:
        return "ERROR: No audio in buffer. Load a sample first."
    
    # Parse pattern
    pattern_str = args[0]
    if ' ' in ' '.join(args[:-1]) and args[-1].replace('.', '').isdigit():
        # Last arg might be duration
        pattern_str = ' '.join(args[:-1])
        duration_beats = float(args[-1])
    else:
        pattern_str = ' '.join(args) if ' ' in args[0] or len(args) == 1 else args[0]
        duration_beats = None
    
    # Check for explicit duration
    if len(args) > 1:
        try:
            duration_beats = float(args[-1])
            if pattern_str.endswith(args[-1]):
                pattern_str = ' '.join(args[:-1])
        except ValueError:
            pass
    
    pattern = parse_pattern_string(pattern_str)
    
    if not pattern:
        return "ERROR: Invalid pattern. Use x.x. format or velocity values."
    
    # Apply pattern
    result = apply_rhythmic_pattern(
        audio,
        pattern,
        session.sample_rate,
        session.bpm,
        duration_beats
    )
    
    # Store result
    _set_current_audio(session, result)
    
    pattern_vis = ''.join(['x' if v > 0 else '.' for v in pattern[:16]])
    if len(pattern) > 16:
        pattern_vis += '...'
    
    return f"OK: Applied pattern [{pattern_vis}] ({len(result)/session.sample_rate:.2f}s)"


# ============================================================================
# BUFFER COMBINING COMMANDS
# ============================================================================

def cmd_cbi(session: "Session", args: List[str]) -> str:
    """Combine buffer indexes (overlay).
    
    Usage:
      /CBI <idx> <idx> ...   Combine buffers by overlay
      /CBI <idx> + <idx>     Append buffers
      
    Examples:
      /CBI 1 2 3             Overlay buffers 1, 2, 3
      /CBI 1 + 2             Append buffer 2 to buffer 1
    """
    from ..dsp.advanced_ops import combine_buffers
    
    if not args:
        return "Usage: /CBI <idx> <idx> ... (overlay) or /CBI <idx> + <idx> (append)"
    
    # Check for append mode
    mode = "overlay"
    if '+' in args:
        mode = "append"
        args = [a for a in args if a != '+']
    
    # Parse buffer indices
    indices = []
    for a in args:
        try:
            idx = int(a)
            indices.append(idx)
        except ValueError:
            pass
    
    if len(indices) < 2:
        return "ERROR: Need at least 2 buffer indices"
    
    # Get buffers
    _ensure_buffer_system(session)
    
    buffers = []
    for idx in indices:
        if idx in session.buffers and session.buffers[idx] is not None:
            buf = session.buffers[idx]
            if len(buf) > 0:
                buffers.append(buf)
    
    if len(buffers) < 2:
        return f"ERROR: Not enough valid buffers. Found {len(buffers)}, need 2+"
    
    # Combine
    result = combine_buffers(buffers, mode, session.sample_rate)
    
    # Store in current buffer
    _set_current_audio(session, result)
    
    return f"OK: Combined {len(buffers)} buffers ({mode}) -> {len(result)/session.sample_rate:.2f}s"


def cmd_bap(session: "Session", args: List[str]) -> str:
    """Append buffer to another buffer.
    
    Usage:
      /BAP <src> <dst>       Append src buffer to dst buffer
      /BAP <src>             Append src to current buffer
    """
    _ensure_buffer_system(session)
    
    if not args:
        return "Usage: /BAP <src> [dst]"
    
    try:
        src_idx = int(args[0])
    except ValueError:
        return "ERROR: Invalid source buffer index"
    
    dst_idx = session.current_buffer_index
    if len(args) > 1:
        try:
            dst_idx = int(args[1])
        except ValueError:
            pass
    
    # Get buffers
    if src_idx not in session.buffers or session.buffers[src_idx] is None:
        return f"ERROR: Source buffer {src_idx} is empty"
    
    src_buf = session.buffers[src_idx]
    
    if dst_idx not in session.buffers:
        session.buffers[dst_idx] = np.array([])
    
    dst_buf = session.buffers[dst_idx]
    if dst_buf is None or len(dst_buf) == 0:
        session.buffers[dst_idx] = src_buf.copy()
    else:
        session.buffers[dst_idx] = np.concatenate([dst_buf, src_buf])
    
    return f"OK: Appended buffer {src_idx} to {dst_idx} ({len(session.buffers[dst_idx])/session.sample_rate:.2f}s)"


# ============================================================================
# BRIDGE COMMANDS
# ============================================================================

def cmd_pr(session: "Session", args: List[str]) -> str:
    """Print (copy) buffer to DJ deck.
    
    Usage:
      /PR [deck]             Copy current buffer to deck
      /PR <buf> <deck>       Copy buffer to deck
    
    Examples:
      /PR                    Copy current buffer to active deck
      /PR 1                  Copy current buffer to deck 1
      /PR 2 3                Copy buffer 2 to deck 3
    """
    try:
        from ..dsp.dj_mode import get_dj_engine, DJDeck
    except ImportError:
        return "ERROR: DJ mode not available"
    
    _ensure_buffer_system(session)
    
    # Parse args
    buf_idx = session.current_buffer_index
    deck_id = None
    
    if len(args) == 1:
        try:
            deck_id = int(args[0])
        except ValueError:
            return "Usage: /PR [buf] [deck]"
    elif len(args) >= 2:
        try:
            buf_idx = int(args[0])
            deck_id = int(args[1])
        except ValueError:
            return "Usage: /PR [buf] [deck]"
    
    # Get buffer
    if buf_idx not in session.buffers or session.buffers[buf_idx] is None:
        return f"ERROR: Buffer {buf_idx} is empty"
    
    buf = session.buffers[buf_idx]
    if len(buf) == 0:
        return f"ERROR: Buffer {buf_idx} is empty"
    
    # Get DJ engine
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        dj.enabled = True
    
    if deck_id is None:
        deck_id = dj.active_deck
    
    # Create deck if needed
    if deck_id not in dj.decks:
        dj.decks[deck_id] = DJDeck(id=deck_id)
    
    deck = dj.decks[deck_id]
    deck.buffer = buf.astype(np.float64) if buf.dtype != np.float64 else buf
    deck.position = 0.0
    deck.playing = False
    deck.analyzed = False
    
    return f"OK: Copied buffer {buf_idx} to deck {deck_id} ({len(buf)/session.sample_rate:.2f}s)"


def cmd_yt(session: "Session", args: List[str]) -> str:
    """Pull YouTube audio to buffer.
    
    Usage:
      /YT <url>              Download and load to buffer
      /YT <url> <buf>        Download to specific buffer
    
    Examples:
      /YT https://youtube.com/watch?v=...
      /YT https://youtu.be/... 2
    """
    if not args:
        return "Usage: /YT <url> [buffer]"
    
    url = args[0]
    
    # Validate URL
    if 'youtube.com' not in url and 'youtu.be' not in url:
        return "ERROR: Invalid YouTube URL"
    
    buf_idx = session.current_buffer_index
    if len(args) > 1:
        try:
            buf_idx = int(args[1])
        except ValueError:
            pass
    
    # Stream to buffer
    try:
        from ..dsp.streaming import stream_youtube
        
        print(f"Downloading from YouTube...")
        buffer = stream_youtube(url, session.sample_rate)
        
        if buffer.error:
            return f"ERROR: {buffer.error}"
        
        if not buffer.ready or buffer.audio is None:
            return "ERROR: Download failed"
        
        # Store in buffer
        _ensure_buffer_system(session)
        session.buffers[buf_idx] = buffer.audio
        
        title = buffer.track_info.title if buffer.track_info else "Unknown"
        dur = len(buffer.audio) / session.sample_rate
        
        # Show registry info if registered
        reg_info = ""
        if buffer.registry_id:
            reg_info = f"\nRegistered as song #{buffer.registry_id}"
        
        return f"OK: Loaded '{title}' to buffer {buf_idx} ({dur:.1f}s){reg_info}"
        
    except ImportError:
        return "ERROR: Streaming requires yt-dlp: pip install yt-dlp"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_sc(session: "Session", args: List[str]) -> str:
    """Pull SoundCloud audio to buffer.
    
    Usage:
      /SC <url>              Download and load to buffer
      /SC <url> <buf>        Download to specific buffer
    """
    if not args:
        return "Usage: /SC <url> [buffer]"
    
    url = args[0]
    
    if 'soundcloud.com' not in url:
        return "ERROR: Invalid SoundCloud URL"
    
    buf_idx = session.current_buffer_index
    if len(args) > 1:
        try:
            buf_idx = int(args[1])
        except ValueError:
            pass
    
    try:
        from ..dsp.streaming import stream_soundcloud
        
        print(f"Downloading from SoundCloud...")
        buffer = stream_soundcloud(url, session.sample_rate)
        
        if buffer.error:
            return f"ERROR: {buffer.error}"
        
        if not buffer.ready or buffer.audio is None:
            return "ERROR: Download failed"
        
        _ensure_buffer_system(session)
        session.buffers[buf_idx] = buffer.audio
        
        title = buffer.track_info.title if buffer.track_info else "Unknown"
        dur = len(buffer.audio) / session.sample_rate
        
        reg_info = ""
        if buffer.registry_id:
            reg_info = f"\nRegistered as song #{buffer.registry_id}"
        
        return f"OK: Loaded '{title}' to buffer {buf_idx} ({dur:.1f}s){reg_info}"
        
    except ImportError:
        return "ERROR: Streaming requires yt-dlp: pip install yt-dlp"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_dk2buf(session: "Session", args: List[str]) -> str:
    """Copy deck audio to buffer.
    
    Usage:
      /DK2BUF [deck] [buf]   Copy deck to buffer
    """
    try:
        from ..dsp.dj_mode import get_dj_engine
    except ImportError:
        return "ERROR: DJ mode not available"
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "ERROR: DJ mode not enabled"
    
    deck_id = dj.active_deck
    buf_idx = session.current_buffer_index
    
    if args:
        try:
            deck_id = int(args[0])
        except ValueError:
            pass
    if len(args) > 1:
        try:
            buf_idx = int(args[1])
        except ValueError:
            pass
    
    if deck_id not in dj.decks:
        return f"ERROR: Deck {deck_id} not found"
    
    deck = dj.decks[deck_id]
    if deck.buffer is None or len(deck.buffer) == 0:
        return f"ERROR: Deck {deck_id} is empty"
    
    _ensure_buffer_system(session)
    session.buffers[buf_idx] = deck.buffer.copy()
    
    return f"OK: Copied deck {deck_id} to buffer {buf_idx} ({len(deck.buffer)/session.sample_rate:.2f}s)"


# ============================================================================
# WAVETABLE COMMANDS
# ============================================================================

def cmd_wt(session: "Session", args: List[str]) -> str:
    """Generate wavetable from current audio.
    
    Usage:
      /WT [frames] [size]    Generate wavetable
      
    Parameters:
      frames   Number of frames (default 256)
      size     Samples per frame (default 2048)
    
    Examples:
      /WT                    256 frames x 2048 samples
      /WT 64                 64 frames
      /WT 128 1024           128 frames x 1024 samples
    """
    from ..dsp.advanced_ops import generate_wavetable
    
    audio = _get_current_audio(session)
    if audio is None or len(audio) == 0:
        return "ERROR: No audio in buffer"
    
    num_frames = 256
    frame_size = 2048
    
    if args:
        try:
            num_frames = int(args[0])
        except ValueError:
            pass
    if len(args) > 1:
        try:
            frame_size = int(args[1])
        except ValueError:
            pass
    
    wavetable = generate_wavetable(audio, session.sample_rate, num_frames, frame_size)
    
    # Store in user stack
    from ..dsp.advanced_ops import get_user_stack
    stack = get_user_stack()
    stack.set('wavetable', wavetable)
    stack.set('wt_frames', num_frames)
    stack.set('wt_size', frame_size)
    
    return f"OK: Generated {num_frames}x{frame_size} wavetable\nAccess via /GET wavetable"


# ============================================================================
# HELPERS
# ============================================================================

def _ensure_buffer_system(session: "Session"):
    """Ensure buffer system is initialized."""
    if not hasattr(session, 'buffers'):
        session.buffers = {}
    if not hasattr(session, 'current_buffer_index'):
        session.current_buffer_index = 1
    if session.current_buffer_index not in session.buffers:
        session.buffers[session.current_buffer_index] = np.array([])


def _get_current_audio(session: "Session") -> Optional[np.ndarray]:
    """Get current audio from buffer or deck."""
    _ensure_buffer_system(session)
    
    buf_idx = session.current_buffer_index
    if buf_idx in session.buffers and session.buffers[buf_idx] is not None:
        buf = session.buffers[buf_idx]
        if len(buf) > 0:
            return buf
    
    # Try last_buffer
    if hasattr(session, 'last_buffer') and session.last_buffer is not None:
        return session.last_buffer
    
    return None


def _set_current_audio(session: "Session", audio: np.ndarray):
    """Set current buffer audio."""
    _ensure_buffer_system(session)
    session.buffers[session.current_buffer_index] = audio
    session.last_buffer = audio


# ============================================================================
# MACRO SYSTEM WITH ARGUMENTS
# ============================================================================

# Global macro storage
_MACROS = {}


def cmd_mc(session: "Session", args: List[str]) -> str:
    """Macro system with argument support.
    
    Usage:
      /MC new <name> [arg1 arg2 ...]   Create macro with named arguments
      /MC run <name> [val1 val2 ...]   Run macro with argument values
      /MC list                          List all macros
      /MC show <name>                   Show macro commands
      /MC del <name>                    Delete macro
    
    Macros can use $1, $2, etc. for positional args or $argname for named args.
    Macros can contain nested function calls and other macros.
    
    Examples:
      /MC new kick bpm vol
        [def:kick]> /bpm $bpm
        [def:kick]> /g sk
        [def:kick]> /amp $vol
        [def:kick]> /end
      
      /MC run kick 140 0.8
    """
    if not args:
        return ("Usage:\n"
                "  /MC new <name> [args]  Create macro\n"
                "  /MC run <name> [vals]  Run macro\n"
                "  /MC list               List macros\n"
                "  /MC show <name>        Show macro\n"
                "  /MC del <name>         Delete macro")
    
    subcmd = args[0].lower()
    
    if subcmd == 'new':
        if len(args) < 2:
            return "Usage: /MC new <name> [arg1 arg2 ...]"
        
        name = args[1].lower()
        macro_args = args[2:] if len(args) > 2 else []
        
        # Start macro definition mode
        session.defining_macro = name
        session.macro_args = macro_args
        session.macro_commands = []
        
        if macro_args:
            return f"Defining macro '{name}' with args: {', '.join(macro_args)}\nEnter commands, use /end to finish"
        else:
            return f"Defining macro '{name}' (no args)\nEnter commands, use /end to finish"

    elif subcmd == 'add':
        # Add a command to an existing macro without entering definition mode
        # Usage: /MC add <name> <command...>
        if len(args) < 3:
            return "Usage: /MC add <name> <command...>"
        name = args[1].lower()
        if name not in _MACROS:
            return f"ERROR: macro '{name}' not found. Use /MC list to see macros."
        # Join the remaining arguments into a command line
        cmd_line = ' '.join(args[2:])
        _MACROS[name]['commands'].append(cmd_line)
        return f"OK: added to macro '{name}': {cmd_line}"

    elif subcmd == 'stp':
        # Schedule an existing macro step with a timing hint
        # Usage: /MC stp <name> <stepIndex> <when>
        if len(args) < 4:
            return "Usage: /MC stp <name> <stepIndex> <when>"
        name = args[1].lower()
        if name not in _MACROS:
            return f"ERROR: macro '{name}' not found. Use /MC list to see macros."
        try:
            idx = int(float(args[2]))
        except Exception:
            return "ERROR: invalid step index"
        when = args[3]
        commands = _MACROS[name]['commands']
        if idx < 1 or idx > len(commands):
            return f"ERROR: step {idx} out of range for macro '{name}'"
        # Prepend schedule token (ensure it starts with '@')
        schedule_token = when if when.startswith('@') else '@' + when
        cmd_line = commands[idx - 1]
        # Avoid duplicating multiple schedule tokens
        # If command already has a schedule at start, replace it
        stripped = cmd_line.strip()
        if stripped.startswith('@'):
            parts = stripped.split(maxsplit=1)
            # Keep the rest after the first token
            rest = parts[1] if len(parts) > 1 else ''
        else:
            rest = cmd_line
        commands[idx - 1] = f"{schedule_token} {rest}".strip()
        return f"OK: scheduled step {idx} of '{name}' at {schedule_token}"
    
    elif subcmd == 'end':
        # End macro definition
        if not hasattr(session, 'defining_macro') or not session.defining_macro:
            return "ERROR: Not defining a macro"
        
        name = session.defining_macro
        _MACROS[name] = {
            'args': session.macro_args,
            'commands': session.macro_commands.copy()
        }
        
        session.defining_macro = None
        session.macro_args = []
        session.macro_commands = []
        
        return f"OK: Macro '{name}' saved with {len(_MACROS[name]['commands'])} commands"
    
    elif subcmd == 'run':
        if len(args) < 2:
            return "Usage: /MC run <name> [values]"
        
        name = args[1].lower()
        if name not in _MACROS:
            return f"ERROR: Macro '{name}' not found. Use /MC list to see macros."
        
        macro = _MACROS[name]
        values = args[2:] if len(args) > 2 else []
        
        # Build argument map
        arg_map = {}
        for i, arg_name in enumerate(macro['args']):
            if i < len(values):
                arg_map[arg_name] = values[i]
                arg_map[str(i + 1)] = values[i]  # Also support $1, $2 etc
            else:
                arg_map[arg_name] = ''
                arg_map[str(i + 1)] = ''
        
        # Also add positional-only args
        for i, val in enumerate(values):
            arg_map[str(i + 1)] = val
        
        # Execute commands with substitution
        from ..dsp.advanced_ops import get_user_stack
        from ..dsp.performance import get_macro_scheduler
        
        stack = get_user_stack()
        scheduler = get_macro_scheduler(session.bpm if hasattr(session, 'bpm') else 120.0)
        
        # Set command handler if session has execute capability
        if hasattr(session, '_execute_line'):
            scheduler.set_command_handler(session._execute_line)
        
        results = []
        for cmd_line in macro['commands']:
            # Check for schedule token at start (e.g., @bar, @beat, @now)
            line = cmd_line.strip()
            schedule_token = "@now"  # Default to immediate
            if line.startswith('@'):
                parts = line.split(maxsplit=1)
                schedule_token = parts[0]
                line = parts[1] if len(parts) > 1 else ''
            
            # Substitute arguments in the command line
            expanded = line
            for arg_name, arg_val in arg_map.items():
                expanded = expanded.replace(f'${arg_name}', str(arg_val))
            
            # Also substitute user variables $var
            for var_name in stack.list_vars():
                expanded = expanded.replace(f'${var_name}', str(stack.get(var_name)))
            
            # Schedule the command with proper timing
            sched_result = scheduler.schedule(expanded, schedule_token)
            results.append(f"  {schedule_token} {expanded}")
            
            # For @now commands, show the result
            if schedule_token == "@now" and "Executed:" in sched_result:
                results.append(f"    -> {sched_result.split('->')[-1].strip()}")
        
        # Show pending count if any scheduled for later
        pending = len(scheduler.pending)
        if pending > 0:
            results.append(f"\n  ({pending} commands scheduled for later)")
        
        return f"Ran macro '{name}':\n" + "\n".join(results)
    
    elif subcmd == 'list':
        if not _MACROS:
            return "No macros defined. Use /MC new <name> to create."
        
        lines = ["=== MACROS ==="]
        for name, macro in sorted(_MACROS.items()):
            args_str = f"({', '.join(macro['args'])})" if macro['args'] else "()"
            lines.append(f"  {name}{args_str}: {len(macro['commands'])} commands")
        return "\n".join(lines)
    
    elif subcmd == 'show':
        if len(args) < 2:
            return "Usage: /MC show <name>"
        
        name = args[1].lower()
        if name not in _MACROS:
            return f"ERROR: Macro '{name}' not found"
        
        macro = _MACROS[name]
        args_str = ', '.join(macro['args']) if macro['args'] else '(none)'
        
        lines = [f"=== MACRO: {name} ==="]
        lines.append(f"Args: {args_str}")
        lines.append("Commands:")
        for i, cmd in enumerate(macro['commands']):
            lines.append(f"  {i+1}: {cmd}")
        return "\n".join(lines)
    
    elif subcmd == 'del':
        if len(args) < 2:
            return "Usage: /MC del <name>"
        
        name = args[1].lower()
        if name not in _MACROS:
            return f"ERROR: Macro '{name}' not found"
        
        del _MACROS[name]
        return f"OK: Deleted macro '{name}'"
    
    elif subcmd == 'queue' or subcmd == 'q':
        # Show scheduled events
        from ..dsp.performance import get_macro_scheduler
        scheduler = get_macro_scheduler()
        return scheduler.list_pending()
    
    elif subcmd == 'clear':
        # Clear scheduled events (not macros, just queue)
        from ..dsp.performance import get_macro_scheduler
        scheduler = get_macro_scheduler()
        scheduler.clear()
        return "OK: Cleared all scheduled events"
    
    elif subcmd == 'tick':
        # Manually advance scheduler (for testing)
        from ..dsp.performance import get_macro_scheduler
        scheduler = get_macro_scheduler()
        
        if len(args) > 1:
            try:
                beat = float(args[1])
                bar = int(args[2]) if len(args) > 2 else int(beat / 4)
            except:
                beat = scheduler.current_beat + 1
                bar = scheduler.current_bar
        else:
            beat = scheduler.current_beat + 1
            bar = int(beat / 4)
        
        results = scheduler.tick(beat, bar)
        if results:
            return f"Tick {beat:.1f} (bar {bar}): executed {len(results)} commands"
        return f"Tick {beat:.1f} (bar {bar}): no commands ready"
    
    else:
        return f"Unknown subcommand: {subcmd}. Use /MC for help."


def cmd_sched(session: "Session", args: List[str]) -> str:
    """Direct scheduler control.
    
    Usage:
      /sched                      Show pending events
      /sched <timing> <cmd>       Schedule a command
      /sched clear                Clear all pending
      /sched tick [beat] [bar]    Manually advance
    
    Timing options:
      @now         Execute immediately
      @beat        Execute on next beat
      @bar         Execute on next bar
      @delay:N     Delay N beats
    
    Examples:
      /sched @bar /fx reverb_large
      /sched @delay:4 /stop
      /sched @beat /vol 80
    """
    from ..dsp.performance import get_macro_scheduler
    scheduler = get_macro_scheduler(session.bpm if hasattr(session, 'bpm') else 120.0)
    
    if hasattr(session, '_execute_line'):
        scheduler.set_command_handler(session._execute_line)
    
    if not args:
        return scheduler.list_pending()
    
    sub = args[0].lower()
    
    if sub == 'clear':
        scheduler.clear()
        return "OK: Cleared scheduled events"
    
    elif sub == 'tick':
        beat = float(args[1]) if len(args) > 1 else scheduler.current_beat + 1
        bar = int(args[2]) if len(args) > 2 else int(beat / 4)
        results = scheduler.tick(beat, bar)
        return f"Tick {beat:.1f}: {len(results)} commands executed"
    
    elif sub.startswith('@'):
        # Schedule command: /sched @timing command...
        timing = sub
        command = ' '.join(args[1:])
        if not command:
            return "Usage: /sched @timing <command>"
        return scheduler.schedule(command, timing)
    
    else:
        return "Usage: /sched [@timing command | clear | tick]"


def cmd_mc_record(session: "Session", cmd_line: str) -> str:
    """Record a command to the current macro definition.
    
    Called by REPL when in macro definition mode.
    """
    if not hasattr(session, 'defining_macro') or not session.defining_macro:
        return None
    
    # Check for /end
    if cmd_line.strip().lower() in ('/end', '/mc end'):
        return cmd_mc(session, ['end'])
    
    # Record the command
    session.macro_commands.append(cmd_line)
    return f"  recorded: {cmd_line}"


# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

def get_advanced_commands() -> dict:
    """Return advanced commands for registration."""
    return {
        # User variables
        '=': cmd_set,
        'set': cmd_set,
        'GET': cmd_get,
        'get': cmd_get,
        'DEL': cmd_del_var,
        
        # Chunking
        'CHK': cmd_chk,
        'chk': cmd_chk,
        'chunk': cmd_chk,
        
        # Remix
        'remix': cmd_remix,
        'rmx': cmd_remix,
        
        # Rhythmic pattern
        'RPAT': cmd_rpat,
        'rpat': cmd_rpat,
        'rp': cmd_rpat,
        
        # Buffer combining
        'CBI': cmd_cbi,
        'cbi': cmd_cbi,
        'combine': cmd_cbi,
        'BAP': cmd_bap,
        'bap': cmd_bap,
        
        # Bridge commands
        'PR': cmd_pr,
        'pr': cmd_pr,
        'print': cmd_pr,
        'YT': cmd_yt,
        'yt': cmd_yt,
        'SC': cmd_sc,
        'sc': cmd_sc,
        'DK2BUF': cmd_dk2buf,
        'dk2buf': cmd_dk2buf,
        
        # Wavetable
        'WT': cmd_wt,
        'wt': cmd_wt,
        'wavetable': cmd_wt,
        
        # Macro system
        'MC': cmd_mc,
        'mc': cmd_mc,
        'macro': cmd_mc,
        
        # Scheduler
        'sched': cmd_sched,
        'schedule': cmd_sched,
    }
