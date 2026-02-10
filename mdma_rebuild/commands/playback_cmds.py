"""Playback Commands for MDMA.

UNIFIED IN-HOUSE PLAYBACK SYSTEM
================================
All playback uses the in-house audio engine (sounddevice) for consistency.

PLAYBACK COMMANDS:
- /P           → Play WORKING BUFFER (always)
- /PB <n>      → Play buffer N
- /PT <n>      → Play track N
- /PTS         → Play SONG (all tracks mixed)
- /PD <n>      → Play deck N
- /PALL        → Play all buffers mixed

OUTPUT FORMAT:
All commands output: ▶ Playing: <source> (<duration>s) @ <sample_rate>Hz

WORKING BUFFER:
- /W <source>  → Load audio to working buffer
- /WBC         → Clear working buffer
- /WFX <fx>    → Add effect with deviation metrics

BUILD ID: playback_cmds_v16.0
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.session import Session


# ============================================================================
# UNIFIED PLAYBACK ENGINE
# ============================================================================

def _unified_play(audio: np.ndarray, label: str, session: "Session") -> str:
    """Unified playback function - ALL playback goes through here.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio data to play (mono or stereo, any dtype)
    label : str
        Human-readable source label (e.g., "working buffer", "buffer 3")
    session : Session
        Session object for sample rate and playback
    
    Returns
    -------
    str
        Status message in format: ▶ Playing: <label> (<duration>s) @ <sr>Hz
    """
    if audio is None or len(audio) == 0:
        return f"ERROR: No audio to play ({label} is empty)"
    
    # Ensure float64 for processing
    audio = np.asarray(audio, dtype=np.float64)
    
    # Calculate duration
    if audio.ndim == 2:
        duration = audio.shape[0] / session.sample_rate
    else:
        duration = len(audio) / session.sample_rate
    
    # Play using session's in-house playback
    if session._play_buffer(audio, 0.8):
        return f"▶ Playing: {label} ({duration:.2f}s) @ {session.sample_rate}Hz"
    else:
        return f"ERROR: Playback failed for {label} - check audio backend"


# ============================================================================
# PLAYBACK CONTEXT TRACKING
# ============================================================================

class PlaybackContext:
    """Track current playback context and working buffer state."""
    
    _instance: Optional['PlaybackContext'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.context = 'working'  # Default to working buffer
            cls._instance.context_id = 0
            cls._instance.working_buffer = None
            cls._instance.working_source = ""
            cls._instance.working_fx_chain = []
        return cls._instance
    
    def set_context(self, ctx: str, ctx_id: int = 0):
        """Set current playback context."""
        self.context = ctx
        self.context_id = ctx_id
    
    def get_context(self) -> tuple:
        """Get current context."""
        return (self.context, self.context_id)
    
    def set_working(self, audio: np.ndarray, source: str = ""):
        """Set working buffer."""
        self.working_buffer = audio.copy() if audio is not None else None
        self.working_source = source
        self.working_fx_chain = []
    
    def get_working(self) -> tuple:
        """Get working buffer and source."""
        return (self.working_buffer, self.working_source)
    
    def clear_working(self):
        """Clear working buffer."""
        self.working_buffer = None
        self.working_source = ""
        self.working_fx_chain = []


def get_playback_context() -> PlaybackContext:
    """Get global playback context."""
    return PlaybackContext()


# ============================================================================
# AUDIO METRICS
# ============================================================================

def compute_metrics(audio: np.ndarray) -> dict:
    """Compute audio metrics for deviation display."""
    if audio is None or len(audio) == 0:
        return {"peak": 0, "rms": 0, "duration": 0, "dc": 0}
    
    # Ensure 1D for analysis
    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio
    
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono ** 2)))
    dc = float(np.mean(mono))
    duration = len(mono)
    
    # Spectral centroid approximation (brightness)
    fft = np.abs(np.fft.rfft(mono[:min(len(mono), 4096)]))
    freqs = np.fft.rfftfreq(min(len(mono), 4096), 1.0/48000)
    if np.sum(fft) > 0:
        centroid = float(np.sum(freqs * fft) / np.sum(fft))
    else:
        centroid = 0
    
    return {
        "peak": peak,
        "rms": rms,
        "dc": dc,
        "duration": duration,
        "centroid": centroid,
        "peak_db": 20 * np.log10(peak + 1e-10),
        "rms_db": 20 * np.log10(rms + 1e-10),
    }


def format_deviation(before: dict, after: dict, sr: int = 48000) -> str:
    """Format deviation metrics between two audio states."""
    lines = []
    
    # Duration change
    dur_before = before["duration"] / sr
    dur_after = after["duration"] / sr
    dur_change = dur_after - dur_before
    if abs(dur_change) > 0.001:
        lines.append(f"  Duration: {dur_before:.3f}s → {dur_after:.3f}s ({dur_change:+.3f}s)")
    
    # Peak change
    peak_change = after["peak_db"] - before["peak_db"]
    if abs(peak_change) > 0.1:
        lines.append(f"  Peak: {before['peak_db']:.1f}dB → {after['peak_db']:.1f}dB ({peak_change:+.1f}dB)")
    
    # RMS change (loudness)
    rms_change = after["rms_db"] - before["rms_db"]
    if abs(rms_change) > 0.1:
        lines.append(f"  RMS: {before['rms_db']:.1f}dB → {after['rms_db']:.1f}dB ({rms_change:+.1f}dB)")
    
    # Brightness change
    cent_change = after["centroid"] - before["centroid"]
    if abs(cent_change) > 50:
        lines.append(f"  Brightness: {before['centroid']:.0f}Hz → {after['centroid']:.0f}Hz ({cent_change:+.0f}Hz)")
    
    # DC offset
    if abs(after["dc"]) > 0.01:
        lines.append(f"  DC offset: {after['dc']:.4f}")
    
    return "\n".join(lines) if lines else "  No significant changes"


# ============================================================================
# SMART PLAYBACK COMMAND
# ============================================================================

def cmd_p(session: "Session", args: List[str]) -> str:
    """Play working buffer.
    
    Usage:
      /P                Play working buffer (session.working_buffer)
    
    The /P command ALWAYS plays the working buffer. This is the central
    audio workspace where generated audio, effects previews, and 
    intermediate results are stored.
    
    For other sources, use explicit commands:
      /PB <n>           Play buffer n
      /PT <n>           Play track n
      /PTS              Play song (all tracks mixed)
      /PD <n>           Play deck n
    
    Working buffer is populated by:
      - /tone, /g_tom, etc. (generators)
      - /mel, /cor (melody/chord commands)
      - /W <source> (load from buffer/deck/file)
      - /fx commands (when applied)
    """
    # Get working buffer from session
    if not hasattr(session, 'working_buffer') or session.working_buffer is None:
        return "ERROR: Working buffer is empty. Generate audio with /tone, /mel, or load with /W"
    
    audio = session.working_buffer
    if len(audio) == 0:
        return "ERROR: Working buffer is empty. Generate audio with /tone, /mel, or load with /W"
    
    # Check if it's just initialization silence
    if hasattr(session, 'working_buffer_source') and session.working_buffer_source == 'init':
        # Check if there's actual audio content
        if np.max(np.abs(audio)) < 1e-7:
            return "ERROR: Working buffer contains only silence. Generate audio first."
    
    # Build label with source info
    source = getattr(session, 'working_buffer_source', 'unknown')
    label = f"working buffer [{source}]"
    
    return _unified_play(audio, label, session)


def cmd_pw(session: "Session", args: List[str]) -> str:
    """Play working buffer (explicit alias for /P).
    
    Usage:
      /PW               Play working buffer
    
    This is an explicit alias for /P. Both commands play the working buffer.
    """
    return cmd_p(session, args)


def cmd_pb(session: "Session", args: List[str]) -> str:
    """Play buffer explicitly.
    
    Usage:
      /PB               Play current buffer
      /PB <n>           Play buffer n
    
    Plays from session.buffers[n]. Falls back to last_buffer if
    the specified buffer is empty.
    """
    if not hasattr(session, 'buffers'):
        session.buffers = {}
    
    buf_idx = session.current_buffer_index if hasattr(session, 'current_buffer_index') else 1
    
    if args:
        try:
            buf_idx = int(args[0])
        except ValueError:
            return f"ERROR: Invalid buffer index '{args[0]}'"
    
    # Get buffer audio
    buf_audio = session.buffers.get(buf_idx)
    
    if buf_audio is not None and len(buf_audio) > 0:
        return _unified_play(buf_audio, f"buffer {buf_idx}", session)
    
    # Fall back to last_buffer
    if hasattr(session, 'last_buffer') and session.last_buffer is not None and len(session.last_buffer) > 0:
        return _unified_play(session.last_buffer, f"buffer {buf_idx} (from last)", session)
    
    return f"ERROR: Buffer {buf_idx} is empty"


def cmd_pt(session: "Session", args: List[str]) -> str:
    """Play track explicitly.
    
    Usage:
      /PT               Play current track (with track + master FX)
      /PT <n>           Play track n (1-indexed)
      /PT <n> dry       Play track n without master FX
    
    Plays a single track from the timeline with its FX chain, gain, and
    pan applied.  By default the master FX chain is also applied so the
    preview matches the final mix.  Add 'dry' to hear the track without
    master processing.
    
    Use /PTS to play all tracks mixed together (song).
    """
    if not hasattr(session, 'tracks') or not session.tracks:
        return "ERROR: No tracks available. Create with /tn or /ta"
    
    # Get track index (1-indexed for user, 0-indexed internally)
    track_idx = session.current_track_index if hasattr(session, 'current_track_index') else 0
    include_master = True
    
    for arg in args:
        if arg.lower() == 'dry':
            include_master = False
        else:
            try:
                track_idx = int(arg) - 1  # Convert to 0-indexed
            except ValueError:
                return f"ERROR: Invalid track index '{arg}'"
    
    if track_idx < 0 or track_idx >= len(session.tracks):
        return f"ERROR: Track {track_idx + 1} not found (have {len(session.tracks)} tracks)"
    
    track = session.tracks[track_idx]
    
    # Check track has audio before calling preview
    raw_audio = track.get('audio')
    if raw_audio is None or (hasattr(raw_audio, '__len__') and len(raw_audio) == 0):
        return f"ERROR: Track {track_idx + 1} is empty"
    if np.max(np.abs(raw_audio)) < 1e-7:
        return f"ERROR: Track {track_idx + 1} contains only silence"
    
    # Use session.preview_track() for full FX pipeline
    audio = session.preview_track(track_idx, include_master=include_master)
    
    if audio is None or (hasattr(audio, '__len__') and len(audio) == 0):
        return f"ERROR: Track {track_idx + 1} produced empty output after FX processing"
    
    # Build label
    track_name = track.get('name', f'Track {track_idx + 1}')
    fx_note = "" if include_master else " (dry, no master FX)"
    chain = track.get('fx_chain', []) or getattr(session, 'track_fx_chain', [])
    master = getattr(session, 'master_fx_chain', [])
    fx_count = len(chain) + (len(master) if include_master else 0)
    fx_info = f" [{fx_count} FX]" if fx_count > 0 else ""
    label = f"track {track_idx + 1} \"{track_name}\"{fx_info}{fx_note}"
    
    return _unified_play(audio, label, session)


def cmd_pts(session: "Session", args: List[str]) -> str:
    """Play song (all tracks mixed).
    
    Usage:
      /PTS              Play all tracks mixed together
    
    Mixes all tracks respecting:
      - Per-track gain and pan
      - Mute/solo states
      - Per-track and master FX chains
    
    This is the "song" - all tracks combined into a final mix.
    """
    if not hasattr(session, 'tracks') or not session.tracks:
        return "ERROR: No tracks available. Create with /tn or /ta"
    
    # Count active tracks
    active_count = sum(1 for t in session.tracks 
                      if t.get('audio') is not None 
                      and len(t.get('audio', [])) > 0
                      and not t.get('mute', False))
    
    if active_count == 0:
        return "ERROR: No active tracks with audio"
    
    # Mix all tracks
    try:
        mixed = session.mix_tracks()
    except Exception as e:
        return f"ERROR: Mix failed - {e}"
    
    if mixed is None or len(mixed) == 0:
        return "ERROR: Mix produced no audio"
    
    # Build label
    total_tracks = len(session.tracks)
    label = f"song [{active_count}/{total_tracks} tracks]"
    
    return _unified_play(mixed, label, session)


def cmd_pall(session: "Session", args: List[str]) -> str:
    """Play all buffers mixed.
    
    Usage:
      /PALL             Play all buffers combined
    
    Mixes all non-empty buffers into a single audio stream.
    """
    if not hasattr(session, 'buffers') or not session.buffers:
        return "ERROR: No buffers available"
    
    # Collect all non-empty buffers
    buffers = []
    for idx in sorted(session.buffers.keys()):
        buf = session.buffers[idx]
        if buf is not None and len(buf) > 0:
            buffers.append((idx, buf))
    
    if not buffers:
        return "ERROR: All buffers are empty"
    
    if len(buffers) == 1:
        idx, buf = buffers[0]
        return _unified_play(buf, f"buffer {idx}", session)
    
    # Mix buffers
    max_len = max(len(buf) for _, buf in buffers)
    mixed = np.zeros(max_len, dtype=np.float64)
    
    for idx, buf in buffers:
        mixed[:len(buf)] += buf.astype(np.float64)
    
    # Normalize if clipping
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    
    label = f"all buffers [{len(buffers)} mixed]"
    return _unified_play(mixed, label, session)


def cmd_pd(session: "Session", args: List[str]) -> str:
    """Play deck explicitly.
    
    Usage:
      /PD               Play active deck
      /PD <n>           Play deck n
    
    Plays deck audio with its FX chain applied.
    Decks are enhanced buffers used in DJ mode.
    """
    try:
        from ..dsp.dj_mode import get_dj_engine
        dj = get_dj_engine(session.sample_rate)
        
        deck_id = dj.active_deck
        if args:
            try:
                deck_id = int(args[0])
            except ValueError:
                return f"ERROR: Invalid deck id '{args[0]}'"
        
        # Initialize deck if it doesn't exist
        if deck_id not in dj.decks:
            dj._ensure_deck(deck_id)
        
        deck = dj.decks.get(deck_id)
        if deck is None:
            return f"ERROR: Deck {deck_id} not found"
        
        # Get deck audio
        audio = deck.buffer if hasattr(deck, 'buffer') else None
        if audio is None:
            audio = deck.audio if hasattr(deck, 'audio') else None
        
        if audio is None or len(audio) == 0:
            # Try to get from session buffers as fallback
            if hasattr(session, 'buffers') and deck_id in session.buffers:
                audio = session.buffers[deck_id]
            else:
                return f"ERROR: Deck {deck_id} is empty"
        
        # Render through deck FX chain if available
        if hasattr(session, 'render_deck'):
            try:
                rendered = session.render_deck(deck_id)
                if rendered is not None and len(rendered) > 0:
                    audio = rendered
            except Exception:
                pass
        
        # Apply deck volume
        audio = audio.astype(np.float64)
        if hasattr(deck, 'volume') and deck.volume != 1.0:
            audio = audio * deck.volume
        
        # Build label with tempo info
        tempo_str = f" @ {deck.tempo:.1f}BPM" if hasattr(deck, 'tempo') and deck.tempo > 0 else ""
        label = f"deck {deck_id}{tempo_str}"
        
        return _unified_play(audio, label, session)
        
    except ImportError:
        return "ERROR: DJ mode module not available"
    except Exception as e:
        return f"ERROR: Deck playback failed: {e}"


# ============================================================================
# WORKING BUFFER COMMANDS
# ============================================================================

def cmd_w(session: "Session", args: List[str]) -> str:
    """Load audio into working buffer.
    
    Usage:
      /W                Load current buffer to working
      /W <n>            Load buffer n to working
      /W deck [n]       Load deck to working
    
    Working buffer is for testing effects without modifying source.
    """
    ctx = get_playback_context()
    
    audio = None
    source = ""
    
    if not args:
        # Load current buffer
        if hasattr(session, 'buffers') and hasattr(session, 'current_buffer_index'):
            buf_idx = session.current_buffer_index
            if buf_idx in session.buffers and session.buffers[buf_idx] is not None:
                audio = session.buffers[buf_idx]
                source = f"buffer {buf_idx}"
        elif hasattr(session, 'last_buffer') and session.last_buffer is not None:
            audio = session.last_buffer
            source = "last buffer"
    else:
        arg = args[0].lower()
        
        if arg in ('deck', 'dk', 'd'):
            # Load from deck
            try:
                from ..dsp.dj_mode import get_dj_engine
                dj = get_dj_engine(session.sample_rate)
                
                deck_id = dj.active_deck
                if len(args) > 1:
                    try:
                        deck_id = int(args[1])
                    except ValueError:
                        pass
                
                if deck_id in dj.decks and dj.decks[deck_id].buffer is not None:
                    audio = dj.decks[deck_id].buffer
                    source = f"deck {deck_id}"
                else:
                    return f"ERROR: Deck {deck_id} is empty"
            except ImportError:
                return "ERROR: DJ mode not available"
        else:
            # Load buffer by index
            try:
                buf_idx = int(args[0])
                if hasattr(session, 'buffers') and buf_idx in session.buffers:
                    audio = session.buffers[buf_idx]
                    source = f"buffer {buf_idx}"
                else:
                    return f"ERROR: Buffer {buf_idx} not found"
            except ValueError:
                return f"ERROR: Invalid argument '{args[0]}'"
    
    if audio is None or len(audio) == 0:
        return "ERROR: No audio to load"
    
    ctx.set_working(audio, source)
    ctx.set_context('working', 0)
    
    metrics = compute_metrics(audio)
    duration = len(audio) / session.sample_rate
    
    return (f"OK: Loaded {source} to working buffer\n"
            f"  Duration: {duration:.3f}s\n"
            f"  Peak: {metrics['peak_db']:.1f}dB  RMS: {metrics['rms_db']:.1f}dB")


def cmd_wbc(session: "Session", args: List[str]) -> str:
    """Clear working buffer.
    
    Usage:
      /WBC              Clear working buffer and FX chain
    """
    ctx = get_playback_context()
    ctx.clear_working()
    
    return "OK: Working buffer cleared"


def cmd_wfx(session: "Session", args: List[str]) -> str:
    """Add effect to working buffer with deviation metrics.
    
    Usage:
      /WFX <effect> [params...]    Add effect to working buffer
      /WFX list                    List available effects
      /WFX undo                    Undo last effect
      /WFX clear                   Clear all effects (reload source)
    
    Shows before/after deviation metrics when effect is applied.
    
    Examples:
      /WFX reverb 0.3              Add reverb with 30% mix
      /WFX dist 50                 Add distortion at 50%
      /WFX lpf 2000                Lowpass at 2kHz
    """
    ctx = get_playback_context()
    working, source = ctx.get_working()
    
    if not args:
        return ("Usage: /WFX <effect> [params...]\n"
                "  /WFX list - show available effects\n"
                "  /WFX undo - undo last effect\n"
                "  /WFX clear - reload from source")
    
    cmd = args[0].lower()
    
    if cmd == 'list':
        return _list_effects()
    
    if cmd == 'undo':
        return _undo_wfx(session, ctx)
    
    if cmd == 'clear':
        # Reload from source (need to re-run /W command)
        ctx.working_fx_chain = []
        return "OK: Effect chain cleared. Use /W to reload source."
    
    if working is None or len(working) == 0:
        return "ERROR: Working buffer is empty. Use /W to load."
    
    # Apply effect
    effect_name = cmd
    effect_params = args[1:] if len(args) > 1 else []
    
    # Compute before metrics
    before_metrics = compute_metrics(working)
    
    # Apply effect
    try:
        result, effect_desc = _apply_effect(working, effect_name, effect_params, session.sample_rate)
    except Exception as e:
        return f"ERROR: Effect failed - {e}"
    
    if result is None:
        return f"ERROR: Unknown effect '{effect_name}'. Use /WFX list"
    
    # Compute after metrics
    after_metrics = compute_metrics(result)
    
    # Store result
    ctx.working_buffer = result
    ctx.working_fx_chain.append((effect_name, effect_params))
    
    # Format output with deviation
    deviation = format_deviation(before_metrics, after_metrics, session.sample_rate)
    
    chain_str = " → ".join([fx[0] for fx in ctx.working_fx_chain])
    
    return (f"OK: Applied {effect_desc}\n"
            f"  Source: {source}\n"
            f"  Chain: {chain_str}\n"
            f"  Deviation:\n{deviation}")


def _list_effects() -> str:
    """List available effects."""
    return """Available Effects:
  DYNAMICS:
    dist <amt>       Distortion (0-100)
    sat <amt>        Saturation (0-100)
    comp <ratio>     Compression
    limit            Limiter
    gate <thresh>    Noise gate
    
  FILTERS:
    lpf <freq>       Lowpass filter
    hpf <freq>       Highpass filter
    bpf <freq> <q>   Bandpass filter
    notch <freq>     Notch filter
    
  MODULATION:
    chorus <rate>    Chorus effect
    flanger <rate>   Flanger
    phaser <rate>    Phaser
    vibrato <rate>   Vibrato
    
  SPACE:
    reverb <mix>     Reverb (0-1)
    delay <ms> <fb>  Delay
    echo <ms>        Simple echo
    
  PITCH/TIME:
    pitch <semi>     Pitch shift
    stretch <ratio>  Time stretch
    
  UTILITY:
    norm             Normalize to 0dB
    gain <db>        Gain adjustment
    fade <in> <out>  Fade in/out (seconds)
    reverse          Reverse audio
    dc               Remove DC offset"""


def _apply_effect(
    audio: np.ndarray, 
    effect: str, 
    params: List[str],
    sr: int
) -> tuple:
    """Apply effect to audio. Returns (result, description)."""
    effect = effect.lower()
    
    # Parse first param as float if possible
    p1 = 0.5
    if params:
        try:
            p1 = float(params[0])
        except ValueError:
            pass
    
    # Parse second param
    p2 = 0.5
    if len(params) > 1:
        try:
            p2 = float(params[1])
        except ValueError:
            pass
    
    # Ensure mono for some effects
    is_stereo = audio.ndim > 1
    if is_stereo:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio
    
    result = None
    desc = ""
    
    # DISTORTION
    if effect in ('dist', 'distortion'):
        amt = p1 / 100 if p1 > 1 else p1
        result = np.tanh(mono * (1 + amt * 10))
        desc = f"distortion @ {amt*100:.0f}%"
    
    elif effect in ('sat', 'saturation'):
        amt = p1 / 100 if p1 > 1 else p1
        result = np.tanh(mono * (1 + amt * 3)) * 0.9
        desc = f"saturation @ {amt*100:.0f}%"
    
    # FILTERS
    elif effect == 'lpf':
        freq = p1 if p1 > 20 else 1000
        result = _simple_lpf(mono, freq, sr)
        desc = f"lowpass @ {freq:.0f}Hz"
    
    elif effect == 'hpf':
        freq = p1 if p1 > 20 else 200
        result = _simple_hpf(mono, freq, sr)
        desc = f"highpass @ {freq:.0f}Hz"
    
    # REVERB (simple)
    elif effect in ('reverb', 'rev'):
        mix = p1 if p1 <= 1 else p1 / 100
        result = _simple_reverb(mono, mix, sr)
        desc = f"reverb @ {mix*100:.0f}% mix"
    
    # DELAY
    elif effect in ('delay', 'del'):
        delay_ms = p1 if p1 > 1 else 250
        feedback = p2 if p2 <= 1 else 0.3
        result = _simple_delay(mono, delay_ms, feedback, sr)
        desc = f"delay {delay_ms:.0f}ms @ {feedback*100:.0f}% fb"
    
    # UTILITY
    elif effect in ('norm', 'normalize'):
        peak = np.max(np.abs(mono))
        if peak > 0:
            result = mono / peak
        else:
            result = mono
        desc = "normalized to 0dB"
    
    elif effect == 'gain':
        db = p1
        result = mono * (10 ** (db / 20))
        desc = f"gain {db:+.1f}dB"
    
    elif effect == 'reverse':
        result = mono[::-1]
        desc = "reversed"
    
    elif effect == 'dc':
        dc = np.mean(mono)
        result = mono - dc
        desc = f"DC removed ({dc:.4f})"
    
    elif effect == 'fade':
        fade_in = int(p1 * sr)
        fade_out = int(p2 * sr) if p2 > 0 else fade_in
        result = mono.copy()
        if fade_in > 0 and fade_in < len(result):
            result[:fade_in] *= np.linspace(0, 1, fade_in)
        if fade_out > 0 and fade_out < len(result):
            result[-fade_out:] *= np.linspace(1, 0, fade_out)
        desc = f"fade {p1:.2f}s in, {p2:.2f}s out"
    
    # Unknown
    else:
        return None, ""
    
    # Convert back to stereo if needed
    if is_stereo and result is not None:
        result = np.column_stack([result, result])
    
    return result, desc


def _simple_lpf(audio: np.ndarray, freq: float, sr: int) -> np.ndarray:
    """Simple one-pole lowpass filter."""
    rc = 1.0 / (2 * np.pi * freq)
    dt = 1.0 / sr
    alpha = dt / (rc + dt)
    
    result = np.zeros_like(audio)
    result[0] = audio[0]
    for i in range(1, len(audio)):
        result[i] = result[i-1] + alpha * (audio[i] - result[i-1])
    return result


def _simple_hpf(audio: np.ndarray, freq: float, sr: int) -> np.ndarray:
    """Simple highpass filter."""
    return audio - _simple_lpf(audio, freq, sr)


def _simple_reverb(audio: np.ndarray, mix: float, sr: int) -> np.ndarray:
    """Simple comb filter reverb."""
    delays = [int(sr * d / 1000) for d in [23, 37, 47, 67]]
    decays = [0.7, 0.65, 0.6, 0.55]
    
    result = audio.copy()
    for delay, decay in zip(delays, decays):
        comb = np.zeros(len(audio) + delay)
        for i in range(delay, len(comb)):
            if i - delay < len(audio):
                comb[i] = audio[i - delay] + comb[i - delay] * decay
        result += comb[:len(audio)] * (mix / len(delays))
    
    # Normalize
    peak = np.max(np.abs(result))
    if peak > 1:
        result /= peak
    
    return result


def _simple_delay(audio: np.ndarray, delay_ms: float, feedback: float, sr: int) -> np.ndarray:
    """Simple delay effect."""
    delay_samples = int(delay_ms * sr / 1000)
    
    result = np.zeros(len(audio) + delay_samples * 4)
    result[:len(audio)] = audio
    
    for i in range(4):  # 4 echoes
        start = delay_samples * (i + 1)
        end = start + len(audio)
        if end <= len(result):
            result[start:end] += audio * (feedback ** (i + 1))
    
    return result[:len(audio)]


def _undo_wfx(session: "Session", ctx: PlaybackContext) -> str:
    """Undo last effect on working buffer."""
    if not ctx.working_fx_chain:
        return "Nothing to undo"
    
    # Can't easily undo, need to reload from source
    # Just remove last from chain and note it
    removed = ctx.working_fx_chain.pop()
    return f"OK: Removed '{removed[0]}' from chain. Use /W to reload and reapply."


# ============================================================================
# STANDARD PLAYBACK COMMANDS
# ============================================================================

def cmd_play(session: "Session", args: List[str]) -> str:
    """Play the current audio buffer.
    
    Usage:
      /play              -> Play at default volume (80%)
      /play 50           -> Play at 50% volume
      /play 100          -> Play at full volume
    
    Uses in-house audio playback (sounddevice/simpleaudio).
    """
    volume = 0.8
    if args:
        try:
            vol_arg = float(args[0])
            if vol_arg > 1.0:
                vol_arg = vol_arg / 100.0
            volume = max(0.0, min(1.0, vol_arg))
        except ValueError:
            pass
    
    return session.play(volume)


def cmd_stop(session: "Session", args: List[str]) -> str:
    """Stop audio playback.
    
    Usage:
      /stop              -> Stop playback
    """
    return session.stop_playback()


def cmd_smart_stop(session: "Session", args: List[str]) -> str:
    """Smart stop - stops ALL audio from ALL sources.
    
    Usage:
      /s                 -> Stop everything
    
    Stops:
      - Active playback (buffer, deck, working buffer)
      - DJ deck playback
      - Live loops
      - Any streaming audio
    """
    stopped = []
    
    # Stop regular playback
    try:
        result = session.stop_playback()
        if 'stopped' in result.lower() or 'ok' in result.lower():
            stopped.append("playback")
    except:
        pass
    
    # Stop DJ decks
    try:
        from ..dsp.dj_mode import get_dj_engine
        dj = get_dj_engine(session.sample_rate)
        if dj.enabled:
            for deck_id, deck in dj.decks.items():
                if deck.playing:
                    deck.playing = False
                    stopped.append(f"deck {deck_id}")
    except:
        pass
    
    # Stop live loops
    try:
        if hasattr(session, 'live_loops'):
            for name, loop in session.live_loops.items():
                if loop.get('playing', False):
                    loop['playing'] = False
                    stopped.append(f"loop '{name}'")
    except:
        pass
    
    # Stop playback engine
    try:
        from ..dsp.playback import get_engine
        engine = get_engine()
        if hasattr(engine, 'stop'):
            engine.stop()
    except:
        pass
    
    # Stop performance engine
    try:
        from ..dsp.performance import get_perf_engine
        perf = get_perf_engine()
        if hasattr(perf, 'stop_all'):
            perf.stop_all()
            stopped.append("performance")
    except:
        pass
    
    if stopped:
        return f"STOP: Stopped {', '.join(stopped)}"
    return "STOP: All audio stopped"


def cmd_status(session: "Session", args: List[str]) -> str:
    """Show playback status with context info.
    
    Usage:
      /status            -> Show playback status
      /st                -> Same as /status
    """
    status = session.playback_status()
    ctx = get_playback_context()
    context, ctx_id = ctx.get_context()
    
    lines = ["=== PLAYBACK STATUS ==="]
    lines.append(f"  Context: {context} {ctx_id}")
    lines.append(f"  State: {status.get('state', 'unknown')}")
    lines.append(f"  Backend: {status.get('backend', 'none')}")
    
    if status.get('duration', 0) > 0:
        lines.append(f"  Duration: {status.get('duration', 0):.2f}s")
        lines.append(f"  Position: {status.get('position', 0):.2f}s")
        lines.append(f"  Progress: {status.get('progress', 0):.1f}%")
    
    lines.append(f"  Volume: {status.get('volume', 0.8):.0%}")
    
    # Working buffer info
    working, source = ctx.get_working()
    if working is not None:
        lines.append(f"\n  Working buffer: {len(working)/session.sample_rate:.2f}s from {source}")
        if ctx.working_fx_chain:
            chain = " → ".join([fx[0] for fx in ctx.working_fx_chain])
            lines.append(f"  FX chain: {chain}")
    
    return '\n'.join(lines)


def cmd_vol(session: "Session", args: List[str]) -> str:
    """Set playback volume."""
    try:
        from ..dsp.playback import get_engine
        engine = get_engine()
        
        if args:
            try:
                vol = float(args[0])
                if vol > 1.0:
                    vol = vol / 100.0
                vol = max(0.0, min(1.0, vol))
                engine.set_volume(vol)
                return f"VOLUME: {vol:.0%}"
            except ValueError:
                return f"ERROR: invalid volume '{args[0]}'"
        else:
            return f"VOLUME: {engine.volume:.0%}"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_backend(session: "Session", args: List[str]) -> str:
    """Show audio backend info."""
    try:
        from ..dsp.playback import get_backend, get_engine
        backend = get_backend()
        engine = get_engine()
        
        lines = ["Audio backend:"]
        lines.append(f"  Active: {backend}")
        lines.append(f"  Sample rate: {engine.sample_rate}Hz")
        
        if backend == "sounddevice":
            try:
                import sounddevice as sd
                default_device = sd.default.device
                lines.append(f"  Default output: {default_device}")
            except Exception:
                pass
        elif backend == "fallback":
            lines.append("  WARNING: No audio backend available")
            lines.append("  Install: pip install sounddevice")
        
        return '\n'.join(lines)
    except Exception as e:
        return f"ERROR: {e}"


def cmd_tone_test(session: "Session", args: List[str]) -> str:
    """Play a test tone to verify audio output."""
    freq = 440.0
    duration = 0.5
    
    if args:
        try:
            freq = float(args[0])
        except ValueError:
            pass
        if len(args) > 1:
            try:
                duration = float(args[1])
            except ValueError:
                pass
    
    try:
        from ..dsp.playback import play_tone
        if play_tone(freq, duration, session.sample_rate, 0.5):
            return f"TEST TONE: {freq:.0f}Hz for {duration:.1f}s"
        else:
            return "ERROR: test tone failed - check audio backend"
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

# Global bypass flag - when True, always use session._play_buffer instead of device output
_BYPASS_DECK_OUTPUT = True  # Default to True for reliability


def get_bypass_mode() -> bool:
    """Get current bypass mode setting."""
    return _BYPASS_DECK_OUTPUT


def set_bypass_mode(enabled: bool) -> None:
    """Set bypass mode."""
    global _BYPASS_DECK_OUTPUT
    _BYPASS_DECK_OUTPUT = enabled
    
    # Also update DJ engine if available
    try:
        from ..dsp.dj_mode import get_dj_engine
        dj = get_dj_engine(48000)  # sample rate doesn't matter for setting
        dj.bypass_deck_output = enabled
    except:
        pass


def _play_buffer(session: "Session", buf_idx: int) -> str:
    """Play a specific buffer with buffer FX chain applied.
    
    Falls back to last_buffer if the selected buffer is empty.
    This ensures /P always plays something if audio was recently generated.
    """
    import numpy as np

    if not hasattr(session, 'buffers'):
        session.buffers = {}
    
    # Check if buffer has actual audio
    buf_audio = session.buffers.get(buf_idx)
    has_buffer = buf_audio is not None and len(buf_audio) > 0
    
    # Check if last_buffer has audio
    has_last = hasattr(session, 'last_buffer') and session.last_buffer is not None and len(session.last_buffer) > 0
    
    # Decide what to play
    if has_buffer:
        audio = buf_audio.astype(np.float64, copy=True)
        source = f"buffer {buf_idx}"
    elif has_last:
        audio = session.last_buffer.astype(np.float64, copy=True)
        source = "last buffer"
    else:
        return f"ERROR: Buffer {buf_idx} is empty (no audio generated yet)"
    
    # Apply buffer FX chain if present
    if hasattr(session, 'buffer_fx_chain') and session.buffer_fx_chain:
        try:
            if audio.ndim == 2 and audio.shape[1] == 2:
                left = session.apply_fx_chain(audio[:, 0], session.buffer_fx_chain)
                right = session.apply_fx_chain(audio[:, 1], session.buffer_fx_chain)
                audio = np.column_stack([left, right])
            else:
                audio = session.apply_fx_chain(audio, session.buffer_fx_chain)
            source += " +FX"
        except Exception:
            pass

    duration = len(audio) / session.sample_rate
    
    if session._play_buffer(audio, 0.8):
        return f"PLAYING: {source} ({duration:.2f}s)"
    return "ERROR: Playback failed - check audio backend"


def _play_deck(session: "Session", deck_id: int) -> str:
    """Play a specific deck using in-house playback with FX applied.
    
    Decks are treated like enhanced buffers. Always uses session
    playback for reliability unless bypass is disabled AND device
    output is explicitly working.
    """
    import numpy as np

    try:
        from ..dsp.dj_mode import get_dj_engine
        dj = get_dj_engine(session.sample_rate)
        
        # Initialize deck if it doesn't exist
        if deck_id not in dj.decks:
            dj._ensure_deck(deck_id)
        
        deck = dj.decks.get(deck_id)
        if deck is None:
            return f"ERROR: Deck {deck_id} not found"
        
        if deck.buffer is None or len(deck.buffer) == 0:
            # Try to get audio from session buffers as fallback
            if hasattr(session, 'buffers') and deck_id in session.buffers:
                deck.buffer = session.buffers[deck_id]
            elif hasattr(session, 'last_buffer') and session.last_buffer is not None:
                deck.buffer = session.last_buffer
            else:
                return f"ERROR: Deck {deck_id} is empty"
        
        # Render deck through its FX chain
        audio = session.render_deck(deck_id)
        if audio is None:
            audio = deck.buffer
        
        duration = len(audio) / session.sample_rate
        
        # Apply deck volume
        audio_out = audio * deck.volume if hasattr(deck, 'volume') else audio
        
        # Always use in-house playback (reliable)
        if session._play_buffer(audio_out, 0.8):
            tempo_str = f" @ {deck.tempo:.1f}BPM" if hasattr(deck, 'tempo') and deck.tempo > 0 else ""
            fx_info = ""
            deck_session = session.decks.get(deck_id, {})
            if deck_session.get('fx_chain'):
                fx_info = " +FX"
            return f"PLAYING: deck {deck_id} ({duration:.2f}s){tempo_str}{fx_info}"
        
        return "ERROR: Playback failed"
        
    except ImportError:
        return "ERROR: DJ mode module not available"
    except Exception as e:
        return f"ERROR: Deck playback failed: {e}"


def _play_clip(session: "Session", clip_id: int) -> str:
    """Play a specific clip."""
    # Clips not fully implemented yet
    return f"ERROR: Clip playback not yet implemented"


# ============================================================================
# BYPASS AND UNIFIED PLAYBACK COMMANDS
# ============================================================================

def cmd_bybd(session: "Session", args: List[str]) -> str:
    """Toggle or set bypass deck output mode.
    
    Usage:
      /BYBD             Toggle bypass mode
      /BYBD on          Enable bypass (use in-house playback)
      /BYBD off         Disable bypass (use device output)
      /BYBD status      Show current mode
    
    When bypass is ON (default):
      - All playback uses session's reliable in-house audio
      - No external device routing needed
      - Works consistently across all systems
    
    When bypass is OFF:
      - DJ mode attempts to use external device output
      - May provide lower latency on supported systems
      - Falls back to in-house if device fails
    
    Aliases: /bypass
    """
    current = get_bypass_mode()
    
    if not args:
        # Toggle
        new_mode = not current
        set_bypass_mode(new_mode)
        status = "ON (in-house playback)" if new_mode else "OFF (device output)"
        return f"OK: Bypass mode {status}"
    
    arg = args[0].lower()
    
    if arg in ('on', 'enable', 'true', '1', 'yes'):
        set_bypass_mode(True)
        return "OK: Bypass mode ON - using in-house playback for all audio"
    elif arg in ('off', 'disable', 'false', '0', 'no'):
        set_bypass_mode(False)
        return "OK: Bypass mode OFF - will attempt device output (falls back if fails)"
    elif arg in ('status', 'st', '?'):
        status = "ON (in-house playback)" if current else "OFF (device output)"
        return f"Bypass mode: {status}"
    else:
        return f"ERROR: Unknown argument '{arg}'. Use on/off/status"


def cmd_play_any(session: "Session", args: List[str]) -> str:
    """Play any audio source by type and index.
    
    Usage:
      /PLAY buf <n>     Play buffer n
      /PLAY deck <n>    Play deck n  
      /PLAY work        Play working buffer
      /PLAY last        Play last generated audio
      /PLAY             Smart play (context-aware)
    
    This unified command works regardless of DJ mode status.
    """
    if not args:
        return cmd_p(session, [])
    
    arg = args[0].lower()
    rest = args[1:] if len(args) > 1 else []
    
    if arg in ('buf', 'buffer', 'b'):
        return cmd_pb(session, rest)
    elif arg in ('deck', 'dk', 'd'):
        return cmd_pd(session, rest)
    elif arg in ('work', 'working', 'w'):
        return cmd_pw(session, [])
    elif arg in ('last', 'l'):
        if hasattr(session, 'last_buffer') and session.last_buffer is not None:
            audio = session.last_buffer
            duration = len(audio) / session.sample_rate
            if session._play_buffer(audio, 0.8):
                return f"PLAYING: last buffer ({duration:.2f}s)"
            return "ERROR: Playback failed"
        return "ERROR: No last buffer"
    else:
        # Try as buffer index
        try:
            idx = int(arg)
            return _play_buffer(session, idx)
        except ValueError:
            return f"ERROR: Unknown target '{arg}'. Use buf/deck/work/last"


def cmd_buf2deck(session: "Session", args: List[str]) -> str:
    """Copy buffer to deck.
    
    Usage:
      /B2D <buf> <deck>   Copy buffer to deck
      /B2D <buf>          Copy buffer to active deck
      /B2D                Copy current buffer (or last audio) to active deck
    
    Example:
      /B2D 1 2            Copy buffer 1 to deck 2
    
    If the specified buffer is empty, uses last generated audio.
    """
    try:
        from ..dsp.dj_mode import get_dj_engine
        dj = get_dj_engine(session.sample_rate)
        
        # Parse arguments
        buf_idx = session.current_buffer_index if hasattr(session, 'current_buffer_index') else 1
        deck_id = dj.active_deck
        
        if len(args) >= 2:
            buf_idx = int(args[0])
            deck_id = int(args[1])
        elif len(args) == 1:
            buf_idx = int(args[0])
        
        # Get buffer audio - check for actual content, not just existence
        if not hasattr(session, 'buffers'):
            session.buffers = {}
        
        audio = None
        source = f"buffer {buf_idx}"
        
        # Try specified buffer first
        buf_audio = session.buffers.get(buf_idx)
        if buf_audio is not None and len(buf_audio) > 0:
            audio = buf_audio
        # Fall back to last_buffer
        elif hasattr(session, 'last_buffer') and session.last_buffer is not None and len(session.last_buffer) > 0:
            audio = session.last_buffer
            source = "last buffer"
        
        if audio is None or len(audio) == 0:
            return f"ERROR: No audio available (buffer {buf_idx} is empty, no last buffer)"
        
        # Copy to deck
        dj._ensure_deck(deck_id)
        dj.decks[deck_id].buffer = audio.copy()
        dj.decks[deck_id].position = 0.0
        
        duration = len(audio) / session.sample_rate
        return f"OK: Copied {source} to deck {deck_id} ({duration:.2f}s)"
        
    except ImportError:
        return "ERROR: DJ mode module not available"
    except ValueError as e:
        return f"ERROR: Invalid index: {e}"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_deck2buf(session: "Session", args: List[str]) -> str:
    """Copy deck to buffer.
    
    Usage:
      /D2B <deck> <buf>   Copy deck to buffer
      /D2B <deck>         Copy deck to current buffer
      /D2B                Copy active deck to current buffer
    
    Example:
      /D2B 1 2            Copy deck 1 to buffer 2
    """
    try:
        from ..dsp.dj_mode import get_dj_engine
        dj = get_dj_engine(session.sample_rate)
        
        # Parse arguments
        deck_id = dj.active_deck
        buf_idx = session.current_buffer_index if hasattr(session, 'current_buffer_index') else 1
        
        if len(args) >= 2:
            deck_id = int(args[0])
            buf_idx = int(args[1])
        elif len(args) == 1:
            deck_id = int(args[0])
        
        # Get deck audio
        if deck_id not in dj.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = dj.decks[deck_id]
        if deck.buffer is None or len(deck.buffer) == 0:
            return f"ERROR: Deck {deck_id} is empty"
        
        # Copy to buffer
        if not hasattr(session, 'buffers'):
            session.buffers = {}
        
        session.buffers[buf_idx] = deck.buffer.copy()
        session.last_buffer = deck.buffer.copy()
        
        duration = len(deck.buffer) / session.sample_rate
        return f"OK: Copied deck {deck_id} to buffer {buf_idx} ({duration:.2f}s)"
        
    except ImportError:
        return "ERROR: DJ mode module not available"
    except ValueError as e:
        return f"ERROR: Invalid index: {e}"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_sync_buffers(session: "Session", args: List[str]) -> str:
    """Sync all buffers with their corresponding decks.
    
    Usage:
      /SYNCBUF           Sync all (buffer 1 <-> deck 1, etc.)
      /SYNCBUF to_decks  Copy buffers to decks
      /SYNCBUF to_bufs   Copy decks to buffers
    
    This ensures buffers and decks have the same content,
    allowing seamless switching between buffer and DJ workflows.
    """
    try:
        from ..dsp.dj_mode import get_dj_engine
        dj = get_dj_engine(session.sample_rate)
        
        if not hasattr(session, 'buffers'):
            session.buffers = {}
        
        direction = 'both'
        if args:
            arg = args[0].lower()
            if arg in ('to_decks', 'todecks', 'decks', 'd'):
                direction = 'to_decks'
            elif arg in ('to_bufs', 'tobufs', 'bufs', 'buffers', 'b'):
                direction = 'to_bufs'
        
        synced = 0
        
        if direction in ('both', 'to_decks'):
            # Buffers -> Decks
            for buf_idx, audio in session.buffers.items():
                if audio is not None and len(audio) > 0:
                    dj._ensure_deck(buf_idx)
                    if dj.decks[buf_idx].buffer is None or len(dj.decks[buf_idx].buffer) == 0:
                        dj.decks[buf_idx].buffer = audio.copy()
                        synced += 1
        
        if direction in ('both', 'to_bufs'):
            # Decks -> Buffers
            for deck_id, deck in dj.decks.items():
                if deck.buffer is not None and len(deck.buffer) > 0:
                    if deck_id not in session.buffers or session.buffers[deck_id] is None:
                        session.buffers[deck_id] = deck.buffer.copy()
                        synced += 1
        
        return f"OK: Synced {synced} items ({direction})"
        
    except ImportError:
        return "ERROR: DJ mode module not available"
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================================
# ALIASES
# ============================================================================

cmd_st = cmd_status
cmd_pv = cmd_p
cmd_back = cmd_backend
cmd_test = cmd_tone_test
cmd_bypass = cmd_bybd
cmd_b2d = cmd_buf2deck
cmd_d2b = cmd_deck2buf
cmd_syncbuf = cmd_sync_buffers


# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

def get_playback_commands() -> dict:
    """Return playback commands for registration."""
    return {
        # Primary playback (working buffer)
        'p': cmd_p,
        'pw': cmd_pw,
        
        # Explicit source playback
        'pb': cmd_pb,
        'pt': cmd_pt,
        'pts': cmd_pts,
        'pd': cmd_pd,
        'pall': cmd_pall,
        
        # Legacy/unified
        'play': cmd_play,
        'play_any': cmd_play_any,
        
        # Working buffer management
        'w': cmd_w,
        'wbc': cmd_wbc,
        'wfx': cmd_wfx,
        
        # Stop/control
        's': cmd_smart_stop,
        'stop': cmd_stop,
        'smartstop': cmd_smart_stop,
        'status': cmd_status,
        'vol': cmd_vol,
        'volume': cmd_vol,
        'backend': cmd_backend,
        'test': cmd_tone_test,
        
        # Bypass and sync
        'bybd': cmd_bybd,
        'bypass': cmd_bypass,
        'b2d': cmd_b2d,
        'buf2deck': cmd_buf2deck,
        'd2b': cmd_d2b,
        'deck2buf': cmd_deck2buf,
        'syncbuf': cmd_sync_buffers,
        
        # Aliases
        'st': cmd_st,
        'pv': cmd_pv,
        'back': cmd_back,
    }
