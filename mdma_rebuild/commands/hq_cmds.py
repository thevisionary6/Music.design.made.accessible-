"""High-Quality Mode Commands for MDMA.

HQ MODE SYSTEM
==============
Controls audio quality processing for output.

COMMANDS:
- /HQ              → Show HQ settings
- /HQ on/off       → Enable/disable HQ mode
- /HQ <setting>    → Configure individual settings

SETTINGS:
- dc        : DC offset removal
- subsonic  : Subsonic filtering (20Hz highpass)
- highend   : High-end smoothing (16kHz shelf)
- saturation: Soft saturation (analog warmth)
- limiting  : Soft limiting (prevent clipping)
- format    : Output format (wav/flac)
- bits      : Bit depth (16/24)
- osc       : Oscillator mode (smooth/fast)

BUILD ID: hq_cmds_v1.0
"""

from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.session import Session


def cmd_hq(session: "Session", args: List[str]) -> str:
    """High-quality mode settings.
    
    Usage:
      /HQ                      Show all HQ settings
      /HQ on                   Enable HQ mode (all processing)
      /HQ off                  Disable HQ mode (raw output)
      /HQ dc on|off            Toggle DC removal
      /HQ subsonic on|off      Toggle subsonic filter
      /HQ subsonic <freq>      Set subsonic cutoff (Hz)
      /HQ highend on|off       Toggle high-end smoothing
      /HQ highend <freq>       Set high-end frequency (Hz)
      /HQ highend <freq> <dB>  Set frequency and reduction
      /HQ saturation on|off    Toggle soft saturation
      /HQ saturation <amount>  Set saturation drive (0-1)
      /HQ limiting on|off      Toggle soft limiting
      /HQ limiting <threshold> Set limiter threshold (0-1)
      /HQ format wav|flac      Set output format
      /HQ bits 16|24           Set bit depth
      /HQ osc smooth|fast      Set oscillator mode
    
    Examples:
      /hq on                   Enable all HQ processing
      /hq subsonic 30          Set subsonic cutoff to 30Hz
      /hq format flac          Set output to FLAC
      /hq bits 24              Set 24-bit output
    """
    if not args:
        return _show_hq_status(session)
    
    cmd = args[0].lower()
    
    # Global on/off
    if cmd in ('on', 'enable', 'true', '1'):
        session.hq_mode = True
        return "OK: HQ mode ENABLED - all quality processing active"
    
    if cmd in ('off', 'disable', 'false', '0'):
        session.hq_mode = False
        return "OK: HQ mode DISABLED - raw output (no processing)"
    
    # Individual settings
    if cmd == 'dc':
        return _toggle_setting(session, args[1:], 'hq_dc_remove', 'DC removal')
    
    if cmd in ('subsonic', 'sub', 'hp', 'highpass'):
        return _config_subsonic(session, args[1:])
    
    if cmd in ('highend', 'he', 'smooth', 'shelf'):
        return _config_highend(session, args[1:])
    
    if cmd in ('saturation', 'sat', 'warm', 'drive'):
        return _config_saturation(session, args[1:])
    
    if cmd in ('limiting', 'limit', 'lim'):
        return _config_limiting(session, args[1:])
    
    if cmd == 'format':
        return _config_format(session, args[1:])
    
    if cmd in ('bits', 'bitdepth', 'depth'):
        return _config_bits(session, args[1:])
    
    if cmd in ('osc', 'oscillator', 'oscillators'):
        return _config_osc(session, args[1:])
    
    if cmd in ('status', 'st', '?'):
        return _show_hq_status(session)
    
    return f"ERROR: Unknown HQ setting '{cmd}'. Use /hq for help."


def _show_hq_status(session: "Session") -> str:
    """Show all HQ settings."""
    lines = ["=== HIGH-QUALITY MODE ==="]
    
    status = "ENABLED" if session.hq_mode else "DISABLED"
    lines.append(f"  HQ Mode: {status}")
    lines.append("")
    
    lines.append("  Processing Chain:")
    lines.append(f"    DC Removal:     {'ON' if session.hq_dc_remove else 'OFF'}")
    lines.append(f"    Subsonic Filter:{'ON' if session.hq_subsonic_filter else 'OFF'} @ {session.hq_subsonic_freq:.0f}Hz")
    lines.append(f"    High-End Smooth:{'ON' if session.hq_highend_smooth else 'OFF'} @ {session.hq_highend_freq:.0f}Hz ({session.hq_highend_reduction:.1f}dB)")
    lines.append(f"    Soft Saturation:{'ON' if session.hq_saturation else 'OFF'} (drive={session.hq_saturation_drive:.2f})")
    lines.append(f"    Soft Limiting:  {'ON' if session.hq_limiting else 'OFF'} (threshold={session.hq_limit_threshold:.2f})")
    lines.append("")
    
    lines.append("  Output Settings:")
    lines.append(f"    Format:    {session.output_format.upper()}")
    lines.append(f"    Bit Depth: {session.output_bit_depth}-bit")
    lines.append(f"    Oscillators: {'Band-limited (smooth)' if session.hq_oscillators else 'Standard (fast)'}")
    lines.append("")
    
    lines.append("  Use: /hq <setting> <value> to configure")
    
    return "\n".join(lines)


def _toggle_setting(session: "Session", args: List[str], attr: str, name: str) -> str:
    """Toggle a boolean HQ setting."""
    current = getattr(session, attr)
    
    if not args:
        # Toggle
        setattr(session, attr, not current)
        new = getattr(session, attr)
        return f"OK: {name} {'ENABLED' if new else 'DISABLED'}"
    
    cmd = args[0].lower()
    if cmd in ('on', 'enable', 'true', '1'):
        setattr(session, attr, True)
        return f"OK: {name} ENABLED"
    elif cmd in ('off', 'disable', 'false', '0'):
        setattr(session, attr, False)
        return f"OK: {name} DISABLED"
    else:
        return f"ERROR: Use on/off for {name}"


def _config_subsonic(session: "Session", args: List[str]) -> str:
    """Configure subsonic filter."""
    if not args:
        status = 'ON' if session.hq_subsonic_filter else 'OFF'
        return f"Subsonic filter: {status} @ {session.hq_subsonic_freq:.0f}Hz"
    
    cmd = args[0].lower()
    
    if cmd in ('on', 'enable', 'true', '1'):
        session.hq_subsonic_filter = True
        return "OK: Subsonic filter ENABLED"
    elif cmd in ('off', 'disable', 'false', '0'):
        session.hq_subsonic_filter = False
        return "OK: Subsonic filter DISABLED"
    else:
        # Try to parse as frequency
        try:
            freq = float(cmd)
            if freq < 1 or freq > 100:
                return "ERROR: Subsonic frequency should be 1-100Hz"
            session.hq_subsonic_freq = freq
            session.hq_subsonic_filter = True
            return f"OK: Subsonic filter @ {freq:.0f}Hz"
        except ValueError:
            return f"ERROR: Invalid value '{cmd}'"


def _config_highend(session: "Session", args: List[str]) -> str:
    """Configure high-end smoothing."""
    if not args:
        status = 'ON' if session.hq_highend_smooth else 'OFF'
        return f"High-end smoothing: {status} @ {session.hq_highend_freq:.0f}Hz ({session.hq_highend_reduction:.1f}dB)"
    
    cmd = args[0].lower()
    
    if cmd in ('on', 'enable', 'true', '1'):
        session.hq_highend_smooth = True
        return "OK: High-end smoothing ENABLED"
    elif cmd in ('off', 'disable', 'false', '0'):
        session.hq_highend_smooth = False
        return "OK: High-end smoothing DISABLED"
    else:
        # Try to parse as frequency
        try:
            freq = float(cmd)
            if freq < 1000 or freq > 22000:
                return "ERROR: High-end frequency should be 1000-22000Hz"
            session.hq_highend_freq = freq
            session.hq_highend_smooth = True
            
            # Check for reduction dB
            if len(args) > 1:
                try:
                    reduction = float(args[1])
                    if reduction > 0:
                        reduction = -reduction  # Ensure negative
                    session.hq_highend_reduction = max(-12, min(0, reduction))
                except ValueError:
                    pass
            
            return f"OK: High-end smoothing @ {freq:.0f}Hz ({session.hq_highend_reduction:.1f}dB)"
        except ValueError:
            return f"ERROR: Invalid value '{cmd}'"


def _config_saturation(session: "Session", args: List[str]) -> str:
    """Configure soft saturation."""
    if not args:
        status = 'ON' if session.hq_saturation else 'OFF'
        return f"Soft saturation: {status} (drive={session.hq_saturation_drive:.2f})"
    
    cmd = args[0].lower()
    
    if cmd in ('on', 'enable', 'true', '1'):
        session.hq_saturation = True
        return "OK: Soft saturation ENABLED"
    elif cmd in ('off', 'disable', 'false', '0'):
        session.hq_saturation = False
        return "OK: Soft saturation DISABLED"
    else:
        # Try to parse as drive amount
        try:
            drive = float(cmd)
            if drive < 0 or drive > 1:
                return "ERROR: Saturation drive should be 0-1"
            session.hq_saturation_drive = drive
            session.hq_saturation = True
            return f"OK: Soft saturation drive={drive:.2f}"
        except ValueError:
            return f"ERROR: Invalid value '{cmd}'"


def _config_limiting(session: "Session", args: List[str]) -> str:
    """Configure soft limiting."""
    if not args:
        status = 'ON' if session.hq_limiting else 'OFF'
        return f"Soft limiting: {status} (threshold={session.hq_limit_threshold:.2f})"
    
    cmd = args[0].lower()
    
    if cmd in ('on', 'enable', 'true', '1'):
        session.hq_limiting = True
        return "OK: Soft limiting ENABLED"
    elif cmd in ('off', 'disable', 'false', '0'):
        session.hq_limiting = False
        return "OK: Soft limiting DISABLED"
    else:
        # Try to parse as threshold
        try:
            threshold = float(cmd)
            if threshold < 0.1 or threshold > 1.0:
                return "ERROR: Limiter threshold should be 0.1-1.0"
            session.hq_limit_threshold = threshold
            session.hq_limiting = True
            return f"OK: Soft limiting threshold={threshold:.2f}"
        except ValueError:
            return f"ERROR: Invalid value '{cmd}'"


def _config_format(session: "Session", args: List[str]) -> str:
    """Configure output format."""
    if not args:
        return f"Output format: {session.output_format.upper()}"
    
    fmt = args[0].lower()
    
    if fmt in ('wav', 'wave'):
        session.output_format = 'wav'
        return "OK: Output format set to WAV"
    elif fmt in ('flac',):
        session.output_format = 'flac'
        return "OK: Output format set to FLAC"
    else:
        return f"ERROR: Unknown format '{fmt}'. Use: wav, flac"


def _config_bits(session: "Session", args: List[str]) -> str:
    """Configure bit depth."""
    if not args:
        return f"Bit depth: {session.output_bit_depth}-bit"
    
    try:
        bits = int(args[0])
        if bits == 16:
            session.output_bit_depth = 16
            return "OK: Bit depth set to 16-bit"
        elif bits == 24:
            session.output_bit_depth = 24
            return "OK: Bit depth set to 24-bit"
        elif bits == 32:
            session.output_bit_depth = 32
            return "OK: Bit depth set to 32-bit (float)"
        else:
            return "ERROR: Bit depth must be 16, 24, or 32"
    except ValueError:
        return f"ERROR: Invalid bit depth '{args[0]}'"


def _config_osc(session: "Session", args: List[str]) -> str:
    """Configure oscillator mode."""
    if not args:
        mode = "Band-limited (smooth)" if session.hq_oscillators else "Standard (fast)"
        return f"Oscillator mode: {mode}"
    
    cmd = args[0].lower()
    
    if cmd in ('smooth', 'bandlimited', 'bl', 'hq', 'on'):
        session.hq_oscillators = True
        return "OK: Using band-limited oscillators (smooth)"
    elif cmd in ('fast', 'standard', 'std', 'off'):
        session.hq_oscillators = False
        return "OK: Using standard oscillators (fast)"
    else:
        return f"ERROR: Unknown mode '{cmd}'. Use: smooth, fast"


def apply_hq_chain(session: "Session", audio: np.ndarray) -> np.ndarray:
    """Apply HQ render chain if enabled.
    
    This is a helper function for other commands to use when
    preparing audio for output.
    
    Parameters
    ----------
    session : Session
        The session with HQ settings
    audio : np.ndarray
        Input audio
    
    Returns
    -------
    np.ndarray
        Processed audio (or original if HQ mode disabled)
    """
    if not session.hq_mode:
        return audio
    
    try:
        from ..dsp.effects import hq_render_chain
        return hq_render_chain(
            audio,
            sr=session.sample_rate,
            dc_remove=session.hq_dc_remove,
            subsonic_filter=session.hq_subsonic_filter,
            subsonic_freq=session.hq_subsonic_freq,
            highend_smooth=session.hq_highend_smooth,
            highend_freq=session.hq_highend_freq,
            highend_reduction=session.hq_highend_reduction,
            saturation=session.hq_saturation,
            saturation_drive=session.hq_saturation_drive,
            limiting=session.hq_limiting,
            limit_threshold=session.hq_limit_threshold
        )
    except ImportError:
        return audio


def save_audio_hq(audio: np.ndarray, path: str, sr: int,
                  format: str = 'wav', bit_depth: int = 16) -> str:
    """Save audio with format and bit depth selection.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio data to save
    path : str
        Output file path (extension will be corrected)
    sr : int
        Sample rate
    format : str
        Output format ('wav' or 'flac')
    bit_depth : int
        Bit depth (16, 24, or 32)
    
    Returns
    -------
    str
        Actual output path used
    """
    import os
    from pathlib import Path
    
    # Ensure correct extension
    path_obj = Path(path)
    if format == 'flac':
        path = str(path_obj.with_suffix('.flac'))
    else:
        path = str(path_obj.with_suffix('.wav'))
    
    try:
        import soundfile as sf
        
        # Determine subtype
        subtype_map = {
            ('wav', 16): 'PCM_16',
            ('wav', 24): 'PCM_24',
            ('wav', 32): 'FLOAT',
            ('flac', 16): 'PCM_16',
            ('flac', 24): 'PCM_24',
        }
        
        subtype = subtype_map.get((format, bit_depth), 'PCM_16')
        
        # Ensure audio is in correct range
        audio = np.clip(audio, -1.0, 1.0)
        
        sf.write(path, audio, sr, format=format.upper(), subtype=subtype)
        return path
        
    except ImportError:
        # Fallback to scipy.io.wavfile for WAV only
        if format == 'flac':
            raise ValueError("FLAC output requires soundfile library. Install with: pip install soundfile")
        
        from scipy.io import wavfile
        
        # Convert to int16 for scipy
        if bit_depth == 16:
            audio_int = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        elif bit_depth == 24:
            # scipy doesn't support 24-bit, use 32-bit float
            audio_int = audio.astype(np.float32)
        else:
            audio_int = audio.astype(np.float32)
        
        wavfile.write(path, sr, audio_int)
        return path


# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

def get_hq_commands() -> dict:
    """Return HQ commands for registration."""
    return {
        'hq': cmd_hq,
        'quality': cmd_hq,
    }
