"""Render and playback commands for the MDMA rebuild.

These commands trigger mixing and playback of the last rendered
buffer and toggle the autoplay preview feature.

All renders are automatically backed up to ~/Documents/MDMA/outputs/
Use /USRP to list and reload past renders.
"""

from __future__ import annotations

from typing import List

from ..core.session import Session


def cmd_play(session: Session, args: List[str]) -> str:
    """Play the current audio (in-house playback).

    Mixes all tracks (if any have audio) and plays via the in-house
    audio engine.  Falls back to last_buffer if no tracks.
    """
    try:
        return session.play(0.8)
    except Exception as exc:
        return f"ERROR: {exc}"


def cmd_pa(session: Session, args: List[str]) -> str:
    """Alias for /play."""
    return cmd_play(session, args)


def cmd_aon(session: Session, args: List[str]) -> str:
    session.autoplay = True
    return "OK: autoplay enabled"


def cmd_aoff(session: Session, args: List[str]) -> str:
    session.autoplay = False
    return "OK: autoplay disabled"


def _backup_render(path: str, name: str = None) -> str:
    """Backup rendered file to user outputs folder."""
    try:
        from ..core.user_data import backup_output
        success, result = backup_output(path, name)
        if success:
            return f" [backed up as #{result}]"
    except Exception:
        pass
    return ""


# Full render commands
def cmd_render(session: Session, args: List[str]) -> str:
    """Render all tracks/timeline with clip processing to WAV file.

    Usage:
      /render           -> Render with current settings
      /render dry       -> Render without effects (clips still processed)
      /render master    -> Render with master effects only
    
    All clips are processed with their stretch/pitch settings before mixing.
    The file is written to the current working directory.
    Automatically backed up to ~/Documents/MDMA/outputs/
    """
    try:
        # Parse render mode
        dry_mode = False
        master_only = False
        
        if args:
            arg = args[0].lower()
            if arg == 'dry':
                dry_mode = True
            elif arg == 'master':
                master_only = True
        
        # Store current effects for dry mode
        saved_effects = None
        if dry_mode:
            saved_effects = session.effects.copy()
            session.effects = []
        
        # Mix tracks - this now processes all clips (stretch, effects, etc.)
        session.mix_tracks()
        
        # Restore effects if dry mode
        if saved_effects is not None:
            session.effects = saved_effects
        
        path = session.full_render()
        
        # Backup to user outputs
        backup_info = _backup_render(path, f"render_{dry_mode and 'dry' or 'full'}")
        
        # Report what was rendered
        track_count = len(session.tracks) if session.tracks else 0
        clip_count = 0  # continuous-track workflow
        stretch = getattr(session, 'clip_stretch', 1.0)
        
        info_parts = [f"rendered to {path}"]
        if track_count > 0:
            info_parts.append(f"{track_count} track(s), {clip_count} clip(s)")
        if stretch != 1.0:
            info_parts.append(f"stretch={stretch:.2f}x")
        if dry_mode:
            info_parts.append("(dry)")
        
        return f"OK: {', '.join(info_parts)}{backup_info}"
    except Exception as exc:
        return f"ERROR: {exc}"


def cmd_b(session: Session, args: List[str]) -> str:
    """Render current buffer to WAV file.
    
    Usage:
      /b [path]          Render buffer to file
      
    Automatically backed up to ~/Documents/MDMA/outputs/
    """
    import numpy as np
    
    if session.last_buffer is None or len(session.last_buffer) == 0:
        return "ERROR: No audio in buffer to render"
    
    try:
        # Get metrics before render
        peak = np.max(np.abs(session.last_buffer))
        rms = np.sqrt(np.mean(session.last_buffer ** 2))
        duration = len(session.last_buffer) / session.sample_rate
        
        path = session.full_render()
        backup_info = _backup_render(path, "buffer")
        
        lines = [f"OK: Rendered to {path}{backup_info}"]
        lines.append(f"  {duration:.3f}s, peak={peak:.3f}, rms={rms:.3f}")
        return '\n'.join(lines)
    except Exception as exc:
        return f"ERROR: {exc}"

def cmd_rn(session: Session, args: List[str]) -> str:
    """Alias for /render."""
    return cmd_render(session, args)


def cmd_mix(session: Session, args: List[str]) -> str:
    """Mix all tracks into the buffer without writing to file.
    
    Usage:
      /mix              -> Mix all tracks with processing
      /mix dry          -> Mix without effects
    
    This updates last_buffer with the mixed result.
    """
    try:
        dry_mode = args and args[0].lower() == 'dry'
        
        saved_effects = None
        if dry_mode:
            saved_effects = session.effects.copy()
            session.effects = []
        
        result = session.mix_tracks()
        
        if saved_effects is not None:
            session.effects = saved_effects
        
        if len(result) == 0:
            return "MIX: no audio (no clips in timeline)"
        
        duration_sec = len(result) / session.sample_rate
        track_count = len(session.tracks) if session.tracks else 0
        
        return f"OK: mixed {track_count} track(s), {duration_sec:.2f}s total"
    except Exception as exc:
        return f"ERROR: {exc}"


def cmd_stop_play(session: Session, args: List[str]) -> str:
    """Stop any playing audio (placeholder for future implementation)."""
    return "OK: playback stopped"