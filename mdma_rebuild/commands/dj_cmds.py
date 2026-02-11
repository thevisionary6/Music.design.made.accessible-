"""MDMA DJ Mode Commands.

Live performance and remix environment commands.

Commands are organized into:
- Device Discovery & Management
- Headphone / Cue Bus System
- Deck Control
- Crossfader & Mixing
- VDA Integration
- NVDA Screen Reader Routing
- Safety & Readiness
"""

from __future__ import annotations

import time
import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.session import Session


# Import DJ engine
try:
    from ..dsp.dj_mode import get_dj_engine, is_dj_mode_enabled
    DJ_AVAILABLE = True
except ImportError:
    DJ_AVAILABLE = False


def _ensure_dj():
    """Check DJ mode availability."""
    if not DJ_AVAILABLE:
        return "ERROR: DJ Mode not available"
    return None


def _get_audio_source(session: "Session") -> Tuple[Optional[np.ndarray], str, int]:
    """Get current audio source based on playback context.
    
    Returns:
        (audio, source_type, source_id)
        source_type is 'buffer', 'deck', or 'last'
        Returns (None, 'empty', -1) if no audio available
    """
    # Get playback context
    try:
        from .playback_cmds import get_playback_context
        ctx = get_playback_context()
        context, ctx_id = ctx.get_context()
    except ImportError:
        context, ctx_id = 'buffer', 1
    
    # Try deck first if context is deck
    if context == 'deck':
        try:
            dj = get_dj_engine(session.sample_rate)
            if ctx_id in dj.decks and dj.decks[ctx_id].buffer is not None:
                return (dj.decks[ctx_id].buffer, 'deck', ctx_id)
        except:
            pass
    
    # Try buffer
    if hasattr(session, 'buffers') and ctx_id in session.buffers:
        buf = session.buffers[ctx_id]
        if buf is not None and len(buf) > 0:
            return (buf, 'buffer', ctx_id)
    
    # Try last_buffer
    if hasattr(session, 'last_buffer') and session.last_buffer is not None and len(session.last_buffer) > 0:
        return (session.last_buffer, 'last', 0)
    
    return (None, 'empty', -1)


def _store_audio_result(session: "Session", audio: np.ndarray, source_type: str, source_id: int) -> str:
    """Store processed audio back to its source.
    
    Returns status message.
    """
    if source_type == 'deck':
        try:
            dj = get_dj_engine(session.sample_rate)
            if source_id in dj.decks:
                dj.decks[source_id].buffer = audio
                return f"deck {source_id}"
        except:
            pass
    
    if source_type == 'buffer':
        if not hasattr(session, 'buffers'):
            session.buffers = {}
        session.buffers[source_id] = audio
        session.last_buffer = audio
        return f"buffer {source_id}"
    
    # Default: store to last_buffer
    session.last_buffer = audio
    return "last buffer"


# ============================================================================
# DJ MODE TOGGLE
# ============================================================================

def cmd_djm(session: "Session", args: List[str]) -> str:
    """Toggle or control DJ Mode.
    
    Usage:
      /djm              Toggle DJ mode
      /djm on           Enable DJ mode
      /djm off          Disable DJ mode
      /djm status       Show DJ mode status
      /djm ready        Run readiness check
    
    DJ Mode is additive - it enables DJ features without
    disabling existing MDMA functionality.
    
    Short aliases: /dj
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not args:
        # Toggle
        if dj.enabled:
            return dj.disable()
        else:
            return dj.enable()
    
    sub = args[0].lower()
    
    if sub in ('on', 'enable', '1'):
        return dj.enable()
    
    elif sub in ('off', 'disable', '0'):
        return dj.disable()
    
    elif sub in ('status', 'stat', 's'):
        if not dj.enabled:
            return "DJ Mode: DISABLED\nUse /djm on to enable"
        return dj.deck_status()
    
    elif sub in ('ready', 'check', 'test'):
        if not dj.enabled:
            return "DJ Mode not enabled. Use /djm on first."
        return dj.readiness_check()

    elif sub in ('bs', 'block', 'blocksize', 'buf'):
        # Set DJ output blocksize (latency vs stability)
        if len(args) < 2:
            return f"DJ blocksize: {getattr(dj, 'block_size', 2048)}"
        try:
            bs = int(float(args[1]))
        except ValueError:
            return f"ERROR: invalid blocksize '{args[1]}'"
        # Clamp to reasonable range
        bs = max(256, min(16384, bs))
        dj.block_size = bs
        return f"OK: DJ blocksize set to {bs}. (Restart DJ output to take effect.)"
    
    return f"Unknown djm command: {sub}"


# ============================================================================
# DEVICE DISCOVERY & MANAGEMENT
# ============================================================================

def cmd_do(session: "Session", args: List[str]) -> str:
    """List audio output devices.
    
    Usage:
      /DO              List all output devices
      /DO scan         Rescan for devices
    
    Devices are grouped as:
    - Master Output Devices
    - Headphones / Cue Devices
    - Virtual / Routing Devices
    
    Short aliases: /devices, /dev
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if args and args[0].lower() == 'scan':
        count = dj._scan_devices()
        return f"OK: found {count} audio devices\n\n" + dj.list_devices()
    
    return dj.list_devices()


def cmd_doc(session: "Session", args: List[str]) -> str:
    """Connect master output device.
    
    Usage:
      /DOC <id>        Connect device by ID
    
    Use /DO to list available devices and their IDs.
    
    Short aliases: /master
    """
    err = _ensure_dj()
    if err:
        return err
    
    if not args:
        return "Usage: /DOC <device_id>\nUse /DO to list devices"
    
    dj = get_dj_engine(session.sample_rate)
    return dj.connect_master(args[0])


def cmd_hep(session: "Session", args: List[str]) -> str:
    """Scan and list headphone devices.
    
    Usage:
      /HEP             Scan for headphone devices
    
    Shows Bluetooth, wired, and virtual headphone options.
    
    Short aliases: /headphones, /phones
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    return dj.scan_headphones()


def cmd_hepc(session: "Session", args: List[str]) -> str:
    """Connect headphones / cue device.
    
    Usage:
      /HEPC <id>       Connect headphones by ID
    
    Headphones receive the cue bus, not master output.
    
    Short aliases: /cuehep
    """
    err = _ensure_dj()
    if err:
        return err
    
    if not args:
        return "Usage: /HEPC <device_id>\nUse /HEP to list devices"
    
    dj = get_dj_engine(session.sample_rate)
    return dj.connect_headphones(args[0])


# ============================================================================
# DECK CONTROL
# ============================================================================

def cmd_deck(session: "Session", args: List[str]) -> str:
    """Deck control and status.
    
    Usage:
      /deck                Show all deck status
      /deck <n>            Select deck
      /deck+ [n]           Add deck(s) or set count
      /deck- <n>           Remove deck
      /deck <n> load [buf] Load buffer to deck
      /deck <n> save [buf] Save deck to buffer
      /deck <n> play       Start playback
      /deck <n> pause      Pause playback
      /deck <n> stop       Stop and reset position
      /deck <n> cue        Toggle cue output
      /deck <n> vol <v>    Set volume (0-1)
      /deck <n> tempo <bpm> Set tempo
      /deck <n> pitch <r>  Set pitch ratio
      /deck <n> analyze    Run AI analysis
    
    Short aliases: /dk, /d
    
    Note: Decks work like enhanced buffers. DJ mode enables
    advanced features but basic deck operations work anytime.
    """
    dj = None
    try:
        dj = get_dj_engine(session.sample_rate)
    except:
        return "ERROR: Could not initialize deck system"
    
    # Store session for playback
    dj._bypass_session = session
    
    if not args:
        # Show deck status even if DJ mode not enabled
        return dj.deck_status()
    
    # Deck count commands
    if args[0] == '+' or args[0].startswith('+'):
        if len(args) > 1:
            try:
                count = int(args[1])
                return dj.set_deck_count(count)
            except ValueError:
                return dj.add_deck()
        return dj.add_deck()
    
    if args[0] == '-' or args[0].startswith('-'):
        if len(args) > 1:
            try:
                deck_id = int(args[1])
                return dj.remove_deck(deck_id)
            except ValueError:
                pass
        return "Usage: /deck- <deck_id>"
    
    # Parse deck number
    try:
        deck_id = int(args[0])
    except ValueError:
        return f"ERROR: invalid deck number '{args[0]}'"
    
    # Ensure deck exists
    dj._ensure_deck(deck_id)
    
    if len(args) == 1:
        # Just select deck - update playback context
        dj.active_deck = deck_id
        
        # Update playback context to deck mode
        try:
            from .playback_cmds import get_playback_context
            ctx = get_playback_context()
            ctx.set_context('deck', deck_id)
        except ImportError:
            pass
        
        deck = dj.decks[deck_id]
        dur = f" ({len(deck.buffer)/session.sample_rate:.2f}s)" if deck.buffer is not None else " (empty)"
        return f"OK: Deck {deck_id} selected{dur}"
    
    action = args[1].lower()
    
    if action == 'load':
        buf_idx = int(args[2]) if len(args) > 2 else None
        return dj.load_from_session_buffer(deck_id, session, buf_idx)
    
    elif action == 'save':
        buf_idx = int(args[2]) if len(args) > 2 else None
        return dj.save_to_session_buffer(deck_id, session, buf_idx)
    
    elif action in ('play', 'start'):
        return dj.play_deck(deck_id)
    
    elif action in ('pause', 'stop'):
        return dj.pause_deck(deck_id)
    
    elif action == 'cue':
        enable = None
        if len(args) > 2:
            enable = args[2].lower() in ('on', '1', 'true')
        return dj.cue_deck(deck_id, enable)
    
    elif action in ('vol', 'volume'):
        if len(args) < 3:
            deck = dj.decks.get(deck_id)
            if deck:
                return f"Deck {deck_id} volume: {deck.volume:.0%}"
            return f"ERROR: Deck {deck_id} not found"
        dj.decks[deck_id].volume = float(args[2])
        return f"OK: Deck {deck_id} volume set to {dj.decks[deck_id].volume:.0%}"
    
    elif action == 'tempo':
        if len(args) < 3:
            deck = dj.decks.get(deck_id)
            if deck:
                return f"Deck {deck_id} tempo: {deck.tempo:.1f} BPM"
            return f"ERROR: Deck {deck_id} not found"
        dj.decks[deck_id].tempo = float(args[2])
        return f"OK: Deck {deck_id} tempo set to {dj.decks[deck_id].tempo:.1f} BPM"
    
    elif action == 'pitch':
        if len(args) < 3:
            deck = dj.decks.get(deck_id)
            if deck:
                return f"Deck {deck_id} pitch: {deck.pitch:.2f}x"
            return f"ERROR: Deck {deck_id} not found"
        dj.decks[deck_id].pitch = float(args[2])
        return f"OK: Deck {deck_id} pitch set to {dj.decks[deck_id].pitch:.2f}x"
    
    elif action in ('analyze', 'ana'):
        return dj.analyze_deck(deck_id)
    
    return f"Unknown deck action: {action}"


def cmd_deckplus(session: "Session", args: List[str]) -> str:
    """Add or set deck count.
    
    Usage:
      /deck+             Add one deck
      /deck+ <n>         Set deck count to N
    
    Short aliases: /dk+
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled"
    
    if args:
        try:
            count = int(args[0])
            return dj.set_deck_count(count)
        except ValueError:
            pass
    
    return dj.add_deck()


def cmd_deckminus(session: "Session", args: List[str]) -> str:
    """Remove a deck.
    
    Usage:
      /deck- <n>         Remove deck N
    
    Short aliases: /dk-
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled"
    
    if not args:
        return "Usage: /deck- <deck_id>"
    
    try:
        deck_id = int(args[0])
        return dj.remove_deck(deck_id)
    except ValueError:
        return f"ERROR: invalid deck number '{args[0]}'"


# ============================================================================
# PLAYBACK CONTROL
# ============================================================================

def cmd_play(session: "Session", args: List[str]) -> str:
    """Start deck playback or play buffer if DJ mode off.
    
    Usage:
      /play              Play active deck (DJ mode) or current buffer
      /play <deck>       Play specific deck
      /play all          Play all decks (DJ mode only)
      /play buf          Force buffer playback
    
    Short aliases: /p, /start
    
    When DJ mode is off, this plays from buffers using in-house playback.
    """
    dj = None
    try:
        dj = get_dj_engine(session.sample_rate)
    except:
        pass
    
    # If DJ mode is not enabled, delegate to buffer/playback system
    if dj is None or not dj.enabled:
        try:
            from .playback_cmds import cmd_p as smart_play
            return smart_play(session, args)
        except ImportError:
            # Ultimate fallback - play last buffer directly
            if session.last_buffer is not None and len(session.last_buffer) > 0:
                if session._play_buffer(session.last_buffer, 0.8):
                    dur = len(session.last_buffer) / session.sample_rate
                    return f"PLAYING: last buffer ({dur:.2f}s)"
            return "ERROR: No audio to play"
    
    # Force buffer playback mode
    if args and args[0].lower() == 'buf':
        try:
            from .playback_cmds import cmd_pb
            return cmd_pb(session, args[1:] if len(args) > 1 else [])
        except ImportError:
            pass
    
    # DJ mode is enabled - play deck
    if args and args[0].lower() == 'all':
        count = 0
        for deck_id, deck in dj.decks.items():
            if deck.buffer is not None:
                deck.playing = True
                count += 1
        return f"OK: Playing {count} decks"
    
    if args:
        try:
            deck_id = int(args[0])
        except ValueError:
            return f"ERROR: invalid deck number '{args[0]}'"
    else:
        deck_id = dj.active_deck
    
    if deck_id not in dj.decks:
        return f"ERROR: Deck {deck_id} not found"
    
    deck = dj.decks[deck_id]
    if deck.buffer is None:
        return f"ERROR: Deck {deck_id} has no audio loaded"
    
    # Use bypass playback for reliability
    dj._bypass_session = session
    return dj._play_deck_bypass(deck_id)


def cmd_stop(session: "Session", args: List[str]) -> str:
    """Stop deck playback.
    
    Usage:
      /stop              Stop active deck
      /stop <deck>       Stop specific deck
      /stop all          Stop all decks
    
    Short aliases: /pause
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled."
    
    if args and args[0].lower() == 'all':
        count = 0
        for deck_id, deck in dj.decks.items():
            if deck.playing:
                deck.playing = False
                count += 1
        return f"OK: Stopped {count} decks"
    
    if args:
        try:
            deck_id = int(args[0])
        except ValueError:
            return f"ERROR: invalid deck number '{args[0]}'"
    else:
        deck_id = dj.active_deck
    
    if deck_id not in dj.decks:
        return f"ERROR: Deck {deck_id} not found"
    
    dj.decks[deck_id].playing = False
    return f"OK: Deck {deck_id} stopped"


def cmd_tempo(session: "Session", args: List[str]) -> str:
    """Get or set deck tempo/BPM.
    
    Usage:
      /tempo             Show session BPM (or deck tempo if DJ mode)
      /tempo <bpm>       Set session/deck BPM
      /tempo <deck> <bpm> Set specific deck tempo (DJ mode)
    
    BPM range: 20-300
    
    Short aliases: /bpm, /t
    
    Works without DJ mode - sets session.bpm directly.
    """
    dj = None
    try:
        dj = get_dj_engine(session.sample_rate)
    except:
        pass
    
    # If no args, show current tempo
    if not args:
        if dj is not None and dj.enabled:
            deck_id = dj.active_deck
            if deck_id in dj.decks:
                return f"Deck {deck_id} tempo: {dj.decks[deck_id].tempo:.1f} BPM (session: {session.bpm:.1f})"
        return f"BPM: {session.bpm:.1f}"
    
    # Parse tempo value
    if len(args) == 1:
        try:
            tempo = float(args[0])
        except ValueError:
            return f"ERROR: invalid tempo '{args[0]}'"
        
        # Clamp tempo
        tempo = max(20, min(300, tempo))
        
        # Always set session BPM
        session.bpm = tempo
        
        # Also set deck tempo if DJ mode enabled
        if dj is not None and dj.enabled:
            deck_id = dj.active_deck
            if deck_id in dj.decks:
                dj.decks[deck_id].tempo = tempo
                return f"OK: BPM set to {tempo:.1f} (deck {deck_id} + session)"
        
        return f"OK: BPM set to {tempo:.1f}"
    
    # Two args: deck_id and tempo (requires DJ mode)
    if len(args) >= 2:
        if dj is None or not dj.enabled:
            # Just treat first arg as tempo
            try:
                tempo = float(args[0])
                tempo = max(20, min(300, tempo))
                session.bpm = tempo
                return f"OK: BPM set to {tempo:.1f}"
            except ValueError:
                return "ERROR: invalid tempo"
        
        try:
            deck_id = int(args[0])
            tempo = float(args[1])
        except ValueError:
            return "ERROR: invalid deck or tempo"
        
        if deck_id not in dj.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        # Clamp tempo
        tempo = max(20, min(300, tempo))
        dj.decks[deck_id].tempo = tempo
        
        return f"OK: Deck {deck_id} tempo set to {tempo:.1f} BPM"
    
    return "Usage: /tempo [deck] <bpm>"


def cmd_vol(session: "Session", args: List[str]) -> str:
    """Get or set deck volume.
    
    Usage:
      /vol               Show active deck volume
      /vol <value>       Set active deck volume (0-100)
      /vol <deck> <val>  Set specific deck volume
      /vol master <val>  Set master volume
    
    Volume: 0-100 (100 = full volume)
    
    Short aliases: /volume, /v
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled."
    
    deck_id = dj.active_deck
    
    if not args:
        # Show volume
        if deck_id in dj.decks:
            vol = dj.decks[deck_id].volume * 100
            return f"Deck {deck_id} volume: {vol:.0f}%"
        return "No active deck"
    
    # Check for master volume
    if args[0].lower() == 'master':
        if len(args) > 1:
            try:
                val = float(args[1])
                val = max(0, min(100, val)) / 100.0
                dj.master_volume = val
                return f"OK: Master volume set to {val*100:.0f}%"
            except ValueError:
                return "ERROR: invalid volume"
        return f"Master volume: {dj.master_volume*100:.0f}%"
    
    # Parse args
    if len(args) == 1:
        try:
            vol = float(args[0])
        except ValueError:
            return f"ERROR: invalid volume '{args[0]}'"
    elif len(args) >= 2:
        try:
            deck_id = int(args[0])
            vol = float(args[1])
        except ValueError:
            return "ERROR: invalid deck or volume"
    else:
        return "Usage: /vol [deck] <0-100>"
    
    if deck_id not in dj.decks:
        return f"ERROR: Deck {deck_id} not found"
    
    # Convert 0-100 to 0-1
    vol = max(0, min(100, vol)) / 100.0
    dj.decks[deck_id].volume = vol
    
    return f"OK: Deck {deck_id} volume set to {vol*100:.0f}%"


def cmd_cf(session: "Session", args: List[str]) -> str:
    """Get or set crossfader position.
    
    Usage:
      /cf                Show crossfader position
      /cf <value>        Set crossfader (0-100)
      /cf left           Full left (deck 1)
      /cf right          Full right (deck 2)
      /cf center         Center (50/50 mix)
    
    Position:
      0   = Full deck 1
      50  = Center (equal mix)
      100 = Full deck 2
    
    Short aliases: /crossfader, /xf
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled."
    
    if not args:
        pos = dj.crossfader * 100
        return f"Crossfader: {pos:.0f}% (0=Deck1, 100=Deck2)"
    
    val = args[0].lower()
    
    if val in ('left', 'l', 'a', '1'):
        dj.crossfader = 0.0
        return "OK: Crossfader full left (Deck 1)"
    elif val in ('right', 'r', 'b', '2'):
        dj.crossfader = 1.0
        return "OK: Crossfader full right (Deck 2)"
    elif val in ('center', 'c', 'mid', 'middle'):
        dj.crossfader = 0.5
        return "OK: Crossfader center (50/50)"
    else:
        try:
            pos = float(val)
            pos = max(0, min(100, pos)) / 100.0
            dj.crossfader = pos
            return f"OK: Crossfader set to {pos*100:.0f}%"
        except ValueError:
            return f"ERROR: invalid crossfader value '{val}'"


# ============================================================================
# STEM COMMANDS
# ============================================================================

def cmd_stem(session: "Session", args: List[str]) -> str:
    """Stem separation and control.
    
    Usage:
      /stem sep <deck>           Separate deck into stems
      /stem <deck> <stem> <vol>  Set stem volume
      /stem <deck> solo <stem>   Solo a stem
      /stem <deck> mute <stem>   Mute a stem
      /stem <deck> remix         Remix with current levels
      /stem <deck> status        Show stem levels
    
    Stems: vocals, drums, bass, other (+ piano, guitar for 6-stem)
    
    Examples:
      /stem sep 1                -> Separate deck 1
      /stem 1 vocals 0           -> Mute vocals
      /stem 1 drums 1.5          -> Boost drums 150%
      /stem 1 solo bass          -> Solo bass
    
    Short aliases: /st
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled"
    
    if not args:
        return ("Usage:\n"
                "  /stem sep <deck>        Separate into stems\n"
                "  /stem <deck> <stem> <v> Set stem volume\n"
                "  /stem <deck> solo <stem>\n"
                "  /stem <deck> remix")
    
    # Handle 'sep' command
    if args[0] == 'sep' and len(args) > 1:
        try:
            deck_id = int(args[1])
            model = args[2] if len(args) > 2 else "htdemucs"
            return dj.separate_stems(deck_id, model)
        except ValueError:
            return f"ERROR: invalid deck number '{args[1]}'"
    
    # Parse deck number
    try:
        deck_id = int(args[0])
    except ValueError:
        return f"ERROR: invalid deck number '{args[0]}'"
    
    if len(args) < 2:
        # Show stem status
        deck = dj.decks.get(deck_id)
        if not deck:
            return f"ERROR: Deck {deck_id} not found"
        if not deck.stems:
            return f"Deck {deck_id} has no stems. Use /stem sep {deck_id}"
        
        lines = [f"=== DECK {deck_id} STEMS ===", ""]
        for stem, level in deck.active_stems.items():
            bar = '█' * int(level * 10) + '░' * (10 - int(level * 10))
            lines.append(f"  {stem:8s} {bar} {level:.0%}")
        return '\n'.join(lines)
    
    action = args[1].lower()
    
    if action == 'remix':
        return dj.remix_stems(deck_id)
    
    elif action == 'solo' and len(args) > 2:
        stem = args[2].lower()
        deck = dj.decks.get(deck_id)
        if not deck or not deck.stems:
            return f"ERROR: Deck {deck_id} has no stems"
        
        # Mute all except target
        for s in deck.active_stems:
            deck.active_stems[s] = 1.0 if s == stem else 0.0
        return f"OK: soloed {stem} on Deck {deck_id}"
    
    elif action == 'mute' and len(args) > 2:
        stem = args[2].lower()
        return dj.set_stem_level(deck_id, stem, 0.0)
    
    elif action == 'status':
        deck = dj.decks.get(deck_id)
        if not deck:
            return f"ERROR: Deck {deck_id} not found"
        if not deck.stems:
            return f"Deck {deck_id} has no stems"
        
        lines = [f"=== DECK {deck_id} STEMS ===", ""]
        for stem, level in deck.active_stems.items():
            lines.append(f"  {stem}: {level:.0%}")
        return '\n'.join(lines)
    
    else:
        # Assume: /stem <deck> <stem> <volume>
        stem = args[1].lower()
        if len(args) > 2:
            try:
                level = float(args[2])
                return dj.set_stem_level(deck_id, stem, level)
            except ValueError:
                return f"ERROR: invalid volume '{args[2]}'"
        else:
            deck = dj.decks.get(deck_id)
            if deck and stem in deck.active_stems:
                return f"Deck {deck_id} {stem}: {deck.active_stems[stem]:.0%}"
            return f"ERROR: stem '{stem}' not found"


# ============================================================================
# SECTION AND CHOP COMMANDS
# ============================================================================

def cmd_section(session: "Session", args: List[str]) -> str:
    """Auto-sectioning and section playback.
    
    Usage:
      /section <deck>            Auto-detect sections
      /section <deck> list       List sections
      /section <deck> <idx>      Jump to section
      /section <deck> play <idx> Play from section
    
    Short aliases: /sec
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled"
    
    if not args:
        return "Usage: /section <deck> [list|<idx>]"
    
    try:
        deck_id = int(args[0])
    except ValueError:
        return f"ERROR: invalid deck number '{args[0]}'"
    
    if len(args) < 2:
        # Auto-section
        return dj.auto_section(deck_id)
    
    action = args[1].lower()
    
    if action == 'list':
        deck = dj.decks.get(deck_id)
        if not deck or not deck.sections:
            return f"No sections on Deck {deck_id}. Run /section {deck_id} first."
        
        lines = [f"=== DECK {deck_id} SECTIONS ===", ""]
        for i, sec in enumerate(deck.sections):
            marker = ">" if i == deck.current_section else " "
            lines.append(f"  {marker}[{i}] {sec.start_time:.1f}s - {sec.end_time:.1f}s ({sec.duration:.1f}s)")
        return '\n'.join(lines)
    
    elif action == 'play' and len(args) > 2:
        try:
            idx = int(args[2])
            return dj.play_section(deck_id, idx)
        except ValueError:
            return f"ERROR: invalid section index '{args[2]}'"
    
    else:
        # Jump to section by index
        try:
            idx = int(action)
            return dj.play_section(deck_id, idx)
        except ValueError:
            return f"Unknown section command: {action}"


def cmd_chop(session: "Session", args: List[str]) -> str:
    """Chop deck audio into slices.
    
    Usage:
      /chop <deck>                 Chop by beat (16 slices)
      /chop <deck> <n>             Chop into N slices
      /chop <deck> beat <n>        Beat-aligned chopping
      /chop <deck> transient       Chop at transients
      /chop <deck> time <n>        Equal time divisions
      /chop <deck> list            List chops
      /chop <deck> get <idx>       Load chop to buffer
    
    Short aliases: /ch (if not taken by chunk)
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled"
    
    if not args:
        return "Usage: /chop <deck> [beat|transient|time] [divisions]"
    
    try:
        deck_id = int(args[0])
    except ValueError:
        return f"ERROR: invalid deck number '{args[0]}'"
    
    if len(args) < 2:
        # Default: beat mode, 16 slices
        return dj.chop(deck_id, "beat", 16)
    
    action = args[1].lower()
    
    if action == 'list':
        deck = dj.decks.get(deck_id)
        if not deck or not deck.chops:
            return f"No chops on Deck {deck_id}. Run /chop {deck_id} first."
        
        lines = [f"=== DECK {deck_id} CHOPS ({len(deck.chops)}) ===", ""]
        for i, chop in enumerate(deck.chops[:20]):
            dur = chop.end_time - chop.start_time
            lines.append(f"  [{i:2d}] {chop.start_time:.2f}s - {chop.end_time:.2f}s ({dur:.3f}s)")
        if len(deck.chops) > 20:
            lines.append(f"  ... and {len(deck.chops) - 20} more")
        return '\n'.join(lines)
    
    elif action == 'get' and len(args) > 2:
        try:
            idx = int(args[2])
            audio = dj.get_chop(deck_id, idx)
            if audio is None:
                return f"ERROR: chop {idx} not found"
            session.last_buffer = audio
            dur = len(audio) / session.sample_rate
            return f"OK: loaded chop {idx} ({dur:.3f}s) to buffer"
        except ValueError:
            return f"ERROR: invalid chop index '{args[2]}'"
    
    elif action in ('beat', 'transient', 'time'):
        divisions = int(args[2]) if len(args) > 2 else 16
        return dj.chop(deck_id, action, divisions)
    
    else:
        # Assume it's a number of divisions
        try:
            divisions = int(action)
            return dj.chop(deck_id, "beat", divisions)
        except ValueError:
            return f"Unknown chop mode: {action}"


# ============================================================================
# STREAMING COMMANDS
# ============================================================================

def cmd_stream(session: "Session", args: List[str]) -> str:
    """Stream audio from URL to deck.
    
    Usage:
      /str <query>              Search SoundCloud/YouTube
      /str use <deck> <idx>     Load search result to deck
      /str <deck> <url>         Stream URL directly to deck
      /str cache clear          Clear stream cache
    
    Supported: SoundCloud, YouTube, direct audio URLs
    All audio is pre-processed to engine format (float64 wave).
    
    Examples:
      /str aphex twin           Search for 'aphex twin'
      /str use 1 0              Load result #0 to deck 1
      /str 1 https://soundcloud.com/artist/track
    
    Short aliases: /stream, /sc
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    # Store search results in dj engine
    if not hasattr(dj, '_stream_results'):
        dj._stream_results = []
    
    if not args:
        return ("Usage:\n"
                "  /str <query>           Search SoundCloud/YouTube\n"
                "  /str use <deck> <idx>  Load result to deck\n"
                "  /str <deck> <url>      Stream URL to deck\n"
                f"\nSearch results: {len(dj._stream_results)} cached")
    
    # Load from search results: /str use <deck> <idx>
    if args[0] == 'use' and len(args) >= 3:
        try:
            deck_id = int(args[1])
            idx = int(args[2])
            
            if not dj._stream_results:
                return "No search results. Run /str <query> first."
            
            if idx < 0 or idx >= len(dj._stream_results):
                return f"ERROR: invalid index {idx}. Range: 0-{len(dj._stream_results)-1}"
            
            result = dj._stream_results[idx]
            url = result.get('url', result.get('webpage_url', ''))
            
            if not url:
                return f"ERROR: no URL for result {idx}"
            
            print(f"Loading: {result.get('title', 'Unknown')}...")
            return dj.stream_to_deck(deck_id, url)
            
        except ValueError:
            return "Usage: /str use <deck> <index>"
    
    # Clear cache
    if args[0] == 'cache' and len(args) > 1 and args[1] == 'clear':
        try:
            from ..dsp.streaming import get_library
            lib = get_library(session.sample_rate)
            freed = lib.clear_cache()
            dj._stream_results = []
            return f"OK: cleared stream cache ({freed / 1024 / 1024:.1f} MB freed)"
        except Exception as e:
            return f"ERROR: {e}"
    
    # Check if first arg is a deck number with URL (direct stream)
    try:
        deck_id = int(args[0])
        if len(args) >= 2 and ('://' in args[1] or args[1].startswith('www.')):
            url = args[1]
            return dj.stream_to_deck(deck_id, url)
        else:
            return f"Usage: /str {deck_id} <url> OR /str use {deck_id} <index>"
    except ValueError:
        pass  # Not a deck number, treat as search query
    
    # Search query
    query = ' '.join(args)
    try:
        from ..dsp.streaming import soundcloud_search, youtube_search
        
        print(f"Searching for: {query}...")
        
        # Try SoundCloud first
        results = []
        try:
            sc_results = soundcloud_search(query, limit=8)
            for r in sc_results:
                r['source'] = 'SoundCloud'
            results.extend(sc_results)
        except Exception:
            pass
        
        # Also try YouTube
        try:
            yt_results = youtube_search(query, limit=5)
            for r in yt_results:
                r['source'] = 'YouTube'
            results.extend(yt_results)
        except Exception:
            pass
        
        if not results:
            return f"No results for '{query}'. Make sure yt-dlp is installed."
        
        # Store results
        dj._stream_results = results
        
        lines = [f"=== STREAM SEARCH: {query} ({len(results)} results) ===", ""]
        for i, r in enumerate(results):
            dur = r.get('duration', 0)
            # Convert to int for formatting (duration comes as float from yt-dlp)
            try:
                dur = int(dur) if dur else 0
            except (TypeError, ValueError):
                dur = 0
            dur_str = f"{dur//60}:{dur%60:02d}" if dur else "??:??"
            source = r.get('source', 'Unknown')
            title = r.get('title', 'Unknown')[:50]
            artist = r.get('uploader', r.get('channel', 'Unknown'))[:30]
            
            lines.append(f"  [{i}] {title}")
            lines.append(f"      {artist} ({dur_str}) [{source}]")
        
        lines.append("")
        lines.append("Use: /str use <deck> <index> to load")
        
        return '\n'.join(lines)
        
    except ImportError:
        return "ERROR: streaming requires yt-dlp: pip install yt-dlp"
    except Exception as e:
        return f"ERROR: search failed: {e}"


# ============================================================================
# TRANSITION COMMANDS
# ============================================================================

def cmd_tran(session: "Session", args: List[str]) -> str:
    """Quick transition between decks.
    
    Usage:
      /tran                    Quick crossfade to next deck (1 bar)
      /tran <deck>             Transition to specific deck
      /tran <deck> <style>     Transition with style
      /tran <d> <s> <dur>      Transition with style and duration (seconds)
      /tran styles             List available styles
    
    Default duration is 1 bar (4 beats) at current tempo.
    Uses equal-power crossfade for smooth ducking.
    
    Styles:
      crossfade    Smooth volume crossfade (default)
      cut          Hard cut (instant)
      echo_out     Echo/delay tail
      filter_sweep Low-pass filter sweep
      spinback     Vinyl spinback effect
      brake        Turntable brake/slowdown
      reverb_tail  Reverb wash out
      stutter      Stutter/glitch
      backspin     Quick backspin then cut
      gate         Rhythmic gate pattern
    
    Examples:
      /tran              -> crossfade to next deck (1 bar)
      /tran 2            -> crossfade to deck 2
      /tran 2 spinback   -> spinback to deck 2
      /tran 2 brake 8    -> 8-second brake to deck 2
    
    The outgoing deck automatically stops after transition completes.
    
    Short aliases: /tr, /trans, /x
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled. Use /djm on first."
    
    # List styles
    if args and args[0] == 'styles':
        styles = dj.get_transition_styles()
        lines = ["=== TRANSITION STYLES ===", ""]
        for name, desc in styles.items():
            lines.append(f"  {name:15s} {desc}")
        return '\n'.join(lines)
    
    # Parse arguments
    to_deck = None
    style = "crossfade"
    duration = None  # None = 1 bar at current tempo
    
    if args:
        try:
            to_deck = int(args[0])
        except ValueError:
            # First arg might be style
            style = args[0]
        
        if len(args) > 1:
            style = args[1]
        
        if len(args) > 2:
            try:
                duration = float(args[2])
            except ValueError:
                pass
    
    return dj.quick_transition(to_deck, style, duration)


def cmd_transition(session: "Session", args: List[str]) -> str:
    """Full transition control.
    
    Usage:
      /transition <from> <to> [style] [dur] [curve]
      /transition styles        List styles
      /transition curves        List curves
      /transition status        Show active transition
      /transition cancel        Cancel active transition
    
    Curves:
      linear       Linear fade (default)
      smooth       S-curve (ease in/out)
      exponential  Exponential curve
      logarithmic  Logarithmic curve
    
    Examples:
      /transition 1 2 crossfade 8 smooth
      /transition 1 2 echo_out 4 exponential
    
    Short aliases: /trans
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled"
    
    if not args:
        return ("Usage: /transition <from> <to> [style] [duration] [curve]\n"
                "  /transition styles - List styles\n"
                "  /transition curves - List curves")
    
    if args[0] == 'styles':
        styles = dj.get_transition_styles()
        lines = ["=== TRANSITION STYLES ===", ""]
        for name, desc in styles.items():
            lines.append(f"  {name:15s} {desc}")
        return '\n'.join(lines)
    
    if args[0] == 'curves':
        return ("=== TRANSITION CURVES ===\n"
                "\n"
                "  linear       Linear fade\n"
                "  smooth       S-curve (ease in/out)\n"
                "  exponential  Exponential curve\n"
                "  logarithmic  Logarithmic curve")
    
    if args[0] == 'status':
        if hasattr(dj, '_active_transition') and dj._active_transition:
            t = dj._active_transition
            elapsed = time.time() - t['started_at']
            progress = min(1.0, elapsed / t['duration'])
            return (f"=== ACTIVE TRANSITION ===\n"
                    f"  From: Deck {t['from_deck']}\n"
                    f"  To: Deck {t['to_deck']}\n"
                    f"  Style: {t['style']}\n"
                    f"  Progress: {progress:.0%}")
        return "No active transition"
    
    if args[0] == 'cancel':
        if hasattr(dj, '_active_transition'):
            dj._active_transition = None
        return "OK: transition cancelled"
    
    # Parse: <from> <to> [style] [dur] [curve]
    if len(args) < 2:
        return "Usage: /transition <from> <to> [style] [duration] [curve]"
    
    try:
        from_deck = int(args[0])
        to_deck = int(args[1])
    except ValueError:
        return "ERROR: invalid deck numbers"
    
    style = args[2] if len(args) > 2 else "crossfade"
    
    # Duration: None = use default duration (user override or 1 bar)
    duration = None
    if len(args) > 3:
        try:
            duration = float(args[3])
        except ValueError:
            pass  # Keep None to use default
    
    curve = args[4] if len(args) > 4 else "equal_power"
    
    return dj.create_transition(from_deck, to_deck, duration, style, curve)


def cmd_drop(session: "Session", args: List[str]) -> str:
    """Instant drop/cut transition.
    
    Usage:
      /drop              Drop to next loaded deck
      /drop <deck>       Drop to specific deck
    
    This is an instant cut with a brief filter sweep on the incoming.
    
    Short aliases: /drp, /!
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled"
    
    to_deck = None
    if args:
        try:
            to_deck = int(args[0])
        except ValueError:
            return f"ERROR: invalid deck number '{args[0]}'"
    
    return dj.quick_transition(to_deck, "cut", 0.1)


# ============================================================================
# FILTER COMMANDS
# ============================================================================

def cmd_filter(session: "Session", args: List[str]) -> str:
    """Filter control - FL Studio style sweeps with resonance.
    
    Usage:
      /fl                  Show filter status
      /fl up               Highpass sweep (cut lows), auto-reset
      /fl down             Lowpass sweep (cut highs), auto-reset
      /fl up <amount>      Highpass sweep with amount (1-100)
      /fl down <amount>    Lowpass sweep with amount (1-100)
      /fl reset            Return to neutral immediately
      /fl <deck> up        Sweep on specific deck
      /fl <deck> <value>   Set filter directly (1-100)
      /fl hold             Keep filter at current position (no auto-reset)
    
    Filter values (1-100):
      1    = Full lowpass (cut all highs)
      50   = Neutral (no filter)
      100  = Full highpass (cut all lows)
    
    Sweep auto-resets to neutral (50) after duration completes.
    Use /flr to set resonance for that classic DJ sweep sound.
    
    Examples:
      /fl up           -> Full highpass sweep, auto-reset
      /fl down 80      -> 80% lowpass sweep
      /fl 1 up         -> Highpass on deck 1
      /fl 2 25         -> Set deck 2 to lowpass at 25
      /fl reset        -> Reset all filters to neutral
    
    Short aliases: /flt, /filter
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled. Use /djm on first."
    
    # No args - show status
    if not args:
        lines = ["=== FILTER STATUS ===", ""]
        for deck_id, deck in dj.decks.items():
            cutoff = deck.filter_cutoff
            resonance = getattr(deck, 'filter_resonance', 20)
            if cutoff < 40:
                mode = "LOWPASS"
            elif cutoff > 60:
                mode = "HIGHPASS"
            else:
                mode = "NEUTRAL"
            # Create visual bar (0-100 scale, 50 in middle)
            pos = int(cutoff / 10)
            bar = "░" * pos + "█" + "░" * (10 - pos)
            lines.append(f"  Deck {deck_id}: [{bar}] {cutoff:.0f} ({mode}) R:{resonance:.0f}")
        
        # Show active sweep
        sweep = getattr(dj, '_active_filter_sweep', None)
        if sweep and not sweep.get('completed'):
            auto_str = " (auto-reset)" if sweep.get('auto_reset') else ""
            reset_str = " [RETURNING]" if sweep.get('reset_started') else ""
            lines.append("")
            lines.append(f"  Active: {sweep.get('filter_type')} on Deck {sweep.get('deck_id')}{auto_str}{reset_str}")
        
        return '\n'.join(lines)
    
    # Parse arguments
    deck_id = dj.active_deck
    direction = None
    amount = None
    direct_value = None
    auto_reset = True
    
    arg_idx = 0
    
    # Check if first arg is a deck number
    if args[arg_idx].isdigit() and int(args[arg_idx]) <= 4:
        deck_id = int(args[arg_idx])
        arg_idx += 1
    
    if arg_idx < len(args):
        arg = args[arg_idx].lower()
        
        if arg in ('up', 'hp', 'highpass', 'high'):
            direction = 'up'
        elif arg in ('down', 'lp', 'lowpass', 'low'):
            direction = 'down'
        elif arg in ('reset', 'off', 'neutral', 'clear'):
            direction = 'reset'
        elif arg in ('hold', 'stay', 'keep'):
            # Set current sweep to not auto-reset
            sweep = getattr(dj, '_active_filter_sweep', None)
            if sweep and not sweep.get('completed'):
                sweep['auto_reset'] = False
                return f"OK: Filter will hold at current position"
            else:
                return "No active sweep to hold"
        else:
            # Try to parse as direct value (1-100)
            try:
                direct_value = float(arg)
            except ValueError:
                return f"ERROR: unknown filter command '{arg}'. Use up/down/reset or 1-100 value."
        
        arg_idx += 1
    
    # Check for amount/duration argument
    if arg_idx < len(args):
        try:
            val = float(args[arg_idx])
            if direction and val >= 1 and val <= 100:
                amount = val
            elif direction:
                # Could be duration in seconds (small number)
                pass  # Will use default duration
        except ValueError:
            # Check for 'hold' modifier
            if args[arg_idx].lower() in ('hold', 'stay', 'keep'):
                auto_reset = False
    
    # Execute command
    if direct_value is not None:
        return dj.set_filter(deck_id, direct_value)
    elif direction:
        return dj.filter_sweep(deck_id, direction, amount, auto_reset=auto_reset)
    else:
        return "Usage: /fl up|down|reset [amount] or /fl <value>"


def cmd_resonance(session: "Session", args: List[str]) -> str:
    """Set filter resonance (Q factor) for DJ filter sweeps.
    
    Usage:
      /flr              Show current resonance
      /flr <value>      Set resonance (1-100)
      /flr <deck> <v>   Set resonance for specific deck
    
    Resonance values (1-100):
      1-30   = Subtle (clean filter)
      30-60  = Moderate (noticeable peak)
      60-80  = Aggressive (classic DJ sound)
      80-100 = Extreme (screaming resonance)
    
    Higher resonance creates that classic DJ filter sweep sound
    with a sharp peak at the cutoff frequency.
    
    Examples:
      /flr 70          -> Aggressive resonance
      /flr 1 50        -> Moderate resonance on deck 1
      /flr 100         -> Maximum resonance (extreme!)
    
    Short aliases: /res, /q
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled. Use /djm on first."
    
    # No args - show status
    if not args:
        lines = ["=== FILTER RESONANCE ===", ""]
        for deck_id, deck in dj.decks.items():
            res = getattr(deck, 'filter_resonance', 20)
            if res < 30:
                style = "subtle"
            elif res < 60:
                style = "moderate"
            elif res < 80:
                style = "aggressive"
            else:
                style = "EXTREME"
            bar = "█" * int(res / 10) + "░" * (10 - int(res / 10))
            lines.append(f"  Deck {deck_id}: [{bar}] {res:.0f} ({style})")
        return '\n'.join(lines)
    
    # Parse arguments
    deck_id = dj.active_deck
    value = None
    
    if len(args) == 1:
        try:
            value = float(args[0])
        except ValueError:
            return f"ERROR: invalid resonance value '{args[0]}'"
    elif len(args) >= 2:
        try:
            deck_id = int(args[0])
            value = float(args[1])
        except ValueError:
            return "ERROR: invalid deck or resonance value"
    
    if value is None:
        return "Usage: /flr <value> or /flr <deck> <value>"
    
    return dj.set_resonance(deck_id, value)


def cmd_duration(session: "Session", args: List[str]) -> str:
    """Set default duration for DJ commands.
    
    Usage:
      /d                   Show current duration setting
      /d <seconds>         Set default duration in seconds
      /d auto              Use automatic (1 bar at current tempo)
      /d 1bar              Set to 1 bar (same as auto)
      /d 2bar              Set to 2 bars
      /d 4bar              Set to 4 bars
    
    The duration affects:
      - Transitions (/tran, /transition)
      - Filter sweeps (/fl up, /fl down)
      - Any other timed DJ commands
    
    Examples:
      /d 4           -> All transitions/sweeps last 4 seconds
      /d auto        -> Calculate from tempo (1 bar)
      /d 2bar        -> 2 bars at current tempo
    
    Short aliases: /dur, /time
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled. Use /djm on first."
    
    # No args - show current
    if not args:
        if dj.default_duration is None:
            eff = dj.get_effective_duration()
            return f"Duration: AUTO (currently {eff:.2f}s = 1 bar)"
        else:
            return f"Duration: {dj.default_duration:.2f}s (fixed)"
    
    arg = args[0].lower()
    
    # Auto/reset
    if arg in ('auto', 'reset', 'default', '1bar'):
        return dj.set_duration(None)
    
    # Bar-based
    if arg.endswith('bar') or arg.endswith('bars'):
        try:
            bars = int(arg.replace('bars', '').replace('bar', ''))
            # Calculate duration for N bars at current tempo
            eff = dj.get_effective_duration()  # This is 1 bar
            duration = eff * bars
            result = dj.set_duration(duration)
            return result + f" ({bars} bar{'s' if bars > 1 else ''} at current tempo)"
        except ValueError:
            return f"ERROR: invalid bar count '{arg}'"
    
    # Numeric
    try:
        duration = float(arg)
        return dj.set_duration(duration)
    except ValueError:
        return f"ERROR: invalid duration '{arg}'. Use seconds or 'auto'."


# ============================================================================
# LOOP AND STUTTER COMMANDS
# ============================================================================

def cmd_lpc(session: "Session", args: List[str]) -> str:
    """Set loop count for counted loops.
    
    Usage:
      /lpc              Show current loop count
      /lpc <count>      Set loop count (1-64)
      /lpc <deck> <n>   Set loop count for specific deck
    
    The loop count determines how many times /lpg will repeat.
    
    Examples:
      /lpc 8           -> Set loop count to 8 for active deck
      /lpc 1 4         -> Set deck 1 loop count to 4
    
    Short aliases: /loopcount
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled. Use /djm on first."
    
    # No args - show current
    if not args:
        lines = ["=== LOOP COUNT ===", ""]
        for deck_id, deck in dj.decks.items():
            count = getattr(deck, 'loop_count', 4)
            remaining = getattr(deck, 'loop_remaining', 0)
            active = "▶" if deck.loop_active else " "
            lines.append(f"  {active} Deck {deck_id}: {count} loops" + 
                        (f" ({remaining} remaining)" if remaining > 0 else ""))
        return '\n'.join(lines)
    
    # Parse arguments
    deck_id = dj.active_deck
    count = None
    
    if len(args) == 1:
        try:
            count = int(args[0])
        except ValueError:
            return f"ERROR: invalid count '{args[0]}'"
    elif len(args) >= 2:
        try:
            deck_id = int(args[0])
            count = int(args[1])
        except ValueError:
            return "ERROR: invalid deck or count"
    
    if count is None:
        return "Usage: /lpc <count> or /lpc <deck> <count>"
    
    return dj.set_loop_count(deck_id, count)


def cmd_lpg(session: "Session", args: List[str]) -> str:
    """Trigger counted loop at current position.
    
    Usage:
      /lpg              Trigger 1-bar loop for loop_count times
      /lpg <beats>      Trigger loop of N beats
      /lpg <deck>       Trigger on specific deck
      /lpg <deck> <b>   Trigger N-beat loop on specific deck
      /lpg off          Release/exit loop
    
    Uses the loop count set by /lpc to determine repetitions.
    
    Examples:
      /lpg              -> Loop 4 beats (1 bar), repeat loop_count times
      /lpg 2            -> Loop 2 beats
      /lpg 1 8          -> Loop 8 beats on deck 1
      /lpg off          -> Release loop
    
    Short aliases: /loopgo, /lg
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled. Use /djm on first."
    
    # Check for off/release
    if args and args[0].lower() in ('off', 'release', 'exit', 'stop'):
        deck_id = dj.active_deck
        if len(args) > 1:
            try:
                deck_id = int(args[1])
            except ValueError:
                pass
        return dj.release_loop(deck_id)
    
    # Parse arguments
    deck_id = dj.active_deck
    beats = 4.0  # Default 1 bar
    
    if len(args) == 1:
        try:
            # Could be deck number or beat count
            val = float(args[0])
            if val <= 4 and val == int(val) and int(val) in dj.decks:
                # Looks like a deck number
                deck_id = int(val)
            else:
                beats = val
        except ValueError:
            return f"ERROR: invalid argument '{args[0]}'"
    elif len(args) >= 2:
        try:
            deck_id = int(args[0])
            beats = float(args[1])
        except ValueError:
            return "ERROR: invalid deck or beats"
    
    return dj.trigger_loop(deck_id, beats)


def cmd_lh(session: "Session", args: List[str]) -> str:
    """Toggle loop-hold (LH) for a deck.

    Loop-hold is an infinite loop over a window starting at the current playhead.
    This is additive: it does not replace /lpc or /lg; it adds a fast "hold" option.

    Usage:
      /lh            Toggle hold on active deck (default 4 beats)
      /lh <beats>    Toggle hold with a custom length in beats
      /lh <deck> <beats>  Specific deck
    """
    err = _ensure_dj()
    if err:
        return err
    dj = get_dj_engine(session.sample_rate)
    if not dj.enabled:
        return "DJ Mode not enabled. Use /djm on first."

    deck_id = dj.active_deck
    beats = None
    if len(args) == 1:
        # Could be beats or deck
        try:
            val = float(args[0])
            if val.is_integer() and int(val) in dj.decks:
                deck_id = int(val)
            else:
                beats = val
        except ValueError:
            return f"ERROR: invalid argument '{args[0]}'"
    elif len(args) >= 2:
        try:
            deck_id = int(args[0])
            beats = float(args[1])
        except ValueError:
            return "ERROR: invalid deck or beat length"

    return dj.toggle_loop_hold(deck_id, beats)


def cmd_ml(session: "Session", args: List[str]) -> str:
    """Master loop wrap for DJ decks.

    When ON, any deck that reaches the end of its source wraps to the beginning
    and continues playing. This is intended as a performance/safety feature:
    it keeps audio running if content runs out.

    Usage:
      /ml           Show status
      /ml on        Enable master wrap
      /ml off       Disable master wrap
    """
    err = _ensure_dj()
    if err:
        return err
    dj = get_dj_engine(session.sample_rate)
    if not dj.enabled:
        return "DJ Mode not enabled. Use /djm on first."

    if not args:
        return f"Master loop wrap: {'ON' if getattr(dj, 'master_loop_wrap', False) else 'OFF'}"
    sub = args[0].lower()
    if sub in ('on', '1', 'enable', 'true'):
        return dj.set_master_loop_wrap(True)
    if sub in ('off', '0', 'disable', 'false'):
        return dj.set_master_loop_wrap(False)
    return "Usage: /ml [on|off]"


def cmd_mdfx(session: "Session", args: List[str]) -> str:
    """Master Deck FX (MDFX): always-on post chain for the DJ master bus.

    Usage:
      /mdfx                    Show chain
      /mdfx add <effect> [amount=50]
      /mdfx rm <n>
      /mdfx clear
      /mdfx <n> <param>=<val>  Update params (e.g., amount)
    """
    err = _ensure_dj()
    if err:
        return err
    dj = get_dj_engine(session.sample_rate)
    if not dj.enabled:
        return "DJ Mode not enabled. Use /djm on first."

    chain = getattr(dj, 'master_deck_fx_chain', [])
    if not args:
        if not chain:
            return "MDFX chain: (empty)"
        lines = [f"MDFX chain ({len(chain)} effects):"]
        for i, (fx, params) in enumerate(chain, 1):
            p = ', '.join(f"{k}={v}" for k, v in (params or {}).items())
            lines.append(f"  {i}. {fx}" + (f" ({p})" if p else ""))
        return '\n'.join(lines)

    sub = args[0].lower()
    if sub == 'add' and len(args) > 1:
        fx_name = args[1].lower()
        # Resolve through DJ effect aliases for convenience
        try:
            fx_name = dj._resolve_effect_name(fx_name)
        except Exception:
            pass
        params = {}
        for tok in args[2:]:
            if '=' in tok:
                k, v = tok.split('=', 1)
                try:
                    params[k] = float(v)
                except ValueError:
                    params[k] = v
        chain.append((fx_name, params))
        dj.master_deck_fx_chain = chain
        return f"OK: added '{fx_name}' to MDFX (position {len(chain)})"

    if sub == 'rm' and len(args) > 1:
        try:
            idx = int(args[1]) - 1
        except ValueError:
            return "ERROR: position must be a number"
        if 0 <= idx < len(chain):
            removed = chain.pop(idx)
            dj.master_deck_fx_chain = chain
            return f"OK: removed '{removed[0]}'"
        return f"ERROR: invalid position {args[1]}"

    if sub == 'clear':
        count = len(chain)
        chain.clear()
        dj.master_deck_fx_chain = chain
        return f"OK: cleared {count} MDFX effects"

    # Update params at position
    try:
        idx = int(sub) - 1
        if 0 <= idx < len(chain):
            fx_name, params = chain[idx]
            for tok in args[1:]:
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    try:
                        params[k] = float(v)
                    except ValueError:
                        params[k] = v
            chain[idx] = (fx_name, params)
            dj.master_deck_fx_chain = chain
            return f"OK: updated '{fx_name}' params: {params}"
    except ValueError:
        pass

    return "ERROR: unknown subcommand. Use: add, rm, clear"


def cmd_stud(session: "Session", args: List[str]) -> str:
    """Trigger stutter effect using loop count.
    
    Usage:
      /stud             Stutter 1 beat, repeat loop_count times
      /stud <beats>     Stutter N beats
      /stud <deck>      Stutter on specific deck
      /stud <d> <b>     Stutter N beats on specific deck
      /stud off         Stop stutter
    
    Creates a rapid repeat of a small audio chunk.
    Duration is set to 1 beat by default.
    Number of repeats comes from loop count (/lpc).
    
    Examples:
      /stud             -> Stutter 1 beat, loop_count times
      /stud 0.5         -> Stutter half a beat
      /stud 1 0.25      -> Stutter quarter beat on deck 1
      /stud off         -> Stop stutter
    
    Short aliases: /stutter, /st
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled. Use /djm on first."
    
    # Check for off/stop
    if args and args[0].lower() in ('off', 'stop', 'release', 'exit'):
        deck_id = dj.active_deck
        if len(args) > 1:
            try:
                deck_id = int(args[1])
            except ValueError:
                pass
        return dj.stop_stutter(deck_id)
    
    # Parse arguments
    deck_id = dj.active_deck
    beats = 1.0  # Default 1 beat
    
    if len(args) == 1:
        try:
            # Could be deck number or beat count
            val = float(args[0])
            if val <= 4 and val == int(val) and int(val) in dj.decks:
                # Looks like a deck number
                deck_id = int(val)
            else:
                beats = val
        except ValueError:
            return f"ERROR: invalid argument '{args[0]}'"
    elif len(args) >= 2:
        try:
            deck_id = int(args[0])
            beats = float(args[1])
        except ValueError:
            return "ERROR: invalid deck or beats"
    
    return dj.trigger_stutter(deck_id, beats)


# ============================================================================
# JUMP COMMAND
# ============================================================================

def cmd_jump(session: "Session", args: List[str]) -> str:
    """Jump to a point in the track.
    
    Usage:
      /j <target>        Jump to named position
      /j <beat>          Jump to beat number
      /j <deck> <target> Jump on specific deck
      /j <time>s         Jump to time in seconds
    
    Named targets:
      start, end, half, quarter, cue
      drop, breakdown, buildup, chorus, verse, outro, intro
    
    AI-detected sections work if track is analyzed.
    Otherwise uses estimated positions.
    
    Examples:
      /j start         -> Jump to start
      /j drop          -> Jump to drop (estimated or detected)
      /j 32            -> Jump to beat 32
      /j 1 chorus      -> Jump deck 1 to chorus
      /j 45s           -> Jump to 45 seconds
    
    Short aliases: /jump
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled. Use /djm on first."
    
    if not args:
        return ("Usage: /j <target> | /j <beat> | /j <time>s\n"
                "Targets: start, end, drop, breakdown, chorus, verse, cue")
    
    # Parse arguments
    deck_id = dj.active_deck
    target = None
    beat = None
    time_sec = None
    
    arg_idx = 0
    
    # Check if first arg is deck number
    if args[0].isdigit() and int(args[0]) <= 4 and int(args[0]) in dj.decks:
        deck_id = int(args[0])
        arg_idx = 1
    
    if arg_idx < len(args):
        arg = args[arg_idx]
        
        # Check for time in seconds (e.g., "45s" or "2.5s")
        if arg.endswith('s'):
            try:
                time_sec = float(arg[:-1])
            except ValueError:
                pass
        
        # Check for beat number
        if time_sec is None and arg.isdigit():
            beat = int(arg)
        
        # Named target
        if time_sec is None and beat is None:
            target = arg
    
    return dj.jump_to(deck_id, target=target, beat=beat, time_sec=time_sec)


# ============================================================================
# SCRATCH COMMAND
# ============================================================================

def cmd_scratch(session: "Session", args: List[str]) -> str:
    """Trigger deterministic scratch pattern.
    
    Usage:
      /scr              Scratch with preset 1 (baby)
      /scr <preset>     Use scratch preset (1-5)
      /scr <deck> <p>   Scratch specific deck
      /scr <preset> <dur>  Scratch with custom duration
    
    Presets:
      1 = Baby scratch (back-forth)
      2 = Forward scratch (push-return)
      3 = Chirp scratch (clipped)
      4 = Transformer (gated)
      5 = Crab scratch (rapid)
    
    Duration uses loop_count × beat duration by default.
    
    Examples:
      /scr              -> Baby scratch, default duration
      /scr 3            -> Chirp scratch
      /scr 1 2          -> Baby scratch on deck 1
      /scr 5 1.0        -> Crab scratch, 1 second
    
    Short aliases: /scratch
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled. Use /djm on first."
    
    # Parse arguments
    deck_id = dj.active_deck
    preset = 1
    duration = None
    
    if len(args) >= 1:
        try:
            val = int(args[0])
            if val <= 5:
                preset = val
            elif val in dj.decks:
                deck_id = val
                if len(args) >= 2:
                    preset = int(args[1])
        except ValueError:
            # Could be duration
            try:
                duration = float(args[0])
            except ValueError:
                return f"ERROR: invalid argument '{args[0]}'"
    
    if len(args) >= 2 and duration is None:
        try:
            # Second arg could be deck (if first was preset) or duration
            val = float(args[1])
            if val <= 8 and val == int(val) and int(val) in dj.decks:
                deck_id = int(val)
            else:
                duration = val
        except ValueError:
            pass
    
    if len(args) >= 3:
        try:
            duration = float(args[2])
        except ValueError:
            pass
    
    return dj.trigger_scratch(deck_id, preset, duration)


# ============================================================================
# DECK EFFECTS COMMAND
# ============================================================================

def cmd_dfx(session: "Session", args: List[str]) -> str:
    """Apply timed effect to deck from main effect modules.
    
    Usage:
      /dfx <effect>             Effect on active deck (1 bar)
      /dfx <effect> <dur>       Effect with custom duration
      /dfx <effect> <amt>       Effect with amount (1-100)
      /dfx <deck> <effect>      Effect on specific deck
      /dfx <deck> <fx> <dur>    Full specification
      /dfx list                 List all available effects
      /dfx list <category>      List effects by category
    
    Built-in (low-latency):
      echo, flanger, phaser, crush, filter
    
    Vamp/Overdrive (aliases: v1-v4, vamp, amp):
      vamp_light, vamp_medium, vamp_heavy, vamp_fuzz
      overdrive_soft, overdrive_classic, overdrive_crunch
    
    Reverb (aliases: r1-r5, reverb):
      reverb_small, reverb_large, reverb_plate, reverb_spring
      conv_hall, conv_shimmer, shimmer
    
    Delay (aliases: d1-d5, delay):
      delay_simple, delay_pingpong, delay_tape, delay_slapback
    
    Saturation (aliases: s1-s5, sat):
      saturate_soft, saturate_hard, saturate_tube, saturate_fuzz
    
    Lo-fi (aliases: l1-l6, lofi):
      lofi_bitcrush, lofi_chorus, lofi_flanger, lofi_phaser
    
    Dynamics (aliases: c1-c3, comp):
      compress_mild, compress_hard, compress_limiter
      gate1-gate5
    
    Examples:
      /dfx vamp           -> Vamp overdrive on active deck
      /dfx r2 4           -> Large reverb for 4 seconds
      /dfx 1 shimmer      -> Shimmer reverb on deck 1
      /dfx delay_tape 70  -> Tape delay at 70% wet
      /dfx 2 v3 2         -> Heavy vamp on deck 2, 2 seconds
    
    Short aliases: /deckfx
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled. Use /djm on first."
    
    if not args:
        return ("Usage: /dfx <effect> [duration]\n"
                "Quick: echo, vamp, reverb, delay, crush, filter\n"
                "Use /dfx list for all effects")
    
    if args[0].lower() == 'list':
        category = args[1].lower() if len(args) > 1 else None
        
        effects_by_category = {
            'builtin': {
                'echo': 'Echo/delay trail',
                'flanger': 'Flanger sweep',
                'phaser': 'Phaser modulation',
                'crush': 'Bitcrush degradation',
                'filter': 'Filter sweep',
            },
            'vamp': {
                'vamp (v2)': 'Medium amp overdrive',
                'vamp_light (v1)': 'Light amp warmth',
                'vamp_heavy (v3)': 'Heavy amp distortion',
                'vamp_fuzz (v4)': 'Fuzz pedal distortion',
                'overdrive (od)': 'Classic overdrive',
                'overdrive_crunch (o3)': 'Crunchy overdrive',
            },
            'reverb': {
                'reverb (r2)': 'Large hall reverb',
                'reverb_small (r1)': 'Small room reverb',
                'reverb_plate (r3)': 'Plate reverb',
                'reverb_spring (r4)': 'Spring reverb',
                'shimmer': 'Shimmer reverb',
                'conv_hall': 'Convolution hall',
            },
            'delay': {
                'delay (d1)': 'Simple delay',
                'delay_pingpong (d2)': 'Ping-pong stereo',
                'delay_tape (d5)': 'Tape echo with wobble',
                'delay_slapback (d4)': 'Short slapback',
            },
            'saturation': {
                'sat (s1)': 'Soft saturation',
                'saturate_hard (s2)': 'Hard clipping',
                'saturate_tube (s5)': 'Tube saturation',
                'saturate_fuzz (s4)': 'Fuzz distortion',
            },
            'lofi': {
                'lofi (l1)': 'Bit crusher',
                'lofi_chorus (l2)': 'Chorus effect',
                'lofi_flanger (l3)': 'Flanger effect',
                'lofi_phaser (l4)': 'Phaser effect',
                'lofi_halftime (l6)': 'Half-speed effect',
            },
            'dynamics': {
                'comp (c1)': 'Mild compressor',
                'compress_hard (c2)': 'Hard compressor',
                'compress_limiter (c3)': 'Brick wall limiter',
                'gate (g3)': 'Medium gate',
            },
        }
        
        if category and category in effects_by_category:
            lines = [f"=== {category.upper()} EFFECTS ===", ""]
            for fx, desc in effects_by_category[category].items():
                lines.append(f"  {fx:24s} - {desc}")
            return '\n'.join(lines)
        
        lines = ["=== DECK EFFECTS ===", ""]
        for cat_name, effects in effects_by_category.items():
            lines.append(f"--- {cat_name.upper()} ---")
            for fx, desc in list(effects.items())[:3]:  # Show first 3
                lines.append(f"  {fx:24s} - {desc}")
            if len(effects) > 3:
                lines.append(f"  ... and {len(effects) - 3} more (use /dfx list {cat_name})")
            lines.append("")
        
        lines.append("Shortcuts: v1-v4, r1-r5, d1-d5, s1-s5, l1-l6, c1-c3, g1-g5")
        return '\n'.join(lines)
    
    # Parse arguments
    deck_id = dj.active_deck
    effect = None
    duration = None
    amount = 50.0
    
    remaining_args = list(args)
    
    # Check if first arg is deck number
    if remaining_args[0].isdigit() and int(remaining_args[0]) <= 4:
        potential_deck = int(remaining_args[0])
        if potential_deck in dj.decks:
            deck_id = potential_deck
            remaining_args = remaining_args[1:]
    
    # Get effect name
    if remaining_args:
        effect = remaining_args[0].lower()
        remaining_args = remaining_args[1:]
    
    # Parse remaining args (could be duration or amount)
    for arg in remaining_args:
        try:
            val = float(arg)
            if val > 10:  # Likely amount (1-100)
                amount = val
            else:  # Likely duration (seconds)
                duration = val
        except ValueError:
            pass
    
    if effect is None:
        return "ERROR: specify effect name"
    
    return dj.apply_deck_effect(deck_id, effect, duration, amount)


def cmd_cue(session: "Session", args: List[str]) -> str:
    """Cue control.
    
    Usage:
      /cue              Show cue status
      /cue <deck>       Toggle cue for deck
      /cue vol <v>      Set cue volume
      /cue mix <v>      Set cue/master mix (0=cue, 1=master)
    
    Cue bus routes to headphones for pre-listening.
    
    Short aliases: /c
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled"
    
    if not args:
        cued = [d.id for d in dj.decks.values() if d.cue_enabled]
        return (f"Cue Status:\n"
                f"  Decks in cue: {cued or 'none'}\n"
                f"  Cue volume: {dj.cue_volume:.0%}\n"
                f"  Cue/Master mix: {dj.cue_mix:.0%}")
    
    if args[0].lower() == 'vol' and len(args) > 1:
        dj.cue_volume = float(args[1])
        return f"OK: Cue volume set to {dj.cue_volume:.0%}"
    
    if args[0].lower() == 'mix' and len(args) > 1:
        dj.cue_mix = float(args[1])
        return f"OK: Cue/Master mix set to {dj.cue_mix:.0%}"
    
    # Toggle deck cue
    try:
        deck_id = int(args[0])
        return dj.cue_deck(deck_id)
    except ValueError:
        return f"ERROR: invalid deck number '{args[0]}'"


def cmd_xfade(session: "Session", args: List[str]) -> str:
    """Crossfader control.
    
    Usage:
      /xfade            Show crossfader position
      /xfade <v>        Set position (0-1, 0=deck1, 1=deck2)
      /xfade left       Full left (deck 1)
      /xfade right      Full right (deck 2)
      /xfade center     Center (50/50 mix)
      /xfade curve <t>  Set curve type (linear, smooth, cut)
    
    Short aliases: /xf, /cross
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled"
    
    if not args:
        return (f"Crossfader: {dj.crossfader:.0%}\n"
                f"  Curve: {dj.crossfader_curve}\n"
                f"  {'█' * int(dj.crossfader * 20)}{'░' * (20 - int(dj.crossfader * 20))}")
    
    arg = args[0].lower()
    
    if arg in ('left', 'l', 'a', '1'):
        return dj.set_crossfader(0.0)
    elif arg in ('right', 'r', 'b', '2'):
        return dj.set_crossfader(1.0)
    elif arg in ('center', 'c', 'mid', 'middle'):
        return dj.set_crossfader(0.5)
    elif arg == 'curve' and len(args) > 1:
        dj.crossfader_curve = args[1].lower()
        return f"OK: Crossfader curve set to {dj.crossfader_curve}"
    else:
        try:
            return dj.set_crossfader(float(arg))
        except ValueError:
            return f"ERROR: invalid crossfader value '{arg}'"


def cmd_sync(session: "Session", args: List[str]) -> str:
    """Sync deck tempos.
    
    Usage:
      /sync             Sync all decks to master tempo
      /sync <d1> <d2>   Sync deck d1 to deck d2
      /sync auto        Enable auto-sync
      /sync off         Disable auto-sync
    
    Short aliases: /sy
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not dj.enabled:
        return "DJ Mode not enabled"
    
    if not args:
        # Sync all to active deck
        master_tempo = dj.decks[dj.active_deck].tempo
        for deck in dj.decks.values():
            deck.tempo = master_tempo
        return f"OK: All decks synced to {master_tempo:.1f} BPM"
    
    if len(args) >= 2:
        try:
            d1, d2 = int(args[0]), int(args[1])
            dj.decks[d1].tempo = dj.decks[d2].tempo
            return f"OK: Deck {d1} synced to Deck {d2} ({dj.decks[d2].tempo:.1f} BPM)"
        except (ValueError, KeyError):
            return "ERROR: invalid deck numbers"
    
    return "Usage: /sync or /sync <deck1> <deck2>"


# ============================================================================
# VDA INTEGRATION
# ============================================================================

def cmd_vda(session: "Session", args: List[str]) -> str:
    """Virtual Audio Device integration.
    
    Usage:
      /VDA              List VDA routes
      /VDA list         List VDA routes
      /VDA route <s> <d> Route source to destination
      /VDA isolate cue  Isolate cue bus to VDA
      /VDA isolate master Isolate master bus to VDA
      /VDA clear        Clear all VDA routes
    
    VDA allows routing to streaming software, separating
    monitoring channels, or isolating analysis.
    
    Short aliases: /virtual
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not args or args[0].lower() == 'list':
        return dj.vda_list()
    
    sub = args[0].lower()
    
    if sub == 'route' and len(args) >= 3:
        return dj.vda_route(args[1], args[2])
    
    elif sub == 'isolate' and len(args) >= 2:
        return dj.vda_isolate(args[1])
    
    elif sub == 'clear':
        dj.vda_routes.clear()
        return "OK: VDA routes cleared"
    
    return f"Unknown VDA command: {sub}"


# ============================================================================
# NVDA SCREEN READER ROUTING
# ============================================================================

def cmd_sr(session: "Session", args: List[str]) -> str:
    """Screen reader (NVDA) audio routing.
    
    Usage:
      /SR              Show screen reader routing menu
      /SR HEP          Route NVDA to headphones
      /SR OUT          Route NVDA to master output
      /SR KEEP         Leave NVDA routing unchanged
      /SR TEST         Test current routing
    
    Use Cases:
    - Studio: DJ output → PA, NVDA → monitors
    - Party: DJ output → speakers, NVDA → headphones
    
    NVDA's built-in audio settings are the primary method.
    MDMA provides guidance and convenience commands.
    
    Short aliases: /nvda, /reader
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not args:
        return dj.sr_menu()
    
    sub = args[0].lower()
    
    if sub in ('hep', 'headphones', 'phones'):
        return dj.sr_route('hep')
    
    elif sub in ('out', 'output', 'master', 'main'):
        return dj.sr_route('out')
    
    elif sub == 'keep':
        return dj.sr_route('keep')
    
    elif sub == 'test':
        return dj.sr_test()
    
    return f"Unknown SR command: {sub}"


# ============================================================================
# SAFETY & FALLBACK
# ============================================================================

def cmd_fallback(session: "Session", args: List[str]) -> str:
    """Slow-fallback safety system control.
    
    Usage:
      /fallback           Show fallback status
      /fallback off       Deactivate fallback (restore control)
      /fallback timeout <s> Set inactivity timeout (seconds)
    
    The slow-fallback system prevents failure during
    distraction or overload by:
    - Extending loops safely
    - Locking tempo
    - Applying neutral filter sweeps
    - Holding energy steady
    
    Short aliases: /fb, /safe
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not args:
        if dj.fallback.active:
            return (f"⚠️ FALLBACK ACTIVE\n"
                    f"  Reason: {dj.fallback.trigger_reason}\n"
                    f"  Loop extended: {dj.fallback.loop_extended}\n"
                    f"  Tempo locked: {dj.fallback.tempo_locked}\n"
                    f"\nUse /fallback off to restore control")
        else:
            return (f"Fallback Status: READY (not active)\n"
                    f"  Timeout: {dj.fallback_timeout:.0f}s of inactivity")
    
    sub = args[0].lower()
    
    if sub == 'off':
        return dj.deactivate_fallback()
    
    elif sub == 'timeout' and len(args) > 1:
        dj.fallback_timeout = float(args[1])
        return f"OK: Fallback timeout set to {dj.fallback_timeout:.0f}s"
    
    return f"Unknown fallback command: {sub}"


def cmd_library(session: "Session", args: List[str]) -> str:
    """Song library management.
    
    Usage:
      /library           List songs in library
      /library add <path> Add song to library
      /library scan <dir> Scan directory for songs
      /library search <q> Search library
      /library load <n>   Load song to deck
    
    Short aliases: /lib
    """
    err = _ensure_dj()
    if err:
        return err
    
    # Stub for now - would integrate with file system
    return ("=== SONG LIBRARY ===\n"
            "\n"
            "Library features coming soon:\n"
            "  - Scan and catalog audio files\n"
            "  - BPM and key detection\n"
            "  - Quick load to decks\n"
            "  - Playlist management")


def cmd_playlist(session: "Session", args: List[str]) -> str:
    """Playlist management.
    
    Usage:
      /playlist            List playlists
      /playlist <n>     Show playlist
      /playlist create <n> Create playlist
      /playlist add <song>   Add song to current playlist
    
    Short aliases: /pl
    """
    err = _ensure_dj()
    if err:
        return err
    
    return ("=== PLAYLISTS ===\n"
            "\n"
            "Playlist features coming soon")


# ============================================================================
# AI AUDIO ENHANCEMENT
# ============================================================================

def cmd_ai(session: "Session", args: List[str]) -> str:
    """Toggle AI audio enhancement on/off.
    
    Usage:
      /ai               Toggle AI enhancement on/off
      /ai on            Enable AI enhancement
      /ai off           Disable AI enhancement
      /ai status        Show detailed status and stats
      /ai passes <n>    Set number of multi-pass iterations (1-5)
    
    AI enhancement uses multi-pass processing to find the best
    result and includes corruption detection to prevent bad output.
    
    Default: OFF (must be enabled explicitly)
    
    Examples:
      /ai               -> Toggle on/off
      /ai on            -> Enable
      /ai passes 5      -> Use 5 enhancement passes
      /ai status        -> Show stats
    """
    try:
        from ..dsp.enhancement import cmd_ai as _cmd_ai
        return _cmd_ai(args)
    except ImportError:
        return "ERROR: Enhancement module not available"


def cmd_enhance(session: "Session", args: List[str]) -> str:
    """AI-powered audio quality enhancement for master output.
    
    Usage:
      /enhance              Show enhancement status
      /enhance on           Enable enhancement
      /enhance off          Disable enhancement
      /enhance <preset>     Set preset (transparent/master/broadcast/loud)
      /enhance bright <v>   Set brightness (-1 to +1)
      /enhance warm <v>     Set warmth (-1 to +1)
      /enhance punch <v>    Set transient punch (0 to 1)
      /enhance target <db>  Set target LUFS
      /enhance passes <n>   Set multi-pass count (1-5)
      /enhance stats        Show processing statistics
    
    Presets:
      transparent - Minimal processing, preserve character
      master      - Balanced mastering enhancement (default)
      broadcast   - Optimized for streaming/broadcast
      loud        - Maximum loudness (may sacrifice dynamics)
    
    The AI enhancement applies:
      - Multi-pass processing (tries variations, picks best)
      - Corruption detection (prevents bad output)
      - Dynamic range optimization
      - Spectral enhancement (brightness/warmth)
      - Transient preservation
      - Intelligent limiting
      - Loudness optimization
    
    Examples:
      /enhance master      -> Balanced mastering
      /enhance loud        -> Maximum loudness
      /enhance bright 0.3  -> Add brightness
      /enhance passes 5    -> Use 5 enhancement passes
    
    Short aliases: /enh
    """
    try:
        from ..dsp.enhancement import cmd_enhance as _cmd_enhance
        return _cmd_enhance(args)
    except ImportError:
        return "ERROR: Enhancement module not available"


# ============================================================================
# SONG REGISTRY
# ============================================================================

def cmd_registry(session: "Session", args: List[str]) -> str:
    """Permanent song registry with quality assurance.
    
    Usage:
      /reg                    Show registry stats
      /reg scan <folder>      Scan folder for songs (recursive)
      /reg scan <folder> flat Scan folder (non-recursive)
      /reg list [query]       List/search songs
      /reg info <id>          Show song details
      /reg load <id|name>     Load song to active deck
      /reg tag <id> <tags>    Add tags to song
      /reg rate <id> <1-5>    Rate song (1-5 stars)
      /reg fav <id>           Toggle favorite
      /reg quality            Show quality breakdown
      /reg fix <id>           Re-analyze and fix quality issues
      /reg remove <id>        Remove from registry
    
    Quality Grades:
      HIGH   - 320kbps+ or lossless, clean audio, ready for performance
      MEDIUM - 128-256kbps, minor issues, usable
      LOW    - < 128kbps, clipped, mono, or severe issues
    
    Songs are automatically:
      - Analyzed for BPM, key, energy, danceability
      - Graded for quality (clipping, dynamic range, etc.)
      - Converted to WAV64 float for high-quality playback
      - Auto-tagged with genre/mood hints
      - Indexed for instant loading by ID or name
    
    Examples:
      /reg scan ~/Music           Scan entire Music folder
      /reg scan ~/DJ recursive    Scan DJ folder recursively
      /reg list techno            Search for techno tracks
      /reg list bpm:120-140       Filter by BPM range
      /reg load 42                Load song #42 to deck
      /reg load "aphex twin"      Load by name search
      /reg tag 42 idm ambient     Add tags
      /reg rate 42 5              Rate 5 stars
    
    Short aliases: /songs, /song
    """
    try:
        from ..core.song_registry import cmd_reg
        return cmd_reg(args)
    except ImportError as e:
        return f"ERROR: Song registry not available: {e}"


# ============================================================================
# UNIVERSAL DJ EFFECTS (work on buffers or decks based on context)
# ============================================================================

def cmd_scr_buf(session: "Session", args: List[str]) -> str:
    """Apply scratch effect to current audio source (buffer or deck).
    
    Usage:
      /scrb                 Apply scratch to current context
      /scrb <preset>        Use preset 1-5
      /scrb <preset> <reps> Repeat the scratch pattern
    
    Presets:
      1 = Baby scratch (gentle back-forth)
      2 = Forward scratch (push-return)  
      3 = Chirp scratch (quick clipped)
      4 = Transformer (gated stutter)
      5 = Crab scratch (rapid multi-cuts)
    
    This is a destructive effect - it modifies the audio in place.
    Works on whichever is selected: buffer or deck.
    """
    audio, source_type, source_id = _get_audio_source(session)
    
    if audio is None:
        return "ERROR: No audio to process - generate or load audio first"
    
    # Parse args
    preset = 1
    repetitions = 2
    
    if args:
        try:
            preset = int(args[0])
            preset = max(1, min(5, preset))
        except ValueError:
            pass
        
        if len(args) > 1:
            try:
                repetitions = int(args[1])
                repetitions = max(1, min(8, repetitions))
            except ValueError:
                pass
    
    # Apply scratch effect with professional-grade smoothing
    sr = session.sample_rate
    
    # Scratch parameters by preset - refined for smoother output
    presets = {
        1: {'name': 'baby', 'chunks': 4, 'reverse_ratio': 0.5, 'speed_var': 0.15, 'wow': 0.02},
        2: {'name': 'forward', 'chunks': 3, 'reverse_ratio': 0.25, 'speed_var': 0.08, 'wow': 0.01},
        3: {'name': 'chirp', 'chunks': 6, 'reverse_ratio': 0.6, 'speed_var': 0.25, 'wow': 0.03},
        4: {'name': 'transformer', 'chunks': 8, 'reverse_ratio': 0.0, 'speed_var': 0.0, 'gate': True, 'wow': 0.0},
        5: {'name': 'crab', 'chunks': 12, 'reverse_ratio': 0.35, 'speed_var': 0.3, 'wow': 0.04},
    }
    
    p = presets.get(preset, presets[1])
    
    # Ensure minimum audio length
    if len(audio) < 1000:
        audio = np.tile(audio, int(np.ceil(1000 / len(audio))))[:1000]
    
    # Calculate chunk parameters
    total_chunks = p['chunks'] * repetitions
    chunk_size = max(int(len(audio) / p['chunks']), 500)  # Min 500 samples
    
    # Long crossfade for smooth transitions (10% of chunk or 512 samples)
    crossfade_len = min(512, chunk_size // 5)
    crossfade_len = max(64, crossfade_len)  # At least 64 samples
    
    # Create smooth crossfade curves (Hann window based)
    t = np.linspace(0, np.pi, crossfade_len)
    fade_in = (1 - np.cos(t)) / 2   # Smooth S-curve fade in
    fade_out = (1 + np.cos(t)) / 2  # Smooth S-curve fade out
    
    # Helper: high-quality resampling with cubic interpolation
    def resample_cubic(data, new_length):
        if len(data) < 4 or new_length < 4:
            return np.interp(np.linspace(0, 1, new_length), np.linspace(0, 1, len(data)), data)
        
        try:
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(data))
            x_new = np.linspace(0, 1, new_length)
            f = interp1d(x_old, data, kind='cubic', fill_value='extrapolate')
            return f(x_new)
        except ImportError:
            # Fall back to linear
            return np.interp(np.linspace(0, 1, new_length), np.linspace(0, 1, len(data)), data)
    
    # Pre-allocate result with overlap
    result_len = int(total_chunks * (chunk_size - crossfade_len // 2) + crossfade_len)
    result = np.zeros(result_len, dtype=np.float64)
    
    write_pos = 0
    rng = np.random.RandomState(42)  # Deterministic for repeatability
    
    for rep in range(repetitions):
        for i in range(p['chunks']):
            # Get source chunk with some overlap for continuity
            start = (i * chunk_size) % (len(audio) - chunk_size)
            chunk = audio[start:start + chunk_size].copy()
            
            # Apply transformations based on preset
            if p.get('gate'):
                # Transformer: smooth amplitude gating
                if i % 2 == 0:
                    # Create smooth ducking envelope
                    env = np.ones(len(chunk))
                    duck_len = min(200, len(chunk) // 3)
                    env[:duck_len] = np.linspace(0.05, 0.05, duck_len)
                    env[-duck_len:] = np.linspace(0.05, 0.05, duck_len)
                    # Smooth transition
                    trans_len = min(100, duck_len // 2)
                    env[duck_len:duck_len+trans_len] = np.linspace(0.05, 1.0, trans_len)
                    env[-(duck_len+trans_len):-duck_len] = np.linspace(1.0, 0.05, trans_len)
                    chunk = chunk * env
            else:
                # Random reverse
                if rng.random() < p['reverse_ratio']:
                    chunk = chunk[::-1]
            
            # Apply speed variation with high-quality resampling
            if p['speed_var'] > 0:
                speed = 1.0 + (rng.random() - 0.5) * p['speed_var'] * 2
                speed = max(0.7, min(1.5, speed))  # Clamp to reasonable range
                
                if abs(speed - 1.0) > 0.01:
                    new_len = max(100, int(len(chunk) / speed))
                    chunk = resample_cubic(chunk, new_len)
            
            # Add subtle wow/flutter (vinyl-like pitch wobble)
            if p.get('wow', 0) > 0:
                wow_freq = 2 + rng.random() * 3  # 2-5 Hz wobble
                wow_depth = p['wow']
                t_wow = np.arange(len(chunk)) / sr
                wow_mod = 1.0 + np.sin(2 * np.pi * wow_freq * t_wow) * wow_depth
                # Apply wow via variable delay (simplified as amplitude mod)
                chunk = chunk * (0.95 + 0.05 * wow_mod)
            
            # Ensure chunk is long enough for crossfade
            if len(chunk) < crossfade_len * 2:
                pad_len = crossfade_len * 2 - len(chunk)
                chunk = np.pad(chunk, (0, pad_len), mode='constant')
            
            # Apply fade envelopes
            chunk[:crossfade_len] = chunk[:crossfade_len] * fade_in
            chunk[-crossfade_len:] = chunk[-crossfade_len:] * fade_out
            
            # Write to result with overlap-add
            write_end = min(write_pos + len(chunk), len(result))
            actual_len = write_end - write_pos
            result[write_pos:write_end] += chunk[:actual_len]
            
            # Move position (overlap by crossfade length)
            write_pos += len(chunk) - crossfade_len
            
            if write_pos >= len(result) - crossfade_len:
                break
        
        if write_pos >= len(result) - crossfade_len:
            break
    
    # Trim to actual content
    result = result[:write_pos + crossfade_len]
    
    # Apply anti-aliasing filter (removes harsh high frequencies from resampling)
    try:
        from scipy.signal import butter, filtfilt, sosfilt, butter as butter_sos
        
        # Two-stage filtering for clean output:
        # 1. Gentle high-shelf cut to reduce harshness
        nyq = sr / 2
        
        # Low-pass at 16kHz to remove aliasing artifacts
        cutoff_lp = min(16000, nyq * 0.85)
        b_lp, a_lp = butter(3, cutoff_lp / nyq, btype='low')
        result = filtfilt(b_lp, a_lp, result)
        
        # High-pass at 20Hz to remove any DC offset
        cutoff_hp = 20
        b_hp, a_hp = butter(2, cutoff_hp / nyq, btype='high')
        result = filtfilt(b_hp, a_hp, result)
        
    except ImportError:
        # Manual simple smoothing if scipy unavailable
        # Moving average to reduce harshness
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        result = np.convolve(result, kernel, mode='same')
    
    # Soft-clip to prevent harsh distortion
    def soft_clip(x, threshold=0.9):
        """Soft clipping with smooth saturation."""
        mask = np.abs(x) > threshold
        x_clipped = x.copy()
        over = np.abs(x[mask]) - threshold
        sign = np.sign(x[mask])
        # Smooth saturation curve
        x_clipped[mask] = sign * (threshold + np.tanh(over * 2) * (1 - threshold))
        return x_clipped
    
    result = soft_clip(result, 0.92)
    
    # Final normalization
    max_val = np.max(np.abs(result))
    if max_val > 0.01:
        target_level = 0.85
        result = result * (target_level / max_val)
    
    # Store back to source
    dest = _store_audio_result(session, result, source_type, source_id)
    session.last_buffer = result
    
    # Update working buffer 
    try:
        from .working_cmds import get_working_buffer
        wb = get_working_buffer()
        wb.set_pending(result, f"scratch:{p['name']}", session)
    except Exception:
        pass
    
    dur = len(result) / sr
    return f"OK: Applied {p['name']} scratch ({repetitions}x) to {dest} ({dur:.2f}s)"


def cmd_stut_buf(session: "Session", args: List[str]) -> str:
    """Apply stutter effect to current audio source.
    
    Usage:
      /stut                 Default stutter (8 repeats, 1/16 beat)
      /stut <repeats>       Set repeat count
      /stut <reps> <size>   Set repeats and chunk size (ms)
    
    Examples:
      /stut 4               4 repeats of default size
      /stut 8 50            8 repeats of 50ms chunks
      /stut 16 25           Rapid 16 repeats of 25ms
    """
    audio, source_type, source_id = _get_audio_source(session)
    
    if audio is None:
        return "ERROR: No audio to process"
    
    sr = session.sample_rate
    
    # Parse args
    repeats = 8
    chunk_ms = 60  # Default ~1/16 at 120bpm
    
    if args:
        try:
            repeats = int(args[0])
            repeats = max(1, min(32, repeats))
        except ValueError:
            pass
        
        if len(args) > 1:
            try:
                chunk_ms = float(args[1])
                chunk_ms = max(10, min(500, chunk_ms))
            except ValueError:
                pass
    
    chunk_samples = int(sr * chunk_ms / 1000)
    
    # Get a random chunk from the audio
    max_start = max(0, len(audio) - chunk_samples)
    start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
    chunk = audio[start:start + chunk_samples]
    
    # Repeat the chunk
    result = np.tile(chunk, repeats)
    
    # Apply envelope to reduce harshness
    fade_samples = min(100, len(result) // 4)
    if fade_samples > 0:
        result[:fade_samples] *= np.linspace(0, 1, fade_samples)
        result[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # Store back
    dest = _store_audio_result(session, result, source_type, source_id)
    session.last_buffer = result
    
    # Update working buffer
    try:
        from .working_cmds import get_working_buffer
        wb = get_working_buffer()
        wb.set_pending(result, f"stutter:{repeats}x", session)
    except Exception:
        pass
    
    dur = len(result) / sr
    return f"OK: Stutter ({repeats}x {chunk_ms:.0f}ms) -> {dest} ({dur:.2f}s)"


def cmd_djfx(session: "Session", args: List[str]) -> str:
    """Apply DJ-style filter effect to current audio.
    
    Usage:
      /djfx hp              Highpass filter (cut lows)
      /djfx lp              Lowpass filter (cut highs)
      /djfx bp <freq>       Bandpass at frequency
      /djfx sweep <dir>     Filter sweep (up/down)
    
    Examples:
      /djfx hp              Cut bass frequencies
      /djfx lp              Cut treble frequencies
      /djfx bp 1000         Focus around 1kHz
      /djfx sweep up        Sweep from low to high
    """
    audio, source_type, source_id = _get_audio_source(session)
    
    if audio is None:
        return "ERROR: No audio to process"
    
    sr = session.sample_rate
    
    if not args:
        return ("Usage: /djfx <type>\n"
                "Types: hp (highpass), lp (lowpass), bp (bandpass), sweep")
    
    effect_type = args[0].lower()
    
    try:
        from scipy import signal
    except ImportError:
        return "ERROR: scipy required for filter effects"
    
    if effect_type in ('hp', 'highpass', 'high'):
        # Highpass at 300Hz
        cutoff = 300 if len(args) < 2 else float(args[1])
        sos = signal.butter(4, cutoff / (sr/2), btype='high', output='sos')
        result = signal.sosfilt(sos, audio)
        effect_name = f"highpass {cutoff:.0f}Hz"
        
    elif effect_type in ('lp', 'lowpass', 'low'):
        # Lowpass at 2000Hz
        cutoff = 2000 if len(args) < 2 else float(args[1])
        sos = signal.butter(4, cutoff / (sr/2), btype='low', output='sos')
        result = signal.sosfilt(sos, audio)
        effect_name = f"lowpass {cutoff:.0f}Hz"
        
    elif effect_type in ('bp', 'bandpass', 'band'):
        # Bandpass around frequency
        center = 1000 if len(args) < 2 else float(args[1])
        low = center * 0.7
        high = min(center * 1.4, sr/2 - 100)
        sos = signal.butter(2, [low / (sr/2), high / (sr/2)], btype='band', output='sos')
        result = signal.sosfilt(sos, audio)
        effect_name = f"bandpass {center:.0f}Hz"
        
    elif effect_type == 'sweep':
        # Filter sweep
        direction = 'up' if len(args) < 2 else args[1].lower()
        
        # Create time-varying filter
        n_frames = 100
        frame_size = len(audio) // n_frames
        result = np.zeros_like(audio)
        
        for i in range(n_frames):
            start = i * frame_size
            end = start + frame_size if i < n_frames - 1 else len(audio)
            
            # Calculate cutoff for this frame
            progress = i / n_frames
            if direction == 'up':
                cutoff = 200 + progress * 4000  # 200Hz -> 4200Hz
            else:
                cutoff = 4200 - progress * 4000  # 4200Hz -> 200Hz
            
            cutoff = min(cutoff, sr/2 - 100)
            
            try:
                sos = signal.butter(2, cutoff / (sr/2), btype='low', output='sos')
                result[start:end] = signal.sosfilt(sos, audio[start:end])
            except:
                result[start:end] = audio[start:end]
        
        effect_name = f"sweep {direction}"
    else:
        return f"ERROR: Unknown effect type '{effect_type}'"
    
    # Store back
    dest = _store_audio_result(session, result, source_type, source_id)
    session.last_buffer = result
    
    # Update working buffer
    try:
        from .working_cmds import get_working_buffer
        wb = get_working_buffer()
        wb.set_pending(result, f"djfx:{effect_name}", session)
    except Exception:
        pass
    
    dur = len(result) / sr
    return f"OK: Applied {effect_name} to {dest} ({dur:.2f}s)"


# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

DJ_COMMANDS = {
    # DJ Mode toggle
    'djm': cmd_djm,
    'dj': cmd_djm,
    'djmode': cmd_djm,
    
    # Device management
    'do': cmd_do,
    'devices': cmd_do,
    'dev': cmd_do,
    'doc': cmd_doc,
    'master': cmd_doc,
    'hep': cmd_hep,
    'headphones': cmd_hep,
    'phones': cmd_hep,
    'hepc': cmd_hepc,
    'cuehep': cmd_hepc,
    
    # Deck control
    'deck': cmd_deck,
    'dk': cmd_deck,
    'd': cmd_deck,
    'deck+': cmd_deckplus,
    'dk+': cmd_deckplus,
    'deck-': cmd_deckminus,
    'dk-': cmd_deckminus,
    
    # Playback control
    'play': cmd_play,
    'p': cmd_play,
    'start': cmd_play,
    'stop': cmd_stop,
    'pause': cmd_stop,
    
    # BPM control
    # Use /bpm to get/set tempo. Removed /tempo and /t aliases to avoid
    # conflicts with track/repeat commands and because BPM already exists.
    'bpm': cmd_tempo,
    
    # Volume control  
    'vol': cmd_vol,
    'volume': cmd_vol,
    'v': cmd_vol,
    
    # Crossfader
    'cf': cmd_cf,
    'crossfader': cmd_cf,
    'xfader': cmd_cf,
    
    # Cue/mixing
    'cue': cmd_cue,
    'c': cmd_cue,
    'xfade': cmd_xfade,
    'xf': cmd_xfade,
    'cross': cmd_xfade,
    'sync': cmd_sync,
    'sy': cmd_sync,
    
    # Transitions
    'tran': cmd_tran,
    'tr': cmd_tran,
    'trans': cmd_tran,
    'x': cmd_tran,  # quick transition
    'transition': cmd_transition,
    'drop': cmd_drop,
    'drp': cmd_drop,
    '!': cmd_drop,  # instant drop
    
    # Filter control
    'fl': cmd_filter,
    'flt': cmd_filter,
    'filter': cmd_filter,
    
    # Filter resonance
    'flr': cmd_resonance,
    'res': cmd_resonance,
    'q': cmd_resonance,
    'resonance': cmd_resonance,
    
    # Duration control
    'dur': cmd_duration,
    'duration': cmd_duration,
    'time': cmd_duration,
    
    # Loop control
    'lpc': cmd_lpc,
    'loopcount': cmd_lpc,
    'lpg': cmd_lpg,
    'loopgo': cmd_lpg,
    'lg': cmd_lpg,
    'lh': cmd_lh,
    'loophold': cmd_lh,
    'ml': cmd_ml,
    'masterloop': cmd_ml,
    'mdfx': cmd_mdfx,
    
    # Stutter
    'stud': cmd_stud,
    'stutter': cmd_stud,
    
    # Jump
    'j': cmd_jump,
    'jump': cmd_jump,
    'goto': cmd_jump,
    
    # Scratch
    'scr': cmd_scratch,
    'scratch': cmd_scratch,
    
    # Deck effects
    'dfx': cmd_dfx,
    'deckfx': cmd_dfx,
    'deckeffect': cmd_dfx,
    
    # Stem separation
    'stem': cmd_stem,
    'st': cmd_stem,
    'stems': cmd_stem,
    
    # Sectioning and chopping
    'section': cmd_section,
    'sec': cmd_section,
    'chop': cmd_chop,
    'slice': cmd_chop,
    
    # Streaming
    'stream': cmd_stream,
    'str': cmd_stream,
    'sc': cmd_stream,  # soundcloud shortcut
    
    # VDA
    'vda': cmd_vda,
    'virtual': cmd_vda,
    
    # Screen reader
    'sr': cmd_sr,
    'nvda': cmd_sr,
    'reader': cmd_sr,
    
    # Safety
    'fallback': cmd_fallback,
    'fb': cmd_fallback,
    'safe': cmd_fallback,
    
    # Library
    'library': cmd_library,
    'lib': cmd_library,
    'playlist': cmd_playlist,
    'pl': cmd_playlist,
    
    # AI Audio Enhancement
    'ai': cmd_ai,  # Quick toggle
    'enhance': cmd_enhance,
    'enh': cmd_enhance,
    'enhancement': cmd_enhance,
    
    # Song Registry
    'reg': cmd_registry,
    'registry': cmd_registry,
    'songs': cmd_registry,
    'song': cmd_registry,
    
    # Universal DJ effects (work on buffers or decks)
    'scrb': cmd_scr_buf,
    'scratchbuf': cmd_scr_buf,
    'stut': cmd_stut_buf,
    'stutter': cmd_stut_buf,
    'djfx': cmd_djfx,
    'djfilter': cmd_djfx,
}


def get_dj_commands():
    """Return DJ commands for registration."""
    return DJ_COMMANDS
