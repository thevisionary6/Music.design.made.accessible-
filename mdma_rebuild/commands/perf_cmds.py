"""MDMA Performance Mode Commands + Quick Deck Loading + Bluetooth.

Performance Mode: Macro-driven layer on top of DJ Mode.
Quick Deck Loading: /DKL (file), /DKS (search)
Bluetooth: /BL (scan), /BDOC (connect)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.session import Session


# Import engines
try:
    from ..dsp.performance import get_perf_engine, is_perf_mode_enabled
    from ..dsp.dj_mode import get_dj_engine, is_dj_mode_enabled, DeviceType
    ENGINES_AVAILABLE = True
except ImportError:
    ENGINES_AVAILABLE = False


def _ensure_dj():
    """Check DJ mode is active."""
    if not ENGINES_AVAILABLE:
        return "ERROR: DJ Mode not available"
    if not is_dj_mode_enabled():
        return "ERROR: DJ Mode not active. Use /djm on first."
    return None


# ============================================================================
# PERFORMANCE MODE COMMANDS
# ============================================================================

def cmd_perf(session: "Session", args: List[str]) -> str:
    """Performance Mode control.
    
    Usage:
      /perf              Toggle Performance Mode
      /perf on           Enable (requires DJ Mode)
      /perf off          Disable
      /perf status       Show status
    
    IMPORTANT: Performance Mode requires DJ Mode to be active.
    
    Short aliases: /pf, /perform
    """
    if not ENGINES_AVAILABLE:
        return "ERROR: Performance Mode not available"
    
    dj = get_dj_engine(session.sample_rate)
    perf = get_perf_engine(dj)
    
    if not args:
        if perf.enabled:
            return perf.disable()
        else:
            return perf.enable()
    
    sub = args[0].lower()
    
    if sub in ('on', 'enable', '1'):
        return perf.enable()
    elif sub in ('off', 'disable', '0'):
        return perf.disable()
    elif sub in ('status', 'stat', 's'):
        if not perf.enabled:
            return ("Performance Mode: DISABLED\n"
                    f"DJ Mode: {'ENABLED' if dj.enabled else 'DISABLED'}")
        return (f"=== PERFORMANCE MODE ===\n"
                f"  Enabled: YES\n"
                f"  Macros: {len(perf.macros)}\n"
                f"  Snapshots: {len(perf.snapshots)}\n"
                f"  Home: {perf.home_snapshot or 'not set'}\n"
                f"  Evolution: {'frozen' if perf.evolution_frozen else 'active'}\n"
                f"  RNG: seed={perf.rng.global_seed}, amount={perf.rng.amount:.0%}")
    
    return f"Unknown perf command: {sub}"


def cmd_mc(session: "Session", args: List[str]) -> str:
    """Macro management.
    
    Usage:
      /mc                    List all macros
      /mc new <name>         Create macro
      /mc add <name> <type> <action>  Add step
      /mc set <name> <prop> <val>     Set property
      /mc arm <name>         Arm macro
      /mc run <name>         Execute macro
      /mc stop <name>        Stop macro
      /mc info <name>        Show details
      /mc del <name>         Delete macro
    
    Step types: command, snapshot, transition, wait, mutate
    Properties: home, seed, rng_amount, rng_domain, mutate_bars
    
    Examples:
      /mc new drop_hit
      /mc add drop_hit command "tran 2 cut"
      /mc add drop_hit command "stem 2 drums 1.5"
      /mc set drop_hit home intro
      /mc run drop_hit
    
    Short aliases: /macro
    """
    if not ENGINES_AVAILABLE:
        return "ERROR: Performance Mode not available"
    
    dj = get_dj_engine(session.sample_rate)
    perf = get_perf_engine(dj)
    
    if not perf.enabled:
        return "ERROR: Performance Mode not enabled. Use /perf on"
    
    if not args:
        return perf.list_macros()
    
    sub = args[0].lower()
    
    if sub == 'list':
        return perf.list_macros()
    elif sub == 'new' and len(args) > 1:
        desc = ' '.join(args[2:]) if len(args) > 2 else ""
        return perf.create_macro(args[1], desc)
    elif sub == 'add' and len(args) > 3:
        name, step_type, action = args[1], args[2], ' '.join(args[3:])
        return perf.add_macro_step(name, step_type, action)
    elif sub == 'set' and len(args) > 3:
        return perf.set_macro_property(args[1], args[2], args[3])
    elif sub == 'arm' and len(args) > 1:
        return perf.arm_macro(args[1])
    elif sub == 'run' and len(args) > 1:
        return perf.run_macro(args[1])
    elif sub == 'stop' and len(args) > 1:
        return perf.stop_macro(args[1])
    elif sub == 'info' and len(args) > 1:
        return perf.macro_info(args[1])
    elif sub == 'del' and len(args) > 1:
        if args[1] in perf.macros:
            del perf.macros[args[1]]
            return f"OK: deleted macro '{args[1]}'"
        return f"ERROR: macro '{args[1]}' not found"
    elif sub == 'panic':
        return perf.panic()
    elif sub == 'freeze':
        return perf.freeze_evolution()
    elif sub == 'unfreeze':
        return perf.unfreeze_evolution()
    
    return f"Unknown mc command: {sub}"


def cmd_snap(session: "Session", args: List[str]) -> str:
    """Snapshot management.
    
    Usage:
      /snap                  List snapshots
      /snap cap <name> [label]  Capture current state
      /snap recall <name>    Recall snapshot
      /snap home <name>      Set home snapshot
      /snap go               Return to home
      /snap del <name>       Delete snapshot
    
    Labels: verse, drop, break, build, outro
    
    Examples:
      /snap cap intro verse
      /snap cap drop1 drop
      /snap home intro
      /snap go
    
    Short aliases: /snapshot, /ss
    """
    if not ENGINES_AVAILABLE:
        return "ERROR: Performance Mode not available"
    
    dj = get_dj_engine(session.sample_rate)
    perf = get_perf_engine(dj)
    
    if not perf.enabled:
        return "ERROR: Performance Mode not enabled"
    
    if not args:
        return perf.list_snapshots()
    
    sub = args[0].lower()
    
    if sub == 'list':
        return perf.list_snapshots()
    elif sub in ('cap', 'capture', 'save') and len(args) > 1:
        name = args[1]
        label = args[2] if len(args) > 2 else ""
        return perf.capture_snapshot(name, label)
    elif sub == 'recall' and len(args) > 1:
        return perf.recall_snapshot(args[1])
    elif sub == 'home':
        if len(args) > 1:
            return perf.set_home_snapshot(args[1])
        return perf.go_home()
    elif sub == 'go':
        return perf.go_home()
    elif sub == 'del' and len(args) > 1:
        if args[1] in perf.snapshots:
            del perf.snapshots[args[1]]
            return f"OK: deleted snapshot '{args[1]}'"
        return f"ERROR: snapshot '{args[1]}' not found"
    
    return f"Unknown snap command: {sub}"


def cmd_panic(session: "Session", args: List[str]) -> str:
    """Emergency panic - immediate safe recovery.
    
    Usage:
      /panic             Stop all macros, freeze evolution, go home
    
    Short aliases: /!panic, /stop!
    """
    if not ENGINES_AVAILABLE:
        return "ERROR: Performance Mode not available"
    
    dj = get_dj_engine(session.sample_rate)
    perf = get_perf_engine(dj)
    
    if not perf.enabled:
        return "Performance Mode not enabled"
    
    return perf.panic()


def cmd_rng(session: "Session", args: List[str]) -> str:
    """RNG (randomness) control.
    
    Usage:
      /rng                   Show status
      /rng seed <n>          Set seed
      /rng amount <0-1>      Set amount
      /rng lock              Lock to seed
      /rng unlock            Unlock
      /rng domain <d1,d2>    Set domains
    
    Domains: filter, gate, volume, tempo, stem, pitch
    
    Short aliases: /random
    """
    if not ENGINES_AVAILABLE:
        return "ERROR: Performance Mode not available"
    
    dj = get_dj_engine(session.sample_rate)
    perf = get_perf_engine(dj)
    
    if not perf.enabled:
        return "ERROR: Performance Mode not enabled"
    
    if not args:
        return (f"=== RNG STATUS ===\n"
                f"  Seed: {perf.rng.global_seed}\n"
                f"  Amount: {perf.rng.amount:.0%}\n"
                f"  Locked: {perf.rng.locked}\n"
                f"  Domain: {', '.join(perf.rng.domain)}")
    
    sub = args[0].lower()
    
    if sub == 'seed' and len(args) > 1:
        perf.rng.set_seed(int(args[1]))
        return f"OK: RNG seed set to {args[1]}"
    elif sub == 'amount' and len(args) > 1:
        perf.rng.set_amount(float(args[1]))
        return f"OK: RNG amount set to {perf.rng.amount:.0%}"
    elif sub == 'lock':
        perf.rng.lock()
        return "OK: RNG locked (deterministic)"
    elif sub == 'unlock':
        perf.rng.unlock()
        return "OK: RNG unlocked"
    elif sub == 'domain' and len(args) > 1:
        perf.rng.set_domain(args[1].split(','))
        return f"OK: RNG domain set to {perf.rng.domain}"
    
    return f"Unknown rng command: {sub}"


# ============================================================================
# QUICK DECK LOADING COMMANDS
# ============================================================================

def cmd_dkl(session: "Session", args: List[str]) -> str:
    """Quick deck load - load local file to deck.
    
    Usage:
      /DKL <deck> <filepath>  Load file to deck
      /DKL <deck>             Show common paths
    
    Supports: wav, mp3, flac, ogg, m4a, aiff
    
    Examples:
      /DKL 1 ~/Music/track.mp3
      /DKL 2 /path/to/song.wav
    
    Short aliases: /deckload, /load
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not args:
        return ("Usage: /DKL <deck> <filepath>\n"
                "\nCommon paths:\n"
                "  ~/Music/\n"
                "  ~/Downloads/\n"
                "  ~/Documents/MDMA/packs/\n"
                "\nPaths with spaces are supported:\n"
                "  /DKL 1 E:/backup/music/virtual riot/track.mp3")
    
    try:
        deck_id = int(args[0])
    except ValueError:
        return f"ERROR: invalid deck number '{args[0]}'"
    
    if len(args) < 2:
        music_dir = Path.home() / 'Music'
        if music_dir.exists():
            try:
                files = list(music_dir.glob('*.mp3'))[:10] + list(music_dir.glob('*.wav'))[:10]
                lines = [f"Files in {music_dir}:", ""]
                for f in files[:20]:
                    lines.append(f"  {f.name}")
                lines.append("")
                lines.append(f"Use: /DKL {deck_id} ~/Music/<filename>")
                return '\n'.join(lines)
            except PermissionError:
                return f"ERROR: Permission denied reading {music_dir}"
        return f"Usage: /DKL {deck_id} <filepath>"
    
    # Join all remaining args to support paths with spaces
    # e.g., /DKL 1 E:/backup/music/virtual riot/track.mp3
    filepath = ' '.join(args[1:])
    filepath = os.path.expanduser(filepath)
    filepath = Path(filepath)
    
    # Check if path is a directory
    if filepath.is_dir():
        return (f"ERROR: '{filepath}' is a directory, not a file.\n"
                f"Use /DKS to search for files, or specify a file path.")
    
    if not filepath.exists():
        return f"ERROR: file not found: {filepath}"
    
    # Check read permission before trying to load
    if not os.access(filepath, os.R_OK):
        return (f"ERROR: Permission denied - cannot read file.\n"
                f"  File: {filepath}\n"
                f"  Try: Check file permissions or run MDMA with appropriate access.")
    
    try:
        from ..dsp.streaming import load_audio_file
        
        audio, sr = load_audio_file(filepath, session.sample_rate)
        
        if deck_id not in dj.decks:
            from ..dsp.dj_mode import DJDeck
            dj.decks[deck_id] = DJDeck(id=deck_id)
        
        deck = dj.decks[deck_id]
        deck.buffer = audio
        deck.position = 0.0
        deck.playing = False
        deck.analyzed = False
        deck.stems = None
        
        dur = len(audio) / session.sample_rate
        return f"OK: loaded '{filepath.name}' ({dur:.1f}s) to Deck {deck_id}"
        
    except PermissionError as e:
        return (f"ERROR: Permission denied.\n"
                f"  File: {filepath}\n"
                f"  {e}")
    except IsADirectoryError:
        return f"ERROR: '{filepath}' is a directory. Specify an audio file."
    except FileNotFoundError:
        return f"ERROR: File not found: {filepath}"
    except Exception as e:
        return f"ERROR: Failed to load file: {e}"


def cmd_dks(session: "Session", args: List[str]) -> str:
    """Quick deck search - search and load files.
    
    Usage:
      /DKS <query>           Search for files
      /DKS <deck> <index>    Load result by index
      /DKS dir <path>        Set search directory
    
    Examples:
      /DKS kick              Search for 'kick'
      /DKS 1 3               Load result #3 to deck 1
      /DKS dir ~/Music       Set search dir
      /DKS dir E:/backup/music/virtual riot
    
    Short aliases: /decksearch, /search
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    # Store search state
    if not hasattr(dj, '_search_results'):
        dj._search_results = []
    if not hasattr(dj, '_search_dir'):
        dj._search_dir = Path.home() / 'Music'
    
    if not args:
        return ("Usage:\n"
                "  /DKS <query>      Search for files\n"
                "  /DKS <deck> <idx> Load result to deck\n"
                "  /DKS dir <path>   Set search directory\n"
                f"\nSearch dir: {dj._search_dir}")
    
    # Set directory - join all args after 'dir' to support spaces
    if args[0] == 'dir' and len(args) > 1:
        # Join all remaining args to support paths with spaces
        path_str = ' '.join(args[1:])
        path = Path(os.path.expanduser(path_str))
        if not path.exists():
            return f"ERROR: directory not found: {path}"
        if not path.is_dir():
            return f"ERROR: not a directory: {path}"
        if not os.access(path, os.R_OK):
            return f"ERROR: Permission denied - cannot read directory: {path}"
        dj._search_dir = path
        return f"OK: search directory set to {path}"
    
    # Load by index: /DKS <deck> <index>
    if len(args) >= 2:
        try:
            deck_id = int(args[0])
            idx = int(args[1])
            
            if not dj._search_results:
                return "No search results. Run /DKS <query> first."
            
            if idx < 0 or idx >= len(dj._search_results):
                return f"ERROR: invalid index {idx}. Range: 0-{len(dj._search_results)-1}"
            
            filepath = dj._search_results[idx]
            
            # Check file still exists
            if not filepath.exists():
                return f"ERROR: File no longer exists: {filepath.name}"
            
            # Check read permission
            if not os.access(filepath, os.R_OK):
                return (f"ERROR: Permission denied.\n"
                        f"  File: {filepath}\n"
                        f"  Check file permissions or try a different file.")
            
            try:
                from ..dsp.streaming import load_audio_file
                
                audio, sr = load_audio_file(filepath, session.sample_rate)
                
                if deck_id not in dj.decks:
                    from ..dsp.dj_mode import DJDeck
                    dj.decks[deck_id] = DJDeck(id=deck_id)
                
                deck = dj.decks[deck_id]
                deck.buffer = audio
                deck.position = 0.0
                deck.playing = False
                
                dur = len(audio) / session.sample_rate
                return f"OK: loaded '{filepath.name}' ({dur:.1f}s) to Deck {deck_id}"
                
            except PermissionError as e:
                return (f"ERROR: Permission denied.\n"
                        f"  File: {filepath}\n"
                        f"  {e}")
            except Exception as e:
                return f"ERROR: Failed to load '{filepath.name}': {e}"
            
        except ValueError:
            pass  # Not loading, treat as search
    
    # Search
    query = ' '.join(args).lower()
    search_dir = dj._search_dir
    
    if not search_dir.exists():
        return f"ERROR: search directory not found: {search_dir}"
    
    if not os.access(search_dir, os.R_OK):
        return f"ERROR: Permission denied - cannot read directory: {search_dir}"
    
    extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.aif'}
    results = []
    
    try:
        for f in search_dir.rglob('*'):
            try:
                if f.suffix.lower() in extensions:
                    if query in f.name.lower() or query in str(f.parent).lower():
                        results.append(f)
                if len(results) >= 50:
                    break
            except PermissionError:
                continue  # Skip files we can't access
    except PermissionError:
        return f"ERROR: Permission denied searching {search_dir}"
    
    dj._search_results = results
    
    if not results:
        return f"No files matching '{query}' in {search_dir}"
    
    lines = [f"=== SEARCH: {query} ({len(results)} results) ===", ""]
    for i, f in enumerate(results[:20]):
        lines.append(f"  [{i}] {f.name}")
    
    if len(results) > 20:
        lines.append(f"  ... and {len(results) - 20} more")
    
    lines.append("")
    lines.append("Use /DKS <deck> <idx> to load")
    
    return '\n'.join(lines)


# ============================================================================
# BLUETOOTH DEVICE COMMANDS
# ============================================================================

def cmd_bl(session: "Session", args: List[str]) -> str:
    """Scan and list Bluetooth audio devices only.
    
    Usage:
      /BL                    Scan for Bluetooth devices
      /BL scan               Rescan Bluetooth
    
    Shows only Bluetooth speakers and headphones.
    Use /BDOC <id> to connect to master output.
    
    Short aliases: /bluetooth, /bt
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if args and args[0].lower() == 'scan':
        dj._scan_devices()
    
    # Filter Bluetooth devices
    bt_devices = [d for d in dj.devices.values() 
                  if d.device_type == DeviceType.BLUETOOTH]
    
    if not bt_devices:
        return ("No Bluetooth audio devices found.\n"
                "Make sure Bluetooth is enabled and devices are paired.\n"
                "Use /BL scan to rescan.")
    
    lines = ["=== BLUETOOTH DEVICES ===", ""]
    for dev in bt_devices:
        marker = ""
        if dev.id == dj.master_device:
            marker = " [MASTER]"
        elif dev.id == dj.headphone_device:
            marker = " [CUE]"
        
        status = "●" if dev.connected else "○"
        lines.append(f"  {status} [{dev.id}] {dev.name}{marker}")
        lines.append(f"      Latency: ~{dev.latency_ms:.0f}ms")
    
    lines.append("")
    lines.append("Commands:")
    lines.append("  /BDOC <id>  Connect to master output")
    lines.append("  /BTHEP <id> Connect to headphones/cue")
    
    return '\n'.join(lines)


def cmd_bdoc(session: "Session", args: List[str]) -> str:
    """Connect Bluetooth device to master output.
    
    Usage:
      /BDOC <id>             Connect Bluetooth as master
      /BDOC                  List Bluetooth devices
    
    Short aliases: /btmaster, /btout
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not args:
        return cmd_bl(session, [])
    
    device_id = args[0]
    
    if device_id in dj.devices:
        dev = dj.devices[device_id]
        if dev.device_type != DeviceType.BLUETOOTH:
            return (f"WARNING: '{dev.name}' is not Bluetooth "
                    f"({dev.device_type.value}). Connecting anyway...")
    
    result = dj.connect_master(device_id)
    
    if "OK:" in result and device_id in dj.devices:
        dev = dj.devices[device_id]
        if dev.latency_ms > 30:
            result += f"\n⚠️ Bluetooth latency (~{dev.latency_ms:.0f}ms) may affect timing"
    
    return result


def cmd_bthep(session: "Session", args: List[str]) -> str:
    """Connect Bluetooth device to headphones/cue.
    
    Usage:
      /BTHEP <id>            Connect Bluetooth headphones
      /BTHEP                 List Bluetooth devices
    
    Short aliases: /btheadphones, /btcue
    """
    err = _ensure_dj()
    if err:
        return err
    
    dj = get_dj_engine(session.sample_rate)
    
    if not args:
        return cmd_bl(session, [])
    
    return dj.connect_headphones(args[0])


# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

PERF_COMMANDS = {
    # Performance Mode
    'perf': cmd_perf,
    'pf': cmd_perf,
    'perform': cmd_perf,
    'performance': cmd_perf,
    
    # Macros
    'mc': cmd_mc,
    'macro': cmd_mc,
    'macros': cmd_mc,
    
    # Panic
    'panic': cmd_panic,
    '!panic': cmd_panic,
    'stop!': cmd_panic,
    
    # Snapshots
    'snap': cmd_snap,
    'snapshot': cmd_snap,
    'ss': cmd_snap,
    
    # RNG
    'rng': cmd_rng,
    'random': cmd_rng,
    
    # Quick deck loading
    'dkl': cmd_dkl,
    'DKL': cmd_dkl,
    'deckload': cmd_dkl,
    # Note: 'load' reserved for project load in general_cmds
    
    'dks': cmd_dks,
    'DKS': cmd_dks,
    'decksearch': cmd_dks,
    'search': cmd_dks,
    
    # Bluetooth
    'bl': cmd_bl,
    'BL': cmd_bl,
    'bluetooth': cmd_bl,
    'bt': cmd_bl,
    
    'bdoc': cmd_bdoc,
    'BDOC': cmd_bdoc,
    'btmaster': cmd_bdoc,
    'btout': cmd_bdoc,
    
    'bthep': cmd_bthep,
    'BTHEP': cmd_bthep,
    'btheadphones': cmd_bthep,
    'btcue': cmd_bthep,
}


def get_perf_commands():
    """Return performance commands for registration."""
    return PERF_COMMANDS
