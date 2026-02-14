"""MIDI commands — Phase 6.0: MIDI as Protocol Writer.

Provides the /midi command family for device management, configuration,
and MIDI input control.  MIDI writes interval tokens into the existing
protocol — it does NOT add sequencing, automation, or a real-time layer.

Commands:
    /midi              Show MIDI status and help
    /midi list         List available MIDI input devices
    /midi select <n>   Connect to device by index
    /midi connect <n>  Alias for select
    /midi disconnect   Close current MIDI device
    /midi status       Show connection and mode status
    /midi mode <m>     Set mode: preview | program
    /midi root <note>  Set root note for interval calc (MIDI # or name)
    /midi window <ms>  Set chord detection window in ms
    /midi rest         Insert rest token (_) into token buffer
    /midi tokens       Show and clear pending tokens from program mode
    /midi flush        Flush pending tokens as protocol string

BUILD ID: midi_cmds_v1.0_phase6
"""

from __future__ import annotations

from typing import Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.session import Session

# ============================================================================
# Lazy controller access
# ============================================================================

def _get_controller(session: Any):
    """Get or create the MIDIInputController on the session."""
    if not hasattr(session, '_midi_controller'):
        from ..dsp.midi_input import MIDIInputController, midi_available
        if not midi_available():
            return None
        session._midi_controller = MIDIInputController(session)
    return session._midi_controller


def _midi_unavailable_msg() -> str:
    return (
        "MIDI support requires mido and python-rtmidi.\n"
        "Install with: pip install mido python-rtmidi\n"
        "Or: pip install mdma[midi]"
    )


# ============================================================================
# /midi command
# ============================================================================

def cmd_midi(session: Any, args: List[str]) -> str:
    """MIDI device management and input control.

    Usage:
      /midi              Show status and help
      /midi list         List available MIDI input devices
      /midi select <n>   Connect to device by index number
      /midi disconnect   Close current MIDI connection
      /midi status       Show connection status
      /midi mode <m>     Set mode: preview or program
      /midi root <note>  Set root note (MIDI number or name like C4)
      /midi window <ms>  Set chord detection window (default 80ms)
      /midi rest         Insert a rest token
      /midi tokens       Show pending program-mode tokens
      /midi flush        Get pending tokens as protocol string
    """
    if not args:
        return _midi_status_help(session)

    sub = args[0].lower()

    if sub == 'list':
        return _midi_list(session)
    elif sub in ('select', 'connect'):
        return _midi_select(session, args[1:])
    elif sub in ('disconnect', 'close', 'off'):
        return _midi_disconnect(session)
    elif sub == 'status':
        return _midi_status(session)
    elif sub == 'mode':
        return _midi_mode(session, args[1:])
    elif sub == 'root':
        return _midi_root(session, args[1:])
    elif sub in ('window', 'win'):
        return _midi_window(session, args[1:])
    elif sub == 'rest':
        return _midi_rest(session)
    elif sub == 'tokens':
        return _midi_tokens(session)
    elif sub == 'flush':
        return _midi_flush(session)
    else:
        return f"Unknown MIDI subcommand: {sub}. Try /midi for help."


# ============================================================================
# Subcommand implementations
# ============================================================================

def _midi_status_help(session: Any) -> str:
    """Show MIDI status and available subcommands."""
    from ..dsp.midi_input import midi_available

    lines = ["MIDI Input — Phase 6.0 Protocol Writer"]
    lines.append("")

    if not midi_available():
        lines.append("STATUS: mido/python-rtmidi NOT installed")
        lines.append(_midi_unavailable_msg())
        return '\n'.join(lines)

    ctrl = _get_controller(session)
    if ctrl and ctrl.is_connected:
        lines.append(f"STATUS: Connected to {ctrl.device_manager.device_name}")
        lines.append(f"MODE: {ctrl.mode}")
    else:
        lines.append("STATUS: Not connected")

    lines.append(f"ROOT: MIDI {session.midi_root_note} ({_note_name(session.midi_root_note)})")
    lines.append(f"CHORD WINDOW: {session.chord_window_ms}ms")
    lines.append("")
    lines.append("Subcommands:")
    lines.append("  /midi list         List devices")
    lines.append("  /midi select <n>   Connect to device")
    lines.append("  /midi disconnect   Close connection")
    lines.append("  /midi mode <m>     Set preview|program")
    lines.append("  /midi root <note>  Set root (C4, 60, etc.)")
    lines.append("  /midi window <ms>  Chord detection window")
    lines.append("  /midi rest         Insert rest token")
    lines.append("  /midi tokens       Show pending tokens")
    lines.append("  /midi flush        Flush tokens as string")

    return '\n'.join(lines)


def _midi_list(session: Any) -> str:
    """List available MIDI input devices."""
    from ..dsp.midi_input import midi_available
    if not midi_available():
        return _midi_unavailable_msg()

    ctrl = _get_controller(session)
    if ctrl is None:
        return _midi_unavailable_msg()

    devices = ctrl.list_devices()
    if not devices:
        return "No MIDI input devices found."

    lines = [f"MIDI Input Devices ({len(devices)}):"]
    for i, name in enumerate(devices):
        marker = " <-- connected" if (ctrl.is_connected and ctrl.device_manager.device_name == name) else ""
        lines.append(f"  {i}: {name}{marker}")
    lines.append("")
    lines.append("Use /midi select <n> to connect.")
    return '\n'.join(lines)


def _midi_select(session: Any, args: List[str]) -> str:
    """Connect to a MIDI device by index."""
    from ..dsp.midi_input import midi_available
    if not midi_available():
        return _midi_unavailable_msg()

    ctrl = _get_controller(session)
    if ctrl is None:
        return _midi_unavailable_msg()

    if not args:
        return "Usage: /midi select <device_number>"

    devices = ctrl.list_devices()
    if not devices:
        return "No MIDI input devices found."

    try:
        idx = int(args[0])
    except ValueError:
        # Try matching by name substring
        query = ' '.join(args).lower()
        matches = [(i, d) for i, d in enumerate(devices) if query in d.lower()]
        if len(matches) == 1:
            idx = matches[0][0]
        elif len(matches) > 1:
            lines = ["Multiple matches:"]
            for i, d in matches:
                lines.append(f"  {i}: {d}")
            return '\n'.join(lines)
        else:
            return f"No device matching '{args[0]}'. Use /midi list."

    if idx < 0 or idx >= len(devices):
        return f"Device index {idx} out of range (0-{len(devices) - 1})."

    name = devices[idx]
    if ctrl.connect(name):
        return f"MIDI: Connected to {name}"
    else:
        return f"MIDI: Failed to connect to {name}"


def _midi_disconnect(session: Any) -> str:
    """Disconnect from the current MIDI device."""
    ctrl = _get_controller(session)
    if ctrl is None or not ctrl.is_connected:
        return "MIDI: Not connected."

    name = ctrl.device_manager.device_name
    ctrl.disconnect()
    return f"MIDI: Disconnected from {name}"


def _midi_status(session: Any) -> str:
    """Show MIDI connection status."""
    from ..dsp.midi_input import midi_available

    lines = []
    if not midi_available():
        lines.append("MIDI libraries: NOT INSTALLED")
        lines.append(_midi_unavailable_msg())
        return '\n'.join(lines)

    ctrl = _get_controller(session)

    lines.append("MIDI libraries: installed")
    if ctrl and ctrl.is_connected:
        lines.append(f"Device: {ctrl.device_manager.device_name}")
        lines.append(f"Mode: {ctrl.mode}")
        pending = len(ctrl.token_buffer)
        if pending:
            lines.append(f"Pending tokens: {pending}")
    else:
        lines.append("Device: not connected")

    lines.append(f"Root note: MIDI {session.midi_root_note} ({_note_name(session.midi_root_note)})")
    lines.append(f"Chord window: {session.chord_window_ms}ms")

    return '\n'.join(lines)


def _midi_mode(session: Any, args: List[str]) -> str:
    """Set MIDI input mode."""
    ctrl = _get_controller(session)
    if ctrl is None:
        return _midi_unavailable_msg()

    if not args:
        mode = ctrl.mode if ctrl else 'preview'
        return f"MIDI mode: {mode}\nUsage: /midi mode preview|program"

    mode = args[0].lower()
    if mode not in ('preview', 'program'):
        return "Mode must be 'preview' or 'program'."

    ctrl.set_mode(mode)
    return f"MIDI mode: {mode}"


def _midi_root(session: Any, args: List[str]) -> str:
    """Set the MIDI root note for interval calculation."""
    if not args:
        return (
            f"MIDI root: {session.midi_root_note} ({_note_name(session.midi_root_note)})\n"
            f"Usage: /midi root <note>  (MIDI number or name like C4, A3, Bb2)"
        )

    token = args[0].strip()

    # Try as MIDI number first
    try:
        note = int(token)
        if 0 <= note <= 127:
            session.midi_root_note = note
            ctrl = _get_controller(session)
            if ctrl:
                ctrl.sync_settings()
            return f"MIDI root: {note} ({_note_name(note)})"
        else:
            return f"MIDI note must be 0-127, got {note}."
    except ValueError:
        pass

    # Try as note name
    try:
        from ..dsp.midi_input import IntervalTranslator
        note = IntervalTranslator.note_name_to_midi(token)
        if 0 <= note <= 127:
            session.midi_root_note = note
            ctrl = _get_controller(session)
            if ctrl:
                ctrl.sync_settings()
            return f"MIDI root: {note} ({_note_name(note)})"
    except Exception:
        pass

    return f"Cannot parse note: '{token}'. Use MIDI number (0-127) or name (C4, A#3)."


def _midi_window(session: Any, args: List[str]) -> str:
    """Set the chord detection window in milliseconds."""
    if not args:
        return (
            f"Chord detection window: {session.chord_window_ms}ms\n"
            f"Usage: /midi window <ms>  (10-500, default 80)"
        )

    try:
        ms = int(args[0])
    except ValueError:
        return "Window must be an integer (milliseconds)."

    if ms < 10 or ms > 500:
        return f"Window must be 10-500ms, got {ms}."

    session.chord_window_ms = ms
    ctrl = _get_controller(session)
    if ctrl:
        ctrl.sync_settings()
    return f"Chord detection window: {ms}ms"


def _midi_rest(session: Any) -> str:
    """Insert a rest token into the program-mode token buffer."""
    ctrl = _get_controller(session)
    if ctrl is None:
        return _midi_unavailable_msg()

    ctrl.translator.insert_rest()
    return "MIDI: Rest token (_) inserted."


def _midi_tokens(session: Any) -> str:
    """Show pending tokens from program mode."""
    ctrl = _get_controller(session)
    if ctrl is None:
        return "MIDI: No controller active."

    tokens = ctrl.token_buffer[:]
    if not tokens:
        return "MIDI: No pending tokens."

    return f"MIDI tokens ({len(tokens)}): {'.'.join(tokens)}"


def _midi_flush(session: Any) -> str:
    """Flush pending tokens as a dot-separated protocol string."""
    ctrl = _get_controller(session)
    if ctrl is None:
        return "MIDI: No controller active."

    result = ctrl.get_token_string()
    if not result:
        return "MIDI: No pending tokens to flush."

    return f"MIDI protocol: {result}"


# ============================================================================
# Helpers
# ============================================================================

def _note_name(midi_note: int) -> str:
    """Convert MIDI note number to display name."""
    names = ['C', 'C#', 'D', 'D#', 'E', 'F',
             'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    return f"{names[midi_note % 12]}{octave}"


# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

MIDI_COMMANDS = {
    'midi': cmd_midi,
}


def get_midi_commands() -> dict:
    """Return all MIDI commands for registration."""
    return MIDI_COMMANDS.copy()
