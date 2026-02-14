"""MIDI Input Module — Phase 6.0: MIDI as Protocol Writer.

Provides MIDI device management, interval translation, chord detection,
and preview triggering for MDMA.  MIDI acts purely as a protocol-writing
accelerator — it converts MIDI note events into interval tokens that the
existing DSL (/mel, /cor) already understands.

MIDI does NOT introduce: sequencing, automation, clips, or a real-time
DAW layer.  MDMA remains deterministic and render-first.

Components:
    MIDIDeviceManager  — enumerate, select, open/close MIDI input ports
    IntervalTranslator — convert MIDI notes to interval tokens with chord detection
    PreviewTrigger     — play short preview tones through the Monolith engine

Dependencies (optional):
    mido >= 1.3.0
    python-rtmidi >= 1.5.0

BUILD ID: midi_input_v1.0_phase6
"""

from __future__ import annotations

import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

# ============================================================================
# Optional MIDI import — graceful degradation when mido is not installed
# ============================================================================

_MIDI_AVAILABLE = False
try:
    import mido  # type: ignore
    _MIDI_AVAILABLE = True
except ImportError:
    mido = None  # type: ignore


def midi_available() -> bool:
    """Return True if mido + rtmidi backend are installed."""
    return _MIDI_AVAILABLE


# ============================================================================
# MIDI DEVICE MANAGER
# ============================================================================

class MIDIDeviceManager:
    """Manage MIDI input device enumeration, selection, and note capture.

    Usage::

        mgr = MIDIDeviceManager()
        devices = mgr.list_devices()
        mgr.open(devices[0])
        mgr.set_callback(my_note_handler)
        ...
        mgr.close()
    """

    def __init__(self) -> None:
        self._port: Any = None
        self._port_name: Optional[str] = None
        self._callback: Optional[Callable[[int, int, bool], None]] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Device enumeration
    # ------------------------------------------------------------------

    @staticmethod
    def list_devices() -> List[str]:
        """Return list of available MIDI input port names."""
        if not _MIDI_AVAILABLE:
            return []
        try:
            return mido.get_input_names()  # type: ignore[union-attr]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------------

    def open(self, port_name: str) -> bool:
        """Open a MIDI input port by name.  Returns True on success."""
        if not _MIDI_AVAILABLE:
            return False
        self.close()
        try:
            self._port = mido.open_input(port_name)  # type: ignore[union-attr]
            self._port_name = port_name
            self._running = True
            self._listener_thread = threading.Thread(
                target=self._listen_loop, daemon=True,
            )
            self._listener_thread.start()
            return True
        except Exception:
            self._port = None
            self._port_name = None
            return False

    def close(self) -> None:
        """Close the currently open MIDI port."""
        self._running = False
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
            self._port = None
            self._port_name = None
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=1.0)
            self._listener_thread = None

    @property
    def is_open(self) -> bool:
        return self._port is not None and self._running

    @property
    def device_name(self) -> Optional[str]:
        return self._port_name

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def set_callback(self, callback: Callable[[int, int, bool], None]) -> None:
        """Set the note callback: callback(note, velocity, is_note_on)."""
        self._callback = callback

    # ------------------------------------------------------------------
    # Internal listener
    # ------------------------------------------------------------------

    def _listen_loop(self) -> None:
        """Background thread: read MIDI messages and dispatch note events."""
        while self._running and self._port is not None:
            try:
                for msg in self._port.iter_pending():
                    if msg.type == 'note_on':
                        is_on = msg.velocity > 0
                        if self._callback:
                            self._callback(msg.note, msg.velocity, is_on)
                    elif msg.type == 'note_off':
                        if self._callback:
                            self._callback(msg.note, 0, False)
            except Exception:
                break
            time.sleep(0.002)  # ~2ms poll interval — low latency


# ============================================================================
# INTERVAL TRANSLATOR
# ============================================================================

class IntervalTranslator:
    """Convert MIDI note events into interval protocol tokens.

    Handles:
    - Single note → interval integer (e.g. ``0``, ``7``, ``-1``)
    - Chord detection → parenthesised group (e.g. ``(0,4,7)``)
    - Duration extension → ``.`` appended on repeated press
    - Rest insertion → ``_`` token

    The chord detection window groups notes arriving within
    ``window_ms`` milliseconds into a single chord token.
    """

    def __init__(self, root_note: int = 60, window_ms: int = 80) -> None:
        self.root_note: int = root_note
        self.window_ms: int = window_ms

        # Pending note buffer for chord detection
        self._pending_notes: List[int] = []
        self._window_start: float = 0.0
        self._flush_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

        # Callback invoked when a complete token is ready
        self._token_callback: Optional[Callable[[str], None]] = None

        # Track last emitted position for duration extension
        self._last_note: Optional[int] = None
        self._last_was_rest: bool = False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_root(self, midi_note: int) -> None:
        """Set the root note for interval calculation."""
        self.root_note = midi_note

    def set_window(self, ms: int) -> None:
        """Set the chord detection window in milliseconds."""
        self.window_ms = max(10, min(500, ms))

    def set_token_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback invoked with each complete protocol token string."""
        self._token_callback = callback

    # ------------------------------------------------------------------
    # Note input
    # ------------------------------------------------------------------

    def note_on(self, midi_note: int) -> None:
        """Process an incoming MIDI note-on event."""
        interval = midi_note - self.root_note
        now = time.monotonic()

        with self._lock:
            # If we're outside the chord window, flush pending and start new
            if self._pending_notes and (now - self._window_start) * 1000 > self.window_ms:
                self._flush_pending()

            if not self._pending_notes:
                self._window_start = now

            self._pending_notes.append(interval)

            # Cancel any existing flush timer and set a new one
            if self._flush_timer is not None:
                self._flush_timer.cancel()
            self._flush_timer = threading.Timer(
                self.window_ms / 1000.0, self._timer_flush,
            )
            self._flush_timer.daemon = True
            self._flush_timer.start()

    def insert_rest(self) -> None:
        """Insert an explicit rest token ``_``."""
        with self._lock:
            self._flush_pending()
        self._last_was_rest = True
        self._last_note = None
        if self._token_callback:
            self._token_callback('_')

    # ------------------------------------------------------------------
    # Token generation
    # ------------------------------------------------------------------

    def _timer_flush(self) -> None:
        """Called by the timer when the chord window expires."""
        with self._lock:
            self._flush_pending()

    def _flush_pending(self) -> None:
        """Flush pending notes into a protocol token."""
        if not self._pending_notes:
            return

        notes = sorted(self._pending_notes)
        self._pending_notes = []

        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None

        if len(notes) == 1:
            token = str(notes[0])
            self._last_note = notes[0]
        else:
            # Chord grouping: (0,4,7)
            token = '(' + ','.join(str(n) for n in notes) + ')'
            self._last_note = None  # chords don't support repeated-press extension

        self._last_was_rest = False

        if self._token_callback:
            self._token_callback(token)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def midi_to_note_name(midi_note: int) -> str:
        """Convert MIDI note number to name, e.g. 60 → 'C4'."""
        names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_note // 12) - 1
        return f"{names[midi_note % 12]}{octave}"

    @staticmethod
    def note_name_to_midi(name: str) -> int:
        """Convert note name to MIDI number, e.g. 'C4' → 60."""
        name = name.strip().upper()
        note_map = {
            'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
        }
        if len(name) >= 3 and name[1] in ('#', 'B'):
            letter = name[0]
            mod = 1 if name[1] == '#' else -1
            octave = int(name[2:])
            base = note_map.get(letter, 0) + mod
        elif len(name) >= 2:
            letter = name[0]
            octave = int(name[1:])
            base = note_map.get(letter, 0)
        else:
            return 60
        return base + (octave + 1) * 12


# ============================================================================
# PREVIEW TRIGGER
# ============================================================================

class PreviewTrigger:
    """Play short preview tones when MIDI notes arrive in Preview Mode.

    Generates a brief tone (~200ms) through the Monolith engine and plays
    it via the session's playback system.  Non-blocking.
    """

    def __init__(self, session: Any) -> None:
        self._session = session
        self._preview_duration: float = 0.2  # seconds
        self._preview_amp: float = 0.5

    def trigger(self, midi_note: int, velocity: int = 100) -> None:
        """Play a preview tone for the given MIDI note.

        Uses the current Monolith engine state (waveform, filters, etc.)
        to render a short tone and plays it non-blocking.
        """
        try:
            freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
            amp = self._preview_amp * (velocity / 127.0)

            session = self._session
            sr = session.sample_rate

            # Render through the Monolith engine if available
            if hasattr(session, 'engine') and hasattr(session.engine, 'generate_tone'):
                audio = session.engine.generate_tone(
                    frequency=freq,
                    duration=self._preview_duration,
                    amplitude=amp,
                    sample_rate=sr,
                )
            else:
                # Fallback: simple sine tone
                samples = int(self._preview_duration * sr)
                t = np.arange(samples) / sr
                audio = np.sin(2 * np.pi * freq * t).astype(np.float64) * amp
                # Quick fade to prevent clicks
                fade = min(256, samples // 4)
                if fade > 0:
                    audio[:fade] *= np.linspace(0, 1, fade)
                    audio[-fade:] *= np.linspace(1, 0, fade)

            # Play non-blocking
            if hasattr(session, 'play_buffer'):
                session.play_buffer(audio)
            elif hasattr(session, 'play'):
                session.play(audio)
            else:
                # Direct sounddevice fallback
                try:
                    import sounddevice as sd  # type: ignore
                    sd.play(audio, sr)
                except Exception:
                    pass
        except Exception:
            pass  # Preview failures are silent — never interrupt workflow


# ============================================================================
# MIDI INPUT CONTROLLER — ties everything together
# ============================================================================

class MIDIInputController:
    """Top-level controller that wires device manager, translator, and preview.

    This is the single object that the command layer creates and stores
    on the session.  It handles:
    - Device lifecycle (list, open, close)
    - Mode switching (preview vs program)
    - Note routing (preview plays tone, program emits tokens)
    """

    def __init__(self, session: Any) -> None:
        self.session = session
        self.device_manager = MIDIDeviceManager()
        self.translator = IntervalTranslator(
            root_note=getattr(session, 'midi_root_note', 60),
            window_ms=getattr(session, 'chord_window_ms', 80),
        )
        self.preview = PreviewTrigger(session)

        # Mode: 'preview' or 'program'
        self.mode: str = 'preview'

        # Token buffer for program mode — collected tokens before user
        # presses enter.  The GUI reads from here to inject into the
        # input field.
        self.token_buffer: List[str] = []
        self._token_lock = threading.Lock()

        # Wire translator callback
        self.translator.set_token_callback(self._on_token)

        # Wire device manager callback
        self.device_manager.set_callback(self._on_midi_note)

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def list_devices(self) -> List[str]:
        return self.device_manager.list_devices()

    def connect(self, port_name: str) -> bool:
        """Open a MIDI input device."""
        ok = self.device_manager.open(port_name)
        if ok:
            self.session.midi_device = port_name
            self.session.midi_enabled = True
        return ok

    def disconnect(self) -> None:
        """Close the current MIDI input device."""
        self.device_manager.close()
        self.session.midi_device = None
        self.session.midi_enabled = False

    @property
    def is_connected(self) -> bool:
        return self.device_manager.is_open

    # ------------------------------------------------------------------
    # Mode control
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Set input mode: 'preview' or 'program'."""
        if mode in ('preview', 'program'):
            self.mode = mode

    # ------------------------------------------------------------------
    # Settings sync
    # ------------------------------------------------------------------

    def sync_settings(self) -> None:
        """Sync translator settings from session state."""
        self.translator.set_root(self.session.midi_root_note)
        self.translator.set_window(self.session.chord_window_ms)

    # ------------------------------------------------------------------
    # Token buffer (for GUI integration)
    # ------------------------------------------------------------------

    def pop_tokens(self) -> List[str]:
        """Return and clear any pending tokens from program mode."""
        with self._token_lock:
            tokens = self.token_buffer[:]
            self.token_buffer.clear()
        return tokens

    def get_token_string(self) -> str:
        """Return pending tokens as a dot-separated protocol string."""
        tokens = self.pop_tokens()
        return '.'.join(tokens)

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _on_midi_note(self, note: int, velocity: int, is_on: bool) -> None:
        """Handle incoming MIDI note event."""
        if not is_on:
            return  # Phase 6.0 only uses note-on

        if self.mode == 'preview':
            self.preview.trigger(note, velocity)
        elif self.mode == 'program':
            self.translator.note_on(note)

    def _on_token(self, token: str) -> None:
        """Handle a completed token from the interval translator."""
        with self._token_lock:
            self.token_buffer.append(token)
