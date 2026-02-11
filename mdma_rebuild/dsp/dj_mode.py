"""MDMA DJ Mode - Live Performance and Remix Environment.

DJ Mode is additive - it loads DJ-specific engines, commands, and workflows
without disabling any existing MDMA functionality.

Core Philosophy:
- Accessibility-first (screen reader native, command-driven)
- Engine-native audio handling (wave float 64 internally)
- AI assists without hijacking control
- Safety and fallback over spectacle

Features:
- Device discovery and routing
- Headphone/cue bus system
- Slow-fallback safety net
- VDA (Virtual Audio Device) integration
- NVDA screen reader routing
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import threading
import time

if TYPE_CHECKING:
    from ..core.session import Session


# ============================================================================
# DEVICE TYPES AND STRUCTURES
# ============================================================================

class DeviceType(Enum):
    MASTER = "master"
    HEADPHONES = "headphones"
    VIRTUAL = "virtual"
    BLUETOOTH = "bluetooth"
    USB = "usb"
    BUILTIN = "builtin"


class BusType(Enum):
    MASTER = "master"
    CUE = "cue"
    MONITOR = "monitor"


@dataclass
class AudioDevice:
    """Represents an audio output device."""
    id: str
    name: str
    device_type: DeviceType
    channels: int = 2
    sample_rate: int = 48000
    latency_ms: float = 10.0
    connected: bool = False
    locked: bool = False
    
    def __str__(self):
        status = "●" if self.connected else "○"
        lock = " [LOCKED]" if self.locked else ""
        return f"{status} [{self.id}] {self.name} ({self.device_type.value}){lock} ~{self.latency_ms:.0f}ms"


@dataclass
class DJDeck:
    """Represents a DJ deck with audio and state."""
    id: int
    buffer: Optional[np.ndarray] = None
    position: float = 0.0  # Current position in seconds
    tempo: float = 120.0
    pitch: float = 1.0  # Pitch adjustment (1.0 = normal)
    playing: bool = False
    cue_point: float = 0.0
    loop_start: float = 0.0
    loop_end: float = 0.0
    loop_active: bool = False
    loop_hold: bool = False  # Infinite loop-hold toggle (LH)
    loop_count: int = 4  # Number of loops to play before releasing
    loop_remaining: int = 0  # Loops remaining (0 = infinite when loop_active)
    volume: float = 1.0
    transition_volume: float = 1.0  # Volume during transition (for ducking)
    eq_low: float = 1.0
    eq_mid: float = 1.0
    eq_high: float = 1.0
    filter_cutoff: float = 50.0  # 1-100, 50 = neutral
    filter_resonance: float = 20.0  # 1-100, higher = more resonance (Q)
    
    # Stutter
    stutter_active: bool = False
    stutter_position: float = 0.0  # Stutter start position
    stutter_length: float = 0.0  # Stutter chunk length in seconds
    stutter_remaining: int = 0  # Stutters remaining
    
    # Routing
    cue_enabled: bool = False  # Send to cue bus
    master_enabled: bool = True  # Send to master bus
    
    # Buffer integration
    source_buffer_idx: Optional[int] = None  # MDMA buffer index this came from
    
    # Stem separation
    stems: Optional[Dict] = None  # StemSet if separated
    active_stems: Dict[str, float] = field(default_factory=lambda: {
        'vocals': 1.0, 'drums': 1.0, 'bass': 1.0, 'other': 1.0
    })
    
    # Sections/chops
    sections: List = field(default_factory=list)
    chops: List = field(default_factory=list)
    current_section: int = 0
    
    # Analysis
    analyzed: bool = False
    analysis_data: Dict = field(default_factory=dict)


@dataclass
class SlowFallbackState:
    """State for the slow-fallback safety system."""
    active: bool = False
    trigger_reason: str = ""
    started_at: float = 0.0
    loop_extended: bool = False
    tempo_locked: bool = False
    filter_sweep_active: bool = False
    energy_held: float = 0.0


# ============================================================================
# DJ MODE ENGINE
# ============================================================================

class DJModeEngine:
    """Core DJ Mode engine managing decks, routing, and safety."""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.enabled = False
        
        # Devices
        self.devices: Dict[str, AudioDevice] = {}
        self.master_device: Optional[str] = None
        self.headphone_device: Optional[str] = None
        self.nvda_device: Optional[str] = None
        
        # Decks (2 by default, expandable)
        self.decks: Dict[int, DJDeck] = {
            1: DJDeck(id=1),
            2: DJDeck(id=2),
        }
        self.active_deck: int = 1
        
        # Crossfader
        self.crossfader: float = 0.5  # 0 = deck 1, 1 = deck 2
        self.crossfader_curve: str = "linear"  # linear, smooth, cut
        
        # Buses
        self.master_volume: float = 1.0

        # Audio output settings
        # block_size controls the PortAudio callback buffer size.
        # Larger values increase stability and reduce CPU spikes at the cost
        # of a little more latency.
        self.block_size: int = 2048

        # Master loop (wrap) behavior
        # When enabled, any playing deck that reaches the end of its buffer
        # will wrap to the beginning and continue playing (even if no loop is
        # active). This is intended as a safety/performance feature.
        self.master_loop_wrap: bool = False

        # Master Deck FX (MDFX)
        # Always-on post-processing applied to the final DJ mix.
        self.master_deck_fx_chain: list[tuple[str, dict]] = []

        # Per-deck timed effect state (so one deck doesn't steal effects)
        self._active_deck_effects: dict[int, dict] = {}
        self.cue_volume: float = 1.0
        self.cue_mix: float = 0.0  # 0 = cue only, 1 = master only
        
        # Safety
        self.fallback = SlowFallbackState()
        self.last_interaction: float = time.time()
        self.fallback_timeout: float = 30.0  # seconds
        
        # Bypass mode - use session playback instead of device output
        self.bypass_deck_output: bool = False
        self._bypass_session: Optional['Session'] = None
        
        # VDA
        self.vda_routes: Dict[str, str] = {}
        
        # Transitions
        self._active_transition: Optional[Dict] = None
        
        # Duration override (None = use tempo-based 1 bar default)
        self.default_duration: Optional[float] = None
        
        # Filter sweeps
        self._active_filter_sweep: Optional[Dict] = None
        
        # Threading
        self._playback_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None
        self._running = False
    
    def enable(self) -> str:
        """Enable DJ Mode."""
        if self.enabled:
            return "DJ Mode already enabled"
        
        self.enabled = True
        self._scan_devices()
        self.last_interaction = time.time()
        
        # Auto-select default output device
        device_msg = ""
        default_id = self.get_default_device()
        if default_id:
            dev = self.devices[default_id]
            self.master_device = default_id
            dev.connected = True
            device_msg = f"\n  Output: {dev.name} (auto-selected)"
        
        return (f"DJ Mode ENABLED\n"
                f"  Decks: {len(self.decks)}\n"
                f"  Devices: {len(self.devices)}{device_msg}\n"
                f"  Use /DO to list output devices\n"
                f"  Use /DECK <n> to select deck")
    
    def disable(self) -> str:
        """Disable DJ Mode."""
        if not self.enabled:
            return "DJ Mode not enabled"
        
        self._stop_all()
        self.enabled = False
        return "DJ Mode DISABLED"
    
    def _scan_devices(self) -> int:
        """Scan for audio devices. Returns count found."""
        self.devices.clear()
        
        # Track the system default device ID
        default_output_id = None
        
        # Try to use sounddevice for real device enumeration
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            
            # Get the default output device
            try:
                default_info = sd.query_devices(kind='output')
                default_output_id = str(sd.default.device[1])  # Output device index
            except Exception:
                pass
            
            for i, dev in enumerate(devices):
                if dev['max_output_channels'] > 0:
                    # Determine device type
                    name_lower = dev['name'].lower()
                    if 'bluetooth' in name_lower or 'bt' in name_lower:
                        dtype = DeviceType.BLUETOOTH
                    elif 'usb' in name_lower:
                        dtype = DeviceType.USB
                    elif 'virtual' in name_lower or 'vda' in name_lower or 'cable' in name_lower:
                        dtype = DeviceType.VIRTUAL
                    elif 'headphone' in name_lower or 'earphone' in name_lower:
                        dtype = DeviceType.HEADPHONES
                    else:
                        dtype = DeviceType.BUILTIN
                    
                    # Mark if this is the system default
                    dev_name = dev['name']
                    is_default = (str(i) == default_output_id)
                    if is_default:
                        dev_name = f"{dev['name']} [Default]"
                    
                    self.devices[str(i)] = AudioDevice(
                        id=str(i),
                        name=dev_name,
                        device_type=dtype,
                        channels=dev['max_output_channels'],
                        sample_rate=int(dev['default_samplerate']),
                        latency_ms=dev['default_low_output_latency'] * 1000,
                    )
            
            # Store the default device ID for reference
            self._system_default_device = default_output_id
            
            return len(self.devices)
            
        except (ImportError, OSError):
            # sounddevice not available or PortAudio not installed
            pass
        except Exception:
            # Any other error, fall back to simulated devices
            pass
        
        # Fallback: create simulated devices
        self.devices = {
            '0': AudioDevice('0', 'System Default Output [Default]', DeviceType.BUILTIN),
            '1': AudioDevice('1', 'Speakers', DeviceType.BUILTIN),
            '2': AudioDevice('2', 'Headphones', DeviceType.HEADPHONES),
            '3': AudioDevice('3', 'Bluetooth Speaker', DeviceType.BLUETOOTH, latency_ms=50),
            '4': AudioDevice('4', 'USB Audio Interface', DeviceType.USB, latency_ms=5),
        }
        self._system_default_device = '0'
        return len(self.devices)
    
    def connect_master(self, device_id: str) -> str:
        """Connect master output to a device."""
        if device_id not in self.devices:
            return f"ERROR: device '{device_id}' not found. Use /DO to list."
        
        dev = self.devices[device_id]
        if dev.locked:
            return f"ERROR: device '{dev.name}' is locked by another process"
        
        # Disconnect previous
        if self.master_device and self.master_device in self.devices:
            self.devices[self.master_device].connected = False
        
        dev.connected = True
        self.master_device = device_id
        
        return (f"OK: Master output connected to {dev.name}\n"
                f"  Latency: ~{dev.latency_ms:.0f}ms\n"
                f"  Channels: {dev.channels}")
    
    def connect_headphones(self, device_id: str) -> str:
        """Connect headphones/cue output."""
        if device_id not in self.devices:
            return f"ERROR: device '{device_id}' not found. Use /HEP to scan."
        
        dev = self.devices[device_id]
        if dev.locked:
            return f"ERROR: device '{dev.name}' is locked"
        
        # Disconnect previous
        if self.headphone_device and self.headphone_device in self.devices:
            self.devices[self.headphone_device].connected = False
        
        dev.connected = True
        self.headphone_device = device_id
        
        return (f"OK: Headphones connected to {dev.name}\n"
                f"  Latency: ~{dev.latency_ms:.0f}ms\n"
                f"  Cue bus routed to headphones")
    
    def get_default_device(self) -> Optional[str]:
        """Get or auto-select default audio device.
        
        Uses system default output device if available. Falls back to:
        1. Device marked as [Default]
        2. First BUILTIN device
        3. USB devices
        4. Any available device
        
        Returns
        -------
        str or None
            Device ID of the default device
        """
        # Return master if already explicitly set by user via /DOC
        if self.master_device and self.master_device in self.devices:
            return self.master_device
        
        # Scan devices if none available
        if not self.devices:
            self._scan_devices()
        
        # No devices at all
        if not self.devices:
            return None
        
        # Use stored system default if available
        if hasattr(self, '_system_default_device') and self._system_default_device:
            if self._system_default_device in self.devices:
                return self._system_default_device
        
        # Look for device marked as [Default] in name
        for dev_id, dev in self.devices.items():
            if '[default]' in dev.name.lower():
                return dev_id
        
        # Look for device named "default" or "system"
        for dev_id, dev in self.devices.items():
            name_lower = dev.name.lower()
            if 'default' in name_lower or 'system' in name_lower:
                return dev_id
        
        # Priority order for auto-selection
        priority_order = [
            DeviceType.BUILTIN,    # System default first
            DeviceType.USB,        # USB interfaces
            DeviceType.HEADPHONES, # Headphones
            DeviceType.BLUETOOTH,  # Bluetooth last (higher latency)
            DeviceType.VIRTUAL,    # Virtual devices
        ]
        
        # Select by priority
        for dtype in priority_order:
            for dev_id, dev in self.devices.items():
                if dev.device_type == dtype and not dev.locked:
                    return dev_id
        
        # Fall back to first available
        return list(self.devices.keys())[0] if self.devices else None
    
    def ensure_playback_device(self) -> str:
        """Ensure a playback device is selected, auto-selecting if needed.
        
        Returns
        -------
        str
            Status message
        """
        if self.master_device and self.master_device in self.devices:
            return f"Using device: {self.devices[self.master_device].name}"
        
        # Auto-select default device
        default_id = self.get_default_device()
        if default_id:
            dev = self.devices[default_id]
            self.master_device = default_id
            dev.connected = True
            return f"Auto-selected output: {dev.name}"
        
        return "WARNING: No audio output device available"
    
    def list_devices(self, filter_type: Optional[DeviceType] = None) -> str:
        """List available devices."""
        if not self.devices:
            self._scan_devices()
        
        lines = ["=== AUDIO OUTPUT DEVICES ===", ""]
        
        # Group by type
        groups = {
            'Master Output Devices': [DeviceType.BUILTIN, DeviceType.USB],
            'Headphones / Cue Devices': [DeviceType.HEADPHONES, DeviceType.BLUETOOTH],
            'Virtual / Routing Devices': [DeviceType.VIRTUAL],
        }
        
        for group_name, types in groups.items():
            group_devs = [d for d in self.devices.values() 
                         if d.device_type in types and (filter_type is None or d.device_type == filter_type)]
            if group_devs:
                lines.append(f"{group_name}:")
                for dev in group_devs:
                    marker = ""
                    if dev.id == self.master_device:
                        marker = " [MASTER]"
                    elif dev.id == self.headphone_device:
                        marker = " [CUE/HEP]"
                    lines.append(f"  {dev}{marker}")
                lines.append("")
        
        lines.append("Commands: /DOC <id> connect master, /HEPC <id> connect headphones")
        return '\n'.join(lines)
    
    def scan_headphones(self) -> str:
        """Scan specifically for headphone devices."""
        self._scan_devices()
        
        hep_types = [DeviceType.HEADPHONES, DeviceType.BLUETOOTH]
        hep_devs = [d for d in self.devices.values() if d.device_type in hep_types]
        
        if not hep_devs:
            return "No headphone devices found. Connect Bluetooth or wired headphones."
        
        lines = ["=== HEADPHONE DEVICES ===", ""]
        for dev in hep_devs:
            marker = " [CONNECTED]" if dev.id == self.headphone_device else ""
            lines.append(f"  {dev}{marker}")
        lines.append("")
        lines.append("Use /HEPC <id> to connect")
        return '\n'.join(lines)
    
    # ========================================================================
    # DECK OPERATIONS
    # ========================================================================
    
    def load_to_deck(self, deck_id: int, audio: np.ndarray) -> str:
        """Load audio to a deck."""
        try:
            if deck_id not in self.decks:
                self.decks[deck_id] = DJDeck(id=deck_id)
            
            deck = self.decks[deck_id]
            
            # Validate audio
            if audio is None:
                return f"ERROR: no audio data provided"
            
            if not isinstance(audio, np.ndarray):
                try:
                    audio = np.array(audio)
                except Exception as e:
                    return f"ERROR: invalid audio data: {e}"
            
            if len(audio) == 0:
                return f"ERROR: audio data is empty"
            
            # Convert to float64, handling edge cases
            if audio.dtype != np.float64:
                audio = audio.astype(np.float64)
            
            # Ensure 1D (mono) - take first channel if stereo
            if audio.ndim > 1:
                audio = audio[:, 0] if audio.shape[1] > audio.shape[0] else audio[0, :]
            
            deck.buffer = audio
            deck.position = 0.0
            deck.playing = False
            deck.transition_volume = 1.0  # Reset transition volume
            
            dur = len(audio) / self.sample_rate
            return f"OK: loaded {dur:.2f}s to Deck {deck_id}"
            
        except Exception as e:
            return f"ERROR: failed to load deck: {e}"
    
    def play_deck(self, deck_id: int, session: Optional['Session'] = None) -> str:
        """Start playback on a deck.
        
        If bypass_deck_output is True or device output fails, uses session playback.
        """
        try:
            if deck_id not in self.decks:
                return f"ERROR: Deck {deck_id} not found"
            
            deck = self.decks[deck_id]
            if deck.buffer is None:
                return f"ERROR: Deck {deck_id} is empty"
            
            if len(deck.buffer) == 0:
                return f"ERROR: Deck {deck_id} has no audio data"
            
            # Ensure position is valid
            max_pos = len(deck.buffer) / self.sample_rate
            if deck.position < 0:
                deck.position = 0.0
            elif deck.position >= max_pos:
                deck.position = 0.0  # Reset to start if at end
            
            # Store session for bypass playback
            if session is not None:
                self._bypass_session = session
            
            # Use bypass mode if enabled
            if self.bypass_deck_output:
                return self._play_deck_bypass(deck_id)
            
            # Auto-select output device if none connected
            device_msg = ""
            if not self.master_device:
                try:
                    device_msg = self.ensure_playback_device()
                    if device_msg.startswith("Auto"):
                        device_msg = f" ({device_msg})"
                    elif device_msg.startswith("WARNING"):
                        device_msg = f"\n⚠️ {device_msg}"
                    else:
                        device_msg = ""
                except Exception as e:
                    device_msg = f"\n⚠️ Device selection error: {e}"
                    # Fall back to bypass mode if device selection fails
                    return self._play_deck_bypass(deck_id, reason="device error")
            
            deck.playing = True
            self.last_interaction = time.time()
            
            # Start audio output thread if not running
            try:
                self._start_audio_output()
            except Exception as e:
                # Fall back to bypass playback
                return self._play_deck_bypass(deck_id, reason=str(e))
            
            return f"OK: Deck {deck_id} PLAYING{device_msg}"
            
        except Exception as e:
            return f"ERROR: play_deck failed: {e}"
    
    def _play_deck_bypass(self, deck_id: int, reason: str = None) -> str:
        """Play deck using session's simpler playback (bypass mode)."""
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = self.decks[deck_id]
        if deck.buffer is None or len(deck.buffer) == 0:
            return f"ERROR: Deck {deck_id} is empty"
        
        # Apply deck settings (volume, EQ, etc.)
        audio = deck.buffer.copy()
        audio = audio * deck.volume
        
        # Get playback position
        start_sample = int(deck.position * self.sample_rate)
        if start_sample >= len(audio):
            start_sample = 0
        audio = audio[start_sample:]
        
        # Try session playback
        if self._bypass_session is not None:
            try:
                if self._bypass_session._play_buffer(audio, 0.8):
                    duration = len(audio) / self.sample_rate
                    reason_str = f" (bypass: {reason})" if reason else " (bypass mode)"
                    return f"PLAYING: Deck {deck_id} ({duration:.2f}s){reason_str}"
            except Exception as e:
                return f"ERROR: Bypass playback failed: {e}"
        
        # Last resort - try using dsp.playback directly
        try:
            from .playback import play
            if play(audio, self.sample_rate, blocking=False, volume=0.8):
                duration = len(audio) / self.sample_rate
                reason_str = f" (direct: {reason})" if reason else " (direct)"
                return f"PLAYING: Deck {deck_id} ({duration:.2f}s){reason_str}"
        except Exception as e:
            return f"ERROR: All playback methods failed: {e}"
        
        return "ERROR: No playback method available"
    
    def pause_deck(self, deck_id: int) -> str:
        """Pause playback on a deck."""
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        self.decks[deck_id].playing = False
        self.last_interaction = time.time()
        return f"OK: Deck {deck_id} PAUSED"
    
    def cue_deck(self, deck_id: int, enable: Optional[bool] = None) -> str:
        """Toggle or set cue for a deck."""
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = self.decks[deck_id]
        if enable is None:
            deck.cue_enabled = not deck.cue_enabled
        else:
            deck.cue_enabled = enable
        
        status = "ON" if deck.cue_enabled else "OFF"
        return f"OK: Deck {deck_id} CUE {status}"
    
    def set_crossfader(self, value: float) -> str:
        """Set crossfader position (0-1)."""
        self.crossfader = max(0.0, min(1.0, value))
        self.last_interaction = time.time()
        
        if self.crossfader < 0.1:
            pos = "DECK 1"
        elif self.crossfader > 0.9:
            pos = "DECK 2"
        else:
            pos = f"MIX ({self.crossfader:.0%})"
        
        return f"OK: Crossfader at {pos}"
    
    def deck_status(self) -> str:
        """Get status of all decks."""
        lines = ["=== DJ DECK STATUS ===", ""]
        
        for deck_id, deck in sorted(self.decks.items()):
            status = "▶" if deck.playing else "⏸"
            cue = "🎧" if deck.cue_enabled else ""
            stems_str = "🎚" if deck.stems else ""
            analyzed_str = "📊" if deck.analyzed else ""
            
            if deck.buffer is not None:
                dur = len(deck.buffer) / self.sample_rate
                pos_str = f"{deck.position:.1f}s / {dur:.1f}s"
            else:
                pos_str = "empty"
            
            lines.append(f"  Deck {deck_id}: {status} {pos_str} {cue}{stems_str}{analyzed_str}")
            lines.append(f"    Tempo: {deck.tempo:.1f} BPM | Pitch: {deck.pitch:.2f}x")
            lines.append(f"    Vol: {deck.volume:.0%} | EQ: L{deck.eq_low:.1f} M{deck.eq_mid:.1f} H{deck.eq_high:.1f}")
            if deck.stems:
                stem_names = list(deck.active_stems.keys())
                lines.append(f"    Stems: {', '.join(stem_names)}")
        
        lines.append("")
        lines.append(f"Crossfader: {'█' * int(self.crossfader * 10)}{'░' * (10 - int(self.crossfader * 10))} ({self.crossfader:.0%})")
        
        return '\n'.join(lines)
    
    # ========================================================================
    # TRANSITION SYSTEM
    # ========================================================================
    
    def create_transition(
        self,
        from_deck: int,
        to_deck: int,
        duration: float = None,  # None = 1 bar at current tempo
        style: str = "crossfade",
        curve: str = "equal_power",  # equal_power gives better ducking
        sync_tempo: bool = True,
    ) -> str:
        """Create and execute a transition between decks.
        
        Parameters
        ----------
        from_deck : int
            Source deck
        to_deck : int
            Destination deck
        duration : float, optional
            Transition duration in seconds. Default is 1 bar (4 beats) at current tempo.
        style : str
            Transition style: crossfade, cut, echo_out, filter_sweep,
            spinback, brake, reverb_tail, stutter, backspin
        curve : str
            Volume curve: equal_power (default, best ducking), linear, smooth, exponential
        sync_tempo : bool
            Whether to sync tempos during transition
        
        Returns
        -------
        str
            Status message
        """
        if from_deck not in self.decks:
            return f"ERROR: Deck {from_deck} not found"
        if to_deck not in self.decks:
            return f"ERROR: Deck {to_deck} not found"
        
        src = self.decks[from_deck]
        dst = self.decks[to_deck]
        
        if dst.buffer is None:
            return f"ERROR: Deck {to_deck} is empty"
        
        # Get duration - use user override if set, otherwise calculate from tempo
        if duration is None:
            duration = self.get_effective_duration(from_deck)
        
        # Sync tempos if requested
        if sync_tempo and src.buffer is not None and src.tempo:
            dst.tempo = src.tempo
        
        self.last_interaction = time.time()
        
        # Store transition state
        self._active_transition = {
            'from_deck': from_deck,
            'to_deck': to_deck,
            'style': style,
            'curve': curve,
            'duration': duration,
            'started_at': time.time(),
            'completed': False,
        }
        
        # Start destination deck if not playing
        if not dst.playing:
            dst.playing = True
        
        # Start audio output if not running
        self._start_audio_output()
        
        tempo_str = f" ({src.tempo:.1f} BPM)" if src.tempo else ""
        return (f"OK: Transition started\n"
                f"  {from_deck} → {to_deck}\n"
                f"  Style: {style}, Duration: {duration:.2f}s{tempo_str}\n"
                f"  Curve: {curve}")
    
    def _get_transition_volumes(self, progress: float, curve: str) -> Tuple[float, float]:
        """Calculate volume levels for source and destination decks.
        
        Returns (source_volume, dest_volume) with proper ducking.
        """
        progress = max(0.0, min(1.0, progress))
        
        if curve == 'equal_power':
            # Equal power crossfade - maintains constant perceived volume
            # Uses sine/cosine curves for smooth ducking
            import math
            angle = progress * math.pi / 2  # 0 to π/2
            src_vol = math.cos(angle)
            dst_vol = math.sin(angle)
        
        elif curve == 'smooth' or curve == 's_curve':
            # S-curve (smoothstep) - gradual start and end
            t = progress
            smooth = t * t * (3 - 2 * t)  # Smoothstep function
            src_vol = 1.0 - smooth
            dst_vol = smooth
        
        elif curve == 'exponential':
            # Exponential - fast out, slow in
            import math
            src_vol = math.exp(-3 * progress)
            dst_vol = 1.0 - math.exp(-3 * progress)
        
        elif curve == 'logarithmic':
            # Logarithmic - slow out, fast in
            import math
            if progress < 0.01:
                src_vol = 1.0
                dst_vol = 0.0
            else:
                src_vol = max(0, 1.0 - math.log10(1 + 9 * progress))
                dst_vol = math.log10(1 + 9 * progress)
        
        else:  # linear
            src_vol = 1.0 - progress
            dst_vol = progress
        
        return (src_vol, dst_vol)
    
    def _process_transition(self) -> None:
        """Process active transition - called from audio loop."""
        try:
            # Safely get _active_transition (might not exist in older instances)
            trans = getattr(self, '_active_transition', None)
            if not trans or trans.get('completed'):
                return
            
            elapsed = time.time() - trans.get('started_at', time.time())
            duration = trans.get('duration', 2.0)
            if duration <= 0:
                duration = 2.0
            progress = min(1.0, elapsed / duration)
            
            from_deck = trans.get('from_deck')
            to_deck = trans.get('to_deck')
            
            if from_deck is None or to_deck is None:
                self._active_transition = None
                return
            
            # Get volume levels based on curve
            curve = trans.get('curve', 'equal_power')
            src_vol, dst_vol = self._get_transition_volumes(progress, curve)
            
            # Apply volumes to decks (safely)
            if from_deck in self.decks:
                self.decks[from_deck].transition_volume = src_vol
            if to_deck in self.decks:
                self.decks[to_deck].transition_volume = dst_vol
            
            # Update crossfader position to match transition
            if from_deck == 1 and to_deck == 2:
                self.crossfader = progress
            elif from_deck == 2 and to_deck == 1:
                self.crossfader = 1.0 - progress
            
            # Check if transition completed
            if progress >= 1.0:
                trans['completed'] = True
                
                # Stop the source deck
                if from_deck in self.decks:
                    self.decks[from_deck].playing = False
                    self.decks[from_deck].transition_volume = 1.0  # Reset for next use
                
                # Ensure destination is at full volume
                if to_deck in self.decks:
                    self.decks[to_deck].transition_volume = 1.0
                
                # Clear active transition
                self._active_transition = None
                
        except Exception:
            # On any error, clear the transition to prevent repeated errors
            self._active_transition = None
    
    def quick_transition(
        self,
        to_deck: int = None,
        style: str = "crossfade",
        duration: float = None,  # None = 1 bar
    ) -> str:
        """Quick transition to next deck or specified deck.
        
        Parameters
        ----------
        to_deck : int, optional
            Target deck (None = next loaded deck)
        style : str
            Transition style
        duration : float, optional
            Transition duration (None = 1 bar at current tempo)
        """
        # Find current playing deck
        playing_decks = [d for d in self.decks.values() if d.playing]
        if not playing_decks:
            return "ERROR: no deck currently playing"
        
        from_deck = playing_decks[0].id
        
        # Find target deck
        if to_deck is None:
            # Find next loaded deck that's not playing
            loaded_decks = [d for d in self.decks.values() 
                          if d.buffer is not None and not d.playing]
            if not loaded_decks:
                return "ERROR: no other loaded deck to transition to"
            to_deck = loaded_decks[0].id
        
        return self.create_transition(from_deck, to_deck, duration, style)
    
    def get_transition_styles(self) -> Dict[str, str]:
        """Get available transition styles with descriptions."""
        return {
            'crossfade': 'Smooth volume crossfade between decks',
            'cut': 'Hard cut (instant switch)',
            'echo_out': 'Echo/delay tail on outgoing deck',
            'filter_sweep': 'Low-pass filter sweep on outgoing',
            'spinback': 'Vinyl spinback effect on outgoing',
            'brake': 'Turntable brake/slowdown effect',
            'reverb_tail': 'Reverb wash on outgoing deck',
            'stutter': 'Stutter/glitch transition',
            'backspin': 'Quick backspin then cut',
            'eq_swap': 'EQ frequency swap between decks',
            'phaser': 'Phaser sweep transition',
            'gate': 'Rhythmic gate pattern transition',
        }
    
    # ========================================================================
    # FILTER CONTROLS
    # ========================================================================
    
    def set_duration(self, duration: Optional[float]) -> str:
        """Set default duration for DJ commands.
        
        Parameters
        ----------
        duration : float or None
            Duration in seconds, or None to use tempo-based default (1 bar)
        """
        if duration is not None and duration <= 0:
            return "ERROR: duration must be positive"
        
        self.default_duration = duration
        
        if duration is None:
            return "OK: Duration set to AUTO (1 bar at current tempo)"
        else:
            return f"OK: Duration set to {duration:.2f}s"
    
    def get_effective_duration(self, deck_id: int = None) -> float:
        """Get the effective duration for commands.
        
        Returns user-set duration, or calculates 1 bar from tempo.
        """
        if self.default_duration is not None:
            return self.default_duration
        
        # Calculate 1 bar from tempo
        tempo = 120.0  # Default
        if deck_id and deck_id in self.decks:
            deck = self.decks[deck_id]
            if deck.tempo and deck.tempo > 0:
                tempo = deck.tempo
        elif self.active_deck in self.decks:
            deck = self.decks[self.active_deck]
            if deck.tempo and deck.tempo > 0:
                tempo = deck.tempo
        
        beats_per_second = tempo / 60.0
        return 4.0 / beats_per_second  # 4 beats = 1 bar
    
    def filter_sweep(
        self,
        deck_id: int,
        direction: str = "down",  # "up" = highpass, "down" = lowpass
        amount: float = None,  # 1-100 sweep amount (None = full sweep)
        duration: float = None,
        auto_reset: bool = True,  # Auto-reset to neutral after duration
    ) -> str:
        """Start a filter sweep on a deck.
        
        Parameters
        ----------
        deck_id : int
            Deck to apply filter to
        direction : str
            "up" = sweep to highpass (cut lows)
            "down" = sweep to lowpass (cut highs)
            "reset" = return to neutral
        amount : float, optional
            Sweep amount 1-100 (None = full sweep to 5 or 95)
        duration : float, optional
            Sweep duration (None = use default duration)
        auto_reset : bool
            If True, auto-reset to neutral after sweep completes
        """
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = self.decks[deck_id]
        
        if duration is None:
            duration = self.get_effective_duration(deck_id)
        
        # Determine target based on direction (1-100 scale, 50 = neutral)
        if direction == "up" or direction == "hp" or direction == "highpass":
            # Highpass - cut lows (cutoff > 50)
            if amount is not None:
                # Amount controls how far from neutral (50) we go
                amount = max(1, min(100, amount))
                target = 50 + (amount / 2)  # 1-100 -> 50.5-100
            else:
                target = 95  # Full highpass
            filter_type = "highpass"
        elif direction == "down" or direction == "lp" or direction == "lowpass":
            # Lowpass - cut highs (cutoff < 50)
            if amount is not None:
                amount = max(1, min(100, amount))
                target = 50 - (amount / 2)  # 1-100 -> 49.5-0
            else:
                target = 5  # Full lowpass
            filter_type = "lowpass"
        elif direction == "reset" or direction == "off" or direction == "neutral":
            target = 50
            filter_type = "neutral"
            duration = min(duration, 0.5)  # Quick reset
            auto_reset = False  # No need to reset after reset
        else:
            return f"ERROR: unknown direction '{direction}'. Use up/down/reset"
        
        # Store sweep state
        self._active_filter_sweep = {
            'deck_id': deck_id,
            'start_cutoff': deck.filter_cutoff,
            'target_cutoff': target,
            'filter_type': filter_type,
            'duration': duration,
            'started_at': time.time(),
            'completed': False,
            'auto_reset': auto_reset,
            'reset_started': False,
        }
        
        self.last_interaction = time.time()
        
        reset_str = " (auto-reset)" if auto_reset else ""
        return (f"OK: Filter sweep {filter_type}{reset_str}\n"
                f"  Deck {deck_id}: {deck.filter_cutoff:.0f} → {target:.0f}\n"
                f"  Duration: {duration:.2f}s, Resonance: {deck.filter_resonance:.0f}")
    
    def set_filter(self, deck_id: int, cutoff: float) -> str:
        """Set filter cutoff directly.
        
        Parameters
        ----------
        deck_id : int
            Deck to apply filter to
        cutoff : float
            Filter cutoff 1-100 (50=neutral, <50=lowpass, >50=highpass)
        """
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        cutoff = max(1, min(100, cutoff))
        self.decks[deck_id].filter_cutoff = cutoff
        
        if cutoff < 40:
            mode = "LOWPASS"
        elif cutoff > 60:
            mode = "HIGHPASS"
        else:
            mode = "NEUTRAL"
        
        return f"OK: Deck {deck_id} filter at {cutoff:.0f} ({mode})"
    
    def set_resonance(self, deck_id: int, resonance: float) -> str:
        """Set filter resonance (Q).
        
        Parameters
        ----------
        deck_id : int
            Deck to set resonance for
        resonance : float
            Resonance amount 1-100 (higher = more resonant peak)
        """
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        resonance = max(1, min(100, resonance))
        self.decks[deck_id].filter_resonance = resonance
        
        if resonance < 30:
            style = "subtle"
        elif resonance < 60:
            style = "moderate"
        elif resonance < 80:
            style = "aggressive"
        else:
            style = "EXTREME"
        
        return f"OK: Deck {deck_id} resonance at {resonance:.0f} ({style})"
    
    def reset_filter(self, deck_id: int = None) -> str:
        """Reset filter to neutral.
        
        Parameters
        ----------
        deck_id : int, optional
            Deck to reset (None = all decks)
        """
        if deck_id is not None:
            if deck_id not in self.decks:
                return f"ERROR: Deck {deck_id} not found"
            self.decks[deck_id].filter_cutoff = 50
            return f"OK: Deck {deck_id} filter reset to neutral (50)"
        else:
            for deck in self.decks.values():
                deck.filter_cutoff = 50
            return f"OK: All filters reset to neutral (50)"
    
    def _process_filter_sweep(self) -> None:
        """Process active filter sweep - called from audio loop."""
        try:
            sweep = getattr(self, '_active_filter_sweep', None)
            if not sweep or sweep.get('completed'):
                return
            
            elapsed = time.time() - sweep.get('started_at', time.time())
            duration = sweep.get('duration', 1.0)
            if duration <= 0:
                duration = 1.0
            
            deck_id = sweep.get('deck_id')
            if deck_id not in self.decks:
                self._active_filter_sweep = None
                return
            
            # Check if we're in reset phase
            if sweep.get('reset_started'):
                # Calculate reset progress
                reset_elapsed = time.time() - sweep.get('reset_start_time', time.time())
                reset_duration = duration * 0.5  # Reset takes half the original duration
                reset_progress = min(1.0, reset_elapsed / reset_duration)
                
                # Interpolate back to neutral (50)
                start = sweep.get('peak_cutoff', 50)
                target = 50
                t = reset_progress
                smooth = t * t * (3 - 2 * t)
                current = start + (target - start) * smooth
                
                self.decks[deck_id].filter_cutoff = current
                
                if reset_progress >= 1.0:
                    sweep['completed'] = True
                    self.decks[deck_id].filter_cutoff = 50  # Ensure exact neutral
                    self._active_filter_sweep = None
                return
            
            # Normal sweep progress
            progress = min(1.0, elapsed / duration)
            
            # Interpolate cutoff
            start = sweep.get('start_cutoff', 50)
            target = sweep.get('target_cutoff', 50)
            
            # Use smooth interpolation
            t = progress
            smooth = t * t * (3 - 2 * t)  # Smoothstep
            current = start + (target - start) * smooth
            
            self.decks[deck_id].filter_cutoff = current
            
            if progress >= 1.0:
                # Sweep complete - check for auto-reset
                if sweep.get('auto_reset', True):
                    # Start reset phase
                    sweep['reset_started'] = True
                    sweep['reset_start_time'] = time.time()
                    sweep['peak_cutoff'] = current
                else:
                    sweep['completed'] = True
                    self._active_filter_sweep = None
                
        except Exception:
            self._active_filter_sweep = None
    
    def _apply_filter(self, audio: np.ndarray, cutoff: float, resonance: float = 20.0) -> np.ndarray:
        """Apply filter to audio based on cutoff position with resonance.
        
        Parameters
        ----------
        audio : np.ndarray
            Audio chunk to filter
        cutoff : float
            Filter cutoff 1-100 (50=neutral, <50=lowpass, >50=highpass)
        resonance : float
            Filter resonance 1-100 (higher = more resonant peak)
        
        Uses resonant filter for classic DJ sweep sound.
        """
        # Stereo: apply filter independently per channel.
        if getattr(audio, 'ndim', 1) == 2 and audio.shape[1] >= 2:
            left = self._apply_filter(audio[:, 0].astype(np.float64), cutoff, resonance)
            right = self._apply_filter(audio[:, 1].astype(np.float64), cutoff, resonance)
            return np.column_stack((left, right))

        # Near neutral - skip filtering
        if abs(cutoff - 50) < 5:
            return audio
        
        # Normalize cutoff to 0-1 range for filter coefficient calculation
        # 1 -> 0.0, 50 -> 0.5, 100 -> 1.0
        norm_cutoff = cutoff / 100.0
        
        # Calculate resonance (Q factor)
        # resonance 1 = Q of 0.5 (no resonance)
        # resonance 100 = Q of 10 (very resonant)
        q = 0.5 + (resonance / 100.0) * 9.5  # Maps 1-100 to 0.5-10
        
        # Calculate filter coefficients
        # Using a simple 2-pole resonant filter (biquad approximation)
        if cutoff < 50:
            # Lowpass mode
            # Map 1-50 to filter frequency (lower = more filtering)
            freq = norm_cutoff * 2  # 0.0-1.0
            freq = max(0.02, freq)  # Prevent instability
            
            # Simple resonant lowpass
            # y[n] = a0*x[n] + a1*x[n-1] + b1*y[n-1] + b2*y[n-2]
            w0 = freq * 3.14159  # Normalized frequency
            alpha = w0 / (2 * q)
            
            # Clamp alpha to prevent instability
            alpha = max(0.01, min(0.99, alpha))
            
            # Apply filter with feedback for resonance
            filtered = np.empty_like(audio)
            y1 = 0.0
            y2 = 0.0
            
            # Resonance feedback coefficient
            fb = min(0.98, (q - 0.5) / 10.0 * 0.95)
            
            for i in range(len(audio)):
                # Lowpass with resonance
                filtered[i] = alpha * audio[i] + (1.0 - alpha) * y1 + fb * (y1 - y2)
                y2 = y1
                y1 = filtered[i]
            
            return filtered
            
        else:
            # Highpass mode
            # Map 50-100 to filter frequency (higher = more bass cut)
            freq = (norm_cutoff - 0.5) * 2  # 0.0-1.0
            freq = max(0.02, min(0.98, freq))
            
            # Simple resonant highpass
            filtered = np.empty_like(audio)
            x1 = 0.0
            y1 = 0.0
            y2 = 0.0
            
            # Highpass coefficient
            hp_coef = 1.0 - freq
            hp_coef = max(0.02, min(0.98, hp_coef))
            
            # Resonance feedback
            fb = min(0.95, (q - 0.5) / 10.0 * 0.9)
            
            for i in range(len(audio)):
                # Highpass with resonance
                hp = hp_coef * (y1 + audio[i] - x1)
                filtered[i] = hp + fb * (y1 - y2)
                x1 = audio[i]
                y2 = y1
                y1 = filtered[i]
            
            return filtered
    
    # ========================================================================
    # LOOP COUNT AND STUTTER
    # ========================================================================
    
    def set_loop_count(self, deck_id: int, count: int) -> str:
        """Set the number of loops to play when loop is triggered.
        
        Parameters
        ----------
        deck_id : int
            Deck to set loop count for
        count : int
            Number of loops (1-64)
        """
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        count = max(1, min(64, count))
        self.decks[deck_id].loop_count = count
        return f"OK: Deck {deck_id} loop count set to {count}"
    
    def get_beat_duration(self, deck_id: int = None) -> float:
        """Get duration of 1 beat in seconds based on deck tempo."""
        tempo = 120.0  # Default
        if deck_id and deck_id in self.decks:
            deck = self.decks[deck_id]
            if deck.tempo and deck.tempo > 0:
                tempo = deck.tempo
        elif self.active_deck in self.decks:
            deck = self.decks[self.active_deck]
            if deck.tempo and deck.tempo > 0:
                tempo = deck.tempo
        
        return 60.0 / tempo  # Seconds per beat
    
    def trigger_loop(self, deck_id: int, beats: float = None) -> str:
        """Trigger a counted loop at current position.
        
        Parameters
        ----------
        deck_id : int
            Deck to loop
        beats : float, optional
            Loop length in beats (None = 4 beats = 1 bar)
        """
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = self.decks[deck_id]
        
        if deck.buffer is None:
            return f"ERROR: Deck {deck_id} is empty"
        
        # Calculate loop duration
        beat_dur = self.get_beat_duration(deck_id)
        if beats is None:
            beats = 4.0  # Default 1 bar
        
        loop_duration = beat_dur * beats
        
        # Set loop points from current position
        deck.loop_start = deck.position
        deck.loop_end = deck.position + loop_duration
        
        # Clamp to buffer length
        max_pos = len(deck.buffer) / self.sample_rate
        if deck.loop_end > max_pos:
            deck.loop_end = max_pos
        
        # Set loop count and activate
        deck.loop_remaining = deck.loop_count
        deck.loop_active = True
        
        self.last_interaction = time.time()
        
        return (f"OK: Deck {deck_id} loop triggered\n"
                f"  {deck.loop_start:.2f}s → {deck.loop_end:.2f}s ({loop_duration:.3f}s)\n"
                f"  {deck.loop_count} loops ({beats} beats each)")

    def toggle_loop_hold(self, deck_id: int, beats: float = None) -> str:
        """Toggle loop-hold (infinite loop) for a deck.

        Loop-hold is additional behavior on top of existing loop-count/LG.
        When enabled, the deck loops indefinitely over a window starting at
        the current playhead.

        Parameters
        ----------
        deck_id : int
            Deck id.
        beats : float, optional
            Loop length in beats (default 4).
        """
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"

        deck = self.decks[deck_id]
        if deck.buffer is None:
            return f"ERROR: Deck {deck_id} is empty"

        if getattr(deck, 'loop_hold', False) and deck.loop_active:
            # Release hold
            deck.loop_hold = False
            deck.loop_active = False
            deck.loop_remaining = 0
            return f"OK: Deck {deck_id} loop-hold released"

        # Enable hold
        beat_dur = self.get_beat_duration(deck_id)
        if beats is None:
            beats = 4.0
        loop_duration = max(0.05, beat_dur * float(beats))
        deck.loop_start = deck.position
        deck.loop_end = deck.position + loop_duration
        max_pos = len(deck.buffer) / self.sample_rate
        if deck.loop_end > max_pos:
            deck.loop_end = max_pos
        deck.loop_active = True
        deck.loop_remaining = 0  # 0 = infinite
        deck.loop_hold = True
        self.last_interaction = time.time()
        return (f"OK: Deck {deck_id} loop-hold ON\n"
                f"  {deck.loop_start:.2f}s → {deck.loop_end:.2f}s ({beats} beats)")

    def set_master_loop_wrap(self, enabled: bool) -> str:
        """Enable or disable master loop wrap.

        When enabled, any deck that reaches the end of its source wraps to the
        beginning and continues playing. This is a safety/performance feature
        and does not replace normal loop controls.
        """
        self.master_loop_wrap = bool(enabled)
        return f"OK: Master loop wrap {'ON' if self.master_loop_wrap else 'OFF'}"
    
    def release_loop(self, deck_id: int) -> str:
        """Release/exit loop on a deck."""
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = self.decks[deck_id]
        deck.loop_active = False
        deck.loop_remaining = 0
        if hasattr(deck, 'loop_hold'):
            deck.loop_hold = False
        
        return f"OK: Deck {deck_id} loop released"
    
    def trigger_stutter(self, deck_id: int, beats: float = 1.0) -> str:
        """Trigger stutter effect using loop count.
        
        Creates a rapid repeat of a small chunk of audio.
        
        Parameters
        ----------
        deck_id : int
            Deck to stutter
        beats : float
            Stutter chunk length in beats (default 1 beat)
        """
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = self.decks[deck_id]
        
        if deck.buffer is None:
            return f"ERROR: Deck {deck_id} is empty"
        
        # Calculate stutter duration
        beat_dur = self.get_beat_duration(deck_id)
        stutter_length = beat_dur * beats
        
        # Set stutter state
        deck.stutter_active = True
        deck.stutter_position = deck.position
        deck.stutter_length = stutter_length
        deck.stutter_remaining = deck.loop_count  # Use loop count for repetitions
        
        self.last_interaction = time.time()
        
        return (f"OK: Deck {deck_id} stutter triggered\n"
                f"  {deck.loop_count}x repeats of {stutter_length:.3f}s ({beats} beat{'s' if beats != 1 else ''})")
    
    def stop_stutter(self, deck_id: int) -> str:
        """Stop stutter effect on a deck."""
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = self.decks[deck_id]
        deck.stutter_active = False
        deck.stutter_remaining = 0
        
        return f"OK: Deck {deck_id} stutter stopped"
    
    # ========================================================================
    # JUMP COMMAND
    # ========================================================================
    
    def jump_to(
        self,
        deck_id: int,
        target: str = None,
        beat: int = None,
        time_sec: float = None,
    ) -> str:
        """Jump to a point in the track.
        
        Parameters
        ----------
        deck_id : int
            Deck to jump
        target : str, optional
            Named target: 'start', 'end', 'drop', 'breakdown', 'chorus', 
            'verse', 'buildup', 'outro', 'half', 'quarter'
        beat : int, optional
            Jump to specific beat number
        time_sec : float, optional
            Jump to specific time in seconds
        """
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = self.decks[deck_id]
        
        if deck.buffer is None:
            return f"ERROR: Deck {deck_id} is empty"
        
        max_time = len(deck.buffer) / self.sample_rate
        new_pos = None
        target_name = ""
        
        # Time in seconds
        if time_sec is not None:
            new_pos = max(0, min(max_time, time_sec))
            target_name = f"{new_pos:.2f}s"
        
        # Beat number
        elif beat is not None:
            beat_dur = self.get_beat_duration(deck_id)
            new_pos = beat * beat_dur
            new_pos = max(0, min(max_time, new_pos))
            target_name = f"beat {beat}"
        
        # Named target
        elif target:
            target = target.lower()
            
            if target == 'start':
                new_pos = 0.0
                target_name = "start"
            elif target == 'end':
                new_pos = max(0, max_time - 0.1)
                target_name = "end"
            elif target == 'half':
                new_pos = max_time / 2
                target_name = "halfway"
            elif target == 'quarter':
                new_pos = max_time / 4
                target_name = "quarter"
            elif target == 'cue':
                new_pos = deck.cue_point
                target_name = f"cue ({new_pos:.2f}s)"
            else:
                # Try to find section in analysis data
                if deck.analyzed and deck.analysis_data:
                    sections = deck.analysis_data.get('sections', [])
                    for sec in sections:
                        if sec.get('type', '').lower() == target:
                            new_pos = sec.get('start', 0)
                            target_name = f"{target} section"
                            break
                
                if new_pos is None:
                    # Estimate based on common structure
                    estimates = {
                        'drop': 0.4,      # 40% into track
                        'breakdown': 0.5,  # Middle
                        'buildup': 0.35,   # Before drop
                        'chorus': 0.3,     # First chorus ~30%
                        'verse': 0.15,     # First verse ~15%
                        'outro': 0.85,     # Near end
                        'intro': 0.0,      # Start
                    }
                    if target in estimates:
                        new_pos = max_time * estimates[target]
                        target_name = f"{target} (estimated)"
        
        if new_pos is None:
            return "ERROR: No valid jump target specified"
        
        # Perform jump
        old_pos = deck.position
        deck.position = new_pos
        
        # Exit any active loop
        if deck.loop_active:
            deck.loop_active = False
        
        self.last_interaction = time.time()
        
        return f"OK: Deck {deck_id} jumped to {target_name}\n  {old_pos:.2f}s → {new_pos:.2f}s"
    
    # ========================================================================
    # SCRATCH COMMAND
    # ========================================================================
    
    def trigger_scratch(
        self,
        deck_id: int,
        preset: int = 1,
        duration: float = None,
    ) -> str:
        """Trigger a deterministic scratch pattern.
        
        Parameters
        ----------
        deck_id : int
            Deck to scratch
        preset : int
            Scratch preset (1-5)
        duration : float, optional
            Scratch duration (None = uses loop_count * beat duration)
        
        Presets:
            1 = Baby scratch (back-forth)
            2 = Forward scratch
            3 = Chirp scratch
            4 = Transformer scratch
            5 = Crab scratch (rapid)
        """
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = self.decks[deck_id]
        
        if deck.buffer is None:
            return f"ERROR: Deck {deck_id} is empty"
        
        # Calculate duration
        if duration is None:
            beat_dur = self.get_beat_duration(deck_id)
            duration = beat_dur * deck.loop_count
        
        preset = max(1, min(5, preset))
        
        preset_names = {
            1: 'Baby',
            2: 'Forward',
            3: 'Chirp',
            4: 'Transformer',
            5: 'Crab',
        }
        
        # Store scratch state with vinyl physics
        self._active_scratch = {
            'deck_id': deck_id,
            'preset': preset,
            'duration': duration,
            'started_at': time.time(),
            'start_position': deck.position,
            'last_position': deck.position,
            'last_time': time.time(),
            'velocity': 0.0,  # Current scratch velocity (samples/sec)
            'completed': False,
        }
        
        self.last_interaction = time.time()
        
        return (f"OK: Deck {deck_id} scratch triggered\n"
                f"  Preset: {preset} ({preset_names[preset]})\n"
                f"  Duration: {duration:.2f}s")
    
    def _process_scratch(self) -> None:
        """Process active scratch with vinyl/tape physics."""
        scratch = getattr(self, '_active_scratch', None)
        if not scratch or scratch.get('completed'):
            return
        
        elapsed = time.time() - scratch.get('started_at', time.time())
        duration = scratch.get('duration', 1.0)
        if duration <= 0:
            duration = 1.0
        
        progress = elapsed / duration
        if progress >= 1.0:
            scratch['completed'] = True
            deck_id = scratch.get('deck_id')
            if deck_id in self.decks:
                self.decks[deck_id].position = scratch.get('start_position', 0)
                self.decks[deck_id].pitch = 1.0  # Reset pitch
            self._active_scratch = None
            return
        
        deck_id = scratch.get('deck_id')
        if deck_id not in self.decks:
            self._active_scratch = None
            return
        
        deck = self.decks[deck_id]
        preset = scratch.get('preset', 1)
        start_pos = scratch.get('start_position', deck.position)
        
        import math
        
        # Scratch range in seconds (how much audio to scratch through)
        scratch_range = 0.15  # 150ms
        
        # Calculate target position based on preset
        # Each preset defines a position curve over time
        if preset == 1:  # Baby scratch - smooth back and forth
            cycles = 4
            pos_curve = math.sin(progress * cycles * 2 * math.pi)
            target_pos = start_pos + scratch_range * pos_curve
        
        elif preset == 2:  # Forward scratch - quick push, slow return
            cycles = 3
            t = (progress * cycles) % 1.0
            if t < 0.25:
                # Quick forward push
                pos_curve = t / 0.25
            else:
                # Slow drag back
                pos_curve = 1.0 - (t - 0.25) / 0.75
            target_pos = start_pos + scratch_range * pos_curve
        
        elif preset == 3:  # Chirp - quick cuts with fader simulation
            cycles = 6
            t = (progress * cycles) % 1.0
            # Quick forward chirp with volume cut simulation
            if t < 0.15:
                pos_curve = t / 0.15
            elif t < 0.3:
                pos_curve = 1.0 - (t - 0.15) / 0.15
            else:
                pos_curve = 0  # "Fader closed"
            target_pos = start_pos + scratch_range * 0.5 * pos_curve
        
        elif preset == 4:  # Transformer - gated scratching
            cycles = 8
            t = (progress * cycles) % 1.0
            # Forward motion with transformer cuts
            base_pos = progress * scratch_range * 0.5
            gate = 1.0 if (int(t * 4) % 2 == 0) else 0.0
            target_pos = start_pos + base_pos
            # Store gate state for audio processing
            scratch['gate'] = gate
        
        else:  # Crab (5) - rapid fluttering
            cycles = 16
            t = progress * cycles
            # Rapid small movements
            pos_curve = math.sin(t * 2 * math.pi) * 0.3
            # Add forward drift
            pos_curve += progress * 0.5
            target_pos = start_pos + scratch_range * pos_curve
        
        # Calculate velocity (for pitch bending)
        current_time = time.time()
        last_time = scratch.get('last_time', current_time)
        last_pos = scratch.get('last_position', target_pos)
        dt = current_time - last_time
        
        if dt > 0.001:  # Avoid division by zero
            # Velocity in position-units per second
            velocity = (target_pos - last_pos) / dt
            # Convert to pitch multiplier
            # Normal playback = 1.0, forward = >1.0, backward = <0
            # Vinyl physics: pitch = velocity / normal_velocity
            normal_velocity = 1.0  # 1 second of audio per second
            pitch = velocity / normal_velocity
            
            # Clamp pitch to reasonable range
            pitch = max(-3.0, min(3.0, pitch))
            
            # Apply pitch to deck
            deck.pitch = pitch
            
            # Store for next iteration
            scratch['velocity'] = velocity
            scratch['last_position'] = target_pos
            scratch['last_time'] = current_time
        
        # Update position
        deck.position = target_pos
        
        # Clamp to valid range
        max_pos = len(deck.buffer) / self.sample_rate
        deck.position = max(0, min(max_pos - 0.01, deck.position))
    
    def _get_scratch_audio(self, deck: "DJDeck", num_samples: int) -> np.ndarray:
        """Get audio for scratching with proper tape/vinyl pitch bending.
        
        This implements proper varispeed playback where pitch changes
        based on scratch velocity.
        """
        scratch = getattr(self, '_active_scratch', None)
        if not scratch or scratch.get('deck_id') != deck.id:
            return None
        
        if deck.buffer is None:
            return None
        
        # Get current pitch (set by _process_scratch)
        pitch = deck.pitch
        
        # Handle transformer gate
        gate = scratch.get('gate', 1.0)
        
        if abs(pitch) < 0.01:
            # Near-zero pitch = silence
            return np.zeros(num_samples)
        
        # Calculate how many source samples we need
        source_samples_needed = int(abs(num_samples * pitch)) + 2
        
        # Get position in samples
        pos_samples = int(deck.position * self.sample_rate)
        
        # Handle forward vs backward playback
        if pitch >= 0:
            # Forward playback
            start = pos_samples
            end = min(start + source_samples_needed, len(deck.buffer))
            if start >= len(deck.buffer):
                return np.zeros(num_samples)
            source = deck.buffer[start:end].astype(np.float64)
        else:
            # Backward playback (reverse)
            end = pos_samples
            start = max(0, end - source_samples_needed)
            if end <= 0:
                return np.zeros(num_samples)
            source = deck.buffer[start:end].astype(np.float64)[::-1]  # Reverse
        
        if len(source) < 2:
            return np.zeros(num_samples)
        
        # Resample to achieve pitch shift (varispeed)
        # Source length / output length = pitch ratio
        x_source = np.linspace(0, 1, len(source))
        x_output = np.linspace(0, 1, num_samples)
        
        try:
            output = np.interp(x_output, x_source, source)
        except Exception:
            output = np.zeros(num_samples)
        
        # Apply transformer gate if active
        if gate < 1.0:
            output = output * gate
        
        return output
    
    # ========================================================================
    # DECK EFFECTS (with main effect module support)
    # ========================================================================
    
    def apply_deck_effect(
        self,
        deck_id: int,
        effect_name: str,
        duration: float = None,
        amount: float = 50.0,
    ) -> str:
        """Apply a timed effect to a deck.
        
        Parameters
        ----------
        deck_id : int
            Deck to apply effect to
        effect_name : str
            Effect name - can be:
            - Built-in: echo, delay, flanger, phaser, crush, filter
            - Vamp: vamp, vamp_light, vamp_medium, vamp_heavy, vamp_fuzz
            - Reverb: reverb, reverb_small, reverb_large, reverb_plate, etc.
            - Delay: delay_simple, delay_pingpong, delay_tape, etc.
            - Saturation: saturate_soft, saturate_hard, saturate_tube, etc.
            - Lo-fi: lofi_bitcrush, lofi_chorus, lofi_flanger, lofi_phaser
            - Dynamics: compress_mild, compress_hard, compress_limiter
            - Aliases: r1, r2, d1, v1, s1, etc.
        duration : float, optional
            Effect duration (None = 1 bar)
        amount : float
            Effect amount 1-100
        """
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        if duration is None:
            duration = self.get_effective_duration(deck_id)
        
        amount = max(1, min(100, amount))
        
        # Resolve effect name through aliases
        resolved_effect = self._resolve_effect_name(effect_name.lower())
        
        # Store per-deck timed effect state (so decks don't stomp each other)
        self._active_deck_effects[deck_id] = {
            'deck_id': deck_id,
            'effect': resolved_effect,
            'original_name': effect_name,
            'duration': duration,
            'amount': amount,
            'started_at': time.time(),
            'completed': False,
            'use_main_fx': self._is_main_effect(resolved_effect),
        }
        
        self.last_interaction = time.time()
        
        return (f"OK: Deck {deck_id} effect {resolved_effect}\n"
                f"  Amount: {amount:.0f}, Duration: {duration:.2f}s")
    
    def _resolve_effect_name(self, name: str) -> str:
        """Resolve effect name through aliases."""
        # Effect alias mapping (same as fx_cmds.py)
        aliases = {
            # Vamp shortcuts
            'vamp': 'vamp_medium',
            'amp': 'vamp_medium',
            'v1': 'vamp_light',
            'v2': 'vamp_medium',
            'v3': 'vamp_heavy',
            'v4': 'vamp_fuzz',
            # Reverb shortcuts
            'reverb': 'reverb_large',
            'r1': 'reverb_small',
            'r2': 'reverb_large',
            'r3': 'reverb_plate',
            'r4': 'reverb_spring',
            'r5': 'reverb_cathedral',
            # Delay shortcuts
            'delay': 'delay_simple',
            'd1': 'delay_simple',
            'd2': 'delay_pingpong',
            'd3': 'delay_multitap',
            'd4': 'delay_slapback',
            'd5': 'delay_tape',
            # Saturation shortcuts
            'sat': 'saturate_soft',
            's1': 'saturate_soft',
            's2': 'saturate_hard',
            's3': 'saturate_overdrive',
            's4': 'saturate_fuzz',
            's5': 'saturate_tube',
            # Overdrive shortcuts
            'od': 'overdrive_classic',
            'overdrive': 'overdrive_classic',
            'o1': 'overdrive_soft',
            'o2': 'overdrive_classic',
            'o3': 'overdrive_crunch',
            # Lo-fi shortcuts
            'lofi': 'lofi_bitcrush',
            'l1': 'lofi_bitcrush',
            'l2': 'lofi_chorus',
            'l3': 'lofi_flanger',
            'l4': 'lofi_phaser',
            'l5': 'lofi_filter',
            'l6': 'lofi_halftime',
            # Dynamics shortcuts
            'comp': 'compress_mild',
            'c1': 'compress_mild',
            'c2': 'compress_hard',
            'c3': 'compress_limiter',
            # Gate shortcuts
            'gate': 'gate3',
            'g1': 'gate1',
            'g2': 'gate2',
            'g3': 'gate3',
            'g4': 'gate4',
            'g5': 'gate5',
            # Convolution reverb
            'shimmer': 'conv_shimmer',
            'hall': 'conv_hall',
            # Built-in aliases
            'echo': 'echo',
            'flanger': 'flanger',
            'phaser': 'phaser',
            'crush': 'crush',
            'bitcrush': 'crush',
            'filter': 'filter',
        }
        return aliases.get(name, name)
    
    def _is_main_effect(self, effect_name: str) -> bool:
        """Check if effect is from main effects module."""
        main_effects = {
            'reverb_small', 'reverb_large', 'reverb_plate', 'reverb_spring', 'reverb_cathedral',
            'conv_hall', 'conv_hall_long', 'conv_room', 'conv_plate', 'conv_spring', 'conv_shimmer', 'conv_reverse',
            'delay_simple', 'delay_pingpong', 'delay_multitap', 'delay_slapback', 'delay_tape',
            'saturate_soft', 'saturate_hard', 'saturate_overdrive', 'saturate_fuzz', 'saturate_tube',
            'vamp_light', 'vamp_medium', 'vamp_heavy', 'vamp_fuzz',
            'overdrive_soft', 'overdrive_classic', 'overdrive_crunch',
            'dual_od_warm', 'dual_od_bright', 'dual_od_heavy',
            'waveshape_fold', 'waveshape_rectify', 'waveshape_sine',
            'compress_mild', 'compress_hard', 'compress_limiter', 'compress_expander', 'compress_softclipper',
            'gate1', 'gate2', 'gate3', 'gate4', 'gate5',
            'lofi_bitcrush', 'lofi_chorus', 'lofi_flanger', 'lofi_phaser', 'lofi_filter', 'lofi_halftime',
        }
        return effect_name in main_effects
    
    def _process_deck_effect(self, audio: np.ndarray, deck_id: int) -> np.ndarray:
        """Apply active deck effect to audio chunk."""
        effect = self._active_deck_effects.get(deck_id)
        if not effect or effect.get('completed'):
            return audio
        
        elapsed = time.time() - effect.get('started_at', time.time())
        duration = effect.get('duration', 1.0)
        
        if elapsed >= duration:
            effect['completed'] = True
            self._active_deck_effects.pop(deck_id, None)
            return audio
        
        progress = elapsed / duration
        amount = effect.get('amount', 50) / 100.0
        effect_type = effect.get('effect', '')
        
        # Check if we should use main effects module
        if effect.get('use_main_fx'):
            return self._apply_main_effect(audio, effect_type, amount, progress)
        
        # Built-in effects (optimized for real-time with any buffer size)
        audio = audio.copy()  # Don't modify original
        
        if effect_type in ('echo', 'delay'):
            # Echo with persistent delay buffer
            delay_ms = 150
            delay_samples = int(delay_ms * self.sample_rate / 1000)
            
            # Initialize delay buffer if needed
            if not hasattr(self, '_delay_buffer') or len(self._delay_buffer) != delay_samples:
                self._delay_buffer = np.zeros(delay_samples, dtype=np.float64)
            
            # Process with circular delay buffer
            feedback = 0.4 * amount
            wet = 0.6 * amount
            
            for i in range(len(audio)):
                # Read from delay buffer (circular)
                delay_idx = i % delay_samples
                delayed = self._delay_buffer[delay_idx]
                
                # Mix: output = dry + wet * delayed
                output_sample = audio[i] + wet * delayed
                
                # Write to delay buffer with feedback
                self._delay_buffer[delay_idx] = audio[i] + feedback * delayed
                
                audio[i] = output_sample
        
        elif effect_type == 'flanger':
            import math
            rate = 0.5  # LFO rate Hz
            max_depth = int(0.007 * self.sample_rate)  # 7ms max delay
            
            # Initialize flanger buffer
            if not hasattr(self, '_flanger_buffer'):
                self._flanger_buffer = np.zeros(max_depth + len(audio), dtype=np.float64)
                self._flanger_write_pos = 0
            
            # Resize if needed
            if len(self._flanger_buffer) < max_depth + len(audio):
                self._flanger_buffer = np.zeros(max_depth + len(audio), dtype=np.float64)
            
            depth = int(max_depth * amount)
            
            for i in range(len(audio)):
                # LFO modulates delay time
                lfo = math.sin(2 * math.pi * rate * (elapsed + i / self.sample_rate))
                delay = int((1 + lfo) * depth / 2) + 1
                
                # Write to buffer
                write_idx = (self._flanger_write_pos + i) % len(self._flanger_buffer)
                self._flanger_buffer[write_idx] = audio[i]
                
                # Read delayed sample
                read_idx = (write_idx - delay) % len(self._flanger_buffer)
                delayed = self._flanger_buffer[read_idx]
                
                # Mix
                audio[i] = audio[i] * 0.7 + delayed * 0.3 * amount
            
            self._flanger_write_pos = (self._flanger_write_pos + len(audio)) % len(self._flanger_buffer)
        
        elif effect_type == 'phaser':
            import math
            rate = 0.3  # LFO rate
            stages = 4  # Number of allpass stages
            
            # Initialize phaser state
            if not hasattr(self, '_phaser_state'):
                self._phaser_state = np.zeros(stages, dtype=np.float64)
            
            for i in range(len(audio)):
                # LFO for modulation
                lfo = math.sin(2 * math.pi * rate * (elapsed + i / self.sample_rate))
                
                # Coefficient varies with LFO
                coef = 0.5 + 0.4 * lfo * amount
                coef = max(-0.99, min(0.99, coef))
                
                # Process through allpass chain
                sample = audio[i]
                for stage in range(stages):
                    allpass_out = -coef * sample + self._phaser_state[stage]
                    self._phaser_state[stage] = sample + coef * allpass_out
                    sample = allpass_out
                
                # Mix dry and wet
                audio[i] = audio[i] * 0.5 + sample * 0.5 * amount
        
        elif effect_type in ('crush', 'bitcrush'):
            # Bit reduction - always works
            bits = int(16 - amount * 12)
            bits = max(2, bits)
            levels = 2 ** bits
            audio = np.round(audio * levels) / levels
        
        elif effect_type == 'filter':
            # Filter sweep based on progress
            cutoff = 50 + (100 - 50) * (1 - progress) * amount
            audio = self._apply_filter(audio, cutoff, 30)
        
        return audio
    
    def _apply_main_effect(
        self,
        audio: np.ndarray,
        effect_name: str,
        amount: float,
        progress: float,
    ) -> np.ndarray:
        """Apply effect from main effects module with wet/dry mix."""
        try:
            from ..dsp.effects import _effect_funcs
            
            if effect_name not in _effect_funcs:
                return audio
            
            effect_func = _effect_funcs[effect_name]
            
            # Process through effect
            dry = audio.copy()
            wet = effect_func(audio)
            
            # Wet/dry mix based on amount
            # Amount fades in over first 10% and out over last 10%
            if progress < 0.1:
                fade = progress / 0.1
            elif progress > 0.9:
                fade = (1.0 - progress) / 0.1
            else:
                fade = 1.0
            
            mix = amount * fade
            return dry * (1 - mix) + wet * mix
            
        except Exception:
            return audio

    def apply_fx_chain(self, audio: np.ndarray, fx_chain: list[tuple[str, dict]]) -> np.ndarray:
        """Apply a chain of main effects to mono or stereo audio.

        Parameters
        ----------
        audio : np.ndarray
            Mono (N,) or stereo (N,2) float buffer.
        fx_chain : list[(str, dict)]
            List of (effect_name, params). Params support 'amount' (0-100).

        Returns
        -------
        np.ndarray
            Processed buffer.
        """
        if audio is None or len(audio) == 0 or not fx_chain:
            return audio

        names = [n for (n, _) in fx_chain]
        params = [p or {} for (_, p) in fx_chain]

        try:
            from ..dsp.effects import apply_effects_with_params
        except Exception:
            return audio

        # Effects module is mono-first; for stereo apply per channel.
        if getattr(audio, 'ndim', 1) == 1:
            try:
                return apply_effects_with_params(audio, names, params)
            except Exception:
                return audio

        if audio.ndim == 2 and audio.shape[1] >= 2:
            left = audio[:, 0].astype(np.float64)
            right = audio[:, 1].astype(np.float64)
            try:
                left_p = apply_effects_with_params(left, names, params)
                right_p = apply_effects_with_params(right, names, params)
                out = np.column_stack((left_p, right_p))
                return out
            except Exception:
                return audio

        return audio
    
    def get_available_deck_effects(self) -> dict:
        """Get all available deck effects with descriptions."""
        return {
            # Built-in (low-latency)
            'echo': 'Echo/delay trail',
            'flanger': 'Flanger sweep',
            'phaser': 'Phaser modulation',
            'crush': 'Bitcrush degradation',
            'filter': 'Filter sweep',
            # Vamp/Overdrive
            'vamp': 'Medium amp overdrive (alias: vamp_medium)',
            'vamp_light': 'Light amp warmth',
            'vamp_medium': 'Medium amp overdrive',
            'vamp_heavy': 'Heavy amp distortion',
            'vamp_fuzz': 'Fuzz pedal distortion',
            # Reverb
            'reverb': 'Large hall reverb (alias: reverb_large)',
            'reverb_small': 'Small room reverb',
            'reverb_large': 'Large hall reverb',
            'reverb_plate': 'Plate reverb',
            'shimmer': 'Shimmer reverb',
            # Delay
            'delay': 'Simple delay (alias: delay_simple)',
            'delay_tape': 'Tape echo with wobble',
            'delay_pingpong': 'Ping-pong stereo delay',
            # Saturation
            'sat': 'Soft saturation',
            'saturate_tube': 'Tube saturation',
            # Lo-fi
            'lofi': 'Bit crusher',
            'lofi_chorus': 'Chorus effect',
            'lofi_halftime': 'Half-speed effect',
            # Dynamics
            'comp': 'Mild compressor',
            'compress_hard': 'Hard compressor',
            'gate': 'Gate (medium threshold)',
        }
    
    def apply_transition_effect(
        self,
        audio: np.ndarray,
        style: str,
        progress: float,  # 0-1
        direction: str = "out",  # "out" or "in"
    ) -> np.ndarray:
        """Apply transition effect to audio chunk.
        
        Parameters
        ----------
        audio : np.ndarray
            Audio chunk to process
        style : str
            Transition style
        progress : float
            Transition progress (0-1)
        direction : str
            "out" for outgoing deck, "in" for incoming
        
        Returns
        -------
        np.ndarray
            Processed audio
        """
        if style == 'cut':
            if direction == 'out' and progress > 0.5:
                return np.zeros_like(audio)
            elif direction == 'in' and progress < 0.5:
                return np.zeros_like(audio)
            return audio
        
        elif style == 'crossfade':
            if direction == 'out':
                volume = 1.0 - progress
            else:
                volume = progress
            return audio * volume
        
        elif style == 'echo_out':
            if direction == 'out':
                # Apply echo with increasing feedback
                volume = 1.0 - progress
                delay_samples = int(0.25 * self.sample_rate)
                
                if len(audio) > delay_samples:
                    feedback = progress * 0.7
                    delayed = np.zeros_like(audio)
                    delayed[delay_samples:] = audio[:-delay_samples] * feedback
                    audio = audio * volume + delayed
                else:
                    audio = audio * volume
            else:
                audio = audio * progress
            return audio
        
        elif style == 'filter_sweep':
            if direction == 'out':
                # Low-pass filter with decreasing cutoff
                cutoff = 1.0 - progress * 0.95  # 1.0 -> 0.05
                # Simple first-order lowpass
                alpha = cutoff
                filtered = np.zeros_like(audio)
                filtered[0] = audio[0]
                for i in range(1, len(audio)):
                    filtered[i] = alpha * audio[i] + (1 - alpha) * filtered[i-1]
                audio = filtered * (1.0 - progress * 0.5)
            else:
                audio = audio * progress
            return audio
        
        elif style == 'spinback':
            if direction == 'out' and progress > 0.3:
                # Reverse and pitch down
                spin_progress = (progress - 0.3) / 0.7
                if spin_progress > 0:
                    # Reverse portion of audio
                    reverse_len = int(len(audio) * spin_progress)
                    if reverse_len > 0:
                        audio[-reverse_len:] = audio[-reverse_len:][::-1]
                    audio = audio * (1.0 - spin_progress)
            elif direction == 'in':
                audio = audio * progress
            return audio
        
        elif style == 'brake':
            if direction == 'out':
                # Slow down playback
                speed = 1.0 - progress * 0.9
                if speed > 0.1:
                    new_len = int(len(audio) / speed)
                    x_old = np.linspace(0, 1, len(audio))
                    x_new = np.linspace(0, 1, new_len)
                    audio = np.interp(x_new, x_old, audio)[:len(audio)]
                    if len(audio) < len(x_new):
                        audio = np.pad(audio, (0, len(x_new) - len(audio)))
                audio = audio * (1.0 - progress)
            else:
                audio = audio * progress
            return audio
        
        elif style == 'reverb_tail':
            if direction == 'out':
                # Simple reverb approximation
                decay = 0.5 + progress * 0.4
                reverb = np.zeros(len(audio) + int(self.sample_rate * 0.5))
                reverb[:len(audio)] = audio
                
                # Multiple delays
                for delay_ms in [50, 100, 150, 200]:
                    delay_samples = int(delay_ms * self.sample_rate / 1000)
                    if delay_samples < len(reverb) - len(audio):
                        reverb[delay_samples:delay_samples + len(audio)] += audio * decay * (1 - delay_ms/250)
                
                audio = reverb[:len(audio)] * (1.0 - progress * 0.7)
            else:
                audio = audio * progress
            return audio
        
        elif style == 'stutter':
            if direction == 'out':
                # Stutter/glitch effect
                stutter_len = max(100, int(len(audio) * (1 - progress * 0.9)))
                if stutter_len < len(audio):
                    repeats = len(audio) // stutter_len
                    pattern = audio[:stutter_len]
                    audio = np.tile(pattern, repeats + 1)[:len(audio)]
                audio = audio * (1.0 - progress * 0.8)
            else:
                audio = audio * progress
            return audio
        
        elif style == 'gate':
            # Rhythmic gate
            gate_freq = 4 + progress * 12  # 4Hz to 16Hz
            t = np.arange(len(audio)) / self.sample_rate
            gate = (np.sin(2 * np.pi * gate_freq * t) > 0).astype(float)
            
            if direction == 'out':
                audio = audio * gate * (1.0 - progress)
            else:
                audio = audio * (1 - gate * (1 - progress))
            return audio
        
        # Default: simple crossfade
        if direction == 'out':
            return audio * (1.0 - progress)
        return audio * progress
    
    # ========================================================================
    # DECK COUNT MANAGEMENT
    # ========================================================================
    
    def set_deck_count(self, count: int) -> str:
        """Set the number of decks.
        
        Parameters
        ----------
        count : int
            Number of decks (1-8)
        
        Returns
        -------
        str
            Status message
        """
        count = max(1, min(8, count))
        
        current_count = len(self.decks)
        
        if count > current_count:
            # Add decks
            for i in range(current_count + 1, count + 1):
                if i not in self.decks:
                    self.decks[i] = DJDeck(id=i)
        elif count < current_count:
            # Remove decks (only empty ones)
            to_remove = []
            for deck_id in sorted(self.decks.keys(), reverse=True):
                if len(self.decks) > count:
                    deck = self.decks[deck_id]
                    if deck.buffer is None and not deck.playing:
                        to_remove.append(deck_id)
                    elif deck_id > count:
                        # Force remove if over limit
                        to_remove.append(deck_id)
            
            for deck_id in to_remove[:current_count - count]:
                del self.decks[deck_id]
        
        return f"OK: deck count set to {len(self.decks)}"
    
    def _ensure_deck(self, deck_id: int) -> None:
        """Ensure a deck exists, creating if needed."""
        if deck_id not in self.decks:
            self.decks[deck_id] = DJDeck(id=deck_id)
    
    def add_deck(self) -> str:
        """Add a new deck."""
        new_id = max(self.decks.keys()) + 1 if self.decks else 1
        if new_id > 8:
            return "ERROR: maximum 8 decks"
        
        self.decks[new_id] = DJDeck(id=new_id)
        return f"OK: added Deck {new_id} (total: {len(self.decks)})"
    
    def remove_deck(self, deck_id: int) -> str:
        """Remove a deck if empty."""
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        if len(self.decks) <= 1:
            return "ERROR: cannot remove last deck"
        
        deck = self.decks[deck_id]
        if deck.playing:
            return f"ERROR: Deck {deck_id} is playing. Stop first."
        
        del self.decks[deck_id]
        return f"OK: removed Deck {deck_id}"
    
    # ========================================================================
    # BUFFER INTEGRATION
    # ========================================================================
    
    def load_from_session_buffer(self, deck_id: int, session: Any, buffer_idx: int = None) -> str:
        """Load audio from session buffer to deck.
        
        Parameters
        ----------
        deck_id : int
            Deck to load to
        session : Session
            MDMA session
        buffer_idx : int, optional
            Specific buffer index (None = last_buffer)
        """
        try:
            if deck_id not in self.decks:
                self.decks[deck_id] = DJDeck(id=deck_id)
            
            deck = self.decks[deck_id]
            
            # Get audio from session
            audio = None
            if buffer_idx is not None:
                if hasattr(session, 'buffers') and buffer_idx in session.buffers:
                    audio = session.buffers[buffer_idx]
                else:
                    return f"ERROR: buffer {buffer_idx} not found"
            else:
                audio = getattr(session, 'last_buffer', None)
            
            if audio is None:
                return "ERROR: no audio in buffer"
            
            if not isinstance(audio, np.ndarray):
                try:
                    audio = np.array(audio)
                except Exception as e:
                    return f"ERROR: invalid audio data: {e}"
            
            if len(audio) == 0:
                return "ERROR: audio buffer is empty"
            
            # Convert to engine format (float64)
            if audio.dtype != np.float64:
                audio = audio.astype(np.float64)
            
            # Preserve stereo when available.
            # - mono: shape (N,)
            # - stereo: shape (N, 2)
            # Some sources may be (2, N); normalize to (N, C).
            if audio.ndim == 2:
                # Transpose if channel-first
                if audio.shape[0] in (1, 2) and audio.shape[1] > audio.shape[0]:
                    # Heuristic: if rows look like channels, transpose
                    # (2, N) -> (N, 2)
                    audio = audio.T
                # If more than 2 channels, keep first two
                if audio.shape[1] > 2:
                    audio = audio[:, :2]
            elif audio.ndim > 2:
                return "ERROR: unsupported audio buffer shape"
            
            deck.buffer = audio
            deck.position = 0.0
            deck.playing = False
            deck.transition_volume = 1.0  # Reset transition volume
            deck.source_buffer_idx = buffer_idx
            deck.analyzed = False
            deck.stems = None
            
            dur = len(deck.buffer) / self.sample_rate
            src = f"buffer {buffer_idx}" if buffer_idx else "last_buffer"
            return f"OK: loaded {dur:.2f}s from {src} to Deck {deck_id}"
            
        except Exception as e:
            return f"ERROR: failed to load from buffer: {e}"
    
    def save_to_session_buffer(self, deck_id: int, session: Any, buffer_idx: int = None) -> str:
        """Save deck audio to session buffer.
        
        Parameters
        ----------
        deck_id : int
            Deck to save from
        session : Session
            MDMA session
        buffer_idx : int, optional
            Target buffer (None = last_buffer)
        """
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = self.decks[deck_id]
        if deck.buffer is None:
            return f"ERROR: Deck {deck_id} is empty"
        
        if buffer_idx is not None:
            if not hasattr(session, 'buffers'):
                session.buffers = {}
            session.buffers[buffer_idx] = deck.buffer.copy()
            return f"OK: saved Deck {deck_id} to buffer {buffer_idx}"
        else:
            session.last_buffer = deck.buffer.copy()
            return f"OK: saved Deck {deck_id} to last_buffer"
    
    # ========================================================================
    # STEM SEPARATION
    # ========================================================================
    
    def separate_stems(self, deck_id: int, model: str = "htdemucs") -> str:
        """Separate deck audio into stems.
        
        Requires demucs: pip install demucs
        """
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = self.decks[deck_id]
        if deck.buffer is None:
            return f"ERROR: Deck {deck_id} is empty"
        
        try:
            from .stems import separate_stems, StemType
            
            print(f"Separating stems for Deck {deck_id}...")
            stem_set = separate_stems(deck.buffer, self.sample_rate, model)
            
            deck.stems = stem_set
            deck.active_stems = {s.value: 1.0 for s in stem_set.stems.keys()}
            
            stems_list = stem_set.list_stems()
            return f"OK: separated Deck {deck_id} into {len(stems_list)} stems: {', '.join(stems_list)}"
            
        except ImportError as e:
            return f"ERROR: stem separation requires demucs: pip install demucs torch"
        except Exception as e:
            return f"ERROR: stem separation failed: {e}"
    
    def set_stem_level(self, deck_id: int, stem: str, level: float) -> str:
        """Set stem volume level.
        
        Parameters
        ----------
        deck_id : int
            Deck ID
        stem : str
            Stem name (vocals, drums, bass, other)
        level : float
            Volume level (0-2)
        """
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = self.decks[deck_id]
        if not deck.stems:
            return f"ERROR: Deck {deck_id} has no stems. Use /stem sep {deck_id} first."
        
        stem_lower = stem.lower()
        if stem_lower not in deck.active_stems:
            available = list(deck.active_stems.keys())
            return f"ERROR: unknown stem '{stem}'. Available: {available}"
        
        deck.active_stems[stem_lower] = max(0.0, min(2.0, level))
        return f"OK: Deck {deck_id} {stem} = {deck.active_stems[stem_lower]:.0%}"
    
    def remix_stems(self, deck_id: int) -> str:
        """Remix stems with current levels and update deck buffer."""
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = self.decks[deck_id]
        if not deck.stems:
            return f"ERROR: Deck {deck_id} has no stems"
        
        try:
            from .stems import StemType
            
            # Convert string keys to StemType
            levels = {}
            for stem_name, level in deck.active_stems.items():
                try:
                    stem_type = StemType(stem_name)
                    levels[stem_type] = level
                except ValueError:
                    pass
            
            # Remix
            remixed = deck.stems.remix(levels)
            deck.buffer = remixed
            
            return f"OK: remixed Deck {deck_id} with current stem levels"
            
        except Exception as e:
            return f"ERROR: remix failed: {e}"
    
    # ========================================================================
    # AUTO-SECTIONING AND CHOPPING
    # ========================================================================
    
    def auto_section(self, deck_id: int, min_seconds: float = 4.0) -> str:
        """Auto-detect sections in deck audio."""
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = self.decks[deck_id]
        if deck.buffer is None:
            return f"ERROR: Deck {deck_id} is empty"
        
        try:
            from .stems import auto_section
            
            sections = auto_section(deck.buffer, self.sample_rate, min_seconds)
            deck.sections = sections
            
            lines = [f"OK: detected {len(sections)} sections in Deck {deck_id}:", ""]
            for i, sec in enumerate(sections[:10]):
                lines.append(f"  [{i}] {sec.start_time:.1f}s - {sec.end_time:.1f}s ({sec.duration:.1f}s)")
            if len(sections) > 10:
                lines.append(f"  ... and {len(sections) - 10} more")
            
            return '\n'.join(lines)
            
        except Exception as e:
            return f"ERROR: auto-section failed: {e}"
    
    def chop(self, deck_id: int, mode: str = "beat", divisions: int = 16) -> str:
        """Chop deck audio into slices."""
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = self.decks[deck_id]
        if deck.buffer is None:
            return f"ERROR: Deck {deck_id} is empty"
        
        try:
            from .stems import chop_audio
            
            chops = chop_audio(deck.buffer, self.sample_rate, mode, divisions)
            deck.chops = chops
            
            return f"OK: chopped Deck {deck_id} into {len(chops)} slices ({mode} mode)"
            
        except Exception as e:
            return f"ERROR: chop failed: {e}"
    
    def get_chop(self, deck_id: int, chop_idx: int) -> Optional[np.ndarray]:
        """Get a specific chop as audio."""
        if deck_id not in self.decks:
            return None
        
        deck = self.decks[deck_id]
        if not deck.chops or chop_idx >= len(deck.chops):
            return None
        
        return deck.chops[chop_idx].audio
    
    def play_section(self, deck_id: int, section_idx: int) -> str:
        """Jump to and play a specific section."""
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = self.decks[deck_id]
        if not deck.sections or section_idx >= len(deck.sections):
            return f"ERROR: section {section_idx} not found"
        
        # Auto-select output device if none connected
        if not self.master_device:
            self.ensure_playback_device()
        
        section = deck.sections[section_idx]
        deck.position = section.start_time
        deck.current_section = section_idx
        deck.playing = True
        
        return f"OK: playing section {section_idx} ({section.start_time:.1f}s - {section.end_time:.1f}s)"
    
    # ========================================================================
    # AI ANALYSIS INTEGRATION
    # ========================================================================
    
    def analyze_deck(self, deck_id: int) -> str:
        """Run AI analysis on deck audio."""
        if deck_id not in self.decks:
            return f"ERROR: Deck {deck_id} not found"
        
        deck = self.decks[deck_id]
        if deck.buffer is None:
            return f"ERROR: Deck {deck_id} is empty"
        
        try:
            from ..ai.analysis import analyze_audio, detect_tempo_key
            from .stems import detect_beats
            
            # Basic analysis
            analysis = analyze_audio(deck.buffer, self.sample_rate)
            
            # Tempo/key detection
            tempo, beats = detect_beats(deck.buffer, self.sample_rate)
            
            deck.tempo = tempo
            deck.analysis_data = analysis
            deck.analyzed = True
            
            # Format result
            lines = [f"=== DECK {deck_id} ANALYSIS ===", ""]
            lines.append(f"Tempo: {tempo:.1f} BPM")
            lines.append(f"Duration: {len(deck.buffer) / self.sample_rate:.2f}s")
            
            # Key attributes
            if 'brightness' in analysis:
                lines.append(f"Brightness: {analysis['brightness']:.0f}%")
            if 'energy' in analysis:
                lines.append(f"Energy: {analysis['energy']:.0f}%")
            if 'attack' in analysis:
                lines.append(f"Attack: {analysis['attack']:.0f}%")
            
            return '\n'.join(lines)
            
        except ImportError:
            # Fallback to basic analysis
            from .stems import detect_beats
            
            tempo, beats = detect_beats(deck.buffer, self.sample_rate)
            deck.tempo = tempo
            deck.analyzed = True
            
            return f"OK: Deck {deck_id} tempo detected: {tempo:.1f} BPM"
        except Exception as e:
            return f"ERROR: analysis failed: {e}"
    
    # ========================================================================
    # STREAMING LIBRARY INTEGRATION
    # ========================================================================
    
    def stream_to_deck(self, deck_id: int, url: str) -> str:
        """Stream audio from URL to deck.
        
        Supports: SoundCloud, YouTube, direct URLs
        All audio is pre-processed to engine format (float64 wave).
        Automatically registers to song registry for future use.
        """
        try:
            from .streaming import stream_url
            
            print(f"Streaming to Deck {deck_id}...")
            buffer = stream_url(url, self.sample_rate)
            
            if buffer.error:
                return f"ERROR: {buffer.error}"
            
            if not buffer.ready:
                return "ERROR: stream not ready"
            
            if deck_id not in self.decks:
                self.decks[deck_id] = DJDeck(id=deck_id)
            
            deck = self.decks[deck_id]
            deck.buffer = buffer.audio  # Already float64
            deck.position = 0.0
            deck.playing = False
            deck.analyzed = False
            deck.stems = None
            
            title = buffer.track_info.title if buffer.track_info else "Unknown"
            dur = buffer.duration
            
            # Show registry ID if registered
            reg_info = ""
            if buffer.registry_id:
                reg_info = f"\n  Registered as song #{buffer.registry_id} - use /reg load {buffer.registry_id} anytime"
            
            return f"OK: streamed '{title}' ({dur:.1f}s) to Deck {deck_id}{reg_info}"
            
        except ImportError:
            return "ERROR: streaming requires yt-dlp: pip install yt-dlp"
        except Exception as e:
            return f"ERROR: stream failed: {e}"
    
    # ========================================================================
    # SLOW-FALLBACK SAFETY SYSTEM
    # ========================================================================
    
    def check_fallback(self) -> Optional[str]:
        """Check if fallback should activate. Returns reason or None."""
        if self.fallback.active:
            return None
        
        # Check for triggers
        elapsed = time.time() - self.last_interaction
        
        if elapsed > self.fallback_timeout:
            return "extended_inactivity"
        
        # Check for tempo drift between decks
        playing_decks = [d for d in self.decks.values() if d.playing]
        if len(playing_decks) >= 2:
            tempos = [d.tempo * d.pitch for d in playing_decks]
            if max(tempos) - min(tempos) > 5:  # 5 BPM drift
                return "tempo_drift"
        
        return None
    
    def activate_fallback(self, reason: str) -> str:
        """Activate slow-fallback safety system."""
        self.fallback.active = True
        self.fallback.trigger_reason = reason
        self.fallback.started_at = time.time()
        
        # Apply fallback behaviors based on reason
        actions = []
        
        if reason == "extended_inactivity":
            # Extend loops safely
            for deck in self.decks.values():
                if deck.playing and not deck.loop_active:
                    # Set a safe loop at current position
                    deck.loop_active = True
                    deck.loop_start = deck.position
                    deck.loop_end = deck.position + 8 * (60 / deck.tempo)  # 8 beats
                    self.fallback.loop_extended = True
            actions.append("loops extended")
        
        elif reason == "tempo_drift":
            # Lock tempos to average
            playing_decks = [d for d in self.decks.values() if d.playing]
            avg_tempo = sum(d.tempo for d in playing_decks) / len(playing_decks)
            for deck in playing_decks:
                deck.tempo = avg_tempo
            self.fallback.tempo_locked = True
            actions.append(f"tempo locked to {avg_tempo:.1f} BPM")
        
        return f"⚠️ SLOW-FALLBACK ACTIVATED: {reason}\n  Actions: {', '.join(actions)}"
    
    def deactivate_fallback(self) -> str:
        """Deactivate fallback and restore control."""
        if not self.fallback.active:
            return "Fallback not active"
        
        self.fallback.active = False
        self.fallback.trigger_reason = ""
        self.fallback.loop_extended = False
        self.fallback.tempo_locked = False
        self.last_interaction = time.time()
        
        return "OK: Fallback deactivated, control restored"
    
    # ========================================================================
    # VDA (VIRTUAL AUDIO DEVICE) INTEGRATION
    # ========================================================================
    
    def vda_list(self) -> str:
        """List VDA routes."""
        lines = ["=== VDA ROUTES ===", ""]
        
        if not self.vda_routes:
            lines.append("No VDA routes configured")
        else:
            for source, dest in self.vda_routes.items():
                lines.append(f"  {source} → {dest}")
        
        lines.append("")
        lines.append("Commands: /VDA route <src> <dst>, /VDA isolate cue")
        return '\n'.join(lines)
    
    def vda_route(self, source: str, dest: str) -> str:
        """Create a VDA route."""
        self.vda_routes[source] = dest
        return f"OK: VDA route created: {source} → {dest}"
    
    def vda_isolate(self, bus: str) -> str:
        """Isolate a bus to its own VDA channel."""
        if bus == "cue":
            self.vda_routes["cue_bus"] = "vda_cue"
            return "OK: Cue bus isolated to VDA channel"
        elif bus == "master":
            self.vda_routes["master_bus"] = "vda_master"
            return "OK: Master bus isolated to VDA channel"
        else:
            return f"ERROR: unknown bus '{bus}'"
    
    # ========================================================================
    # NVDA SCREEN READER ROUTING
    # ========================================================================
    
    def sr_route(self, target: str) -> str:
        """Route NVDA screen reader output."""
        if target == "hep" or target == "headphones":
            self.nvda_device = self.headphone_device
            return ("OK: NVDA routed to headphones\n"
                    "  Note: Set in NVDA Settings → Audio → Output Device")
        
        elif target == "out" or target == "master":
            self.nvda_device = self.master_device
            return "OK: NVDA routed to master output"
        
        elif target == "keep":
            return "OK: NVDA routing unchanged"
        
        else:
            return f"ERROR: unknown target '{target}'. Use: hep, out, keep"
    
    def sr_test(self) -> str:
        """Test screen reader routing."""
        return ("NVDA Routing Test:\n"
                "  If you hear this through your expected device, routing is correct.\n"
                "  Current NVDA target: " + (self.nvda_device or "system default"))
    
    def sr_menu(self) -> str:
        """Show screen reader routing menu."""
        return ("=== SCREEN READER ROUTING ===\n"
                "\n"
                "  /SR HEP   Route NVDA to headphones\n"
                "  /SR OUT   Route NVDA to master output\n"
                "  /SR KEEP  Leave NVDA routing unchanged\n"
                "  /SR TEST  Test current routing\n"
                "\n"
                "Current NVDA device: " + (self.nvda_device or "system default"))
    
    # ========================================================================
    # READINESS CHECK
    # ========================================================================
    
    def readiness_check(self) -> str:
        """Verify DJ Mode setup integrity."""
        lines = ["=== DJ MODE READINESS CHECK ===", ""]
        issues = []
        warnings = []
        
        # Check master device
        if self.master_device:
            dev = self.devices.get(self.master_device)
            if dev:
                lines.append(f"✓ Master Output: {dev.name} (~{dev.latency_ms:.0f}ms)")
            else:
                issues.append("Master device not found")
        else:
            issues.append("No master output connected")
        
        # Check headphones
        if self.headphone_device:
            dev = self.devices.get(self.headphone_device)
            if dev:
                lines.append(f"✓ Headphones: {dev.name} (~{dev.latency_ms:.0f}ms)")
            else:
                issues.append("Headphone device not found")
        else:
            warnings.append("No headphones connected (cue unavailable)")
        
        # Check decks
        loaded_decks = sum(1 for d in self.decks.values() if d.buffer is not None)
        lines.append(f"✓ Decks: {len(self.decks)} configured, {loaded_decks} loaded")
        
        # Check latency
        total_latency = 0
        if self.master_device and self.master_device in self.devices:
            total_latency = self.devices[self.master_device].latency_ms
        
        if total_latency > 30:
            warnings.append(f"High latency detected ({total_latency:.0f}ms)")
        else:
            lines.append(f"✓ Latency: ~{total_latency:.0f}ms (acceptable)")
        
        # NVDA routing
        if self.nvda_device:
            lines.append(f"✓ NVDA: routed to {self.nvda_device}")
        else:
            lines.append("○ NVDA: using system default")
        
        # Fallback status
        if self.fallback.active:
            warnings.append("Slow-fallback is currently ACTIVE")
        else:
            lines.append("✓ Safety: fallback ready")
        
        lines.append("")
        
        if issues:
            lines.append("ISSUES (must fix):")
            for i in issues:
                lines.append(f"  ✗ {i}")
        
        if warnings:
            lines.append("WARNINGS:")
            for w in warnings:
                lines.append(f"  ⚠ {w}")
        
        if not issues:
            lines.append("")
            lines.append("✓ READY FOR PERFORMANCE")
        else:
            lines.append("")
            lines.append("✗ NOT READY - fix issues above")
        
        return '\n'.join(lines)
    
    def _stop_all(self):
        """Stop all playback."""
        self._running = False
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)
        for deck in self.decks.values():
            deck.playing = False
    
    def _start_audio_output(self) -> str:
        """Start the audio output thread if not already running."""
        if self._running and self._playback_thread and self._playback_thread.is_alive():
            return ""  # Already running
        
        self._running = True
        self._playback_thread = threading.Thread(target=self._audio_output_loop, daemon=True)
        self._playback_thread.start()
        return "Audio output started"
    
    def _audio_output_loop(self):
        """Main audio output loop - runs in background thread."""
        try:
            import sounddevice as sd
        except ImportError:
            print("WARNING: sounddevice not available, audio output disabled")
            self._running = False
            return
        except OSError as e:
            print(f"WARNING: PortAudio not available ({e}), audio output disabled")
            self._running = False
            return
        
        # Audio parameters
        block_size = int(getattr(self, 'block_size', 2048) or 2048)
        sample_rate = self.sample_rate
        
        # Get device ID
        device_id = None
        if self.master_device:
            try:
                device_id = int(self.master_device)
            except ValueError:
                pass
        
        def audio_callback(outdata, frames, time_info, status):
            """Callback function for sounddevice stream."""
            try:
                if status:
                    print(f"Audio status: {status}")
                
                # Process any active transition (updates volumes)
                try:
                    self._process_transition()
                except Exception:
                    pass  # Don't let transition errors stop audio
                
                # Process any active filter sweep
                try:
                    self._process_filter_sweep()
                except Exception:
                    pass  # Don't let filter errors stop audio
                
                # Process any active scratch (updates position/pitch)
                try:
                    self._process_scratch()
                except Exception:
                    pass  # Don't let scratch errors stop audio
                
                # Mix audio from all playing decks (stereo)
                output = np.zeros((frames, 2), dtype=np.float64)
                
                # Safely get playing decks
                try:
                    playing_decks = [(d_id, d) for d_id, d in self.decks.items() 
                                    if d.playing and d.buffer is not None and len(d.buffer) > 0]
                except Exception:
                    playing_decks = []
                
                if not playing_decks:
                    outdata[:] = np.zeros((frames, 2), dtype=np.float32)
                    return
                
                for deck_id, deck in playing_decks:
                    try:
                        buffer_len = deck.buffer.shape[0] if hasattr(deck.buffer, 'shape') else len(deck.buffer)
                        
                        # Check if scratch is active for this deck
                        scratch = getattr(self, '_active_scratch', None)
                        is_scratching = scratch and scratch.get('deck_id') == deck_id and not scratch.get('completed')
                        
                        # Handle stutter mode
                        if getattr(deck, 'stutter_active', False) and deck.stutter_remaining > 0:
                            # In stutter mode - read from stutter position
                            stutter_pos = int(deck.stutter_position * sample_rate)
                            stutter_len = int(deck.stutter_length * sample_rate)
                            
                            if stutter_len <= 0:
                                stutter_len = int(sample_rate * 0.1)  # Default 100ms
                            
                            # Get chunk from stutter position
                            chunk = np.zeros((frames, 2), dtype=np.float64)
                            write_pos = 0
                            while write_pos < frames:
                                remaining = frames - write_pos
                                read_len = min(remaining, stutter_len)
                                
                                if stutter_pos + read_len <= buffer_len:
                                    src = deck.buffer[stutter_pos:stutter_pos + read_len]
                                    if src.ndim == 1:
                                        src = np.column_stack((src, src))
                                    chunk[write_pos:write_pos + read_len] = src
                                else:
                                    avail = buffer_len - stutter_pos
                                    if avail > 0:
                                        src = deck.buffer[stutter_pos:buffer_len]
                                        if src.ndim == 1:
                                            src = np.column_stack((src, src))
                                        chunk[write_pos:write_pos + avail] = src
                                
                                write_pos += read_len
                                # Decrement stutter count after each repeat
                                deck.stutter_remaining -= 1
                                if deck.stutter_remaining <= 0:
                                    deck.stutter_active = False
                                    break
                            
                            # Advance actual position so we continue from somewhere sensible
                            deck.position += frames / sample_rate
                        
                        elif is_scratching:
                            # Scratch mode - use varispeed playback based on pitch
                            chunk = self._get_scratch_audio(deck, frames)
                            if chunk is None:
                                chunk = np.zeros((frames, 2), dtype=np.float64)
                            else:
                                chunk = chunk.astype(np.float64)
                                if chunk.ndim == 1:
                                    chunk = np.column_stack((chunk, chunk))
                            
                        else:
                            # Normal playback
                            pos = int(deck.position * sample_rate)
                            
                            # Bounds check
                            if pos < 0:
                                pos = 0
                                deck.position = 0.0
                            
                            # Check for loop end crossing
                            if deck.loop_active and deck.loop_end is not None and deck.loop_end > 0:
                                loop_end_samples = int(deck.loop_end * sample_rate)
                                loop_start_samples = int(deck.loop_start * sample_rate) if deck.loop_start else 0
                                
                                if pos >= loop_end_samples:
                                    # Loop point reached
                                    loop_remaining = getattr(deck, 'loop_remaining', 0)
                                    if loop_remaining > 0:
                                        deck.loop_remaining = loop_remaining - 1
                                        if deck.loop_remaining <= 0:
                                            # Done looping, release
                                            deck.loop_active = False
                                            deck.loop_remaining = 0
                                        else:
                                            # Continue looping
                                            deck.position = deck.loop_start if deck.loop_start else 0.0
                                            pos = loop_start_samples
                                    else:
                                        # Infinite loop (loop_remaining was 0)
                                        deck.position = deck.loop_start if deck.loop_start else 0.0
                                        pos = loop_start_samples
                            
                            # Get audio chunk
                            end_pos = pos + frames
                            if end_pos <= buffer_len:
                                chunk = deck.buffer[pos:end_pos].copy()
                                if chunk.ndim == 1:
                                    chunk = np.column_stack((chunk, chunk))
                                deck.position = end_pos / sample_rate
                            elif pos < buffer_len:
                                # Partial chunk at end
                                available = buffer_len - pos
                                chunk = np.zeros((frames, 2), dtype=np.float64)
                                src = deck.buffer[pos:buffer_len]
                                if src.ndim == 1:
                                    src = np.column_stack((src, src))
                                chunk[:available] = src
                                
                                if deck.loop_active and deck.loop_start is not None:
                                    # Loop back
                                    deck.position = max(0.0, deck.loop_start)
                                elif getattr(self, 'master_loop_wrap', False):
                                    # Master loop wrap: fill the rest from start
                                    wrap_needed = frames - available
                                    if wrap_needed > 0 and buffer_len > 0:
                                        wrap_src = deck.buffer[:min(wrap_needed, buffer_len)]
                                        if wrap_src.ndim == 1:
                                            wrap_src = np.column_stack((wrap_src, wrap_src))
                                        chunk[available:available + len(wrap_src)] = wrap_src
                                    deck.position = (wrap_needed / sample_rate)
                                else:
                                    # Stop at end
                                    deck.playing = False
                                    deck.position = 0.0
                            else:
                                # Past end
                                if getattr(self, 'master_loop_wrap', False) and buffer_len > 0:
                                    deck.position = 0.0
                                    chunk = deck.buffer[:frames].copy()
                                    if chunk.ndim == 1:
                                        chunk = np.column_stack((chunk, chunk))
                                else:
                                    deck.playing = False
                                    deck.position = 0.0
                                    continue
                        
                        # Ensure float64 for processing
                        chunk = chunk.astype(np.float64)
                        
                        # Apply deck effect if active
                        try:
                            chunk = self._process_deck_effect(chunk, deck_id)
                        except Exception:
                            pass  # Don't let effect errors stop audio
                        
                        # Apply filter if not neutral (1-100 scale, 50 = neutral)
                        filter_cutoff = getattr(deck, 'filter_cutoff', 50.0)
                        filter_resonance = getattr(deck, 'filter_resonance', 20.0)
                        if abs(filter_cutoff - 50) > 5:
                            try:
                                chunk = self._apply_filter(chunk, filter_cutoff, filter_resonance)
                            except Exception:
                                pass  # Skip filter on error
                        
                        # Apply deck volume, transition volume, and crossfader
                        trans_vol = getattr(deck, 'transition_volume', 1.0)
                        volume = deck.volume * trans_vol
                        
                        # Apply crossfader (only when NOT in active transition)
                        active_trans = getattr(self, '_active_transition', None)
                        if active_trans is None and len(self.decks) >= 2:
                            if deck_id == 1:
                                volume *= (1.0 - self.crossfader)
                            elif deck_id == 2:
                                volume *= self.crossfader
                        
                        output += chunk * volume
                        
                    except Exception as e:
                        # Skip this deck on error, don't crash audio
                        continue
                
                # Apply always-on Master Deck FX (MDFX) before output gain/clip
                try:
                    if getattr(self, 'master_deck_fx_chain', None):
                        output = self.apply_fx_chain(output, self.master_deck_fx_chain)
                except Exception:
                    pass

                # Clip and convert to float32 for output
                output = np.clip(output * self.master_volume, -0.95, 0.95)
                
                # Apply AI audio enhancement if enabled
                try:
                    from ..dsp.enhancement import get_enhancer, get_enhancement_settings
                    settings = get_enhancement_settings()
                    if settings.enabled:
                        enhancer = get_enhancer(sample_rate)
                        output = enhancer.process(output)
                        output = np.clip(output, -0.95, 0.95)
                except Exception:
                    pass  # Skip enhancement on error
                
                outdata[:] = output.astype(np.float32)
                
            except Exception as e:
                # Last resort - output silence on any error
                outdata[:] = np.zeros((frames, 2), dtype=np.float32)
        
        try:
            with sd.OutputStream(
                samplerate=sample_rate,
                channels=2,
                dtype=np.float32,
                blocksize=block_size,
                device=device_id,
                callback=audio_callback,
            ):
                while self._running:
                    # Check if any deck is playing
                    any_playing = any(d.playing for d in self.decks.values())
                    if not any_playing:
                        time.sleep(0.1)
                    else:
                        time.sleep(0.01)
        except Exception as e:
            print(f"Audio output error: {e}")
        finally:
            self._running = False


# ============================================================================
# GLOBAL DJ ENGINE
# ============================================================================

_dj_engine: Optional[DJModeEngine] = None


def get_dj_engine(sample_rate: int = 48000) -> DJModeEngine:
    """Get or create the global DJ engine."""
    global _dj_engine
    if _dj_engine is None:
        _dj_engine = DJModeEngine(sample_rate)
    return _dj_engine


def is_dj_mode_enabled() -> bool:
    """Check if DJ mode is enabled."""
    return _dj_engine is not None and _dj_engine.enabled
