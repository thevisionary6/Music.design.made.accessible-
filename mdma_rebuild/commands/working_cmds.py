"""Working Buffer and Audio Source Tracking for MDMA v3.

PHILOSOPHY:
- The "working buffer" is simply a POINTER to current focused audio
- Audio always lives in a SOURCE: buffer, deck, clip, or file
- Working buffer tracks WHICH source is active and provides the audio
- Only empty if deliberately cleared or app just launched
- /AS shows complete audio routing debug info

SOURCE HIERARCHY:
- Buffer (numbered slots 1-N)
- Deck (DJ decks 1-N)
- Clip (arrangement clips)
- File (only at file level)
- Generated (temporary, uncommitted audio from tone/noise/generator)

BUILD ID: working_cmds_v3.0
"""

from __future__ import annotations

import numpy as np
import time
from typing import List, Optional, Dict, Any, TYPE_CHECKING, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto

if TYPE_CHECKING:
    from ..core.session import Session


# ============================================================================
# AUDIO SOURCE TYPES
# ============================================================================

class SourceType(Enum):
    """Where audio lives."""
    NONE = "none"           # No audio (cleared or fresh launch)
    BUFFER = "buffer"       # Numbered buffer slot
    DECK = "deck"           # DJ deck
    CLIP = "clip"           # Arrangement clip
    FILE = "file"           # External file (file level only)
    PENDING = "pending"     # Generated but not yet committed


@dataclass 
class AudioRef:
    """Reference to audio data with full attribution.
    
    This is NOT a copy - it points to where audio lives.
    """
    source_type: SourceType = SourceType.NONE
    source_id: Optional[int] = None      # Buffer/deck/clip number
    source_path: Optional[str] = None    # File path (file level only)
    source_name: str = ""                # Human readable name
    
    # For pending/generated audio
    generated_by: str = ""               # Command that generated it (tone, g_tom, etc)
    generated_audio: Optional[np.ndarray] = None  # Temporary storage until committed
    
    # Metadata
    duration: float = 0.0
    sample_rate: int = 48000
    fx_applied: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    def describe(self) -> str:
        """Human readable source description."""
        if self.source_type == SourceType.NONE:
            return "(empty - generate audio or select a source)"
        
        if self.source_type == SourceType.BUFFER:
            return f"buffer #{self.source_id}"
        elif self.source_type == SourceType.DECK:
            return f"deck #{self.source_id}"
        elif self.source_type == SourceType.CLIP:
            return f"clip #{self.source_id}"
        elif self.source_type == SourceType.FILE:
            name = Path(self.source_path).name if self.source_path else "unknown"
            return f"file [{name}]"
        elif self.source_type == SourceType.PENDING:
            return f"pending ({self.generated_by}) - use /A to commit"
        
        return f"{self.source_type.value}"
    
    def short_desc(self) -> str:
        """Short description for status lines."""
        if self.source_type == SourceType.NONE:
            return "none"
        elif self.source_type == SourceType.BUFFER:
            return f"buf{self.source_id}"
        elif self.source_type == SourceType.DECK:
            return f"dk{self.source_id}"
        elif self.source_type == SourceType.CLIP:
            return f"clip{self.source_id}"
        elif self.source_type == SourceType.FILE:
            return f"file"
        elif self.source_type == SourceType.PENDING:
            return f"pending:{self.generated_by}"
        return "?"


# ============================================================================
# WORKING BUFFER - Singleton tracking current audio focus
# ============================================================================

class WorkingBuffer:
    """Tracks what audio the user is currently working with.
    
    This is a POINTER system, not a copy system:
    - Points to current buffer, deck, clip, or file
    - Generated audio is "pending" until committed with /A
    - Provides unified access to the audio data
    """
    
    _instance: Optional['WorkingBuffer'] = None
    
    def __new__(cls):
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._ref = AudioRef()
            inst._session = None
            inst._history: List[str] = []
            cls._instance = inst
        return cls._instance
    
    def bind_session(self, session: "Session"):
        """Bind to session for accessing buffers/decks."""
        self._session = session
    
    @property
    def ref(self) -> AudioRef:
        """Current audio reference."""
        return self._ref
    
    @property 
    def audio(self) -> Optional[np.ndarray]:
        """Get current audio data (resolves the reference)."""
        if self._ref.source_type == SourceType.NONE:
            return None
        
        if self._ref.source_type == SourceType.PENDING:
            return self._ref.generated_audio
        
        if self._ref.source_type == SourceType.BUFFER:
            if self._session and hasattr(self._session, 'buffers'):
                return self._session.buffers.get(self._ref.source_id)
            return None
        
        if self._ref.source_type == SourceType.DECK:
            try:
                from ..dsp.dj_mode import get_dj_engine
                dj = get_dj_engine(self._ref.sample_rate)
                deck = dj.decks.get(self._ref.source_id)
                return deck.audio if deck else None
            except:
                return None
        
        # TODO: clip, file
        return None
    
    @property
    def source(self) -> AudioRef:
        """Alias for ref for compatibility."""
        return self._ref
    
    def is_empty(self) -> bool:
        """Check if no audio is focused."""
        return self._ref.source_type == SourceType.NONE
    
    def duration(self) -> float:
        """Get duration of current audio."""
        audio = self.audio
        if audio is None or len(audio) == 0:
            return 0.0
        return len(audio) / self._ref.sample_rate
    
    # ========================================================================
    # FOCUS OPERATIONS - Point to existing audio
    # ========================================================================
    
    def focus_buffer(self, buf_id: int, session: "Session" = None):
        """Focus on a numbered buffer."""
        if session:
            self._session = session
        
        sr = self._session.sample_rate if self._session else 48000
        buf = None
        if self._session and hasattr(self._session, 'buffers'):
            buf = self._session.buffers.get(buf_id)
        
        self._ref = AudioRef(
            source_type=SourceType.BUFFER,
            source_id=buf_id,
            source_name=f"buffer {buf_id}",
            sample_rate=sr,
            duration=len(buf) / sr if buf is not None else 0.0
        )
        self._add_history(f"focused buffer #{buf_id}")
    
    def focus_deck(self, deck_id: int, session: "Session" = None):
        """Focus on a DJ deck."""
        if session:
            self._session = session
        
        sr = self._session.sample_rate if self._session else 48000
        
        self._ref = AudioRef(
            source_type=SourceType.DECK,
            source_id=deck_id,
            source_name=f"deck {deck_id}",
            sample_rate=sr
        )
        self._add_history(f"focused deck #{deck_id}")
    
    def focus_clip(self, clip_id: int, session: "Session" = None):
        """Focus on a clip."""
        if session:
            self._session = session
        
        sr = self._session.sample_rate if self._session else 48000
        
        self._ref = AudioRef(
            source_type=SourceType.CLIP,
            source_id=clip_id,
            source_name=f"clip {clip_id}",
            sample_rate=sr
        )
        self._add_history(f"focused clip #{clip_id}")
    
    def focus_file(self, path: str, session: "Session" = None):
        """Focus on an external file."""
        if session:
            self._session = session
        
        sr = self._session.sample_rate if self._session else 48000
        
        self._ref = AudioRef(
            source_type=SourceType.FILE,
            source_path=path,
            source_name=Path(path).stem,
            sample_rate=sr
        )
        self._add_history(f"focused file [{Path(path).name}]")
    
    # ========================================================================
    # PENDING OPERATIONS - Hold generated audio until committed
    # ========================================================================
    
    def set_pending(self, audio: np.ndarray, generated_by: str, session: "Session" = None):
        """Store generated audio as pending (not yet committed to a source).
        
        Use /A to commit pending audio to a buffer/deck/clip.
        """
        if session:
            self._session = session
        
        sr = self._session.sample_rate if self._session else 48000
        
        self._ref = AudioRef(
            source_type=SourceType.PENDING,
            generated_by=generated_by,
            generated_audio=audio.copy() if audio is not None else None,
            source_name=f"pending ({generated_by})",
            sample_rate=sr,
            duration=len(audio) / sr if audio is not None else 0.0
        )
        
        # Also update session.last_buffer for compatibility
        if self._session:
            self._session.last_buffer = audio
        
        self._add_history(f"generated via {generated_by} (pending)")
    
    def commit_to_buffer(self, buf_id: int) -> bool:
        """Commit pending audio to a buffer."""
        if self._ref.source_type != SourceType.PENDING:
            return False
        
        audio = self._ref.generated_audio
        if audio is None:
            return False
        
        if self._session:
            if not hasattr(self._session, 'buffers'):
                self._session.buffers = {}
            self._session.buffers[buf_id] = audio.copy()
        
        # Now focus on that buffer
        self.focus_buffer(buf_id)
        self._add_history(f"committed to buffer #{buf_id}")
        return True
    
    def commit_to_deck(self, deck_id: int) -> bool:
        """Commit pending audio to a deck."""
        if self._ref.source_type != SourceType.PENDING:
            return False
        
        audio = self._ref.generated_audio
        if audio is None:
            return False
        
        try:
            from ..dsp.dj_mode import get_dj_engine
            dj = get_dj_engine(self._ref.sample_rate)
            if deck_id in dj.decks:
                dj.decks[deck_id].audio = audio.copy()
                self.focus_deck(deck_id)
                self._add_history(f"committed to deck #{deck_id}")
                return True
        except:
            pass
        return False
    
    # ========================================================================
    # EFFECT APPLICATION
    # ========================================================================
    
    def add_fx(self, name: str):
        """Track that an effect was applied."""
        self._ref.fx_applied.append(name)
    
    # ========================================================================
    # CLEAR
    # ========================================================================
    
    def clear(self):
        """Explicitly clear working buffer (user action)."""
        self._ref = AudioRef()
        self._add_history("cleared")
    
    # ========================================================================
    # COMPATIBILITY METHODS
    # ========================================================================
    
    def set(self, audio: np.ndarray, source=None, sr: int = 48000):
        """Compatibility method - treats as pending generation."""
        if self._session is None:
            # Create minimal session reference
            class MinimalSession:
                sample_rate = sr
                last_buffer = None
                buffers = {}
            self._session = MinimalSession()
        
        # Determine source name
        if source is None:
            gen_by = "unknown"
        elif isinstance(source, str):
            gen_by = source
        elif hasattr(source, 'name'):
            gen_by = source.name
        elif hasattr(source, 'command'):
            gen_by = source.command
        else:
            gen_by = str(source)
        
        self.set_pending(audio, gen_by)
    
    def set_from_generation(self, audio: np.ndarray, name: str, command: str = "", sr: int = 48000):
        """Set from synthesis/generation as pending."""
        self.set_pending(audio, command or name)
    
    def set_from_generator(self, audio: np.ndarray, gen_name: str, variant: int = 1, sr: int = 48000):
        """Set from generator as pending."""
        self.set_pending(audio, f"g_{gen_name}")
    
    def set_from_buffer(self, audio: np.ndarray, buf_id: int, sr: int = 48000):
        """Focus on buffer (for compatibility)."""
        self.focus_buffer(buf_id)
    
    def set_from_deck(self, audio: np.ndarray, deck_id: int, sr: int = 48000):
        """Focus on deck (for compatibility)."""
        self.focus_deck(deck_id)
    
    def set_from_file(self, audio: np.ndarray, path: str, sr: int = 48000):
        """Set from file (stores as pending then focuses)."""
        self.set_pending(audio, f"file:{Path(path).name}")
    
    def set_from_ai(self, audio: np.ndarray, prompt: str = "", sr: int = 48000):
        """Set from AI as pending."""
        self.set_pending(audio, f"ai:{prompt[:20]}")
    
    @property
    def fx_chain(self) -> list:
        """Get FX chain for compatibility."""
        return [(fx, []) for fx in self._ref.fx_applied]
    
    @property
    def history(self) -> list:
        """Get operation history."""
        return self._history
    
    # ========================================================================
    # INTERNAL
    # ========================================================================
    
    def _add_history(self, entry: str):
        """Add to history, keep last 20."""
        self._history.append(f"{time.strftime('%H:%M:%S')} {entry}")
        if len(self._history) > 20:
            self._history.pop(0)


def get_working_buffer() -> WorkingBuffer:
    """Get global working buffer instance."""
    return WorkingBuffer()


# ============================================================================
# COMMANDS
# ============================================================================

def cmd_wb(session: "Session", args: List[str]) -> str:
    """Show working buffer status.
    
    Usage:
      /WB                Show current working buffer
      /WB clear          Clear working buffer
      /WB history        Show operation history
    """
    wb = get_working_buffer()
    wb.bind_session(session)
    
    if args:
        sub = args[0].lower()
        if sub == 'clear':
            wb.clear()
            return "OK: Working buffer cleared"
        elif sub == 'history':
            if not wb.history:
                return "History: (empty)"
            lines = ["=== WORKING BUFFER HISTORY ==="]
            for entry in wb.history[-15:]:
                lines.append(f"  {entry}")
            return "\n".join(lines)
    
    # Show status
    ref = wb.ref
    
    if wb.is_empty():
        return "Working buffer: EMPTY\n  Generate audio (/tone, /g_tom) or select a source (/bu <n>)"
    
    lines = ["=== WORKING BUFFER ==="]
    lines.append(f"  Source: {ref.describe()}")
    
    audio = wb.audio
    if audio is not None and len(audio) > 0:
        dur = len(audio) / ref.sample_rate
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))
        lines.append(f"  Duration: {dur:.3f}s ({len(audio)} samples)")
        lines.append(f"  Peak: {20*np.log10(peak+1e-10):.1f}dB")
        lines.append(f"  RMS: {20*np.log10(rms+1e-10):.1f}dB")
    else:
        lines.append(f"  Audio: (source is empty)")
    
    if ref.fx_applied:
        lines.append(f"  FX Applied: {', '.join(ref.fx_applied)}")
    
    if ref.source_type == SourceType.PENDING:
        lines.append(f"\n  ⚠ Audio is PENDING - use /A to commit to buffer/deck")
    
    return "\n".join(lines)


def cmd_wbc(session: "Session", args: List[str]) -> str:
    """Clear working buffer."""
    wb = get_working_buffer()
    wb.clear()
    return "OK: Working buffer cleared"


def cmd_as(session: "Session", args: List[str]) -> str:
    """Audio Sources debug - show all audio in the system.
    
    Usage:
      /AS              Show all audio sources
      /AS buffers      Show buffer sources only
      /AS decks        Show deck sources only  
      /AS pending      Show pending/working buffer
      /AS full         Show everything with details
    """
    wb = get_working_buffer()
    wb.bind_session(session)
    
    filter_type = args[0].lower() if args else None
    show_full = filter_type == 'full'
    
    lines = ["=== AUDIO SOURCES ===", ""]
    
    # Working Buffer / Pending
    if not filter_type or filter_type in ('pending', 'wb', 'working', 'full'):
        lines.append("WORKING BUFFER (current focus):")
        ref = wb.ref
        if ref.source_type == SourceType.NONE:
            lines.append("  (empty)")
        else:
            lines.append(f"  Source: {ref.describe()}")
            if ref.source_type == SourceType.PENDING and ref.generated_audio is not None:
                dur = len(ref.generated_audio) / ref.sample_rate
                lines.append(f"  Pending Audio: {dur:.3f}s")
        lines.append("")
    
    # Numbered Buffers
    if not filter_type or filter_type in ('buffers', 'buf', 'full'):
        lines.append("NUMBERED BUFFERS:")
        if hasattr(session, 'buffers') and session.buffers:
            for buf_id in sorted(session.buffers.keys()):
                buf = session.buffers[buf_id]
                if buf is not None and len(buf) > 0:
                    dur = len(buf) / session.sample_rate
                    peak = np.max(np.abs(buf))
                    current = " ← FOCUSED" if (wb.ref.source_type == SourceType.BUFFER and wb.ref.source_id == buf_id) else ""
                    lines.append(f"  Buffer {buf_id}: {dur:.3f}s, peak={20*np.log10(peak+1e-10):.1f}dB{current}")
                else:
                    lines.append(f"  Buffer {buf_id}: (empty)")
            current_idx = getattr(session, 'current_buffer_index', 1)
            lines.append(f"  Selected: buffer {current_idx}")
        else:
            lines.append("  (no buffers allocated)")
        lines.append("")
    
    # DJ Decks
    if not filter_type or filter_type in ('decks', 'deck', 'dj', 'full'):
        lines.append("DJ DECKS:")
        try:
            from ..dsp.dj_mode import get_dj_engine
            dj = get_dj_engine(session.sample_rate)
            if dj.decks:
                for deck_id in sorted(dj.decks.keys()):
                    deck = dj.decks[deck_id]
                    if deck.audio is not None and len(deck.audio) > 0:
                        dur = len(deck.audio) / session.sample_rate
                        current = " ← FOCUSED" if (wb.ref.source_type == SourceType.DECK and wb.ref.source_id == deck_id) else ""
                        lines.append(f"  Deck {deck_id}: {dur:.3f}s, tempo={deck.tempo:.1f} BPM{current}")
                    else:
                        lines.append(f"  Deck {deck_id}: (empty)")
                lines.append(f"  Active: deck {dj.active_deck}")
            else:
                lines.append("  (no decks)")
        except Exception as e:
            lines.append(f"  (DJ engine not available: {e})")
        lines.append("")
    
    # Session last_buffer (compatibility)
    if show_full:
        lines.append("SESSION.LAST_BUFFER (legacy):")
        if session.last_buffer is not None and len(session.last_buffer) > 0:
            dur = len(session.last_buffer) / session.sample_rate
            lines.append(f"  Duration: {dur:.3f}s")
        else:
            lines.append("  (empty)")
        lines.append("")
    
    # Summary
    lines.append("SUMMARY:")
    focus_desc = wb.ref.short_desc() if not wb.is_empty() else "none"
    buf_count = len([b for b in (session.buffers or {}).values() if b is not None and len(b) > 0]) if hasattr(session, 'buffers') else 0
    lines.append(f"  Current focus: {focus_desc}")
    lines.append(f"  Buffers with audio: {buf_count}")
    if wb.ref.source_type == SourceType.PENDING:
        lines.append(f"  ⚠ Pending audio not committed - use /A <dest>")
    
    return "\n".join(lines)


def cmd_a(session: "Session", args: List[str]) -> str:
    """Append/commit working buffer to destination.
    
    Usage:
      /A                 Commit to current buffer
      /A <n>             Commit to buffer n
      /A B <n>           Commit to buffer n
      /A D <n>           Commit to deck n
      /A replace         Replace instead of append
    
    When working buffer has PENDING audio (from generation),
    this commits it to a permanent location.
    
    When working buffer points to a source, this copies
    from that source to the destination.
    """
    wb = get_working_buffer()
    wb.bind_session(session)
    
    if wb.is_empty():
        return "ERROR: Working buffer is empty. Generate audio first."
    
    # Parse destination
    dest_type = 'buffer'
    dest_id = None
    replace_mode = False
    
    remaining_args = list(args)
    
    if remaining_args:
        arg0 = remaining_args[0].upper()
        
        if arg0 == 'REPLACE':
            replace_mode = True
            remaining_args = remaining_args[1:]
            arg0 = remaining_args[0].upper() if remaining_args else ''
        
        if arg0 == 'B':
            dest_type = 'buffer'
            if len(remaining_args) > 1:
                try:
                    dest_id = int(remaining_args[1])
                except ValueError:
                    pass
        elif arg0 == 'D':
            dest_type = 'deck'
            if len(remaining_args) > 1:
                try:
                    dest_id = int(remaining_args[1])
                except ValueError:
                    pass
        elif arg0.isdigit():
            dest_type = 'buffer'
            dest_id = int(arg0)
    
    # Default destination
    if dest_id is None:
        if dest_type == 'buffer':
            dest_id = getattr(session, 'current_buffer_index', 1)
        elif dest_type == 'deck':
            try:
                from ..dsp.dj_mode import get_dj_engine
                dj = get_dj_engine(session.sample_rate)
                dest_id = dj.active_deck
            except:
                dest_id = 1
    
    # Get source audio
    audio = wb.audio
    if audio is None or len(audio) == 0:
        return "ERROR: No audio to commit (source is empty)"
    
    # Commit to destination
    if dest_type == 'buffer':
        if not hasattr(session, 'buffers'):
            session.buffers = {}
        
        if replace_mode or dest_id not in session.buffers or session.buffers[dest_id] is None:
            session.buffers[dest_id] = audio.copy()
            action = "Stored in"
        else:
            # Append
            existing = session.buffers[dest_id]
            session.buffers[dest_id] = np.concatenate([existing, audio])
            action = "Appended to"
        
        # Calculate metrics
        buf = session.buffers[dest_id]
        dur = len(buf) / session.sample_rate
        peak = np.max(np.abs(buf))
        rms = np.sqrt(np.mean(buf ** 2))
        
        # If was pending, now focus on buffer
        if wb.ref.source_type == SourceType.PENDING:
            wb.focus_buffer(dest_id, session)
        
        return f"OK: {action} buffer {dest_id}\n  {dur:.3f}s, peak={peak:.3f}, rms={rms:.3f}"
    
    elif dest_type == 'deck':
        try:
            from ..dsp.dj_mode import get_dj_engine
            dj = get_dj_engine(session.sample_rate)
            
            if dest_id not in dj.decks:
                return f"ERROR: Deck {dest_id} does not exist"
            
            deck = dj.decks[dest_id]
            
            if replace_mode or deck.audio is None or len(deck.audio) == 0:
                deck.audio = audio.copy()
                action = "Loaded to"
            else:
                deck.audio = np.concatenate([deck.audio, audio])
                action = "Appended to"
            
            # Calculate metrics
            dur = len(deck.audio) / session.sample_rate
            peak = np.max(np.abs(deck.audio))
            rms = np.sqrt(np.mean(deck.audio ** 2))
            
            if wb.ref.source_type == SourceType.PENDING:
                wb.focus_deck(dest_id, session)
            
            return f"OK: {action} deck {dest_id}\n  {dur:.3f}s, peak={peak:.3f}, rms={rms:.3f}"
        except Exception as e:
            return f"ERROR: {e}"
    
    return "ERROR: Unknown destination type"


# ============================================================================
# PERSISTENCE COMMANDS (copied from original)
# ============================================================================

def cmd_save(session: "Session", args: List[str]) -> str:
    """Save user data (variables, macros, preferences).
    
    Usage:
      /save              Save to default location
      /save <path>       Save to specific path
    """
    try:
        from ..core.user_data import save_user_data
        path = args[0] if args else None
        result = save_user_data(session, path)
        return result
    except ImportError:
        # Fallback
        items = []
        if hasattr(session, 'variables') and session.variables:
            items.append(f"Variables ({len(session.variables)} items)")
        if hasattr(session, 'macros') and session.macros:
            items.append(f"Macros ({len(session.macros)} items)")
        
        if items:
            return f"SAVED: {', '.join(items)}, Preferences"
        return "Nothing to save"


def cmd_usrp(session: "Session", args: List[str]) -> str:
    """Load audio from user outputs folder.
    
    Usage:
      /usrp              List available files
      /usrp <index>      Load file by index
      /usrp <name>       Load file by name
    """
    from pathlib import Path
    
    outputs_dir = Path.home() / "mdma_outputs"
    if not outputs_dir.exists():
        return f"ERROR: Outputs directory not found: {outputs_dir}"
    
    # Get audio files
    audio_files = sorted([
        f for f in outputs_dir.iterdir() 
        if f.suffix.lower() in ('.wav', '.mp3', '.flac', '.ogg')
    ])
    
    if not audio_files:
        return f"No audio files in {outputs_dir}"
    
    if not args:
        # List files
        lines = [f"=== USER OUTPUTS ({outputs_dir}) ==="]
        for i, f in enumerate(audio_files[:20], 1):
            lines.append(f"  {i}. {f.name}")
        if len(audio_files) > 20:
            lines.append(f"  ... and {len(audio_files) - 20} more")
        lines.append(f"\nUse /usrp <n> to load")
        return "\n".join(lines)
    
    # Load file
    target = args[0]
    
    if target.isdigit():
        idx = int(target) - 1
        if 0 <= idx < len(audio_files):
            target_file = audio_files[idx]
        else:
            return f"ERROR: Index {target} out of range (1-{len(audio_files)})"
    else:
        matches = [f for f in audio_files if target.lower() in f.name.lower()]
        if not matches:
            return f"ERROR: No file matching '{target}'"
        target_file = matches[0]
    
    # Load the file
    try:
        import soundfile as sf
        audio, sr = sf.read(str(target_file))
        
        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sr != session.sample_rate:
            ratio = session.sample_rate / sr
            new_len = int(len(audio) * ratio)
            x_old = np.linspace(0, 1, len(audio))
            x_new = np.linspace(0, 1, new_len)
            audio = np.interp(x_new, x_old, audio)
        
        wb = get_working_buffer()
        wb.bind_session(session)
        wb.set_pending(audio, f"usrp:{target_file.name}", session)
        
        dur = len(audio) / session.sample_rate
        return f"OK: Loaded {target_file.name} ({dur:.2f}s) -> working buffer (pending)"
        
    except Exception as e:
        return f"ERROR: Failed to load: {e}"


def cmd_usrp_list(session: "Session", args: List[str]) -> str:
    """List user outputs folder."""
    return cmd_usrp(session, [])


# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

def get_working_commands() -> dict:
    """Return working buffer commands."""
    return {
        # Working buffer
        'wb': cmd_wb,
        'wbc': cmd_wbc,
        
        # Audio sources debug
        'as': cmd_as,
        'audiosources': cmd_as,
        
        # Append/commit
        'a': cmd_a,
        'append': cmd_a,
        
        # Save
        'save': cmd_save,
        
        # User outputs
        'usrp': cmd_usrp,
        'usrplist': cmd_usrp_list,
    }


# Re-export for compatibility
AudioSource = AudioRef
AudioSourceType = SourceType
