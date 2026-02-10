"""Audio Playback Module - In-House Audio Playback for MDMA.

This module provides cross-platform audio playback without relying on
external media players. Uses sounddevice as primary backend with
fallbacks for compatibility.

FEATURES:
- Direct audio buffer playback
- Non-blocking playback with stop control
- Playback status tracking
- Volume control
- Cross-platform support (Windows, macOS, Linux)

BUILD ID: playback_v14.2_chunk3
"""

from __future__ import annotations

import threading
import time
import queue
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 1
DEFAULT_VOLUME = 0.8
FADE_SAMPLES = 512  # Fade in/out to prevent clicks


# ============================================================================
# ENUMS
# ============================================================================

class PlaybackState(Enum):
    """Playback state."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    FINISHED = "finished"


# ============================================================================
# PLAYBACK ENGINE
# ============================================================================

@dataclass
class PlaybackEngine:
    """In-house audio playback engine.
    
    Provides direct audio playback without calling external media players.
    Uses sounddevice as the primary backend.
    
    Attributes
    ----------
    sample_rate : int
        Audio sample rate
    channels : int
        Number of audio channels (1=mono, 2=stereo)
    volume : float
        Playback volume (0.0-1.0)
    state : PlaybackState
        Current playback state
    """
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    volume: float = DEFAULT_VOLUME
    
    # Internal state
    _state: PlaybackState = field(default=PlaybackState.STOPPED, repr=False)
    _buffer: Optional[np.ndarray] = field(default=None, repr=False)
    _position: int = field(default=0, repr=False)
    _stream: Any = field(default=None, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _backend: str = field(default="none", repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def __post_init__(self):
        self._detect_backend()
    
    @property
    def state(self) -> PlaybackState:
        """Get current playback state."""
        return self._state
    
    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._state == PlaybackState.PLAYING
    
    @property
    def position(self) -> float:
        """Get current playback position in seconds."""
        if self._buffer is None:
            return 0.0
        return self._position / self.sample_rate
    
    @property
    def duration(self) -> float:
        """Get total duration of loaded buffer in seconds."""
        if self._buffer is None:
            return 0.0
        return len(self._buffer) / self.sample_rate
    
    @property
    def progress(self) -> float:
        """Get playback progress as percentage (0-100)."""
        if self._buffer is None or len(self._buffer) == 0:
            return 0.0
        return (self._position / len(self._buffer)) * 100.0
    
    @property
    def backend(self) -> str:
        """Get name of active audio backend."""
        return self._backend
    
    def _detect_backend(self) -> None:
        """Detect available audio backend."""
        # Try sounddevice first (best cross-platform support)
        try:
            import sounddevice as sd
            sd.query_devices()  # Test if audio devices available
            self._backend = "sounddevice"
            return
        except Exception:
            pass
        
        # Try simpleaudio
        try:
            import simpleaudio
            self._backend = "simpleaudio"
            return
        except Exception:
            pass
        
        # Try pyaudio
        try:
            import pyaudio
            self._backend = "pyaudio"
            return
        except Exception:
            pass
        
        # Fallback to wave file + system
        self._backend = "fallback"
    
    def _apply_volume_and_fade(self, buffer: np.ndarray) -> np.ndarray:
        """Apply volume and fade in/out to prevent clicks."""
        out = buffer.astype(np.float64) * self.volume
        
        # Fade in at start
        fade_len = min(FADE_SAMPLES, len(out) // 4)
        if fade_len > 0:
            fade_in = np.linspace(0, 1, fade_len)
            out[:fade_len] *= fade_in
        
        # Fade out at end
        if fade_len > 0:
            fade_out = np.linspace(1, 0, fade_len)
            out[-fade_len:] *= fade_out
        
        return out
    
    def play(
        self,
        buffer: np.ndarray,
        sample_rate: Optional[int] = None,
        blocking: bool = False,
        on_complete: Optional[Callable] = None,
    ) -> bool:
        """Play an audio buffer.
        
        Parameters
        ----------
        buffer : np.ndarray
            Audio buffer to play (float64, -1.0 to 1.0)
        sample_rate : int, optional
            Sample rate (uses engine default if not specified)
        blocking : bool
            If True, wait for playback to complete
        on_complete : Callable, optional
            Callback function when playback completes
            
        Returns
        -------
        bool
            True if playback started successfully
        """
        # Stop any existing playback
        self.stop()
        
        if buffer is None or len(buffer) == 0:
            return False
        
        # Update sample rate if specified
        if sample_rate is not None:
            self.sample_rate = sample_rate
        
        # Store and prepare buffer
        with self._lock:
            self._buffer = self._apply_volume_and_fade(buffer.copy())
            self._position = 0
            self._stop_event.clear()
            self._state = PlaybackState.PLAYING
        
        # Choose playback method based on backend
        if self._backend == "sounddevice":
            success = self._play_sounddevice(blocking, on_complete)
        elif self._backend == "simpleaudio":
            success = self._play_simpleaudio(blocking, on_complete)
        elif self._backend == "pyaudio":
            success = self._play_pyaudio(blocking, on_complete)
        else:
            success = self._play_fallback(blocking, on_complete)
        
        if not success:
            self._state = PlaybackState.STOPPED
        
        return success
    
    def _play_sounddevice(
        self,
        blocking: bool,
        on_complete: Optional[Callable],
    ) -> bool:
        """Play using sounddevice backend."""
        try:
            import sounddevice as sd
            
            buffer = self._buffer
            if buffer is None:
                return False
            
            # Ensure proper shape for sounddevice
            if buffer.ndim == 1:
                buffer = buffer.reshape(-1, 1)
            
            # Convert to float32 for sounddevice
            buffer = buffer.astype(np.float32)
            
            def callback_finished():
                with self._lock:
                    self._state = PlaybackState.FINISHED
                if on_complete:
                    on_complete()
            
            if blocking:
                sd.play(buffer, self.sample_rate)
                sd.wait()
                callback_finished()
            else:
                def play_thread():
                    try:
                        sd.play(buffer, self.sample_rate)
                        sd.wait()
                        if not self._stop_event.is_set():
                            callback_finished()
                    except Exception:
                        pass
                
                thread = threading.Thread(target=play_thread, daemon=True)
                thread.start()
            
            return True
            
        except Exception as e:
            print(f"[playback] sounddevice error: {e}")
            return False
    
    def _play_simpleaudio(
        self,
        blocking: bool,
        on_complete: Optional[Callable],
    ) -> bool:
        """Play using simpleaudio backend."""
        try:
            import simpleaudio as sa
            
            buffer = self._buffer
            if buffer is None:
                return False
            
            # Convert to int16
            audio_int16 = np.int16(np.clip(buffer, -1.0, 1.0) * 32767)
            
            play_obj = sa.play_buffer(
                audio_int16.tobytes(),
                num_channels=1,
                bytes_per_sample=2,
                sample_rate=self.sample_rate,
            )
            
            self._stream = play_obj
            
            def callback_finished():
                with self._lock:
                    self._state = PlaybackState.FINISHED
                if on_complete:
                    on_complete()
            
            if blocking:
                play_obj.wait_done()
                callback_finished()
            else:
                def wait_thread():
                    play_obj.wait_done()
                    if not self._stop_event.is_set():
                        callback_finished()
                
                thread = threading.Thread(target=wait_thread, daemon=True)
                thread.start()
            
            return True
            
        except Exception as e:
            print(f"[playback] simpleaudio error: {e}")
            return False
    
    def _play_pyaudio(
        self,
        blocking: bool,
        on_complete: Optional[Callable],
    ) -> bool:
        """Play using pyaudio backend."""
        try:
            import pyaudio
            
            buffer = self._buffer
            if buffer is None:
                return False
            
            # Convert to int16
            audio_int16 = np.int16(np.clip(buffer, -1.0, 1.0) * 32767)
            
            p = pyaudio.PyAudio()
            
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
            )
            
            def callback_finished():
                with self._lock:
                    self._state = PlaybackState.FINISHED
                if on_complete:
                    on_complete()
            
            def play_audio():
                try:
                    chunk_size = 1024
                    for i in range(0, len(audio_int16), chunk_size):
                        if self._stop_event.is_set():
                            break
                        chunk = audio_int16[i:i + chunk_size]
                        stream.write(chunk.tobytes())
                        self._position = i
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    if not self._stop_event.is_set():
                        callback_finished()
                except Exception:
                    pass
            
            if blocking:
                play_audio()
            else:
                thread = threading.Thread(target=play_audio, daemon=True)
                thread.start()
            
            return True
            
        except Exception as e:
            print(f"[playback] pyaudio error: {e}")
            return False
    
    def _play_fallback(
        self,
        blocking: bool,
        on_complete: Optional[Callable],
    ) -> bool:
        """Fallback playback - writes to temp file and reports error."""
        print("[playback] WARNING: No audio backend available")
        print("[playback] Install sounddevice: pip install sounddevice")
        self._state = PlaybackState.STOPPED
        return False
    
    def stop(self) -> None:
        """Stop playback."""
        self._stop_event.set()
        
        with self._lock:
            self._state = PlaybackState.STOPPED
        
        # Stop sounddevice if active
        if self._backend == "sounddevice":
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass
        
        # Stop simpleaudio if active
        if self._stream is not None and self._backend == "simpleaudio":
            try:
                self._stream.stop()
            except Exception:
                pass
        
        self._stream = None
    
    def set_volume(self, volume: float) -> None:
        """Set playback volume (0.0-1.0)."""
        self.volume = max(0.0, min(1.0, volume))
    
    def get_status(self) -> dict:
        """Get playback status."""
        return {
            "state": self._state.value,
            "backend": self._backend,
            "position": self.position,
            "duration": self.duration,
            "progress": self.progress,
            "volume": self.volume,
            "sample_rate": self.sample_rate,
        }


# ============================================================================
# GLOBAL PLAYBACK ENGINE
# ============================================================================

# Singleton playback engine
_engine: Optional[PlaybackEngine] = None


def get_engine() -> PlaybackEngine:
    """Get or create the global playback engine."""
    global _engine
    if _engine is None:
        _engine = PlaybackEngine()
    return _engine


def play(
    buffer: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    blocking: bool = False,
    volume: float = DEFAULT_VOLUME,
) -> bool:
    """Play an audio buffer using the global engine.
    
    Parameters
    ----------
    buffer : np.ndarray
        Audio buffer to play
    sample_rate : int
        Sample rate
    blocking : bool
        Wait for playback to complete
    volume : float
        Playback volume (0.0-1.0)
        
    Returns
    -------
    bool
        True if playback started
    """
    engine = get_engine()
    engine.set_volume(volume)
    return engine.play(buffer, sample_rate, blocking)


def stop() -> None:
    """Stop any active playback."""
    engine = get_engine()
    engine.stop()


def is_playing() -> bool:
    """Check if audio is currently playing."""
    engine = get_engine()
    return engine.is_playing


def get_status() -> dict:
    """Get playback status."""
    engine = get_engine()
    return engine.get_status()


def get_backend() -> str:
    """Get name of active audio backend."""
    engine = get_engine()
    return engine.backend


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def play_tone(
    frequency: float = 440.0,
    duration: float = 0.5,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    volume: float = 0.5,
) -> bool:
    """Play a simple sine tone (useful for testing).
    
    Parameters
    ----------
    frequency : float
        Tone frequency in Hz
    duration : float
        Duration in seconds
    sample_rate : int
        Sample rate
    volume : float
        Volume (0.0-1.0)
        
    Returns
    -------
    bool
        True if playback started
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    buffer = np.sin(2 * np.pi * frequency * t) * volume
    return play(buffer, sample_rate, blocking=False, volume=1.0)


def play_file(
    path: str,
    volume: float = DEFAULT_VOLUME,
    blocking: bool = False,
) -> bool:
    """Play an audio file.
    
    Parameters
    ----------
    path : str
        Path to audio file (WAV)
    volume : float
        Playback volume
    blocking : bool
        Wait for playback to complete
        
    Returns
    -------
    bool
        True if playback started
    """
    try:
        import wave
        
        with wave.open(path, 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            
            raw_data = wf.readframes(n_frames)
        
        # Convert to numpy array
        if sample_width == 2:
            buffer = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64) / 32767.0
        elif sample_width == 1:
            buffer = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float64) - 128) / 128.0
        else:
            return False
        
        # Convert stereo to mono if needed
        if n_channels == 2:
            buffer = buffer.reshape(-1, 2).mean(axis=1)
        
        return play(buffer, sample_rate, blocking, volume)
        
    except Exception as e:
        print(f"[playback] Error loading file: {e}")
        return False


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Classes
    'PlaybackEngine',
    'PlaybackState',
    # Global functions
    'get_engine',
    'play',
    'stop',
    'is_playing',
    'get_status',
    'get_backend',
    # Convenience
    'play_tone',
    'play_file',
    # Constants
    'DEFAULT_SAMPLE_RATE',
    'DEFAULT_VOLUME',
]
