"""Audio I/O helpers for the Session object.

This module holds the playback, preview, and wav-writing methods that used
to live directly on ``core/session.py``. They were extracted during the
backend V2 repo-prep to keep Session focused on orchestration and to give
the SignalFlow-backed backend a narrower surface to grow against.

Every function in this module takes ``self`` (a Session instance) as its
first argument and is re-bound onto the Session class at the bottom of
``core/session.py``. Callers still use ``session.play(...)``,
``session.play_last_buffer()``, etc. — no behavioural change, only
physical location.

BUILD ID: audio_io_v1 (extracted from session_v14.2_chunk3)
"""

from __future__ import annotations

import os
import wave
from typing import TYPE_CHECKING

import numpy as np  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .session import Session


def _play_buffer(self: "Session", buffer: np.ndarray, volume: float = 0.8) -> bool:
    """Play audio buffer using in-house playback engine.

    Uses sounddevice/simpleaudio/pyaudio for direct playback.
    No external media player calls.

    Parameters
    ----------
    buffer : np.ndarray
        Audio buffer to play
    volume : float
        Playback volume (0.0-1.0)

    Returns
    -------
    bool
        True if playback started successfully
    """
    try:
        from ..dsp.playback import play
        # Normalize buffer before playback
        data = self._normalize_output(buffer, target_db=-3.0)
        return play(data, self.sample_rate, blocking=False, volume=volume)
    except ImportError:
        # Fallback to file-based playback
        return self._play_via_file(buffer)
    except Exception as e:
        print(f"[session] playback error: {e}")
        return self._play_via_file(buffer)


def _play_via_file(self: "Session", buffer: np.ndarray) -> bool:
    """Fallback: write to temp file and open with system player.

    Only used if in-house playback fails.
    Supports mono (1D) and stereo (N,2) buffers.
    """
    try:
        path = os.path.join(self.temp_dir, "preview.wav")
        data = self._normalize_output(buffer, target_db=-3.0)
        # Determine channel count
        if data.ndim == 2 and data.shape[1] == 2:
            n_channels = 2
            # Interleave stereo samples for WAV
            interleaved = np.empty(data.shape[0] * 2, dtype=np.float64)
            interleaved[0::2] = data[:, 0]
            interleaved[1::2] = data[:, 1]
            data_int16 = np.int16(np.clip(interleaved * 32767, -32767, 32767))
        else:
            n_channels = 1
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            data_int16 = np.int16(np.clip(data * 32767, -32767, 32767))
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(data_int16.tobytes())
        self._open_file(path)
        return True
    except Exception:
        return False


def _open_file(self: "Session", path: str) -> None:
    """Fallback: Open a file using the operating system's default handler.

    Only used if in-house playback is unavailable.
    On Windows this uses os.startfile(), on macOS 'open', and on
    Linux 'xdg-open'.  Errors are silently ignored.
    """
    import platform
    import subprocess
    import os as _os
    try:
        system = platform.system()
        if system == 'Windows':
            _os.startfile(path)  # type: ignore[attr-defined]
        elif system == 'Darwin':
            # Use Popen to avoid blocking and suppress output
            subprocess.Popen(['open', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Attempt to use xdg-open; suppress output
            subprocess.Popen(['xdg-open', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        # If opening fails, ignore
        pass


def play(self: "Session", volume: float = 0.8) -> str:
    """Play audio using in-house playback (track-first, FX-aware).

    Priority:
    1) Mix all tracks (applies per-track FX, gain, pan, and master FX)
    2) Last buffer processed through buffer FX chain
    """
    audio = None
    source = ''

    # Prefer mixed track output (includes FX, gain, pan)
    try:
        track = self.get_current_track()
        t_audio = track.get('audio')
        if t_audio is not None:
            if t_audio.ndim == 2:
                peak = float(np.max(np.abs(t_audio)))
            elif t_audio.ndim == 1 and len(t_audio) > 0:
                peak = float(np.max(np.abs(t_audio)))
            else:
                peak = 0.0
            if peak > 0.0:
                # Use mix_tracks to get fully-processed audio
                audio = self.mix_tracks()
                source = f"mix ({len(self.tracks)} tracks)"
    except Exception:
        pass

    # Fallback: last_buffer with buffer FX chain
    if audio is None and self.last_buffer is not None:
        if hasattr(self.last_buffer, '__len__') and len(self.last_buffer) > 0:
            audio = self.last_buffer
            # Apply buffer FX chain if present
            if self.buffer_fx_chain:
                try:
                    if audio.ndim == 2 and audio.shape[1] == 2:
                        left = self.apply_fx_chain(audio[:, 0], self.buffer_fx_chain)
                        right = self.apply_fx_chain(audio[:, 1], self.buffer_fx_chain)
                        audio = np.column_stack([left, right])
                    else:
                        audio = self.apply_fx_chain(audio, self.buffer_fx_chain)
                except Exception:
                    pass
            source = 'last'

    if audio is None or (hasattr(audio, '__len__') and len(audio) == 0):
        return "ERROR: No audio to play. Create audio with /tone or write into a track with /TWRITE."

    duration = audio.shape[0] / self.sample_rate if audio.ndim >= 1 else len(audio) / self.sample_rate

    if self._play_buffer(audio, volume):
        try:
            from ..dsp.playback import get_status
            st = get_status()
            state = st.get('state', 'playing')
        except Exception:
            state = 'playing'
        return f"OK: {state} {source} ({duration:.2f}s)"
    return "ERROR: Playback failed. Try /play (file preview) as fallback."


def preview_track(self: "Session", track_index: int = None, include_master: bool = True) -> np.ndarray:
    """Render a single track with its FX chain, gain, pan, and
    optionally the master FX chain applied.

    When *include_master* is True (the default), the preview goes
    through the same signal path as ``mix_tracks()`` so users hear
    effects exactly as they would in the final render.  Pass
    ``include_master=False`` to hear the track in isolation.

    Returns processed stereo ``(N, 2)`` buffer.
    """
    if track_index is None:
        track_index = self.current_track_index
    if track_index < 0 or track_index >= len(self.tracks):
        return np.zeros((0, 2), dtype=np.float64)

    t = self.tracks[track_index]
    buf = t.get('audio')
    if buf is None or buf.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)

    # Ensure stereo
    if buf.ndim == 1:
        buf = np.column_stack([buf, buf])
    buf = buf.astype(np.float64, copy=True)

    # Apply gain
    gain = float(t.get('gain', 1.0))
    if gain != 1.0:
        buf = buf * gain

    # Apply per-track FX chain
    chain = t.get('fx_chain', []) or self.track_fx_chain
    if chain:
        try:
            left = self.apply_fx_chain(buf[:, 0], chain)
            right = self.apply_fx_chain(buf[:, 1], chain)
            buf = np.column_stack([left, right])
        except Exception:
            pass

    # Apply pan
    pan = float(t.get('pan', 0.0))
    pan = max(-1.0, min(1.0, pan))
    angle = (pan + 1.0) * 0.25 * np.pi
    buf[:, 0] *= float(np.cos(angle))
    buf[:, 1] *= float(np.sin(angle))

    # Apply master FX chain so preview matches final mix
    if include_master and self.master_fx_chain:
        try:
            left = self.apply_fx_chain(buf[:, 0], self.master_fx_chain)
            right = self.apply_fx_chain(buf[:, 1], self.master_fx_chain)
            buf = np.column_stack([left, right])
        except Exception:
            pass

    return buf


def stop_playback(self: "Session") -> str:
    """Stop any active playback.

    Returns
    -------
    str
        Status message
    """
    try:
        from ..dsp.playback import stop
        stop()
        return "STOPPED"
    except Exception:
        return "OK (no active playback)"


def playback_status(self: "Session") -> dict:
    """Get current playback status.

    Returns
    -------
    dict
        Playback status info
    """
    try:
        from ..dsp.playback import get_status
        return get_status()
    except Exception:
        return {"state": "unknown", "backend": "none"}


def play_last_buffer(self: "Session") -> str:
    """Legacy method: Write buffer to WAV file and return path.

    Prefer using play() for direct playback instead.
    If tracks have audio, renders via mix_tracks() to include FX.
    Supports mono and stereo buffers.
    """
    # Prefer mix_tracks output (includes all FX)
    data_source = None
    try:
        has_track_audio = any(
            t.get('audio') is not None and float(np.max(np.abs(t['audio']))) > 0
            for t in self.tracks
        )
        if has_track_audio:
            data_source = self.mix_tracks()
    except Exception:
        pass

    if data_source is None or (hasattr(data_source, '__len__') and len(data_source) == 0):
        data_source = self.last_buffer

    if data_source is None or (hasattr(data_source, '__len__') and len(data_source) == 0):
        raise RuntimeError("no audio buffer to play")
    path = os.path.join(self.temp_dir, "preview.wav")
    # Normalize to consistent output level (-3dB) and convert to int16
    data = self._normalize_output(data_source, target_db=-3.0)
    if data.ndim == 2 and data.shape[1] == 2:
        n_channels = 2
        interleaved = np.empty(data.shape[0] * 2, dtype=np.float64)
        interleaved[0::2] = data[:, 0]
        interleaved[1::2] = data[:, 1]
        data_int16 = np.int16(np.clip(interleaved * 32767, -32767, 32767))
    else:
        n_channels = 1
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data_int16 = np.int16(np.clip(data * 32767, -32767, 32767))
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data_int16.tobytes())
    return path


def full_render(self: "Session") -> str:
    """Write audio to a WAV file for final output.

    If tracks contain audio, renders via ``mix_tracks()`` (which
    applies per-track FX, gain, pan, and master FX).  Otherwise
    falls back to ``last_buffer``.

    The file is written in the current working directory with a
    unique name based on the project and sketch counters.
    Supports mono and stereo buffers.
    Returns the path to the created file.
    """
    # Prefer mix_tracks output (includes all FX)
    data_source = None
    try:
        has_track_audio = any(
            t.get('audio') is not None and float(np.max(np.abs(t['audio']))) > 0
            for t in self.tracks
        )
        if has_track_audio:
            data_source = self.mix_tracks()
    except Exception:
        pass

    if data_source is None or (hasattr(data_source, '__len__') and len(data_source) == 0):
        data_source = self.last_buffer

    if data_source is None or (hasattr(data_source, '__len__') and len(data_source) == 0):
        raise RuntimeError("no audio buffer to render")

    # Determine file format and extension
    fmt = getattr(self, 'output_format', 'wav').lower()
    bit_depth = getattr(self, 'output_bit_depth', 16)
    ext = 'flac' if fmt == 'flac' else 'wav'

    # Determine file name
    base_name = f"render_{self.project_count}_{self.sketch_count}_{self.file_count}.{ext}"
    path = os.path.join(os.getcwd(), base_name)
    # Normalize to consistent output level (-3dB) and convert to target format
    data = self._normalize_output(data_source, target_db=-3.0)
    if data.ndim == 2 and data.shape[1] == 2:
        n_channels = 2
        interleaved = np.empty(data.shape[0] * 2, dtype=np.float64)
        interleaved[0::2] = data[:, 0]
        interleaved[1::2] = data[:, 1]
    else:
        n_channels = 1
        if data.ndim > 1:
            interleaved = np.mean(data, axis=1)
        else:
            interleaved = data

    if fmt == 'flac':
        # Write FLAC using soundfile if available, fall back to WAV
        try:
            import soundfile as sf
            # Reshape for soundfile: (samples, channels)
            if n_channels == 2:
                sf_data = np.column_stack([interleaved[0::2], interleaved[1::2]])
            else:
                sf_data = interleaved
            subtype = 'PCM_24' if bit_depth == 24 else 'PCM_16'
            sf.write(path, sf_data, self.sample_rate, subtype=subtype, format='FLAC')
            return path
        except ImportError:
            # Fall back to WAV — warn user instead of silent downgrade
            print("[WARNING] FLAC export requires 'soundfile' package. "
                  "Install with: pip install soundfile")
            print("[WARNING] Falling back to WAV format.")
            ext = 'wav'
            base_name = f"render_{self.project_count}_{self.sketch_count}_{self.file_count}.wav"
            path = os.path.join(os.getcwd(), base_name)

    # WAV output
    if bit_depth == 24:
        # 24-bit WAV: pack as 3-byte samples
        samples_clipped = np.clip(interleaved * 8388607, -8388607, 8388607).astype(np.int32)
        raw = bytearray()
        for s in samples_clipped:
            val = int(s) & 0xFFFFFF
            raw.append(val & 0xFF)
            raw.append((val >> 8) & 0xFF)
            raw.append((val >> 16) & 0xFF)
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(3)  # 24-bit = 3 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(bytes(raw))
    else:
        # Standard 16-bit WAV
        data_int16 = np.int16(np.clip(interleaved * 32767, -32767, 32767))
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(data_int16.tobytes())
    return path


def bind_to(session_cls) -> None:
    """Attach every public audio I/O function above to ``session_cls``.

    Called once from ``core/session.py`` after the class body finishes, so
    instances end up with bound methods that behave identically to the
    originals.
    """
    session_cls._play_buffer = _play_buffer
    session_cls._play_via_file = _play_via_file
    session_cls._open_file = _open_file
    session_cls.play = play
    session_cls.preview_track = preview_track
    session_cls.stop_playback = stop_playback
    session_cls.playback_status = playback_status
    session_cls.play_last_buffer = play_last_buffer
    session_cls.full_render = full_render
