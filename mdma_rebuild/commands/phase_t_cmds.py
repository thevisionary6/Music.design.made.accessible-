"""Phase T: Full System Audit — Commands for song-ready production.

T.1  Undo/Redo System          /undo, /redo, /snapshot
T.2  Song Structure            /section, /pchain
T.3  Render & Export           /export, /master_gain
T.4  Missing Commands          /crossover, /dup, /metronome, /swap
T.5  Generative Output Routing /commit
T.6  Save/Load Persistence     /autosave
T.7  Playback Fixes            /pos, /seek
T.8  Audio Import              (sample rate conversion helper)

BUILD ID: phase_t_cmds_v1.0
"""

from __future__ import annotations

import os
import time
import json
import threading
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.session import Session


# ============================================================================
# T.1 — UNDO / REDO / SNAPSHOT
# ============================================================================

def cmd_undo(session: "Session", args: List[str]) -> str:
    """Undo last destructive operation.

    Usage:
      /undo                Undo working buffer
      /undo track [n]      Undo track n (default: current)
      /undo snapshot       Restore last parameter snapshot
    """
    if args:
        sub = args[0].lower()
        if sub in ('track', 't'):
            idx = session.current_track_index
            if len(args) > 1:
                try:
                    idx = int(args[1]) - 1
                except ValueError:
                    return f"ERROR: Invalid track index '{args[1]}'"
            if session.pop_undo('track', idx):
                return f"OK: Undid last operation on track {idx + 1}"
            return f"ERROR: Nothing to undo on track {idx + 1}"
        elif sub in ('snapshot', 'snap', 'ss'):
            if session.restore_snapshot(-1):
                return "OK: Restored last parameter snapshot"
            return "ERROR: No snapshots saved"
    # Default: undo working buffer
    if session.pop_undo('working'):
        dur = len(session.working_buffer) / session.sample_rate
        return f"OK: Undid last operation on working buffer ({dur:.2f}s)"
    return "ERROR: Nothing to undo"


def cmd_redo(session: "Session", args: List[str]) -> str:
    """Redo previously undone operation.

    Usage:
      /redo                Redo working buffer
      /redo track [n]      Redo track n (default: current)
    """
    if args:
        sub = args[0].lower()
        if sub in ('track', 't'):
            idx = session.current_track_index
            if len(args) > 1:
                try:
                    idx = int(args[1]) - 1
                except ValueError:
                    return f"ERROR: Invalid track index '{args[1]}'"
            if session.pop_redo('track', idx):
                return f"OK: Redid operation on track {idx + 1}"
            return f"ERROR: Nothing to redo on track {idx + 1}"
    if session.pop_redo('working'):
        dur = len(session.working_buffer) / session.sample_rate
        return f"OK: Redid operation on working buffer ({dur:.2f}s)"
    return "ERROR: Nothing to redo"


def cmd_snapshot(session: "Session", args: List[str]) -> str:
    """Save or restore a session parameter snapshot.

    Usage:
      /snapshot            Save current parameters
      /snapshot save       Same as above
      /snapshot list       List saved snapshots
      /snapshot restore [n] Restore snapshot n (default: last)
    """
    if not args or args[0].lower() == 'save':
        idx = session.save_snapshot()
        return f"OK: Saved snapshot #{idx} (BPM={session.bpm}, {len(session.effects)} FX)"

    sub = args[0].lower()
    if sub == 'list':
        if not session._snapshots:
            return "No snapshots saved"
        lines = ["Saved Snapshots:"]
        for i, snap in enumerate(session._snapshots):
            lines.append(f"  #{i}: BPM={snap['bpm']}, "
                         f"voices={snap['voice_count']}, "
                         f"gain={snap['master_gain']}dB")
        return '\n'.join(lines)

    if sub in ('restore', 'load'):
        idx = -1
        if len(args) > 1:
            try:
                idx = int(args[1])
            except ValueError:
                return f"ERROR: Invalid snapshot index '{args[1]}'"
        if session.restore_snapshot(idx):
            display_idx = idx if idx >= 0 else len(session._snapshots) + idx
            return f"OK: Restored snapshot #{display_idx}"
        return "ERROR: Snapshot not found"

    return "Usage: /snapshot [save|list|restore [n]]"


# ============================================================================
# T.2 — SONG STRUCTURE & ARRANGEMENT
# ============================================================================

def cmd_section(session: "Session", args: List[str]) -> str:
    """Manage song sections for arrangement.

    Usage:
      /section add <name> <start_bar> <end_bar>   Define a section
      /section list                                List all sections
      /section goto <name>                         Move write position to section start
      /section copy <name> <to_bar>                Copy section audio to new position
      /section move <name> <to_bar>                Move section audio to new position
      /section remove <name>                       Remove section definition
    """
    if not args:
        return ("Usage: /section add <name> <start> <end>\n"
                "       /section list | goto | copy | move | remove")

    sub = args[0].lower()

    if sub == 'add':
        if len(args) < 4:
            return "Usage: /section add <name> <start_bar> <end_bar>"
        name = args[1]
        try:
            start_bar = int(args[2])
            end_bar = int(args[3])
        except ValueError:
            return "ERROR: start_bar and end_bar must be integers"
        if end_bar <= start_bar:
            return "ERROR: end_bar must be greater than start_bar"
        # Remove existing section with same name
        session.sections = [s for s in session.sections if s['name'] != name]
        session.sections.append({
            'name': name,
            'start_bar': start_bar,
            'end_bar': end_bar
        })
        session.sections.sort(key=lambda s: s['start_bar'])
        dur_bars = end_bar - start_bar
        return f"OK: Section '{name}' — bars {start_bar}-{end_bar} ({dur_bars} bars)"

    if sub == 'list':
        if not session.sections:
            return "No sections defined. Use /section add <name> <start> <end>"
        lines = ["Song Sections:"]
        for s in session.sections:
            bars = s['end_bar'] - s['start_bar']
            start_sec = _bar_to_seconds(s['start_bar'], session.bpm)
            end_sec = _bar_to_seconds(s['end_bar'], session.bpm)
            lines.append(f"  {s['name']:12s} bars {s['start_bar']:3d}-{s['end_bar']:3d} "
                         f"({bars} bars, {start_sec:.1f}s-{end_sec:.1f}s)")
        return '\n'.join(lines)

    if sub == 'goto':
        if len(args) < 2:
            return "Usage: /section goto <name>"
        name = args[1]
        sec = _find_section(session, name)
        if sec is None:
            return f"ERROR: Section '{name}' not found"
        seconds = _bar_to_seconds(sec['start_bar'], session.bpm)
        session.set_track_cursor_seconds(seconds)
        return f"OK: Write position moved to bar {sec['start_bar']} ({seconds:.2f}s)"

    if sub == 'copy':
        if len(args) < 3:
            return "Usage: /section copy <name> <to_bar>"
        name = args[1]
        try:
            to_bar = int(args[2])
        except ValueError:
            return "ERROR: to_bar must be an integer"
        return _section_copy_move(session, name, to_bar, move=False)

    if sub == 'move':
        if len(args) < 3:
            return "Usage: /section move <name> <to_bar>"
        name = args[1]
        try:
            to_bar = int(args[2])
        except ValueError:
            return "ERROR: to_bar must be an integer"
        return _section_copy_move(session, name, to_bar, move=True)

    if sub in ('remove', 'rm', 'delete', 'del'):
        if len(args) < 2:
            return "Usage: /section remove <name>"
        name = args[1]
        before = len(session.sections)
        session.sections = [s for s in session.sections if s['name'] != name]
        if len(session.sections) < before:
            return f"OK: Removed section '{name}'"
        return f"ERROR: Section '{name}' not found"

    return "Usage: /section add|list|goto|copy|move|remove"


def _bar_to_seconds(bar: int, bpm: float) -> float:
    """Convert bar number to seconds (4 beats per bar)."""
    return bar * 4.0 * 60.0 / bpm


def _bar_to_samples(bar: int, bpm: float, sr: int) -> int:
    """Convert bar number to sample position."""
    return int(_bar_to_seconds(bar, bpm) * sr)


def _find_section(session: "Session", name: str) -> Optional[dict]:
    """Find section by name (case-insensitive)."""
    name_lower = name.lower()
    for s in session.sections:
        if s['name'].lower() == name_lower:
            return s
    return None


def _section_copy_move(session: "Session", name: str, to_bar: int, move: bool) -> str:
    """Copy or move section audio to a new bar position."""
    sec = _find_section(session, name)
    if sec is None:
        return f"ERROR: Section '{name}' not found"

    sr = session.sample_rate
    bpm = session.bpm
    src_start = _bar_to_samples(sec['start_bar'], bpm, sr)
    src_end = _bar_to_samples(sec['end_bar'], bpm, sr)
    dst_start = _bar_to_samples(to_bar, bpm, sr)

    for i, track in enumerate(session.tracks):
        audio = track.get('audio')
        if audio is None:
            continue
        # Clamp source range
        actual_end = min(src_end, len(audio))
        if src_start >= actual_end:
            continue
        # Push undo before modifying each track
        session.push_undo('track', i)
        section_audio = audio[src_start:actual_end].copy()
        # Write to destination
        dst_end = min(dst_start + len(section_audio), len(audio))
        if dst_start < dst_end:
            track['audio'][dst_start:dst_end] = section_audio[:dst_end - dst_start]
        # Clear source if moving
        if move:
            track['audio'][src_start:actual_end] = 0.0

    action = "Moved" if move else "Copied"
    return f"OK: {action} section '{name}' to bar {to_bar}"


def cmd_pchain(session: "Session", args: List[str]) -> str:
    """Chain patterns into a longer sequence on the current track.

    Usage:
      /pchain <buf_a> <repeat_a> <buf_b> <repeat_b> ...

    Example:
      /pchain 1 4 2 2 1 4     Chain: buf1 x4, buf2 x2, buf1 x4
    """
    if not args or len(args) < 2:
        return ("Usage: /pchain <buf> <repeats> [<buf> <repeats> ...]\n"
                "  Example: /pchain 1 4 2 2 — buffer 1 x4 then buffer 2 x2")

    # Parse pairs
    pairs = []
    i = 0
    while i < len(args) - 1:
        try:
            buf_idx = int(args[i])
            repeats = int(args[i + 1])
            pairs.append((buf_idx, repeats))
            i += 2
        except ValueError:
            return f"ERROR: Invalid argument at position {i + 1}: '{args[i]}'"

    if not pairs:
        return "ERROR: No valid buffer/repeat pairs"

    # Build chained audio
    chain_parts = []
    for buf_idx, repeats in pairs:
        buf = session.buffers.get(buf_idx)
        if buf is None or len(buf) == 0:
            return f"ERROR: Buffer {buf_idx} is empty"
        for _ in range(max(1, repeats)):
            chain_parts.append(buf.copy())

    chained = np.concatenate(chain_parts)

    # Push undo for both targets BEFORE any mutations
    session.push_undo('working')
    session.push_undo('track', session.current_track_index)

    # Write to current track
    start, end = session.write_to_track(chained)
    dur = len(chained) / session.sample_rate

    # Also put in working buffer
    session.working_buffer = chained.copy()
    session.working_buffer_source = 'pchain'
    session.last_buffer = chained.copy()

    desc = " + ".join(f"buf{b}x{r}" for b, r in pairs)
    return f"OK: Chained {desc} — {dur:.2f}s, written to track at sample {start}"


# ============================================================================
# T.3 — RENDER & EXPORT PIPELINE
# ============================================================================

def cmd_export(session: "Session", args: List[str]) -> str:
    """Export stems, individual tracks, or render sections.

    Usage:
      /export stems [path]          Export each track as separate file
      /export track <n> [path]      Export single track
      /export section <name> [path] Render a single section
    """
    if not args:
        return ("Usage:\n"
                "  /export stems [path]           Export all tracks as stems\n"
                "  /export track <n> [path]       Export single track\n"
                "  /export section <name> [path]  Export a section")

    sub = args[0].lower()
    out_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'MDMA', 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    if sub == 'stems':
        path = args[1] if len(args) > 1 else out_dir
        os.makedirs(path, exist_ok=True)
        exported = 0
        for i, track in enumerate(session.tracks):
            audio = track.get('audio')
            if audio is None or np.max(np.abs(audio)) < 1e-7:
                continue
            # Apply per-track FX, gain, pan
            processed = session.preview_track(i, include_master=False)
            if processed is None or len(processed) == 0:
                continue
            fname = f"stem_{i + 1}_{track.get('name', f'track_{i + 1}')}.wav"
            fpath = os.path.join(path, fname)
            _write_wav(processed, session.sample_rate, fpath)
            exported += 1
        if exported == 0:
            return "ERROR: No tracks with audio to export"
        return f"OK: Exported {exported} stems to {path}"

    if sub == 'track':
        if len(args) < 2:
            return "Usage: /export track <n> [path]"
        try:
            track_idx = int(args[1]) - 1
        except ValueError:
            return f"ERROR: Invalid track index '{args[1]}'"
        if track_idx < 0 or track_idx >= len(session.tracks):
            return f"ERROR: Track {track_idx + 1} not found"
        processed = session.preview_track(track_idx, include_master=False)
        if processed is None or len(processed) == 0:
            return f"ERROR: Track {track_idx + 1} is empty"
        path = args[2] if len(args) > 2 else out_dir
        if os.path.isdir(path):
            t = session.tracks[track_idx]
            fname = f"track_{track_idx + 1}_{t.get('name', '')}.wav"
            path = os.path.join(path, fname)
        _write_wav(processed, session.sample_rate, path)
        dur = len(processed) / session.sample_rate
        return f"OK: Exported track {track_idx + 1} to {path} ({dur:.2f}s)"

    if sub == 'section':
        if len(args) < 2:
            return "Usage: /export section <name> [path]"
        name = args[1]
        sec = _find_section(session, name)
        if sec is None:
            return f"ERROR: Section '{name}' not found"
        sr = session.sample_rate
        bpm = session.bpm
        src_start = _bar_to_samples(sec['start_bar'], bpm, sr)
        src_end = _bar_to_samples(sec['end_bar'], bpm, sr)
        # Mix all tracks for that range
        mixed = _mix_range(session, src_start, src_end)
        if mixed is None or np.max(np.abs(mixed)) < 1e-7:
            return f"ERROR: Section '{name}' contains no audio"
        path = args[2] if len(args) > 2 else out_dir
        if os.path.isdir(path):
            path = os.path.join(path, f"section_{name}.wav")
        _write_wav(mixed, sr, path)
        dur = len(mixed) / sr
        return f"OK: Exported section '{name}' to {path} ({dur:.2f}s)"

    return "Usage: /export stems|track|section"


def _mix_range(session: "Session", start: int, end: int) -> np.ndarray:
    """Mix all tracks for a sample range."""
    length = end - start
    if length <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    mixed = np.zeros((length, 2), dtype=np.float64)
    has_solo = any(t.get('solo', False) for t in session.tracks)
    for i, track in enumerate(session.tracks):
        if track.get('mute', False):
            continue
        if has_solo and not track.get('solo', False):
            continue
        audio = track.get('audio')
        if audio is None:
            continue
        actual_end = min(end, len(audio))
        if start >= actual_end:
            continue
        seg = audio[start:actual_end].copy()
        if seg.ndim == 1:
            seg = np.column_stack([seg, seg])
        gain = float(track.get('gain', 1.0))
        seg = seg * gain
        pan = float(track.get('pan', 0.0))
        pan = max(-1.0, min(1.0, pan))
        angle = (pan + 1.0) * 0.25 * np.pi
        seg[:, 0] *= float(np.cos(angle))
        seg[:, 1] *= float(np.sin(angle))
        out_len = min(len(seg), length)
        mixed[:out_len] += seg[:out_len]
    # Apply master gain
    mg = getattr(session, 'master_gain', 0.0)
    if mg != 0.0:
        mixed *= 10.0 ** (mg / 20.0)
    return mixed


def _write_wav(audio: np.ndarray, sr: int, path: str) -> None:
    """Write audio to WAV file (16-bit)."""
    import wave as _wave
    if audio.size == 0:
        raise ValueError("Cannot write empty audio to WAV file")
    if audio.ndim == 2 and audio.shape[1] == 2:
        n_channels = 2
        interleaved = np.empty(audio.shape[0] * 2, dtype=np.float64)
        interleaved[0::2] = audio[:, 0]
        interleaved[1::2] = audio[:, 1]
    else:
        n_channels = 1
        interleaved = audio.ravel() if audio.ndim > 1 else audio
    peak = np.max(np.abs(interleaved))
    if peak > 1.0:
        interleaved = interleaved / peak
    data_int16 = np.int16(np.clip(interleaved * 32767, -32767, 32767))
    with _wave.open(path, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data_int16.tobytes())


def cmd_master_gain(session: "Session", args: List[str]) -> str:
    """Set or show master gain.

    Usage:
      /master_gain           Show current master gain
      /master_gain <dB>      Set master gain in dB (e.g., -3, 0, +6)
    """
    if not args:
        return f"Master gain: {session.master_gain:+.1f}dB"
    try:
        db = float(args[0])
        session.master_gain = db
        return f"OK: Master gain set to {db:+.1f}dB"
    except ValueError:
        return f"ERROR: Invalid dB value '{args[0]}'"


# ============================================================================
# T.4 — MISSING COMMAND IMPLEMENTATIONS
# ============================================================================

def cmd_crossover(session: "Session", args: List[str]) -> str:
    """Crossover two buffers using genetic breeding crossover.

    Usage:
      /crossover <buf_a> <buf_b> [method]

    Methods: temporal, spectral, blend, morphological, multi_point
    Default: temporal

    Result stored in working buffer and next empty buffer.
    """
    if len(args) < 2:
        return ("Usage: /crossover <buf_a> <buf_b> [method]\n"
                "  Methods: temporal, spectral, blend, morphological, multi_point")

    try:
        buf_a_idx = int(args[0])
        buf_b_idx = int(args[1])
    except ValueError:
        return "ERROR: Buffer indices must be integers"

    method = args[2].lower() if len(args) > 2 else 'temporal'

    audio_a = session.buffers.get(buf_a_idx)
    audio_b = session.buffers.get(buf_b_idx)

    if audio_a is None or len(audio_a) == 0:
        return f"ERROR: Buffer {buf_a_idx} is empty"
    if audio_b is None or len(audio_b) == 0:
        return f"ERROR: Buffer {buf_b_idx} is empty"

    try:
        from ..ai.breeding import (
            crossover_temporal, crossover_spectral, crossover_blend,
            crossover_morphological, crossover_multi_point
        )
    except ImportError:
        return "ERROR: Breeding module not available"

    # Normalize lengths
    max_len = max(len(audio_a), len(audio_b))
    a = np.zeros(max_len, dtype=np.float64)
    b = np.zeros(max_len, dtype=np.float64)
    a[:len(audio_a)] = audio_a[:max_len].astype(np.float64).ravel()[:max_len]
    b[:len(audio_b)] = audio_b[:max_len].astype(np.float64).ravel()[:max_len]

    methods = {
        'temporal': crossover_temporal,
        'spectral': crossover_spectral,
        'blend': crossover_blend,
        'morphological': crossover_morphological,
        'multi_point': crossover_multi_point,
    }

    fn = methods.get(method)
    if fn is None:
        return f"ERROR: Unknown method '{method}'. Use: {', '.join(methods)}"

    child = fn(a, b)

    # Normalize
    peak = np.max(np.abs(child))
    if peak > 0.95:
        child = child * (0.95 / peak)

    # Store result
    session.push_undo('working')
    session.working_buffer = child.copy()
    session.working_buffer_source = f'crossover_{method}'
    session.last_buffer = child.copy()

    dest = session.get_lowest_empty_buffer()
    session.store_in_buffer(dest, child)

    dur = len(child) / session.sample_rate
    return (f"OK: Crossover ({method}) of buffers {buf_a_idx} + {buf_b_idx}\n"
            f"  Result: {dur:.2f}s, stored in buffer {dest} and working buffer")


def cmd_dup(session: "Session", args: List[str]) -> str:
    """Duplicate a buffer.

    Usage:
      /dup [source] [dest]     Copy buffer source to dest
      /dup [source]            Copy to next empty buffer
      /dup                     Duplicate current buffer
    """
    src_idx = session.current_buffer_index
    if args:
        try:
            src_idx = int(args[0])
        except ValueError:
            return f"ERROR: Invalid buffer index '{args[0]}'"

    src = session.buffers.get(src_idx)
    if src is None or len(src) == 0:
        return f"ERROR: Buffer {src_idx} is empty"

    if len(args) > 1:
        try:
            dst_idx = int(args[1])
        except ValueError:
            return f"ERROR: Invalid destination index '{args[1]}'"
    else:
        dst_idx = session.get_lowest_empty_buffer()

    session.store_in_buffer(dst_idx, src.copy())
    dur = len(src) / session.sample_rate
    return f"OK: Duplicated buffer {src_idx} to buffer {dst_idx} ({dur:.2f}s)"


def cmd_metronome(session: "Session", args: List[str]) -> str:
    """Generate a metronome click track.

    Usage:
      /metronome [bars] [bpm]   Generate click track
      /metronome off             Clear metronome from working buffer

    Example:
      /metronome 4               4 bars at session BPM
      /metronome 8 120           8 bars at 120 BPM
    """
    if args and args[0].lower() == 'off':
        return "OK: Metronome cleared"

    bars = 4
    bpm = session.bpm

    if args:
        try:
            bars = int(args[0])
        except ValueError:
            pass
        if len(args) > 1:
            try:
                bpm = float(args[1])
            except ValueError:
                pass

    sr = session.sample_rate
    beats_per_bar = 4
    total_beats = bars * beats_per_bar
    beat_dur = 60.0 / bpm
    total_samples = int(total_beats * beat_dur * sr)

    click = np.zeros(total_samples, dtype=np.float64)

    for beat in range(total_beats):
        pos = int(beat * beat_dur * sr)
        # Downbeat is higher pitch
        freq = 1200.0 if beat % beats_per_bar == 0 else 800.0
        amp = 0.7 if beat % beats_per_bar == 0 else 0.4
        click_len = min(int(0.02 * sr), total_samples - pos)  # 20ms click
        if click_len > 0:
            t = np.arange(click_len) / sr
            tone = amp * np.sin(2 * np.pi * freq * t)
            # Apply fast decay envelope
            env = np.exp(-t * 80)
            click[pos:pos + click_len] += tone * env

    session.push_undo('working')
    session.working_buffer = click
    session.working_buffer_source = 'metronome'
    session.last_buffer = click.copy()

    dur = total_samples / sr
    return f"OK: Metronome — {bars} bars, {bpm:.0f} BPM, {total_beats} clicks, {dur:.2f}s"


def cmd_swap(session: "Session", args: List[str]) -> str:
    """Swap contents of two buffers.

    Usage:
      /swap <a> <b>        Swap buffers a and b
    """
    if len(args) < 2:
        return "Usage: /swap <a> <b>"
    try:
        a = int(args[0])
        b = int(args[1])
    except ValueError:
        return "ERROR: Buffer indices must be integers"
    if a < 0 or b < 0:
        return "ERROR: Buffer indices must be non-negative"
    if a == b:
        return "ERROR: Cannot swap a buffer with itself"

    buf_a = session.buffers.get(a, np.zeros(0, dtype=np.float64)).copy()
    buf_b = session.buffers.get(b, np.zeros(0, dtype=np.float64)).copy()

    session.store_in_buffer(a, buf_b)
    session.store_in_buffer(b, buf_a)

    dur_a = len(buf_a) / session.sample_rate if len(buf_a) > 0 else 0
    dur_b = len(buf_b) / session.sample_rate if len(buf_b) > 0 else 0
    return f"OK: Swapped buffer {a} ({dur_a:.2f}s) with buffer {b} ({dur_b:.2f}s)"


# ============================================================================
# T.5 — GENERATIVE OUTPUT ROUTING
# ============================================================================

def cmd_commit(session: "Session", args: List[str]) -> str:
    """Commit working buffer to current track and advance write position.

    Usage:
      /commit [track_n]      Write working buffer to track (default: current)

    The working buffer audio is written to the track at the current write
    position. The write position advances to the end of the written region.
    """
    if not session.has_real_working_audio():
        return "ERROR: Working buffer is empty. Generate audio first."

    track_idx_1based = None
    if args:
        try:
            track_idx_1based = int(args[0])
        except ValueError:
            return f"ERROR: Invalid track index '{args[0]}'"

    audio = session.working_buffer.copy()

    if track_idx_1based is not None:
        session.ensure_track_index(track_idx_1based)
        session.push_undo('track', track_idx_1based - 1)
        start, end = session.write_to_track(audio, track_idx=track_idx_1based)
        track_display = track_idx_1based
    else:
        track_display = session.current_track_index + 1
        session.push_undo('track', session.current_track_index)
        start, end = session.write_to_track(audio)

    dur = len(audio) / session.sample_rate
    start_sec = start / session.sample_rate
    end_sec = end / session.sample_rate
    return (f"OK: Committed {dur:.2f}s to track {track_display}\n"
            f"  Written at {start_sec:.2f}s-{end_sec:.2f}s\n"
            f"  Write position advanced to {end_sec:.2f}s")


# ============================================================================
# T.6 — SAVE/LOAD PERSISTENCE
# ============================================================================

def _autosave_tick(session: "Session") -> None:
    """Background autosave timer callback."""
    if not getattr(session, '_autosave_enabled', False):
        return
    try:
        proj_name = getattr(session, 'current_project', None) or 'autosave'
        save_path = os.path.join(
            os.path.expanduser('~'), 'Documents', 'MDMA',
            f"{proj_name}_autosave.mdma")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Use the save infrastructure if available
        from mdma_rebuild.commands.general_cmds import cmd_save
        cmd_save(session, [save_path])
    except Exception:
        pass  # Autosave should never crash the session
    finally:
        # Reschedule if still enabled
        if getattr(session, '_autosave_enabled', False):
            interval = getattr(session, '_autosave_interval', 5)
            t = threading.Timer(interval * 60, _autosave_tick, args=(session,))
            t.daemon = True
            session._autosave_timer = t
            t.start()


def cmd_autosave(session: "Session", args: List[str]) -> str:
    """Configure auto-save.

    Usage:
      /autosave on              Enable auto-save (5 min default)
      /autosave off             Disable auto-save
      /autosave interval <min>  Set interval in minutes
      /autosave status          Show status
    """
    if not args:
        status = "ON" if session._autosave_enabled else "OFF"
        return f"Auto-save: {status}, interval: {session._autosave_interval} min"

    sub = args[0].lower()
    if sub in ('on', 'enable'):
        session._autosave_enabled = True
        # Cancel existing timer if any
        old_timer = getattr(session, '_autosave_timer', None)
        if old_timer is not None:
            old_timer.cancel()
        # Start new timer
        interval = session._autosave_interval
        t = threading.Timer(interval * 60, _autosave_tick, args=(session,))
        t.daemon = True
        session._autosave_timer = t
        t.start()
        return f"OK: Auto-save enabled (every {interval} min)"
    elif sub in ('off', 'disable'):
        session._autosave_enabled = False
        old_timer = getattr(session, '_autosave_timer', None)
        if old_timer is not None:
            old_timer.cancel()
            session._autosave_timer = None
        return "OK: Auto-save disabled"
    elif sub == 'interval':
        if len(args) < 2:
            return f"Auto-save interval: {session._autosave_interval} min"
        try:
            mins = int(args[1])
            session._autosave_interval = max(1, mins)
            # Restart timer if running
            if session._autosave_enabled:
                old_timer = getattr(session, '_autosave_timer', None)
                if old_timer is not None:
                    old_timer.cancel()
                t = threading.Timer(session._autosave_interval * 60,
                                    _autosave_tick, args=(session,))
                t.daemon = True
                session._autosave_timer = t
                t.start()
            return f"OK: Auto-save interval set to {session._autosave_interval} min"
        except ValueError:
            return f"ERROR: Invalid interval '{args[1]}'"
    elif sub == 'status':
        status = "ON" if session._autosave_enabled else "OFF"
        return f"Auto-save: {status}, interval: {session._autosave_interval} min"

    return "Usage: /autosave on|off|interval|status"


# ============================================================================
# T.7 — PLAYBACK FIXES
# ============================================================================

def cmd_pos(session: "Session", args: List[str]) -> str:
    """Show or set write position.

    Usage:
      /pos                Show current write position
      /pos <seconds>      Set write position in seconds
      /pos <bar>b         Set write position in bars
    """
    track = session.get_current_track()
    if not args:
        pos_samples = track.get('write_pos', 0)
        pos_sec = pos_samples / session.sample_rate
        pos_bars = pos_sec / (4.0 * 60.0 / session.bpm)
        return (f"Write position: {pos_sec:.2f}s (bar {pos_bars:.1f})\n"
                f"  Track: {session.current_track_index + 1}\n"
                f"  Project length: {session.project_length_seconds:.1f}s")

    arg = args[0]
    if arg.endswith('b'):
        try:
            bars = float(arg[:-1])
            seconds = _bar_to_seconds(bars, session.bpm)
        except ValueError:
            return f"ERROR: Invalid bar position '{arg}'"
    else:
        try:
            seconds = float(arg)
        except ValueError:
            return f"ERROR: Invalid position '{arg}'"

    session.set_track_cursor_seconds(seconds)
    pos_bars = seconds / (4.0 * 60.0 / session.bpm)
    return f"OK: Write position set to {seconds:.2f}s (bar {pos_bars:.1f})"


def cmd_seek(session: "Session", args: List[str]) -> str:
    """Seek to a position (alias for /pos with position).

    Usage:
      /seek <seconds>     Jump to position in seconds
      /seek <bar>b        Jump to bar position
    """
    if not args:
        return "Usage: /seek <seconds> or /seek <bar>b"
    return cmd_pos(session, args)


# ============================================================================
# T.8 — AUDIO IMPORT FIX
# ============================================================================

def resample_audio(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample audio from one sample rate to another.

    Uses linear interpolation as a simple fallback. Tries scipy first
    for higher quality.
    """
    if from_sr == to_sr:
        return audio

    ratio = to_sr / from_sr
    if audio.ndim == 2:
        new_len = int(audio.shape[0] * ratio)
        result = np.zeros((new_len, audio.shape[1]), dtype=np.float64)
        for ch in range(audio.shape[1]):
            result[:, ch] = _resample_channel(audio[:, ch], new_len)
        return result
    else:
        new_len = int(len(audio) * ratio)
        return _resample_channel(audio, new_len)


def _resample_channel(channel: np.ndarray, new_len: int) -> np.ndarray:
    """Resample a single channel."""
    try:
        from scipy.signal import resample as scipy_resample
        return scipy_resample(channel.astype(np.float64), new_len)
    except ImportError:
        # Linear interpolation fallback
        old_indices = np.linspace(0, len(channel) - 1, new_len)
        return np.interp(old_indices, np.arange(len(channel)), channel.astype(np.float64))


# ============================================================================
# T.10 — FILE FX CHAIN CLEANUP
# ============================================================================

def cmd_filefx(session: "Session", args: List[str]) -> str:
    """Manage file FX chain (applied during import/export).

    Usage:
      /filefx list             Show file FX chain
      /filefx add <effect>     Add effect to file FX chain
      /filefx clear            Clear file FX chain
      /filefx remove           Remove last effect
    """
    if not args or args[0].lower() == 'list':
        if not session.file_fx_chain:
            return "File FX chain: (empty)"
        lines = ["File FX chain:"]
        for i, (name, params) in enumerate(session.file_fx_chain):
            lines.append(f"  {i + 1}. {name} {params}")
        return '\n'.join(lines)

    sub = args[0].lower()
    if sub == 'add':
        if len(args) < 2:
            return "Usage: /filefx add <effect_name> [params]"
        fx_name = args[1]
        fx_params = {}
        if len(args) > 2:
            for p in args[2:]:
                if '=' in p:
                    k, v = p.split('=', 1)
                    try:
                        fx_params[k] = float(v)
                    except ValueError:
                        fx_params[k] = v
        session.file_fx_chain.append((fx_name, fx_params))
        return f"OK: Added '{fx_name}' to file FX chain"

    if sub == 'clear':
        session.file_fx_chain = []
        return "OK: File FX chain cleared"

    if sub in ('remove', 'rm'):
        if session.file_fx_chain:
            removed = session.file_fx_chain.pop()
            return f"OK: Removed '{removed[0]}' from file FX chain"
        return "File FX chain is already empty"

    return "Usage: /filefx list|add|clear|remove"


# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

def get_phase_t_commands() -> dict:
    """Return all Phase T commands for registration."""
    return {
        # T.1 Undo/Redo
        'undo': cmd_undo,
        'redo': cmd_redo,
        'snapshot': cmd_snapshot,

        # T.2 Song Structure
        'section': cmd_section,
        'pchain': cmd_pchain,

        # T.3 Render/Export
        'export': cmd_export,
        'master_gain': cmd_master_gain,
        'mgain': cmd_master_gain,

        # T.4 Missing Commands
        'crossover': cmd_crossover,
        'xover': cmd_crossover,
        'dup': cmd_dup,
        'duplicate': cmd_dup,
        'metronome': cmd_metronome,
        'metro': cmd_metronome,
        'click': cmd_metronome,
        'swap': cmd_swap,

        # T.5 Output Routing
        'commit': cmd_commit,
        'cm': cmd_commit,

        # T.6 Persistence
        'autosave': cmd_autosave,

        # T.7 Playback
        'pos': cmd_pos,
        'seek': cmd_seek,

        # T.10 File FX
        'filefx': cmd_filefx,
    }
