"""Buffer store helpers for the Session object.

Extracted from ``core/session.py`` during the backend V2 repo-prep. Owns:

- Numbered buffer slots (``session.buffers``) and their append positions.
- The working buffer (``session.working_buffer``) and its lifecycle.
- Buffer priority / source resolution (the ``_get_source_audio`` family).
- Undo / redo stacks for working and track audio.
- Parameter snapshots (save / restore of session state, without audio).

Every function takes ``self`` (a Session) as its first argument and is
re-bound onto the Session class at the bottom of ``core/session.py``.
Behaviour is unchanged; only the physical location of the code moved.

BUILD ID: buffer_store_v1 (extracted from session_v14.2_chunk3)
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .session import Session


# ------------ Buffer management methods ------------

def get_buffer(self: "Session", idx: int) -> Optional[np.ndarray]:
    """Get a buffer by index (1-indexed).

    Parameters
    ----------
    idx : int
        Buffer index (1-indexed for accessibility)

    Returns
    -------
    np.ndarray or None
        The buffer contents, or None if empty/nonexistent
    """
    buf = self.buffers.get(idx)
    if buf is None or len(buf) == 0:
        return None
    return buf


def store_in_buffer(self: "Session", idx: int, audio: Optional[np.ndarray] = None) -> None:
    """Store audio in a buffer.

    Parameters
    ----------
    idx : int
        Buffer index (1-indexed)
    audio : np.ndarray, optional
        Audio to store. If None, uses self.last_buffer
    """
    if audio is None:
        audio = self.last_buffer
    if audio is None:
        return

    # Ensure buffer exists
    if idx not in self.buffers:
        self.buffers[idx] = np.zeros(0, dtype=np.float64)
        self.buffer_append_positions[idx] = 0

    self.buffers[idx] = audio.astype(np.float64).copy()
    self.buffer_append_positions[idx] = 0


def append_to_buffer(self: "Session", idx: int, audio: Optional[np.ndarray] = None) -> int:
    """Append audio to a buffer.

    Parameters
    ----------
    idx : int
        Buffer index (1-indexed)
    audio : np.ndarray, optional
        Audio to append. If None, uses self.last_buffer

    Returns
    -------
    int
        The append position (start of new audio)
    """
    if audio is None:
        audio = self.last_buffer
    if audio is None or len(audio) == 0:
        return 0

    # Ensure buffer exists
    if idx not in self.buffers:
        self.buffers[idx] = np.zeros(0, dtype=np.float64)
        self.buffer_append_positions[idx] = 0

    existing = self.buffers[idx]
    append_pos = len(existing)

    if len(existing) == 0:
        self.buffers[idx] = audio.astype(np.float64).copy()
    else:
        self.buffers[idx] = np.concatenate([existing, audio.astype(np.float64)])

    self.buffer_append_positions[idx] = append_pos
    return append_pos


def clear_buffer(self: "Session", idx: int) -> None:
    """Clear a buffer.

    Parameters
    ----------
    idx : int
        Buffer index (1-indexed)
    """
    self.buffers[idx] = np.zeros(0, dtype=np.float64)
    self.buffer_append_positions[idx] = 0


def ensure_buffer_count(self: "Session", count: int) -> None:
    """Ensure at least 'count' buffers exist.

    Parameters
    ----------
    count : int
        Minimum number of buffers to have (1-indexed, so count=4 means buffers 1-4)
    """
    for i in range(1, count + 1):
        if i not in self.buffers:
            self.buffers[i] = np.zeros(0, dtype=np.float64)
            self.buffer_append_positions[i] = 0


# ------------ Working buffer management ------------

def ensure_working_buffer(self: "Session") -> np.ndarray:
    """Ensure working_buffer exists and is not empty.

    Returns the working buffer, initializing with silence if needed.
    """
    if self.working_buffer is None or len(self.working_buffer) == 0:
        self.working_buffer = np.zeros(self.sample_rate, dtype=np.float64)
        self.working_buffer_source = 'init'
    return self.working_buffer


def append_to_working(self: "Session", audio: np.ndarray) -> int:
    """Append audio to the working buffer.

    If the working buffer only has init silence, replaces it instead
    of prepending useless silence.

    Parameters
    ----------
    audio : np.ndarray
        Audio to append

    Returns
    -------
    int
        The append position (start of new audio)
    """
    # If working buffer is just init silence, replace it
    if not self.has_real_working_audio():
        self.working_buffer = audio.astype(np.float64)
        self.working_buffer_source = 'generated'
        return 0

    # Otherwise truly append
    append_pos = len(self.working_buffer)
    self.working_buffer = np.concatenate([
        self.working_buffer.astype(np.float64),
        audio.astype(np.float64)
    ])
    self.working_buffer_source = 'appended'
    return append_pos


def set_working_buffer(self: "Session", audio: np.ndarray, source: str = 'generated') -> None:
    """Set the working buffer to new audio.

    Parameters
    ----------
    audio : np.ndarray
        Audio data
    source : str
        Source description (e.g., 'generated', 'melody', 'out_block')
    """
    self.working_buffer = audio.astype(np.float64)
    self.working_buffer_source = source
    self.working_buffer_source_id = None


def get_filled_buffers(self: "Session") -> list[int]:
    """Get list of buffer indices that have audio.

    Returns
    -------
    list[int]
        List of buffer indices with non-empty audio
    """
    filled = []
    for idx, buf in self.buffers.items():
        if buf is not None and len(buf) > 0:
            filled.append(idx)
    return sorted(filled)


def get_any_audio(self: "Session") -> tuple[Optional[np.ndarray], str]:
    """Get audio from any available source.

    Priority: working_buffer > last_buffer > first filled buffer

    Returns
    -------
    tuple[np.ndarray, str]
        (audio_data, source_description) or (None, 'none')
    """
    # Try working buffer first (but only if it has real content, not just init silence)
    if (self.working_buffer is not None and len(self.working_buffer) > 0
        and self.working_buffer_source != 'init'):
        return self.working_buffer, 'working'

    # Try last_buffer
    if self.last_buffer is not None and len(self.last_buffer) > 0:
        return self.last_buffer, 'last'

    # Try any filled buffer
    filled = self.get_filled_buffers()
    if filled:
        idx = filled[0]
        return self.buffers[idx], f'buffer_{idx}'

    return None, 'none'


def get_playable_buffer(self: "Session", idx: Optional[int] = None) -> tuple[Optional[np.ndarray], str]:
    """Get a buffer suitable for playback.

    If idx is specified, tries that buffer first.
    Otherwise picks from available audio sources.
    If multiple buffers have audio and none specified, picks randomly.

    Parameters
    ----------
    idx : int, optional
        Specific buffer index to try first

    Returns
    -------
    tuple[np.ndarray, str]
        (audio_data, source_description) or (None, 'none')
    """
    import random

    # If specific buffer requested
    if idx is not None:
        if idx in self.buffers and len(self.buffers[idx]) > 0:
            return self.buffers[idx], f'buffer_{idx}'
        # Fall through to find any audio

    # Collect all sources with audio
    sources = []

    # Working buffer only if it has real content (not init silence)
    if (self.working_buffer is not None and len(self.working_buffer) > 0
        and self.working_buffer_source != 'init'):
        sources.append((self.working_buffer, 'working'))

    if self.last_buffer is not None and len(self.last_buffer) > 0:
        sources.append((self.last_buffer, 'last'))

    for buf_idx in self.get_filled_buffers():
        sources.append((self.buffers[buf_idx], f'buffer_{buf_idx}'))

    if not sources:
        return None, 'none'

    if len(sources) == 1:
        return sources[0]

    # Multiple sources - pick randomly
    return random.choice(sources)


def get_lowest_empty_buffer(self: "Session") -> int:
    """Get the lowest numbered empty buffer.

    Returns
    -------
    int
        Buffer index (creates new one if all are full)
    """
    # Check buffers 1-10 for empty one
    for i in range(1, 11):
        if i not in self.buffers or len(self.buffers[i]) == 0:
            # Ensure it exists
            if i not in self.buffers:
                self.buffers[i] = np.zeros(0, dtype=np.float64)
                self.buffer_append_positions[i] = 0
            return i
    # All 1-10 are full, return 11
    self.buffers[11] = np.zeros(0, dtype=np.float64)
    self.buffer_append_positions[11] = 0
    return 11


def has_real_working_audio(self: "Session") -> bool:
    """Check if working buffer has real audio (not just init silence)."""
    return (self.working_buffer is not None
            and len(self.working_buffer) > 0
            and self.working_buffer_source != 'init')


# ------------ Undo/Redo helpers (Phase T.1) ------------

def push_undo(self: "Session", target: str = 'working', track_idx: int = 0) -> None:
    """Push current state onto undo stack before a destructive operation."""
    if target == 'working':
        if self.working_buffer is not None and len(self.working_buffer) > 0:
            self._undo_stack.append(self.working_buffer.copy())
            if len(self._undo_stack) > self._undo_max_depth:
                self._undo_stack.pop(0)
            self._redo_stack.clear()
    elif target == 'track':
        idx = track_idx
        if idx < len(self.tracks):
            audio = self.tracks[idx].get('audio')
            if audio is not None:
                if idx not in self._track_undo_stacks:
                    self._track_undo_stacks[idx] = []
                    self._track_redo_stacks[idx] = []
                self._track_undo_stacks[idx].append(audio.copy())
                if len(self._track_undo_stacks[idx]) > self._undo_max_depth:
                    self._track_undo_stacks[idx].pop(0)
                self._track_redo_stacks[idx].clear()


def pop_undo(self: "Session", target: str = 'working', track_idx: int = 0) -> bool:
    """Restore previous state from undo stack. Returns True if successful."""
    if target == 'working':
        if not self._undo_stack:
            return False
        self._redo_stack.append(self.working_buffer.copy())
        self.working_buffer = self._undo_stack.pop()
        self.working_buffer_source = 'undo'
        self.last_buffer = self.working_buffer.copy()
        return True
    elif target == 'track':
        idx = track_idx
        stack = self._track_undo_stacks.get(idx, [])
        if not stack:
            return False
        if idx not in self._track_redo_stacks:
            self._track_redo_stacks[idx] = []
        self._track_redo_stacks[idx].append(self.tracks[idx]['audio'].copy())
        self.tracks[idx]['audio'] = stack.pop()
        return True
    return False


def pop_redo(self: "Session", target: str = 'working', track_idx: int = 0) -> bool:
    """Re-apply undone operation. Returns True if successful."""
    if target == 'working':
        if not self._redo_stack:
            return False
        self._undo_stack.append(self.working_buffer.copy())
        self.working_buffer = self._redo_stack.pop()
        self.working_buffer_source = 'redo'
        self.last_buffer = self.working_buffer.copy()
        return True
    elif target == 'track':
        idx = track_idx
        stack = self._track_redo_stacks.get(idx, [])
        if not stack:
            return False
        self._track_undo_stacks[idx].append(self.tracks[idx]['audio'].copy())
        self.tracks[idx]['audio'] = stack.pop()
        return True
    return False


def save_snapshot(self: "Session") -> int:
    """Save lightweight session parameter snapshot (no audio). Returns snapshot index."""
    snap = {
        'bpm': self.bpm,
        'step': self.step,
        'attack': self.attack, 'decay': self.decay,
        'sustain': self.sustain, 'release': self.release,
        'carrier_count': self.carrier_count,
        'mod_count': self.mod_count,
        'voice_count': self.voice_count,
        'voice_algorithm': self.voice_algorithm,
        'dt': self.dt, 'rand': self.rand, 'v_mod': self.v_mod,
        'filter_types': dict(self.filter_types),
        'filter_cutoffs': dict(self.filter_cutoffs),
        'filter_resonances': dict(self.filter_resonances),
        'filter_enabled': dict(self.filter_enabled),
        'effects': list(self.effects),
        'master_gain': self.master_gain,
        'master_fx_chain': list(self.master_fx_chain),
        'track_gains': [t.get('gain', 1.0) for t in self.tracks],
        'track_pans': [t.get('pan', 0.0) for t in self.tracks],
        'track_mutes': [t.get('mute', False) for t in self.tracks],
        'track_solos': [t.get('solo', False) for t in self.tracks],
    }
    self._snapshots.append(snap)
    return len(self._snapshots) - 1


def restore_snapshot(self: "Session", index: int = -1) -> bool:
    """Restore session parameters from snapshot. Returns True if successful."""
    if not self._snapshots:
        return False
    if index < 0:
        index = len(self._snapshots) + index
    if index < 0 or index >= len(self._snapshots):
        return False
    snap = self._snapshots[index]
    for key in ('bpm', 'step', 'attack', 'decay', 'sustain', 'release',
                 'carrier_count', 'mod_count', 'voice_count', 'voice_algorithm',
                 'dt', 'rand', 'v_mod', 'master_gain'):
        if key in snap:
            setattr(self, key, snap[key])
    for key in ('filter_types', 'filter_cutoffs', 'filter_resonances', 'filter_enabled'):
        if key in snap:
            setattr(self, key, dict(snap[key]))
    if 'effects' in snap:
        self.effects = list(snap['effects'])
    if 'master_fx_chain' in snap:
        self.master_fx_chain = list(snap['master_fx_chain'])
    for i, t in enumerate(self.tracks):
        if i < len(snap.get('track_gains', [])):
            t['gain'] = snap['track_gains'][i]
        if i < len(snap.get('track_pans', [])):
            t['pan'] = snap['track_pans'][i]
        if i < len(snap.get('track_mutes', [])):
            t['mute'] = snap['track_mutes'][i]
        if i < len(snap.get('track_solos', [])):
            t['solo'] = snap['track_solos'][i]
    return True


def bind_to(session_cls) -> None:
    """Attach every buffer-store function above to ``session_cls``."""
    session_cls.get_buffer = get_buffer
    session_cls.store_in_buffer = store_in_buffer
    session_cls.append_to_buffer = append_to_buffer
    session_cls.clear_buffer = clear_buffer
    session_cls.ensure_buffer_count = ensure_buffer_count
    session_cls.ensure_working_buffer = ensure_working_buffer
    session_cls.append_to_working = append_to_working
    session_cls.set_working_buffer = set_working_buffer
    session_cls.get_filled_buffers = get_filled_buffers
    session_cls.get_any_audio = get_any_audio
    session_cls.get_playable_buffer = get_playable_buffer
    session_cls.get_lowest_empty_buffer = get_lowest_empty_buffer
    session_cls.has_real_working_audio = has_real_working_audio
    session_cls.push_undo = push_undo
    session_cls.pop_undo = pop_undo
    session_cls.pop_redo = pop_redo
    session_cls.save_snapshot = save_snapshot
    session_cls.restore_snapshot = restore_snapshot
