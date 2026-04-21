"""Session management for the MDMA rebuild.

The Session class captures all state needed to build sound objects,
apply modulation and envelope settings, and generate simple audio
buffers.  It is intentionally minimal: the goal is to provide a
sandbox where monolith and playback commands can be exercised without
depending on the original MDMA codebase.  This session uses the
MonolithEngine to generate tones and exposes properties for
operators, filters and envelopes.  Clips and tracks are simplified
to a single buffer for demonstration purposes.

PARAMETER SCALING:
-----------------
Abstract parameters use unified 1-100 scaling:
- resonance: 0-100 (filter intensity)
- rand: 0-100 (voice amplitude variation)
- v_mod: 0-100 (voice modulation scaling)
- effect amounts: 0-100

Real-world units preserved:
- dt: Hz (detune)
- cutoff: Hz
- envelope times: seconds
- frequency: Hz

PLAYBACK:
---------
In-house audio playback via sounddevice/simpleaudio/pyaudio.
No calls to external media players (os.startfile, xdg-open, etc.)
Use session.play() for direct buffer playback.

BUILD ID: session_v14.2_chunk3
"""

from __future__ import annotations

import os
import wave
import struct
from typing import Any, Optional

import numpy as np  # type: ignore

from ..dsp.monolith import MonolithEngine
from ..dsp.envelopes import ADSREnvelope
from ..dsp.scaling import parse_param, clamp_param, is_wacky, validate_param


class Session:
    """Maintain state for the MDMA REPL.

    A session holds global parameters such as BPM, step size, and
    sample rate, along with operator parameters, filter settings and
    envelope values.  It also keeps track of the last rendered
    buffer so that playback commands can write it to disk.  Tracks
    and projects are not persisted to disk in this simplified
    implementation.
    """

    def __init__(self, sample_rate: int = 48_000) -> None:
        self.sample_rate: int = sample_rate
        # Tempo and step: 1/32 note is default (0.125 beats)
        self.bpm: float = 128.0
        self.step: float = 0.125
        # Monolith engine manages operator definitions and renders
        self.engine = MonolithEngine(sample_rate=self.sample_rate)
        # Currently selected operator index
        self.current_operator: int = 0
        # Operator counts and voice settings
        self.carrier_count: int = 1
        self.mod_count: int = 0
        self.voice_count: int = 1
        self.voice_algorithm: str = "unison"  # algo 1 — random phase spread
        
        # --- Filter bank settings ---
        # Filter type 0=lowpass (default), supports multiple filter slots.
        # Each slot has its own type, cutoff, and resonance.
        self.filter_count: int = 1
        self.selected_filter: int = 0  # Currently selected filter slot
        self.filter_types: dict[int, int] = {0: 0}  # slot -> filter type index
        self.filter_cutoffs: dict[int, float] = {0: 4500.0}  # slot -> cutoff Hz (default 4.5kHz, not muffled)
        self.filter_resonances: dict[int, float] = {0: 50.0}  # slot -> resonance (0-100 scale)
        self.filter_enabled: dict[int, bool] = {0: True}  # slot -> enabled
        
        # Filter type names for lookup
        self.filter_type_names: dict[int, str] = {
            # Basic filters (0-6)
            0: 'lowpass', 1: 'highpass', 2: 'bandpass', 3: 'notch',
            4: 'peak', 5: 'ringmod', 6: 'allpass',
            # Comb filters (7-9)
            7: 'comb_ff', 8: 'comb_fb', 9: 'comb_both',
            # Analog modeled (10-11)
            10: 'analog', 11: 'acid',
            # Formant filters (12-16)
            12: 'formant_a', 13: 'formant_e', 14: 'formant_i',
            15: 'formant_o', 16: 'formant_u',
            # Shelf filters (17-18)
            17: 'lowshelf', 18: 'highshelf',
            # Moog/SVF (19-22)
            19: 'moog', 20: 'svf_lp', 21: 'svf_hp', 22: 'svf_bp',
            # Destructive (23-24)
            23: 'bitcrush', 24: 'downsample',
            # Utility (25-26)
            25: 'dc_block', 26: 'tilt',
            # Character (27-29)
            27: 'resonant', 28: 'vocal', 29: 'telephone',
        }
        self.filter_type_aliases: dict[str, int] = {
            # Basic (0-6)
            'lp': 0, 'lowpass': 0, 'low': 0, 'lpf': 0,
            'hp': 1, 'highpass': 1, 'high': 1, 'hpf': 1,
            'bp': 2, 'bandpass': 2, 'band': 2, 'bpf': 2,
            'notch': 3, 'bs': 3, 'bandstop': 3, 'reject': 3,
            'peak': 4, 'bell': 4, 'eq': 4,
            'ring': 5, 'ringmod': 5, 'rm': 5,
            'ap': 6, 'allpass': 6, 'all': 6, 'phase': 6,
            # Comb (7-9)
            'comb': 7, 'comb_ff': 7, 'combff': 7, 'cff': 7,
            'comb_fb': 8, 'combfb': 8, 'cfb': 8,
            'comb_both': 9, 'combboth': 9, 'cb': 9,
            # Analog (10-11)
            'analog': 10, 'ana': 10, '4pole': 10,
            'acid': 11, '303': 11, 'tb303': 11, 'reso': 11,
            # Formant (12-16)
            'formant_a': 12, 'fa': 12, 'ah': 12,
            'formant_e': 13, 'fe': 13, 'eh': 13,
            'formant_i': 14, 'fi': 14, 'ee': 14,
            'formant_o': 15, 'fo': 15, 'oh': 15,
            'formant_u': 16, 'fu': 16, 'oo': 16,
            # Shelf (17-18)
            'lowshelf': 17, 'lshelf': 17, 'ls': 17, 'bass': 17,
            'highshelf': 18, 'hshelf': 18, 'hs': 18, 'treble': 18,
            # Moog/SVF (19-22)
            'moog': 19, 'ladder': 19, 'mg': 19,
            'svf_lp': 20, 'svflp': 20, 'svf': 20,
            'svf_hp': 21, 'svfhp': 21,
            'svf_bp': 22, 'svfbp': 22,
            # Destructive (23-24)
            'bitcrush': 23, 'crush': 23, 'bit': 23, 'bc': 23,
            'downsample': 24, 'ds': 24, 'decimate': 24, 'srr': 24,
            # Utility (25-26)
            'dc_block': 25, 'dc': 25, 'dcb': 25,
            'tilt': 26, 'tilteq': 26,
            # Character (27-29)
            'resonant': 27, 'res': 27, 'rez': 27,
            'vocal': 28, 'vox': 28, 'voice': 28,
            'telephone': 29, 'phone': 29, 'tel': 29, 'lofi': 29,
        }
        
        # --- Global amplitude envelope (default) ---
        self.attack: float = 0.01
        self.decay: float = 0.1
        self.sustain: float = 0.8
        self.release: float = 0.1
        
        # --- Per-operator envelopes (keyed by operator index) ---
        # When synth_level=2 (operator), envelope commands edit these.
        # If an operator has no custom envelope, the global one is used.
        self.operator_envelopes: dict[int, dict[str, float]] = {}
        
        # --- Filter envelope bank ---
        # Each filter slot can have its own envelope for filter modulation.
        # selected_filter_envelope determines which slot's envelope is edited.
        self.selected_filter_envelope: int = 0
        self.filter_envelopes: dict[int, dict[str, float]] = {
            0: {'attack': 0.01, 'decay': 0.1, 'sustain': 1.0, 'release': 0.1}
        }

        # Voice algorithm parameters (use 1-100 scaling for abstract params).
        # These control unison behaviour when multiple voices are active.
        # ``dt`` detunes each subsequent voice (Hz - real units).
        # ``rand`` applies random amplitude variation per voice (0-100 scale).
        # ``v_mod`` scales modulation indices (0-100 scale).
        self.dt: float = 0.0      # Detune in Hz (real units, 0 = no detune)
        self.rand: float = 0.0    # Amplitude variation (0-100, 0 = no variation)
        self.v_mod: float = 0.0   # Mod scaling (0-100, 0 = no extra scaling)
        # Last rendered audio buffer (numpy array) and location for temp files
        self.last_buffer: Optional[np.ndarray] = None
        self.autoplay: bool = False
        # Directory for temporary preview files; ensure it exists
        self.temp_dir: str = os.path.join(os.getcwd(), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        # Effects chain: a list of effect names applied after synthesis
        self.effects: list[str] = []

        # Selected effect index and parameters for each effect.  When editing
        # effect parameters via the /fx command the current selection is
        # stored here.  ``selected_effect`` is -1 when no effect is
        # selected.  ``effect_params`` parallels ``effects`` and holds
        # dictionaries of effect parameters (e.g., mix, amount).  These
        # parameters are not currently used in audio processing but are
        # stored for future extension.
        self.selected_effect: int = -1
        self.effect_params: list[dict[str, float]] = []

        # --- FX Chain Positions ---
        # Separate FX chains for different processing stages
        # BFX = Buffer effects (applied to current buffer)
        # TFX = Track effects (applied during track render)
        # MFX = Master effects (applied to final mix)
        # FFX = File effects (applied when processing files)
        self.buffer_fx_chain: list[tuple[str, dict]] = []  # (effect_name, params)
        self.track_fx_chain: list[tuple[str, dict]] = []
        self.master_fx_chain: list[tuple[str, dict]] = []
        self.file_fx_chain: list[tuple[str, dict]] = []

        # --- Deck support ---
        # A deck represents an external playback target similar to a buffer but
        # may be routed to a custom device. Each deck holds a buffer and its
        # own FX chain. Decks are indexed by integers and can be expanded as
        # needed. This design allows future extension to send audio to
        # external hardware while sharing the same processing pipeline as
        # buffers. Use session.decks[idx]['buffer'] to set audio and
        # session.decks[idx]['fx_chain'] to adjust effects.
        self.decks: dict[int, dict] = {}
        
        # --- Buffer system ---
        # Multi-buffer workspace for building complex sounds.
        # Buffers are 1-indexed for accessibility (screen reader friendly).
        # Buffer 1 is the default working buffer.
        self.buffers: dict[int, np.ndarray] = {1: np.zeros(0, dtype=np.float64)}
        self.current_buffer_index: int = 1
        self.buffer_append_positions: dict[int, int] = {1: 0}
        
        # Working buffer state - ALWAYS has audio (never None)
        # Initialize with 1 second of silence so there's always something
        self.working_buffer: np.ndarray = np.zeros(self.sample_rate, dtype=np.float64)
        self.working_buffer_source: str = 'init'  # 'init', 'buffer', 'deck', 'file', 'generated'
        self.working_buffer_source_id: Optional[int] = None
        
        # Current FX chain target (default to buffer)
        self.fx_chain_target: str = 'buffer'  # 'buffer', 'track', 'master', 'file'

        # --- High-Quality (HQ) Mode Settings ---
        # Controls audio quality processing for output
        self.hq_mode: bool = True  # Enable HQ render chain by default
        self.hq_dc_remove: bool = True  # DC offset removal
        self.hq_subsonic_filter: bool = True  # 20Hz highpass
        self.hq_subsonic_freq: float = 20.0  # Subsonic cutoff frequency
        self.hq_highend_smooth: bool = True  # High-end rolloff
        self.hq_highend_freq: float = 16000.0  # High-end shelf frequency
        self.hq_highend_reduction: float = -1.5  # High-end reduction in dB
        self.hq_saturation: bool = True  # Soft saturation for warmth
        self.hq_saturation_drive: float = 0.1  # Saturation amount (0-1)
        self.hq_limiting: bool = True  # Soft limiting
        self.hq_limit_threshold: float = 0.95  # Limiter threshold
        
        # Output format settings (v49: defaults changed to FLAC 24-bit)
        self.output_format: str = 'flac'  # 'wav' or 'flac' - default FLAC for quality
        self.output_bit_depth: int = 24  # 16 or 24 - default 24-bit for headroom
        
        # Band-limited oscillators (smoother waveforms)
        self.hq_oscillators: bool = True  # Use band-limited waveforms

        # --- Track timeline (simplified) ---
        # MDMA v40: tracks are the primary audio containers.
        # Each track holds ONE continuous STEREO audio array for the project
        # duration. You write/overwrite into the track at a cursor position.
        #
        # This intentionally replaces the old multi-buffer + clip timeline for
        # core workflows, because buffers/working routing caused confusion and
        # test friction.
        #
        # Track schema:
        #   {
        #     'name': str,
        #     'audio': np.ndarray (float64, shape=(project_length_samples, 2)),
        #     'fx_chain': list[(effect_name, params_dict)],
        #     'write_pos': int (cursor in samples),
        #     'gain': float (linear, default 1.0),
        #     'pan': float (-1.0=left, 0.0=center, 1.0=right),
        #     'mute': bool,
        #     'solo': bool,
        #   }
        #
        # Project length defaults to 30 seconds. Use /TLEN to set.
        self.project_length_seconds: float = 30.0
        self.project_length_samples: int = int(self.project_length_seconds * self.sample_rate)

        self.tracks: list[dict[str, Any]] = []
        self.current_track_index: int = 0

        # Create a default track
        self._init_default_tracks()

        # Note: legacy files/clips are kept for compatibility with existing
        # commands, but the preferred workflow is track-first.
        self.files: dict[str, np.ndarray] = {}
        self.clips: dict[str, np.ndarray] = {}

        # --- Hierarchical project structure ---
        # The session maintains a notion of the current level in a
        # hierarchy: global -> project -> sketch -> file -> clip.  Each
        # level can be used to create nested entities with the /new
        # command.  When the level changes, subordinate levels are
        # reset.  The counters and names below track the number of
        # created items and the currently selected entity at each level.
        self.current_level: str = 'global'
        self.synth_level: int = 1  # 1=global parameters, 2=operator, 3=voice
        self.project_count: int = 0
        self.sketch_count: int = 0
        self.file_count: int = 0
        self.clip_count: int = 0
        self.current_project: str = ''
        self.current_sketch: str = ''
        self.current_file: str = ''
        self.current_clip: str = ''

        # --- User-defined functions and chains ---
        # These are populated by bmdma.py at startup (from persisted data)
        # and updated at runtime when the user defines new functions/chains.
        self.user_functions: dict[str, list[str]] = {}
        self.function_commands: list[str] = []
        self.defining_function: Optional[str] = None  # name of function being defined, or None
        self.chains: dict[str, list[str]] = {}

        # --- Phase T: Undo/Redo System (T.1) ---
        self._undo_stack: list[np.ndarray] = []  # Working buffer undo stack
        self._redo_stack: list[np.ndarray] = []  # Working buffer redo stack
        self._undo_max_depth: int = 10  # Max undo levels
        self._track_undo_stacks: dict[int, list[np.ndarray]] = {}  # Per-track undo
        self._track_redo_stacks: dict[int, list[np.ndarray]] = {}
        self._snapshots: list[dict] = []  # Session parameter snapshots

        # --- Phase T: Song Structure (T.2) ---
        self.sections: list[dict] = []  # [{name, start_bar, end_bar}]

        # --- Phase T: Master Gain (T.3) ---
        self.master_gain: float = 0.0  # dB, applied before limiting in render

        # --- Phase T: Auto-save (T.6) ---
        self._autosave_enabled: bool = False
        self._autosave_interval: int = 5  # minutes


    # ------------ Track timeline helpers (simplified) ------------

    def _init_default_tracks(self, track_count: int = 1) -> None:
        """Initialize the track list with empty continuous stereo tracks."""
        self.tracks = []
        for i in range(track_count):
            self.tracks.append({
                'name': f'track_{i+1}',
                'audio': np.zeros((self.project_length_samples, 2), dtype=np.float64),
                'fx_chain': [],
                'write_pos': 0,
                'gain': 1.0,
                'pan': 0.0,
                'mute': False,
                'solo': False,
            })
        self.current_track_index = 0

    def set_project_length_seconds(self, seconds: float, reset_audio: bool = True) -> None:
        """Set project duration for all tracks."""
        seconds = float(max(0.1, seconds))
        self.project_length_seconds = seconds
        self.project_length_samples = int(round(seconds * self.sample_rate))
        if reset_audio:
            existing_n = len(self.tracks) if self.tracks else 1
            self._init_default_tracks(existing_n)

    def get_current_track(self) -> dict:
        if not self.tracks:
            self._init_default_tracks(1)
        self.current_track_index = int(np.clip(self.current_track_index, 0, len(self.tracks)-1))
        return self.tracks[self.current_track_index]

    def ensure_track_index(self, idx_1based: int) -> None:
        """Ensure tracks list has at least idx_1based tracks."""
        idx_1based = max(1, int(idx_1based))
        while len(self.tracks) < idx_1based:
            i = len(self.tracks)
            self.tracks.append({
                'name': f'track_{i+1}',
                'audio': np.zeros((self.project_length_samples, 2), dtype=np.float64),
                'fx_chain': [],
                'write_pos': 0,
                'gain': 1.0,
                'pan': 0.0,
                'mute': False,
                'solo': False,
            })

    def set_track_cursor_seconds(self, seconds: float, track_idx: int | None = None) -> None:
        """Set write cursor position for a track."""
        if track_idx is None:
            track = self.get_current_track()
        else:
            self.ensure_track_index(track_idx)
            track = self.tracks[track_idx-1]
        pos = int(round(float(seconds) * self.sample_rate))
        track['write_pos'] = int(np.clip(pos, 0, max(0, self.project_length_samples-1)))

    def write_to_track(self, audio: np.ndarray, mode: str = 'overwrite',
                       track_idx: int | None = None, start_sample: int | None = None) -> tuple[int, int]:
        """Write audio into a continuous stereo track at the cursor.

        mode: 'overwrite' (default) replaces samples; 'add' sums into the track.
        Input can be mono (1D) or stereo (N,2). Mono is duplicated to both channels.
        Returns (start_sample, end_sample).
        """
        if audio is None:
            return (0, 0)
        audio = audio.astype(np.float64, copy=False)

        # Ensure stereo (N, 2)
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        elif audio.ndim == 2:
            if audio.shape[1] == 1:
                audio = np.column_stack([audio[:, 0], audio[:, 0]])
            elif audio.shape[1] > 2:
                audio = audio[:, :2]
            # shape[0] < shape[1] heuristic: channels-first → transpose
            if audio.ndim == 2 and audio.shape[0] in (1, 2) and audio.shape[1] > 2:
                audio = audio.T
        else:
            # 3D+ just take first 2D slice
            audio = audio.reshape(-1, audio.shape[-1])[:, :2]

        if track_idx is None:
            track = self.get_current_track()
        else:
            self.ensure_track_index(track_idx)
            track = self.tracks[track_idx-1]

        # Ensure track audio is stereo (migrate from old mono tracks)
        t_audio = track.get('audio')
        if t_audio is not None and t_audio.ndim == 1:
            track['audio'] = np.column_stack([t_audio, t_audio])

        if start_sample is None:
            start_sample = int(track.get('write_pos', 0))
        start_sample = int(np.clip(start_sample, 0, self.project_length_samples))

        end_sample = min(self.project_length_samples, start_sample + len(audio))
        if end_sample <= start_sample:
            return (start_sample, start_sample)

        seg = audio[: end_sample - start_sample]
        if mode == 'add':
            track['audio'][start_sample:end_sample] += seg
        else:
            track['audio'][start_sample:end_sample] = seg

        track['write_pos'] = end_sample
        return (start_sample, end_sample)

    def get_track_mix(self) -> np.ndarray:
        """Return a mixed buffer of all tracks (track FX + master FX)."""
        return self.mix_tracks()
# ------------ Audio processing helpers ------------
    def _apply_micro_fades(self, buf: np.ndarray, fade_ms: float = 5.0) -> np.ndarray:
        """Apply micro fade-in and fade-out to prevent click artifacts.

        Parameters
        ----------
        buf : np.ndarray
            The audio buffer to process (mono or stereo).
        fade_ms : float
            Fade duration in milliseconds (default 5ms).

        Returns
        -------
        np.ndarray
            Buffer with fades applied.
        """
        if buf.shape[0] == 0:
            return buf
        n = buf.shape[0]
        fade_samples = int(fade_ms * self.sample_rate / 1000.0)
        fade_samples = min(fade_samples, n // 2)
        if fade_samples < 2:
            return buf
        out = buf.copy()
        # Fade curves
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_samples)))
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_samples)))
        if out.ndim == 2:
            out[:fade_samples] *= fade_in[:, np.newaxis]
            out[-fade_samples:] *= fade_out[:, np.newaxis]
        else:
            out[:fade_samples] *= fade_in
            out[-fade_samples:] *= fade_out
        return out

    def _apply_smoothing(self, buf: np.ndarray) -> np.ndarray:
        """Apply simple smoothing/anti-aliasing to reduce harsh artifacts.

        Uses a gentle 3-sample moving average that preserves transients
        while reducing aliasing and high-frequency hash.
        Handles both mono (N,) and stereo (N,2) buffers.
        """
        if buf.shape[0] < 3:
            return buf
        out = buf.copy()
        if out.ndim == 2:
            for ch in range(out.shape[1]):
                out[1:-1, ch] = 0.25 * buf[:-2, ch] + 0.5 * buf[1:-1, ch] + 0.25 * buf[2:, ch]
        else:
            out[1:-1] = 0.25 * buf[:-2] + 0.5 * buf[1:-1] + 0.25 * buf[2:]
        return out

    def _normalize_output(self, buf: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """Normalize buffer to a consistent output level.

        Parameters
        ----------
        buf : np.ndarray
            The audio buffer to normalize.
        target_db : float
            Target peak level in dB (default -3dB for headroom).

        Returns
        -------
        np.ndarray
            Normalized buffer.
        """
        if len(buf) == 0:
            return buf
        max_val = np.max(np.abs(buf))
        if max_val < 1e-10:  # Essentially silent
            return buf
        target_linear = 10.0 ** (target_db / 20.0)
        return buf * (target_linear / max_val)

    # ------------ Unified FX processing helpers ------------
    def apply_fx_chain(self, buf: np.ndarray, fx_chain: list[tuple[str, dict]]) -> np.ndarray:
        """
        Apply an ordered FX chain to a buffer.  Each chain item is a
        tuple of (effect_name, params) where params is a dictionary of
        effect parameters.  If the chain is empty, the buffer is
        returned unchanged.  Unknown effect names are skipped.

        This helper centralizes effect application so that buffers,
        tracks, master mixes and decks all share a common pipeline.  It
        uses apply_effects_with_params from dsp.effects, passing along
        any provided parameters.  Aliases should be resolved before
        calling this method.

        Parameters
        ----------
        buf : np.ndarray
            Input audio buffer to process.
        fx_chain : list of (str, dict)
            List of effect names and their parameter dicts.

        Returns
        -------
        np.ndarray
            Processed audio buffer.
        """
        if buf is None or len(buf) == 0 or not fx_chain:
            return buf

        # Unpack names and params
        names = []
        params = []
        for fx_name, fx_params in fx_chain:
            names.append(fx_name)
            params.append(fx_params or {})

        try:
            from ..dsp.effects import apply_effects_with_params
            # No filter parameters applied here; filtering should be
            # performed separately via filter slots if needed.
            processed = apply_effects_with_params(buf, names, params)
            return processed
        except Exception:
            # If DSP module unavailable or effect fails, fall back to input
            return buf

    def render_buffer(self, buf: np.ndarray) -> np.ndarray:
        """Process a buffer through the buffer FX chain.

        This convenience method applies the session's buffer FX chain
        using apply_fx_chain().  It returns the processed buffer and
        updates last_buffer.
        """
        if buf is None:
            return np.zeros(0, dtype=np.float64)
        processed = self.apply_fx_chain(buf, self.buffer_fx_chain)
        self.last_buffer = processed.astype(np.float64)
        return self.last_buffer

    def render_deck(self, deck_index: int) -> Optional[np.ndarray]:
        """Render a deck through its own FX chain.

        Decks behave similarly to buffers but are intended to route
        audio to external devices.  If the deck at the given index
        contains a buffer, this method applies its FX chain and
        returns the processed buffer.  If the deck does not exist or
        has no buffer, None is returned.  Decks can be populated
        manually by external code.

        Parameters
        ----------
        deck_index : int
            Index of the deck to render.

        Returns
        -------
        np.ndarray | None
            Processed deck buffer or None if not available.
        """
        deck = self.decks.get(deck_index)
        if not deck:
            return None
        buf = deck.get('buffer')
        fx_chain = deck.get('fx_chain', [])
        if buf is None or len(buf) == 0:
            return None
        processed = self.apply_fx_chain(buf, fx_chain)
        # Optionally assign back to deck
        deck['buffer'] = processed
        return processed

    # Audio-rate modulation helper
    def _apply_audiorate_config(self, duration_sec: float) -> None:
        """Apply audio-rate modulation configuration to the engine.
        
        This reads from session.audiorate_config (set by /audiorate commands)
        and generates the appropriate modulation signals for the current render.
        """
        if not hasattr(self, 'audiorate_config') or not self.audiorate_config:
            return
        
        try:
            from ..commands.audiorate_cmds import generate_mod_signal, generate_interval_mod_from_pattern
        except ImportError:
            return
        
        n_samples = int(duration_sec * self.sample_rate)
        
        for key, config in self.audiorate_config.items():
            if key.startswith('interval_'):
                # Interval modulation for specific operator
                op_idx = int(key.split('_')[1])
                mod_signal = generate_mod_signal(
                    config['type'],
                    duration_sec,
                    self.sample_rate,
                    config['rate'],
                    1.0  # Normalized signal
                )
                self.engine.set_interval_mod(op_idx, mod_signal, config['depth'])
                
            elif key.startswith('pattern_'):
                # Pattern-based interval modulation
                op_idx = int(key.split('_')[1])
                mod_signal = generate_interval_mod_from_pattern(
                    config['pattern'],
                    duration_sec,
                    self.sample_rate,
                    config.get('bpm', self.bpm)
                )
                self.engine.set_interval_mod(op_idx, mod_signal, 12.0)  # Direct semitone values
                
            elif key == 'filter':
                # Filter modulation
                mod_signal = generate_mod_signal(
                    config['type'],
                    duration_sec,
                    self.sample_rate,
                    config['rate'],
                    1.0
                )
                self.engine.set_filter_mod(mod_signal, config['depth'])

    # Tone generation
    def generate_tone(self, freq: float, beats: float, amp: float) -> np.ndarray:
        """Generate a tone using the current operator and synthesizer parameters.

        The beats argument is interpreted relative to the current BPM.  This
        method updates the current operator's frequency and amplitude, then
        renders all operators with modulation algorithms and the current
        voice count.  ADSR envelopes are applied intelligently based on
        synth level - per-operator if available, otherwise global.
        Filter bank processing applies all enabled filters with their
        respective envelopes.
        """
        beats = float(beats)
        if beats <= 0:
            beats = 1.0
        duration_sec = beats * 60.0 / self.bpm
        
        # Sync HQ oscillators setting with engine
        self.engine.hq_oscillators = self.hq_oscillators
        
        # Update current operator parameters
        self.set_frequency(freq)
        self.set_amplitude(amp)
        
        # Get filter settings for synthesis-time filtering
        # Use the first enabled filter for synthesis, or None
        synth_filter_type = None
        synth_cutoff = 1000.0
        synth_resonance = 50.0  # 0-100 scale default
        for slot in range(self.filter_count):
            if self.filter_enabled.get(slot, False):
                synth_filter_type = self.filter_types.get(slot, 0)
                synth_cutoff = self.filter_cutoffs.get(slot, 1000.0)
                synth_resonance = self.filter_resonances.get(slot, 50.0)  # 0-100 scale
                break
        
        # Apply audio-rate modulation if configured
        self._apply_audiorate_config(duration_sec)
        
        # Render through the monolith engine using voice_count
        # Note: engine.render() expects 0-100 scaling for resonance, rand, mod
        # Gather optional voice params from session (set by /stereo, /vphase)
        _stereo = getattr(self, 'stereo_spread', None)
        _vphase = getattr(self, 'voice_phase_offset', None)
        # Convert degrees to radians for engine
        _phase_rad = (_vphase * 3.14159265 / 180.0) if _vphase else None
        buf = self.engine.render(
            duration_sec,
            voice_count=self.voice_count,
            carrier_count=self.carrier_count,
            mod_count=self.mod_count,
            filter_type=synth_filter_type,
            cutoff=synth_cutoff,
            resonance=synth_resonance,  # Already 0-100
            dt=self.dt,                  # Hz (real units)
            rand=self.rand,              # 0-100
            mod=self.v_mod,              # 0-100
            stereo_spread=_stereo,       # 0-100 or None
            phase_spread=_phase_rad,     # radians or None
            voice_algorithm=self.voice_algorithm,
        )
        
        # Engine may return stereo (N,2) if stereo_spread is active.
        # All downstream processing must handle both shapes.
        is_stereo = buf.ndim == 2 and buf.shape[1] == 2
        
        # Apply amplitude envelope - use per-operator if at operator level
        # and operator has custom envelope, otherwise use global
        env_params = self.get_envelope_for_operator(self.current_operator)
        env = ADSREnvelope(
            env_params['attack'], env_params['decay'],
            env_params['sustain'], env_params['release'],
            self.sample_rate
        )
        if is_stereo:
            n = buf.shape[0]
            env_curve = env.apply(np.ones(n, dtype=np.float64))
            buf = buf * env_curve[:, np.newaxis]
        else:
            buf = env.apply(buf)
        
        # Apply filter bank processing - process all enabled filters
        # with their respective envelopes
        try:
            from ..dsp.effects import _apply_filter
            from ..dsp.scaling import scale_resonance
            for slot in range(self.filter_count):
                if not self.filter_enabled.get(slot, False):
                    continue
                    
                f_type = self.filter_types.get(slot, 0)
                f_cutoff = self.filter_cutoffs.get(slot, 1000.0)
                f_res_scaled = self.filter_resonances.get(slot, 50.0)  # 0-100 scale
                # Scale resonance from 0-100 to Q value for _apply_filter
                f_res_q = scale_resonance(f_res_scaled)
                f_env = self.filter_envelopes.get(slot, {
                    'attack': 0.01, 'decay': 0.1, 'sustain': 1.0, 'release': 0.1
                })
                
                # Check if this filter has a non-trivial envelope
                has_env = any([
                    f_env['attack'] > 0.01,
                    f_env['decay'] > 0.01,
                    f_env['release'] > 0.01,
                    f_env['sustain'] != 1.0
                ])
                
                if is_stereo:
                    # Process each channel independently
                    for ch in range(2):
                        ch_buf = buf[:, ch].copy()
                        if has_env:
                            env_adsr = ADSREnvelope(
                                f_env['attack'], f_env['decay'],
                                f_env['sustain'], f_env['release'],
                                self.sample_rate
                            )
                            env_curve = env_adsr.apply(np.ones_like(ch_buf))
                            filtered = _apply_filter(ch_buf, f_type, f_cutoff, f_res_q)
                            buf[:, ch] = (env_curve * filtered) + ((1.0 - env_curve) * ch_buf)
                        else:
                            buf[:, ch] = _apply_filter(ch_buf, f_type, f_cutoff, f_res_q)
                else:
                    if has_env:
                        # Apply filter with envelope modulation
                        env_adsr = ADSREnvelope(
                            f_env['attack'], f_env['decay'],
                            f_env['sustain'], f_env['release'],
                            self.sample_rate
                        )
                        env_curve = env_adsr.apply(np.ones_like(buf))
                        filtered = _apply_filter(buf.astype(np.float64), f_type, f_cutoff, f_res_q)
                        # Crossfade: env_curve=1 means fully filtered, 0 means dry
                        buf = (env_curve * filtered) + ((1.0 - env_curve) * buf)
                    else:
                        # Apply filter directly without envelope
                        buf = _apply_filter(buf.astype(np.float64), f_type, f_cutoff, f_res_q)
        except Exception:
            pass
        
        # Apply effects chain
        try:
            from ..dsp.effects import apply_effects
            if is_stereo:
                # Process each channel
                for ch in range(2):
                    buf[:, ch] = apply_effects(
                        buf[:, ch].astype(np.float64),
                        self.effects,
                        filter_type=None,
                        cutoff=None,
                        resonance=None,
                    ).astype(np.float64)
            else:
                buf = apply_effects(
                    buf.astype(np.float64),
                    self.effects,
                    filter_type=None,
                    cutoff=None,
                    resonance=None,
                ).astype(np.float64)
        except Exception:
            pass
            
        # Apply smoothing to reduce aliasing artifacts
        buf = self._apply_smoothing(buf)
        # Apply micro fades to prevent click artifacts at start/end
        buf = self._apply_micro_fades(buf)
        self.last_buffer = buf
        # Persist the generated tone into the current file or clip, if any.
        if self.current_file:
            self.files[self.current_file] = buf.copy()
        if self.current_clip:
            self.clips[self.current_clip] = buf.copy()
        if self.autoplay:
            self._play_buffer(buf)
        return buf

    # ------------ Pattern and clip management ------------
    def apply_pattern(self, tokens: list[str], algorithm: str = 'general') -> str:
        """Apply a pattern to the last rendered buffer or current clip.

        Pattern tokens can include integers (pitch offsets in semitones),
        'R' for rest (mute), and 'D' or 'D<n>' to extend the duration of
        the previous note.  Currently this method does not modify the
        audio buffer; it simply stores the pattern and algorithm for
        future use.  A more sophisticated implementation could
        re‑render the audio according to the pattern.

        Parameters
        ----------
        tokens : list[str]
            Pattern tokens.
        algorithm : str, optional
            The pattern algorithm name ("general" or "chaotic").  Ignored
            in this simplified implementation.

        Returns
        -------
        str
            A status message.
        """
        # Store the pattern and algorithm for inspection
        self.last_pattern = list(tokens)
        self.last_pattern_alg = algorithm
        # If a buffer exists, apply gating and pitch‑shift per step.  Each
        # token corresponds to one step of length ``step`` (in beats).
        if self.last_buffer is not None and len(self.last_buffer) > 0:
            import scipy.signal  # type: ignore
            step_beats = self.step
            if step_beats <= 0:
                step_beats = 1.0
            step_samples = int(step_beats * 60.0 / self.bpm * self.sample_rate)
            orig = self.last_buffer.astype(np.float64)
            out = np.zeros_like(orig)
            total_steps = max(len(tokens), int(len(orig) / step_samples) + 1)
            last_seg = None
            last_pitch = 0  # last semitone offset
            step_idx = 0
            i = 0
            while step_idx * step_samples < len(out) and i < len(tokens):
                tok = tokens[i]
                # Determine duration in steps; handle D tokens
                duration_steps = 1
                semitone = None
                if tok.upper().startswith('R'):
                    semitone = None
                elif tok.upper().startswith('D'):
                    # Extend previous pitch
                    dur_str = tok[1:]
                    try:
                        duration_steps = int(float(dur_str)) if dur_str else 1
                    except Exception:
                        duration_steps = 1
                    semitone = last_pitch
                else:
                    try:
                        semitone = int(float(tok))
                        last_pitch = semitone
                    except Exception:
                        # Unrecognised token: treat as no pitch change
                        semitone = 0
                        last_pitch = semitone
                # For each duration step, process segment
                for _ in range(duration_steps):
                    start = step_idx * step_samples
                    end = start + step_samples
                    if start >= len(out):
                        break
                    seg = orig[start:end]
                    if seg.size == 0:
                        break
                    if semitone is None:
                        # Rest: silence
                        out[start:end] = 0.0
                    else:
                        # Pitch shift segment.  Compute factor and resample
                        factor = 2.0 ** (semitone / 12.0)
                        # Resample to new length
                        new_len = int(len(seg) / factor)
                        if new_len < 1:
                            new_len = 1
                        resampled = scipy.signal.resample(seg, new_len)
                        # Pad or trim to step_samples
                        actual_len = min(step_samples, len(out) - start)
                        if new_len < step_samples:
                            # Repeat or pad to fill
                            rep = int(np.ceil(step_samples / new_len))
                            tile = np.tile(resampled, rep)[:actual_len]
                            out[start:start + len(tile)] = tile
                        elif new_len > step_samples:
                            # Trim centre portion
                            offset = (new_len - step_samples) // 2
                            trimmed = resampled[offset: offset + step_samples]
                            out[start:start + min(len(trimmed), len(out) - start)] = trimmed[:len(out) - start]
                        else:
                            out[start:start + min(new_len, len(out) - start)] = resampled[:len(out) - start]
                    step_idx += 1
                i += 1
            # If pattern shorter than buffer, copy remaining original samples
            leftover_start = step_idx * step_samples
            if leftover_start < len(out):
                out[leftover_start:] = orig[leftover_start:]
            # Update last_buffer and current file data if applicable
            self.last_buffer = out.astype(np.float64)
            # Update current file buffer if selected
            if self.current_file and self.current_file in self.files:
                self.files[self.current_file] = self.last_buffer.copy()
            # Update current clip buffer if selected
            if self.current_clip and self.current_clip in self.clips:
                self.clips[self.current_clip] = self.last_buffer.copy()
        return f"pattern stored: {' '.join(tokens)} with algorithm {algorithm}"

    # ------------ Level and hierarchy management ------------

    def set_level(self, level: str) -> None:
        """Change the current hierarchy level and reset lower levels.

        Valid levels are 'global', 'project', 'sketch', 'file' and
        'clip'.  Moving to a higher level than the current one will
        leave deeper context intact.  Moving to a lower level will
        clear the deeper context.
        """
        level = level.lower()
        levels = ['global', 'project', 'sketch', 'file', 'clip']
        if level not in levels:
            raise ValueError(f"unknown level '{level}'")
        # If the new level is higher in the hierarchy (closer to root)
        # than the current one, clear subordinate contexts.
        current_index = levels.index(self.current_level)
        new_index = levels.index(level)
        self.current_level = level
        if new_index <= current_index:
            # Clear deeper context values.  Do not reset counts so that
            # file and clip names remain unique within a project.
            if new_index <= 0:
                # global: clear project/sketch/file/clip names and counters
                self.current_project = ''
                self.project_count = 0
                self.current_sketch = ''
                self.sketch_count = 0
                self.current_file = ''
                self.current_clip = ''
                # Reset counters and lists only when fully resetting the project
                self.file_count = 0
                self.clip_count = 0
                self.tracks = []
                self.current_track_index = 0
            elif new_index == 1:
                # project: clear sketch/file/clip names; retain counters
                self.current_sketch = ''
                self.sketch_count = 0
                self.current_file = ''
                self.current_clip = ''
                # Reset file/clip counters for new sketch tree? preserve file_count to keep unique names
            elif new_index == 2:
                # sketch: clear file/clip names; retain counters
                self.current_file = ''
                self.current_clip = ''
            elif new_index == 3:
                # file: clear clip name; retain counter
                self.current_clip = ''

    def new_item(self, name: Optional[str] = None) -> str:
        """Create a new entity based on the current level.

        Parameters
        ----------
        name : str, optional
            Custom name for the entity. If not provided, auto-generates.

        - At global level: creates a new project and enters project level.
        - At project level: creates a new sketch and enters sketch level.
        - At sketch level: creates a new file and enters file level.
        - At file level: creates a new clip and enters clip level.
        - At clip level: raises an error; cannot create deeper entities.
        Returns a descriptive string indicating the created entity.
        """
        level = self.current_level
        if level == 'global':
            # Create a new project
            self.project_count += 1
            if name:
                self.current_project = name
            else:
                self.current_project = f"project_{self.project_count}"
            self.current_level = 'project'
            # Create a directory for this project with output and temp subdirs
            proj_dir = os.path.join(os.getcwd(), 'projects', self.current_project)
            # Ensure subdirectories exist
            try:
                os.makedirs(os.path.join(proj_dir, 'output'), exist_ok=True)
                os.makedirs(os.path.join(proj_dir, 'temp'), exist_ok=True)
                os.makedirs(os.path.join(proj_dir, 'data'), exist_ok=True)
            except Exception:
                pass
            # Update temp directory for previews to project temp
            self.temp_dir = os.path.join(proj_dir, 'temp')
            # Reset track list for new project and create a default track
            self.tracks = [{'name': 'track_1', 'effects': [], 'clips': []}]
            self.current_track_index = 0
            return f"project created: {self.current_project}"
        elif level == 'project':
            # Create a new sketch
            self.sketch_count += 1
            if name:
                self.current_sketch = name
            else:
                self.current_sketch = f"sketch_{self.sketch_count}"
            self.current_level = 'sketch'
            return f"sketch created: {self.current_sketch}"
        elif level == 'sketch':
            # Create a new file
            self.file_count += 1
            if name:
                file_name = name
            else:
                file_name = f"file_{self.file_count}"
            self.current_file = file_name
            self.current_level = 'file'
            # Persist the current last_buffer into the files dict.  If
            # no buffer exists, create a zero‑length array.
            if self.last_buffer is not None:
                self.files[file_name] = self.last_buffer.copy()
            else:
                self.files[file_name] = np.zeros(0, dtype=np.float64)
            return f"file created: {self.current_file}"
        elif level == 'file':
            # Create a new clip
            self.clip_count += 1
            if name:
                self.current_clip = name
            else:
                self.current_clip = f"clip_{self.clip_count}"
            self.current_level = 'clip'
            # Associate the current file buffer as the clip if present
            if self.current_file and self.current_file in self.files:
                self.clips[self.current_clip] = self.files[self.current_file].copy()
            return f"clip created: {self.current_clip}"
        else:
            # Already at the deepest level
            raise RuntimeError("cannot create new entity at clip level")

    def set_synth_level(self, value: int) -> None:
        """Set the synth level used for operator/voice parameter editing.

        Valid values are 1 (global), 2 (operator), and 3 (voice).
        """
        if value not in (1, 2, 3):
            raise ValueError("synth level must be 1, 2, or 3")
        self.synth_level = value

    # Ensure the engine has at least `count` operators.  Missing operators
    # are created with default parameters.  Operators are indexed
    # sequentially from 0.
    def _ensure_operators(self, count: int) -> None:
        current_count = len(self.engine.operators)
        # Add new operators as needed
        for i in range(current_count, count):
            # Default parameters: sine wave at frequency 440*(i+1), full amplitude
            freq = 440.0 * (i + 1)
            self.engine.set_operator(i, wave_type='sine', freq=freq, amp=1.0, phase=0.0)

    # ------------ Effects management ------------
    def add_effect(self, name: str) -> None:
        """Add an effect to the processing chain.

        Valid effect names are defined in ``mdma_rebuild.dsp.effects``.
        """
        try:
            from ..dsp.effects import _effect_funcs
            if name not in _effect_funcs:
                raise ValueError(f"unknown effect '{name}'")
            self.effects.append(name)
            # Append a corresponding empty parameter dict
            self.effect_params.append({})
            # Auto select the newly added effect
            self.selected_effect = len(self.effects) - 1
        except Exception:
            # Re-raise as ValueError for consistency
            raise ValueError(f"unknown effect '{name}'")

    def clear_effects(self) -> None:
        """Remove all effects from the chain."""
        self.effects = []
        self.effect_params = []
        self.selected_effect = -1

    def list_effects(self) -> list[str]:
        """Return a copy of the current effects chain."""
        return list(self.effects)

    # ------------ Track management ------------
    def new_track(self) -> str:
        """Create a new track in the current project and select it.

        Uses the continuous stereo array schema (v40).
        Returns a descriptive message.
        """
        name = f"track_{len(self.tracks) + 1}"
        self.tracks.append({
            'name': name,
            'audio': np.zeros((self.project_length_samples, 2), dtype=np.float64),
            'fx_chain': [],
            'write_pos': 0,
            'gain': 1.0,
            'pan': 0.0,
            'mute': False,
            'solo': False,
        })
        self.current_track_index = len(self.tracks) - 1
        return f"track created: {name}"

    def list_tracks(self) -> list[str]:
        """Return a list of track names for the current project."""
        return [t['name'] for t in self.tracks]

    def select_track(self, idx: int) -> str:
        """Select an existing track by index (0‑based).

        Raises ValueError if index is out of range.
        """
        if idx < 0 or idx >= len(self.tracks):
            raise ValueError("invalid track index")
        self.current_track_index = idx
        return self.tracks[idx]['name']

    def insert_clip_to_track(self, clip_name: str, start_beats: float, **kwargs) -> str:
        """Insert a named clip into the current track at a beat position.

        Parameters
        ----------
        clip_name : str
            Name of the clip in ``self.clips`` or ``self.files``.  If
            the clip does not exist, a ValueError is raised.
        start_beats : float
            Position (in beats) where the clip should start.
        **kwargs : dict
            Optional processing parameters:
            - stretch: float (time stretch factor, 1.0 = no change)
            - pitch_locked: bool (preserve pitch during stretch)
            - effects: list (clip-level effects to apply)
            - gain: float (volume multiplier)

        Returns
        -------
        str
            A status message.
        """
        if clip_name in self.clips:
            buf = self.clips[clip_name]
        elif clip_name in self.files:
            buf = self.files[clip_name]
        else:
            raise ValueError(f"unknown clip or file '{clip_name}'")
        
        # Convert beat position to sample index
        start_beats = float(start_beats)
        if start_beats < 0:
            start_beats = 0.0
        start_sample = int(start_beats * 60.0 / self.bpm * self.sample_rate)
        
        # Create clip data dict with processing metadata
        clip_data = {
            'start': start_sample,
            'buffer': buf.copy(),
            'stretch': kwargs.get('stretch', getattr(self, 'clip_stretch', 1.0)),
            'pitch_locked': kwargs.get('pitch_locked', getattr(self, 'pitch_locked', True)),
            'effects': kwargs.get('effects', []),
            'gain': kwargs.get('gain', 1.0),
            'name': clip_name,
        }
        
        # Append to current track's clip list
        track = self.tracks[self.current_track_index]
        track['clips'].append(clip_data)
        
        stretch_info = f" (stretch={clip_data['stretch']:.2f}x)" if clip_data['stretch'] != 1.0 else ""
        return f"clip '{clip_name}' inserted at {start_beats:.3f} beats on {track['name']}{stretch_info}"

    def _process_clip(self, clip_data) -> np.ndarray:
        """Process a clip applying stretch, pitch shift, and effects.
        
        Parameters
        ----------
        clip_data : tuple or dict
            Either (start_sample, buffer) tuple or dict with processing params
        
        Returns
        -------
        tuple
            (start_sample, processed_buffer)
        """
        # Handle both old-style tuples and new-style dicts
        if isinstance(clip_data, dict):
            start = clip_data.get('start', 0)
            buf = clip_data.get('buffer', np.zeros(0)).astype(np.float64)
            stretch = clip_data.get('stretch', 1.0)
            pitch_locked = clip_data.get('pitch_locked', True)
            clip_effects = clip_data.get('effects', [])
            gain = clip_data.get('gain', 1.0)
        elif isinstance(clip_data, (list, tuple)) and len(clip_data) >= 2:
            start = clip_data[0]
            buf = clip_data[1].astype(np.float64)
            # Use session defaults for processing
            stretch = getattr(self, 'clip_stretch', 1.0)
            pitch_locked = getattr(self, 'pitch_locked', True)
            clip_effects = []
            gain = 1.0
        else:
            return (0, np.zeros(0, dtype=np.float64))
        
        # Apply time stretch if needed
        if stretch != 1.0 and len(buf) > 0:
            import scipy.signal
            new_len = int(len(buf) * stretch)
            if new_len > 0:
                if pitch_locked:
                    # Time stretch with pitch preservation using phase vocoder approach:
                    # 1. Resample to new length (changes both time and pitch)
                    # 2. Pitch-shift back to compensate for the pitch change
                    x_old = np.linspace(0, 1, len(buf))
                    x_new = np.linspace(0, 1, new_len)
                    stretched = np.interp(x_new, x_old, buf)
                    
                    # Pitch correction: resample to compensate for pitch shift
                    # When we stretch by factor S, pitch shifts by 1/S
                    # So we need to resample by factor S to restore original pitch
                    # This means the final length = new_len * stretch = original * stretch^2
                    # ... which isn't right. Instead, use STFT-based approach:
                    hop = 256
                    win_len = 1024
                    window = np.hanning(win_len)
                    
                    # Phase vocoder time stretch
                    n_frames_in = (len(buf) - win_len) // hop + 1
                    if n_frames_in < 2:
                        # Too short for phase vocoder, fall back to simple interp
                        buf = stretched
                    else:
                        # Compute STFT
                        stft_frames = []
                        for i in range(n_frames_in):
                            start = i * hop
                            frame = buf[start:start + win_len] * window
                            stft_frames.append(np.fft.rfft(frame))
                        
                        # Synthesize with modified hop size
                        out_hop = int(hop * stretch)
                        out_len = (n_frames_in - 1) * out_hop + win_len
                        output = np.zeros(out_len)
                        window_sum = np.zeros(out_len)
                        
                        # Phase accumulator for phase vocoder
                        phase_acc = np.angle(stft_frames[0]) if stft_frames else np.zeros(win_len // 2 + 1)
                        
                        for i, frame_fft in enumerate(stft_frames):
                            mag = np.abs(frame_fft)
                            phase = np.angle(frame_fft)
                            
                            if i > 0:
                                # Phase advance
                                prev_phase = np.angle(stft_frames[i - 1])
                                freq_bins = np.arange(len(phase))
                                expected_advance = 2.0 * np.pi * freq_bins * hop / win_len
                                phase_diff = phase - prev_phase - expected_advance
                                # Wrap to [-pi, pi]
                                phase_diff = phase_diff - 2.0 * np.pi * np.round(phase_diff / (2.0 * np.pi))
                                true_freq = expected_advance + phase_diff
                                phase_acc += true_freq * stretch
                            
                            # Reconstruct frame
                            synth_fft = mag * np.exp(1j * phase_acc)
                            frame_out = np.fft.irfft(synth_fft, n=win_len) * window
                            
                            start = i * out_hop
                            end = start + win_len
                            if end <= out_len:
                                output[start:end] += frame_out
                                window_sum[start:end] += window ** 2
                        
                        # Normalize by window sum
                        nonzero = window_sum > 1e-8
                        output[nonzero] /= window_sum[nonzero]
                        
                        # Trim to target length
                        if len(output) > new_len:
                            buf = output[:new_len]
                        else:
                            buf = np.pad(output, (0, max(0, new_len - len(output))))
                else:
                    # Tape-style stretch (pitch follows length)
                    x_old = np.linspace(0, 1, len(buf))
                    x_new = np.linspace(0, 1, new_len)
                    buf = np.interp(x_new, x_old, buf)
        
        # Apply clip-level effects
        if clip_effects:
            try:
                from ..dsp.effects import apply_effects
                buf = apply_effects(buf, clip_effects)
            except Exception:
                pass
        
        # Apply gain
        if gain != 1.0:
            buf = buf * gain
        
        return (start, buf)

    def mix_tracks(self) -> np.ndarray:
        """Mix all tracks into a stereo buffer (track FX + master FX).

        Track audio is stored as stereo (N, 2) arrays.
        Mixing rules:
        - Apply per-track fx_chain (if any) then apply gain and pan
        - Sum all active tracks into a stereo mix
        - Apply session.master_fx_chain on the final mix
        - Normalize if needed

        Returns np.ndarray with shape (N, 2) for stereo output.
        """
        if not self.tracks:
            return np.zeros((0, 2), dtype=np.float64)

        any_solo = any(bool(t.get('solo')) for t in self.tracks)
        max_len = 0
        for t in self.tracks:
            a = t.get('audio')
            if a is not None:
                max_len = max(max_len, a.shape[0] if a.ndim >= 1 else len(a))
        if max_len == 0:
            return np.zeros((0, 2), dtype=np.float64)

        mix = np.zeros((max_len, 2), dtype=np.float64)

        for t in self.tracks:
            if t.get('mute'):
                continue
            if any_solo and not t.get('solo'):
                continue

            buf = t.get('audio')
            if buf is None or (buf.ndim >= 1 and buf.shape[0] == 0):
                continue

            # Ensure stereo (N, 2)
            if buf.ndim == 1:
                buf = np.column_stack([buf, buf])
            buf = buf.astype(np.float64, copy=True)

            # Apply per-track gain
            gain = float(t.get('gain', 1.0))
            if gain != 1.0:
                buf = buf * gain

            # Apply per-track FX chain
            chain = t.get('fx_chain', []) or self.track_fx_chain
            if chain:
                try:
                    # Apply FX to each channel (most effects are mono-pipeline)
                    left = self.apply_fx_chain(buf[:, 0], chain)
                    right = self.apply_fx_chain(buf[:, 1], chain)
                    buf = np.column_stack([left, right])
                except Exception:
                    pass

            # Apply per-track pan using equal-power pan law
            pan = float(t.get('pan', 0.0))  # -1.0 to 1.0
            pan = max(-1.0, min(1.0, pan))
            # Equal-power: cos/sin panning
            angle = (pan + 1.0) * 0.25 * np.pi  # 0 to pi/2
            pan_l = float(np.cos(angle))
            pan_r = float(np.sin(angle))
            buf[:, 0] *= pan_l
            buf[:, 1] *= pan_r

            n = min(buf.shape[0], max_len)
            mix[:n] += buf[:n]

        # Apply master FX chain (per-channel)
        if self.master_fx_chain:
            try:
                left = self.apply_fx_chain(mix[:, 0], self.master_fx_chain)
                right = self.apply_fx_chain(mix[:, 1], self.master_fx_chain)
                mix = np.column_stack([left, right])
            except Exception:
                pass

        # Apply master gain (T.3.4)
        mg = getattr(self, 'master_gain', 0.0)
        if mg != 0.0:
            mix = mix * (10.0 ** (mg / 20.0))

        # Normalize if clipping
        max_val = float(np.max(np.abs(mix))) if mix.size else 0.0
        if max_val > 1.0:
            mix = mix / max_val

        self.last_buffer = mix.astype(np.float64)
        return self.last_buffer


# ---------------------------------------------------------------------------
# V2 backend repo-prep: attach helper modules to Session.
#
# Playback / wav-writing lives in ``core/audio_io.py``, numbered-buffer and
# working-buffer management in ``core/buffer_store.py``, and the long tail
# of filter / envelope / voice / operator parameter setters in
# ``core/session_params.py``. Each of those modules exposes a ``bind_to``
# function that walks its public functions and assigns them onto the class
# here, so ``session.play_last_buffer()``, ``session.get_buffer(...)``,
# ``session.set_cutoff(...)``, etc. continue to resolve as before.
# ---------------------------------------------------------------------------
from . import audio_io as _audio_io  # noqa: E402
from . import backend_bridge as _backend_bridge  # noqa: E402
from . import buffer_store as _buffer_store  # noqa: E402
from . import session_params as _session_params  # noqa: E402

_audio_io.bind_to(Session)
_buffer_store.bind_to(Session)
_session_params.bind_to(Session)
_backend_bridge.bind_to(Session)
