"""MDMA First-Class Object Model.

Defines the base data classes for all MDMA objects. Every significant
entity (pattern, beat, loop, clip, patch, effect chain, track, song,
template) inherits from MDMAObject and carries a stable identity, name,
creation history, and version counter.

This module is Phase 1 of the Object Model migration — data classes only.
No generation or command code is modified by importing this module.

See: docs/specs/OBJECT_MODEL_SPEC.md for the full specification.

BUILD ID: objects_v1.0_phase1
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


# ============================================================================
# SUPPORTING DATA TYPES
# ============================================================================

@dataclass
class NoteEvent:
    """A single note event within a Pattern.

    Attributes:
        pitch: MIDI note number (0-127).
        velocity: Note velocity (0-127).
        start: Start position in beats.
        duration: Duration in beats.
    """
    pitch: int
    velocity: int = 100
    start: float = 0.0
    duration: float = 0.25


@dataclass
class DrumHit:
    """A single drum hit within a BeatPattern.

    Attributes:
        instrument: Instrument name (e.g. 'kick', 'snare', 'hihat').
        step: Step index within the pattern grid.
        velocity: Hit velocity (0-127).
    """
    instrument: str
    step: int
    velocity: int = 100


@dataclass
class OperatorState:
    """Snapshot of a single FM operator's parameters.

    Attributes:
        index: Operator index in the engine.
        waveform: Waveform type name.
        ratio: Frequency ratio relative to base.
        level: Output level (0.0-1.0).
        feedback: Self-modulation feedback amount.
        is_carrier: Whether this operator outputs directly.
        is_modulator: Whether this operator modulates another.
        mod_target: Index of the operator this modulates (if modulator).
    """
    index: int = 0
    waveform: str = "sine"
    ratio: float = 1.0
    level: float = 1.0
    feedback: float = 0.0
    is_carrier: bool = True
    is_modulator: bool = False
    mod_target: int = -1


@dataclass
class EnvelopeState:
    """ADSR envelope parameter snapshot.

    Attributes:
        attack: Attack time in seconds.
        decay: Decay time in seconds.
        sustain: Sustain level (0.0-1.0).
        release: Release time in seconds.
    """
    attack: float = 0.01
    decay: float = 0.1
    sustain: float = 0.8
    release: float = 0.1


@dataclass
class ModRoute:
    """A modulation routing assignment.

    Attributes:
        source: Source type ('lfo', 'envelope', 'macro').
        source_id: Identifier for the specific source instance.
        target_param: Target parameter name.
        depth: Modulation depth (0.0-1.0).
        rate: LFO rate in Hz (for LFO sources).
        shape: LFO shape name (for LFO sources).
    """
    source: str = "lfo"
    source_id: str = ""
    target_param: str = ""
    depth: float = 0.5
    rate: float = 1.0
    shape: str = "sine"


@dataclass
class EffectSlot:
    """A single effect in an effect chain.

    Attributes:
        effect_name: Name of the effect (must match dsp/effects.py names).
        params: Parameter dict for this effect instance.
        enabled: Whether this slot is active.
    """
    effect_name: str = ""
    params: dict = field(default_factory=dict)
    enabled: bool = True


@dataclass
class Placement:
    """An object placed on a track at a specific position.

    Attributes:
        object_id: ID of the placed object (Pattern, AudioClip, etc.).
        position_beats: Start position in beats from the track origin.
        length_override: Optional length override in beats (None = use object's natural length).
    """
    object_id: str = ""
    position_beats: float = 0.0
    length_override: Optional[float] = None


@dataclass
class TemplateField:
    """A single field definition within a Template.

    Attributes:
        name: Internal field name.
        field_type: Type tag ('string', 'int', 'float', 'choice', 'object_ref').
        label: Screen-reader-friendly short label.
        description: Longer description for new users.
        default: Default value.
        choices: List of valid choices (for 'choice' type).
        min_value: Minimum value (for numeric types).
        max_value: Maximum value (for numeric types).
        required: Whether this field must be filled.
    """
    name: str = ""
    field_type: str = "string"
    label: str = ""
    description: str = ""
    default: Any = None
    choices: list = field(default_factory=list)
    min_value: float = 0.0
    max_value: float = 100.0
    required: bool = False


# ============================================================================
# HELPER
# ============================================================================

def _new_id() -> str:
    """Generate a new UUID string for object identity."""
    return str(uuid.uuid4())


def _now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.utcnow()


# ============================================================================
# BASE OBJECT
# ============================================================================

@dataclass
class MDMAObject:
    """Base class for all first-class MDMA objects.

    Every object in the registry inherits from this base and carries:
    - A stable UUID identity
    - A human-readable, user-editable name
    - A type tag for registry queries
    - Creation and modification timestamps
    - User-defined tags for organisation
    - Source provenance (parameters and parent object IDs)
    - A version counter that increments on each modification
    """

    id: str = field(default_factory=_new_id)
    name: str = ""
    obj_type: str = "base"
    created_at: datetime = field(default_factory=_now)
    modified_at: datetime = field(default_factory=_now)
    tags: list[str] = field(default_factory=list)
    source_params: dict = field(default_factory=dict)
    source_object_ids: list[str] = field(default_factory=list)
    notes: str = ""
    version: int = 1
    is_template: bool = False

    def touch(self) -> None:
        """Update the modification timestamp and increment version."""
        self.modified_at = _now()
        self.version += 1

    def summary(self) -> str:
        """Return a short human-readable summary string."""
        return f"{self.obj_type}:{self.name} (v{self.version})"


# ============================================================================
# CONCRETE OBJECT TYPES
# ============================================================================

@dataclass
class Pattern(MDMAObject):
    """A sequence of musical events — notes, rhythms, velocities, durations.

    The fundamental unit of musical content. Produced by the Generation
    window, Mutation window, Pattern editor, and CLI commands like
    /gen2, /mel, /cor.
    """

    obj_type: str = "pattern"
    events: list[NoteEvent] = field(default_factory=list)
    length_beats: float = 4.0
    scale: str = "minor"
    root: str = "C"
    bpm: float = 120.0
    time_signature: tuple[int, int] = (4, 4)
    pattern_kind: str = "melody"  # melody, chord, bassline, arpeggio, drone


@dataclass
class BeatPattern(MDMAObject):
    """A rhythmic pattern defined in terms of drum hits.

    Produced by the Generation window (beat generator panel) and
    the CLI /beat command.
    """

    obj_type: str = "beat_pattern"
    hits: list[DrumHit] = field(default_factory=list)
    steps: int = 16
    genre: str = ""
    bars: int = 4
    bpm: float = 120.0
    swing: float = 0.0  # 0.0 to 1.0


@dataclass
class Loop(MDMAObject):
    """A composite object bundling multiple patterns into a reusable loop.

    Layers map layer names to Pattern or BeatPattern IDs in the registry.
    Produced by the Generation window (loop generator panel) and
    the CLI /loop command.
    """

    obj_type: str = "loop"
    layers: dict[str, str] = field(default_factory=dict)
    bars: int = 4
    bpm: float = 120.0
    genre: str = ""


@dataclass
class AudioClip(MDMAObject):
    """Rendered audio data — a concrete waveform.

    Audio data is stored as a numpy float64 stereo array (N x 2) at
    the given sample rate. Produced by rendering patterns, importing
    audio files, AI generation, and CLI commands like /render, /tone.

    Note: The ``data`` field is not serialised to JSON — audio is
    persisted as a referenced WAV file in the project package.
    """

    obj_type: str = "audio_clip"
    # data is intentionally not in __init__ defaults to avoid large
    # default allocations; set it after construction.
    data: Any = None  # np.ndarray (N x 2) float64 — set post-init
    sample_rate: int = 48000
    duration_seconds: float = 0.0
    bit_depth: int = 24
    render_source_id: str = ""
    render_patch_id: str = ""


@dataclass
class Patch(MDMAObject):
    """A synthesizer configuration — full Monolith engine parameter state.

    Produced by the Synthesis window, preset saves, and
    the CLI /preset save command.
    """

    obj_type: str = "patch"
    engine: str = "monolith"
    operators: list[OperatorState] = field(default_factory=list)
    waveform: str = "sine"
    envelope: EnvelopeState = field(default_factory=EnvelopeState)
    modulation: list[ModRoute] = field(default_factory=list)
    effects_chain_id: str = ""
    params: dict = field(default_factory=dict)


@dataclass
class EffectChain(MDMAObject):
    """An ordered sequence of effects with parameters.

    A named, reusable processing pipeline. Produced by the Effects
    window (chain builder panel) and CLI /chain save.
    """

    obj_type: str = "effect_chain"
    effects: list[EffectSlot] = field(default_factory=list)
    category: str = ""  # spatial, dynamics, tone, texture, damage, utility


@dataclass
class Track(MDMAObject):
    """A single lane in an arrangement — object placements over time.

    Produced by the Arrangement window.
    """

    obj_type: str = "track"
    track_kind: str = "midi"  # audio, beat, midi
    patch_id: str = ""
    placements: list[Placement] = field(default_factory=list)
    volume: float = 1.0  # 0.0 to 1.0
    pan: float = 0.0     # -1.0 to 1.0
    muted: bool = False
    soloed: bool = False
    effects_chain_id: str = ""


@dataclass
class Song(MDMAObject):
    """Top-level project object — composition of all tracks.

    Produced by the Arrangement window.
    """

    obj_type: str = "song"
    track_ids: list[str] = field(default_factory=list)
    bpm: float = 120.0
    time_signature: tuple[int, int] = (4, 4)
    key: str = "C"
    scale: str = "minor"
    total_length_beats: float = 0.0
    output_format: str = "wav_24"
    hq_mode: bool = False


@dataclass
class Template(MDMAObject):
    """A parameterised recipe for creating any other object type.

    Templates are themselves objects — they live in the registry and
    can be saved, shared, and versioned. They provide screen-reader
    users with a structured, form-based interface for object creation.
    """

    obj_type: str = "template"
    target_type: str = ""  # Type of object this template creates
    fields: list[TemplateField] = field(default_factory=list)
    description: str = ""


# ============================================================================
# TYPE REGISTRY — maps type tags to classes
# ============================================================================

OBJECT_TYPE_MAP: dict[str, type] = {
    "pattern": Pattern,
    "beat_pattern": BeatPattern,
    "loop": Loop,
    "audio_clip": AudioClip,
    "patch": Patch,
    "effect_chain": EffectChain,
    "track": Track,
    "song": Song,
    "template": Template,
}
