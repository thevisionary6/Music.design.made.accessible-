# MDMA First-Class Object Model Specification

**Version:** 1.0
**Status:** Planning
**Depended on by:** GUI_WINDOW_ARCHITECTURE_SPEC.md
**Author:** MDMA Development

-----

## Overview

This document specifies the first-class object system for MDMA. The core shift it defines is: **every significant thing that MDMA creates or manages is a named, stored, versioned object with a stable identity.**

Currently, most generation and processing operations are destructive or transient — they produce audio buffers that are consumed or overwritten without being tracked as discrete entities. This makes it impossible for users to browse, organise, reuse, or compose the things they've made.

The object model changes this. Every generated pattern, every synthesizer patch, every effect chain, every rendered clip becomes a first-class object: it has a unique identity, a human-readable name, a type, a creation history, and relationships to other objects. The CLI and GUI both become *views* into the same object registry.

### Relationship to Existing Architecture

The existing codebase already has registry-like patterns:

- `core/song_registry.py` — Persistent song database with quality grading
- `core/banks.py` — Synth preset bank system
- `Session` object — Central state holder with buffers, tracks, effects

This object model builds on those foundations:
- `SongRegistry` becomes one consumer of the new `ObjectRegistry`
- Bank presets become `Patch` objects registered in the shared registry
- Session buffers become `AudioClip` objects with stable identities

-----

## Design Principles

**Identity over data.** Objects are not just saved audio blobs. They are entities with IDs, names, types, and metadata. Two objects can contain identical audio but remain distinct entities.

**Reproducibility.** Every object records the parameters and source objects that produced it. Given the same parameters, the same object can be reproduced. This supports undo, variation, and template systems.

**Composability.** Objects can reference other objects. A `Loop` contains `Pattern` references. An `Arrangement` references `Track` objects which reference `Pattern` and `AudioClip` objects. The registry manages these relationships.

**Non-destructive by default.** Operations that transform objects produce new objects rather than overwriting the source. The source is preserved unless the user explicitly chooses to overwrite.

**CLI and GUI parity.** Every object operation available in the GUI is also expressible as a CLI command, and vice versa. The object system is the shared substrate both interfaces operate on.

**Templates as a first-class feature.** Any object type can have an associated template — a parameterised recipe for creating instances of that type. Templates are themselves objects and can be saved, shared, and versioned.

-----

## Object Taxonomy

### Base Object

All MDMA objects share a common base structure:

```python
@dataclass
class MDMAObject:
    id: str                        # UUID, generated at creation
    name: str                      # Human-readable, user-editable
    type: str                      # Type tag — see type registry below
    created_at: datetime
    modified_at: datetime
    tags: list[str]                # User-defined tags for organisation
    source_params: dict            # Parameters that produced this object
    source_object_ids: list[str]   # IDs of objects this was derived from
    notes: str                     # User-editable freetext notes
    version: int                   # Increments on each modification
    is_template: bool              # True if this is a template instance
```

Every subclass adds its own data fields on top of this base.

-----

### Object Types

#### Pattern

A sequence of musical events — notes, rhythms, velocities, durations. The fundamental unit of musical content.

```python
@dataclass
class Pattern(MDMAObject):
    type = "pattern"
    notes: list[NoteEvent]         # List of (pitch, velocity, start, duration)
    length_beats: float
    scale: str                     # e.g. "minor", "dorian"
    root: str                      # e.g. "C", "F#"
    bpm: float
    time_signature: tuple[int,int] # e.g. (4, 4)
    pattern_kind: str              # "melody", "chord", "bassline", "arpeggio", "drone"
```

**Produced by:** Generation window (melody/harmony panel, generative theory panel), Mutation window (transform/adapt panels), Pattern editor, CLI `/gen2`, `/mel`, `/cor`

-----

#### BeatPattern

A rhythmic pattern defined in terms of drum hits rather than pitched notes.

```python
@dataclass
class BeatPattern(MDMAObject):
    type = "beat_pattern"
    hits: list[DrumHit]            # List of (instrument, step, velocity)
    steps: int                     # e.g. 16 for 16-step pattern
    genre: str
    bars: int
    bpm: float
    swing: float                   # 0.0-1.0
```

**Produced by:** Generation window (beat generator panel), CLI `/beat`

-----

#### Loop

A composite object that bundles multiple patterns into a single reusable unit — a full musical loop with all its layers intact.

```python
@dataclass
class Loop(MDMAObject):
    type = "loop"
    layers: dict[str, str]         # layer name -> Pattern or BeatPattern ID
    # e.g. {"drums": "uuid-...", "bass": "uuid-...", "chords": "uuid-..."}
    bars: int
    bpm: float
    genre: str
```

**Produced by:** Generation window (loop generator panel), CLI `/loop`

-----

#### AudioClip

Rendered audio data — a concrete waveform. This is what you hear.

```python
@dataclass
class AudioClip(MDMAObject):
    type = "audio_clip"
    data: np.ndarray               # Float64 stereo (N x 2)
    sample_rate: int               # Always 44100 or 48000
    duration_seconds: float
    bit_depth: int                 # 16, 24, or 32
    render_source_id: str          # ID of Pattern/BeatPattern/Loop rendered to produce this
    render_patch_id: str           # ID of Patch used during render (if applicable)
```

**Produced by:** Rendering any Pattern/BeatPattern/Loop, importing audio files, AI generation, CLI `/render`, `/import`, `/tone`

-----

#### Patch

A synthesizer configuration — the full parameter state of the Monolith engine for a particular sound.

```python
@dataclass
class Patch(MDMAObject):
    type = "patch"
    engine: str                    # "monolith", "wavetable", "physical"
    operators: list[OperatorState] # FM operators if applicable
    waveform: str
    envelope: EnvelopeState        # ADSR
    modulation: list[ModRoute]     # LFO -> target assignments
    effects_chain_id: str          # Optional: ID of an EffectChain to auto-apply
    params: dict                   # All other engine parameters
```

**Produced by:** Synthesis window, preset saves, CLI `/preset save`

-----

#### EffectChain

An ordered sequence of effects with their parameters — a named, reusable processing pipeline.

```python
@dataclass
class EffectChain(MDMAObject):
    type = "effect_chain"
    effects: list[EffectSlot]      # Ordered list of (effect_name, params dict)
    category: str                  # "spatial", "dynamics", "tone", "texture", "damage", "utility"
```

**Produced by:** Effects window (chain builder panel), CLI `/chain save`

-----

#### Track

A single lane in an arrangement — a sequence of object placements over time.

```python
@dataclass
class Track(MDMAObject):
    type = "track"
    track_kind: str                # "audio", "beat", "midi"
    patch_id: str                  # For MIDI tracks: which Patch to use for playback
    placements: list[Placement]    # List of (object_id, position_beats, length_override)
    volume: float                  # 0.0-1.0
    pan: float                     # -1.0 to 1.0
    muted: bool
    soloed: bool
    effects_chain_id: str          # Track-level effects chain
```

-----

#### Song / Arrangement

The top-level project object — the composition of all tracks into a complete piece.

```python
@dataclass
class Song(MDMAObject):
    type = "song"
    tracks: list[str]              # Ordered list of Track IDs
    bpm: float
    time_signature: tuple[int,int]
    key: str
    scale: str
    total_length_beats: float
    output_format: str             # "wav_16", "wav_24", "flac_24"
    hq_mode: bool
```

-----

#### Template

A parameterised recipe for creating any other object type. Templates are objects too — they live in the registry and can be saved, shared, and versioned.

```python
@dataclass
class Template(MDMAObject):
    type = "template"
    target_type: str               # Type of object this template creates
    fields: list[TemplateField]    # Named slots with types, defaults, and descriptions
    description: str               # Human-readable explanation of what this template makes
```

```python
@dataclass
class TemplateField:
    name: str
    field_type: str                # "string", "int", "float", "choice", "object_ref"
    label: str                     # Screen reader friendly label
    description: str               # Longer description for new users
    default: Any
    choices: list                  # For "choice" type fields
    min_value: float               # For numeric fields
    max_value: float
    required: bool
```

**Example — Beat Template:**

```
name: "My Beat Template"
target_type: "beat_pattern"
fields:
  - name: genre,    type: choice,   choices: [hip_hop, house, dnb, techno, ...],  default: hip_hop
  - name: bars,     type: int,      min: 1, max: 32,  default: 4
  - name: swing,    type: float,    min: 0, max: 100, default: 0
  - name: bpm,      type: float,    min: 60, max: 200, default: 120
```

Filling out a template and submitting it calls the generator with those parameters and produces a new `BeatPattern` object.

-----

## The Object Registry

The registry is the single, authoritative store for all objects in a project. Both the CLI and the GUI read from and write to it. Nothing exists outside the registry except transient intermediate computation.

```python
class ObjectRegistry:
    def register(obj: MDMAObject) -> str          # Returns ID
    def get(id: str) -> MDMAObject
    def get_by_name(name: str) -> MDMAObject
    def list(type: str = None) -> list[MDMAObject]
    def update(id: str, updates: dict) -> MDMAObject
    def delete(id: str)
    def duplicate(id: str, new_name: str) -> MDMAObject
    def search(query: str, type: str = None) -> list[MDMAObject]
    def get_dependents(id: str) -> list[MDMAObject]  # What references this object?
    def export(id: str, path: str)                   # Export to file
    def import_file(path: str) -> MDMAObject         # Import from file
    def subscribe(callback, event_type: str)          # Event subscription
    def fire(event: RegistryEvent)                    # Internal event dispatch
```

### Registry Events

```python
class RegistryEvent:
    OBJECT_CREATED  = "object_created"
    OBJECT_UPDATED  = "object_updated"
    OBJECT_DELETED  = "object_deleted"
    OBJECT_RENAMED  = "object_renamed"
```

All GUI windows subscribe to registry events to keep their views current. When a new `BeatPattern` is created in the CLI, the Generation window's output list and the Inspector's object tree both update automatically.

-----

## CLI Integration

The object system maps cleanly onto CLI commands. The existing command layer becomes a thin wrapper that creates and retrieves objects rather than manipulating raw buffers.

### New CLI commands introduced by the object model

```
/obj list [type]                  -- List all objects, optionally filtered by type
/obj list patterns                -- List all Pattern objects
/obj list beats                   -- List all BeatPattern objects
/obj info <name|id>               -- Show full object detail
/obj rename <name|id> <new_name>  -- Rename an object
/obj tag <name|id> <tag>          -- Add a tag to an object
/obj dup <name|id> [new_name]     -- Duplicate an object
/obj delete <name|id>             -- Delete an object
/obj export <name|id> <path>      -- Export to file
/obj import <path>                -- Import from file

/template list                    -- List all saved templates
/template show <name>             -- Show template fields
/template fill <name>             -- Interactive template filler (walks through fields)
/template save <name>             -- Save current context as a template
```

### Modified generation commands

Existing commands gain an optional `--name` parameter and always produce an object:

```
/beat hip_hop 4              -> Creates BeatPattern named "beat_001" (auto-named)
/beat hip_hop 4 --name my_beat  -> Creates BeatPattern named "my_beat"
/gen2 melody minor 8         -> Creates Pattern named "melody_001"
/loop house 8 --name main_loop  -> Creates Loop named "main_loop"
```

The audio output behaviour is unchanged — you still hear the result immediately. The difference is that the object also persists in the registry.

-----

## Object Naming

Auto-naming follows the pattern `{type}_{sequence}` — `beat_001`, `melody_002`, `patch_003`. Names must be unique within a type. Users can rename at any time. Names are the primary human-facing identifier; IDs are the internal stable reference.

-----

## Persistence

The registry serialises to a project file alongside audio data. Object metadata is stored as JSON. Audio data (`AudioClip.data`) is stored as referenced WAV files within the project package. On project load, the registry is fully reconstructed including all object relationships.

```
my_project/
  project.json         # Registry metadata -- all objects, relationships, names
  audio/
    clip_uuid1.wav
    clip_uuid2.wav
  patches/
    patch_uuid1.json
  templates/
    template_uuid1.json
```

-----

## Template System — Extended Design

Templates are the user-facing tool for creating reproducible, customisable workflows. They are especially valuable for accessibility — they give screen reader users a structured, predictable form-based interface for creating complex objects without needing to remember command syntax.

### Template lifecycle

1. **Discover** — User browses available templates via `/template list` or the Template Browser in any generation window.
2. **Fill** — User fills template fields. In CLI: interactive prompt per field. In GUI: form panel with labeled controls.
3. **Submit** — Template engine calls the appropriate generator with the filled parameters.
4. **Object created** — A new first-class object appears in the registry.
5. **Save as new template** — User can save any set of filled parameters as a new template for future reuse.

### Built-in templates (initial set)

These ship with MDMA as starting points:

```
Beat Templates:
  - Basic Hip Hop Beat (4 bars, moderate swing)
  - Minimal Techno Loop (8 bars, straight)
  - Breakbeat Template (4 bars, heavy swing)

Melody Templates:
  - Simple Pentatonic Hook (8 bars, major pentatonic)
  - Dark Arpeggio (4 bars, minor)
  - Bass Line Template (4 bars, root-focused)

Loop Templates:
  - Full Hip Hop Loop (drums + bass + chords, 4 bars)
  - House Loop (drums + bass, 8 bars)

Patch Templates:
  - Basic FM Lead
  - Sub Bass
  - Pad Template

Effect Chain Templates:
  - Lo-fi Processing Chain
  - Room Reverb Chain
  - Mastering Chain
```

### User-created templates

Any parameter set can be saved as a template. Templates are themselves objects in the registry — they can be named, tagged, exported, and shared. This means a user can build up a personal library of creative starting points over time.

-----

## Migration Path from Current Architecture

The object model is designed to be introduced incrementally without breaking existing functionality.

**Step 1 — Define data classes only.**
Create the `MDMAObject` base and all subtype classes in `core/objects.py`. No generation code is changed yet. Tests can verify the data structures.

**Step 2 — Implement the registry.**
Create `core/registry.py` with the full `ObjectRegistry` class. Attach it to the `Session` object. No existing code changes — the registry just exists.

**Step 3 — Wrap the highest-value generators.**
Modify `/beat` and `/loop` to create and register objects as a side effect. Audio output behaviour is unchanged. CLI feedback adds the object name: `"Generated: beat_001 (BeatPattern, 4 bars, hip_hop)"`.

**Step 4 — Wire the Inspector window.**
The object tree in the Inspector window subscribes to registry events. From this point on, generated objects appear visually in real time. This is the first user-visible change.

**Step 5 — Progressively migrate all generators.**
`/gen2`, `/mel`, `/cor`, `/xform`, `/adapt` and others are updated to produce objects. This happens window-by-window as the GUI refactor progresses.

**Step 6 — Formalise AudioClip tracking.**
Raw buffer operations are wrapped to produce `AudioClip` objects. This is the most widespread change and is last because it touches the most code paths.

**Step 7 — Introduce templates.**
Once object types are stable, template definitions are added and the template filler system is implemented in CLI and GUI.

-----

## Cross-References

- **GUI Architecture:** See `docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md` for the window system that consumes these objects.
- **Current Song Registry:** See `mdma_rebuild/core/song_registry.py` for the existing registry pattern this builds on.
- **Preset Banks:** See `mdma_rebuild/core/banks.py` for existing preset storage that will migrate to `Patch` objects.
- **DSP Effects:** See `mdma_rebuild/dsp/effects.py` for the 113+ effects that `EffectChain` objects will wrap.
- **Generation Commands:** See `mdma_rebuild/commands/gen_cmds.py` for the generators that will produce first-class objects.
