# MDMA GUI Window Architecture Specification

**Version:** 1.0
**Status:** Planning
**Depends on:** OBJECT_MODEL_SPEC.md
**Author:** MDMA Development
**Supersedes:** GUI_SPEC_v0.2.md (Action Panel Client MVP)

-----

## Overview

This document specifies the modular sub-window architecture for the MDMA GUI. The design replaces the current monolithic action panel with a set of focused, workflow-sorted sub-windows — each representing a distinct creative mode. This mirrors the FL Studio dockable window model but organises windows by **workflow category** rather than function, reducing cognitive load and improving debuggability.

Each window is independently spawnable, closeable, and internally self-contained. Windows communicate exclusively through the **Bridge layer** and the **Object Registry** — never directly with each other.

### Relationship to Existing Architecture

The current `mdma_gui.py` monolith (~8,859 lines) implements a two-pane action panel client as specified in `GUI_SPEC_v0.2.md`. This architecture replaces that single-file approach with a modular system where:

- `mdma_gui.py` is progressively deprecated in favour of `gui/shell.py`
- The existing `Session` object remains the authoritative state holder
- All DSP and command dispatch continues through the existing `mdma_rebuild/` package
- The GUI remains a **client** — it never invents new execution semantics

-----

## Design Principles

**Workflow separation over function grouping.** A user in generation mode should only see generation controls. They should not be distracted by synthesis or effects controls they are not currently using.

**Creation, transformation, and processing are distinct modes.** Generating something new, mutating something that exists, and applying signal processing to audio are three fundamentally different cognitive tasks. They belong in separate windows.

**All windows operate on first-class objects.** Windows do not work with raw buffers or ephemeral state. They consume objects from the registry and produce objects back into it. See OBJECT_MODEL_SPEC.md for the full object taxonomy.

**Accessibility is structural, not cosmetic.** Screen reader compatibility is built into the widget layer and panel structure — not added as an afterthought. Every interactive element in every window must be operable without a mouse.

**Windows are independently debuggable.** A bug in the Mutation window does not require scanning the Generation window. Each window's panel class owns only its own concerns.

-----

## Window Inventory

### 1. Generation Window

**Purpose:** Create new first-class objects from scratch. This window is strictly creative — it produces objects but does not modify existing ones.

**Objects produced:** `Pattern`, `AudioClip`, `BeatPattern`, `Loop`

**Panels within this window:**

*Beat Generator Panel* — Wraps `/beat` command. Genre selector, bar count, variation controls. Output always produces a named `BeatPattern` object registered in the object tree.

*Melody & Harmony Panel* — Wraps `/gen2` for melody, chords, bassline, arpeggio, drone. Scale selector, length, root note, density controls. Produces `Pattern` objects.

*Loop Generator Panel* — Wraps `/loop`. Genre, bar count, layer toggles (drums/bass/chords/melody). Produces `Loop` objects containing linked sub-patterns.

*Generative Theory Panel* — Music theory tools: scale explorer, chord progression builder, voice leading suggestions. Produces `Pattern` objects from theory-first input.

**Strict constraints:**

- No transform, mutation, or effects controls anywhere in this window.
- No direct buffer editing.
- Every "Generate" action must produce a named object in the registry before any audio is played.

-----

### 2. Mutation & Editing Window

**Purpose:** Transform existing objects. Takes objects from the registry as input, produces new or modified objects as output. This is where creative variation and algorithmic editing happen.

**Objects consumed:** `Pattern`, `AudioClip`, `BeatPattern`, `Loop`
**Objects produced:** New or modified versions of the above (non-destructive by default — original preserved unless user explicitly overwrites)

**Panels within this window:**

*Transform Panel* — Wraps `/xform`. Retrograde, inversion, stutter, time-stretch, pitch-shift, rhythmic augmentation/diminution. Object selector input, transform selector, parameter controls, output naming field.

*Adapt Panel* — Wraps `/adapt`. Key change, scale change, tempo change, style reinterpretation. Takes a source object and produces an adapted variant.

*Pattern Editor Panel* — Direct step-level editing of `Pattern` objects. Grid view of note/beat steps, drag-to-adjust velocity, step toggle. Keyboard navigable for screen reader use.

*Splice & Combine Panel* — Merge or splice two objects together. Stitch patterns end-to-end, layer them, or interleave. Produces a new composite object.

**Non-destructive by default:** All operations produce a new object unless the user explicitly selects "overwrite." Original objects are preserved in the registry.

-----

### 3. Effects Window

**Purpose:** Apply signal processing to audio objects. This window is strictly for DSP — it does not generate or mutate musical content.

**Objects consumed:** `AudioClip`, `Pattern` (rendered to audio for processing)
**Objects produced:** Processed `AudioClip`, named `EffectChain`

**Panels within this window:**

*Effect Browser Panel* — Browsable, categorised list of all 113+ effects. Categories: Spatial (reverb, delay), Dynamics (compress, gate, limit), Tone (filters, EQ, saturate), Texture (granular, chorus, flanger, phaser), Damage (bitcrush, distort, decimate), Utility (normalize, trim, fade). Search field for fast access.

*Effect Chain Builder Panel* — Visual ordered list of effects applied in sequence. Add/remove/reorder effects. Each effect slot shows its parameters inline. Save the current chain as a named `EffectChain` object.

*Convolution Panel* — Dedicated panel for the convolution reverb engine. IR preset browser (17 presets), IR import, neural enhancement controls (irenhance modes), semantic IR transform (irtransform descriptors).

*Parameter Inspector Panel* — When an effect slot is selected in the chain builder, this panel shows its full parameter set. Unified 1-100 scale. LFO assignment per parameter.

**Effect chains as objects:** Any chain built in this window can be saved as a named `EffectChain` object, recalled by name, and applied to any compatible object later via CLI or GUI.

-----

### 4. Synthesis Window

**Purpose:** Sound design. Create and edit synthesizer patches. This window is entirely about defining *how something sounds*, not *what notes it plays*.

**Objects produced:** `Patch`
**Objects consumed:** `Patch` (for editing)

**Panels within this window:**

*Operator Panel* — FM synthesis operator grid. Carrier/modulator assignment, operator ratios, levels, feedback. Visual operator graph (text-representable for screen readers).

*Waveform Panel* — Waveform type selector across all 22 types. Preview waveform display. Wavetable position control for wavetable mode.

*Envelope Panel* — ADSR controls per operator or globally. Visual envelope shape. Named envelope presets.

*Modulation Panel* — LFO assignment, rate, depth, target parameter. Macro controls. Modulation matrix overview.

*Physical Modeling Panel* — Physical modeling parameters when relevant synthesis mode is active.

*Preset Browser Panel* — Browse, load, save, and name `Patch` presets. Tag-based filtering. Favourites.

-----

### 5. Arrangement Window

**Purpose:** Assemble objects into tracks and song structure. This is where patterns, clips, and beats become a piece of music.

**Objects consumed:** `Pattern`, `AudioClip`, `BeatPattern`, `Loop`, `Patch`
**Objects produced:** `Track`, `Song`

**Panels within this window:**

*Track List Panel* — Ordered list of tracks. Each track has a name, type (audio/MIDI/beat), assigned patch (for MIDI tracks), mute/solo toggles, volume and pan.

*Pattern Lane Panel* — Per-track view of which patterns/clips are placed at which positions. Keyboard navigable. Each slot shows the object name.

*Song Settings Panel* — BPM, time signature, key, total length, output format settings.

*Object Drop Zone* — Drag-or-assign area where objects from the registry can be placed onto tracks. Screen reader users can use a command-style input: "place [object name] on [track] at [position]."

-----

### 6. Mixing & DJ Window

**Purpose:** Output mixing, deck-based DJ operations, mastering. Distinct from arrangement — this is real-time performance and final output shaping.

**Panels within this window:**

*Deck Panel (xN)* — Per-deck controls: load track, play/stop, BPM, pitch, loop points, deck-specific effects.

*Crossfader Panel* — Visual crossfader between decks. Keyboard operable.

*Master Channel Panel* — Volume, master effects chain, output format, HQ mode toggle, render-to-file controls.

*Stem Panel* — Stem separation controls when stem mode is active.

-----

### 7. Inspector & Console Window

**Purpose:** Persistent accessibility anchor. Always available regardless of what other windows are open. This is the primary orientation point for screen reader users.

**Panels within this window:**

*Object Tree Panel* — Full hierarchical view of the object registry. Browse all objects by type or by project. Select any object to inspect or send to another window for editing. Keyboard navigable with full screen reader labels.

*Parameter Inspector Panel* — Shows the full detail of whatever object or parameter is currently selected anywhere in the application.

*Console/CLI Mirror Panel* — Live mirror of CLI output. All actions taken in any GUI window produce the same text feedback as the CLI equivalent. Users can also type CLI commands directly here.

*Status Bar* — Last action description, current selection, active BPM, current key/scale. Updated on every action. Screen readers can poll this for orientation.

-----

## Window Communication Model

Windows never call each other directly. All inter-window communication flows through two shared systems:

**The Bridge** (`gui/bridge.py`) — A single adapter object that all windows call for session/DSP operations. The bridge translates GUI actions into engine calls and publishes results as events.

**The Object Registry** (`core/registry.py`) — The shared object store. Windows read from and write to the registry. When one window creates a new `Pattern`, the object tree in the Inspector window updates automatically because both subscribe to registry change events.

**Event flow:**

```
User action in Window A
    -> Window A calls Bridge method
        -> Bridge calls DSP/session
            -> Bridge creates/updates Object in Registry
                -> Registry fires RegistryChangeEvent
                    -> All subscribed windows update their views
```

No window holds a reference to another window. No window holds audio data directly — all data lives in objects in the registry.

-----

## Accessibility Requirements Per Window

Every window must satisfy the following:

**Keyboard navigation** — Full tab order through all controls. No action requires a mouse. Arrow keys navigate within lists and grids.

**Screen reader labels** — Every control has a meaningful `wx.Accessible` name and description. Labels are not just icon tooltips — they are full descriptive strings.

**Status announcements** — Every action that changes state produces a screen-reader-readable status string, surfaced in the Inspector console and as an accessible live region.

**No visual-only information** — No information is conveyed by colour, position, or visual shape alone. All states are also expressed as text.

**Consistent control patterns** — The same type of action (e.g., selecting an object, setting a parameter value) uses the same keyboard interaction in every window.

-----

## File Structure

```
gui/
  __init__.py                 # Package init
  shell.py                    # Top-level wx.Frame, window manager, menu bar
  bridge.py                   # Session/DSP adapter — all engine calls route here
  events.py                   # Custom wx event types
  windows/
    __init__.py
    generation_window.py
    mutation_window.py
    effects_window.py
    synthesis_window.py
    arrangement_window.py
    mixing_window.py
    inspector_window.py
  panels/
    __init__.py
    generation/
      __init__.py
      beat_generator.py
      melody_harmony.py
      loop_generator.py
      generative_theory.py
    mutation/
      __init__.py
      transform_panel.py
      adapt_panel.py
      pattern_editor.py
      splice_combine.py
    effects/
      __init__.py
      effect_browser.py
      chain_builder.py
      convolution_panel.py
      param_inspector.py
    synthesis/
      __init__.py
      operator_panel.py
      waveform_panel.py
      envelope_panel.py
      modulation_panel.py
      physical_modeling.py
      preset_browser.py
    arrangement/
      __init__.py
      track_list.py
      pattern_lane.py
      song_settings.py
    mixing/
      __init__.py
      deck_panel.py
      crossfader_panel.py
      master_channel.py
      stem_panel.py
    inspector/
      __init__.py
      object_tree.py
      parameter_inspector.py
      console_panel.py
  widgets/
    __init__.py
    param_slider.py           # Accessible 1-100 parameter control
    object_selector.py        # Registry-aware object picker
    labeled_control.py        # Accessible label+control pair
    step_grid.py              # Keyboard-navigable step grid widget
    waveform_view.py          # Text-representable waveform display
```

-----

## Implementation Phases

**Phase 1 — Foundation (no existing code broken)**
Set up `gui/bridge.py`, `gui/events.py`, and `gui/shell.py` as empty skeletons. Establish the event system. Stand up the Inspector window with the console mirror only — this immediately gives a debuggable, always-available anchor window that works alongside the existing monolithic GUI.

**Phase 2 — Object Registry integration**
Implement the object registry (see OBJECT_MODEL_SPEC.md). Wire the Inspector's object tree to it. From this point on, new objects appear in the tree automatically.

**Phase 3 — Extract Generation Window**
Extract beat gen, loop gen, and melody gen panels from the monolith into the Generation window. Wire them through the bridge. Verify they produce registry objects.

**Phase 4 — Extract Effects Window**
Extract the effect chain builder and convolution panel. This is self-contained enough to extract cleanly without touching synthesis or arrangement concerns.

**Phase 5 — Extract Synthesis Window**
Extract the Monolith/FM operator panels. These are the most internally complex but also fairly self-contained once the bridge handles session calls.

**Phase 6 — Extract Mutation Window**
Extract transform and adapt panels. By this phase the registry and event system are mature, so non-destructive operation should be straightforward to implement.

**Phase 7 — Extract Arrangement and Mixing Windows**
These are last because they depend on objects produced by all other windows being stable and well-formed.

**Phase 8 — Retire the monolith**
Once all panels are extracted and verified, `mdma_gui.py` is deprecated. The shell becomes the new entry point.

-----

## Cross-References

- **Object Model:** See `docs/specs/OBJECT_MODEL_SPEC.md` for the full object taxonomy, registry API, and persistence format.
- **Current GUI Spec:** See `GUI_SPEC_v0.2.md` (root) for the prior action-panel architecture this spec supersedes.
- **Interface Transition:** See `INTERFACE_TRANSITION_SPEC.md` for the CLI-as-authority design principle this architecture preserves.
- **Roadmap:** See `ROADMAP_FULL_RELEASE.md` for the overall v1.0 phase plan this work falls within.
