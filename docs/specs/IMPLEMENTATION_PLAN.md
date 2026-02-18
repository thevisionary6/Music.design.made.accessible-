# MDMA Modular GUI & Object Model — Implementation Plan

**Version:** 1.0
**Status:** Phases 1-8 Complete
**Depends on:** GUI_WINDOW_ARCHITECTURE_SPEC.md, OBJECT_MODEL_SPEC.md

-----

## Summary

This document tracks the phase-by-phase implementation of the MDMA modular GUI
architecture and first-class object model, with comprehensive bug sweeps
between each phase.

-----

## Phase 1 — Foundation (COMPLETE)

**Goal:** Stand up the Bridge, event system, and shell as working skeletons.

**Deliverables:**
- [x] `gui/bridge.py` — Session/DSP adapter with real command dispatch
- [x] `gui/events.py` — 10 custom wx event types with headless stubs
- [x] `gui/shell.py` — Top-level frame with menu bar, status bar, window manager
- [x] Bridge `execute_command()` implements full command table dispatch
  (mirrors `mdma_gui.CommandExecutor.execute` via `bmdma.build_command_table()`)
- [x] Bridge uses session's `ObjectRegistry` (shared CLI/GUI state)
- [x] 622 commands available through Bridge dispatch

**Bug Sweep 1 — PASSED:**
- Bridge imports and initialises cleanly
- Command table loads 622 commands
- `/beat`, `/gen2`, `/obj` execute correctly through Bridge
- Unknown commands return proper error messages
- Registry events fire on object creation

-----

## Phase 2 — Object Registry Completion (COMPLETE)

**Goal:** Complete registry CRUD, add export/import, template CLI commands.

**Deliverables:**
- [x] `registry.export(id, path)` — JSON + companion WAV for AudioClips
- [x] `registry.import_file(path)` — Reconstruct objects from exported files
- [x] `/obj export <name> <path>` — CLI export command
- [x] `/obj import <path>` — CLI import command
- [x] `/template save <name> <type> [desc]` — Save template with auto-fields
- [x] `/template list` — List all templates
- [x] `/template show <name>` — Show template fields
- [x] `/template fill <name> k=v ...` — Fill template and generate object
- [x] Template registration in `bmdma.py` command table (COMMAND_OWNERS)

**Bug Sweep 2 — PASSED:**
- Export creates JSON file, re-imports successfully
- Templates save with correct default fields per target type
- Template fill dispatches correct generator command
- Registry count tracks all object types accurately

-----

## Phase 3 — Generation Window (COMPLETE)

**Goal:** Extract beat/melody/loop generation into modular panels.

**Deliverables:**
- [x] `gui/panels/generation/beat_generator.py` — BeatGeneratorPanel
  - Genre dropdown (11 genres), bars spinner (1-32), name field
  - Quick buttons for 4/8 bar generation
- [x] `gui/panels/generation/melody_harmony.py` — MelodyHarmonyPanel
  - Content type selector (melody/chords/bassline/arp/drone)
  - Dynamic parameter controls per type
  - Scale/progression/chord dropdowns, note count/bars/octave spinners
- [x] `gui/panels/generation/loop_generator.py` — LoopGeneratorPanel
  - Genre dropdown, bars spinner, layer checkboxes
- [x] `gui/panels/generation/generative_theory.py` — GenerativeTheoryPanel
  - Root note + scale selectors, theory query buttons, info display
- [x] `gui/windows/generation_window.py` — Notebook with all 4 panels
- [x] Shell window manager wired to GenerationWindow

**Bug Sweep 3 — PASSED:**
- All 5 files pass syntax check
- Panel constants match engine capabilities
- Bridge dispatches /beat, /gen2, /loop, /theory correctly
- Registry captures BeatPattern, Pattern, Loop, AudioClip objects

-----

## Phase 4 — Effects Window (COMPLETE)

**Goal:** Extract effect browser, chain builder, convolution into modular panels.

**Deliverables:**
- [x] `gui/panels/effects/effect_browser.py` — EffectBrowserPanel
  - 10 categories, 68+ effects, search field, tree view
- [x] `gui/panels/effects/chain_builder.py` — ChainBuilderPanel
  - Ordered effect list, add/remove/reorder, save as EffectChain object
- [x] `gui/panels/effects/convolution_panel.py` — ConvolutionPanel
  - IR preset selector, neural enhance modes, semantic transform
- [x] `gui/panels/effects/param_inspector.py` — ParamInspectorPanel
  - Effect parameter display, info queries
- [x] `gui/windows/effects_window.py` — Notebook with all 4 panels

**Bug Sweep 4 — PASSED:**
- All files pass syntax check
- Effect categories and IR presets import correctly
- /fx reverb_small applies successfully through Bridge
- /hfx returns effect info

-----

## Phase 5 — Synthesis Window (COMPLETE)

**Goal:** Extract FM synthesis and sound design panels from monolith.

**Deliverables:**
- [x] `gui/panels/synthesis/operator_panel.py` — FM operator grid
  - Operator index selector (0-7), waveform dropdown (22 types)
  - Ratio, level, feedback controls, carrier/modulator toggle
- [x] `gui/panels/synthesis/waveform_panel.py` — Waveform selector + preview
- [x] `gui/panels/synthesis/envelope_panel.py` — ADSR controls
- [x] `gui/panels/synthesis/modulation_panel.py` — LFO assignment
- [x] `gui/panels/synthesis/physical_modeling.py` — Physical modeling params
- [x] `gui/panels/synthesis/preset_browser.py` — Preset load/save/browse
- [x] `gui/windows/synthesis_window.py` — Notebook with all 6 panels

-----

## Phase 6 — Mutation Window (COMPLETE)

**Goal:** Extract transform/adapt/pattern editing into modular panels.

**Deliverables:**
- [x] `gui/panels/mutation/transform_panel.py` — TransformPanel
  - Transform type dropdown, amount controls, non-destructive output
- [x] `gui/panels/mutation/adapt_panel.py` — AdaptPanel
  - Key/tempo/style/develop subcommands with dynamic controls
- [x] `gui/panels/mutation/pattern_editor.py` — PatternEditorPanel
  - Step-level pattern editing (text-based grid view)
- [x] `gui/panels/mutation/splice_combine.py` — SpliceCombinePanel
  - Merge/splice two objects, operation type selector
- [x] `gui/windows/mutation_window.py` — Notebook with all 4 panels

-----

## Phase 7 — Arrangement & Mixing Windows (COMPLETE)

**Goal:** Build track arrangement and DJ/mixing windows.

**Deliverables:**
- [x] `gui/panels/arrangement/track_list.py` — TrackListPanel
- [x] `gui/panels/arrangement/pattern_lane.py` — PatternLanePanel
- [x] `gui/panels/arrangement/song_settings.py` — SongSettingsPanel
- [x] `gui/windows/arrangement_window.py` — Notebook with 3 panels
- [x] `gui/panels/mixing/deck_panel.py` — DeckPanel
- [x] `gui/panels/mixing/crossfader_panel.py` — CrossfaderPanel
- [x] `gui/panels/mixing/master_channel.py` — MasterChannelPanel
- [x] `gui/panels/mixing/stem_panel.py` — StemPanel
- [x] `gui/windows/mixing_window.py` — Notebook with 4 panels

-----

## Phase 8 — Inspector Extraction & Monolith Retirement Prep (COMPLETE)

**Goal:** Extract Inspector into proper panel modules, wire template system.

**Deliverables:**
- [x] `gui/panels/inspector/object_tree.py` — ObjectTreePanel
  - Hierarchical registry browser with type grouping
  - Refresh/Inspect/Delete buttons
  - Tree item data stores object ID for selection
- [x] `gui/panels/inspector/console_panel.py` — ConsolePanel
  - CLI mirror with direct command input
  - append_text() method for programmatic output
- [x] `gui/panels/inspector/parameter_inspector.py` — ParameterInspectorPanel
  - Detailed object view, rename/duplicate/tag actions
- [x] `gui/windows/inspector_window.py` — Split layout window
  - Left: tree + detail (vertical split)
  - Right: console mirror
  - Tree selection drives parameter inspector

**Bug Sweep 5-8 — PASSED:**
- 54 Python files in gui/, all valid syntax
- 27/27 panel classes import successfully (headless)
- 7/7 window classes import successfully
- Full Bridge integration: beat/melody/loop/fx/obj/template commands work
- Registry tracks all object types correctly

-----

## Implementation Statistics

| Category | Count |
|----------|-------|
| Total gui/ Python files | 54 |
| Panel modules | 27 |
| Window modules | 7 |
| Widget modules | 5 |
| Core infrastructure (bridge, events, shell) | 3 |
| Package __init__.py | 12 |
| CLI commands added | /obj (10 subcommands), /template (4 subcommands) |
| Registry methods added | export(), import_file() |
| Bridge commands available | 622 |

-----

## Architecture Compliance

| Spec Requirement | Status |
|------------------|--------|
| 7 modular sub-windows | COMPLETE — all 7 implemented |
| 25+ panel modules | COMPLETE — 27 panels |
| Bridge as single adapter | COMPLETE — full command dispatch |
| Object Registry integration | COMPLETE — shared CLI/GUI state |
| Event system | COMPLETE — 10 wx event types |
| Non-destructive by default | COMPLETE — new objects on transform |
| CLI/GUI parity | COMPLETE — all GUI actions are CLI commands |
| Template system | COMPLETE — save/list/show/fill |
| Export/Import | COMPLETE — JSON + WAV |
| Accessibility (name= attrs) | COMPLETE — all controls named |
| Shell window manager | COMPLETE — dynamic window open/close |
| wx headless stubs | COMPLETE — all modules importable without wx |

-----

## Remaining Work (Future Phases)

1. **Object Drop Zone** — Drag-and-drop area in Arrangement window
2. **Live parameter sliders** — Replace text info display with actual sliders
3. **Project persistence** — Full project save/load via registry serialisation
4. **Built-in template library** — Ship default templates for common workflows
5. **Monolith retirement** — Gradually deprecate `mdma_gui.py` as new GUI matures
6. **Thread safety** — Worker thread pool for long DSP operations
7. **Full step grid editing** — Painted step grid with velocity visualization

-----

## Cross-References

- **GUI Architecture Spec:** `docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md`
- **Object Model Spec:** `docs/specs/OBJECT_MODEL_SPEC.md`
- **Current GUI:** `mdma_gui.py` (8,859 lines — to be retired)
- **CLI Launcher:** `bmdma.py`
