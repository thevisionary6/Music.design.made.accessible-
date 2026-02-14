# MDMA Full Release Roadmap

**Music Design Made Accessible**
**Document Version:** 1.5
**Baseline:** v52.0 (2026-02-03)
**Last Updated:** 2026-02-13 (Phase T: Full System Audit added between Phase 4b and Phase 5)
**Target:** v1.0 Full Release

---

## How to Read This Roadmap

Each phase is a self-contained milestone that delivers usable value on its own.
Phases are ordered by dependency — later phases build on earlier ones.
Within each phase, features are listed by priority (highest first).

**Status Key:**
- **[DONE]** — Implemented and tested
- **[EXISTS]** — Already implemented (may need refinement)
- **[PARTIAL]** — Foundation exists, significant work remains
- **[NEW]** — Not yet started

---

## Phase 1: Core Interface & Workflow -- COMPLETE

> **Goal:** Make the GUI a real production surface, not just a command launcher.
> **Depends on:** v52 GUI MVP
> **Delivers:** A navigable, visual workspace for all MDMA objects.
> **Completed:** 2026-02-11

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 1.1 | Object tree view | **[DONE]** | 10-category hierarchical tree: Engine, Synthesizer, Filter, Envelope, Tracks, Buffers, Decks, Effects, Presets, Banks |
| 1.2 | Dynamic object list | **[DONE]** | Rich tree data with inline metadata; auto-refresh timer (500ms); full rebuild after every command |
| 1.3 | Preview views | **[DONE]** | Inline previews: `Track 1: Drums [2.50s, stereo, peak: -3.2dB] [SOLO]` for all object types |
| 1.4 | Expanded detail views | **[DONE]** | InspectorPanel with property grid, contextual action buttons, tabbed alongside ActionPanel |
| 1.5 | Step grid highlighting | **[DONE]** | StepGridPanel with beat grid, playhead/write-pos indicators, 16/32/64/128 steps, color legend |
| 1.6 | Context menus | **[DONE]** | Right-click on tracks, buffers, decks, effects, operators, presets, chains, categories |
| 1.7 | Selection-based FX | **[DONE]** | FX picker dialog targeting specific objects (track/buffer/deck/working/global) |
| 1.8 | Accessibility markers | **[DONE]** | Deck section markers (intro/body/outro) as navigable child nodes with timestamps |

### Milestone Deliverable
A GUI where users can browse every object in their session, inspect its state, right-click to act on it, and see step-by-step playback position — all keyboard-navigable and screen-reader compatible.

---

## Phase 2: Monolith Engine & Synthesis Expansion -- COMPLETE

> **Goal:** Elevate the Monolith synth from functional to professional-grade.
> **Depends on:** Phase 1 (GUI views to expose new parameters)
> **Delivers:** A deep, visual synth engine with extended waveform capabilities.
> **Completed:** 2026-02-11

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 2.1 | Full Monolith patch builder view | **[DONE]** | PatchBuilderPanel: operator list + routing list + inline param editing + wave type picker + quick action buttons |
| 2.2 | Carrier/modulator GUI | **[DONE]** | RoutingPanel: visual signal flow diagram with operator boxes, bezier-curve routing arrows, color-coded by modulation type |
| 2.3 | Full parameter exposure | **[DONE]** | All 40+ Monolith parameters exposed via /wm key=value syntax and GUI widgets. Param map covers supersaw, additive, formant, harmonic, waveguide, wavetable, compound |
| 2.4 | Oscillator list view | **[DONE]** | OscillatorListPanel: scrollable card-based browser with category filter (Basic/Noise/Physical/Extended/Waveguide/Wavetable/Compound), per-op select button |
| 2.5 | Extended wave models | **[DONE]** | Added supersaw (JP-8000 style, 3-11 detuned saws), additive (harmonic rolloff), formant (vowel-shaped oscillator a/e/i/o/u) |
| 2.6 | Physical modeling waveforms | **[DONE]** | 4 waveguide models: Karplus-Strong string (damping/brightness/position), tube/pipe (reflection/bore), membrane/drum (tension/strike), plate/bar (thickness/material) |
| 2.7 | Odd/even harmonic simulation | **[DONE]** | Harmonic wave type with independent odd_level, even_level, odd_decay, even_decay. Clarinet-like (odd only) to organ-like (even emphasis) |
| 2.8 | Wavetable import support | **[DONE]** | /wt load command imports Serum-format .wav wavetables. Frame interpolation, per-frame normalization, frame position 0.0-1.0. Engine stores named wavetables |
| 2.9 | Compound wave creation | **[DONE]** | /compound new/add/use/morph commands. Layer multiple wave types with detune/amp/phase per layer. Morph mode crossfades between 2 layers |

### Milestone Deliverable
A fully visual synth-design experience — users can build patches from scratch, import wavetables, use physical models, and hear the results immediately.

### Phase 2 Implementation Notes

**New wave types added to `monolith.py`:**
- `supersaw` (ssaw): 7 detuned saws, JP-8000 style, params: num_saws, detune_spread, mix
- `additive` (add): Harmonic rolloff synthesis, params: num_harmonics, rolloff
- `formant` (vowel): Vocal formant oscillator, params: vowel (a/e/i/o/u)
- `harmonic` (harm): Independent odd/even control, params: odd_level, even_level, odd_decay, even_decay
- `waveguide_string` (string/pluck): Karplus-Strong, params: damping, brightness, position
- `waveguide_tube` (tube/pipe): Waveguide tube, params: damping, reflection, bore_shape
- `waveguide_membrane` (membrane/drum): Drum model, params: tension, damping, strike_pos
- `waveguide_plate` (plate/bar): Vibraphone/marimba, params: thickness, damping, material
- `wavetable` (wt): Imported wavetable playback, params: wavetable_name, frame_pos
- `compound` (comp/layer): Multi-wave layer/morph, params: compound_name, morph, layers

**New commands:** /ssaw, /harm, /wg, /wt, /compound, /waveinfo
**GUI panels:** PatchBuilderPanel, RoutingPanel, OscillatorListPanel (3 new tabs)
**Total wave types:** 22 (up from 8)

---

## Phase 3: Modulation, Impulse & Convolution -- COMPLETE

> **Goal:** Professional-grade modulation sources and convolution processing.
> **Depends on:** Phase 2 (synth engine to modulate)
> **Delivers:** Deep modulation routing and studio-quality convolution reverb.
> **Completed:** 2026-02-12

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 3.1 | Pulse/impulse importing for LFOs | **[DONE]** | Import audio files as LFO waveshapes — any waveform becomes a modulation source. `/impulselfo load/file/apply/filter/clear` |
| 3.2 | Impulse importing for envelopes | **[DONE]** | Import impulse recordings as envelope shapes — transient captures become amplitude contours. `/impenv load/file/apply/operator` |
| 3.3 | Advanced convolution reverb engine | **[DONE]** | ConvolutionEngine class with early/late reflection split, stereo width (Haas decorrelation), IR bank, 17 presets. `/conv load/preset/apply/params/split/save` |
| 3.4 | Neural-enhanced convolution | **[DONE]** | Spectral analysis + resynthesis for IR extension (decay curve extrapolation), spectral gating denoiser, gap-fill interpolation. `/irenhance extend/denoise/fill` |
| 3.5 | AI-assisted impulse transformation | **[DONE]** | 15 semantic descriptors (bigger/darker/metallic/cathedral/ethereal/etc.) with spectral reshaping, resonance modeling, and chain support. `/irtransform <descriptor> [intensity]` |
| 3.6 | Granular impulse tools | **[DONE]** | IR-specific granular workflow: stretch, morph (interleaved grains from 2 IRs), redesign (granular decomposition/resynthesis), freeze. `/irgranular stretch/morph/redesign/freeze` |

### Milestone Deliverable
A modulation and convolution system where any audio can become a modulation source, convolution reverbs sound studio-grade, and AI can intelligently reshape acoustic spaces.

### Phase 3 Implementation Notes

**New DSP module:** `mdma_rebuild/dsp/convolution.py`
- `impulse_to_lfo()`: Resamples impulse audio to single-cycle waveshape for table lookup LFO
- `lfo_from_waveshape()`: Phase-accumulator LFO generator from arbitrary waveshape
- `impulse_to_envelope()`: Envelope follower (attack/release smoothing) → amplitude contour
- `ConvolutionEngine`: Advanced reverb with early/late split (configurable split point), Haas-effect stereo width, per-band EQ on IR, IR bank management
- `ir_extend()`: Analyses tail decay rate + spectral shape, synthesises matching extension with crossfade
- `ir_denoise()`: STFT-based spectral gating using noise floor estimation from tail
- `ir_fill_gaps()`: Detects envelope dips below threshold, interpolates across gaps
- `ir_transform()`: 15 descriptor mappings → spectral modifications (boost/cut/resonance/saturation/shift/reverse)
- `granular_ir_stretch/morph/redesign()`: Wraps GranularEngine for IR-specific workflows

**New commands module:** `mdma_rebuild/commands/convolution_cmds.py`
- `/impulselfo` (aliases: `/ilfo`, `/lfoimport`): Import impulse as LFO + apply to operators/filter
- `/impenv` (aliases: `/ienv`, `/envimport`): Import impulse as envelope + apply to buffer/operator
- `/conv` (aliases: `/convolution`, `/convrev`): Full convolution reverb management
- `/irenhance` (aliases: `/ire`, `/enhance`): Neural-inspired IR processing
- `/irtransform` (aliases: `/irt`, `/transform`): Semantic descriptor transforms
- `/irgranular` (aliases: `/irg`, `/irgrains`): Granular IR tools

**GUI additions (mdma_gui.py v3.0.0):**
- 6 new ACTIONS categories: impulse_lfo, impulse_env, convolution, ir_enhance, ir_transform, ir_granular
- 6 new ObjectBrowser categories with live state display
- InspectorPanel methods for convolution, IR bank, LFO shapes, impulse envelopes
- Category summaries for all Phase 3 objects

**Total new commands:** 18 (6 commands × 3 aliases each)
**Total IR presets:** 17 (hall×4, room×3, plate×3, spring×3, shimmer×2, reverse×2)
**Total descriptors:** 15 (bigger/smaller/brighter/darker/warmer/metallic/wooden/glass/cathedral/intimate/ethereal/haunted/telephone/underwater/vintage)

### Phase 3 Readiness Notes (system audit 2026-02-12)

**Existing foundation:**
- `audiorate_cmds.py` already has `/audiorate interval`, `/audiorate filter`, `/audiorate pattern`, `/audiorate clear` — audio-rate LFO for both interval and filter modulation
- `UmpulseBank` class in audiorate_cmds.py has `/ump load`, `/ump buffer`, `/ump use wave|ir`, `/ump wavetable` — impulse import pipeline exists
- `MonolithEngine` has `set_interval_mod()`, `set_interval_lfo()`, `set_filter_mod()`, `set_op_filter_mod()` — engine hooks are ready
- Enhanced chunking (`/chke`) supports 9 algorithms: auto, transient, beat, zero, equal, wavetable, energy, spectral, syllable
- Granular engine exists in the codebase (Phase 3.6 marked PARTIAL)

**Known gaps to resolve BEFORE Phase 3 implementation:**

**A. Critical GUI–Engine gaps (27 ability gaps found):**

| Priority | Gap | Engine | Commands | GUI |
|----------|-----|--------|----------|-----|
| HIGH | Preset Bank (0-127) | `save_preset/load_preset/list/delete` | `/preset` | MISSING — no save/load/delete controls |
| HIGH | Audio-Rate Interval Mod | `set_interval_mod/set_interval_lfo` | `/audiorate interval` | MISSING — zero coverage |
| HIGH | Audio-Rate Filter Mod | `set_filter_mod/set_op_filter_mod` | `/audiorate filter` | MISSING — zero coverage |
| HIGH | HQ Mode Controls | 12 params (dc, subsonic, saturation, etc.) | `/hq` | DISPLAY ONLY — no toggle/edit |
| HIGH | Filter Types (30) | 30 types in `_apply_filter` | `/ft` (all 30) | Only 6 in ACTIONS dropdown |
| HIGH | Filter Envelope (ADSR) | Per-slot filter ADSR | `/fatk /fdec /fsus /frel` | MISSING — no filter ADSR controls |
| HIGH | Algorithm Bank System | routing presets | `/bk /al` | PARTIAL — ObjectBrowser "Banks" empty |
| HIGH | Per-Operator Envelope | operator_envelopes dict | `/venv` + level-aware ADSR | Read-only in inspector |
| HIGH | Wavetable Management | load/list/delete methods | `/wt load/del/info` | DISPLAY ONLY — no load/delete |
| HIGH | Compound Wave Mgmt | create/list/delete methods | `/compound new/add/del` | DISPLAY ONLY — no create/delete |
| MEDIUM | Voice Algorithm | stack/unison/wide | `/va` | DISPLAY ONLY — no control |
| MEDIUM | Stereo/Phase Spread | render params | `/stereo /vphase` | MISSING |
| MEDIUM | Rand / VMod | voice variation params | `/rand /vmod` | MISSING |
| MEDIUM | Waveform dropdown | 18 wave types | `/wm` (all 18) | 5 of 18 in ACTIONS |
| MEDIUM | Umpulse System | UmpulseBank class | `/ump /impulse` | MISSING — no ObjectBrowser category |
| MEDIUM | Enhanced Chunking | 9 algorithms | `/chke` | MISSING |
| MEDIUM | Pattern Interval Mod | pattern-to-signal | `/audiorate pattern` | MISSING |
| MEDIUM | Modulation Envelope | mod ADSR | `/menv` | MISSING |
| MEDIUM | Filter Slot Count/Enable | 1-8 slots | `/fcount /fs /fe` | PARTIAL — no count/enable control |
| MEDIUM | Modulation Routing | 5 types | `/fm /tfm /am /rm /pm /rt` | PARTIAL — text entry only |
| MEDIUM | Wave Params in PatchBuilder | 40+ params | full `/wm key=val` | Missing many per-type params |
| LOW | Musical Key/Scale | session key | `/key` | MISSING |
| LOW | Noise / Note Sequence | engine | `/noise /ns` | MISSING |
| LOW | Physical Model Params | phys/phys2 | `/phys /phys2` | Not in PatchBuilder |
| LOW | Pulse Width | pw param | `/pw` | Display only |
| LOW | Clear Modulation | clear_modulation() | `/audiorate clear` | MISSING |
| LOW | OpInfo / WaveInfo | engine | `/opinfo /waveinfo` | Partial |

**B. Phase 3 specific blockers:**
- Feature 3.1 (LFO import): Needs `/ump` impulse system exposed in GUI. Currently zero GUI coverage of UmpulseBank.
- Feature 3.2 (Envelope import): Same dependency — UmpulseBank has the import pipeline, GUI needs management panel.
- Feature 3.3 (Convolution reverb): No convolution engine exists yet. Must build from scratch in dsp/.
- Feature 3.4/3.5 (Neural/AI convolution): Depends on 3.3. AI model serving infrastructure needed.
- Feature 3.6 (Granular impulse): Granular engine exists but needs IR-specific workflow wrapper.

**C. Recommended pre-Phase 3 work:**
1. Close the HIGH-priority GUI gaps — bring the GUI to parity with the CLI
2. Wire audio-rate modulation into GUI (prerequisite for 3.1/3.2)
3. Wire umpulse/impulse system into GUI (prerequisite for 3.1/3.2)
4. Add ObjectBrowser categories for: Wavetables, Compound Waves, Umpulses, Audio-Rate Config
5. Expand ACTIONS waveform dropdown from 5 → 18 types
6. Expand ACTIONS filter type dropdown from 6 → 30 types

---

## Phase 4: Generative Systems -- COMPLETE

> **Goal:** AI and algorithmic tools that produce musically useful output.
> **Depends on:** Phase 2 (synth engine for rendering), Phase 3 (modulation for expression)
> **Delivers:** Generation engines that produce patterns, loops, and beats ready for arrangement.
> **Completed:** 2026-02-12

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 4.1 | Improved content generation | **[DONE]** | Overhauled generation via `/gen2` — melody, chord progressions, bassline, arpeggio, drone with musical intelligence (voice leading, contour shaping, genre awareness) |
| 4.2 | Pattern adaptation algorithms | **[DONE]** | `/adapt` command — key/scale/tempo/style adaptation, motivic development chains, fundamental detection |
| 4.3 | Full loop generation engine | **[DONE]** | `/loop` command — multi-layer loop generation (drums, bass, chords, melody) with genre templates, mood modifiers, chord progressions |
| 4.4 | Drum beat generation engine | **[DONE]** | `/beat` command — 17 sound generators, 11 genre templates, humanization, fill generation (buildup/breakdown/roll/crash) |
| 4.5 | Transformation engines | **[DONE]** | `/xform` command — note-level (retrograde, inversion, augmentation, diminution, permute, motivic development) and audio-level (pitch shift, stutter, chop, granular freeze) with presets |

### Implementation Details
- **Music theory foundation** (`dsp/music_theory.py`): 21 scales, 21 chord types, 12 progressions, voice leading, key detection, melody generation
- **Beat generation** (`dsp/beat_gen.py`): 17 sound generators (kick, snare, hihat, tom, cymbal, clap, snap, shaker, stab, bass, bell, riser, downlifter, fx, silence, click, sweep), genre templates (house, techno, hiphop, trap, dnb, lofi, reggaeton, breakbeat, dubstep, afrobeat, minimal)
- **Loop generation** (`dsp/loop_gen.py`): Multi-layer synthesis with mood modifiers, chord progressions, voice-led pads
- **Transforms** (`dsp/transforms.py`): 10 note-level + 8 audio-level transforms, 5 note presets + 10 audio presets
- **Commands** (`commands/gen_cmds.py`): `/beat`, `/loop`, `/gen2`, `/xform`, `/adapt`, `/theory`
- **GUI**: Full Generative tree category with subcategories (Beat Generation, Loop Generation, Transforms, Music Theory, Content Generation), context menus, inspector panels, generation dialogs

### Milestone Deliverable
Users can say "generate a 4-bar drum loop in this style" or "adapt this melody to minor key" and get musically coherent, production-ready results. **ACHIEVED.**

---

## Phase 4b: Microtonal Support (Generative Systems Expansion)

> **Goal:** Extend Phase 4 generative systems and the music theory engine to support microtonal tuning systems beyond 12-TET.
> **Depends on:** Phase 4 (generative systems), Phase 2 (synth engine frequency pipeline)
> **Delivers:** Full microtonal composition workflow — custom tuning systems, microtonal scales, non-Western theory integration, and microtonal-aware generation.

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 4b.1 | Tuning system engine | **[NEW]** | Pluggable tuning backend that replaces 12-TET assumptions — support for equal divisions of the octave (EDO: 19, 22, 24, 31, 53, etc.), just intonation (rational frequency ratios), and custom cent-based tunings. Session-level `tuning` property with hot-swap support |
| 4b.2 | Microtonal frequency pipeline | **[NEW]** | Replace `midi_to_freq()` 12-TET formula with tuning-aware conversion — `tuning_to_freq(degree, tuning_system)`. All synthesis, pattern, and generative code routes through the new pipeline so tuning changes propagate everywhere |
| 4b.3 | Microtonal scale definitions | **[NEW]** | Extend `SCALES` dict to support non-12-TET interval sets — scales defined as cent offsets or frequency ratios instead of semitone integers. Built-in microtonal scales: quarter-tone Arabic maqamat, Turkish makam, Indonesian slendro/pelog, Indian shruti (22-tone), Bohlen-Pierce, Wendy Carlos Alpha/Beta/Gamma |
| 4b.4 | Microtonal notation and input | **[NEW]** | New note input syntax for microtonal pitches — cent offsets (`+50c`), EDO step numbers (`19edo:7`), ratio notation (`3/2`, `5/4`), and quarter-tone accidentals (`C4+`, `Eb4-`). Pattern commands (`/mel`, `/cor`, `/pat`) accept microtonal tokens |
| 4b.5 | Configurable reference frequency | **[NEW]** | Session-level A4 reference (default 440Hz) — `/ref 432`, `/ref 415` (baroque), `/ref 444` (orchestral). All pitch calculations respect the reference. Persistent across save/load |
| 4b.6 | Microtonal-aware generative systems | **[NEW]** | Extend `/gen2`, `/beat`, `/loop`, `/adapt`, `/xform` to generate content using the active tuning system — melody generation respects EDO step counts, chord voicings use tuning-correct intervals, adaptation algorithms handle non-12 pitch spaces |
| 4b.7 | Tuning table import/export | **[NEW]** | Import tuning definitions from standard formats — Scala `.scl`/`.kbm` files, TUN files, AnaMark tuning. Export MDMA tunings to `.scl` for use in other software |
| 4b.8 | Microtonal theory extensions | **[NEW]** | Extend music theory module — interval measurement in cents, consonance/dissonance ranking for arbitrary tunings, microtonal chord detection, key detection for non-12-TET pitch sets, MOS (Moment of Symmetry) scale generation |

### Implementation Notes

**Architectural approach:** The current `music_theory.py` uses semitone-integer offsets for all scales and chords. The microtonal extension redefines intervals as **cent values** (float) internally while maintaining backward compatibility — 12-TET semitone `1` = `100.0` cents. This means existing 12-TET scales and commands work unchanged.

**Key extension points identified in current codebase:**
- `music_theory.py:midi_to_freq()` — replace with tuning-aware lookup
- `music_theory.py:SCALES` — extend to accept cent-based intervals
- `music_theory.py:snap_to_scale()` — generalize for non-12 pitch class sets
- `monolith.py` generate functions — already accept arbitrary Hz, no changes needed
- `gen_cmds.py` — pass tuning context to all generation functions
- `session.py` — add `tuning_system` and `reference_freq` session properties

**Backward compatibility:** All existing 12-TET workflows remain the default. Microtonal features activate only when a non-standard tuning is loaded via `/tuning` command. No breaking changes.

### Milestone Deliverable
Users can load any tuning system (EDO, just intonation, or custom), compose with microtonal scales from diverse musical traditions, generate microtonal content with the full suite of generative tools, and import/export tuning definitions for interoperability with other music software.

---

## Phase T: Full System Audit — Patch All Gaps for Complete Song Production

> **Goal:** Close every gap that prevents a user from making a complete, polished song from start to finish using only MDMA. After Phase T, the product is song-ready — not demo-ready.
> **Depends on:** Phases 1-4 (all complete)
> **Delivers:** A rock-solid production workflow where a user can go from blank project → structured arrangement → mixed tracks → mastered render without hitting dead ends.
> **Blocking:** Phase T MUST complete before Phase 5. Every subsequent phase builds on a solid foundation, not on workarounds.

### Audit Summary

A full-system audit identified **7 critical gaps**, **9 high-priority fixes**, and **6 medium-priority improvements** that collectively prevent end-to-end song production. These fall into four categories:

1. **Workflow blockers** — missing song structure, undo, and arrangement features
2. **Broken wiring** — GUI actions referencing commands that don't exist
3. **Silent failures** — operations that fail without user feedback
4. **Missing plumbing** — features that exist in the engine but have no command interface

### T.1 — Undo/Redo System (CRITICAL)

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| T.1.1 | Working buffer undo stack | **[NEW]** | Before any destructive buffer operation (`/fx`, `/xform`, `/pitch`, `/stretch`, `/chop`, `/reverse`), push a copy of the working buffer onto an undo stack. `/undo` restores the previous state, `/redo` re-applies. Stack depth configurable (default 10) |
| T.1.2 | Track undo stack | **[NEW]** | Same pattern for track audio — `/btw back`, `/ta`, and destructive track ops push to a per-track undo stack. `/undo track` or `/undo t` restores |
| T.1.3 | Session snapshot undo | **[NEW]** | `/snapshot` saves a full lightweight session snapshot (parameters, FX chains, operator state — NOT audio). `/undo snapshot` restores. Complements audio-level undo for parameter changes |

**Implementation notes:** Audio undo copies are expensive. Use copy-on-write or ring buffer with configurable max depth. Parameter snapshots are cheap (JSON dict). Existing `/WFX undo` in playback_cmds.py covers single-effect undo on working buffer — extend this pattern to a proper stack.

### T.2 — Song Structure & Arrangement (CRITICAL)

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| T.2.1 | Section markers | **[NEW]** | `/section add <name> <start_bar> <end_bar>` — define named sections (intro, verse, chorus, bridge, outro, or custom). Stored on session, persisted in save/load. `/section list` shows all. `/section goto <name>` moves write position |
| T.2.2 | Section rendering | **[NEW]** | `/render section <name>` renders only that section. `/render all` renders the full arrangement. Sections are contiguous bar ranges on the track timeline |
| T.2.3 | Pattern chaining | **[NEW]** | `/chain <pattern_a> <repeat_a> <pattern_b> <repeat_b> ...` — chain named patterns into a longer sequence. Output to working buffer or track. This is the missing link between "I have a 4-bar loop" and "I have a song" |
| T.2.4 | Section copy/move | **[NEW]** | `/section copy <name> <to_bar>` duplicates a section's audio at a new position. `/section move <name> <to_bar>` moves it. Essential for arranging verse/chorus/verse/chorus structures |

### T.3 — Render & Export Pipeline (HIGH)

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| T.3.1 | Track stem export | **[NEW]** | `/export stems <path>` renders each track as a separate WAV/FLAC file. Essential for collaboration, remixing, and multi-DAW workflows |
| T.3.2 | Individual track export | **[NEW]** | `/export track <n> <path>` renders a single track to file with its FX chain applied |
| T.3.3 | FLAC export fix | **[NEW]** | Fix silent fallback to WAV when soundfile unavailable. If FLAC requested but unsupported, warn the user explicitly instead of silently downgrading |
| T.3.4 | Master gain control | **[NEW]** | `/master_gain <dB>` — session-level master gain applied before limiting in the render chain. Currently no way to control master output level. Add `master_gain` to session state, persist in save/load |

### T.4 — Missing Command Implementations (HIGH)

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| T.4.1 | `/crossover` command | **[NEW]** | Implement `/crossover <buf_a> <buf_b> <method>` in ai_cmds.py — the GUI already has full ActionDef and context menus for this but the command doesn't exist. Methods: temporal, spectral, blend, morphological, multi_point. Wire to `breeding.py` crossover functions which are already implemented |
| T.4.2 | `/duplicate` buffer command | **[NEW]** | Implement `/dup <source> <dest>` to copy buffer contents. Currently GUI uses `/w <n>` + `/a` as workaround but a direct command is needed for clarity and atomicity |
| T.4.3 | `/metronome` command | **[NEW]** | `/metronome on/off`, `/metronome vol <1-100>`, `/metronome sound click/beep/wood`. Generate a click track at session BPM overlaid on playback. Essential for performing and recording in time |
| T.4.4 | `/swap <a> <b>` buffer swap | **[NEW]** | Swap contents of two buffers atomically |
| T.4.5 | Stop playback fix | **[NEW]** | `/stop` in render_cmds.py is a stub (returns "OK" without stopping). Wire it to `playback.stop()` so in-progress playback actually halts |

### T.5 — Generative Output Routing (HIGH)

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| T.5.1 | Generation `route=` parameter | **[NEW]** | Add optional `route=` to `/beat`, `/loop`, `/gen2`, `/adapt`, `/xform`. Values: `working` (default), `track`, `track:<n>`, `buffer:<n>`. Example: `/beat house 4 route=track:2` |
| T.5.2 | `/commit` command | **[NEW]** | `/commit` moves working buffer to current track at write position and clears working. `/commit <n>` targets track N. Shortcut for the common "I generated something, now put it on a track" workflow |
| T.5.3 | Auto-advance write position | **[NEW]** | After committing audio to a track, auto-advance the write position to the end of the committed region. Currently write_pos stays at 0, so the next commit overwrites |

### T.6 — Save/Load & Persistence Gaps (MEDIUM)

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| T.6.1 | Auto-save timer | **[NEW]** | Auto-save project every N minutes (configurable, default 5) to `~/Documents/MDMA/autosave/`. `/autosave on/off/interval <min>`. Prevent total loss from crashes |
| T.6.2 | DJ state persistence | **[NEW]** | Persist deck audio references, crossfader position, and deck effects in save/load. Currently DJ state is lost on reload |
| T.6.3 | Section markers persistence | **[NEW]** | Persist section definitions (T.2.1) in save/load format |

### T.7 — Playback Fixes (MEDIUM)

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| T.7.1 | Playback error feedback | **[NEW]** | When playback fails (device missing, buffer empty, format error), print a clear error message instead of failing silently. Currently `_play_buffer()` returns False with no user-visible output |
| T.7.2 | Playback position query | **[NEW]** | `/pos` shows current playback position in beats and seconds. `/seek <beat>` jumps to a position. Foundation for non-linear playback |

### T.8 — Audio Import Fix (MEDIUM)

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| T.8.1 | Sample rate conversion on import | **[NEW]** | When importing audio at a different sample rate than the session, resample using `scipy.signal.resample` or `resample_poly` to match session rate. Currently imported audio plays at wrong pitch/speed if rates don't match |

### T.9 — GUI Wiring Fixes (HIGH)

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| T.9.1 | Wire `/crossover` GUI to engine | **[NEW]** | After T.4.1 implements the command, verify the GUI ActionDef template, context menus, and breed picker dialog all produce correct commands |
| T.9.2 | Breeding action category validation | **[NEW]** | Verify all 6 breeding actions, 11 tree items, and 3 dialog flows produce commands that the engine accepts. Fix any mismatched parameter names or orderings |
| T.9.3 | Buffer action category validation | **[NEW]** | Verify all 10 buffer actions produce working commands. Test duplicate, import, bounce, and info flows end-to-end |

### T.10 — File FX Chain Cleanup (LOW)

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| T.10.1 | Remove or wire file_fx_chain | **[NEW]** | `session.file_fx_chain` is initialized but never used anywhere. Either remove the dead code or implement `/filefx` commands that apply effects to audio during file import/export |

### Implementation Priority Order

```
CRITICAL (do first — these are dealbreakers):
  T.1  Undo/Redo System
  T.2  Song Structure & Arrangement

HIGH (do second — these fix broken or missing functionality):
  T.4  Missing Command Implementations
  T.5  Generative Output Routing
  T.3  Render & Export Pipeline
  T.9  GUI Wiring Fixes

MEDIUM (do third — quality of life):
  T.6  Save/Load Persistence
  T.7  Playback Fixes
  T.8  Audio Import Fix

LOW (do last — cleanup):
  T.10 File FX Chain Cleanup
```

### Milestone Deliverable

A user can open MDMA, set a tempo, generate drums and bass, arrange them into intro/verse/chorus/bridge/outro sections, mix tracks with pan/gain/mute/solo, undo mistakes, render to WAV or FLAC, and export individual stems — all without hitting a single dead end, silent failure, or missing command. **The product is song-ready.**

### Verification Criteria

Phase T is complete when ALL of the following end-to-end workflows succeed without manual workarounds:

1. **Generate → Arrange → Render**: `/beat house 4` → `/commit 1` → `/section add verse 1 4` → `/gen2 melody` → `/commit 2` → `/section add chorus 5 8` → `/render all`
2. **Undo workflow**: `/tone 440 1` → `/fx reverb 80` → `/undo` → (buffer has no reverb)
3. **Stem export**: 4-track project → `/export stems ./stems/` → 4 WAV files on disk
4. **Breeding workflow**: `/breed 1 2 4` → `/crossover 1 2 spectral` → children in buffers
5. **Section copy**: `/section copy verse 9` → verse audio duplicated at bar 9

---

## Phase 5: Advanced Sound Engines

> **Goal:** Next-generation synthesis engines that push beyond traditional DSP.
> **Depends on:** Phase 2 (Monolith foundation), Phase 4 (generation integration)
> **Delivers:** Neural and physics-based sound engines for unprecedented timbres.

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 5.1 | Full NEURO oscillator integration | **[NEW]** | Neural-network-based oscillator that learns and reproduces timbres from audio samples — train on any sound, play it chromatically |
| 5.2 | Neural network resonator system | **[NEW]** | AI resonator that models the frequency response of physical and virtual spaces — feed it audio, it learns the resonance profile |
| 5.3 | Physics simulation sound extension | **[NEW]** | Future-facing engine for physics-simulated sound — fluid dynamics, rigid body collisions, wave propagation in arbitrary media |

### Milestone Deliverable
Sound engines that can learn timbres from recordings, model resonant spaces with AI, and simulate physics-based acoustics — tools that don't exist in any other accessible DAW.

---

## Phase 6: MIDI as Protocol Writer

> **Goal:** Add MIDI input as a protocol-writing accelerator and preview layer — NOT a new sequencing system.
> **Depends on:** Phase 1 (GUI surface with console input field), Phase 2 (synth engine for preview/render)
> **Delivers:** MIDI keyboard input that writes interval tokens into the existing protocol, plus audible preview.
> **Design spec:** v6.0 (MIDI as Protocol Writer — Final Locked Spec)

### Core Principle

MIDI writes directly into the existing interval protocol text field. It does NOT introduce a new sequencing system, clip system, automation lanes, or real-time DAW layer. MDMA remains deterministic and render-first.

### MIDI Modes

**Preview Mode (default):** When the cursor is NOT in the protocol input field, pressing a MIDI key plays a preview tone through the current synth backend (Monolith). No text is written. Used for patch testing.

**Program Mode:** When the cursor IS inside the protocol input field, pressing a MIDI key converts the pitch to an interval relative to the current root and inserts it at the cursor. Repeated presses at the same position append `.` to extend duration. Multiple keys pressed within the chord detection window insert a chord grouping token.

### Phase 6.0 Features

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 6.0.1 | MIDI device manager | **[NEW]** | List available MIDI input devices, select active device, basic note-on/note-off capture via `mido` + `python-rtmidi`. `/midi list`, `/midi select <n>`, `/midi status` |
| 6.0.2 | Interval translator | **[NEW]** | Convert MIDI note number to semitone interval relative to session root (`interval = midi_note - root_note`). Detect chord clusters within configurable window (default 80ms). Format as protocol tokens: single notes (`0`), chords (`(0,4,7)`), duration extension (`.`), rest (`_`) |
| 6.0.3 | MIDI preview trigger | **[NEW]** | Route MIDI note-on through current Monolith engine for audible feedback in Preview Mode. Short preview tone (~200ms) at the note's frequency, non-blocking |
| 6.0.4 | Protocol token injection | **[NEW]** | In Program Mode, inject formatted interval tokens at cursor position in the GUI console input field. Supports single notes, chords (parentheses grouping), duration extension, and rest insertion |
| 6.0.5 | Rest token `_` | **[NEW]** | Add `_` as explicit rest symbol to the interval protocol DSL. Parsed by `parse_melody_pattern()` and `parse_chord_pattern()` alongside existing `r` rest. Removes ambiguity from space-based separation |
| 6.0.6 | Chord grouping `(0,4,7)` | **[NEW]** | Add parentheses-based chord grouping to the melody parser so MIDI chord input produces valid protocol text. `(0,4,7)..` = chord sustained for 2 beats |
| 6.0.7 | Chord detection window | **[NEW]** | Configurable chord detection window in session settings — `session.chord_window_ms` (default 80ms). `/midi window <ms>` to adjust. Notes arriving within the window are grouped as a chord |
| 6.0.8 | MIDI root setting | **[NEW]** | `/midi root <note>` sets the MIDI root note for interval calculation (default: 60 / C4). Stored in session, persisted in save/load. Root changes do NOT retroactively alter stored intervals |

### What Phase 6.0 Does NOT Include

- CC mapping, automation, pitch bend, aftertouch, MPE, transport clock sync
- MIDI output to external hardware
- VST hosting (deferred to future phase)
- Accessible surface transform (deferred to future phase)
- Real-time recording mode

### Interval Conversion Rules

```
interval = midi_note - session.midi_root_note

Example (root = MIDI 60 / C4):
  MIDI 60 → interval 0
  MIDI 62 → interval 2
  MIDI 59 → interval -1
  MIDI 72 → interval 12 (octave)
```

Root changes do NOT rewrite existing intervals. Any integer interval is allowed (no clamping).

### New Protocol Tokens

| Token | Meaning | Example |
|-------|---------|---------|
| `_` | Explicit rest (1 beat) | `0._. 2` |
| `(N,N,N)` | Chord grouping | `(0,4,7)` = C major triad |
| `(N,N,N)..` | Extended chord | `(0,4,7)..` = 2 beats |

### Phase 6.1 (Future)

| # | Feature | Description |
|---|---------|-------------|
| 6.1.1 | Real-time MIDI recording | Record MIDI input to protocol in real time with quantization |
| 6.1.2 | CC handling | Map MIDI CC messages to synth/effect parameters |
| 6.1.3 | MIDI output | Send patterns as MIDI to external hardware/software |
| 6.1.4 | VST render backend | Load VST instruments as alternative synth backend |
| 6.1.5 | Accessible surface transform | Flat, navigable parameter panels for external plugin GUIs |

### Technical Components

1. **MIDI Device Manager** (`mdma_rebuild/dsp/midi_input.py`) — Device enumeration, selection, note capture via `mido`/`python-rtmidi`
2. **Interval Translator** (integrated into midi_input.py) — Note-to-interval conversion, chord cluster detection, token formatting
3. **Preview Trigger** (integrated into midi_input.py) — Route notes through Monolith for audible feedback
4. **MIDI Commands** (`mdma_rebuild/commands/midi_cmds.py`) — `/midi` command family for device management and configuration

### Milestone Deliverable
A MIDI keyboard connected to MDMA writes interval tokens directly into the protocol input field. Users can preview patches by playing keys, then switch to the input field and type melodies and chords at keyboard speed. The protocol text is the same format used by `/mel` and `/cor` — no new abstractions, no timeline, no sequencer.

---

## Phase 7: Presets & Content Tools

> **Goal:** Make it trivial to create, share, and use sound content.
> **Depends on:** Phase 2 (synth engine), Phase 4 (generation), Phase 6 (VST for hybrid presets)
> **Delivers:** One-click preset and sample pack creation tools.

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 7.1 | Quick preset pack generator | **[NEW]** | Generate themed preset packs from a seed patch — auto-create variations across parameter ranges, export as shareable pack |
| 7.2 | Sample pack generator | **[NEW]** | Render presets and patterns into organized sample packs — export as WAV/FLAC folders with metadata, ready for distribution |
| 7.3 | Built-in prefab library | **[PARTIAL]** | Curated library of ready-to-use building blocks — drum kits, bass patches, pad textures, FX chains, pattern templates — factory presets exist (15), needs significant expansion |

### Milestone Deliverable
Users can generate complete preset packs and sample packs from their MDMA sessions with a single command, and start new projects from a rich built-in library.

---

## Phase 8: DJ & Performance Tools

> **Goal:** Make MDMA a capable live performance and DJ environment.
> **Depends on:** Phase 1 (GUI), Phase 6 (hardware controllers)
> **Delivers:** Professional DJ workflow with real-time control and hardware integration.

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 8.1 | Extended DJ performance tools | **[PARTIAL]** | Expand DJ mode beyond current deck/crossfade model — cue points, hot cues, beat jump, loop rolls, slip mode |
| 8.2 | Granular real-time control | **[PARTIAL]** | Real-time granular parameter manipulation during performance — grain size, density, pitch, position all controllable live |
| 8.3 | High-quality scratching effects | **[NEW]** | Vinyl-accurate scratch simulation with platter physics, needle drop, and back-spin — mapped to MIDI controller or keyboard |
| 8.4 | Tape effects | **[NEW]** | Tape stop, tape start, wow/flutter, tape saturation — all with authentic analog behavior and real-time control |
| 8.5 | Improved performance schedulers | **[PARTIAL]** | Quantized event scheduling for live performance — trigger clips, FX, and transitions on beat/bar boundaries |
| 8.6 | Hardware controller support | **[NEW]** | Native support for DJ controllers (HID/MIDI) — auto-mapping for common controllers, custom mapping editor |

### Milestone Deliverable
A live performance environment where DJs and performers can mix, scratch, and manipulate audio in real time using hardware controllers — all fully accessible.

---

## Phase 9: Recording & Input Configuration

> **Goal:** Make audio input and recording effortless.
> **Depends on:** Phase 1 (GUI for config), Phase 6 (hardware integration)
> **Delivers:** Zero-friction recording workflow with device management.

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 9.1 | Easy recording configuration | **[NEW]** | One-panel recording setup — select input device, set levels, arm track, hit record — no hidden settings |
| 9.2 | Mini controller integration | **[NEW]** | Support for compact MIDI/USB controllers as recording triggers — footswitches, pad controllers, transport remotes |
| 9.3 | Real audio input support | **[NEW]** | Live audio input from microphones and instruments — monitor through MDMA's FX chain in real time with low latency |
| 9.4 | Device tracking support | **[NEW]** | Automatic device detection, hot-plug handling, and session-persistent device assignments — MDMA remembers your setup |

### Milestone Deliverable
Users can plug in a mic or instrument, see it appear in MDMA, arm a track, and record — with live monitoring through effects — all without touching a config file.

---

## Phase 10: Visualization & Media Integration

> **Goal:** See the music — accessible visual feedback and creative media output.
> **Depends on:** Phase 1 (GUI framework), Phase 8 (performance context)
> **Delivers:** Real-time visualization and AI-driven video generation.

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 10.1 | Built-in accessible visualizer tools | **[PARTIAL]** | Audio-reactive visualizations built with accessibility in mind — high contrast, configurable color schemes, screen-reader descriptors for visual elements |
| 10.2 | ProjectM integration | **[NEW]** | Embed the ProjectM (Milkdrop) visualization engine — thousands of community presets, audio-reactive, full-screen capable |
| 10.3 | External screen visualization | **[NEW]** | Output visualizations to a secondary display — fullscreen visuals for live performance while keeping the DAW on the primary screen |
| 10.4 | AI/algorithmic video generation | **[NEW]** | Generate abstract video content synchronized to audio — algorithmic visuals driven by frequency, amplitude, and musical structure |

### Milestone Deliverable
Live audio-reactive visuals on external screens, accessible visualizer tools for all users, and AI-generated video content synced to the music.

---

## Phase Summary & Dependency Map

```
Phase 1: Core Interface & Workflow         [DONE]
  |
  +---> Phase 2: Monolith Engine & Synthesis     [DONE]
  |       |
  |       +---> Phase 3: Modulation, Impulse & Convolution  [DONE]
  |       |       |
  |       |       +---> Phase 4: Generative Systems          [DONE]
  |       |               |
  |       |               +---> Phase 4b: Microtonal Support [NEW]
  |       |               |
  |       |               +---> Phase T: Full System Audit   [NEW] ← GATE
  |       |                       |
  |       |                       +---> Phase 5: Advanced Sound Engines
  |       |
  |       +---> Phase 7: Presets & Content Tools
  |
  +---> Phase 6: MIDI as Protocol Writer       [IN PROGRESS]
  |       |
  |       +---> Phase 6.1: Advanced MIDI/VST (future)
  |       +---> Phase 7: Presets & Content Tools
  |       +---> Phase 8: DJ & Performance Tools
  |       +---> Phase 9: Recording & Input Configuration
  |
  +---> Phase 10: Visualization & Media Integration
```

**Phase T is a gate.** It does not add new capabilities — it makes existing capabilities reliable and complete. No phase after T should inherit broken wiring or silent failures.

### Parallelization Opportunities

These phase groups can be developed concurrently:

| Track A (Sound Engine) | Track B (Integration) | Track C (Media) |
|------------------------|-----------------------|-----------------|
| ~~Phase 2: Synthesis~~ DONE | **Phase 6.0: MIDI Protocol Writer** IN PROGRESS | Phase 10: Visualization |
| ~~Phase 3: Modulation~~ DONE | Phase 6.1: Advanced MIDI/VST (future) | |
| ~~Phase 4: Generative~~ DONE | Phase 8: DJ & Performance | |
| | Phase 9: Recording | |
| Phase 4b: Microtonal | | |
| **Phase T: System Audit** (GATE) | | |
| Phase 5: Neural Engines | | |
| Phase 7: Presets (after both tracks) | | |

**Note:** Phase T and Phase 4b can run in parallel. Phase T and Track B/C can also run in parallel since Track B/C are independent integration work. Phase 5 waits on Phase T.

---

## Feature Count Summary

| Phase | Status | Features | Done | Remaining |
|-------|--------|----------|------|-----------|
| 1. Core Interface | **COMPLETE** | 8 | 8 | 0 |
| 2. Monolith & Synthesis | **COMPLETE** | 9 | 9 | 0 |
| 3. Modulation & Convolution | **COMPLETE** | 6 | 6 | 0 |
| 4. Generative Systems | **COMPLETE** | 5 | 5 | 0 |
| 4b. Microtonal Support | NEW | 8 | 0 | 8 |
| **T. Full System Audit** | **NEW (GATE)** | **22** | **0** | **22** |
| 5. Advanced Sound Engines | NEW | 3 | 0 | 3 |
| 6.0 MIDI Protocol Writer | **IN PROGRESS** | 8 | 0 | 8 |
| 6.1 Advanced MIDI/VST | FUTURE | 5 | 0 | 5 |
| 7. Presets & Content | PARTIAL | 3 | 0 | 3 |
| 8. DJ & Performance | PARTIAL | 6 | 0 | 6 |
| 9. Recording & Input | NEW | 4 | 0 | 4 |
| 10. Visualization & Media | PARTIAL | 4 | 0 | 4 |
| **Total** | | **91** | **28** | **63** |

---

## Guiding Principles

1. **Accessibility is not optional.** Every feature ships with keyboard navigation, screen-reader labels, and non-visual feedback. If a feature can't be made accessible, it doesn't ship.

2. **CLI parity.** Every GUI feature has a corresponding CLI command. The GUI never invents behavior that the CLI can't reproduce.

3. **Ship usable increments.** Each phase delivers value on its own. No phase requires all subsequent phases to be useful.

4. **Sound quality over feature count.** A smaller number of professional-quality engines beats a large number of toy implementations.

5. **The engine is the truth.** The GUI is a client. All state lives in the MDMA session. All operations go through the command system.

---

*MDMA — Music Design Made Accessible*
*Built for vision-optional audio production*
