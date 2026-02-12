# MDMA Full Release Roadmap

**Music Design Made Accessible**
**Document Version:** 1.2
**Baseline:** v52.0 (2026-02-03)
**Last Updated:** 2026-02-11 (Phase 1 & 2 complete)
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

## Phase 3: Modulation, Impulse & Convolution

> **Goal:** Professional-grade modulation sources and convolution processing.
> **Depends on:** Phase 2 (synth engine to modulate)
> **Delivers:** Deep modulation routing and studio-quality convolution reverb.

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 3.1 | Pulse/impulse importing for LFOs | **[NEW]** | Import audio files as LFO waveshapes — any waveform becomes a modulation source |
| 3.2 | Impulse importing for envelopes | **[NEW]** | Import impulse recordings as envelope shapes — transient captures become amplitude contours |
| 3.3 | Advanced convolution reverb engine | **[NEW]** | Full convolution reverb with IR loading, pre-delay, decay control, early/late reflection split, and stereo width |
| 3.4 | Neural-enhanced convolution | **[NEW]** | AI processing layer that enhances or extends impulse responses — fill gaps, extend tails, denoise captured IRs |
| 3.5 | AI-assisted impulse transformation | **[NEW]** | Transform impulse responses using AI descriptors — "make this room bigger," "add metallic quality," "darken the tail" |
| 3.6 | Granular impulse tools | **[PARTIAL]** | Granular destruction, stretching, and resampling of impulse responses — granular engine exists, needs IR-specific workflow |

### Milestone Deliverable
A modulation and convolution system where any audio can become a modulation source, convolution reverbs sound studio-grade, and AI can intelligently reshape acoustic spaces.

---

## Phase 4: Generative Systems

> **Goal:** AI and algorithmic tools that produce musically useful output.
> **Depends on:** Phase 2 (synth engine for rendering), Phase 3 (modulation for expression)
> **Delivers:** Generation engines that produce patterns, loops, and beats ready for arrangement.

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 4.1 | Improved content generation | **[PARTIAL]** | Overhaul existing generation algorithms for higher musical quality — better voice leading, harmonic awareness, rhythm coherence |
| 4.2 | Pattern adaptation algorithms | **[PARTIAL]** | Algorithms that adapt existing patterns to new keys, scales, time signatures, and styles while preserving musical intent |
| 4.3 | Full loop generation engine | **[NEW]** | End-to-end loop generator — specify genre, tempo, key, mood and receive a rendered, mixable loop |
| 4.4 | Drum beat generation engine | **[NEW]** | Dedicated beat generator with genre templates, humanization, fill generation, and polyrhythm support |
| 4.5 | Transformation engines | **[PARTIAL]** | Musically aware transformations — retrograde, inversion, augmentation, diminution, motivic development — that produce usable results, not just math |

### Milestone Deliverable
Users can say "generate a 4-bar drum loop in this style" or "adapt this melody to minor key" and get musically coherent, production-ready results.

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

## Phase 6: MIDI, VST & Hardware Integration

> **Goal:** Connect MDMA to the broader music production ecosystem.
> **Depends on:** Phase 1 (GUI surface for configuration), Phase 2 (synth engine for MIDI targets)
> **Delivers:** MIDI I/O, VST hosting, and accessible hardware control surfaces.

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 6.1 | MIDI input | **[NEW]** | Receive MIDI from external controllers and keyboards — note on/off, CC, pitch bend routed to MDMA parameters |
| 6.2 | MIDI output | **[NEW]** | Send MIDI to external synths and hardware — pattern playback, note sequencing, CC automation |
| 6.3 | VST support | **[NEW]** | Host VST2/VST3 plugins as effects or instruments — scan, load, configure, and automate plugin parameters |
| 6.4 | Accessible surface transform | **[NEW]** | Expose raw hardware/plugin parameters as flat, navigable panels — no nested menus, every knob is a labeled widget accessible by keyboard and screen reader |

### Milestone Deliverable
MDMA talks to MIDI hardware, hosts VST plugins, and makes every external parameter accessible through flat, navigable panels — no visual-only plugin GUIs required.

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
Phase 1: Core Interface & Workflow
  |
  +---> Phase 2: Monolith Engine & Synthesis
  |       |
  |       +---> Phase 3: Modulation, Impulse & Convolution
  |       |       |
  |       |       +---> Phase 4: Generative Systems
  |       |               |
  |       |               +---> Phase 5: Advanced Sound Engines
  |       |
  |       +---> Phase 7: Presets & Content Tools
  |
  +---> Phase 6: MIDI, VST & Hardware Integration
  |       |
  |       +---> Phase 7: Presets & Content Tools
  |       +---> Phase 8: DJ & Performance Tools
  |       +---> Phase 9: Recording & Input Configuration
  |
  +---> Phase 10: Visualization & Media Integration
```

### Parallelization Opportunities

These phase groups can be developed concurrently:

| Track A (Sound Engine) | Track B (Integration) | Track C (Media) |
|------------------------|-----------------------|-----------------|
| Phase 2: Synthesis | Phase 6: MIDI/VST/HW | Phase 10: Visualization |
| Phase 3: Modulation | Phase 8: DJ & Performance | |
| Phase 4: Generative | Phase 9: Recording | |
| Phase 5: Neural Engines | | |
| Phase 7: Presets (after both tracks) | | |

---

## Feature Count Summary

| Phase | New Features | Partial/Existing | Total |
|-------|-------------|------------------|-------|
| 1. Core Interface | 8 DONE | 0 | 8 |
| 2. Monolith & Synthesis | 9 DONE | 0 | 9 |
| 3. Modulation & Convolution | 5 | 1 | 6 |
| 4. Generative Systems | 2 | 3 | 5 |
| 5. Advanced Sound Engines | 3 | 0 | 3 |
| 6. MIDI, VST & Hardware | 4 | 0 | 4 |
| 7. Presets & Content | 2 | 1 | 3 |
| 8. DJ & Performance | 3 | 3 | 6 |
| 9. Recording & Input | 4 | 0 | 4 |
| 10. Visualization & Media | 3 | 1 | 4 |
| **Total** | **38** | **14** | **52** |

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
