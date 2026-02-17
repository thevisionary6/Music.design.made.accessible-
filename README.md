# MDMA - Music Design Made Accessible

A command-line audio production environment built for screen reader accessibility and zero-vision workflows. Create music, design sounds, and mix tracks entirely from the terminal — no visual interface required.

**Version:** 52.0
**Build:** mdma_v52.0_20260203
**Python:** 3.10+
**Platform:** Windows, macOS, Linux
**Phases Complete:** 1-4 of 10 (Core Interface, Synthesis, Modulation/Convolution, Generative Systems)

---

## What It Does

MDMA is a text-based digital audio workstation (DAW) that runs in your terminal. It provides:

- A **synthesizer engine** with FM synthesis, 22 waveform types, physical modeling, and wavetable support
- **113+ audio effects** (reverb, delay, distortion, filters, granular, convolution, and more)
- **Advanced convolution reverb** with 17 IR presets and neural-enhanced processing
- **Generative systems** — algorithmic melody, chord progressions, beat generation, loop creation, and transformation engines
- **Music theory engine** — 21 scales, 21 chord types, 12 progressions, voice leading, key detection
- **Multi-track project management** with full-state save/load
- **DJ mode** with multi-deck mixing, crossfading, and deck-specific effects
- **AI-assisted** sound generation and analysis
- An optional **wxPython GUI** with object browser, patch builder, step grid, and inspector panels

Everything is designed so screen readers can access every feature without any visual dependency.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/thevisionary6/Music.design.made.accessible-.git
cd Music.design.made.accessible-

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install core dependencies
pip install numpy scipy soundfile sounddevice

# Optional: GUI support
pip install wxPython

# Optional: AI features
pip install anthropic
```

### Verify Installation

```bash
python -c "from mdma_rebuild import __version__; print(f'MDMA v{__version__}')"
python -c "from mdma_rebuild.dsp.playback import get_backend; print(f'Audio backend: {get_backend()}')"
```

---

## Quick Start

### Launch the CLI

```bash
python bmdma.py
```

### Basic Workflow

```
/bpm 120              # Set tempo
/tone 440 1           # Generate a 440 Hz tone, 1 beat long
/play                 # Listen to it
/fx reverb 60         # Add reverb at 60% intensity
/play                 # Listen again
/save myproject       # Save your work
```

### Launch the GUI (optional)

```bash
python mdma_gui.py
```

---

## Commands Overview

MDMA uses slash-prefixed commands. There are 569+ commands across these categories:

### Playback

| Command | Description |
|---------|-------------|
| `/p` | Play working buffer |
| `/pb <n>` | Play buffer N |
| `/pt <n>` | Play track N |
| `/pts` | Play all tracks (song) |
| `/stop` | Stop playback |
| `/vol <n>` | Set volume (1-100) |

### Synthesis

| Command | Description |
|---------|-------------|
| `/tone <freq> <dur>` | Generate a tone |
| `/n <freq>` | Quick note |
| `/mel <pattern>` | Melody from dot-separated intervals (e.g. `0.4.7`) |
| `/cor <pattern>` | Chord pattern |
| `/preset <name>` | Load a synth preset |

### Effects

| Command | Description |
|---------|-------------|
| `/fx <name> [amount]` | Apply an effect (amount 1-100) |
| `/chain <name>` | Apply a named effect chain |
| `/conv preset:<name>` | Apply convolution reverb (17 IR presets) |

Some of the 113+ effects: `reverb`, `delay`, `chorus`, `flanger`, `phaser`, `saturate_tube`, `bitcrush`, `lowpass`, `highpass`, `compress`, `granular`, `normalize`, and many more.

### Generative Systems

| Command | Description |
|---------|-------------|
| `/gen2 <type> <scale> <n>` | Generate melody, chords, bassline, arpeggio, or drone |
| `/beat <genre> <bars>` | Generate drum beat (11 genre templates) |
| `/loop <genre> <bars>` | Generate multi-layer loop (drums, bass, chords, melody) |
| `/xform <transform>` | Apply note or audio transforms (retrograde, inversion, stutter, etc.) |
| `/adapt <mode> <args>` | Adapt patterns — key/scale/tempo/style changes |
| `/theory scales` | List available scales, chords, and progressions |

### Convolution & Impulse

| Command | Description |
|---------|-------------|
| `/conv preset:<name>` | Apply convolution reverb (hall, room, plate, spring, shimmer, reverse) |
| `/impulselfo load <file>` | Import audio as LFO waveshape |
| `/impenv load <file>` | Import impulse as envelope shape |
| `/irenhance <mode>` | Neural-enhanced IR processing (extend, denoise, fill) |
| `/irtransform <descriptor>` | Semantic IR transform (bigger, darker, metallic, cathedral, etc.) |

### Parameters

| Command | Description |
|---------|-------------|
| `/parm` | List parameter categories |
| `/parm <cat>` | List parameters in a category |
| `/cha <value>` | Change selected parameter |
| `/cha lfo sin 4 50` | Assign LFO modulation |

### DJ Mode

| Command | Description |
|---------|-------------|
| `/deck <n>` | Select a deck |
| `/xfade <n>` | Crossfade between decks |
| `/djm` | DJ master settings |

### Project

| Command | Description |
|---------|-------------|
| `/save <name>` | Save project |
| `/load <name>` | Load project |
| `/import <file>` | Import an audio file |
| `/bpm <tempo>` | Set tempo |
| `/hq on` | Enable high-quality output mode |

Run `/help` inside the REPL for the full command list.

---

## Project Structure

```
Music.design.made.accessible-/
|
|-- bmdma.py                    # Main CLI entry point (REPL)
|-- mdma_gui.py                 # wxPython GUI
|-- mad_tui.py                  # TUI launcher
|-- run_mdma.py                 # Main executable entry
|-- test_mad_dsl.py             # Test suite
|
|-- mdma_rebuild/               # Core package
|   |-- __init__.py             # Version and package metadata
|   |
|   |-- core/                   # State and session management
|   |   |-- session.py          # Session object (holds all state)
|   |   |-- objects.py          # First-class object model (Pattern, Patch, etc.)
|   |   |-- registry.py         # Object registry (shared object store)
|   |   |-- user_data.py        # User profile and preferences
|   |   |-- banks.py            # Preset bank system
|   |   |-- pack.py             # Project/pack management
|   |   |-- song_registry.py    # Song tracking
|   |
|   |-- dsp/                    # Audio processing (~29k lines)
|   |   |-- monolith.py         # FM synthesis engine (22 wave types)
|   |   |-- effects.py          # 113+ DSP effects
|   |   |-- music_theory.py     # Scales, chords, progressions, voice leading
|   |   |-- beat_gen.py         # Drum beat generation (17 generators, 11 genres)
|   |   |-- loop_gen.py         # Multi-layer loop generation
|   |   |-- transforms.py       # Note and audio transforms
|   |   |-- convolution.py      # Convolution reverb engine (17 IR presets)
|   |   |-- pattern.py          # Pattern and modulation system
|   |   |-- granular.py         # Granular synthesis
|   |   |-- playback.py         # In-house audio playback
|   |   |-- generators.py       # Audio generators
|   |   |-- envelopes.py        # ADSR envelopes
|   |   |-- scaling.py          # Unified 1-100 parameter scaling
|   |   |-- dj_mode.py          # DJ engine (decks, mixing)
|   |   |-- streaming.py        # Real-time streaming
|   |   |-- stems.py            # Stem separation
|   |   |-- enhancement.py      # Audio enhancement
|   |   |-- performance.py      # Performance tools
|   |   |-- visualization.py    # Audio visualization
|   |   |-- advanced_ops.py     # Advanced DSP operations
|   |
|   |-- commands/               # Command modules (24 modules, 569+ commands)
|   |   |-- general_cmds.py     # Help, save, load, project setup
|   |   |-- synth_cmds.py       # Tone and synthesis commands
|   |   |-- fx_cmds.py          # Effect commands
|   |   |-- playback_cmds.py    # Playback control
|   |   |-- buffer_cmds.py      # Buffer operations
|   |   |-- pattern_cmds.py     # Pattern commands
|   |   |-- param_cmds.py       # Parameter system
|   |   |-- gen_cmds.py         # Generative systems (/beat, /loop, /gen2, /xform, /adapt)
|   |   |-- convolution_cmds.py # Convolution reverb and impulse commands
|   |   |-- dsl_cmds.py         # DSL block processing
|   |   |-- dj_cmds.py          # DJ mode commands
|   |   |-- ai_cmds.py          # AI-powered commands
|   |   |-- render_cmds.py      # Output and rendering
|   |   |-- ... and more
|   |
|   |-- ai/                     # AI integration
|       |-- core.py             # AI engine core
|       |-- generation.py       # AI sound generation
|       |-- analysis.py         # AI audio analysis
|       |-- breeding.py         # Sound breeding/evolution
|       |-- router.py           # AI command routing
|       |-- descriptors.py      # AI descriptors
|
|-- gui/                         # Modular GUI package (new architecture)
|   |-- shell.py                 # Top-level window manager
|   |-- bridge.py                # Session/DSP adapter
|   |-- events.py                # Custom wx event types
|   |-- windows/                 # Sub-window modules
|   |-- panels/                  # Panel modules (per window)
|   |-- widgets/                 # Accessible reusable widgets
|
|-- docs/
|   |-- specs/
|       |-- GUI_WINDOW_ARCHITECTURE_SPEC.md  # Modular window architecture
|       |-- OBJECT_MODEL_SPEC.md             # First-class object model
|
|-- COMMANDS.md                  # Full command reference (569+ commands)
|-- VERSION.md                   # Version history
|-- RELEASE_NOTES.md             # Release notes
|-- CHANGELOG_v52.md             # v52 changelog
|-- ROADMAP_FULL_RELEASE.md      # 10-phase roadmap to v1.0
|-- GUI_SPEC_v0.2.md             # GUI specification (superseded by window arch spec)
|-- INTERFACE_TRANSITION_SPEC.md  # Architecture rationale
```

---

## Architecture

MDMA follows a **single-source-of-truth** design:

- **Session** (`core/session.py`) holds all mutable state: buffers, tracks, parameters, effects, playback.
- **DSP modules** (`dsp/`) handle audio generation and processing. They are independent of the command layer.
- **Command modules** (`commands/`) are thin wrappers that parse user input and call into DSP/session.
- **Interfaces** (CLI, GUI, TUI) are all clients of the same engine. The CLI is the primary, authoritative interface.

Audio is processed internally as **64-bit float stereo** (N x 2 numpy arrays). Output supports 16/24/32-bit WAV and 24-bit FLAC.

---

## Audio Backends

MDMA plays audio directly (no external media player). Backends are tried in this order:

1. **sounddevice** (recommended) - `pip install sounddevice`
2. **simpleaudio** - `pip install simpleaudio`
3. **pyaudio** - `pip install pyaudio`
4. **fallback** - File-based output if no backend is available

---

## Accessibility

MDMA was built from the ground up for accessible audio production:

- **No visual dependencies** - every feature works via text
- **Screen reader compatible** - consistent, readable output
- **In-house playback** - no popup windows or external players
- **Unified 1-100 scaling** - all parameters use the same intuitive range
- **Human-readable presets** - named values alongside numeric ones
- **Clear text feedback** - every action reports what it did

---

## Running Tests

```bash
python test_mad_dsl.py
```

---

## Documentation

| File | Contents |
|------|----------|
| `COMMANDS.md` | Full command reference (569+ commands) |
| `VERSION.md` | Detailed version history (v38 through v52) |
| `RELEASE_NOTES.md` | Release notes for major versions |
| `CHANGELOG_v52.md` | v52 changes (GUI MVP, interface transition) |
| `ROADMAP_FULL_RELEASE.md` | 10-phase roadmap to v1.0 (Phases 1-4 complete) |
| `GUI_SPEC_v0.2.md` | GUI design specification (MVP, superseded) |
| `INTERFACE_TRANSITION_SPEC.md` | Architecture and design rationale |
| `docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md` | Modular sub-window GUI architecture |
| `docs/specs/OBJECT_MODEL_SPEC.md` | First-class object model and registry |

---

## License

MDMA - Music Design Made Accessible
Built for vision-optional audio production.
