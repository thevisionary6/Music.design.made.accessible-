# MDMA - Music Design Made Accessible

A command-line audio production environment built for screen reader accessibility and zero-vision workflows. Create music, design sounds, and mix tracks entirely from the terminal — no visual interface required.

**Version:** 52.0
**Python:** 3.x
**Platform:** Windows, macOS, Linux

---

## What It Does

MDMA is a text-based digital audio workstation (DAW) that runs in your terminal. It provides:

- A **synthesizer engine** with FM synthesis and multi-voice support
- **113+ audio effects** (reverb, delay, distortion, filters, granular, and more)
- **Multi-track project management** with save/load
- **DJ mode** with multi-deck mixing and crossfading
- **AI-assisted** sound generation and analysis
- An optional **wxPython GUI** for visual interaction

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

Some of the 113+ effects: `reverb`, `delay`, `chorus`, `flanger`, `phaser`, `saturate_tube`, `bitcrush`, `lowpass`, `highpass`, `compress`, `granular`, `normalize`, and many more.

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
|-- test_mad_dsl.py             # Test suite
|
|-- mdma_rebuild/               # Core package
|   |-- __init__.py             # Version and package metadata
|   |
|   |-- core/                   # State and session management
|   |   |-- session.py          # Session object (holds all state)
|   |   |-- user_data.py        # User profile and preferences
|   |   |-- banks.py            # Preset bank system
|   |   |-- pack.py             # Project/pack management
|   |   |-- song_registry.py    # Song tracking
|   |
|   |-- dsp/                    # Audio processing
|   |   |-- monolith.py         # FM synthesis engine
|   |   |-- effects.py          # 113+ DSP effects
|   |   |-- pattern.py          # Pattern and modulation system
|   |   |-- playback.py         # In-house audio playback
|   |   |-- generators.py       # Audio generators
|   |   |-- granular.py         # Granular synthesis
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
|   |-- commands/               # Command modules (23 modules)
|   |   |-- general_cmds.py     # Help, save, load, project setup
|   |   |-- synth_cmds.py       # Tone and synthesis commands
|   |   |-- fx_cmds.py          # Effect commands
|   |   |-- playback_cmds.py    # Playback control
|   |   |-- buffer_cmds.py      # Buffer operations
|   |   |-- pattern_cmds.py     # Pattern commands
|   |   |-- param_cmds.py       # Parameter system
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
|-- Documentation
    |-- COMMANDS.md              # Full command reference
    |-- VERSION.md               # Version history
    |-- RELEASE_NOTES.md         # Release notes
    |-- CHANGELOG_v52.md         # v52 changelog
    |-- ROADMAP_v45.md           # Development roadmap
    |-- GUI_SPEC_v0.2.md         # GUI specification
    |-- INTERFACE_TRANSITION_SPEC.md  # Architecture document
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
| `VERSION.md` | Detailed version history |
| `RELEASE_NOTES.md` | Release notes for major versions |
| `CHANGELOG_v52.md` | Latest changes (GUI, interface transition) |
| `GUI_SPEC_v0.2.md` | GUI design specification |
| `INTERFACE_TRANSITION_SPEC.md` | Architecture and design rationale |

---

## License

MDMA - Music Design Made Accessible
Built for vision-optional audio production.
