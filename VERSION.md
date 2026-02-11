# MDMA Version History

## Version 46 — Unified Playback, Parameter System, HQ Audio (Current)

**Release Date:** 2026-02-03  
**Build ID:** bmdma_v46.0  
**Total Commands:** 569+  
**DSP Effects:** 113+  
**New Commands:** 7 (parm, cha, plist, pt, pts, pall, bout)

### Unified Playback System

All playback now uses in-house audio engine (sounddevice):

| Command | Description |
|---------|-------------|
| `/p` | Play working buffer (always) |
| `/pb <n>` | Play buffer N |
| `/pt <n>` | Play track N |
| `/pts` | Play song (all tracks mixed) |
| `/pd <n>` | Play deck N |
| `/pall` | Play all buffers mixed |

Output format: `▶ Playing: <source> (<duration>s) @ <sample_rate>Hz`

### Parameter System

Explicit parameter management with modulation support:

| Command | Description |
|---------|-------------|
| `/parm` | List all parameter categories |
| `/parm <cat>` | List parameters in category |
| `/parm <cat> <n>` | Select parameter N |
| `/parm ?` | Show selected parameter |
| `/cha <value>` | Set parameter value |
| `/cha +/-/*// <v>` | Math operations |
| `/cha rand` | Random value |
| `/cha lfo <shape> <rate> <depth>` | Assign LFO modulation |
| `/cha env <a> <d> <s> <r>` | Assign envelope modulation |
| `/cha clear` | Remove modulation |
| `/plist` | List all parameters |

Categories: SYNTH, FILTER, GLOBAL, VOICE
LFO shapes: sin, tri, saw, sqr, rnd

### HQ Audio Mode

High-quality render chain with format selection:

| Setting | Description |
|---------|-------------|
| `/hq on/off` | Enable/disable HQ chain |
| `/hq format wav/flac` | Set output format |
| `/hq bits 16/24/32` | Set bit depth |
| `/hq dc/sub/high/sat/lim` | Individual chain stages |

HQ Chain: DC removal → 20Hz highpass → 16kHz smoothing → saturation → limiting

### Internal Improvements

- All audio processing uses 64-bit float
- Buffer iteration covers buffers 1-10
- Unified `_unified_play()` function for all playback

---

## Version 44 — Persistence, Import, Keybindings, Auto-Save (Finalized)

**Release Date:** 2026-02-02  
**Build ID:** bmdma_v44.0  
**Total Commands:** 559  
**DSP Effects:** 113  
**Effect Aliases:** 167  
**Factory Presets:** 15  
**Test Suite:** 34/34 passing

### Full-State Project Persistence

`/save` and `/load` now persist the complete session state:
- Session parameters (BPM, envelope, synthesis settings)
- Track audio (base64-encoded numpy arrays with shape preservation)
- Buffers and clips
- Effects chains (buffer, track, master, file)
- Operators and modulation routing
- **SyDef definitions** (user-created, factory excluded)
- **Named chains**
- **User functions** (/fn blocks)
- **Working buffer** (if non-silent)

Track audio is serialized using `__ndarray__` dict format with base64
encoding and shape metadata for correct stereo reconstruction.

### /import — File and Data Import

| Usage | Description |
|-------|-------------|
| `/import kick.wav` | Import audio to working buffer |
| `/import bass.wav track` | Import audio directly to current track |
| `/import patches.json` | Import SyDef/chain/function definitions |
| `/import other.mdma` | Merge definitions from another project |

Supports `.wav` (8/16/24/32-bit), `.json` (sydefs, chains, functions),
and `.mdma` (project merge — definitions only, no audio replacement).
Audio is automatically resampled to the session sample rate.

### Keybindings

| Key | Action |
|-----|--------|
| `Ctrl+S` | Save project (prompts for path on first save) |
| `Ctrl+O` | Open project (lists .mdma files or prompts for path) |
| `Ctrl+N` | New project (prompts for name) |
| `Ctrl+R` | Re-run last command (or execute highlighted text) |
| `Ctrl+K` | Kill to end of line |
| `Ctrl+U` | Kill whole line |
| `Ctrl+W` | Kill word backward |
| `Ctrl+Y` | Yank (paste from kill ring) |
| `Tab` | Auto-complete command names |

### Auto-Save (Immediate Persistence)

All definition data is auto-saved immediately when modified, not on
project save or render:

| Data | Trigger | Storage |
|------|---------|---------|
| SyDef definitions | `/syt`, `/sydef` end, delete | `~/Documents/MDMA/sydefs.json` |
| User functions | `/fn`, `/def` end | `~/Documents/MDMA/user_functions.json` |
| Named chains | `/chain` add, modify | `~/Documents/MDMA/chains.json` |
| Command history | Every command | `~/.mdma_history` |

Auto-save also triggers on `/import` (which may bring in new definitions)
and on exit (`/q`).

### /syt — Synth Block Definitions

`/syt` is now documented as the primary command for defining synth
blocks (alias for `/sydef`).  The `[syt:name]>` prompt appears during
block recording.

---

## Version 42 — BPM-Locked Timing, Parser Fix, Interval Range, /use Cleanup

**Release Date:** 2026-02-02  
**Build ID:** bmdma_v42.0  
**Total Commands:** 558  
**DSP Effects:** 113 (was 96)  
**Effect Aliases:** 167 (was 135)

### Auto-Persistence System

SyDefs, user functions (`/fn`), and named chains now **auto-save
immediately** to `~/Documents/MDMA/` when created, modified, or
deleted — no need to `/save` first.

| Data | File | Auto-saved on |
|------|------|---------------|
| SyDef definitions | `sydefs.json` | `/end` of block, `/sydef del`, copy, load |
| User functions | `user_functions.json` | `/end` of `/fn` block, exit |
| Named chains | `chains.json` | Chain modify, exit |

All three are auto-loaded on startup.  Factory SyDef presets are
excluded from the user file (they're regenerated on boot).

### Project Save/Load Enhanced

`/save` and `/load` now persist:
- Track audio (base64, with non-silent detection)
- SyDef definitions (user-created)
- User function blocks
- Named chain definitions
- Working buffer + source tag
- All existing data (buffers, clips, operators, FX chains, etc.)

### Readline-Powered REPL

| Keybinding | Action |
|------------|--------|
| **Ctrl+S** | Save project (same as `/save`) |
| **Ctrl+O** | Open project (file picker if no path) |
| **Ctrl+K** | Clear current input line |
| **Ctrl+R** | Re-run last command |
| **Tab** | Autocomplete command names |
| **Up/Down** | History navigation (persisted to `~/.mdma_history`) |
| **Ctrl+C** | Cancel / interrupt |

Copy/paste uses standard terminal bindings (Ctrl+Shift+C/V on
Linux, Cmd+C/V on macOS).  History is saved across sessions.

### /syt — Synth Block Alias

`/syt` is now an alias for `/sydef`.  Use it to define reusable
synth blocks:

```
/syt my_pad freq=440 width=50
  /op 1
  /wm sine
  /fr $freq
  /vc 4
  /dt 2
/end
```

The prompt shows `[syt:name]>` during definition.

### Granular Effect Presets (7 new effects)

Granular synthesis effects that process the entire source buffer (not
just a slice).  Each preset tunes grain size, density, spread, pitch
and envelope for a distinct character:

| Effect | Alias | Description |
|--------|-------|-------------|
| granular_cloud | granular, grain, cloud, gr1 | Dense ethereal texture |
| granular_scatter | scatter, gr2 | Sparse random grains |
| granular_stretch | grstretch, gr3 | Time-stretch to 2x duration |
| granular_freeze | grfreeze, gr4 | Sustains source midpoint |
| granular_shimmer | grshimmer, gr5 | Pitch-shifted sparkle overlay |
| granular_reverse | grrev, gr6 | 70% reverse-grain probability |
| granular_stutter | grstutter, gr7 | Tiny glitchy repetition |

Usage: `/fx granular`, `/fx scatter`, `/fx gr5`

### Utility Effects (10 new effects)

| Effect | Alias | Description |
|--------|-------|-------------|
| util_normalize | normalize, norm | Peak normalize to -1dB |
| util_normalize_rms | normrms, lufs | RMS normalize to -14 LUFS |
| util_declip | declip | Repair hard-clipped regions |
| util_declick | declick | Remove transient clicks/pops |
| util_smooth | smooth, warmth | Gentle LP at 8kHz |
| util_smooth_heavy | smoothheavy, muffle | Heavy LP at 4kHz |
| util_dc_remove | dc, dcremove | Remove DC offset (HP 10Hz) |
| util_fade_in | fadein | 50ms fade-in |
| util_fade_out | fadeout | 50ms fade-out |
| util_fade_both | fade, fades | 50ms fade in+out |

Usage: `/fx normalize`, `/fx declip`, `/fx smooth`, `/fx fade`

### Track Bounce: /btw

Bounce entire track audio into the working buffer, process it,
then write it back — the fastest way to apply effects to an
existing track.

```
/btw               Bounce current track -> working
/btw 2             Bounce track 2 -> working
/btw back          Write working back (overwrite)
/btw back add      Write back additively
```

Typical workflow:
```
/btw               Pull track into working
/fx reverb         Process
/fx normalize      Fix levels
/btw back          Put it back
```

### Playback: External Player Removed

`render_cmds.cmd_play` now uses the in-house playback engine
(`session.play()`) instead of writing to WAV and calling
`xdg-open`/`open`/`startfile`.  File-based fallback remains
available only if the audio backend is missing.

### Interval Range Extended to ±24

`note_to_hz()` now treats -24 through 24 as semitone intervals
from root.  Values outside that range are MIDI note numbers.

Old behaviour: 0-11 = interval, 12+ = MIDI
New behaviour: -24…24 = interval, outside = MIDI

This means you can now write:

| Value | Meaning | Frequency (root=440) |
|-------|---------|---------------------|
| 0     | Root    | 440 Hz              |
| 7     | 5th     | 659 Hz              |
| 12    | Octave  | 880 Hz              |
| 19    | Oct+5th | 1319 Hz             |
| 24    | 2 oct   | 1760 Hz             |
| -12   | Oct below | 220 Hz            |
| -24   | 2 oct below | 110 Hz          |
| 60    | MIDI C4 | 262 Hz              |
| 69    | MIDI A4 | 440 Hz              |

Patterns like `/mel -12.0.12.24` (two-octave sweep) and
`/cor 0,12,19` (root+octave+5th chord) now work correctly.

### BPM-Locked Timing Grid

All pattern commands now quantize to BPM:

    1 dot-unit = 1 beat = 60/BPM seconds

| BPM | 1 beat | /mel 0.4.7 (3 beats) |
|-----|--------|----------------------|
| 60  | 1.000s | 3.000s               |
| 120 | 0.500s | 1.500s               |
| 140 | 0.429s | 1.286s               |
| 240 | 0.250s | 0.750s               |

Affected commands: /mel, /cor, /wa mel, /wa cor, /ns (/out block),
/tone (already BPM-locked via generate_tone).

### Dot-As-Separator Parser Rewrite

BREAKING: Dots are now note/chord separators, not duration extenders.

Old behaviour (v41-):
  0.4.7 → note 0 (2 beats), note 4 (2 beats), note 7 (1 beat) = 5 beats

New behaviour (v42):
  0.4.7 → note 0 (1 beat), note 4 (1 beat), note 7 (1 beat) = 3 beats

To hold a note longer, use extra consecutive dots:
  0..4.7 → note 0 (2 beats), note 4 (1), note 7 (1) = 4 beats
  0...4..7 → note 0 (3 beats), note 4 (2), note 7 (1) = 6 beats

This applies to both melody patterns (/mel) and chord patterns (/cor):
  0,4,7.0,3,7     → 2 chords × 1 beat = 2 beats
  0,4,7..0,3,7    → first chord 2 beats, second 1 = 3 beats
  0,4,7...0,3,7.. → first 3 beats, second 2 = 5 beats

Rests also work consistently:
  0.r.4    → note, rest, note (1 beat each = 3 beats)
  0.r..4   → note (1), rest (2), note (1) = 4 beats

### Timing Pipeline Summary

  /bpm 140        → sets global clock (60/140 = 0.429s per beat)
  /mel 0.4.7      → 3 notes × 1 beat = 3 × 0.429s = 1.286s
  /cor 0,4,7..    → 1 chord × 2 beats = 2 × 0.429s = 0.857s
  /tone 440 2     → 2 beats at BPM = 0.857s
  /ns (out block) → each command = 1 beat at BPM

### /use Is Now Purely SyDef

`/use` (and `/u`) now only instantiates synth definitions.
The old `synth`/`chain` sub-commands have been removed:

- `/use synth 1` → no longer valid (use `/synth` for DSL synth selection)
- `/use chain test` → no longer valid (use `/chain` for FX chains)
- `/use pad` → instantiate the "pad" SyDef ✓
- `/use acid 440 1200` → SyDef with params ✓

This eliminates the conflict where `/use` had overloaded meaning
for both synth engine configuration and effects routing.

### Effect System Audit & Fix

**Validation**: All effect commands now validate names before adding.
Invalid names throw a clear error with "Did you mean?" suggestions:

```
/fx foobar        -> ERROR: Unknown effect 'foobar'. Use /hfx to see all.
/fx revert        -> ERROR: Unknown effect 'revert'. Did you mean: reverb, reverse
/chain_id add xyz -> ERROR: Unknown effect 'xyz'. Did you mean: ...
```

**Changes**:
- Removed `FX_ALIAS_MAP` (conflicting duplicate). `_effect_aliases` is now
  the single source of truth (135 aliases → 96 DSP effects).
- Added `resolve_effect_name()` helper used by all FX commands:
  `/fx`, `/fxa`, `/fxq`, chain `add`, and slot commands.
- Chain `add` now validates effect names (was silently accepting anything).
- Fuzzy matching suggests close names on typos via `difflib.get_close_matches`.

**DSP Audit Results**:
- 96/96 registered effects confirmed working (audio changes verified).
- Full pipeline test: `/fx reverb`, `/fx delay`, `/fx compress`,
  `/fx distort`, `/fx chorus`, `/fx bitcrush` all modify audio correctly.
- `stereo_wide`/`stereo_narrow` return (N,2) stereo — pipeline handles this.

### Track Append Commands: /ta and /wta

New commands for writing audio directly to tracks, mirroring the
existing /wa (working append) conventions.

`/ta` appends audio to the current track at the cursor:
  /ta                    Append working buffer to track
  /ta tone 440 2         Append 2-beat tone to track
  /ta mel 0.4.7          Append melody to track
  /ta cor 0,4,7.0,3,7    Append chords to track
  /ta s 4                Append 4 beats silence
  /ta add mel 0.7.12     Additive (sum into existing audio)

`/wta` commits working buffer to track and clears working:
  /wta                   Commit to current track
  /wta add               Commit additive
  /wta 2                 Commit to track 2
  /wta 2 add             Commit to track 2, additive

All subcommands support sydef= and quantize to BPM.

### Stats

- **556 commands** (up from 554)
- 96 effects
- 44 generators
- 14 factory SyDef presets
- 3 voice algorithms (stack, unison, wide)
- BPM-locked timing for all pattern/melody/chord commands

---

## Version 41 — Voice Algorithms, FX Playback, Simple Presets

**Release Date:** 2026-02-02  
**Build ID:** bmdma_v41.0  
**Total Commands:** 554

- Voice algorithm system: stack (0), unison (1), wide (2)
- Default algorithm = unison (prevents phase-locking)
- rand in unison mode spreads phase + amplitude
- FX-aware playback: play(), full_render(), _play_buffer(), _play_deck()
- Stereo-aware generate_tone (ADSR, filters, effects per-channel)
- 14 factory SyDef presets (9 simple + 5 full)
- New aliases: /vc, /ps, /ss
- Session passes stereo_spread, voice_phase_offset, voice_algorithm to engine

---

## Version 40 — Stereo Pipeline, SyDef System, Track Controls

**Release Date:** 2026-02-02  
**Build ID:** bmdma_v40.0  
**Total Commands:** 550+

- Full stereo pipeline rewrite (N,2 track arrays, equal-power pan law)
- SyDef system with block-based parameterized synth patches
- /mel and /cor use full synth engine with SyDef support
- Track commands: /tgain, /tpan, /tmute, /tsolo, /tinfo
- Fixed: play() scope, new_track() schema, /tracks clear crash

---

## Version 39 — MAD DSL v2, Live Loops, Buffer Overhaul

**Release Date:** 2026-02-01  
**Build ID:** bmdma_v39.0  
**Total Commands:** 510+

- /play searches last_buffer, working_buffer, numbered buffers
- Live Loops (/live, /kill, /ka)
- /render, /loop, /mutate, /b buffer overview

---

## Version 38 — Macro Timing, Audio-Rate Modulation, Umpulse System

- Macro Timing System (@now, @beat, @bar, @delay:N)
- Audio-Rate Modulation (/audiorate, /ar)

---

## Previous Versions

See full history in VERSION_HISTORY.md
