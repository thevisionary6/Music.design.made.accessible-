# MDMA v14.1 - Complete Release Notes

---

# MDMA v37 - DJ Stabilization Pack (2026-01-29)

Focus: DJ reliability + stereo output + performance-buffer style deck behavior.

- Fix /d<N> shorthand (e.g., /d2) to reliably select decks (no alias collisions).
- DJ engine now preserves stereo buffers and outputs true stereo (2-channel stream).
- Add adjustable DJ blocksize via /djm bs <n> (stability vs latency).
- Add Master Loop Wrap (/ml on|off) to wrap decks when content runs out.
- Add Loop Hold (/lh) infinite looping without removing loop-count /lg.
- Add Master Deck FX chain (/mdfx) always-on post-processing for the DJ master bus.
- Per-deck timed effects state (deck effects no longer stomp each other).

---

**Version:** 14.1
**Build ID:** mdma_v14.1_20260127
**Date:** 2026-01-27
**Status:** ✅ STABLE RELEASE

---

## Summary

MDMA v14.1 implements two major feature chunks:
- **Chunk 1:** Unified 1-100 parameter scaling across all modules
- **Chunk 2:** Audio-rate pattern modulation system

---

## Chunk 1: Unified Parameter Scaling

### New Module: `dsp/scaling.py`

Single source of truth for parameter scaling functions.

**Scaling Functions (1-100 → internal):**

| Function | Output Range | Purpose |
|----------|--------------|---------|
| `scale_drive()` | 1.0-20.0 | Drive/saturation multiplier |
| `scale_wet()` | 0.0-1.0 | Wet/dry mix |
| `scale_feedback()` | 0.0-0.95 | Delay/reverb feedback |
| `scale_resonance()` | 0.1-20.0 | Filter Q (log curve) |
| `scale_modulation_index()` | 0.0-10.0 | FM/AM/PM depth |
| `scale_threshold_db()` | -60 to 0dB | Compressor threshold |
| `scale_ratio()` | 1:1 to 20:1 | Compression ratio |
| `scale_bits()` | 16 to 2 bits | Bit reduction |

**Preset Names:**

| Preset | Value |
|--------|-------|
| `off` | 0 |
| `subtle` | 15 |
| `light` | 25 |
| `moderate` / `default` | 50 |
| `strong` | 65 |
| `heavy` | 85 |
| `max` / `extreme` | 100 |
| `wacky` | 120 |
| `insane` | 175 |
| `broken` | 200 |

### Updated: `dsp/effects.py`

**New Direct Helpers:**
```python
from mdma_rebuild.dsp.effects import fx_buffer, fx_file, list_effects

# Apply effects to buffer
result = fx_buffer(audio, ['vamp_medium', 'reverb_plate'], amount=75)

# Apply effects to file
fx_file('input.wav', ['saturate_tube'], 'output.wav', amount=60)

# List available effects
effects = list_effects()  # Returns 51 effects
```

**Effects with 1-100 Scaling:**
- `vamp_process()` - drive, gain, mix
- `dual_overdrive()` - drive_low, drive_high, blend
- `compress()` - threshold, ratio, makeup
- `gate()` - threshold
- `bitcrush()` - bits, downsample, mix

### Updated: `dsp/monolith.py`

**Voice Algorithm Parameters (1-100 scaled):**
| Param | Description | Scaling |
|-------|-------------|---------|
| `rand` | Amplitude variation | 0-100 → 0.0-1.0 |
| `mod` | Modulation depth | 0-100 → 0.0-1.0 |
| `stereo_spread` | Stereo width | 0-100 → 0.0-1.0 |
| `resonance` | Filter Q | 0-100 → 0.1-20.0 |

**Real Units Preserved:**
| Param | Units |
|-------|-------|
| `dt` | Hz (detune) |
| `phase_spread` | radians |
| `cutoff` | Hz |

**Modulation Amount:**
- `add_algorithm(amount)` - 0-100 scaled → 0-10 modulation index
- `add_algorithm_raw()` - Direct index control for advanced users

### Updated: `core/session.py`

**Session Defaults:**
- `filter_resonances` default: 50.0 (was 0.5)
- `rand` default: 0.0 (0-100 scale)
- `v_mod` default: 0.0 (0-100 scale)

**New Properties:**
- `session.resonance` - Get current filter resonance (0-100)
- `session.cutoff` - Get current filter cutoff (Hz)

**Updated Methods:**
- `set_resonance(value)` - Accepts 0-100 or preset names
- `set_rand(value)` - 0-100 scale with validation
- `set_mod(value)` - 0-100 scale with validation

### Updated: `commands/fx_cmds.py`

**Command Output Format Changes:**
- Filter commands show `res=` instead of `Q=`
- Example: `FILTER[0]: ON | lowpass | cut=1000Hz | res=50`

---

## Chunk 2: Audio-Rate Pattern Modulation

### New Module: `dsp/pattern.py`

**Classes:**
- `PatternNote` - Single note in sequence (pitch, rest, hold, glide)
- `PatternClip` - Pattern with source audio and note sequence
- `PlaybackMode` - oneshot, loop, pingpong, reverse
- `NoteType` - pitch, rest, hold, glide

**Parsing Functions:**
```python
from mdma_rebuild.dsp.pattern import parse_pattern, pattern_to_string

# Parse pattern tokens
notes = parse_pattern('0 7 12 R D G5')
# Produces: [Pitch(0), Pitch(7), Pitch(12), Rest, Hold, Glide(5)]

# Convert back to string
pattern_str = pattern_to_string(notes)
# Returns: "0 7 12 R D G5"
```

**Pattern Token Syntax:**
| Token | Meaning |
|-------|---------|
| `0`, `7`, `-5` | Pitch offset in semitones |
| `R`, `.`, `-` | Rest (silence) |
| `D`, `D3` | Hold/extend previous note (D3 = 3 steps) |
| `G7` | Glide to semitone 7 |
| `V80:5` | Velocity 80, pitch 5 |

**Rendering Functions:**
```python
from mdma_rebuild.dsp.pattern import quick_pattern, arpeggiate, render_pattern

# Quick pattern application
result = quick_pattern(source_audio, '0 7 12 5', bpm=120, step=0.25)

# Arpeggio from chord
arp = arpeggiate(source_audio, [0, 4, 7, 12], repeats=2)

# Full control with PatternClip
clip = PatternClip(
    source_buffer=audio,
    notes=parse_pattern('0 4 7 12'),
    bpm=120,
    step=0.25
)
result = render_pattern(clip, mix=100)
```

**Pattern Presets (18 total):**

*Scales:*
- `major_up`, `major_down`
- `minor_up`, `minor_down`
- `chromatic`

*Arpeggios:*
- `major_arp`, `minor_arp`
- `maj7_arp`, `min7_arp`, `dom7_arp`

*Rhythmic:*
- `pulse`, `offbeat`, `dotted`

*Melodic:*
- `bounce`, `rise`, `fall`
- `octaves`, `fifths`

### New Module: `commands/pattern_cmds.py`

**Pattern Commands:**

| Command | Description |
|---------|-------------|
| `/pat <tokens>` | Apply pattern to buffer |
| `/arp <chord>` | Quick arpeggio |
| `/patlist` | List pattern presets |
| `/patinfo` | Show current settings |
| `/reverse` | Reverse buffer |
| `/chop <indices>` | Rearrange steps |
| `/stretch <factor>` | Time-stretch |
| `/pitch <semitones>` | Pitch-shift |

**Chord Presets for /arp:**
- `maj`, `min`, `dim`, `aug`
- `maj7`, `min7`, `dom7`
- `sus2`, `sus4`, `add9`, `6`, `m6`

---

## Files Changed

### New Files
| File | Description |
|------|-------------|
| `dsp/scaling.py` | Parameter scaling system |
| `dsp/pattern.py` | Pattern modulation system |
| `commands/pattern_cmds.py` | Pattern commands |

### Modified Files
| File | Changes |
|------|---------|
| `dsp/__init__.py` | Added pattern module |
| `dsp/effects.py` | 1-100 scaling, direct helpers |
| `dsp/monolith.py` | Voice algorithm scaling |
| `core/session.py` | Session defaults, properties |
| `commands/__init__.py` | Added pattern_cmds |
| `commands/fx_cmds.py` | Filter command format |

---

## Breaking Changes

### Parameter Scaling Migration

| Parameter | Old Default | New Default | Migration |
|-----------|-------------|-------------|-----------|
| resonance | 0.5 (Q) | 50 | Multiply by ~25 |
| rand | 0.0-1.0 | 0-100 | Multiply by 100 |
| v_mod | 0.0-1.0 | 0-100 | Multiply by 100 |
| drive | 1.0-20.0 | 0-100 | ~5x old value |

### Filter Output Format
- Old: `FILTER[0]: ON | lowpass | cut=1000Hz | Q=2.50`
- New: `FILTER[0]: ON | lowpass | cut=1000Hz | res=50`

---

## Quick Start Examples

### Basic Pattern Usage
```bash
/tone 440 1 0.8        # Generate 440Hz tone
/pat 0 7 12 5          # Apply major arpeggio pattern
/play                  # Preview result
```

### Arpeggio
```bash
/tone 220 0.5 0.8      # Generate bass tone
/arp min7 4            # Minor 7th arpeggio, 4 repeats
```

### Preset Pattern
```bash
/pat preset:major_up   # Major scale ascending
```

### Effects with Scaling
```bash
/fx vamp_medium 75     # Vamp effect at 75
/sfr 60                # Resonance at 60
```

---

## Test Summary

```
✅ All DSP modules import correctly
✅ Scaling functions work (51 effects available)
✅ Pattern parsing (6 token types)
✅ Pattern presets (18 presets)
✅ quick_pattern() rendering
✅ arpeggiate() rendering
✅ Pattern commands (13 commands)
✅ Buffer manipulation (reverse, pitch, stretch, chop)
✅ Session defaults updated
✅ Effect helpers (fx_buffer, fx_file)
```

---

## Dependencies

```
Python 3.10+
numpy>=1.20.0
scipy>=1.7.0
soundfile>=0.10.0 (optional)
```

**Quick Install:**
```bash
pip install numpy scipy soundfile
```

---

## Credits

MDMA - Music Design Made Accessible
Built for vision-optional audio production
