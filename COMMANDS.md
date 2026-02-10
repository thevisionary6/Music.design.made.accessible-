# MDMA Command Reference

## Quick Reference

Use `/help` in MDMA to see this help interactively.
Use `/help <category>` for category details.
Use `/help <command>` for specific command help.

---

## Smart Playback

Context-aware playback that automatically detects what to play.

| Command | Description |
|---------|-------------|
| `/P` | Smart play - plays based on current context (buffer/deck/working) |
| `/PB [n]` | Play buffer explicitly |
| `/PD [n]` | Play deck explicitly |
| `/PW` | Play working buffer |
| `/stop` | Stop playback |

---

## Working Buffer & Append (v30)

The working buffer is your audio cursor. All generators and processing goes here first.
Use `/A` to append/commit to permanent storage.

| Command | Description |
|---------|-------------|
| `/WB` | Show working buffer status |
| `/A` | Append working buffer to current buffer |
| `/A <n>` | Append to buffer n |
| `/A B <n>` | Append to buffer explicitly |
| `/A D <n>` | Append to deck |
| `/A replace` | Replace instead of append |

### Workflow
```
/g_kick 1        # Generate kick -> working buffer
/A               # Append to current buffer
/g_snare 2       # Generate snare -> working buffer
/A 2             # Append to buffer 2
```

---

## Generators (v30)

All generators output to working buffer. Use `/A` to append to destination.

### Drums
| Command | Types | Description |
|---------|-------|-------------|
| `/g_kick [1-4]` | 808, punchy, sub, acoustic | Kick drum |
| `/g_snare [1-4]` | 808, acoustic, tight, clicky | Snare |
| `/g_hat [1-4]` | closed, open, pedal, crispy | Hi-hat |
| `/g_tom [1-4]` | low, mid, high, electronic | Tom |
| `/g_clp [1-4]` | 808, acoustic, layered, tight | Clap |
| `/g_cym [1-4]` | crash, ride, china, splash | Cymbal |
| `/g_snp [1-2]` | dry, reverb | Finger snap |
| `/g_shk [1-3]` | 16th, 8th, triplet | Shaker |

### FX & Sounds
| Command | Types | Description |
|---------|-------|-------------|
| `/g_zap [1-4]` | short, long, sweep, retro | Zap sound |
| `/g_lsr [1-3]` | pew, beam, charge | Laser FX |
| `/g_bel [1-4]` | bright, dark, tubular, glass | Bell/ping |
| `/g_bas [1-4]` | 808, sub, punch, growl | Bass hit |
| `/g_rsr [1-4]` | noise, tonal, sweep, tension | Riser |
| `/g_dwn [1-4]` | drop, sweep, impact, reverse | Downlifter |
| `/g_wsh [1-3]` | fast, slow, textured | Whoosh |
| `/g_glt [1-4]` | stutter, buffer, bitcrush, random | Glitch |
| `/g_vnl [1-2]` | crackle, hiss | Vinyl texture |
| `/g_wnd [1-3]` | gentle, harsh, filtered | Wind/noise |
| `/g_blp [1-4]` | short, pitched, glitch, zap | Bleep/blip |
| `/g_stb [1-4]` | chord, brass, synth, orchestral | Stab |

### Utility
| Command | Parameters | Description |
|---------|------------|-------------|
| `/g_sil [duration]` | seconds | Silence/spacer |
| `/g_clk [bpm] [bars]` | BPM, bar count | Click track |
| `/g_cal [freq] [dur]` | Hz, seconds | Calibration tone |
| `/g_swp [f1] [f2] [dur]` | start Hz, end Hz, seconds | Frequency sweep |

---

## Voice Parameters (v30)

Control voice spread, phase, and envelopes.

| Command | Description |
|---------|-------------|
| `/stereo <0-100>` | Voice stereo spread (0=mono, 100=full width) |
| `/vphase <0-360>` | Voice phase offset (degrees per voice) |
| `/venv <ms>` | Voice envelope offset (stagger for humanization) |
| `/fenv <a> <d> <s> <r>` | Frequency envelope (pitch modulation) |
| `/menv <a> <d> <s> <r>` | Modulation envelope (FM depth over time) |

### Examples
```
/stereo 75          Wide stereo spread
/vphase 45          45° phase offset per voice
/fenv 0.01 0.1 0 0.1 amount 12   Pitch envelope +12 semitones
/menv 0.05 0.2 0.5 0.3           Mod envelope
```

---

## Persistence (v30)

Save and restore user data.

| Command | Description |
|---------|-------------|
| `/save` | Save all user data (variables, macros, prefs) |
| `/USRP` | List user outputs (render backups) |
| `/USRP <id>` | Load output by ID to working buffer |

Renders are automatically backed up to `~/Documents/MDMA/outputs/`

---

## Legacy Working Buffer

Temporary audio workspace for testing effects without modifying source.

| Command | Description |
|---------|-------------|
| `/W [src]` | Load audio to working buffer (buffer/deck) |
| `/WBC` | Clear working buffer |
| `/WFX <effect> [params]` | Add effect with deviation metrics |
| `/WFX list` | List available effects |
| `/WFX undo` | Undo last effect |

### Effect Examples
```
/WFX dist 50       Distortion at 50%
/WFX lpf 2000      Lowpass at 2kHz
/WFX reverb 0.3    Reverb at 30% mix
/WFX delay 250 0.5 Delay 250ms @ 50% feedback
/WFX norm          Normalize to 0dB
```

---

## Auto-Chunking

Automatically segment audio using various algorithms.

| Command | Description |
|---------|-------------|
| `/CHK [algo] [num]` | Auto-chunk current buffer |
| `/CHK use <n>` | Load chunk n to buffer |
| `/CHK use all` | Concatenate all chunks |

### Algorithms
- `auto` - Automatic selection (default)
- `transient` - Split at transients
- `beat` - Split at detected beats
- `zero` - Split at zero crossings
- `equal` - Equal-sized chunks
- `wavetable` - Single-cycle extraction
- `energy` - Energy-based segmentation

---

## Remix

Remix audio using AI and deterministic algorithms.

| Command | Description |
|---------|-------------|
| `/remix [algo] [intensity]` | Remix current buffer |

### Algorithms
- `shuffle` - Random chunk shuffle (default)
- `reverse` - Reverse random chunks
- `stutter` - Add stutter effects
- `glitch` - Glitch/noise effects
- `chop` - Beat chop/rearrange
- `layer` - Layer chunks
- `evolve` - AI-style evolution
- `granular` - Granular reconstruction

Intensity: 0.0 - 1.0 (default 0.5)

---

## Rhythmic Patterns (RPAT)

Apply rhythmic patterns to audio - plays audio at pattern positions.

| Command | Description |
|---------|-------------|
| `/RPAT <pattern> [beats]` | Apply pattern to current audio |

### Pattern Formats
```
x.x.x.x.         Binary (x=hit, .=rest)
1010             Numeric (1=hit, 0=rest)
1 0.5 0 0.8      Space-separated velocities (0-1)
h.m.l.           Letters (h=high 0.9, m=medium 0.6, l=low 0.3)
```

### Examples
```
/RPAT x.x.x.x.        Basic 8th notes
/RPAT x..x..x.        Syncopated pattern
/RPAT 1 0.5 0 0.8 4   Velocity pattern, 4 beats
```

---

## Buffer Combining

| Command | Description |
|---------|-------------|
| `/CBI <i> <j> ...` | Combine buffers by overlay |
| `/BAP <src> [dst]` | Append src buffer to dst |

---

## User Variables

Store and retrieve values across commands.

| Command | Description |
|---------|-------------|
| `/= name value` | Set variable |
| `/GET [name]` | Get variable (or list all) |
| `/GET name.key` | Get nested value (dict key or list index) |
| `/DEL name` | Delete variable |
| `/DEL *` | Clear all variables |

### Examples
```
/= mybpm 128         Set numeric
/= pattern x.x.x.x.  Set string
/= mylist [1,2,3]    Set list
/GET mybpm           Get value -> 128
/GET mylist.0        Get index -> 1
```

---

## Bridge Commands

Transfer audio between systems.

| Command | Description |
|---------|-------------|
| `/PR [buf] [deck]` | Print (copy) buffer to deck |
| `/YT <url> [buf]` | Download YouTube to buffer |
| `/SC <url> [buf]` | Download SoundCloud to buffer |
| `/DK2BUF [deck] [buf]` | Copy deck to buffer |

---

## Wavetable

| Command | Description |
|---------|-------------|
| `/WT [frames] [size]` | Generate wavetable from audio |

---

## DJ Mode & Mixing

### Mode Control
| Command | Aliases | Description |
|---------|---------|-------------|
| `/djm` | `/dj`, `/djmode` | Toggle/control DJ mode |

### Playback Control
| Command | Aliases | Description |
|---------|---------|-------------|
| `/play` | `/p`, `/start` | Start deck playback |
| `/stop` | `/pause` | Stop deck playback |

### Deck Selection
| Command | Aliases | Description |
|---------|---------|-------------|
| `/deck` | `/dk`, `/d` | Select active deck (1-4) |
| `/deck+` | `/dk+` | Add new deck |
| `/deck-` | `/dk-` | Remove deck |

### Tempo & Timing
| Command | Aliases | Description |
|---------|---------|-------------|
| `/tempo` | `/bpm`, `/t` | Get/set deck tempo (20-300 BPM) |
| `/sync` | `/sy` | Sync deck tempos |

### Volume & Mix
| Command | Aliases | Description |
|---------|---------|-------------|
| `/vol` | `/volume`, `/v` | Get/set deck volume (0-100) |
| `/vol master` | - | Set master volume |
| `/cf` | `/crossfader`, `/xfader` | Get/set crossfader (0-100) |
| `/cf left` | - | Full left (deck 1) |
| `/cf right` | - | Full right (deck 2) |
| `/cf center` | - | Center (50/50) |
| `/xfade` | `/xf`, `/cross` | Crossfade between decks |

### Filter
| Command | Aliases | Description |
|---------|---------|-------------|
| `/fl` | `/flt`, `/filter` | Filter cutoff (1-100) |
| `/flr` | `/res`, `/q` | Filter resonance (1-100) |

### Navigation
| Command | Aliases | Description |
|---------|---------|-------------|
| `/j` | `/jump`, `/goto` | Jump to position |
| `/j start` | - | Jump to start |
| `/j drop` | - | Jump to drop |
| `/j <beat>` | - | Jump to beat number |
| `/j <time>` | - | Jump to time (e.g., 1:30) |

### Effects
| Command | Aliases | Description |
|---------|---------|-------------|
| `/dfx` | `/deckfx` | Apply deck effect |
| `/dfx echo` | - | Echo effect |
| `/dfx flanger` | - | Flanger effect |
| `/dfx phaser` | - | Phaser effect |
| `/dfx crush` | - | Bitcrush effect |
| `/dfx filter` | - | Filter sweep |
| `/dfx vamp` | - | Vamp overdrive |
| `/dfx reverb` | - | Reverb effect |
| `/scr` | `/scratch` | Trigger scratch preset (1-5) |
| `/stud` | `/stutter` | Trigger stutter |

### Loops
| Command | Aliases | Description |
|---------|---------|-------------|
| `/lpc` | `/loopcount` | Set loop repeat count |
| `/lpg` | `/lg`, `/loopgo` | Jump to loop start |
| `/dur` | `/duration`, `/time` | Set effect duration |

### Transitions
| Command | Aliases | Description |
|---------|---------|-------------|
| `/tran` | `/tr`, `/trans`, `/x` | Quick transition |
| `/transition` | - | Full transition control |
| `/drop` | `/drp`, `/!` | Instant drop |

### Cue Points
| Command | Aliases | Description |
|---------|---------|-------------|
| `/cue` | `/c` | Set/trigger cue point |

### Stems
| Command | Aliases | Description |
|---------|---------|-------------|
| `/stem` | `/st`, `/stems` | Stem separation/control |
| `/stem sep <deck>` | - | Separate into stems |
| `/stem <deck> <stem> <vol>` | - | Set stem volume |
| `/stem <deck> solo <stem>` | - | Solo a stem |

### Sections
| Command | Aliases | Description |
|---------|---------|-------------|
| `/section` | `/sec` | Navigate to section |
| `/chop` | `/slice` | Chop audio into sections |

### Streaming
| Command | Aliases | Description |
|---------|---------|-------------|
| `/stream` | `/str`, `/sc` | Stream audio source |

#### Stream Auto-Registration
When you stream from SoundCloud or YouTube:
- Audio is automatically saved to `~/Documents/MDMA/songs/downloads/`
- Full analysis runs (BPM, key, energy, quality)
- Song is registered for instant future loading via `/reg load`

```bash
# Stream and auto-register
/str 1 https://soundcloud.com/artist/track
# Output: OK: streamed 'Track Name' (180.5s) to Deck 1
#         Registered as song #42 - use /reg load 42 anytime

# Load again instantly
/reg load 42
```

### AI Enhancement
| Command | Aliases | Description |
|---------|---------|-------------|
| `/ai` | - | Toggle AI audio enhancement |
| `/ai on` | - | Enable AI enhancement |
| `/ai off` | - | Disable AI enhancement |
| `/ai status` | - | Show enhancement stats |
| `/ai passes <n>` | - | Set multi-pass count (1-5) |
| `/enhance` | `/enh` | Enhancement settings |
| `/enhance <preset>` | - | Set preset (transparent/master/broadcast/loud) |
| `/enhance bright <v>` | - | Set brightness (-1 to +1) |
| `/enhance warm <v>` | - | Set warmth (-1 to +1) |
| `/enhance punch <v>` | - | Set transient punch (0 to 1) |

### Devices
| Command | Aliases | Description |
|---------|---------|-------------|
| `/do` | `/devices`, `/dev` | List audio devices |
| `/doc` | `/master` | Set master output device |
| `/hep` | `/headphones`, `/phones` | Headphone output |

### Screen Reader
| Command | Aliases | Description |
|---------|---------|-------------|
| `/sr` | `/nvda`, `/reader` | Screen reader settings |

### Safety
| Command | Aliases | Description |
|---------|---------|-------------|
| `/fallback` | `/fb`, `/safe` | Enable fallback mode |

### Library
| Command | Aliases | Description |
|---------|---------|-------------|
| `/library` | `/lib` | Track library |
| `/playlist` | `/pl` | Playlist management |
| `/reg` | `/registry`, `/songs` | Song registry system |

---

## Song Registry & Quality Assurance

The song registry provides permanent storage of your music library with automatic quality analysis, conversion, and tagging.

### Scanning
| Command | Description |
|---------|-------------|
| `/reg scan <folder>` | Scan folder for songs (recursive) |
| `/reg rescan` | Re-check all registered songs |

### Browsing
| Command | Description |
|---------|-------------|
| `/reg` | Show registry stats |
| `/reg list [query]` | List/search songs |
| `/reg info <id>` | Show song details |
| `/reg high` | List high-quality songs only |
| `/reg recent` | Show recently added |
| `/reg played` | Show most played |
| `/reg favs` | Show favorites |

### Filtering
| Command | Description |
|---------|-------------|
| `/reg bpm <range>` | Filter by BPM (e.g., 120-140) |
| `/reg key <key>` | Filter by musical key (e.g., Am, C, "F# minor") |
| `/reg genre <genre>` | Filter by genre hint |
| `/reg quality` | Show quality breakdown |

### Loading
| Command | Description |
|---------|-------------|
| `/reg load <id>` | Load song to active deck |
| `/reg load <name>` | Load by name search |
| `/reg load <id> <deck>` | Load to specific deck |

### Metadata
| Command | Description |
|---------|-------------|
| `/reg tag <id> <tags...>` | Add tags to song |
| `/reg rate <id> <1-5>` | Rate song (1-5 stars) |
| `/reg fav <id>` | Toggle favorite status |

### Management
| Command | Description |
|---------|-------------|
| `/reg fix <id>` | Re-analyze and fix quality issues |
| `/reg remove <id>` | Remove from registry |
| `/reg cache` | Show cache status |
| `/reg cache rebuild` | Rebuild all cache files |
| `/reg verify` | Verify registry integrity and fix issues |

### Quality Grades

| Grade | Description | Criteria |
|-------|-------------|----------|
| **HIGH** | Ready for performance | 320kbps+ or lossless, no clipping, good dynamic range |
| **MEDIUM** | Usable with minor issues | 128-256kbps, minor clipping or compression |
| **LOW** | Not recommended | < 128kbps, severe clipping, mono, or major issues |

### Auto-Analysis

Songs are automatically analyzed for:
- **BPM** - Beat detection with confidence score
- **Key** - Musical key detection
- **Energy** - Overall loudness/intensity (0-100%)
- **Danceability** - Rhythmic content score
- **Genre hints** - Based on BPM and characteristics
- **Mood hints** - Based on energy and tempo
- **Quality issues** - Clipping, DC offset, dynamic range

### WAV64 Conversion

High and medium quality songs are automatically converted to:
- 48kHz sample rate
- 32-bit float
- Stereo
- Cached for instant loading

### Examples

```bash
# Scan your music folder
/reg scan ~/Music

# Browse songs
/reg list
/reg list aphex
/reg bpm 120-140
/reg key Am
/reg genre techno

# Load to deck
/reg load 42
/reg load "selected ambient" 1

# Rate and tag
/reg rate 42 5
/reg tag 42 idm ambient favorite
/reg fav 42

# Check quality
/reg quality
/reg high
/reg info 42
```

---

## Effects & Processing

### Quick Effect Shortcuts
| Command | Effect Name |
|---------|-------------|
| `/r1` | reverb_small |
| `/r2` | reverb_large |
| `/r3` | reverb_plate |
| `/r4` | reverb_spring |
| `/r5` | reverb_cathedral |
| `/d1` | delay_simple |
| `/d2` | delay_pingpong |
| `/d3` | delay_multitap |
| `/d4` | delay_slapback |
| `/d5` | delay_tape |
| `/s1` | saturate_soft |
| `/s2` | saturate_hard |
| `/s3` | saturate_overdrive |
| `/s4` | saturate_fuzz |
| `/s5` | saturate_tube |
| `/v1` | vamp_light |
| `/v2` | vamp_medium |
| `/v3` | vamp_heavy |
| `/v4` | vamp_fuzz |
| `/l1` | lofi_bitcrush |
| `/l2` | lofi_chorus |
| `/l3` | lofi_flanger |
| `/l4` | lofi_phaser |
| `/l5` | lofi_filter |
| `/l6` | lofi_halftime |
| `/c1` | compress_mild |
| `/c2` | compress_hard |
| `/c3` | compress_limiter |
| `/c4` | compress_expander |
| `/c5` | compress_softclipper |
| `/g1-g5` | gate patterns |

### Granular Effect Presets
| Alias | Effect | Description |
|-------|--------|-------------|
| `/fx granular` | granular_cloud | Dense ethereal cloud |
| `/fx scatter` | granular_scatter | Sparse random grains |
| `/fx grstretch` | granular_stretch | Time-stretch 2x |
| `/fx grfreeze` | granular_freeze | Sustain midpoint |
| `/fx grshimmer` | granular_shimmer | Pitch-shifted sparkle |
| `/fx grrev` | granular_reverse | Reversed grains |
| `/fx grstutter` | granular_stutter | Tiny glitch repeat |
| `/fx gr1`-`gr7` | (numbered shortcuts) | Same order as above |

### Utility Effects
| Alias | Effect | Description |
|-------|--------|-------------|
| `/fx normalize` | util_normalize | Peak normalize -1dB |
| `/fx lufs` | util_normalize_rms | RMS normalize -14 LUFS |
| `/fx declip` | util_declip | Repair clipped audio |
| `/fx declick` | util_declick | Remove clicks/pops |
| `/fx smooth` | util_smooth | Gentle LP 8kHz |
| `/fx muffle` | util_smooth_heavy | Heavy LP 4kHz |
| `/fx dc` | util_dc_remove | Remove DC offset |
| `/fx fadein` | util_fade_in | 50ms fade-in |
| `/fx fadeout` | util_fade_out | 50ms fade-out |
| `/fx fade` | util_fade_both | 50ms fade in+out |

### Effect Control
| Command | Aliases | Description |
|---------|---------|-------------|
| `/fx` | - | Apply effect to buffer |
| `/fxl` | - | List all effects |
| `/fxa` | - | Apply effect with amount |
| `/amt` | - | Set effect wet/dry amount |

### Filter Slots
| Command | Aliases | Description |
|---------|---------|-------------|
| `/sfc` | - | Set filter slot count |
| `/sf` | - | Select active filter slot |
| `/e0` - `/e9` | - | Quick filter slot select |

### Specific Effects
| Command | Description |
|---------|-------------|
| `/vamp` | Vamp/overdrive effect |
| `/reverb`, `/verb` | Reverb effect |
| `/delay` | Delay effect |
| `/saturate`, `/sat` | Saturation effect |
| `/compress`, `/comp` | Compression effect |
| `/lofi` | Lo-fi effect |
| `/gate` | Gate effect |
| `/shimmer` | Shimmer reverb |
| `/conv` | Convolution reverb |

### EQ/Filter
| Command | Aliases | Description |
|---------|---------|-------------|
| `/lp` | `/lpf` | Low-pass filter |
| `/hp` | `/hpf` | High-pass filter |
| `/bp` | - | Band-pass filter |

### Modulation
| Command | Description |
|---------|-------------|
| `/chorus` | Chorus effect |
| `/flanger` | Flanger effect |
| `/phaser` | Phaser effect |

### Distortion
| Command | Aliases | Description |
|---------|---------|-------------|
| `/od` | `/overdrive` | Overdrive |
| `/dist` | `/distort` | Distortion |
| `/fuzz` | - | Fuzz effect |
| `/crush` | `/bitcrush` | Bitcrusher |

### Time Effects
| Command | Description |
|---------|-------------|
| `/half` | Half-speed effect |
| `/double` | Double-speed effect |
| `/reverse`, `/rev` | Reverse audio |

### Chain
| Command | Description |
|---------|-------------|
| `/chain` | Effect chain management |
| `/bypass` | Bypass effects |
| `/dry` | Set dry level |

### Vocoder & Spectral (v29.1)
| Effect Name | Description |
|-------------|-------------|
| `vocoder_synth` | Vocoder with sawtooth carrier |
| `vocoder_noise` | Vocoder with noise carrier (robot voice) |
| `vocoder_chord` | Vocoder with chord carrier |
| `spc_freeze` | Freeze spectrum (drone effect) |
| `spc_blur` | Spectral blur/smear |
| `spc_shift_up` | Pitch up 5 semitones |
| `spc_shift_down` | Pitch down 5 semitones |

### LFO & Stereo Effects (v29.1)
| Effect Name | Description |
|-------------|-------------|
| `lfo_filter_slow` | Slow filter sweep |
| `lfo_filter_fast` | Fast wah-wah sweep |
| `lfo_tremolo` | Amplitude tremolo |
| `lfo_vibrato` | Pitch vibrato |
| `stereo_wide` | Wide stereo spread |
| `stereo_narrow` | Collapse to mono |

---

## Synthesis & Sound Design

### Oscillator
| Command | Aliases | Description |
|---------|---------|-------------|
| `/wave` | - | Set waveform (sine/saw/square/tri/noise) |
| `/osc` | - | Oscillator settings |
| `/freq` | - | Set frequency (Hz) |
| `/note` | - | Set note (c4, a#3, etc.) |
| `/det` | `/detune` | Detune amount |

### Envelope (ADSR)
| Command | Aliases | Description |
|---------|---------|-------------|
| `/att` | `/atk` | Attack time |
| `/dec` | - | Decay time |
| `/sus` | - | Sustain level |
| `/rel` | - | Release time |

### Filter
| Command | Aliases | Description |
|---------|---------|-------------|
| `/cut` | `/cutoff` | Filter cutoff |
| `/res` | `/reso` | Filter resonance |
| `/fenv` | - | Filter envelope |
| `/fatk` | - | Filter attack |
| `/fdec` | - | Filter decay |
| `/frel` | - | Filter release |

### Modulation
| Command | Description |
|---------|-------------|
| `/lfo` | LFO settings |
| `/mod` | Modulation amount |
| `/fm` | FM synthesis |
| `/pm` | Phase modulation |

### FM/Operators
| Command | Aliases | Description |
|---------|---------|-------------|
| `/op` | - | Select operator |
| `/car` | - | Carrier settings |
| `/alg` | - | FM algorithm |
| `/ratio` | - | Operator ratio |
| `/fb` | `/feedback` | FM feedback |

### Generation
| Command | Aliases | Description |
|---------|---------|-------------|
| `/gen` | - | Generate sound |
| `/render` | `/rn` | Render to buffer |

### Presets
| Command | Aliases | Description |
|---------|---------|-------------|
| `/preset` | `/pre` | Load/save preset |
| `/bank` | `/bk` | Preset bank |

---

## AI & Generation

### Enhancement
| Command | Aliases | Description |
|---------|---------|-------------|
| `/ai` | - | Toggle AI enhancement |
| `/enhance` | `/enh` | Enhancement settings |

### Generation
| Command | Description |
|---------|-------------|
| `/gen` | AI audio generation |
| `/prompt` | Set generation prompt |
| `/style` | Set generation style |
| `/seed` | Set random seed |

### Analysis
| Command | Description |
|---------|-------------|
| `/analyze` | Analyze audio |
| `/describe` | AI describe sound |
| `/classify` | Classify audio type |

### Breeding
| Command | Description |
|---------|-------------|
| `/breed` | Breed two sounds |
| `/mutate` | Mutate sound |
| `/evolve` | Evolve sound population |

---

## Buffer & Audio Management

### Buffer Control
| Command | Aliases | Description |
|---------|---------|-------------|
| `/buf` | `/b` | Select buffer |
| `/new` | - | Create new buffer |
| `/clear` | `/clr` | Clear buffer |
| `/copy` | `/cp` | Copy buffer |
| `/len` | - | Get/set buffer length |

### Operations
| Command | Aliases | Description |
|---------|---------|-------------|
| `/norm` | `/normalize` | Normalize audio |
| `/gain` | `/g` | Apply gain |
| `/fade` | - | Apply fade in/out |
| `/trim` | - | Trim silence |
| `/crop` | - | Crop to selection |

### Analysis
| Command | Description |
|---------|-------------|
| `/peak` | Show peak level |
| `/rms` | Show RMS level |
| `/dc` | Remove DC offset |

---

## Pattern & Sequencing

| Command | Aliases | Description |
|---------|---------|-------------|
| `/pat` | - | Apply pattern |
| `/apat` | - | Advanced pattern |
| `/seq` | - | Sequence editor |
| `/euclid` | - | Euclidean rhythm |
| `/poly` | - | Polyrhythm |
| `/swing` | - | Apply swing |
| `/grid` | - | Set grid resolution |
| `/quant` | - | Quantize |
| `/step` | - | Step sequencer |

---

## Playback & Recording

| Command | Aliases | Description |
|---------|---------|-------------|
| `/play` | `/p` | Play audio |
| `/stop` | - | Stop playback |
| `/pause` | - | Pause playback |
| `/loop` | - | Toggle loop mode |
| `/pos` | - | Get/set position |
| `/seek` | - | Seek to time |
| `/rewind` | `/rw` | Rewind |
| `/rec` | - | Start recording |
| `/load` | - | Load project from .mdma file |
| `/save` | - | Save project (all state, audio, definitions) |
| `/import` | - | Import audio/data files |
| `/export` | - | Export audio |

### Project Commands (v43)

| Command | Description |
|---------|-------------|
| `/save` | Save project to .mdma (full state) |
| `/save <path>` | Save to specific path |
| `/load <path>` | Load project from .mdma |
| `/import <path>` | Import audio (.wav) to working buffer |
| `/import <path> track` | Import audio to current track |
| `/import <path.json>` | Import SyDef/chain/function definitions |
| `/import <path.mdma>` | Merge definitions from another project |
| `/new [name]` | Create new project/sketch |

### Keybindings

| Key | Action |
|-----|--------|
| `Ctrl+S` | Save project |
| `Ctrl+O` | Open/load project |
| `Ctrl+N` | New project |
| `Ctrl+R` | Re-run last command |
| `Ctrl+K` | Kill to end of line |
| `Ctrl+U` | Kill whole line |
| `Ctrl+W` | Kill word backward |
| `Ctrl+Y` | Yank (paste from kill ring) |
| `Tab` | Auto-complete commands |

---

## Performance & Live

| Command | Description |
|---------|-------------|
| `/perf` | Performance mode |
| `/snap` | Snapshot state |
| `/recall` | Recall snapshot |
| `/macro` | Create macro |
| `/meter` | Level meter |
| `/panic` | Audio panic (stop all) |
| `/mute` | Mute output |
| `/solo` | Solo channel |

---

## Help System

| Command | Aliases | Description |
|---------|---------|-------------|
| `/help` | `/h`, `/?` | Show command help |
| `/help <category>` | - | Show category commands |
| `/help <command>` | - | Show specific command help |
| `/help all` | - | Show all commands |

---

## Scratch Presets

| Preset | Name | Description |
|--------|------|-------------|
| 1 | Baby | Smooth sine wave motion |
| 2 | Forward | Quick push, slow return |
| 3 | Chirp | Short fader cuts |
| 4 | Transformer | Gated on/off |
| 5 | Crab | Rapid flutter |

---

## AI Enhancement Presets

| Preset | Target LUFS | Dynamics | Use Case |
|--------|-------------|----------|----------|
| `transparent` | -16.0 | Gentle | Minimal processing |
| `master` | -14.0 | Balanced | Default mastering |
| `broadcast` | -14.0 | Balanced | Streaming optimized |
| `loud` | -11.0 | Aggressive | Maximum loudness |

---

## Examples

```bash
# Start DJ mode
/djm on

# Load and play
/deck 1
/play

# Set tempo and crossfade
/tempo 128
/cf center

# Apply effects
/dfx vamp 2
/scr 3

# Jump around
/j drop
/j 32    # Beat 32

# AI enhancement
/ai on
/enhance loud

# Filter sweep
/fl 20
/fl 100
```

---

## Advanced Audio Operations (v28+)

### Auto-Chunking
Automatically split audio into chunks for wavetables, sample slicing, or remix.

| Command | Description |
|---------|-------------|
| `/CHK` | Auto-chunk current buffer (algorithm: auto) |
| `/CHK <algo>` | Chunk with specific algorithm |
| `/CHK <algo> <n>` | Chunk into N pieces |
| `/CHK use <idx>` | Load chunk by index to buffer |
| `/CHK use all` | Concatenate all chunks |

**Algorithms:**
- `auto` - Automatic selection based on content
- `transient` - Split at transients (percussive)
- `beat` - Split at detected beats
- `zero` - Split at zero crossings
- `equal` - Equal-sized chunks
- `wavetable` - Single-cycle extraction
- `energy` - Energy-based segmentation
- `spectral` - Spectral similarity

### Remix
Remix audio using various algorithms.

| Command | Description |
|---------|-------------|
| `/remix` | Shuffle remix at 50% intensity |
| `/remix <algo>` | Remix with algorithm |
| `/remix <algo> <int>` | Remix with intensity (0-1) |

**Algorithms:**
- `shuffle` - Random chunk shuffle
- `reverse` - Reverse random chunks
- `stutter` - Add stutter effects
- `glitch` - Glitch/noise effects
- `chop` - Beat chop/rearrange
- `layer` - Layer chunks
- `evolve` - AI-style evolution
- `granular` - Granular reconstruction

### Rhythmic Pattern (RPAT)
Apply rhythmic patterns to audio for drum programming.

| Command | Description |
|---------|-------------|
| `/RPAT <pattern>` | Apply pattern using global duration |
| `/RPAT <pattern> <beats>` | Apply pattern for N beats |

**Pattern Formats:**
- Binary: `x.x.x.x.` (x=hit, .=rest)
- Numeric: `10101010` (1=hit, 0=rest)
- Velocity: `1 0.5 0 0.8` (space-separated 0-1)
- Letters: `h.m.l.` (h=high 0.9, m=medium 0.6, l=low 0.3)

**Examples:**
```bash
# Basic 8th notes
/RPAT x.x.x.x.

# Syncopated kick
/RPAT x..x..x.

# Velocity pattern (4 beats)
/RPAT "1 0.5 0 0.8" 4

# Hi-hat pattern with accents
/RPAT h.l.m.l.h.l.m.l.
```

### Buffer Combining
Combine multiple buffers into one.

| Command | Description |
|---------|-------------|
| `/CBI <idx> <idx> ...` | Overlay/mix buffers |
| `/CBI <idx> + <idx>` | Append buffers |
| `/BAP <src> [dst]` | Append src buffer to dst |

### Wavetable Generation
Generate wavetables from audio.

| Command | Description |
|---------|-------------|
| `/WT` | Generate 256x2048 wavetable |
| `/WT <frames>` | Custom frame count |
| `/WT <frames> <size>` | Custom frames and size |

---

## User Variables

Store and retrieve values for use in commands, functions, and macros.

| Command | Description |
|---------|-------------|
| `/= name value` | Set variable |
| `/GET name` | Get variable value |
| `/GET name.key` | Get dict key or list index |
| `/GET` | List all variables |
| `/DEL name` | Delete variable |
| `/DEL *` | Clear all variables |

**Examples:**
```bash
# Set values
/= mybpm 128
/= mydict {a:1,b:2}
/= mylist [10,20,30]

# Get values
/GET mybpm        # 128
/GET mydict.a     # 1
/GET mylist.1     # 20
```

---

## Advanced Macros

Macros with argument support for reusable command sequences.

| Command | Description |
|---------|-------------|
| `/MC new <n> [args]` | Create macro with named arguments |
| `/MC run <n> [vals]` | Execute macro with values |
| `/MC list` | List all macros |
| `/MC show <n>` | Show macro commands |
| `/MC del <n>` | Delete macro |

**Creating Macros:**
```bash
# Create macro with bpm and vol arguments
/MC new kick bpm vol
[def:kick]> /bpm $bpm
[def:kick]> /g sk
[def:kick]> /amp $vol
[def:kick]> /end

# Run with values
/MC run kick 140 0.8
```

**Argument Substitution:**
- `$argname` - Named argument
- `$1`, `$2` - Positional arguments
- `$varname` - User variable (from /=)

---

## Bridge Commands

Connect different subsystems (buffers, decks, streaming).

| Command | Description |
|---------|-------------|
| `/PR [buf] [deck]` | Print buffer to DJ deck |
| `/YT <url> [buf]` | Download YouTube to buffer |
| `/SC <url> [buf]` | Download SoundCloud to buffer |
| `/DK2BUF [deck] [buf]` | Copy deck audio to buffer |

**Examples:**
```bash
# Copy buffer 1 to deck 2
/PR 1 2

# Download YouTube to buffer
/YT https://youtu.be/...

# Copy deck to buffer for processing
/DK2BUF 1 3
/remix glitch 0.5
/PR 3 1
```

---

## Track System (v40-v42)

Tracks are continuous stereo audio lanes. Audio is written at a cursor
that advances automatically.  Tracks are mixed with per-track gain, pan,
mute/solo and FX chains.

### Track Management

| Command | Description |
|---------|-------------|
| `/tracks` | List all tracks with duration, peak, pan, mute/solo |
| `/tracks clear` | Reset to 1 empty track |
| `/tn` | Create a new track and select it |
| `/ti` | List all tracks (short) |
| `/ti <n>` | Select track by 1-based index |
| `/tsel <n>` | Select track (1-based) |
| `/tlen` | Show project length |
| `/tlen <sec>` | Set project length in seconds (resets audio) |
| `/rc` | Clear (silence) current track, reset cursor |

### Writing Audio to Tracks

| Command | Description |
|---------|-------------|
| `/twrite` | Write last_buffer to track at cursor (overwrite) |
| `/twrite add` | Write last_buffer to track at cursor (additive/sum) |
| `/tpos` | Show cursor position |
| `/tpos <sec>` | Set cursor position in seconds |

### Track Append — /ta

Append audio directly to track at cursor. Like `/wa` but goes to
the timeline instead of the working buffer.  Cursor advances after
each write.

| Command | Description |
|---------|-------------|
| `/ta` | Append working buffer to track |
| `/ta tone <hz> [beats]` | Append synth tone to track |
| `/ta t <hz> [beats]` | Short for /ta tone |
| `/ta silence <beats>` | Append silence |
| `/ta s <beats>` | Short for /ta silence |
| `/ta mel <pat> [hz]` | Append melody pattern to track |
| `/ta m <pat> [hz]` | Short for /ta mel |
| `/ta cor <pat> [hz]` | Append chord sequence to track |
| `/ta c <pat> [hz]` | Short for /ta cor |
| `/ta add tone ...` | Additive mode (sum into existing audio) |
| `/ta add mel ...` | Additive melody |

All `/ta` subcommands support `sydef=<name>` for SyDef patches:
```
/ta mel 0.4.7.12 sydef=acid
/ta cor 0,4,7.0,3,7 sydef=pad
```

### Working → Track Commit — /wta

Commit the working buffer to the current track and clear working.
This is the track equivalent of `/wa` (which commits to numbered buffers).

| Command | Description |
|---------|-------------|
| `/wta` | Commit working to current track (overwrite) |
| `/wta add` | Commit working to current track (additive) |
| `/wta <n>` | Commit working to track n (1-based) |
| `/wta <n> add` | Commit to track n, additive |

### Track Mixing

| Command | Description |
|---------|-------------|
| `/tgain` | Show current track gain |
| `/tgain <dB>` | Set gain (0=unity, -6=half, +6=double) |
| `/tgain <n> <dB>` | Set gain for track n |
| `/tpan` | Show current pan |
| `/tpan <val>` | Set pan (-100=left, 0=center, 100=right) |
| `/tpan <n> <val>` | Set pan for track n |
| `/tpan L` / `C` / `R` | Pan shortcuts |
| `/tmute` | Toggle mute on current track |
| `/tmute <n>` | Toggle mute on track n |
| `/tsolo` | Toggle solo on current track |
| `/tsolo <n>` | Toggle solo on track n |
| `/tinfo` | Detailed info for current track |
| `/tinfo <n>` | Detailed info for track n |

### Timeline View

| Command | Description |
|---------|-------------|
| `/tl` | Show timeline overview (all tracks) |
| `/tl <n>` | Show specific track |
| `/tl beats` | Show with beat positions |
| `/tl samples` | Show with sample positions |

### Track ↔ Working Bounce — /btw

| Command | Description |
|---------|-------------|
| `/btw` | Bounce current track → working buffer |
| `/btw <n>` | Bounce track n → working buffer |
| `/btw back` | Write working → track (overwrite from pos 0) |
| `/btw back add` | Write working → track (additive) |

### Typical Track Workflow

```
/bpm 120                    Set tempo
/tlen 30                    30 second project
/tsel 1                     Select track 1

# Build directly on track
/ta mel 0.4.7.12            Melody -> track at cursor
/ta s 2                     2 beats silence
/ta cor 0,4,7..0,3,7        Chords with hold

# Or build in working buffer, then commit
/mel 0.4.7.0.3.7            Build melody in working
/wta                         Commit to track

# Bounce for processing
/btw                         Pull track into working
/fx reverb                   Apply reverb
/fx normalize                Fix levels
/btw back                    Put it back

# Switch tracks for layering
/tn                          New track
/ta mel 0.12.7 sydef=pad     Pad melody on track 2

# Mix
/tsel 1
/tgain -3                    Track 1 down 3dB
/tsel 2
/tpan 30                     Track 2 slightly right
/play                        Hear the mix
```

---

## Melody & Chord Patterns (v40-v42)

### /mel — Melody

| Command | Description |
|---------|-------------|
| `/mel <pat>` | Render melody at 440Hz -> working |
| `/mel <pat> <hz>` | Melody at custom root |
| `/mel <pat> sydef=<n>` | Melody using SyDef patch |

Pattern syntax: dots separate notes, extra dots extend.
Values -24 to 24 = semitone intervals, outside = MIDI.

```
/mel 0.4.7              3 notes x 1 beat (root, 3rd, 5th)
/mel 0..4.7             Root held 2 beats
/mel -12.0.12.24        Two-octave sweep
/mel 0.4.7 sydef=acid   With acid SyDef
```

### /cor — Chords

| Command | Description |
|---------|-------------|
| `/cor <pat>` | Render chord sequence at 440Hz -> working |
| `/cor <pat> <hz>` | Chords at custom root |
| `/cor <pat> sydef=<n>` | Chords using SyDef patch |

```
/cor 0,4,7.0,3,7        Major then minor (1 beat each)
/cor 0,4,7..0,3,7       Major held 2 beats, minor 1
/cor 0,4,7.r.0,3,7      Major, rest, minor
```

### /wa — Working Buffer Append

| Command | Description |
|---------|-------------|
| `/wa` | Commit working to lowest empty buffer |
| `/wa <n>` | Commit working to buffer n |
| `/wa tone <hz> [beats]` | Append tone to working |
| `/wa t <hz> [beats]` | Short for /wa tone |
| `/wa silence <beats>` | Append silence to working |
| `/wa s <beats>` | Short for /wa silence |
| `/wa mel <pat> [hz]` | Append melody to working |
| `/wa cor <pat> [hz]` | Append chord sequence to working |

### /use — SyDef Instantiation (v42)

| Command | Description |
|---------|-------------|
| `/use` | List available SyDef presets |
| `/use <name>` | Instantiate SyDef with defaults |
| `/use <name> <v1> <v2>` | Positional param overrides |
| `/use <name> p=val` | Named param overrides |
| `/use <name> 4x` | Instantiate 4 times (layering) |

14 factory presets: saw, square, sub, lead, bass, bell, string,
nz, hihat, sine, acid, pad, pluck, kick.

### Voice & Synthesis

| Command | Description |
|---------|-------------|
| `/va` | Show/set voice algorithm (stack/unison/wide) |
| `/v` or `/vc` | Voice count |
| `/vphase` or `/ps` | Phase spread (degrees) |
| `/stereo` or `/ss` | Stereo spread (0-100) |

### Timing

| Command | Description |
|---------|-------------|
| `/bpm` | Show/set BPM (all patterns quantize to this) |
| `/step` | Show/set step length in beats |

1 pattern unit = 1 beat = 60/BPM seconds.
