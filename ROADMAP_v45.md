# MDMA v45 Implementation Roadmap

## Overview

Three major feature areas to implement:
1. **Playback System Overhaul** - Transparent, consistent in-house playback
2. **Parameter System** - Explicit parameter management with modulation
3. **Audio Quality Improvements** - High-quality rendering and output

---

## 1. PLAYBACK SYSTEM OVERHAUL

### Current State Analysis
- Multiple `cmd_p` definitions scattered across modules (buffer_cmds, dj_cmds, render_cmds)
- `PlaybackContext` singleton exists in `playback_cmds.py` but not consistently used
- No unified playback routing - different commands go different paths
- Track playback not well-defined
- Working buffer playback exists (`/pw`) but `/p` alone doesn't default to it

### Proposed Architecture

#### 1.1 Playback Router (New: `playback_cmds.py`)
```
/p              → Always play WORKING BUFFER (clear output: "▶ Playing: working buffer (2.5s)")
/pb <n>         → Play buffer N
/pt <n>         → Play track N
/pts            → Play SONG (all tracks mixed)
/pd <n>         → Play deck N
/pall           → Play all buffers mixed
```

#### 1.2 In-House Playback Engine
- **Single playback function**: `play_audio(audio, source_label, session)`
- Uses `sounddevice` exclusively (no subprocess calls)
- Always outputs: `"▶ Playing: {source} ({duration}s) @ {sample_rate}Hz"`
- Handles stereo/mono conversion automatically
- Applies master chain if enabled

#### 1.3 Implementation Steps

| Step | File | Changes |
|------|------|---------|
| 1.1 | `playback_cmds.py` | Create `_unified_play(audio, label, session)` function |
| 1.2 | `playback_cmds.py` | Rewrite `cmd_p()` to ONLY play working buffer |
| 1.3 | `playback_cmds.py` | Add `cmd_pt(session, args)` for track playback |
| 1.4 | `playback_cmds.py` | Add `cmd_pts(session, args)` for song (all tracks) playback |
| 1.5 | `playback_cmds.py` | Update `cmd_pb()`, `cmd_pd()` to use unified player |
| 1.6 | `__init__.py` | Update command registration to remove duplicates |
| 1.7 | Remove | Delete duplicate `cmd_p` from `buffer_cmds.py`, `render_cmds.py` |

#### 1.4 Track Mixing for Song Playback
```python
def mix_all_tracks(session) -> np.ndarray:
    """Mix all tracks respecting gain, pan, mute, solo."""
    # Get max length
    # Apply per-track gain/pan
    # Respect mute/solo flags
    # Sum to stereo master
    # Apply master chain if present
    return mixed_audio
```

#### 1.5 Output Format
All playback commands output consistently:
```
▶ Playing: working buffer (2.50s) @ 48000Hz
▶ Playing: buffer 3 (1.25s) @ 48000Hz
▶ Playing: track 1 "Drums" (30.00s) @ 48000Hz
▶ Playing: song [4 tracks] (30.00s) @ 48000Hz
▶ Playing: deck 2 (45.30s) @ 48000Hz
```

---

## 2. PARAMETER SYSTEM OVERHAUL

### Current State Analysis
- Parameters scattered across `session` object (attack, decay, cutoff, etc.)
- No unified listing or categorization
- No modulation assignment system
- `RandomnessController` exists in `performance.py` but limited
- `LFO` class exists in `effects.py` but not integrated with parameters
- `ADSREnvelope` exists but only for audio shaping, not parameter modulation

### Proposed Architecture

#### 2.1 Parameter Categories
```
SYNTH:
  1. attack      (0.001 - 5.0 sec)
  2. decay       (0.001 - 5.0 sec)
  3. sustain     (0.0 - 1.0)
  4. release     (0.001 - 10.0 sec)
  5. amp         (0.0 - 1.0)
  6. dt          (0.0 - 100.0 Hz) - detune
  7. rand        (0 - 100) - voice randomization
  
FILTER:
  1. cutoff      (20 - 20000 Hz)
  2. resonance   (0 - 100)
  3. env_amount  (0 - 100) - filter envelope depth

GLOBAL:
  1. bpm         (20 - 300)
  2. master_vol  (0.0 - 2.0)
  3. pan         (-1.0 - 1.0)

VOICE:
  1. voice_count (1 - 16)
  2. v_mod       (0 - 100)
  
OPERATOR (per-operator):
  1. ratio       (0.01 - 32.0)
  2. level       (0.0 - 1.0)
  3. feedback    (0.0 - 1.0)
```

#### 2.2 New File: `param_cmds.py`

##### Data Structures
```python
@dataclass
class ParameterDef:
    name: str
    category: str
    min_val: float
    max_val: float
    default: float
    unit: str  # "Hz", "sec", "%", ""
    getter: Callable  # lambda session: session.attack
    setter: Callable  # lambda session, v: setattr(session, 'attack', v)

@dataclass 
class ParameterModulation:
    mod_type: str  # 'lfo', 'envelope', 'random', 'none'
    # LFO params
    lfo_shape: str = 'sin'
    lfo_rate: float = 1.0
    lfo_depth: float = 50.0
    # Envelope params
    env_attack: float = 0.1
    env_decay: float = 0.2
    env_sustain: float = 0.5
    env_release: float = 0.3
    env_depth: float = 100.0
    # Random params
    rand_min: float = 0.0
    rand_max: float = 1.0
    rand_mode: str = 'uniform'  # 'uniform', 'gaussian', 'walk'

class ParameterSystem:
    """Singleton managing all parameters."""
    _instance = None
    
    def __init__(self):
        self.params: Dict[str, ParameterDef] = {}
        self.modulations: Dict[str, ParameterModulation] = {}
        self.selected_param: Optional[str] = None
        self._register_all_params()
```

##### Commands

**`/PARM` - List/Select Parameters**
```
/parm                    → List all categories
/parm synth              → List SYNTH parameters
/parm synth 1            → Select attack (synth param #1)
/parm filter 2           → Select resonance (filter param #2)
/parm ?                  → Show currently selected parameter
```

**`/CHA` - Change Parameter**
```
/cha 0.5                 → Set selected param to 0.5
/cha +0.1                → Add 0.1 to current value
/cha -0.1                → Subtract 0.1
/cha *2                  → Multiply by 2
/cha /2                  → Divide by 2
/cha rand                → Random value in param's range
/cha rand 0.2 0.8        → Random between 0.2 and 0.8
/cha lfo sin 2 50        → Assign LFO (shape, rate Hz, depth %)
/cha lfo tri 0.5 30      → Triangle LFO at 0.5Hz, 30% depth
/cha env 0.1 0.2 0.5 0.3 → Assign envelope (A, D, S, R in seconds)
/cha env 0.1 0.2 0.5 0.3 80  → With depth %
/cha clear               → Remove modulation
```

#### 2.3 Modulation Application

Modulations are applied at render time:
```python
def get_modulated_value(param_name: str, base_value: float, 
                        sample_idx: int, total_samples: int) -> float:
    """Get parameter value with modulation applied."""
    mod = param_system.modulations.get(param_name)
    if not mod or mod.mod_type == 'none':
        return base_value
    
    if mod.mod_type == 'lfo':
        lfo = LFO(mod.lfo_shape, mod.lfo_rate, mod.lfo_depth)
        mod_value = lfo.generate(1)[0]  # Single sample
        return base_value * (1 + mod_value)
    
    if mod.mod_type == 'envelope':
        # Calculate envelope position
        position = sample_idx / total_samples
        env_value = calculate_adsr(position, mod)
        return base_value + (env_value * mod.env_depth / 100)
    
    if mod.mod_type == 'random':
        return random_in_range(mod.rand_min, mod.rand_max, mod.rand_mode)
```

#### 2.4 Implementation Steps

| Step | File | Changes |
|------|------|---------|
| 2.1 | `param_cmds.py` | Create new file with ParameterDef, ParameterModulation, ParameterSystem |
| 2.2 | `param_cmds.py` | Register all parameters from session |
| 2.3 | `param_cmds.py` | Implement `cmd_parm()` for listing/selecting |
| 2.4 | `param_cmds.py` | Implement `cmd_cha()` for changing/modulating |
| 2.5 | `session.py` | Add `param_system` reference |
| 2.6 | `session.py` | Modify `generate_tone()` to apply modulations |
| 2.7 | `effects.py` | Modify effects to respect parameter modulation |
| 2.8 | `__init__.py` | Register new commands |

#### 2.5 Random System Audit

Current random implementations found:
- `RandomnessController` in `performance.py` - good foundation
- `np.random.randn()` scattered throughout
- `rng` command in `perf_cmds.py`

**Unified Random System:**
```python
class UnifiedRNG:
    """Centralized random number generation with reproducibility."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.locked = False
    
    def uniform(self, low: float, high: float) -> float:
        return self._rng.uniform(low, high)
    
    def gaussian(self, mean: float, std: float) -> float:
        return self._rng.normal(mean, std)
    
    def walk(self, current: float, step_size: float) -> float:
        return current + self._rng.uniform(-step_size, step_size)
    
    def choice(self, options: list):
        return self._rng.choice(options)
    
    def reseed(self, seed: int = None):
        self.seed = seed or int(time.time() * 1000) % (2**32)
        self._rng = np.random.default_rng(self.seed)
```

---

## 3. AUDIO QUALITY IMPROVEMENTS

### Current State Analysis
- Already using float64 internally (after v44 fix)
- WAV output is standard format
- No FLAC support for output
- No high-quality filtering on output
- Waveforms can have harsh high-end
- No DC offset removal or ultra-low filtering

### Proposed Improvements

#### 3.1 Output Format Support

**Add FLAC output to `/b`, `/bo`, `/ba` commands:**
```
/b                   → WAV output (default)
/b flac              → FLAC output
/b wav 24            → 24-bit WAV
/b flac 24           → 24-bit FLAC
```

Implementation:
```python
def save_audio(audio: np.ndarray, path: str, sr: int, 
               format: str = 'wav', bit_depth: int = 16):
    """Save audio with format selection."""
    import soundfile as sf
    
    subtype_map = {
        ('wav', 16): 'PCM_16',
        ('wav', 24): 'PCM_24',
        ('wav', 32): 'FLOAT',
        ('flac', 16): 'PCM_16',
        ('flac', 24): 'PCM_24',
    }
    
    subtype = subtype_map.get((format, bit_depth), 'PCM_16')
    sf.write(path, audio, sr, format=format.upper(), subtype=subtype)
```

#### 3.2 High-Quality Render Chain

New render pipeline applied before final output:

```python
def hq_render_chain(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply high-quality processing before output."""
    
    # 1. DC Offset Removal
    audio = remove_dc_offset(audio)
    
    # 2. Ultra-Low Cut (subsonic filter at 20Hz)
    audio = highpass_filter(audio, cutoff=20, sr=sr, order=4)
    
    # 3. High-End Smoothing (gentle rolloff above 16kHz)
    audio = gentle_highshelf(audio, freq=16000, gain_db=-1.5, sr=sr)
    
    # 4. Analog-Style Saturation (subtle warmth)
    audio = soft_saturation(audio, drive=0.1)
    
    # 5. Final Limiting (prevent clipping)
    audio = soft_limiter(audio, threshold=0.95)
    
    return audio
```

#### 3.3 Waveform Improvements

**Smoother Oscillators in `monolith.py`:**
```python
def generate_smooth_saw(freq: float, duration: float, sr: int) -> np.ndarray:
    """Band-limited sawtooth using additive synthesis."""
    t = np.linspace(0, duration, int(duration * sr), dtype=np.float64)
    nyquist = sr / 2
    max_harmonic = int(nyquist / freq)
    
    wave = np.zeros_like(t)
    for k in range(1, min(max_harmonic, 64) + 1):
        wave += ((-1) ** (k + 1)) * np.sin(2 * np.pi * k * freq * t) / k
    
    return wave * (2 / np.pi)

def generate_smooth_square(freq: float, duration: float, sr: int) -> np.ndarray:
    """Band-limited square using additive synthesis."""
    t = np.linspace(0, duration, int(duration * sr), dtype=np.float64)
    nyquist = sr / 2
    max_harmonic = int(nyquist / freq)
    
    wave = np.zeros_like(t)
    for k in range(1, min(max_harmonic, 64) + 1, 2):  # odd harmonics only
        wave += np.sin(2 * np.pi * k * freq * t) / k
    
    return wave * (4 / np.pi)
```

**Anti-Aliased Waveforms:**
- Use PolyBLEP for real-time
- Use additive synthesis for rendered output
- Configurable via session flag: `session.hq_oscillators = True`

#### 3.4 Implementation Steps

| Step | File | Changes |
|------|------|---------|
| 3.1 | `render_cmds.py` | Add format/bit-depth args to `/b`, `/bo`, `/ba` |
| 3.2 | `effects.py` | Add `hq_render_chain()` function |
| 3.3 | `effects.py` | Add `remove_dc_offset()`, `gentle_highshelf()`, `soft_saturation()`, `soft_limiter()` |
| 3.4 | `buffer_cmds.py` | Update `cmd_b()` to use `save_audio()` with format selection |
| 3.5 | `monolith.py` | Add band-limited oscillator options |
| 3.6 | `session.py` | Add `hq_mode: bool = True` flag |
| 3.7 | `session.py` | Add `output_format: str = 'wav'` and `output_bit_depth: int = 16` |

#### 3.5 Quality Settings Command

```
/hq                  → Show HQ settings
/hq on               → Enable HQ render chain
/hq off              → Disable (raw output)
/hq format wav       → Set output format
/hq format flac      → Set output format  
/hq bits 24          → Set bit depth
/hq osc smooth       → Enable band-limited oscillators
/hq osc fast         → Use standard oscillators
```

---

## Implementation Order

### Phase 1: Playback (Est. 300 lines)
1. Create unified playback function
2. Rewrite `/p` to play working buffer only
3. Add `/pt`, `/pts` commands
4. Remove duplicate playback commands
5. Test all playback paths

### Phase 2: Parameters (Est. 600 lines)
1. Create `param_cmds.py` with data structures
2. Register all existing parameters
3. Implement `/parm` command
4. Implement `/cha` command (basic)
5. Add LFO modulation support
6. Add envelope modulation support
7. Integrate with render pipeline
8. Test parameter changes and modulation

### Phase 3: Audio Quality (Est. 400 lines)
1. Add FLAC output support
2. Implement HQ render chain functions
3. Add band-limited oscillators
4. Create `/hq` command
5. Integrate into output pipeline
6. Test audio quality improvements

---

## Files to Modify

| File | Phase | Changes |
|------|-------|---------|
| `playback_cmds.py` | 1 | Major rewrite |
| `buffer_cmds.py` | 1, 3 | Remove duplicate /p, add format support |
| `render_cmds.py` | 1, 3 | Remove duplicate /p, add format support |
| `param_cmds.py` | 2 | **NEW FILE** |
| `session.py` | 2, 3 | Add param_system, hq settings |
| `effects.py` | 3 | Add HQ chain functions |
| `monolith.py` | 3 | Add band-limited oscillators |
| `commands/__init__.py` | 1, 2 | Update command registration |

---

## New Commands Summary

### Playback
- `/p` - Play working buffer (changed behavior)
- `/pt <n>` - Play track N
- `/pts` - Play song (all tracks)

### Parameters
- `/parm [category] [index]` - List/select parameters
- `/cha <value|op|mod>` - Change selected parameter

### Quality
- `/hq [on|off|setting]` - High-quality mode settings

---

## Approval Checklist

Please confirm:
- [ ] Playback: `/p` should ONLY play working buffer
- [ ] Playback: Track/Song playback commands approved
- [ ] Parameters: Category structure acceptable
- [ ] Parameters: `/cha` syntax acceptable  
- [ ] Parameters: LFO shapes (sin, tri, saw, sqr, rnd) sufficient
- [ ] Quality: FLAC output desired
- [ ] Quality: 24-bit output desired
- [ ] Quality: HQ render chain (DC removal, subsonic filter, high-end smoothing) approved
- [ ] Quality: Band-limited oscillators desired

---

**Ready to implement upon approval.**
