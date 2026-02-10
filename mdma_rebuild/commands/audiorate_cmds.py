"""Audio-Rate Modulation & Umpulse System Commands.

Implements:
- Audio-rate interval modulation (pitch at audio rate)
- Audio-rate filter modulation
- Umpulse (custom impulse/wavetable import system)
- Auto-chunking enhancements

BUILD ID: audiorate_v1.0
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np
import os

if TYPE_CHECKING:
    from ..core.session import Session


# ============================================================================
# UMPULSE STORAGE
# ============================================================================

class UmpulseBank:
    """Storage for custom impulses and wavetables."""
    
    def __init__(self):
        # Named impulses: name -> (audio, sample_rate, metadata)
        self.impulses: Dict[str, Tuple[np.ndarray, int, dict]] = {}
        # Wavetables: name -> np.ndarray (2D: frames x samples_per_frame)
        self.wavetables: Dict[str, np.ndarray] = {}
        # IRs for convolution: name -> np.ndarray
        self.irs: Dict[str, np.ndarray] = {}
        
    def load(self, name: str, audio: np.ndarray, sr: int, 
             metadata: Optional[dict] = None) -> None:
        """Load an impulse into the bank."""
        if metadata is None:
            metadata = {}
        metadata['duration'] = len(audio) / sr
        metadata['samples'] = len(audio)
        self.impulses[name] = (audio.copy(), sr, metadata)
    
    def get(self, name: str) -> Optional[Tuple[np.ndarray, int, dict]]:
        """Get an impulse by name."""
        return self.impulses.get(name)
    
    def delete(self, name: str) -> bool:
        """Delete an impulse."""
        if name in self.impulses:
            del self.impulses[name]
            return True
        return False
    
    def list_all(self) -> List[str]:
        """List all impulse names."""
        return list(self.impulses.keys())
    
    def normalize(self, name: str) -> bool:
        """Normalize an impulse to peak amplitude."""
        if name not in self.impulses:
            return False
        audio, sr, metadata = self.impulses[name]
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
            metadata['normalized'] = True
            self.impulses[name] = (audio, sr, metadata)
        return True
    
    def to_wavetable(self, name: str, frames: int = 256, 
                     samples_per_frame: int = 2048) -> Optional[np.ndarray]:
        """Convert impulse to wavetable format."""
        if name not in self.impulses:
            return None
        
        audio, sr, _ = self.impulses[name]
        total_samples = len(audio)
        
        wavetable = np.zeros((frames, samples_per_frame))
        
        for i in range(frames):
            # Calculate start position for this frame
            start = int(i * (total_samples - samples_per_frame) / max(1, frames - 1))
            end = start + samples_per_frame
            
            if end <= total_samples:
                wavetable[i] = audio[start:end]
            else:
                # Wrap around
                available = total_samples - start
                wavetable[i, :available] = audio[start:]
                wavetable[i, available:] = audio[:samples_per_frame - available]
        
        # Normalize each frame
        for i in range(frames):
            peak = np.max(np.abs(wavetable[i]))
            if peak > 0:
                wavetable[i] /= peak
        
        self.wavetables[name] = wavetable
        return wavetable


# Global umpulse bank instance
_umpulse_bank: Optional[UmpulseBank] = None

def get_umpulse_bank() -> UmpulseBank:
    """Get the global umpulse bank."""
    global _umpulse_bank
    if _umpulse_bank is None:
        _umpulse_bank = UmpulseBank()
    return _umpulse_bank


# ============================================================================
# AUDIO-RATE MODULATION GENERATORS
# ============================================================================

def generate_mod_signal(
    mod_type: str,
    duration_sec: float,
    sr: int = 48000,
    rate: float = 1.0,
    depth: float = 1.0,
    **kwargs
) -> np.ndarray:
    """Generate a modulation signal for audio-rate modulation.
    
    Parameters
    ----------
    mod_type : str
        Type of modulation: sine, saw, square, triangle, noise, 
        step, random, buffer, envelope
    duration_sec : float
        Duration in seconds
    sr : int
        Sample rate
    rate : float
        Rate/frequency of modulation (Hz or steps/sec for step types)
    depth : float
        Depth/amplitude of modulation (-1 to 1 range output)
    **kwargs : dict
        Additional parameters for specific mod types
    
    Returns
    -------
    np.ndarray
        Modulation signal in -1 to 1 range
    """
    n_samples = int(duration_sec * sr)
    t = np.arange(n_samples) / sr
    
    if mod_type == 'sine':
        mod = np.sin(2 * np.pi * rate * t)
        
    elif mod_type == 'saw' or mod_type == 'sawtooth':
        mod = 2 * ((rate * t) % 1) - 1
        
    elif mod_type == 'square':
        mod = np.sign(np.sin(2 * np.pi * rate * t))
        
    elif mod_type == 'triangle' or mod_type == 'tri':
        mod = 2 * np.abs(2 * ((rate * t) % 1) - 1) - 1
        
    elif mod_type == 'noise':
        # Filtered noise for smoother modulation
        cutoff = kwargs.get('cutoff', rate * 2)
        noise = np.random.randn(n_samples)
        try:
            from scipy import signal
            b, a = signal.butter(2, min(cutoff, sr/2.5) / (sr/2), 'low')
            mod = signal.filtfilt(b, a, noise)
            # Normalize to -1, 1
            peak = np.max(np.abs(mod))
            if peak > 0:
                mod = mod / peak
        except:
            mod = noise / np.max(np.abs(noise))
    
    elif mod_type == 'step' or mod_type == 'quantize':
        # Stepped modulation (like sample & hold)
        steps_per_sec = rate
        samples_per_step = int(sr / steps_per_sec)
        values = np.random.uniform(-1, 1, n_samples // samples_per_step + 1)
        mod = np.repeat(values, samples_per_step)[:n_samples]
        
    elif mod_type == 'random':
        # Random walk
        steps = np.random.randn(n_samples) * (rate / sr)
        mod = np.cumsum(steps)
        # Normalize and wrap
        mod = np.tanh(mod)  # Soft limit to -1, 1
        
    elif mod_type == 'envelope' or mod_type == 'env':
        # Attack-decay envelope modulation
        attack = kwargs.get('attack', 0.1)
        decay = kwargs.get('decay', duration_sec - attack)
        attack_samples = int(attack * sr)
        decay_samples = int(decay * sr)
        
        mod = np.zeros(n_samples)
        if attack_samples > 0:
            mod[:attack_samples] = np.linspace(0, 1, attack_samples)
        if decay_samples > 0 and attack_samples < n_samples:
            decay_start = attack_samples
            decay_end = min(decay_start + decay_samples, n_samples)
            mod[decay_start:decay_end] = np.exp(-np.linspace(0, 5, decay_end - decay_start))
        
        # Scale to -1, 1 range
        mod = mod * 2 - 1
        
    elif mod_type == 'blackmidi':
        # Ultra-fast stepped modulation for glitchy textures
        # Random intervals with hyper-fast rate changes
        base_rate = rate * 100  # Much faster
        intervals = np.random.randint(1, int(sr / base_rate) + 1, n_samples // 10 + 1)
        values = np.random.uniform(-1, 1, len(intervals))
        
        mod = np.zeros(n_samples)
        pos = 0
        for i, (interval, val) in enumerate(zip(intervals, values)):
            end = min(pos + interval, n_samples)
            mod[pos:end] = val
            pos = end
            if pos >= n_samples:
                break
    
    else:
        # Default to sine
        mod = np.sin(2 * np.pi * rate * t)
    
    # Apply depth
    return mod * depth


def generate_interval_mod_from_pattern(
    pattern: str,
    duration_sec: float,
    sr: int = 48000,
    bpm: float = 120.0
) -> np.ndarray:
    """Generate interval modulation from a pattern string.
    
    Pattern format: space-separated semitone offsets
    Example: "0 12 7 0 -12 5" = cycle through these intervals
    
    Parameters
    ----------
    pattern : str
        Space-separated semitone values
    duration_sec : float
        Duration in seconds
    sr : int
        Sample rate
    bpm : float
        BPM for timing
    
    Returns
    -------
    np.ndarray
        Interval modulation signal (in semitones, needs scaling)
    """
    try:
        intervals = [float(x) for x in pattern.split()]
    except:
        return np.zeros(int(duration_sec * sr))
    
    if not intervals:
        return np.zeros(int(duration_sec * sr))
    
    n_samples = int(duration_sec * sr)
    beat_duration = 60.0 / bpm
    samples_per_beat = int(beat_duration * sr)
    samples_per_step = max(1, samples_per_beat // 4)  # 16th notes
    
    mod = np.zeros(n_samples)
    
    for i in range(0, n_samples, samples_per_step):
        step_idx = (i // samples_per_step) % len(intervals)
        end = min(i + samples_per_step, n_samples)
        mod[i:end] = intervals[step_idx] / 12.0  # Normalize to -1 to 1 (ish)
    
    return mod


# ============================================================================
# COMMAND IMPLEMENTATIONS
# ============================================================================

def cmd_audiorate(session: "Session", args: List[str]) -> str:
    """Audio-rate modulation for interval and filter.
    
    Usage:
      /audiorate                         Show status
      /audiorate interval <op> <type> <rate> <depth>
                                         Set interval (pitch) modulation
      /audiorate filter <type> <rate> <depth>
                                         Set filter cutoff modulation
      /audiorate pattern <op> <pattern>  Pattern-based interval mod
      /audiorate clear [op|filter|all]   Clear modulation
      /audiorate presets                 Show preset configurations
    
    Mod Types:
      sine, saw, square, triangle, noise, step, random, envelope, blackmidi
    
    Examples:
      /audiorate interval 0 sine 5 12    Op 0: sine LFO, 5Hz, ±12 semitones
      /audiorate interval 0 blackmidi 10 24
                                         Op 0: black MIDI style, ±24 semitones
      /audiorate filter saw 2 2          Filter: saw LFO, 2Hz, 2 octaves
      /audiorate pattern 0 "0 12 7 0 -12 5"
                                         Op 0: pattern sequence
    """
    if not args:
        return _audiorate_status(session)
    
    sub = args[0].lower()
    
    if sub == 'interval' or sub == 'int' or sub == 'i':
        return _audiorate_interval(session, args[1:])
    elif sub == 'filter' or sub == 'flt' or sub == 'f':
        return _audiorate_filter(session, args[1:])
    elif sub == 'pattern' or sub == 'pat' or sub == 'p':
        return _audiorate_pattern(session, args[1:])
    elif sub == 'clear' or sub == 'clr':
        return _audiorate_clear(session, args[1:])
    elif sub == 'presets' or sub == 'preset':
        return _audiorate_presets()
    elif sub == 'help':
        return cmd_audiorate.__doc__
    else:
        return f"ERROR: Unknown subcommand '{sub}'. Use /audiorate help"


def _audiorate_status(session: "Session") -> str:
    """Show current audio-rate modulation status."""
    lines = ["=== AUDIO-RATE MODULATION STATUS ==="]
    
    engine = session.engine
    
    # Check interval modulation
    has_interval = False
    for idx, op in engine.operators.items():
        if op.get('interval_mod') is not None:
            depth = op.get('interval_mod_depth', 12)
            lines.append(f"  Operator {idx}: Interval mod active, depth={depth} semitones")
            has_interval = True
        elif op.get('interval_lfo_rate', 0) > 0:
            rate = op.get('interval_lfo_rate')
            depth = op.get('interval_lfo_depth')
            wave = op.get('interval_lfo_wave', 'sine')
            lines.append(f"  Operator {idx}: Interval LFO {wave}, {rate}Hz, {depth} semitones")
            has_interval = True
    
    if not has_interval:
        lines.append("  Interval modulation: Not active")
    
    # Check filter modulation
    if engine.filter_mod is not None:
        lines.append(f"  Filter mod: Active, depth={engine.filter_mod_depth} octaves")
    else:
        lines.append("  Filter modulation: Not active")
    
    lines.append("")
    lines.append("Use /audiorate interval|filter|pattern|clear|presets")
    
    return "\n".join(lines)


def _audiorate_interval(session: "Session", args: List[str]) -> str:
    """Set interval modulation for an operator."""
    if len(args) < 4:
        return ("Usage: /audiorate interval <op> <type> <rate> <depth>\n"
                "  op: operator index (0-7)\n"
                "  type: sine, saw, square, triangle, noise, step, random, blackmidi\n"
                "  rate: modulation rate in Hz\n"
                "  depth: depth in semitones")
    
    try:
        op_idx = int(args[0])
        mod_type = args[1].lower()
        rate = float(args[2])
        depth = float(args[3])
    except (ValueError, IndexError) as e:
        return f"ERROR: Invalid arguments: {e}"
    
    if op_idx < 0 or op_idx > 7:
        return "ERROR: Operator index must be 0-7"
    
    # Store in session for next render
    # We'll generate the mod signal at render time based on duration
    if not hasattr(session, 'audiorate_config'):
        session.audiorate_config = {}
    
    session.audiorate_config[f'interval_{op_idx}'] = {
        'type': mod_type,
        'rate': rate,
        'depth': depth
    }
    
    # Also set up LFO-based modulation in engine (works for constant-rate)
    session.engine.set_interval_lfo(op_idx, rate, depth, 
                                     'sine' if mod_type == 'sine' else 'triangle')
    
    return f"OK: Interval mod on Op{op_idx}: {mod_type}, {rate}Hz, ±{depth} semitones"


def _audiorate_filter(session: "Session", args: List[str]) -> str:
    """Set filter cutoff modulation."""
    if len(args) < 3:
        return ("Usage: /audiorate filter <type> <rate> <depth>\n"
                "  type: sine, saw, square, triangle, noise, step, random\n"
                "  rate: modulation rate in Hz\n"
                "  depth: depth in octaves")
    
    try:
        mod_type = args[0].lower()
        rate = float(args[1])
        depth = float(args[2])
    except (ValueError, IndexError) as e:
        return f"ERROR: Invalid arguments: {e}"
    
    # Store configuration
    if not hasattr(session, 'audiorate_config'):
        session.audiorate_config = {}
    
    session.audiorate_config['filter'] = {
        'type': mod_type,
        'rate': rate,
        'depth': depth
    }
    
    # Generate a short preview signal and apply
    preview_signal = generate_mod_signal(mod_type, 0.1, session.sample_rate, rate, 1.0)
    session.engine.set_filter_mod(preview_signal, depth)
    
    return f"OK: Filter mod: {mod_type}, {rate}Hz, ±{depth} octaves"


def _audiorate_pattern(session: "Session", args: List[str]) -> str:
    """Set pattern-based interval modulation."""
    if len(args) < 2:
        return ("Usage: /audiorate pattern <op> <pattern>\n"
                "  op: operator index\n"
                "  pattern: semitone sequence, e.g., '0 12 7 0 -12 5'\n"
                "\nPattern steps are 16th notes at current BPM")
    
    try:
        op_idx = int(args[0])
        pattern = ' '.join(args[1:]).strip('"\'')
    except (ValueError, IndexError) as e:
        return f"ERROR: Invalid arguments: {e}"
    
    if not hasattr(session, 'audiorate_config'):
        session.audiorate_config = {}
    
    session.audiorate_config[f'pattern_{op_idx}'] = {
        'pattern': pattern,
        'bpm': session.bpm
    }
    
    # Parse and validate pattern
    try:
        intervals = [float(x) for x in pattern.split()]
    except:
        return f"ERROR: Invalid pattern format. Use space-separated numbers."
    
    return f"OK: Pattern mod on Op{op_idx}: {len(intervals)} steps [{pattern[:30]}{'...' if len(pattern) > 30 else ''}]"


def _audiorate_clear(session: "Session", args: List[str]) -> str:
    """Clear audio-rate modulation."""
    target = args[0].lower() if args else 'all'
    
    if target == 'all':
        session.engine.clear_modulation()
        if hasattr(session, 'audiorate_config'):
            session.audiorate_config.clear()
        return "OK: All audio-rate modulation cleared"
    
    elif target == 'filter':
        session.engine.set_filter_mod(None)
        if hasattr(session, 'audiorate_config') and 'filter' in session.audiorate_config:
            del session.audiorate_config['filter']
        return "OK: Filter modulation cleared"
    
    else:
        try:
            op_idx = int(target)
            session.engine.set_interval_mod(op_idx, None)
            if hasattr(session, 'audiorate_config'):
                session.audiorate_config.pop(f'interval_{op_idx}', None)
                session.audiorate_config.pop(f'pattern_{op_idx}', None)
            return f"OK: Interval modulation on Op{op_idx} cleared"
        except ValueError:
            return f"ERROR: Invalid target '{target}'. Use: op number, 'filter', or 'all'"


def _audiorate_presets() -> str:
    """Show audio-rate modulation presets."""
    return """=== AUDIO-RATE MODULATION PRESETS ===

VIBRATO (subtle pitch wobble):
  /audiorate interval 0 sine 5 0.5

TRILL (fast alternation):
  /audiorate pattern 0 "0 2 0 2"

OCTAVE JUMP:
  /audiorate pattern 0 "0 12"

ARPEGGIO SIMULATION:
  /audiorate pattern 0 "0 4 7 12 7 4"

GLITCH/BLACK MIDI:
  /audiorate interval 0 blackmidi 20 24

FILTER WOBBLE (dubstep):
  /audiorate filter sine 0.5 3

FILTER SWEEP:
  /audiorate filter saw 0.25 4

RANDOM CHAOS:
  /audiorate interval 0 random 10 12
  /audiorate filter noise 5 2

Use /audiorate clear all to reset"""


# ============================================================================
# UMPULSE COMMANDS
# ============================================================================

def cmd_ump(session: "Session", args: List[str]) -> str:
    """Umpulse - Custom impulse/wavetable import system.
    
    Usage:
      /ump load <path> [name]     Load impulse from file
      /ump buffer [name]          Load from current buffer
      /ump list                   List named impulses
      /ump info <name>            Show impulse details
      /ump use <name> wave <op>   Use as operator waveform
      /ump use <name> ir          Use for convolution reverb
      /ump del <name>             Delete impulse
      /ump norm <name>            Normalize impulse
      /ump wavetable <name> [frames]
                                  Convert to wavetable
    
    Examples:
      /ump buffer mysound         Save buffer as 'mysound'
      /ump use mysound wave 0     Use as waveform for op 0
      /ump wavetable mysound 64   Create 64-frame wavetable
    """
    if not args:
        return _ump_list(session)
    
    sub = args[0].lower()
    
    if sub == 'load':
        return _ump_load(session, args[1:])
    elif sub == 'buffer' or sub == 'buf':
        return _ump_from_buffer(session, args[1:])
    elif sub == 'list' or sub == 'ls':
        return _ump_list(session)
    elif sub == 'info':
        return _ump_info(session, args[1:])
    elif sub == 'use':
        return _ump_use(session, args[1:])
    elif sub == 'del' or sub == 'delete' or sub == 'rm':
        return _ump_delete(session, args[1:])
    elif sub == 'norm' or sub == 'normalize':
        return _ump_normalize(session, args[1:])
    elif sub == 'wavetable' or sub == 'wt':
        return _ump_wavetable(session, args[1:])
    elif sub == 'help':
        return cmd_ump.__doc__
    else:
        return f"ERROR: Unknown subcommand '{sub}'. Use /ump help"


def _ump_load(session: "Session", args: List[str]) -> str:
    """Load impulse from file."""
    if not args:
        return "Usage: /ump load <path> [name]"
    
    path = args[0]
    name = args[1] if len(args) > 1 else os.path.splitext(os.path.basename(path))[0]
    
    # Check if file exists
    if not os.path.exists(path):
        return f"ERROR: File not found: {path}"
    
    try:
        import soundfile as sf
        audio, sr = sf.read(path)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample to session rate if needed
        if sr != session.sample_rate:
            from scipy import signal
            num_samples = int(len(audio) * session.sample_rate / sr)
            audio = signal.resample(audio, num_samples)
            sr = session.sample_rate
        
        bank = get_umpulse_bank()
        bank.load(name, audio, sr, {'source': path})
        
        dur_ms = len(audio) / sr * 1000
        return f"OK: Loaded '{name}' ({dur_ms:.1f}ms, {len(audio)} samples)"
        
    except ImportError:
        # Fallback: try scipy.io.wavfile
        try:
            from scipy.io import wavfile
            sr, audio = wavfile.read(path)
            
            # Normalize
            if audio.dtype == np.int16:
                audio = audio.astype(np.float64) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float64) / 2147483648.0
            
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            bank = get_umpulse_bank()
            bank.load(name, audio, sr, {'source': path})
            
            dur_ms = len(audio) / sr * 1000
            return f"OK: Loaded '{name}' ({dur_ms:.1f}ms, {len(audio)} samples)"
            
        except Exception as e:
            return f"ERROR: Could not load file: {e}"
    except Exception as e:
        return f"ERROR: Could not load file: {e}"


def _ump_from_buffer(session: "Session", args: List[str]) -> str:
    """Load impulse from current buffer."""
    name = args[0] if args else f"impulse_{len(get_umpulse_bank().impulses)}"
    
    audio = session.last_buffer
    if audio is None or len(audio) == 0:
        return "ERROR: No audio in buffer. Generate or load audio first."
    
    bank = get_umpulse_bank()
    bank.load(name, audio, session.sample_rate, {'source': 'buffer'})
    
    dur_ms = len(audio) / session.sample_rate * 1000
    return f"OK: Saved buffer as '{name}' ({dur_ms:.1f}ms, {len(audio)} samples)"


def _ump_list(session: "Session") -> str:
    """List all impulses."""
    bank = get_umpulse_bank()
    names = bank.list_all()
    
    if not names:
        return "No impulses loaded. Use /ump load <file> or /ump buffer <name>"
    
    lines = ["=== UMPULSE LIBRARY ==="]
    for name in names:
        data = bank.get(name)
        if data:
            audio, sr, meta = data
            dur_ms = len(audio) / sr * 1000
            lines.append(f"  {name}: {dur_ms:.1f}ms ({len(audio)} samples)")
    
    lines.append(f"\nTotal: {len(names)} impulses")
    lines.append("Use /ump info <name> for details")
    
    return "\n".join(lines)


def _ump_info(session: "Session", args: List[str]) -> str:
    """Show impulse details."""
    if not args:
        return "Usage: /ump info <name>"
    
    name = args[0]
    bank = get_umpulse_bank()
    data = bank.get(name)
    
    if not data:
        return f"ERROR: Impulse '{name}' not found"
    
    audio, sr, meta = data
    
    lines = [f"=== IMPULSE: {name} ==="]
    lines.append(f"  Samples: {len(audio)}")
    lines.append(f"  Sample rate: {sr} Hz")
    lines.append(f"  Duration: {len(audio) / sr * 1000:.2f} ms")
    lines.append(f"  Peak amplitude: {np.max(np.abs(audio)):.4f}")
    lines.append(f"  RMS: {np.sqrt(np.mean(audio**2)):.4f}")
    
    if meta:
        lines.append("  Metadata:")
        for k, v in meta.items():
            lines.append(f"    {k}: {v}")
    
    return "\n".join(lines)


def _ump_use(session: "Session", args: List[str]) -> str:
    """Use impulse as waveform or IR."""
    if len(args) < 2:
        return ("Usage:\n"
                "  /ump use <name> wave <op>  - Use as waveform for operator\n"
                "  /ump use <name> ir         - Use for convolution reverb")
    
    name = args[0]
    use_type = args[1].lower()
    
    bank = get_umpulse_bank()
    data = bank.get(name)
    
    if not data:
        return f"ERROR: Impulse '{name}' not found"
    
    audio, sr, meta = data
    
    if use_type == 'wave' or use_type == 'waveform':
        if len(args) < 3:
            return "Usage: /ump use <name> wave <op_idx>"
        
        try:
            op_idx = int(args[2])
        except:
            return f"ERROR: Invalid operator index: {args[2]}"
        
        # Store as custom waveform in operator
        # This requires the engine to support custom wavetables
        if not hasattr(session, 'custom_wavetables'):
            session.custom_wavetables = {}
        
        session.custom_wavetables[op_idx] = audio
        
        # Set wave type to 'custom' (would need engine support)
        session.engine.set_wave(op_idx, 'custom')
        session.engine.operators[op_idx]['custom_wave'] = audio
        
        return f"OK: Using '{name}' as waveform for operator {op_idx}"
    
    elif use_type == 'ir':
        # Store as impulse response for convolution
        bank.irs[name] = audio
        
        # Store reference in session for effects
        if not hasattr(session, 'convolution_ir'):
            session.convolution_ir = {}
        session.convolution_ir['active'] = name
        session.convolution_ir['data'] = audio
        
        return f"OK: Using '{name}' as convolution IR ({len(audio)/sr*1000:.1f}ms)"
    
    else:
        return f"ERROR: Unknown use type '{use_type}'. Use 'wave' or 'ir'"


def _ump_delete(session: "Session", args: List[str]) -> str:
    """Delete an impulse."""
    if not args:
        return "Usage: /ump del <name>"
    
    name = args[0]
    bank = get_umpulse_bank()
    
    if bank.delete(name):
        return f"OK: Deleted impulse '{name}'"
    else:
        return f"ERROR: Impulse '{name}' not found"


def _ump_normalize(session: "Session", args: List[str]) -> str:
    """Normalize an impulse."""
    if not args:
        return "Usage: /ump norm <name>"
    
    name = args[0]
    bank = get_umpulse_bank()
    
    if bank.normalize(name):
        return f"OK: Normalized impulse '{name}'"
    else:
        return f"ERROR: Impulse '{name}' not found"


def _ump_wavetable(session: "Session", args: List[str]) -> str:
    """Convert impulse to wavetable."""
    if not args:
        return "Usage: /ump wavetable <name> [frames] [samples_per_frame]"
    
    name = args[0]
    frames = int(args[1]) if len(args) > 1 else 256
    samples_per_frame = int(args[2]) if len(args) > 2 else 2048
    
    bank = get_umpulse_bank()
    wt = bank.to_wavetable(name, frames, samples_per_frame)
    
    if wt is None:
        return f"ERROR: Impulse '{name}' not found"
    
    # Store reference
    if not hasattr(session, 'wavetables'):
        session.wavetables = {}
    session.wavetables[name] = wt
    
    return f"OK: Created wavetable '{name}': {frames} frames × {samples_per_frame} samples"


# ============================================================================
# ENHANCED CHUNKING COMMANDS
# ============================================================================

def cmd_chk_enhanced(session: "Session", args: List[str]) -> str:
    """Enhanced auto-chunking with more algorithms.
    
    Usage:
      /chk [algo] [num]          Chunk with algorithm
      /chk use <idx|range|all>   Load chunk(s) to buffer
      /chk shuffle               Shuffle chunk order
      /chk reverse               Reverse chunk order
      /chk sort energy|length|peak
                                 Sort chunks by property
      /chk export <prefix>       Export chunks as files
      /chk remix <pattern>       Remix chunks by pattern
    
    Algorithms:
      auto       Automatic selection (default)
      transient  Split at transients (drums, percussive)
      beat       Split at detected beats
      zero       Split at zero crossings
      equal      Equal-sized chunks
      wavetable  Single-cycle extraction
      energy     Energy-based segmentation
      spectral   Spectral similarity
      syllable   Speech-like syllable detection
    
    Examples:
      /chk beat                  Chunk at beats
      /chk equal 16              16 equal chunks
      /chk use 0-3               Load chunks 0,1,2,3
      /chk remix "0 2 1 3 0 0"   Reorder chunks
    """
    from ..dsp.advanced_ops import auto_chunk, AudioChunk
    
    if not args:
        return _chk_status(session)
    
    sub = args[0].lower()
    
    if sub == 'use':
        return _chk_use(session, args[1:])
    elif sub == 'shuffle':
        return _chk_shuffle(session)
    elif sub == 'reverse':
        return _chk_reverse(session)
    elif sub == 'sort':
        return _chk_sort(session, args[1:])
    elif sub == 'export':
        return _chk_export(session, args[1:])
    elif sub == 'remix':
        return _chk_remix(session, args[1:])
    elif sub == 'help':
        return cmd_chk_enhanced.__doc__
    else:
        # Assume it's an algorithm or number
        return _chk_run(session, args)


def _chk_status(session: "Session") -> str:
    """Show chunking status."""
    if not hasattr(session, 'chunks') or 'last' not in session.chunks:
        return ("No chunks available.\n"
                "Usage: /chk [algorithm] [num_chunks]\n"
                "Algorithms: auto, transient, beat, zero, equal, wavetable, energy")
    
    chunks = session.chunks['last']
    lines = [f"=== {len(chunks)} CHUNKS AVAILABLE ==="]
    
    for i, chunk in enumerate(chunks[:15]):
        dur_ms = len(chunk.audio) / session.sample_rate * 1000
        lines.append(f"  {i:2d}: {dur_ms:6.1f}ms  E={chunk.energy:.3f}  P={chunk.peak:.3f}")
    
    if len(chunks) > 15:
        lines.append(f"  ... and {len(chunks) - 15} more")
    
    lines.append("\nCommands: use, shuffle, reverse, sort, export, remix")
    
    return "\n".join(lines)


def _chk_run(session: "Session", args: List[str]) -> str:
    """Run chunking algorithm."""
    from ..dsp.advanced_ops import auto_chunk
    
    # Get current audio
    audio = session.last_buffer
    if audio is None or len(audio) == 0:
        # Try working buffer
        if hasattr(session, 'working_buffer') and session.working_buffer is not None:
            audio = session.working_buffer
        else:
            return "ERROR: No audio in buffer. Generate or load audio first."
    
    # Parse args
    algorithm = "auto"
    num_chunks = 0
    
    if args:
        try:
            num_chunks = int(args[0])
            algorithm = "equal"
        except ValueError:
            algorithm = args[0].lower()
    
    if len(args) > 1:
        try:
            num_chunks = int(args[1])
        except:
            pass
    
    # Run chunking
    chunks = auto_chunk(audio, session.sample_rate, algorithm=algorithm, num_chunks=num_chunks)
    
    if not chunks:
        return f"ERROR: Could not chunk audio with algorithm '{algorithm}'"
    
    # Store
    if not hasattr(session, 'chunks'):
        session.chunks = {}
    session.chunks['last'] = chunks
    
    lines = [f"=== CHUNKED: {len(chunks)} chunks ({algorithm}) ==="]
    for i, chunk in enumerate(chunks[:10]):
        dur_ms = len(chunk.audio) / session.sample_rate * 1000
        lines.append(f"  {i:2d}: {dur_ms:6.1f}ms  E={chunk.energy:.3f}")
    
    if len(chunks) > 10:
        lines.append(f"  ... and {len(chunks) - 10} more")
    
    return "\n".join(lines)


def _chk_use(session: "Session", args: List[str]) -> str:
    """Load chunks to buffer."""
    if not hasattr(session, 'chunks') or 'last' not in session.chunks:
        return "ERROR: No chunks. Run /chk first."
    
    chunks = session.chunks['last']
    
    if not args:
        return f"Usage: /chk use <idx|range|all>\n  {len(chunks)} chunks available"
    
    spec = args[0].lower()
    
    if spec == 'all':
        audio = np.concatenate([c.audio for c in chunks])
    elif '-' in spec:
        # Range: "0-5"
        try:
            start, end = map(int, spec.split('-'))
            selected = chunks[start:end+1]
            audio = np.concatenate([c.audio for c in selected])
        except:
            return f"ERROR: Invalid range '{spec}'"
    elif ',' in spec:
        # List: "0,2,4"
        try:
            indices = [int(x) for x in spec.split(',')]
            audio = np.concatenate([chunks[i].audio for i in indices])
        except:
            return f"ERROR: Invalid list '{spec}'"
    else:
        # Single index
        try:
            idx = int(spec)
            if idx < 0 or idx >= len(chunks):
                return f"ERROR: Index {idx} out of range (0-{len(chunks)-1})"
            audio = chunks[idx].audio
        except:
            return f"ERROR: Invalid index '{spec}'"
    
    session.last_buffer = audio
    dur_ms = len(audio) / session.sample_rate * 1000
    return f"OK: Loaded {dur_ms:.1f}ms to buffer"


def _chk_shuffle(session: "Session") -> str:
    """Shuffle chunk order."""
    if not hasattr(session, 'chunks') or 'last' not in session.chunks:
        return "ERROR: No chunks. Run /chk first."
    
    chunks = session.chunks['last']
    np.random.shuffle(chunks)
    
    # Update indices
    for i, chunk in enumerate(chunks):
        chunk.index = i
    
    return f"OK: Shuffled {len(chunks)} chunks"


def _chk_reverse(session: "Session") -> str:
    """Reverse chunk order."""
    if not hasattr(session, 'chunks') or 'last' not in session.chunks:
        return "ERROR: No chunks. Run /chk first."
    
    session.chunks['last'] = session.chunks['last'][::-1]
    
    # Update indices
    for i, chunk in enumerate(session.chunks['last']):
        chunk.index = i
    
    return f"OK: Reversed {len(session.chunks['last'])} chunks"


def _chk_sort(session: "Session", args: List[str]) -> str:
    """Sort chunks by property."""
    if not hasattr(session, 'chunks') or 'last' not in session.chunks:
        return "ERROR: No chunks. Run /chk first."
    
    key = args[0].lower() if args else 'energy'
    
    chunks = session.chunks['last']
    
    if key == 'energy':
        chunks.sort(key=lambda c: c.energy, reverse=True)
    elif key == 'length' or key == 'len':
        chunks.sort(key=lambda c: len(c.audio), reverse=True)
    elif key == 'peak':
        chunks.sort(key=lambda c: c.peak, reverse=True)
    else:
        return f"ERROR: Unknown sort key '{key}'. Use: energy, length, peak"
    
    # Update indices
    for i, chunk in enumerate(chunks):
        chunk.index = i
    
    return f"OK: Sorted {len(chunks)} chunks by {key}"


def _chk_export(session: "Session", args: List[str]) -> str:
    """Export chunks as files."""
    if not hasattr(session, 'chunks') or 'last' not in session.chunks:
        return "ERROR: No chunks. Run /chk first."
    
    prefix = args[0] if args else 'chunk'
    
    chunks = session.chunks['last']
    
    try:
        import soundfile as sf
        
        for i, chunk in enumerate(chunks):
            filename = f"{prefix}_{i:03d}.wav"
            sf.write(filename, chunk.audio, session.sample_rate)
        
        return f"OK: Exported {len(chunks)} chunks as {prefix}_XXX.wav"
    
    except ImportError:
        return "ERROR: soundfile not available for export"
    except Exception as e:
        return f"ERROR: Export failed: {e}"


def _chk_remix(session: "Session", args: List[str]) -> str:
    """Remix chunks by pattern."""
    if not hasattr(session, 'chunks') or 'last' not in session.chunks:
        return "ERROR: No chunks. Run /chk first."
    
    if not args:
        return "Usage: /chk remix <pattern>\n  Pattern: space-separated indices, e.g., '0 2 1 3 0 0'"
    
    pattern = ' '.join(args).strip('"\'')
    chunks = session.chunks['last']
    
    try:
        indices = [int(x) for x in pattern.split()]
    except:
        return f"ERROR: Invalid pattern. Use space-separated indices."
    
    # Validate indices
    for idx in indices:
        if idx < 0 or idx >= len(chunks):
            return f"ERROR: Index {idx} out of range (0-{len(chunks)-1})"
    
    # Build remixed audio
    audio = np.concatenate([chunks[i].audio for i in indices])
    
    session.last_buffer = audio
    dur_ms = len(audio) / session.sample_rate * 1000
    return f"OK: Remixed {len(indices)} chunks -> {dur_ms:.1f}ms"


# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

def get_audiorate_commands() -> Dict[str, callable]:
    """Get all audiorate and umpulse commands."""
    return {
        'audiorate': cmd_audiorate,
        'ar': cmd_audiorate,
        'ump': cmd_ump,
        'impulse': cmd_ump,
        'chke': cmd_chk_enhanced,
    }
