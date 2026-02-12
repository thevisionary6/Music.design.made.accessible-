"""Phase 3 Commands: Modulation, Impulse & Convolution.

Implements:
- /impulselfo  — Import impulse as LFO waveshape (3.1)
- /impenv      — Import impulse as envelope shape (3.2)
- /conv        — Advanced convolution reverb engine (3.3)
- /irenhance   — Neural-enhanced IR processing (3.4)
- /irtransform — AI-descriptor IR transformation (3.5)
- /irgranular  — Granular IR tools (3.6)

BUILD ID: convolution_cmds_v1.0_phase3
"""

from __future__ import annotations

from typing import List, Dict, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..core.session import Session


# ============================================================================
# FEATURE 3.1: IMPULSE → LFO
# ============================================================================

def cmd_impulselfo(session: "Session", args: List[str]) -> str:
    """Import an impulse as an LFO waveshape and apply as modulation.

    Usage:
      /impulselfo                    Show status
      /impulselfo load <ump_name>    Convert umpulse to LFO waveshape
      /impulselfo file <path>        Load audio file as LFO waveshape
      /impulselfo apply <op> <rate> [depth]
                                     Apply loaded LFO to operator interval mod
      /impulselfo filter <rate> [depth]
                                     Apply loaded LFO to filter mod
      /impulselfo list               List available LFO waveshapes
      /impulselfo clear              Clear LFO modulation
      /impulselfo info               Show current LFO waveshape info

    Examples:
      /impulselfo load kick          Use 'kick' umpulse as LFO shape
      /impulselfo apply 0 4 12       Op 0: 4Hz LFO, ±12 semitones
      /impulselfo filter 2 2         Filter: 2Hz LFO, 2 octaves
    """
    from ..dsp.convolution import impulse_to_lfo, lfo_from_waveshape

    # Store LFO waveshapes on session
    if not hasattr(session, '_lfo_waveshapes'):
        session._lfo_waveshapes = {}
    if not hasattr(session, '_current_lfo_shape'):
        session._current_lfo_shape = None
        session._current_lfo_name = ''

    if not args:
        return _impulselfo_status(session)

    sub = args[0].lower()

    if sub == 'load':
        if len(args) < 2:
            return "Usage: /impulselfo load <umpulse_name>"
        return _impulselfo_load_ump(session, args[1])

    elif sub == 'file':
        if len(args) < 2:
            return "Usage: /impulselfo file <path>"
        return _impulselfo_load_file(session, args[1])

    elif sub == 'apply':
        if len(args) < 3:
            return "Usage: /impulselfo apply <op_index> <rate_hz> [depth_semitones]"
        return _impulselfo_apply(session, args[1:])

    elif sub == 'filter':
        if len(args) < 2:
            return "Usage: /impulselfo filter <rate_hz> [depth_octaves]"
        return _impulselfo_filter(session, args[1:])

    elif sub == 'list':
        shapes = list(session._lfo_waveshapes.keys())
        if not shapes:
            return "No LFO waveshapes loaded. Use /impulselfo load <name>"
        return "LFO Waveshapes:\n" + '\n'.join(f"  {s}" for s in shapes)

    elif sub == 'clear':
        session.engine.clear_modulation()
        return "Cleared all LFO modulation."

    elif sub == 'info':
        if session._current_lfo_shape is None:
            return "No LFO waveshape loaded."
        name = session._current_lfo_name
        samples = len(session._current_lfo_shape)
        return f"Current LFO: '{name}' ({samples} samples per cycle)"

    elif sub == 'help':
        return cmd_impulselfo.__doc__

    return f"ERROR: Unknown subcommand '{sub}'. Use /impulselfo help"


def _impulselfo_status(session: "Session") -> str:
    lines = ["=== IMPULSE LFO STATUS ==="]
    if session._current_lfo_shape is not None:
        lines.append(f"  Current shape: {session._current_lfo_name}")
        lines.append(f"  Cycle samples: {len(session._current_lfo_shape)}")
    else:
        lines.append("  No LFO shape loaded")
    shapes = list(getattr(session, '_lfo_waveshapes', {}).keys())
    if shapes:
        lines.append(f"  Available: {', '.join(shapes)}")
    return '\n'.join(lines)


def _impulselfo_load_ump(session: "Session", name: str) -> str:
    from ..commands.audiorate_cmds import get_umpulse_bank
    from ..dsp.convolution import impulse_to_lfo

    bank = get_umpulse_bank()
    data = bank.get(name)
    if data is None:
        return f"ERROR: Umpulse '{name}' not found. Use /ump list"
    audio, sr, _ = data
    shape = impulse_to_lfo(audio, sr)
    session._lfo_waveshapes[name] = shape
    session._current_lfo_shape = shape
    session._current_lfo_name = name
    return f"Loaded umpulse '{name}' as LFO waveshape ({len(shape)} samples/cycle)"


def _impulselfo_load_file(session: "Session", path: str) -> str:
    import os
    from ..dsp.convolution import impulse_to_lfo

    if not os.path.isfile(path):
        return f"ERROR: File not found: {path}"
    try:
        import wave
        with wave.open(path, 'rb') as wf:
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        if sw == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
        elif sw == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float64) / 2147483648.0
        else:
            audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float64) / 128.0 - 1.0
        if ch == 2:
            audio = (audio[::2] + audio[1::2]) / 2.0
    except Exception as e:
        return f"ERROR: Could not load audio: {e}"

    name = os.path.splitext(os.path.basename(path))[0]
    shape = impulse_to_lfo(audio)
    session._lfo_waveshapes[name] = shape
    session._current_lfo_shape = shape
    session._current_lfo_name = name
    return f"Loaded '{name}' as LFO waveshape ({len(shape)} samples/cycle)"


def _impulselfo_apply(session: "Session", args: List[str]) -> str:
    from ..dsp.convolution import lfo_from_waveshape

    if session._current_lfo_shape is None:
        return "ERROR: No LFO shape loaded. Use /impulselfo load first"

    try:
        op_idx = int(args[0])
    except ValueError:
        return "ERROR: Operator index must be an integer"

    try:
        rate = float(args[1])
    except (ValueError, IndexError):
        return "ERROR: Rate (Hz) required"

    depth = float(args[2]) if len(args) > 2 else 12.0

    sr = getattr(session, 'sample_rate', 48000)
    duration = 10.0  # generate 10 seconds of modulation
    mod = lfo_from_waveshape(session._current_lfo_shape, duration, rate, 1.0, sr)
    session.engine.set_interval_mod(op_idx, mod, depth)
    return (f"Applied impulse LFO to Op {op_idx}: "
            f"{rate:.1f}Hz, ±{depth:.1f} semitones "
            f"(shape: {session._current_lfo_name})")


def _impulselfo_filter(session: "Session", args: List[str]) -> str:
    from ..dsp.convolution import lfo_from_waveshape

    if session._current_lfo_shape is None:
        return "ERROR: No LFO shape loaded. Use /impulselfo load first"

    try:
        rate = float(args[0])
    except (ValueError, IndexError):
        return "ERROR: Rate (Hz) required"

    depth = float(args[1]) if len(args) > 1 else 2.0

    sr = getattr(session, 'sample_rate', 48000)
    mod = lfo_from_waveshape(session._current_lfo_shape, 10.0, rate, 1.0, sr)
    session.engine.set_filter_mod(mod, depth)
    return (f"Applied impulse LFO to filter: "
            f"{rate:.1f}Hz, ±{depth:.1f} octaves "
            f"(shape: {session._current_lfo_name})")


# ============================================================================
# FEATURE 3.2: IMPULSE → ENVELOPE
# ============================================================================

def cmd_impenv(session: "Session", args: List[str]) -> str:
    """Import an impulse as an amplitude envelope shape.

    Usage:
      /impenv                       Show status
      /impenv load <ump_name>       Extract envelope from umpulse
      /impenv file <path>           Extract envelope from audio file
      /impenv apply [duration_sec]  Apply envelope to working buffer
      /impenv operator <op> [dur]   Set as operator amplitude envelope
      /impenv list                  List extracted envelopes
      /impenv info                  Show current envelope info

    Examples:
      /impenv load snare            Extract envelope from 'snare' umpulse
      /impenv apply 2.0             Apply envelope over 2 seconds
      /impenv operator 0 1.0        Apply to op 0, 1 second duration
    """
    from ..dsp.convolution import impulse_to_envelope

    if not hasattr(session, '_imp_envelopes'):
        session._imp_envelopes = {}
    if not hasattr(session, '_current_imp_env'):
        session._current_imp_env = None
        session._current_imp_env_name = ''

    if not args:
        return _impenv_status(session)

    sub = args[0].lower()

    if sub == 'load':
        if len(args) < 2:
            return "Usage: /impenv load <umpulse_name>"
        return _impenv_load(session, args[1])

    elif sub == 'file':
        if len(args) < 2:
            return "Usage: /impenv file <path>"
        return _impenv_file(session, args[1])

    elif sub == 'apply':
        dur = float(args[1]) if len(args) > 1 else None
        return _impenv_apply(session, dur)

    elif sub == 'operator' or sub == 'op':
        if len(args) < 2:
            return "Usage: /impenv operator <op_index> [duration_sec]"
        return _impenv_operator(session, args[1:])

    elif sub == 'list':
        envs = list(session._imp_envelopes.keys())
        if not envs:
            return "No impulse envelopes extracted. Use /impenv load <name>"
        return "Impulse Envelopes:\n" + '\n'.join(f"  {e}" for e in envs)

    elif sub == 'info':
        if session._current_imp_env is None:
            return "No impulse envelope loaded."
        return (f"Current envelope: '{session._current_imp_env_name}' "
                f"({len(session._current_imp_env)} samples)")

    elif sub == 'help':
        return cmd_impenv.__doc__

    return f"ERROR: Unknown subcommand '{sub}'. Use /impenv help"


def _impenv_status(session: "Session") -> str:
    lines = ["=== IMPULSE ENVELOPE STATUS ==="]
    if session._current_imp_env is not None:
        lines.append(f"  Current: {session._current_imp_env_name}")
        lines.append(f"  Samples: {len(session._current_imp_env)}")
    else:
        lines.append("  No envelope loaded")
    envs = list(getattr(session, '_imp_envelopes', {}).keys())
    if envs:
        lines.append(f"  Available: {', '.join(envs)}")
    return '\n'.join(lines)


def _impenv_load(session: "Session", name: str) -> str:
    from ..commands.audiorate_cmds import get_umpulse_bank
    from ..dsp.convolution import impulse_to_envelope

    bank = get_umpulse_bank()
    data = bank.get(name)
    if data is None:
        return f"ERROR: Umpulse '{name}' not found. Use /ump list"
    audio, sr, _ = data
    env = impulse_to_envelope(audio, sr)
    session._imp_envelopes[name] = env
    session._current_imp_env = env
    session._current_imp_env_name = name
    dur = len(env) / sr
    return f"Extracted envelope from '{name}' ({dur:.3f}s, {len(env)} samples)"


def _impenv_file(session: "Session", path: str) -> str:
    import os
    from ..dsp.convolution import impulse_to_envelope

    if not os.path.isfile(path):
        return f"ERROR: File not found: {path}"
    try:
        import wave
        with wave.open(path, 'rb') as wf:
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        if sw == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
        elif sw == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float64) / 2147483648.0
        else:
            audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float64) / 128.0 - 1.0
        if ch == 2:
            audio = (audio[::2] + audio[1::2]) / 2.0
    except Exception as e:
        return f"ERROR: Could not load audio: {e}"

    name = os.path.splitext(os.path.basename(path))[0]
    env = impulse_to_envelope(audio, sr)
    session._imp_envelopes[name] = env
    session._current_imp_env = env
    session._current_imp_env_name = name
    return f"Extracted envelope from '{name}' ({len(env)/sr:.3f}s)"


def _impenv_apply(session: "Session", duration: float = None) -> str:
    if session._current_imp_env is None:
        return "ERROR: No envelope loaded. Use /impenv load first"

    buf = getattr(session, 'working_buffer', None)
    if buf is None or len(buf) == 0:
        return "ERROR: Working buffer is empty"

    sr = getattr(session, 'sample_rate', 48000)
    env = session._current_imp_env

    if duration:
        target = int(duration * sr)
    else:
        target = len(buf) if buf.ndim == 1 else buf.shape[0]

    # Resample envelope to match buffer length
    from ..dsp.convolution import impulse_to_envelope
    if len(env) != target:
        x_old = np.linspace(0, 1, len(env))
        x_new = np.linspace(0, 1, target)
        env = np.interp(x_new, x_old, env)

    if buf.ndim == 2:
        for ch in range(buf.shape[1]):
            buf[:target, ch] *= env[:min(target, buf.shape[0])]
    else:
        buf[:target] *= env[:min(target, len(buf))]

    return (f"Applied impulse envelope '{session._current_imp_env_name}' "
            f"to working buffer ({target/sr:.2f}s)")


def _impenv_operator(session: "Session", args: List[str]) -> str:
    if session._current_imp_env is None:
        return "ERROR: No envelope loaded. Use /impenv load first"

    try:
        op_idx = int(args[0])
    except ValueError:
        return "ERROR: Operator index must be an integer"

    duration = float(args[1]) if len(args) > 1 else 1.0
    sr = getattr(session, 'sample_rate', 48000)
    target = int(duration * sr)

    env = session._current_imp_env
    if len(env) != target:
        x_old = np.linspace(0, 1, len(env))
        x_new = np.linspace(0, 1, target)
        env = np.interp(x_new, x_old, env)

    # Store as operator amplitude envelope
    if op_idx in session.engine.operators:
        session.engine.operators[op_idx]['amp_envelope'] = env
        return (f"Set impulse envelope on Op {op_idx} "
                f"({duration:.2f}s, shape: {session._current_imp_env_name})")
    return f"ERROR: Operator {op_idx} not found"


# ============================================================================
# FEATURE 3.3: ADVANCED CONVOLUTION REVERB
# ============================================================================

def cmd_conv(session: "Session", args: List[str]) -> str:
    """Advanced convolution reverb engine.

    Usage:
      /conv                          Show status
      /conv load <file>              Load IR from WAV file
      /conv preset <name>            Load built-in IR preset
      /conv save <name>              Save current IR to bank
      /conv use <name>               Use IR from bank
      /conv apply [wet] [dry]        Apply convolution to working buffer
      /conv params <key=value ...>   Set engine parameters
      /conv split <ms>               Set early/late split point
      /conv bank                     List IRs in bank
      /conv presets                  List available presets
      /conv info                     Detailed engine info
      /conv clear                    Reset engine

    Parameters for /conv params:
      wet=0-100, dry=0-100, pre_delay_ms=0-500, decay=0.1-5.0,
      stereo_width=0-100, early_level=0-100, late_level=0-100,
      high_cut=20-20000, low_cut=20-20000

    Examples:
      /conv preset hall              Load hall preset
      /conv params wet=70 decay=1.5  Set wet level and decay
      /conv apply                    Process working buffer
      /conv split 60                 Split early/late at 60ms
    """
    from ..dsp.convolution import get_convolution_engine

    engine = get_convolution_engine()

    if not args:
        return _conv_status(engine)

    sub = args[0].lower()

    if sub == 'load':
        if len(args) < 2:
            return "Usage: /conv load <file_path>"
        return engine.load_ir_from_file(args[1])

    elif sub == 'preset':
        if len(args) < 2:
            return "Usage: /conv preset <name>. Use /conv presets to list."
        return engine.load_preset(args[1])

    elif sub == 'save':
        if len(args) < 2:
            return "Usage: /conv save <name>"
        return engine.save_ir(args[1])

    elif sub == 'use':
        if len(args) < 2:
            return "Usage: /conv use <name>"
        name = args[1]
        if name in engine.ir_bank:
            engine.load_ir(engine.ir_bank[name], name)
            return f"Using IR '{name}'"
        return f"ERROR: IR '{name}' not in bank. Use /conv bank"

    elif sub == 'apply':
        return _conv_apply(session, engine, args[1:])

    elif sub == 'params' or sub == 'param' or sub == 'set':
        return _conv_set_params(engine, args[1:])

    elif sub == 'split':
        if len(args) < 2:
            return f"Current split: {engine.early_late_split_ms:.0f}ms"
        try:
            ms = float(args[1])
            engine.set_params(early_late_split_ms=ms)
            return f"Early/late split set to {ms:.0f}ms"
        except ValueError:
            return "ERROR: Split value must be a number (ms)"

    elif sub == 'bank' or sub == 'list':
        names = engine.list_irs()
        if not names:
            return "IR bank is empty. Use /conv load or /conv preset"
        return "IR Bank:\n" + '\n'.join(f"  {n}" for n in names)

    elif sub == 'presets':
        try:
            from ..dsp.effects import IR_PRESETS
            names = sorted(IR_PRESETS.keys())
            return "Available IR Presets:\n" + '\n'.join(f"  {n}" for n in names)
        except ImportError:
            return "ERROR: Could not load IR presets"

    elif sub == 'info':
        info = engine.get_info()
        lines = ["=== CONVOLUTION ENGINE INFO ==="]
        for k, v in info.items():
            lines.append(f"  {k}: {v}")
        return '\n'.join(lines)

    elif sub == 'clear' or sub == 'reset':
        from ..dsp.convolution import ConvolutionEngine
        engine.__init__(engine.sr)
        return "Convolution engine reset."

    elif sub == 'help':
        return cmd_conv.__doc__

    return f"ERROR: Unknown subcommand '{sub}'. Use /conv help"


def _conv_status(engine) -> str:
    info = engine.get_info()
    lines = ["=== CONVOLUTION REVERB ==="]
    if info['ir_loaded']:
        lines.append(f"  IR: {info['ir_name']} ({info['ir_duration']:.2f}s)")
    else:
        lines.append("  No IR loaded")
    lines.append(f"  Wet/Dry: {info['wet']:.0f}/{info['dry']:.0f}")
    lines.append(f"  Pre-delay: {info['pre_delay_ms']:.0f}ms")
    lines.append(f"  Decay: {info['decay']:.2f}x")
    lines.append(f"  Width: {info['stereo_width']:.0f}")
    lines.append(f"  Early/Late: {info['early_level']:.0f}/{info['late_level']:.0f} (split {info['early_late_split_ms']:.0f}ms)")
    lines.append(f"  Bank: {info['bank_count']} IRs")
    return '\n'.join(lines)


def _conv_apply(session: "Session", engine, args: List[str]) -> str:
    if engine.ir is None:
        return "ERROR: No IR loaded. Use /conv load or /conv preset first"

    buf = getattr(session, 'working_buffer', None)
    if buf is None or (hasattr(buf, '__len__') and len(buf) == 0):
        return "ERROR: Working buffer is empty"

    # Optional wet/dry overrides
    if len(args) >= 1:
        try:
            engine.wet = float(args[0])
        except ValueError:
            pass
    if len(args) >= 2:
        try:
            engine.dry = float(args[1])
        except ValueError:
            pass

    result = engine.process(buf)
    session.working_buffer = result
    dur = len(result) / getattr(session, 'sample_rate', 48000)
    return f"Applied convolution reverb ({dur:.2f}s, IR: {engine.ir_name})"


def _conv_set_params(engine, args: List[str]) -> str:
    if not args:
        return "Usage: /conv params key=value ..."

    changes = []
    for arg in args:
        if '=' not in arg:
            continue
        key, val = arg.split('=', 1)
        key = key.strip()
        try:
            val = float(val.strip())
            engine.set_params(**{key: val})
            changes.append(f"{key}={val}")
        except (ValueError, AttributeError) as e:
            return f"ERROR: Invalid param '{key}': {e}"

    if changes:
        return f"Set: {', '.join(changes)}"
    return "No parameters changed."


# ============================================================================
# FEATURE 3.4: NEURAL-ENHANCED IR
# ============================================================================

def cmd_irenhance(session: "Session", args: List[str]) -> str:
    """Neural-enhanced impulse response processing.

    Usage:
      /irenhance extend <target_sec> Extend IR tail to target duration
      /irenhance denoise [threshold]  Remove noise floor from IR
      /irenhance fill [threshold]     Fill gaps/dropouts in IR
      /irenhance info                 Show current IR state

    Examples:
      /irenhance extend 5.0          Extend IR to 5 seconds
      /irenhance denoise -50         Denoise with -50dB threshold
      /irenhance fill                Auto-fill gaps
    """
    from ..dsp.convolution import (
        get_convolution_engine, ir_extend, ir_denoise, ir_fill_gaps
    )

    engine = get_convolution_engine()

    if not args:
        return cmd_irenhance.__doc__

    sub = args[0].lower()

    if sub == 'extend':
        if engine.ir is None:
            return "ERROR: No IR loaded. Use /conv load first"
        target = float(args[1]) if len(args) > 1 else 5.0
        current_dur = len(engine.ir) / engine.sr
        engine.ir = ir_extend(engine.ir, target, engine.sr)
        engine._split_ir()
        new_dur = len(engine.ir) / engine.sr
        return (f"Extended IR from {current_dur:.2f}s to {new_dur:.2f}s "
                f"(target: {target:.1f}s)")

    elif sub == 'denoise':
        if engine.ir is None:
            return "ERROR: No IR loaded"
        threshold = float(args[1]) if len(args) > 1 else -60.0
        engine.ir = ir_denoise(engine.ir, engine.sr, threshold)
        engine._split_ir()
        return f"Denoised IR (threshold: {threshold:.0f}dB)"

    elif sub == 'fill':
        if engine.ir is None:
            return "ERROR: No IR loaded"
        threshold = float(args[1]) if len(args) > 1 else -40.0
        engine.ir = ir_fill_gaps(engine.ir, engine.sr, threshold)
        engine._split_ir()
        return f"Filled gaps in IR (threshold: {threshold:.0f}dB)"

    elif sub == 'info':
        if engine.ir is None:
            return "No IR loaded."
        dur = len(engine.ir) / engine.sr
        peak = np.max(np.abs(engine.ir))
        rms = np.sqrt(np.mean(engine.ir ** 2))
        return (f"IR: {engine.ir_name}\n"
                f"  Duration: {dur:.3f}s ({len(engine.ir)} samples)\n"
                f"  Peak: {20*np.log10(max(peak,1e-10)):.1f}dB\n"
                f"  RMS: {20*np.log10(max(rms,1e-10)):.1f}dB")

    elif sub == 'help':
        return cmd_irenhance.__doc__

    return f"ERROR: Unknown subcommand '{sub}'. Use /irenhance help"


# ============================================================================
# FEATURE 3.5: AI-DESCRIPTOR TRANSFORMATION
# ============================================================================

def cmd_irtransform(session: "Session", args: List[str]) -> str:
    """Transform an IR using semantic descriptors.

    Usage:
      /irtransform <descriptor> [intensity]
                                     Apply named transformation
      /irtransform list              Show available descriptors
      /irtransform chain <d1> <d2> ...
                                     Chain multiple transformations

    Descriptors:
      bigger, smaller, brighter, darker, warmer, metallic, wooden,
      glass, cathedral, intimate, ethereal, haunted, telephone,
      underwater, vintage

    Examples:
      /irtransform bigger             Make the space sound bigger
      /irtransform metallic 0.5       Add subtle metallic quality
      /irtransform chain bigger darker Apply bigger then darker
    """
    from ..dsp.convolution import (
        get_convolution_engine, ir_transform, list_descriptors
    )

    engine = get_convolution_engine()

    if not args:
        return cmd_irtransform.__doc__

    sub = args[0].lower()

    if sub == 'list':
        descriptors = list_descriptors()
        return "Available Descriptors:\n" + '\n'.join(f"  {d}" for d in descriptors)

    elif sub == 'chain':
        if engine.ir is None:
            return "ERROR: No IR loaded. Use /conv load first"
        if len(args) < 2:
            return "Usage: /irtransform chain <desc1> <desc2> ..."
        results = []
        for desc in args[1:]:
            engine.ir = ir_transform(engine.ir, desc, 1.0, engine.sr)
            results.append(desc)
        engine._split_ir()
        return f"Applied chain: {' → '.join(results)}"

    elif sub == 'help':
        return cmd_irtransform.__doc__

    else:
        # Direct descriptor application
        descriptor = sub
        if engine.ir is None:
            return "ERROR: No IR loaded. Use /conv load first"

        available = list_descriptors()
        if descriptor not in available:
            return (f"ERROR: Unknown descriptor '{descriptor}'. "
                    f"Available: {', '.join(available)}")

        intensity = float(args[1]) if len(args) > 1 else 1.0
        intensity = max(0.0, min(intensity, 2.0))

        original_dur = len(engine.ir) / engine.sr
        engine.ir = ir_transform(engine.ir, descriptor, intensity, engine.sr)
        engine._split_ir()
        new_dur = len(engine.ir) / engine.sr

        return (f"Transformed IR: '{descriptor}' (intensity {intensity:.1f}). "
                f"Duration: {original_dur:.2f}s → {new_dur:.2f}s")


# ============================================================================
# FEATURE 3.6: GRANULAR IR TOOLS
# ============================================================================

def cmd_irgranular(session: "Session", args: List[str]) -> str:
    """Granular processing tools for impulse responses.

    Usage:
      /irgranular stretch <factor> [grain_ms] [density]
                                     Stretch IR using grains
      /irgranular morph <ir_a> <ir_b> [pos]
                                     Morph between two IRs
      /irgranular redesign [grain_ms] [density] [scatter] [reverse_prob]
                                     Redesign IR with granular resynthesis
      /irgranular freeze <position>  Freeze IR at position (0-1)

    Examples:
      /irgranular stretch 2.0 40 8   Stretch IR to 2x with 40ms grains
      /irgranular morph hall plate 0.5
                                     50/50 morph of hall and plate
      /irgranular redesign 20 6 0.3  Redesign with small scattered grains
      /irgranular freeze 0.3         Freeze at 30% position
    """
    from ..dsp.convolution import (
        get_convolution_engine, granular_ir_stretch,
        granular_ir_morph, granular_ir_redesign
    )

    engine = get_convolution_engine()

    if not args:
        return cmd_irgranular.__doc__

    sub = args[0].lower()

    if sub == 'stretch':
        if engine.ir is None:
            return "ERROR: No IR loaded"
        factor = float(args[1]) if len(args) > 1 else 2.0
        grain_ms = float(args[2]) if len(args) > 2 else 40.0
        density = float(args[3]) if len(args) > 3 else 8.0
        original = len(engine.ir) / engine.sr
        engine.ir = granular_ir_stretch(engine.ir, factor, grain_ms, density, engine.sr)
        engine._split_ir()
        new_dur = len(engine.ir) / engine.sr
        return (f"Granular stretched IR: {original:.2f}s → {new_dur:.2f}s "
                f"(grains: {grain_ms:.0f}ms, density: {density:.0f})")

    elif sub == 'morph':
        if len(args) < 3:
            return "Usage: /irgranular morph <ir_name_a> <ir_name_b> [morph_pos]"
        name_a, name_b = args[1], args[2]
        pos = float(args[3]) if len(args) > 3 else 0.5

        ir_a = engine.ir_bank.get(name_a)
        ir_b = engine.ir_bank.get(name_b)
        if ir_a is None:
            return f"ERROR: IR '{name_a}' not in bank"
        if ir_b is None:
            return f"ERROR: IR '{name_b}' not in bank"

        engine.ir = granular_ir_morph(ir_a, ir_b, pos, sr=engine.sr)
        engine.ir_name = f"morph({name_a},{name_b},{pos:.1f})"
        engine._split_ir()
        return f"Morphed IRs: {name_a} ←{pos:.0%}→ {name_b}"

    elif sub == 'redesign':
        if engine.ir is None:
            return "ERROR: No IR loaded"
        grain_ms = float(args[1]) if len(args) > 1 else 20.0
        density = float(args[2]) if len(args) > 2 else 6.0
        scatter = float(args[3]) if len(args) > 3 else 0.3
        rev_prob = float(args[4]) if len(args) > 4 else 0.2
        engine.ir = granular_ir_redesign(
            engine.ir, grain_ms, density, scatter, rev_prob, engine.sr)
        engine._split_ir()
        return (f"Redesigned IR (grains: {grain_ms:.0f}ms, "
                f"density: {density:.0f}, scatter: {scatter:.1f}, "
                f"reverse: {rev_prob:.0%})")

    elif sub == 'freeze':
        if engine.ir is None:
            return "ERROR: No IR loaded"
        pos = float(args[1]) if len(args) > 1 else 0.5
        pos = max(0.0, min(pos, 1.0))
        try:
            from ..dsp.granular import GranularEngine
            gr = GranularEngine(engine.sr)
            gr.set_params(grain_size_ms=30, density=10, envelope='hann')
            target_dur = len(engine.ir) / engine.sr
            engine.ir = gr.freeze(engine.ir, target_dur, pos)
            engine._split_ir()
            return f"Frozen IR at position {pos:.0%}"
        except Exception as e:
            return f"ERROR: Granular freeze failed: {e}"

    elif sub == 'help':
        return cmd_irgranular.__doc__

    return f"ERROR: Unknown subcommand '{sub}'. Use /irgranular help"


# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

def get_convolution_commands() -> Dict[str, callable]:
    """Get all Phase 3 convolution commands for registration."""
    return {
        # Feature 3.1: Impulse → LFO
        'impulselfo': cmd_impulselfo,
        'ilfo': cmd_impulselfo,
        'lfoimport': cmd_impulselfo,
        # Feature 3.2: Impulse → Envelope
        'impenv': cmd_impenv,
        'ienv': cmd_impenv,
        'envimport': cmd_impenv,
        # Feature 3.3: Advanced Convolution
        'conv': cmd_conv,
        'convolution': cmd_conv,
        'convrev': cmd_conv,
        # Feature 3.4: Neural IR Enhancement
        'irenhance': cmd_irenhance,
        'ire': cmd_irenhance,
        'enhance': cmd_irenhance,
        # Feature 3.5: AI Descriptor Transformation
        'irtransform': cmd_irtransform,
        'irt': cmd_irtransform,
        'transform': cmd_irtransform,
        # Feature 3.6: Granular IR
        'irgranular': cmd_irgranular,
        'irg': cmd_irgranular,
        'irgrains': cmd_irgranular,
    }
