"""Stub commands for unimplemented MDMA features.

Per Anti-Drift rule L2: If a feature is not implemented yet, it still 
exists as a stub command/module. Stub must explain ownership + planned behavior.

These stubs ensure the command interface is stable and users know what's
coming. Each stub returns a clear message about the feature's status.

NOTE: Many generator commands are now IMPLEMENTED using generators.py
"""

from __future__ import annotations
from typing import List
import numpy as np
from ..core.session import Session


def _stub_response(feature: str, owner: str, planned: str) -> str:
    """Generate a consistent stub response."""
    return (f"STUB: {feature}\n"
            f"  Owner: {owner}\n"
            f"  Status: Not yet implemented\n"
            f"  Planned: {planned}")


def _generate_and_store(session: Session, gen_name: str, variant: int = 1,
                        duration: float = None, freq: float = None) -> str:
    """Generate audio and store in session buffer."""
    try:
        from ..dsp.generators import generate
        
        audio = generate(gen_name, variant=variant, duration=duration, 
                        freq=freq, sr=session.sample_rate)
        
        if audio is None:
            return f"ERROR: Generator '{gen_name}' not found"
        
        # Store in session
        session.last_buffer = audio
        
        # Store in working buffer with proper source tracking
        try:
            from .working_cmds import get_working_buffer
            wb = get_working_buffer()
            wb.set_from_generator(audio, gen_name, variant, session.sample_rate)
        except Exception:
            pass
        
        duration_s = len(audio) / session.sample_rate
        return f"OK: generated {gen_name}_{variant} ({duration_s:.3f}s) -> working buffer"
        
    except Exception as e:
        return f"ERROR: Generation failed: {e}"


# ============================================================================
# A) CORE SYNTH STUBS
# ============================================================================

def cmd_pm(session: Session, args: List[str]) -> str:
    """Phase Modulation routing - NOW IMPLEMENTED.
    
    This stub exists for backwards compatibility.
    The actual implementation is in synth_cmds.py.
    
    Usage: /pm <source> <target> <amount>
    """
    # This should not be called as the real implementation exists
    # but keeping stub for documentation
    return ("PM routing is now implemented in synth_cmds.\n"
            "Usage: /pm <source> <target> <amount>\n"
            "Example: /pm 1 0 2.5")


def cmd_stereo(session: Session, args: List[str]) -> str:
    """Voice stereo spread - NOW IMPLEMENTED.
    
    Usage: /stereo <spread>
    
    See synth_cmds.py for full implementation.
    """
    return ("Stereo spread is now implemented in synth_cmds.\n"
            "Usage: /stereo <0-100>  (0=mono, 100=full width)")


def cmd_vphase(session: Session, args: List[str]) -> str:
    """Voice phase offset - NOW IMPLEMENTED.
    
    Usage: /vphase <offset>
    
    See synth_cmds.py for full implementation.
    """
    return ("Voice phase offset is now implemented in synth_cmds.\n"
            "Usage: /vphase <0-360>  (degrees per voice)")


def cmd_venv(session: Session, args: List[str]) -> str:
    """Voice envelope offset - NOW IMPLEMENTED.
    
    Usage: /venv <offset_ms>
    
    See synth_cmds.py for full implementation.
    """
    return ("Voice envelope offset is now implemented in synth_cmds.\n"
            "Usage: /venv <ms>  (envelope start offset per voice)")


def cmd_fenv(session: Session, args: List[str]) -> str:
    """Frequency envelope - NOW IMPLEMENTED.
    
    Usage: /fenv <atk> <dec> <sus> <rel> [amount]
    
    See synth_cmds.py for full implementation.
    """
    return ("Frequency envelope is now implemented in synth_cmds.\n"
            "Usage: /fenv <a> <d> <s> <r>  or  /fenv amount <semitones>")


def cmd_menv(session: Session, args: List[str]) -> str:
    """Modulation envelope - NOW IMPLEMENTED.
    
    Usage: /menv <atk> <dec> <sus> <rel> [target]
    
    See synth_cmds.py for full implementation.
    """
    return ("Modulation envelope is now implemented in synth_cmds.\n"
            "Usage: /menv <a> <d> <s> <r>  or  /menv depth <amount>")


# ============================================================================
# B) WAVE MODEL STUBS
# ============================================================================

def cmd_pulse(session: Session, args: List[str]) -> str:
    """Set operator to pulse wave with PWM - NOW IMPLEMENTED.
    
    Use /wm pulse to set wave type, then /pw to adjust width.
    Or: /wm pulse pw=0.3
    """
    return ("Pulse wave is now implemented.\n"
            "Usage: /wm pulse        -> Set wave to pulse\n"
            "       /pw <0-1>        -> Set pulse width\n"
            "       /wm pulse pw=0.3 -> Combined")


def cmd_pink(session: Session, args: List[str]) -> str:
    """Set operator to pink noise - NOW IMPLEMENTED.
    
    Use /wm pink to set wave type.
    """
    return ("Pink noise is now implemented.\n"
            "Usage: /wm pink -> Set wave to pink noise")


def cmd_phys(session: Session, args: List[str]) -> str:
    """Physical modeling wave - NOW IMPLEMENTED.
    
    Use /wm physical or /wm phys, then /phys to configure.
    """
    return ("Physical modeling is now implemented.\n"
            "Usage: /wm phys          -> Set wave to physical\n"
            "       /phys <even> <odd> -> Configure harmonics\n"
            "See /phys for full parameter list")


def cmd_phys2(session: Session, args: List[str]) -> str:
    """Physical modeling wave variant 2 - NOW IMPLEMENTED.
    
    Use /wm physical2 or /wm phys2, then /phys2 to configure.
    """
    return ("Physical2 modeling is now implemented.\n"
            "Usage: /wm phys2             -> Set wave to physical2\n"
            "       /phys2 <inharm> <n>   -> Configure inharmonicity\n"
            "See /phys2 for full parameter list")


# ============================================================================
# B3) UMPULSE SYSTEM STUBS
# ============================================================================

def cmd_ump(session: Session, args: List[str]) -> str:
    """Umpulse system for custom impulse/wavetable import - NOW IMPLEMENTED.
    
    Usage:
      /ump load <path> [name]  - Load impulse from file
      /ump buffer [name]       - Load from current buffer
      /ump list                - List named impulses
      /ump use <n> wave <op>   - Use as operator waveform
      /ump use <n> ir          - Use for convolution
      /ump del <n>             - Delete impulse
      /ump norm <n>            - Normalize impulse
      /ump wavetable <n>       - Convert to wavetable
    
    See audiorate_cmds.py for full implementation.
    """
    # Delegate to real implementation
    try:
        from .audiorate_cmds import cmd_ump as real_cmd
        return real_cmd(session, args)
    except ImportError:
        if not args:
            return ("Umpulse is now implemented.\n"
                    "Subcommands: load, buffer, list, info, use, del, norm, wavetable")
        return f"Umpulse subcommand: {args[0]} (delegating to audiorate_cmds)"


# ============================================================================
# C) MAJOR EFFECTS STUBS
# ============================================================================

def cmd_vamp(session: Session, args: List[str]) -> str:
    """Vamp Overdrive saturation effects.
    
    Usage: 
      /vamp light | medium | heavy | fuzz  (presets)
      /vamp                                 (show available presets)
    
    These effects are implemented - applies immediately to buffer.
    """
    if session.last_buffer is None:
        return "ERROR: no buffer to process. Generate audio first."
    
    presets = {
        'light': 'vamp_light',
        'medium': 'vamp_medium',
        'heavy': 'vamp_heavy',
        'fuzz': 'vamp_fuzz',
    }
    
    if not args:
        return "VAMP OVERDRIVE:\n  /vamp light - subtle warmth\n  /vamp medium - classic overdrive\n  /vamp heavy - aggressive distortion\n  /vamp fuzz - fuzzy saturation"
    
    preset = args[0].lower()
    if preset not in presets:
        return f"ERROR: unknown preset '{preset}'. Use: light, medium, heavy, fuzz"
    
    # Use fx system
    from .fx_cmds import cmd_fx
    return cmd_fx(session, [presets[preset]])


def cmd_bbe(session: Session, args: List[str]) -> str:
    """Big Brain Evolver convolution reverb - NOW USES UMPULSE.
    
    Usage: /bbe <impulse_name> [wet] [predelay]
    
    Load an IR with /ump first, then use /bbe to apply convolution.
    
    Examples:
      /ump load hall.wav hall_ir
      /bbe hall_ir 50        Apply with 50% wet
    """
    if not args:
        return ("Big Brain Evolver - Convolution Reverb\n"
                "Usage: /bbe <impulse_name> [wet_percent]\n\n"
                "First load an impulse with /ump:\n"
                "  /ump load <ir_file> <n>\n"
                "  /ump buffer <n>     (use current buffer)\n\n"
                "Then apply convolution:\n"
                "  /bbe <n> [wet]")
    
    impulse_name = args[0]
    wet = float(args[1]) / 100.0 if len(args) > 1 else 0.5
    
    # Get impulse from umpulse bank
    try:
        from .audiorate_cmds import get_umpulse_bank
        bank = get_umpulse_bank()
        data = bank.get(impulse_name)
        
        if not data:
            return f"ERROR: Impulse '{impulse_name}' not found. Use /ump list to see available."
        
        ir_audio, ir_sr, _ = data
        
        # Get audio to process
        if session.last_buffer is None or len(session.last_buffer) == 0:
            return "ERROR: No audio in buffer. Generate audio first."
        
        audio = session.last_buffer
        
        # Simple convolution reverb
        from scipy import signal
        convolved = signal.fftconvolve(audio, ir_audio, mode='full')[:len(audio)]
        
        # Normalize
        peak = np.max(np.abs(convolved))
        if peak > 0:
            convolved = convolved / peak * np.max(np.abs(audio))
        
        # Mix
        result = audio * (1 - wet) + convolved * wet
        
        session.last_buffer = result
        
        return f"OK: Applied convolution with '{impulse_name}' (wet={wet*100:.0f}%)"
        
    except ImportError:
        return "ERROR: audiorate_cmds not available"
    except Exception as e:
        return f"ERROR: Convolution failed: {e}"


def cmd_fc(session: Session, args: List[str]) -> str:
    """Forever Compression multiband OTT-style compressor.
    
    Usage: 
      /fc punch | glue | loud | soft | ott  (presets)
      /fc                                    (show available presets)
    
    These effects are implemented - applies immediately to buffer.
    """
    if session.last_buffer is None:
        return "ERROR: no buffer to process. Generate audio first."
    
    presets = {
        'punch': 'fc_punch',
        'glue': 'fc_glue',
        'loud': 'fc_loud',
        'soft': 'fc_soft',
        'ott': 'fc_ott',
    }
    
    if not args:
        return "FOREVER COMPRESSION:\n  /fc punch - punchy attack\n  /fc glue - glue/bus compression\n  /fc loud - loudness maximizer\n  /fc soft - gentle multiband\n  /fc ott - OTT-style extreme"
    
    preset = args[0].lower()
    if preset not in presets:
        return f"ERROR: unknown preset '{preset}'. Use: punch, glue, loud, soft, ott"
    
    # Use fx system
    from .fx_cmds import cmd_fx
    return cmd_fx(session, [presets[preset]])


def cmd_gg(session: Session, args: List[str]) -> str:
    """Giga Gate pattern-based gating and effects.
    
    Usage:
      /gg half              - Half pattern
      /gg quarter           - Quarter pattern
      /gg tresillo          - Tresillo rhythm
      /gg glitch            - Glitch pattern
      /gg stutter           - Stutter effect
      /gg halftime          - Halftime effect
      /gg tape              - Tape stop effect
      /gg                   - Show available patterns
    
    These effects are implemented - applies immediately to buffer.
    """
    if session.last_buffer is None:
        return "ERROR: no buffer loaded. Use /tone, /g, or /upl first"
    
    presets = {
        'half': 'gg_half',
        'quarter': 'gg_quarter',
        'tresillo': 'gg_tresillo',
        'glitch': 'gg_glitch',
        'stutter': 'gg_stutter',
        'halftime': 'gg_halftime',
        'tape': 'gg_tape_stop',
        'tapestop': 'gg_tape_stop',
    }
    
    if not args:
        return "GIGA GATE:\n  /gg half - half pattern\n  /gg quarter - quarter pattern\n  /gg tresillo - tresillo rhythm\n  /gg glitch - glitch pattern\n  /gg stutter - stutter effect\n  /gg halftime - halftime\n  /gg tape - tape stop"
    
    preset = args[0].lower()
    if preset not in presets:
        return f"ERROR: unknown pattern '{preset}'. Use: half, quarter, tresillo, glitch, stutter, halftime, tape"
    
    # Use fx system
    from .fx_cmds import cmd_fx
    return cmd_fx(session, [presets[preset]])


def cmd_voc(session: Session, args: List[str]) -> str:
    """Vocoder effect.
    
    Usage: 
      /voc synth | noise | chord  (presets)
      /voc                        (show available presets)
    
    These effects are implemented - applies immediately to buffer.
    """
    if session.last_buffer is None and args:
        return "ERROR: no buffer to process. Generate audio first."
    
    presets = {
        'synth': 'vocoder_synth',
        'noise': 'vocoder_noise',
        'chord': 'vocoder_chord',
    }
    
    if not args:
        return "VOCODER:\n  /voc synth - sawtooth carrier\n  /voc noise - noise carrier (robot)\n  /voc chord - chord carrier\n\nApply to buffer with voice/speech for best results."
    
    preset = args[0].lower()
    if preset not in presets:
        return f"ERROR: unknown preset '{preset}'. Use: synth, noise, chord"
    
    # Use fx system
    from .fx_cmds import cmd_fx
    return cmd_fx(session, [presets[preset]])


def cmd_spc(session: Session, args: List[str]) -> str:
    """Spectral processing effects.
    
    Usage:
      /spc freeze | blur | up | down  (presets)
      /spc                            (show available presets)
    
    These effects are implemented - applies immediately to buffer.
    """
    if session.last_buffer is None and args:
        return "ERROR: no buffer to process. Generate audio first."
    
    presets = {
        'freeze': 'spc_freeze',
        'blur': 'spc_blur',
        'up': 'spc_shift_up',
        'down': 'spc_shift_down',
        'shiftup': 'spc_shift_up',
        'shiftdown': 'spc_shift_down',
    }
    
    if not args:
        return "SPECTRAL PROCESSING:\n  /spc freeze - freeze spectrum (drone)\n  /spc blur - spectral blur/smear\n  /spc up - shift up 5 semitones\n  /spc down - shift down 5 semitones"
    
    preset = args[0].lower()
    if preset not in presets:
        return f"ERROR: unknown preset '{preset}'. Use: freeze, blur, up, down"
    
    # Use fx system
    from .fx_cmds import cmd_fx
    return cmd_fx(session, [presets[preset]])


# ============================================================================
# D) FILTER/LFO STUBS  
# ============================================================================

def cmd_lfo(session: Session, args: List[str]) -> str:
    """LFO effects for parameter modulation.
    
    Usage:
      /lfo slow | fast | tremolo | vibrato  (presets)
      /lfo                                   (show available presets)
    
    These effects are implemented - applies immediately to buffer.
    """
    if session.last_buffer is None and args:
        return "ERROR: no buffer to process. Generate audio first."
    
    presets = {
        'slow': 'lfo_filter_slow',
        'fast': 'lfo_filter_fast',
        'tremolo': 'lfo_tremolo',
        'vibrato': 'lfo_vibrato',
        'filter': 'lfo_filter_slow',
        'wah': 'lfo_filter_fast',
    }
    
    if not args:
        return "LFO EFFECTS:\n  /lfo slow - slow filter sweep\n  /lfo fast - fast wah-wah sweep\n  /lfo tremolo - amplitude tremolo\n  /lfo vibrato - pitch vibrato"
    
    preset = args[0].lower()
    if preset not in presets:
        return f"ERROR: unknown preset '{preset}'. Use: slow, fast, tremolo, vibrato"
    
    # Use fx system
    from .fx_cmds import cmd_fx
    return cmd_fx(session, [presets[preset]])


def cmd_audiorate(session: Session, args: List[str]) -> str:
    """Audio-rate modulation - NOW IMPLEMENTED.
    
    Usage: /audiorate <subcommand>
    
    See audiorate_cmds.py for full implementation.
    Subcommands: interval, filter, pattern, clear, presets
    """
    # Delegate to real implementation
    try:
        from .audiorate_cmds import cmd_audiorate as real_cmd
        return real_cmd(session, args)
    except ImportError:
        return ("Audio-rate modulation is now implemented.\n"
                "Usage: /audiorate interval <op> <type> <rate> <depth>\n"
                "       /audiorate filter <type> <rate> <depth>\n"
                "       /audiorate pattern <op> <pattern>\n"
                "       /audiorate clear [all|filter|<op>]")


# ============================================================================
# E) ROUTING BANK STUBS
# ============================================================================

def cmd_bk(session: Session, args: List[str]) -> str:
    """Routing bank selection (STUB).
    
    Usage:
      /bk           - Show current bank
      /bk <index>   - Select bank by index (0-7)
      /bk list      - List all banks
    
    Owner: E1 - Routing Banks
    Planned: 8 banks of 16 algorithms each.
    """
    if not args:
        return ("STUB: /bk - Routing bank system\n"
                "  Current bank: (not implemented)\n"
                "  Banks: DMX, Modern Clean, Modern Aggressive, Noisy,\n"
                "         Physical, Glitch, Hybrid, Custom\n"
                "  Section E1: Not yet implemented")
    
    if args[0].lower() == 'list':
        return _stub_response(
            "Routing bank list",
            "Section E1: Routing Banks",
            "8 banks: DMX/Classic, Modern Clean, Modern Aggressive, "
            "Noisy/Experimental, Physical-Model, Glitch/Metallic, Hybrid, Custom"
        )
    
    try:
        idx = int(args[0])
        return _stub_response(
            f"Select routing bank {idx}",
            "Section E1: Routing Banks",
            f"Load bank {idx} with 16 preset algorithms"
        )
    except ValueError:
        return f"ERROR: invalid bank index: {args[0]}"


def cmd_al(session: Session, args: List[str]) -> str:
    """Algorithm selection from current bank (STUB).
    
    Usage:
      /al           - Show current algorithm
      /al <index>   - Load algorithm by index (0-15)
      /al list      - List algorithms in current bank
    
    Owner: E2 - Algorithm Count Targets
    Planned: 16 algorithms per bank.
    """
    if not args:
        return ("STUB: /al - Algorithm selection\n"
                "  Current algorithm: (not implemented)\n"
                "  Section E2: Not yet implemented")
    
    if args[0].lower() == 'list':
        return _stub_response(
            "Algorithm list",
            "Section E2: Algorithm Count Targets",
            "List 16 algorithms in current routing bank"
        )
    
    try:
        idx = int(args[0])
        return _stub_response(
            f"Load algorithm {idx}",
            "Section E2: Algorithm Count Targets",
            f"Load algorithm {idx} from current bank, setting up operator routing"
        )
    except ValueError:
        return f"ERROR: invalid algorithm index: {args[0]}"


def cmd_preset(session: Session, args: List[str]) -> str:
    """Load preset from preset pack (STUB).
    
    Usage: /preset <category> <index>
    
    Categories: bass, lead, pad, pluck, perc, atmos, noise, fx, physical
    
    Owner: E3 - Base Preset Pack
    Planned: 60-100 built-in patches.
    """
    if not args:
        return ("STUB: /preset - Monolith preset pack\n"
                "  Usage: /preset <category> <index>\n"
                "  Categories: bass, lead, pad, pluck, perc, atmos, noise, fx, physical\n"
                "  Section E3: Not yet implemented")
    
    cat = args[0].lower()
    idx = args[1] if len(args) > 1 else "1"
    return _stub_response(
        f"Load preset: {cat} #{idx}",
        "Section E3: Base Preset Pack",
        f"Load {cat} preset {idx} from built-in Monolith patch library"
    )


# ============================================================================
# J) GENERATOR COMMANDS (IMPLEMENTED)
# ============================================================================

def _gen_help(name: str, desc: str, variants: str) -> str:
    """Generate help text for a generator."""
    return (f"/g_{name} - {desc}\n"
            f"  Usage: /g_{name} [variant] [duration]\n"
            f"  Variants: {variants}\n"
            f"  Audio is stored in working buffer, use /A to append to buffer")


def cmd_g_tom(session: Session, args: List[str]) -> str:
    """Generate tom sound.
    
    Usage: /g_tom [variant] [duration]
    Variants: 1=low, 2=mid, 3=high, 4=electronic
    """
    if args and args[0].lower() in ('help', '?'):
        return _gen_help("tom", "Tom drums", "1=low, 2=mid, 3=high, 4=electronic")
    
    variant = int(args[0]) if args and args[0].isdigit() else 1
    duration = float(args[1]) if len(args) > 1 else None
    return _generate_and_store(session, 'tom', variant, duration)


def cmd_g_cym(session: Session, args: List[str]) -> str:
    """Generate cymbal sound.
    
    Usage: /g_cym [variant] [duration]
    Variants: 1=crash, 2=ride, 3=china, 4=splash
    """
    if args and args[0].lower() in ('help', '?'):
        return _gen_help("cym", "Cymbals", "1=crash, 2=ride, 3=china, 4=splash")
    
    variant = int(args[0]) if args and args[0].isdigit() else 1
    duration = float(args[1]) if len(args) > 1 else None
    return _generate_and_store(session, 'cym', variant, duration)


def cmd_g_clp(session: Session, args: List[str]) -> str:
    """Generate clap sound.
    
    Usage: /g_clp [variant]
    Variants: 1=808, 2=acoustic, 3=layered, 4=tight
    """
    if args and args[0].lower() in ('help', '?'):
        return _gen_help("clp", "Claps", "1=808, 2=acoustic, 3=layered, 4=tight")
    
    variant = int(args[0]) if args and args[0].isdigit() else 1
    return _generate_and_store(session, 'clp', variant)


def cmd_g_snp(session: Session, args: List[str]) -> str:
    """Generate snap sound.
    
    Usage: /g_snp [variant]
    Variants: 1=dry, 2=reverb
    """
    if args and args[0].lower() in ('help', '?'):
        return _gen_help("snp", "Finger snaps", "1=dry, 2=reverb")
    
    variant = int(args[0]) if args and args[0].isdigit() else 1
    return _generate_and_store(session, 'snp', variant)


def cmd_g_shk(session: Session, args: List[str]) -> str:
    """Generate shaker sound.
    
    Usage: /g_shk [variant]
    Variants: 1=16th, 2=8th, 3=triplet
    """
    if args and args[0].lower() in ('help', '?'):
        return _gen_help("shk", "Shakers", "1=16th, 2=8th, 3=triplet")
    
    variant = int(args[0]) if args and args[0].isdigit() else 1
    return _generate_and_store(session, 'shk', variant)


def cmd_g_blp(session: Session, args: List[str]) -> str:
    """Generate bleep hit.
    
    Usage: /g_blp [variant] [freq]
    Variants: 1=short, 2=pitched, 3=glitch, 4=zap
    """
    if args and args[0].lower() in ('help', '?'):
        return _gen_help("blp", "Bleeps/blips", "1=short, 2=pitched, 3=glitch, 4=zap")
    
    variant = int(args[0]) if args and args[0].isdigit() else 1
    freq = float(args[1]) if len(args) > 1 and args[1].replace('.','').isdigit() else None
    return _generate_and_store(session, 'blp', variant, freq=freq)


def cmd_g_stb(session: Session, args: List[str]) -> str:
    """Generate stab sound.
    
    Usage: /g_stb [variant] [freq]
    Variants: 1=chord, 2=brass, 3=synth, 4=orchestral
    """
    if args and args[0].lower() in ('help', '?'):
        return _gen_help("stb", "Stabs", "1=chord, 2=brass, 3=synth, 4=orchestral")
    
    variant = int(args[0]) if args and args[0].isdigit() else 1
    freq = float(args[1]) if len(args) > 1 and args[1].replace('.','').isdigit() else None
    return _generate_and_store(session, 'stb', variant, freq=freq)


def cmd_g_zap(session: Session, args: List[str]) -> str:
    """Generate zap sound.
    
    Usage: /g_zap [variant]
    Variants: 1=short, 2=long, 3=sweep, 4=retro
    """
    if args and args[0].lower() in ('help', '?'):
        return _gen_help("zap", "Zaps/lasers", "1=short, 2=long, 3=sweep, 4=retro")
    
    variant = int(args[0]) if args and args[0].isdigit() else 1
    return _generate_and_store(session, 'zap', variant)


def cmd_g_lsr(session: Session, args: List[str]) -> str:
    """Generate laser FX.
    
    Usage: /g_lsr [variant]
    Variants: 1=pew, 2=beam, 3=charge
    """
    if args and args[0].lower() in ('help', '?'):
        return _gen_help("lsr", "Laser FX", "1=pew, 2=beam, 3=charge")
    
    variant = int(args[0]) if args and args[0].isdigit() else 1
    return _generate_and_store(session, 'lsr', variant)


def cmd_g_bel(session: Session, args: List[str]) -> str:
    """Generate bell ping.
    
    Usage: /g_bel [variant] [freq]
    Variants: 1=bright, 2=dark, 3=tubular, 4=glass
    """
    if args and args[0].lower() in ('help', '?'):
        return _gen_help("bel", "Bells", "1=bright, 2=dark, 3=tubular, 4=glass")
    
    variant = int(args[0]) if args and args[0].isdigit() else 1
    freq = float(args[1]) if len(args) > 1 and args[1].replace('.','').isdigit() else None
    return _generate_and_store(session, 'bel', variant, freq=freq)


def cmd_g_bas(session: Session, args: List[str]) -> str:
    """Generate short bass hit.
    
    Usage: /g_bas [variant] [freq]
    Variants: 1=808, 2=sub, 3=punch, 4=growl
    """
    if args and args[0].lower() in ('help', '?'):
        return _gen_help("bas", "Bass hits", "1=808, 2=sub, 3=punch, 4=growl")
    
    variant = int(args[0]) if args and args[0].isdigit() else 1
    freq = float(args[1]) if len(args) > 1 and args[1].replace('.','').isdigit() else None
    return _generate_and_store(session, 'bas', variant, freq=freq)


def cmd_g_rsr(session: Session, args: List[str]) -> str:
    """Generate riser.
    
    Usage: /g_rsr [variant] [duration]
    Variants: 1=noise, 2=tonal, 3=sweep, 4=tension
    """
    if args and args[0].lower() in ('help', '?'):
        return _gen_help("rsr", "Risers", "1=noise, 2=tonal, 3=sweep, 4=tension")
    
    variant = int(args[0]) if args and args[0].isdigit() else 1
    duration = float(args[1]) if len(args) > 1 else 2.0
    return _generate_and_store(session, 'rsr', variant, duration)


def cmd_g_dwn(session: Session, args: List[str]) -> str:
    """Generate downlifter.
    
    Usage: /g_dwn [variant] [duration]
    Variants: 1=drop, 2=sweep, 3=impact, 4=reverse
    """
    if args and args[0].lower() in ('help', '?'):
        return _gen_help("dwn", "Downlifters", "1=drop, 2=sweep, 3=impact, 4=reverse")
    
    variant = int(args[0]) if args and args[0].isdigit() else 1
    duration = float(args[1]) if len(args) > 1 else 1.0
    return _generate_and_store(session, 'dwn', variant, duration)


def cmd_g_wsh(session: Session, args: List[str]) -> str:
    """Generate whoosh.
    
    Usage: /g_wsh [variant] [duration]
    Variants: 1=fast, 2=slow, 3=textured
    """
    if args and args[0].lower() in ('help', '?'):
        return _gen_help("wsh", "Whooshes", "1=fast, 2=slow, 3=textured")
    
    variant = int(args[0]) if args and args[0].isdigit() else 1
    duration = float(args[1]) if len(args) > 1 else 0.5
    return _generate_and_store(session, 'wsh', variant, duration)


def cmd_g_glt(session: Session, args: List[str]) -> str:
    """Generate glitch.
    
    Usage: /g_glt [variant]
    Variants: 1=stutter, 2=buffer, 3=bitcrush, 4=random
    """
    if args and args[0].lower() in ('help', '?'):
        return _gen_help("glt", "Glitches", "1=stutter, 2=buffer, 3=bitcrush, 4=random")
    
    variant = int(args[0]) if args and args[0].isdigit() else 1
    return _generate_and_store(session, 'glt', variant)


def cmd_g_vnl(session: Session, args: List[str]) -> str:
    """Generate vinyl texture.
    
    Usage: /g_vnl [variant] [duration]
    Variants: 1=crackle, 2=hiss
    """
    if args and args[0].lower() in ('help', '?'):
        return _gen_help("vnl", "Vinyl texture", "1=crackle, 2=hiss")
    
    variant = int(args[0]) if args and args[0].isdigit() else 1
    duration = float(args[1]) if len(args) > 1 else 2.0
    return _generate_and_store(session, 'vnl', variant, duration)


def cmd_g_wnd(session: Session, args: List[str]) -> str:
    """Generate wind/noise sweep.
    
    Usage: /g_wnd [variant] [duration]
    Variants: 1=gentle, 2=harsh, 3=filtered
    """
    if args and args[0].lower() in ('help', '?'):
        return _gen_help("wnd", "Wind/noise", "1=gentle, 2=harsh, 3=filtered")
    
    variant = int(args[0]) if args and args[0].isdigit() else 1
    duration = float(args[1]) if len(args) > 1 else 2.0
    return _generate_and_store(session, 'wnd', variant, duration)


def cmd_g_sil(session: Session, args: List[str]) -> str:
    """Generate silence/spacer.
    
    Usage: /g_sil [duration]
    """
    duration = float(args[0]) if args else 1.0
    return _generate_and_store(session, 'sil', 1, duration)


def cmd_g_clk(session: Session, args: List[str]) -> str:
    """Generate click track.
    
    Usage: /g_clk [bpm] [bars]
    """
    try:
        from ..dsp.generators import generate_click
        
        bpm = float(args[0]) if args else 120
        bars = int(args[1]) if len(args) > 1 else 4
        
        audio = generate_click(bpm, bars, session.sample_rate)
        session.last_buffer = audio
        
        # Store in working buffer
        try:
            from .working_cmds import get_working_buffer
            wb = get_working_buffer()
            wb.set(audio, f"click_{bpm}bpm", session.sample_rate)
        except ImportError:
            pass
        
        duration = len(audio) / session.sample_rate
        return f"OK: Generated click track @ {bpm}BPM, {bars} bars ({duration:.2f}s)"
        
    except Exception as e:
        return f"ERROR: {e}"


def cmd_g_cal(session: Session, args: List[str]) -> str:
    """Generate calibration tone.
    
    Usage: /g_cal [freq] [duration]
    """
    try:
        from ..dsp.generators import generate_calibration
        
        freq = float(args[0]) if args else 1000
        duration = float(args[1]) if len(args) > 1 else 1.0
        
        audio = generate_calibration(freq, duration, session.sample_rate)
        session.last_buffer = audio
        
        # Store in working buffer
        try:
            from .working_cmds import get_working_buffer
            wb = get_working_buffer()
            wb.set(audio, f"cal_{freq}hz", session.sample_rate)
        except ImportError:
            pass
        
        return f"OK: Generated {freq}Hz calibration tone ({duration}s)"
        
    except Exception as e:
        return f"ERROR: {e}"


def cmd_g_swp(session: Session, args: List[str]) -> str:
    """Generate test sweep.
    
    Usage: /g_swp [start_freq] [end_freq] [duration]
    """
    try:
        from ..dsp.generators import generate_sweep
        
        start_freq = float(args[0]) if args else 20
        end_freq = float(args[1]) if len(args) > 1 else 20000
        duration = float(args[2]) if len(args) > 2 else 5.0
        
        audio = generate_sweep(start_freq, end_freq, duration, session.sample_rate)
        session.last_buffer = audio
        
        # Store in working buffer
        try:
            from .working_cmds import get_working_buffer
            wb = get_working_buffer()
            wb.set(audio, f"sweep_{start_freq}-{end_freq}hz", session.sample_rate)
        except ImportError:
            pass
        
        return f"OK: Generated {start_freq}-{end_freq}Hz sweep ({duration}s)"
        
    except Exception as e:
        return f"ERROR: {e}"



# ============================================================================
# STUB COMMAND REGISTRY
# ============================================================================

# NOTE: Generator commands (cmd_g_*) are now IMPLEMENTED above using generators.py
# DJ Mode commands are also IMPLEMENTED in dj_cmds.py

# STUB_COMMANDS dictionary is defined at the end of this file after all functions


# ============================================================================
# DJ MODE STUBS (Future Feature Set)
# ============================================================================

def cmd_djm(session: Session, args: List[str]) -> str:
    """DJ Mode - Dual-deck mixing interface (STUB).
    
    Usage:
      /djm                    -> Toggle DJ mode on/off
      /djm on                 -> Enable DJ mode
      /djm off                -> Disable DJ mode
      /djm status             -> Show DJ mode status
    
    DJ Mode provides:
    - Dual deck playback (A/B)
    - Crossfader control
    - BPM sync and beatmatching
    - Cue points and loops
    - Song library integration
    - Playlist management
    
    Owner: DJ Feature Set (Future)
    Planned: Full DJ mixing workflow with screen-reader feedback
    """
    if not args:
        return ("=== DJ MODE ===\n"
                "STUB: DJ Mode is planned but not yet implemented.\n"
                "\n"
                "Planned Features:\n"
                "  - Dual deck playback (Deck A / Deck B)\n"
                "  - Crossfader with curves\n"
                "  - BPM detection and sync\n"
                "  - Cue points and hot cues\n"
                "  - Loop controls\n"
                "  - Song library with BPM/key tagging\n"
                "  - Playlist management\n"
                "  - Effects per deck\n"
                "\n"
                "Related commands (all stubs):\n"
                "  /deck, /cue, /sync, /xfade, /library, /playlist")
    
    sub = args[0].lower()
    if sub in ('on', 'enable'):
        return _stub_response(
            "DJ Mode Enable",
            "DJ Feature Set",
            "Enter DJ mode with dual decks and crossfader"
        )
    elif sub in ('off', 'disable'):
        return _stub_response(
            "DJ Mode Disable",
            "DJ Feature Set",
            "Exit DJ mode and return to normal operation"
        )
    elif sub == 'status':
        return _stub_response(
            "DJ Mode Status",
            "DJ Feature Set",
            "Show current DJ mode state, deck status, crossfader position"
        )
    return "STUB: DJ Mode - use /djm for help"


def cmd_deck(session: Session, args: List[str]) -> str:
    """Deck control for DJ mode (STUB).
    
    Usage:
      /deck                   -> Show both deck status
      /deck a                 -> Select deck A
      /deck b                 -> Select deck B
      /deck a load <file>     -> Load file to deck A
      /deck a play            -> Play deck A
      /deck a stop            -> Stop deck A
      /deck a pause           -> Pause deck A
      /deck a pitch <n>       -> Set pitch/tempo on deck A
    
    Owner: DJ Feature Set (Future)
    """
    if not args:
        return _stub_response(
            "Deck Status",
            "DJ Feature Set",
            "Show status of Deck A and Deck B (track, position, BPM, pitch)"
        )
    
    deck = args[0].upper()
    if deck not in ('A', 'B'):
        return f"ERROR: invalid deck '{deck}'. Use A or B."
    
    if len(args) < 2:
        return _stub_response(
            f"Deck {deck} Status",
            "DJ Feature Set",
            f"Show current state of Deck {deck}"
        )
    
    action = args[1].lower()
    return _stub_response(
        f"Deck {deck} {action.title()}",
        "DJ Feature Set",
        f"Perform '{action}' on Deck {deck}"
    )


def cmd_cue(session: Session, args: List[str]) -> str:
    """Cue point management for DJ mode (STUB).
    
    Usage:
      /cue                    -> List cue points for current deck
      /cue set <n>            -> Set cue point at current position
      /cue goto <n>           -> Jump to cue point
      /cue del <n>            -> Delete cue point
      /cue main               -> Set main cue point
    
    Owner: DJ Feature Set (Future)
    """
    if not args:
        return _stub_response(
            "Cue Points",
            "DJ Feature Set",
            "Manage cue points and hot cues for beatmatching"
        )
    
    sub = args[0].lower()
    return _stub_response(
        f"Cue {sub.title()}",
        "DJ Feature Set",
        f"Cue point operation: {sub}"
    )


def cmd_sync(session: Session, args: List[str]) -> str:
    """BPM sync for DJ mode (STUB).
    
    Usage:
      /sync                   -> Show sync status
      /sync on                -> Enable auto-sync
      /sync off               -> Disable auto-sync
      /sync a                 -> Sync deck B to deck A's tempo
      /sync b                 -> Sync deck A to deck B's tempo
      /sync tap               -> Tap tempo input
    
    Owner: DJ Feature Set (Future)
    """
    if not args:
        return _stub_response(
            "BPM Sync",
            "DJ Feature Set",
            "Automatic BPM synchronization between decks"
        )
    
    sub = args[0].lower()
    return _stub_response(
        f"Sync {sub.title()}",
        "DJ Feature Set",
        f"Sync operation: {sub}"
    )


def cmd_xfade(session: Session, args: List[str]) -> str:
    """Crossfader control for DJ mode (STUB).
    
    Usage:
      /xfade                  -> Show crossfader position
      /xfade <0-100>          -> Set crossfader position (0=A, 100=B)
      /xfade a                -> Snap to deck A
      /xfade b                -> Snap to deck B
      /xfade mid              -> Center crossfader
      /xfade curve <type>     -> Set crossfader curve (linear, log, cut)
    
    Owner: DJ Feature Set (Future)
    """
    if not args:
        return _stub_response(
            "Crossfader",
            "DJ Feature Set",
            "Control crossfader between Deck A and Deck B"
        )
    
    arg = args[0].lower()
    return _stub_response(
        f"Crossfader {arg}",
        "DJ Feature Set",
        f"Set crossfader: {arg}"
    )


def cmd_library(session: Session, args: List[str]) -> str:
    """Song library management for DJ mode (STUB).
    
    Usage:
      /library                -> Show library overview
      /library scan <path>    -> Scan folder for songs
      /library search <query> -> Search library
      /library add <file>     -> Add single file to library
      /library tag <file> bpm=<n> key=<k> -> Tag a song
      /library analyze <file> -> Auto-detect BPM and key
    
    Songs are stored in: ~/Documents/MDMA/songs/
    
    Owner: DJ Feature Set (Future)
    """
    if not args:
        return ("=== SONG LIBRARY ===\n"
                "STUB: Library management for DJ mode.\n"
                "\n"
                "Planned Features:\n"
                "  - Scan folders for audio files\n"
                "  - BPM detection and tagging\n"
                "  - Key detection\n"
                "  - Genre/mood tagging\n"
                "  - Search by BPM range, key, genre\n"
                "  - Favorites and ratings\n"
                "\n"
                "Songs folder: ~/Documents/MDMA/songs/")
    
    sub = args[0].lower()
    return _stub_response(
        f"Library {sub.title()}",
        "DJ Feature Set",
        f"Library operation: {sub}"
    )


def cmd_playlist(session: Session, args: List[str]) -> str:
    """Playlist management for DJ mode (STUB).
    
    Usage:
      /playlist               -> List all playlists
      /playlist <n>        -> Show playlist contents
      /playlist new <n>    -> Create new playlist
      /playlist add <n> <song> -> Add song to playlist
      /playlist rm <n> <idx>-> Remove song from playlist
      /playlist del <n>    -> Delete playlist
      /playlist load <n>   -> Load playlist to deck queue
    
    Playlists are stored in: ~/Documents/MDMA/songs/playlists/
    
    Owner: DJ Feature Set (Future)
    """
    if not args:
        return _stub_response(
            "Playlists",
            "DJ Feature Set",
            "View and manage playlists for DJ sets"
        )
    
    sub = args[0].lower()
    return _stub_response(
        f"Playlist {sub.title()}",
        "DJ Feature Set",
        f"Playlist operation: {sub}"
    )


# ============================================================================
# PACK MANAGEMENT STUBS
# ============================================================================

def cmd_pack(session: Session, args: List[str]) -> str:
    """Sound pack management with auto-generation.
    
    Usage:
      /pack                   -> List installed packs
      /pack info <n>       -> Show pack details
      /pack samples <n>    -> List samples in pack
      /pack load <n> <idx> -> Load sample from pack to buffer
      /pack create <n>     -> Create new empty pack
      /pack gen <n>        -> Generate pack with algorithms
      /pack gen <n> <type> -> Generate specific type
    
    Generation Types (for /pack gen):
      kicks      - Kick drum variations
      snares     - Snare variations  
      hats       - Hi-hat variations
      percs      - Percussion one-shots
      bass       - Bass synth hits
      leads      - Lead synth stabs
      pads       - Pad textures
      fx         - FX and impacts
      all        - All types (default)
    
    Packs are stored in: ~/Documents/MDMA/packs/
    
    Examples:
      /pack gen my_kit kicks     -> Generate kick variations
      /pack gen drums all        -> Generate full drum kit
      /pack gen synths bass,leads -> Generate bass and leads
    """
    from ..core.user_data import list_packs, get_pack_samples, get_packs_dir, create_pack_manifest
    
    if not args:
        packs = list_packs()
        if not packs:
            return f"No packs installed. Packs folder: {get_packs_dir()}"
        
        lines = ["=== SOUND PACKS ===", ""]
        for p in packs:
            lines.append(f"  {p['name']} ({p.get('sample_count', 0)} samples)")
            if p.get('author') and p.get('author') != 'Unknown':
                lines.append(f"    by {p['author']}")
        
        lines.append("")
        lines.append(f"Pack folder: {get_packs_dir()}")
        lines.append("")
        lines.append("Use /pack gen <name> to generate, /pack info <name> for details")
        return '\n'.join(lines)
    
    sub = args[0].lower()
    
    if sub == 'info' and len(args) > 1:
        pack_name = args[1]
        packs = [p for p in list_packs() if p['name'].lower() == pack_name.lower()]
        if not packs:
            return f"ERROR: pack '{pack_name}' not found"
        p = packs[0]
        lines = [
            f"=== PACK: {p['name']} ===",
            f"Author: {p.get('author', 'Unknown')}",
            f"Version: {p.get('version', '1.0')}",
            f"Description: {p.get('description', 'N/A')}",
            f"Samples: {p.get('sample_count', 0)}",
            f"Path: {p['path']}"
        ]
        return '\n'.join(lines)
    
    elif sub == 'samples' and len(args) > 1:
        pack_name = args[1]
        samples = get_pack_samples(pack_name)
        if not samples:
            return f"ERROR: pack '{pack_name}' not found or empty"
        
        lines = [f"=== SAMPLES IN {pack_name.upper()} ===", ""]
        for i, s in enumerate(samples[:50]):  # Limit to 50
            lines.append(f"  [{i}] {s.name}")
        
        if len(samples) > 50:
            lines.append(f"  ... and {len(samples) - 50} more")
        
        return '\n'.join(lines)
    
    elif sub == 'load' and len(args) > 2:
        pack_name = args[1]
        try:
            sample_idx = int(args[2])
        except ValueError:
            return f"ERROR: invalid sample index '{args[2]}'"
        
        samples = get_pack_samples(pack_name)
        if not samples:
            return f"ERROR: pack '{pack_name}' not found"
        if sample_idx < 0 or sample_idx >= len(samples):
            return f"ERROR: sample index {sample_idx} out of range (0-{len(samples)-1})"
        
        # Load sample to session buffer
        sample_path = samples[sample_idx]
        try:
            import soundfile as sf
            audio, sr = sf.read(str(sample_path))
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Mono
            session.last_buffer = audio.astype(np.float64)
            dur = len(audio) / sr
            return f"OK: loaded '{sample_path.name}' ({dur:.3f}s) to buffer"
        except Exception as e:
            return f"ERROR: could not load sample: {e}"
    
    elif sub == 'create' and len(args) > 1:
        pack_name = args[1]
        try:
            pack_dir = get_packs_dir() / pack_name
            pack_dir.mkdir(parents=True, exist_ok=True)
            create_pack_manifest(pack_name)
            return f"OK: created empty pack '{pack_name}' at {pack_dir}"
        except Exception as e:
            return f"ERROR: could not create pack: {e}"
    
    elif sub in ('gen', 'generate') and len(args) > 1:
        pack_name = args[1]
        gen_types = args[2].lower().split(',') if len(args) > 2 else ['all']
        return _generate_pack(session, pack_name, gen_types)
    
    return _stub_response(
        f"Pack {sub}",
        "Pack System",
        f"Pack operation: {sub}"
    )


def _generate_pack(session: Session, pack_name: str, gen_types: List[str]) -> str:
    """Generate a sample pack using synthesis algorithms.
    
    Parameters
    ----------
    session : Session
        Current session
    pack_name : str
        Name for the new pack
    gen_types : list
        Types to generate: kicks, snares, hats, percs, bass, leads, pads, fx, all
    
    Returns
    -------
    str
        Status message
    """
    import numpy as np
    from ..core.user_data import get_packs_dir, create_pack_manifest
    
    # Create pack directory
    try:
        pack_dir = get_packs_dir() / pack_name
        pack_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return f"ERROR: could not create pack directory: {e}"
    
    sr = session.sample_rate
    generated = []
    
    # Expand 'all' type
    if 'all' in gen_types:
        gen_types = ['kicks', 'snares', 'hats', 'percs', 'bass']
    
    # Generation algorithms
    def gen_kick(variation: int) -> np.ndarray:
        """Generate kick drum variation."""
        dur = 0.3 + (variation * 0.05)
        t = np.arange(int(sr * dur)) / sr
        
        # Pitch envelope (high to low)
        start_freq = 150 + (variation * 20)
        end_freq = 40 + (variation * 5)
        freq = start_freq * np.exp(-t * (8 + variation))
        freq = np.maximum(freq, end_freq)
        
        # Oscillator
        phase = np.cumsum(2 * np.pi * freq / sr)
        osc = np.sin(phase)
        
        # Amplitude envelope
        amp = np.exp(-t * (4 + variation * 0.5))
        
        # Add click transient
        click_dur = 0.005
        click_samples = int(click_dur * sr)
        click = np.random.randn(click_samples) * np.exp(-np.linspace(0, 5, click_samples))
        
        out = osc * amp
        out[:click_samples] += click * 0.3
        
        return np.clip(out, -1, 1)
    
    def gen_snare(variation: int) -> np.ndarray:
        """Generate snare drum variation."""
        dur = 0.25 + (variation * 0.03)
        t = np.arange(int(sr * dur)) / sr
        
        # Body (pitched component)
        body_freq = 180 + (variation * 15)
        body = np.sin(2 * np.pi * body_freq * t) * np.exp(-t * 20)
        
        # Noise (snare wires)
        noise = np.random.randn(len(t))
        noise_env = np.exp(-t * (10 + variation * 2))
        noise = noise * noise_env
        
        # Highpass the noise
        from scipy import signal
        try:
            b, a = signal.butter(2, 2000 / (sr/2), 'high')
            noise = signal.lfilter(b, a, noise)
        except:
            pass
        
        # Mix
        mix = (0.4 + variation * 0.05)
        out = body * (1 - mix) + noise * mix
        
        return np.clip(out * 0.8, -1, 1)
    
    def gen_hat(variation: int) -> np.ndarray:
        """Generate hi-hat variation."""
        is_open = variation >= 3
        dur = 0.5 if is_open else 0.1 + (variation * 0.02)
        t = np.arange(int(sr * dur)) / sr
        
        # Multiple high-frequency oscillators
        freqs = [4000 + (variation * 500), 7000 + (variation * 300), 10000]
        osc = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
        
        # Noise component
        noise = np.random.randn(len(t)) * 0.5
        
        # Mix
        out = osc * 0.3 + noise * 0.7
        
        # Envelope
        decay = 5 if is_open else (30 + variation * 5)
        out = out * np.exp(-t * decay)
        
        # Highpass
        try:
            from scipy import signal
            b, a = signal.butter(2, 5000 / (sr/2), 'high')
            out = signal.lfilter(b, a, out)
        except:
            pass
        
        return np.clip(out, -1, 1)
    
    def gen_perc(variation: int) -> np.ndarray:
        """Generate percussion one-shot."""
        dur = 0.15 + (variation * 0.05)
        t = np.arange(int(sr * dur)) / sr
        
        # Pitched metallic component
        freq = 300 + (variation * 100)
        freq2 = freq * (1.4 + variation * 0.1)  # Inharmonic
        
        osc1 = np.sin(2 * np.pi * freq * t)
        osc2 = np.sin(2 * np.pi * freq2 * t) * 0.5
        
        # Envelope
        env = np.exp(-t * (12 + variation * 2))
        
        out = (osc1 + osc2) * env
        
        return np.clip(out * 0.7, -1, 1)
    
    def gen_bass(variation: int) -> np.ndarray:
        """Generate bass synth hit."""
        dur = 0.4 + (variation * 0.1)
        t = np.arange(int(sr * dur)) / sr
        
        # Base frequency
        freq = 55 * (2 ** (variation / 12))  # Different notes
        
        # FM synthesis
        mod_ratio = 1 + (variation * 0.5)
        mod_depth = 2 + (variation * 0.5)
        
        mod = np.sin(2 * np.pi * freq * mod_ratio * t) * mod_depth
        carrier = np.sin(2 * np.pi * freq * t + mod)
        
        # Add sub
        sub = np.sin(2 * np.pi * freq * 0.5 * t) * 0.5
        
        # Envelope
        env = np.exp(-t * 3) * (1 - np.exp(-t * 50))  # Attack + decay
        
        out = (carrier + sub) * env
        
        return np.clip(out, -1, 1)
    
    # Generate samples
    generators = {
        'kicks': (gen_kick, 5, 'kick'),
        'snares': (gen_snare, 5, 'snare'),
        'hats': (gen_hat, 5, 'hat'),
        'percs': (gen_perc, 5, 'perc'),
        'bass': (gen_bass, 5, 'bass'),
    }
    
    for gen_type in gen_types:
        if gen_type not in generators:
            continue
        
        gen_func, count, prefix = generators[gen_type]
        
        for i in range(count):
            try:
                audio = gen_func(i)
                filename = f"{prefix}_{i+1:02d}.wav"
                out_path = pack_dir / filename
                
                try:
                    import soundfile as sf
                    sf.write(str(out_path), audio, sr)
                    generated.append(filename)
                except ImportError:
                    from scipy.io import wavfile
                    wavfile.write(str(out_path), sr, (audio * 32767).astype(np.int16))
                    generated.append(filename)
            except Exception as e:
                generated.append(f"{prefix}_{i+1:02d}.wav (ERROR: {e})")
    
    # Create manifest
    try:
        create_pack_manifest(pack_name, author='MDMA Generator',
                           description=f'Auto-generated pack with {len(generated)} samples')
    except:
        pass
    
    if not generated:
        return f"ERROR: no samples generated. Valid types: kicks, snares, hats, percs, bass, all"
    
    return f"OK: generated pack '{pack_name}' with {len(generated)} samples:\n  " + '\n  '.join(generated[:20]) + (f"\n  ... and {len(generated)-20} more" if len(generated) > 20 else "")


# ============================================================================
# STUB COMMAND REGISTRY
# ============================================================================

# NOTE: Most "stubs" are now functional shortcuts that delegate to the fx system
# or to full implementations. They're kept here for organization and documentation.

# Commands that delegate to real implementations:
# - /audiorate -> audiorate_cmds.py
# - /ump -> audiorate_cmds.py  
# - /bbe -> uses umpulse for convolution

# Commands that are shortcuts to fx system:
# - /vamp, /fc, /gg, /voc, /spc, /lfo -> apply effects directly

# DJ commands are fully implemented in dj_cmds.py:
# - /cue, /sync, /xfade, /library, /playlist

STUB_COMMANDS = {
    # Effect shortcuts (functional - delegate to fx system)
    'vamp': cmd_vamp,
    'fc': cmd_fc,
    'gg': cmd_gg,
    'voc': cmd_voc,
    'spc': cmd_spc,
    'lfo': cmd_lfo,
    
    # Advanced features (functional - delegate to implementations)
    'ump': cmd_ump,
    'audiorate': cmd_audiorate,
    'bbe': cmd_bbe,
    
    # Pack management
    'pack': cmd_pack,
}
