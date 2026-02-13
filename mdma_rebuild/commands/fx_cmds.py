"""FX and filter commands for the MDMA rebuild.

These commands manage the filter bank settings exposed on the Session.
The actual filter processing is not implemented in this simplified
version; only the counts and parameters are stored.
"""

from __future__ import annotations

from typing import List

# ---------------------------------------------------------------------------
# FX alias mapping
#
# The canonical alias map is ``_effect_aliases`` defined below.  The old
# ``FX_ALIAS_MAP`` has been removed to eliminate conflicting aliases.
# ---------------------------------------------------------------------------
import numpy as np

from ..core.session import Session


def calculate_deviation(original: np.ndarray, processed: np.ndarray) -> dict:
    """Calculate deviation metric showing how much audio changed.
    
    Returns dict with:
    - pct: percentage change (0-100+)
    - rms_delta: RMS level change in dB
    - peak_delta: peak level change in dB
    - correlation: similarity (1.0 = identical, 0.0 = uncorrelated)
    """
    if original is None or processed is None:
        return {'pct': 0, 'rms_delta': 0, 'peak_delta': 0, 'correlation': 1.0}
    
    # Ensure same length
    min_len = min(len(original), len(processed))
    orig = original[:min_len].astype(np.float64)
    proc = processed[:min_len].astype(np.float64)
    
    # RMS levels
    orig_rms = np.sqrt(np.mean(orig ** 2)) + 1e-10
    proc_rms = np.sqrt(np.mean(proc ** 2)) + 1e-10
    rms_delta = 20 * np.log10(proc_rms / orig_rms)
    
    # Peak levels
    orig_peak = np.max(np.abs(orig)) + 1e-10
    proc_peak = np.max(np.abs(proc)) + 1e-10
    peak_delta = 20 * np.log10(proc_peak / orig_peak)
    
    # Correlation (normalized cross-correlation at zero lag)
    orig_norm = orig - np.mean(orig)
    proc_norm = proc - np.mean(proc)
    denom = np.sqrt(np.sum(orig_norm**2) * np.sum(proc_norm**2))
    if denom > 1e-10:
        correlation = np.sum(orig_norm * proc_norm) / denom
    else:
        correlation = 1.0
    
    # Percentage change (based on mean absolute difference)
    diff = np.abs(proc - orig)
    mean_diff = np.mean(diff)
    orig_range = np.max(np.abs(orig)) + 1e-10
    pct = (mean_diff / orig_range) * 100
    
    return {
        'pct': min(999, pct),  # Cap at 999%
        'rms_delta': rms_delta,
        'peak_delta': peak_delta,
        'correlation': correlation,
    }


def format_deviation(dev: dict, compact: bool = True) -> str:
    """Format deviation metric for display."""
    if compact:
        pct = dev.get('pct', 0)
        rms = dev.get('rms_delta', 0)
        if pct < 1:
            return "Δ: ~0%"
        elif pct < 10:
            return f"Δ: {pct:.1f}%"
        else:
            sign = "+" if rms > 0 else ""
            return f"Δ: {pct:.0f}% ({sign}{rms:.1f}dB)"
    else:
        return (f"Change: {dev['pct']:.1f}%, "
                f"RMS: {dev['rms_delta']:+.1f}dB, "
                f"Peak: {dev['peak_delta']:+.1f}dB, "
                f"Similarity: {dev['correlation']:.2f}")


def format_fx_result(effect_name: str, deviation: dict = None, extra: str = None) -> str:
    """Format clean FX result output."""
    parts = [f"✓ {effect_name}"]
    if deviation:
        parts.append(format_deviation(deviation))
    if extra:
        parts.append(extra)
    return " | ".join(parts)

# Mapping of short effect aliases to full effect names.  These allow
# quick selection via codes like r1 (reverb_small) or d3 (delay_multitap).
# Effect parameter documentation for /hfx
# All parameters use unified 0-100 scale where 100 = max output
EFFECT_PARAMS = {
    # Reverbs - amount controls wet/dry mix
    'reverb_small': {'desc': 'Small room reverb', 'params': {'amount': '0-100 (wet mix)'}},
    'reverb_large': {'desc': 'Large hall reverb', 'params': {'amount': '0-100 (wet mix)'}},
    'reverb_plate': {'desc': 'Plate reverb', 'params': {'amount': '0-100 (wet mix)'}},
    'reverb_spring': {'desc': 'Spring reverb', 'params': {'amount': '0-100 (wet mix)'}},
    'reverb_cathedral': {'desc': 'Cathedral reverb', 'params': {'amount': '0-100 (wet mix)'}},
    # Convolution reverbs
    'conv_hall': {'desc': 'Convolution hall reverb', 'params': {'amount': '0-100'}},
    'conv_hall_long': {'desc': 'Long convolution hall', 'params': {'amount': '0-100'}},
    'conv_room': {'desc': 'Convolution room reverb', 'params': {'amount': '0-100'}},
    'conv_plate': {'desc': 'Convolution plate reverb', 'params': {'amount': '0-100'}},
    'conv_spring': {'desc': 'Convolution spring reverb', 'params': {'amount': '0-100'}},
    'conv_shimmer': {'desc': 'Shimmer reverb with pitch shift', 'params': {'amount': '0-100'}},
    'conv_reverse': {'desc': 'Reverse reverb', 'params': {'amount': '0-100'}},
    # Delays - amount controls wet/feedback
    'delay_simple': {'desc': 'Basic delay', 'params': {'amount': '0-100 (wet+feedback)'}},
    'delay_pingpong': {'desc': 'Ping-pong stereo delay', 'params': {'amount': '0-100'}},
    'delay_multitap': {'desc': 'Multi-tap delay', 'params': {'amount': '0-100'}},
    'delay_slapback': {'desc': 'Short slapback delay', 'params': {'amount': '0-100'}},
    'delay_tape': {'desc': 'Tape echo with wobble', 'params': {'amount': '0-100'}},
    # Saturation - amount controls drive
    'saturate_soft': {'desc': 'Soft saturation', 'params': {'amount': '0-100 (drive)'}},
    'saturate_hard': {'desc': 'Hard clipping', 'params': {'amount': '0-100 (drive)'}},
    'saturate_overdrive': {'desc': 'Overdrive', 'params': {'amount': '0-100 (drive)'}},
    'saturate_fuzz': {'desc': 'Fuzz distortion', 'params': {'amount': '0-100 (drive)'}},
    'saturate_tube': {'desc': 'Tube saturation', 'params': {'amount': '0-100 (drive)'}},
    # Vamp/Overdrive - amount controls drive intensity
    'vamp_light': {'desc': 'Light amp warmth', 'params': {'amount': '0-100'}},
    'vamp_medium': {'desc': 'Medium amp overdrive', 'params': {'amount': '0-100'}},
    'vamp_heavy': {'desc': 'Heavy amp distortion', 'params': {'amount': '0-100'}},
    'vamp_fuzz': {'desc': 'Fuzz pedal distortion', 'params': {'amount': '0-100'}},
    'overdrive_soft': {'desc': 'Soft overdrive', 'params': {'amount': '0-100'}},
    'overdrive_classic': {'desc': 'Classic overdrive tone', 'params': {'amount': '0-100'}},
    'overdrive_crunch': {'desc': 'Crunchy overdrive', 'params': {'amount': '0-100'}},
    'dual_od_warm': {'desc': 'Warm dual-stage OD', 'params': {'amount': '0-100'}},
    'dual_od_bright': {'desc': 'Bright dual-stage OD', 'params': {'amount': '0-100'}},
    'dual_od_heavy': {'desc': 'Heavy dual-stage OD', 'params': {'amount': '0-100'}},
    'waveshape_fold': {'desc': 'Wave folding distortion', 'params': {'amount': '0-100'}},
    'waveshape_rectify': {'desc': 'Rectifier distortion', 'params': {'amount': '0-100'}},
    'waveshape_sine': {'desc': 'Sine waveshaping harmonics', 'params': {'amount': '0-100'}},
    # Dynamics - amount controls intensity
    'compress_mild': {'desc': 'Mild compressor (4:1)', 'params': {'amount': '0-100'}},
    'compress_hard': {'desc': 'Hard compressor (8:1)', 'params': {'amount': '0-100'}},
    'compress_limiter': {'desc': 'Brick wall limiter', 'params': {'amount': '0-100'}},
    'compress_expander': {'desc': 'Expander/gate', 'params': {'amount': '0-100'}},
    'compress_softclipper': {'desc': 'Soft clipper', 'params': {'amount': '0-100'}},
    # Gates - amount controls threshold
    'gate1': {'desc': 'Gate light', 'params': {'amount': '0-100 (threshold)'}},
    'gate2': {'desc': 'Gate medium-light', 'params': {'amount': '0-100'}},
    'gate3': {'desc': 'Gate medium', 'params': {'amount': '0-100'}},
    'gate4': {'desc': 'Gate medium-heavy', 'params': {'amount': '0-100'}},
    'gate5': {'desc': 'Gate heavy', 'params': {'amount': '0-100'}},
    # Lo-fi - amount controls intensity
    'lofi_bitcrush': {'desc': 'Bit crusher', 'params': {'amount': '0-100 (crush)'}},
    'lofi_chorus': {'desc': 'Chorus effect', 'params': {'amount': '0-100 (depth)'}},
    'lofi_flanger': {'desc': 'Flanger effect', 'params': {'amount': '0-100 (depth)'}},
    'lofi_phaser': {'desc': 'Phaser effect', 'params': {'amount': '0-100 (depth)'}},
    'lofi_filter': {'desc': 'Lo-fi filter', 'params': {'amount': '0-100 (cutoff)'}},
    'lofi_halftime': {'desc': 'Half-speed effect', 'params': {'amount': '0-100 (mix)'}},
}

# Parameter scale info for help
PARAM_SCALE_INFO = """
=== UNIFIED PARAMETER SCALE (0-100) ===
All effect 'amount' parameters use 0-100 scale:
  0   = minimum/off
  25  = subtle/light  
  50  = moderate/default
  75  = strong
  100 = maximum

Presets: subtle, light, moderate, strong, heavy, extreme, off, full, half
"""

_effect_aliases = {
    # Common single-word shortcuts (user-friendly defaults)
    'reverb': 'reverb_small',
    'delay': 'delay_simple',
    'compress': 'compress_mild',
    'compressor': 'compress_mild',
    'distort': 'saturate_hard',
    'distortion': 'saturate_hard',
    'saturate': 'saturate_soft',
    'saturation': 'saturate_soft',
    'fuzz': 'saturate_fuzz',
    'gate': 'gate2',
    'lofi': 'lofi_bitcrush',
    'bitcrush': 'lofi_bitcrush',
    'chorus': 'lofi_chorus',
    'flanger': 'lofi_flanger',
    'phaser': 'lofi_phaser',
    'halftime': 'lofi_halftime',
    'limit': 'compress_limiter',
    'limiter': 'compress_limiter',
    'expand': 'compress_expander',
    'expander': 'compress_expander',
    'clip': 'compress_softclipper',
    'softclip': 'compress_softclipper',
    'hall': 'conv_hall',
    'room': 'conv_room',
    'plate': 'conv_plate',
    'spring': 'conv_spring',
    'tape': 'delay_tape',
    'echo': 'delay_simple',
    'pingpong': 'delay_pingpong',
    'slap': 'delay_slapback',
    'slapback': 'delay_slapback',
    # Basic reverbs
    'r1': 'reverb_small',
    'r2': 'reverb_large',
    'r3': 'reverb_plate',
    'r4': 'reverb_spring',
    'r5': 'reverb_cathedral',
    # Convolution reverbs
    'cr1': 'conv_hall',
    'cr2': 'conv_hall_long',
    'cr3': 'conv_room',
    'cr4': 'conv_plate',
    'cr5': 'conv_spring',
    'cr6': 'conv_shimmer',
    'cr7': 'conv_reverse',
    # Alternative conv aliases
    'cvhall': 'conv_hall',
    'cvroom': 'conv_room',
    'cvplate': 'conv_plate',
    'cvspring': 'conv_spring',
    'shimmer': 'conv_shimmer',
    'reverse': 'conv_reverse',
    'd1': 'delay_simple',
    'd2': 'delay_pingpong',
    'd3': 'delay_multitap',
    'd4': 'delay_slapback',
    'd5': 'delay_tape',
    's1': 'saturate_soft',
    's2': 'saturate_hard',
    's3': 'saturate_overdrive',
    's4': 'saturate_fuzz',
    's5': 'saturate_tube',
    # Vamp/Overdrive aliases
    'v1': 'vamp_light',
    'v2': 'vamp_medium',
    'v3': 'vamp_heavy',
    'v4': 'vamp_fuzz',
    'vamp': 'vamp_medium',
    'amp': 'vamp_medium',
    'o1': 'overdrive_soft',
    'o2': 'overdrive_classic',
    'o3': 'overdrive_crunch',
    'od': 'overdrive_classic',
    'overdrive': 'overdrive_classic',
    'crunch': 'overdrive_crunch',
    'do1': 'dual_od_warm',
    'do2': 'dual_od_bright',
    'do3': 'dual_od_heavy',
    'dual': 'dual_od_warm',
    'ws1': 'waveshape_fold',
    'ws2': 'waveshape_rectify',
    'ws3': 'waveshape_sine',
    'fold': 'waveshape_fold',
    'rect': 'waveshape_rectify',
    # Dynamics
    'c1': 'compress_mild',
    'c2': 'compress_hard',
    'c3': 'compress_limiter',
    'c4': 'compress_expander',
    'c5': 'compress_softclipper',
    # Dynamics aliases (di prefix)
    'di1': 'compress_mild',
    'di2': 'compress_hard',
    'di3': 'compress_limiter',
    'di4': 'compress_expander',
    'di5': 'compress_softclipper',
    'g1': 'gate1',
    'g2': 'gate2',
    'g3': 'gate3',
    'g4': 'gate4',
    'g5': 'gate5',
    'l1': 'lofi_bitcrush',
    'l2': 'lofi_chorus',
    'l3': 'lofi_flanger',
    'l4': 'lofi_phaser',
    'l5': 'lofi_filter',
    'l6': 'lofi_halftime',
    # Filter aliases
    'lowpass': 'filter_lowpass',
    'lp': 'filter_lowpass',
    'lp1': 'filter_lowpass_soft',
    'lp2': 'filter_lowpass',
    'lp3': 'filter_lowpass_hard',
    'highpass': 'filter_highpass',
    'hp': 'filter_highpass',
    'hp1': 'filter_highpass_soft',
    'hp2': 'filter_highpass',
    'hp3': 'filter_highpass_hard',
    'bandpass': 'filter_bandpass',
    'bp': 'filter_bandpass',
    'bp1': 'filter_bandpass',
    'bp2': 'filter_bandpass_narrow',
    'telephone': 'filter_bandpass_narrow',
    # Pitch shift aliases
    'pitchup': 'pitch_up_5',
    'pitchdown': 'pitch_down_5',
    'pitchshift': 'pitch_up_5',
    'pu2': 'pitch_up_2',
    'pu5': 'pitch_up_5',
    'pu7': 'pitch_up_7',
    'pu12': 'pitch_up_12',
    'pd2': 'pitch_down_2',
    'pd5': 'pitch_down_5',
    'pd7': 'pitch_down_7',
    'pd12': 'pitch_down_12',
    'octave_up': 'pitch_up_12',
    'octave_down': 'pitch_down_12',
    # Harmonizer aliases
    'harmonize': 'harmonizer_5th',
    'harmonizer': 'harmonizer_5th',
    'h3': 'harmonizer_3rd',
    'h5': 'harmonizer_5th',
    'h8': 'harmonizer_octave',
    'hchord': 'harmonizer_chord',
    # Granular preset aliases
    'granular': 'granular_cloud',
    'grain': 'granular_cloud',
    'cloud': 'granular_cloud',
    'scatter': 'granular_scatter',
    'grstretch': 'granular_stretch',
    'grfreeze': 'granular_freeze',
    'grshimmer': 'granular_shimmer',
    'grrev': 'granular_reverse',
    'grstutter': 'granular_stutter',
    'gr1': 'granular_cloud',
    'gr2': 'granular_scatter',
    'gr3': 'granular_stretch',
    'gr4': 'granular_freeze',
    'gr5': 'granular_shimmer',
    'gr6': 'granular_reverse',
    'gr7': 'granular_stutter',
    # Utility aliases
    'normalize': 'util_normalize',
    'norm': 'util_normalize',
    'normrms': 'util_normalize_rms',
    'lufs': 'util_normalize_rms',
    'declip': 'util_declip',
    'declick': 'util_declick',
    'smooth': 'util_smooth',
    'warmth': 'util_smooth',
    'smoothheavy': 'util_smooth_heavy',
    'muffle': 'util_smooth_heavy',
    'dcremove': 'util_dc_remove',
    'dc': 'util_dc_remove',
    'fadein': 'util_fade_in',
    'fadeout': 'util_fade_out',
    'fade': 'util_fade_both',
    'fades': 'util_fade_both',
}


def resolve_effect_name(name: str) -> tuple[str | None, str | None]:
    """Resolve an effect name or alias to a valid DSP effect.

    Returns (resolved_name, error_msg).
    On success: (resolved_name, None).
    On failure: (None, human-readable error string).
    """
    from ..dsp.effects import _effect_funcs

    low = name.lower()
    # Try alias first
    resolved = _effect_aliases.get(low, low)
    if resolved in _effect_funcs:
        return resolved, None
    # Case-insensitive fallback
    for key in _effect_funcs:
        if key.lower() == resolved.lower():
            return key, None
    # Build suggestion
    import difflib
    close = difflib.get_close_matches(low, list(_effect_funcs.keys()) + list(_effect_aliases.keys()), n=3, cutoff=0.5)
    suggestion = f"  Did you mean: {', '.join(close)}" if close else "  Use /hfx to see all effects."
    return None, f"ERROR: Unknown effect '{name}'.\n{suggestion}"


def cmd_sfc(session: Session, args: List[str]) -> str:
    """Get or set the number of available filter slots.

    Usage:
      /sfc          -> show current number of filter slots
      /sfc <int>    -> set number of filter slots
    """
    if not args:
        return f"FILTER SLOTS: {session.filter_count}"
    try:
        n = int(float(args[0]))
        if n <= 0:
            return "ERROR: filter slot count must be positive"
        session.set_filter_count(n)
        return f"OK: filter slots set to {session.filter_count}"
    except Exception:
        return "ERROR: invalid filter slot count"


def cmd_sf(session: Session, args: List[str]) -> str:
    """Get or set the selected filter index.

    Usage:
      /sf        -> show selected filter index
      /sf <int>  -> select filter slot by index
    """
    if not args:
        return f"FILTER INDEX: {session.selected_filter}"
    try:
        idx = int(float(args[0]))
        session.select_filter(idx)
        return f"OK: selected filter set to {session.selected_filter}"
    except Exception as exc:
        return f"ERROR: {exc}"


def cmd_sfr(session: Session, args: List[str]) -> str:
    """Get or set filter resonance (0-100 scale).

    Usage:
      /sfr        -> show resonance
      /sfr <val>  -> set resonance (0-100 or preset name)
    
    Values:
      0 = no resonance
      50 = moderate resonance (default)
      100 = high resonance (near self-oscillation)
      >100 = wacky territory
    
    Preset names: subtle, light, moderate, strong, heavy, extreme, wacky
    """
    if not args:
        return f"RESONANCE: {session.resonance:.1f}"
    try:
        val = float(args[0])
        session.set_resonance(val)
        return f"OK: resonance set to {session.resonance:.1f}"
    except Exception:
        return "ERROR: invalid resonance value"


def cmd_sff(session: Session, args: List[str]) -> str:
    """Get or set filter cutoff frequency (Hz).

    Usage:
      /sff        -> show cutoff frequency
      /sff <Hz>   -> set cutoff frequency
    """
    if not args:
        return f"CUTOFF: {session.cutoff:.1f} Hz"
    try:
        val = float(args[0])
        if val <= 0:
            return "ERROR: cutoff frequency must be positive"
        session.set_cutoff(val)
        return f"OK: cutoff set to {session.cutoff:.1f} Hz"
    except Exception:
        return "ERROR: invalid cutoff value"


# Effects commands - Simplified v34
def cmd_fx(session: Session, args: List[str]) -> str:
    """Apply an effect to the current audio buffer.

    Usage:
      /fx <effect>         Apply effect immediately with deviation metrics
      /fx list             Show available effects
      /fx chain            Show current effect chain
      /fx clear            Clear effect chain
    
    Common effects (use short names):
      reverb, delay, compress, distort, chorus, flanger, phaser
      
    Full effect names:
      reverb_small, reverb_large, reverb_plate, delay_simple, delay_pingpong
      saturate_soft, saturate_hard, compress_mild, compress_hard
      lofi_bitcrush, lofi_chorus, gate1, gate2, gate3

    Examples:
      /fx reverb           Apply reverb to buffer
      /fx delay            Apply delay to buffer
      /fx compress         Apply compression to buffer
    """
    import numpy as np
    
    if not args:
        return ("Usage: /fx <effect>\n"
                "Common: reverb, delay, compress, distort, chorus, flanger, phaser\n"
                "Use /fx list for full list, /fx chain to see current chain")
    
    sub = args[0].lower()
    
    if sub == 'list':
        from ..dsp.effects import _effect_funcs
        lines = ["=== AVAILABLE EFFECTS ==="]
        
        categories = {
            'REVERB': [e for e in _effect_funcs if 'reverb' in e or e.startswith('conv_')],
            'DELAY': [e for e in _effect_funcs if 'delay' in e],
            'SATURATION': [e for e in _effect_funcs if 'saturate' in e or 'vamp' in e or 'overdrive' in e],
            'DYNAMICS': [e for e in _effect_funcs if 'compress' in e or 'gate' in e],
            'LOFI': [e for e in _effect_funcs if 'lofi' in e],
        }
        
        for cat, effects in categories.items():
            if effects:
                lines.append(f"  {cat}: {', '.join(sorted(effects)[:5])}...")
        
        lines.append("\nAliases: reverb, delay, compress, distort, chorus, flanger, phaser")
        return '\n'.join(lines)
    
    if sub == 'chain':
        effects = session.list_effects()
        if not effects:
            return "CHAIN: (empty)"
        return "CHAIN: " + " -> ".join(effects)
    
    if sub == 'clear':
        session.clear_effects()
        return "OK: effect chain cleared"
    
    # Legacy: /fx add <name> still works
    if sub == 'add' and len(args) >= 2:
        sub = args[1].lower()
    
    # Resolve alias and validate
    try:
        from ..dsp.effects import apply_effects_with_params
        effect_name, err = resolve_effect_name(sub)
        if err:
            return err
    except ImportError as e:
        return f"ERROR: effects module not available: {e}"
    
    # Check if we have audio to process
    if session.last_buffer is None or len(session.last_buffer) == 0:
        session.add_effect(effect_name)
        return f"OK: '{effect_name}' added to chain (no audio to apply yet)"
    
    # Get before metrics
    before_peak = np.max(np.abs(session.last_buffer))
    before_rms = np.sqrt(np.mean(session.last_buffer ** 2))
    before_samples = len(session.last_buffer)
    
    # Apply effect immediately AND track in chain
    try:
        session.last_buffer = apply_effects_with_params(session.last_buffer, [effect_name], [{}])

        # Track in effects chain so tree/inspector reflect the change
        session.add_effect(effect_name)

        # Update working buffer
        try:
            from .working_cmds import get_working_buffer
            wb = get_working_buffer()
            wb.set_pending(session.last_buffer, f"fx:{effect_name}", session)
        except:
            pass
        
        # Get after metrics
        after_peak = np.max(np.abs(session.last_buffer))
        after_rms = np.sqrt(np.mean(session.last_buffer ** 2))
        after_samples = len(session.last_buffer)
        
        # Calculate changes
        peak_db = 20 * np.log10(after_peak / before_peak) if before_peak > 0 else 0
        rms_db = 20 * np.log10(after_rms / before_rms) if before_rms > 0 else 0
        
        # Format output with deviation metrics
        lines = [f"OK: Applied '{effect_name}'"]
        lines.append(f"  Peak: {before_peak:.3f} -> {after_peak:.3f} ({peak_db:+.1f} dB)")
        lines.append(f"  RMS:  {before_rms:.3f} -> {after_rms:.3f} ({rms_db:+.1f} dB)")
        if after_samples != before_samples:
            lines.append(f"  Length: {before_samples} -> {after_samples} samples")
        
        return '\n'.join(lines)
        
    except Exception as e:
        return f"ERROR applying '{effect_name}': {e}"

def cmd_hfx(session: Session, args: List[str]) -> str:
    """List all effects with aliases and parameters.
    
    Usage:
      /hfx              -> List all effects grouped by category
      /hfx <n>       -> Show details for specific effect
      /hfx aliases      -> List all aliases
      /hfx reverb       -> List reverb effects
      /hfx delay        -> List delay effects
      /hfx sat          -> List saturation effects
      /hfx vamp         -> List vamp/overdrive effects
      /hfx comp         -> List dynamics/compressor effects
      /hfx gate         -> List gate effects
      /hfx lofi         -> List lo-fi effects
      /hfx conv         -> List convolution effects
    """
    from ..dsp.effects import _effect_funcs
    
    # Build reverse alias map
    alias_by_effect = {}
    for alias, effect in _effect_aliases.items():
        if effect not in alias_by_effect:
            alias_by_effect[effect] = []
        alias_by_effect[effect].append(alias)
    
    if not args:
        # List all effects by category
        lines = ["=== EFFECTS LIST ===", ""]
        
        categories = {
            'REVERB (basic)': ['reverb_small', 'reverb_large', 'reverb_plate', 'reverb_spring', 'reverb_cathedral'],
            'REVERB (convolution)': ['conv_hall', 'conv_hall_long', 'conv_room', 'conv_plate', 'conv_spring', 'conv_shimmer', 'conv_reverse'],
            'DELAY': ['delay_simple', 'delay_pingpong', 'delay_multitap', 'delay_slapback', 'delay_tape'],
            'SATURATION': ['saturate_soft', 'saturate_hard', 'saturate_overdrive', 'saturate_fuzz', 'saturate_tube'],
            'VAMP/OVERDRIVE': ['vamp_light', 'vamp_medium', 'vamp_heavy', 'vamp_fuzz', 
                              'overdrive_soft', 'overdrive_classic', 'overdrive_crunch',
                              'dual_od_warm', 'dual_od_bright', 'dual_od_heavy',
                              'waveshape_fold', 'waveshape_rectify', 'waveshape_sine'],
            'DYNAMICS': ['compress_mild', 'compress_hard', 'compress_limiter', 'compress_expander', 'compress_softclipper'],
            'GATE': ['gate1', 'gate2', 'gate3', 'gate4', 'gate5'],
            'LO-FI': ['lofi_bitcrush', 'lofi_chorus', 'lofi_flanger', 'lofi_phaser', 'lofi_filter', 'lofi_halftime'],
            'GRANULAR': ['granular_cloud', 'granular_scatter', 'granular_stretch', 'granular_freeze',
                        'granular_shimmer', 'granular_reverse', 'granular_stutter'],
            'UTILITY': ['util_normalize', 'util_normalize_rms', 'util_declip', 'util_declick',
                       'util_smooth', 'util_smooth_heavy', 'util_dc_remove',
                       'util_fade_in', 'util_fade_out', 'util_fade_both'],
        }
        
        for cat_name, effects in categories.items():
            lines.append(f"--- {cat_name} ---")
            for effect in effects:
                aliases = alias_by_effect.get(effect, [])
                alias_str = ', '.join(aliases[:3]) if aliases else ''
                info = EFFECT_PARAMS.get(effect, {})
                desc = info.get('desc', '')
                if alias_str:
                    lines.append(f"  {effect} ({alias_str})")
                else:
                    lines.append(f"  {effect}")
                if desc:
                    lines.append(f"    {desc}")
            lines.append("")
        
        lines.append("Use /hfx <n> for parameters")
        return '\n'.join(lines)
    
    query = args[0].lower()
    
    # Show aliases
    if query == 'aliases':
        lines = ["=== EFFECT ALIASES ==="]
        sorted_aliases = sorted(_effect_aliases.items())
        for alias, effect in sorted_aliases:
            lines.append(f"  {alias} -> {effect}")
        return '\n'.join(lines)
    
    # Filter by category
    category_filters = {
        'reverb': lambda e: 'reverb' in e or e.startswith('conv_'),
        'conv': lambda e: e.startswith('conv_'),
        'delay': lambda e: 'delay' in e,
        'sat': lambda e: 'saturate' in e,
        'saturation': lambda e: 'saturate' in e,
        'vamp': lambda e: e.startswith('vamp_') or e.startswith('overdrive_') or e.startswith('dual_od') or e.startswith('waveshape_'),
        'overdrive': lambda e: e.startswith('overdrive_') or e.startswith('dual_od'),
        'od': lambda e: e.startswith('overdrive_') or e.startswith('dual_od'),
        'waveshape': lambda e: e.startswith('waveshape_'),
        'ws': lambda e: e.startswith('waveshape_'),
        'comp': lambda e: 'compress' in e,
        'dynamics': lambda e: 'compress' in e,
        'gate': lambda e: e.startswith('gate'),
        'lofi': lambda e: 'lofi' in e,
        'granular': lambda e: e.startswith('granular_'),
        'grain': lambda e: e.startswith('granular_'),
        'gr': lambda e: e.startswith('granular_'),
        'util': lambda e: e.startswith('util_'),
        'utility': lambda e: e.startswith('util_'),
        'repair': lambda e: e in ('util_declip', 'util_declick', 'util_dc_remove'),
        'smooth': lambda e: e in ('util_smooth', 'util_smooth_heavy'),
        'normalize': lambda e: e in ('util_normalize', 'util_normalize_rms'),
    }
    
    if query in category_filters:
        filter_fn = category_filters[query]
        lines = [f"=== {query.upper()} EFFECTS ==="]
        for effect in _effect_funcs.keys():
            if filter_fn(effect):
                aliases = alias_by_effect.get(effect, [])
                alias_str = ', '.join(aliases[:3]) if aliases else ''
                info = EFFECT_PARAMS.get(effect, {})
                desc = info.get('desc', '')
                params = info.get('params', {})
                
                if alias_str:
                    lines.append(f"\n{effect} ({alias_str})")
                else:
                    lines.append(f"\n{effect}")
                if desc:
                    lines.append(f"  {desc}")
                if params:
                    param_strs = [f"{k}: {v}" for k, v in params.items()]
                    lines.append(f"  Params: {', '.join(param_strs)}")
        return '\n'.join(lines)
    
    # Try to resolve as effect name or alias
    effect_name = _effect_aliases.get(query) or query
    
    if effect_name in _effect_funcs or effect_name in EFFECT_PARAMS:
        info = EFFECT_PARAMS.get(effect_name, {})
        aliases = alias_by_effect.get(effect_name, [])
        
        lines = [f"=== {effect_name.upper()} ==="]
        if aliases:
            lines.append(f"Aliases: {', '.join(aliases)}")
        
        desc = info.get('desc', 'No description')
        lines.append(f"Description: {desc}")
        
        params = info.get('params', {})
        if params:
            lines.append("\nParameters:")
            for param, range_str in params.items():
                lines.append(f"  {param}: {range_str}")
            lines.append("\nSet with: /fx set <param> <value>")
            lines.append("Or: /fx <param> <value>")
        else:
            lines.append("\nNo adjustable parameters")
        
        return '\n'.join(lines)
    
    return f"ERROR: unknown effect or category '{query}'\nUse /hfx to list all effects"


def cmd_conv(session: Session, args: List[str]) -> str:
    """Apply convolution reverb with advanced parameters.
    
    Usage:
      /conv                     -> Apply default hall convolution
      /conv <preset>            -> Apply preset (hall, room, plate, spring, shimmer, reverse)
      /conv file <path>         -> Use WAV file as impulse response
      /conv <preset> wet <val>  -> Set wet level (0-1)
      /conv <preset> dry <val>  -> Set dry level (0-1)
      /conv <preset> stretch <val> -> Time-stretch IR (0.5-2.0)
      /conv <preset> predelay <ms> -> Set pre-delay in ms
      /conv <preset> highcut <Hz>  -> High frequency cutoff
      /conv <preset> lowcut <Hz>   -> Low frequency cutoff
    
    Presets: hall, hall_long, hall_bright, hall_dark, room, room_small, room_large,
             plate, plate_bright, plate_dark, spring, spring_tight, spring_loose,
             shimmer, shimmer_fifth, reverse, reverse_long
    """
    if session.last_buffer is None:
        return "ERROR: no buffer to process. Generate audio first."
    
    from ..dsp.effects import convolve_reverb, IR_PRESETS
    
    # Parse arguments
    preset = 'hall'
    ir_file = None
    wet = 0.5
    dry = 0.6
    stretch = 1.0
    pre_delay = 0.0
    high_cut = 20000.0
    low_cut = 20.0
    
    i = 0
    while i < len(args):
        arg = args[i].lower()
        
        if arg == 'file' and i + 1 < len(args):
            ir_file = args[i + 1]
            i += 2
            continue
        elif arg in IR_PRESETS:
            preset = arg
            i += 1
            continue
        elif arg == 'wet' and i + 1 < len(args):
            try:
                wet = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        elif arg == 'dry' and i + 1 < len(args):
            try:
                dry = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        elif arg == 'stretch' and i + 1 < len(args):
            try:
                stretch = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        elif arg in ('predelay', 'pd') and i + 1 < len(args):
            try:
                pre_delay = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        elif arg in ('highcut', 'hc') and i + 1 < len(args):
            try:
                high_cut = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        elif arg in ('lowcut', 'lc') and i + 1 < len(args):
            try:
                low_cut = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        else:
            # Try as preset name
            if arg in IR_PRESETS:
                preset = arg
            i += 1
    
    try:
        session.last_buffer = convolve_reverb(
            session.last_buffer,
            preset=preset,
            ir_file=ir_file,
            wet=wet,
            dry=dry,
            stretch=stretch,
            pre_delay=pre_delay,
            high_cut=high_cut,
            low_cut=low_cut,
        )
        
        src = f"file:{ir_file}" if ir_file else f"preset:{preset}"
        return f"OK: convolution reverb applied ({src}, wet={wet:.2f}, dry={dry:.2f}, stretch={stretch:.2f}x)"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_vamp(session: Session, args: List[str]) -> str:
    """Apply vamp/overdrive/waveshaping with advanced parameters.
    
    Usage:
      /vamp                           -> Apply default medium vamp
      /vamp <preset>                  -> Apply preset (light, medium, heavy, fuzz)
      /vamp drive <val>               -> Set drive (1-20)
      /vamp shape <name>              -> Set waveshape (tanh, hard, tube, fuzz, fold, rectify, cubic, sine)
      /vamp pre <Hz>                  -> Set pre-filter cutoff (hp by default)
      /vamp pre <Hz> lp               -> Set pre-filter as lowpass
      /vamp post <Hz>                 -> Set post-filter cutoff (lp by default)
      /vamp post <Hz> hp              -> Set post-filter as highpass
      /vamp bias <val>                -> Set DC bias (-1 to 1)
      /vamp gain <val>                -> Set gain makeup (0.5-4)
      /vamp mix <val>                 -> Set dry/wet mix (0-1)
      /vamp file <path>               -> Use WAV file as custom waveshape
    
    Presets: light, medium, heavy, fuzz
    Waveshapes: tanh, hard, tube, fuzz, fold, rectify, cubic, sine
    
    Example: /vamp heavy shape fuzz pre 100 post 4000 gain 1.2
    """
    if session.last_buffer is None:
        return "ERROR: no buffer to process. Generate audio first."
    
    from ..dsp.effects import vamp_process, WAVESHAPE_PRESETS
    import numpy as np
    
    # Defaults
    preset_settings = {
        'light': {'drive': 2.0, 'waveshape': 'tube', 'post_filter': 8000},
        'medium': {'drive': 5.0, 'waveshape': 'tube', 'post_filter': 6000},
        'heavy': {'drive': 10.0, 'waveshape': 'tube', 'pre_filter': 100, 'post_filter': 4000},
        'fuzz': {'drive': 15.0, 'waveshape': 'fuzz', 'post_filter': 3000, 'gain_makeup': 0.8},
    }
    
    drive = 5.0
    waveshape = 'tube'
    custom_curve = None
    pre_filter = None
    pre_filter_type = 'hp'
    post_filter = None
    post_filter_type = 'lp'
    gain_makeup = 1.0
    bias = 0.0
    mix = 1.0
    
    i = 0
    while i < len(args):
        arg = args[i].lower()
        
        # Check for preset
        if arg in preset_settings:
            settings = preset_settings[arg]
            drive = settings.get('drive', drive)
            waveshape = settings.get('waveshape', waveshape)
            pre_filter = settings.get('pre_filter', pre_filter)
            post_filter = settings.get('post_filter', post_filter)
            gain_makeup = settings.get('gain_makeup', gain_makeup)
            i += 1
            continue
        
        # Parse parameters
        if arg == 'drive' and i + 1 < len(args):
            try:
                drive = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        elif arg in ('shape', 'waveshape') and i + 1 < len(args):
            waveshape = args[i + 1].lower()
            i += 2
            continue
        elif arg == 'pre' and i + 1 < len(args):
            try:
                pre_filter = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            # Check for filter type
            if i < len(args) and args[i].lower() in ('lp', 'hp'):
                pre_filter_type = args[i].lower()
                i += 1
            continue
        elif arg == 'post' and i + 1 < len(args):
            try:
                post_filter = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            # Check for filter type
            if i < len(args) and args[i].lower() in ('lp', 'hp'):
                post_filter_type = args[i].lower()
                i += 1
            continue
        elif arg == 'bias' and i + 1 < len(args):
            try:
                bias = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        elif arg in ('gain', 'makeup') and i + 1 < len(args):
            try:
                gain_makeup = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        elif arg == 'mix' and i + 1 < len(args):
            try:
                mix = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        elif arg == 'file' and i + 1 < len(args):
            # Load custom waveshape from file
            try:
                import wave
                filepath = args[i + 1]
                with wave.open(filepath, 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    sampwidth = wf.getsampwidth()
                    if sampwidth == 2:
                        data = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
                    elif sampwidth == 4:
                        data = np.frombuffer(frames, dtype=np.int32).astype(np.float64) / 2147483648.0
                    else:
                        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float64) / 128.0 - 1.0
                    # Resample to 2048 points
                    x_old = np.linspace(0, 1, len(data))
                    x_new = np.linspace(0, 1, 2048)
                    custom_curve = np.interp(x_new, x_old, data)
            except Exception as e:
                return f"ERROR: failed to load waveshape file: {e}"
            i += 2
            continue
        else:
            i += 1
    
    try:
        session.last_buffer = vamp_process(
            session.last_buffer,
            drive=drive,
            waveshape=waveshape,
            custom_curve=custom_curve,
            pre_filter=pre_filter,
            pre_filter_type=pre_filter_type,
            post_filter=post_filter,
            post_filter_type=post_filter_type,
            gain=gain_makeup,
            bias=bias,
            mix=mix,
        )
        
        parts = [f"drive={drive:.1f}", f"shape={waveshape}"]
        if pre_filter:
            parts.append(f"pre={pre_filter:.0f}Hz({pre_filter_type})")
        if post_filter:
            parts.append(f"post={post_filter:.0f}Hz({post_filter_type})")
        if gain_makeup != 1.0:
            parts.append(f"gain={gain_makeup:.2f}")
        if mix < 1.0:
            parts.append(f"mix={mix:.2f}")
        
        return f"OK: vamp applied ({', '.join(parts)})"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_dual(session: Session, args: List[str]) -> str:
    """Apply dual-stage overdrive with crossover.
    
    Usage:
      /dual                           -> Apply default warm dual OD
      /dual <preset>                  -> Apply preset (warm, bright, heavy)
      /dual drive1 <val>              -> Set drive for low band (1-15)
      /dual drive2 <val>              -> Set drive for high band (1-15)
      /dual shape1 <name>             -> Set waveshape for low band
      /dual shape2 <name>             -> Set waveshape for high band
      /dual crossover <Hz>            -> Set crossover frequency
      /dual blend <val>               -> Set band blend (0-1)
      /dual gain <val>                -> Set gain makeup
    
    Presets: warm, bright, heavy
    """
    if session.last_buffer is None:
        return "ERROR: no buffer to process. Generate audio first."
    
    from ..dsp.effects import dual_overdrive
    
    preset_settings = {
        'warm': {'drive1': 3.0, 'drive2': 4.0, 'shape1': 'tube', 'shape2': 'cubic', 'crossover': 600},
        'bright': {'drive1': 2.0, 'drive2': 6.0, 'shape1': 'cubic', 'shape2': 'tanh', 'crossover': 1200},
        'heavy': {'drive1': 6.0, 'drive2': 10.0, 'shape1': 'tube', 'shape2': 'fuzz', 'crossover': 400},
    }
    
    drive1 = 3.0
    drive2 = 4.0
    shape1 = 'tube'
    shape2 = 'cubic'
    crossover = 600.0
    blend = 0.5
    gain_makeup = 1.0
    
    i = 0
    while i < len(args):
        arg = args[i].lower()
        
        if arg in preset_settings:
            settings = preset_settings[arg]
            drive1 = settings.get('drive1', drive1)
            drive2 = settings.get('drive2', drive2)
            shape1 = settings.get('shape1', shape1)
            shape2 = settings.get('shape2', shape2)
            crossover = settings.get('crossover', crossover)
            i += 1
            continue
        
        if arg == 'drive1' and i + 1 < len(args):
            try:
                drive1 = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        elif arg == 'drive2' and i + 1 < len(args):
            try:
                drive2 = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        elif arg == 'shape1' and i + 1 < len(args):
            shape1 = args[i + 1].lower()
            i += 2
            continue
        elif arg == 'shape2' and i + 1 < len(args):
            shape2 = args[i + 1].lower()
            i += 2
            continue
        elif arg in ('crossover', 'xover', 'cross') and i + 1 < len(args):
            try:
                crossover = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        elif arg == 'blend' and i + 1 < len(args):
            try:
                blend = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        elif arg in ('gain', 'makeup') and i + 1 < len(args):
            try:
                gain_makeup = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        else:
            i += 1
    
    try:
        session.last_buffer = dual_overdrive(
            session.last_buffer,
            drive_low=drive1,
            drive_high=drive2,
            shape_low=shape1,
            shape_high=shape2,
            crossover=crossover,
            blend=blend,
            gain=gain_makeup,
        )
        
        return (f"OK: dual OD applied (lo: {shape1}@{drive1:.1f}, hi: {shape2}@{drive2:.1f}, "
                f"xover={crossover:.0f}Hz, gain={gain_makeup:.2f})")
    except Exception as e:
        return f"ERROR: {e}"


def cmd_amt(session: Session, args: List[str]) -> str:
    """Get or set the effect amount (unified 0-100 scale).
    
    Usage:
      /amt              -> Show current effect amount
      /amt <0-100>      -> Set amount (0=off, 50=moderate, 100=max)
      /amt <preset>     -> Use preset (subtle, light, moderate, strong, heavy, extreme)
    
    The amount parameter controls the intensity/mix of the currently selected effect.
    All effects use the same 0-100 scale for consistency.
    """
    if not hasattr(session, 'effect_amount'):
        session.effect_amount = 50.0  # Default to moderate
    
    if not args:
        return f"EFFECT AMOUNT: {session.effect_amount:.0f} (0=off, 50=moderate, 100=max)"
    
    # Parse amount value or preset name
    arg = args[0].lower().strip()
    
    presets = {
        'off': 0, 'subtle': 25, 'light': 35, 'moderate': 50,
        'medium': 50, 'strong': 70, 'heavy': 85, 'extreme': 100,
        'full': 100, 'half': 50, 'quarter': 25,
    }
    
    if arg in presets:
        session.effect_amount = float(presets[arg])
        return f"OK: effect amount set to {session.effect_amount:.0f} ({arg})"
    
    try:
        val = float(arg)
        val = max(0.0, min(100.0, val))
        session.effect_amount = val
        return f"OK: effect amount set to {session.effect_amount:.0f}"
    except ValueError:
        return f"ERROR: invalid amount '{arg}'. Use 0-100 or preset name (subtle, light, moderate, strong, heavy, extreme)"


def cmd_st(session: Session, args: List[str]) -> str:
    """Get or set the clip stretch factor.
    
    Usage:
      /st              -> Show current stretch factor
      /st <factor>     -> Set stretch (0.5=half speed, 1.0=normal, 2.0=double)
      /st <percent>%   -> Set stretch as percentage (50%=half, 100%=normal, 200%=double)
    
    The stretch factor is applied to clips when rendered/mixed.
    Use /lk to toggle pitch lock (preserve pitch during stretch).
    """
    if not hasattr(session, 'clip_stretch'):
        session.clip_stretch = 1.0
    
    if not args:
        percent = session.clip_stretch * 100
        pitch_mode = "pitch-locked" if getattr(session, 'pitch_locked', True) else "tape-style"
        return f"STRETCH: {session.clip_stretch:.2f}x ({percent:.0f}%) [{pitch_mode}]"
    
    arg = args[0].strip()
    
    try:
        if arg.endswith('%'):
            # Parse as percentage
            val = float(arg[:-1]) / 100.0
        else:
            val = float(arg)
        
        # Clamp to reasonable range
        val = max(0.1, min(10.0, val))
        session.clip_stretch = val
        percent = val * 100
        return f"OK: stretch set to {val:.2f}x ({percent:.0f}%)"
    except ValueError:
        return f"ERROR: invalid stretch value '{arg}'. Use decimal (0.5-10.0) or percentage (50%-1000%)"


def cmd_fxa(session: Session, args: List[str]) -> str:
    """Apply effects to the current buffer immediately.
    
    Usage:
      /fxa              -> Apply all effects in chain with their amounts
      /fxa <n>       -> Apply single effect to buffer
      /fxa <n> <amt> -> Apply effect with specific amount (0-100)
    
    This processes the buffer immediately rather than storing in the effects chain.
    """
    if session.last_buffer is None:
        return "ERROR: no buffer to process. Generate audio first."
    
    from ..dsp.effects import apply_effects_with_params, _effect_funcs
    
    if not args:
        # Apply all effects in chain with their params
        if not session.effects:
            return "ERROR: no effects in chain. Use /fx add <effect> first."
        try:
            session.last_buffer = apply_effects_with_params(
                session.last_buffer, 
                session.effects,
                session.effect_params
            )
            # Build summary
            parts = []
            for i, fx in enumerate(session.effects):
                amt = session.effect_params[i].get('amount', 50) if i < len(session.effect_params) else 50
                parts.append(f"{fx}@{amt:.0f}")
            return f"OK: applied {len(session.effects)} effect(s): {', '.join(parts)}"
        except Exception as e:
            return f"ERROR: {e}"
    
    # Apply single effect
    name = args[0].lower()
    target_name, err = resolve_effect_name(name)
    if err:
        return err
    
    # Get amount if specified
    amount = 50.0
    if len(args) > 1:
        try:
            amount = float(args[1])
            amount = max(0.0, min(100.0, amount))
        except ValueError:
            # Check for preset name
            presets = {'off': 0, 'subtle': 25, 'light': 35, 'moderate': 50, 
                      'strong': 70, 'heavy': 85, 'extreme': 100, 'full': 100}
            amount = presets.get(args[1].lower(), 50.0)
    
    try:
        session.last_buffer = apply_effects_with_params(
            session.last_buffer, 
            [target_name],
            [{'amount': amount}]
        )
        return f"OK: applied {target_name} (amount={amount:.0f})"
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================================
# INDEX-BASED FX COMMANDS (Fast workflow)
# ============================================================================

def cmd_fxs(session: Session, args: List[str]) -> str:
    """Select effect by index (shortcut for /fx sel).
    
    Usage:
      /fxs <n>          -> Select effect at index n
      /fxs              -> Show current selection
    """
    if not args:
        if session.selected_effect < 0 or session.selected_effect >= len(session.effects):
            return "FX SELECTION: none"
        fx = session.effects[session.selected_effect]
        params = session.effect_params[session.selected_effect] if session.selected_effect < len(session.effect_params) else {}
        amt = params.get('amount', 50)
        return f"FX SELECTION: [{session.selected_effect}] {fx} (amount={amt:.0f})"
    
    try:
        idx = int(args[0])
        if idx < 0 or idx >= len(session.effects):
            return f"ERROR: index {idx} out of range (0-{len(session.effects)-1})"
        session.selected_effect = idx
        fx = session.effects[idx]
        # Ensure params exist
        while len(session.effect_params) < len(session.effects):
            session.effect_params.append({'amount': 50.0})
        params = session.effect_params[idx]
        amt = params.get('amount', 50)
        return f"OK: selected [{idx}] {fx} (amount={amt:.0f})"
    except ValueError:
        return f"ERROR: invalid index '{args[0]}'"


def cmd_fxr(session: Session, args: List[str]) -> str:
    """Remove effect by index.
    
    Usage:
      /fxr <n>          -> Remove effect at index n
      /fxr              -> Remove currently selected effect
      /fxr last         -> Remove last effect in chain
    """
    if not session.effects:
        return "ERROR: no effects to remove"
    
    if not args:
        # Remove selected
        if session.selected_effect < 0 or session.selected_effect >= len(session.effects):
            return "ERROR: no effect selected. Use /fxs <n> first or /fxr <n>"
        idx = session.selected_effect
    elif args[0].lower() == 'last':
        idx = len(session.effects) - 1
    else:
        try:
            idx = int(args[0])
        except ValueError:
            return f"ERROR: invalid index '{args[0]}'"
    
    if idx < 0 or idx >= len(session.effects):
        return f"ERROR: index {idx} out of range (0-{len(session.effects)-1})"
    
    removed = session.effects.pop(idx)
    if idx < len(session.effect_params):
        session.effect_params.pop(idx)
    
    # Adjust selection
    if session.selected_effect >= len(session.effects):
        session.selected_effect = len(session.effects) - 1
    
    if session.effects:
        return f"OK: removed [{idx}] {removed}. Remaining: {', '.join(session.effects)}"
    return f"OK: removed [{idx}] {removed}. Chain empty."


def cmd_fxm(session: Session, args: List[str]) -> str:
    """Move effect in chain (reorder).
    
    Usage:
      /fxm <from> <to>  -> Move effect from index to new position
      /fxm up           -> Move selected effect up one position
      /fxm down         -> Move selected effect down one position
      /fxm top          -> Move selected effect to start of chain
      /fxm end          -> Move selected effect to end of chain
    """
    if len(session.effects) < 2:
        return "ERROR: need at least 2 effects to reorder"
    
    if not args:
        return "ERROR: usage: /fxm <from> <to> or /fxm up|down|top|end"
    
    arg = args[0].lower()
    
    if arg in ('up', 'down', 'top', 'end'):
        if session.selected_effect < 0 or session.selected_effect >= len(session.effects):
            return "ERROR: no effect selected"
        from_idx = session.selected_effect
        
        if arg == 'up':
            to_idx = max(0, from_idx - 1)
        elif arg == 'down':
            to_idx = min(len(session.effects) - 1, from_idx + 1)
        elif arg == 'top':
            to_idx = 0
        else:  # end
            to_idx = len(session.effects) - 1
    else:
        if len(args) < 2:
            return "ERROR: usage: /fxm <from> <to>"
        try:
            from_idx = int(args[0])
            to_idx = int(args[1])
        except ValueError:
            return "ERROR: invalid indices"
    
    if from_idx < 0 or from_idx >= len(session.effects):
        return f"ERROR: from index {from_idx} out of range"
    if to_idx < 0 or to_idx >= len(session.effects):
        return f"ERROR: to index {to_idx} out of range"
    
    if from_idx == to_idx:
        return "OK: no change needed"
    
    # Move effect
    fx = session.effects.pop(from_idx)
    params = session.effect_params.pop(from_idx) if from_idx < len(session.effect_params) else {'amount': 50.0}
    
    session.effects.insert(to_idx, fx)
    while len(session.effect_params) < to_idx:
        session.effect_params.append({'amount': 50.0})
    session.effect_params.insert(to_idx, params)
    
    # Update selection
    session.selected_effect = to_idx
    
    return f"OK: moved {fx} from [{from_idx}] to [{to_idx}]. Chain: {', '.join(session.effects)}"


def cmd_fxp(session: Session, args: List[str]) -> str:
    """View or set effect parameters by index.
    
    Usage:
      /fxp                    -> Show all effects with parameters
      /fxp <n>                -> Show parameters for effect at index n
      /fxp <n> <amt>          -> Set amount for effect at index n (0-100)
      /fxp <n> <param> <val>  -> Set specific parameter for effect
    
    Common parameters:
      amount (0-100)  - Effect intensity/mix
      wet (0-100)     - Wet signal level
      drive (0-100)   - Drive/gain amount
      time (0-100)    - Delay/reverb time
      feedback (0-100)- Feedback amount
    """
    # Ensure params list matches effects list
    while len(session.effect_params) < len(session.effects):
        session.effect_params.append({'amount': 50.0})
    
    if not args:
        # Show all effects with params
        if not session.effects:
            return "EFFECTS: (none)"
        lines = ["FX CHAIN:"]
        for i, fx in enumerate(session.effects):
            mark = "*" if i == session.selected_effect else " "
            params = session.effect_params[i] if i < len(session.effect_params) else {}
            amt = params.get('amount', 50)
            param_str = ", ".join(f"{k}={v:.0f}" for k, v in params.items())
            if param_str:
                lines.append(f"  [{i}]{mark} {fx}: {param_str}")
            else:
                lines.append(f"  [{i}]{mark} {fx}: amount={amt:.0f}")
        return '\n'.join(lines)
    
    # Parse index
    try:
        idx = int(args[0])
    except ValueError:
        # Not a number — assume it's a param name targeting selected effect
        if session.selected_effect >= 0 and session.selected_effect < len(session.effects):
            idx = session.selected_effect
            # Shift args: treat args[0] as param_name, args[1] as value
            args = [str(idx)] + list(args)
        else:
            return f"ERROR: '{args[0]}' is not a valid index. Select an effect first with /fxs <n>, or use /fxp <n> <param> <val>"
    
    if idx < 0 or idx >= len(session.effects):
        return f"ERROR: index {idx} out of range (0-{len(session.effects)-1})"
    
    fx = session.effects[idx]
    params = session.effect_params[idx]
    
    if len(args) == 1:
        # Show params for this effect
        param_str = ", ".join(f"{k}={v:.0f}" for k, v in params.items())
        if not param_str:
            param_str = "amount=50 (default)"
        return f"[{idx}] {fx}: {param_str}"
    
    if len(args) == 2:
        # Set amount (shortcut)
        try:
            amt = float(args[1])
            amt = max(0.0, min(100.0, amt))
            params['amount'] = amt
            return f"OK: [{idx}] {fx} amount={amt:.0f}"
        except ValueError:
            return f"ERROR: invalid amount '{args[1]}'"
    
    # Set specific parameter
    param_name = args[1].lower()
    try:
        value = float(args[2])
        value = max(0.0, min(100.0, value))
        params[param_name] = value
        return f"OK: [{idx}] {fx} {param_name}={value:.0f}"
    except ValueError:
        return f"ERROR: invalid value '{args[2]}'"


def cmd_fxc(session: Session, args: List[str]) -> str:
    """Copy effect (duplicate in chain).
    
    Usage:
      /fxc <n>          -> Duplicate effect at index n
      /fxc              -> Duplicate currently selected effect
    """
    if not session.effects:
        return "ERROR: no effects to copy"
    
    if not args:
        if session.selected_effect < 0 or session.selected_effect >= len(session.effects):
            return "ERROR: no effect selected"
        idx = session.selected_effect
    else:
        try:
            idx = int(args[0])
        except ValueError:
            return f"ERROR: invalid index '{args[0]}'"
    
    if idx < 0 or idx >= len(session.effects):
        return f"ERROR: index {idx} out of range"
    
    fx = session.effects[idx]
    params = session.effect_params[idx].copy() if idx < len(session.effect_params) else {'amount': 50.0}
    
    session.effects.append(fx)
    session.effect_params.append(params)
    
    new_idx = len(session.effects) - 1
    return f"OK: duplicated [{idx}] {fx} to [{new_idx}]"


def cmd_fxl(session: Session, args: List[str]) -> str:
    """List effects chain (compact format).
    
    Usage:
      /fxl              -> List all effects with amounts
    """
    if not session.effects:
        return "FX: (none)"
    
    while len(session.effect_params) < len(session.effects):
        session.effect_params.append({'amount': 50.0})
    
    parts = []
    for i, fx in enumerate(session.effects):
        mark = "*" if i == session.selected_effect else ""
        amt = session.effect_params[i].get('amount', 50)
        # Get short alias
        alias = None
        for a, name in _effect_aliases.items():
            if name == fx and len(a) <= 3:
                alias = a
                break
        if alias:
            parts.append(f"[{i}]{mark}{alias}@{amt:.0f}")
        else:
            short = fx[:8] if len(fx) > 8 else fx
            parts.append(f"[{i}]{mark}{short}@{amt:.0f}")
    
    return "FX: " + " ".join(parts)


# Quick add with amount
def cmd_fxq(session: Session, args: List[str]) -> str:
    """Quick add effect with amount.
    
    Usage:
      /fxq <effect> [amount]  -> Add effect with optional amount (0-100)
      /fxq r1 75              -> Add reverb_small at 75% intensity
      /fxq v2                 -> Add vamp_medium at default 50%
    """
    if not args:
        return "ERROR: usage: /fxq <effect> [amount]"
    
    name = args[0].lower()
    amount = 50.0
    
    if len(args) > 1:
        try:
            amount = float(args[1])
            amount = max(0.0, min(100.0, amount))
        except ValueError:
            pass
    
    # Resolve alias and validate
    target_name, err = resolve_effect_name(name)
    if err:
        return err
    
    try:
        session.effects.append(target_name)
        session.effect_params.append({'amount': amount})
        idx = len(session.effects) - 1
        session.selected_effect = idx
        
        return f"OK: [{idx}] {target_name} added (amount={amount:.0f})"
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================================
# CHAIN BUILDER - Fast effect chain construction
# ============================================================================

def cmd_chain(session: Session, args: List[str]) -> str:
    """Build effect chain quickly with compact syntax.
    
    Usage:
      /chain <fx1>[@amt] <fx2>[@amt] ...  -> Build chain
      /chain r1@30 v2@75 d1@50            -> Reverb 30%, vamp 75%, delay 50%
      /chain clear                         -> Clear chain
      /chain save <name>                   -> Save current chain as preset
      /chain load <name>                   -> Load saved chain preset
      /chain presets                       -> List saved presets
    
    Examples:
      /chain r1 v2              -> Add with default 50% amounts
      /chain r1@25 s1@80        -> Reverb at 25%, saturation at 80%
      /chain clear r1@30 v2@60  -> Clear then build new chain
    """
    if not args:
        # Show current chain
        if not session.effects:
            return "CHAIN: (empty). Use /chain <fx1>[@amt] <fx2>[@amt] ..."
        
        parts = []
        for i, fx in enumerate(session.effects):
            amt = session.effect_params[i].get('amount', 50) if i < len(session.effect_params) else 50
            # Get short alias
            alias = None
            for a, name in _effect_aliases.items():
                if name == fx and len(a) <= 3:
                    alias = a
                    break
            if alias:
                parts.append(f"{alias}@{amt:.0f}")
            else:
                parts.append(f"{fx}@{amt:.0f}")
        return "CHAIN: " + " ".join(parts)
    
    # Handle subcommands
    if args[0].lower() == 'clear':
        session.effects = []
        session.effect_params = []
        session.selected_effect = -1
        if len(args) == 1:
            return "OK: chain cleared"
        # Continue with rest of args to build new chain
        args = args[1:]
    
    if args and args[0].lower() == 'presets':
        presets = getattr(session, 'chain_presets', {})
        if not presets:
            return "CHAIN PRESETS: (none). Use /chain save <name> to create."
        return "CHAIN PRESETS: " + ", ".join(presets.keys())
    
    if args and args[0].lower() == 'save' and len(args) >= 2:
        name = args[1].lower()
        if not session.effects:
            return "ERROR: no chain to save"
        if not hasattr(session, 'chain_presets'):
            session.chain_presets = {}
        session.chain_presets[name] = {
            'effects': session.effects.copy(),
            'params': [p.copy() for p in session.effect_params]
        }
        return f"OK: chain saved as '{name}'"
    
    if args and args[0].lower() == 'load' and len(args) >= 2:
        name = args[1].lower()
        presets = getattr(session, 'chain_presets', {})
        if name not in presets:
            return f"ERROR: preset '{name}' not found"
        preset = presets[name]
        session.effects = preset['effects'].copy()
        session.effect_params = [p.copy() for p in preset['params']]
        session.selected_effect = len(session.effects) - 1 if session.effects else -1
        return f"OK: loaded chain '{name}' ({len(session.effects)} effects)"
    
    # Parse effect specs and build chain
    from ..dsp.effects import _effect_funcs
    
    added = []
    for spec in args:
        # Check if it's a pure numeric index — select that chain slot
        try:
            slot_idx = int(spec)
            if 0 <= slot_idx < len(session.effects):
                session.selected_effect = slot_idx
                fx = session.effects[slot_idx]
                return f"OK: selected chain slot [{slot_idx}] ({fx})"
            else:
                return f"ERROR: slot {slot_idx} out of range (0-{len(session.effects)-1})" if session.effects else "ERROR: chain is empty"
        except ValueError:
            pass  # Not numeric, continue with effect name parsing
        
        # Parse fx@amount or just fx
        if '@' in spec:
            fx_part, amt_part = spec.split('@', 1)
            try:
                amount = float(amt_part)
                amount = max(0.0, min(100.0, amount))
            except ValueError:
                amount = 50.0
        else:
            fx_part = spec
            amount = 50.0
        
        # Resolve alias and validate
        fx_name = fx_part.lower()
        target_name, err = resolve_effect_name(fx_name)
        if err:
            continue  # Skip unknown effects in batch
        
        session.effects.append(target_name)
        session.effect_params.append({'amount': amount})
        added.append(f"{fx_part}@{amount:.0f}")
    
    if session.effects:
        session.selected_effect = len(session.effects) - 1
    
    if added:
        return f"OK: chain built: {' '.join(added)}"
    return "ERROR: no valid effects specified"


# ============================================================================  
# QUICK EFFECT SLOTS (numbered shortcuts)
# ============================================================================

def cmd_e0(session: Session, args: List[str]) -> str:
    """Quick slot 0 - select/apply/set effect at index 0."""
    return _quick_effect_slot(session, 0, args)

def cmd_e1(session: Session, args: List[str]) -> str:
    """Quick slot 1 - select/apply/set effect at index 1."""
    return _quick_effect_slot(session, 1, args)

def cmd_e2(session: Session, args: List[str]) -> str:
    """Quick slot 2 - select/apply/set effect at index 2."""
    return _quick_effect_slot(session, 2, args)

def cmd_e3(session: Session, args: List[str]) -> str:
    """Quick slot 3 - select/apply/set effect at index 3."""
    return _quick_effect_slot(session, 3, args)

def cmd_e4(session: Session, args: List[str]) -> str:
    """Quick slot 4 - select/apply/set effect at index 4."""
    return _quick_effect_slot(session, 4, args)

def cmd_e5(session: Session, args: List[str]) -> str:
    """Quick slot 5 - select/apply/set effect at index 5."""
    return _quick_effect_slot(session, 5, args)

def cmd_e6(session: Session, args: List[str]) -> str:
    """Quick slot 6 - select/apply/set effect at index 6."""
    return _quick_effect_slot(session, 6, args)

def cmd_e7(session: Session, args: List[str]) -> str:
    """Quick slot 7 - select/apply/set effect at index 7."""
    return _quick_effect_slot(session, 7, args)


def _quick_effect_slot(session: Session, idx: int, args: List[str]) -> str:
    """Handle quick effect slot command.
    
    Usage:
      /e0           -> Select effect at slot 0
      /e0 <amt>     -> Set amount for slot 0
      /e0 <fx>      -> Replace slot 0 with new effect (or add if empty)
      /e0 <fx> <amt>-> Replace with new effect and amount
    """
    # Ensure params list
    while len(session.effect_params) < len(session.effects):
        session.effect_params.append({'amount': 50.0})
    
    if not args:
        # Just select
        if idx >= len(session.effects):
            return f"ERROR: slot {idx} empty (chain has {len(session.effects)} effects)"
        session.selected_effect = idx
        fx = session.effects[idx]
        amt = session.effect_params[idx].get('amount', 50)
        return f"SLOT {idx}: {fx} (amount={amt:.0f})"
    
    arg = args[0]
    
    # Check if it's a number (set amount)
    try:
        amount = float(arg)
        amount = max(0.0, min(100.0, amount))
        if idx >= len(session.effects):
            return f"ERROR: slot {idx} empty"
        session.effect_params[idx]['amount'] = amount
        return f"OK: slot {idx} amount={amount:.0f}"
    except ValueError:
        pass
    
    # It's an effect name - replace or add
    fx_name = arg.lower()
    target_name, err = resolve_effect_name(fx_name)
    if err:
        return err
    
    # Get amount if provided
    amount = 50.0
    if len(args) > 1:
        try:
            amount = float(args[1])
            amount = max(0.0, min(100.0, amount))
        except ValueError:
            pass
    
    # Add or replace
    if idx < len(session.effects):
        old = session.effects[idx]
        session.effects[idx] = target_name
        session.effect_params[idx] = {'amount': amount}
        return f"OK: slot {idx} replaced {old} with {target_name}@{amount:.0f}"
    elif idx == len(session.effects):
        session.effects.append(target_name)
        session.effect_params.append({'amount': amount})
        session.selected_effect = idx
        return f"OK: slot {idx} added {target_name}@{amount:.0f}"
    else:
        return f"ERROR: slot {idx} too far ahead (chain has {len(session.effects)} effects)"


# ============================================================================
# EFFECT INTENSITY PRESETS
# ============================================================================

def cmd_dry(session: Session, args: List[str]) -> str:
    """Set all effects to 0% (dry/bypass)."""
    if not session.effects:
        return "EFFECTS: (none)"
    for params in session.effect_params:
        params['amount'] = 0.0
    return f"OK: all {len(session.effects)} effects set to 0% (dry)"

def cmd_wet(session: Session, args: List[str]) -> str:
    """Set all effects to 100% (full wet)."""
    if not session.effects:
        return "EFFECTS: (none)"
    for params in session.effect_params:
        params['amount'] = 100.0
    return f"OK: all {len(session.effects)} effects set to 100% (wet)"

def cmd_half(session: Session, args: List[str]) -> str:
    """Set all effects to 50% (moderate)."""
    if not session.effects:
        return "EFFECTS: (none)"
    for params in session.effect_params:
        params['amount'] = 50.0
    return f"OK: all {len(session.effects)} effects set to 50%"


# ============================================================================
# EFFECT BYPASS
# ============================================================================

def cmd_bypass(session: Session, args: List[str]) -> str:
    """Toggle effect bypass (store amounts and set to 0, or restore).
    
    Usage:
      /bypass           -> Toggle bypass on/off
      /bypass on        -> Enable bypass (mute effects)
      /bypass off       -> Disable bypass (restore effects)
      /bypass <n>       -> Toggle bypass for single effect at index
    """
    if not hasattr(session, 'effect_bypass_store'):
        session.effect_bypass_store = {}
        session.effects_bypassed = False
    
    if not session.effects:
        return "EFFECTS: (none)"
    
    if not args:
        # Toggle all
        if session.effects_bypassed:
            # Restore
            for i, params in enumerate(session.effect_params):
                if i in session.effect_bypass_store:
                    params['amount'] = session.effect_bypass_store[i]
            session.effect_bypass_store = {}
            session.effects_bypassed = False
            return "OK: effects restored (bypass off)"
        else:
            # Store and bypass
            session.effect_bypass_store = {}
            for i, params in enumerate(session.effect_params):
                session.effect_bypass_store[i] = params.get('amount', 50.0)
                params['amount'] = 0.0
            session.effects_bypassed = True
            return "OK: effects bypassed (amounts stored)"
    
    arg = args[0].lower()
    
    if arg == 'on':
        if session.effects_bypassed:
            return "OK: already bypassed"
        session.effect_bypass_store = {}
        for i, params in enumerate(session.effect_params):
            session.effect_bypass_store[i] = params.get('amount', 50.0)
            params['amount'] = 0.0
        session.effects_bypassed = True
        return "OK: effects bypassed"
    
    if arg == 'off':
        if not session.effects_bypassed:
            return "OK: already active"
        for i, params in enumerate(session.effect_params):
            if i in session.effect_bypass_store:
                params['amount'] = session.effect_bypass_store[i]
        session.effect_bypass_store = {}
        session.effects_bypassed = False
        return "OK: effects restored"
    
    # Single effect toggle
    try:
        idx = int(arg)
        if idx < 0 or idx >= len(session.effects):
            return f"ERROR: index {idx} out of range"
        
        params = session.effect_params[idx]
        current = params.get('amount', 50.0)
        
        if current == 0.0:
            # Restore from store if available
            stored = session.effect_bypass_store.get(idx, 50.0)
            params['amount'] = stored
            if idx in session.effect_bypass_store:
                del session.effect_bypass_store[idx]
            return f"OK: effect {idx} restored to {stored:.0f}%"
        else:
            # Store and bypass
            session.effect_bypass_store[idx] = current
            params['amount'] = 0.0
            return f"OK: effect {idx} bypassed (was {current:.0f}%)"
    except ValueError:
        return f"ERROR: invalid argument '{arg}'"


# ============================================================================
# ENHANCED FILTER COMMANDS (Unified interface)
# ============================================================================

def cmd_flt(session: Session, args: List[str]) -> str:
    """Combined filter command - set type, cutoff, and resonance in one call.
    
    Usage:
      /flt                          -> Show current filter settings
      /flt <type>                   -> Set filter type (also enables)
      /flt <type> <cutoff>          -> Set type and cutoff
      /flt <type> <cutoff> <res>    -> Set type, cutoff, and resonance
      /flt off                      -> Disable filter
      /flt on                       -> Enable filter
      /flt <cutoff>Hz               -> Set cutoff only (append Hz)
      /flt q<res>                   -> Set resonance only (prefix q)
      /flt slot <n>                 -> Select filter slot
      /flt list                     -> List filter types
    
    Examples:
      /flt lp 800 75                  -> Lowpass at 800Hz, resonance=75
      /flt moog 500 85                -> Moog ladder at 500Hz, resonance=85
      /flt acid 300                   -> Acid (303) filter at 300Hz
      /flt 1200Hz                     -> Set cutoff to 1200Hz
      /flt q75                        -> Set resonance to 75
    
    Resonance: 0-100 scale (0=none, 50=moderate, 100=high, >100=wacky)
    """
    slot = session.selected_filter
    
    if not args:
        # Show current settings
        f_type = session.filter_types.get(slot, 0)
        f_name = session.filter_type_names.get(f_type, 'unknown')
        cutoff = session.filter_cutoffs.get(slot, 1000.0)
        res = session.filter_resonances.get(slot, 50.0)  # 0-100 scale
        enabled = session.filter_enabled.get(slot, False)
        status = "ON" if enabled else "OFF"
        return f"FILTER[{slot}]: {status} | {f_name} ({f_type}) | cut={cutoff:.0f}Hz | res={res:.0f}"
    
    arg0 = args[0].lower()
    
    # Handle special commands
    if arg0 == 'off' or arg0 == 'none' or arg0 == 'disable':
        session.filter_enabled[slot] = False
        return f"OK: filter[{slot}] disabled"
    
    if arg0 == 'on' or arg0 == 'enable':
        session.filter_enabled[slot] = True
        f_type = session.filter_types.get(slot, 0)
        f_name = session.filter_type_names.get(f_type, 'unknown')
        return f"OK: filter[{slot}] enabled ({f_name})"
    
    if arg0 == 'list' or arg0 == 'types':
        lines = ["Filter types (30):"]
        lines.append("  Basic: lp hp bp notch peak ring allpass")
        lines.append("  Comb: comb_ff comb_fb comb_both")
        lines.append("  Analog: analog acid/303 moog")
        lines.append("  SVF: svf_lp svf_hp svf_bp")
        lines.append("  Formant: fa fe fi fo fu (ah eh ee oh oo)")
        lines.append("  Shelf: lowshelf/bass highshelf/treble")
        lines.append("  Destroy: bitcrush downsample")
        lines.append("  Utility: dc_block tilt")
        lines.append("  Character: resonant vocal telephone")
        return '\n'.join(lines)
    
    if arg0 == 'slot' and len(args) > 1:
        try:
            idx = int(args[1])
            session.select_filter(idx)
            return f"OK: filter slot set to {idx}"
        except Exception as e:
            return f"ERROR: {e}"
    
    # Handle cutoff-only (ends with Hz or is pure number > 29)
    if arg0.endswith('hz'):
        try:
            cutoff = float(arg0[:-2])
            session.set_cutoff(cutoff)
            return f"OK: filter[{slot}] cutoff set to {cutoff:.0f}Hz"
        except ValueError:
            return f"ERROR: invalid cutoff '{arg0}'"
    
    # Handle resonance-only (starts with q)
    if arg0.startswith('q') and len(arg0) > 1:
        try:
            res = float(arg0[1:])
            session.set_resonance(res)
            return f"OK: filter[{slot}] resonance set to {res:.0f}"
        except ValueError:
            return f"ERROR: invalid resonance '{arg0}'"
    
    # Try to parse as type, cutoff, resonance
    type_set = False
    cutoff_set = False
    res_set = False
    
    # First arg: could be type or cutoff number
    try:
        # Check if it's a pure number (could be type index or cutoff)
        val = float(arg0)
        if val <= 29 and val == int(val):
            # Treat as filter type index
            session.set_filter_type(int(val))
            type_set = True
        else:
            # Treat as cutoff
            session.set_cutoff(val)
            cutoff_set = True
    except ValueError:
        # Not a number, try as filter type alias
        try:
            session.set_filter_type(arg0)
            type_set = True
        except ValueError as e:
            return f"ERROR: {e}"
    
    # Second arg: cutoff (if type was set) or resonance (if cutoff was set)
    if len(args) > 1:
        try:
            val = float(args[1])
            if type_set and not cutoff_set:
                session.set_cutoff(val)
                cutoff_set = True
            elif cutoff_set and not res_set:
                session.set_resonance(val)
                res_set = True
        except ValueError:
            pass
    
    # Third arg: resonance
    if len(args) > 2:
        try:
            val = float(args[2])
            session.set_resonance(val)
            res_set = True
        except ValueError:
            pass
    
    # Build response
    f_type = session.filter_types.get(slot, 0)
    f_name = session.filter_type_names.get(f_type, 'unknown')
    cutoff = session.filter_cutoffs.get(slot, 1000.0)
    res = session.filter_resonances.get(slot, 50.0)  # 0-100 scale
    enabled = session.filter_enabled.get(slot, False)
    
    return f"OK: filter[{slot}] = {f_name} | cut={cutoff:.0f}Hz | res={res:.0f} | enabled={enabled}"


def cmd_f0(session: Session, args: List[str]) -> str:
    """Quick filter slot 0."""
    return _quick_filter_slot(session, 0, args)

def cmd_f1(session: Session, args: List[str]) -> str:
    """Quick filter slot 1."""
    return _quick_filter_slot(session, 1, args)

def cmd_f2(session: Session, args: List[str]) -> str:
    """Quick filter slot 2."""
    return _quick_filter_slot(session, 2, args)

def cmd_f3(session: Session, args: List[str]) -> str:
    """Quick filter slot 3."""
    return _quick_filter_slot(session, 3, args)

def cmd_f4(session: Session, args: List[str]) -> str:
    """Quick filter slot 4."""
    return _quick_filter_slot(session, 4, args)

def cmd_f5(session: Session, args: List[str]) -> str:
    """Quick filter slot 5."""
    return _quick_filter_slot(session, 5, args)

def cmd_f6(session: Session, args: List[str]) -> str:
    """Quick filter slot 6."""
    return _quick_filter_slot(session, 6, args)

def cmd_f7(session: Session, args: List[str]) -> str:
    """Quick filter slot 7."""
    return _quick_filter_slot(session, 7, args)


def _quick_filter_slot(session: Session, slot: int, args: List[str]) -> str:
    """Handle quick filter slot command.
    
    Usage:
      /f0                   -> Select and show slot 0
      /f0 lp                -> Set slot 0 to lowpass
      /f0 lp 800            -> Set to lowpass at 800Hz
      /f0 lp 800 75         -> Set to lowpass at 800Hz, res=75
      /f0 off               -> Disable slot 0
      /f0 on                -> Enable slot 0
    
    Resonance: 0-100 scale (0=none, 50=moderate, 100=high)
    """
    # Ensure slot exists
    if slot >= session.filter_count:
        session.set_filter_count(slot + 1)
    
    # Select this slot
    session.select_filter(slot)
    
    if not args:
        # Just show status
        f_type = session.filter_types.get(slot, 0)
        f_name = session.filter_type_names.get(f_type, 'unknown')
        cutoff = session.filter_cutoffs.get(slot, 1000.0)
        res = session.filter_resonances.get(slot, 50.0)  # 0-100 scale
        enabled = session.filter_enabled.get(slot, False)
        status = "ON" if enabled else "OFF"
        return f"FILTER[{slot}]: {status} | {f_name} | cut={cutoff:.0f}Hz | res={res:.0f}"
    
    # Delegate to cmd_flt for the actual work
    return cmd_flt(session, args)


# Quick filter type shortcuts
def cmd_lp(session: Session, args: List[str]) -> str:
    """Quick lowpass filter.
    
    Usage:
      /lp [cutoff] [res]    -> Enable lowpass filter
      /lp 800               -> Lowpass at 800Hz
      /lp 800 2.0           -> Lowpass at 800Hz, Q=2.0
    """
    return _quick_filter_type(session, 'lp', args)

def cmd_hp(session: Session, args: List[str]) -> str:
    """Quick highpass filter.
    
    Usage:
      /hp [cutoff] [res]    -> Enable highpass filter
    """
    return _quick_filter_type(session, 'hp', args)

def cmd_bp(session: Session, args: List[str]) -> str:
    """Quick bandpass filter.
    
    Usage:
      /bp [cutoff] [res]    -> Enable bandpass filter
    """
    return _quick_filter_type(session, 'bp', args)

def cmd_notch(session: Session, args: List[str]) -> str:
    """Quick notch filter.
    
    Usage:
      /notch [cutoff] [res] -> Enable notch filter
    """
    return _quick_filter_type(session, 'notch', args)


def _quick_filter_type(session: Session, filter_type: str, args: List[str]) -> str:
    """Set filter type with optional cutoff and resonance."""
    slot = session.selected_filter
    
    try:
        session.set_filter_type(filter_type)
    except ValueError as e:
        return f"ERROR: {e}"
    
    # Parse cutoff
    if len(args) > 0:
        try:
            cutoff = float(args[0])
            session.set_cutoff(cutoff)
        except ValueError:
            pass
    
    # Parse resonance
    if len(args) > 1:
        try:
            res = float(args[1])
            session.set_resonance(res)
        except ValueError:
            pass
    
    f_type = session.filter_types.get(slot, 0)
    f_name = session.filter_type_names.get(f_type, 'unknown')
    cutoff = session.filter_cutoffs.get(slot, 1000.0)
    res = session.filter_resonances.get(slot, 50.0)  # 0-100 scale
    
    return f"OK: filter[{slot}] = {f_name} | cut={cutoff:.0f}Hz | res={res:.0f}"


# Filter apply command
def cmd_fa(session: Session, args: List[str]) -> str:
    """Apply current filter to buffer immediately.
    
    Usage:
      /fa                   -> Apply current filter settings to buffer
      /fa <type> [cut] [q]  -> Set and apply filter in one step
      /fa lp 800 2.0        -> Apply lowpass at 800Hz, Q=2.0
    """
    if session.last_buffer is None:
        return "ERROR: no buffer to filter. Generate audio first."
    
    # If args provided, set filter first
    if args:
        result = cmd_flt(session, args)
        if result.startswith("ERROR"):
            return result
    
    slot = session.selected_filter
    f_type = session.filter_types.get(slot, 0)
    cutoff = session.filter_cutoffs.get(slot, 1000.0)
    res_100 = session.filter_resonances.get(slot, 50.0)  # 0-100 scale
    enabled = session.filter_enabled.get(slot, True)
    
    if not enabled:
        return "ERROR: filter is disabled. Use /flt on to enable."
    
    try:
        from ..dsp.effects import _apply_filter
        from ..dsp.scaling import scale_resonance
        # Scale resonance from 0-100 to Q value
        res_q = scale_resonance(res_100)
        session.last_buffer = _apply_filter(session.last_buffer, f_type, cutoff, res_q)
        f_name = session.filter_type_names.get(f_type, 'unknown')
        return f"OK: applied {f_name} filter (cut={cutoff:.0f}Hz, res={res_100:.0f})"
    except Exception as e:
        return f"ERROR: {e}"


# Filter sweep command
def cmd_sweep(session: Session, args: List[str]) -> str:
    """Apply filter sweep to buffer.
    
    Usage:
      /sweep <start> <end> [time]   -> Sweep cutoff from start to end Hz
      /sweep 200 8000               -> Sweep from 200Hz to 8000Hz
      /sweep 8000 200 2.0           -> Reverse sweep over 2 seconds
    """
    if session.last_buffer is None:
        return "ERROR: no buffer. Generate audio first."
    
    if len(args) < 2:
        return "ERROR: usage: /sweep <start_hz> <end_hz> [time_seconds]"
    
    try:
        start_hz = float(args[0])
        end_hz = float(args[1])
        duration = float(args[2]) if len(args) > 2 else len(session.last_buffer) / 48000
    except ValueError:
        return "ERROR: invalid sweep parameters"
    
    slot = session.selected_filter
    f_type = session.filter_types.get(slot, 0)
    res_100 = session.filter_resonances.get(slot, 50.0)  # 0-100 scale
    
    import numpy as np
    from ..dsp.effects import _apply_filter
    from ..dsp.scaling import scale_resonance
    
    # Scale resonance from 0-100 to Q value
    res_q = scale_resonance(res_100)
    
    # Create logarithmic sweep
    samples = len(session.last_buffer)
    freqs = np.logspace(np.log10(max(20, start_hz)), np.log10(min(20000, end_hz)), samples)
    
    # Apply time-varying filter (process in chunks)
    chunk_size = 1024
    output = np.zeros_like(session.last_buffer)
    
    for i in range(0, samples, chunk_size):
        end_i = min(i + chunk_size, samples)
        chunk = session.last_buffer[i:end_i].copy()
        # Use midpoint frequency for this chunk
        mid_freq = freqs[(i + end_i) // 2]
        filtered = _apply_filter(chunk, f_type, mid_freq, res_q)
        output[i:end_i] = filtered[:end_i - i]
    
    session.last_buffer = output
    f_name = session.filter_type_names.get(f_type, 'unknown')
    return f"OK: swept {f_name} from {start_hz:.0f}Hz to {end_hz:.0f}Hz"


# Filter info command
def cmd_hflt(session: Session, args: List[str]) -> str:
    """Comprehensive filter help.
    
    Usage:
      /hflt           -> Show filter commands and current settings
      /hflt types     -> List all 30 filter types
      /hflt all       -> Show all filter slots
      /hflt <type>    -> Show info about specific filter type
    """
    if args:
        arg = args[0].lower()
        
        if arg == 'types':
            return _filter_types_help(session)
        
        if arg == 'all':
            return _filter_slots_help(session)
        
        # Check if it's a filter type
        if arg in session.filter_type_aliases:
            idx = session.filter_type_aliases[arg]
            name = session.filter_type_names.get(idx, 'unknown')
            aliases = [k for k, v in session.filter_type_aliases.items() if v == idx]
            return f"FILTER TYPE {idx}: {name}\n  Aliases: {', '.join(sorted(aliases))}"
    
    # Default help
    lines = [
        "=== FILTER COMMANDS ===",
        "",
        "QUICK ACCESS:",
        "  /flt <type> [cut] [q]   Combined filter command",
        "  /f0-/f7                 Quick filter slots",
        "  /lp /hp /bp /notch      Quick filter types",
        "  /fa [type] [cut] [q]    Apply filter to buffer",
        "  /sweep <start> <end>    Filter sweep",
        "",
        "CONTROL:",
        "  /ft <type>      Set filter type (0-29 or alias)",
        "  /cut <Hz>       Set cutoff (20-20000)",
        "  /res <Q>        Set resonance (0-20)",
        "  /fen <0|1>      Enable/disable filter",
        "  /fs <n>         Select filter slot",
        "  /fc <n>         Set filter count",
        "",
        "FILTER TYPES (30 total):",
        "  Basic:    lp hp bp notch peak ring allpass",
        "  Comb:     comb_ff comb_fb comb_both",
        "  Analog:   analog acid/303 moog",
        "  SVF:      svf_lp svf_hp svf_bp",
        "  Formant:  fa fe fi fo fu (ah eh ee oh oo)",
        "  Shelf:    lowshelf highshelf (bass treble)",
        "  Destroy:  bitcrush downsample",
        "  Utility:  dc_block tilt",
        "  Character: resonant vocal telephone",
        "",
        "Use /hflt types for full list, /hflt all for all slots",
    ]
    
    # Add current settings
    slot = session.selected_filter
    f_type = session.filter_types.get(slot, 0)
    f_name = session.filter_type_names.get(f_type, 'unknown')
    cutoff = session.filter_cutoffs.get(slot, 1000.0)
    res = session.filter_resonances.get(slot, 50.0)
    enabled = session.filter_enabled.get(slot, False)
    
    lines.append("")
    lines.append(f"CURRENT FILTER[{slot}]: {f_name} | cut={cutoff:.0f}Hz | Q={res:.2f} | {'ON' if enabled else 'OFF'}")
    
    return '\n'.join(lines)


def _filter_types_help(session: Session) -> str:
    """Generate filter types help."""
    lines = [
        "=== ALL FILTER TYPES (30) ===",
        "",
        "BASIC (0-6):",
    ]
    
    categories = [
        ("BASIC", range(0, 7)),
        ("COMB", range(7, 10)),
        ("ANALOG", range(10, 12)),
        ("FORMANT", range(12, 17)),
        ("SHELF", range(17, 19)),
        ("MOOG/SVF", range(19, 23)),
        ("DESTRUCTIVE", range(23, 25)),
        ("UTILITY", range(25, 27)),
        ("CHARACTER", range(27, 30)),
    ]
    
    for cat_name, idx_range in categories:
        lines.append(f"\n{cat_name} ({idx_range.start}-{idx_range.stop-1}):")
        for idx in idx_range:
            name = session.filter_type_names.get(idx, 'unknown')
            aliases = [k for k, v in session.filter_type_aliases.items() if v == idx]
            alias_str = ', '.join(sorted(aliases)[:4])
            lines.append(f"  {idx:2d}: {name:12s} ({alias_str})")
    
    lines.append("")
    lines.append("Special: none, off, -1, bypass -> disable filter")
    return '\n'.join(lines)


def _filter_slots_help(session: Session) -> str:
    """Generate filter slots help."""
    lines = ["=== ALL FILTER SLOTS ==="]
    
    for slot in range(session.filter_count):
        f_type = session.filter_types.get(slot, 0)
        f_name = session.filter_type_names.get(f_type, 'unknown')
        cutoff = session.filter_cutoffs.get(slot, 1000.0)
        res = session.filter_resonances.get(slot, 50.0)
        enabled = session.filter_enabled.get(slot, False)
        status = "ON " if enabled else "OFF"
        mark = "*" if slot == session.selected_filter else " "
        lines.append(f"  [{slot}]{mark} {status} {f_name:12s} cut={cutoff:6.0f}Hz Q={res:.2f}")
    
    return '\n'.join(lines)


# ============================================================================
# FX CHAIN POSITION COMMANDS - BFX, TFX, MFX, FFX
# ============================================================================

def _get_fx_chain(session: Session, target: str) -> list:
    """Get FX chain for specified target."""
    if target == 'buffer':
        return session.buffer_fx_chain
    elif target == 'track':
        return session.track_fx_chain
    elif target == 'master':
        return session.master_fx_chain
    elif target == 'file':
        return session.file_fx_chain
    return []


def _format_fx_chain(chain: list, name: str) -> str:
    """Format FX chain for display."""
    if not chain:
        return f"{name} FX chain: (empty)"
    
    lines = [f"{name} FX chain ({len(chain)} effects):"]
    for i, (fx_name, params) in enumerate(chain, 1):
        param_str = ', '.join(f"{k}={v}" for k, v in params.items()) if params else ""
        lines.append(f"  {i}. {fx_name}" + (f" ({param_str})" if param_str else ""))
    return '\n'.join(lines)


def _parse_fx_args(args: list) -> tuple:
    """Parse effect name and parameters from args.
    
    Returns (effect_name, params_dict) or (None, None) for show command.
    """
    if not args:
        return None, None
    
    effect_name = args[0].lower()
    # Resolve aliases via the canonical _effect_aliases map
    effect_name = _effect_aliases.get(effect_name, effect_name)
    params = {}
    
    i = 1
    while i < len(args):
        arg = args[i]
        if '=' in arg:
            key, val = arg.split('=', 1)
            try:
                params[key] = float(val)
            except ValueError:
                params[key] = val
        elif i + 1 < len(args):
            # Check if it's a key-value pair
            try:
                params[args[i]] = float(args[i + 1])
                i += 1
            except ValueError:
                pass
        i += 1
    
    return effect_name, params


def cmd_bfx(session: Session, args: List[str]) -> str:
    """Buffer FX chain - effects applied to current buffer.
    
    Usage:
      /bfx                    -> Show buffer FX chain
      /bfx add <effect> [params] -> Add effect to chain
      /bfx rm <n>             -> Remove effect at position
      /bfx clear              -> Clear entire chain
      /bfx apply              -> Apply chain to current buffer
      /bfx <n> <param>=<val>  -> Modify effect params at position
    
    Examples:
      /bfx add reverb amount=50
      /bfx add saturate_soft amount=30
      /bfx rm 1
      /bfx apply
    """
    if not args:
        return _format_fx_chain(session.buffer_fx_chain, "Buffer")
    
    cmd = args[0].lower()
    
    if cmd == 'add' and len(args) > 1:
        effect_name, params = _parse_fx_args(args[1:])
        if effect_name:
            session.buffer_fx_chain.append((effect_name, params))
            return f"OK: added '{effect_name}' to buffer FX chain (position {len(session.buffer_fx_chain)})"
        return "ERROR: no effect name specified"
    
    elif cmd == 'rm' and len(args) > 1:
        try:
            idx = int(args[1]) - 1  # 1-indexed
            if 0 <= idx < len(session.buffer_fx_chain):
                removed = session.buffer_fx_chain.pop(idx)
                return f"OK: removed '{removed[0]}' from buffer FX chain"
            return f"ERROR: invalid position {args[1]}"
        except ValueError:
            return "ERROR: position must be a number"
    
    elif cmd == 'clear':
        count = len(session.buffer_fx_chain)
        session.buffer_fx_chain.clear()
        return f"OK: cleared {count} effects from buffer FX chain"
    
    elif cmd == 'apply':
        if session.last_buffer is None:
            return "ERROR: no buffer to process"
        if not session.buffer_fx_chain:
            return "ERROR: buffer FX chain is empty"
        
        from ..dsp.effects import apply_effects_with_params
        try:
            # Unpack tuples into separate lists
            effect_names = [fx[0] for fx in session.buffer_fx_chain]
            effect_params = [fx[1] for fx in session.buffer_fx_chain]
            session.last_buffer = apply_effects_with_params(
                session.last_buffer, 
                effect_names,
                effect_params
            )
            return f"OK: applied {len(session.buffer_fx_chain)} effects to buffer"
        except Exception as e:
            return f"ERROR: {e}"
    
    # Modify params at position
    try:
        idx = int(cmd) - 1  # 1-indexed
        if 0 <= idx < len(session.buffer_fx_chain):
            fx_name, old_params = session.buffer_fx_chain[idx]
            _, new_params = _parse_fx_args(args[1:])
            old_params.update(new_params)
            return f"OK: updated '{fx_name}' params: {old_params}"
        return f"ERROR: invalid position {cmd}"
    except ValueError:
        pass
    
    return "ERROR: unknown subcommand. Use: add, rm, clear, apply"


def cmd_tfx(session: Session, args: List[str]) -> str:
    """Track FX chain - effects applied during track render.
    
    Usage:
      /tfx                    -> Show track FX chain
      /tfx add <effect> [params] -> Add effect to chain
      /tfx rm <n>             -> Remove effect at position
      /tfx clear              -> Clear entire chain
      /tfx <n> <param>=<val>  -> Modify effect params at position
    
    Track effects are applied when rendering the current track.
    """
    if not args:
        return _format_fx_chain(session.track_fx_chain, "Track")
    
    cmd = args[0].lower()
    
    if cmd == 'add' and len(args) > 1:
        effect_name, params = _parse_fx_args(args[1:])
        if effect_name:
            session.track_fx_chain.append((effect_name, params))
            return f"OK: added '{effect_name}' to track FX chain (position {len(session.track_fx_chain)})"
        return "ERROR: no effect name specified"
    
    elif cmd == 'rm' and len(args) > 1:
        try:
            idx = int(args[1]) - 1
            if 0 <= idx < len(session.track_fx_chain):
                removed = session.track_fx_chain.pop(idx)
                return f"OK: removed '{removed[0]}' from track FX chain"
            return f"ERROR: invalid position {args[1]}"
        except ValueError:
            return "ERROR: position must be a number"
    
    elif cmd == 'clear':
        count = len(session.track_fx_chain)
        session.track_fx_chain.clear()
        return f"OK: cleared {count} effects from track FX chain"
    
    # Modify params at position
    try:
        idx = int(cmd) - 1
        if 0 <= idx < len(session.track_fx_chain):
            fx_name, old_params = session.track_fx_chain[idx]
            _, new_params = _parse_fx_args(args[1:])
            old_params.update(new_params)
            return f"OK: updated '{fx_name}' params: {old_params}"
        return f"ERROR: invalid position {cmd}"
    except ValueError:
        pass
    
    return "ERROR: unknown subcommand. Use: add, rm, clear"


def cmd_mfx(session: Session, args: List[str]) -> str:
    """Master FX chain - effects applied to final mix.
    
    Usage:
      /mfx                    -> Show master FX chain
      /mfx add <effect> [params] -> Add effect to chain
      /mfx rm <n>             -> Remove effect at position
      /mfx clear              -> Clear entire chain
      /mfx <n> <param>=<val>  -> Modify effect params at position
    
    Master effects are applied when building the final mix.
    """
    if not args:
        return _format_fx_chain(session.master_fx_chain, "Master")
    
    cmd = args[0].lower()
    
    if cmd == 'add' and len(args) > 1:
        effect_name, params = _parse_fx_args(args[1:])
        if effect_name:
            session.master_fx_chain.append((effect_name, params))
            return f"OK: added '{effect_name}' to master FX chain (position {len(session.master_fx_chain)})"
        return "ERROR: no effect name specified"
    
    elif cmd == 'rm' and len(args) > 1:
        try:
            idx = int(args[1]) - 1
            if 0 <= idx < len(session.master_fx_chain):
                removed = session.master_fx_chain.pop(idx)
                return f"OK: removed '{removed[0]}' from master FX chain"
            return f"ERROR: invalid position {args[1]}"
        except ValueError:
            return "ERROR: position must be a number"
    
    elif cmd == 'clear':
        count = len(session.master_fx_chain)
        session.master_fx_chain.clear()
        return f"OK: cleared {count} effects from master FX chain"
    
    # Modify params at position
    try:
        idx = int(cmd) - 1
        if 0 <= idx < len(session.master_fx_chain):
            fx_name, old_params = session.master_fx_chain[idx]
            _, new_params = _parse_fx_args(args[1:])
            old_params.update(new_params)
            return f"OK: updated '{fx_name}' params: {old_params}"
        return f"ERROR: invalid position {cmd}"
    except ValueError:
        pass
    
    return "ERROR: unknown subcommand. Use: add, rm, clear"


def cmd_ffx(session: Session, args: List[str]) -> str:
    """File FX chain - effects applied when processing files.
    
    Usage:
      /ffx                    -> Show file FX chain
      /ffx add <effect> [params] -> Add effect to chain
      /ffx rm <n>             -> Remove effect at position
      /ffx clear              -> Clear entire chain
      /ffx apply [path]       -> Apply chain to file (or last loaded)
      /ffx <n> <param>=<val>  -> Modify effect params at position
    
    File effects are applied when processing external audio files.
    """
    if not args:
        return _format_fx_chain(session.file_fx_chain, "File")
    
    cmd = args[0].lower()
    
    if cmd == 'add' and len(args) > 1:
        effect_name, params = _parse_fx_args(args[1:])
        if effect_name:
            session.file_fx_chain.append((effect_name, params))
            return f"OK: added '{effect_name}' to file FX chain (position {len(session.file_fx_chain)})"
        return "ERROR: no effect name specified"
    
    elif cmd == 'rm' and len(args) > 1:
        try:
            idx = int(args[1]) - 1
            if 0 <= idx < len(session.file_fx_chain):
                removed = session.file_fx_chain.pop(idx)
                return f"OK: removed '{removed[0]}' from file FX chain"
            return f"ERROR: invalid position {args[1]}"
        except ValueError:
            return "ERROR: position must be a number"
    
    elif cmd == 'clear':
        count = len(session.file_fx_chain)
        session.file_fx_chain.clear()
        return f"OK: cleared {count} effects from file FX chain"
    
    elif cmd == 'apply':
        if session.last_buffer is None:
            return "ERROR: no buffer to process"
        if not session.file_fx_chain:
            return "ERROR: file FX chain is empty"
        
        from ..dsp.effects import apply_effects_with_params
        try:
            # Unpack tuples into separate lists
            effect_names = [fx[0] for fx in session.file_fx_chain]
            effect_params = [fx[1] for fx in session.file_fx_chain]
            session.last_buffer = apply_effects_with_params(
                session.last_buffer,
                effect_names,
                effect_params
            )
            return f"OK: applied {len(session.file_fx_chain)} effects to buffer"
        except Exception as e:
            return f"ERROR: {e}"
    
    # Modify params at position
    try:
        idx = int(cmd) - 1
        if 0 <= idx < len(session.file_fx_chain):
            fx_name, old_params = session.file_fx_chain[idx]
            _, new_params = _parse_fx_args(args[1:])
            old_params.update(new_params)
            return f"OK: updated '{fx_name}' params: {old_params}"
        return f"ERROR: invalid position {cmd}"
    except ValueError:
        pass
    
    return "ERROR: unknown subcommand. Use: add, rm, clear, apply"


# ============================================================================
# FOREVER COMPRESSION COMMAND (Section C2)
# ============================================================================

def cmd_fc(session: Session, args: List[str]) -> str:
    """Forever Compression - Multiband OTT-style compressor.
    
    Usage:
      /fc                     -> Show current FC settings
      /fc <preset>            -> Apply preset to buffer
      /fc <depth> [params]    -> Apply with custom settings
      
    Presets:
      punch  - Punchy, transient-focused (drums)
      glue   - Gentle cohesion (bus/group)
      loud   - Aggressive maximizer (EDM)
      soft   - Subtle dynamics (vocals, acoustic)
      ott    - Classic OTT 50% mix
    
    Parameters (all 0-100):
      depth=<n>     Overall compression depth
      low=<n>       Low band amount
      mid=<n>       Mid band amount  
      high=<n>      High band amount
      up=<n>        Upward compression strength
      down=<n>      Downward compression strength
      mix=<n>       Wet/dry mix
      out=<n>       Output level (50=unity)
    
    Crossovers (Hz):
      lowx=<hz>     Low/mid crossover (default 120)
      highx=<hz>    Mid/high crossover (default 2500)
    
    Examples:
      /fc punch              -> Apply punch preset
      /fc 75                 -> Apply at 75% depth
      /fc 60 low=80 high=40  -> Custom band amounts
      /fc loud mix=50        -> Loud preset at 50% wet
    """
    from ..dsp.effects import forever_compression
    
    # Default settings
    params = {
        'depth': 50.0,
        'low_xover': 120.0,
        'high_xover': 2500.0,
        'low_amount': 50.0,
        'mid_amount': 50.0,
        'high_amount': 50.0,
        'upward': 50.0,
        'downward': 50.0,
        'mix': 100.0,
        'output': 50.0,
    }
    
    # Preset definitions
    presets = {
        'punch': {'depth': 60, 'low_amount': 70, 'mid_amount': 50, 'high_amount': 40, 
                  'upward': 30, 'downward': 70, 'mix': 75},
        'glue': {'depth': 35, 'low_amount': 45, 'mid_amount': 50, 'high_amount': 45,
                 'upward': 40, 'downward': 40, 'mix': 60},
        'loud': {'depth': 85, 'low_amount': 80, 'mid_amount': 90, 'high_amount': 85,
                 'upward': 70, 'downward': 80, 'mix': 100, 'output': 60},
        'soft': {'depth': 25, 'low_amount': 30, 'mid_amount': 35, 'high_amount': 25,
                 'upward': 50, 'downward': 30, 'mix': 50},
        'ott': {'depth': 100, 'low_amount': 100, 'mid_amount': 100, 'high_amount': 100,
                'upward': 100, 'downward': 100, 'mix': 50},
    }
    
    # Param aliases
    param_aliases = {
        'low': 'low_amount', 'mid': 'mid_amount', 'high': 'high_amount',
        'up': 'upward', 'down': 'downward', 'out': 'output',
        'lowx': 'low_xover', 'highx': 'high_xover',
    }
    
    if not args:
        # Show help/status
        lines = [
            "=== FOREVER COMPRESSION ===",
            "Multiband OTT-style compressor (Section C2)",
            "",
            "Presets: punch, glue, loud, soft, ott",
            "",
            "Usage: /fc <preset>",
            "       /fc <depth> [low=n] [mid=n] [high=n] [up=n] [down=n] [mix=n]",
            "",
            "Example: /fc punch",
            "         /fc 75 low=80 high=40 mix=75",
        ]
        return '\n'.join(lines)
    
    # Check for preset
    first_arg = args[0].lower()
    if first_arg in presets:
        params.update(presets[first_arg])
        preset_name = first_arg
        args = args[1:]  # Allow additional params to override preset
    else:
        preset_name = None
        # Try parsing first arg as depth
        try:
            params['depth'] = float(first_arg)
            args = args[1:]
        except ValueError:
            pass  # Not a number, will be parsed as param
    
    # Parse remaining param=value arguments
    for arg in args:
        if '=' in arg:
            key, val = arg.split('=', 1)
            key = key.lower()
            # Apply alias
            if key in param_aliases:
                key = param_aliases[key]
            if key in params:
                try:
                    params[key] = float(val)
                except ValueError:
                    return f"ERROR: invalid value for {key}: {val}"
            else:
                return f"ERROR: unknown parameter: {key}"
    
    # Check for buffer
    if session.last_buffer is None or len(session.last_buffer) == 0:
        return "ERROR: no buffer loaded. Use /tone, /g, or /upl first"
    
    # Apply compression
    try:
        session.last_buffer = forever_compression(
            session.last_buffer,
            depth=params['depth'],
            low_xover=params['low_xover'],
            high_xover=params['high_xover'],
            low_amount=params['low_amount'],
            mid_amount=params['mid_amount'],
            high_amount=params['high_amount'],
            upward=params['upward'],
            downward=params['downward'],
            mix=params['mix'],
            output=params['output'],
        )
        
        # Build response
        if preset_name:
            resp = f"OK: Forever Compression [{preset_name}]"
        else:
            resp = f"OK: Forever Compression"
        
        details = []
        if params['depth'] != 50:
            details.append(f"depth={params['depth']:.0f}")
        if params['mix'] != 100:
            details.append(f"mix={params['mix']:.0f}")
        if params['low_amount'] != 50 or params['mid_amount'] != 50 or params['high_amount'] != 50:
            details.append(f"L/M/H={params['low_amount']:.0f}/{params['mid_amount']:.0f}/{params['high_amount']:.0f}")
        
        if details:
            resp += f" ({', '.join(details)})"
        
        return resp
        
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================================
# GIGA GATE COMMAND (Section C2)
# ============================================================================

def cmd_gg(session: Session, args: List[str]) -> str:
    """Giga Gate - Pattern-based gating and stutter engine.
    
    Usage:
      /gg                      -> Show help and patterns
      /gg <pattern>            -> Apply gate pattern
      /gg stutter <n>          -> Stutter effect (n repeats)
      /gg half                 -> Halftime effect
      /gg double               -> Doubletime effect  
      /gg tape stop [dur]      -> Tape stop effect
      /gg tape bend [amt]      -> Tape wobble effect
    
    Gate Patterns (binary or named):
      Binary: 1010101010101010 (1=on, 0=off)
      Shorthand: x.x.x.x. (x=on, .=off)
      
    Named patterns:
      half        - Alternating (1010...)
      quarter     - Every 4th (1000...)
      four_floor  - 4-on-floor kick pattern
      offbeat     - Offbeat pattern
      tresillo    - Tresillo rhythm
      son_clave   - Son clave
      rumba_clave - Rumba clave  
      bossa       - Bossa nova
      shuffle     - Shuffle feel
      glitch1/2   - Glitchy patterns
      sparse      - Minimal hits
      dense       - Busy pattern
    
    Parameters:
      steps=<n>    Number of steps (default 16)
      shape=<s>    Gate shape: square, saw, ramp, triangle, sine, exp
      atk=<ms>     Attack time in ms
      rel=<ms>     Release time in ms
      mix=<n>      Wet/dry mix (0-100)
    
    Examples:
      /gg half                  -> Apply half pattern
      /gg 1010110010100110      -> Custom binary pattern
      /gg tresillo shape=sine   -> Tresillo with sine shape
      /gg stutter 8             -> 8x stutter
      /gg tape stop 0.5         -> 0.5s tape stop
    """
    from ..dsp.effects import (
        giga_gate, giga_stutter, giga_halftime, giga_doubletime,
        giga_tape_stop, giga_tape_bend
    )
    
    if not args:
        lines = [
            "=== GIGA GATE ===",
            "Pattern-based gating and stutter engine (Section C2)",
            "",
            "Patterns: half, quarter, four_floor, offbeat, tresillo,",
            "          son_clave, rumba_clave, bossa, shuffle, glitch1,",
            "          glitch2, sparse, dense, stutter2/4/8",
            "",
            "Or use binary: /gg 1010101010101010",
            "",
            "Effects:",
            "  /gg stutter <n>     - Repeat first slice n times",
            "  /gg half            - Halftime (1 octave down)", 
            "  /gg double          - Doubletime (1 octave up)",
            "  /gg tape stop [s]   - Tape stop effect",
            "  /gg tape bend [amt] - Tape wobble",
            "",
            "Shapes: square, saw, ramp, triangle, sine, exp",
        ]
        return '\n'.join(lines)
    
    # Check for buffer
    if session.last_buffer is None or len(session.last_buffer) == 0:
        return "ERROR: no buffer loaded. Use /tone, /g, or /upl first"
    
    # Default params
    params = {
        'steps': 16,
        'shape': 'square',
        'attack': 1.0,
        'release': 5.0,
        'mix': 100.0,
    }
    
    # Parse param=value args
    pattern_args = []
    for arg in args:
        if '=' in arg:
            key, val = arg.split('=', 1)
            key = key.lower()
            if key == 'steps':
                params['steps'] = int(val)
            elif key == 'shape':
                params['shape'] = val.lower()
            elif key in ('atk', 'attack'):
                params['attack'] = float(val)
            elif key in ('rel', 'release'):
                params['release'] = float(val)
            elif key == 'mix':
                params['mix'] = float(val)
        else:
            pattern_args.append(arg)
    
    if not pattern_args:
        return "ERROR: no pattern or command specified"
    
    first = pattern_args[0].lower()
    
    try:
        # Special commands
        if first == 'stutter':
            repeats = int(pattern_args[1]) if len(pattern_args) > 1 else 4
            decay = float(pattern_args[2]) if len(pattern_args) > 2 else 0.85
            session.last_buffer = giga_stutter(
                session.last_buffer, repeats=repeats, decay=decay, mix=params['mix']
            )
            return f"OK: Giga Gate stutter (repeats={repeats}, decay={decay:.2f})"
        
        elif first == 'half' and len(pattern_args) == 1:
            session.last_buffer = giga_halftime(session.last_buffer, mix=params['mix'])
            return "OK: Giga Gate halftime"
        
        elif first == 'double':
            session.last_buffer = giga_doubletime(session.last_buffer, mix=params['mix'])
            return "OK: Giga Gate doubletime"
        
        elif first == 'tape':
            if len(pattern_args) < 2:
                return "ERROR: specify 'tape stop' or 'tape bend'"
            
            sub = pattern_args[1].lower()
            if sub == 'stop':
                dur = float(pattern_args[2]) if len(pattern_args) > 2 else 0.5
                session.last_buffer = giga_tape_stop(
                    session.last_buffer, duration=dur, mix=params['mix']
                )
                return f"OK: Giga Gate tape stop ({dur:.2f}s)"
            
            elif sub == 'bend':
                amt = float(pattern_args[2]) if len(pattern_args) > 2 else 50.0
                rate = float(pattern_args[3]) if len(pattern_args) > 3 else 2.0
                session.last_buffer = giga_tape_bend(
                    session.last_buffer, amount=amt, rate=rate, mix=params['mix']
                )
                return f"OK: Giga Gate tape bend (amt={amt:.0f}, rate={rate:.1f}Hz)"
            
            else:
                return f"ERROR: unknown tape command: {sub}"
        
        else:
            # Treat as gate pattern
            pattern = pattern_args[0]
            session.last_buffer = giga_gate(
                session.last_buffer,
                pattern=pattern,
                steps=params['steps'],
                shape=params['shape'],
                attack=params['attack'],
                release=params['release'],
                mix=params['mix'],
            )
            return f"OK: Giga Gate [{pattern}] (steps={params['steps']}, shape={params['shape']})"
    
    except Exception as e:
        return f"ERROR: {e}"


def cmd_fxall(session: Session, args: List[str]) -> str:
    """Show all FX chains.
    
    Usage:
      /fxall             -> Show all FX chains
    """
    lines = ["=== ALL FX CHAINS ===", ""]
    
    lines.append(_format_fx_chain(session.buffer_fx_chain, "BFX (Buffer)"))
    lines.append("")
    lines.append(_format_fx_chain(session.track_fx_chain, "TFX (Track)"))
    lines.append("")
    lines.append(_format_fx_chain(session.master_fx_chain, "MFX (Master)"))
    lines.append("")
    lines.append(_format_fx_chain(session.file_fx_chain, "FFX (File)"))
    
    total = (len(session.buffer_fx_chain) + len(session.track_fx_chain) + 
             len(session.master_fx_chain) + len(session.file_fx_chain))
    lines.append(f"\nTotal effects across all chains: {total}")
    
    return '\n'.join(lines)


# Quick effect commands - apply effect directly to last_buffer
def cmd_reverb(session: Session, args: List[str]) -> str:
    """Quick reverb. Usage: /reverb [mix] [size]"""
    return cmd_fx(session, ['reverb'] + list(args))

def cmd_delay(session: Session, args: List[str]) -> str:
    """Quick delay. Usage: /delay [time] [feedback]"""
    return cmd_fx(session, ['delay'] + list(args))

def cmd_compress(session: Session, args: List[str]) -> str:
    """Quick compression. Usage: /compress [threshold] [ratio]"""
    return cmd_fx(session, ['compress'] + list(args))

def cmd_distort(session: Session, args: List[str]) -> str:
    """Quick distortion. Usage: /distort [amount]"""
    return cmd_fx(session, ['distort'] + list(args))

def cmd_chorus(session: Session, args: List[str]) -> str:
    """Quick chorus. Usage: /chorus"""
    return cmd_fx(session, ['chorus'] + list(args))

def cmd_flanger(session: Session, args: List[str]) -> str:
    """Quick flanger. Usage: /flanger"""
    return cmd_fx(session, ['flanger'] + list(args))

def cmd_phaser(session: Session, args: List[str]) -> str:
    """Quick phaser. Usage: /phaser"""
    return cmd_fx(session, ['phaser'] + list(args))


def get_fxchain_commands() -> dict:
    """Return FX chain position commands for registration."""
    return {
        'bfx': cmd_bfx,
        'tfx': cmd_tfx,
        'mfx': cmd_mfx,
        'ffx': cmd_ffx,
        'fxall': cmd_fxall,
    }


def get_fx_commands() -> dict:
    """Return all FX commands for registration."""
    return {
        # Main fx command
        'fx': cmd_fx,
        'hfx': cmd_hfx,
        
        # Quick effects
        'reverb': cmd_reverb,
        'delay': cmd_delay,
        'compress': cmd_compress,
        'distort': cmd_distort,
        'chorus': cmd_chorus,
        'flanger': cmd_flanger,
        'phaser': cmd_phaser,
        
        # FX chains
        'bfx': cmd_bfx,
        'tfx': cmd_tfx,
        'mfx': cmd_mfx,
        'ffx': cmd_ffx,
        'fxall': cmd_fxall,
        
        # Apply command (shortcut) - /ap since /p conflicts with play
        'ap': cmd_apply,
        'apply': cmd_apply,
        'fxap': cmd_apply,
        
        # Named effect commands
        'fxa': cmd_fxa,
        'fxs': cmd_fxs,
        'fxr': cmd_fxr,
        'fxm': cmd_fxm,
        'fxp': cmd_fxp,
        'fxc': cmd_fxc,
        'fxl': cmd_fxl,
        'fxq': cmd_fxq,
        
        # Vamp/saturation
        'vamp': cmd_vamp,
        'dual': cmd_dual,
        'conv': cmd_conv,
        
        # Dynamics
        'fc': cmd_fc,
        'gg': cmd_gg,
    }



def cmd_apply(session: "Session", args: List[str]) -> str:
    """Apply the current FX chain to the buffer.
    
    Usage:
      /p                 Apply effects chain to buffer
      /p clear           Apply chain then clear it
      /apply             Same as /p
    
    This applies all effects added via /fx to the current audio buffer.
    
    Examples:
      /fx add reverb     Add reverb to chain
      /fx add delay      Add delay to chain
      /p                 Apply both effects to buffer
    """
    if session.last_buffer is None:
        return "ERROR: no audio in buffer. Generate audio first."
    
    # If an effect name is given (not 'clear'), apply it directly
    if args and args[0].lower() != 'clear':
        from ..dsp.effects import resolve_effect_name, apply_effects_with_params
        effect_name, err = resolve_effect_name(args[0].lower())
        if err:
            return f"ERROR: unknown effect '{args[0]}'. {err}"
        import numpy as np
        before_peak = np.max(np.abs(session.last_buffer))
        try:
            session.last_buffer = apply_effects_with_params(
                session.last_buffer, [effect_name], [{}]
            )
            after_peak = np.max(np.abs(session.last_buffer))
            return f"OK: applied {effect_name} (peak {before_peak:.3f} -> {after_peak:.3f})"
        except Exception as e:
            return f"ERROR: {e}"
    
    # Check session.effects (used by /fx add)
    if not session.effects:
        return "ERROR: FX chain is empty. Add effects with /fx add <n>"
    
    # Get before metrics
    import numpy as np
    before_peak = np.max(np.abs(session.last_buffer))
    before_rms = np.sqrt(np.mean(session.last_buffer ** 2))
    before_samples = len(session.last_buffer)
    
    from ..dsp.effects import apply_effects_with_params
    try:
        # Use session.effects and session.effect_params
        effect_names = list(session.effects)
        effect_params = list(session.effect_params)
        
        session.last_buffer = apply_effects_with_params(
            session.last_buffer, 
            effect_names,
            effect_params
        )
        
        # Update working buffer
        try:
            from .working_cmds import get_working_buffer
            wb = get_working_buffer()
            fx_str = "+".join(effect_names[:3])
            if len(effect_names) > 3:
                fx_str += f"+{len(effect_names)-3}more"
            wb.set_pending(session.last_buffer, f"fx:{fx_str}", session)
        except:
            pass
        
        # Get after metrics
        after_peak = np.max(np.abs(session.last_buffer))
        after_rms = np.sqrt(np.mean(session.last_buffer ** 2))
        after_samples = len(session.last_buffer)
        
        # Calculate deviation
        peak_change = 20 * np.log10(after_peak / before_peak) if before_peak > 0 else 0
        rms_change = 20 * np.log10(after_rms / before_rms) if before_rms > 0 else 0
        length_change = after_samples - before_samples
        
        # Format effect list
        fx_list = ", ".join(effect_names)
        
        # Clear chain if requested
        clear_note = ""
        if args and args[0].lower() == 'clear':
            session.clear_effects()
            clear_note = " (chain cleared)"
        
        lines = [f"OK: Applied {len(effect_names)} effect(s): {fx_list}{clear_note}"]
        lines.append(f"  Peak: {before_peak:.3f} -> {after_peak:.3f} ({peak_change:+.1f} dB)")
        lines.append(f"  RMS:  {before_rms:.3f} -> {after_rms:.3f} ({rms_change:+.1f} dB)")
        if length_change != 0:
            lines.append(f"  Length: {before_samples} -> {after_samples} ({length_change:+d} samples)")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"ERROR: {e}"
