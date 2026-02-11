"""MDMA Unified Parameter Scaling System.

This module defines the canonical 1-100 scaling standard for MDMA.
All abstract parameters (amounts, depths, drives, mixes, etc.) should
use this system for consistency and predictability.

SCALING RULES:
--------------
1. Range 1-100 is the "normal" operating range
2. 100 = general maximum for clean/intended audio
3. Values > 100 are ALLOWED but enter "wacky/scuffed" territory
4. 0 = off/minimum (where applicable)

WHAT USES 1-100 SCALING:
------------------------
- Effect amounts/mix/wet-dry
- Drive/saturation amounts
- Modulation depths
- Voice algorithm params (detune spread, random, mod scaling)
- Compression ratios, thresholds (as abstract amounts)
- Gate thresholds (as amounts)
- LFO depths
- Feedback amounts
- Resonance/Q (as abstract "intensity")
- Any "intensity" or "amount" parameter

WHAT KEEPS REAL UNITS:
----------------------
- Frequency (Hz) - filter cutoff, LFO rate, oscillator freq
- Time (ms, seconds) - attack, decay, delay time, etc.
- Sample rate, BPM
- Semitones, octaves
- Pan position (-1 to +1 or -100 to +100)

BUILD ID: scaling_v1_chunk1.1
"""

from __future__ import annotations

import numpy as np
from typing import Union, Optional

# ============================================================================
# CORE SCALING CONSTANTS
# ============================================================================

# The standard range
PARAM_MIN = 0.0
PARAM_NORMAL_MAX = 100.0
PARAM_ABSOLUTE_MAX = 200.0  # Hard cap even for "wacky" mode

# Threshold where we warn about entering wacky territory
PARAM_WACKY_THRESHOLD = 100.0


# ============================================================================
# PRESET VALUES (for named presets)
# ============================================================================

PARAM_PRESETS = {
    # Intensity levels
    'off': 0,
    'zero': 0,
    'none': 0,
    'min': 1,
    'minimum': 1,
    'subtle': 15,
    'light': 25,
    'low': 25,
    'gentle': 30,
    'mild': 35,
    'moderate': 50,
    'medium': 50,
    'half': 50,
    'default': 50,
    'strong': 70,
    'high': 75,
    'heavy': 85,
    'intense': 90,
    'full': 100,
    'max': 100,
    'maximum': 100,
    'extreme': 100,
    # Wacky presets (explicitly beyond normal)
    'wacky': 120,
    'crazy': 140,
    'insane': 175,
    'broken': 200,
}


# ============================================================================
# CORE PARSING FUNCTIONS
# ============================================================================

def parse_param(value: Union[str, int, float], default: float = 50.0) -> float:
    """Parse a parameter value from user input.
    
    Accepts:
    - Numeric values (int, float, or string representation)
    - Preset names ('subtle', 'heavy', 'max', etc.)
    
    Parameters
    ----------
    value : str, int, or float
        The input value to parse
    default : float
        Default value if parsing fails (default: 50.0)
    
    Returns
    -------
    float
        Parameter value (clamped to 0-200 range)
    
    Examples
    --------
    >>> parse_param(75)
    75.0
    >>> parse_param("heavy")
    85.0
    >>> parse_param("120")  # Wacky but allowed
    120.0
    >>> parse_param("garbage")
    50.0
    """
    if isinstance(value, (int, float)):
        return clamp_param(float(value))
    
    if isinstance(value, str):
        # Check preset names first
        lower_val = value.lower().strip()
        if lower_val in PARAM_PRESETS:
            return float(PARAM_PRESETS[lower_val])
        
        # Try parsing as number
        try:
            return clamp_param(float(value))
        except ValueError:
            pass
    
    return default


def clamp_param(value: float, allow_wacky: bool = True) -> float:
    """Clamp a parameter to valid range.
    
    Parameters
    ----------
    value : float
        Input value
    allow_wacky : bool
        If True, allows values up to 200. If False, clamps at 100.
    
    Returns
    -------
    float
        Clamped value
    """
    max_val = PARAM_ABSOLUTE_MAX if allow_wacky else PARAM_NORMAL_MAX
    return max(PARAM_MIN, min(max_val, float(value)))


def is_wacky(value: float) -> bool:
    """Check if a parameter value is in the 'wacky' range (>100)."""
    return value > PARAM_WACKY_THRESHOLD


def validate_param(value: float, name: str = "parameter") -> tuple[float, Optional[str]]:
    """Validate a parameter and return any warning message.
    
    Parameters
    ----------
    value : float
        The parameter value
    name : str
        Name of the parameter (for warning message)
    
    Returns
    -------
    tuple[float, Optional[str]]
        The clamped value and optional warning message
    """
    clamped = clamp_param(value)
    warning = None
    
    if value > PARAM_ABSOLUTE_MAX:
        warning = f"WARNING: {name}={value:.1f} clamped to {PARAM_ABSOLUTE_MAX:.0f} (absolute max)"
    elif is_wacky(clamped):
        warning = f"NOTE: {name}={clamped:.1f} is in wacky range (>100) - expect scuffed audio"
    
    return clamped, warning


# ============================================================================
# SCALING FUNCTIONS (1-100 to internal ranges)
# ============================================================================

def scale_to_range(value: float, min_out: float, max_out: float) -> float:
    """Scale a 0-100 value to an arbitrary output range.
    
    Parameters
    ----------
    value : float
        Input value (0-100, or higher for wacky)
    min_out : float
        Minimum output value
    max_out : float
        Maximum output value (at input=100)
    
    Returns
    -------
    float
        Scaled value
    
    Notes
    -----
    For values > 100, continues scaling linearly beyond max_out.
    """
    # Normalize to 0-1 (with extension for wacky)
    normalized = value / 100.0
    return min_out + normalized * (max_out - min_out)


def scale_drive(amount: float) -> float:
    """Scale 0-100 amount to drive multiplier (1.0-20.0).
    
    At 100: drive = 20.0
    Above 100: continues scaling (wacky territory)
    """
    return scale_to_range(amount, 1.0, 20.0)


def scale_wet(amount: float) -> float:
    """Scale 0-100 amount to wet mix (0.0-1.0).
    
    Clamped at 1.0 even for wacky values (100% wet is max).
    """
    return min(1.0, amount / 100.0)


def scale_dry(amount: float) -> float:
    """Scale 0-100 amount to dry mix (inverse of wet)."""
    return 1.0 - scale_wet(amount)


def scale_feedback(amount: float) -> float:
    """Scale 0-100 amount to feedback coefficient (0.0-0.95).
    
    At 100: feedback = 0.95 (prevents runaway)
    Above 100: caps at 0.99 (very dangerous but allowed)
    """
    if amount <= 100:
        return amount / 100.0 * 0.95
    else:
        # Wacky: allow up to 0.99
        extra = (amount - 100) / 100.0 * 0.04  # additional 0-0.04
        return min(0.99, 0.95 + extra)


def scale_resonance(amount: float) -> float:
    """Scale 0-100 amount to resonance/Q (0.1-20.0).
    
    This is for abstract "resonance intensity", not raw Q values.
    """
    # Logarithmic scaling for more usable range
    if amount <= 0:
        return 0.1
    normalized = amount / 100.0
    # Map 0-1 to 0.1-20 with log curve for better control in lower range
    return 0.1 + (normalized ** 1.5) * 19.9


def scale_threshold_db(amount: float) -> float:
    """Scale 0-100 amount to threshold in dB (-60 to 0).
    
    0 = -60dB (very quiet threshold)
    100 = 0dB (max threshold)
    """
    return scale_to_range(amount, -60.0, 0.0)


def scale_ratio(amount: float) -> float:
    """Scale 0-100 amount to compression ratio (1:1 to 20:1).
    
    0 = 1:1 (no compression)
    100 = 20:1 (heavy limiting)
    """
    return scale_to_range(amount, 1.0, 20.0)


def scale_bits(amount: float) -> int:
    """Scale 0-100 amount to bit depth (1-16 bits).
    
    0 = 1 bit (extreme crush)
    100 = 16 bits (clean)
    """
    return int(max(1, min(16, scale_to_range(amount, 1, 16))))


def scale_detune_hz(amount: float, max_hz: float = 10.0) -> float:
    """Scale 0-100 amount to detune in Hz.
    
    Parameters
    ----------
    amount : float
        Detune amount (0-100)
    max_hz : float
        Maximum detune at 100 (default 10 Hz)
    
    Returns
    -------
    float
        Detune in Hz
    """
    return scale_to_range(amount, 0.0, max_hz)


def scale_detune_cents(amount: float, max_cents: float = 50.0) -> float:
    """Scale 0-100 amount to detune in cents.
    
    Parameters
    ----------
    amount : float
        Detune amount (0-100)
    max_cents : float
        Maximum detune at 100 (default 50 cents)
    """
    return scale_to_range(amount, 0.0, max_cents)


def scale_pan(amount: float) -> float:
    """Scale 0-100 amount to stereo pan position (-1.0 to +1.0).
    
    0 = full left (-1.0)
    50 = center (0.0)
    100 = full right (+1.0)
    """
    return (amount - 50.0) / 50.0


def scale_spread(amount: float) -> float:
    """Scale 0-100 amount to stereo spread (0.0-1.0).
    
    0 = mono
    100 = full stereo width
    """
    return min(1.0, amount / 100.0)


def scale_modulation_index(amount: float, max_index: float = 10.0) -> float:
    """Scale 0-100 amount to modulation index (0 to max_index).
    
    Used for FM/AM/PM modulation depth.
    """
    return scale_to_range(amount, 0.0, max_index)


def scale_attack_release_mult(amount: float) -> float:
    """Scale 0-100 to attack/release time multiplier (0.1x to 10x).
    
    Used for envelope time scaling.
    50 = 1x (no change)
    0 = 0.1x (10% of original)
    100 = 10x (10x original)
    """
    if amount <= 50:
        # 0-50 maps to 0.1-1.0
        return 0.1 + (amount / 50.0) * 0.9
    else:
        # 50-100 maps to 1.0-10.0
        return 1.0 + ((amount - 50) / 50.0) * 9.0


# ============================================================================
# INVERSE SCALING (internal values back to 1-100)
# ============================================================================

def unscale_from_range(value: float, min_in: float, max_in: float) -> float:
    """Convert an internal value back to 0-100 range.
    
    Parameters
    ----------
    value : float
        Internal value
    min_in : float
        Minimum internal value (maps to 0)
    max_in : float
        Maximum internal value (maps to 100)
    
    Returns
    -------
    float
        Value in 0-100 range
    """
    if max_in == min_in:
        return 50.0
    return (value - min_in) / (max_in - min_in) * 100.0


def unscale_drive(drive: float) -> float:
    """Convert drive multiplier (1-20) back to 0-100."""
    return unscale_from_range(drive, 1.0, 20.0)


def unscale_wet(wet: float) -> float:
    """Convert wet mix (0-1) back to 0-100."""
    return wet * 100.0


def unscale_feedback(fb: float) -> float:
    """Convert feedback coefficient back to 0-100."""
    if fb <= 0.95:
        return fb / 0.95 * 100.0
    else:
        return 100.0 + (fb - 0.95) / 0.04 * 100.0


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def format_param(value: float, name: str = "", include_warning: bool = True) -> str:
    """Format a parameter value for display.
    
    Parameters
    ----------
    value : float
        Parameter value
    name : str
        Parameter name (optional)
    include_warning : bool
        Whether to include wacky warning
    
    Returns
    -------
    str
        Formatted string for display
    """
    prefix = f"{name}=" if name else ""
    wacky_suffix = " [WACKY]" if is_wacky(value) and include_warning else ""
    return f"{prefix}{value:.1f}{wacky_suffix}"


def get_preset_name(value: float, tolerance: float = 2.0) -> Optional[str]:
    """Get the preset name for a value if it matches one.
    
    Parameters
    ----------
    value : float
        Parameter value
    tolerance : float
        How close the value needs to be to match (default: 2.0)
    
    Returns
    -------
    Optional[str]
        Preset name or None if no match
    """
    for name, preset_val in PARAM_PRESETS.items():
        if abs(value - preset_val) <= tolerance:
            return name
    return None


# ============================================================================
# BATCH OPERATIONS
# ============================================================================

def parse_params(values: list, defaults: list = None) -> list[float]:
    """Parse multiple parameter values at once.
    
    Parameters
    ----------
    values : list
        List of values to parse
    defaults : list, optional
        List of default values (same length as values)
    
    Returns
    -------
    list[float]
        Parsed parameter values
    """
    if defaults is None:
        defaults = [50.0] * len(values)
    
    return [
        parse_param(v, d) 
        for v, d in zip(values, defaults)
    ]


def validate_params(params: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Validate a dictionary of parameters.
    
    Parameters
    ----------
    params : dict
        Dictionary of parameter_name -> value
    
    Returns
    -------
    tuple[dict, list]
        Validated params dict and list of warning messages
    """
    validated = {}
    warnings = []
    
    for name, value in params.items():
        clamped, warning = validate_param(value, name)
        validated[name] = clamped
        if warning:
            warnings.append(warning)
    
    return validated, warnings
