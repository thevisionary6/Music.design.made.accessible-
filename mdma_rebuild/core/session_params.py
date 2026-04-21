"""Session parameter accessors.

Extracted from ``core/session.py`` during the backend V2 repo-prep. Owns
the long tail of filter / envelope / voice / operator setter-and-getter
methods that used to live on Session directly. Every function takes
``self`` (a Session) as its first argument and is re-bound onto the
Session class at the bottom of ``core/session.py`` via :func:`bind_to`.

Behaviour is unchanged; this is purely a code-location move. Two entries
(``resonance`` and ``cutoff``) were ``@property`` on Session and stay
that way — :func:`bind_to` wraps them in :class:`property` before
assigning.

BUILD ID: session_params_v1 (extracted from session_v14.2_chunk3)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..dsp.scaling import parse_param, validate_param

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .session import Session


# ------------ Filter management methods ------------

def set_filter_count(self: "Session", n: int) -> None:
    """Set the number of filter slots available."""
    n = max(1, min(8, int(n)))  # Clamp between 1-8
    self.filter_count = n
    # Initialize any new slots with defaults
    for i in range(n):
        if i not in self.filter_types:
            self.filter_types[i] = 0  # lowpass
            self.filter_cutoffs[i] = 1000.0
            self.filter_resonances[i] = 50.0
            self.filter_enabled[i] = False  # New slots disabled by default
        if i not in self.filter_envelopes:
            self.filter_envelopes[i] = {
                'attack': 0.01, 'decay': 0.1, 'sustain': 1.0, 'release': 0.1
            }
    # Clamp selected filter to valid range
    if self.selected_filter >= n:
        self.selected_filter = n - 1
    if self.selected_filter_envelope >= n:
        self.selected_filter_envelope = n - 1


def set_filter_type(self: "Session", type_spec) -> None:
    """Set the filter type for the currently selected filter slot.

    Parameters
    ----------
    type_spec : str | int
        Filter type as index (0-29), alias ('lp', 'hp', etc.), full name,
        or special values: 'none', 'off', '-1' to disable filter.
    """
    slot = self.selected_filter

    # Handle "none"/"off" values to disable filter
    if isinstance(type_spec, str):
        lower_spec = type_spec.lower().strip()
        if lower_spec in ('none', 'off', 'disable', 'disabled', '-1', 'bypass'):
            self.filter_enabled[slot] = False
            return

    # Handle -1 as disable
    if isinstance(type_spec, int) and type_spec == -1:
        self.filter_enabled[slot] = False
        return

    # Resolve to index
    if isinstance(type_spec, int):
        idx = type_spec
    else:
        # Check aliases first (handles cases like '303' which is an alias for acid)
        lower_spec = type_spec.lower()
        if lower_spec in self.filter_type_aliases:
            idx = self.filter_type_aliases[lower_spec]
        elif type_spec.isdigit() or (type_spec.startswith('-') and type_spec[1:].isdigit()):
            idx = int(type_spec)
            if idx == -1:
                self.filter_enabled[slot] = False
                return
        else:
            # List valid options in error
            valid_aliases = sorted(set(self.filter_type_aliases.keys()))[:15]
            raise ValueError(f"unknown filter type: {type_spec}\n"
                           f"  Valid: 0-29, {', '.join(valid_aliases)}, none, off")
    if idx < 0 or idx > 29:
        raise ValueError(f"filter type index must be 0-29 (or -1/none/off to disable), got {idx}")
    self.filter_types[slot] = idx
    self.filter_enabled[slot] = True  # Enable filter when type is set


def select_filter(self: "Session", idx: int) -> None:
    """Select a filter slot for editing."""
    if idx < 0 or idx >= self.filter_count:
        raise ValueError(f"filter index must be 0-{self.filter_count - 1}")
    self.selected_filter = idx


def select_filter_envelope(self: "Session", idx: int) -> None:
    """Select a filter envelope slot for editing."""
    if idx < 0 or idx >= self.filter_count:
        raise ValueError(f"filter envelope index must be 0-{self.filter_count - 1}")
    self.selected_filter_envelope = idx


def get_current_filter_settings(self: "Session") -> tuple:
    """Get settings for the currently selected filter slot.

    Returns (type_index, cutoff, resonance, enabled)
    """
    slot = self.selected_filter
    return (
        self.filter_types.get(slot, 0),
        self.filter_cutoffs.get(slot, 1000.0),
        self.filter_resonances.get(slot, 0.5),
        self.filter_enabled.get(slot, False)
    )


def set_cutoff(self: "Session", freq: float) -> None:
    """Set cutoff frequency for the currently selected filter.

    Parameters
    ----------
    freq : float
        Cutoff frequency in Hz (real units, 20-20000)
    """
    self.filter_cutoffs[self.selected_filter] = max(20.0, min(20000.0, float(freq)))


def set_resonance(self: "Session", value: float) -> None:
    """Set resonance for the currently selected filter.

    Parameters
    ----------
    value : float
        Resonance amount (0-100 scale)
        0 = no resonance
        50 = moderate resonance
        100 = high resonance (near self-oscillation)
        >100 = wacky territory (allowed)
    """
    # Parse accepts numbers or preset names like 'heavy', 'subtle', etc.
    parsed = parse_param(value, default=50.0)
    clamped, warning = validate_param(parsed, "resonance")
    if warning:
        print(f"[session] {warning}")
    self.filter_resonances[self.selected_filter] = clamped


def _resonance_getter(self: "Session") -> float:
    """Get resonance of currently selected filter (0-100 scale)."""
    return self.filter_resonances.get(self.selected_filter, 50.0)


def _cutoff_getter(self: "Session") -> float:
    """Get cutoff of currently selected filter (Hz)."""
    return self.filter_cutoffs.get(self.selected_filter, 1000.0)


def enable_filter(self: "Session", enabled: bool = True) -> None:
    """Enable or disable the currently selected filter."""
    self.filter_enabled[self.selected_filter] = enabled


# ------------ Envelope management methods ------------

def get_envelope_for_operator(self: "Session", op_idx: int) -> dict:
    """Get the envelope for a specific operator, falling back to global."""
    if op_idx in self.operator_envelopes:
        return self.operator_envelopes[op_idx]
    return {
        'attack': self.attack,
        'decay': self.decay,
        'sustain': self.sustain,
        'release': self.release
    }


def set_envelope_param(self: "Session", param: str, value: float) -> None:
    """Set an envelope parameter based on current synth level.

    At synth_level=1 (global): Sets the global envelope.
    At synth_level=2 (operator): Sets the current operator's envelope.
    """
    value = float(value)
    if param == 'sustain':
        value = max(0.0, min(1.0, value))
    else:
        value = max(0.0, value)

    if self.synth_level == 2:
        # Operator level: set per-operator envelope
        op = self.current_operator
        if op not in self.operator_envelopes:
            # Initialize from global
            self.operator_envelopes[op] = {
                'attack': self.attack,
                'decay': self.decay,
                'sustain': self.sustain,
                'release': self.release
            }
        self.operator_envelopes[op][param] = value
    else:
        # Global level
        setattr(self, param, value)


def get_envelope_param(self: "Session", param: str) -> float:
    """Get an envelope parameter based on current synth level."""
    if self.synth_level == 2 and self.current_operator in self.operator_envelopes:
        return self.operator_envelopes[self.current_operator].get(
            param, getattr(self, param)
        )
    return getattr(self, param)


def set_filter_envelope_param(self: "Session", param: str, value: float) -> None:
    """Set a filter envelope parameter for the selected filter envelope slot."""
    value = float(value)
    if param == 'sustain':
        value = max(0.0, min(1.0, value))
    else:
        value = max(0.0, value)
    slot = self.selected_filter_envelope
    if slot not in self.filter_envelopes:
        self.filter_envelopes[slot] = {
            'attack': 0.01, 'decay': 0.1, 'sustain': 1.0, 'release': 0.1
        }
    self.filter_envelopes[slot][param] = value


def get_filter_envelope(self: "Session", slot: int = None) -> dict:
    """Get the filter envelope for a slot (default: selected slot)."""
    if slot is None:
        slot = self.selected_filter_envelope
    return self.filter_envelopes.get(slot, {
        'attack': 0.01, 'decay': 0.1, 'sustain': 1.0, 'release': 0.1
    })


# ------------ Parameter setters and getters ------------

def set_carrier_count(self: "Session", n: int) -> None:
    n = max(1, int(n))
    # Ensure enough operators exist
    self._ensure_operators(n + self.mod_count)
    self.carrier_count = n
    # Clamp current operator index into valid range
    max_idx = max(0, self.carrier_count + self.mod_count - 1)
    if self.current_operator > max_idx:
        self.current_operator = max_idx


def set_mod_count(self: "Session", n: int) -> None:
    n = max(0, int(n))
    # Ensure enough operators exist
    self._ensure_operators(self.carrier_count + n)
    self.mod_count = n
    # Clamp current operator index into valid range
    max_idx = max(0, self.carrier_count + self.mod_count - 1)
    if self.current_operator > max_idx:
        self.current_operator = max_idx


def set_voice_count(self: "Session", n: int) -> None:
    self.voice_count = max(1, int(n))


def set_voice_algorithm(self: "Session", alg: str) -> None:
    self.voice_algorithm = alg


def select_operator(self: "Session", idx: int) -> None:
    if idx < 0:
        raise ValueError("operator index must be non-negative")
    self.current_operator = idx


# Operator parameter updates
def set_waveform(self: "Session", wave_type: str) -> None:
    self.engine.set_wave(self.current_operator, wave_type)


def set_frequency(self: "Session", freq: float) -> None:
    self.engine.set_wave(self.current_operator, self.engine.operators.get(self.current_operator, {}).get('wave', 'sine'), freq=freq)


def set_amplitude(self: "Session", amp: float) -> None:
    self.engine.set_wave(self.current_operator, self.engine.operators.get(self.current_operator, {}).get('wave', 'sine'), amp=amp)


def set_phase(self: "Session", phase: float) -> None:
    self.engine.set_wave(self.current_operator, self.engine.operators.get(self.current_operator, {}).get('wave', 'sine'), phase=phase)


# Modulation algorithms
def add_modulation(self: "Session", algo_type: str, source: int, target: int, amount: float) -> None:
    self.engine.add_algorithm(algo_type.upper(), int(source), int(target), float(amount))


def clear_algorithms(self: "Session") -> None:
    self.engine.clear_algorithms()


# Envelope settings - level-aware
def set_attack(self: "Session", val: float) -> None:
    self.set_envelope_param('attack', val)


def set_decay(self: "Session", val: float) -> None:
    self.set_envelope_param('decay', val)


def set_sustain(self: "Session", val: float) -> None:
    self.set_envelope_param('sustain', val)


def set_release(self: "Session", val: float) -> None:
    self.set_envelope_param('release', val)


# Filter envelope setters - use selected filter envelope slot
def set_f_attack(self: "Session", val: float) -> None:
    """Set filter attack time in seconds."""
    self.set_filter_envelope_param('attack', val)


def set_f_decay(self: "Session", val: float) -> None:
    """Set filter decay time in seconds."""
    self.set_filter_envelope_param('decay', val)


def set_f_sustain(self: "Session", val: float) -> None:
    """Set filter sustain level (0-1)."""
    self.set_filter_envelope_param('sustain', val)


def set_f_release(self: "Session", val: float) -> None:
    """Set filter release time in seconds."""
    self.set_filter_envelope_param('release', val)


# Voice algorithm parameter setters
def set_dt(self: "Session", val: float) -> None:
    """Set voice detune amount.

    Parameters
    ----------
    val : float
        Detune in Hz (real units). A non-zero value detunes
        successive voices by this amount. Typical range: 0-10 Hz.
    """
    self.dt = float(val)


def set_rand(self: "Session", val: float) -> None:
    """Set voice random amplitude variation.

    Parameters
    ----------
    val : float
        Amplitude variation (0-100 scale)
        0 = no variation
        50 = +/-50% variation per voice
        100 = +/-100% variation per voice
        >100 = wacky territory (allowed)
    """
    parsed = parse_param(val, default=0.0)
    clamped, warning = validate_param(parsed, "rand")
    if warning:
        print(f"[session] {warning}")
    self.rand = clamped


def set_mod(self: "Session", val: float) -> None:
    """Set voice modulation scaling.

    Parameters
    ----------
    val : float
        Modulation scaling (0-100 scale)
        0 = same modulation depth all voices
        50 = 1.5x on last voice
        100 = 2x on last voice
        >100 = wacky territory (allowed)
    """
    parsed = parse_param(val, default=0.0)
    clamped, warning = validate_param(parsed, "v_mod")
    if warning:
        print(f"[session] {warning}")
    self.v_mod = clamped


def bind_to(session_cls) -> None:
    """Attach every parameter function above to ``session_cls``."""
    # Filter
    session_cls.set_filter_count = set_filter_count
    session_cls.set_filter_type = set_filter_type
    session_cls.select_filter = select_filter
    session_cls.select_filter_envelope = select_filter_envelope
    session_cls.get_current_filter_settings = get_current_filter_settings
    session_cls.set_cutoff = set_cutoff
    session_cls.set_resonance = set_resonance
    session_cls.enable_filter = enable_filter
    # Two accessors stay ``@property`` — wrap before binding.
    session_cls.resonance = property(_resonance_getter)
    session_cls.cutoff = property(_cutoff_getter)
    # Envelopes
    session_cls.get_envelope_for_operator = get_envelope_for_operator
    session_cls.set_envelope_param = set_envelope_param
    session_cls.get_envelope_param = get_envelope_param
    session_cls.set_filter_envelope_param = set_filter_envelope_param
    session_cls.get_filter_envelope = get_filter_envelope
    # Voice / operator counts
    session_cls.set_carrier_count = set_carrier_count
    session_cls.set_mod_count = set_mod_count
    session_cls.set_voice_count = set_voice_count
    session_cls.set_voice_algorithm = set_voice_algorithm
    session_cls.select_operator = select_operator
    # Operator parameter updates
    session_cls.set_waveform = set_waveform
    session_cls.set_frequency = set_frequency
    session_cls.set_amplitude = set_amplitude
    session_cls.set_phase = set_phase
    # Modulation algorithms
    session_cls.add_modulation = add_modulation
    session_cls.clear_algorithms = clear_algorithms
    # Envelope setters
    session_cls.set_attack = set_attack
    session_cls.set_decay = set_decay
    session_cls.set_sustain = set_sustain
    session_cls.set_release = set_release
    session_cls.set_f_attack = set_f_attack
    session_cls.set_f_decay = set_f_decay
    session_cls.set_f_sustain = set_f_sustain
    session_cls.set_f_release = set_f_release
    # Voice algorithm params
    session_cls.set_dt = set_dt
    session_cls.set_rand = set_rand
    session_cls.set_mod = set_mod
