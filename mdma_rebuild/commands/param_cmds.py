"""Parameter Management Commands for MDMA.

EXPLICIT PARAMETER SYSTEM
=========================
Centralized parameter management with modulation support.

COMMANDS:
- /PARM [category] [index]  → List/select parameters
- /CHA <value|op|mod>       → Change selected parameter

CATEGORIES:
- synth   : attack, decay, sustain, release, amp, dt, rand
- filter  : cutoff, resonance, env_amount
- global  : bpm, master_vol, pan
- voice   : voice_count, v_mod
- operator: ratio, level, feedback (per-operator)

MODULATION:
- LFO: /cha lfo <shape> <rate> <depth>
- Envelope: /cha env <a> <d> <s> <r> [depth]
- Random: /cha rand [min] [max]

BUILD ID: param_cmds_v1.0
"""

from __future__ import annotations

import time
import numpy as np
from typing import List, Dict, Optional, Callable, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ..core.session import Session


# ============================================================================
# PARAMETER DEFINITIONS
# ============================================================================

@dataclass
class ParameterDef:
    """Definition of a controllable parameter."""
    name: str
    category: str
    min_val: float
    max_val: float
    default: float
    unit: str  # "Hz", "sec", "%", "", "BPM"
    description: str
    getter: Callable[["Session"], float]
    setter: Callable[["Session", float], None]


@dataclass
class ParameterModulation:
    """Modulation assigned to a parameter."""
    mod_type: str = 'none'  # 'lfo', 'envelope', 'random', 'none'
    
    # LFO parameters
    lfo_shape: str = 'sin'  # sin, tri, saw, sqr, rnd
    lfo_rate: float = 1.0   # Hz
    lfo_depth: float = 50.0 # 0-100%
    
    # Envelope parameters
    env_attack: float = 0.1
    env_decay: float = 0.2
    env_sustain: float = 0.5
    env_release: float = 0.3
    env_depth: float = 100.0  # 0-100%
    
    # Random parameters
    rand_min: float = 0.0
    rand_max: float = 1.0
    rand_mode: str = 'uniform'  # 'uniform', 'gaussian', 'walk'
    
    # State
    phase: float = 0.0
    last_value: float = 0.0


# ============================================================================
# PARAMETER SYSTEM SINGLETON
# ============================================================================

class ParameterSystem:
    """Centralized parameter management system."""
    
    _instance: Optional['ParameterSystem'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.params: Dict[str, ParameterDef] = {}
        self.modulations: Dict[str, ParameterModulation] = {}
        self.selected_param: Optional[str] = None
        self.selected_category: Optional[str] = None
        
        # Categories in display order
        self.categories = ['synth', 'filter', 'global', 'voice', 'operator', 'effect']
        
        # Register all parameters
        self._register_all_params()
    
    def _register_all_params(self):
        """Register all controllable parameters."""
        
        # Helper for safe attribute access
        def safe_get(attr, default):
            return lambda s: getattr(s, attr, default)
        
        def safe_set(attr):
            return lambda s, v: setattr(s, attr, v)
        
        # === SYNTH PARAMETERS ===
        self._register('attack', 'synth', 0.001, 5.0, 0.01, 'sec',
                      'Amplitude envelope attack time',
                      safe_get('attack', 0.01),
                      safe_set('attack'))
        
        self._register('decay', 'synth', 0.001, 5.0, 0.1, 'sec',
                      'Amplitude envelope decay time',
                      safe_get('decay', 0.1),
                      safe_set('decay'))
        
        self._register('sustain', 'synth', 0.0, 1.0, 0.8, '',
                      'Amplitude envelope sustain level',
                      safe_get('sustain', 0.8),
                      safe_set('sustain'))
        
        self._register('release', 'synth', 0.001, 10.0, 0.1, 'sec',
                      'Amplitude envelope release time',
                      safe_get('release', 0.1),
                      safe_set('release'))
        
        self._register('dt', 'synth', 0.0, 100.0, 0.0, 'Hz',
                      'Voice detune amount',
                      safe_get('dt', 0.0),
                      safe_set('dt'))
        
        self._register('rand', 'synth', 0.0, 100.0, 0.0, '%',
                      'Voice amplitude randomization',
                      safe_get('rand', 0.0),
                      safe_set('rand'))
        
        self._register('v_mod', 'synth', 0.0, 100.0, 0.0, '%',
                      'Voice modulation scaling',
                      safe_get('v_mod', 0.0),
                      safe_set('v_mod'))
        
        # === FILTER PARAMETERS ===
        def get_cutoff(s):
            cutoffs = getattr(s, 'filter_cutoffs', {0: 4500.0})
            sel = getattr(s, 'selected_filter', 0)
            return cutoffs.get(sel, 4500.0)
        
        def set_cutoff(s, v):
            if not hasattr(s, 'filter_cutoffs'):
                s.filter_cutoffs = {0: 4500.0}
            sel = getattr(s, 'selected_filter', 0)
            s.filter_cutoffs[sel] = v
        
        self._register('cutoff', 'filter', 20.0, 20000.0, 4500.0, 'Hz',
                      'Filter cutoff frequency',
                      get_cutoff, set_cutoff)
        
        def get_resonance(s):
            resos = getattr(s, 'filter_resonances', {0: 50.0})
            sel = getattr(s, 'selected_filter', 0)
            return resos.get(sel, 50.0)
        
        def set_resonance(s, v):
            if not hasattr(s, 'filter_resonances'):
                s.filter_resonances = {0: 50.0}
            sel = getattr(s, 'selected_filter', 0)
            s.filter_resonances[sel] = v
        
        self._register('resonance', 'filter', 0.0, 100.0, 50.0, '%',
                      'Filter resonance',
                      get_resonance, set_resonance)
        
        # === GLOBAL PARAMETERS ===
        self._register('bpm', 'global', 20.0, 300.0, 128.0, 'BPM',
                      'Tempo in beats per minute',
                      safe_get('bpm', 128.0),
                      safe_set('bpm'))
        
        self._register('step', 'global', 0.0625, 4.0, 0.125, 'beats',
                      'Step duration in beats',
                      safe_get('step', 0.125),
                      safe_set('step'))
        
        # === VOICE PARAMETERS ===
        self._register('voice_count', 'voice', 1, 16, 1, '',
                      'Number of unison voices',
                      lambda s: float(getattr(s, 'voice_count', 1)),
                      lambda s, v: setattr(s, 'voice_count', int(v)))
        
        self._register('carrier_count', 'voice', 1, 8, 1, '',
                      'Number of carrier operators',
                      lambda s: float(getattr(s, 'carrier_count', 1)),
                      lambda s, v: setattr(s, 'carrier_count', int(v)))
        
        self._register('mod_count', 'voice', 0, 8, 0, '',
                      'Number of modulator operators',
                      lambda s: float(getattr(s, 'mod_count', 0)),
                      lambda s, v: setattr(s, 'mod_count', int(v)))
        
        # === OPERATOR PARAMETERS ===
        def get_op_ratio(s):
            op = getattr(s, 'current_operator', 0)
            engine = getattr(s, 'engine', None)
            if engine and hasattr(engine, 'operators'):
                return engine.operators.get(op, {}).get('ratio', 1.0)
            return 1.0
        
        def set_op_ratio(s, v):
            op = getattr(s, 'current_operator', 0)
            engine = getattr(s, 'engine', None)
            if engine and hasattr(engine, 'operators'):
                engine.operators.setdefault(op, {})['ratio'] = v
        
        self._register('op_ratio', 'operator', 0.0, 32.0, 1.0, '',
                      'Operator frequency ratio',
                      get_op_ratio, set_op_ratio)
        
        def get_op_level(s):
            op = getattr(s, 'current_operator', 0)
            engine = getattr(s, 'engine', None)
            if engine and hasattr(engine, 'operators'):
                return engine.operators.get(op, {}).get('amp', 1.0)
            return 1.0
        
        def set_op_level(s, v):
            op = getattr(s, 'current_operator', 0)
            engine = getattr(s, 'engine', None)
            if engine and hasattr(engine, 'operators'):
                engine.operators.setdefault(op, {})['amp'] = v
        
        self._register('op_level', 'operator', 0.0, 1.0, 1.0, '',
                      'Operator amplitude level',
                      get_op_level, set_op_level)
        
        def get_op_feedback(s):
            op = getattr(s, 'current_operator', 0)
            engine = getattr(s, 'engine', None)
            if engine and hasattr(engine, 'operators'):
                return engine.operators.get(op, {}).get('feedback', 0.0)
            return 0.0
        
        def set_op_feedback(s, v):
            op = getattr(s, 'current_operator', 0)
            engine = getattr(s, 'engine', None)
            if engine and hasattr(engine, 'operators'):
                engine.operators.setdefault(op, {})['feedback'] = v
        
        self._register('op_feedback', 'operator', 0.0, 1.0, 0.0, '',
                      'Operator self-feedback amount',
                      get_op_feedback, set_op_feedback)
    
    def _register(self, name: str, category: str, min_val: float, max_val: float,
                  default: float, unit: str, description: str,
                  getter: Callable, setter: Callable):
        """Register a parameter."""
        self.params[name] = ParameterDef(
            name=name,
            category=category,
            min_val=min_val,
            max_val=max_val,
            default=default,
            unit=unit,
            description=description,
            getter=getter,
            setter=setter
        )
    
    def get_by_category(self, category: str, session: "Session" = None) -> List[ParameterDef]:
        """Get all parameters in a category.
        
        For the 'effect' category, dynamically refreshes parameters
        based on the current effect chains in the session.
        """
        if category == 'effect' and session is not None:
            self._refresh_effect_params(session)
        return [p for p in self.params.values() if p.category == category]
    
    def _refresh_effect_params(self, session: "Session") -> None:
        """Dynamically register parameters for all effects in the active chains.
        
        Scans session.effects (the /fx chain), buffer_fx_chain, 
        track_fx_chain, and master_fx_chain.  For each effect, registers
        parameters from the effect metadata (actual DSP params like drive,
        wet, threshold, etc.) plus the universal 'amount' wet/dry control.
        
        Parameters are named like:
          fx0_amount        — amount for effect 0 in session.effects
          bfx1_wet          — wet param for buffer_fx_chain[1]
          mfx0_threshold    — threshold for master_fx_chain[0]
        """
        # Clear old effect params
        to_remove = [k for k, v in self.params.items() if v.category == 'effect']
        for k in to_remove:
            del self.params[k]
        
        # Import metadata from effects module
        try:
            from ..dsp.effects import get_effect_dsp_params, _get_param_range
        except ImportError:
            return
        
        # Collect all chains with their prefixes
        chains: list[tuple[str, str, list]] = []
        
        # session.effects + session.effect_params (the /fx chain)
        if hasattr(session, 'effects') and session.effects:
            fx_chain = list(zip(
                session.effects,
                session.effect_params if hasattr(session, 'effect_params') else [{}] * len(session.effects)
            ))
            chains.append(('fx', 'FX chain', fx_chain))
        
        # buffer_fx_chain
        if hasattr(session, 'buffer_fx_chain') and session.buffer_fx_chain:
            chains.append(('bfx', 'Buffer FX', session.buffer_fx_chain))
        
        # track_fx_chain (default for all tracks)
        if hasattr(session, 'track_fx_chain') and session.track_fx_chain:
            chains.append(('tfx', 'Track FX', session.track_fx_chain))
        
        # master_fx_chain
        if hasattr(session, 'master_fx_chain') and session.master_fx_chain:
            chains.append(('mfx', 'Master FX', session.master_fx_chain))
        
        # Per-track chains
        if hasattr(session, 'tracks'):
            for t_idx, track in enumerate(session.tracks):
                per_track_chain = track.get('fx_chain', [])
                if per_track_chain:
                    chains.append((f't{t_idx}fx', f'Track {t_idx+1} FX', per_track_chain))
        
        for prefix, chain_label, chain_items in chains:
            for idx, item in enumerate(chain_items):
                # Unpack — session.effects uses (name, params_dict) or just name+separate params
                if isinstance(item, tuple) and len(item) == 2:
                    fx_name, fx_params = item
                elif isinstance(item, str):
                    fx_name = item
                    fx_params = {}
                else:
                    continue
                
                if not isinstance(fx_params, dict):
                    fx_params = {}
                
                param_prefix = f"{prefix}{idx}"
                display_name = f"{chain_label}[{idx}] {fx_name}"
                
                # Get all DSP parameters for this effect from metadata
                dsp_params = get_effect_dsp_params(fx_name)
                
                for pname, default_val in dsp_params.items():
                    min_v, max_v = _get_param_range(pname, default_val)
                    current = fx_params.get(pname, default_val)
                    
                    # Determine unit hint
                    unit = ''
                    pn = pname.lower()
                    if any(k in pn for k in ('freq', 'cutoff', 'crossover', 'filter', 'high_cut', 'low_cut')):
                        unit = 'Hz'
                    elif 'time' in pn or 'delay' in pn:
                        unit = 'sec'
                    elif any(k in pn for k in ('amount', 'wet', 'dry', 'mix', 'drive', 'gain',
                                                'threshold', 'ratio', 'makeup', 'depth')):
                        unit = '%'
                    elif 'semitone' in pn:
                        unit = 'st'
                    
                    self._register_effect_param(
                        f'{param_prefix}_{pname}', fx_name, idx, prefix,
                        min_v, max_v, float(current), unit,
                        f'{display_name} {pname}',
                        pname, session, float(default_val)
                    )
    
    def _register_effect_param(self, name: str, fx_name: str, fx_idx: int,
                                chain_prefix: str, min_val: float, max_val: float,
                                default: float, unit: str, description: str,
                                param_key: str, session: "Session",
                                fallback_default: float = 50.0) -> None:
        """Register a single effect parameter with getter/setter closures."""
        
        def make_getter(fx_n, fx_i, ch_prefix, p_key, p_default):
            def getter(s):
                chain_data = self._get_chain_params(s, ch_prefix, fx_i)
                if chain_data is not None:
                    return float(chain_data.get(p_key, p_default))
                return float(p_default)
            return getter
        
        def make_setter(fx_n, fx_i, ch_prefix, p_key):
            def setter(s, v):
                self._set_chain_param(s, ch_prefix, fx_i, p_key, v)
            return setter
        
        self._register(name, 'effect', min_val, max_val, default, unit,
                       description,
                       make_getter(fx_name, fx_idx, chain_prefix, param_key, fallback_default),
                       make_setter(fx_name, fx_idx, chain_prefix, param_key))
    
    def _get_chain_params(self, session: "Session", prefix: str, idx: int) -> Optional[dict]:
        """Get the params dict for an effect at the given chain position."""
        if prefix == 'fx':
            if hasattr(session, 'effect_params') and idx < len(session.effect_params):
                return session.effect_params[idx]
        elif prefix == 'bfx':
            if hasattr(session, 'buffer_fx_chain') and idx < len(session.buffer_fx_chain):
                return session.buffer_fx_chain[idx][1]
        elif prefix == 'tfx':
            if hasattr(session, 'track_fx_chain') and idx < len(session.track_fx_chain):
                return session.track_fx_chain[idx][1]
        elif prefix == 'mfx':
            if hasattr(session, 'master_fx_chain') and idx < len(session.master_fx_chain):
                return session.master_fx_chain[idx][1]
        elif prefix.startswith('t') and prefix.endswith('fx'):
            # Per-track: t0fx, t1fx, etc.
            try:
                t_idx = int(prefix[1:-2])
                if hasattr(session, 'tracks') and t_idx < len(session.tracks):
                    chain = session.tracks[t_idx].get('fx_chain', [])
                    if idx < len(chain):
                        return chain[idx][1]
            except (ValueError, IndexError):
                pass
        return None
    
    def _set_chain_param(self, session: "Session", prefix: str, idx: int,
                          param_key: str, value: float) -> None:
        """Set a parameter value in the appropriate chain's params dict."""
        if prefix == 'fx':
            if hasattr(session, 'effect_params') and idx < len(session.effect_params):
                session.effect_params[idx][param_key] = value
        elif prefix == 'bfx':
            if hasattr(session, 'buffer_fx_chain') and idx < len(session.buffer_fx_chain):
                session.buffer_fx_chain[idx] = (session.buffer_fx_chain[idx][0],
                    {**session.buffer_fx_chain[idx][1], param_key: value})
        elif prefix == 'tfx':
            if hasattr(session, 'track_fx_chain') and idx < len(session.track_fx_chain):
                session.track_fx_chain[idx] = (session.track_fx_chain[idx][0],
                    {**session.track_fx_chain[idx][1], param_key: value})
        elif prefix == 'mfx':
            if hasattr(session, 'master_fx_chain') and idx < len(session.master_fx_chain):
                session.master_fx_chain[idx] = (session.master_fx_chain[idx][0],
                    {**session.master_fx_chain[idx][1], param_key: value})
        elif prefix.startswith('t') and prefix.endswith('fx'):
            try:
                t_idx = int(prefix[1:-2])
                if hasattr(session, 'tracks') and t_idx < len(session.tracks):
                    chain = session.tracks[t_idx].get('fx_chain', [])
                    if idx < len(chain):
                        chain[idx] = (chain[idx][0], {**chain[idx][1], param_key: value})
            except (ValueError, IndexError):
                pass
    
    @staticmethod
    def _infer_param_range(name: str, default: float) -> tuple:
        """Infer reasonable min/max range from a parameter name and default.
        
        Delegates to _get_param_range from dsp.effects for consistency.
        Falls back to a simple heuristic if unavailable.
        """
        try:
            from ..dsp.effects import _get_param_range
            return _get_param_range(name, default)
        except ImportError:
            if 0.0 <= default <= 1.0:
                return (0.0, 1.0)
            if 0.0 <= default <= 100.0:
                return (0.0, 100.0)
            return (min(0.0, default * 0.5), max(default * 2, 1.0))
    
    def get_value(self, name: str, session: "Session") -> float:
        """Get current value of a parameter."""
        if name not in self.params:
            raise KeyError(f"Unknown parameter: {name}")
        return self.params[name].getter(session)
    
    def set_value(self, name: str, session: "Session", value: float) -> float:
        """Set value of a parameter (clamped to range)."""
        if name not in self.params:
            raise KeyError(f"Unknown parameter: {name}")
        
        param = self.params[name]
        value = max(param.min_val, min(param.max_val, value))
        param.setter(session, value)
        return value
    
    def get_modulated_value(self, name: str, session: "Session", 
                           sample_idx: int = 0, total_samples: int = 1) -> float:
        """Get parameter value with modulation applied."""
        base_value = self.get_value(name, session)
        
        if name not in self.modulations:
            return base_value
        
        mod = self.modulations[name]
        if mod.mod_type == 'none':
            return base_value
        
        param = self.params[name]
        param_range = param.max_val - param.min_val
        
        if mod.mod_type == 'lfo':
            return self._apply_lfo(base_value, mod, param_range, sample_idx, total_samples, session)
        elif mod.mod_type == 'envelope':
            return self._apply_envelope(base_value, mod, param_range, sample_idx, total_samples)
        elif mod.mod_type == 'random':
            return self._apply_random(base_value, mod, param)
        
        return base_value
    
    def _apply_lfo(self, base: float, mod: ParameterModulation, 
                   param_range: float, sample_idx: int, total_samples: int,
                   session: "Session") -> float:
        """Apply LFO modulation."""
        sr = session.sample_rate if hasattr(session, 'sample_rate') else 48000
        t = sample_idx / sr
        phase = 2 * np.pi * mod.lfo_rate * t + mod.phase
        
        if mod.lfo_shape == 'sin':
            wave = np.sin(phase)
        elif mod.lfo_shape == 'tri':
            wave = 2 * np.abs(2 * ((t * mod.lfo_rate) % 1) - 1) - 1
        elif mod.lfo_shape == 'saw':
            wave = 2 * ((t * mod.lfo_rate) % 1) - 1
        elif mod.lfo_shape == 'sqr':
            wave = np.sign(np.sin(phase))
        elif mod.lfo_shape == 'rnd':
            # Sample-and-hold
            period = int(sr / max(0.001, mod.lfo_rate))
            if sample_idx % period == 0:
                mod.last_value = np.random.uniform(-1, 1)
            wave = mod.last_value
        else:
            wave = np.sin(phase)
        
        depth = mod.lfo_depth / 100.0
        return base + wave * depth * param_range * 0.5
    
    def _apply_envelope(self, base: float, mod: ParameterModulation,
                       param_range: float, sample_idx: int, total_samples: int) -> float:
        """Apply envelope modulation."""
        if total_samples <= 0:
            return base
        
        # Calculate envelope position (0.0 to 1.0)
        position = sample_idx / total_samples
        
        # Calculate ADSR envelope value
        a_end = mod.env_attack / (mod.env_attack + mod.env_decay + mod.env_release + 0.001)
        d_end = a_end + mod.env_decay / (mod.env_attack + mod.env_decay + mod.env_release + 0.001)
        r_start = 1.0 - mod.env_release / (mod.env_attack + mod.env_decay + mod.env_release + 0.001)
        
        if position < a_end:
            # Attack phase
            env_value = position / a_end if a_end > 0 else 1.0
        elif position < d_end:
            # Decay phase
            decay_pos = (position - a_end) / (d_end - a_end) if (d_end - a_end) > 0 else 1.0
            env_value = 1.0 - (1.0 - mod.env_sustain) * decay_pos
        elif position < r_start:
            # Sustain phase
            env_value = mod.env_sustain
        else:
            # Release phase
            release_pos = (position - r_start) / (1.0 - r_start) if (1.0 - r_start) > 0 else 1.0
            env_value = mod.env_sustain * (1.0 - release_pos)
        
        depth = mod.env_depth / 100.0
        return base + env_value * depth * param_range * 0.5
    
    def _apply_random(self, base: float, mod: ParameterModulation, 
                      param: ParameterDef) -> float:
        """Apply random modulation."""
        if mod.rand_mode == 'uniform':
            # Random value in range
            rand_range = mod.rand_max - mod.rand_min
            value = mod.rand_min + np.random.random() * rand_range
            # Scale to parameter range
            return param.min_val + value * (param.max_val - param.min_val)
        elif mod.rand_mode == 'gaussian':
            mean = (mod.rand_min + mod.rand_max) / 2
            std = (mod.rand_max - mod.rand_min) / 4
            value = np.clip(np.random.normal(mean, std), mod.rand_min, mod.rand_max)
            return param.min_val + value * (param.max_val - param.min_val)
        elif mod.rand_mode == 'walk':
            step = (mod.rand_max - mod.rand_min) * 0.1
            mod.last_value = np.clip(mod.last_value + np.random.uniform(-step, step),
                                     mod.rand_min, mod.rand_max)
            return param.min_val + mod.last_value * (param.max_val - param.min_val)
        return base
    
    def set_modulation(self, name: str, mod: ParameterModulation):
        """Set modulation for a parameter."""
        if name not in self.params:
            raise KeyError(f"Unknown parameter: {name}")
        self.modulations[name] = mod
    
    def clear_modulation(self, name: str):
        """Clear modulation from a parameter."""
        if name in self.modulations:
            del self.modulations[name]


def get_param_system() -> ParameterSystem:
    """Get global parameter system instance."""
    return ParameterSystem()


# ============================================================================
# COMMANDS
# ============================================================================

def cmd_parm(session: "Session", args: List[str]) -> str:
    """List and select parameters.
    
    Usage:
      /PARM                    List all categories
      /PARM <category>         List parameters in category
      /PARM <category> <n>     Select parameter n from category
      /PARM ?                  Show currently selected parameter
    
    Categories: synth, filter, global, voice, operator, effect
    
    The 'effect' category is dynamic — it shows parameters for all effects
    currently in the active chains (FX chain, buffer FX, track FX, master FX).
    Parameters are named like: fx0_amount, mfx1_wet, bfx0_drive, etc.
    
    Examples:
      /parm synth              List synth parameters
      /parm synth 1            Select attack (synth param #1)
      /parm filter 1           Select cutoff (filter param #1)
      /parm effect             List all active effect parameters
      /parm ?                  Show current selection
    """
    ps = get_param_system()
    
    if not args:
        # List all categories
        lines = ["=== PARAMETER CATEGORIES ==="]
        for cat in ps.categories:
            params = ps.get_by_category(cat, session)
            lines.append(f"  {cat.upper()}: {len(params)} parameters")
        lines.append("")
        lines.append("Use: /parm <category> to list parameters")
        lines.append("     /parm <category> <n> to select")
        if ps.selected_param:
            lines.append(f"\nCurrently selected: {ps.selected_param}")
        return "\n".join(lines)
    
    arg = args[0].lower()
    
    # Show current selection
    if arg == '?':
        if not ps.selected_param:
            return "No parameter selected. Use /parm <category> <n> to select."
        
        param = ps.params[ps.selected_param]
        value = param.getter(session)
        
        lines = [f"=== SELECTED: {ps.selected_param.upper()} ==="]
        lines.append(f"  Category: {param.category}")
        lines.append(f"  Value: {value:.4f} {param.unit}")
        lines.append(f"  Range: {param.min_val} - {param.max_val} {param.unit}")
        lines.append(f"  {param.description}")
        
        # Show modulation if any
        if ps.selected_param in ps.modulations:
            mod = ps.modulations[ps.selected_param]
            if mod.mod_type != 'none':
                lines.append(f"\n  Modulation: {mod.mod_type.upper()}")
                if mod.mod_type == 'lfo':
                    lines.append(f"    Shape: {mod.lfo_shape}, Rate: {mod.lfo_rate}Hz, Depth: {mod.lfo_depth}%")
                elif mod.mod_type == 'envelope':
                    lines.append(f"    ADSR: {mod.env_attack}/{mod.env_decay}/{mod.env_sustain}/{mod.env_release}")
                    lines.append(f"    Depth: {mod.env_depth}%")
                elif mod.mod_type == 'random':
                    lines.append(f"    Mode: {mod.rand_mode}, Range: {mod.rand_min} - {mod.rand_max}")
        
        lines.append("\nUse /cha <value> to change")
        return "\n".join(lines)
    
    # List category
    if arg in ps.categories:
        params = ps.get_by_category(arg, session)
        
        lines = [f"=== {arg.upper()} PARAMETERS ==="]
        for i, param in enumerate(params, 1):
            value = param.getter(session)
            mod_str = ""
            if param.name in ps.modulations and ps.modulations[param.name].mod_type != 'none':
                mod_str = f" [MOD:{ps.modulations[param.name].mod_type}]"
            selected = " ← SELECTED" if ps.selected_param == param.name else ""
            lines.append(f"  {i}. {param.name}: {value:.4f} {param.unit}{mod_str}{selected}")
        
        lines.append("")
        lines.append(f"Use: /parm {arg} <n> to select")
        
        # If second argument, select that parameter
        if len(args) > 1:
            try:
                idx = int(args[1]) - 1
                if 0 <= idx < len(params):
                    ps.selected_param = params[idx].name
                    ps.selected_category = arg
                    lines.append(f"\n→ Selected: {params[idx].name}")
                else:
                    lines.append(f"\nERROR: Invalid index {args[1]} (1-{len(params)})")
            except ValueError:
                lines.append(f"\nERROR: Invalid index '{args[1]}'")
        
        return "\n".join(lines)
    
    return f"ERROR: Unknown category '{arg}'. Use: synth, filter, global, voice, operator, effect"


def cmd_cha(session: "Session", args: List[str]) -> str:
    """Change selected parameter value or assign modulation.
    
    Usage:
      /CHA <value>             Set to exact value
      /CHA +<value>            Add to current value
      /CHA -<value>            Subtract from current value
      /CHA *<value>            Multiply by value
      /CHA /<value>            Divide by value
      /CHA rand                Random value in parameter's range
      /CHA rand <min> <max>    Random value in specified range
      /CHA lfo <shape> <rate> <depth>    Assign LFO
      /CHA env <a> <d> <s> <r> [depth]   Assign envelope
      /CHA clear               Remove modulation
    
    LFO shapes: sin, tri, saw, sqr, rnd
    
    Examples:
      /cha 0.5                 Set to 0.5
      /cha +0.1                Increase by 0.1
      /cha *2                  Double the value
      /cha lfo sin 2 50        Sine LFO at 2Hz, 50% depth
      /cha env 0.1 0.2 0.5 0.3 Set envelope modulation
    """
    ps = get_param_system()
    
    if not ps.selected_param:
        return "ERROR: No parameter selected. Use /parm <category> <n> first."
    
    if not args:
        # Show current value
        param = ps.params[ps.selected_param]
        value = param.getter(session)
        return f"{ps.selected_param}: {value:.4f} {param.unit}"
    
    param = ps.params[ps.selected_param]
    current_value = param.getter(session)
    
    cmd = args[0].lower()
    
    # Clear modulation
    if cmd == 'clear':
        ps.clear_modulation(ps.selected_param)
        return f"OK: Cleared modulation from {ps.selected_param}"
    
    # Random value
    if cmd == 'rand' or cmd == 'random':
        if len(args) >= 3:
            try:
                rand_min = float(args[1])
                rand_max = float(args[2])
            except ValueError:
                return "ERROR: Invalid range values"
        else:
            rand_min = param.min_val
            rand_max = param.max_val
        
        new_value = rand_min + np.random.random() * (rand_max - rand_min)
        new_value = ps.set_value(ps.selected_param, session, new_value)
        return f"OK: {ps.selected_param} = {new_value:.4f} {param.unit} (random)"
    
    # LFO modulation
    if cmd == 'lfo':
        if len(args) < 4:
            return "Usage: /cha lfo <shape> <rate> <depth>\n  Shapes: sin, tri, saw, sqr, rnd"
        
        shape = args[1].lower()
        if shape not in ('sin', 'tri', 'saw', 'sqr', 'rnd'):
            return f"ERROR: Unknown LFO shape '{shape}'. Use: sin, tri, saw, sqr, rnd"
        
        try:
            rate = float(args[2])
            depth = float(args[3])
        except ValueError:
            return "ERROR: Invalid rate or depth values"
        
        mod = ParameterModulation(
            mod_type='lfo',
            lfo_shape=shape,
            lfo_rate=rate,
            lfo_depth=depth
        )
        ps.set_modulation(ps.selected_param, mod)
        return f"OK: {ps.selected_param} modulated by LFO ({shape}, {rate}Hz, {depth}%)"
    
    # Envelope modulation
    if cmd == 'env' or cmd == 'envelope':
        if len(args) < 5:
            return "Usage: /cha env <attack> <decay> <sustain> <release> [depth]"
        
        try:
            attack = float(args[1])
            decay = float(args[2])
            sustain = float(args[3])
            release = float(args[4])
            depth = float(args[5]) if len(args) > 5 else 100.0
        except ValueError:
            return "ERROR: Invalid envelope values"
        
        mod = ParameterModulation(
            mod_type='envelope',
            env_attack=attack,
            env_decay=decay,
            env_sustain=sustain,
            env_release=release,
            env_depth=depth
        )
        ps.set_modulation(ps.selected_param, mod)
        return f"OK: {ps.selected_param} modulated by envelope (ADSR: {attack}/{decay}/{sustain}/{release}, {depth}%)"
    
    # Math operations
    try:
        if cmd.startswith('+'):
            delta = float(cmd[1:]) if len(cmd) > 1 else float(args[1]) if len(args) > 1 else 0
            new_value = current_value + delta
        elif cmd.startswith('-'):
            delta = float(cmd[1:]) if len(cmd) > 1 else float(args[1]) if len(args) > 1 else 0
            new_value = current_value - delta
        elif cmd.startswith('*'):
            factor = float(cmd[1:]) if len(cmd) > 1 else float(args[1]) if len(args) > 1 else 1
            new_value = current_value * factor
        elif cmd.startswith('/'):
            divisor = float(cmd[1:]) if len(cmd) > 1 else float(args[1]) if len(args) > 1 else 1
            if divisor == 0:
                return "ERROR: Division by zero"
            new_value = current_value / divisor
        else:
            # Direct value assignment
            new_value = float(cmd)
    except ValueError:
        return f"ERROR: Invalid value '{cmd}'"
    
    # Apply the new value
    new_value = ps.set_value(ps.selected_param, session, new_value)
    change = new_value - current_value
    change_str = f" ({change:+.4f})" if abs(change) > 0.0001 else ""
    
    return f"OK: {ps.selected_param} = {new_value:.4f} {param.unit}{change_str}"


def cmd_plist(session: "Session", args: List[str]) -> str:
    """List all parameters with current values.
    
    Usage:
      /PLIST              List all parameters
      /PLIST mod          List only modulated parameters
    """
    ps = get_param_system()
    show_mod_only = args and args[0].lower() in ('mod', 'modulated')
    
    lines = ["=== ALL PARAMETERS ==="]
    
    for cat in ps.categories:
        params = ps.get_by_category(cat)
        if not params:
            continue
        
        cat_lines = []
        for param in params:
            value = param.getter(session)
            has_mod = param.name in ps.modulations and ps.modulations[param.name].mod_type != 'none'
            
            if show_mod_only and not has_mod:
                continue
            
            mod_str = ""
            if has_mod:
                mod = ps.modulations[param.name]
                if mod.mod_type == 'lfo':
                    mod_str = f" [LFO:{mod.lfo_shape}@{mod.lfo_rate}Hz]"
                elif mod.mod_type == 'envelope':
                    mod_str = f" [ENV]"
                elif mod.mod_type == 'random':
                    mod_str = f" [RND]"
            
            cat_lines.append(f"    {param.name}: {value:.4f} {param.unit}{mod_str}")
        
        if cat_lines:
            lines.append(f"\n  {cat.upper()}:")
            lines.extend(cat_lines)
    
    if len(lines) == 1:
        return "No modulated parameters" if show_mod_only else "No parameters found"
    
    return "\n".join(lines)


# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

def get_param_commands() -> dict:
    """Return parameter commands for registration."""
    return {
        'parm': cmd_parm,
        'param': cmd_parm,
        'cha': cmd_cha,
        'change': cmd_cha,
        'plist': cmd_plist,
    }
