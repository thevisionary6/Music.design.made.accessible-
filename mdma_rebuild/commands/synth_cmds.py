"""Synth/monolith commands for the MDMA rebuild.

These commands allow the user to manipulate operator counts, select
operators, change waveform parameters and add modulation algorithms.
They also expose the tone generator and envelope settings.
"""

from __future__ import annotations

from typing import List

from ..core.session import Session


def _parse_int(arg: str) -> int:
    return int(float(arg))  # support integers passed as floats (e.g. "4.0")

def _parse_float(arg: str) -> float:
    return float(arg)


# Carrier count
def cmd_car(session: Session, args: List[str]) -> str:
    if not args:
        return f"CARRIERS: {session.carrier_count}"
    try:
        val = _parse_int(args[0])
        session.set_carrier_count(val)
        return f"OK: carriers set to {session.carrier_count}"
    except Exception:
        return "ERROR: invalid carrier count"


# Modulator count
def cmd_mod(session: Session, args: List[str]) -> str:
    if not args:
        return f"MODULATORS: {session.mod_count}"
    try:
        val = _parse_int(args[0])
        session.set_mod_count(val)
        return f"OK: modulators set to {session.mod_count}"
    except Exception:
        return "ERROR: invalid modulator count"


# Voice count
def cmd_v(session: Session, args: List[str]) -> str:
    if not args:
        return f"VOICES: {session.voice_count}"
    try:
        val = _parse_int(args[0])
        session.set_voice_count(val)
        return f"OK: voices set to {session.voice_count}"
    except Exception:
        return "ERROR: invalid voice count"


# Voice algorithm
def cmd_va(session: Session, args: List[str]) -> str:
    """Set voice algorithm for multi-voice behaviour.

    Usage:
      /va              Show current algorithm
      /va 0 | stack    Classic stack (phase-locks without /vphase or /dt)
      /va 1 | unison   Normal unison — rand spreads phase, prevents lock
      /va 2 | wide     Wide unison — auto stereo + detune jitter

    Algorithm 1 is recommended for most use.  With /rand > 0 each
    voice gets a random initial phase so identical waveforms don't
    destructively or constructively interfere.
    """
    ALIASES = {
        '0': 'stack', 'stack': 'stack',
        '1': 'unison', 'unison': 'unison',
        '2': 'wide', 'wide': 'wide',
    }
    if not args:
        cur = session.voice_algorithm or 'stack'
        return f"VOICE ALG: {cur} (0=stack, 1=unison, 2=wide)"
    alg = args[0].lower()
    resolved = ALIASES.get(alg, alg)
    session.set_voice_algorithm(resolved)
    return f"OK: voice algorithm → {resolved}"


# Operator selection
def cmd_op(session: Session, args: List[str]) -> str:
    if not args:
        return f"OPERATOR: {session.current_operator}"
    try:
        idx = _parse_int(args[0])
        session.select_operator(idx)
        return f"OK: operator set to {session.current_operator}"
    except Exception:
        return "ERROR: invalid operator index"


# Waveform selection
def cmd_wm(session: Session, args: List[str]) -> str:
    """Set operator waveform.
    
    Usage:
      /wm              -> Show current wave and list all types
      /wm <type>       -> Set waveform type
      /wm <type> <param>=<val>  -> Set wave with parameters
    
    Wave Types:
      sine, sin        - Pure sine wave
      triangle, tri    - Triangle wave
      saw, sawtooth    - Sawtooth wave (carrier-only)
      pulse, pwm, square - Pulse wave with width control (carrier-only)
      noise, white     - White noise
      pink             - Pink noise (-3dB/octave)
      physical, phys   - Physical modeling (even/odd harmonics)
      physical2, phys2 - Physical modeling with inharmonicity
      supersaw, ssaw   - Stacked detuned saws (JP-8000 style)
      additive, add    - Additive synthesis with harmonic rolloff
      formant, vowel   - Formant-shaped oscillator (vocal timbres)
      harmonic, harm   - Independent odd/even harmonic control
      waveguide_string, string, pluck - Karplus-Strong string model
      waveguide_tube, tube, pipe      - Waveguide tube/pipe model
      waveguide_membrane, membrane, drum - Drum membrane model
      waveguide_plate, plate, bar     - Plate/bar model (vibraphone)
      wavetable, wt    - Wavetable playback (load with /wt)
      compound, comp   - Compound wave (define with /compound)

    Parameters (wave-specific):
      pw=0.5           - Pulse width (0.0-1.0, default 0.5)
      even=8           - Even harmonics count (physical)
      odd=4            - Odd harmonics count (physical)
      ewt=1.0          - Even harmonic weight (physical)
      decay=0.7        - Harmonic decay rate (physical)
      inharm=0.01      - Inharmonicity (physical2)
      partials=12      - Partial count (physical2)
      curve=exp        - Decay curve: exp/linear/sqrt (physical2)
      saws=7           - Number of saw oscillators (supersaw)
      spread=0.5       - Detune spread in semitones (supersaw)
      mix=0.75         - Center vs. stack mix (supersaw)
      rolloff=1.0      - Harmonic rolloff exponent (additive)
      nharm=16         - Harmonic count (additive/harmonic)
      vowel=a          - Vowel shape: a/e/i/o/u (formant)
      oddlvl=1.0       - Odd harmonic level (harmonic)
      evenlvl=1.0      - Even harmonic level (harmonic)
      odecay=0.8       - Odd harmonic decay (harmonic)
      edecay=0.8       - Even harmonic decay (harmonic)
      damp=0.996       - Damping (waveguide models)
      bright=0.5       - Brightness (waveguide_string)
      pos=0.28         - Pluck/strike position (string/membrane)
      reflect=0.7      - End reflection (tube)
      bore=0.5         - Bore shape (tube)
      tension=0.5      - Membrane tension (membrane)
      thick=0.5        - Plate thickness (plate)
      mat=0.5          - Material type (plate: 0=wood, 1=metal)
      frame=0.0        - Wavetable frame position 0.0-1.0
      wtn=name         - Wavetable name to use
      morph=0.0        - Morph position for compound waves
    """
    # Available wave types for display
    wave_types = [
        ('sine', 'sin', 'Pure sine wave'),
        ('triangle', 'tri', 'Triangle wave'),
        ('saw', '', 'Sawtooth (carrier-only)'),
        ('pulse', 'pwm/square', 'Pulse with width control'),
        ('noise', 'white', 'White noise'),
        ('pink', '', 'Pink noise (-3dB/oct)'),
        ('physical', 'phys', 'Harmonic modeling'),
        ('physical2', 'phys2', 'Inharmonic modeling'),
        ('supersaw', 'ssaw', 'Stacked detuned saws'),
        ('additive', 'add', 'Additive synthesis'),
        ('formant', 'vowel', 'Vocal formant oscillator'),
        ('harmonic', 'harm', 'Odd/even harmonic control'),
        ('waveguide_string', 'string/pluck', 'Plucked string model'),
        ('waveguide_tube', 'tube/pipe', 'Tube/pipe model'),
        ('waveguide_membrane', 'membrane/drum', 'Drum membrane model'),
        ('waveguide_plate', 'plate/bar', 'Plate/bar model'),
        ('wavetable', 'wt', 'Wavetable playback'),
        ('compound', 'comp/layer', 'Compound wave layers'),
    ]
    
    if not args:
        # Show current waveform for selected operator
        op_idx = session.current_operator
        params = session.engine.operators.get(op_idx)
        
        lines = [f"=== OPERATOR {op_idx} WAVE ==="]
        if params is None:
            lines.append("  (no operator defined)")
        else:
            wave = params.get('wave', 'sine')
            lines.append(f"  Type: {wave}")
            
            # Show relevant params based on wave type
            if wave == 'pulse':
                lines.append(f"  Pulse Width: {params.get('pw', 0.5):.2f}")
            elif wave == 'physical':
                lines.append(f"  Even Harmonics: {params.get('even_harmonics', 8)}")
                lines.append(f"  Odd Harmonics: {params.get('odd_harmonics', 4)}")
                lines.append(f"  Even Weight: {params.get('even_weight', 1.0):.2f}")
                lines.append(f"  Decay: {params.get('decay', 0.7):.2f}")
            elif wave == 'physical2':
                lines.append(f"  Inharmonicity: {params.get('inharmonicity', 0.01):.4f}")
                lines.append(f"  Partials: {params.get('partials', 12)}")
                lines.append(f"  Decay Curve: {params.get('decay_curve', 'exp')}")
            elif wave == 'supersaw':
                lines.append(f"  Saws: {params.get('num_saws', 7)}")
                lines.append(f"  Detune Spread: {params.get('detune_spread', 0.5):.2f} st")
                lines.append(f"  Mix: {params.get('mix', 0.75):.2f}")
            elif wave == 'additive':
                lines.append(f"  Harmonics: {params.get('num_harmonics', 16)}")
                lines.append(f"  Rolloff: {params.get('rolloff', 1.0):.2f}")
            elif wave == 'formant':
                lines.append(f"  Vowel: {params.get('vowel', 'a')}")
            elif wave == 'harmonic':
                lines.append(f"  Odd Level: {params.get('odd_level', 1.0):.2f}")
                lines.append(f"  Even Level: {params.get('even_level', 1.0):.2f}")
                lines.append(f"  Harmonics: {params.get('num_harmonics', 16)}")
                lines.append(f"  Odd Decay: {params.get('odd_decay', 0.8):.2f}")
                lines.append(f"  Even Decay: {params.get('even_decay', 0.8):.2f}")
            elif wave == 'waveguide_string':
                lines.append(f"  Damping: {params.get('damping', 0.996):.4f}")
                lines.append(f"  Brightness: {params.get('brightness', 0.5):.2f}")
                lines.append(f"  Position: {params.get('position', 0.28):.2f}")
            elif wave == 'waveguide_tube':
                lines.append(f"  Damping: {params.get('damping', 0.995):.4f}")
                lines.append(f"  Reflection: {params.get('reflection', 0.7):.2f}")
                lines.append(f"  Bore Shape: {params.get('bore_shape', 0.5):.2f}")
            elif wave == 'waveguide_membrane':
                lines.append(f"  Tension: {params.get('tension', 0.5):.2f}")
                lines.append(f"  Damping: {params.get('damping', 0.995):.4f}")
                lines.append(f"  Strike Pos: {params.get('strike_pos', 0.3):.2f}")
            elif wave == 'waveguide_plate':
                lines.append(f"  Thickness: {params.get('thickness', 0.5):.2f}")
                lines.append(f"  Damping: {params.get('damping', 0.997):.4f}")
                lines.append(f"  Material: {params.get('material', 0.5):.2f}")
            elif wave == 'wavetable':
                lines.append(f"  Table: {params.get('wavetable_name', '(none)')}")
                lines.append(f"  Frame Pos: {params.get('frame_pos', 0.0):.3f}")
            elif wave == 'compound':
                lines.append(f"  Compound: {params.get('compound_name', '(none)')}")
                lines.append(f"  Morph: {params.get('morph', 0.0):.3f}")
        
        lines.append("")
        lines.append("Available wave types:")
        for name, alias, desc in wave_types:
            alias_str = f" ({alias})" if alias else ""
            lines.append(f"  {name:10s}{alias_str:12s} - {desc}")
        
        return '\n'.join(lines)
    
    # Parse wave type and optional parameters
    wave = args[0].lower()
    kwargs = {}
    
    # Parse additional key=value parameters
    for arg in args[1:]:
        if '=' in arg:
            key, val = arg.split('=', 1)
            key = key.lower()
            
            # Map short param names to full names
            param_map = {
                'pw': 'pw',
                'even': 'even_harmonics',
                'odd': 'odd_harmonics',
                'ewt': 'even_weight',
                'decay': 'decay',
                'inharm': 'inharmonicity',
                'partials': 'partials',
                'curve': 'decay_curve',
                # Phase 2: supersaw
                'saws': 'num_saws',
                'spread': 'detune_spread',
                'mix': 'mix',
                # Phase 2: additive / harmonic
                'rolloff': 'rolloff',
                'nharm': 'num_harmonics',
                'vowel': 'vowel',
                'oddlvl': 'odd_level',
                'evenlvl': 'even_level',
                'odecay': 'odd_decay',
                'edecay': 'even_decay',
                # Phase 2: waveguide
                'damp': 'damping',
                'bright': 'brightness',
                'pos': 'position',
                'reflect': 'reflection',
                'bore': 'bore_shape',
                'tension': 'tension',
                'strike': 'strike_pos',
                'thick': 'thickness',
                'mat': 'material',
                # Phase 2: wavetable / compound
                'frame': 'frame_pos',
                'wtn': 'wavetable_name',
                'morph': 'morph',
                'cmpn': 'compound_name',
            }

            # String-type params (no float conversion)
            STRING_PARAMS = {'decay_curve', 'vowel', 'wavetable_name', 'compound_name'}
            # Integer-type params
            INT_PARAMS = {'even_harmonics', 'odd_harmonics', 'partials', 'num_saws', 'num_harmonics'}

            if key in param_map:
                full_key = param_map[key]
                # Convert to appropriate type
                if full_key in INT_PARAMS:
                    kwargs[full_key] = int(float(val))
                elif full_key in STRING_PARAMS:
                    kwargs[full_key] = val
                else:
                    kwargs[full_key] = float(val)
    
    # Set the waveform
    try:
        session.engine.set_wave(session.current_operator, wave, **kwargs)
        
        result = f"OK: wave set to {wave} on operator {session.current_operator}"
        if kwargs:
            param_str = ', '.join(f"{k}={v}" for k, v in kwargs.items())
            result += f" ({param_str})"
        return result
    except Exception as e:
        return f"ERROR: {e}"


# Frequency
def cmd_fr(session: Session, args: List[str]) -> str:
    if not args:
        params = session.engine.operators.get(session.current_operator)
        return f"FREQ: {params.get('freq') if params else 'unset'}"
    try:
        val = _parse_float(args[0])
        session.set_frequency(val)
        return f"OK: frequency set to {val} on operator {session.current_operator}"
    except Exception:
        return "ERROR: invalid frequency"


# Amplitude
def cmd_amp(session: Session, args: List[str]) -> str:
    if not args:
        params = session.engine.operators.get(session.current_operator)
        return f"AMP: {params.get('amp') if params else 'unset'}"
    try:
        val = _parse_float(args[0])
        if val < 0:
            return "ERROR: amplitude must be non-negative"
        session.set_amplitude(val)
        return f"OK: amplitude set to {val} on operator {session.current_operator}"
    except Exception:
        return "ERROR: invalid amplitude"


# Phase
def cmd_ph(session: Session, args: List[str]) -> str:
    if not args:
        params = session.engine.operators.get(session.current_operator)
        return f"PHASE: {params.get('phase') if params else 'unset'}"
    try:
        val = _parse_float(args[0])
        session.set_phase(val)
        return f"OK: phase set to {val} on operator {session.current_operator}"
    except Exception:
        return "ERROR: invalid phase"


# ============================================================================
# WAVE-SPECIFIC PARAMETER COMMANDS
# ============================================================================

def cmd_pw(session: Session, args: List[str]) -> str:
    """Set pulse width for pulse wave.
    
    Usage:
      /pw          -> Show current pulse width
      /pw <0-1>    -> Set pulse width (0.5 = square)
    
    Values: 0.0 = very thin pulse, 0.5 = square, 1.0 = inverted thin pulse
    """
    op_idx = session.current_operator
    params = session.engine.operators.get(op_idx)
    
    if not args:
        if params is None:
            return "PW: (no operator defined)"
        pw = params.get('pw', 0.5)
        return f"PW: {pw:.2f} (0.5=square)"
    
    try:
        val = _parse_float(args[0])
        val = max(0.01, min(0.99, val))  # Clamp to avoid DC
        session.engine.set_param(op_idx, 'pw', val)
        return f"OK: pulse width set to {val:.2f} on operator {op_idx}"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_phys(session: Session, args: List[str]) -> str:
    """Configure physical modeling wave parameters.
    
    Usage:
      /phys                    -> Show current settings
      /phys <even> <odd>       -> Set harmonic counts
      /phys <even> <odd> <weight> <decay>  -> Full config
    
    Parameters:
      even   - Number of even harmonics (2,4,6,8...) default: 8
      odd    - Number of odd harmonics (3,5,7...) default: 4  
      weight - Even harmonic weight (1.0 = equal) default: 1.0
      decay  - Amplitude decay per harmonic default: 0.7
    
    Creates bell-like, marimba-like, or wood-block tones.
    """
    op_idx = session.current_operator
    params = session.engine.operators.get(op_idx)
    
    if not args:
        if params is None:
            return "PHYS: (no operator defined)"
        
        lines = [
            f"=== PHYSICAL WAVE op{op_idx} ===",
            f"  Even Harmonics: {params.get('even_harmonics', 8)}",
            f"  Odd Harmonics: {params.get('odd_harmonics', 4)}",
            f"  Even Weight: {params.get('even_weight', 1.0):.2f}",
            f"  Decay: {params.get('decay', 0.7):.2f}",
        ]
        return '\n'.join(lines)
    
    try:
        even = _parse_int(args[0]) if len(args) > 0 else 8
        odd = _parse_int(args[1]) if len(args) > 1 else 4
        weight = _parse_float(args[2]) if len(args) > 2 else 1.0
        decay = _parse_float(args[3]) if len(args) > 3 else 0.7
        
        session.engine.set_param(op_idx, 'even_harmonics', even)
        session.engine.set_param(op_idx, 'odd_harmonics', odd)
        session.engine.set_param(op_idx, 'even_weight', weight)
        session.engine.set_param(op_idx, 'decay', decay)
        
        return f"OK: physical params set (even={even}, odd={odd}, weight={weight:.2f}, decay={decay:.2f})"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_phys2(session: Session, args: List[str]) -> str:
    """Configure physical modeling wave variant 2 parameters.
    
    Usage:
      /phys2                      -> Show current settings
      /phys2 <inharm> <partials>  -> Set inharmonicity and partial count
      /phys2 <inharm> <partials> <curve>  -> Full config
    
    Parameters:
      inharm   - Inharmonicity coefficient (0.0=pure, 0.05=very stretched)
      partials - Number of partials to generate (default: 12)
      curve    - Decay curve type: exp, linear, sqrt (default: exp)
    
    Creates piano-like, metallic, or bell-like tones with beating.
    Higher inharmonicity = more metallic/bell-like.
    """
    op_idx = session.current_operator
    params = session.engine.operators.get(op_idx)
    
    if not args:
        if params is None:
            return "PHYS2: (no operator defined)"
        
        lines = [
            f"=== PHYSICAL2 WAVE op{op_idx} ===",
            f"  Inharmonicity: {params.get('inharmonicity', 0.01):.4f}",
            f"  Partials: {params.get('partials', 12)}",
            f"  Decay Curve: {params.get('decay_curve', 'exp')}",
        ]
        return '\n'.join(lines)
    
    try:
        inharm = _parse_float(args[0]) if len(args) > 0 else 0.01
        partials = _parse_int(args[1]) if len(args) > 1 else 12
        curve = args[2].lower() if len(args) > 2 else 'exp'
        
        if curve not in ('exp', 'linear', 'sqrt'):
            return "ERROR: curve must be exp, linear, or sqrt"
        
        session.engine.set_param(op_idx, 'inharmonicity', inharm)
        session.engine.set_param(op_idx, 'partials', partials)
        session.engine.set_param(op_idx, 'decay_curve', curve)
        
        return f"OK: physical2 params set (inharm={inharm:.4f}, partials={partials}, curve={curve})"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_opinfo(session: Session, args: List[str]) -> str:
    """Show detailed info for an operator.
    
    Usage:
      /opinfo        -> Show info for current operator
      /opinfo <n>    -> Show info for operator n
      /opinfo all    -> Show info for all operators
    """
    if args and args[0].lower() == 'all':
        if not session.engine.operators:
            return "No operators defined."
        
        lines = ["=== ALL OPERATORS ==="]
        for idx in sorted(session.engine.operators.keys()):
            op = session.engine.operators[idx]
            wave = op.get('wave', 'sine')
            freq = op.get('freq', 440.0)
            amp = op.get('amp', 1.0)
            lines.append(f"  [{idx}] {wave:10s} freq={freq:.1f}Hz amp={amp:.2f}")
        
        if session.engine.algorithms:
            lines.append("")
            lines.append("=== ROUTING ===")
            for algo, src, tgt, amt in session.engine.algorithms:
                lines.append(f"  {algo}: op{src} -> op{tgt} (amt={amt:.2f})")
        
        return '\n'.join(lines)
    
    # Single operator
    op_idx = session.current_operator
    if args:
        try:
            op_idx = _parse_int(args[0])
        except:
            pass
    
    params = session.engine.operators.get(op_idx)
    if params is None:
        return f"OPERATOR {op_idx}: (not defined)"
    
    lines = [f"=== OPERATOR {op_idx} ==="]
    lines.append(f"  Wave: {params.get('wave', 'sine')}")
    lines.append(f"  Freq: {params.get('freq', 440.0):.2f} Hz")
    lines.append(f"  Amp: {params.get('amp', 1.0):.3f}")
    lines.append(f"  Phase: {params.get('phase', 0.0):.3f} rad")
    
    wave = params.get('wave', 'sine')
    if wave == 'pulse':
        lines.append(f"  Pulse Width: {params.get('pw', 0.5):.2f}")
    elif wave == 'physical':
        lines.append(f"  Even Harmonics: {params.get('even_harmonics', 8)}")
        lines.append(f"  Odd Harmonics: {params.get('odd_harmonics', 4)}")
        lines.append(f"  Even Weight: {params.get('even_weight', 1.0):.2f}")
        lines.append(f"  Decay: {params.get('decay', 0.7):.2f}")
    elif wave == 'physical2':
        lines.append(f"  Inharmonicity: {params.get('inharmonicity', 0.01):.4f}")
        lines.append(f"  Partials: {params.get('partials', 12)}")
        lines.append(f"  Decay Curve: {params.get('decay_curve', 'exp')}")
    
    return '\n'.join(lines)


# ============================================================================
# MODULATION ALGORITHMS
# ============================================================================

# Modulation algorithms
def _handle_alg(session: Session, args: List[str], algo_name: str) -> str:
    if len(args) < 3:
        return f"ERROR: usage /{algo_name.lower()} <source> <target> <amount>"
    try:
        source = _parse_int(args[0])
        target = _parse_int(args[1])
        amount = _parse_float(args[2])
        session.add_modulation(algo_name, source, target, amount)
        # Provide more verbose feedback: include total algorithm count
        count = len(session.engine.algorithms)
        return f"OK: {algo_name} added ({source}->{target} amount={amount}); total algorithms: {count}"
    except Exception as e:
        return f"ERROR: invalid modulation parameters - {e}"

def cmd_fm(session: Session, args: List[str]) -> str:
    """Frequency Modulation routing.
    
    Usage: /fm <source> <target> <amount>
    
    Source operator modulates target's frequency.
    Classic DX7-style FM synthesis.
    """
    return _handle_alg(session, args, 'FM')

def cmd_tfm(session: Session, args: List[str]) -> str:
    """Through-Zero Frequency Modulation routing.
    
    Usage: /tfm <source> <target> <amount>
    
    More aggressive than standard FM, allows negative frequencies.
    Creates harsher, more metallic timbres.
    """
    return _handle_alg(session, args, 'TFM')

def cmd_am(session: Session, args: List[str]) -> str:
    """Amplitude Modulation routing.
    
    Usage: /am <source> <target> <amount>
    
    Source operator modulates target's amplitude.
    Creates tremolo and sidebands.
    """
    return _handle_alg(session, args, 'AM')

def cmd_rm(session: Session, args: List[str]) -> str:
    """Ring Modulation routing.
    
    Usage: /rm <source> <target> <amount>
    
    Multiplies carrier and modulator directly.
    Creates inharmonic, metallic, bell-like tones.
    """
    return _handle_alg(session, args, 'RM')

def cmd_pm(session: Session, args: List[str]) -> str:
    """Phase Modulation routing.
    
    Usage: /pm <source> <target> <amount>
    
    Source operator directly offsets target's phase.
    Similar to FM but with different harmonic characteristics.
    Commonly used in Yamaha synths (often labeled as FM).
    """
    return _handle_alg(session, args, 'PM')


# Routing inspection
def cmd_rt(session: Session, args: List[str]) -> str:
    """View or add modulation routing.
    
    Usage:
      /rt               -> Show all current routings
      /rt clear         -> Clear all routings
      /rt <s> <t> <type> [amt]  -> Add routing (type: fm/tfm/am/rm/pm)
    
    Examples:
      /rt 1 0 fm 2.5    -> FM: op1 modulates op0, amount=2.5
      /rt 2 1 pm 1.0    -> PM: op2 modulates op1
    """
    if not args:
        # Show current routing
        return session.engine.get_routing_info()
    
    if args[0].lower() == 'clear':
        session.clear_algorithms()
        return "OK: all routings cleared"
    
    if len(args) < 3:
        return ("ERROR: usage /rt <source> <target> <type> [amount]\n"
                "  Types: fm, tfm, am, rm, pm\n"
                "  Example: /rt 1 0 fm 2.5")
    
    try:
        source = _parse_int(args[0])
        target = _parse_int(args[1])
        algo_type = args[2].upper()
        amount = _parse_float(args[3]) if len(args) > 3 else 1.0
        
        valid_types = {'FM', 'TFM', 'AM', 'RM', 'PM'}
        if algo_type not in valid_types:
            return f"ERROR: unknown routing type '{algo_type}'. Valid: {', '.join(valid_types)}"
        
        session.add_modulation(algo_type, source, target, amount)
        count = len(session.engine.algorithms)
        return f"OK: {algo_type} routing added ({source}->{target} amount={amount}); total: {count}"
    except Exception as e:
        return f"ERROR: {e}"


# Clear algorithms
def cmd_clearalg(session: Session, args: List[str]) -> str:
    session.clear_algorithms()
    return "OK: modulation algorithms cleared"


# Tone generation
def _note_to_freq(note: str) -> float:
    """Convert note name to frequency (e.g., 'A4' -> 440.0, 'C4' -> 261.63)."""
    note = note.strip().upper()
    
    # Note names and their semitones from C
    note_map = {
        'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11
    }
    
    # Parse note name
    if len(note) < 2:
        raise ValueError(f"Invalid note: {note}")
    
    # Get base note
    base = note[0]
    if base not in note_map:
        raise ValueError(f"Invalid note: {note}")
    
    semitone = note_map[base]
    rest = note[1:]
    
    # Check for sharps/flats
    if rest.startswith('#') or rest.startswith('S'):
        semitone += 1
        rest = rest[1:]
    elif rest.startswith('B') and len(rest) > 1:  # Flat, not B note
        semitone -= 1
        rest = rest[1:]
    
    # Parse octave
    try:
        octave = int(rest)
    except ValueError:
        raise ValueError(f"Invalid note: {note}")
    
    # Calculate frequency (A4 = 440Hz)
    # A4 is semitone 9 in octave 4
    semitones_from_a4 = (octave - 4) * 12 + (semitone - 9)
    freq = 440.0 * (2 ** (semitones_from_a4 / 12))
    
    return freq


def _parse_freq(arg: str) -> float:
    """Parse frequency from number or note name."""
    arg = arg.strip()
    
    # Try as number first
    try:
        return float(arg)
    except ValueError:
        pass
    
    # Try as note name
    try:
        return _note_to_freq(arg)
    except ValueError:
        pass
    
    raise ValueError(f"Cannot parse frequency: {arg}")


def cmd_tone(session: Session, args: List[str]) -> str:
    """Generate a tone with optional frequency/note, beats and amplitude.

    Usage:
      /tone <freq> <beats> <amp>
      /tone <note> <beats> <amp>
    
    Frequency can be Hz (e.g., 440) or note name (e.g., A4, C#3, Bb2).
    Values omitted default to 440Hz, 1 beat, amplitude 1.
    
    Examples:
      /tone 440 1           -> 440Hz for 1 beat
      /tone A4 2            -> A4 (440Hz) for 2 beats
      /tone C#4 0.5         -> C#4 for half a beat
    """
    import numpy as np
    
    # Defaults
    freq = 440.0
    beats = 1.0
    amp = 1.0
    freq_str = "440"
    
    try:
        if args:
            freq = _parse_freq(args[0])
            freq_str = args[0]
        if len(args) > 1:
            beats = _parse_float(args[1])
        if len(args) > 2:
            amp = _parse_float(args[2])
        session.generate_tone(freq, beats, amp)
        
        # Update working buffer with source tracking
        if session.last_buffer is not None:
            try:
                from .working_cmds import get_working_buffer
                wb = get_working_buffer()
                wb.set_pending(session.last_buffer, f"tone:{freq_str}", session)
            except Exception:
                pass
            
            # Show deviation metrics
            peak = np.max(np.abs(session.last_buffer))
            rms = np.sqrt(np.mean(session.last_buffer ** 2))
            dur = len(session.last_buffer) / session.sample_rate
            
            lines = [f"OK: tone {freq_str} ({freq:.1f}Hz) for {beats} beats"]
            lines.append(f"  {dur:.3f}s, peak={peak:.3f}, rms={rms:.3f}")
            return '\n'.join(lines)
        
        return f"OK: tone generated ({freq_str} = {freq:.1f}Hz for {beats} beats)"
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"


# Shortcut: /n for note (same as /tone)
def cmd_n(session: Session, args: List[str]) -> str:
    """Generate a note (alias for /tone with note-friendly syntax).
    
    Usage:
      /n <note> [beats] [amp]
      /n C4            -> Middle C for 1 beat
      /n A4 2          -> A4 for 2 beats
      /n G#3 0.5 0.8   -> G#3, half beat, 80% amp
    """
    return cmd_tone(session, args)


# Envelope settings - level-aware (global at level 1, per-operator at level 2)
def cmd_atk(session: Session, args: List[str]) -> str:
    level = "OP" if session.synth_level == 2 else "GLOBAL"
    op = session.current_operator
    if not args:
        val = session.get_envelope_param('attack')
        if session.synth_level == 2:
            return f"ATTACK[op{op}]: {val:.3f}s"
        return f"ATTACK[{level}]: {val:.3f}s"
    try:
        val = _parse_float(args[0])
        session.set_attack(val)
        if session.synth_level == 2:
            return f"OK: attack[op{op}] set to {val:.3f}s"
        return f"OK: attack[{level}] set to {val:.3f}s"
    except Exception:
        return "ERROR: invalid attack value"

def cmd_dec(session: Session, args: List[str]) -> str:
    level = "OP" if session.synth_level == 2 else "GLOBAL"
    op = session.current_operator
    if not args:
        val = session.get_envelope_param('decay')
        if session.synth_level == 2:
            return f"DECAY[op{op}]: {val:.3f}s"
        return f"DECAY[{level}]: {val:.3f}s"
    try:
        val = _parse_float(args[0])
        session.set_decay(val)
        if session.synth_level == 2:
            return f"OK: decay[op{op}] set to {val:.3f}s"
        return f"OK: decay[{level}] set to {val:.3f}s"
    except Exception:
        return "ERROR: invalid decay value"

def cmd_sus(session: Session, args: List[str]) -> str:
    level = "OP" if session.synth_level == 2 else "GLOBAL"
    op = session.current_operator
    if not args:
        val = session.get_envelope_param('sustain')
        if session.synth_level == 2:
            return f"SUSTAIN[op{op}]: {val:.3f}"
        return f"SUSTAIN[{level}]: {val:.3f}"
    try:
        val = _parse_float(args[0])
        session.set_sustain(val)
        if session.synth_level == 2:
            return f"OK: sustain[op{op}] set to {val:.3f}"
        return f"OK: sustain[{level}] set to {val:.3f}"
    except Exception:
        return "ERROR: invalid sustain value"

def cmd_rel(session: Session, args: List[str]) -> str:
    level = "OP" if session.synth_level == 2 else "GLOBAL"
    op = session.current_operator
    if not args:
        val = session.get_envelope_param('release')
        if session.synth_level == 2:
            return f"RELEASE[op{op}]: {val:.3f}s"
        return f"RELEASE[{level}]: {val:.3f}s"
    try:
        val = _parse_float(args[0])
        session.set_release(val)
        if session.synth_level == 2:
            return f"OK: release[op{op}] set to {val:.3f}s"
        return f"OK: release[{level}] set to {val:.3f}s"
    except Exception:
        return "ERROR: invalid release value"


def cmd_env(session: Session, args: List[str]) -> str:
    """Set or show envelope (ADSR).
    
    Usage:
      /env                   Show current envelope
      /env <a> <d> <s> <r>   Set ADSR values
      /env preset <name>     Apply preset envelope
    
    Presets: pluck, pad, organ, perc, slow, fast
    
    Examples:
      /env 0.01 0.1 0.7 0.3   Fast attack, short decay
      /env preset pluck       Plucky envelope
    """
    # Envelope presets
    presets = {
        'pluck': (0.001, 0.1, 0.0, 0.1),
        'pad': (0.5, 0.3, 0.7, 1.0),
        'organ': (0.01, 0.01, 1.0, 0.1),
        'perc': (0.001, 0.2, 0.0, 0.05),
        'slow': (1.0, 0.5, 0.6, 2.0),
        'fast': (0.001, 0.05, 0.8, 0.1),
        'string': (0.1, 0.2, 0.8, 0.5),
        'brass': (0.05, 0.1, 0.9, 0.2),
    }
    
    if not args:
        a = session.get_envelope_param('attack')
        d = session.get_envelope_param('decay')
        s = session.get_envelope_param('sustain')
        r = session.get_envelope_param('release')
        return f"ENVELOPE: A={a:.3f}s D={d:.3f}s S={s:.2f} R={r:.3f}s"
    
    # Check for preset
    if args[0].lower() == 'preset' and len(args) > 1:
        name = args[1].lower()
        if name not in presets:
            return f"ERROR: unknown preset. Available: {', '.join(presets.keys())}"
        a, d, s, r = presets[name]
        session.set_attack(a)
        session.set_decay(d)
        session.set_sustain(s)
        session.set_release(r)
        return f"OK: envelope preset '{name}' applied (A={a:.3f} D={d:.3f} S={s:.2f} R={r:.3f})"
    
    # Direct preset name
    if args[0].lower() in presets:
        a, d, s, r = presets[args[0].lower()]
        session.set_attack(a)
        session.set_decay(d)
        session.set_sustain(s)
        session.set_release(r)
        return f"OK: envelope preset '{args[0]}' applied"
    
    # Parse ADSR values
    try:
        a = float(args[0]) if len(args) > 0 else 0.01
        d = float(args[1]) if len(args) > 1 else 0.1
        s = float(args[2]) if len(args) > 2 else 0.7
        r = float(args[3]) if len(args) > 3 else 0.3
        
        session.set_attack(max(0.001, a))
        session.set_decay(max(0.001, d))
        session.set_sustain(max(0, min(1, s)))
        session.set_release(max(0.001, r))
        
        return f"OK: envelope A={a:.3f} D={d:.3f} S={s:.2f} R={r:.3f}"
    except ValueError:
        return "ERROR: invalid ADSR values"


# Alias for envelope
cmd_adsr = cmd_env
cmd_e = cmd_env


def cmd_key(session: Session, args: List[str]) -> str:
    """Set or show musical key/scale.
    
    Usage:
      /key              Show current key
      /key <note>       Set key (C, D, E, F, G, A, B + optional #/b)
      /key <note> <scale>  Set key and scale
    
    Scales: major, minor, dorian, phrygian, lydian, mixolydian, locrian, 
            pentatonic, blues, harmonic, melodic
    
    Examples:
      /key Am           A minor
      /key C major      C major
      /key F# dorian    F# dorian
    """
    if not hasattr(session, 'musical_key'):
        session.musical_key = 'C'
    if not hasattr(session, 'musical_scale'):
        session.musical_scale = 'major'
    
    if not args:
        return f"KEY: {session.musical_key} {session.musical_scale}"
    
    key_input = args[0].upper()
    
    # Parse note and possible minor indicator
    if key_input.endswith('M'):
        key = key_input[:-1]
        scale = 'minor'
    elif key_input.endswith('MAJ'):
        key = key_input[:-3]
        scale = 'major'
    elif key_input.endswith('MIN'):
        key = key_input[:-3]
        scale = 'minor'
    else:
        key = key_input
        scale = args[1].lower() if len(args) > 1 else session.musical_scale
    
    # Validate key
    valid_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 
                   'C#', 'D#', 'F#', 'G#', 'A#',
                   'DB', 'EB', 'GB', 'AB', 'BB']
    if key not in valid_notes:
        return f"ERROR: invalid key '{key}'"
    
    valid_scales = ['major', 'minor', 'dorian', 'phrygian', 'lydian', 
                    'mixolydian', 'locrian', 'pentatonic', 'blues',
                    'harmonic', 'melodic']
    if scale not in valid_scales:
        return f"ERROR: invalid scale '{scale}'. Use: {', '.join(valid_scales)}"
    
    session.musical_key = key
    session.musical_scale = scale
    
    return f"OK: key set to {key} {scale}"


# Filter management commands
def cmd_fcount(session: Session, args: List[str]) -> str:
    """Get or set filter count (number of filter slots).
    
    Usage:
      /fcount         -> show current filter count
      /fcount <n>     -> set filter count (1-8)
    """
    if not args:
        return f"FILTER COUNT: {session.filter_count}"
    try:
        val = _parse_int(args[0])
        session.set_filter_count(val)
        return f"OK: filter count set to {session.filter_count}"
    except Exception as e:
        return f"ERROR: {e}"

def cmd_ft(session: Session, args: List[str]) -> str:
    """Get or set filter type for the currently selected filter slot.
    
    Usage:
      /ft              -> show current filter type
      /ft <type>       -> set filter type (index, alias, or name)
      /ft list         -> list available filter types
      /ft none         -> disable filter (same as /ft off, /ft -1)
    
    Filter types (30 total):
      Basic:     0=lp, 1=hp, 2=bp, 3=notch, 4=peak, 5=ring, 6=allpass
      Comb:      7=comb_ff, 8=comb_fb, 9=comb_both
      Analog:    10=analog, 11=acid (303-style)
      Formant:   12=fa(ah), 13=fe(eh), 14=fi(ee), 15=fo(oh), 16=fu(oo)
      Shelf:     17=lowshelf, 18=highshelf
      Moog/SVF:  19=moog, 20=svf_lp, 21=svf_hp, 22=svf_bp
      Destroy:   23=bitcrush, 24=downsample
      Utility:   25=dc_block, 26=tilt
      Character: 27=resonant, 28=vocal, 29=telephone
    
    Special values: none, off, -1, disable, bypass -> disable filter
    """
    if not args:
        slot = session.selected_filter
        f_type = session.filter_types.get(slot, 0)
        f_name = session.filter_type_names.get(f_type, 'unknown')
        enabled = session.filter_enabled.get(slot, False)
        return f"FILTER[{slot}]: type={f_type} ({f_name}), enabled={enabled}"
    
    arg = args[0].lower()
    
    if arg == 'list':
        lines = ["Available filter types:"]
        for idx, name in session.filter_type_names.items():
            aliases = [k for k, v in session.filter_type_aliases.items() if v == idx]
            alias_str = ', '.join(aliases[:3])
            lines.append(f"  {idx}: {name} ({alias_str})")
        lines.append("")
        lines.append("Special: none, off, -1, disable, bypass -> disable filter")
        return '\n'.join(lines)
    
    try:
        session.set_filter_type(args[0])
        slot = session.selected_filter
        
        # Check if filter was disabled
        if not session.filter_enabled.get(slot, False):
            return f"OK: filter[{slot}] disabled"
        
        f_type = session.filter_types.get(slot, 0)
        f_name = session.filter_type_names.get(f_type, 'unknown')
        return f"OK: filter[{slot}] type set to {f_type} ({f_name})"
    except Exception as e:
        return f"ERROR: {e}\nUse /ft list to see available types"

def cmd_fs(session: Session, args: List[str]) -> str:
    """Select a filter slot for editing.
    
    Usage:
      /fs         -> show current filter slot
      /fs <n>     -> select filter slot n
    """
    if not args:
        slot = session.selected_filter
        f_type = session.filter_types.get(slot, 0)
        f_name = session.filter_type_names.get(f_type, 'unknown')
        cutoff = session.filter_cutoffs.get(slot, 1000.0)
        res = session.filter_resonances.get(slot, 0.5)
        enabled = session.filter_enabled.get(slot, False)
        return f"FILTER SLOT: {slot} | type={f_name} cutoff={cutoff:.0f}Hz Q={res:.2f} enabled={enabled}"
    try:
        idx = _parse_int(args[0])
        session.select_filter(idx)
        return f"OK: filter slot set to {session.selected_filter}"
    except Exception as e:
        return f"ERROR: {e}"

def cmd_fe(session: Session, args: List[str]) -> str:
    """Select a filter envelope slot for editing.
    
    Usage:
      /fe         -> show current filter envelope slot and values
      /fe <n>     -> select filter envelope slot n
    """
    if not args:
        slot = session.selected_filter_envelope
        env = session.get_filter_envelope(slot)
        return (f"FILTER ENV[{slot}]: A={env['attack']:.3f}s D={env['decay']:.3f}s "
                f"S={env['sustain']:.3f} R={env['release']:.3f}s")
    try:
        idx = _parse_int(args[0])
        session.select_filter_envelope(idx)
        slot = session.selected_filter_envelope
        env = session.get_filter_envelope(slot)
        return (f"OK: filter envelope slot set to {slot} | "
                f"A={env['attack']:.3f} D={env['decay']:.3f} S={env['sustain']:.3f} R={env['release']:.3f}")
    except Exception as e:
        return f"ERROR: {e}"

def cmd_fen(session: Session, args: List[str]) -> str:
    """Enable or disable the currently selected filter.
    
    Usage:
      /fen            -> show filter enabled state
      /fen 1          -> enable filter (also: on, yes, true)
      /fen 0          -> disable filter (also: off, no, false, none)
      /fen toggle     -> toggle current state
    """
    slot = session.selected_filter
    if not args:
        enabled = session.filter_enabled.get(slot, False)
        return f"FILTER[{slot}] ENABLED: {enabled}"
    
    arg = args[0].lower()
    
    # Toggle mode
    if arg == 'toggle':
        current = session.filter_enabled.get(slot, False)
        session.enable_filter(not current)
        return f"OK: filter[{slot}] {'enabled' if not current else 'disabled'}"
    
    # Enable values
    if arg in ('1', 'true', 'on', 'yes', 'enable', 'enabled'):
        session.enable_filter(True)
        return f"OK: filter[{slot}] enabled"
    
    # Disable values
    if arg in ('0', 'false', 'off', 'no', 'none', 'disable', 'disabled'):
        session.enable_filter(False)
        return f"OK: filter[{slot}] disabled"
    
    return f"ERROR: unknown value '{arg}'\nUse: 1/on/yes to enable, 0/off/no/none to disable, or toggle"

def cmd_cut(session: Session, args: List[str]) -> str:
    """Get or set cutoff frequency for current filter slot.
    
    Usage:
      /cut            -> show current cutoff
      /cut <freq>     -> set cutoff in Hz (20-20000)
      /cut none       -> reset to default (1000 Hz)
    """
    slot = session.selected_filter
    if not args:
        cutoff = session.filter_cutoffs.get(slot, 1000.0)
        return f"CUTOFF[{slot}]: {cutoff:.1f} Hz"
    
    arg = args[0].lower()
    
    # Handle reset/none values
    if arg in ('none', 'reset', 'default', 'off'):
        session.filter_cutoffs[slot] = 1000.0
        return f"OK: filter[{slot}] cutoff reset to 1000.0 Hz"
    
    try:
        val = _parse_float(args[0])
        session.set_cutoff(val)
        return f"OK: filter[{slot}] cutoff set to {session.filter_cutoffs[slot]:.1f} Hz"
    except Exception as e:
        return f"ERROR: {e}\nUsage: /cut <freq> (20-20000 Hz) or /cut none to reset"

def cmd_res(session: Session, args: List[str]) -> str:
    """Get or set resonance/Q for current filter slot.
    
    Usage:
      /res            -> show current resonance
      /res <q>        -> set resonance (0-20)
      /res none       -> reset to default (0.5)
    """
    slot = session.selected_filter
    if not args:
        res = session.filter_resonances.get(slot, 0.5)
        return f"RESONANCE[{slot}]: {res:.2f}"
    
    arg = args[0].lower()
    
    # Handle reset/none values
    if arg in ('none', 'reset', 'default', 'off'):
        session.filter_resonances[slot] = 0.5
        return f"OK: filter[{slot}] resonance reset to 0.50"
    
    try:
        val = _parse_float(args[0])
        session.set_resonance(val)
        return f"OK: filter[{slot}] resonance set to {session.filter_resonances[slot]:.2f}"
    except Exception as e:
        return f"ERROR: {e}\nUsage: /res <q> (0-20) or /res none to reset"

# Filter envelope commands - use new bank system
def cmd_fatk(session: Session, args: List[str]) -> str:
    """Get or set filter attack time for selected filter envelope slot."""
    slot = session.selected_filter_envelope
    env = session.get_filter_envelope(slot)
    if not args:
        return f"F-ATTACK[{slot}]: {env['attack']:.3f}s"
    try:
        val = _parse_float(args[0])
        session.set_f_attack(val)
        return f"OK: filter[{slot}] attack set to {val:.3f}s"
    except Exception:
        return "ERROR: invalid filter attack value"

def cmd_fdec(session: Session, args: List[str]) -> str:
    """Get or set filter decay time for selected filter envelope slot."""
    slot = session.selected_filter_envelope
    env = session.get_filter_envelope(slot)
    if not args:
        return f"F-DECAY[{slot}]: {env['decay']:.3f}s"
    try:
        val = _parse_float(args[0])
        session.set_f_decay(val)
        return f"OK: filter[{slot}] decay set to {val:.3f}s"
    except Exception:
        return "ERROR: invalid filter decay value"

def cmd_fsus(session: Session, args: List[str]) -> str:
    """Get or set filter sustain level for selected filter envelope slot."""
    slot = session.selected_filter_envelope
    env = session.get_filter_envelope(slot)
    if not args:
        return f"F-SUSTAIN[{slot}]: {env['sustain']:.3f}"
    try:
        val = _parse_float(args[0])
        session.set_f_sustain(val)
        return f"OK: filter[{slot}] sustain set to {val:.3f}"
    except Exception:
        return "ERROR: invalid filter sustain value"

def cmd_frel(session: Session, args: List[str]) -> str:
    """Get or set filter release time for selected filter envelope slot."""
    slot = session.selected_filter_envelope
    env = session.get_filter_envelope(slot)
    if not args:
        return f"F-RELEASE[{slot}]: {env['release']:.3f}s"
    try:
        val = _parse_float(args[0])
        session.set_f_release(val)
        return f"OK: filter[{slot}] release set to {val:.3f}s"
    except Exception:
        return "ERROR: invalid filter release value"

# Voice algorithm parameter commands
def cmd_dt(session: Session, args: List[str]) -> str:
    """Get or set voice detune amount (Hz)."""
    if not args:
        return f"DT: {session.dt:.3f} Hz"
    try:
        val = _parse_float(args[0])
        session.set_dt(val)
        return f"OK: detune set to {session.dt:.3f} Hz"
    except Exception:
        return "ERROR: invalid detune value"

def cmd_rand(session: Session, args: List[str]) -> str:
    """Set voice randomness (0-100).

    In voice algorithm 0 (stack): only varies amplitude.
    In voice algorithm 1 (unison) / 2 (wide): also randomizes
    per-voice phase to prevent phase-locking.
    """
    if not args:
        return f"RAND: {session.rand:.3f}"
    try:
        val = _parse_float(args[0])
        session.set_rand(val)
        return f"OK: random variation set to {session.rand:.3f}"
    except Exception:
        return "ERROR: invalid random variation value"

def cmd_vmod(session: Session, args: List[str]) -> str:
    """Get or set modulation scaling factor for voices."""
    if not args:
        return f"VMOD: {session.v_mod:.3f}"
    try:
        val = _parse_float(args[0])
        session.set_mod(val)
        return f"OK: voice modulation scaling set to {session.v_mod:.3f}"
    except Exception:
        return "ERROR: invalid voice modulation value"

def cmd_hf(session: Session, args: List[str]) -> str:
    """Show comprehensive filter help and current settings.
    
    Usage:
      /hf           -> Show all filter commands and current settings
      /hf all       -> Show settings for all filter slots
      /hf types     -> List all filter types with aliases
    """
    if args and args[0].lower() == 'types':
        lines = [
            "=== FILTER TYPES (30 total) ===",
            "",
            "BASIC (0-6):",
        ]
        for idx in range(7):
            name = session.filter_type_names.get(idx, 'unknown')
            aliases = [k for k, v in session.filter_type_aliases.items() if v == idx]
            alias_str = ', '.join(sorted(aliases)[:4])
            lines.append(f"  {idx:2d}: {name:12s} ({alias_str})")
        
        lines.append("\nCOMB (7-9):")
        for idx in range(7, 10):
            name = session.filter_type_names.get(idx, 'unknown')
            aliases = [k for k, v in session.filter_type_aliases.items() if v == idx]
            alias_str = ', '.join(sorted(aliases)[:4])
            lines.append(f"  {idx:2d}: {name:12s} ({alias_str})")
        
        lines.append("\nANALOG (10-11):")
        for idx in range(10, 12):
            name = session.filter_type_names.get(idx, 'unknown')
            aliases = [k for k, v in session.filter_type_aliases.items() if v == idx]
            alias_str = ', '.join(sorted(aliases)[:4])
            lines.append(f"  {idx:2d}: {name:12s} ({alias_str})")
        
        lines.append("\nFORMANT (12-16):")
        for idx in range(12, 17):
            name = session.filter_type_names.get(idx, 'unknown')
            aliases = [k for k, v in session.filter_type_aliases.items() if v == idx]
            alias_str = ', '.join(sorted(aliases)[:4])
            lines.append(f"  {idx:2d}: {name:12s} ({alias_str})")
        
        lines.append("\nSHELF (17-18):")
        for idx in range(17, 19):
            name = session.filter_type_names.get(idx, 'unknown')
            aliases = [k for k, v in session.filter_type_aliases.items() if v == idx]
            alias_str = ', '.join(sorted(aliases)[:4])
            lines.append(f"  {idx:2d}: {name:12s} ({alias_str})")
        
        lines.append("\nMOOG/SVF (19-22):")
        for idx in range(19, 23):
            name = session.filter_type_names.get(idx, 'unknown')
            aliases = [k for k, v in session.filter_type_aliases.items() if v == idx]
            alias_str = ', '.join(sorted(aliases)[:4])
            lines.append(f"  {idx:2d}: {name:12s} ({alias_str})")
        
        lines.append("\nDESTRUCTIVE (23-24):")
        for idx in range(23, 25):
            name = session.filter_type_names.get(idx, 'unknown')
            aliases = [k for k, v in session.filter_type_aliases.items() if v == idx]
            alias_str = ', '.join(sorted(aliases)[:4])
            lines.append(f"  {idx:2d}: {name:12s} ({alias_str})")
        
        lines.append("\nUTILITY (25-26):")
        for idx in range(25, 27):
            name = session.filter_type_names.get(idx, 'unknown')
            aliases = [k for k, v in session.filter_type_aliases.items() if v == idx]
            alias_str = ', '.join(sorted(aliases)[:4])
            lines.append(f"  {idx:2d}: {name:12s} ({alias_str})")
        
        lines.append("\nCHARACTER (27-29):")
        for idx in range(27, 30):
            name = session.filter_type_names.get(idx, 'unknown')
            aliases = [k for k, v in session.filter_type_aliases.items() if v == idx]
            alias_str = ', '.join(sorted(aliases)[:4])
            lines.append(f"  {idx:2d}: {name:12s} ({alias_str})")
        
        lines.append("")
        lines.append("Special: none, off, -1, disable, bypass -> disable filter")
        return '\n'.join(lines)
    
    if args and args[0].lower() == 'all':
        lines = ["=== ALL FILTER SLOTS ==="]
        for slot in range(session.filter_count):
            f_type = session.filter_types.get(slot, 0)
            f_name = session.filter_type_names.get(f_type, 'unknown')
            cutoff = session.filter_cutoffs.get(slot, 1000.0)
            res = session.filter_resonances.get(slot, 0.5)
            enabled = session.filter_enabled.get(slot, False)
            status = "ON " if enabled else "OFF"
            mark = "*" if slot == session.selected_filter else " "
            lines.append(f"  [{slot}]{mark} {status} {f_name:12s} cut={cutoff:6.0f}Hz Q={res:.2f}")
        return '\n'.join(lines)
    
    # Default: show help and current slot settings
    slot = session.selected_filter
    f_type = session.filter_types.get(slot, 0)
    f_name = session.filter_type_names.get(f_type, 'unknown')
    cutoff = session.filter_cutoffs.get(slot, 1000.0)
    res = session.filter_resonances.get(slot, 0.5)
    enabled = session.filter_enabled.get(slot, False)
    
    lines = [
        "=== FILTER COMMANDS ===",
        "",
        "Selection:",
        "  /fs <n>         Select filter slot (0-7)",
        "  /fc <n>         Set filter count (1-8)",
        "",
        "Type & Enable:",
        "  /ft <type>      Set filter type (0-29, alias, or 'none' to disable)",
        "  /ft list        List all filter types",
        "  /fen <0|1>      Enable/disable filter (also: on/off/toggle)",
        "",
        "Parameters:",
        "  /cut <Hz>       Set cutoff frequency (20-20000, or 'none' to reset)",
        "  /res <Q>        Set resonance (0-20, or 'none' to reset)",
        "",
        "Envelope:",
        "  /fe <n>         Select filter envelope slot",
        "  /fatk <s>       Set filter envelope attack",
        "  /fdec <s>       Set filter envelope decay",
        "  /fsus <0-1>     Set filter envelope sustain",
        "  /frel <s>       Set filter envelope release",
        "",
        f"=== CURRENT SLOT [{slot}] ===",
        f"  Type:      {f_type} ({f_name})",
        f"  Cutoff:    {cutoff:.1f} Hz",
        f"  Resonance: {res:.2f}",
        f"  Enabled:   {enabled}",
        "",
        "Use /hf all to see all slots, /hf types to see all 30 filter types"
    ]
    return '\n'.join(lines)


# ============================================================================
# ROUTING BANK COMMANDS (Section E)
# ============================================================================

def cmd_bk(session: Session, args: List[str]) -> str:
    """Routing bank selection and management.
    
    Usage:
      /bk                -> Show current bank and list available
      /bk <name>         -> Select bank by name or alias
      /bk list           -> List all available banks
      /bk info <name>    -> Show bank details
      /bk save <name>    -> Save current routings as user bank
    
    Factory Banks:
      classic_fm (dx7, fm)     - Classic DX7-style FM
      modern_aggressive (bass) - Modern bass music
      thru_zero_fm (tfm)       - Through-zero FM
      phase_mod (pm, casio)    - Phase modulation
      ring_am (rm, ring)       - Ring/amplitude mod
      hybrid (mix)             - Mixed modulation
      physical (phys)          - Physical modeling
      experimental (exp, glitch) - Experimental
    """
    from ..core.banks import (
        get_bank, list_bank_names, list_bank_algorithms,
        BANK_ALIASES, FACTORY_BANKS
    )
    from ..core.user_data import list_banks, save_bank
    
    # Ensure current_bank attribute exists
    if not hasattr(session, 'current_bank'):
        session.current_bank = 'classic_fm'
    if not hasattr(session, 'current_algorithm'):
        session.current_algorithm = 0
    
    if not args:
        # Show current bank and list
        bank = get_bank(session.current_bank)
        bank_name = bank.get('name', session.current_bank) if bank else session.current_bank
        bank_desc = bank.get('description', '') if bank else ''
        
        lines = [
            "=== ROUTING BANKS ===",
            f"Current: {bank_name}",
            f"  {bank_desc}",
            "",
            "Factory Banks:",
        ]
        
        for name in list_bank_names():
            b = FACTORY_BANKS.get(name, {})
            algo_count = len(b.get('algorithms', []))
            vibe = b.get('vibe', '')
            lines.append(f"  {name} ({algo_count} algos) - {vibe}")
        
        # List user banks
        user_banks = [b for b in list_banks() if not b.get('factory')]
        if user_banks:
            lines.append("")
            lines.append("User Banks:")
            for b in user_banks:
                lines.append(f"  {b['name']} ({b.get('algorithm_count', 0)} algos)")
        
        lines.append("")
        lines.append("Use /bk <name> to select, /al to choose algorithm")
        
        return '\n'.join(lines)
    
    cmd = args[0].lower()
    
    if cmd == 'list':
        lines = ["=== ALL BANKS ===", ""]
        lines.append("Factory:")
        for name in list_bank_names():
            aliases = [k for k, v in BANK_ALIASES.items() if v == name]
            alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
            lines.append(f"  {name}{alias_str}")
        
        user_banks = [b for b in list_banks() if not b.get('factory')]
        if user_banks:
            lines.append("")
            lines.append("User:")
            for b in user_banks:
                lines.append(f"  {b['name']}")
        
        return '\n'.join(lines)
    
    elif cmd == 'info' and len(args) > 1:
        bank_name = args[1]
        bank = get_bank(bank_name)
        if not bank:
            return f"ERROR: bank '{bank_name}' not found"
        
        lines = [
            f"=== BANK: {bank.get('name', bank_name)} ===",
            f"Description: {bank.get('description', 'N/A')}",
            f"Vibe: {bank.get('vibe', 'N/A')}",
            "",
            "Algorithms:"
        ]
        
        for i, algo in enumerate(bank.get('algorithms', [])):
            lines.append(f"  [{i}] {algo.get('name', f'algo_{i}')} - {algo.get('description', '')}")
        
        return '\n'.join(lines)
    
    elif cmd == 'save' and len(args) > 1:
        bank_name = args[1]
        # Get current engine routings
        engine = session.mono
        if not engine.algorithms:
            return "ERROR: no routings to save. Add routings with /fm, /am, /pm, etc."
        
        # Build bank data
        bank_data = {
            'name': bank_name,
            'description': f'User bank created from current session',
            'vibe': 'Custom',
            'algorithms': [{
                'name': 'custom_1',
                'description': 'User-defined routing',
                'operator_count': len(engine.operators),
                'carriers': list(range(session.carrier_count)),
                'routings': [
                    {'type': t, 'src': s, 'dst': d, 'amount': a * 10}  # Convert back to 0-100
                    for t, s, d, a in engine.algorithms
                ],
                'wave_hints': {
                    i: op.get('wave', 'sine')
                    for i, op in engine.operators.items()
                }
            }]
        }
        
        if save_bank(bank_name, bank_data):
            return f"OK: saved bank '{bank_name}' with current routing"
        else:
            return f"ERROR: failed to save bank '{bank_name}'"
    
    else:
        # Select bank by name
        bank = get_bank(cmd)
        if not bank:
            return f"ERROR: bank '{cmd}' not found. Use /bk list to see available banks."
        
        session.current_bank = bank.get('name', cmd)
        session.current_algorithm = 0
        
        algo_count = len(bank.get('algorithms', []))
        return f"OK: selected bank '{session.current_bank}' ({algo_count} algorithms). Use /al to select algorithm."


def cmd_al(session: Session, args: List[str]) -> str:
    """Algorithm selection from current bank.
    
    Usage:
      /al                -> Show current algorithm
      /al <n>            -> Load algorithm by index
      /al <name>         -> Load algorithm by name
      /al list           -> List algorithms in current bank
      /al apply          -> Apply algorithm to engine
    
    Examples:
      /al 0              -> Load first algorithm
      /al bell           -> Load algorithm named 'bell'
      /al list           -> See all algorithms in bank
    """
    from ..core.banks import (
        get_bank, get_algorithm, get_algorithm_by_name,
        list_bank_algorithms, apply_algorithm_to_engine
    )
    
    # Ensure attributes exist
    if not hasattr(session, 'current_bank'):
        session.current_bank = 'classic_fm'
    if not hasattr(session, 'current_algorithm'):
        session.current_algorithm = 0
    if not hasattr(session, 'current_algorithm_data'):
        session.current_algorithm_data = None
    
    bank = get_bank(session.current_bank)
    if not bank:
        return f"ERROR: bank '{session.current_bank}' not found. Use /bk to select a bank."
    
    if not args:
        # Show current algorithm
        if session.current_algorithm_data:
            algo = session.current_algorithm_data
            lines = [
                f"=== ALGORITHM: {algo.get('name', 'unnamed')} ===",
                f"Bank: {session.current_bank}",
                f"Index: {session.current_algorithm}",
                f"Description: {algo.get('description', 'N/A')}",
                f"Operators: {algo.get('operator_count', 2)}",
                f"Carriers: {algo.get('carriers', [0])}",
                f"Routings: {len(algo.get('routings', []))}",
            ]
            return '\n'.join(lines)
        else:
            return f"ALGORITHM: none selected. Bank: {session.current_bank}. Use /al <n> to select."
    
    cmd = args[0].lower()
    
    if cmd == 'list':
        algos = list_bank_algorithms(session.current_bank)
        if not algos:
            return f"ERROR: bank '{session.current_bank}' has no algorithms"
        
        lines = [f"=== ALGORITHMS IN {session.current_bank.upper()} ===", ""]
        for idx, name, desc in algos:
            marker = " *" if idx == session.current_algorithm and session.current_algorithm_data else ""
            lines.append(f"  [{idx}] {name}{marker}")
            if desc:
                lines.append(f"      {desc}")
        
        lines.append("")
        lines.append("Use /al <n> to load, /al apply to apply to engine")
        return '\n'.join(lines)
    
    elif cmd == 'apply':
        if not session.current_algorithm_data:
            return "ERROR: no algorithm selected. Use /al <n> first."
        
        result = apply_algorithm_to_engine(session.mono, session.current_algorithm_data)
        
        # Update session carrier count based on algorithm
        carriers = session.current_algorithm_data.get('carriers', [0])
        session.set_carrier_count(len(carriers))
        
        return f"OK: {result}"
    
    else:
        # Try to load by index or name
        algo = None
        try:
            idx = int(cmd)
            algo = get_algorithm(session.current_bank, idx)
            if algo:
                session.current_algorithm = idx
        except ValueError:
            # Try by name
            algo = get_algorithm_by_name(session.current_bank, cmd)
            if algo:
                # Find index
                for i, (_, name, _) in enumerate(list_bank_algorithms(session.current_bank)):
                    if name.lower() == cmd:
                        session.current_algorithm = i
                        break
        
        if not algo:
            return f"ERROR: algorithm '{cmd}' not found in bank '{session.current_bank}'"
        
        session.current_algorithm_data = algo
        
        # Auto-apply if requested or show info
        lines = [
            f"OK: loaded algorithm '{algo.get('name', 'unnamed')}'",
            f"  {algo.get('description', '')}",
            f"  Operators: {algo.get('operator_count', 2)}, Carriers: {algo.get('carriers', [0])}",
            "",
            "Use /al apply to apply to engine, or /tone to test"
        ]
        return '\n'.join(lines)


def cmd_preset(session: Session, args: List[str]) -> str:
    """Load synth preset from preset library.
    
    Usage:
      /preset                  -> Show preset categories
      /preset <category>       -> List presets in category
      /preset <category> <n>   -> Load preset
      /preset <name>           -> Load preset by name
      /preset save <name>      -> Save current as preset
    
    Categories: bass, lead, pad, pluck, perc, atmos, noise, fx, physical
    """
    from ..core.user_data import list_presets, load_preset, save_preset
    
    categories = ['bass', 'lead', 'pad', 'pluck', 'perc', 'atmos', 'noise', 'fx', 'physical']
    
    if not args:
        lines = [
            "=== PRESETS ===",
            "",
            "Categories:",
        ]
        for cat in categories:
            count = len(list_presets(category=cat))
            lines.append(f"  {cat}: {count} presets")
        
        lines.append("")
        lines.append("Use /preset <category> to list, /preset <cat> <n> to load")
        return '\n'.join(lines)
    
    cmd = args[0].lower()
    
    if cmd == 'save' and len(args) > 1:
        name = args[1]
        category = args[2] if len(args) > 2 else 'misc'
        
        # Build preset from current session
        preset_data = {
            'name': name,
            'category': category,
            'description': 'User preset',
            'bank': getattr(session, 'current_bank', 'classic_fm'),
            'algorithm': getattr(session, 'current_algorithm', 0),
            'carrier_count': session.carrier_count,
            'voice_count': session.voice_count,
            'operators': {
                str(i): {
                    'wave': op.get('wave', 'sine'),
                    'freq': op.get('freq', 440.0),
                    'amp': op.get('amp', 1.0),
                }
                for i, op in session.mono.operators.items()
            },
            'envelope': {
                'attack': session.attack,
                'decay': session.decay,
                'sustain': session.sustain,
                'release': session.release,
            }
        }
        
        if save_preset(name, preset_data):
            return f"OK: saved preset '{name}' in category '{category}'"
        else:
            return f"ERROR: failed to save preset"
    
    elif cmd in categories:
        # List presets in category
        presets = list_presets(category=cmd)
        if not presets:
            return f"No presets in category '{cmd}'"
        
        if len(args) > 1:
            # Load by index
            try:
                idx = int(args[1])
                if 0 <= idx < len(presets):
                    preset_name = presets[idx]['name']
                    preset = load_preset(preset_name)
                    if preset:
                        # Apply preset (simplified)
                        return f"OK: loaded preset '{preset_name}' - use /al apply if algorithm-based"
                    else:
                        return f"ERROR: could not load preset '{preset_name}'"
                else:
                    return f"ERROR: index {idx} out of range (0-{len(presets)-1})"
            except ValueError:
                return f"ERROR: invalid index '{args[1]}'"
        else:
            # List category
            lines = [f"=== {cmd.upper()} PRESETS ===", ""]
            for i, p in enumerate(presets):
                factory = " [F]" if p.get('factory') else ""
                lines.append(f"  [{i}] {p['name']}{factory}")
            return '\n'.join(lines)
    
    else:
        # Try to load by name directly
        preset = load_preset(cmd)
        if preset:
            return f"OK: loaded preset '{cmd}'"
        else:
            return f"ERROR: preset '{cmd}' not found. Use /preset to see categories."


# ============================================================================
# VOICE PARAMETER COMMANDS
# ============================================================================

def cmd_stereo(session: Session, args: List[str]) -> str:
    """Set voice stereo spread.
    
    Usage:
      /stereo           Show current spread
      /stereo <0-100>   Set spread (0=mono, 100=full width)
    
    Spreads multiple voices across the stereo field.
    """
    if not hasattr(session, 'stereo_spread'):
        session.stereo_spread = 0.0
    
    if not args:
        return f"STEREO: {session.stereo_spread:.0f}%"
    
    try:
        spread = float(args[0])
        spread = max(0, min(100, spread))
        session.stereo_spread = spread
        
        if spread == 0:
            return "OK: stereo spread 0% (mono)"
        elif spread < 30:
            return f"OK: stereo spread {spread:.0f}% (narrow)"
        elif spread < 70:
            return f"OK: stereo spread {spread:.0f}% (moderate)"
        else:
            return f"OK: stereo spread {spread:.0f}% (wide)"
    except ValueError:
        return f"ERROR: invalid spread value '{args[0]}'"


def cmd_vphase(session: Session, args: List[str]) -> str:
    """Set voice phase offset.
    
    Usage:
      /vphase           Show current offset
      /vphase <0-360>   Set phase offset per voice (degrees)
    
    Each voice gets cumulative phase offset for thickness.
    """
    if not hasattr(session, 'voice_phase_offset'):
        session.voice_phase_offset = 0.0
    
    if not args:
        return f"VOICE PHASE: {session.voice_phase_offset:.1f}°"
    
    try:
        offset = float(args[0])
        offset = offset % 360  # Wrap to valid range
        session.voice_phase_offset = offset
        
        if offset == 0:
            return "OK: voice phase offset 0° (in phase)"
        elif offset < 45:
            return f"OK: voice phase offset {offset:.1f}° (slight)"
        elif offset < 90:
            return f"OK: voice phase offset {offset:.1f}° (moderate)"
        else:
            return f"OK: voice phase offset {offset:.1f}° (heavy)"
    except ValueError:
        return f"ERROR: invalid phase value '{args[0]}'"


def cmd_venv(session: Session, args: List[str]) -> str:
    """Set voice envelope offset.
    
    Usage:
      /venv            Show current offset
      /venv <ms>       Set envelope start offset per voice (milliseconds)
    
    Each voice envelope starts progressively later for smearing/humanization.
    """
    if not hasattr(session, 'voice_env_offset'):
        session.voice_env_offset = 0.0
    
    if not args:
        return f"VOICE ENV OFFSET: {session.voice_env_offset:.1f}ms"
    
    try:
        offset = float(args[0])
        offset = max(0, min(100, offset))  # Clamp to reasonable range
        session.voice_env_offset = offset
        
        if offset == 0:
            return "OK: voice envelope offset 0ms (sync)"
        elif offset < 10:
            return f"OK: voice envelope offset {offset:.1f}ms (subtle)"
        elif offset < 30:
            return f"OK: voice envelope offset {offset:.1f}ms (moderate)"
        else:
            return f"OK: voice envelope offset {offset:.1f}ms (smeared)"
    except ValueError:
        return f"ERROR: invalid offset value '{args[0]}'"


def cmd_fenv(session: Session, args: List[str]) -> str:
    """Set frequency envelope (pitch envelope).
    
    Usage:
      /fenv                    Show current settings
      /fenv <a> <d> <s> <r>    Set ADSR (attack, decay, sustain, release)
      /fenv amount <semitones> Set modulation amount
      /fenv off                Disable frequency envelope
    
    Modulates operator frequency over time.
    """
    if not hasattr(session, 'freq_env'):
        session.freq_env = {
            'enabled': False,
            'attack': 0.01,
            'decay': 0.1,
            'sustain': 0.0,
            'release': 0.1,
            'amount': 12.0  # semitones
        }
    
    if not args:
        env = session.freq_env
        if not env['enabled']:
            return "FREQ ENV: off"
        return (f"FREQ ENV: A={env['attack']:.3f}s D={env['decay']:.3f}s "
                f"S={env['sustain']:.2f} R={env['release']:.3f}s "
                f"Amount={env['amount']:.1f}st")
    
    if args[0].lower() == 'off':
        session.freq_env['enabled'] = False
        return "OK: frequency envelope disabled"
    
    if args[0].lower() == 'amount' and len(args) > 1:
        try:
            amount = float(args[1])
            session.freq_env['amount'] = amount
            session.freq_env['enabled'] = True
            return f"OK: frequency envelope amount set to {amount:.1f} semitones"
        except ValueError:
            return f"ERROR: invalid amount '{args[1]}'"
    
    # Parse ADSR
    try:
        a = float(args[0]) if len(args) > 0 else 0.01
        d = float(args[1]) if len(args) > 1 else 0.1
        s = float(args[2]) if len(args) > 2 else 0.0
        r = float(args[3]) if len(args) > 3 else 0.1
        
        session.freq_env['attack'] = max(0.001, a)
        session.freq_env['decay'] = max(0.001, d)
        session.freq_env['sustain'] = max(0, min(1, s))
        session.freq_env['release'] = max(0.001, r)
        session.freq_env['enabled'] = True
        
        return f"OK: frequency envelope A={a:.3f} D={d:.3f} S={s:.2f} R={r:.3f}"
    except ValueError:
        return "ERROR: invalid ADSR values"


def cmd_menv(session: Session, args: List[str]) -> str:
    """Set modulation envelope.
    
    Usage:
      /menv                    Show current settings
      /menv <a> <d> <s> <r>    Set ADSR (attack, decay, sustain, release)
      /menv depth <amount>     Set modulation depth (0-100)
      /menv off                Disable modulation envelope
    
    Modulates FM/AM/PM depth over time.
    """
    if not hasattr(session, 'mod_env'):
        session.mod_env = {
            'enabled': False,
            'attack': 0.01,
            'decay': 0.2,
            'sustain': 0.5,
            'release': 0.2,
            'depth': 100.0
        }
    
    if not args:
        env = session.mod_env
        if not env['enabled']:
            return "MOD ENV: off"
        return (f"MOD ENV: A={env['attack']:.3f}s D={env['decay']:.3f}s "
                f"S={env['sustain']:.2f} R={env['release']:.3f}s "
                f"Depth={env['depth']:.0f}%")
    
    if args[0].lower() == 'off':
        session.mod_env['enabled'] = False
        return "OK: modulation envelope disabled"
    
    if args[0].lower() == 'depth' and len(args) > 1:
        try:
            depth = float(args[1])
            session.mod_env['depth'] = max(0, min(200, depth))
            session.mod_env['enabled'] = True
            return f"OK: modulation envelope depth set to {depth:.0f}%"
        except ValueError:
            return f"ERROR: invalid depth '{args[1]}'"
    
    # Parse ADSR
    try:
        a = float(args[0]) if len(args) > 0 else 0.01
        d = float(args[1]) if len(args) > 1 else 0.2
        s = float(args[2]) if len(args) > 2 else 0.5
        r = float(args[3]) if len(args) > 3 else 0.2
        
        session.mod_env['attack'] = max(0.001, a)
        session.mod_env['decay'] = max(0.001, d)
        session.mod_env['sustain'] = max(0, min(1, s))
        session.mod_env['release'] = max(0.001, r)
        session.mod_env['enabled'] = True
        
        return f"OK: modulation envelope A={a:.3f} D={d:.3f} S={s:.2f} R={r:.3f}"
    except ValueError:
        return "ERROR: invalid ADSR values"


# ============================================================================
# COMMAND ALIASES AND EXPORTS
# ============================================================================

# Oscillator/waveform aliases
cmd_o = cmd_wm
cmd_osc = cmd_wm
cmd_wave = cmd_wm

# Noise generator
def cmd_noise(session: Session, args: List[str]) -> str:
    """Generate noise.
    
    Usage:
      /noise [duration] [type]
      
    Types: white (default), pink, brown
    """
    duration = 1.0
    noise_type = 'white'
    
    if args:
        try:
            duration = float(args[0])
        except ValueError:
            noise_type = args[0]
    if len(args) > 1:
        noise_type = args[1]
    
    import numpy as np
    samples = int(duration * session.sample_rate)
    
    if noise_type == 'pink':
        # Pink noise approximation
        white = np.random.randn(samples)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        try:
            from scipy.signal import lfilter
            noise = lfilter(b, a, white)
        except:
            noise = white
    elif noise_type == 'brown':
        white = np.random.randn(samples)
        noise = np.cumsum(white)
        noise = noise / np.max(np.abs(noise) + 1e-10)
    else:
        noise = np.random.randn(samples)
    
    noise = noise / (np.max(np.abs(noise)) + 1e-10) * 0.8
    session.last_buffer = noise.astype(np.float64)
    
    # Update working buffer
    try:
        from .working_cmds import get_working_buffer
        wb = get_working_buffer()
        wb.set_pending(session.last_buffer, f"noise:{noise_type}", session)
    except Exception:
        pass
    
    return f"OK: {noise_type} noise generated ({duration:.2f}s)"


# ---------------------------------------------------------------------------
# NOTE SEQUENCE GENERATOR
# ---------------------------------------------------------------------------

def cmd_ns(session: Session, args: List[str]) -> str:
    """Generate a sequence of notes using the built-in tone generator.

    This command allows you to specify a list of note names or frequencies and
    optionally a common duration and amplitude for all notes. Each note in
    the sequence will be rendered one after the other and the resulting
    audio will be concatenated into a single buffer.

    Usage:
      /ns <note1> <note2> ... [beats] [amp]

    * ``note`` can be a musical note name (e.g. C4, D#3, Bb5) or a
      frequency in Hz (e.g. 440). Note names are case-insensitive.
    * ``beats`` (optional) is the length of each note in beats. If omitted
      it defaults to 1 beat. This value applies to all notes in the
      sequence.
    * ``amp`` (optional) is the amplitude (0-1) for all notes. If omitted
      it defaults to 1.0.

    Examples:
      /ns C4 D4 E4 F4          -> four quarter notes at full volume
      /ns A3 C4 E4 0.5         -> each note lasts half a beat
      /ns G3 B3 D4 0.5 0.8     -> half‑beat notes at 80% amplitude

    On completion, the combined audio is stored in ``session.last_buffer``
    and, if possible, also in the working buffer for later use. The
    command reports duration, peak and RMS of the resulting sequence.
    """
    import numpy as np
    from .working_cmds import get_working_buffer  # Lazy import to avoid cycle

    # Defaults
    beats = 1.0
    amp = 1.0
    notes: list[str] = []

    # Separate note tokens from optional numeric tokens. Numeric tokens are
    # interpreted as beats and amplitude in the order they appear. Any
    # non-numeric token is treated as a note or frequency. We intentionally
    # treat tokens that fail numeric parsing as notes to support note names
    # like C4 or F#3. If a token contains only digits and possibly a dot it
    # will be interpreted as a float for beats/amp.
    numeric_vals: list[float] = []
    for tok in args:
        # Try parsing as float
        try:
            # Only treat as numeric if it doesn't contain any letters.
            # This avoids mistaking note names like C4 or A#3 for numbers.
            if tok.replace('.', '').replace('-', '').isdigit():
                numeric_vals.append(float(tok))
            else:
                notes.append(tok)
        except ValueError:
            notes.append(tok)

    # Assign beats and amplitude from numeric values if provided
    if numeric_vals:
        beats = numeric_vals[0]
        if len(numeric_vals) > 1:
            amp = numeric_vals[1]

    # Early exit if no notes provided
    if not notes:
        return "ERROR: no notes specified. Usage: /ns <note1> <note2> ... [beats] [amp]"

    # Save original last buffer to restore if anything fails
    orig_last_buffer = session.last_buffer
    generated_buffers: list[np.ndarray] = []

    # Generate each note in sequence
    for note in notes:
        try:
            # Use existing _parse_freq from this module to convert note name
            # to frequency. If it fails, _parse_freq will raise a ValueError.
            freq = _parse_freq(note)
        except Exception:
            # If parsing fails, try interpreting as a float (Hz)
            try:
                freq = float(note)
            except Exception:
                session.last_buffer = orig_last_buffer
                return f"ERROR: invalid note or frequency '{note}'"
        try:
            session.generate_tone(freq, beats, amp)
        except Exception as e:
            session.last_buffer = orig_last_buffer
            return f"ERROR: {e}"
        # Copy last_buffer to avoid overwriting when the next tone is generated
        if session.last_buffer is not None:
            generated_buffers.append(np.copy(session.last_buffer))

    # Concatenate all generated note buffers
    if not generated_buffers:
        return "ERROR: failed to generate sequence (no audio)"

    combined = np.concatenate(generated_buffers)
    session.last_buffer = combined.astype(np.float64)

    # Attempt to store in working buffer
    try:
        wb = get_working_buffer()
        label = 'sequence:' + ','.join(notes)
        wb.set_pending(session.last_buffer, label, session)
    except Exception:
        pass

    # Compute metrics
    try:
        peak = float(np.max(np.abs(session.last_buffer)))
        rms = float(np.sqrt(np.mean(session.last_buffer ** 2)))
    except Exception:
        peak = 0.0
        rms = 0.0
    # Duration in seconds
    duration = len(session.last_buffer) / session.sample_rate if session.last_buffer is not None else 0.0

    return (f"OK: generated sequence ({len(notes)} notes)\n"
            f"  beats per note: {beats}\n"
            f"  amplitude: {amp}\n"
            f"  duration: {duration:.3f}s\n"
            f"  peak: {peak:.3f}\n"
            f"  rms: {rms:.3f}")


# ===========================================================================
# PHASE 2: NEW SYNTH COMMANDS
# ===========================================================================

def cmd_ssaw(session: Session, args: List[str]) -> str:
    """Configure supersaw parameters.

    Usage:
      /ssaw                       -> Show current supersaw settings
      /ssaw <saws> <spread> <mix> -> Set supersaw params

    Parameters:
      saws   - Number of saw oscillators (3-11, default 7)
      spread - Detune spread in semitones (0.0-2.0, default 0.5)
      mix    - Center vs. stack balance (0.0-1.0, default 0.75)
    """
    op_idx = session.current_operator
    params = session.engine.operators.get(op_idx)

    if not args:
        if params is None:
            return "SSAW: (no operator defined)"
        lines = [
            f"=== SUPERSAW op{op_idx} ===",
            f"  Saws: {params.get('num_saws', 7)}",
            f"  Detune Spread: {params.get('detune_spread', 0.5):.2f} st",
            f"  Mix: {params.get('mix', 0.75):.2f}",
        ]
        return '\n'.join(lines)

    try:
        saws = _parse_int(args[0]) if len(args) > 0 else 7
        spread = _parse_float(args[1]) if len(args) > 1 else 0.5
        mix = _parse_float(args[2]) if len(args) > 2 else 0.75
        session.engine.set_param(op_idx, 'num_saws', max(3, min(11, saws)))
        session.engine.set_param(op_idx, 'detune_spread', spread)
        session.engine.set_param(op_idx, 'mix', mix)
        return f"OK: supersaw params set (saws={saws}, spread={spread:.2f}, mix={mix:.2f})"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_harm(session: Session, args: List[str]) -> str:
    """Configure harmonic model (odd/even independent control).

    Usage:
      /harm                                   -> Show settings
      /harm <odd_lvl> <even_lvl>              -> Set levels
      /harm <odd_lvl> <even_lvl> <nharm>      -> Set levels + count
      /harm <odd_lvl> <even_lvl> <nharm> <odecay> <edecay>

    Parameters:
      odd_lvl  - Odd harmonic level (0.0-2.0, default 1.0)
      even_lvl - Even harmonic level (0.0-2.0, default 1.0)
      nharm    - Number of harmonics (1-64, default 16)
      odecay   - Odd harmonic decay rate (0.1-1.0, default 0.8)
      edecay   - Even harmonic decay rate (0.1-1.0, default 0.8)

    Tip: Set even_lvl=0 for clarinet/square-like timbres (odd only).
         Set odd_lvl low for warm organ-like timbres (even emphasis).
    """
    op_idx = session.current_operator
    params = session.engine.operators.get(op_idx)

    if not args:
        if params is None:
            return "HARM: (no operator defined)"
        lines = [
            f"=== HARMONIC MODEL op{op_idx} ===",
            f"  Odd Level: {params.get('odd_level', 1.0):.2f}",
            f"  Even Level: {params.get('even_level', 1.0):.2f}",
            f"  Harmonics: {params.get('num_harmonics', 16)}",
            f"  Odd Decay: {params.get('odd_decay', 0.8):.2f}",
            f"  Even Decay: {params.get('even_decay', 0.8):.2f}",
        ]
        return '\n'.join(lines)

    try:
        odd_lvl = max(0.0, min(2.0, _parse_float(args[0]))) if len(args) > 0 else 1.0
        even_lvl = max(0.0, min(2.0, _parse_float(args[1]))) if len(args) > 1 else 1.0
        nharm = _parse_int(args[2]) if len(args) > 2 else 16
        odecay = max(0.1, min(1.0, _parse_float(args[3]))) if len(args) > 3 else 0.8
        edecay = max(0.1, min(1.0, _parse_float(args[4]))) if len(args) > 4 else 0.8
        session.engine.set_param(op_idx, 'odd_level', odd_lvl)
        session.engine.set_param(op_idx, 'even_level', even_lvl)
        session.engine.set_param(op_idx, 'num_harmonics', max(1, min(64, nharm)))
        session.engine.set_param(op_idx, 'odd_decay', odecay)
        session.engine.set_param(op_idx, 'even_decay', edecay)
        return (f"OK: harmonic model set (odd={odd_lvl:.2f}, even={even_lvl:.2f}, "
                f"nharm={nharm}, odecay={odecay:.2f}, edecay={edecay:.2f})")
    except Exception as e:
        return f"ERROR: {e}"


def cmd_waveguide(session: Session, args: List[str]) -> str:
    """Configure waveguide model parameters.

    Usage:
      /wg                           -> Show current waveguide settings
      /wg <param> <value>           -> Set single parameter

    Parameters (context-dependent on current wave type):
      damp <0.9-0.999>    - Damping
      bright <0.0-1.0>    - Brightness (string)
      pos <0.0-0.5>       - Pluck/strike position (string/membrane)
      reflect <0.0-1.0>   - End reflection (tube)
      bore <0.0-1.0>      - Bore shape (tube: 0=cyl, 1=conical)
      tension <0.0-1.0>   - Membrane tension (membrane)
      strike <0.0-1.0>    - Strike position (membrane)
      thick <0.0-1.0>     - Plate thickness (plate)
      mat <0.0-1.0>       - Material (plate: 0=wood, 1=metal)
    """
    op_idx = session.current_operator
    params = session.engine.operators.get(op_idx)

    if not args:
        if params is None:
            return "WG: (no operator defined)"
        wave = params.get('wave', '')
        lines = [f"=== WAVEGUIDE op{op_idx} ({wave}) ==="]
        lines.append(f"  Damping: {params.get('damping', 0.996):.4f}")
        if wave == 'waveguide_string':
            lines.append(f"  Brightness: {params.get('brightness', 0.5):.2f}")
            lines.append(f"  Position: {params.get('position', 0.28):.2f}")
        elif wave == 'waveguide_tube':
            lines.append(f"  Reflection: {params.get('reflection', 0.7):.2f}")
            lines.append(f"  Bore Shape: {params.get('bore_shape', 0.5):.2f}")
        elif wave == 'waveguide_membrane':
            lines.append(f"  Tension: {params.get('tension', 0.5):.2f}")
            lines.append(f"  Strike Pos: {params.get('strike_pos', 0.3):.2f}")
        elif wave == 'waveguide_plate':
            lines.append(f"  Thickness: {params.get('thickness', 0.5):.2f}")
            lines.append(f"  Material: {params.get('material', 0.5):.2f}")
        return '\n'.join(lines)

    param_name = args[0].lower()
    if len(args) < 2:
        return "Usage: /wg <param> <value>"

    wg_param_map = {
        'damp': 'damping', 'damping': 'damping',
        'bright': 'brightness', 'brightness': 'brightness',
        'pos': 'position', 'position': 'position',
        'reflect': 'reflection', 'reflection': 'reflection',
        'bore': 'bore_shape', 'bore_shape': 'bore_shape',
        'tension': 'tension',
        'strike': 'strike_pos', 'strike_pos': 'strike_pos',
        'thick': 'thickness', 'thickness': 'thickness',
        'mat': 'material', 'material': 'material',
    }

    full_name = wg_param_map.get(param_name)
    if not full_name:
        return f"ERROR: Unknown waveguide param '{param_name}'. Use: {', '.join(sorted(set(wg_param_map.values())))}"

    try:
        val = _parse_float(args[1])
        session.engine.set_param(op_idx, full_name, val)
        return f"OK: {full_name} = {val:.4f} on op{op_idx}"
    except Exception as e:
        return f"ERROR: {e}"


def cmd_wt(session: Session, args: List[str]) -> str:
    """Wavetable management.

    Usage:
      /wt                       -> List loaded wavetables
      /wt load <name> <path>    -> Load wavetable from .wav file
      /wt load <name> <path> <framesize> -> Load with custom frame size
      /wt use <name>            -> Set current operator to use wavetable
      /wt frame <pos>           -> Set frame position (0.0-1.0)
      /wt del <name>            -> Delete a loaded wavetable
      /wt info <name>           -> Show wavetable details
    """
    if not args:
        tables = session.engine.list_wavetables()
        if not tables:
            return "No wavetables loaded. Use /wt load <name> <path>"
        lines = ["=== WAVETABLES ==="]
        for name, frames, size in tables:
            lines.append(f"  {name}: {frames} frames x {size} samples")
        return '\n'.join(lines)

    sub = args[0].lower()

    if sub == 'load' and len(args) >= 3:
        name = args[1]
        path = args[2]
        frame_size = _parse_int(args[3]) if len(args) > 3 else 2048
        result = session.engine.load_wavetable_from_file(name, path, frame_size)
        return result

    elif sub == 'use' and len(args) >= 2:
        name = args[1]
        if name not in session.engine.wavetables:
            available = ', '.join(session.engine.wavetables.keys()) or '(none)'
            return f"ERROR: Wavetable '{name}' not found. Available: {available}"
        op_idx = session.current_operator
        session.engine.set_wave(op_idx, 'wavetable', wavetable_name=name)
        return f"OK: op{op_idx} set to wavetable '{name}'"

    elif sub == 'frame' and len(args) >= 2:
        pos = _parse_float(args[1])
        op_idx = session.current_operator
        session.engine.set_param(op_idx, 'frame_pos', max(0.0, min(1.0, pos)))
        return f"OK: frame position = {pos:.3f} on op{op_idx}"

    elif sub == 'del' and len(args) >= 2:
        name = args[1]
        if session.engine.delete_wavetable(name):
            return f"OK: wavetable '{name}' deleted"
        return f"ERROR: wavetable '{name}' not found"

    elif sub == 'info' and len(args) >= 2:
        name = args[1]
        if name not in session.engine.wavetables:
            return f"ERROR: wavetable '{name}' not found"
        frames = session.engine.wavetables[name]
        lines = [
            f"=== WAVETABLE: {name} ===",
            f"  Frames: {frames.shape[0]}",
            f"  Frame Size: {frames.shape[1]} samples",
            f"  Total Samples: {frames.shape[0] * frames.shape[1]}",
        ]
        return '\n'.join(lines)

    return ("Usage: /wt [load|use|frame|del|info]\n"
            "  /wt load <name> <path> [framesize]\n"
            "  /wt use <name>\n"
            "  /wt frame <0.0-1.0>\n"
            "  /wt del <name>\n"
            "  /wt info <name>")


def cmd_compound(session: Session, args: List[str]) -> str:
    """Compound wave management.

    Usage:
      /compound                       -> List compound waves
      /compound new <name>            -> Create new compound
      /compound add <name> <wave> [detune] [amp] [phase]  -> Add layer
      /compound use <name>            -> Set operator to use compound
      /compound morph <0.0-1.0>       -> Set morph position (2-layer)
      /compound del <name>            -> Delete compound
      /compound show <name>           -> Show layers
    """
    if not args:
        compounds = session.engine.list_compounds()
        if not compounds:
            return "No compound waves defined. Use /compound new <name>"
        lines = ["=== COMPOUND WAVES ==="]
        for name, n_layers in compounds:
            lines.append(f"  {name}: {n_layers} layers")
        return '\n'.join(lines)

    sub = args[0].lower()

    if sub == 'new' and len(args) >= 2:
        name = args[1]
        session.engine.create_compound(name, [])
        return f"OK: compound '{name}' created (empty). Add layers with /compound add"

    elif sub == 'add' and len(args) >= 3:
        name = args[1]
        wave = args[2].lower()
        detune = _parse_float(args[3]) if len(args) > 3 else 0.0
        layer_amp = _parse_float(args[4]) if len(args) > 4 else 1.0
        layer_phase = _parse_float(args[5]) if len(args) > 5 else 0.0

        if name not in session.engine.compound_waves:
            session.engine.compound_waves[name] = []

        layer = {'wave': wave, 'amp': layer_amp, 'detune': detune, 'phase': layer_phase}
        session.engine.compound_waves[name].append(layer)
        n = len(session.engine.compound_waves[name])
        return f"OK: added {wave} layer to '{name}' (now {n} layers)"

    elif sub == 'use' and len(args) >= 2:
        name = args[1]
        if name not in session.engine.compound_waves:
            available = ', '.join(session.engine.compound_waves.keys()) or '(none)'
            return f"ERROR: Compound '{name}' not found. Available: {available}"
        op_idx = session.current_operator
        session.engine.set_wave(op_idx, 'compound', compound_name=name)
        return f"OK: op{op_idx} set to compound '{name}'"

    elif sub == 'morph' and len(args) >= 2:
        pos = _parse_float(args[1])
        op_idx = session.current_operator
        session.engine.set_param(op_idx, 'morph', max(0.0, min(1.0, pos)))
        return f"OK: morph position = {pos:.3f} on op{op_idx}"

    elif sub == 'del' and len(args) >= 2:
        name = args[1]
        if session.engine.delete_compound(name):
            return f"OK: compound '{name}' deleted"
        return f"ERROR: compound '{name}' not found"

    elif sub == 'show' and len(args) >= 2:
        name = args[1]
        if name not in session.engine.compound_waves:
            return f"ERROR: compound '{name}' not found"
        layers = session.engine.compound_waves[name]
        lines = [f"=== COMPOUND: {name} ({len(layers)} layers) ==="]
        for i, layer in enumerate(layers):
            lines.append(
                f"  [{i}] {layer.get('wave', 'sine')}: "
                f"amp={layer.get('amp', 1.0):.2f}, "
                f"detune={layer.get('detune', 0.0):.2f}st, "
                f"phase={layer.get('phase', 0.0):.2f}"
            )
        return '\n'.join(lines)

    return ("Usage: /compound [new|add|use|morph|del|show]\n"
            "  /compound new <name>\n"
            "  /compound add <name> <wave> [detune] [amp] [phase]\n"
            "  /compound use <name>\n"
            "  /compound morph <0.0-1.0>\n"
            "  /compound del <name>\n"
            "  /compound show <name>")


def cmd_waveinfo(session: Session, args: List[str]) -> str:
    """Show all available wave types and their parameters.

    Usage:
      /waveinfo         -> List all wave types
    """
    return session.engine.get_wave_info()


def get_synth_commands() -> dict:
    """Return synth commands for registration."""
    return {
        # Note/tone
        'n': cmd_n,
        'note': cmd_n,
        'tone': cmd_tone,
        'noise': cmd_noise,
        'ns': cmd_ns,
        'seq': cmd_ns,
        
        # Oscillator/waveform
        'o': cmd_wm,
        'osc': cmd_wm,
        'wave': cmd_wm,
        'wm': cmd_wm,
        
        # Operators
        'op': cmd_op,
        'car': cmd_car,
        'mod': cmd_mod,
        'v': cmd_v,
        'vc': cmd_v,     # alias voice count
        'va': cmd_va,
        'ps': cmd_vphase, # alias phase spread
        'ss': cmd_stereo, # alias stereo spread
        
        # Envelope
        'atk': cmd_atk,
        'dec': cmd_dec,
        'sus': cmd_sus,
        'rel': cmd_rel,
        'env': cmd_env,
        'adsr': cmd_adsr,
        # Use /ev for envelope to avoid conflict with /e (function execution)
        'ev': cmd_env,
        
        # Musical key/scale
        'key': cmd_key,
        
        # Filter
        'fcount': cmd_fcount,
        'ft': cmd_ft,
        'fs': cmd_fs,
        'fe': cmd_fe,
        'fen': cmd_fen,
        'cut': cmd_cut,
        'res': cmd_res,
        'fatk': cmd_fatk,
        'fdec': cmd_fdec,
        'fsus': cmd_fsus,
        'frel': cmd_frel,
        
        # Modulation/routing
        'fm': cmd_fm,
        'tfm': cmd_tfm,
        'am': cmd_am,
        'rm': cmd_rm,
        'pm': cmd_pm,
        'rt': cmd_rt,
        
        # Other
        'fr': cmd_fr,
        'amp': cmd_amp,
        'ph': cmd_ph,
        'pw': cmd_pw,
        'phys': cmd_phys,
        'phys2': cmd_phys2,
        'opinfo': cmd_opinfo,
        'clearalg': cmd_clearalg,
        'dt': cmd_dt,
        'rand': cmd_rand,
        'vmod': cmd_vmod,
        'hf': cmd_hf,
        'bk': cmd_bk,
        'al': cmd_al,
        'preset': cmd_preset,
        
        # Voice params
        'stereo': cmd_stereo,
        'vphase': cmd_vphase,
        'venv': cmd_venv,
        'fenv': cmd_fenv,
        'menv': cmd_menv,

        # Phase 2: Extended wave commands
        'ssaw': cmd_ssaw,
        'supersaw': cmd_ssaw,
        'harm': cmd_harm,
        'harmonic': cmd_harm,
        'wg': cmd_waveguide,
        'waveguide': cmd_waveguide,
        'wt': cmd_wt,
        'wavetable': cmd_wt,
        'compound': cmd_compound,
        'comp': cmd_compound,
        'waveinfo': cmd_waveinfo,
        'waves': cmd_waveinfo,
    }
