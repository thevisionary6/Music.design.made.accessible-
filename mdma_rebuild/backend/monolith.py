"""V2 SignalFlow-native monolith voice builder.

The legacy ``dsp/monolith.py`` is a 2100-LOC numpy FM synth. Per
SKILL.md's Phase 8+ notes, it's being rebuilt piece-by-piece as a
SignalFlow graph — the numpy version stays around for reference and
for advanced wave types (supersaw, wavetable, formant, waveguide,
compound) until those are ported.

This module implements the **core path**: sine / triangle / saw /
pulse / noise operators with FM / AM / RM routing, ASR envelope from
``session.attack/.decay/.sustain/.release``, and the session's
currently-selected SVFilter slot. That's enough to make
``/patn`` sound like whatever the user configures via ``/wm``,
``/fr``, ``/atk``, ``/fm``, ``/am``, ``/rm``, ``/cut``, ``/res`` —
the cleanest subset of the existing synth REPL surface that maps
naturally to SignalFlow nodes.

Extensions (supersaw, additive, formants, waveguide strings, etc.)
can hang off :func:`_build_oscillator` as new branches without
touching the envelope / modulation / filter code.

Public surface:

- :func:`build_voice(note, session) -> Node` — the shape callable
  for ``Pattern.play(shape, graph)`` when the user wants their
  configured monolith. :func:`.utils.build_default_shape` is the
  recommended entry point; it picks this automatically when the
  session has operators configured.
"""

from __future__ import annotations

from typing import Optional

from ..dsp.monolith import WAVE_ALIASES  # tiny alias table, no hot code
from .utils import note_to_freq


# ---------------------------------------------------------------------------
# Wave type dispatch
# ---------------------------------------------------------------------------


_SUPPORTED_WAVES = {"sine", "triangle", "saw", "pulse", "noise"}


def _resolve_wave(raw: str) -> str:
    """Map user-typed wave names (sin, sawtooth, square, ...) to the
    five SignalFlow-backed core types. Unknown or unsupported names
    fall back to ``"sine"`` with a one-liner warning so the user
    hears *something* instead of silently getting silence.
    """
    wave = (raw or "sine").lower()
    if wave in WAVE_ALIASES:
        wave = WAVE_ALIASES[wave]
    if wave not in _SUPPORTED_WAVES:
        print(
            f"[monolith] wave {raw!r} not supported in the V2 graph yet; "
            "falling back to 'sine'. Use the legacy numpy monolith for "
            "supersaw / wavetable / formant / waveguide / compound."
        )
        return "sine"
    return wave


def _build_oscillator(op: dict, note: float, freq_offset=None):
    """Build one operator's base oscillator node.

    Frequency resolution:

    - ``op['ratio']`` (if set, float) is a multiplier on the played
      note frequency. ``ratio=2.0`` means one octave above the note,
      ``ratio=0.5`` one octave below.
    - Otherwise ``op['freq']`` (absolute Hz) is used — matching the
      legacy monolith's contract.
    - Otherwise the played note's fundamental.

    ``freq_offset`` is a SignalFlow node added to the base frequency
    (this is how FM routes through; see :func:`build_voice`).
    """
    import signalflow as sf

    wave = _resolve_wave(op.get("wave", "sine"))

    if "ratio" in op:
        base_freq = note_to_freq(note) * float(op["ratio"])
    elif "freq" in op:
        base_freq = float(op["freq"])
    else:
        base_freq = note_to_freq(note)

    if freq_offset is not None:
        freq_node = base_freq + freq_offset
    else:
        freq_node = base_freq

    amp = float(op.get("amp", 1.0))
    phase = float(op.get("phase", 0.0)) if wave != "noise" else 0.0

    if wave == "sine":
        osc = sf.SineOscillator(freq_node, phase_offset=phase)
    elif wave == "triangle":
        osc = sf.TriangleOscillator(freq_node)
    elif wave == "saw":
        osc = sf.SawOscillator(freq_node)
    elif wave == "pulse":
        osc = sf.SquareOscillator(freq_node)
    else:  # noise
        # SignalFlow's WhiteNoise has no frequency input; the wave
        # type is just "noise" regardless of the op's freq setting.
        if hasattr(sf, "WhiteNoise"):
            osc = sf.WhiteNoise()
        else:  # pragma: no cover - older SignalFlow builds
            osc = sf.SineOscillator(freq_node)

    return osc * amp if amp != 1.0 else osc


# ---------------------------------------------------------------------------
# Modulation routing
# ---------------------------------------------------------------------------


def _apply_modulations(
    base_nodes: dict,
    operators: dict,
    algorithms: list,
    note: float,
):
    """Wire every ``(algo, source, target, amount)`` algorithm entry.

    Mutates ``base_nodes`` in place. Supported algorithms:

    - ``FM`` — source signal modulates target's frequency.
    - ``AM`` — target is multiplied by ``(1 + source * amount)``.
    - ``RM`` — target is multiplied by ``(source * amount)`` (ring
      modulation).

    ``PM`` (phase modulation) and ``TFM`` (through-zero FM) are
    recognised but not yet implemented; they print a warning and
    skip. A future pass can add PM once the spec for phase-offset
    routing on SignalFlow oscillators stabilises.
    """
    for algo, source, target, amount in algorithms:
        if source not in base_nodes or target not in base_nodes:
            continue
        amt = float(amount)
        mod = base_nodes[source]
        algo_u = algo.upper()
        if algo_u == "FM":
            # Rebuild the target oscillator with a frequency-modulated
            # input. We can't just mutate the existing node's
            # frequency input reliably across SignalFlow versions, so
            # we produce a fresh oscillator with the same wave /
            # amp / phase but a new frequency signal.
            target_op = operators[target]
            base_nodes[target] = _build_oscillator(
                target_op, note, freq_offset=mod * amt
            )
        elif algo_u == "AM":
            base_nodes[target] = base_nodes[target] * (1.0 + mod * amt)
        elif algo_u == "RM":
            base_nodes[target] = base_nodes[target] * (mod * amt)
        elif algo_u in ("PM", "TFM"):
            print(
                f"[monolith] {algo_u} routing not yet supported in the V2 "
                f"graph; skipping {source}->{target} @ {amt}."
            )
        else:
            print(
                f"[monolith] unknown algorithm {algo!r}; "
                f"skipping {source}->{target}."
            )


# ---------------------------------------------------------------------------
# Envelope + filter
# ---------------------------------------------------------------------------


def _apply_envelope(voice, session):
    """Wrap ``voice`` in an ASREnvelope sized from the session's
    ADSR. SignalFlow's ``ASREnvelope`` collapses decay/sustain into
    "sustain at 1.0 for hold time", so we approximate:

    - attack = session.attack
    - sustain_time = session.decay + some hold — matches the
      perceptual "body" of a note under the old ADSR.
    - release = session.release

    A later pass can swap in a full ADSR node once SignalFlow exposes
    one directly (or via a breakpoint Envelope).
    """
    attack = float(getattr(session, "attack", 0.01))
    decay = float(getattr(session, "decay", 0.1))
    release = float(getattr(session, "release", 0.1))

    import signalflow as sf
    env = sf.ASREnvelope(
        attack=max(0.001, attack),
        sustain=max(0.0, decay),  # sustain time, not level
        release=max(0.001, release),
    )
    return voice * env


def _apply_session_filter(voice, session):
    """If the session's selected filter slot is enabled and its type
    maps to an SVFilter string, wrap ``voice``. Unsupported slot
    types (formant, moog-special, etc.) pass through so the voice
    stays audible."""
    enabled = session.filter_enabled.get(session.selected_filter, False)
    if not enabled:
        return voice

    type_names = getattr(session, "filter_type_names", {})
    type_idx = session.filter_types.get(session.selected_filter, 0)
    type_name = type_names.get(type_idx, "lowpass")

    svf_map = {
        "lowpass": "low_pass",
        "highpass": "high_pass",
        "bandpass": "band_pass",
        "notch": "notch",
        "peak": "peak",
        "lowshelf": "low_shelf",
        "highshelf": "high_shelf",
    }
    sf_type = svf_map.get(type_name)
    if sf_type is None:
        # Unsupported filter type (e.g. formant, ringmod, acid) —
        # skip rather than crash. Users who want those can bind a
        # custom DSP callable via .fx() / session.custom_effects.
        return voice

    cutoff = float(session.filter_cutoffs.get(session.selected_filter, 1000.0))
    res_0_100 = float(
        session.filter_resonances.get(session.selected_filter, 50.0)
    )
    # Convert the UI 0-100 scale to SVFilter's [0, 1] resonance.
    res_unit = max(0.0, min(0.95, res_0_100 / 100.0))

    import signalflow as sf
    return sf.SVFilter(voice, sf_type, cutoff, res_unit)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_voice(note: float, session) -> object:
    """Build a SignalFlow voice node for one MIDI note.

    Reads operator definitions from ``session.engine.operators``,
    modulation routing from ``session.engine.algorithms``, and the
    envelope / filter state from the session itself. Returns a node
    suitable for ``Pattern.play`` / ``Session.play_pattern``.

    If no carriers are configured (``session.carrier_count`` is 0 or
    no operator indices below ``carrier_count`` exist), falls back
    to a plain sine at the played note's frequency so the voice is
    never silent.
    """
    import signalflow as sf

    engine = session.engine
    operators = dict(engine.operators)
    algorithms = list(getattr(engine, "algorithms", []))

    base_nodes = {
        idx: _build_oscillator(op, note)
        for idx, op in operators.items()
    }
    _apply_modulations(base_nodes, operators, algorithms, note)

    carrier_count = int(getattr(session, "carrier_count", 1))
    carrier_indices = [i for i in range(carrier_count) if i in base_nodes]

    if not carrier_indices:
        voice = sf.SineOscillator(note_to_freq(note))
    else:
        voice = base_nodes[carrier_indices[0]]
        for idx in carrier_indices[1:]:
            voice = voice + base_nodes[idx]
        # Average so N carriers don't clip proportionally to N.
        if len(carrier_indices) > 1:
            voice = voice * (1.0 / len(carrier_indices))

    voice = _apply_session_filter(voice, session)
    voice = _apply_envelope(voice, session)

    return voice


__all__ = ["build_voice"]
