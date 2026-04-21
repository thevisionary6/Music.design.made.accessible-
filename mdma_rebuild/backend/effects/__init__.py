"""Ported SignalFlow-native effects for the V2 backend.

This subpackage is where old ``mdma_rebuild/dsp/`` effects get their
V2 form — SignalFlow-node DSP callables matching the spec signature::

    def my_effect(input_node: Node, **params) -> Node:
        ...

Drop any callable with that shape into :meth:`Pattern.mod`,
:meth:`Pattern.dist`, :meth:`Pattern.spec`, :meth:`Pattern.ir`, or
:meth:`Pattern.fx` and it composes into the chain.

Porting convention (for future additions):

1. **One effect per module.** Keeps the surface area small and the
   SignalFlow import cost local. Good names read as verbs:
   ``soft_clip``, ``delay``, ``chorus``.
2. **First argument is always ``input_node``.** Everything else is
   keyword-only by convention (pass via ``**params``).
3. **Return a new Node — never mutate the input.** Patterns rely on
   that immutability when the chain is applied per event.
4. **Parameters in MDMA scaling (0-100) where sensible**, in real
   units (Hz, seconds) otherwise. Use ``mdma_rebuild.dsp.scaling``
   helpers if a 0-100 param needs coercion from a preset name.
5. **SignalFlow import is lazy** (``import signalflow as sf`` inside
   the function body), so the subpackage is safe to import in
   environments without SignalFlow.
6. **Ship tests that exercise the call signature** against a node
   mock. Audio-rate correctness tests live under ``tests/audio/``
   and are opt-in.

Current inventory:

**Distortion** (``Pattern.dist``)
- :mod:`.soft_clip` — tanh saturation.
- :mod:`.hard_clip` — symmetric brick-wall clipping.
- :mod:`.foldback` — reflective folding.
- :mod:`.bitcrush` — bit-depth + sample-rate reduction.

**Saturation** (``Pattern.dist``) — the softer / warmer side of
distortion.
- :func:`.saturation.tube`, :func:`.saturation.tape`,
  :func:`.saturation.fuzz`.

**Delay family** (``Pattern.fx`` / ``Pattern.ir``)
- :mod:`.delay` — single-tap feedback delay.
- :mod:`.reverb` — Schroeder-style algorithmic reverb.
- :func:`.delay_family.slapback` — single short repeat.
- :func:`.delay_family.ping_pong` — stereo-bouncing delay.
- :func:`.delay_family.multitap` — parallel taps at different times.
- :func:`.delay_family.tape_echo` — delay with tanh + high-cut on
  the feedback path.

**Extended filters** (``Pattern.fx`` / ``Pattern.spec``; ``Pattern.bf``
is spec-locked to lpf/hpf/bpf/notch)
- :func:`.filters.peak`, :func:`.filters.low_shelf`,
  :func:`.filters.high_shelf` — EQ family.
- :func:`.filters.moog` — Moog-ladder VCF.
- :func:`.filters.allpass_filter` — phase rotator.
- :func:`.filters.comb_filter` — feedback comb.

**Modulation** (``Pattern.fx``)
- :func:`.modulation.chorus`, :func:`.modulation.flanger`,
  :func:`.modulation.phaser` — LFO-modulated delay family.
- :func:`.modulation.tremolo`, :func:`.modulation.autopan` —
  amplitude / stereo LFO effects.

**Pitch / frequency** (``Pattern.fx``)
- :func:`.pitch_freq.ring_mod` — ring modulation.
- :func:`.pitch_freq.amplitude_mod` — amplitude modulation.
- :func:`.pitch_freq.detune_unison` — stacked detuned voices via
  short LFO-driven delays.

**Spatial / stereo** (``Pattern.fx``)
- :func:`.spatial.haas`, :func:`.spatial.stereo_widen`,
  :func:`.spatial.mono`, :func:`.spatial.balance`.

**Dynamics** (``Pattern.fx``)
- :func:`.dynamics.compressor` — feed-forward compressor.
- :func:`.dynamics_extra.limiter` — brick-wall-ish 20:1.
- :func:`.dynamics_extra.noise_gate` — downward gate.
- :func:`.dynamics_extra.expander` — upward expander.

Custom effects load via :mod:`..fx_loader` / ``/loadfx``. See the
Phase 9+ note in ``SKILL.md`` for the roadmap on spectral,
granular, and vocoder effects (still deferred).
"""

from .bitcrush import bitcrush
from .delay import delay
from .delay_family import multitap, ping_pong, slapback, tape_echo
from .dynamics import compressor, db_to_amplitude
from .dynamics_extra import expander, limiter, noise_gate
from .filters import (
    allpass_filter,
    comb_filter,
    high_shelf,
    low_shelf,
    moog,
    peak,
)
from .foldback import foldback
from .hard_clip import hard_clip
from .modulation import (
    autopan,
    chorus,
    flanger,
    phaser,
    tremolo,
)
from .pitch_freq import amplitude_mod, detune_unison, ring_mod
from .reverb import reverb
from .saturation import fuzz, tape, tube
from .soft_clip import soft_clip
from .spatial import balance, haas, mono, stereo_widen

__all__ = [
    # distortion
    "bitcrush",
    "foldback",
    "hard_clip",
    "soft_clip",
    # saturation (softer / warmer)
    "fuzz",
    "tape",
    "tube",
    # delay / ambient
    "delay",
    "multitap",
    "ping_pong",
    "reverb",
    "slapback",
    "tape_echo",
    # filters
    "allpass_filter",
    "comb_filter",
    "high_shelf",
    "low_shelf",
    "moog",
    "peak",
    # modulation
    "autopan",
    "chorus",
    "flanger",
    "phaser",
    "tremolo",
    # pitch / freq
    "amplitude_mod",
    "detune_unison",
    "ring_mod",
    # spatial
    "balance",
    "haas",
    "mono",
    "stereo_widen",
    # dynamics
    "compressor",
    "db_to_amplitude",
    "expander",
    "limiter",
    "noise_gate",
]
