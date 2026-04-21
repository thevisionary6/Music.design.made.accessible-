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

**Delay / ambient** (``Pattern.fx`` / ``Pattern.ir``)
- :mod:`.delay` — single-tap feedback delay.
- :mod:`.reverb` — Schroeder-style algorithmic reverb.

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

**Dynamics** (``Pattern.fx``)
- :func:`.dynamics.compressor` — feed-forward compressor.

More effects get ported on demand — see the ``Phase 8+`` note in
the SKILL.md roadmap.
"""

from .bitcrush import bitcrush
from .delay import delay
from .dynamics import compressor, db_to_amplitude
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
from .reverb import reverb
from .soft_clip import soft_clip

__all__ = [
    # distortion
    "bitcrush",
    "foldback",
    "hard_clip",
    "soft_clip",
    # delay / ambient
    "delay",
    "reverb",
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
    # dynamics
    "compressor",
    "db_to_amplitude",
]
