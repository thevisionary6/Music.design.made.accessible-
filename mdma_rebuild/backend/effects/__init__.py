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

Phase 8 ships with :mod:`.soft_clip` and :mod:`.delay` as reference
implementations. More effects get ported on demand — see the
``Phase 8+`` note in the SKILL.md roadmap.
"""

from .soft_clip import soft_clip
from .delay import delay

__all__ = ["soft_clip", "delay"]
