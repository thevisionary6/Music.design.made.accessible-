"""Pattern class for the MDMA backend V2.

Implements ``references/pattern_spec.md``. A Pattern holds:

- ``events`` — a list of ``(note, duration)`` float-tuples.
- ``chain`` — an ordered list of DSP transform specs that get applied at
  ``.play()`` time in order; index 0 is innermost / first applied.

All transforms return a **new Pattern**. Self is never mutated. Long
chains stay safe because every step copies.

Scope after Phase 2:

- Compositional transforms (``.fast``, ``.slow``, ``.t_p``, ``.l``,
  ``.gs``, ``.ex``, ``.cat``, ``.pre``) — fully implemented.
- Synthesis transforms (``.mod``, ``.dist``, ``.bf``, ``.spec``, ``.ir``,
  ``.gate``, ``.fx``) — fully implemented at the chain-registration
  level; chain entries are the exact tuple shapes the spec specifies.
- ``.play()`` walks the events, applies the chain via
  :meth:`Pattern._apply_chain`, and dispatches sequentially via
  ``graph.wait(dur)``.
- :meth:`Pattern._apply_builtin_filter` wraps the voice in a SignalFlow
  :class:`SVFilter`.
- :meth:`Pattern._apply_gate` turns the rhythm pattern into a
  single-cycle sample buffer and plays it through a looping
  :class:`BufferPlayer`, so rhythm patterns shorter than the host event
  simply repeat until the voice stops.

SignalFlow imports are deferred to the helpers so environments that only
want the compositional transforms (tests, offline analysis) don't pay
the SignalFlow import cost.
"""

from __future__ import annotations

from typing import Callable


# Filter types accepted by :meth:`Pattern.bf`. Kept at module scope so the
# validation can be read by callers and covered by tests without poking
# Pattern internals.
_BF_TYPES: frozenset[str] = frozenset({"lpf", "hpf", "bpf", "notch"})

# Mapping from the spec's short filter names to SignalFlow's ``SVFilter``
# ``filter_type`` strings. The SignalFlow API accepts these strings directly
# (see ``help(signalflow.SVFilter)``).
_BF_SIGNALFLOW_NAMES: dict[str, str] = {
    "lpf": "low_pass",
    "hpf": "high_pass",
    "bpf": "band_pass",
    "notch": "notch",
}


def _current_sample_rate() -> float:
    """Back-compat shim. The real implementation now lives in
    :func:`.utils.current_sample_rate` so every effect module can
    share it without creating circular imports.
    """
    from .utils import current_sample_rate
    return current_sample_rate()


class Pattern:
    """A sequence of ``(note, duration)`` events with a DSP chain.

    Construct from a list of ``(note, duration)`` tuples; both values are
    coerced to ``float`` so microtonal notes and sub-millisecond durations
    round-trip cleanly. Empty patterns are legal and act as the identity
    for :meth:`cat` and :meth:`pre`.

    Every transform returns a brand-new ``Pattern``; ``self`` is never
    mutated.
    """

    # -- Construction --------------------------------------------------

    def __init__(self, patt: list[tuple[float, float]]) -> None:
        """Build a Pattern from a list of ``(note, duration)`` tuples.

        Both ``note`` and ``duration`` are coerced to ``float``. The
        ``chain`` is always initialised empty; transforms that touch the
        chain (``.mod``, ``.dist``, ``.bf``, ``.spec``, ``.ir``, ``.gate``,
        ``.fx``) return a new Pattern with a copied-and-extended chain.
        """
        self.events: list[tuple[float, float]] = [
            (float(note), float(dur)) for (note, dur) in patt
        ]
        self.chain: list = []

    @classmethod
    def _from_parts(cls, events: list[tuple[float, float]], chain: list) -> "Pattern":
        """Internal constructor used by every transform.

        Avoids re-running ``float`` coercion on data that's already clean
        and lets transforms propagate chain state without re-validating
        it. ``events`` and ``chain`` are copied defensively so the caller
        can keep reusing its own lists.
        """
        p = cls.__new__(cls)
        p.events = list(events)
        p.chain = list(chain)
        return p

    # -- Compositional transforms --------------------------------------

    def fast(self, n: float) -> "Pattern":
        """Scale all durations by ``1/n``. ``n=2`` is twice as fast.

        ``n`` must be positive; non-positive raises :class:`ValueError`.
        Notes are unchanged. Chain is preserved.

        Test examples (from spec):
            Pattern([(60, 0.5), (62, 0.5)]).fast(2).events
                == [(60.0, 0.25), (62.0, 0.25)]
            Pattern([(60, 0.5)]).fast(0.5).events == [(60.0, 1.0)]
            Pattern([]).fast(2).events == []
        """
        if n <= 0:
            raise ValueError(f"fast factor must be positive, got {n}")
        new_events = [(note, dur / n) for (note, dur) in self.events]
        return Pattern._from_parts(new_events, self.chain)

    def slow(self, n: float) -> "Pattern":
        """Scale all durations by ``n``. Inverse of :meth:`fast`.

        ``n`` must be positive; non-positive raises :class:`ValueError`.

        Test examples (from spec):
            Pattern([(60, 0.25), (62, 0.25)]).slow(2).events
                == [(60.0, 0.5), (62.0, 0.5)]
            Pattern([(60, 0.5)]).slow(0.5).events == [(60.0, 0.25)]
        """
        if n <= 0:
            raise ValueError(f"slow factor must be positive, got {n}")
        new_events = [(note, dur * n) for (note, dur) in self.events]
        return Pattern._from_parts(new_events, self.chain)

    def t_p(self, n: float) -> "Pattern":
        """Transpose every note by ``n`` semitones (float-valued).

        Durations are unchanged. Chain is preserved.

        Test examples (from spec):
            Pattern([(60, 0.25), (64, 0.25)]).t_p(7).events
                == [(67.0, 0.25), (71.0, 0.25)]
            Pattern([(60, 0.25)]).t_p(-12).events == [(48.0, 0.25)]
            Pattern([(60, 0.25)]).t_p(0.5).events == [(60.5, 0.25)]
        """
        new_events = [(note + n, dur) for (note, dur) in self.events]
        return Pattern._from_parts(new_events, self.chain)

    def l(self, target_duration: float) -> "Pattern":
        """Rescale durations so the pattern's total length equals ``target_duration``.

        Uses full floating-point precision — no grid snapping. If the
        pattern is empty, returns an empty pattern. If the total duration
        is zero, returns an unchanged copy (no meaningful scale exists).

        ``target_duration`` must be positive; non-positive raises
        :class:`ValueError`.

        Test examples (from spec):
            Pattern([(60, 0.25), (62, 0.25), (64, 0.5)]).l(2.0).events
                == [(60.0, 0.5), (62.0, 0.5), (64.0, 1.0)]
            Pattern([(60, 1.0), (62, 1.0)]).l(1.0).events
                == [(60.0, 0.5), (62.0, 0.5)]
            Pattern([]).l(2.0).events == []
        """
        if target_duration <= 0:
            raise ValueError(
                f"target_duration must be positive, got {target_duration}"
            )
        if not self.events:
            return Pattern._from_parts([], self.chain)
        total = sum(dur for (_, dur) in self.events)
        if total == 0:
            return Pattern._from_parts(self.events, self.chain)
        scale = target_duration / total
        new_events = [(note, dur * scale) for (note, dur) in self.events]
        return Pattern._from_parts(new_events, self.chain)

    def gs(self, a: float, b: float) -> "Pattern":
        """Groove-swing: even indices scaled by ``a``, odd indices by ``b``.

        A single-event pattern takes ``a`` (it's at even index 0).
        Notes are unchanged.

        Test examples (from spec):
            Pattern([(60,.25),(62,.25),(64,.25),(65,.25)]).gs(1.5, 0.5).events
                == [(60.0, 0.375), (62.0, 0.125),
                    (64.0, 0.375), (65.0, 0.125)]
            Pattern([(60, 1.0)]).gs(2.0, 0.5).events == [(60.0, 2.0)]
        """
        new_events = []
        for i, (note, dur) in enumerate(self.events):
            mult = a if i % 2 == 0 else b
            new_events.append((note, dur * mult))
        return Pattern._from_parts(new_events, self.chain)

    def ex(self, note: float, dur: float) -> "Pattern":
        """Append a single ``(note, dur)`` event.

        Test examples (from spec):
            Pattern([(60, 0.25)]).ex(62, 0.5).events
                == [(60.0, 0.25), (62.0, 0.5)]
            Pattern([]).ex(60, 1.0).events == [(60.0, 1.0)]
        """
        new_events = list(self.events)
        new_events.append((float(note), float(dur)))
        return Pattern._from_parts(new_events, self.chain)

    def cat(self, other: "Pattern") -> "Pattern":
        """Concatenate ``other`` onto the end of this pattern.

        The result's ``chain`` is ``self.chain`` — ``other``'s chain is
        dropped. Mixing two chains ambiguously is worse than an explicit
        rule; the left-hand chain wins.

        Test examples (from spec):
            a, b = Pattern([(60, 0.25)]), Pattern([(62, 0.5)])
            a.cat(b).events == [(60.0, 0.25), (62.0, 0.5)]
            a.cat(Pattern([])).events == [(60.0, 0.25)]
            Pattern([]).cat(a).events == [(60.0, 0.25)]
        """
        new_events = list(self.events) + list(other.events)
        return Pattern._from_parts(new_events, self.chain)

    def pre(self, other: "Pattern") -> "Pattern":
        """Prepend ``other`` to the beginning of this pattern.

        Inverse of :meth:`cat`. Chain handling is the same — result uses
        ``self.chain``.

        Test example (from spec):
            a, b = Pattern([(60, 0.25)]), Pattern([(62, 0.5)])
            a.pre(b).events == [(62.0, 0.5), (60.0, 0.25)]
        """
        new_events = list(other.events) + list(self.events)
        return Pattern._from_parts(new_events, self.chain)

    # -- Synthesis transforms ------------------------------------------

    def mod(self, dsp_fn: Callable, **params) -> "Pattern":
        """Append a modulation stage (FM / AM / RM / arbitrary).

        ``dsp_fn`` must match the fixed DSP-callable signature
        ``fn(input_node, **params) -> Node``. The call site's kwargs are
        stored on the chain entry and re-applied at play time.

        Chain entry: ``("mod", dsp_fn, params)``.

        Test example (from spec):
            p = Pattern([(60, 0.25)]).mod(fm_mod, mod_index=3.0)
            p.chain == [("mod", fm_mod, {"mod_index": 3.0})]
            p.events == [(60.0, 0.25)]  # events untouched
        """
        new_chain = list(self.chain) + [("mod", dsp_fn, dict(params))]
        return Pattern._from_parts(self.events, new_chain)

    def dist(self, dsp_fn: Callable, **params) -> "Pattern":
        """Append a distortion / waveshaping / clipping stage.

        Semantically identical to :meth:`mod`; the separate name is for
        readability in chains and for future differentiation (e.g.,
        default oversampling for distortion).

        Chain entry: ``("dist", dsp_fn, params)``.
        """
        new_chain = list(self.chain) + [("dist", dsp_fn, dict(params))]
        return Pattern._from_parts(self.events, new_chain)

    def bf(self, filter_type: str, cutoff: float, resonance: float = 0.0) -> "Pattern":
        """Append a built-in filter stage.

        ``filter_type`` must be one of ``"lpf"``, ``"hpf"``, ``"bpf"``,
        ``"notch"``; anything else raises :class:`ValueError`. Allpass,
        low-shelf, and high-shelf are deliberately excluded (they are
        delay primitives and EQ shaping, not transformer filters).

        Chain entry: ``("bf", filter_type, cutoff, resonance)`` with both
        numeric fields coerced to ``float``.

        Test examples (from spec):
            Pattern([(60, 0.25)]).bf("lpf", 1000).chain
                == [("bf", "lpf", 1000.0, 0.0)]
            Pattern([(60, 0.25)]).bf("bpf", 800, 0.7).chain
                == [("bf", "bpf", 800.0, 0.7)]
            Pattern([(60, 0.25)]).bf("badtype", 1000)  # raises ValueError
        """
        if filter_type not in _BF_TYPES:
            raise ValueError(
                f"unknown filter type: {filter_type!r}; "
                f"must be one of {sorted(_BF_TYPES)}"
            )
        new_chain = list(self.chain) + [
            ("bf", filter_type, float(cutoff), float(resonance))
        ]
        return Pattern._from_parts(self.events, new_chain)

    def spec(self, dsp_fn: Callable, **params) -> "Pattern":
        """Append a spectral / FFT-based stage (excluding convolution).

        Chain entry: ``("spec", dsp_fn, params)``.
        """
        new_chain = list(self.chain) + [("spec", dsp_fn, dict(params))]
        return Pattern._from_parts(self.events, new_chain)

    def ir(self, dsp_fn: Callable, **params) -> "Pattern":
        """Append a convolution / impulse-response / granular stage.

        Covers convolution reverbs, IR-based effects, and granular
        synthesis. Distinct from :meth:`spec` so the scheduler can make
        different optimisation decisions for each.

        Chain entry: ``("ir", dsp_fn, params)``.
        """
        new_chain = list(self.chain) + [("ir", dsp_fn, dict(params))]
        return Pattern._from_parts(self.events, new_chain)

    def gate(self, rhythm_pattern: "Pattern") -> "Pattern":
        """Append a rhythmic-gate stage.

        ``rhythm_pattern`` is a Pattern whose events are
        ``(gate_state, segment_duration)`` with ``gate_state`` being
        ``1`` (open) or ``0`` (closed). The rhythm loops for the full
        duration of the host pattern's playback.

        Chain entry: ``("gate", rhythm_pattern)``.

        Test example (from spec):
            g = Pattern([(1, 0.25), (0, 0.25)])
            p = Pattern([(60, 2.0)]).gate(g)
            p.chain == [("gate", g)]
        """
        new_chain = list(self.chain) + [("gate", rhythm_pattern)]
        return Pattern._from_parts(self.events, new_chain)

    def fx(self, dsp_fn: Callable, **params) -> "Pattern":
        """Append a catch-all custom-DSP stage.

        Covers anything that doesn't fit :meth:`mod`, :meth:`dist`,
        :meth:`bf`, :meth:`spec`, :meth:`ir`, or :meth:`gate` — delays,
        custom filters, odd utilities.

        Chain entry: ``("fx", dsp_fn, params)``.
        """
        new_chain = list(self.chain) + [("fx", dsp_fn, dict(params))]
        return Pattern._from_parts(self.events, new_chain)

    # -- Playback ------------------------------------------------------

    def play(self, shape: Callable, graph) -> None:
        """Play the pattern sequentially on ``graph`` using ``shape``.

        ``shape(note) -> Node`` builds a voice for one event. For each
        ``(note, dur)`` the voice is built, :meth:`_apply_chain` wraps it
        with the chain stages, the result is played, ``graph.wait(dur)``
        stalls until the event is done, and the voice is stopped.
        Empty patterns return immediately.
        """
        if not self.events:
            return
        for (note, dur) in self.events:
            voice = shape(note)
            output = self._apply_chain(voice)
            output.play()
            graph.wait(dur)
            output.stop()

    def _apply_chain(self, voice):
        """Walk ``self.chain`` and wrap ``voice`` with each stage.

        Chain entries are applied left-to-right: index 0 is innermost /
        first applied, the last entry is outermost. ``mod``, ``dist``,
        ``spec``, ``ir``, and ``fx`` all dispatch to the registered
        DSP callable using the fixed signature
        ``fn(input_node, **params) -> Node``. ``bf`` delegates to
        :meth:`_apply_builtin_filter`, and ``gate`` delegates to
        :meth:`_apply_gate`.
        """
        final, _stages = self._apply_chain_with_nodes(voice)
        return final

    def _apply_chain_with_nodes(self, voice):
        """Apply the chain and return both the final node and a list of
        per-stage output nodes.

        ``stages[i]`` is the node produced by chain entry ``i``. The
        scheduler uses this trace to route chain-parameter automation
        to the live node that owns the target parameter — see
        :meth:`Scheduler.bind_chain_param`.
        """
        current = voice
        stages: list = []
        for entry in self.chain:
            kind = entry[0]
            if kind in ("mod", "dist", "spec", "ir", "fx"):
                _, fn, params = entry
                current = fn(current, **params)
            elif kind == "bf":
                _, ftype, cutoff, res = entry
                current = self._apply_builtin_filter(current, ftype, cutoff, res)
            elif kind == "gate":
                _, rhythm = entry
                current = self._apply_gate(current, rhythm)
            else:
                raise ValueError(f"unknown chain entry kind: {kind!r}")
            stages.append(current)
        return current, stages

    def _apply_builtin_filter(self, voice, ftype: str, cutoff: float, res: float):
        """Wrap ``voice`` in a SignalFlow :class:`SVFilter`.

        The spec's short filter names map to SignalFlow's ``filter_type``
        strings via :data:`_BF_SIGNALFLOW_NAMES`. ``cutoff`` is in Hz and
        ``res`` is passed through as-is — SignalFlow expects resonance in
        ``[0, 1]`` but larger values are allowed by :meth:`Pattern.bf`,
        which is consistent with the spec's "wacky territory" convention.

        SignalFlow import is lazy so environments without it can still
        import this module.
        """
        import signalflow as sf

        sf_type = _BF_SIGNALFLOW_NAMES.get(ftype)
        if sf_type is None:
            # Should never happen — ``.bf`` validates on the way in. Guard
            # anyway so a corrupted chain surfaces loudly.
            raise ValueError(f"unknown filter type in chain: {ftype!r}")
        return sf.SVFilter(voice, sf_type, cutoff, res)

    def _apply_gate(self, voice, rhythm: "Pattern"):
        """Apply a rhythmic gate as an amplitude envelope.

        ``rhythm.events`` is a list of ``(gate_state, segment_duration)``
        tuples. ``gate_state`` is treated as binary: any value >= 0.5
        opens the gate (amplitude 1.0), anything else closes it (0.0).
        Durations are in seconds.

        The gate loops for the full length of the host event. It does not
        have to be the same length as the host pattern — we build a
        single-cycle sample buffer and hand it to a
        :class:`signalflow.BufferPlayer` with ``loop=True``, so the gate
        repeats forever until the voice is stopped by :meth:`play`.

        An empty rhythm is treated as a no-op (voice passes through
        unchanged). SignalFlow import is lazy.
        """
        if not rhythm.events:
            return voice

        import signalflow as sf

        sample_rate = _current_sample_rate()
        samples: list[float] = []
        for gate_state, dur in rhythm.events:
            if dur <= 0:
                # Zero-length segments contribute no samples; skip them so
                # the buffer never becomes degenerate.
                continue
            n = max(1, int(round(sample_rate * dur)))
            value = 1.0 if float(gate_state) >= 0.5 else 0.0
            samples.extend([value] * n)

        if not samples:
            return voice

        gate_buffer = sf.Buffer(samples)
        gate_player = sf.BufferPlayer(gate_buffer, loop=True)
        return voice * gate_player

    # -- Misc ----------------------------------------------------------

    def __repr__(self) -> str:
        return f"Pattern(events={self.events!r}, chain_len={len(self.chain)})"
