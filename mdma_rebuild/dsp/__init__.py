"""DSP primitives for the MDMA rebuild.

This subpackage contains simple signal generation and processing code
used by the Session object.  The MonolithEngine provides basic
operator handling and tone rendering.  The envelopes module
implements a simple ADSR envelope.

Modules:
- scaling: Unified 1-100 parameter scaling system (canonical source of truth)
- monolith: Operator-based offline synth engine
- envelopes: ADSR envelope implementation
- effects: Audio effects and filters
- buffer_rearranger: Audio-rate buffer-rearrangement engine (was dsp/pattern.py; renamed for the V2 backend merge because the old "Pattern" was actually a buffer slicer, not a sequencer Pattern)
- playback: In-house audio playback (no external media player calls)

BUILD ID: dsp_v14.2
"""

__all__ = [
    "scaling",
    "monolith",
    "envelopes",
    "effects",
    "buffer_rearranger",
    "playback",
    "advanced_ops",
    "dj_mode",
    "enhancement",
    "generators",
    "granular",
    "performance",
    "stems",
    "streaming",
    "music_theory",
    "beat_gen",
    "loop_gen",
    "transforms",
]