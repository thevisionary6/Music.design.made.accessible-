"""Core functionality for the MDMA rebuild.

This subpackage contains the Session class, which maintains state for
projects, operators, filters, envelopes and playback.  It also
provides minimal support for tone generation and writing WAV files.

The object model (objects.py) and registry (registry.py) provide the
first-class object system described in docs/specs/OBJECT_MODEL_SPEC.md.
"""

from .session import Session  # noqa: F401

# Object model — no heavy dependencies (no numpy/scipy required)
from .objects import MDMAObject, Pattern, BeatPattern, Loop, AudioClip  # noqa: F401
from .objects import Patch, EffectChain, Track, Song, Template  # noqa: F401
from .registry import ObjectRegistry  # noqa: F401