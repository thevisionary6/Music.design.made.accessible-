"""Concrete controllers for :mod:`mdma_rebuild.backend.input_controller`.

Each concrete controller lives in its own module so its external
dependency (``pynput``, ``mido``, ``python-osc``) is imported lazily.
Callers that only use one controller pay import cost for only that
one.

Gamepad is explicitly deferred (spec §v1.1). The interface is open via
:class:`mdma_rebuild.backend.input_controller.Controller` when the
implementation lands.
"""

from .keyboard import KeyboardController
from .midi import MidiController
from .osc import OSCController

__all__ = [
    "KeyboardController",
    "MidiController",
    "OSCController",
]
