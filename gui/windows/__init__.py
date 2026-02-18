"""MDMA GUI Sub-Windows.

Each module defines a wx.Frame subclass representing one workflow
window as specified in GUI_WINDOW_ARCHITECTURE_SPEC.md:

- generation_window: Create new objects (beats, melodies, loops)
- mutation_window: Transform existing objects
- effects_window: Apply signal processing
- synthesis_window: Sound design / patch editing
- arrangement_window: Track and song assembly
- mixing_window: DJ decks and master output
- inspector_window: Object tree, console, status (always available)
"""

from .generation_window import GenerationWindow
from .effects_window import EffectsWindow
from .synthesis_window import SynthesisWindow
from .mutation_window import MutationWindow
from .arrangement_window import ArrangementWindow
from .mixing_window import MixingWindow
from .inspector_window import InspectorWindow

__all__ = [
    'GenerationWindow',
    'EffectsWindow',
    'SynthesisWindow',
    'MutationWindow',
    'ArrangementWindow',
    'MixingWindow',
    'InspectorWindow',
]
