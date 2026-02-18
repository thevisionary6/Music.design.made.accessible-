"""Generation window panels.

Panels:
- BeatGeneratorPanel — /beat command wrapper
- MelodyHarmonyPanel — /gen2 melody/chords/bassline/arp/drone
- LoopGeneratorPanel — /loop command wrapper
- GenerativeTheoryPanel — /theory and /key tools
"""

from .beat_generator import BeatGeneratorPanel
from .melody_harmony import MelodyHarmonyPanel
from .loop_generator import LoopGeneratorPanel
from .generative_theory import GenerativeTheoryPanel

__all__ = [
    'BeatGeneratorPanel',
    'MelodyHarmonyPanel',
    'LoopGeneratorPanel',
    'GenerativeTheoryPanel',
]
