"""Mixing window panels.

Panels:
- DeckPanel — per-deck load/play/stop/pitch controls
- CrossfaderPanel — crossfader slider between decks A and B
- MasterChannelPanel — master volume, effects chain, render
- StemPanel — stem separation from audio sources
"""

from .deck_panel import DeckPanel
from .crossfader_panel import CrossfaderPanel
from .master_channel import MasterChannelPanel
from .stem_panel import StemPanel

__all__ = [
    'DeckPanel',
    'CrossfaderPanel',
    'MasterChannelPanel',
    'StemPanel',
]
