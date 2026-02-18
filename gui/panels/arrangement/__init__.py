"""Arrangement window panels.

Panels:
- TrackListPanel — ordered track list with mute/solo/volume/pan
- PatternLanePanel — per-track pattern/clip placement view
- SongSettingsPanel — BPM, time signature, key, scale, output format
"""

from .track_list import TrackListPanel
from .pattern_lane import PatternLanePanel
from .song_settings import SongSettingsPanel

__all__ = [
    'TrackListPanel',
    'PatternLanePanel',
    'SongSettingsPanel',
]
