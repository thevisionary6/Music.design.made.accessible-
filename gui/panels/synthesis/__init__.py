"""Synthesis window panels — sound design and patch editing.

Panels:
- OperatorPanel — FM operator grid (/op)
- WaveformPanel — waveform type selector (/wave)
- EnvelopePanel — ADSR controls (/attack, /decay, /sustain, /release)
- ModulationPanel — LFO modulation routing (/lfo)
- PhysicalModelingPanel — waveguide synthesis (/waveguide)
- PresetBrowserPanel — patch preset management (/preset)
"""

from .operator_panel import OperatorPanel
from .waveform_panel import WaveformPanel
from .envelope_panel import EnvelopePanel
from .modulation_panel import ModulationPanel
from .physical_modeling import PhysicalModelingPanel
from .preset_browser import PresetBrowserPanel

__all__ = [
    'OperatorPanel',
    'WaveformPanel',
    'EnvelopePanel',
    'ModulationPanel',
    'PhysicalModelingPanel',
    'PresetBrowserPanel',
]
