"""Effects window panels.

Panels:
- EffectBrowserPanel — browsable categorised effect list
- ChainBuilderPanel — ordered effect chain with save
- ConvolutionPanel — convolution reverb engine
- ParamInspectorPanel — per-effect parameter editor
"""

from .effect_browser import EffectBrowserPanel
from .chain_builder import ChainBuilderPanel
from .convolution_panel import ConvolutionPanel
from .param_inspector import ParamInspectorPanel

__all__ = [
    'EffectBrowserPanel',
    'ChainBuilderPanel',
    'ConvolutionPanel',
    'ParamInspectorPanel',
]
