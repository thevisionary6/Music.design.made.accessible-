"""Effects Window — Apply signal processing to audio objects.

Strictly DSP — does not generate or mutate musical content.
Contains four panels in a notebook:

- Effect Browser   → browsable categorised list
- Chain Builder    → build/save EffectChain objects
- Convolution      → IR reverb engine
- Param Inspector  → per-effect parameter editing

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Effects Window

BUILD ID: window_effects_v1.0_phase4
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import wx
    _WX_AVAILABLE = True
except ImportError:
    _WX_AVAILABLE = False

if _WX_AVAILABLE:

    class EffectsWindow(wx.Frame):
        """Effects workflow window with tabbed panels."""

        def __init__(
            self,
            parent: wx.Window,
            bridge: Any,
            theme: dict | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                parent, title="MDMA — Effects", size=(750, 600), **kwargs,
            )
            self.bridge = bridge
            self._theme = theme or {}
            if self._theme:
                self.SetBackgroundColour(
                    wx.Colour(*self._theme.get("BG_DARK", (30, 30, 35)))
                )
            self._build_ui()
            logger.info("EffectsWindow created")

        def _build_ui(self) -> None:
            from gui.panels.effects.effect_browser import EffectBrowserPanel
            from gui.panels.effects.chain_builder import ChainBuilderPanel
            from gui.panels.effects.convolution_panel import ConvolutionPanel
            from gui.panels.effects.param_inspector import ParamInspectorPanel

            notebook = wx.Notebook(self, name="Effects Panels")

            self._browser = EffectBrowserPanel(notebook, self.bridge)
            notebook.AddPage(self._browser, "Effect Browser")

            self._chain = ChainBuilderPanel(notebook, self.bridge)
            notebook.AddPage(self._chain, "Chain Builder")

            self._conv = ConvolutionPanel(notebook, self.bridge)
            notebook.AddPage(self._conv, "Convolution")

            self._params = ParamInspectorPanel(notebook, self.bridge)
            notebook.AddPage(self._params, "Parameters")

            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(notebook, 1, wx.EXPAND)
            self.SetSizer(sizer)

else:
    class EffectsWindow:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
