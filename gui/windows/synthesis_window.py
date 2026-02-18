"""Synthesis Window — Sound design and patch editing.

This window handles all synthesis-related tasks: FM operator
configuration, waveform selection, ADSR envelopes, LFO modulation,
physical modeling, and preset management.  Contains six panels
in a tabbed notebook:

- Operator Panel     → FM operator grid (/op)
- Waveform Panel     → waveform type selector (/wave)
- Envelope Panel     → ADSR controls (/attack, /decay, /sustain, /release)
- Modulation Panel   → LFO routing (/lfo)
- Physical Modeling  → waveguide synthesis (/waveguide)
- Preset Browser     → patch presets (/preset)

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Synthesis Window

BUILD ID: window_synthesis_v1.0_phase5
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

    class SynthesisWindow(wx.Frame):
        """Synthesis workflow window with tabbed panels.

        Houses six panels in a notebook for tabbed navigation.
        All panels dispatch through the shared Bridge instance.
        """

        def __init__(
            self,
            parent: wx.Window,
            bridge: Any,
            theme: dict | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                parent,
                title="MDMA — Synthesis",
                size=(750, 650),
                **kwargs,
            )
            self.bridge = bridge
            self._theme = theme or {}

            if self._theme:
                self.SetBackgroundColour(
                    wx.Colour(*self._theme.get("BG_DARK", (30, 30, 35)))
                )

            self._build_ui()
            logger.info("SynthesisWindow created")

        def _build_ui(self) -> None:
            from gui.panels.synthesis.operator_panel import OperatorPanel
            from gui.panels.synthesis.waveform_panel import WaveformPanel
            from gui.panels.synthesis.envelope_panel import EnvelopePanel
            from gui.panels.synthesis.modulation_panel import ModulationPanel
            from gui.panels.synthesis.physical_modeling import PhysicalModelingPanel
            from gui.panels.synthesis.preset_browser import PresetBrowserPanel

            notebook = wx.Notebook(self, name="Synthesis Panels")

            self._operator = OperatorPanel(notebook, self.bridge)
            notebook.AddPage(self._operator, "Operators")

            self._waveform = WaveformPanel(notebook, self.bridge)
            notebook.AddPage(self._waveform, "Waveforms")

            self._envelope = EnvelopePanel(notebook, self.bridge)
            notebook.AddPage(self._envelope, "Envelope")

            self._modulation = ModulationPanel(notebook, self.bridge)
            notebook.AddPage(self._modulation, "Modulation")

            self._physical = PhysicalModelingPanel(notebook, self.bridge)
            notebook.AddPage(self._physical, "Physical Model")

            self._presets = PresetBrowserPanel(notebook, self.bridge)
            notebook.AddPage(self._presets, "Presets")

            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(notebook, 1, wx.EXPAND)
            self.SetSizer(sizer)

else:
    class SynthesisWindow:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
