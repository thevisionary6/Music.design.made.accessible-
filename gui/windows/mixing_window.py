"""Mixing Window — DJ decks and master output.

Houses four panels in a notebook for tabbed navigation:

- Deck Panel       -> per-deck load/play/stop/pitch controls
- Crossfader       -> crossfader slider between decks A and B
- Master Channel   -> master volume, effects chain, render
- Stem Separation  -> stem extraction from audio sources

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Mixing Window

BUILD ID: window_mixing_v1.0_phase5
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

    class MixingWindow(wx.Frame):
        """Mixing workflow window with tabbed panels."""

        def __init__(
            self,
            parent: wx.Window,
            bridge: Any,
            theme: dict | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                parent, title="MDMA — Mixing", size=(750, 600), **kwargs,
            )
            self.bridge = bridge
            self._theme = theme or {}
            if self._theme:
                self.SetBackgroundColour(
                    wx.Colour(*self._theme.get("BG_DARK", (30, 30, 35)))
                )
            self._build_ui()
            logger.info("MixingWindow created")

        def _build_ui(self) -> None:
            from gui.panels.mixing.deck_panel import DeckPanel
            from gui.panels.mixing.crossfader_panel import CrossfaderPanel
            from gui.panels.mixing.master_channel import MasterChannelPanel
            from gui.panels.mixing.stem_panel import StemPanel

            notebook = wx.Notebook(self, name="Mixing Panels")

            self._deck = DeckPanel(notebook, self.bridge)
            notebook.AddPage(self._deck, "Decks")

            self._crossfader = CrossfaderPanel(notebook, self.bridge)
            notebook.AddPage(self._crossfader, "Crossfader")

            self._master = MasterChannelPanel(notebook, self.bridge)
            notebook.AddPage(self._master, "Master")

            self._stem = StemPanel(notebook, self.bridge)
            notebook.AddPage(self._stem, "Stems")

            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(notebook, 1, wx.EXPAND)
            self.SetSizer(sizer)

else:
    class MixingWindow:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
