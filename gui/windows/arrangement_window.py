"""Arrangement Window — Track and song assembly.

Houses three panels in a notebook for tabbed navigation:

- Track List     -> ordered track management with mute/solo/volume/pan
- Pattern Lane   -> per-track pattern/clip placements
- Song Settings  -> BPM, key, scale, time signature, output format

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Arrangement Window

BUILD ID: window_arrangement_v1.0_phase5
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

    class ArrangementWindow(wx.Frame):
        """Arrangement workflow window with tabbed panels."""

        def __init__(
            self,
            parent: wx.Window,
            bridge: Any,
            theme: dict | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                parent, title="MDMA — Arrangement", size=(750, 600), **kwargs,
            )
            self.bridge = bridge
            self._theme = theme or {}
            if self._theme:
                self.SetBackgroundColour(
                    wx.Colour(*self._theme.get("BG_DARK", (30, 30, 35)))
                )
            self._build_ui()
            logger.info("ArrangementWindow created")

        def _build_ui(self) -> None:
            from gui.panels.arrangement.track_list import TrackListPanel
            from gui.panels.arrangement.pattern_lane import PatternLanePanel
            from gui.panels.arrangement.song_settings import SongSettingsPanel

            notebook = wx.Notebook(self, name="Arrangement Panels")

            self._track_list = TrackListPanel(notebook, self.bridge)
            notebook.AddPage(self._track_list, "Track List")

            self._pattern_lane = PatternLanePanel(notebook, self.bridge)
            notebook.AddPage(self._pattern_lane, "Pattern Lane")

            self._song_settings = SongSettingsPanel(notebook, self.bridge)
            notebook.AddPage(self._song_settings, "Song Settings")

            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(notebook, 1, wx.EXPAND)
            self.SetSizer(sizer)

else:
    class ArrangementWindow:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
