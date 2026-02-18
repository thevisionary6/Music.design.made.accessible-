"""Generation Window — Create new first-class objects from scratch.

This window is strictly creative — it produces objects but does not
modify existing ones.  Contains four panels:

- Beat Generator   → BeatPattern + AudioClip
- Melody & Harmony → Pattern + AudioClip
- Loop Generator   → Loop + AudioClip
- Generative Theory → Pattern (theory-driven)

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Generation Window

BUILD ID: window_generation_v1.0_phase3
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

    class GenerationWindow(wx.Frame):
        """Generation workflow window.

        Houses four panels in a notebook for tabbed navigation.
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
                title="MDMA — Generation",
                size=(700, 600),
                **kwargs,
            )
            self.bridge = bridge
            self._theme = theme or {}

            if self._theme:
                self.SetBackgroundColour(wx.Colour(*self._theme.get("BG_DARK", (30, 30, 35))))

            self._build_ui()
            logger.info("GenerationWindow created")

        def _build_ui(self) -> None:
            from gui.panels.generation.beat_generator import BeatGeneratorPanel
            from gui.panels.generation.melody_harmony import MelodyHarmonyPanel
            from gui.panels.generation.loop_generator import LoopGeneratorPanel
            from gui.panels.generation.generative_theory import GenerativeTheoryPanel

            notebook = wx.Notebook(self, name="Generation Panels")

            self._beat_panel = BeatGeneratorPanel(notebook, self.bridge)
            notebook.AddPage(self._beat_panel, "Beat Generator")

            self._melody_panel = MelodyHarmonyPanel(notebook, self.bridge)
            notebook.AddPage(self._melody_panel, "Melody && Harmony")

            self._loop_panel = LoopGeneratorPanel(notebook, self.bridge)
            notebook.AddPage(self._loop_panel, "Loop Generator")

            self._theory_panel = GenerativeTheoryPanel(notebook, self.bridge)
            notebook.AddPage(self._theory_panel, "Music Theory")

            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(notebook, 1, wx.EXPAND)
            self.SetSizer(sizer)

else:
    class GenerationWindow:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
