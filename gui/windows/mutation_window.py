"""Mutation Window — Transform existing objects.

This window is strictly transformative — it operates on objects that
already exist in the object tree.  It does not generate new content
from scratch.  All operations are non-destructive by default: the
source object is preserved and a new output object is created.

Contains four panels:

- Transform      -> /xform: reverse, invert, stretch, etc.
- Adapt          -> /adapt: key, tempo, style, develop
- Pattern Editor -> direct step-level editing of Pattern objects
- Splice & Combine -> merge two objects (stitch, layer, interleave)

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Mutation Window

BUILD ID: window_mutation_v1.0_phase5
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

    class MutationWindow(wx.Frame):
        """Mutation workflow window with tabbed panels.

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
                title="MDMA — Mutation && Editing",
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
            logger.info("MutationWindow created")

        def _build_ui(self) -> None:
            from gui.panels.mutation.transform_panel import TransformPanel
            from gui.panels.mutation.adapt_panel import AdaptPanel
            from gui.panels.mutation.pattern_editor import PatternEditorPanel
            from gui.panels.mutation.splice_combine import SpliceCombinePanel

            notebook = wx.Notebook(self, name="Mutation Panels")

            self._transform = TransformPanel(notebook, self.bridge)
            notebook.AddPage(self._transform, "Transform")

            self._adapt = AdaptPanel(notebook, self.bridge)
            notebook.AddPage(self._adapt, "Adapt")

            self._editor = PatternEditorPanel(notebook, self.bridge)
            notebook.AddPage(self._editor, "Pattern Editor")

            self._splice = SpliceCombinePanel(notebook, self.bridge)
            notebook.AddPage(self._splice, "Splice && Combine")

            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(notebook, 1, wx.EXPAND)
            self.SetSizer(sizer)

else:
    class MutationWindow:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
