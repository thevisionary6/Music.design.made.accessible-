"""Inspector & Console Window — Persistent accessibility anchor.

Always available regardless of what other windows are open.
Primary orientation point for screen reader users.

Contains three panels:
- Object Tree — registry browser
- Parameter Inspector — selected object detail
- Console/CLI Mirror — command input and output

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Inspector Window

BUILD ID: window_inspector_v1.0_phase8
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

    class InspectorWindow(wx.Frame):
        """Inspector & Console window with split layout.

        Left: Object tree + parameter inspector (vertical split)
        Right: Console/CLI mirror
        """

        def __init__(
            self,
            parent: wx.Window,
            bridge: Any,
            theme: dict | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                parent, title="MDMA — Inspector & Console",
                size=(900, 700), **kwargs,
            )
            self.bridge = bridge
            self._theme = theme or {}
            if self._theme:
                self.SetBackgroundColour(
                    wx.Colour(*self._theme.get("BG_DARK", (30, 30, 35)))
                )
            self._build_ui()
            logger.info("InspectorWindow created")

        def _build_ui(self) -> None:
            from gui.panels.inspector.object_tree import ObjectTreePanel
            from gui.panels.inspector.console_panel import ConsolePanel
            from gui.panels.inspector.parameter_inspector import ParameterInspectorPanel

            # Main horizontal splitter: tree/detail | console
            main_splitter = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE)
            main_splitter.SetMinimumPaneSize(200)

            # Left side: vertical splitter for tree and detail
            left_splitter = wx.SplitterWindow(main_splitter, style=wx.SP_LIVE_UPDATE)
            left_splitter.SetMinimumPaneSize(150)

            self._tree_panel = ObjectTreePanel(left_splitter, self.bridge)
            self._detail_panel = ParameterInspectorPanel(left_splitter, self.bridge)

            left_splitter.SplitHorizontally(self._tree_panel, self._detail_panel, 350)

            # Right side: console
            self._console_panel = ConsolePanel(main_splitter, self.bridge)

            main_splitter.SplitVertically(left_splitter, self._console_panel, 350)

            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(main_splitter, 1, wx.EXPAND)
            self.SetSizer(sizer)

            # Wire tree selection to parameter inspector
            self._tree_panel._tree.Bind(
                wx.EVT_TREE_SEL_CHANGED, self._on_tree_select
            )

        def _on_tree_select(self, event: wx.TreeEvent) -> None:
            obj_id = self._tree_panel._get_selected_object_id()
            if obj_id:
                self._detail_panel.show_object(obj_id)

        def refresh_tree(self) -> None:
            """Refresh the object tree from registry."""
            self._tree_panel.refresh()

        def append_console(self, text: str) -> None:
            """Add text to the console output."""
            self._console_panel.append_text(text)

        # Backward compatibility with shell.py references
        @property
        def _tree_ctrl(self) -> wx.TreeCtrl:
            return self._tree_panel._tree

        @property
        def _console_output(self) -> wx.TextCtrl:
            return self._console_panel._output

        @property
        def _cmd_input(self) -> wx.TextCtrl:
            return self._console_panel._input

else:
    class InspectorWindow:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
