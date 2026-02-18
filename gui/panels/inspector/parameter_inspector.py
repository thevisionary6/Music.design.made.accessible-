"""Parameter Inspector Panel — Inspector Window.

Shows the full detail of whatever object is currently selected
anywhere in the application.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Inspector Window

BUILD ID: panel_parameter_inspector_v1.0_phase8
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

    class ParameterInspectorPanel(wx.Panel):
        """Detailed parameter view of a selected object.

        Displays all fields of the currently selected registry object.
        Updates when a new object is selected in the tree.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._current_id: str = ""
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            label = wx.StaticText(self, label="Object Detail")
            label.SetFont(label.GetFont().Bold())
            sizer.Add(label, 0, wx.ALL, 5)

            self._detail = wx.TextCtrl(
                self,
                style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
                name="Object Detail",
            )
            sizer.Add(self._detail, 1, wx.EXPAND | wx.ALL, 4)

            # Action buttons
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

            btn_rename = wx.Button(self, label="Rename")
            btn_rename.Bind(wx.EVT_BUTTON, self._on_rename)
            btn_sizer.Add(btn_rename, 0, wx.ALL, 2)

            btn_dup = wx.Button(self, label="Duplicate")
            btn_dup.Bind(wx.EVT_BUTTON, self._on_duplicate)
            btn_sizer.Add(btn_dup, 0, wx.ALL, 2)

            btn_tag = wx.Button(self, label="Add Tag")
            btn_tag.Bind(wx.EVT_BUTTON, self._on_tag)
            btn_sizer.Add(btn_tag, 0, wx.ALL, 2)

            sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            self._status = wx.StaticText(self, label="Select an object to inspect")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 4)

            self.SetSizer(sizer)

        def show_object(self, object_id: str) -> None:
            """Display details for the given object ID."""
            self._current_id = object_id
            result = self.bridge.execute_command('/obj', ['info', object_id[:8]])
            self._detail.SetValue(result)

        def _on_rename(self, event: wx.CommandEvent) -> None:
            if not self._current_id:
                return
            dlg = wx.TextEntryDialog(self, "New name:", "Rename Object")
            if dlg.ShowModal() == wx.ID_OK:
                new_name = dlg.GetValue().strip()
                if new_name:
                    result = self.bridge.execute_command(
                        '/obj', ['rename', self._current_id[:8], new_name]
                    )
                    self._status.SetLabel(result[:120])
                    self.show_object(self._current_id)
            dlg.Destroy()

        def _on_duplicate(self, event: wx.CommandEvent) -> None:
            if not self._current_id:
                return
            result = self.bridge.execute_command('/obj', ['dup', self._current_id[:8]])
            self._status.SetLabel(result[:120])

        def _on_tag(self, event: wx.CommandEvent) -> None:
            if not self._current_id:
                return
            dlg = wx.TextEntryDialog(self, "Tag:", "Add Tag")
            if dlg.ShowModal() == wx.ID_OK:
                tag = dlg.GetValue().strip()
                if tag:
                    result = self.bridge.execute_command(
                        '/obj', ['tag', self._current_id[:8], tag]
                    )
                    self._status.SetLabel(result[:120])
                    self.show_object(self._current_id)
            dlg.Destroy()

else:
    class ParameterInspectorPanel:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
