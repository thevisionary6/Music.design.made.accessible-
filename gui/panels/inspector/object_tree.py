"""Object Tree Panel — Inspector Window.

Full hierarchical view of the object registry.  Browse all objects by
type.  Select any object to inspect or send to another window for editing.
Keyboard navigable with full screen reader labels.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Inspector Window

BUILD ID: panel_object_tree_v1.0_phase8
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

    class ObjectTreePanel(wx.Panel):
        """Hierarchical object registry browser.

        Displays objects grouped by type in a tree control.
        Auto-refreshes when registry events fire.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            label = wx.StaticText(self, label="Object Registry")
            label.SetFont(label.GetFont().Bold())
            sizer.Add(label, 0, wx.ALL, 5)

            self._tree = wx.TreeCtrl(
                self,
                style=wx.TR_DEFAULT_STYLE | wx.TR_HIDE_ROOT,
                name="Object Tree",
            )
            sizer.Add(self._tree, 1, wx.EXPAND | wx.ALL, 2)

            # Action buttons
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
            btn_refresh = wx.Button(self, label="Refresh")
            btn_refresh.Bind(wx.EVT_BUTTON, lambda e: self.refresh())
            btn_sizer.Add(btn_refresh, 0, wx.ALL, 2)

            btn_info = wx.Button(self, label="Inspect")
            btn_info.Bind(wx.EVT_BUTTON, self._on_inspect)
            btn_sizer.Add(btn_info, 0, wx.ALL, 2)

            btn_delete = wx.Button(self, label="Delete")
            btn_delete.Bind(wx.EVT_BUTTON, self._on_delete)
            btn_sizer.Add(btn_delete, 0, wx.ALL, 2)

            sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            self._status = wx.StaticText(self, label="")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 4)

            self.SetSizer(sizer)
            self.refresh()

        def refresh(self) -> None:
            """Rebuild tree from registry."""
            self._tree.DeleteAllItems()
            root = self._tree.AddRoot("Registry")

            from mdma_rebuild.core.objects import OBJECT_TYPE_MAP
            registry = self.bridge.registry

            total = 0
            for type_name in OBJECT_TYPE_MAP:
                objects = registry.list_objects(type_name)
                label = type_name.replace("_", " ").title()
                if objects:
                    label += f" ({len(objects)})"
                type_node = self._tree.AppendItem(root, label)

                for obj in objects:
                    detail = obj.name
                    if hasattr(obj, "genre") and obj.genre:
                        detail += f" — {obj.genre}"
                    elif hasattr(obj, "pattern_kind") and obj.pattern_kind:
                        detail += f" — {obj.pattern_kind}"
                    elif hasattr(obj, "duration_seconds") and obj.duration_seconds > 0:
                        detail += f" — {obj.duration_seconds:.2f}s"
                    item = self._tree.AppendItem(type_node, detail)
                    self._tree.SetItemData(item, obj.id)
                    total += 1

                if objects:
                    self._tree.Expand(type_node)

            self._status.SetLabel(f"{total} objects")

        def _get_selected_object_id(self) -> str:
            item = self._tree.GetSelection()
            if not item.IsOk():
                return ""
            data = self._tree.GetItemData(item)
            return data if data else ""

        def _on_inspect(self, event: wx.CommandEvent) -> None:
            obj_id = self._get_selected_object_id()
            if not obj_id:
                self._status.SetLabel("Select an object first")
                return
            result = self.bridge.execute_command('/obj', ['info', obj_id[:8]])
            self._status.SetLabel(result[:200] if len(result) > 200 else result)

        def _on_delete(self, event: wx.CommandEvent) -> None:
            obj_id = self._get_selected_object_id()
            if not obj_id:
                self._status.SetLabel("Select an object first")
                return
            result = self.bridge.execute_command('/obj', ['delete', obj_id[:8]])
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self.refresh()

else:
    class ObjectTreePanel:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
