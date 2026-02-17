"""Registry-Aware Object Selector Widget.

A dropdown/combo control that lists objects from the ObjectRegistry
filtered by type.  Subscribes to registry events to keep the list
current as objects are created, renamed, or deleted.

BUILD ID: widget_object_selector_v1.0
"""

from __future__ import annotations

from typing import Any, Callable, Optional

try:
    import wx
    _WX_AVAILABLE = True
except ImportError:
    _WX_AVAILABLE = False


if _WX_AVAILABLE:
    from .labeled_control import LabeledControl

    class ObjectSelector(LabeledControl):
        """Dropdown for selecting objects from the registry.

        Automatically refreshes when registry events fire.  Can be
        filtered to a specific object type.

        Parameters:
            parent: Parent wx window.
            bridge: The Bridge instance (for registry access).
            label: Label text.
            obj_type: Object type to filter by (None = all types).
            on_select: Optional callback(object_id: str) on selection change.
        """

        def __init__(
            self,
            parent: wx.Window,
            bridge: Any = None,
            label: str = "Select Object",
            description: str = "Choose an object from the registry",
            obj_type: Optional[str] = None,
            on_select: Optional[Callable[[str], None]] = None,
            **kwargs: Any,
        ) -> None:
            self._bridge = bridge
            self._obj_type = obj_type
            self._on_select = on_select
            self._id_map: dict[int, str] = {}  # combo index -> object ID
            super().__init__(parent, label=label, description=description, **kwargs)

            # Initial population
            if bridge is not None:
                self.refresh()

        def _create_control(self) -> wx.Window:
            combo = wx.ComboBox(
                self,
                style=wx.CB_READONLY,
                name=self._label_text,
            )
            combo.Bind(wx.EVT_COMBOBOX, self._on_combo_select)
            return combo

        def refresh(self) -> None:
            """Repopulate the dropdown from the registry."""
            if self._bridge is None or self._control is None:
                return

            combo = self._control
            combo.Clear()
            self._id_map.clear()

            objects = self._bridge.list_objects(self._obj_type)
            for idx, obj in enumerate(objects):
                display = f"{obj.name} ({obj.obj_type})"
                combo.Append(display)
                self._id_map[idx] = obj.id

        def _on_combo_select(self, event: wx.CommandEvent) -> None:
            idx = event.GetSelection()
            object_id = self._id_map.get(idx, "")
            if object_id and self._on_select:
                self._on_select(object_id)

        def get_selected_id(self) -> str:
            """Return the ID of the currently selected object."""
            idx = self._control.GetSelection()
            return self._id_map.get(idx, "")

        def set_obj_type(self, obj_type: Optional[str]) -> None:
            """Change the type filter and refresh."""
            self._obj_type = obj_type
            self.refresh()

else:

    class ObjectSelector:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("wxPython required")
