"""Pattern Lane Panel — Arrangement Window.

Per-track view of pattern and clip placements along the timeline.
Allows placing and removing objects at specific positions within
a track.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Arrangement Window

BUILD ID: panel_pattern_lane_v1.0_phase5
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

    class PatternLanePanel(wx.Panel):
        """Per-track pattern/clip placement view.

        Controls:
        - Track selector (choose which track to view)
        - Position spinner (bar/beat position for placement)
        - Object name field (name of pattern/clip to place)
        - Place button (add object at position)
        - Remove button (remove object at position)
        - Placement list (shows current placements)

        All actions dispatch through the Bridge.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="Pattern Lane")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # ---- Track selector ----
            track_sizer = wx.BoxSizer(wx.HORIZONTAL)
            track_label = wx.StaticText(self, label="Track:")
            track_sizer.Add(track_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._track_selector = wx.TextCtrl(self, name="Track Selector")
            self._track_selector.SetHint("Track name")
            track_sizer.Add(self._track_selector, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)

            self._refresh_btn = wx.Button(self, label="Refresh", name="Refresh Placements")
            self._refresh_btn.Bind(wx.EVT_BUTTON, self._on_refresh)
            track_sizer.Add(self._refresh_btn, 0, wx.LEFT | wx.RIGHT, 4)
            sizer.Add(track_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Placement list ----
            list_label = wx.StaticText(self, label="Placements:")
            sizer.Add(list_label, 0, wx.LEFT | wx.TOP, 8)
            self._placement_list = wx.ListBox(
                self, choices=[], name="Placement List",
            )
            sizer.Add(self._placement_list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # ---- Position spinner ----
            pos_sizer = wx.BoxSizer(wx.HORIZONTAL)
            pos_label = wx.StaticText(self, label="Position (bar):")
            pos_sizer.Add(pos_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._position_spin = wx.SpinCtrl(
                self, min=1, max=9999, initial=1, name="Placement Position",
            )
            pos_sizer.Add(self._position_spin, 0, wx.LEFT | wx.RIGHT, 4)
            sizer.Add(pos_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Object name field ----
            obj_sizer = wx.BoxSizer(wx.HORIZONTAL)
            obj_label = wx.StaticText(self, label="Object:")
            obj_sizer.Add(obj_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._object_name = wx.TextCtrl(self, name="Object Name")
            self._object_name.SetHint("Pattern or clip name")
            obj_sizer.Add(self._object_name, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)
            sizer.Add(obj_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Place / Remove buttons ----
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._place_btn = wx.Button(self, label="Place", name="Place Object")
            self._place_btn.Bind(wx.EVT_BUTTON, self._on_place)
            btn_sizer.Add(self._place_btn, 0, wx.ALL, 4)

            self._remove_btn = wx.Button(self, label="Remove", name="Remove Object")
            self._remove_btn.Bind(wx.EVT_BUTTON, self._on_remove)
            btn_sizer.Add(self._remove_btn, 0, wx.ALL, 4)

            sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            # ---- Status ----
            self._status = wx.StaticText(self, label="", name="Pattern Lane Status")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        # ---- Helpers ----

        def _set_status(self, text: str) -> None:
            display = text[:160] if len(text) > 160 else text
            self._status.SetLabel(display)
            self._status.Wrap(self.GetSize().width - 16)

        # ---- Event handlers ----

        def _on_place(self, event: wx.CommandEvent) -> None:
            """Place an object on the track at the given position."""
            track = self._track_selector.GetValue().strip()
            obj_name = self._object_name.GetValue().strip()
            position = self._position_spin.GetValue()

            if not track:
                self._set_status("Enter a track name first")
                return
            if not obj_name:
                self._set_status("Enter an object name to place")
                return

            result = self.bridge.execute_command(
                '/place', [track, obj_name, str(position)],
            )
            self._set_status(result)
            # Add to visible list
            self._placement_list.Append(
                f"Bar {position}: {obj_name}"
            )

        def _on_remove(self, event: wx.CommandEvent) -> None:
            """Remove the object at the given position from the track."""
            track = self._track_selector.GetValue().strip()
            position = self._position_spin.GetValue()

            if not track:
                self._set_status("Enter a track name first")
                return

            result = self.bridge.execute_command(
                '/unplace', [track, str(position)],
            )
            self._set_status(result)

            # Remove matching entry from visible list
            for i in range(self._placement_list.GetCount()):
                if self._placement_list.GetString(i).startswith(f"Bar {position}:"):
                    self._placement_list.Delete(i)
                    break

        def _on_refresh(self, event: wx.CommandEvent) -> None:
            """Refresh the placement list for the current track."""
            track = self._track_selector.GetValue().strip()
            if not track:
                self._set_status("Enter a track name first")
                return
            result = self.bridge.execute_command('/showtrack', [track])
            self._set_status(result)
            # Clear and repopulate from result
            self._placement_list.Clear()
            for line in result.splitlines():
                line = line.strip()
                if line:
                    self._placement_list.Append(line)

else:
    class PatternLanePanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
