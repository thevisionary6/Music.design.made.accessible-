"""Track List Panel — Arrangement Window.

Ordered list of tracks in the current song.  Provides controls for
adding, removing, and configuring tracks (mute, solo, volume, pan).

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Arrangement Window

BUILD ID: panel_track_list_v1.0_phase5
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

# Supported track types
TRACK_TYPES = ['audio', 'beat', 'midi']

if _WX_AVAILABLE:

    class TrackListPanel(wx.Panel):
        """Ordered track list with per-track controls.

        Controls:
        - Track list display (ListBox)
        - Track type selector (audio/beat/midi)
        - Name field for new tracks
        - Add Track button
        - Remove Track button
        - Mute / Solo toggles
        - Volume / Pan sliders

        All actions dispatch through the Bridge.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="Track List")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # ---- Track list display ----
            list_label = wx.StaticText(self, label="Tracks:")
            sizer.Add(list_label, 0, wx.LEFT | wx.TOP, 8)
            self._track_list = wx.ListBox(
                self, choices=[], name="Track List",
            )
            sizer.Add(self._track_list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # ---- Add-track row ----
            add_sizer = wx.BoxSizer(wx.HORIZONTAL)

            name_label = wx.StaticText(self, label="Name:")
            add_sizer.Add(name_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._name_field = wx.TextCtrl(self, name="Track Name")
            self._name_field.SetHint("New track name")
            add_sizer.Add(self._name_field, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)

            type_label = wx.StaticText(self, label="Type:")
            add_sizer.Add(type_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 4)
            self._type_selector = wx.Choice(
                self, choices=TRACK_TYPES, name="Track Type",
            )
            self._type_selector.SetSelection(0)
            add_sizer.Add(self._type_selector, 0, wx.LEFT | wx.RIGHT, 4)

            sizer.Add(add_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Action buttons ----
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._add_btn = wx.Button(self, label="Add Track", name="Add Track")
            self._add_btn.Bind(wx.EVT_BUTTON, self._on_add_track)
            btn_sizer.Add(self._add_btn, 0, wx.ALL, 4)

            self._remove_btn = wx.Button(self, label="Remove Track", name="Remove Track")
            self._remove_btn.Bind(wx.EVT_BUTTON, self._on_remove_track)
            btn_sizer.Add(self._remove_btn, 0, wx.ALL, 4)

            sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            # ---- Mute / Solo toggles ----
            toggle_sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._mute_btn = wx.ToggleButton(self, label="Mute", name="Mute Toggle")
            self._mute_btn.Bind(wx.EVT_TOGGLEBUTTON, self._on_mute)
            toggle_sizer.Add(self._mute_btn, 0, wx.ALL, 4)

            self._solo_btn = wx.ToggleButton(self, label="Solo", name="Solo Toggle")
            self._solo_btn.Bind(wx.EVT_TOGGLEBUTTON, self._on_solo)
            toggle_sizer.Add(self._solo_btn, 0, wx.ALL, 4)

            sizer.Add(toggle_sizer, 0, wx.LEFT, 4)

            # ---- Volume slider ----
            vol_sizer = wx.BoxSizer(wx.HORIZONTAL)
            vol_label = wx.StaticText(self, label="Volume:")
            vol_sizer.Add(vol_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._volume_slider = wx.Slider(
                self, value=80, minValue=0, maxValue=100,
                style=wx.SL_HORIZONTAL | wx.SL_LABELS,
                name="Track Volume",
            )
            self._volume_slider.Bind(wx.EVT_SLIDER, self._on_volume)
            vol_sizer.Add(self._volume_slider, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)
            sizer.Add(vol_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Pan slider ----
            pan_sizer = wx.BoxSizer(wx.HORIZONTAL)
            pan_label = wx.StaticText(self, label="Pan:")
            pan_sizer.Add(pan_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._pan_slider = wx.Slider(
                self, value=50, minValue=0, maxValue=100,
                style=wx.SL_HORIZONTAL | wx.SL_LABELS,
                name="Track Pan",
            )
            self._pan_slider.Bind(wx.EVT_SLIDER, self._on_pan)
            pan_sizer.Add(self._pan_slider, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)
            sizer.Add(pan_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Status ----
            self._status = wx.StaticText(self, label="", name="Track List Status")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        # ---- Helpers ----

        def _selected_track_name(self) -> str:
            """Return the name of the currently selected track, or ''."""
            sel = self._track_list.GetSelection()
            if sel == wx.NOT_FOUND:
                return ""
            return self._track_list.GetString(sel)

        def _set_status(self, text: str) -> None:
            display = text[:160] if len(text) > 160 else text
            self._status.SetLabel(display)
            self._status.Wrap(self.GetSize().width - 16)

        # ---- Event handlers ----

        def _on_add_track(self, event: wx.CommandEvent) -> None:
            """Add a new track to the song."""
            name = self._name_field.GetValue().strip()
            track_type = TRACK_TYPES[self._type_selector.GetSelection()]
            args = [track_type]
            if name:
                args.append(name)
            result = self.bridge.execute_command('/addtrack', args)
            self._set_status(result)
            # Refresh track list
            if name:
                self._track_list.Append(f"{name} ({track_type})")
            self._name_field.SetValue("")

        def _on_remove_track(self, event: wx.CommandEvent) -> None:
            """Remove the selected track."""
            track = self._selected_track_name()
            if not track:
                self._set_status("Select a track first")
                return
            result = self.bridge.execute_command('/rmtrack', [track])
            self._set_status(result)
            sel = self._track_list.GetSelection()
            if sel != wx.NOT_FOUND:
                self._track_list.Delete(sel)

        def _on_mute(self, event: wx.CommandEvent) -> None:
            """Toggle mute on the selected track."""
            track = self._selected_track_name()
            if not track:
                self._set_status("Select a track first")
                return
            state = "on" if self._mute_btn.GetValue() else "off"
            result = self.bridge.execute_command('/mute', [track, state])
            self._set_status(result)

        def _on_solo(self, event: wx.CommandEvent) -> None:
            """Toggle solo on the selected track."""
            track = self._selected_track_name()
            if not track:
                self._set_status("Select a track first")
                return
            state = "on" if self._solo_btn.GetValue() else "off"
            result = self.bridge.execute_command('/solo', [track, state])
            self._set_status(result)

        def _on_volume(self, event: wx.CommandEvent) -> None:
            """Adjust track volume."""
            track = self._selected_track_name()
            if not track:
                return
            vol = self._volume_slider.GetValue()
            result = self.bridge.execute_command('/tvol', [track, str(vol)])
            self._set_status(result)

        def _on_pan(self, event: wx.CommandEvent) -> None:
            """Adjust track pan."""
            track = self._selected_track_name()
            if not track:
                return
            pan = self._pan_slider.GetValue()
            result = self.bridge.execute_command('/tpan', [track, str(pan)])
            self._set_status(result)

else:
    class TrackListPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
