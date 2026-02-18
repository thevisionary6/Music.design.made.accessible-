"""Master Channel Panel — Mixing Window.

Master output controls: volume, effects chain display, output format,
HQ mode toggle, and render-to-file.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Mixing Window

BUILD ID: panel_master_channel_v1.0_phase5
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

# Output formats matching song settings
OUTPUT_FORMATS = ['wav_16', 'wav_24', 'flac_24']

if _WX_AVAILABLE:

    class MasterChannelPanel(wx.Panel):
        """Master output channel controls.

        Controls:
        - Master volume slider (0-100)
        - Master effects chain display (read-only list)
        - Output format selector (wav_16, wav_24, flac_24)
        - HQ mode toggle checkbox
        - Render to file button

        All actions dispatch through the Bridge.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="Master Channel")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # ---- Master volume slider ----
            vol_sizer = wx.BoxSizer(wx.HORIZONTAL)
            vol_label = wx.StaticText(self, label="Master Volume:")
            vol_sizer.Add(vol_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._volume_slider = wx.Slider(
                self, value=80, minValue=0, maxValue=100,
                style=wx.SL_HORIZONTAL | wx.SL_LABELS,
                name="Master Volume",
            )
            self._volume_slider.Bind(wx.EVT_SLIDER, self._on_volume)
            vol_sizer.Add(self._volume_slider, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)
            sizer.Add(vol_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Master effects chain display ----
            chain_label = wx.StaticText(self, label="Master Effects Chain:")
            sizer.Add(chain_label, 0, wx.LEFT | wx.TOP, 8)
            self._chain_list = wx.ListBox(
                self, choices=[], name="Master Effects Chain",
            )
            sizer.Add(self._chain_list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            refresh_btn = wx.Button(
                self, label="Refresh Chain", name="Refresh Effects Chain",
            )
            refresh_btn.Bind(wx.EVT_BUTTON, self._on_refresh_chain)
            sizer.Add(refresh_btn, 0, wx.LEFT | wx.TOP, 8)

            # ---- Output format selector ----
            fmt_sizer = wx.BoxSizer(wx.HORIZONTAL)
            fmt_label = wx.StaticText(self, label="Output Format:")
            fmt_sizer.Add(fmt_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._format_choice = wx.Choice(
                self, choices=OUTPUT_FORMATS, name="Output Format",
            )
            self._format_choice.SetSelection(0)
            fmt_sizer.Add(self._format_choice, 0, wx.LEFT | wx.RIGHT, 4)
            sizer.Add(fmt_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- HQ mode toggle ----
            self._hq_check = wx.CheckBox(
                self, label="HQ Mode", name="HQ Mode Toggle",
            )
            self._hq_check.Bind(wx.EVT_CHECKBOX, self._on_hq_toggle)
            sizer.Add(self._hq_check, 0, wx.LEFT | wx.TOP, 8)

            # ---- Render to file button ----
            sizer.AddSpacer(12)
            render_sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._render_btn = wx.Button(
                self, label="Render to File", name="Render to File",
            )
            self._render_btn.Bind(wx.EVT_BUTTON, self._on_render)
            render_sizer.Add(self._render_btn, 0, wx.ALL, 4)

            sizer.Add(render_sizer, 0, wx.LEFT, 4)

            # ---- Status ----
            self._status = wx.StaticText(self, label="", name="Master Channel Status")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        # ---- Helpers ----

        def _set_status(self, text: str) -> None:
            display = text[:160] if len(text) > 160 else text
            self._status.SetLabel(display)
            self._status.Wrap(self.GetSize().width - 16)

        # ---- Event handlers ----

        def _on_volume(self, event: wx.CommandEvent) -> None:
            """Adjust master volume."""
            vol = self._volume_slider.GetValue()
            result = self.bridge.execute_command('/mgain', [str(vol)])
            self._set_status(result)

        def _on_refresh_chain(self, event: wx.CommandEvent) -> None:
            """Refresh the master effects chain display."""
            result = self.bridge.execute_command('/mfxlist', [])
            self._chain_list.Clear()
            for line in result.splitlines():
                line = line.strip()
                if line:
                    self._chain_list.Append(line)
            self._set_status("Effects chain refreshed")

        def _on_hq_toggle(self, event: wx.CommandEvent) -> None:
            """Toggle HQ mode."""
            state = "on" if self._hq_check.GetValue() else "off"
            result = self.bridge.execute_command('/hq', [state])
            self._set_status(result)

        def _on_render(self, event: wx.CommandEvent) -> None:
            """Render the mix to a file."""
            fmt = OUTPUT_FORMATS[self._format_choice.GetSelection()]
            args = [fmt]
            result = self.bridge.execute_command('/render', args)
            self._set_status(result)

else:
    class MasterChannelPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
