"""Crossfader Panel — Mixing Window.

Crossfader slider for blending between decks A and B.  Fully
keyboard operable for accessibility.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Mixing Window

BUILD ID: panel_crossfader_v1.0_phase5
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

    class CrossfaderPanel(wx.Panel):
        """Crossfader between Deck A and Deck B.

        Controls:
        - Crossfader slider (0 = full A, 50 = center, 100 = full B)
        - Current position display
        - Quick buttons: Full A, Center, Full B

        Keyboard operable: arrow keys adjust the slider by 1 unit,
        Page Up/Down by 10 units.

        All actions dispatch through the Bridge.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="Crossfader")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # ---- Labels for deck sides ----
            label_sizer = wx.BoxSizer(wx.HORIZONTAL)
            a_label = wx.StaticText(self, label="Deck A")
            label_sizer.Add(a_label, 0, wx.LEFT, 8)
            label_sizer.AddStretchSpacer()
            center_label = wx.StaticText(self, label="Center")
            label_sizer.Add(center_label, 0)
            label_sizer.AddStretchSpacer()
            b_label = wx.StaticText(self, label="Deck B")
            label_sizer.Add(b_label, 0, wx.RIGHT, 8)
            sizer.Add(label_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Crossfader slider ----
            self._xfade_slider = wx.Slider(
                self, value=50, minValue=0, maxValue=100,
                style=wx.SL_HORIZONTAL | wx.SL_LABELS,
                name="Crossfader",
            )
            self._xfade_slider.Bind(wx.EVT_SLIDER, self._on_xfade)
            sizer.Add(self._xfade_slider, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # ---- Position display ----
            pos_sizer = wx.BoxSizer(wx.HORIZONTAL)
            pos_label = wx.StaticText(self, label="Position:")
            pos_sizer.Add(pos_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._pos_display = wx.StaticText(
                self, label="50 (Center)", name="Crossfader Position",
            )
            pos_sizer.Add(self._pos_display, 0, wx.LEFT, 4)
            sizer.Add(pos_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Quick buttons ----
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._full_a_btn = wx.Button(self, label="Full A", name="Full A")
            self._full_a_btn.Bind(wx.EVT_BUTTON, self._on_full_a)
            btn_sizer.Add(self._full_a_btn, 0, wx.ALL, 4)

            self._center_btn = wx.Button(self, label="Center", name="Center Crossfader")
            self._center_btn.Bind(wx.EVT_BUTTON, self._on_center)
            btn_sizer.Add(self._center_btn, 0, wx.ALL, 4)

            self._full_b_btn = wx.Button(self, label="Full B", name="Full B")
            self._full_b_btn.Bind(wx.EVT_BUTTON, self._on_full_b)
            btn_sizer.Add(self._full_b_btn, 0, wx.ALL, 4)

            sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            # ---- Status ----
            self._status = wx.StaticText(self, label="", name="Crossfader Status")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        # ---- Helpers ----

        def _set_status(self, text: str) -> None:
            display = text[:160] if len(text) > 160 else text
            self._status.SetLabel(display)
            self._status.Wrap(self.GetSize().width - 16)

        def _update_position_label(self, value: int) -> None:
            """Update the position display text."""
            if value == 0:
                desc = "Full A"
            elif value == 50:
                desc = "Center"
            elif value == 100:
                desc = "Full B"
            elif value < 50:
                desc = f"A bias ({value})"
            else:
                desc = f"B bias ({value})"
            self._pos_display.SetLabel(f"{value} ({desc})")

        def _dispatch_xfade(self, value: int) -> None:
            """Send crossfader value through the bridge."""
            self._xfade_slider.SetValue(value)
            self._update_position_label(value)
            result = self.bridge.execute_command('/xfade', [str(value)])
            self._set_status(result)

        # ---- Event handlers ----

        def _on_xfade(self, event: wx.CommandEvent) -> None:
            """Handle crossfader slider change."""
            value = self._xfade_slider.GetValue()
            self._update_position_label(value)
            result = self.bridge.execute_command('/xfade', [str(value)])
            self._set_status(result)

        def _on_full_a(self, event: wx.CommandEvent) -> None:
            """Snap crossfader to full Deck A."""
            self._dispatch_xfade(0)

        def _on_center(self, event: wx.CommandEvent) -> None:
            """Snap crossfader to center."""
            self._dispatch_xfade(50)

        def _on_full_b(self, event: wx.CommandEvent) -> None:
            """Snap crossfader to full Deck B."""
            self._dispatch_xfade(100)

else:
    class CrossfaderPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
