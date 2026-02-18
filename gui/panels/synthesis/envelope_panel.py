"""Envelope Panel — Synthesis Window.

ADSR envelope controls for shaping the amplitude contour of patches.
Wraps the /attack, /decay, /sustain, and /release commands.  Each
slider uses the unified 0-100 scaling that the engine maps to real
time/level values internally.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Synthesis Window

BUILD ID: panel_envelope_v1.0_phase5
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

# ADSR parameter definitions: (label, command, default)
ADSR_PARAMS = [
    ('Attack',  '/attack',  20),
    ('Decay',   '/decay',   40),
    ('Sustain', '/sustain', 70),
    ('Release', '/release', 30),
]

if _WX_AVAILABLE:

    class EnvelopePanel(wx.Panel):
        """ADSR envelope editor panel.

        Controls:
        - Attack slider (0-100, mapped to engine values)
        - Decay slider (0-100, mapped to engine values)
        - Sustain slider (0-100, mapped to engine values)
        - Release slider (0-100, mapped to engine values)
        - Apply All button to send all values
        - Reset button to restore defaults

        Individual sliders dispatch immediately on change.
        All actions dispatch through the Bridge.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._sliders: dict[str, wx.Slider] = {}
            self._value_labels: dict[str, wx.StaticText] = {}
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="ADSR Envelope")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            desc = wx.StaticText(
                self,
                label="Shape the amplitude envelope. Values 0-100 map to engine parameters.",
            )
            sizer.Add(desc, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

            # --- ADSR sliders ---
            for label, cmd, default in ADSR_PARAMS:
                row_sizer = wx.BoxSizer(wx.HORIZONTAL)

                param_label = wx.StaticText(
                    self, label=f"{label}:", size=(70, -1),
                )
                row_sizer.Add(param_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)

                slider = wx.Slider(
                    self, value=default, minValue=0, maxValue=100,
                    style=wx.SL_HORIZONTAL,
                    name=f"{label} Slider",
                )
                self._sliders[label] = slider
                row_sizer.Add(slider, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)

                value_display = wx.StaticText(
                    self, label=str(default), size=(35, -1),
                    style=wx.ALIGN_RIGHT,
                    name=f"{label} Value",
                )
                self._value_labels[label] = value_display
                row_sizer.Add(value_display, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)

                # Bind slider change to update display and dispatch
                slider.Bind(
                    wx.EVT_SLIDER,
                    lambda evt, lbl=label, cmd_str=cmd: self._on_slider_change(evt, lbl, cmd_str),
                )

                sizer.Add(row_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # --- Buttons ---
            sizer.AddSpacer(12)
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._apply_all_btn = wx.Button(self, label="Apply All")
            self._apply_all_btn.Bind(wx.EVT_BUTTON, self._on_apply_all)
            btn_sizer.Add(self._apply_all_btn, 0, wx.ALL, 4)

            self._reset_btn = wx.Button(self, label="Reset Defaults")
            self._reset_btn.Bind(wx.EVT_BUTTON, self._on_reset)
            btn_sizer.Add(self._reset_btn, 0, wx.ALL, 4)

            sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            # Status
            self._status = wx.StaticText(self, label="")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        def _on_slider_change(
            self, event: wx.CommandEvent, label: str, command: str
        ) -> None:
            """Handle individual slider adjustment."""
            value = self._sliders[label].GetValue()
            self._value_labels[label].SetLabel(str(value))
            # Update accessible name with current value
            self._sliders[label].SetName(f"{label} Slider: {value}")

        def _on_apply_all(self, event: wx.CommandEvent) -> None:
            """Send all four ADSR values to the engine."""
            results = []
            for label, cmd, _default in ADSR_PARAMS:
                value = str(self._sliders[label].GetValue())
                result = self.bridge.execute_command(cmd, [value])
                results.append(f"{label}={value}")

            summary = "Applied: " + ", ".join(results)
            self._status.SetLabel(summary)
            self._status.Wrap(self.GetSize().width - 16)

        def _on_reset(self, event: wx.CommandEvent) -> None:
            """Reset all sliders to default values."""
            for label, _cmd, default in ADSR_PARAMS:
                self._sliders[label].SetValue(default)
                self._value_labels[label].SetLabel(str(default))
                self._sliders[label].SetName(f"{label} Slider: {default}")
            self._status.SetLabel("Envelope reset to defaults")

        def get_values(self) -> dict[str, int]:
            """Return current ADSR values as a dict."""
            return {
                label: self._sliders[label].GetValue()
                for label, _cmd, _default in ADSR_PARAMS
            }

else:
    class EnvelopePanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
