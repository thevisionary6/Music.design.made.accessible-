"""Modulation Panel — Synthesis Window.

LFO modulation routing controls.  Wraps the /lfo command.  Allows
assigning an LFO to a target parameter with configurable rate, depth,
and waveform shape.  Multiple LFO assignments can be stacked by
applying different targets.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Synthesis Window

BUILD ID: panel_modulation_v1.0_phase5
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

# LFO waveform shapes
LFO_SHAPES = [
    'sine', 'square', 'triangle', 'saw', 'random', 'sample_hold',
]

# Modulation target parameters
LFO_TARGETS = [
    'pitch', 'amplitude', 'filter_cutoff', 'filter_resonance',
    'pan', 'fm_depth', 'pwm_width', 'wavetable_position',
    'feedback', 'ratio', 'level', 'delay_time', 'reverb_mix',
]

if _WX_AVAILABLE:

    class ModulationPanel(wx.Panel):
        """LFO modulation routing panel.

        Controls:
        - LFO shape dropdown
        - Rate spinner (0.01 - 50.0 Hz)
        - Depth slider (0-100)
        - Target parameter dropdown
        - Apply / Clear buttons

        All actions dispatch through the Bridge.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="LFO Modulation")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            desc = wx.StaticText(
                self,
                label="Assign low-frequency oscillators to modulate synthesis parameters.",
            )
            sizer.Add(desc, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

            # --- LFO Shape ---
            shape_label = wx.StaticText(self, label="LFO Shape:")
            sizer.Add(shape_label, 0, wx.LEFT | wx.TOP, 8)
            self._shape = wx.Choice(
                self, choices=LFO_SHAPES, name="LFO Shape",
            )
            self._shape.SetSelection(0)  # default: sine
            sizer.Add(self._shape, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # --- Rate spinner ---
            rate_label = wx.StaticText(self, label="Rate (Hz):")
            sizer.Add(rate_label, 0, wx.LEFT | wx.TOP, 8)
            self._rate = wx.SpinCtrlDouble(
                self, min=0.01, max=50.0, initial=1.0, inc=0.1,
                name="LFO Rate",
            )
            sizer.Add(self._rate, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # --- Depth slider ---
            depth_label = wx.StaticText(self, label="Depth:")
            sizer.Add(depth_label, 0, wx.LEFT | wx.TOP, 8)

            depth_row = wx.BoxSizer(wx.HORIZONTAL)
            self._depth = wx.Slider(
                self, value=50, minValue=0, maxValue=100,
                style=wx.SL_HORIZONTAL,
                name="LFO Depth",
            )
            self._depth_display = wx.StaticText(
                self, label="50", size=(35, -1),
                style=wx.ALIGN_RIGHT,
                name="LFO Depth Value",
            )
            self._depth.Bind(wx.EVT_SLIDER, self._on_depth_change)
            depth_row.Add(self._depth, 1, wx.EXPAND | wx.RIGHT, 4)
            depth_row.Add(self._depth_display, 0, wx.ALIGN_CENTER_VERTICAL)
            sizer.Add(depth_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # --- Target parameter ---
            target_label = wx.StaticText(self, label="Target Parameter:")
            sizer.Add(target_label, 0, wx.LEFT | wx.TOP, 8)
            self._target = wx.Choice(
                self, choices=LFO_TARGETS, name="LFO Target",
            )
            self._target.SetSelection(0)  # default: pitch
            sizer.Add(self._target, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # --- Buttons ---
            sizer.AddSpacer(12)
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._apply_btn = wx.Button(self, label="Apply LFO")
            self._apply_btn.Bind(wx.EVT_BUTTON, self._on_apply)
            btn_sizer.Add(self._apply_btn, 0, wx.ALL, 4)

            self._clear_btn = wx.Button(self, label="Clear LFO")
            self._clear_btn.Bind(wx.EVT_BUTTON, self._on_clear)
            btn_sizer.Add(self._clear_btn, 0, wx.ALL, 4)

            self._reset_btn = wx.Button(self, label="Reset")
            self._reset_btn.Bind(wx.EVT_BUTTON, self._on_reset)
            btn_sizer.Add(self._reset_btn, 0, wx.ALL, 4)

            sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            # Status
            self._status = wx.StaticText(self, label="")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        def _on_depth_change(self, event: wx.CommandEvent) -> None:
            """Update depth value display."""
            value = self._depth.GetValue()
            self._depth_display.SetLabel(str(value))
            self._depth.SetName(f"LFO Depth: {value}")

        def _on_apply(self, event: wx.CommandEvent) -> None:
            """Apply LFO assignment via bridge."""
            shape = LFO_SHAPES[self._shape.GetSelection()]
            rate = str(self._rate.GetValue())
            depth = str(self._depth.GetValue())
            target = LFO_TARGETS[self._target.GetSelection()]

            args = [
                '--shape', shape,
                '--rate', rate,
                '--depth', depth,
                '--target', target,
            ]
            result = self.bridge.execute_command('/lfo', args)
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)

        def _on_clear(self, event: wx.CommandEvent) -> None:
            """Clear LFO from current target via bridge."""
            target = LFO_TARGETS[self._target.GetSelection()]
            result = self.bridge.execute_command('/lfo', ['--clear', '--target', target])
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)

        def _on_reset(self, event: wx.CommandEvent) -> None:
            """Reset all controls to defaults."""
            self._shape.SetSelection(0)
            self._rate.SetValue(1.0)
            self._depth.SetValue(50)
            self._depth_display.SetLabel("50")
            self._depth.SetName("LFO Depth: 50")
            self._target.SetSelection(0)
            self._status.SetLabel("LFO controls reset to defaults")

else:
    class ModulationPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
