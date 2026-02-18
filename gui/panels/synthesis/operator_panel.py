"""Operator Panel — Synthesis Window.

FM synthesis operator grid.  Wraps the /op command.  Provides controls
for selecting and configuring up to 8 operators with waveform type,
ratio, level, feedback, and carrier/modulator routing.

Each operator maps to one slot in the FM matrix.  Changes are applied
through the Bridge so the engine stays in sync.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Synthesis Window

BUILD ID: panel_operator_v1.0_phase5
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

# 22 waveform types supported by the engine
WAVEFORM_TYPES = [
    'sine', 'square', 'triangle', 'saw', 'pulse', 'noise',
    'supersaw', 'additive', 'formant', 'harmonic', 'wavetable',
    'compound', 'pwm', 'sync', 'ring', 'fold', 'clip',
    'half_rect', 'full_rect', 'staircase', 'sample_hold', 'resonant',
]

# Operator indices 0-7
OPERATOR_COUNT = 8

if _WX_AVAILABLE:

    class OperatorPanel(wx.Panel):
        """FM synthesis operator configuration panel.

        Controls:
        - Operator index selector (0-7)
        - Waveform type dropdown (22 types)
        - Ratio spinner (0.5 - 32.0)
        - Level slider (0-100)
        - Feedback slider (0-100)
        - Carrier/Modulator toggle
        - Apply button

        All actions dispatch through the Bridge.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="FM Operator Grid")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # --- Operator index selector ---
            op_label = wx.StaticText(self, label="Operator Index:")
            sizer.Add(op_label, 0, wx.LEFT | wx.TOP, 8)
            self._op_index = wx.SpinCtrl(
                self, min=0, max=OPERATOR_COUNT - 1, initial=0,
                name="Operator Index",
            )
            sizer.Add(self._op_index, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # --- Waveform type ---
            wave_label = wx.StaticText(self, label="Waveform Type:")
            sizer.Add(wave_label, 0, wx.LEFT | wx.TOP, 8)
            self._waveform = wx.Choice(
                self, choices=WAVEFORM_TYPES, name="Waveform Type",
            )
            self._waveform.SetSelection(0)  # default: sine
            sizer.Add(self._waveform, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # --- Ratio spinner ---
            ratio_label = wx.StaticText(self, label="Frequency Ratio:")
            sizer.Add(ratio_label, 0, wx.LEFT | wx.TOP, 8)
            self._ratio = wx.SpinCtrlDouble(
                self, min=0.5, max=32.0, initial=1.0, inc=0.5,
                name="Frequency Ratio",
            )
            sizer.Add(self._ratio, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # --- Level slider ---
            level_label = wx.StaticText(self, label="Level:")
            sizer.Add(level_label, 0, wx.LEFT | wx.TOP, 8)
            self._level = wx.Slider(
                self, value=80, minValue=0, maxValue=100,
                style=wx.SL_HORIZONTAL | wx.SL_VALUE_LABEL,
                name="Operator Level",
            )
            sizer.Add(self._level, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # --- Feedback slider ---
            fb_label = wx.StaticText(self, label="Feedback:")
            sizer.Add(fb_label, 0, wx.LEFT | wx.TOP, 8)
            self._feedback = wx.Slider(
                self, value=0, minValue=0, maxValue=100,
                style=wx.SL_HORIZONTAL | wx.SL_VALUE_LABEL,
                name="Operator Feedback",
            )
            sizer.Add(self._feedback, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # --- Carrier / Modulator toggle ---
            self._carrier_toggle = wx.CheckBox(
                self, label="Carrier (unchecked = Modulator)",
                name="Carrier Modulator Toggle",
            )
            self._carrier_toggle.SetValue(True)
            sizer.Add(self._carrier_toggle, 0, wx.ALL, 8)

            # --- Apply button ---
            sizer.AddSpacer(8)
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
            self._apply_btn = wx.Button(self, label="Apply Operator")
            self._apply_btn.Bind(wx.EVT_BUTTON, self._on_apply)
            btn_sizer.Add(self._apply_btn, 0, wx.ALL, 4)

            self._reset_btn = wx.Button(self, label="Reset Operator")
            self._reset_btn.Bind(wx.EVT_BUTTON, self._on_reset)
            btn_sizer.Add(self._reset_btn, 0, wx.ALL, 4)

            sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            # Status
            self._status = wx.StaticText(self, label="")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        def _on_apply(self, event: wx.CommandEvent) -> None:
            """Apply current operator settings via bridge."""
            op_idx = self._op_index.GetValue()
            waveform = WAVEFORM_TYPES[self._waveform.GetSelection()]
            ratio = str(self._ratio.GetValue())
            level = str(self._level.GetValue())
            feedback = str(self._feedback.GetValue())
            role = 'carrier' if self._carrier_toggle.GetValue() else 'modulator'

            args = [
                str(op_idx),
                '--wave', waveform,
                '--ratio', ratio,
                '--level', level,
                '--feedback', feedback,
                '--role', role,
            ]
            result = self.bridge.execute_command('/op', args)
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)

        def _on_reset(self, event: wx.CommandEvent) -> None:
            """Reset controls to defaults."""
            self._op_index.SetValue(0)
            self._waveform.SetSelection(0)
            self._ratio.SetValue(1.0)
            self._level.SetValue(80)
            self._feedback.SetValue(0)
            self._carrier_toggle.SetValue(True)
            self._status.SetLabel("Operator reset to defaults")

else:
    class OperatorPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
