"""Physical Modeling Panel — Synthesis Window.

Physical modeling parameters for waveguide synthesis.  Active when a
physical modeling mode is selected.  Wraps the /waveguide command.
Controls material type, excitation method, and damping characteristics
for simulating acoustic instrument behaviour.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Synthesis Window

BUILD ID: panel_physical_modeling_v1.0_phase5
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

# Material types for waveguide simulation
MATERIAL_TYPES = [
    'steel', 'nylon', 'gut', 'brass', 'wood',
    'glass', 'membrane', 'reed', 'air_column', 'bar',
]

# Excitation types
EXCITATION_TYPES = [
    'pluck', 'bow', 'strike', 'blow', 'scrape',
    'tap', 'noise_burst', 'impulse',
]

# Waveguide model presets
MODEL_PRESETS = [
    'string', 'tube', 'plate', 'bar', 'membrane',
    'flute', 'clarinet', 'marimba', 'kalimba', 'custom',
]

if _WX_AVAILABLE:

    class PhysicalModelingPanel(wx.Panel):
        """Physical modeling / waveguide synthesis panel.

        Controls:
        - Model preset selector
        - Material type dropdown
        - Excitation type dropdown
        - Damping slider (0-100)
        - Brightness slider (0-100)
        - Tension slider (0-100)
        - Inharmonicity slider (0-100)
        - Apply / Reset buttons

        All actions dispatch through the Bridge.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="Physical Modeling")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            desc = wx.StaticText(
                self,
                label="Waveguide synthesis parameters for acoustic simulation.",
            )
            sizer.Add(desc, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

            # --- Model preset ---
            model_label = wx.StaticText(self, label="Model Preset:")
            sizer.Add(model_label, 0, wx.LEFT | wx.TOP, 8)
            self._model = wx.Choice(
                self, choices=MODEL_PRESETS, name="Model Preset",
            )
            self._model.SetSelection(0)  # default: string
            sizer.Add(self._model, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # --- Material type ---
            mat_label = wx.StaticText(self, label="Material:")
            sizer.Add(mat_label, 0, wx.LEFT | wx.TOP, 8)
            self._material = wx.Choice(
                self, choices=MATERIAL_TYPES, name="Material Type",
            )
            self._material.SetSelection(0)  # default: steel
            sizer.Add(self._material, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # --- Excitation type ---
            exc_label = wx.StaticText(self, label="Excitation:")
            sizer.Add(exc_label, 0, wx.LEFT | wx.TOP, 8)
            self._excitation = wx.Choice(
                self, choices=EXCITATION_TYPES, name="Excitation Type",
            )
            self._excitation.SetSelection(0)  # default: pluck
            sizer.Add(self._excitation, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # --- Parameter sliders ---
            self._param_sliders: dict[str, wx.Slider] = {}
            self._param_displays: dict[str, wx.StaticText] = {}

            slider_defs = [
                ('Damping',       50),
                ('Brightness',    60),
                ('Tension',       70),
                ('Inharmonicity', 10),
            ]

            for param_name, default in slider_defs:
                row_sizer = wx.BoxSizer(wx.HORIZONTAL)

                plabel = wx.StaticText(
                    self, label=f"{param_name}:", size=(100, -1),
                )
                row_sizer.Add(plabel, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)

                slider = wx.Slider(
                    self, value=default, minValue=0, maxValue=100,
                    style=wx.SL_HORIZONTAL,
                    name=f"{param_name} Slider",
                )
                self._param_sliders[param_name] = slider
                row_sizer.Add(slider, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)

                val_display = wx.StaticText(
                    self, label=str(default), size=(35, -1),
                    style=wx.ALIGN_RIGHT,
                    name=f"{param_name} Value",
                )
                self._param_displays[param_name] = val_display
                row_sizer.Add(val_display, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)

                slider.Bind(
                    wx.EVT_SLIDER,
                    lambda evt, pn=param_name: self._on_param_change(evt, pn),
                )

                sizer.Add(row_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # --- Buttons ---
            sizer.AddSpacer(12)
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._apply_btn = wx.Button(self, label="Apply Waveguide")
            self._apply_btn.Bind(wx.EVT_BUTTON, self._on_apply)
            btn_sizer.Add(self._apply_btn, 0, wx.ALL, 4)

            self._reset_btn = wx.Button(self, label="Reset")
            self._reset_btn.Bind(wx.EVT_BUTTON, self._on_reset)
            btn_sizer.Add(self._reset_btn, 0, wx.ALL, 4)

            sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            # Status
            self._status = wx.StaticText(self, label="")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        def _on_param_change(self, event: wx.CommandEvent, param_name: str) -> None:
            """Update value display on slider change."""
            value = self._param_sliders[param_name].GetValue()
            self._param_displays[param_name].SetLabel(str(value))
            self._param_sliders[param_name].SetName(f"{param_name} Slider: {value}")

        def _on_apply(self, event: wx.CommandEvent) -> None:
            """Apply waveguide settings via bridge."""
            model = MODEL_PRESETS[self._model.GetSelection()]
            material = MATERIAL_TYPES[self._material.GetSelection()]
            excitation = EXCITATION_TYPES[self._excitation.GetSelection()]
            damping = str(self._param_sliders['Damping'].GetValue())
            brightness = str(self._param_sliders['Brightness'].GetValue())
            tension = str(self._param_sliders['Tension'].GetValue())
            inharm = str(self._param_sliders['Inharmonicity'].GetValue())

            args = [
                '--model', model,
                '--material', material,
                '--excitation', excitation,
                '--damping', damping,
                '--brightness', brightness,
                '--tension', tension,
                '--inharmonicity', inharm,
            ]
            result = self.bridge.execute_command('/waveguide', args)
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)

        def _on_reset(self, event: wx.CommandEvent) -> None:
            """Reset all controls to defaults."""
            self._model.SetSelection(0)
            self._material.SetSelection(0)
            self._excitation.SetSelection(0)

            defaults = {'Damping': 50, 'Brightness': 60, 'Tension': 70, 'Inharmonicity': 10}
            for param_name, default in defaults.items():
                self._param_sliders[param_name].SetValue(default)
                self._param_displays[param_name].SetLabel(str(default))
                self._param_sliders[param_name].SetName(f"{param_name} Slider: {default}")

            self._status.SetLabel("Physical modeling reset to defaults")

else:
    class PhysicalModelingPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
