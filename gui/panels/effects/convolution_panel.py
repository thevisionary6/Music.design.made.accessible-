"""Convolution Panel — Effects Window.

Dedicated panel for the convolution reverb engine.  IR preset browser,
neural enhancement controls, semantic IR transform.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Effects Window

BUILD ID: panel_convolution_v1.0_phase4
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

IR_PRESETS = [
    'conv_hall', 'conv_hall_long', 'conv_room', 'conv_plate',
    'conv_spring', 'conv_shimmer', 'conv_reverse',
]

IRENHANCE_MODES = [
    'extend', 'denoise', 'fill', 'brighten', 'darken',
    'widen', 'narrow', 'warmth', 'air',
]

IRTRANSFORM_DESCRIPTORS = [
    'cathedral', 'cave', 'telephone', 'underwater',
    'metallic', 'wooden', 'glass', 'ethereal',
    'dark', 'bright', 'tight', 'vast',
    'vintage', 'futuristic', 'organic',
]

if _WX_AVAILABLE:

    class ConvolutionPanel(wx.Panel):
        """Convolution reverb panel.

        Controls:
        - IR preset selector
        - Mix (dry/wet) slider
        - Neural enhance mode selector
        - Semantic IR transform descriptor
        - Apply/Preview buttons
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            title = wx.StaticText(self, label="Convolution Reverb")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # IR preset
            preset_label = wx.StaticText(self, label="IR Preset:")
            sizer.Add(preset_label, 0, wx.LEFT | wx.TOP, 8)
            self._preset = wx.Choice(self, choices=IR_PRESETS, name="IR Preset")
            self._preset.SetSelection(0)
            sizer.Add(self._preset, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Apply preset button
            btn_apply = wx.Button(self, label="Apply Convolution")
            btn_apply.Bind(wx.EVT_BUTTON, self._on_apply_preset)
            sizer.Add(btn_apply, 0, wx.LEFT | wx.TOP, 8)

            sizer.AddSpacer(16)

            # Neural enhance section
            enhance_label = wx.StaticText(self, label="Neural IR Enhance:")
            enhance_label.SetFont(enhance_label.GetFont().Bold())
            sizer.Add(enhance_label, 0, wx.LEFT | wx.TOP, 8)

            self._enhance_mode = wx.Choice(self, choices=IRENHANCE_MODES,
                                            name="Enhance Mode")
            self._enhance_mode.SetSelection(0)
            sizer.Add(self._enhance_mode, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            btn_enhance = wx.Button(self, label="Enhance IR")
            btn_enhance.Bind(wx.EVT_BUTTON, self._on_enhance)
            sizer.Add(btn_enhance, 0, wx.LEFT | wx.TOP, 8)

            sizer.AddSpacer(16)

            # Semantic transform section
            transform_label = wx.StaticText(self, label="Semantic IR Transform:")
            transform_label.SetFont(transform_label.GetFont().Bold())
            sizer.Add(transform_label, 0, wx.LEFT | wx.TOP, 8)

            self._transform = wx.Choice(self, choices=IRTRANSFORM_DESCRIPTORS,
                                         name="Transform Descriptor")
            self._transform.SetSelection(0)
            sizer.Add(self._transform, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            btn_transform = wx.Button(self, label="Transform IR")
            btn_transform.Bind(wx.EVT_BUTTON, self._on_transform)
            sizer.Add(btn_transform, 0, wx.LEFT | wx.TOP, 8)

            # Status
            self._status = wx.StaticText(self, label="")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        def _on_apply_preset(self, event: wx.CommandEvent) -> None:
            preset = IR_PRESETS[self._preset.GetSelection()]
            result = self.bridge.execute_command('/fx', [preset])
            self._status.SetLabel(result[:120] if len(result) > 120 else result)

        def _on_enhance(self, event: wx.CommandEvent) -> None:
            mode = IRENHANCE_MODES[self._enhance_mode.GetSelection()]
            result = self.bridge.execute_command('/irenhance', [mode])
            self._status.SetLabel(result[:120] if len(result) > 120 else result)

        def _on_transform(self, event: wx.CommandEvent) -> None:
            desc = IRTRANSFORM_DESCRIPTORS[self._transform.GetSelection()]
            result = self.bridge.execute_command('/irtransform', [desc])
            self._status.SetLabel(result[:120] if len(result) > 120 else result)

else:
    class ConvolutionPanel:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
