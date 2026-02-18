"""Parameter Inspector Panel — Effects Window.

When an effect slot is selected in the chain builder, this panel shows
its full parameter set.  Unified 1-100 scale.  LFO assignment per
parameter.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Effects Window

BUILD ID: panel_param_inspector_v1.0_phase4
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

    class ParamInspectorPanel(wx.Panel):
        """Display and edit parameters for a selected effect.

        Shows all parameters with sliders on the unified 1-100 scale.
        Allows LFO assignment per parameter.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._current_effect: str = ""
            self._build_ui()

        def _build_ui(self) -> None:
            self._main_sizer = wx.BoxSizer(wx.VERTICAL)

            title = wx.StaticText(self, label="Parameter Inspector")
            title.SetFont(title.GetFont().Bold())
            self._main_sizer.Add(title, 0, wx.ALL, 8)

            # Effect name display
            self._effect_label = wx.StaticText(self, label="No effect selected")
            self._main_sizer.Add(self._effect_label, 0, wx.LEFT, 8)

            # Scrolled parameter area
            self._scroll = wx.ScrolledWindow(self, style=wx.VSCROLL)
            self._scroll.SetScrollRate(0, 10)
            self._param_sizer = wx.BoxSizer(wx.VERTICAL)
            self._scroll.SetSizer(self._param_sizer)
            self._main_sizer.Add(self._scroll, 1, wx.EXPAND | wx.ALL, 4)

            # Info button
            btn_info = wx.Button(self, label="Show Full Info")
            btn_info.Bind(wx.EVT_BUTTON, self._on_show_info)
            self._main_sizer.Add(btn_info, 0, wx.LEFT | wx.BOTTOM, 8)

            # Status
            self._status = wx.StaticText(self, label="")
            self._main_sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(self._main_sizer)

        def set_effect(self, effect_name: str) -> None:
            """Load parameters for the given effect."""
            self._current_effect = effect_name
            self._effect_label.SetLabel(f"Effect: {effect_name}")
            self._param_sizer.Clear(True)

            # Query effect info from bridge
            result = self.bridge.execute_command('/hfx', [effect_name])

            # Display as text for now — full parameter sliders in future
            info_text = wx.StaticText(self._scroll, label=result[:500])
            info_text.Wrap(self._scroll.GetSize().width - 20)
            self._param_sizer.Add(info_text, 0, wx.ALL, 8)
            self._scroll.Layout()
            self._scroll.FitInside()

        def _on_show_info(self, event: wx.CommandEvent) -> None:
            if self._current_effect:
                result = self.bridge.execute_command('/hfx', [self._current_effect])
                self._status.SetLabel(result[:200] if len(result) > 200 else result)
            else:
                self._status.SetLabel("No effect selected")

else:
    class ParamInspectorPanel:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
