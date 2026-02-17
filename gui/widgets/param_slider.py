"""Accessible 1-100 Parameter Slider Widget.

A labeled slider control that maps to MDMA's unified 1-100 parameter
scaling system.  Provides keyboard increment/decrement, screen reader
value announcements, and optional value display.

Consistent with the parameter scaling defined in mdma_rebuild/dsp/scaling.py.

BUILD ID: widget_param_slider_v1.0
"""

from __future__ import annotations

from typing import Any, Callable, Optional

try:
    import wx
    _WX_AVAILABLE = True
except ImportError:
    _WX_AVAILABLE = False


if _WX_AVAILABLE:
    from .labeled_control import LabeledControl

    class ParamSlider(LabeledControl):
        """Accessible slider for 1-100 parameter values.

        Displays a slider with a value readout.  Keyboard controls:
        - Left/Right arrows: adjust by 1
        - Page Up/Down: adjust by 10
        - Home/End: jump to min/max

        Parameters:
            parent: Parent wx window.
            label: Parameter name shown as label.
            description: Screen reader description.
            min_val: Minimum value (default 0).
            max_val: Maximum value (default 100).
            default: Initial value (default 50).
            on_change: Optional callback(value: int) on slider change.
        """

        def __init__(
            self,
            parent: wx.Window,
            label: str = "Parameter",
            description: str = "",
            min_val: int = 0,
            max_val: int = 100,
            default: int = 50,
            on_change: Optional[Callable[[int], None]] = None,
            **kwargs: Any,
        ) -> None:
            self._min_val = min_val
            self._max_val = max_val
            self._default = default
            self._on_change = on_change
            super().__init__(parent, label=label, description=description, **kwargs)

        def _create_control(self) -> wx.Window:
            panel = wx.Panel(self)
            sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._slider = wx.Slider(
                panel,
                value=self._default,
                minValue=self._min_val,
                maxValue=self._max_val,
                style=wx.SL_HORIZONTAL,
                name=self._label_text,
            )

            self._value_display = wx.StaticText(
                panel,
                label=str(self._default),
                size=(40, -1),
                style=wx.ALIGN_RIGHT,
            )

            sizer.Add(self._slider, 1, wx.EXPAND | wx.RIGHT, 5)
            sizer.Add(self._value_display, 0, wx.ALIGN_CENTER_VERTICAL)
            panel.SetSizer(sizer)

            # Bind events
            self._slider.Bind(wx.EVT_SLIDER, self._on_slider_change)

            return panel

        def _on_slider_change(self, event: wx.CommandEvent) -> None:
            value = self._slider.GetValue()
            self._value_display.SetLabel(str(value))

            # Update accessible name with current value
            self._slider.SetName(f"{self._label_text}: {value}")

            if self._on_change:
                self._on_change(value)

        def get_value(self) -> int:
            return self._slider.GetValue()

        def set_value(self, value: int) -> None:
            value = max(self._min_val, min(self._max_val, int(value)))
            self._slider.SetValue(value)
            self._value_display.SetLabel(str(value))

        def reset(self) -> None:
            """Reset to default value."""
            self.set_value(self._default)

else:

    class ParamSlider:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("wxPython required")
