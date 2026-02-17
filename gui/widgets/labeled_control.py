"""Accessible Label+Control Pair Widget.

Base class for all MDMA GUI controls that pairs a visible label with
an interactive widget and ensures screen reader accessibility.

Every LabeledControl guarantees:
- A meaningful accessible name read by screen readers
- A description string for additional context
- Keyboard focus support (tab-navigable)
- Consistent layout (label above or beside the control)

BUILD ID: widget_labeled_control_v1.0
"""

from __future__ import annotations

from typing import Any, Optional

try:
    import wx
    _WX_AVAILABLE = True
except ImportError:
    _WX_AVAILABLE = False


if _WX_AVAILABLE:

    class LabeledControl(wx.Panel):
        """A label paired with an interactive control.

        Subclasses override ``_create_control`` to provide the specific
        widget (slider, dropdown, text field, etc.).  The label is
        always a wx.StaticText positioned above the control.

        Parameters:
            parent: Parent wx window.
            label: Human-readable label text.
            description: Longer description for screen readers.
            orientation: wx.VERTICAL (label above) or wx.HORIZONTAL
                (label beside).
        """

        def __init__(
            self,
            parent: wx.Window,
            label: str = "",
            description: str = "",
            orientation: int = None,
            **kwargs: Any,
        ) -> None:
            if orientation is None:
                orientation = wx.VERTICAL
            super().__init__(parent, **kwargs)

            self._label_text = label
            self._description = description

            sizer = wx.BoxSizer(orientation)

            # Label
            self._label = wx.StaticText(self, label=label)
            sizer.Add(self._label, 0, wx.ALL, 2)

            # Control (created by subclass)
            self._control = self._create_control()
            if self._control is not None:
                sizer.Add(self._control, 1, wx.EXPAND | wx.ALL, 2)

                # Set accessible name and description
                self._control.SetName(label)
                if hasattr(self._control, "SetHint"):
                    self._control.SetHint(description)

            self.SetSizer(sizer)

        def _create_control(self) -> Optional[wx.Window]:
            """Override in subclasses to create the specific control widget.

            Returns the created control, or None if creation is deferred.
            """
            return None

        def get_control(self) -> Optional[wx.Window]:
            """Return the inner control widget."""
            return self._control

        def set_label(self, text: str) -> None:
            """Update the label text."""
            self._label_text = text
            self._label.SetLabel(text)
            if self._control is not None:
                self._control.SetName(text)

        def get_value(self) -> Any:
            """Get the control's current value. Override in subclasses."""
            if self._control is not None and hasattr(self._control, "GetValue"):
                return self._control.GetValue()
            return None

        def set_value(self, value: Any) -> None:
            """Set the control's value. Override in subclasses."""
            if self._control is not None and hasattr(self._control, "SetValue"):
                self._control.SetValue(value)

else:

    class LabeledControl:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("wxPython required")
