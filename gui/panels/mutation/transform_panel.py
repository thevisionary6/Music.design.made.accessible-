"""Transform Panel — Mutation & Editing Window.

Wraps the /xform command.  Applies musical transformations to existing
objects (reverse, retrograde, invert, etc.) and produces a new named
object.  Non-destructive by default — the source object is never
modified unless the user explicitly overwrites.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Mutation Window

BUILD ID: panel_transform_v1.0_phase5
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

# Available transform types (matches xform engine)
TRANSFORM_TYPES = [
    'reverse',
    'retrograde',
    'invert',
    'stutter',
    'stretch',
    'compress',
    'augment',
    'diminish',
    'shuffle',
    'rotate',
]

if _WX_AVAILABLE:

    class TransformPanel(wx.Panel):
        """Panel for applying musical transforms to existing objects.

        Controls:
        - Object selector (text field for source object name)
        - Transform type dropdown (10 transform types)
        - Amount/factor spinner (0.1 - 16.0, default 1.0)
        - Output name field (auto-named if empty)
        - Apply button

        All actions dispatch through the Bridge via /xform.
        Non-destructive: always produces a new object unless
        the output name matches the source.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="Transform")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # Source object selector
            src_label = wx.StaticText(self, label="Source Object:")
            sizer.Add(src_label, 0, wx.LEFT | wx.TOP, 8)
            self._source = wx.TextCtrl(self, name="Source Object")
            self._source.SetHint("Enter object name")
            sizer.Add(self._source, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Transform type selector
            type_label = wx.StaticText(self, label="Transform Type:")
            sizer.Add(type_label, 0, wx.LEFT | wx.TOP, 8)
            self._xform_type = wx.Choice(
                self, choices=TRANSFORM_TYPES, name="Transform Type",
            )
            self._xform_type.SetSelection(0)
            sizer.Add(self._xform_type, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Amount / factor spinner
            amount_label = wx.StaticText(self, label="Amount / Factor:")
            sizer.Add(amount_label, 0, wx.LEFT | wx.TOP, 8)
            self._amount = wx.SpinCtrlDouble(
                self, min=0.1, max=16.0, initial=1.0, inc=0.1,
                name="Amount",
            )
            sizer.Add(self._amount, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Output name
            out_label = wx.StaticText(self, label="Output Name (optional):")
            sizer.Add(out_label, 0, wx.LEFT | wx.TOP, 8)
            self._output_name = wx.TextCtrl(self, name="Output Name")
            self._output_name.SetHint("Auto-named if empty")
            sizer.Add(self._output_name, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Buttons
            sizer.AddSpacer(12)
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._apply_btn = wx.Button(self, label="Apply Transform")
            self._apply_btn.Bind(wx.EVT_BUTTON, self._on_apply)
            btn_sizer.Add(self._apply_btn, 0, wx.ALL, 4)

            self._preview_btn = wx.Button(self, label="Preview")
            self._preview_btn.Bind(wx.EVT_BUTTON, self._on_preview)
            btn_sizer.Add(self._preview_btn, 0, wx.ALL, 4)

            sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            # Status
            self._status = wx.StaticText(self, label="")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        def _on_apply(self, event: wx.CommandEvent) -> None:
            """Dispatch /xform {type} with amount through the bridge."""
            source = self._source.GetValue().strip()
            if not source:
                self._status.SetLabel("Enter a source object name")
                return

            xform = TRANSFORM_TYPES[self._xform_type.GetSelection()]
            amount = self._amount.GetValue()
            output = self._output_name.GetValue().strip()

            args = [source, xform, str(amount)]
            if output:
                args.extend(['--name', output])

            result = self.bridge.execute_command('/xform', args)
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)

        def _on_preview(self, event: wx.CommandEvent) -> None:
            """Preview the transform without committing."""
            source = self._source.GetValue().strip()
            if not source:
                self._status.SetLabel("Enter a source object name")
                return

            xform = TRANSFORM_TYPES[self._xform_type.GetSelection()]
            amount = self._amount.GetValue()

            args = [source, xform, str(amount), '--preview']
            result = self.bridge.execute_command('/xform', args)
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)

else:
    class TransformPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
