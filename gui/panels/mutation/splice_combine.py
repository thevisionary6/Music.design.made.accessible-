"""Splice & Combine Panel — Mutation & Editing Window.

Merges or splices two existing objects together using various
combination strategies (stitch, layer, interleave).  Produces a new
named object — the source objects are never modified.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Mutation Window

BUILD ID: panel_splice_combine_v1.0_phase5
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

# Combination operations
OPERATION_TYPES = ['stitch', 'layer', 'interleave']

OPERATION_DESCRIPTIONS = {
    'stitch': "Join B after A end-to-end",
    'layer': "Stack A and B on top of each other",
    'interleave': "Alternate steps/bars between A and B",
}

if _WX_AVAILABLE:

    class SpliceCombinePanel(wx.Panel):
        """Panel for merging or splicing two objects together.

        Controls:
        - Source A object selector (text field)
        - Source B object selector (text field)
        - Operation type dropdown (stitch, layer, interleave)
        - Operation description (updates dynamically)
        - Output name field (auto-named if empty)
        - Combine button

        All actions dispatch through the Bridge.
        Non-destructive: always produces a new object.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="Splice && Combine")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # Source A
            src_a_label = wx.StaticText(self, label="Source A:")
            sizer.Add(src_a_label, 0, wx.LEFT | wx.TOP, 8)
            self._source_a = wx.TextCtrl(self, name="Source A")
            self._source_a.SetHint("Enter first object name")
            sizer.Add(self._source_a, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Source B
            src_b_label = wx.StaticText(self, label="Source B:")
            sizer.Add(src_b_label, 0, wx.LEFT | wx.TOP, 8)
            self._source_b = wx.TextCtrl(self, name="Source B")
            self._source_b.SetHint("Enter second object name")
            sizer.Add(self._source_b, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Operation type
            op_label = wx.StaticText(self, label="Operation:")
            sizer.Add(op_label, 0, wx.LEFT | wx.TOP, 8)
            self._operation = wx.Choice(
                self, choices=OPERATION_TYPES, name="Operation Type",
            )
            self._operation.SetSelection(0)
            self._operation.Bind(wx.EVT_CHOICE, self._on_operation_change)
            sizer.Add(self._operation, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Operation description
            self._op_desc = wx.StaticText(
                self,
                label=OPERATION_DESCRIPTIONS['stitch'],
                name="Operation Description",
            )
            self._op_desc.SetForegroundColour(wx.Colour(120, 120, 120))
            sizer.Add(self._op_desc, 0, wx.LEFT | wx.TOP, 8)

            # Swap button
            swap_sizer = wx.BoxSizer(wx.HORIZONTAL)
            self._swap_btn = wx.Button(self, label="Swap A <-> B")
            self._swap_btn.Bind(wx.EVT_BUTTON, self._on_swap)
            swap_sizer.Add(self._swap_btn, 0, wx.ALL, 4)
            sizer.Add(swap_sizer, 0, wx.LEFT, 4)

            # Output name
            out_label = wx.StaticText(self, label="Output Name (optional):")
            sizer.Add(out_label, 0, wx.LEFT | wx.TOP, 8)
            self._output_name = wx.TextCtrl(self, name="Output Name")
            self._output_name.SetHint("Auto-named if empty")
            sizer.Add(self._output_name, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Action buttons
            sizer.AddSpacer(12)
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._combine_btn = wx.Button(self, label="Combine")
            self._combine_btn.Bind(wx.EVT_BUTTON, self._on_combine)
            btn_sizer.Add(self._combine_btn, 0, wx.ALL, 4)

            self._preview_btn = wx.Button(self, label="Preview")
            self._preview_btn.Bind(wx.EVT_BUTTON, self._on_preview)
            btn_sizer.Add(self._preview_btn, 0, wx.ALL, 4)

            sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            # Status
            self._status = wx.StaticText(self, label="")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        # -- Event handlers ---------------------------------------------------

        def _on_operation_change(self, event: wx.CommandEvent) -> None:
            """Update description when operation selection changes."""
            op = OPERATION_TYPES[self._operation.GetSelection()]
            self._op_desc.SetLabel(OPERATION_DESCRIPTIONS.get(op, ""))

        def _on_swap(self, event: wx.CommandEvent) -> None:
            """Swap Source A and Source B values."""
            val_a = self._source_a.GetValue()
            val_b = self._source_b.GetValue()
            self._source_a.SetValue(val_b)
            self._source_b.SetValue(val_a)
            self._status.SetLabel("Sources swapped")

        def _on_combine(self, event: wx.CommandEvent) -> None:
            """Dispatch the splice/combine command through the bridge."""
            source_a = self._source_a.GetValue().strip()
            source_b = self._source_b.GetValue().strip()

            if not source_a:
                self._status.SetLabel("Enter Source A object name")
                return
            if not source_b:
                self._status.SetLabel("Enter Source B object name")
                return

            op = OPERATION_TYPES[self._operation.GetSelection()]
            output = self._output_name.GetValue().strip()

            args = [source_a, source_b, op]
            if output:
                args.extend(['--name', output])

            result = self.bridge.execute_command('/splice', args)
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)

        def _on_preview(self, event: wx.CommandEvent) -> None:
            """Preview the combination without committing."""
            source_a = self._source_a.GetValue().strip()
            source_b = self._source_b.GetValue().strip()

            if not source_a:
                self._status.SetLabel("Enter Source A object name")
                return
            if not source_b:
                self._status.SetLabel("Enter Source B object name")
                return

            op = OPERATION_TYPES[self._operation.GetSelection()]

            args = [source_a, source_b, op, '--preview']
            result = self.bridge.execute_command('/splice', args)
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)

else:
    class SpliceCombinePanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
