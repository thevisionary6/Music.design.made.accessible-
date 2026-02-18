"""Waveform Panel — Synthesis Window.

Waveform type selector across 22 oscillator types with preview display.
Wraps the /wave command.  Provides wavetable position control when
wavetable mode is active, and a preview region for visualising the
current waveform shape.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Synthesis Window

BUILD ID: panel_waveform_v1.0_phase5
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

# 22 waveform types — must match operator_panel.WAVEFORM_TYPES
WAVEFORM_TYPES = [
    'sine', 'square', 'triangle', 'saw', 'pulse', 'noise',
    'supersaw', 'additive', 'formant', 'harmonic', 'wavetable',
    'compound', 'pwm', 'sync', 'ring', 'fold', 'clip',
    'half_rect', 'full_rect', 'staircase', 'sample_hold', 'resonant',
]

# Categories for browsing convenience
WAVEFORM_CATEGORIES = {
    'Classic':   ['sine', 'square', 'triangle', 'saw', 'pulse', 'noise'],
    'Extended':  ['supersaw', 'additive', 'formant', 'harmonic', 'wavetable', 'compound'],
    'Shaped':    ['pwm', 'sync', 'ring', 'fold', 'clip'],
    'Rectified': ['half_rect', 'full_rect', 'staircase', 'sample_hold', 'resonant'],
}

if _WX_AVAILABLE:

    class WaveformPanel(wx.Panel):
        """Waveform type selector with preview and wavetable controls.

        Controls:
        - Waveform type list grouped by category
        - Preview display area
        - Wavetable position slider (active in wavetable mode)
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
            title = wx.StaticText(self, label="Waveform Selector")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # --- Waveform type tree ---
            wave_label = wx.StaticText(self, label="Waveform Type:")
            sizer.Add(wave_label, 0, wx.LEFT | wx.TOP, 8)

            self._tree = wx.TreeCtrl(
                self,
                style=wx.TR_DEFAULT_STYLE | wx.TR_HIDE_ROOT | wx.TR_SINGLE,
                name="Waveform Types",
            )
            self._populate_tree()
            self._tree.Bind(wx.EVT_TREE_SEL_CHANGED, self._on_selection_changed)
            sizer.Add(self._tree, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # --- Preview display ---
            preview_label = wx.StaticText(self, label="Preview:")
            sizer.Add(preview_label, 0, wx.LEFT | wx.TOP, 8)

            self._preview = wx.StaticText(
                self, label="(select a waveform to preview)",
                style=wx.ST_NO_AUTORESIZE | wx.BORDER_SUNKEN,
                size=(-1, 60),
                name="Waveform Preview",
            )
            sizer.Add(self._preview, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # --- Wavetable position (only active in wavetable mode) ---
            wt_label = wx.StaticText(self, label="Wavetable Position:")
            sizer.Add(wt_label, 0, wx.LEFT | wx.TOP, 8)

            self._wt_position = wx.Slider(
                self, value=0, minValue=0, maxValue=100,
                style=wx.SL_HORIZONTAL | wx.SL_VALUE_LABEL,
                name="Wavetable Position",
            )
            self._wt_position.Enable(False)  # disabled until wavetable selected
            sizer.Add(self._wt_position, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # --- Buttons ---
            sizer.AddSpacer(8)
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._apply_btn = wx.Button(self, label="Apply Waveform")
            self._apply_btn.Bind(wx.EVT_BUTTON, self._on_apply)
            btn_sizer.Add(self._apply_btn, 0, wx.ALL, 4)

            self._preview_btn = wx.Button(self, label="Preview Sound")
            self._preview_btn.Bind(wx.EVT_BUTTON, self._on_preview)
            btn_sizer.Add(self._preview_btn, 0, wx.ALL, 4)

            sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            # Status
            self._status = wx.StaticText(self, label="")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        def _populate_tree(self) -> None:
            """Build the categorised waveform tree."""
            self._tree.DeleteAllItems()
            root = self._tree.AddRoot("Waveforms")

            for cat_name, waves in WAVEFORM_CATEGORIES.items():
                cat_node = self._tree.AppendItem(root, cat_name)
                for wave in waves:
                    self._tree.AppendItem(cat_node, wave)
                self._tree.Expand(cat_node)

        def _get_selected_waveform(self) -> str:
            """Return selected waveform name, or empty string if category."""
            item = self._tree.GetSelection()
            if not item.IsOk():
                return ""
            label = self._tree.GetItemText(item)
            # Skip category nodes
            if label in WAVEFORM_CATEGORIES:
                return ""
            return label

        def _on_selection_changed(self, event: wx.TreeEvent) -> None:
            """Update preview and wavetable controls on selection."""
            waveform = self._get_selected_waveform()
            if not waveform:
                self._preview.SetLabel("(select a waveform to preview)")
                self._wt_position.Enable(False)
                return

            self._preview.SetLabel(f"Selected: {waveform}")
            # Enable wavetable position slider only in wavetable mode
            self._wt_position.Enable(waveform == 'wavetable')

        def _on_apply(self, event: wx.CommandEvent) -> None:
            """Apply selected waveform via bridge."""
            waveform = self._get_selected_waveform()
            if not waveform:
                self._status.SetLabel("Select a waveform first")
                return

            args = [waveform]
            if waveform == 'wavetable':
                pos = str(self._wt_position.GetValue())
                args.extend(['--position', pos])

            result = self.bridge.execute_command('/wave', args)
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)

        def _on_preview(self, event: wx.CommandEvent) -> None:
            """Preview selected waveform sound via bridge."""
            waveform = self._get_selected_waveform()
            if not waveform:
                self._status.SetLabel("Select a waveform first")
                return

            args = [waveform, '--preview']
            if waveform == 'wavetable':
                pos = str(self._wt_position.GetValue())
                args.extend(['--position', pos])

            result = self.bridge.execute_command('/wave', args)
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)

else:
    class WaveformPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
