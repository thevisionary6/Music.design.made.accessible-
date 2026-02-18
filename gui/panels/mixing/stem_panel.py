"""Stem Panel — Mixing Window.

Stem separation controls.  Allows separating a source audio object
into individual stems (vocals, drums, bass, other).

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Mixing Window

BUILD ID: panel_stem_v1.0_phase5
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

# Available stem types
STEM_TYPES = ['vocals', 'drums', 'bass', 'other']

if _WX_AVAILABLE:

    class StemPanel(wx.Panel):
        """Stem separation controls.

        Controls:
        - Source selector (text field for object name)
        - Stem type checkboxes (vocals, drums, bass, other)
        - Separate button
        - Select All / Deselect All quick buttons

        All actions dispatch through the Bridge.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._stem_checks: dict[str, wx.CheckBox] = {}
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="Stem Separation")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # ---- Source selector ----
            src_sizer = wx.BoxSizer(wx.HORIZONTAL)
            src_label = wx.StaticText(self, label="Source:")
            src_sizer.Add(src_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._source_field = wx.TextCtrl(self, name="Stem Source")
            self._source_field.SetHint("Audio object name or file")
            src_sizer.Add(self._source_field, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)
            sizer.Add(src_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Stem type checkboxes ----
            stems_label = wx.StaticText(self, label="Stems to extract:")
            sizer.Add(stems_label, 0, wx.LEFT | wx.TOP, 8)

            check_sizer = wx.BoxSizer(wx.VERTICAL)
            for stem in STEM_TYPES:
                cb = wx.CheckBox(
                    self, label=stem.capitalize(), name=f"Stem {stem.capitalize()}",
                )
                cb.SetValue(True)
                self._stem_checks[stem] = cb
                check_sizer.Add(cb, 0, wx.LEFT | wx.TOP, 12)
            sizer.Add(check_sizer, 0, wx.EXPAND)

            # ---- Select All / Deselect All ----
            sel_sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._sel_all_btn = wx.Button(
                self, label="Select All", name="Select All Stems",
            )
            self._sel_all_btn.Bind(wx.EVT_BUTTON, self._on_select_all)
            sel_sizer.Add(self._sel_all_btn, 0, wx.ALL, 4)

            self._desel_all_btn = wx.Button(
                self, label="Deselect All", name="Deselect All Stems",
            )
            self._desel_all_btn.Bind(wx.EVT_BUTTON, self._on_deselect_all)
            sel_sizer.Add(self._desel_all_btn, 0, wx.ALL, 4)

            sizer.Add(sel_sizer, 0, wx.LEFT, 4)

            # ---- Separate button ----
            sizer.AddSpacer(12)
            self._separate_btn = wx.Button(
                self, label="Separate", name="Separate Stems",
            )
            self._separate_btn.Bind(wx.EVT_BUTTON, self._on_separate)
            sizer.Add(self._separate_btn, 0, wx.LEFT, 8)

            # ---- Status ----
            self._status = wx.StaticText(self, label="", name="Stem Status")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        # ---- Helpers ----

        def _set_status(self, text: str) -> None:
            display = text[:160] if len(text) > 160 else text
            self._status.SetLabel(display)
            self._status.Wrap(self.GetSize().width - 16)

        def _selected_stems(self) -> list[str]:
            """Return list of checked stem types."""
            return [
                stem for stem, cb in self._stem_checks.items()
                if cb.GetValue()
            ]

        # ---- Event handlers ----

        def _on_select_all(self, event: wx.CommandEvent) -> None:
            """Check all stem type checkboxes."""
            for cb in self._stem_checks.values():
                cb.SetValue(True)

        def _on_deselect_all(self, event: wx.CommandEvent) -> None:
            """Uncheck all stem type checkboxes."""
            for cb in self._stem_checks.values():
                cb.SetValue(False)

        def _on_separate(self, event: wx.CommandEvent) -> None:
            """Run stem separation on the source with selected stems."""
            source = self._source_field.GetValue().strip()
            if not source:
                self._set_status("Enter a source audio object name")
                return

            stems = self._selected_stems()
            if not stems:
                self._set_status("Select at least one stem type")
                return

            args = [source] + stems
            result = self.bridge.execute_command('/stem', args)
            self._set_status(result)

else:
    class StemPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
