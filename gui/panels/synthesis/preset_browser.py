"""Preset Browser Panel — Synthesis Window.

Browse, load, and save Patch presets.  Wraps the /preset command.
Provides a searchable list of available presets with load/save/delete
actions.  Presets are Patch objects stored in the ObjectRegistry.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Synthesis Window

BUILD ID: panel_preset_browser_v1.0_phase5
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

# Built-in preset categories for organisation
PRESET_CATEGORIES = [
    'All', 'Bass', 'Lead', 'Pad', 'Keys', 'Pluck',
    'Strings', 'Brass', 'Percussion', 'FX', 'User',
]

if _WX_AVAILABLE:

    class PresetBrowserPanel(wx.Panel):
        """Patch preset browser with load, save, and management.

        Controls:
        - Category filter dropdown
        - Search field
        - Preset list
        - Preset name field for saving
        - Load / Save / Delete buttons
        - Refresh button

        All actions dispatch through the Bridge.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()
            self._refresh_presets()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="Preset Browser")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # --- Category filter ---
            cat_sizer = wx.BoxSizer(wx.HORIZONTAL)
            cat_label = wx.StaticText(self, label="Category:")
            cat_sizer.Add(cat_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._category = wx.Choice(
                self, choices=PRESET_CATEGORIES, name="Preset Category",
            )
            self._category.SetSelection(0)  # default: All
            self._category.Bind(wx.EVT_CHOICE, self._on_filter_changed)
            cat_sizer.Add(self._category, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)
            sizer.Add(cat_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # --- Search field ---
            search_sizer = wx.BoxSizer(wx.HORIZONTAL)
            search_label = wx.StaticText(self, label="Search:")
            search_sizer.Add(search_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._search = wx.TextCtrl(self, name="Preset Search")
            self._search.SetHint("Filter presets...")
            self._search.Bind(wx.EVT_TEXT, self._on_filter_changed)
            search_sizer.Add(self._search, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)
            sizer.Add(search_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # --- Preset list ---
            list_label = wx.StaticText(self, label="Available Presets:")
            sizer.Add(list_label, 0, wx.LEFT | wx.TOP, 8)
            self._preset_list = wx.ListBox(
                self, style=wx.LB_SINGLE, name="Preset List",
            )
            sizer.Add(self._preset_list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # --- Name field for saving ---
            name_label = wx.StaticText(self, label="Preset Name:")
            sizer.Add(name_label, 0, wx.LEFT | wx.TOP, 8)
            self._name_field = wx.TextCtrl(self, name="Preset Name")
            self._name_field.SetHint("Enter name for saving...")
            sizer.Add(self._name_field, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # --- Buttons ---
            sizer.AddSpacer(8)
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._load_btn = wx.Button(self, label="Load Preset")
            self._load_btn.Bind(wx.EVT_BUTTON, self._on_load)
            btn_sizer.Add(self._load_btn, 0, wx.ALL, 4)

            self._save_btn = wx.Button(self, label="Save Preset")
            self._save_btn.Bind(wx.EVT_BUTTON, self._on_save)
            btn_sizer.Add(self._save_btn, 0, wx.ALL, 4)

            self._delete_btn = wx.Button(self, label="Delete")
            self._delete_btn.Bind(wx.EVT_BUTTON, self._on_delete)
            btn_sizer.Add(self._delete_btn, 0, wx.ALL, 4)

            self._refresh_btn = wx.Button(self, label="Refresh")
            self._refresh_btn.Bind(wx.EVT_BUTTON, self._on_refresh)
            btn_sizer.Add(self._refresh_btn, 0, wx.ALL, 4)

            sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            # Status
            self._status = wx.StaticText(self, label="")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        def _refresh_presets(self) -> None:
            """Fetch preset list from the engine via bridge."""
            result = self.bridge.execute_command('/preset', ['--list'])
            # Parse newline-separated preset names from result
            self._all_presets = [
                line.strip() for line in result.splitlines()
                if line.strip() and not line.startswith('Error')
            ]
            self._apply_filter()

        def _apply_filter(self) -> None:
            """Apply category and search filters to the preset list."""
            category = PRESET_CATEGORIES[self._category.GetSelection()]
            search_text = self._search.GetValue().lower()

            filtered = self._all_presets
            if category != 'All':
                cat_lower = category.lower()
                filtered = [p for p in filtered if cat_lower in p.lower()]
            if search_text:
                filtered = [p for p in filtered if search_text in p.lower()]

            self._preset_list.Set(filtered)

        def _on_filter_changed(self, event: wx.CommandEvent) -> None:
            """Re-apply filters when category or search changes."""
            self._apply_filter()

        def _get_selected_preset(self) -> str:
            """Return currently selected preset name."""
            sel = self._preset_list.GetSelection()
            if sel == wx.NOT_FOUND:
                return ""
            return self._preset_list.GetString(sel)

        def _on_load(self, event: wx.CommandEvent) -> None:
            """Load selected preset via bridge."""
            preset = self._get_selected_preset()
            if not preset:
                self._status.SetLabel("Select a preset to load")
                return
            result = self.bridge.execute_command('/preset', ['--load', preset])
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)

        def _on_save(self, event: wx.CommandEvent) -> None:
            """Save current patch as a preset via bridge."""
            name = self._name_field.GetValue().strip()
            if not name:
                self._status.SetLabel("Enter a preset name to save")
                return
            result = self.bridge.execute_command('/preset', ['--save', name])
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)
            # Refresh list to include newly saved preset
            self._refresh_presets()

        def _on_delete(self, event: wx.CommandEvent) -> None:
            """Delete selected preset via bridge."""
            preset = self._get_selected_preset()
            if not preset:
                self._status.SetLabel("Select a preset to delete")
                return

            dlg = wx.MessageDialog(
                self,
                f"Delete preset '{preset}'?",
                "Confirm Delete",
                wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING,
            )
            if dlg.ShowModal() == wx.ID_YES:
                result = self.bridge.execute_command('/preset', ['--delete', preset])
                self._status.SetLabel(result[:120] if len(result) > 120 else result)
                self._status.Wrap(self.GetSize().width - 16)
                self._refresh_presets()
            dlg.Destroy()

        def _on_refresh(self, event: wx.CommandEvent) -> None:
            """Manually refresh the preset list."""
            self._refresh_presets()
            self._status.SetLabel(f"Refreshed: {len(self._all_presets)} presets found")

else:
    class PresetBrowserPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
