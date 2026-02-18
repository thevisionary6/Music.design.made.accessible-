"""Effect Browser Panel — Effects Window.

Browsable, categorised list of all 113+ effects.  Search field for
fast access.  Selecting an effect shows its parameters and allows
adding it to the current chain.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Effects Window

BUILD ID: panel_effect_browser_v1.0_phase4
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

# Effect categories matching fx_cmds.py
EFFECT_CATEGORIES = {
    'Reverb': ['reverb_small', 'reverb_large', 'reverb_plate', 'reverb_spring', 'reverb_cathedral'],
    'Convolution': ['conv_hall', 'conv_hall_long', 'conv_room', 'conv_plate', 'conv_spring', 'conv_shimmer', 'conv_reverse'],
    'Delay': ['delay_simple', 'delay_pingpong', 'delay_multitap', 'delay_slapback', 'delay_tape'],
    'Saturation': ['saturate_soft', 'saturate_hard', 'saturate_overdrive', 'saturate_fuzz', 'saturate_tube'],
    'Overdrive': ['vamp_light', 'vamp_medium', 'vamp_heavy', 'vamp_fuzz',
                  'overdrive_soft', 'overdrive_classic', 'overdrive_crunch',
                  'dual_od_warm', 'dual_od_bright', 'dual_od_heavy',
                  'waveshape_fold', 'waveshape_rectify', 'waveshape_sine'],
    'Dynamics': ['compress_mild', 'compress_hard', 'compress_limiter', 'compress_expander', 'compress_softclipper'],
    'Gate': ['gate1', 'gate2', 'gate3', 'gate4', 'gate5'],
    'Lo-Fi': ['lofi_bitcrush', 'lofi_chorus', 'lofi_flanger', 'lofi_phaser', 'lofi_filter', 'lofi_halftime'],
    'Granular': ['granular_cloud', 'granular_scatter', 'granular_stretch', 'granular_freeze',
                 'granular_shimmer', 'granular_reverse', 'granular_stutter'],
    'Utility': ['util_normalize', 'util_normalize_rms', 'util_declip', 'util_declick',
                'util_smooth', 'util_smooth_heavy', 'util_dc_remove',
                'util_fade_in', 'util_fade_out', 'util_fade_both'],
}

if _WX_AVAILABLE:

    class EffectBrowserPanel(wx.Panel):
        """Browsable, categorised effect list with search.

        Controls:
        - Search field for filtering
        - Tree view of categories and effects
        - Apply button to add selected effect
        - Info display for selected effect
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            title = wx.StaticText(self, label="Effect Browser")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # Search
            search_sizer = wx.BoxSizer(wx.HORIZONTAL)
            search_label = wx.StaticText(self, label="Search:")
            search_sizer.Add(search_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._search = wx.TextCtrl(self, name="Effect Search")
            self._search.SetHint("Type to filter effects...")
            self._search.Bind(wx.EVT_TEXT, self._on_search)
            search_sizer.Add(self._search, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)
            sizer.Add(search_sizer, 0, wx.EXPAND | wx.ALL, 4)

            # Tree
            self._tree = wx.TreeCtrl(
                self, style=wx.TR_DEFAULT_STYLE | wx.TR_HIDE_ROOT,
                name="Effect Categories",
            )
            self._populate_tree()
            sizer.Add(self._tree, 1, wx.EXPAND | wx.ALL, 4)

            # Apply button
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
            self._apply_btn = wx.Button(self, label="Apply Effect")
            self._apply_btn.Bind(wx.EVT_BUTTON, self._on_apply)
            btn_sizer.Add(self._apply_btn, 0, wx.ALL, 4)

            self._info_btn = wx.Button(self, label="Show Info")
            self._info_btn.Bind(wx.EVT_BUTTON, self._on_info)
            btn_sizer.Add(self._info_btn, 0, wx.ALL, 4)
            sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            # Status
            self._status = wx.StaticText(self, label="")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        def _populate_tree(self, filter_text: str = "") -> None:
            self._tree.DeleteAllItems()
            root = self._tree.AddRoot("Effects")
            filt = filter_text.lower()

            for cat_name, effects in EFFECT_CATEGORIES.items():
                matching = [e for e in effects if not filt or filt in e.lower() or filt in cat_name.lower()]
                if not matching:
                    continue
                cat_node = self._tree.AppendItem(root, f"{cat_name} ({len(matching)})")
                for effect in matching:
                    self._tree.AppendItem(cat_node, effect)
                if filt:
                    self._tree.Expand(cat_node)

        def _on_search(self, event: wx.CommandEvent) -> None:
            self._populate_tree(self._search.GetValue())

        def _get_selected_effect(self) -> str:
            item = self._tree.GetSelection()
            if not item.IsOk():
                return ""
            label = self._tree.GetItemText(item)
            if '(' in label:
                return ""  # Category node
            return label

        def _on_apply(self, event: wx.CommandEvent) -> None:
            effect = self._get_selected_effect()
            if not effect:
                self._status.SetLabel("Select an effect first")
                return
            result = self.bridge.execute_command('/fx', [effect])
            self._status.SetLabel(result[:120] if len(result) > 120 else result)

        def _on_info(self, event: wx.CommandEvent) -> None:
            effect = self._get_selected_effect()
            if not effect:
                self._status.SetLabel("Select an effect first")
                return
            result = self.bridge.execute_command('/hfx', [effect])
            self._status.SetLabel(result[:200] if len(result) > 200 else result)

else:
    class EffectBrowserPanel:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
