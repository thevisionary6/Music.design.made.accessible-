"""Effect Chain Builder Panel — Effects Window.

Visual ordered list of effects applied in sequence.  Add/remove/reorder
effects.  Save the current chain as a named EffectChain object.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Effects Window

BUILD ID: panel_chain_builder_v1.0_phase4
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

    class ChainBuilderPanel(wx.Panel):
        """Build and manage effect chains.

        Controls:
        - Ordered list of effects in the chain
        - Add/Remove/Move Up/Move Down buttons
        - Save chain as named EffectChain object
        - Apply chain to current buffer
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._chain: list[str] = []
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            title = wx.StaticText(self, label="Effect Chain Builder")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # Chain list
            self._list = wx.ListBox(self, name="Effect Chain", style=wx.LB_SINGLE)
            sizer.Add(self._list, 1, wx.EXPAND | wx.ALL, 4)

            # Control buttons
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

            btn_add = wx.Button(self, label="Add Effect")
            btn_add.Bind(wx.EVT_BUTTON, self._on_add)
            btn_sizer.Add(btn_add, 0, wx.ALL, 2)

            btn_remove = wx.Button(self, label="Remove")
            btn_remove.Bind(wx.EVT_BUTTON, self._on_remove)
            btn_sizer.Add(btn_remove, 0, wx.ALL, 2)

            btn_up = wx.Button(self, label="Move Up")
            btn_up.Bind(wx.EVT_BUTTON, self._on_move_up)
            btn_sizer.Add(btn_up, 0, wx.ALL, 2)

            btn_down = wx.Button(self, label="Move Down")
            btn_down.Bind(wx.EVT_BUTTON, self._on_move_down)
            btn_sizer.Add(btn_down, 0, wx.ALL, 2)

            sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            # Action buttons
            action_sizer = wx.BoxSizer(wx.HORIZONTAL)

            btn_apply = wx.Button(self, label="Apply Chain")
            btn_apply.Bind(wx.EVT_BUTTON, self._on_apply)
            action_sizer.Add(btn_apply, 0, wx.ALL, 4)

            btn_clear = wx.Button(self, label="Clear")
            btn_clear.Bind(wx.EVT_BUTTON, self._on_clear)
            action_sizer.Add(btn_clear, 0, wx.ALL, 4)

            sizer.Add(action_sizer, 0, wx.LEFT, 4)

            # Save as EffectChain object
            save_sizer = wx.BoxSizer(wx.HORIZONTAL)
            save_label = wx.StaticText(self, label="Save as:")
            save_sizer.Add(save_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._save_name = wx.TextCtrl(self, name="Chain Name")
            self._save_name.SetHint("chain_name")
            save_sizer.Add(self._save_name, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)
            btn_save = wx.Button(self, label="Save Chain")
            btn_save.Bind(wx.EVT_BUTTON, self._on_save)
            save_sizer.Add(btn_save, 0, wx.RIGHT, 4)
            sizer.Add(save_sizer, 0, wx.EXPAND | wx.ALL, 4)

            # Status
            self._status = wx.StaticText(self, label="Chain: empty")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        def _refresh_list(self) -> None:
            self._list.Clear()
            for i, fx in enumerate(self._chain):
                self._list.Append(f"[{i+1}] {fx}")
            count = len(self._chain)
            self._status.SetLabel(f"Chain: {count} effect(s)" if count else "Chain: empty")

        def _on_add(self, event: wx.CommandEvent) -> None:
            from gui.panels.effects.effect_browser import EFFECT_CATEGORIES
            all_effects = []
            for effects in EFFECT_CATEGORIES.values():
                all_effects.extend(effects)
            all_effects.sort()

            dlg = wx.SingleChoiceDialog(self, "Select effect to add:",
                                        "Add Effect", all_effects)
            if dlg.ShowModal() == wx.ID_OK:
                self._chain.append(dlg.GetStringSelection())
                self._refresh_list()
            dlg.Destroy()

        def _on_remove(self, event: wx.CommandEvent) -> None:
            idx = self._list.GetSelection()
            if idx != wx.NOT_FOUND and idx < len(self._chain):
                self._chain.pop(idx)
                self._refresh_list()

        def _on_move_up(self, event: wx.CommandEvent) -> None:
            idx = self._list.GetSelection()
            if idx > 0:
                self._chain[idx-1], self._chain[idx] = self._chain[idx], self._chain[idx-1]
                self._refresh_list()
                self._list.SetSelection(idx - 1)

        def _on_move_down(self, event: wx.CommandEvent) -> None:
            idx = self._list.GetSelection()
            if idx != wx.NOT_FOUND and idx < len(self._chain) - 1:
                self._chain[idx], self._chain[idx+1] = self._chain[idx+1], self._chain[idx]
                self._refresh_list()
                self._list.SetSelection(idx + 1)

        def _on_apply(self, event: wx.CommandEvent) -> None:
            if not self._chain:
                self._status.SetLabel("Chain is empty — add effects first")
                return
            fx_str = '+'.join(self._chain)
            result = self.bridge.execute_command('/fx', [fx_str])
            self._status.SetLabel(result[:120] if len(result) > 120 else result)

        def _on_clear(self, event: wx.CommandEvent) -> None:
            self._chain.clear()
            self._refresh_list()

        def _on_save(self, event: wx.CommandEvent) -> None:
            name = self._save_name.GetValue().strip()
            if not name:
                self._status.SetLabel("Enter a name for the chain")
                return
            if not self._chain:
                self._status.SetLabel("Chain is empty — add effects first")
                return
            # Register as EffectChain object
            from mdma_rebuild.core.objects import EffectChain, EffectSlot
            chain_obj = EffectChain(
                name=name,
                effects=[EffectSlot(effect_name=fx) for fx in self._chain],
            )
            self.bridge.register_object(chain_obj, auto_name=False)
            self._status.SetLabel(f"Saved chain '{name}' ({len(self._chain)} effects)")

        def add_effect(self, effect_name: str) -> None:
            """Programmatically add an effect to the chain."""
            self._chain.append(effect_name)
            self._refresh_list()

else:
    class ChainBuilderPanel:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
