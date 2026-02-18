"""Loop Generator Panel — Generation Window.

Wraps the /loop command.  Genre, bar count, layer toggles
(drums/bass/chords/melody).  Produces Loop objects containing linked
sub-patterns.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Generation Window

BUILD ID: panel_loop_generator_v1.0_phase3
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

GENRES = [
    'afrobeat', 'breakbeat', 'dnb', 'dubstep', 'hiphop',
    'house', 'lofi', 'minimal', 'reggaeton', 'techno', 'trap',
]

LAYERS = ['full', 'drums', 'bass', 'chords', 'melody']

if _WX_AVAILABLE:

    class LoopGeneratorPanel(wx.Panel):
        """Panel for generating multi-layer loops.

        Controls:
        - Genre dropdown (11 genres)
        - Bars spinner (1-32, default 4)
        - Layer checkboxes (drums, bass, chords, melody)
        - Name field (optional)
        - Generate button
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="Loop Generator")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # Genre
            genre_label = wx.StaticText(self, label="Genre:")
            sizer.Add(genre_label, 0, wx.LEFT | wx.TOP, 8)
            self._genre = wx.Choice(self, choices=GENRES, name="Genre")
            self._genre.SetSelection(GENRES.index('hiphop'))
            sizer.Add(self._genre, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Bars
            bars_label = wx.StaticText(self, label="Bars:")
            sizer.Add(bars_label, 0, wx.LEFT | wx.TOP, 8)
            self._bars = wx.SpinCtrl(self, min=1, max=32, initial=4, name="Bars")
            sizer.Add(self._bars, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Layer checkboxes
            layer_label = wx.StaticText(self, label="Layers:")
            sizer.Add(layer_label, 0, wx.LEFT | wx.TOP, 8)

            self._layer_checks: dict[str, wx.CheckBox] = {}
            layer_sizer = wx.BoxSizer(wx.HORIZONTAL)
            for layer in ['drums', 'bass', 'chords', 'melody']:
                cb = wx.CheckBox(self, label=layer.title(), name=f"Layer {layer}")
                cb.SetValue(True)
                self._layer_checks[layer] = cb
                layer_sizer.Add(cb, 0, wx.ALL, 4)
            sizer.Add(layer_sizer, 0, wx.LEFT, 4)

            # Name
            name_label = wx.StaticText(self, label="Name (optional):")
            sizer.Add(name_label, 0, wx.LEFT | wx.TOP, 8)
            self._name = wx.TextCtrl(self, name="Loop Name")
            self._name.SetHint("Auto-named if empty")
            sizer.Add(self._name, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Generate button
            sizer.AddSpacer(12)
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
            self._gen_btn = wx.Button(self, label="Generate Loop")
            self._gen_btn.Bind(wx.EVT_BUTTON, self._on_generate)
            btn_sizer.Add(self._gen_btn, 0, wx.ALL, 4)

            btn_full = wx.Button(self, label="Full Loop")
            btn_full.Bind(wx.EVT_BUTTON, lambda e: self._quick_gen('full'))
            btn_sizer.Add(btn_full, 0, wx.ALL, 4)

            sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            # Status
            self._status = wx.StaticText(self, label="")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        def _on_generate(self, event: wx.CommandEvent) -> None:
            genre = GENRES[self._genre.GetSelection()]
            bars = self._bars.GetValue()
            name = self._name.GetValue().strip()

            # Build layer string from checkboxes
            active = [l for l, cb in self._layer_checks.items() if cb.GetValue()]
            if len(active) == 4:
                layer_arg = 'full'
            elif active:
                layer_arg = ' '.join(active)
            else:
                layer_arg = 'full'

            args = [genre, str(bars)]
            if layer_arg != 'full':
                args.append(layer_arg)
            if name:
                args.extend(['--name', name])

            result = self.bridge.execute_command('/loop', args)
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)

        def _quick_gen(self, layer: str) -> None:
            genre = GENRES[self._genre.GetSelection()]
            bars = self._bars.GetValue()
            args = [genre, str(bars)]
            result = self.bridge.execute_command('/loop', args)
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)

else:
    class LoopGeneratorPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
