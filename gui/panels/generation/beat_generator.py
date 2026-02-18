"""Beat Generator Panel — Generation Window.

Wraps the /beat command.  Provides genre selector, bar count, and
variation controls.  Output always produces a named BeatPattern object
registered in the object tree.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Generation Window

BUILD ID: panel_beat_generator_v1.0_phase3
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

# Available genres (matches beat_gen.GENRE_TEMPLATES)
GENRES = [
    'afrobeat', 'breakbeat', 'dnb', 'dubstep', 'hiphop',
    'house', 'lofi', 'minimal', 'reggaeton', 'techno', 'trap',
]

if _WX_AVAILABLE:

    class BeatGeneratorPanel(wx.Panel):
        """Panel for generating beat patterns.

        Controls:
        - Genre dropdown (11 genres)
        - Bars spinner (1-32, default 4)
        - Name field (optional, auto-named if empty)
        - Generate button

        All actions dispatch through the Bridge.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="Beat Generator")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # Genre selector
            genre_label = wx.StaticText(self, label="Genre:")
            sizer.Add(genre_label, 0, wx.LEFT | wx.TOP, 8)
            self._genre = wx.Choice(self, choices=GENRES, name="Genre")
            self._genre.SetSelection(GENRES.index('hiphop'))
            sizer.Add(self._genre, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Bars spinner
            bars_label = wx.StaticText(self, label="Bars:")
            sizer.Add(bars_label, 0, wx.LEFT | wx.TOP, 8)
            self._bars = wx.SpinCtrl(
                self, min=1, max=32, initial=4, name="Bars"
            )
            sizer.Add(self._bars, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Name field
            name_label = wx.StaticText(self, label="Name (optional):")
            sizer.Add(name_label, 0, wx.LEFT | wx.TOP, 8)
            self._name = wx.TextCtrl(self, name="Beat Name")
            self._name.SetHint("Auto-named if empty")
            sizer.Add(self._name, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Generate button
            sizer.AddSpacer(12)
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
            self._gen_btn = wx.Button(self, label="Generate Beat")
            self._gen_btn.Bind(wx.EVT_BUTTON, self._on_generate)
            btn_sizer.Add(self._gen_btn, 0, wx.ALL, 4)

            # Quick buttons
            btn4 = wx.Button(self, label="4 Bars")
            btn4.Bind(wx.EVT_BUTTON, lambda e: self._quick_gen(4))
            btn_sizer.Add(btn4, 0, wx.ALL, 4)

            btn8 = wx.Button(self, label="8 Bars")
            btn8.Bind(wx.EVT_BUTTON, lambda e: self._quick_gen(8))
            btn_sizer.Add(btn8, 0, wx.ALL, 4)

            sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            # Status
            self._status = wx.StaticText(self, label="")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        def _on_generate(self, event: wx.CommandEvent) -> None:
            """Handle Generate button click."""
            genre = GENRES[self._genre.GetSelection()]
            bars = self._bars.GetValue()
            name = self._name.GetValue().strip()
            args = [genre, str(bars)]
            if name:
                args.extend(['--name', name])
            result = self.bridge.execute_command('/beat', args)
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)

        def _quick_gen(self, bars: int) -> None:
            """Quick-generate with current genre and specified bars."""
            genre = GENRES[self._genre.GetSelection()]
            result = self.bridge.execute_command('/beat', [genre, str(bars)])
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)

else:
    class BeatGeneratorPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
