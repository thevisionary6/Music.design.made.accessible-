"""Generative Theory Panel — Generation Window.

Music theory tools: scale explorer, chord progression builder, voice
leading suggestions.  Produces Pattern objects from theory-first input.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Generation Window

BUILD ID: panel_generative_theory_v1.0_phase3
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

ROOT_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

SCALES = [
    'major', 'minor', 'dorian', 'phrygian', 'lydian',
    'mixolydian', 'aeolian', 'locrian', 'pentatonic',
    'blues', 'harmonic_minor', 'melodic_minor', 'chromatic',
]

THEORY_QUERIES = ['scales', 'chords', 'progressions']

if _WX_AVAILABLE:

    class GenerativeTheoryPanel(wx.Panel):
        """Panel for music theory exploration and theory-driven generation.

        Controls:
        - Root note dropdown (12 notes)
        - Scale selector (13 scales)
        - Theory query buttons (scales, chords, progressions)
        - Key/scale setter
        - Theory info display area
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="Music Theory")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # Key selector
            key_sizer = wx.BoxSizer(wx.HORIZONTAL)

            root_label = wx.StaticText(self, label="Root:")
            key_sizer.Add(root_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._root = wx.Choice(self, choices=ROOT_NOTES, name="Root Note")
            self._root.SetSelection(0)  # C
            key_sizer.Add(self._root, 0, wx.LEFT, 4)

            scale_label = wx.StaticText(self, label="Scale:")
            key_sizer.Add(scale_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
            self._scale = wx.Choice(self, choices=SCALES, name="Scale")
            self._scale.SetSelection(1)  # minor
            key_sizer.Add(self._scale, 0, wx.LEFT, 4)

            set_key_btn = wx.Button(self, label="Set Key")
            set_key_btn.Bind(wx.EVT_BUTTON, self._on_set_key)
            key_sizer.Add(set_key_btn, 0, wx.LEFT, 8)

            sizer.Add(key_sizer, 0, wx.ALL, 4)

            # Theory query buttons
            query_sizer = wx.BoxSizer(wx.HORIZONTAL)
            for query in THEORY_QUERIES:
                btn = wx.Button(self, label=f"Show {query.title()}")
                btn.Bind(wx.EVT_BUTTON, lambda e, q=query: self._on_theory_query(q))
                query_sizer.Add(btn, 0, wx.ALL, 4)
            sizer.Add(query_sizer, 0, wx.LEFT, 4)

            # Info display
            info_label = wx.StaticText(self, label="Theory Info:")
            sizer.Add(info_label, 0, wx.LEFT | wx.TOP, 8)
            self._info = wx.TextCtrl(
                self,
                style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
                size=(-1, 200),
                name="Theory Info",
            )
            sizer.Add(self._info, 1, wx.EXPAND | wx.ALL, 8)

            self.SetSizer(sizer)

        def _on_set_key(self, event: wx.CommandEvent) -> None:
            root = ROOT_NOTES[self._root.GetSelection()]
            scale = SCALES[self._scale.GetSelection()]
            result = self.bridge.execute_command('/key', [root, scale])
            self._info.SetValue(result)

        def _on_theory_query(self, query: str) -> None:
            result = self.bridge.execute_command('/theory', [query])
            self._info.SetValue(result)

else:
    class GenerativeTheoryPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
