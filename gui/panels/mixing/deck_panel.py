"""Deck Panel — Mixing Window.

Per-deck controls for the DJ workflow.  Supports 4 decks with
load, play, stop, BPM display, and pitch adjustment.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Mixing Window

BUILD ID: panel_deck_v1.0_phase5
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

# Available deck IDs
DECK_IDS = ['1', '2', '3', '4']

if _WX_AVAILABLE:

    class DeckPanel(wx.Panel):
        """Per-deck DJ controls.

        Controls:
        - Deck selector (1-4)
        - Load track button
        - Play / Stop buttons
        - BPM display (read-only)
        - Pitch slider (-12 to +12 semitones, center = 0)

        All actions dispatch through the Bridge.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="Deck Controls")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # ---- Deck selector ----
            deck_sizer = wx.BoxSizer(wx.HORIZONTAL)
            deck_label = wx.StaticText(self, label="Deck:")
            deck_sizer.Add(deck_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._deck_choice = wx.Choice(
                self, choices=DECK_IDS, name="Deck Selector",
            )
            self._deck_choice.SetSelection(0)
            deck_sizer.Add(self._deck_choice, 0, wx.LEFT | wx.RIGHT, 4)
            sizer.Add(deck_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Load track ----
            load_sizer = wx.BoxSizer(wx.HORIZONTAL)
            track_label = wx.StaticText(self, label="Track:")
            load_sizer.Add(track_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._track_field = wx.TextCtrl(self, name="Track to Load")
            self._track_field.SetHint("Track or file name")
            load_sizer.Add(self._track_field, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)

            self._load_btn = wx.Button(self, label="Load", name="Load Track")
            self._load_btn.Bind(wx.EVT_BUTTON, self._on_load)
            load_sizer.Add(self._load_btn, 0, wx.RIGHT, 4)
            sizer.Add(load_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Play / Stop buttons ----
            transport_sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._play_btn = wx.Button(self, label="Play", name="Play Deck")
            self._play_btn.Bind(wx.EVT_BUTTON, self._on_play)
            transport_sizer.Add(self._play_btn, 0, wx.ALL, 4)

            self._stop_btn = wx.Button(self, label="Stop", name="Stop Deck")
            self._stop_btn.Bind(wx.EVT_BUTTON, self._on_stop)
            transport_sizer.Add(self._stop_btn, 0, wx.ALL, 4)

            sizer.Add(transport_sizer, 0, wx.LEFT, 4)

            # ---- BPM display ----
            bpm_sizer = wx.BoxSizer(wx.HORIZONTAL)
            bpm_label = wx.StaticText(self, label="BPM:")
            bpm_sizer.Add(bpm_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._bpm_display = wx.TextCtrl(
                self, value="---", style=wx.TE_READONLY, name="Deck BPM Display",
            )
            bpm_sizer.Add(self._bpm_display, 0, wx.LEFT | wx.RIGHT, 4)
            sizer.Add(bpm_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Pitch slider ----
            pitch_sizer = wx.BoxSizer(wx.HORIZONTAL)
            pitch_label = wx.StaticText(self, label="Pitch:")
            pitch_sizer.Add(pitch_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._pitch_slider = wx.Slider(
                self, value=0, minValue=-12, maxValue=12,
                style=wx.SL_HORIZONTAL | wx.SL_LABELS,
                name="Deck Pitch",
            )
            self._pitch_slider.Bind(wx.EVT_SLIDER, self._on_pitch)
            pitch_sizer.Add(self._pitch_slider, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)
            sizer.Add(pitch_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Status ----
            self._status = wx.StaticText(self, label="", name="Deck Status")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        # ---- Helpers ----

        def _current_deck(self) -> str:
            return DECK_IDS[self._deck_choice.GetSelection()]

        def _set_status(self, text: str) -> None:
            display = text[:160] if len(text) > 160 else text
            self._status.SetLabel(display)
            self._status.Wrap(self.GetSize().width - 16)

        # ---- Event handlers ----

        def _on_load(self, event: wx.CommandEvent) -> None:
            """Load a track onto the selected deck."""
            deck = self._current_deck()
            track = self._track_field.GetValue().strip()
            if not track:
                self._set_status("Enter a track name to load")
                return
            result = self.bridge.execute_command('/deck', [deck, track])
            self._set_status(result)

        def _on_play(self, event: wx.CommandEvent) -> None:
            """Start playback on the selected deck."""
            deck = self._current_deck()
            result = self.bridge.execute_command('/play', [deck])
            self._set_status(result)

        def _on_stop(self, event: wx.CommandEvent) -> None:
            """Stop playback on the selected deck."""
            deck = self._current_deck()
            result = self.bridge.execute_command('/stop', [deck])
            self._set_status(result)

        def _on_pitch(self, event: wx.CommandEvent) -> None:
            """Adjust pitch on the selected deck."""
            deck = self._current_deck()
            pitch = self._pitch_slider.GetValue()
            result = self.bridge.execute_command(
                '/pitch', [deck, str(pitch)],
            )
            self._set_status(result)

else:
    class DeckPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
