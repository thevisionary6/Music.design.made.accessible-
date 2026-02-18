"""Song Settings Panel — Arrangement Window.

Global song parameters: BPM, time signature, key, scale, output
format, and HQ mode.  Save-song button commits current settings
to the session.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Arrangement Window

BUILD ID: panel_song_settings_v1.0_phase5
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

# Constants
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

SCALES = [
    'major', 'minor', 'dorian', 'phrygian', 'lydian', 'mixolydian',
    'aeolian', 'locrian', 'harmonic_minor', 'melodic_minor',
    'pentatonic_major', 'pentatonic_minor', 'blues', 'chromatic',
    'whole_tone',
]

TIME_SIG_NUMERATORS = [str(n) for n in range(1, 17)]
TIME_SIG_DENOMINATORS = ['2', '4', '8', '16']

OUTPUT_FORMATS = ['wav_16', 'wav_24', 'flac_24']

if _WX_AVAILABLE:

    class SongSettingsPanel(wx.Panel):
        """Global song settings panel.

        Controls:
        - BPM spinner (60-300, default 120)
        - Time signature: numerator (1-16), denominator (2/4/8/16)
        - Key dropdown (12 notes)
        - Scale dropdown
        - Output format choice (wav_16, wav_24, flac_24)
        - HQ mode checkbox
        - Save Song button

        All actions dispatch through the Bridge.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="Song Settings")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # ---- BPM ----
            bpm_sizer = wx.BoxSizer(wx.HORIZONTAL)
            bpm_label = wx.StaticText(self, label="BPM:")
            bpm_sizer.Add(bpm_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._bpm_spin = wx.SpinCtrl(
                self, min=60, max=300, initial=120, name="BPM",
            )
            bpm_sizer.Add(self._bpm_spin, 0, wx.LEFT | wx.RIGHT, 4)
            sizer.Add(bpm_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Time signature ----
            ts_sizer = wx.BoxSizer(wx.HORIZONTAL)
            ts_label = wx.StaticText(self, label="Time Signature:")
            ts_sizer.Add(ts_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)

            self._ts_num = wx.Choice(
                self, choices=TIME_SIG_NUMERATORS, name="Time Signature Numerator",
            )
            self._ts_num.SetSelection(3)  # default = 4
            ts_sizer.Add(self._ts_num, 0, wx.LEFT, 4)

            slash_label = wx.StaticText(self, label="/")
            ts_sizer.Add(slash_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 2)

            self._ts_den = wx.Choice(
                self, choices=TIME_SIG_DENOMINATORS, name="Time Signature Denominator",
            )
            self._ts_den.SetSelection(1)  # default = 4
            ts_sizer.Add(self._ts_den, 0, wx.RIGHT, 4)
            sizer.Add(ts_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Key ----
            key_sizer = wx.BoxSizer(wx.HORIZONTAL)
            key_label = wx.StaticText(self, label="Key:")
            key_sizer.Add(key_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._key_choice = wx.Choice(
                self, choices=NOTES, name="Key",
            )
            self._key_choice.SetSelection(0)  # default = C
            key_sizer.Add(self._key_choice, 0, wx.LEFT | wx.RIGHT, 4)
            sizer.Add(key_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Scale ----
            scale_sizer = wx.BoxSizer(wx.HORIZONTAL)
            scale_label = wx.StaticText(self, label="Scale:")
            scale_sizer.Add(scale_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._scale_choice = wx.Choice(
                self, choices=SCALES, name="Scale",
            )
            self._scale_choice.SetSelection(0)  # default = major
            scale_sizer.Add(self._scale_choice, 0, wx.LEFT | wx.RIGHT, 4)
            sizer.Add(scale_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- Output format ----
            fmt_sizer = wx.BoxSizer(wx.HORIZONTAL)
            fmt_label = wx.StaticText(self, label="Output Format:")
            fmt_sizer.Add(fmt_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._format_choice = wx.Choice(
                self, choices=OUTPUT_FORMATS, name="Output Format",
            )
            self._format_choice.SetSelection(0)  # default = wav_16
            fmt_sizer.Add(self._format_choice, 0, wx.LEFT | wx.RIGHT, 4)
            sizer.Add(fmt_sizer, 0, wx.EXPAND | wx.TOP, 4)

            # ---- HQ mode ----
            self._hq_check = wx.CheckBox(self, label="HQ Mode", name="HQ Mode")
            sizer.Add(self._hq_check, 0, wx.LEFT | wx.TOP, 8)

            # ---- Save Song button ----
            sizer.AddSpacer(12)
            self._save_btn = wx.Button(self, label="Save Song", name="Save Song")
            self._save_btn.Bind(wx.EVT_BUTTON, self._on_save)
            sizer.Add(self._save_btn, 0, wx.LEFT, 8)

            # ---- Status ----
            self._status = wx.StaticText(self, label="", name="Song Settings Status")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

        # ---- Helpers ----

        def _set_status(self, text: str) -> None:
            display = text[:160] if len(text) > 160 else text
            self._status.SetLabel(display)
            self._status.Wrap(self.GetSize().width - 16)

        # ---- Event handlers ----

        def _on_save(self, event: wx.CommandEvent) -> None:
            """Commit all song settings through the bridge."""
            bpm = self._bpm_spin.GetValue()
            ts_num = TIME_SIG_NUMERATORS[self._ts_num.GetSelection()]
            ts_den = TIME_SIG_DENOMINATORS[self._ts_den.GetSelection()]
            key = NOTES[self._key_choice.GetSelection()]
            scale = SCALES[self._scale_choice.GetSelection()]
            fmt = OUTPUT_FORMATS[self._format_choice.GetSelection()]
            hq = self._hq_check.GetValue()

            # Set BPM
            result = self.bridge.execute_command('/bpm', [str(bpm)])

            # Set time signature
            result = self.bridge.execute_command(
                '/timesig', [ts_num, ts_den],
            )

            # Set key and scale
            result = self.bridge.execute_command('/key', [key, scale])

            # Set output format
            result = self.bridge.execute_command('/format', [fmt])

            # Set HQ mode
            if hq:
                result = self.bridge.execute_command('/hq', ['on'])
            else:
                result = self.bridge.execute_command('/hq', ['off'])

            # Save song
            result = self.bridge.execute_command('/save', [])
            self._set_status(f"Song saved: BPM={bpm}, {ts_num}/{ts_den}, "
                             f"key={key} {scale}, format={fmt}, HQ={'on' if hq else 'off'}")

else:
    class SongSettingsPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
