"""Melody & Harmony Panel — Generation Window.

Wraps /gen2 for melody, chords, bassline, arpeggio, drone.
Scale selector, length, root note, density controls.
Produces Pattern objects.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Generation Window

BUILD ID: panel_melody_harmony_v1.0_phase3
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

CONTENT_TYPES = ['melody', 'chords', 'bassline', 'arp', 'drone']

SCALES = [
    'major', 'minor', 'dorian', 'phrygian', 'lydian', 'mixolydian',
    'pentatonic_major', 'pentatonic_minor', 'blues', 'harmonic_minor',
    'melodic_minor', 'whole_tone', 'japanese', 'arabic', 'hungarian_minor',
]

CHORD_PROGS = [
    'I_IV_V', 'I_V_vi_IV', 'ii_V_I', 'I_vi_IV_V', 'vi_IV_I_V',
    'I_IV_vi_V', 'i_iv_v', 'i_VI_III_VII', 'i_iv_VII_III',
    'i_VII_VI_V', '12bar',
]

CHORD_TYPES = [
    'maj', 'min', 'dim', 'aug', 'maj7', 'min7', 'dom7',
    'sus2', 'sus4', 'add9', 'min9', 'maj9',
]

if _WX_AVAILABLE:

    class MelodyHarmonyPanel(wx.Panel):
        """Panel for generating melodic and harmonic content.

        Dynamically adjusts controls based on selected content type:
        - melody: scale dropdown, note count spinner
        - chords: progression dropdown, bars spinner
        - bassline: scale dropdown, bars spinner
        - arp: chord type dropdown, octaves spinner
        - drone: duration spinner
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            self._main_sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="Melody & Harmony")
            title.SetFont(title.GetFont().Bold())
            self._main_sizer.Add(title, 0, wx.ALL, 8)

            # Content type selector
            type_label = wx.StaticText(self, label="Type:")
            self._main_sizer.Add(type_label, 0, wx.LEFT | wx.TOP, 8)
            self._type = wx.Choice(
                self, choices=[t.title() for t in CONTENT_TYPES],
                name="Content Type",
            )
            self._type.SetSelection(0)
            self._type.Bind(wx.EVT_CHOICE, self._on_type_changed)
            self._main_sizer.Add(self._type, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Dynamic parameter panel
            self._param_panel = wx.Panel(self)
            self._param_sizer = wx.BoxSizer(wx.VERTICAL)
            self._param_panel.SetSizer(self._param_sizer)
            self._main_sizer.Add(self._param_panel, 0, wx.EXPAND)

            # Name field
            name_label = wx.StaticText(self, label="Name (optional):")
            self._main_sizer.Add(name_label, 0, wx.LEFT | wx.TOP, 8)
            self._name = wx.TextCtrl(self, name="Pattern Name")
            self._name.SetHint("Auto-named if empty")
            self._main_sizer.Add(self._name, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Generate button
            self._main_sizer.AddSpacer(12)
            self._gen_btn = wx.Button(self, label="Generate")
            self._gen_btn.Bind(wx.EVT_BUTTON, self._on_generate)
            self._main_sizer.Add(self._gen_btn, 0, wx.LEFT, 12)

            # Status
            self._status = wx.StaticText(self, label="")
            self._main_sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(self._main_sizer)
            self._build_params_for_type('melody')

        def _on_type_changed(self, event: wx.CommandEvent) -> None:
            ctype = CONTENT_TYPES[self._type.GetSelection()]
            self._build_params_for_type(ctype)

        def _build_params_for_type(self, ctype: str) -> None:
            """Rebuild parameter controls for the selected content type."""
            self._param_sizer.Clear(True)
            self._controls: dict[str, Any] = {}

            if ctype == 'melody':
                self._add_choice('scale', 'Scale:', SCALES, 'minor')
                self._add_spin('length', 'Note Count:', 2, 64, 8)
            elif ctype == 'chords':
                self._add_choice('progression', 'Progression:', CHORD_PROGS, 'I_IV_V')
                self._add_spin('bars', 'Bars:', 1, 32, 4)
            elif ctype == 'bassline':
                self._add_choice('scale', 'Scale:',
                                 ['major', 'minor', 'dorian', 'mixolydian',
                                  'pentatonic_minor', 'blues'], 'minor')
                self._add_spin('bars', 'Bars:', 1, 32, 4)
            elif ctype == 'arp':
                self._add_choice('chord', 'Chord Type:', CHORD_TYPES, 'min')
                self._add_spin('octaves', 'Octaves:', 1, 4, 2)
            elif ctype == 'drone':
                self._add_spin('duration', 'Duration (beats):', 1, 64, 8)

            self._param_panel.Layout()
            self.Layout()

        def _add_choice(self, key: str, label: str,
                        choices: list[str], default: str) -> None:
            lbl = wx.StaticText(self._param_panel, label=label)
            self._param_sizer.Add(lbl, 0, wx.LEFT | wx.TOP, 8)
            ctrl = wx.Choice(self._param_panel, choices=choices, name=label.rstrip(':'))
            idx = choices.index(default) if default in choices else 0
            ctrl.SetSelection(idx)
            self._param_sizer.Add(ctrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)
            self._controls[key] = (ctrl, choices)

        def _add_spin(self, key: str, label: str,
                      min_val: int, max_val: int, default: int) -> None:
            lbl = wx.StaticText(self._param_panel, label=label)
            self._param_sizer.Add(lbl, 0, wx.LEFT | wx.TOP, 8)
            ctrl = wx.SpinCtrl(
                self._param_panel, min=min_val, max=max_val,
                initial=default, name=label.rstrip(':'),
            )
            self._param_sizer.Add(ctrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)
            self._controls[key] = ctrl

        def _on_generate(self, event: wx.CommandEvent) -> None:
            ctype = CONTENT_TYPES[self._type.GetSelection()]
            name = self._name.GetValue().strip()

            if ctype == 'melody':
                ctrl, choices = self._controls['scale']
                scale = choices[ctrl.GetSelection()]
                length = self._controls['length'].GetValue()
                args = ['melody', scale, str(length)]
            elif ctype == 'chords':
                ctrl, choices = self._controls['progression']
                prog = choices[ctrl.GetSelection()]
                bars = self._controls['bars'].GetValue()
                args = ['chords', prog, str(bars)]
            elif ctype == 'bassline':
                ctrl, choices = self._controls['scale']
                scale = choices[ctrl.GetSelection()]
                bars = self._controls['bars'].GetValue()
                args = ['bassline', scale, str(bars)]
            elif ctype == 'arp':
                ctrl, choices = self._controls['chord']
                chord = choices[ctrl.GetSelection()]
                octaves = self._controls['octaves'].GetValue()
                args = ['arp', chord, str(octaves)]
            elif ctype == 'drone':
                duration = self._controls['duration'].GetValue()
                args = ['drone', str(duration)]
            else:
                args = [ctype]

            if name:
                args.extend(['--name', name])

            result = self.bridge.execute_command('/gen2', args)
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)

else:
    class MelodyHarmonyPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
