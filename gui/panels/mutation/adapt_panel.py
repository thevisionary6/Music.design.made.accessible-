"""Adapt Panel — Mutation & Editing Window.

Wraps the /adapt command.  Adapts existing objects to a new key, tempo,
style, or development direction.  Each subcommand reveals dynamic
controls specific to that adaptation mode.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Mutation Window

BUILD ID: panel_adapt_v1.0_phase5
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

# Subcommands for /adapt
ADAPT_SUBCOMMANDS = ['key', 'tempo', 'style', 'develop']

# Key choices for key adaptation
KEY_CHOICES = [
    'C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F',
    'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B',
]

# Scale/mode qualifiers
SCALE_CHOICES = [
    'major', 'minor', 'dorian', 'phrygian', 'lydian',
    'mixolydian', 'aeolian', 'locrian', 'harmonic_minor',
    'melodic_minor', 'pentatonic', 'blues',
]

# Development types
DEVELOP_TYPES = [
    'variation', 'extension', 'reduction', 'fragmentation',
    'sequence', 'modulation', 'intensify', 'simplify',
]

if _WX_AVAILABLE:

    class AdaptPanel(wx.Panel):
        """Panel for adapting existing objects to new musical contexts.

        Controls:
        - Source object name field
        - Subcommand selector (key, tempo, style, develop)
        - Dynamic controls per subcommand:
            key    -> target key dropdown + scale dropdown
            tempo  -> target BPM spinner
            style  -> style descriptor text field
            develop -> development type dropdown
        - Output name field (auto-named if empty)
        - Apply button

        All actions dispatch through the Bridge via /adapt.
        Non-destructive: always produces a new object.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._dynamic_ctrls: list[wx.Window] = []
            self._build_ui()

        def _build_ui(self) -> None:
            self._main_sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="Adapt")
            title.SetFont(title.GetFont().Bold())
            self._main_sizer.Add(title, 0, wx.ALL, 8)

            # Source object
            src_label = wx.StaticText(self, label="Source Object:")
            self._main_sizer.Add(src_label, 0, wx.LEFT | wx.TOP, 8)
            self._source = wx.TextCtrl(self, name="Source Object")
            self._source.SetHint("Enter object name")
            self._main_sizer.Add(self._source, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Subcommand selector
            sub_label = wx.StaticText(self, label="Adaptation Mode:")
            self._main_sizer.Add(sub_label, 0, wx.LEFT | wx.TOP, 8)
            self._subcmd = wx.Choice(
                self, choices=ADAPT_SUBCOMMANDS, name="Adaptation Mode",
            )
            self._subcmd.SetSelection(0)
            self._subcmd.Bind(wx.EVT_CHOICE, self._on_subcmd_change)
            self._main_sizer.Add(self._subcmd, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Dynamic controls container
            self._dynamic_sizer = wx.BoxSizer(wx.VERTICAL)
            self._main_sizer.Add(self._dynamic_sizer, 0, wx.EXPAND)

            # Output name
            out_label = wx.StaticText(self, label="Output Name (optional):")
            self._main_sizer.Add(out_label, 0, wx.LEFT | wx.TOP, 8)
            self._output_name = wx.TextCtrl(self, name="Output Name")
            self._output_name.SetHint("Auto-named if empty")
            self._main_sizer.Add(
                self._output_name, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8,
            )

            # Apply button
            self._main_sizer.AddSpacer(12)
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
            self._apply_btn = wx.Button(self, label="Apply Adaptation")
            self._apply_btn.Bind(wx.EVT_BUTTON, self._on_apply)
            btn_sizer.Add(self._apply_btn, 0, wx.ALL, 4)
            self._main_sizer.Add(btn_sizer, 0, wx.LEFT, 4)

            # Status
            self._status = wx.StaticText(self, label="")
            self._main_sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(self._main_sizer)

            # Build initial dynamic controls for default subcommand
            self._show_key_controls()

        # -- Dynamic control builders ----------------------------------------

        def _clear_dynamic(self) -> None:
            """Remove all dynamic controls."""
            for ctrl in self._dynamic_ctrls:
                ctrl.Destroy()
            self._dynamic_ctrls.clear()
            self._dynamic_sizer.Clear()

        def _show_key_controls(self) -> None:
            """Show controls for key adaptation."""
            self._clear_dynamic()

            key_label = wx.StaticText(self, label="Target Key:")
            self._dynamic_sizer.Add(key_label, 0, wx.LEFT | wx.TOP, 8)
            self._target_key = wx.Choice(
                self, choices=KEY_CHOICES, name="Target Key",
            )
            self._target_key.SetSelection(0)
            self._dynamic_sizer.Add(
                self._target_key, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8,
            )
            self._dynamic_ctrls.extend([key_label, self._target_key])

            scale_label = wx.StaticText(self, label="Scale / Mode:")
            self._dynamic_sizer.Add(scale_label, 0, wx.LEFT | wx.TOP, 8)
            self._target_scale = wx.Choice(
                self, choices=SCALE_CHOICES, name="Scale Mode",
            )
            self._target_scale.SetSelection(0)
            self._dynamic_sizer.Add(
                self._target_scale, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8,
            )
            self._dynamic_ctrls.extend([scale_label, self._target_scale])

            self.Layout()

        def _show_tempo_controls(self) -> None:
            """Show controls for tempo adaptation."""
            self._clear_dynamic()

            bpm_label = wx.StaticText(self, label="Target BPM:")
            self._dynamic_sizer.Add(bpm_label, 0, wx.LEFT | wx.TOP, 8)
            self._target_bpm = wx.SpinCtrl(
                self, min=20, max=300, initial=120, name="Target BPM",
            )
            self._dynamic_sizer.Add(
                self._target_bpm, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8,
            )
            self._dynamic_ctrls.extend([bpm_label, self._target_bpm])

            self.Layout()

        def _show_style_controls(self) -> None:
            """Show controls for style adaptation."""
            self._clear_dynamic()

            style_label = wx.StaticText(self, label="Style Descriptor:")
            self._dynamic_sizer.Add(style_label, 0, wx.LEFT | wx.TOP, 8)
            self._style_desc = wx.TextCtrl(self, name="Style Descriptor")
            self._style_desc.SetHint("e.g. jazz, aggressive, laid-back")
            self._dynamic_sizer.Add(
                self._style_desc, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8,
            )
            self._dynamic_ctrls.extend([style_label, self._style_desc])

            self.Layout()

        def _show_develop_controls(self) -> None:
            """Show controls for development adaptation."""
            self._clear_dynamic()

            dev_label = wx.StaticText(self, label="Development Type:")
            self._dynamic_sizer.Add(dev_label, 0, wx.LEFT | wx.TOP, 8)
            self._develop_type = wx.Choice(
                self, choices=DEVELOP_TYPES, name="Development Type",
            )
            self._develop_type.SetSelection(0)
            self._dynamic_sizer.Add(
                self._develop_type, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8,
            )
            self._dynamic_ctrls.extend([dev_label, self._develop_type])

            self.Layout()

        # -- Event handlers ---------------------------------------------------

        def _on_subcmd_change(self, event: wx.CommandEvent) -> None:
            """Switch dynamic controls when subcommand changes."""
            subcmd = ADAPT_SUBCOMMANDS[self._subcmd.GetSelection()]
            builders = {
                'key': self._show_key_controls,
                'tempo': self._show_tempo_controls,
                'style': self._show_style_controls,
                'develop': self._show_develop_controls,
            }
            builders[subcmd]()

        def _on_apply(self, event: wx.CommandEvent) -> None:
            """Dispatch /adapt {subcmd} with parameters through the bridge."""
            source = self._source.GetValue().strip()
            if not source:
                self._status.SetLabel("Enter a source object name")
                return

            subcmd = ADAPT_SUBCOMMANDS[self._subcmd.GetSelection()]
            output = self._output_name.GetValue().strip()

            args = [source, subcmd]

            if subcmd == 'key':
                key = KEY_CHOICES[self._target_key.GetSelection()]
                scale = SCALE_CHOICES[self._target_scale.GetSelection()]
                args.extend([key, scale])
            elif subcmd == 'tempo':
                bpm = self._target_bpm.GetValue()
                args.append(str(bpm))
            elif subcmd == 'style':
                style = self._style_desc.GetValue().strip()
                if not style:
                    self._status.SetLabel("Enter a style descriptor")
                    return
                args.append(style)
            elif subcmd == 'develop':
                dev = DEVELOP_TYPES[self._develop_type.GetSelection()]
                args.append(dev)

            if output:
                args.extend(['--name', output])

            result = self.bridge.execute_command('/adapt', args)
            self._status.SetLabel(result[:120] if len(result) > 120 else result)
            self._status.Wrap(self.GetSize().width - 16)

else:
    class AdaptPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
