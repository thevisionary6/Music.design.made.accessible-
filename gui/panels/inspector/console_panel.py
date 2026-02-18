"""Console/CLI Mirror Panel — Inspector Window.

Live mirror of CLI output.  All actions taken in any GUI window produce
the same text feedback as the CLI equivalent.  Users can also type CLI
commands directly here.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Inspector Window

BUILD ID: panel_console_v1.0_phase8
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

    class ConsolePanel(wx.Panel):
        """CLI mirror and direct command input.

        Shows all command output.  Allows direct CLI command entry.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            label = wx.StaticText(self, label="Console")
            label.SetFont(label.GetFont().Bold())
            sizer.Add(label, 0, wx.ALL, 5)

            self._output = wx.TextCtrl(
                self,
                style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
                name="Console Output",
            )
            sizer.Add(self._output, 1, wx.EXPAND | wx.ALL, 2)

            # Command input
            input_sizer = wx.BoxSizer(wx.HORIZONTAL)
            prompt = wx.StaticText(self, label="/")
            input_sizer.Add(prompt, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)

            self._input = wx.TextCtrl(
                self, style=wx.TE_PROCESS_ENTER, name="Command Input",
            )
            self._input.SetHint("Type a CLI command and press Enter")
            self._input.Bind(wx.EVT_TEXT_ENTER, self._on_command)
            input_sizer.Add(self._input, 1, wx.EXPAND | wx.ALL, 2)

            sizer.Add(input_sizer, 0, wx.EXPAND)
            self.SetSizer(sizer)

        def _on_command(self, event: wx.CommandEvent) -> None:
            cmd = self._input.GetValue().strip()
            if not cmd:
                return
            self._input.Clear()
            self.append_text(f"> /{cmd}")
            result = self.bridge.execute_command(f"/{cmd}")
            self.append_text(result)
            self.append_text("")

        def append_text(self, text: str) -> None:
            """Add text to the console output."""
            self._output.AppendText(text + "\n")

else:
    class ConsolePanel:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
