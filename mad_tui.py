#!/usr/bin/env python
"""MDMA Textual TUI (NVDA-friendly).

- Multiline script editor
- Output log
- Command input
- /mad runs the script (non-slash lines auto-prefixed with '/')

Run:
    python bmdma.py
    /tui

Or directly:
    python mad_tui.py
"""

from __future__ import annotations

import os
import sys
from typing import List, Callable, Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from mdma_rebuild.core.session import Session
from mdma_rebuild.commands import advanced_cmds
from bmdma import build_command_table

# Import repeat block support
try:
    from mdma_rebuild.commands.math_cmds import (
        is_recording_repeat,
        record_repeat_command,
        cmd_repeat_end,
    )
    REPEAT_SUPPORT = True
except ImportError:
    REPEAT_SUPPORT = False

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Log, TextArea, Static
from textual.containers import Horizontal


class MDMA_TUI(App):
    CSS = """
    Screen { layout: vertical; }
    #top { height: 1fr; }
    #script { width: 1fr; height: 1fr; }
    #right { width: 28; min-width: 28; }
    #out { height: 14; }
    #cmd { height: 3; }
    """

    BINDINGS = [
        ("ctrl+r", "run_mad", "Run /mad"),
        ("ctrl+enter", "run_cmd", "Run cmd"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.session = Session()
        self.commands: Dict[str, Callable[..., str]] = build_command_table()
        # Store command executor in session for repeat blocks
        self.session.command_executor = self.execute_command

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="top"):
            yield TextArea(id="script")
            yield Static(
                "Keys:\n"
                "- Ctrl+R : run /mad\n"
                "- Ctrl+Enter : run command\n"
                "- Ctrl+Q : quit\n\n"
                "Tip:\nType /mad in the command line\n"
                "to run the script pane.\n\n"
                "Repeat blocks:\n"
                "/8 ... /end (8x)\n"
                "/16 ... /end (16x)\n",
                id="right",
            )
        yield Log(id="out", highlight=True)
        yield Input(placeholder="MDMA command here. /mad runs script pane.", id="cmd")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#script", TextArea).focus()
        self._log("MDMA TUI ready. Ctrl+R runs /mad.\n")

    def _log(self, text: str) -> None:
        self.query_one("#out", Log).write(text)

    def execute_command(self, cmd_line: str) -> str:
        """Execute a single command line. Handles repeat blocks."""
        if not cmd_line.startswith("/"):
            return "ERROR: Commands must start with /"
        parts = cmd_line[1:].split()
        if not parts:
            return ""
        cmd = parts[0].lower()
        args: List[str] = parts[1:]

        # Handle repeat block recording
        if REPEAT_SUPPORT and is_recording_repeat():
            if cmd == 'end':
                # End repeat block and execute
                return cmd_repeat_end(self.session, [])
            else:
                # Record the command
                result = record_repeat_command(cmd_line)
                return result if result else ""

        if len(cmd) >= 2 and cmd[0] == "d" and cmd[1:].isdigit():
            args = [cmd[1:]] + args
            cmd = "deck"

        if cmd in ("q", "quit", "exit"):
            return "EXIT"

        func = self.commands.get(cmd)
        if func is not None:
            try:
                return func(self.session, args) or ""
            except Exception as exc:
                msg = f"ERROR: {exc}"
                try:
                    help_text = advanced_cmds.get_command_help(cmd) if hasattr(advanced_cmds, "get_command_help") else ""
                except Exception:
                    help_text = ""
                if help_text:
                    msg += f"\n\n--- Help for /{cmd} ---\n{help_text}"
                return msg

        if cmd.isdigit():
            try:
                return advanced_cmds.cmd_stack_get(self.session, args, int(cmd))
            except Exception as exc:
                return f"ERROR: {exc}"

        similar = [c for c in self.commands.keys() if cmd in c or c.startswith(cmd[:2])][:5]
        if similar:
            return f"ERROR: Unknown command /{cmd}. Did you mean: {', '.join('/' + s for s in similar)}?"
        return f"ERROR: Unknown command /{cmd}"

    def action_run_cmd(self) -> None:
        inp = self.query_one("#cmd", Input)
        line = (inp.value or "").strip()
        if not line:
            return
        inp.value = ""
        self._log(f"> {line}\n")

        if line.lower().startswith("/mad"):
            self.action_run_mad()
            return

        out = self.execute_command(line)
        if out == "EXIT":
            self.exit()
            return
        if out:
            self._log(out + "\n")

    def action_run_mad(self) -> None:
        """Run the script pane contents as a batch of commands."""
        area = self.query_one("#script", TextArea)
        text = area.text or ""
        for raw in text.splitlines():
            line = raw.rstrip()
            if not line:
                continue
            if not line.startswith("/"):
                line = "/" + line
            out = self.execute_command(line)
            if out == "EXIT":
                self.exit()
                return
            if out:
                self._log(out + "\n")


def main() -> None:
    MDMA_TUI().run()


if __name__ == "__main__":
    main()
