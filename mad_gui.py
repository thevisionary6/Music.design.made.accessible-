"""
DEPRECATED: This Tkinter GUI is superseded by mdma_gui.py (wxPython Action Panel).
Use `python mdma_gui.py` instead. This file is retained for reference only.

Simple graphical front‑end for the MDMA rebuild with DSL support.

This script provides a minimal GUI around the existing MDMA session
and command dispatch mechanism.  It adds a multi‑line editor for
authoring DSL scripts and a command entry field for running normal
MDMA commands.  When you type ``/mad`` into the command entry or
click the "Run /mad" button, the contents of the script editor are
scanned line by line.  Each line that does not begin with a slash
will be prefixed with one so it can be passed straight through the
existing MDMA command table.  Lines are executed in order and their
results printed to the output console.

To aid users of screen readers the script editor displays a narrow
column of line numbers alongside the text.  These numbers are
prefaced by an ``INVISIBLE SEPARATOR`` (U+2063) so NVDA and other
assistive technologies will ignore them while still giving sighted
users visual context.  If this behaviour is undesirable you can
change the ``_LINE_PREFIX`` constant below.

This front‑end does not replace the existing bmdma.py REPL; it
simply wraps the same session object and command table in a
Tkinter interface.  All baseline MDMA commands, including pattern
(``/mel``), output blocks (``/out``) and effect chains (``/chain``),
are dispatched directly to the MDMA engine.  Additional DSL
abstractions can be layered on top of this by modifying the
``run_mad_script`` method to perform pre‑processing before
execution.

To launch the GUI run this module directly from the root of the
mdma_v38 package:

    python -m mdma_v38.mad_gui

"""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk
from typing import List, Callable, Dict

try:
    # Import the MDMA session and command builder from the local package.
    from .bmdma import build_command_table
    from .mdma_rebuild.core.session import Session
    from .mdma_rebuild.commands import advanced_cmds  # type: ignore
except ImportError:
    # Fallback when running as a standalone script from inside mdma_v38
    from bmdma import build_command_table  # type: ignore
    from mdma_rebuild.core.session import Session  # type: ignore
    from mdma_rebuild.commands import advanced_cmds  # type: ignore


class MadGui:
    """Tkinter front‑end for MDMA with DSL script support."""

    # Invisible separator for line number prefix.  This is used to
    # prevent screen readers from reading the line numbers aloud.  If
    # you wish NVDA to read the numbers, set this to an empty string.
    _LINE_PREFIX: str = "\u2063"

    def __init__(self) -> None:
        # Create MDMA session and command table
        self.session: Session = Session()
        self.commands: Dict[str, Callable[..., str]] = build_command_table()

        # Build the GUI
        self.root: tk.Tk = tk.Tk()
        self.root.title("MDMA GUI")
        self.root.geometry("800x600")

        # Main layout frames
        self._build_script_editor()
        self._build_output_console()
        self._build_command_entry()

        # Redirect standard output/error to the console
        sys.stdout = self  # type: ignore
        sys.stderr = self  # type: ignore

        # Initialise line numbers
        self._update_line_numbers()

    # ------------------------------------------------------------------
    # GUI Construction
    # ------------------------------------------------------------------
    def _build_script_editor(self) -> None:
        """Create the multi‑line script editor with line numbers."""
        outer_frame = ttk.Frame(self.root)
        outer_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Line numbers: narrow read‑only text widget
        self.ln_text = tk.Text(outer_frame,
                               width=4,
                               padx=4,
                               takefocus=0,
                               bd=0,
                               bg=self.root.cget('bg'),
                               state='disabled',
                               highlightthickness=0)
        self.ln_text.pack(side=tk.LEFT, fill=tk.Y)

        # Script editing area
        self.script = tk.Text(outer_frame,
                              wrap='none',
                              undo=True)
        self.script.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind to update line numbers on changes
        self.script.bind('<KeyRelease>', self._on_script_change)
        self.script.bind('<MouseWheel>', self._sync_scroll)
        self.ln_text.bind('<MouseWheel>', self._sync_scroll)

        # Add a vertical scrollbar linking both widgets
        vsb = ttk.Scrollbar(outer_frame, orient='vertical', command=self._on_scroll)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.script.configure(yscrollcommand=vsb.set)
        self.ln_text.configure(yscrollcommand=vsb.set)

    def _build_output_console(self) -> None:
        """Create the read‑only output console."""
        # Separator line
        sep = ttk.Separator(self.root, orient='horizontal')
        sep.pack(fill=tk.X, pady=(0, 4))

        # Output area
        self.output = tk.Text(self.root,
                              height=10,
                              wrap='word',
                              state='disabled',
                              bg='#1e1e1e',
                              fg='#00ff00')
        self.output.pack(fill=tk.BOTH, expand=False, padx=4, pady=(0, 4))

    def _build_command_entry(self) -> None:
        """Create the command entry field and run button."""
        cmd_frame = ttk.Frame(self.root)
        cmd_frame.pack(fill=tk.X, padx=4, pady=(0, 4))
        self.cmd_entry = ttk.Entry(cmd_frame)
        self.cmd_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cmd_entry.bind('<Return>', self._on_command_entered)
        run_btn = ttk.Button(cmd_frame, text="Run /mad", command=self.run_mad_script)
        run_btn.pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # Output Redirection
    # ------------------------------------------------------------------
    def write(self, text: str) -> None:
        """Implement file‑like API so `print()` calls append to output."""
        self.output.configure(state='normal')
        self.output.insert(tk.END, text)
        self.output.see(tk.END)
        self.output.configure(state='disabled')

    def flush(self) -> None:  # pragma: no cover
        # No buffering, so nothing to flush
        return

    # ------------------------------------------------------------------
    # Line number management
    # ------------------------------------------------------------------
    def _update_line_numbers(self) -> None:
        """Refresh the line numbers to match the script content."""
        # Count lines in the script editor
        end_index = self.script.index('end-1c')
        try:
            line_count = int(end_index.split('.')[0])
        except Exception:
            line_count = 1
        # Update the line number widget
        self.ln_text.configure(state='normal')
        self.ln_text.delete('1.0', tk.END)
        for i in range(1, line_count + 1):
            # Prepend invisible separator to avoid screen reader reading
            self.ln_text.insert(tk.END, f"{self._LINE_PREFIX}{i}\n")
        self.ln_text.configure(state='disabled')

    def _on_script_change(self, event: tk.Event) -> None:
        """Callback when the script editor content changes."""
        # Update line numbers on user input
        self._update_line_numbers()

    def _on_scroll(self, *args: str) -> None:
        """Link scrolling between script and line numbers."""
        self.script.yview(*args)
        self.ln_text.yview(*args)

    def _sync_scroll(self, event: tk.Event) -> None:
        """Synchronise scroll when using mouse wheel."""
        self.script.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        self.ln_text.yview_scroll(int(-1 * (event.delta / 120)), 'units')

    # ------------------------------------------------------------------
    # Command entry handling
    # ------------------------------------------------------------------
    def _on_command_entered(self, event: tk.Event) -> None:
        """Handle the user pressing Return in the command entry."""
        cmd_line = self.cmd_entry.get().strip()
        if not cmd_line:
            return
        self.cmd_entry.delete(0, tk.END)
        self.write(f"> {cmd_line}\n")

        # Intercept /mad to run script editor contents
        if cmd_line.lstrip().lower().startswith('/mad'):
            self.run_mad_script()
            return
        # Otherwise, execute as single command
        result = self._execute_command(cmd_line)
        if result == "EXIT":
            self.root.quit()
        elif result:
            self.write(result + '\n')

    # ------------------------------------------------------------------
    # MDMA command dispatch
    # ------------------------------------------------------------------
    def _execute_command(self, cmd_line: str) -> str:
        """Run a single MDMA command and return its result string."""
        if not cmd_line.startswith('/'):
            return "ERROR: Commands must start with /"
        parts = cmd_line[1:].split()
        if not parts:
            return ""
        cmd = parts[0].lower()
        args: List[str] = parts[1:]
        # Deck shorthand (/d2 -> /deck 2)
        if len(cmd) >= 2 and cmd[0] == 'd' and cmd[1:].isdigit():
            deck_num = cmd[1:]
            cmd = 'deck'
            args = [deck_num] + args
        if cmd in ('q', 'quit', 'exit'):
            return "EXIT"
        # Numeric stack access
        if cmd.isdigit():
            try:
                idx = int(cmd)
                return advanced_cmds.cmd_stack_get(self.session, args, idx)
            except Exception as exc:
                return f"ERROR: {exc}"
        func = self.commands.get(cmd)
        if func is None:
            # Suggest similar command names
            similar = [c for c in self.commands.keys() if cmd in c or c.startswith(cmd[:2])][:5]
            if similar:
                return f"ERROR: Unknown command /{cmd}. Did you mean: {', '.join('/' + s for s in similar)}?"
            return f"ERROR: Unknown command /{cmd}"
        try:
            result = func(self.session, args)
            return result or ""
        except Exception as exc:
            # Append help text if available
            error_msg = f"ERROR: {exc}"
            help_text = ''
            if hasattr(advanced_cmds, 'get_command_help'):
                try:
                    help_text = advanced_cmds.get_command_help(cmd)
                except Exception:
                    help_text = ''
            if help_text:
                error_msg += f"\n\n--- Help for /{cmd} ---\n{help_text}"
            return error_msg

    # ------------------------------------------------------------------
    # DSL execution
    # ------------------------------------------------------------------
    def run_mad_script(self) -> None:
        """Compile and run the contents of the script editor."""
        script_text = self.script.get('1.0', tk.END)
        for raw_line in script_text.splitlines():
            # Remove invisible separator prefix and whitespace
            line = raw_line.lstrip(self._LINE_PREFIX).rstrip()
            if not line:
                continue
            # Ensure line starts with '/'
            if not line.startswith('/'):
                line = '/' + line
            result = self._execute_command(line)
            if result == "EXIT":
                self.root.quit()
                return
            elif result:
                self.write(result + '\n')

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Enter the Tkinter main loop."""
        self.root.mainloop()


def main() -> None:
    """Entry point for the GUI when run as a module."""
    app = MadGui()
    app.run()


if __name__ == '__main__':
    main()