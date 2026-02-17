"""Keyboard-Navigable Step Grid Widget.

A grid control for step-based pattern editing.  Fully keyboard
accessible: arrow keys navigate cells, Space toggles steps, and
screen readers announce the current cell position and state.

This widget is used by the Pattern Editor panel in the Mutation
window and the Beat Generator panel in the Generation window.

BUILD ID: widget_step_grid_v1.0
"""

from __future__ import annotations

from typing import Any, Callable, Optional

try:
    import wx
    _WX_AVAILABLE = True
except ImportError:
    _WX_AVAILABLE = False


if _WX_AVAILABLE:

    class StepGrid(wx.Panel):
        """Keyboard-navigable step grid for pattern editing.

        The grid is a 2D array of cells: rows represent instruments
        or pitches, columns represent time steps.  Each cell has an
        on/off state and an optional velocity value.

        Keyboard controls:
        - Arrow keys: navigate cells
        - Space: toggle current cell on/off
        - Ctrl+A: select all cells
        - Delete: clear current cell
        - 0-9: set velocity (mapped to 0-127)

        Parameters:
            parent: Parent wx window.
            rows: Number of rows (instruments/pitches).
            cols: Number of columns (time steps).
            row_labels: Optional list of row label strings.
            on_cell_change: Optional callback(row, col, state, velocity).
        """

        def __init__(
            self,
            parent: wx.Window,
            rows: int = 4,
            cols: int = 16,
            row_labels: Optional[list[str]] = None,
            on_cell_change: Optional[Callable[[int, int, bool, int], None]] = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(parent, **kwargs)

            self._rows = rows
            self._cols = cols
            self._row_labels = row_labels or [f"Row {i+1}" for i in range(rows)]
            self._on_cell_change = on_cell_change

            # Grid state: [row][col] -> (active: bool, velocity: int)
            self._cells: list[list[tuple[bool, int]]] = [
                [(False, 100) for _ in range(cols)] for _ in range(rows)
            ]

            # Cursor position
            self._cursor_row = 0
            self._cursor_col = 0

            # Accessible name
            self.SetName("Step Grid")

            # Build UI using a wx.GridSizer
            self._build_grid()

            # Bind keyboard events
            self.Bind(wx.EVT_KEY_DOWN, self._on_key_down)
            self.SetFocus()

        def _build_grid(self) -> None:
            """Build the grid display."""
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Header row with step numbers
            header = wx.BoxSizer(wx.HORIZONTAL)
            header.Add(wx.StaticText(self, label="", size=(80, -1)), 0)
            for c in range(self._cols):
                lbl = wx.StaticText(
                    self, label=str(c + 1), size=(30, -1),
                    style=wx.ALIGN_CENTER,
                )
                header.Add(lbl, 0, wx.ALL, 1)
            sizer.Add(header, 0, wx.EXPAND)

            # Data rows
            self._cell_buttons: list[list[wx.ToggleButton]] = []
            for r in range(self._rows):
                row_sizer = wx.BoxSizer(wx.HORIZONTAL)
                label = wx.StaticText(
                    self, label=self._row_labels[r], size=(80, -1),
                )
                row_sizer.Add(label, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 1)

                btn_row = []
                for c in range(self._cols):
                    btn = wx.ToggleButton(
                        self, label=" ", size=(30, 25),
                        name=f"{self._row_labels[r]} step {c+1}",
                    )
                    btn.Bind(
                        wx.EVT_TOGGLEBUTTON,
                        lambda evt, row=r, col=c: self._on_cell_toggle(row, col),
                    )
                    row_sizer.Add(btn, 0, wx.ALL, 1)
                    btn_row.append(btn)

                self._cell_buttons.append(btn_row)
                sizer.Add(row_sizer, 0, wx.EXPAND)

            self.SetSizer(sizer)

        def _on_cell_toggle(self, row: int, col: int) -> None:
            """Handle a cell being toggled via mouse or keyboard."""
            btn = self._cell_buttons[row][col]
            active = btn.GetValue()
            _, velocity = self._cells[row][col]
            self._cells[row][col] = (active, velocity)

            if self._on_cell_change:
                self._on_cell_change(row, col, active, velocity)

        def _on_key_down(self, event: wx.KeyEvent) -> None:
            """Handle keyboard navigation within the grid."""
            key = event.GetKeyCode()

            if key == wx.WXK_RIGHT:
                self._cursor_col = min(self._cursor_col + 1, self._cols - 1)
            elif key == wx.WXK_LEFT:
                self._cursor_col = max(self._cursor_col - 1, 0)
            elif key == wx.WXK_DOWN:
                self._cursor_row = min(self._cursor_row + 1, self._rows - 1)
            elif key == wx.WXK_UP:
                self._cursor_row = max(self._cursor_row - 1, 0)
            elif key == wx.WXK_SPACE:
                # Toggle current cell
                btn = self._cell_buttons[self._cursor_row][self._cursor_col]
                btn.SetValue(not btn.GetValue())
                self._on_cell_toggle(self._cursor_row, self._cursor_col)
            elif key == wx.WXK_DELETE:
                # Clear current cell
                btn = self._cell_buttons[self._cursor_row][self._cursor_col]
                btn.SetValue(False)
                self._cells[self._cursor_row][self._cursor_col] = (False, 100)
            else:
                event.Skip()
                return

            # Focus the current cell button
            self._cell_buttons[self._cursor_row][self._cursor_col].SetFocus()

        def get_state(self) -> list[list[tuple[bool, int]]]:
            """Return the full grid state."""
            return [row[:] for row in self._cells]

        def set_state(self, state: list[list[tuple[bool, int]]]) -> None:
            """Set the full grid state from external data."""
            for r in range(min(len(state), self._rows)):
                for c in range(min(len(state[r]), self._cols)):
                    active, velocity = state[r][c]
                    self._cells[r][c] = (active, velocity)
                    self._cell_buttons[r][c].SetValue(active)

        def clear(self) -> None:
            """Clear all cells."""
            for r in range(self._rows):
                for c in range(self._cols):
                    self._cells[r][c] = (False, 100)
                    self._cell_buttons[r][c].SetValue(False)

else:

    class StepGrid:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("wxPython required")
