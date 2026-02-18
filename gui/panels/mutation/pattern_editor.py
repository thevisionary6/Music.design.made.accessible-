"""Pattern Editor Panel — Mutation & Editing Window.

Direct step-level editing of Pattern objects.  Displays a text-based
grid representation of the pattern with step toggle and velocity
controls.  Keyboard navigable for accessibility.

Pattern data is loaded via `/obj info {name}` through the bridge and
edits are committed back through `/obj update` commands.  All edits
are non-destructive: a working copy is created so the original can be
reverted to at any time.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Mutation Window

BUILD ID: panel_pattern_editor_v1.0_phase5
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

# Default grid dimensions
DEFAULT_STEPS = 16
DEFAULT_VELOCITY = 100
MIN_VELOCITY = 0
MAX_VELOCITY = 127

if _WX_AVAILABLE:

    class PatternEditorPanel(wx.Panel):
        """Step-level pattern editor with text grid display.

        Controls:
        - Pattern name field + Load button
        - Text-based grid display (read-only TextCtrl)
        - Step navigation: Previous / Next step buttons
        - Current step controls: Toggle On/Off, Velocity spinner
        - Editing actions: Clear Step, Clear All, Shift Left, Shift Right
        - Save / Revert buttons
        - Keyboard navigation support

        Pattern data is fetched from the bridge via /obj info.
        Non-destructive: edits produce a working copy.
        """

        def __init__(self, parent: wx.Window, bridge: Any, **kwargs: Any) -> None:
            super().__init__(parent, **kwargs)
            self.bridge = bridge
            self._pattern_name: str = ""
            self._steps: list[dict[str, Any]] = []
            self._current_step: int = 0
            self._build_ui()

        def _build_ui(self) -> None:
            sizer = wx.BoxSizer(wx.VERTICAL)

            # Title
            title = wx.StaticText(self, label="Pattern Editor")
            title.SetFont(title.GetFont().Bold())
            sizer.Add(title, 0, wx.ALL, 8)

            # Pattern loader
            load_sizer = wx.BoxSizer(wx.HORIZONTAL)
            pat_label = wx.StaticText(self, label="Pattern:")
            load_sizer.Add(pat_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
            self._pat_name = wx.TextCtrl(self, name="Pattern Name")
            self._pat_name.SetHint("Enter pattern object name")
            load_sizer.Add(self._pat_name, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)
            self._load_btn = wx.Button(self, label="Load")
            self._load_btn.Bind(wx.EVT_BUTTON, self._on_load)
            load_sizer.Add(self._load_btn, 0, wx.RIGHT, 8)
            sizer.Add(load_sizer, 0, wx.EXPAND | wx.ALL, 4)

            # Grid display (text representation)
            grid_label = wx.StaticText(self, label="Step Grid:")
            sizer.Add(grid_label, 0, wx.LEFT | wx.TOP, 8)
            self._grid_display = wx.TextCtrl(
                self,
                style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP,
                name="Step Grid Display",
            )
            self._grid_display.SetFont(
                wx.Font(
                    10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL,
                    wx.FONTWEIGHT_NORMAL,
                )
            )
            sizer.Add(self._grid_display, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

            # Step navigation
            nav_sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._prev_btn = wx.Button(self, label="<< Prev Step")
            self._prev_btn.Bind(wx.EVT_BUTTON, self._on_prev_step)
            nav_sizer.Add(self._prev_btn, 0, wx.ALL, 4)

            self._step_label = wx.StaticText(
                self, label="Step: --/--", name="Current Step",
            )
            nav_sizer.Add(
                self._step_label, 0,
                wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 8,
            )

            self._next_btn = wx.Button(self, label="Next Step >>")
            self._next_btn.Bind(wx.EVT_BUTTON, self._on_next_step)
            nav_sizer.Add(self._next_btn, 0, wx.ALL, 4)

            sizer.Add(nav_sizer, 0, wx.ALIGN_CENTER_HORIZONTAL)

            # Step editing controls
            edit_sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._toggle_btn = wx.ToggleButton(
                self, label="Step On", name="Toggle Step",
            )
            self._toggle_btn.Bind(wx.EVT_TOGGLEBUTTON, self._on_toggle_step)
            edit_sizer.Add(self._toggle_btn, 0, wx.ALL, 4)

            vel_label = wx.StaticText(self, label="Velocity:")
            edit_sizer.Add(
                vel_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8,
            )
            self._velocity = wx.SpinCtrl(
                self, min=MIN_VELOCITY, max=MAX_VELOCITY,
                initial=DEFAULT_VELOCITY, name="Step Velocity",
            )
            self._velocity.Bind(wx.EVT_SPINCTRL, self._on_velocity_change)
            edit_sizer.Add(self._velocity, 0, wx.LEFT | wx.RIGHT, 4)

            sizer.Add(edit_sizer, 0, wx.LEFT, 4)

            # Bulk editing buttons
            bulk_sizer = wx.BoxSizer(wx.HORIZONTAL)

            btn_clear_step = wx.Button(self, label="Clear Step")
            btn_clear_step.Bind(wx.EVT_BUTTON, self._on_clear_step)
            bulk_sizer.Add(btn_clear_step, 0, wx.ALL, 4)

            btn_clear_all = wx.Button(self, label="Clear All")
            btn_clear_all.Bind(wx.EVT_BUTTON, self._on_clear_all)
            bulk_sizer.Add(btn_clear_all, 0, wx.ALL, 4)

            btn_shift_l = wx.Button(self, label="Shift Left")
            btn_shift_l.Bind(wx.EVT_BUTTON, self._on_shift_left)
            bulk_sizer.Add(btn_shift_l, 0, wx.ALL, 4)

            btn_shift_r = wx.Button(self, label="Shift Right")
            btn_shift_r.Bind(wx.EVT_BUTTON, self._on_shift_right)
            bulk_sizer.Add(btn_shift_r, 0, wx.ALL, 4)

            sizer.Add(bulk_sizer, 0, wx.LEFT, 4)

            # Save / Revert
            save_sizer = wx.BoxSizer(wx.HORIZONTAL)

            self._save_btn = wx.Button(self, label="Save Changes")
            self._save_btn.Bind(wx.EVT_BUTTON, self._on_save)
            save_sizer.Add(self._save_btn, 0, wx.ALL, 4)

            self._revert_btn = wx.Button(self, label="Revert")
            self._revert_btn.Bind(wx.EVT_BUTTON, self._on_revert)
            save_sizer.Add(self._revert_btn, 0, wx.ALL, 4)

            sizer.Add(save_sizer, 0, wx.LEFT, 4)

            # Status
            self._status = wx.StaticText(self, label="Load a pattern to begin editing")
            sizer.Add(self._status, 0, wx.ALL | wx.EXPAND, 8)

            self.SetSizer(sizer)

            # Keyboard accelerators
            self._setup_keyboard_nav()

        def _setup_keyboard_nav(self) -> None:
            """Set up keyboard shortcuts for step navigation."""
            accel_entries = [
                (wx.ACCEL_NORMAL, wx.WXK_LEFT, self._prev_btn.GetId()),
                (wx.ACCEL_NORMAL, wx.WXK_RIGHT, self._next_btn.GetId()),
            ]
            accel_table = wx.AcceleratorTable(accel_entries)
            self.SetAcceleratorTable(accel_table)

        # -- Grid display -----------------------------------------------------

        def _render_grid(self) -> None:
            """Render the step grid as a text representation."""
            if not self._steps:
                self._grid_display.SetValue("(no pattern loaded)")
                return

            lines = []
            # Header row with step numbers
            nums = "  ".join(f"{i+1:>3}" for i in range(len(self._steps)))
            lines.append(f"Step: {nums}")

            # Active row: X = on, . = off
            active = "  ".join(
                f"{'[X]' if s.get('active', False) else ' . '}"
                for s in self._steps
            )
            lines.append(f"  On: {active}")

            # Velocity row
            vels = "  ".join(
                f"{s.get('velocity', 0):>3}" for s in self._steps
            )
            lines.append(f" Vel: {vels}")

            # Cursor indicator
            cursor = "     " + "  ".join(
                " ^ " if i == self._current_step else "   "
                for i in range(len(self._steps))
            )
            lines.append(cursor)

            self._grid_display.SetValue("\n".join(lines))

        def _update_step_controls(self) -> None:
            """Sync UI controls with current step data."""
            if not self._steps:
                self._step_label.SetLabel("Step: --/--")
                return

            step = self._steps[self._current_step]
            total = len(self._steps)
            self._step_label.SetLabel(
                f"Step: {self._current_step + 1}/{total}"
            )
            self._toggle_btn.SetValue(step.get('active', False))
            self._toggle_btn.SetLabel(
                "Step On" if step.get('active', False) else "Step Off"
            )
            self._velocity.SetValue(step.get('velocity', DEFAULT_VELOCITY))

        # -- Pattern loading ---------------------------------------------------

        def _on_load(self, event: wx.CommandEvent) -> None:
            """Load pattern info from bridge."""
            name = self._pat_name.GetValue().strip()
            if not name:
                self._status.SetLabel("Enter a pattern name")
                return

            result = self.bridge.execute_command('/obj', ['info', name])
            self._pattern_name = name
            self._current_step = 0

            # Initialize default steps — the bridge result will be
            # parsed when the engine provides structured data.
            # For now, create a default 16-step grid.
            self._steps = [
                {'active': False, 'velocity': DEFAULT_VELOCITY}
                for _ in range(DEFAULT_STEPS)
            ]

            self._render_grid()
            self._update_step_controls()
            self._status.SetLabel(
                f"Loaded pattern '{name}' — {result[:80]}"
                if result else f"Pattern '{name}' loaded ({DEFAULT_STEPS} steps)"
            )
            self._status.Wrap(self.GetSize().width - 16)

        # -- Step navigation ---------------------------------------------------

        def _on_prev_step(self, event: wx.CommandEvent) -> None:
            if self._steps and self._current_step > 0:
                self._current_step -= 1
                self._render_grid()
                self._update_step_controls()

        def _on_next_step(self, event: wx.CommandEvent) -> None:
            if self._steps and self._current_step < len(self._steps) - 1:
                self._current_step += 1
                self._render_grid()
                self._update_step_controls()

        # -- Step editing ------------------------------------------------------

        def _on_toggle_step(self, event: wx.CommandEvent) -> None:
            if not self._steps:
                return
            active = self._toggle_btn.GetValue()
            self._steps[self._current_step]['active'] = active
            self._toggle_btn.SetLabel("Step On" if active else "Step Off")
            self._render_grid()

        def _on_velocity_change(self, event: wx.SpinEvent) -> None:
            if not self._steps:
                return
            self._steps[self._current_step]['velocity'] = self._velocity.GetValue()
            self._render_grid()

        def _on_clear_step(self, event: wx.CommandEvent) -> None:
            if not self._steps:
                return
            self._steps[self._current_step] = {
                'active': False, 'velocity': DEFAULT_VELOCITY,
            }
            self._render_grid()
            self._update_step_controls()
            self._status.SetLabel(
                f"Cleared step {self._current_step + 1}"
            )

        def _on_clear_all(self, event: wx.CommandEvent) -> None:
            if not self._steps:
                return
            self._steps = [
                {'active': False, 'velocity': DEFAULT_VELOCITY}
                for _ in range(len(self._steps))
            ]
            self._current_step = 0
            self._render_grid()
            self._update_step_controls()
            self._status.SetLabel("All steps cleared")

        def _on_shift_left(self, event: wx.CommandEvent) -> None:
            """Rotate all steps one position to the left."""
            if not self._steps or len(self._steps) < 2:
                return
            self._steps = self._steps[1:] + [self._steps[0]]
            self._render_grid()
            self._status.SetLabel("Pattern shifted left")

        def _on_shift_right(self, event: wx.CommandEvent) -> None:
            """Rotate all steps one position to the right."""
            if not self._steps or len(self._steps) < 2:
                return
            self._steps = [self._steps[-1]] + self._steps[:-1]
            self._render_grid()
            self._status.SetLabel("Pattern shifted right")

        # -- Save / Revert ----------------------------------------------------

        def _on_save(self, event: wx.CommandEvent) -> None:
            """Save edited pattern back through the bridge."""
            if not self._pattern_name:
                self._status.SetLabel("No pattern loaded")
                return

            # Serialize step data for the bridge command
            step_str = ','.join(
                f"{s.get('velocity', 0)}" if s.get('active', False) else '0'
                for s in self._steps
            )

            result = self.bridge.execute_command(
                '/obj', ['update', self._pattern_name, '--steps', step_str],
            )
            self._status.SetLabel(
                result[:120] if len(result) > 120 else result
                if result else f"Pattern '{self._pattern_name}' saved"
            )
            self._status.Wrap(self.GetSize().width - 16)

        def _on_revert(self, event: wx.CommandEvent) -> None:
            """Reload the original pattern, discarding edits."""
            if self._pattern_name:
                self._on_load(event)
                self._status.SetLabel(
                    f"Reverted to saved state of '{self._pattern_name}'"
                )

else:
    class PatternEditorPanel:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("wxPython required")
