"""MDMA GUI Shell — Top-Level Window Manager.

The Shell is the top-level wx.Frame that owns the menu bar, the window
manager, and the Bridge instance.  It is the entry point for the modular
GUI and the event target for all Bridge-posted events.

The Shell manages the lifecycle of sub-windows (Generation, Mutation,
Effects, Synthesis, Arrangement, Mixing, Inspector) and provides a
menu for spawning and closing them.

Phase 1 implementation: Shell + Inspector window only.  The Inspector
provides a console mirror and object tree — enough to validate the
event system and Bridge without touching the existing monolith.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md

BUILD ID: gui_shell_v1.0_phase1
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# wxPython import guard
try:
    import wx
    _WX_AVAILABLE = True
except ImportError:
    _WX_AVAILABLE = False

if _WX_AVAILABLE:
    from .bridge import Bridge
    from .events import (
        EVT_OBJECT_CREATED,
        EVT_OBJECT_UPDATED,
        EVT_OBJECT_DELETED,
        EVT_OBJECT_RENAMED,
        EVT_STATUS_MESSAGE,
        EVT_COMMAND_EXECUTED,
    )


# ============================================================================
# THEME — matches the existing mdma_gui.py "Serum-ish" muted dark theme
# ============================================================================

THEME = {
    "BG_DARK": (30, 30, 35),
    "BG_MID": (42, 42, 48),
    "BG_LIGHT": (55, 55, 62),
    "TEXT": (220, 220, 225),
    "TEXT_DIM": (140, 140, 150),
    "ACCENT": (100, 180, 255),
    "SUCCESS": (100, 220, 120),
    "ERROR": (255, 100, 100),
    "WARNING": (255, 200, 80),
    "BORDER": (65, 65, 72),
}


# ============================================================================
# WINDOW REGISTRY — tracks which sub-windows are open
# ============================================================================

# Window type identifiers matching the spec
WINDOW_TYPES = [
    "generation",
    "mutation",
    "effects",
    "synthesis",
    "arrangement",
    "mixing",
    "inspector",
]


if _WX_AVAILABLE:

    class MDMAShell(wx.Frame):
        """Top-level frame and window manager for the modular MDMA GUI.

        Responsibilities:
        - Create and own the Bridge instance
        - Provide menu bar with Window menu for spawning sub-windows
        - Track open sub-windows
        - Serve as the wx event target for all Bridge events
        - Forward events to open sub-windows via the wx event system
        """

        def __init__(
            self,
            session: Any,
            parent: Optional[wx.Window] = None,
            title: str = "MDMA — Music Design Made Accessible",
            size: tuple[int, int] = (1200, 800),
        ) -> None:
            super().__init__(parent, title=title, size=size)

            # Apply theme
            self.SetBackgroundColour(wx.Colour(*THEME["BG_DARK"]))

            # Bridge — the single adapter for all engine operations
            # Use the session's ObjectRegistry so CLI and GUI share state
            self._registry = getattr(session, 'object_registry', None)
            if self._registry is None:
                from mdma_rebuild.core.registry import ObjectRegistry
                self._registry = ObjectRegistry()
            self.bridge = Bridge(
                session=session,
                registry=self._registry,
                event_target=self,
            )

            # Open sub-windows tracker: type_name -> window instance
            self._windows: dict[str, wx.Frame] = {}

            # Build the UI
            self._build_menu_bar()
            self._build_status_bar()

            # Bind Bridge events
            self.Bind(EVT_STATUS_MESSAGE, self._on_status_message)
            self.Bind(EVT_OBJECT_CREATED, self._on_object_event)
            self.Bind(EVT_OBJECT_UPDATED, self._on_object_event)
            self.Bind(EVT_OBJECT_DELETED, self._on_object_event)
            self.Bind(EVT_OBJECT_RENAMED, self._on_object_event)

            # Phase 1: auto-open the Inspector window
            self._open_window("inspector")

            logger.info("MDMAShell initialised")

        # ---- Menu bar -------------------------------------------------------

        def _build_menu_bar(self) -> None:
            """Create the menu bar with File and Window menus."""
            menu_bar = wx.MenuBar()

            # File menu
            file_menu = wx.Menu()
            file_menu.Append(wx.ID_EXIT, "E&xit\tCtrl+Q", "Close MDMA")
            self.Bind(wx.EVT_MENU, self._on_exit, id=wx.ID_EXIT)
            menu_bar.Append(file_menu, "&File")

            # Window menu — one item per sub-window type
            self._window_menu = wx.Menu()
            self._window_menu_ids: dict[int, str] = {}
            for wtype in WINDOW_TYPES:
                label = wtype.replace("_", " ").title()
                menu_id = wx.NewIdRef()
                self._window_menu.Append(menu_id, f"&{label}", f"Open {label} window")
                self._window_menu_ids[menu_id.GetId()] = wtype
                self.Bind(wx.EVT_MENU, self._on_window_menu, id=menu_id)
            menu_bar.Append(self._window_menu, "&Windows")

            self.SetMenuBar(menu_bar)

        def _build_status_bar(self) -> None:
            """Create the status bar for last-action display."""
            self._status_bar = self.CreateStatusBar(3)
            self._status_bar.SetStatusWidths([-3, -1, -1])
            self._status_bar.SetStatusText("Ready", 0)
            self._status_bar.SetStatusText("", 1)
            self._status_bar.SetStatusText("", 2)

        # ---- Window management ----------------------------------------------

        def _open_window(self, window_type: str) -> None:
            """Open a sub-window by type.  If already open, raise it."""
            if window_type in self._windows:
                win = self._windows[window_type]
                if win and not win.IsBeingDeleted():
                    win.Raise()
                    win.SetFocus()
                    return
                else:
                    del self._windows[window_type]

            if window_type == "inspector":
                win = self._create_inspector_window()
            elif window_type == "generation":
                from gui.windows.generation_window import GenerationWindow
                win = GenerationWindow(self, self.bridge, theme=THEME)
            elif window_type == "mutation":
                from gui.windows.mutation_window import MutationWindow
                win = MutationWindow(self, self.bridge, theme=THEME)
            elif window_type == "effects":
                from gui.windows.effects_window import EffectsWindow
                win = EffectsWindow(self, self.bridge, theme=THEME)
            elif window_type == "synthesis":
                from gui.windows.synthesis_window import SynthesisWindow
                win = SynthesisWindow(self, self.bridge, theme=THEME)
            elif window_type == "arrangement":
                from gui.windows.arrangement_window import ArrangementWindow
                win = ArrangementWindow(self, self.bridge, theme=THEME)
            elif window_type == "mixing":
                from gui.windows.mixing_window import MixingWindow
                win = MixingWindow(self, self.bridge, theme=THEME)
            else:
                label = window_type.replace("_", " ").title()
                win = wx.Frame(
                    self,
                    title=f"MDMA — {label} (Coming Soon)",
                    size=(800, 600),
                )
                win.SetBackgroundColour(wx.Colour(*THEME["BG_DARK"]))
                sizer = wx.BoxSizer(wx.VERTICAL)
                placeholder = wx.StaticText(
                    win,
                    label=f"{label} window will be implemented in a future phase.",
                )
                placeholder.SetForegroundColour(wx.Colour(*THEME["TEXT_DIM"]))
                sizer.Add(placeholder, 0, wx.ALL | wx.ALIGN_CENTER, 40)
                win.SetSizer(sizer)

            self._windows[window_type] = win
            win.Bind(wx.EVT_CLOSE, lambda evt, wt=window_type: self._on_window_close(evt, wt))
            win.Show()
            logger.info("Opened %s window", window_type)

        def _on_window_close(self, event: wx.CloseEvent, window_type: str) -> None:
            """Handle a sub-window being closed."""
            self._windows.pop(window_type, None)
            event.Skip()

        def _create_inspector_window(self) -> wx.Frame:
            """Create the Inspector & Console window (Phase 1).

            Phase 1 provides:
            - Console/CLI mirror panel with command input
            - Object tree panel (registry browser)
            - Status bar
            """
            win = wx.Frame(
                self,
                title="MDMA — Inspector & Console",
                size=(900, 700),
            )
            win.SetBackgroundColour(wx.Colour(*THEME["BG_DARK"]))

            # Main sizer with splitter
            splitter = wx.SplitterWindow(win, style=wx.SP_LIVE_UPDATE)
            splitter.SetMinimumPaneSize(200)

            # Left: Object tree
            tree_panel = wx.Panel(splitter)
            tree_panel.SetBackgroundColour(wx.Colour(*THEME["BG_MID"]))
            tree_sizer = wx.BoxSizer(wx.VERTICAL)

            tree_label = wx.StaticText(tree_panel, label="Object Registry")
            tree_label.SetForegroundColour(wx.Colour(*THEME["TEXT"]))
            tree_sizer.Add(tree_label, 0, wx.ALL, 5)

            tree_ctrl = wx.TreeCtrl(
                tree_panel,
                style=wx.TR_DEFAULT_STYLE | wx.TR_HIDE_ROOT,
                name="Object Tree",
            )
            tree_ctrl.SetBackgroundColour(wx.Colour(*THEME["BG_DARK"]))
            tree_ctrl.SetForegroundColour(wx.Colour(*THEME["TEXT"]))
            root = tree_ctrl.AddRoot("Registry")

            # Add type categories
            from mdma_rebuild.core.objects import OBJECT_TYPE_MAP
            for type_name in OBJECT_TYPE_MAP:
                tree_ctrl.AppendItem(root, type_name.replace("_", " ").title())

            tree_sizer.Add(tree_ctrl, 1, wx.EXPAND | wx.ALL, 2)
            tree_panel.SetSizer(tree_sizer)

            # Right: Console mirror
            console_panel = wx.Panel(splitter)
            console_panel.SetBackgroundColour(wx.Colour(*THEME["BG_MID"]))
            console_sizer = wx.BoxSizer(wx.VERTICAL)

            console_label = wx.StaticText(console_panel, label="Console")
            console_label.SetForegroundColour(wx.Colour(*THEME["TEXT"]))
            console_sizer.Add(console_label, 0, wx.ALL, 5)

            console_output = wx.TextCtrl(
                console_panel,
                style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
                name="Console Output",
            )
            console_output.SetBackgroundColour(wx.Colour(*THEME["BG_DARK"]))
            console_output.SetForegroundColour(wx.Colour(*THEME["TEXT"]))
            console_sizer.Add(console_output, 1, wx.EXPAND | wx.ALL, 2)

            # Command input
            input_sizer = wx.BoxSizer(wx.HORIZONTAL)
            cmd_label = wx.StaticText(console_panel, label="/")
            cmd_label.SetForegroundColour(wx.Colour(*THEME["ACCENT"]))
            input_sizer.Add(cmd_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)

            cmd_input = wx.TextCtrl(
                console_panel,
                style=wx.TE_PROCESS_ENTER,
                name="Command Input",
            )
            cmd_input.SetBackgroundColour(wx.Colour(*THEME["BG_LIGHT"]))
            cmd_input.SetForegroundColour(wx.Colour(*THEME["TEXT"]))
            input_sizer.Add(cmd_input, 1, wx.EXPAND | wx.ALL, 2)

            console_sizer.Add(input_sizer, 0, wx.EXPAND)
            console_panel.SetSizer(console_sizer)

            splitter.SplitVertically(tree_panel, console_panel, 300)

            # Store references for event handlers
            win._tree_ctrl = tree_ctrl
            win._console_output = console_output
            win._cmd_input = cmd_input

            # Bind command input
            cmd_input.Bind(wx.EVT_TEXT_ENTER, lambda evt: self._on_console_command(evt, win))

            # Accessibility
            cmd_input.SetHint("Type a CLI command and press Enter")
            tree_ctrl.GetAccessible()  # Ensure accessible object exists

            return win

        # ---- Event handlers -------------------------------------------------

        def _on_exit(self, event: wx.CommandEvent) -> None:
            """Handle File > Exit."""
            self.Close()

        def _on_window_menu(self, event: wx.CommandEvent) -> None:
            """Handle Window menu item selection."""
            wtype = self._window_menu_ids.get(event.GetId())
            if wtype:
                self._open_window(wtype)

        def _on_status_message(self, event: Any) -> None:
            """Handle StatusMessageEvent — update status bar and console."""
            message = getattr(event, "message", "")
            self._status_bar.SetStatusText(message, 0)

            # Also push to Inspector console if open
            inspector = self._windows.get("inspector")
            if inspector and hasattr(inspector, "_console_output"):
                inspector._console_output.AppendText(f"{message}\n")

        def _on_object_event(self, event: Any) -> None:
            """Handle object registry events — refresh Inspector tree."""
            obj_type = getattr(event, "obj_type", "")
            name = getattr(event, "name", "")
            self._status_bar.SetStatusText(f"{obj_type}: {name}", 1)

            # Live-update the Inspector object tree
            self._refresh_inspector_tree()

        def _refresh_inspector_tree(self) -> None:
            """Rebuild the Inspector's object tree from the registry."""
            inspector = self._windows.get("inspector")
            if not inspector or not hasattr(inspector, "_tree_ctrl"):
                return

            tree = inspector._tree_ctrl
            tree.DeleteAllItems()
            root = tree.AddRoot("Registry")

            from mdma_rebuild.core.objects import OBJECT_TYPE_MAP
            registry = self._registry

            for type_name in OBJECT_TYPE_MAP:
                objects = registry.list_objects(type_name)
                label = type_name.replace("_", " ").title()
                if objects:
                    label += f" ({len(objects)})"
                type_node = tree.AppendItem(root, label)

                for obj in objects:
                    # Show name and key details
                    detail = obj.name
                    if hasattr(obj, "genre") and obj.genre:
                        detail += f" — {obj.genre}"
                    elif hasattr(obj, "pattern_kind") and obj.pattern_kind:
                        detail += f" — {obj.pattern_kind}"
                    elif hasattr(obj, "duration_seconds") and obj.duration_seconds > 0:
                        detail += f" — {obj.duration_seconds:.2f}s"
                    tree.AppendItem(type_node, detail)

                if objects:
                    tree.Expand(type_node)

        def _on_console_command(self, event: Any, inspector_win: wx.Frame) -> None:
            """Handle command input from the Inspector console."""
            cmd_input = inspector_win._cmd_input
            command = cmd_input.GetValue().strip()
            if not command:
                return

            cmd_input.Clear()
            console = inspector_win._console_output
            console.AppendText(f"> /{command}\n")

            # Execute through bridge
            result = self.bridge.execute_command(f"/{command}")
            console.AppendText(f"{result}\n\n")


    def launch_shell(session: Any) -> None:
        """Launch the modular GUI shell.

        This is the entry point for the new modular GUI.  It can run
        alongside the existing mdma_gui.py during the transition.

        Parameters:
            session: The MDMA Session instance.
        """
        app = wx.App()
        shell = MDMAShell(session=session)
        shell.Show()
        app.MainLoop()

else:
    # Stubs for when wx is not available

    class MDMAShell:  # type: ignore[no-redef]
        """Placeholder when wxPython is not installed."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "wxPython is required for the MDMA GUI. "
                "Install it with: pip install wxPython"
            )

    def launch_shell(session: Any) -> None:
        """Placeholder launcher when wx is not available."""
        raise RuntimeError(
            "wxPython is required for the MDMA GUI. "
            "Install it with: pip install wxPython"
        )
