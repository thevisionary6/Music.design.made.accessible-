#!/usr/bin/env python3
"""MDMA - Music Design Made Accessible
Unified launcher with automatic interface detection.

Usage:
    python run_mdma.py              Auto-detect best available interface
    python run_mdma.py --repl       Force REPL mode (always available)
    python run_mdma.py --gui        Force wxPython GUI
    python run_mdma.py --tui        Force Textual TUI
    python run_mdma.py --help       Show this help

BUILD ID: launcher_v1.0
"""

import sys
import os
import argparse

# Ensure the project root is on the path
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# ── Dependency checks ──────────────────────────────────────────────────

def _check_core_deps():
    """Verify core dependencies are installed. Exit with helpful message if not."""
    missing = []
    for pkg, import_name in [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("soundfile", "soundfile"),
        ("sounddevice", "sounddevice"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("MDMA: Missing core dependencies:")
        for pkg in missing:
            print(f"  - {pkg}")
        print(f"\nInstall with:  pip install {' '.join(missing)}")
        print("Or install all:  pip install -r requirements.txt")
        sys.exit(1)


def _has_wx():
    """Check if wxPython is available."""
    try:
        import wx  # noqa: F401
        return True
    except ImportError:
        return False


def _has_textual():
    """Check if Textual is available."""
    try:
        import textual  # noqa: F401
        return True
    except ImportError:
        return False


# ── Launchers ──────────────────────────────────────────────────────────

def launch_repl():
    """Launch the REPL (bmdma.py)."""
    print("MDMA: Starting REPL...")
    # Import and run bmdma's main loop
    import bmdma
    if hasattr(bmdma, 'main'):
        bmdma.main()
    else:
        # Fallback: bmdma runs on import via if __name__ == '__main__' guard,
        # so we call its REPL loop directly
        from mdma_rebuild.core.session import Session
        session = Session()
        cmd_table = bmdma.build_command_table()
        bmdma.repl(session, cmd_table)


def launch_gui():
    """Launch the wxPython GUI (mdma_gui.py)."""
    if not _has_wx():
        print("MDMA: wxPython is not installed.")
        print("Install with:  pip install wxPython")
        print("\nFalling back to REPL...")
        launch_repl()
        return
    print("MDMA: Starting GUI...")
    import mdma_gui
    if hasattr(mdma_gui, 'main'):
        mdma_gui.main()
    else:
        app = mdma_gui.wx.App()
        frame = mdma_gui.MDMAFrame(None)
        frame.Show()
        app.MainLoop()


def launch_tui():
    """Launch the Textual TUI (mad_tui.py)."""
    if not _has_textual():
        print("MDMA: Textual is not installed.")
        print("Install with:  pip install textual")
        print("\nFalling back to REPL...")
        launch_repl()
        return
    print("MDMA: Starting TUI...")
    import mad_tui
    mad_tui.main()


def auto_detect():
    """Pick the best available interface automatically.

    Priority: GUI > TUI > REPL
    GUI requires wxPython + a display server.
    TUI requires Textual.
    REPL always works.
    """
    # Check for a display (X11/Wayland) — GUI needs one
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    if has_display and _has_wx():
        launch_gui()
    elif _has_textual():
        launch_tui()
    else:
        launch_repl()


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="mdma",
        description="MDMA - Music Design Made Accessible",
        epilog=(
            "Interfaces:\n"
            "  REPL  Terminal command line (always available)\n"
            "  GUI   wxPython visual interface (pip install wxPython)\n"
            "  TUI   Textual terminal UI (pip install textual)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--repl", action="store_true", help="Launch REPL (terminal)")
    group.add_argument("--gui", action="store_true", help="Launch wxPython GUI")
    group.add_argument("--tui", action="store_true", help="Launch Textual TUI")
    parser.add_argument(
        "--check", action="store_true",
        help="Check dependencies and available interfaces, then exit",
    )
    args = parser.parse_args()

    # Always verify core deps first
    _check_core_deps()

    if args.check:
        _print_status()
        return

    if args.repl:
        launch_repl()
    elif args.gui:
        launch_gui()
    elif args.tui:
        launch_tui()
    else:
        auto_detect()


def _print_status():
    """Print dependency and interface availability status."""
    print("MDMA - Music Design Made Accessible")
    print("=" * 40)
    print()

    # Core deps
    print("Core dependencies:")
    for pkg, import_name in [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("soundfile", "soundfile"),
        ("sounddevice", "sounddevice"),
        ("pydub", "pydub"),
    ]:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  {pkg:20s} {ver}")
        except ImportError:
            print(f"  {pkg:20s} NOT INSTALLED")

    print()
    print("Interfaces:")

    # REPL
    print(f"  {'REPL':20s} always available")

    # GUI
    if _has_wx():
        import wx
        print(f"  {'GUI (wxPython)':20s} {wx.__version__}")
    else:
        print(f"  {'GUI (wxPython)':20s} NOT INSTALLED  (pip install wxPython)")

    # TUI
    if _has_textual():
        import textual
        ver = getattr(textual, "__version__", "installed")
        print(f"  {'TUI (Textual)':20s} {ver}")
    else:
        print(f"  {'TUI (Textual)':20s} NOT INSTALLED  (pip install textual)")

    # Display
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    print()
    print(f"Display server: {'detected' if has_display else 'not detected (GUI unavailable)'}")

    # Optional extras
    print()
    print("Optional packages:")
    for pkg, import_name in [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("librosa", "librosa"),
        ("demucs", "demucs"),
        ("yt-dlp", "yt_dlp"),
    ]:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  {pkg:20s} {ver}")
        except ImportError:
            print(f"  {pkg:20s} not installed")


if __name__ == "__main__":
    main()
