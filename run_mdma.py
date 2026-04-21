#!/usr/bin/env python3
"""MDMA - Music Design Made Accessible
Unified launcher.

Usage:
    python run_mdma.py              Launch the REPL (default)
    python run_mdma.py --repl       Same as no-args
    python run_mdma.py --gui        DEPRECATED wxPython GUI
    python run_mdma.py --tui        DEPRECATED Textual TUI
    python run_mdma.py --help       Show this help

The REPL is the supported entry point. ``--gui`` and ``--tui``
remain so existing workflows don't break, but they print a
DeprecationWarning and are not accepting new features. New V2
backend commands (``/patn``, ``/loadfx``, ``/listfx``) are only
wired into the REPL.

BUILD ID: launcher_v2.0
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


def _warn_deprecated_interface(name: str) -> None:
    """Print + warn when a deprecated interface is requested."""
    import warnings
    msg = (
        f"MDMA: the {name} interface is deprecated. "
        "The REPL is the supported entry point; V2 backend commands "
        "(/patn, /loadfx, /listfx) are only wired there."
    )
    print(f"[MDMA] DEPRECATION: {msg}")
    warnings.warn(msg, DeprecationWarning, stacklevel=2)


def launch_gui():
    """Launch the wxPython GUI (DEPRECATED)."""
    _warn_deprecated_interface("GUI")
    if not _has_wx():
        print("MDMA: wxPython is not installed.")
        print("Install with:  pip install wxPython")
        print("\nFalling back to REPL...")
        launch_repl()
        return
    print("MDMA: Starting GUI (deprecated)...")
    import mdma_gui
    if hasattr(mdma_gui, 'main'):
        mdma_gui.main()
    else:
        app = mdma_gui.wx.App()
        frame = mdma_gui.MDMAFrame(None)
        frame.Show()
        app.MainLoop()


def launch_tui():
    """Launch the Textual TUI (DEPRECATED)."""
    _warn_deprecated_interface("TUI")
    if not _has_textual():
        print("MDMA: Textual is not installed.")
        print("Install with:  pip install textual")
        print("\nFalling back to REPL...")
        launch_repl()
        return
    print("MDMA: Starting TUI (deprecated)...")
    import mad_tui
    mad_tui.main()


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="mdma",
        description="MDMA - Music Design Made Accessible",
        epilog=(
            "Interfaces:\n"
            "  REPL  Terminal command line (default, supported)\n"
            "  GUI   DEPRECATED wxPython visual interface\n"
            "  TUI   DEPRECATED Textual terminal UI\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--repl", action="store_true",
        help="Launch REPL (default; flag kept for explicitness)",
    )
    group.add_argument(
        "--gui", action="store_true",
        help="DEPRECATED: launch wxPython GUI",
    )
    group.add_argument(
        "--tui", action="store_true",
        help="DEPRECATED: launch Textual TUI",
    )
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

    if args.gui:
        launch_gui()
    elif args.tui:
        launch_tui()
    else:
        # Default and --repl both go straight to the REPL now; the
        # previous auto-detect preferred GUI > TUI > REPL, which
        # routed around the supported interface.
        launch_repl()


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
    print(f"  {'REPL':20s} supported; default entry point")

    if _has_wx():
        import wx
        print(f"  {'GUI (wxPython)':20s} DEPRECATED; {wx.__version__} installed")
    else:
        print(f"  {'GUI (wxPython)':20s} DEPRECATED; not installed")

    if _has_textual():
        import textual
        ver = getattr(textual, "__version__", "installed")
        print(f"  {'TUI (Textual)':20s} DEPRECATED; {ver} installed")
    else:
        print(f"  {'TUI (Textual)':20s} DEPRECATED; not installed")

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
