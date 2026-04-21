#!/usr/bin/env python3
"""Environment installer for MDMA.

Usage::

    python scripts/setup_env.py                 # core + V2 backend
    python scripts/setup_env.py --profile full  # everything
    python scripts/setup_env.py --profile ai    # core + V2 + AI
    python scripts/setup_env.py --list          # show available profiles
    python scripts/setup_env.py --dry-run       # print the pip commands without running them

Run this once inside the Python environment you want to use — either
the system interpreter, a `venv`, or a Conda environment. The
installer does not create venvs for you; pick your environment
first, then run the script inside it.

The authoritative dependency list is :mod:`pyproject.toml`; the
profiles below map to ``[project.optional-dependencies]`` extras.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

#: Dependency profiles. Each name maps to the list of ``pyproject.toml``
#: extras to install. ``default`` is what you get when ``--profile`` is
#: omitted.
PROFILES: dict[str, list[str]] = {
    # Base + V2 backend. This is the minimum that exercises the
    # SignalFlow-native Pattern/Scheduler path and /patn.
    "default": ["v2"],

    # Everything except the giant ML wheels. Useful for full UI work
    # without needing a GPU.
    "studio": ["v2", "gui", "tui", "streaming"],

    # Everything that involves ML/AI. Pulls torch, transformers,
    # diffusers, librosa. On CUDA hosts, install torch via the right
    # CUDA index URL first — see README / setup notes.
    "ai": ["v2", "ai"],

    # Stem-separation toolkit. Heavy: pulls demucs + spleeter.
    "stems": ["v2", "stems"],

    # Kitchen sink. Matches ``pip install mdma[all]``.
    "full": ["v2", "gui", "tui", "ai", "stems", "streaming"],

    # Dev-only: core + V2 + nothing else. Fastest install for test
    # runs.
    "dev": ["v2"],

    # Minimum viable install: legacy numpy-buffer path only, no V2.
    "legacy": [],
}

#: Optional Python tools that are nice-to-have for development but
#: live outside pyproject.toml's extras. Installed on ``--with-dev``.
DEV_TOOLS: list[str] = [
    "pytest>=7.0",
    "pip-tools>=7.0",
]


# ---------------------------------------------------------------------------
# Pip driver
# ---------------------------------------------------------------------------

def _pip(args: list[str], *, dry_run: bool) -> None:
    """Run ``python -m pip <args>``. Honours ``dry_run``."""
    cmd = [sys.executable, "-m", "pip", *args]
    print("  $", " ".join(cmd))
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise SystemExit(
            f"pip exited with code {result.returncode} while running: "
            + " ".join(cmd)
        )


def _install_profile(
    profile: str,
    *,
    with_dev: bool,
    dry_run: bool,
    upgrade: bool,
) -> None:
    extras = PROFILES[profile]
    extras_spec = f"[{','.join(extras)}]" if extras else ""
    target = f".{extras_spec}"
    print(f"\n== installing profile {profile!r} ==")
    pip_args = ["install", "-e", target]
    if upgrade:
        pip_args.append("--upgrade")
    _pip(pip_args, dry_run=dry_run)

    if with_dev and DEV_TOOLS:
        print("\n== installing dev tools ==")
        _pip(["install", *DEV_TOOLS], dry_run=dry_run)


# ---------------------------------------------------------------------------
# OS-specific notes
# ---------------------------------------------------------------------------

def _os_notes() -> None:
    system = platform.system()
    print("\n== OS-specific notes ==")
    if system == "Linux":
        print(
            "  Linux: python-rtmidi needs libasound and libjack headers.\n"
            "    apt:   sudo apt-get install -y libasound2-dev libjack-jackd2-dev\n"
            "    dnf:   sudo dnf install -y alsa-lib-devel jack-audio-connection-kit-devel\n"
            "  signalflow needs a working ALSA/Pulse output for live\n"
            "  playback. Headless CI skips the audio-integration tests\n"
            "  unless you set MDMA_AUDIO_TESTS=1."
        )
    elif system == "Darwin":
        print(
            "  macOS: portaudio (for sounddevice) comes via homebrew.\n"
            "    brew install portaudio"
        )
    elif system == "Windows":
        print(
            "  Windows: all dependencies install from wheels.\n"
            "  NVDA users: the KeyboardController is focus-scoped and does\n"
            "  NOT install a global keyboard hook — MDMA intentionally\n"
            "  never grabs keystrokes system-wide."
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

SMOKE_IMPORTS = [
    "numpy",
    "scipy",
    "soundfile",
    "sounddevice",
    "mdma_rebuild",
    "mdma_rebuild.backend",
    "mdma_rebuild.backend.pattern",
]

OPTIONAL_IMPORTS = {
    "signalflow": "v2",
    "mido": "v2",
    "pythonosc": "v2",
    "torch": "ai",
    "transformers": "ai",
    "librosa": "ai",
    "demucs": "stems",
    "yt_dlp": "streaming",
    "wx": "gui",
    "textual": "tui",
}


def _smoke_test() -> None:
    print("\n== smoke test ==")
    ok: list[str] = []
    missing: list[tuple[str, str]] = []
    for name in SMOKE_IMPORTS:
        try:
            __import__(name)
            ok.append(name)
        except Exception as exc:
            print(f"  FAIL: required module {name!r} did not import: {exc}")
            raise SystemExit(1)
    for name, profile in OPTIONAL_IMPORTS.items():
        try:
            __import__(name)
            ok.append(name)
        except Exception:
            missing.append((name, profile))
    print(f"  OK   : {len(ok)} modules imported.")
    if missing:
        print("  note : optional modules not installed:")
        for name, profile in missing:
            print(f"           {name}  (pulled in by profile '{profile}')")

    # Readline compat status — important for the REPL on Windows where
    # stdlib readline is absent and pyreadline3 carries the load.
    try:
        from mdma_rebuild.core.readline_compat import platform_summary
        print("  " + platform_summary())
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"  readline: shim import failed: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        "-p",
        choices=sorted(PROFILES),
        default="default",
        help="Which dependency profile to install (default: %(default)s).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available profiles and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the pip commands that would run, don't execute them.",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Pass --upgrade to pip so existing packages get bumped.",
    )
    parser.add_argument(
        "--with-dev",
        action="store_true",
        help="Also install dev tools (pytest, pip-tools).",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the post-install smoke test.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list:
        print("Available profiles:")
        for name, extras in PROFILES.items():
            rendered = (
                f"pip install .[{','.join(extras)}]" if extras else "pip install ."
            )
            print(f"  {name:<8} -> {rendered}")
        return 0

    if sys.version_info < (3, 9):
        raise SystemExit(
            f"MDMA requires Python 3.9+, but this interpreter is "
            f"{sys.version.split()[0]}."
        )

    print(f"MDMA setup — Python {sys.version.split()[0]} on {platform.system()}")
    print(f"Repo root: {REPO_ROOT}")

    _install_profile(
        args.profile,
        with_dev=args.with_dev,
        dry_run=args.dry_run,
        upgrade=args.upgrade,
    )
    _os_notes()

    if not args.dry_run and not args.skip_smoke:
        _smoke_test()

    print("\nDone. Next steps:")
    print("  python -m unittest discover tests       # run unit tests")
    print("  python run_mdma.py                      # start the REPL")
    print("  /patn 60,0.25 64,0.25 67,0.5            # play a Pattern via V2")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
