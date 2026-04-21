"""Cross-platform readline shim.

On Linux and macOS the stdlib ``readline`` module is the right answer;
its libedit-vs-GNU differences are already handled downstream. On
Windows, the stdlib module isn't available — this shim falls back to
:mod:`pyreadline3`, a pure-Python fork maintained specifically to
keep the Python REPL usable on Windows for screen-reader users.

Exports:

- :data:`readline` — the chosen module, or ``None`` if neither is
  available.
- :data:`HAS_READLINE` — ``True`` when the shim found something
  usable. Callers branch on this so the REPL still runs (without
  completion / history) in minimal environments.
- :data:`IS_PYREADLINE3` — ``True`` when the backing implementation
  is pyreadline3 (Windows). Handy for skipping GNU/libedit-only
  bindings whose syntax pyreadline3 can't parse.
- :data:`IS_LIBEDIT` — ``True`` when the stdlib readline is the
  libedit-linked variant (typical on macOS).
- :data:`IS_GNU` — ``True`` when the backing implementation is GNU
  readline (Linux).

Usage::

    from mdma_rebuild.core.readline_compat import readline, HAS_READLINE

    if HAS_READLINE:
        readline.parse_and_bind("tab: complete")

``parse_and_bind`` strings that target libedit (``bind ^K ed-kill-line``)
or GNU-specific escape sequences work on their native backend only;
the shim doesn't translate between syntaxes — that's the caller's
responsibility. See ``bmdma.py`` for the pattern.
"""

from __future__ import annotations

import platform
from types import ModuleType
from typing import Optional


def _load_readline() -> tuple[Optional[ModuleType], bool]:
    """Return ``(module, is_pyreadline3)``.

    Prefers the stdlib ``readline`` (works on Linux / macOS). Falls
    back to ``pyreadline3`` on any platform if the stdlib import
    fails. Returns ``(None, False)`` if neither is available.
    """
    # Stdlib first — gives us libedit/GNU detection for free.
    try:
        import readline as _stdlib_readline
        return _stdlib_readline, False
    except ImportError:
        pass
    # Windows (or any env without stdlib readline) -> pyreadline3.
    try:
        import pyreadline3 as _pyreadline3
        return _pyreadline3, True
    except ImportError:
        return None, False


readline, IS_PYREADLINE3 = _load_readline()
HAS_READLINE: bool = readline is not None

# Libedit detection: the stdlib readline's docstring mentions
# "libedit" when it's linked against Apple's libedit shim instead of
# real GNU readline. pyreadline3 is GNU-flavored, so the libedit
# branch is never relevant for it.
IS_LIBEDIT: bool = bool(
    HAS_READLINE
    and not IS_PYREADLINE3
    and readline is not None
    and getattr(readline, "__doc__", None)
    and "libedit" in readline.__doc__
)

IS_GNU: bool = HAS_READLINE and not IS_LIBEDIT and not IS_PYREADLINE3


def parse_and_bind(spec: str) -> bool:
    """Best-effort :func:`readline.parse_and_bind`.

    Returns ``True`` on success, ``False`` when the shim silently
    dropped the binding (either no readline at all, or the backend
    rejected the syntax).

    pyreadline3 occasionally can't parse every GNU escape sequence;
    rather than crash the launcher, this wrapper swallows the
    exception and returns ``False`` so callers can log if they want.
    """
    if not HAS_READLINE:
        return False
    try:
        readline.parse_and_bind(spec)  # type: ignore[union-attr]
        return True
    except Exception:
        return False


def platform_summary() -> str:
    """One-line human-readable summary. Handy for the launcher's
    --check output."""
    if not HAS_READLINE:
        return (
            "readline: not available "
            f"(install pyreadline3 on {platform.system()} to enable "
            "history + completion)"
        )
    if IS_PYREADLINE3:
        return "readline: pyreadline3 (Windows-compatible)"
    if IS_LIBEDIT:
        return "readline: stdlib readline (libedit backend, macOS)"
    return "readline: stdlib readline (GNU backend)"


__all__ = [
    "HAS_READLINE",
    "IS_GNU",
    "IS_LIBEDIT",
    "IS_PYREADLINE3",
    "parse_and_bind",
    "platform_summary",
    "readline",
]
