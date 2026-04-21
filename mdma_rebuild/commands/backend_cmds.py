"""Commands that drive the V2 SignalFlow backend.

Current commands:

- ``/patn`` — build a :class:`mdma_rebuild.backend.Pattern` from the
  args and play it via :meth:`Session.play_pattern`.
- ``/loadfx`` — load custom SignalFlow effect callables from a
  drop-in file or directory. They land in
  ``session.custom_effects`` where Python-level callers can pick
  them up via ``pattern.fx(session.custom_effects[name], ...)``.
- ``/listfx`` — list the currently-loaded custom effects.

The existing numpy-buffer commands (``/tone``, ``/mel``, ``/pat``,
etc.) stay untouched.

Command argument grammar for ``/patn``:

    /patn <note>,<dur> <note>,<dur> ...

where ``note`` is a MIDI note number (float, so microtones work) and
``dur`` is a duration in seconds (float). Example::

    /patn 60,0.25 64,0.25 67,0.5

Edge cases:

- No args -> returns a usage hint without playing anything.
- Malformed token -> returns an error message with the offending
  token, does not play.
- ``session.backend_graph()`` raises when SignalFlow isn't installed;
  the command surfaces that as a clear message rather than a stack
  trace.
"""

from __future__ import annotations

from typing import List, Tuple

from ..backend.fx_loader import (
    DEFAULT_USER_DIR,
    format_summary,
    load_from,
)
from ..backend.pattern import Pattern


_USAGE = (
    "Usage: /patn <note>,<dur> [<note>,<dur> ...]\n"
    "  note: MIDI note number (float, microtones ok)\n"
    "  dur : seconds (float)\n"
    "  e.g. /patn 60,0.25 64,0.25 67,0.5"
)


def _parse_events(args: List[str]) -> Tuple[List[Tuple[float, float]], str]:
    """Return ``(events, error)`` where ``error`` is empty on success."""
    events: List[Tuple[float, float]] = []
    for tok in args:
        if "," not in tok:
            return [], f"ERROR: expected '<note>,<dur>', got {tok!r}"
        note_str, _, dur_str = tok.partition(",")
        try:
            note = float(note_str)
        except ValueError:
            return [], f"ERROR: could not parse note from {tok!r}"
        try:
            dur = float(dur_str)
        except ValueError:
            return [], f"ERROR: could not parse duration from {tok!r}"
        if dur <= 0:
            return [], f"ERROR: duration must be positive in {tok!r}"
        events.append((note, dur))
    return events, ""


def _default_shape(session):
    """Pick the right default voice factory.

    When the session has monolith operators configured, the user
    probably wants the V2 graph to reflect their REPL settings —
    ``/wm saw``, ``/fr 440``, ``/fm 1 0 0.5``, etc. When not, a
    plain SawOscillator is a sensible least-surprise default.
    See :func:`mdma_rebuild.backend.utils.build_default_shape`.
    """
    from ..backend.utils import build_default_shape
    return build_default_shape(session)


def cmd_patn(session, args: List[str]) -> str:
    """Play a pattern through the V2 backend.

    See the module docstring for the argument grammar.
    """
    if not args:
        return _USAGE

    events, err = _parse_events(args)
    if err:
        return err

    pattern = Pattern(events)

    try:
        graph = session.backend_graph()
    except ImportError:
        return (
            "ERROR: V2 backend needs signalflow installed "
            "(pip install signalflow)."
        )
    except Exception as exc:
        return f"ERROR: backend graph unavailable: {exc}"

    shape = _default_shape(session)
    try:
        session.play_pattern(pattern, shape)
    except Exception as exc:
        return f"ERROR: /patn playback failed: {exc}"

    total = sum(d for _n, d in events)
    return f"OK: played {len(events)} events ({total:.3f}s)"


# ---------------------------------------------------------------------------
# /loadfx and /listfx
# ---------------------------------------------------------------------------

_LOADFX_USAGE = (
    "Usage: /loadfx [path]\n"
    "  path defaults to ~/.mdma/effects (the standard drop-in dir).\n"
    "  Loads every top-level callable matching fn(input_node, **params).\n"
    "  Loaded effects land in session.custom_effects; see /listfx."
)


def _ensure_registry(session) -> dict:
    """Lazy-init ``session.custom_effects``. Matches the spec convention
    that the Session object carries live state the backend needs."""
    registry = getattr(session, "custom_effects", None)
    if registry is None:
        registry = {}
        session.custom_effects = registry
    return registry


def cmd_loadfx(session, args: List[str]) -> str:
    """Load drop-in custom effects.

    ``/loadfx`` with no args reads from ``~/.mdma/effects``.
    ``/loadfx path/to/file.py`` or ``/loadfx path/to/dir`` works too.
    """
    if args and args[0] in ("-h", "--help", "help"):
        return _LOADFX_USAGE

    path = args[0] if args else str(DEFAULT_USER_DIR)
    registry = _ensure_registry(session)
    results = load_from(registry, path)
    return format_summary(results)


def cmd_listfx(session, args: List[str]) -> str:
    """List currently-registered custom effects."""
    registry = getattr(session, "custom_effects", None) or {}
    if not registry:
        return (
            "No custom effects loaded. Try /loadfx to load from "
            f"{DEFAULT_USER_DIR}."
        )
    lines = [f"{len(registry)} custom effect(s) loaded:"]
    for name in sorted(registry):
        fn = registry[name]
        doc = (fn.__doc__ or "").strip().splitlines()
        summary = doc[0] if doc else "<no docstring>"
        lines.append(f"  {name:<20} {summary}")
    return "\n".join(lines)


def get_backend_commands() -> dict:
    """Dict of name -> callable for the router's late-phase loader.

    Keeping this small so the router's ``COMMAND_OWNERS`` stays easy
    to reason about.
    """
    return {
        "patn": cmd_patn,
        "loadfx": cmd_loadfx,
        "listfx": cmd_listfx,
    }


__all__ = [
    "cmd_patn",
    "cmd_loadfx",
    "cmd_listfx",
    "get_backend_commands",
]
