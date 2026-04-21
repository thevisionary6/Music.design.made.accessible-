"""Commands that drive the V2 SignalFlow backend.

Currently exposes one command:

- ``/patn`` — build a :class:`mdma_rebuild.backend.Pattern` from the
  args and play it via :meth:`Session.play_pattern`.

The existing numpy-buffer commands (``/tone``, ``/mel``, ``/pat``,
etc.) stay untouched. ``/patn`` is the first V2-backed entry point —
more will follow once Scheduler / PAR / automation commands are
decided on.

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


def _default_shape(note: float):
    """Build a default SawOscillator voice at ``note``.

    Imported lazily so the command module stays importable when
    SignalFlow is missing. The import error surfaces at call time,
    not at module load.
    """
    import signalflow as sf
    freq = 440.0 * (2.0 ** ((note - 69.0) / 12.0))
    return sf.SawOscillator(freq)


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

    try:
        session.play_pattern(pattern, _default_shape)
    except Exception as exc:
        return f"ERROR: /patn playback failed: {exc}"

    total = sum(d for _n, d in events)
    return f"OK: played {len(events)} events ({total:.3f}s)"


def get_backend_commands() -> dict:
    """Dict of name -> callable for the router's late-phase loader.

    Keeping this small so the router's ``COMMAND_OWNERS`` stays easy
    to reason about.
    """
    return {
        "patn": cmd_patn,
    }


__all__ = ["cmd_patn", "get_backend_commands"]
