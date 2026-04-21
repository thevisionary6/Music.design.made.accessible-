"""Command router for the MDMA rebuild.

This module owns the slash-command discovery, ownership arbitration, and
dispatch for every ``cmd_*`` function defined across the
``mdma_rebuild.commands`` package. It was extracted from ``bmdma.py`` so
that every launcher and harness (the REPL, the GUI, the TUI, the
InputController handlers, the scheduler's render path, test suites) can
consume the same command table without importing the REPL entry point.

Public surface:

- :data:`COMMAND_OWNERS` — explicit module-owns-command map used to resolve
  collisions where multiple modules define ``cmd_<same>``.
- :func:`load_command_modules` — import every command module and cache the
  resulting ``ModuleType`` objects on the module-level ``_CMD_MODULES`` dict.
- :func:`build_command_table` — assemble the ``{'name': callable}`` dict the
  launcher feeds to its input loop.
- :func:`dispatch` — convenience helper for callers that just want to fire
  one command by name (used by InputController handlers, etc.).

BUILD ID: router_v1 (extracted from bmdma_v52.0)
"""

from __future__ import annotations

from types import ModuleType
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

# Cache of imported command modules. Keys are the short module names
# (e.g. "synth_cmds"); values are the imported module objects or None when
# the import failed. Populated by :func:`load_command_modules`.
_CMD_MODULES: dict[str, Optional[ModuleType]] = {}

# Names of every command module we try to eagerly import. Optional modules
# (like ``ai_cmds``) are handled by :func:`load_command_modules` and do not
# prevent the rest from loading if missing.
_EAGER_MODULES: tuple[str, ...] = (
    "general_cmds",
    "synth_cmds",
    "fx_cmds",
    "render_cmds",
    "advanced_cmds",
    "pattern_cmds",
    "playback_cmds",
    "buffer_cmds",
    "math_cmds",
    "param_cmds",
    "hq_cmds",
    "gen_cmds",
)

# Set to True once :func:`load_command_modules` confirms ``ai_cmds`` imported
# cleanly. Callers can read this to decide whether to wire up AI features.
AI_AVAILABLE: bool = False


def _import_module(name: str) -> Optional[ModuleType]:
    """Import one command module by short name. Returns None on failure.

    Mirrors the previous ``bmdma._import_module`` behaviour: the failure is
    logged to stdout but does not raise, so a missing optional module does
    not abort the launcher.
    """
    try:
        mod = __import__(f"mdma_rebuild.commands.{name}", fromlist=[name])
    except ImportError as exc:
        print(f"Warning: Could not import {name}: {exc}")
        _CMD_MODULES[name] = None
        return None
    _CMD_MODULES[name] = mod
    return mod


def load_command_modules() -> dict[str, Optional[ModuleType]]:
    """Eagerly import the command modules and populate :data:`_CMD_MODULES`.

    Safe to call multiple times; re-imports refresh the cache. Returns the
    cache dict so callers can look up specific modules without reaching for
    the module-level name.
    """
    global AI_AVAILABLE

    for mod_name in _EAGER_MODULES:
        _import_module(mod_name)

    ai_mod = _import_module("ai_cmds")
    AI_AVAILABLE = ai_mod is not None

    return _CMD_MODULES


# ---------------------------------------------------------------------------
# Ownership map
# ---------------------------------------------------------------------------

# Explicit ownership for conflicting commands. When two modules both define
# ``cmd_play`` the one listed here wins; commands not in this dict use the
# "last loaded wins" behaviour of :func:`build_command_table`.
#
# Keep this dict at module scope (rather than inside ``build_command_table``)
# so tests and editor tooling can import it without running the full table
# assembly.
COMMAND_OWNERS: dict[str, str] = {
    # Playback - playback_cmds owns unified playback
    "play": "playback_cmds",
    "stop": "playback_cmds",
    "p": "playback_cmds",
    "pw": "playback_cmds",
    "stop_play": "playback_cmds",
    "s": "playback_cmds",

    # Buffer operations - buffer_cmds owns
    "a": "buffer_cmds",
    "b": "buffer_cmds",
    "pa": "buffer_cmds",
    "buf": "buffer_cmds",
    "bu": "buffer_cmds",
    "clr": "buffer_cmds",

    # Working buffer - working_cmds owns
    "wb": "working_cmds",
    "wbc": "working_cmds",
    "w": "working_cmds",
    "as": "working_cmds",

    # Pattern - pattern_cmds owns
    "pat": "pattern_cmds",
    "apat": "pattern_cmds",
    "chop": "pattern_cmds",
    "arp": "pattern_cmds",

    # Synth - synth_cmds owns
    "preset": "synth_cmds",
    "tone": "synth_cmds",
    "n": "synth_cmds",

    # Effects - fx_cmds owns
    "fx": "fx_cmds",
    "st": "fx_cmds",
    "vamp": "fx_cmds",
    "fc": "fx_cmds",
    "gg": "fx_cmds",

    # DJ - dj_cmds owns DJ-specific
    "vol": "dj_cmds",
    "deck": "dj_cmds",
    "djm": "dj_cmds",
    "xfade": "dj_cmds",

    # Performance - perf_cmds owns
    "mc": "perf_cmds",
    "snap": "perf_cmds",
    "perf": "perf_cmds",

    # General - general_cmds owns
    "save": "general_cmds",
    "load": "general_cmds",
    "import": "general_cmds",
    "bpm": "general_cmds",
    "help": "general_cmds",
    "h": "general_cmds",

    # Render - render_cmds owns
    "rn": "render_cmds",
    "render": "render_cmds",

    # AI - ai_cmds owns
    "ai": "ai_cmds",
    "enhance": "ai_cmds",
    "gen": "ai_cmds",
    "ask": "ai_cmds",
    "high": "ai_cmds",

    # Audio-rate modulation - audiorate_cmds owns
    "audiorate": "audiorate_cmds",
    "ar": "audiorate_cmds",
    "ump": "audiorate_cmds",
    "impulse": "audiorate_cmds",
    "chke": "audiorate_cmds",

    # Phase 3: Convolution & impulse commands - convolution_cmds owns
    "conv": "convolution_cmds",
    "convolution": "convolution_cmds",
    "convrev": "convolution_cmds",
    "impulselfo": "convolution_cmds",
    "ilfo": "convolution_cmds",
    "lfoimport": "convolution_cmds",
    "impenv": "convolution_cmds",
    "ienv": "convolution_cmds",
    "envimport": "convolution_cmds",
    "irenhance": "convolution_cmds",
    "ire": "convolution_cmds",
    "irtransform": "convolution_cmds",
    "irt": "convolution_cmds",
    "irgranular": "convolution_cmds",
    "irg": "convolution_cmds",
    "irgrains": "convolution_cmds",

    # Phase T: System audit commands - phase_t_cmds owns
    "undo": "phase_t_cmds",
    "redo": "phase_t_cmds",
    "snapshot": "phase_t_cmds",
    "section": "phase_t_cmds",
    "pchain": "phase_t_cmds",
    "export": "phase_t_cmds",
    "master_gain": "phase_t_cmds",
    "mgain": "phase_t_cmds",
    "crossover": "phase_t_cmds",
    "xover": "phase_t_cmds",
    "dup": "phase_t_cmds",
    "duplicate": "phase_t_cmds",
    "metronome": "phase_t_cmds",
    "metro": "phase_t_cmds",
    "commit": "phase_t_cmds",
    "cm": "phase_t_cmds",
    "autosave": "phase_t_cmds",
    "pos": "phase_t_cmds",
    "seek": "phase_t_cmds",
    "swap": "phase_t_cmds",
    "filefx": "phase_t_cmds",

    # Phase 4: Generative commands - gen_cmds owns
    "beat": "gen_cmds",
    "loop": "gen_cmds",
    "xform": "gen_cmds",
    "transform": "gen_cmds",
    "adapt": "gen_cmds",
    "theory": "gen_cmds",
    "gen2": "gen_cmds",
    "generate": "gen_cmds",

    # V2 backend commands - backend_cmds owns
    "patn": "backend_cmds",
    "loadfx": "backend_cmds",
    "listfx": "backend_cmds",
}


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------

def _module_short_name(module: Optional[ModuleType]) -> Optional[str]:
    """Return the trailing path component of ``module.__name__`` or None."""
    if module is None:
        return None
    name = getattr(module, "__name__", str(module))
    return name.split(".")[-1] if "." in name else name


def _should_register(cmd_name: str, owner_name: Optional[str]) -> bool:
    """True when ``owner_name`` is the recognised owner for ``cmd_name``.

    Commands without an entry in :data:`COMMAND_OWNERS` always register
    (last-loaded-wins), so this returns True in that case too.
    """
    if cmd_name in COMMAND_OWNERS:
        return COMMAND_OWNERS[cmd_name] == owner_name
    return True


def build_command_table() -> dict[str, Callable[..., object]]:
    """Collect all ``cmd_*`` functions from every command module.

    Loading order is significant — later phases override earlier ones for
    commands without an explicit owner. The priority ladder below matches
    the bmdma v52 behaviour; keep it stable unless you are intentionally
    rewiring the REPL.

    Priority order for non-owned commands (lowest to highest):

    1. ``general_cmds``, ``synth_cmds``, ``fx_cmds``, ``render_cmds``
    2. ``advanced_cmds`` / ``adv_cmds``
    3. ``buffer_cmds``, ``pattern_cmds``, ``working_cmds``
    4. ``math_cmds``
    5. ``playback_cmds``
    6. ``generator_cmds``
    7. ``dj_cmds``, ``perf_cmds``
    8. ``pack_cmds``, ``audiorate_cmds``, ``convolution_cmds``, ``param_cmds``,
       ``hq_cmds``, ``gen_cmds``, ``phase_t_cmds``
    9. ``ai_cmds`` (highest)
    """
    if not _CMD_MODULES:
        load_command_modules()

    commands: dict[str, Callable[..., object]] = {}

    def register_from_module(module: Optional[ModuleType], source_name: Optional[str] = None) -> None:
        """Register ``cmd_*`` attributes of ``module`` into the table."""
        if module is None:
            return
        mod_name = source_name or _module_short_name(module)
        for attr_name in dir(module):
            if not attr_name.startswith("cmd_"):
                continue
            func = getattr(module, attr_name)
            if not callable(func):
                continue
            cmd_name = attr_name[4:].lower()
            if _should_register(cmd_name, mod_name):
                commands[cmd_name] = func

    def register_from_dict(cmd_dict: object, mod_name: str) -> None:
        """Register entries from a ``{'name': func}`` dict returned by a module."""
        if not isinstance(cmd_dict, dict):
            return
        for cmd_name, func in cmd_dict.items():
            if not callable(func):
                continue
            if cmd_name in COMMAND_OWNERS:
                if COMMAND_OWNERS[cmd_name] == mod_name:
                    commands[cmd_name] = func
            else:
                commands[cmd_name] = func

    # PHASE 1: base modules (stub_cmds was removed during Phase 0 triage).
    for mod_name in ("general_cmds", "synth_cmds", "fx_cmds", "render_cmds"):
        register_from_module(_CMD_MODULES.get(mod_name), mod_name)

    # PHASE 3: advanced_cmds plus its ADVANCED_COMMANDS dict.
    register_from_module(_CMD_MODULES.get("advanced_cmds"), "advanced_cmds")
    adv_mod = _CMD_MODULES.get("advanced_cmds")
    if adv_mod is not None and hasattr(adv_mod, "ADVANCED_COMMANDS"):
        register_from_dict(adv_mod.ADVANCED_COMMANDS, "advanced_cmds")

    # adv_cmds (separate module from advanced_cmds).
    try:
        from mdma_rebuild.commands.adv_cmds import get_advanced_commands
        register_from_dict(get_advanced_commands(), "adv_cmds")
    except ImportError:
        pass

    # PHASE 4: buffer / pattern / working modules.
    register_from_module(_CMD_MODULES.get("buffer_cmds"), "buffer_cmds")
    register_from_module(_CMD_MODULES.get("pattern_cmds"), "pattern_cmds")

    try:
        from mdma_rebuild.commands.working_cmds import get_working_commands
        register_from_dict(get_working_commands(), "working_cmds")
    except ImportError:
        pass

    try:
        from mdma_rebuild.commands import working_cmds as _wc_mod
        register_from_module(_wc_mod, "working_cmds")
    except ImportError:
        pass

    # PHASE 4.5: math + utility commands.
    register_from_module(_CMD_MODULES.get("math_cmds"), "math_cmds")

    # PHASE 5: playback (owns play/stop).
    register_from_module(_CMD_MODULES.get("playback_cmds"), "playback_cmds")
    pb_mod = _CMD_MODULES.get("playback_cmds")
    if pb_mod is not None and hasattr(pb_mod, "get_playback_commands"):
        try:
            register_from_dict(pb_mod.get_playback_commands(), "playback_cmds")
        except Exception:
            pass

    # PHASE 6: generators.
    try:
        from mdma_rebuild.commands.generator_cmds import get_generator_commands
        register_from_dict(get_generator_commands(), "generator_cmds")
    except ImportError:
        pass

    # PHASE 7: DJ and performance.
    try:
        from mdma_rebuild.commands.dj_cmds import get_dj_commands
        register_from_dict(get_dj_commands(), "dj_cmds")
    except ImportError:
        pass

    try:
        from mdma_rebuild.commands.perf_cmds import get_perf_commands
        register_from_dict(get_perf_commands(), "perf_cmds")
    except ImportError:
        pass

    # PHASE 8: pack commands.
    try:
        from mdma_rebuild.commands.pack_cmds import get_pack_commands
        register_from_dict(get_pack_commands(), "pack_cmds")
    except ImportError:
        pass

    # PHASE 8.5: audio-rate modulation / umpulse.
    try:
        from mdma_rebuild.commands.audiorate_cmds import get_audiorate_commands
        register_from_dict(get_audiorate_commands(), "audiorate_cmds")
    except ImportError:
        pass

    # PHASE 8.55: convolution / impulse commands.
    try:
        from mdma_rebuild.commands.convolution_cmds import get_convolution_commands
        register_from_dict(get_convolution_commands(), "convolution_cmds")
    except ImportError:
        pass

    # PHASE 8.6: parameter system (v45).
    register_from_module(_CMD_MODULES.get("param_cmds"), "param_cmds")
    param_mod = _CMD_MODULES.get("param_cmds")
    if param_mod is not None and hasattr(param_mod, "get_param_commands"):
        try:
            register_from_dict(param_mod.get_param_commands(), "param_cmds")
        except Exception:
            pass

    # PHASE 8.7: HQ audio (v45).
    register_from_module(_CMD_MODULES.get("hq_cmds"), "hq_cmds")
    hq_mod = _CMD_MODULES.get("hq_cmds")
    if hq_mod is not None and hasattr(hq_mod, "get_hq_commands"):
        try:
            register_from_dict(hq_mod.get_hq_commands(), "hq_cmds")
        except Exception:
            pass

    # PHASE 8.8: generative (Phase 4).
    try:
        from mdma_rebuild.commands.gen_cmds import get_gen_commands
        register_from_dict(get_gen_commands(), "gen_cmds")
    except ImportError:
        pass

    # PHASE 8.9: Phase T commands.
    try:
        from mdma_rebuild.commands.phase_t_cmds import get_phase_t_commands
        register_from_dict(get_phase_t_commands(), "phase_t_cmds")
    except ImportError:
        pass

    # PHASE 8.95: V2 backend commands (/patn, ...). Kept ahead of
    # ai_cmds so the backend surface stays visible when AI is on.
    try:
        from mdma_rebuild.commands.backend_cmds import get_backend_commands
        register_from_dict(get_backend_commands(), "backend_cmds")
    except ImportError:
        pass

    # PHASE 9: AI commands (highest).
    ai_mod = _CMD_MODULES.get("ai_cmds")
    if AI_AVAILABLE and ai_mod is not None:
        register_from_module(ai_mod, "ai_cmds")
        if hasattr(ai_mod, "AI_COMMANDS"):
            register_from_dict(ai_mod.AI_COMMANDS, "ai_cmds")

    # PHASE 10: explicit voice parameter commands.
    try:
        from mdma_rebuild.commands import synth_cmds as _synth_cmds
        voice_cmds = {
            "stereo": _synth_cmds.cmd_stereo,
            "vphase": _synth_cmds.cmd_vphase,
            "venv": _synth_cmds.cmd_venv,
            "fenv": _synth_cmds.cmd_fenv,
            "menv": _synth_cmds.cmd_menv,
        }
        for cmd_name, func in voice_cmds.items():
            commands[cmd_name] = func

        if hasattr(_synth_cmds, "get_synth_commands"):
            register_from_dict(_synth_cmds.get_synth_commands(), "synth_cmds")
    except (ImportError, AttributeError):
        pass

    # PHASE 10.5: MAD DSL.
    try:
        from mdma_rebuild.commands.dsl_cmds import get_dsl_commands
        register_from_dict(get_dsl_commands(), "dsl_cmds")
    except ImportError:
        pass

    # PHASE 10.6: SyDef commands.
    try:
        from mdma_rebuild.commands.sydef_cmds import get_sydef_commands
        register_from_dict(get_sydef_commands(), "sydef_cmds")
    except ImportError:
        pass

    # PHASE 11: help alias fallback.
    if "help" in commands and "h" not in commands:
        commands["h"] = commands["help"]

    return commands


# ---------------------------------------------------------------------------
# Convenience dispatch
# ---------------------------------------------------------------------------

def dispatch(
    session: object,
    command_name: str,
    args: Optional[list[str]] = None,
    commands: Optional[dict[str, Callable[..., object]]] = None,
) -> object:
    """Fire a single command by name.

    Handy for non-REPL callers (InputController handlers, test harnesses,
    GUI buttons) that want "run the slash command named X" without
    reimplementing the launcher's input loop.

    Raises :class:`KeyError` if the command is unknown. Unknown-command
    handling is left to the caller so they can choose their own UX (an
    autocomplete hint, an error dialog, a no-op for MIDI handlers, etc.).
    """
    if commands is None:
        commands = build_command_table()
    if command_name not in commands:
        raise KeyError(f"Unknown command: {command_name}")
    return commands[command_name](session, args or [])


__all__ = [
    "AI_AVAILABLE",
    "COMMAND_OWNERS",
    "build_command_table",
    "dispatch",
    "load_command_modules",
]
