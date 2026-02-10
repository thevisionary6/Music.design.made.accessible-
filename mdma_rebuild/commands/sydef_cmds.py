"""Synth Definition System (SyDef) for MDMA v39.

Provides /sydef to define reusable synth patches as named blocks,
and /use to instantiate them with argument substitution.

A SyDef captures a sequence of MDMA commands (oscillators, filters,
envelopes, effects, routings, etc.) as a named template.  Arguments
declared in the header become substitutable tokens inside the block.

Usage:
    /sydef bass freq=220 cutoff=800 res=60
      /op 1
      /wm saw
      /fr $freq
      /ft acid
      /cut $cutoff
      /res $res
      /atk 0.01
      /dec 0.3
      /sus 0.6
      /rel 0.2
      /fx reverb
    /end

    /use bass                    -- use all defaults
    /use bass 440                -- freq=440, rest defaults
    /use bass freq=880 res=80    -- named overrides
    /use bass 440 1200 90        -- positional overrides

SyDefs are stored on the session and can be listed, inspected,
deleted, exported, and imported.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mdma_rebuild.core.session import Session


# ---------------------------------------------------------------------------
# Auto-persistence helpers
# ---------------------------------------------------------------------------

def _autosave_sydefs(session: Session) -> None:
    """Persist all user-created SyDefs to ~/Documents/MDMA/sydefs.json.

    Factory presets are excluded (they're re-loaded on boot).
    Called automatically whenever a SyDef is created, modified, or deleted.
    """
    try:
        from mdma_rebuild.core.user_data import save_sydefs, get_sydefs_path
        # Only persist non-factory definitions
        user_defs = {}
        factory_names = set(getattr(session, '_factory_sydef_names', []))
        
        for name, sd in session.sydefs.items():
            if name not in factory_names:
                user_defs[name] = sd
        
        if user_defs:
            success = save_sydefs(user_defs)
            if not success:
                print(f"[sydef] WARNING: Failed to save {len(user_defs)} user SyDefs")
        # If no user defs, that's fine - nothing to save
    except Exception as e:
        print(f"[sydef] WARNING: Auto-save error: {e}")


def _autoload_sydefs(session: Session) -> int:
    """Load persisted user SyDefs from disk into the session.

    Returns the number of definitions loaded.
    """
    try:
        from mdma_rebuild.core.user_data import load_sydefs
        data = load_sydefs()
        count = 0
        for name, d in data.items():
            session.sydefs[name.lower()] = SyDef.from_dict(d)
            count += 1
        return count
    except Exception:
        return 0

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class SyDefParam:
    """A single parameter declared in a sydef header."""
    __slots__ = ('name', 'default')

    def __init__(self, name: str, default: str) -> None:
        self.name = name
        self.default = default

    def to_dict(self) -> dict:
        return {'name': self.name, 'default': self.default}

    @classmethod
    def from_dict(cls, d: dict) -> 'SyDefParam':
        return cls(d['name'], d['default'])

    def __repr__(self) -> str:
        return f"{self.name}={self.default}"


class SyDef:
    """A synth definition: named block of commands with parameters."""
    __slots__ = ('name', 'params', 'commands', 'description')

    def __init__(self, name: str, params: List[SyDefParam],
                 commands: List[str], description: str = '') -> None:
        self.name = name
        self.params = params
        self.commands = commands
        self.description = description

    def param_names(self) -> List[str]:
        return [p.name for p in self.params]

    def default_map(self) -> Dict[str, str]:
        return {p.name: p.default for p in self.params}

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'params': [p.to_dict() for p in self.params],
            'commands': self.commands,
            'description': self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SyDef':
        return cls(
            name=d['name'],
            params=[SyDefParam.from_dict(p) for p in d.get('params', [])],
            commands=d.get('commands', []),
            description=d.get('description', ''),
        )

    def __repr__(self) -> str:
        pstr = ', '.join(str(p) for p in self.params)
        return f"SyDef({self.name}({pstr}), {len(self.commands)} cmds)"


# ---------------------------------------------------------------------------
# Session attribute helpers
# ---------------------------------------------------------------------------

def _ensure_sydef_attrs(session: Session) -> None:
    """Ensure sydef-related attributes exist on the session."""
    if not hasattr(session, 'sydefs'):
        session.sydefs: Dict[str, SyDef] = {}
    if not hasattr(session, 'defining_sydef'):
        session.defining_sydef: Optional[str] = None
    if not hasattr(session, 'sydef_commands_buffer'):
        session.sydef_commands_buffer: List[str] = []
    if not hasattr(session, 'sydef_pending_params'):
        session.sydef_pending_params: List[SyDefParam] = []


def is_defining_sydef(session: Session) -> bool:
    """Return True if we're currently inside a /sydef block."""
    _ensure_sydef_attrs(session)
    return session.defining_sydef is not None


# ---------------------------------------------------------------------------
# /sydef command — start or manage synth definitions
# ---------------------------------------------------------------------------

def cmd_sydef(session: Session, args: List[str]) -> str:
    """Define a reusable synth patch.

    Usage:
      /sydef <name> [param=default ...]   Start defining a new SyDef
      /sydef list                         List all defined SyDefs
      /sydef show <name>                  Show contents of a SyDef
      /sydef del <name>                   Delete a SyDef
      /sydef save <path>                  Save all SyDefs to JSON
      /sydef load <path>                  Load SyDefs from JSON
      /sydef desc <name> <text>           Set description for a SyDef
      /sydef copy <src> <dst>             Copy a SyDef to new name

    Inside a /sydef block, use $param_name to reference arguments.
    End the block with /end.

    Examples:
      /sydef acid_bass freq=220 cutoff=1000 res=70
        /op 1
        /wm saw
        /fr $freq
        /ft acid
        /cut $cutoff
        /res $res
      /end

      /sydef pad freq=440 width=50 verb=0.6
        /op 1
        /wm sine
        /fr $freq
        /vc 4
        /dt 2
        /stereo $width
        /fx reverb $verb
      /end
    """
    _ensure_sydef_attrs(session)

    if not args:
        # No args: show summary
        if not session.sydefs:
            return ("SYDEF: No synth definitions yet.\n"
                    "  Create one: /sydef <name> [param=default ...]\n"
                    "  Example:    /sydef bass freq=220 cutoff=800")
        lines = ["SYDEF: Defined synth patches:"]
        for name, sd in session.sydefs.items():
            pstr = ' '.join(str(p) for p in sd.params) if sd.params else '(no params)'
            desc = f' — {sd.description}' if sd.description else ''
            lines.append(f"  {name}({pstr}) [{len(sd.commands)} cmds]{desc}")
        lines.append(f"\n  Total: {len(session.sydefs)} | /sydef show <name> for details")
        return '\n'.join(lines)

    sub = args[0].lower()

    # --- /sydef list ---
    if sub == 'list':
        return cmd_sydef(session, [])

    # --- /sydef show <name> ---
    if sub == 'show':
        if len(args) < 2:
            return "Usage: /sydef show <name>"
        name = args[1].lower()
        if name not in session.sydefs:
            return _suggest_sydef(session, name)
        sd = session.sydefs[name]
        lines = [f"SYDEF: '{name}'"]
        if sd.description:
            lines.append(f"  Description: {sd.description}")
        if sd.params:
            lines.append(f"  Parameters:")
            for p in sd.params:
                lines.append(f"    ${p.name} = {p.default}")
        else:
            lines.append(f"  Parameters: (none)")
        lines.append(f"  Commands ({len(sd.commands)}):")
        for i, cmd in enumerate(sd.commands, 1):
            lines.append(f"    {i:3d}. {cmd}")
        return '\n'.join(lines)

    # --- /sydef del <name> ---
    if sub in ('del', 'delete', 'rm', 'remove'):
        if len(args) < 2:
            return "Usage: /sydef del <name>"
        name = args[1].lower()
        if name not in session.sydefs:
            return _suggest_sydef(session, name)
        del session.sydefs[name]
        _autosave_sydefs(session)
        return f"SYDEF: Deleted '{name}'"

    # --- /sydef copy <src> <dst> ---
    if sub == 'copy':
        if len(args) < 3:
            return "Usage: /sydef copy <source> <destination>"
        src = args[1].lower()
        dst = args[2].lower()
        if src not in session.sydefs:
            return _suggest_sydef(session, src)
        sd = session.sydefs[src]
        session.sydefs[dst] = SyDef(
            name=dst,
            params=copy.deepcopy(sd.params),
            commands=list(sd.commands),
            description=sd.description,
        )
        _autosave_sydefs(session)
        return f"SYDEF: Copied '{src}' -> '{dst}'"

    # --- /sydef desc <name> <text> ---
    if sub == 'desc':
        if len(args) < 3:
            return "Usage: /sydef desc <name> <description text>"
        name = args[1].lower()
        if name not in session.sydefs:
            return _suggest_sydef(session, name)
        session.sydefs[name].description = ' '.join(args[2:])
        return f"SYDEF: Description set for '{name}'"

    # --- /sydef save <path> ---
    if sub == 'save':
        path = args[1] if len(args) > 1 else 'sydefs.json'
        if not path.endswith('.json'):
            path += '.json'
        data = {name: sd.to_dict() for name, sd in session.sydefs.items()}
        try:
            Path(path).write_text(json.dumps(data, indent=2))
            return f"SYDEF: Saved {len(data)} definitions to {path}"
        except Exception as e:
            return f"ERROR: Could not save: {e}"

    # --- /sydef load <path> ---
    if sub == 'load':
        if len(args) < 2:
            return "Usage: /sydef load <path>"
        path = args[1]
        if not path.endswith('.json'):
            path += '.json'
        try:
            data = json.loads(Path(path).read_text())
            count = 0
            for name, d in data.items():
                session.sydefs[name.lower()] = SyDef.from_dict(d)
                count += 1
            _autosave_sydefs(session)
            return f"SYDEF: Loaded {count} definitions from {path}"
        except FileNotFoundError:
            return f"ERROR: File not found: {path}"
        except Exception as e:
            return f"ERROR: Could not load: {e}"

    # --- Start a new definition: /sydef <name> [param=default ...] ---
    name = sub  # First arg is the name

    # Validate name
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        return (f"ERROR: Invalid SyDef name '{name}'. "
                "Use letters, numbers, underscores (start with letter).")

    # Reserved sub-commands
    if name in ('list', 'show', 'del', 'delete', 'rm', 'remove',
                'copy', 'desc', 'save', 'load', 'help'):
        return f"ERROR: '{name}' is reserved. Choose a different name."

    # Parse parameters: param=default
    params = []
    for arg in args[1:]:
        if '=' in arg:
            pname, pdefault = arg.split('=', 1)
            pname = pname.strip().lower()
            if not re.match(r'^[a-zA-Z_]\w*$', pname):
                return f"ERROR: Invalid param name '{pname}'"
            params.append(SyDefParam(pname, pdefault.strip()))
        else:
            # Positional param with no default — default is empty
            pname = arg.strip().lower()
            if not re.match(r'^[a-zA-Z_]\w*$', pname):
                return f"ERROR: Invalid param name '{pname}'"
            params.append(SyDefParam(pname, ''))

    # Check for overwrite
    overwrite_note = ''
    if name in session.sydefs:
        overwrite_note = f' (overwriting existing)'

    # Enter definition mode
    session.defining_sydef = name
    session.sydef_commands_buffer = []
    session.sydef_pending_params = params

    pstr = ', '.join(f'${p.name}={p.default}' for p in params)
    if not pstr:
        pstr = '(no parameters)'

    return (f"SYDEF: Defining '{name}'{overwrite_note}\n"
            f"  Parameters: {pstr}\n"
            f"  Enter synth commands now. Use $param_name for arguments.\n"
            f"  Type /end to finish definition.")


# ---------------------------------------------------------------------------
# Recording — called from the REPL when defining_sydef is set
# ---------------------------------------------------------------------------

def record_sydef_command(session: Session, line: str) -> str:
    """Record a command line into the active sydef block.

    Returns a confirmation string for the REPL to print.
    """
    _ensure_sydef_attrs(session)
    session.sydef_commands_buffer.append(line)

    # Show which params are referenced
    refs = re.findall(r'\$([a-zA-Z_]\w*)', line)
    param_names = {p.name for p in session.sydef_pending_params}
    ref_note = ''
    if refs:
        matched = [r for r in refs if r in param_names]
        unmatched = [r for r in refs if r not in param_names]
        parts = []
        if matched:
            parts.append(f"params: {', '.join(matched)}")
        if unmatched:
            parts.append(f"WARNING unbound: {', '.join(unmatched)}")
        ref_note = f'  ({"; ".join(parts)})'

    n = len(session.sydef_commands_buffer)
    return f"  [{n}] {line}{ref_note}"


def end_sydef(session: Session) -> str:
    """Finalize the current sydef definition (called on /end)."""
    _ensure_sydef_attrs(session)

    name = session.defining_sydef
    cmds = session.sydef_commands_buffer
    params = session.sydef_pending_params

    if not cmds:
        session.defining_sydef = None
        session.sydef_commands_buffer = []
        session.sydef_pending_params = []
        return f"SYDEF: '{name}' discarded (empty definition)"

    sd = SyDef(name=name, params=params, commands=list(cmds))
    session.sydefs[name] = sd
    
    # If user overwrites a factory preset, remove it from factory list
    # so it gets persisted as a user sydef
    if hasattr(session, '_factory_sydef_names') and name in session._factory_sydef_names:
        session._factory_sydef_names.remove(name)

    # Auto-save SyDefs to disk
    _autosave_sydefs(session)

    # Clear definition state
    session.defining_sydef = None
    session.sydef_commands_buffer = []
    session.sydef_pending_params = []

    pstr = ', '.join(f'${p.name}={p.default}' for p in params)
    return (f"SYDEF: '{name}' saved — {len(cmds)} commands"
            + (f", params: {pstr}" if pstr else ""))


# ---------------------------------------------------------------------------
# /use command — instantiate a SyDef
# ---------------------------------------------------------------------------

def cmd_use(session: Session, args: List[str]) -> str:
    """Instantiate a synth definition.

    Usage:
      /use <name>                          Use with all defaults
      /use <name> <v1> <v2> ...            Positional argument override
      /use <name> param=value ...          Named argument override
      /use <name> <v1> param=value ...     Mixed positional + named
      /use <name> Nx                       Instantiate N times

    Positional arguments map to parameters in declaration order.
    Named arguments override by name.  Unspecified params use defaults.

    Examples:
      /use bass                -- all defaults
      /use bass 440            -- freq=440 (first param)
      /use bass freq=880       -- named override
      /use bass 440 1200 90    -- freq=440, cutoff=1200, res=90
      /use bass 4x             -- run 4 times (layering)
    """
    _ensure_sydef_attrs(session)

    if not args:
        # No args: list available sydefs
        if not session.sydefs:
            return ("USE: No synth definitions available.\n"
                    "  Define one first with /sydef <name> [param=default ...]")
        lines = ["USE: Available synth definitions:"]
        for name, sd in session.sydefs.items():
            pstr = ' '.join(f'{p.name}={p.default}' for p in sd.params)
            if not pstr:
                pstr = '(no params)'
            desc = f' — {sd.description}' if sd.description else ''
            lines.append(f"  /use {name} {pstr}{desc}")
        return '\n'.join(lines)

    name = args[0].lower()

    if name not in session.sydefs:
        return _suggest_sydef(session, name)

    sd = session.sydefs[name]

    # Parse call arguments
    loop_count = 1
    positional = []
    named = {}

    for arg in args[1:]:
        # Loop modifier: 4x
        if arg.endswith('x') and arg[:-1].isdigit():
            loop_count = int(arg[:-1])
            continue
        # Named: param=value
        if '=' in arg:
            key, val = arg.split('=', 1)
            named[key.lower()] = val
        else:
            positional.append(arg)

    # Build substitution map: start with defaults, override with call args
    subs = sd.default_map()

    # Override with positional args (in declaration order)
    for i, val in enumerate(positional):
        if i < len(sd.params):
            subs[sd.params[i].name] = val

    # Override with named args
    for key, val in named.items():
        if key in subs:
            subs[key] = val
        else:
            # Unknown param — warn but continue
            known = ', '.join(sd.param_names())
            return (f"ERROR: Unknown parameter '${key}' for '{name}'.\n"
                    f"  Known params: {known}")

    # Check for empty required params
    missing = [k for k, v in subs.items() if v == '']
    if missing:
        return (f"ERROR: Missing required arguments for '{name}': "
                f"{', '.join('$' + m for m in missing)}\n"
                f"  Usage: /use {name} {' '.join(f'{p.name}=<value>' for p in sd.params if p.default == '')}")

    # Get command executor
    if not hasattr(session, 'command_executor'):
        return "ERROR: Command executor not available (internal error)"

    # Execute
    all_results = []

    for loop_i in range(loop_count):
        results = []
        for cmd_line in sd.commands:
            # Substitute $param_name tokens
            expanded = _substitute(cmd_line, subs)

            # Warn about unresolved references
            unresolved = re.findall(r'\$([a-zA-Z_]\w*)', expanded)
            if unresolved:
                results.append(f"  WARNING: unresolved ${', $'.join(unresolved)} in: {expanded}")

            try:
                result = session.command_executor(expanded)
                if result and result.startswith('ERROR'):
                    results.append(f"  {result}")
                elif result and not result.startswith('OK'):
                    results.append(f"  {result}")
            except Exception as e:
                results.append(f"  ERROR in '{expanded}': {e}")

        if results and loop_count > 1:
            all_results.append(f"[{loop_i+1}/{loop_count}]")
            all_results.extend(results[:5])
        elif results:
            all_results.extend(results)

    # Summary
    param_summary = ', '.join(f'{k}={v}' for k, v in subs.items())
    loop_note = f' x{loop_count}' if loop_count > 1 else ''
    header = f"USE: '{name}'{loop_note}"
    if param_summary:
        header += f" ({param_summary})"

    if all_results:
        return header + '\n' + '\n'.join(all_results[:20])
    return header + " — OK"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _substitute(template: str, subs: Dict[str, str]) -> str:
    """Replace $param_name tokens in a command string.

    Handles $name tokens. Longer names are replaced first to avoid
    partial matches (e.g. $freq_mod before $freq).
    """
    result = template
    # Sort by length descending to avoid partial matches
    for key in sorted(subs.keys(), key=len, reverse=True):
        result = result.replace(f'${key}', str(subs[key]))
    return result


def _suggest_sydef(session: Session, name: str) -> str:
    """Error message with suggestions for unknown sydef names."""
    _ensure_sydef_attrs(session)
    if not session.sydefs:
        return (f"ERROR: '{name}' not found. No SyDefs defined yet.\n"
                f"  Create one: /sydef <name> [param=default ...]")
    similar = [n for n in session.sydefs if name in n or n in name
               or n[:3] == name[:3]]
    if similar:
        return (f"ERROR: '{name}' not found. Did you mean: "
                f"{', '.join(similar)}?\n"
                f"  All: {', '.join(session.sydefs.keys())}")
    return (f"ERROR: '{name}' not found.\n"
            f"  Available: {', '.join(session.sydefs.keys())}")


# ---------------------------------------------------------------------------
# Factory presets — built-in SyDefs for common patches
# ---------------------------------------------------------------------------

def _make_factory_presets() -> Dict[str, SyDef]:
    """Create a set of factory SyDef presets.

    Organised into two tiers:
      • Simple presets (1-2 params): saw, square, sub, lead, bass, bell,
        string, noise, hihat — quick one-liners for fast sketching.
      • Full presets (3-4 params): sine, acid, pad, pluck, kick —
        deeper sound design with filter and envelope control.
    """
    presets = {}

    # =================================================================
    #  SIMPLE PRESETS (1-2 params, fast sketching)
    # =================================================================

    # --- Raw saw (warm analog default) ---
    presets['saw'] = SyDef(
        name='saw',
        params=[SyDefParam('freq', '220'), SyDefParam('amp', '0.8')],
        commands=[
            '/op 1',
            '/wm saw',
            '/fr $freq',
            '/amp $amp',
            '/va unison',
            '/atk 0.005',
            '/dec 0.1',
            '/sus 0.7',
            '/rel 0.15',
        ],
        description='Raw saw wave',
    )

    # --- Square (hollow, NES-like) ---
    presets['square'] = SyDef(
        name='square',
        params=[SyDefParam('freq', '220'), SyDefParam('pw', '0.5')],
        commands=[
            '/op 1',
            '/wm pulse',
            '/fr $freq',
            '/amp 0.7',
            '/pw $pw',
            '/va unison',
            '/atk 0.005',
            '/dec 0.1',
            '/sus 0.8',
            '/rel 0.1',
        ],
        description='Square/pulse wave',
    )

    # --- Sub (deep sine sub-bass) ---
    presets['sub'] = SyDef(
        name='sub',
        params=[SyDefParam('freq', '55')],
        commands=[
            '/op 1',
            '/wm sine',
            '/fr $freq',
            '/amp 1.0',
            '/atk 0.01',
            '/dec 0.05',
            '/sus 0.9',
            '/rel 0.2',
        ],
        description='Deep sub-bass sine',
    )

    # --- Lead (bright filtered saw) ---
    presets['lead'] = SyDef(
        name='lead',
        params=[SyDefParam('freq', '440'), SyDefParam('bright', '3000')],
        commands=[
            '/op 1',
            '/wm saw',
            '/fr $freq',
            '/amp 0.8',
            '/ft lp',
            '/cut $bright',
            '/res 40',
            '/atk 0.01',
            '/dec 0.15',
            '/sus 0.6',
            '/rel 0.2',
            '/va unison',
        ],
        description='Bright mono lead',
    )

    # --- Bass (filtered saw bass) ---
    presets['bass'] = SyDef(
        name='bass',
        params=[SyDefParam('freq', '110'), SyDefParam('cut', '800')],
        commands=[
            '/op 1',
            '/wm saw',
            '/fr $freq',
            '/amp 0.9',
            '/ft lp',
            '/cut $cut',
            '/res 30',
            '/atk 0.005',
            '/dec 0.2',
            '/sus 0.4',
            '/rel 0.15',
        ],
        description='Filtered saw bass',
    )

    # --- Bell (FM sine on sine) ---
    presets['bell'] = SyDef(
        name='bell',
        params=[SyDefParam('freq', '880'), SyDefParam('mod', '3')],
        commands=[
            '/op 1',
            '/wm sine',
            '/fr $freq',
            '/amp 0.7',
            '/op 2',
            '/wm sine',
            '/fr $freq',
            '/amp 0.5',
            '/fm 2 1 $mod',
            '/car 1',
            '/atk 0.001',
            '/dec 0.4',
            '/sus 0.0',
            '/rel 0.6',
        ],
        description='FM bell tone',
    )

    # --- String (detuned saw ensemble) ---
    presets['string'] = SyDef(
        name='string',
        params=[SyDefParam('freq', '220'), SyDefParam('voices', '4')],
        commands=[
            '/op 1',
            '/wm saw',
            '/fr $freq',
            '/amp 0.5',
            '/vc $voices',
            '/dt 1.5',
            '/rand 30',
            '/va unison',
            '/ft lp',
            '/cut 3000',
            '/res 15',
            '/atk 0.3',
            '/dec 0.2',
            '/sus 0.7',
            '/rel 0.5',
        ],
        description='String ensemble',
    )

    # --- Noise (white noise hit/texture) ---
    presets['nz'] = SyDef(
        name='nz',
        params=[SyDefParam('cut', '4000')],
        commands=[
            '/op 1',
            '/wm noise',
            '/amp 0.6',
            '/ft lp',
            '/cut $cut',
            '/res 10',
            '/atk 0.001',
            '/dec 0.1',
            '/sus 0.0',
            '/rel 0.05',
        ],
        description='Filtered noise burst',
    )

    # --- Hihat (noise + HP filter) ---
    presets['hihat'] = SyDef(
        name='hihat',
        params=[SyDefParam('cut', '8000')],
        commands=[
            '/op 1',
            '/wm noise',
            '/amp 0.5',
            '/ft hp',
            '/cut $cut',
            '/res 20',
            '/atk 0.001',
            '/dec 0.06',
            '/sus 0.0',
            '/rel 0.04',
        ],
        description='Hihat from noise',
    )

    # =================================================================
    #  FULL PRESETS (3-4 params, deeper control)
    # =================================================================

    # --- Basic sine tone ---
    presets['sine'] = SyDef(
        name='sine',
        params=[SyDefParam('freq', '440'), SyDefParam('dur', '1')],
        commands=[
            '/op 1',
            '/wm sine',
            '/fr $freq',
            '/amp 0.8',
            '/tone $freq $dur',
        ],
        description='Simple sine tone',
    )

    # --- Acid bass ---
    presets['acid'] = SyDef(
        name='acid',
        params=[
            SyDefParam('freq', '110'),
            SyDefParam('cutoff', '600'),
            SyDefParam('res', '80'),
            SyDefParam('dur', '0.5'),
        ],
        commands=[
            '/op 1',
            '/wm saw',
            '/fr $freq',
            '/amp 0.9',
            '/ft acid',
            '/cut $cutoff',
            '/res $res',
            '/atk 0.005',
            '/dec 0.2',
            '/sus 0.3',
            '/rel 0.15',
            '/tone $freq $dur',
        ],
        description='303-style acid bass',
    )

    # --- Pad ---
    presets['pad'] = SyDef(
        name='pad',
        params=[
            SyDefParam('freq', '220'),
            SyDefParam('voices', '4'),
            SyDefParam('detune', '2'),
            SyDefParam('dur', '3'),
        ],
        commands=[
            '/op 1',
            '/wm saw',
            '/fr $freq',
            '/amp 0.5',
            '/vc $voices',
            '/dt $detune',
            '/rand 25',
            '/va unison',
            '/ft lp',
            '/cut 2000',
            '/res 20',
            '/atk 0.5',
            '/dec 0.3',
            '/sus 0.8',
            '/rel 1.0',
            '/tone $freq $dur',
        ],
        description='Detuned pad with filter',
    )

    # --- Pluck ---
    presets['pluck'] = SyDef(
        name='pluck',
        params=[
            SyDefParam('freq', '440'),
            SyDefParam('bright', '4000'),
            SyDefParam('dur', '0.8'),
        ],
        commands=[
            '/op 1',
            '/wm tri',
            '/fr $freq',
            '/amp 0.8',
            '/ft lp',
            '/cut $bright',
            '/res 30',
            '/atk 0.001',
            '/dec 0.15',
            '/sus 0.2',
            '/rel 0.4',
            '/tone $freq $dur',
        ],
        description='Plucked string sound',
    )

    # --- Kick ---
    presets['kick'] = SyDef(
        name='kick',
        params=[
            SyDefParam('freq', '55'),
            SyDefParam('punch', '200'),
            SyDefParam('dur', '0.4'),
        ],
        commands=[
            '/op 1',
            '/wm sine',
            '/fr $freq',
            '/amp 1.0',
            '/atk 0.001',
            '/dec 0.15',
            '/sus 0.0',
            '/rel 0.2',
            '/tone $freq $dur',
        ],
        description='Basic kick drum',
    )

    return presets


def load_factory_presets(session: Session) -> int:
    """Load factory presets into session (only if not already defined)."""
    _ensure_sydef_attrs(session)
    presets = _make_factory_presets()
    loaded = 0
    for name, sd in presets.items():
        if name not in session.sydefs:
            session.sydefs[name] = sd
            loaded += 1
    # Record factory names so auto-save can exclude them
    if not hasattr(session, '_factory_sydef_names'):
        session._factory_sydef_names = []
    session._factory_sydef_names = list(presets.keys())

    # Auto-load user SyDefs from disk
    user_count = _autoload_sydefs(session)
    if user_count > 0:
        loaded += user_count
    return loaded


# ---------------------------------------------------------------------------
# /syt alias — defines synth blocks (alias for /sydef)
# ---------------------------------------------------------------------------

def cmd_syt(session: Session, args: List[str]) -> str:
    """Define a synth block (alias for /sydef).

    Usage:
      /syt <name> [param=default ...]   Start defining a synth block
      /syt list                         List all synth blocks
      /syt show <name>                  Show block contents
      /syt del <name>                   Delete a synth block

    Identical to /sydef — defines reusable synth patches.
    """
    return cmd_sydef(session, args)


# ---------------------------------------------------------------------------
# Command registration
# ---------------------------------------------------------------------------

def get_sydef_commands() -> Dict[str, Any]:
    """Return command dict for registration in bmdma.py."""
    return {
        'sydef': cmd_sydef,
        'sd': cmd_sydef,       # short alias
        'synthdef': cmd_sydef,  # long alias
        'syt': cmd_syt,        # synth block alias
        'use': cmd_use,
        'u': cmd_use,          # short alias
    }
