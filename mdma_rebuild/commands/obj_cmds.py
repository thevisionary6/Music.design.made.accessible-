"""Object registry CLI commands.

Commands:
  /obj list [type]              — list all objects, optionally filtered by type
  /obj info <name|id>           — show full object detail
  /obj rename <name|id> <new>   — rename an object
  /obj tag <name|id> <tag>      — add a tag to an object
  /obj dup <name|id> [new_name] — duplicate an object
  /obj delete <name|id>         — delete an object (with dependency check)
  /obj types                    — show object type summary
  /obj search <query>           — search by name or tag

See docs/specs/OBJECT_MODEL_SPEC.md for the full specification.
"""

from __future__ import annotations

from typing import List

try:
    from ..core.session import Session
except Exception:
    Session = object


def _resolve_object(session, name_or_id: str):
    """Look up an object by name or ID prefix."""
    reg = session.object_registry

    # Try exact ID match
    obj = reg.get(name_or_id)
    if obj:
        return obj

    # Try name match
    obj = reg.get_by_name(name_or_id)
    if obj:
        return obj

    # Try ID prefix match
    for oid, o in reg._objects.items():
        if oid.startswith(name_or_id):
            return o

    # Try partial name match
    results = reg.search(name_or_id)
    if len(results) == 1:
        return results[0]

    return None


def _format_object_summary(obj) -> str:
    """Format a single-line summary of an object."""
    tags = f" [{', '.join(obj.tags)}]" if obj.tags else ""
    return f"  {obj.name:20s}  {obj.obj_type:14s}  v{obj.version}{tags}"


def _format_object_detail(obj) -> str:
    """Format full detail view of an object."""
    lines = [
        f"OBJECT: {obj.name}",
        f"  Type:     {obj.obj_type}",
        f"  ID:       {obj.id}",
        f"  Version:  {obj.version}",
        f"  Created:  {obj.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Modified: {obj.modified_at.strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    if obj.tags:
        lines.append(f"  Tags:     {', '.join(obj.tags)}")
    if obj.notes:
        lines.append(f"  Notes:    {obj.notes}")
    if obj.source_params:
        params_str = ', '.join(f'{k}={v}' for k, v in obj.source_params.items())
        lines.append(f"  Params:   {params_str}")
    if obj.source_object_ids:
        lines.append(f"  Sources:  {', '.join(obj.source_object_ids[:5])}")

    # Type-specific fields
    if obj.obj_type == 'pattern':
        lines.append(f"  Kind:     {obj.pattern_kind}")
        lines.append(f"  Key:      {obj.root} {obj.scale}")
        lines.append(f"  Length:   {obj.length_beats} beats")
        lines.append(f"  BPM:      {obj.bpm}")
        lines.append(f"  Events:   {len(obj.events)} notes")
    elif obj.obj_type == 'beat_pattern':
        lines.append(f"  Genre:    {obj.genre}")
        lines.append(f"  Steps:    {obj.steps}")
        lines.append(f"  Bars:     {obj.bars}")
        lines.append(f"  BPM:      {obj.bpm}")
        lines.append(f"  Swing:    {obj.swing}")
        lines.append(f"  Hits:     {len(obj.hits)}")
    elif obj.obj_type == 'loop':
        lines.append(f"  Genre:    {obj.genre}")
        lines.append(f"  Bars:     {obj.bars}")
        lines.append(f"  BPM:      {obj.bpm}")
        layers_str = ', '.join(obj.layers.keys()) if obj.layers else '(none)'
        lines.append(f"  Layers:   {layers_str}")
    elif obj.obj_type == 'audio_clip':
        lines.append(f"  Duration: {obj.duration_seconds:.2f}s")
        lines.append(f"  Rate:     {obj.sample_rate} Hz")
        lines.append(f"  Depth:    {obj.bit_depth}-bit")
        if obj.render_source_id:
            lines.append(f"  Source:   {obj.render_source_id[:12]}...")
    elif obj.obj_type == 'effect_chain':
        lines.append(f"  Category: {obj.category}")
        lines.append(f"  Effects:  {len(obj.effects)}")
        for i, slot in enumerate(obj.effects):
            lines.append(f"    [{i+1}] {slot.effect_name} (enabled={slot.enabled})")
    elif obj.obj_type == 'patch':
        lines.append(f"  Engine:   {obj.engine}")
        lines.append(f"  Waveform: {obj.waveform}")
        lines.append(f"  Ops:      {len(obj.operators)}")

    return '\n'.join(lines)


def cmd_obj(session: Session, args: List[str]) -> str:
    """Manage the object registry.

    /obj                            — show summary
    /obj list [type]                — list objects
    /obj info <name>                — show object detail
    /obj rename <name> <new_name>   — rename
    /obj tag <name> <tag>           — add tag
    /obj dup <name> [new_name]      — duplicate
    /obj delete <name>              — delete
    /obj types                      — type summary
    /obj search <query>             — search by name/tag
    """
    reg = session.object_registry

    if not args:
        total = reg.count()
        if total == 0:
            return ("OBJECT REGISTRY — Empty\n"
                    "  No objects registered yet.\n"
                    "  Use /beat, /loop, /gen2 to create objects.\n"
                    "  Use /obj list to browse.")
        summary = reg.types_summary()
        lines = [f"OBJECT REGISTRY — {total} object(s)\n"]
        for t, c in sorted(summary.items()):
            lines.append(f"  {t:14s}  {c}")
        lines.append(f"\n  Use /obj list [type] to browse.")
        return '\n'.join(lines)

    sub = args[0].lower()

    # /obj list [type]
    if sub == 'list':
        obj_type = args[1].lower() if len(args) > 1 else None
        objects = reg.list_objects(obj_type)
        if not objects:
            if obj_type:
                return f"No {obj_type} objects in the registry."
            return "Registry is empty. Use /beat, /loop, /gen2 to create objects."
        lines = [f"OBJECTS{' (' + obj_type + ')' if obj_type else ''} "
                 f"— {len(objects)} item(s)\n"]
        for obj in objects:
            lines.append(_format_object_summary(obj))
        return '\n'.join(lines)

    # /obj types
    if sub == 'types':
        summary = reg.types_summary()
        if not summary:
            return "Registry is empty."
        lines = ["OBJECT TYPES\n"]
        for t, c in sorted(summary.items()):
            lines.append(f"  {t:14s}  {c}")
        lines.append(f"\n  Total: {reg.count()}")
        return '\n'.join(lines)

    # /obj info <name|id>
    if sub == 'info':
        if len(args) < 2:
            return "Usage: /obj info <name>"
        obj = _resolve_object(session, args[1])
        if not obj:
            return f"ERROR: object '{args[1]}' not found."
        return _format_object_detail(obj)

    # /obj rename <name|id> <new_name>
    if sub == 'rename':
        if len(args) < 3:
            return "Usage: /obj rename <name> <new_name>"
        obj = _resolve_object(session, args[1])
        if not obj:
            return f"ERROR: object '{args[1]}' not found."
        old_name = obj.name
        try:
            reg.rename(obj.id, args[2])
        except ValueError as e:
            return f"ERROR: {e}"
        return f"OK: Renamed '{old_name}' → '{args[2]}'"

    # /obj tag <name|id> <tag>
    if sub == 'tag':
        if len(args) < 3:
            return "Usage: /obj tag <name> <tag>"
        obj = _resolve_object(session, args[1])
        if not obj:
            return f"ERROR: object '{args[1]}' not found."
        tag = args[2]
        if tag not in obj.tags:
            obj.tags.append(tag)
            obj.touch()
        return f"OK: Tagged '{obj.name}' with '{tag}'. Tags: {', '.join(obj.tags)}"

    # /obj dup <name|id> [new_name]
    if sub in ('dup', 'duplicate'):
        if len(args) < 2:
            return "Usage: /obj dup <name> [new_name]"
        obj = _resolve_object(session, args[1])
        if not obj:
            return f"ERROR: object '{args[1]}' not found."
        new_name = args[2] if len(args) > 2 else ''
        dup = reg.duplicate(obj.id, new_name)
        if dup:
            return f"OK: Duplicated '{obj.name}' → '{dup.name}'"
        return "ERROR: duplication failed."

    # /obj delete <name|id>
    if sub == 'delete':
        if len(args) < 2:
            return "Usage: /obj delete <name>"
        obj = _resolve_object(session, args[1])
        if not obj:
            return f"ERROR: object '{args[1]}' not found."
        # Check dependents
        deps = reg.get_dependents(obj.id)
        if deps:
            dep_names = ', '.join(d.name for d in deps[:5])
            return (f"WARNING: '{obj.name}' is referenced by: {dep_names}. "
                    f"Use /obj delete {args[1]} --force to delete anyway.")
        if len(args) > 2 and args[2] == '--force':
            pass  # Force delete even with deps
        reg.delete(obj.id)
        return f"OK: Deleted '{obj.name}' ({obj.obj_type})"

    # /obj search <query>
    if sub == 'search':
        if len(args) < 2:
            return "Usage: /obj search <query>"
        query = ' '.join(args[1:])
        results = reg.search(query)
        if not results:
            return f"No objects matching '{query}'."
        lines = [f"SEARCH RESULTS for '{query}' — {len(results)} match(es)\n"]
        for obj in results:
            lines.append(_format_object_summary(obj))
        return '\n'.join(lines)

    return f"ERROR: unknown subcommand '{sub}'. Use /obj for help."


def get_obj_commands() -> dict:
    """Return object registry commands for registration."""
    return {
        'obj': cmd_obj,
        'object': cmd_obj,
    }
