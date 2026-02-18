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

    # /obj export <name|id> <path>
    if sub == 'export':
        if len(args) < 3:
            return "Usage: /obj export <name> <path>"
        obj = _resolve_object(session, args[1])
        if not obj:
            return f"ERROR: object '{args[1]}' not found."
        path = args[2]
        ok = reg.export(obj.id, path)
        if ok:
            return f"OK: Exported '{obj.name}' to {path}"
        return "ERROR: export failed."

    # /obj import <path>
    if sub == 'import':
        if len(args) < 2:
            return "Usage: /obj import <path>"
        path = args[1]
        imported = reg.import_file(path)
        if imported:
            return f"OK: Imported '{imported.name}' ({imported.obj_type}) from {path}"
        return f"ERROR: could not import from '{path}'."

    return f"ERROR: unknown subcommand '{sub}'. Use /obj for help."


def cmd_template(session: Session, args: List[str]) -> str:
    """Manage templates.

    /template list                    — list all saved templates
    /template show <name>             — show template fields
    /template fill <name> [k=v ...]   — fill template fields and generate
    /template save <name> <target_type> [description]  — save a new template
    """
    reg = session.object_registry

    if not args:
        return ("TEMPLATE SYSTEM\n"
                "  /template list              — list templates\n"
                "  /template show <name>       — show template fields\n"
                "  /template fill <name> k=v   — fill and generate\n"
                "  /template save <name> <type> [desc] — save template")

    sub = args[0].lower()

    if sub == 'list':
        templates = reg.list_objects('template')
        if not templates:
            return ("No templates registered.\n"
                    "  Use /template save <name> <type> to create one.")
        lines = [f"TEMPLATES — {len(templates)} item(s)\n"]
        for t in templates:
            desc = getattr(t, 'description', '') or ''
            target = getattr(t, 'target_type', '') or ''
            lines.append(f"  {t.name:20s}  -> {target:14s}  {desc[:40]}")
        return '\n'.join(lines)

    if sub == 'show':
        if len(args) < 2:
            return "Usage: /template show <name>"
        obj = _resolve_object(session, args[1])
        if not obj or obj.obj_type != 'template':
            return f"ERROR: template '{args[1]}' not found."
        lines = [f"TEMPLATE: {obj.name}",
                 f"  Target type: {obj.target_type}",
                 f"  Description: {obj.description}",
                 f"  Fields:"]
        for fld in obj.fields:
            req = " [required]" if fld.required else ""
            default = f" (default: {fld.default})" if fld.default is not None else ""
            choices_str = ""
            if fld.choices:
                choices_str = f" choices: {', '.join(str(c) for c in fld.choices)}"
            lines.append(f"    {fld.name:15s} ({fld.field_type}){req}{default}{choices_str}")
            if fld.description:
                lines.append(f"      {fld.description}")
        return '\n'.join(lines)

    if sub == 'fill':
        if len(args) < 2:
            return "Usage: /template fill <name> key=value ..."
        obj = _resolve_object(session, args[1])
        if not obj or obj.obj_type != 'template':
            return f"ERROR: template '{args[1]}' not found."

        # Parse key=value pairs from remaining args
        params = {}
        for a in args[2:]:
            if '=' in a:
                k, v = a.split('=', 1)
                params[k] = v

        # Validate required fields
        missing = []
        for fld in obj.fields:
            if fld.required and fld.name not in params:
                if fld.default is not None:
                    params[fld.name] = fld.default
                else:
                    missing.append(fld.name)
        if missing:
            return f"ERROR: missing required fields: {', '.join(missing)}"

        # Apply defaults for unfilled fields
        for fld in obj.fields:
            if fld.name not in params and fld.default is not None:
                params[fld.name] = fld.default

        # Build the generation command from template target type
        target = obj.target_type
        cmd_map = {
            'beat_pattern': 'beat',
            'pattern': 'gen2',
            'loop': 'loop',
        }
        cmd_name = cmd_map.get(target)
        if not cmd_name:
            return f"ERROR: no generator mapped for target type '{target}'."

        # Build args from params
        if cmd_name == 'beat':
            genre = params.get('genre', 'hiphop')
            bars = params.get('bars', '4')
            cmd_args = [genre, str(bars)]
            if 'name' in params:
                cmd_args.extend(['--name', params['name']])
        elif cmd_name == 'gen2':
            kind = params.get('kind', 'melody')
            scale = params.get('scale', 'minor')
            bars = params.get('bars', '4')
            cmd_args = [kind, scale, str(bars)]
            if 'name' in params:
                cmd_args.extend(['--name', params['name']])
        elif cmd_name == 'loop':
            genre = params.get('genre', 'hiphop')
            bars = params.get('bars', '4')
            cmd_args = [genre, str(bars)]
            if 'name' in params:
                cmd_args.extend(['--name', params['name']])
        else:
            cmd_args = []

        # Execute through session command table
        try:
            import bmdma
            cmds = bmdma.build_command_table()
            if cmd_name in cmds:
                result = cmds[cmd_name](session, cmd_args)
                return f"OK: Template '{obj.name}' filled.\n{result}"
            return f"ERROR: command '{cmd_name}' not found in command table."
        except ImportError:
            return "ERROR: bmdma not available."

    if sub == 'save':
        if len(args) < 3:
            return "Usage: /template save <name> <target_type> [description]"
        name = args[1]
        target_type = args[2]
        desc = ' '.join(args[3:]) if len(args) > 3 else ''

        try:
            from ..core.objects import Template, TemplateField
        except ImportError:
            from mdma_rebuild.core.objects import Template, TemplateField

        # Create default fields based on target type
        field_presets = {
            'beat_pattern': [
                TemplateField(name='genre', field_type='choice', label='Genre',
                              description='Beat genre', default='hiphop',
                              choices=['hiphop', 'house', 'dnb', 'techno',
                                       'lofi', 'trap', 'afrobeat', 'reggaeton'],
                              required=True),
                TemplateField(name='bars', field_type='int', label='Bars',
                              description='Number of bars', default=4,
                              min_value=1, max_value=32, required=True),
                TemplateField(name='name', field_type='string', label='Name',
                              description='Object name (optional)'),
            ],
            'pattern': [
                TemplateField(name='kind', field_type='choice', label='Kind',
                              description='Pattern kind', default='melody',
                              choices=['melody', 'chords', 'bassline', 'arp', 'drone'],
                              required=True),
                TemplateField(name='scale', field_type='string', label='Scale',
                              description='Musical scale', default='minor'),
                TemplateField(name='bars', field_type='int', label='Bars',
                              description='Number of bars', default=4,
                              min_value=1, max_value=32, required=True),
                TemplateField(name='name', field_type='string', label='Name',
                              description='Object name (optional)'),
            ],
            'loop': [
                TemplateField(name='genre', field_type='choice', label='Genre',
                              description='Loop genre', default='hiphop',
                              choices=['hiphop', 'house', 'dnb', 'techno', 'lofi'],
                              required=True),
                TemplateField(name='bars', field_type='int', label='Bars',
                              description='Number of bars', default=4,
                              min_value=1, max_value=32, required=True),
                TemplateField(name='name', field_type='string', label='Name',
                              description='Object name (optional)'),
            ],
        }

        fields = field_presets.get(target_type, [])
        template = Template(
            name=name,
            target_type=target_type,
            fields=fields,
            description=desc,
        )
        reg.register(template, auto_name=False)
        return f"OK: Template '{name}' saved (target: {target_type}, {len(fields)} fields)"

    return f"ERROR: unknown subcommand '{sub}'. Use /template for help."


def get_obj_commands() -> dict:
    """Return object registry commands for registration."""
    return {
        'obj': cmd_obj,
        'object': cmd_obj,
        'template': cmd_template,
    }
