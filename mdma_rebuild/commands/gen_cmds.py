"""Generative system commands — Phase 4.

Commands:
  /beat       — drum beat generation (4.4)
  /loop       — full loop generation (4.3)
  /xform      — musical transformations (4.5)
  /adapt      — pattern adaptation (4.2)
  /theory     — music theory queries
  /gen2       — improved content generation (4.1)

Object Model Integration:
  All generation commands now produce first-class objects in the
  ObjectRegistry alongside the audio output.  The --name parameter
  can be used to assign a custom name.  See OBJECT_MODEL_SPEC.md.
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

try:
    from ..core.session import Session
except Exception:
    Session = object

# Lazy imports to avoid circular deps
def _mt():
    from ..dsp import music_theory
    return music_theory

def _bg():
    from ..dsp import beat_gen
    return beat_gen

def _lg():
    from ..dsp import loop_gen
    return loop_gen

def _tf():
    from ..dsp import transforms
    return transforms

def _objects():
    from ..core import objects
    return objects


def _parse_name_arg(args: list) -> Optional[str]:
    """Extract --name <value> from args, return the name or None."""
    for i, arg in enumerate(args):
        if arg == '--name' and i + 1 < len(args):
            return args[i + 1]
        if isinstance(arg, str) and arg.startswith('--name='):
            return arg.split('=', 1)[1]
    return None


def _register_audio_clip(session, audio, source_label: str,
                          name: str = '', source_id: str = '',
                          source_params: Optional[dict] = None) -> str:
    """Create and register an AudioClip object. Returns the object ID."""
    obj = _objects()
    clip = obj.AudioClip(
        name=name,
        data=audio,
        sample_rate=session.sample_rate,
        duration_seconds=len(audio) / session.sample_rate,
        bit_depth=24,
        render_source_id=source_id,
        source_params=source_params or {},
    )
    return session.object_registry.register(clip, auto_name=not name)


def _apply_route(session: Session, audio, args: list, default_source: str = 'generated') -> str:
    """Apply route= parameter from args. Returns routing info string.

    Supported: route=working (default), route=track, route=track:N, route=buffer:N
    """
    route = 'working'
    for arg in args:
        if isinstance(arg, str) and arg.lower().startswith('route='):
            route = arg.split('=', 1)[1].lower()
            break

    extra = ''
    if route.startswith('track'):
        parts = route.split(':')
        if len(parts) > 1:
            try:
                track_n = int(parts[1])
                session.ensure_track_index(track_n)
                session.push_undo('track', track_n - 1)
                start, end = session.write_to_track(audio, track_idx=track_n)
                extra = f"\n  Routed to track {track_n} at {start / session.sample_rate:.2f}s"
            except (ValueError, AttributeError):
                pass
        else:
            session.push_undo('track', session.current_track_index)
            start, end = session.write_to_track(audio)
            t_n = session.current_track_index + 1
            extra = f"\n  Routed to track {t_n} at {start / session.sample_rate:.2f}s"
    elif route.startswith('buffer'):
        parts = route.split(':')
        if len(parts) > 1:
            try:
                buf_n = int(parts[1])
                session.store_in_buffer(buf_n, audio)
                extra = f"\n  Routed to buffer {buf_n}"
            except ValueError:
                pass

    return extra


# ======================================================================
# /beat — drum beat generation (4.4)
# ======================================================================

def cmd_beat(session: Session, args: List[str]) -> str:
    """Generate drum beats from genre templates.

    /beat                     — list genres
    /beat <genre>             — generate 1-bar beat
    /beat <genre> <bars>      — generate N-bar beat
    /beat <genre> <bars> <bpm> — custom BPM
    /beat fill <type>         — generate a fill (buildup/breakdown/roll/crash)
    /beat humanize <amount>   — humanize current beat pattern
    /beat list                — list all generators and variants
    """
    bg = _bg()

    if not args:
        genres = bg.list_genres()
        return ("BEAT GENERATOR — Genre Templates\n"
                f"  Available: {', '.join(genres)}\n"
                f"  Usage: /beat <genre> [bars] [bpm]\n"
                f"  Fills: /beat fill buildup|breakdown|roll|crash\n"
                f"  Info:  /beat list")

    sub = args[0].lower()

    if sub == 'list':
        lines = ["BEAT GENERATOR — Sound Library\n"]
        for name, info in sorted(bg.GENERATORS.items()):
            vars_str = ', '.join(info['variants']) if info['variants'] else '(none)'
            lines.append(f"  {name:12s}  {info['desc']:30s}  variants: {vars_str}")
        lines.append(f"\n  Total: {len(bg.GENERATORS)} generators")
        return '\n'.join(lines)

    if sub == 'fill':
        fill_type = args[1] if len(args) > 1 else 'buildup'
        last_genre = getattr(session, '_last_beat_genre', 'house')
        template = bg.GENRE_TEMPLATES.get(last_genre, bg.GENRE_TEMPLATES['house'])
        import copy
        fp = bg.generate_fill(copy.deepcopy(template), fill_type,
                               seed=random.randint(0, 99999))
        audio = bg.render_beat(fp, bars=1, sr=session.sample_rate)
        session.last_buffer = audio
        session.set_working_buffer(audio, f'beat_fill_{fill_type}')
        dur = len(audio) / session.sample_rate
        return (f"OK: Generated {fill_type} fill — {dur:.2f}s, "
                f"peak {np.max(np.abs(audio)):.3f}")

    if sub == 'humanize':
        amt = float(args[1]) if len(args) > 1 else 5.0
        last_genre = getattr(session, '_last_beat_genre', 'house')
        template = bg.GENRE_TEMPLATES.get(last_genre, bg.GENRE_TEMPLATES['house'])
        import copy
        hp = bg.humanize_pattern(copy.deepcopy(template), amt, amt)
        bars = getattr(session, '_last_beat_bars', 1)
        audio = bg.render_beat(hp, bars=bars, sr=session.sample_rate)
        session.last_buffer = audio
        session.set_working_buffer(audio, f'beat_humanized')
        return f"OK: Humanized {last_genre} beat — timing: {amt:.0f}%, velocity: {amt:.0f}%"

    # Genre-based generation
    genre = sub
    if genre not in bg.GENRE_TEMPLATES:
        # Fuzzy match
        matches = [g for g in bg.GENRE_TEMPLATES if g.startswith(genre)]
        if matches:
            genre = matches[0]
        else:
            return (f"ERROR: unknown genre '{genre}'. "
                    f"Available: {', '.join(bg.list_genres())}")

    bars = int(args[1]) if len(args) > 1 else 1
    bars = max(1, min(32, bars))

    import copy
    template = copy.deepcopy(bg.GENRE_TEMPLATES[genre])

    if len(args) > 2:
        try:
            template.bpm = float(args[2])
        except ValueError:
            pass

    audio = bg.render_beat(template, bars=bars, sr=session.sample_rate)
    session.last_buffer = audio
    session.set_working_buffer(audio, f'beat_{genre}_{bars}bar')
    session._last_beat_genre = genre
    session._last_beat_bars = bars
    dur = len(audio) / session.sample_rate
    peak = np.max(np.abs(audio))
    route_info = _apply_route(session, audio, args, f'beat_{genre}')

    # --- Register BeatPattern object ---
    obj = _objects()
    custom_name = _parse_name_arg(args)
    hits = []
    for step_idx, hit_list in template.grid.items():
        for h in hit_list:
            hits.append(obj.DrumHit(
                instrument=h.instrument,
                step=step_idx,
                velocity=int(h.velocity),
            ))
    beat_obj = obj.BeatPattern(
        name=custom_name or '',
        hits=hits,
        steps=template.steps,
        genre=genre,
        bars=bars,
        bpm=template.bpm,
        swing=template.swing,
        source_params={'genre': genre, 'bars': bars, 'bpm': template.bpm},
    )
    beat_id = session.object_registry.register(beat_obj, auto_name=not custom_name)
    clip_id = _register_audio_clip(
        session, audio, f'beat_{genre}',
        source_id=beat_id,
        source_params={'genre': genre, 'bars': bars, 'bpm': template.bpm},
    )
    obj_name = session.object_registry.get(beat_id).name

    return (f"OK: {genre} beat — {bars} bar(s), {template.bpm:.0f} BPM, "
            f"{dur:.2f}s, peak {peak:.3f}\n"
            f"  Registered: {obj_name} (BeatPattern){route_info}")


# ======================================================================
# /loop — full loop generation (4.3)
# ======================================================================

def cmd_loop(session: Session, args: List[str]) -> str:
    """Generate a full loop with multiple layers.

    /loop                          — show help
    /loop <genre>                  — 4-bar loop (drums only)
    /loop <genre> full             — drums + bass + chords + melody
    /loop <genre> <layers...>      — custom layers
    /loop <genre> bars=N bpm=M key=C4 scale=minor mood=dark
    """
    lg = _lg()
    bg = _bg()

    if not args:
        genres = bg.list_genres()
        return ("LOOP GENERATOR\n"
                f"  Genres: {', '.join(genres)}\n"
                "  Layers: drums, bass, chords, melody\n"
                "  Usage:  /loop <genre> [layers...] [key=C4] "
                "[scale=minor] [mood=dark] [bars=4] [bpm=128]\n"
                "  Moods:  neutral, bright, dark, aggressive, chill\n"
                "  Quick:  /loop house full\n"
                "  Quick:  /loop trap drums bass")

    genre = args[0].lower()

    # Parse keyword args and layer names
    spec = lg.LoopSpec(genre=genre)
    layers = []
    for arg in args[1:]:
        low = arg.lower()
        if low == 'full':
            layers = ['drums', 'bass', 'chords', 'melody']
        elif '=' in low:
            key, val = low.split('=', 1)
            if key == 'bars':
                spec.bars = max(1, min(32, int(val)))
            elif key == 'bpm':
                spec.bpm = float(val)
            elif key == 'key':
                try:
                    mt = _mt()
                    spec.root_note = mt.note_name_to_midi(val.upper())
                except Exception:
                    spec.root_note = 60
            elif key == 'scale':
                spec.scale = val
            elif key == 'mood':
                spec.mood = val
            elif key == 'humanize':
                spec.humanize = float(val)
            elif key == 'seed':
                spec.seed = int(val)
        elif low in ('drums', 'beat', 'rhythm', 'bass', 'bassline', 'sub',
                     'chords', 'pad', 'harmony', 'melody', 'lead', 'top'):
            layers.append(low)

    if not layers:
        layers = ['drums']
    spec.layers = layers

    audio = lg.generate_loop(spec, sr=session.sample_rate)
    session.last_buffer = audio
    source = f'loop_{genre}_{"_".join(layers)}'
    session.set_working_buffer(audio, source)
    dur = len(audio) / session.sample_rate
    peak = np.max(np.abs(audio))
    info = lg.format_loop_info(spec)
    route_info = _apply_route(session, audio, args, source)

    # --- Register Loop object ---
    obj = _objects()
    custom_name = _parse_name_arg(args)
    loop_obj = obj.Loop(
        name=custom_name or '',
        layers={layer: '' for layer in layers},  # Layer IDs filled when sub-objects tracked
        bars=spec.bars,
        bpm=spec.bpm if spec.bpm > 0 else session.bpm,
        genre=genre,
        source_params={
            'genre': genre, 'layers': layers, 'bars': spec.bars,
            'bpm': spec.bpm, 'root_note': spec.root_note,
            'scale': spec.scale, 'mood': spec.mood,
        },
    )
    loop_id = session.object_registry.register(loop_obj, auto_name=not custom_name)
    _register_audio_clip(
        session, audio, source,
        source_id=loop_id,
        source_params=loop_obj.source_params,
    )
    obj_name = session.object_registry.get(loop_id).name

    return (f"OK: {info}\n  Duration: {dur:.2f}s, peak: {peak:.3f}\n"
            f"  Registered: {obj_name} (Loop){route_info}")


# ======================================================================
# /xform — musical transformations (4.5)
# ======================================================================

def cmd_xform(session: Session, args: List[str]) -> str:
    """Apply musical transformations to the working buffer.

    /xform                        — list transforms
    /xform <name>                 — apply named transform
    /xform <name> <param=value>   — apply with parameters
    /xform preset <name>          — apply a transform preset
    /xform presets                — list presets
    """
    tf = _tf()

    if not args:
        audio_t = tf.list_audio_transforms()
        note_t = tf.list_note_transforms()
        presets = tf.list_transform_presets()
        return ("MUSICAL TRANSFORMS\n"
                f"  Audio:   {', '.join(audio_t)}\n"
                f"  Note:    {', '.join(note_t)}\n"
                f"  Presets: {', '.join(presets['audio'])}\n"
                "  Usage:   /xform <name> [param=value ...]\n"
                "  Preset:  /xform preset <name>")

    sub = args[0].lower()

    if sub == 'presets':
        presets = tf.list_transform_presets()
        lines = ["TRANSFORM PRESETS\n", "Audio:"]
        for name in presets['audio']:
            steps = tf.AUDIO_TRANSFORM_PRESETS[name]
            desc = ' → '.join(s[0] for s in steps)
            lines.append(f"  {name:15s} {desc}")
        lines.append("\nNote:")
        for name in presets['note']:
            steps = tf.TRANSFORM_PRESETS[name]
            desc = ' → '.join(s[0] for s in steps)
            lines.append(f"  {name:15s} {desc}")
        return '\n'.join(lines)

    if sub == 'preset':
        preset_name = args[1] if len(args) > 1 else 'reverse'
        if preset_name in tf.AUDIO_TRANSFORM_PRESETS:
            transforms = tf.AUDIO_TRANSFORM_PRESETS[preset_name]
            buf = session.ensure_working_buffer()
            result = tf.apply_audio_transforms(
                buf, transforms, session.sample_rate, session.bpm)
            session.last_buffer = result
            session.set_working_buffer(result, f'xform_{preset_name}')

            # --- Register AudioClip object ---
            custom_name = _parse_name_arg(args)
            clip_id = _register_audio_clip(
                session, result, f'xform_{preset_name}',
                name=custom_name or '',
                source_params={'preset': preset_name},
            )
            obj_name = session.object_registry.get(clip_id).name

            return (f"OK: Applied preset '{preset_name}' — "
                    f"{len(result)/session.sample_rate:.2f}s\n"
                    f"  Registered: {obj_name} (AudioClip)")
        elif preset_name in tf.TRANSFORM_PRESETS:
            return (f"OK: Note preset '{preset_name}' loaded. "
                    f"Use /mel or /pat to render notes.")
        else:
            return (f"ERROR: unknown preset '{preset_name}'. "
                    f"Use /xform presets to list.")

    # Direct transform
    transform_name = sub
    params = {}
    for arg in args[1:]:
        if '=' in arg:
            k, v = arg.split('=', 1)
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = v

    if transform_name in ('retrograde', 'reverse'):
        buf = session.ensure_working_buffer()
        result = tf.audio_retrograde(buf)
    elif transform_name in ('augmentation', 'stretch', 'halftime'):
        factor = params.get('factor', 2.0)
        buf = session.ensure_working_buffer()
        result = tf.audio_augmentation(buf, factor, session.sample_rate)
    elif transform_name in ('diminution', 'compress', 'doubletime'):
        factor = params.get('factor', 0.5)
        buf = session.ensure_working_buffer()
        result = tf.audio_diminution(buf, factor, session.sample_rate)
    elif transform_name in ('pitch_shift', 'pitch'):
        semitones = params.get('semitones', 0)
        buf = session.ensure_working_buffer()
        result = tf.audio_pitch_shift(buf, semitones, session.sample_rate)
    elif transform_name == 'stutter':
        buf = session.ensure_working_buffer()
        result = tf.audio_stutter(
            buf, int(params.get('repeats', 4)),
            params.get('grain_beats', 0.25),
            session.bpm, session.sample_rate)
    elif transform_name in ('chop', 'chop_rearrange'):
        buf = session.ensure_working_buffer()
        result = tf.audio_chop_and_rearrange(
            buf, int(params.get('n_slices', 8)),
            seed=int(params.get('seed', random.randint(0, 99999))))
    elif transform_name in ('freeze', 'granular_freeze'):
        buf = session.ensure_working_buffer()
        result = tf.audio_granular_freeze(
            buf, params.get('position', 0.5),
            int(params.get('grain_size', 2048)),
            int(params.get('n_grains', 50)),
            session.sample_rate)
    elif transform_name == 'inversion':
        buf = session.ensure_working_buffer()
        result = tf.audio_inversion(buf, session.sample_rate)
    else:
        return (f"ERROR: unknown transform '{transform_name}'. "
                f"Use /xform to list available transforms.")

    session.last_buffer = result
    session.set_working_buffer(result, f'xform_{transform_name}')
    dur = len(result) / session.sample_rate

    # --- Register AudioClip object ---
    custom_name = _parse_name_arg(args)
    clip_id = _register_audio_clip(
        session, result, f'xform_{transform_name}',
        name=custom_name or '',
        source_params={'transform': transform_name, **params},
    )
    obj_name = session.object_registry.get(clip_id).name

    return (f"OK: Applied '{transform_name}' — {dur:.2f}s\n"
            f"  Registered: {obj_name} (AudioClip)")


# ======================================================================
# /adapt — pattern adaptation (4.2)
# ======================================================================

def cmd_adapt(session: Session, args: List[str]) -> str:
    """Adapt existing audio/patterns to new keys, scales, styles.

    /adapt key <note> [scale]     — adapt working buffer to new key
    /adapt scale <name>           — change scale while keeping key
    /adapt tempo <bpm>            — change tempo
    /adapt style <genre>          — re-render in new genre style
    /adapt develop [technique...] — motivic development
    /adapt detect                 — detect key of working buffer
    """
    mt = _mt()
    tf = _tf()

    if not args:
        return ("PATTERN ADAPTATION\n"
                "  /adapt key C4 minor    — adapt to key\n"
                "  /adapt scale blues     — change scale\n"
                "  /adapt tempo 140       — change tempo\n"
                "  /adapt style trap      — re-render in genre\n"
                "  /adapt develop         — motivic development\n"
                "  /adapt detect          — detect current key\n"
                f"  Scales:  {', '.join(mt.list_scales()[:12])}...\n"
                f"  Chords:  {', '.join(mt.list_chords()[:12])}...")

    sub = args[0].lower()

    if sub == 'detect':
        buf = session.ensure_working_buffer()
        # Simple frequency-based key detection
        from ..dsp.pattern import detect_fundamental_frequency
        freq = detect_fundamental_frequency(buf, session.sample_rate)
        if freq > 0:
            midi = mt.freq_to_midi(freq)
            note = mt.midi_to_note_name(midi)
            return f"Detected fundamental: {freq:.1f} Hz ({note}, MIDI {midi})"
        return "Could not detect fundamental frequency."

    if sub == 'key':
        if len(args) < 2:
            return "Usage: /adapt key <note> [scale]"
        try:
            new_root = mt.note_name_to_midi(args[1].upper())
        except ValueError:
            return f"ERROR: invalid note '{args[1]}'"
        new_scale = args[2].lower() if len(args) > 2 else 'major'
        if new_scale not in mt.SCALES:
            return f"ERROR: unknown scale '{new_scale}'. Use: {', '.join(mt.list_scales()[:10])}..."

        buf = session.ensure_working_buffer()
        # Pitch-shift to new root
        from ..dsp.pattern import detect_fundamental_frequency
        current_freq = detect_fundamental_frequency(buf, session.sample_rate)
        if current_freq > 0:
            current_midi = mt.freq_to_midi(current_freq)
            shift = (new_root % 12) - (current_midi % 12)
            if shift > 6:
                shift -= 12
            elif shift < -6:
                shift += 12
            result = tf.audio_pitch_shift(buf, shift, session.sample_rate)
            session.last_buffer = result
            session.set_working_buffer(result, f'adapt_key_{args[1]}')

            # --- Register AudioClip object ---
            custom_name = _parse_name_arg(args)
            clip_id = _register_audio_clip(
                session, result, f'adapt_key_{args[1]}',
                name=custom_name or '',
                source_params={'key': args[1], 'scale': new_scale, 'shift': shift},
            )
            obj_name = session.object_registry.get(clip_id).name

            return (f"OK: Adapted to {mt.midi_to_note_name(new_root)} {new_scale} "
                    f"(shifted {shift:+d} semitones)\n"
                    f"  Registered: {obj_name} (AudioClip)")
        return "ERROR: could not detect pitch for adaptation."

    if sub == 'scale':
        scale_name = args[1].lower() if len(args) > 1 else 'minor'
        if scale_name not in mt.SCALES:
            return f"ERROR: unknown scale. Available: {', '.join(mt.list_scales()[:15])}..."
        return f"OK: Scale set to '{scale_name}'. Use /mel or /pat to render with new scale."

    if sub == 'tempo':
        if len(args) < 2:
            return "Usage: /adapt tempo <bpm>"
        try:
            new_bpm = float(args[1])
        except ValueError:
            return f"ERROR: invalid BPM '{args[1]}'"
        ratio = session.bpm / new_bpm
        buf = session.ensure_working_buffer()
        result = tf.audio_augmentation(buf, ratio, session.sample_rate)
        old_bpm = session.bpm
        session.bpm = new_bpm
        session.last_buffer = result
        session.set_working_buffer(result, f'adapt_tempo_{new_bpm:.0f}')

        # --- Register AudioClip object ---
        custom_name = _parse_name_arg(args)
        clip_id = _register_audio_clip(
            session, result, f'adapt_tempo_{new_bpm:.0f}',
            name=custom_name or '',
            source_params={'old_bpm': old_bpm, 'new_bpm': new_bpm, 'ratio': ratio},
        )
        obj_name = session.object_registry.get(clip_id).name

        return (f"OK: Tempo adapted {old_bpm:.0f} → {new_bpm:.0f} BPM "
                f"(ratio {ratio:.3f}, {len(result)/session.sample_rate:.2f}s)\n"
                f"  Registered: {obj_name} (AudioClip)")

    if sub == 'style':
        bg = _bg()
        genre = args[1].lower() if len(args) > 1 else 'house'
        if genre not in bg.GENRE_TEMPLATES:
            return f"ERROR: unknown genre. Available: {', '.join(bg.list_genres())}"
        import copy
        template = copy.deepcopy(bg.GENRE_TEMPLATES[genre])
        template.bpm = session.bpm
        audio = bg.render_beat(template, bars=4, sr=session.sample_rate)
        session.last_buffer = audio
        session.set_working_buffer(audio, f'adapt_style_{genre}')
        dur = len(audio) / session.sample_rate

        # --- Register AudioClip object ---
        custom_name = _parse_name_arg(args)
        clip_id = _register_audio_clip(
            session, audio, f'adapt_style_{genre}',
            name=custom_name or '',
            source_params={'genre': genre, 'bpm': session.bpm},
        )
        obj_name = session.object_registry.get(clip_id).name

        return (f"OK: Re-rendered in {genre} style — {dur:.2f}s at {session.bpm:.0f} BPM\n"
                f"  Registered: {obj_name} (AudioClip)")

    if sub == 'develop':
        techniques = args[1:] if len(args) > 1 else None
        buf = session.ensure_working_buffer()
        # Chain multiple audio transforms for motivic development
        transforms_list = []
        if techniques:
            for t in techniques:
                if t in ('retrograde', 'reverse'):
                    transforms_list.append(('retrograde', {}))
                elif t in ('augmentation', 'stretch'):
                    transforms_list.append(('augmentation', {'factor': 1.5}))
                elif t in ('diminution', 'compress'):
                    transforms_list.append(('diminution', {'factor': 0.75}))
                elif t.startswith('pitch'):
                    transforms_list.append(('pitch_shift', {'semitones': 7}))
                elif t == 'stutter':
                    transforms_list.append(('stutter', {'repeats': 4}))
                elif t in ('chop', 'glitch'):
                    transforms_list.append(('chop_rearrange', {'n_slices': 8}))
                elif t == 'freeze':
                    transforms_list.append(('granular_freeze', {'position': 0.5}))
        else:
            # Auto-develop: pick interesting transforms
            rng = random.Random()
            techniques_pool = [
                ('retrograde', {}),
                ('augmentation', {'factor': 1.3}),
                ('pitch_shift', {'semitones': rng.choice([3, 4, 5, 7])}),
            ]
            transforms_list = rng.sample(techniques_pool,
                                          k=min(2, len(techniques_pool)))

        result = tf.apply_audio_transforms(
            buf, transforms_list, session.sample_rate, session.bpm)
        session.last_buffer = result
        names = [t[0] for t in transforms_list]
        session.set_working_buffer(result, f'develop_{"_".join(names)}')

        # --- Register AudioClip object ---
        custom_name = _parse_name_arg(args)
        clip_id = _register_audio_clip(
            session, result, f'develop_{"_".join(names)}',
            name=custom_name or '',
            source_params={'techniques': names},
        )
        obj_name = session.object_registry.get(clip_id).name

        return (f"OK: Motivic development applied: {' → '.join(names)} — "
                f"{len(result)/session.sample_rate:.2f}s\n"
                f"  Registered: {obj_name} (AudioClip)")

    return f"ERROR: unknown subcommand '{sub}'. Use /adapt for help."


# ======================================================================
# /theory — music theory queries
# ======================================================================

def cmd_theory(session: Session, args: List[str]) -> str:
    """Query music theory — scales, chords, progressions.

    /theory scales              — list all scales
    /theory chords              — list all chords
    /theory progressions        — list chord progressions
    /theory scale <name>        — show scale degrees
    /theory chord <name> [root] — show chord notes
    /theory prog <name> [root] [scale] — show progression chords
    /theory diatonic [root] [scale]     — show diatonic chords
    /theory key <notes...>      — detect key from notes
    """
    mt = _mt()

    if not args:
        return ("MUSIC THEORY\n"
                "  /theory scales     — list scales\n"
                "  /theory chords     — list chords\n"
                "  /theory progressions — list progressions\n"
                "  /theory scale <name> — show scale intervals\n"
                "  /theory chord <name> [root] — show chord notes\n"
                "  /theory prog <name> [root] [scale] — resolve progression\n"
                "  /theory diatonic [root] [scale] — show diatonic chords")

    sub = args[0].lower()

    if sub == 'scales':
        lines = ["AVAILABLE SCALES\n"]
        for name, intervals in sorted(mt.SCALES.items()):
            lines.append(f"  {name:20s}  {' '.join(str(i) for i in intervals)}")
        return '\n'.join(lines)

    if sub == 'chords':
        lines = ["AVAILABLE CHORDS\n"]
        for name, intervals in sorted(mt.CHORDS.items()):
            lines.append(f"  {name:10s}  {' '.join(str(i) for i in intervals)}")
        return '\n'.join(lines)

    if sub in ('progressions', 'progs'):
        lines = ["CHORD PROGRESSIONS\n"]
        for name, prog in sorted(mt.PROGRESSIONS.items()):
            desc = ' | '.join(f"{d}:{q}" for d, q in prog[:6])
            if len(prog) > 6:
                desc += '...'
            lines.append(f"  {name:18s}  {desc}")
        return '\n'.join(lines)

    if sub == 'scale':
        scale_name = args[1].lower() if len(args) > 1 else 'major'
        root = 'C4'
        if len(args) > 2:
            root = args[2]
        try:
            root_midi = mt.note_name_to_midi(root)
        except ValueError:
            root_midi = 60
        try:
            notes = mt.get_scale(root_midi, scale_name)
        except ValueError as e:
            return f"ERROR: {e}"
        names = [mt.midi_to_note_name(n) for n in notes]
        intervals = mt.SCALES[scale_name]
        return (f"Scale: {scale_name} from {mt.midi_to_note_name(root_midi)}\n"
                f"  Intervals: {' '.join(str(i) for i in intervals)}\n"
                f"  Notes:     {' '.join(names)}")

    if sub == 'chord':
        chord_name = args[1].lower() if len(args) > 1 else 'maj'
        root = args[2] if len(args) > 2 else 'C4'
        try:
            root_midi = mt.note_name_to_midi(root)
        except ValueError:
            root_midi = 60
        try:
            notes = mt.get_chord(root_midi, chord_name)
        except ValueError as e:
            return f"ERROR: {e}"
        names = [mt.midi_to_note_name(n) for n in notes]
        freqs = [mt.midi_to_freq(n) for n in notes]
        return (f"Chord: {chord_name} rooted at {mt.midi_to_note_name(root_midi)}\n"
                f"  MIDI:  {' '.join(str(n) for n in notes)}\n"
                f"  Notes: {' '.join(names)}\n"
                f"  Hz:    {' '.join(f'{f:.1f}' for f in freqs)}")

    if sub == 'prog':
        prog_name = args[1] if len(args) > 1 else 'I_V_vi_IV'
        root = args[2] if len(args) > 2 else 'C4'
        scale = args[3] if len(args) > 3 else 'major'
        try:
            root_midi = mt.note_name_to_midi(root)
        except ValueError:
            root_midi = 60
        try:
            chords = mt.resolve_progression(root_midi, scale, prog_name)
        except ValueError as e:
            return f"ERROR: {e}"
        lines = [f"Progression: {prog_name} in {mt.midi_to_note_name(root_midi)} {scale}\n"]
        for i, chord in enumerate(chords):
            names = [mt.midi_to_note_name(n) for n in chord]
            lines.append(f"  Bar {i+1}: {' '.join(names)}")
        return '\n'.join(lines)

    if sub == 'diatonic':
        root = args[1] if len(args) > 1 else 'C4'
        scale = args[2] if len(args) > 2 else 'major'
        try:
            root_midi = mt.note_name_to_midi(root)
        except ValueError:
            root_midi = 60
        try:
            chords = mt.diatonic_chords(root_midi, scale, seventh=True)
        except Exception as e:
            return f"ERROR: {e}"
        lines = [f"Diatonic chords: {mt.midi_to_note_name(root_midi)} {scale}\n"]
        for i, (chord_root, quality, notes) in enumerate(chords):
            names = [mt.midi_to_note_name(n) for n in notes]
            lines.append(f"  {i+1}. {mt.midi_to_note_name(chord_root):4s} "
                         f"{quality:6s}  {' '.join(names)}")
        return '\n'.join(lines)

    if sub == 'key':
        midi_notes = []
        for arg in args[1:]:
            try:
                midi_notes.append(mt.note_name_to_midi(arg))
            except ValueError:
                try:
                    midi_notes.append(int(arg))
                except ValueError:
                    pass
        if not midi_notes:
            return "Usage: /theory key C4 E4 G4 B4"
        root, scale, conf = mt.detect_key(midi_notes)
        return (f"Detected key: {mt.midi_to_note_name(root)} {scale} "
                f"(confidence: {conf:.0%})")

    return f"ERROR: unknown subcommand '{sub}'. Use /theory for help."


# ======================================================================
# /gen2 — improved content generation (4.1)
# ======================================================================

def cmd_gen2(session: Session, args: List[str]) -> str:
    """Musically-aware content generation.

    /gen2                       — show help
    /gen2 melody [key] [scale] [length] [contour] — generate melody
    /gen2 chord_prog [key] [prog] — generate chord progression audio
    /gen2 bassline [key] [scale] [genre] — generate bass-line
    /gen2 arp <chord> [pattern] — generate arpeggiation
    /gen2 drone [key] [dur]     — generate drone/ambient texture
    """
    mt = _mt()
    tf = _tf()

    if not args:
        return ("CONTENT GENERATOR v2 — Musically-Aware\n"
                "  /gen2 melody [key=C4] [scale=minor] [length=8] [contour=arch]\n"
                "  /gen2 chord_prog [key=C4] [prog=I_V_vi_IV]\n"
                "  /gen2 bassline [key=C4] [scale=minor] [genre=house]\n"
                "  /gen2 arp <chord> [key=C4]\n"
                "  /gen2 drone [key=C4] [dur=4]\n"
                "  Contours: arch, ascending, descending, wave, random")

    sub = args[0].lower()
    # Parse keyword args
    kw = {}
    positional = []
    for arg in args[1:]:
        if '=' in arg:
            k, v = arg.split('=', 1)
            kw[k.lower()] = v
        else:
            positional.append(arg)

    root_str = kw.get('key', 'C4')
    try:
        root_midi = mt.note_name_to_midi(root_str.upper())
    except ValueError:
        root_midi = 60
    scale = kw.get('scale', 'minor')

    if sub == 'melody':
        length = int(kw.get('length', '8'))
        contour = kw.get('contour', 'arch')
        melody_notes = mt.generate_melody_from_scale(
            root_midi, scale, length, contour=contour)
        # Render using session's synth engine
        sr = session.sample_rate
        beat_dur = 60.0 / session.bpm
        total = np.zeros(0)
        for midi_note in melody_notes:
            freq = mt.midi_to_freq(midi_note)
            try:
                tone = session.generate_tone(freq, 1.0, 0.6)
            except Exception:
                t = np.arange(int(sr * beat_dur)) / sr
                tone = np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.2) * 0.5
            total = np.concatenate([total, tone])
        session.last_buffer = total
        session.set_working_buffer(total, f'gen2_melody_{scale}')
        names = [mt.midi_to_note_name(n) for n in melody_notes]
        dur = len(total) / sr

        # --- Register Pattern object ---
        obj = _objects()
        custom_name = _parse_name_arg(args)
        events = []
        for i, midi_note in enumerate(melody_notes):
            events.append(obj.NoteEvent(
                pitch=midi_note, velocity=100,
                start=float(i), duration=1.0,
            ))
        pat = obj.Pattern(
            name=custom_name or '',
            events=events,
            length_beats=float(length),
            scale=scale, root=root_str,
            bpm=session.bpm,
            pattern_kind='melody',
            source_params={'scale': scale, 'key': root_str,
                           'length': length, 'contour': contour},
        )
        pat_id = session.object_registry.register(pat, auto_name=not custom_name)
        _register_audio_clip(session, total, f'gen2_melody_{scale}',
                              source_id=pat_id, source_params=pat.source_params)
        obj_name = session.object_registry.get(pat_id).name

        return (f"OK: Generated {length}-note {scale} melody ({contour})\n"
                f"  Notes: {' '.join(names)}\n"
                f"  Duration: {dur:.2f}s\n"
                f"  Registered: {obj_name} (Pattern)")

    if sub in ('chord_prog', 'chords', 'progression'):
        prog_name = kw.get('prog', positional[0] if positional else 'I_V_vi_IV')
        try:
            chords = mt.resolve_progression(root_midi, scale, prog_name)
        except ValueError as e:
            return f"ERROR: {e}"
        sr = session.sample_rate
        beat_dur = 60.0 / session.bpm
        bar_samples = int(sr * beat_dur * 4)
        total = np.zeros(0)
        prev_voicing = None
        for chord_midi in chords:
            if prev_voicing:
                chord_midi = mt.voice_lead(prev_voicing, chord_midi)
            prev_voicing = chord_midi
            bar = np.zeros(bar_samples)
            t = np.arange(bar_samples) / sr
            for midi_note in chord_midi:
                freq = mt.midi_to_freq(midi_note)
                for dt_cents in [-15, 0, 15]:
                    f = freq * (2.0 ** (dt_cents / 1200.0))
                    tone = np.sin(2 * np.pi * f * t)
                    # Soft pad envelope
                    atk = min(int(sr * 0.08), bar_samples // 4)
                    rel = min(int(sr * 0.12), bar_samples // 4)
                    env = np.ones(bar_samples)
                    if atk > 0:
                        env[:atk] = np.linspace(0, 1, atk)
                    if rel > 0:
                        env[-rel:] = np.linspace(1, 0, rel)
                    bar += tone * env * 0.1 / max(1, len(chord_midi))
            total = np.concatenate([total, bar])
        peak = np.max(np.abs(total))
        if peak > 1e-10:
            total *= 0.9 / peak
        session.last_buffer = total
        session.set_working_buffer(total, f'gen2_chords_{prog_name}')
        dur = len(total) / sr

        # --- Register Pattern object ---
        obj = _objects()
        custom_name = _parse_name_arg(args)
        events = []
        for bar_idx, chord_midi in enumerate(chords):
            for midi_note in chord_midi:
                events.append(obj.NoteEvent(
                    pitch=midi_note, velocity=100,
                    start=float(bar_idx * 4), duration=4.0,
                ))
        pat = obj.Pattern(
            name=custom_name or '',
            events=events,
            length_beats=float(len(chords) * 4),
            scale=scale, root=root_str,
            bpm=session.bpm,
            pattern_kind='chord',
            source_params={'scale': scale, 'key': root_str, 'prog': prog_name},
        )
        pat_id = session.object_registry.register(pat, auto_name=not custom_name)
        _register_audio_clip(session, total, f'gen2_chords_{prog_name}',
                              source_id=pat_id, source_params=pat.source_params)
        obj_name = session.object_registry.get(pat_id).name

        return (f"OK: Generated {prog_name} progression in "
                f"{mt.midi_to_note_name(root_midi)} {scale}\n"
                f"  {len(chords)} chords, {dur:.2f}s\n"
                f"  Registered: {obj_name} (Pattern)")

    if sub == 'bassline':
        genre = kw.get('genre', 'house')
        lg = _lg()
        spec = lg.LoopSpec(genre=genre, bpm=session.bpm,
                            bars=int(kw.get('bars', '4')),
                            root_note=root_midi, scale=scale)
        audio = lg._generate_bassline(spec, session.sample_rate)
        session.last_buffer = audio
        session.set_working_buffer(audio, f'gen2_bass_{genre}')
        dur = len(audio) / session.sample_rate

        # --- Register Pattern object ---
        obj = _objects()
        custom_name = _parse_name_arg(args)
        pat = obj.Pattern(
            name=custom_name or '',
            length_beats=float(spec.bars * 4),
            scale=scale, root=root_str,
            bpm=session.bpm,
            pattern_kind='bassline',
            source_params={'scale': scale, 'key': root_str,
                           'genre': genre, 'bars': spec.bars},
        )
        pat_id = session.object_registry.register(pat, auto_name=not custom_name)
        _register_audio_clip(session, audio, f'gen2_bass_{genre}',
                              source_id=pat_id, source_params=pat.source_params)
        obj_name = session.object_registry.get(pat_id).name

        return (f"OK: Generated {genre} bassline — {dur:.2f}s\n"
                f"  Registered: {obj_name} (Pattern)")

    if sub == 'arp':
        chord_name = positional[0] if positional else 'min7'
        try:
            chord_notes = mt.get_chord(root_midi, chord_name)
        except ValueError as e:
            return f"ERROR: {e}"
        sr = session.sample_rate
        beat_dur = 60.0 / session.bpm
        total = np.zeros(0)
        # Arpeggiate up then down
        arp_seq = chord_notes + list(reversed(chord_notes[1:-1]))
        repeats = int(kw.get('repeats', '2'))
        for _ in range(repeats):
            for midi_note in arp_seq:
                freq = mt.midi_to_freq(midi_note)
                note_len = int(sr * beat_dur * 0.5)
                t = np.arange(note_len) / sr
                tone = np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.1)
                tone += np.sin(2 * np.pi * freq * 2 * t) * 0.2 * np.exp(-t / 0.06)
                total = np.concatenate([total, tone * 0.5])
        session.last_buffer = total
        session.set_working_buffer(total, f'gen2_arp_{chord_name}')
        dur = len(total) / sr
        names = [mt.midi_to_note_name(n) for n in chord_notes]

        # --- Register Pattern object ---
        obj = _objects()
        custom_name = _parse_name_arg(args)
        events = []
        beat_pos = 0.0
        for _ in range(repeats):
            for midi_note in arp_seq:
                events.append(obj.NoteEvent(
                    pitch=midi_note, velocity=100,
                    start=beat_pos, duration=0.5,
                ))
                beat_pos += 0.5
        pat = obj.Pattern(
            name=custom_name or '',
            events=events,
            length_beats=beat_pos,
            scale=scale, root=root_str,
            bpm=session.bpm,
            pattern_kind='arpeggio',
            source_params={'scale': scale, 'key': root_str,
                           'chord': chord_name, 'repeats': repeats},
        )
        pat_id = session.object_registry.register(pat, auto_name=not custom_name)
        _register_audio_clip(session, total, f'gen2_arp_{chord_name}',
                              source_id=pat_id, source_params=pat.source_params)
        obj_name = session.object_registry.get(pat_id).name

        return (f"OK: Arpeggiated {chord_name} ({' '.join(names)}) "
                f"x{repeats} — {dur:.2f}s\n"
                f"  Registered: {obj_name} (Pattern)")

    if sub == 'drone':
        dur_beats = float(kw.get('dur', '4'))
        sr = session.sample_rate
        dur_sec = dur_beats * 60.0 / session.bpm
        n = int(sr * dur_sec)
        t = np.arange(n) / sr
        freq = mt.midi_to_freq(root_midi - 12)  # One octave below root
        out = np.zeros(n)
        # Stack detuned oscillators
        for detune in [-0.1, -0.05, 0, 0.05, 0.1]:
            f = freq * (2.0 ** (detune / 12.0))
            out += np.sin(2 * np.pi * f * t) * 0.15
        # Add fifth
        f5 = freq * 1.5
        out += np.sin(2 * np.pi * f5 * t) * 0.08
        # Slow LFO amplitude modulation
        lfo = 0.7 + 0.3 * np.sin(2 * np.pi * 0.15 * t)
        out *= lfo
        # Fade in/out
        fade = min(int(sr * 0.5), n // 4)
        if fade > 0:
            out[:fade] *= np.linspace(0, 1, fade)
            out[-fade:] *= np.linspace(1, 0, fade)
        peak = np.max(np.abs(out))
        if peak > 1e-10:
            out *= 0.9 / peak
        session.last_buffer = out
        session.set_working_buffer(out, f'gen2_drone')

        # --- Register Pattern object ---
        obj = _objects()
        custom_name = _parse_name_arg(args)
        pat = obj.Pattern(
            name=custom_name or '',
            events=[obj.NoteEvent(pitch=root_midi - 12, velocity=100,
                                   start=0.0, duration=dur_beats)],
            length_beats=dur_beats,
            scale=scale, root=root_str,
            bpm=session.bpm,
            pattern_kind='drone',
            source_params={'scale': scale, 'key': root_str, 'dur': dur_beats},
        )
        pat_id = session.object_registry.register(pat, auto_name=not custom_name)
        _register_audio_clip(session, out, 'gen2_drone',
                              source_id=pat_id, source_params=pat.source_params)
        obj_name = session.object_registry.get(pat_id).name

        return (f"OK: Generated drone in {mt.midi_to_note_name(root_midi)} — {dur_sec:.2f}s\n"
                f"  Registered: {obj_name} (Pattern)")

    return f"ERROR: unknown sub-command '{sub}'. Use /gen2 for help."


# ======================================================================
# Command registration
# ======================================================================

def get_gen_commands() -> dict:
    """Return all generative commands for registration."""
    return {
        'beat': cmd_beat,
        'loop': cmd_loop,
        'xform': cmd_xform, 'transform': cmd_xform,
        'adapt': cmd_adapt,
        'theory': cmd_theory,
        'gen2': cmd_gen2, 'generate': cmd_gen2,
    }
