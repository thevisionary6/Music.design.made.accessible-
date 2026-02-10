"""MDMA Pack Commands.

Commands for sample pack management, generation, and dictionary access.

Commands:
- /pack - List/manage sound packs
- /dict - Sample dictionary access
"""

from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.session import Session


def cmd_pack(session: "Session", args: List[str]) -> str:
    """Sound pack management and generation.
    
    Usage:
      /pack                   -> List installed packs
      /pack info <n>          -> Show pack details
      /pack samples <n>       -> List samples in pack
      /pack load <n> <idx>    -> Load sample from pack to buffer
      /pack create <n>        -> Create new empty pack
      /pack save <n> <name>   -> Save buffer to pack as sample
      /pack gen <n>           -> Generate pack with algorithms
      /pack gen <n> <type>    -> Generate specific type
      /pack genai <n> <prompts...> -> Generate with AI
      /pack dict              -> Sample dictionary operations
    
    Generation Types (for /pack gen):
      kicks      - Kick drum variations
      snares     - Snare variations  
      hats       - Hi-hat variations (closed + open)
      percs      - Percussion one-shots
      bass       - Bass synth hits
      leads      - Lead synth stabs
      pads       - Pad textures
      risers     - Risers and transitions
      impacts    - Impact hits
      sweeps     - Filter sweeps
      all        - All types (default)
    
    AI Generation (requires torch, diffusers):
      /pack genai <n> "prompt1" "prompt2" ...
    
    Dictionary Commands:
      /pack dict list [pack]  -> List dictionary entries
      /pack dict get <n>   -> Load from dictionary to buffer
      /pack dict add <n>   -> Save buffer to dictionary
      /pack dict search <q>   -> Search dictionary
      /pack dict clear        -> Clear dictionary
    
    Packs are stored in: ~/Documents/MDMA/packs/
    
    Examples:
      /pack gen my_kit kicks        -> Generate kick variations
      /pack gen drums all           -> Generate full drum kit
      /pack gen synths bass,leads   -> Generate bass and leads
      /pack genai ai_drums "punchy kick" "snappy snare"
      /pack save my_kit custom_kick -> Save buffer to pack
    """
    # Import pack module
    try:
        from ..core.pack import (
            get_packs_dir, list_packs, get_pack_info, get_pack_samples,
            create_pack_manifest, generate_pack, generate_pack_ai,
            get_sample_dictionary, load_sample_to_session, save_buffer_to_dictionary,
        )
        PACK_MODULE = True
    except ImportError:
        PACK_MODULE = False
    
    if not PACK_MODULE:
        # Fallback to user_data module
        try:
            from ..core.user_data import list_packs as list_packs_old, get_pack_samples, get_packs_dir
            packs = list_packs_old()
            if not packs:
                return f"No packs installed. Packs folder: {get_packs_dir()}"
            lines = ["=== SOUND PACKS ===", ""]
            for p in packs:
                lines.append(f"  {p['name']} ({p.get('sample_count', 0)} samples)")
            return '\n'.join(lines)
        except ImportError:
            return "ERROR: Pack module not available"
    
    # No args - list packs
    if not args:
        pack_names = list_packs()
        if not pack_names:
            return f"No packs installed. Packs folder: {get_packs_dir()}\nUse /pack gen <n> to create one."
        
        lines = ["=== SOUND PACKS ===", ""]
        for pname in pack_names:
            info = get_pack_info(pname)
            count = info.get('sample_count', 0) if info else len(get_pack_samples(pname))
            lines.append(f"  {pname} ({count} samples)")
            if info and info.get('author') and info.get('author') not in ('Unknown', 'MDMA User'):
                lines.append(f"    by {info['author']}")
        
        lines.append("")
        lines.append(f"Pack folder: {get_packs_dir()}")
        lines.append("")
        lines.append("Commands: gen, genai, samples, load, save, info, dict")
        return '\n'.join(lines)
    
    sub = args[0].lower()
    
    # === INFO ===
    if sub == 'info' and len(args) > 1:
        pack_name = args[1]
        info = get_pack_info(pack_name)
        samples = get_pack_samples(pack_name)
        if not info and not samples:
            return f"ERROR: pack '{pack_name}' not found"
        
        if not info:
            info = {'name': pack_name}
        
        lines = [
            f"=== PACK: {info.get('name', pack_name)} ===",
            f"Author: {info.get('author', 'Unknown')}",
            f"Version: {info.get('version', '1.0')}",
            f"Description: {info.get('description', 'N/A')}",
            f"Samples: {len(samples)}",
            f"Tags: {', '.join(info.get('tags', [])) or 'none'}",
            f"Path: {get_packs_dir() / pack_name}",
        ]
        return '\n'.join(lines)
    
    # === SAMPLES ===
    elif sub == 'samples' and len(args) > 1:
        pack_name = args[1]
        samples = get_pack_samples(pack_name)
        if not samples:
            return f"ERROR: pack '{pack_name}' not found or empty"
        
        lines = [f"=== SAMPLES IN {pack_name.upper()} ({len(samples)} total) ===", ""]
        for i, s in enumerate(samples[:50]):
            lines.append(f"  [{i:2d}] {s.name}")
        
        if len(samples) > 50:
            lines.append(f"  ... and {len(samples) - 50} more")
        
        lines.append("")
        lines.append(f"Use /pack load {pack_name} <index> to load")
        return '\n'.join(lines)
    
    # === LOAD ===
    elif sub == 'load' and len(args) > 2:
        pack_name = args[1]
        try:
            sample_idx = int(args[2])
        except ValueError:
            return f"ERROR: invalid sample index '{args[2]}'"
        
        samples = get_pack_samples(pack_name)
        if not samples:
            return f"ERROR: pack '{pack_name}' not found"
        if sample_idx < 0 or sample_idx >= len(samples):
            return f"ERROR: index {sample_idx} out of range (0-{len(samples)-1})"
        
        sample_path = samples[sample_idx]
        try:
            import soundfile as sf
            audio, sr = sf.read(str(sample_path))
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample if needed
            if sr != session.sample_rate:
                ratio = session.sample_rate / sr
                new_len = int(len(audio) * ratio)
                x_old = np.linspace(0, 1, len(audio))
                x_new = np.linspace(0, 1, new_len)
                audio = np.interp(x_new, x_old, audio)
            
            audio = audio.astype(np.float64)
            session.last_buffer = audio
            
            # Add to dictionary
            sample_dict = get_sample_dictionary()
            dict_name = f"{pack_name}/{sample_path.stem}"
            sample_dict.add(dict_name, audio, session.sample_rate, pack_name)
            
            dur = len(audio) / session.sample_rate
            return f"OK: loaded '{sample_path.name}' ({dur:.3f}s) to buffer and dictionary"
            
        except ImportError:
            from scipy.io import wavfile
            sr, audio = wavfile.read(str(sample_path))
            if audio.dtype == np.int16:
                audio = audio.astype(np.float64) / 32768.0
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            session.last_buffer = audio.astype(np.float64)
            dur = len(audio) / sr
            return f"OK: loaded '{sample_path.name}' ({dur:.3f}s)"
        except Exception as e:
            return f"ERROR: {e}"
    
    # === CREATE ===
    elif sub == 'create' and len(args) > 1:
        pack_name = args[1]
        try:
            pack_dir = get_packs_dir() / pack_name
            pack_dir.mkdir(parents=True, exist_ok=True)
            create_pack_manifest(pack_name)
            return f"OK: created empty pack '{pack_name}' at {pack_dir}"
        except Exception as e:
            return f"ERROR: {e}"
    
    # === SAVE ===
    elif sub == 'save' and len(args) > 2:
        pack_name = args[1]
        sample_name = args[2]
        
        if session.last_buffer is None or len(session.last_buffer) == 0:
            return "ERROR: no audio in buffer"
        
        pack_dir = get_packs_dir() / pack_name
        pack_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = pack_dir / f"{sample_name}.wav"
        try:
            import soundfile as sf
            sf.write(str(out_path), session.last_buffer, session.sample_rate)
        except ImportError:
            from scipy.io import wavfile
            audio_int = (session.last_buffer * 32767).astype(np.int16)
            wavfile.write(str(out_path), session.sample_rate, audio_int)
        
        create_pack_manifest(pack_name)
        
        # Add to dictionary
        sample_dict = get_sample_dictionary()
        dict_name = f"{pack_name}/{sample_name}"
        sample_dict.add(dict_name, session.last_buffer, session.sample_rate, pack_name)
        
        dur = len(session.last_buffer) / session.sample_rate
        return f"OK: saved '{sample_name}' ({dur:.3f}s) to pack '{pack_name}'"
    
    # === GENERATE (Algorithmic) ===
    elif sub in ('gen', 'generate') and len(args) > 1:
        pack_name = args[1]
        gen_types = args[2].lower().split(',') if len(args) > 2 else ['all']
        
        print(f"Generating pack '{pack_name}' with types: {gen_types}")
        
        generated, msg = generate_pack(
            pack_name=pack_name,
            gen_types=gen_types,
            sample_rate=session.sample_rate,
            add_to_dict=True,
        )
        
        if generated:
            lines = [msg, "", "Generated:"]
            for f in generated[:20]:
                lines.append(f"  {f}")
            if len(generated) > 20:
                lines.append(f"  ... and {len(generated) - 20} more")
            lines.append("")
            lines.append(f"Use /pack samples {pack_name} to list all")
            return '\n'.join(lines)
        return msg
    
    # === GENERATE (AI) ===
    elif sub == 'genai' and len(args) > 2:
        pack_name = args[1]
        prompts = args[2:]
        
        print(f"AI generating pack '{pack_name}' with {len(prompts)} prompts...")
        
        try:
            generated, msg = generate_pack_ai(
                pack_name=pack_name,
                prompts=prompts,
                session=session,
                samples_per_prompt=1,
                duration=3.0,
            )
            
            if generated:
                lines = [msg, "", "Generated:"]
                for f in generated[:20]:
                    lines.append(f"  {f}")
                if len(generated) > 20:
                    lines.append(f"  ... and {len(generated) - 20} more")
                return '\n'.join(lines)
            return msg
        except Exception as e:
            return f"ERROR: AI generation failed: {e}"
    
    # === DICTIONARY ===
    elif sub == 'dict':
        sample_dict = get_sample_dictionary()
        
        if len(args) < 2:
            # Show stats
            count = len(sample_dict.entries)
            packs = set(e.pack for e in sample_dict.entries.values() if e.pack)
            return (f"Sample Dictionary: {count} samples from {len(packs)} packs\n"
                    f"\n"
                    f"Commands:\n"
                    f"  /pack dict list [pack]  - List entries\n"
                    f"  /pack dict get <n>   - Load to buffer\n"
                    f"  /pack dict add <n>   - Save buffer\n"
                    f"  /pack dict search <q>   - Search\n"
                    f"  /pack dict clear        - Clear all")
        
        dict_sub = args[1].lower()
        
        if dict_sub == 'list':
            pack_filter = args[2] if len(args) > 2 else None
            samples = sample_dict.list(pack=pack_filter)
            if not samples:
                return "Dictionary is empty."
            lines = [f"=== DICTIONARY ({len(samples)} entries) ===", ""]
            for s in samples[:50]:
                entry = sample_dict.entries.get(s)
                dur = entry.duration if entry else 0
                lines.append(f"  {s} ({dur:.2f}s)")
            if len(samples) > 50:
                lines.append(f"  ... and {len(samples) - 50} more")
            return '\n'.join(lines)
        
        elif dict_sub == 'get' and len(args) > 2:
            name = args[2]
            success, msg = load_sample_to_session(session, name)
            return msg
        
        elif dict_sub == 'add' and len(args) > 2:
            name = args[2]
            pack = args[3] if len(args) > 3 else ""
            success, msg = save_buffer_to_dictionary(session, name, pack=pack)
            return msg
        
        elif dict_sub == 'search' and len(args) > 2:
            query = args[2]
            results = sample_dict.search(query)
            if not results:
                return f"No matches for '{query}'"
            lines = [f"=== SEARCH: {query} ({len(results)} results) ===", ""]
            for s in results[:20]:
                entry = sample_dict.entries.get(s)
                dur = entry.duration if entry else 0
                lines.append(f"  {s} ({dur:.2f}s)")
            if len(results) > 20:
                lines.append(f"  ... and {len(results) - 20} more")
            return '\n'.join(lines)
        
        elif dict_sub == 'clear':
            count = sample_dict.clear()
            return f"OK: cleared {count} entries from dictionary"
        
        return f"Unknown dict command: {dict_sub}"
    
    return (f"Unknown pack command: {sub}\n"
            f"Commands: gen, genai, samples, load, save, create, info, dict")


def cmd_dict(session: "Session", args: List[str]) -> str:
    """Sample dictionary quick access.
    
    Usage:
      /dict                -> Show dictionary stats
      /dict list [pack]    -> List entries
      /dict get <n>     -> Load to buffer
      /dict add <n>     -> Save buffer
      /dict search <q>     -> Search
    
    Shortcut for /pack dict commands.
    """
    # Prepend 'dict' to args and use cmd_pack
    new_args = ['dict'] + list(args)
    return cmd_pack(session, new_args)


# Command registry
PACK_COMMANDS = {
    'pack': cmd_pack,
    'dict': cmd_dict,
}


def get_pack_commands():
    """Return pack commands for registration."""
    return PACK_COMMANDS
