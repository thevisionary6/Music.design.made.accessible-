"""MDMA AI Commands.

Commands for AI-powered audio generation, analysis, and breeding.

Commands:
- /gen <prompt> [duration] - Generate audio from text
- /analyze [detailed] - Deep attribute analysis
- /breed <buf1> <buf2> [n] - Breed two buffers
- /evolve <target_attrs> [gens] - Evolve toward target
- /gpu - Show GPU info
- /attr <name> - Show attribute info
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.session import Session


# Check if AI modules are available
AI_AVAILABLE = False
AI_ERROR = None

try:
    from ..ai import (
        detect_gpu,
        get_optimal_settings,
        generate_audio,
        generate_variations,
        enhance_prompt,
        get_negative_prompt,
        PRESET_PROMPTS,
        ATTRIBUTE_POOL,
        AttributeVector,
        get_analyzer,
        format_analysis,
        format_comparison,
        get_breeder,
        breed_samples,
        evolve_samples,
        format_breeding_result,
        # GPU Settings
        SCHEDULERS,
        MODELS,
        get_gpu_settings,
        set_gpu_setting,
        get_gpu_status,
        list_schedulers,
        list_models,
    )
    AI_AVAILABLE = True
except ImportError as e:
    AI_ERROR = str(e)


def _ensure_ai() -> Optional[str]:
    """Check if AI modules are available."""
    if not AI_AVAILABLE:
        return (f"ERROR: AI modules not available.\n"
                f"Install dependencies: pip install torch diffusers transformers librosa\n"
                f"Error: {AI_ERROR}")
    return None


# ============================================================================
# GPU COMMAND
# ============================================================================

def cmd_gpu(session: "Session", args: List[str]) -> str:
    """Configure GPU and AI model settings.
    
    Usage:
      /gpu                    Show full GPU status and settings
      /gpu test               Test GPU availability
      
    Settings commands:
      /gpu steps <n>          Set inference steps (1-500, default 150)
      /gpu cfg <n>            Set CFG scale (1-30, default 10)
      /gpu sk <n>             Set scheduler by index (0-9)
      /gpu sk list            List available schedulers
      /gpu model <n>          Set model by index (0-2)
      /gpu model list         List available models
      /gpu device <dev>       Set device (cuda/cpu/mps)
      /gpu fp16 <on/off>      Toggle half precision
      /gpu dur <seconds>      Set default duration
      /gpu neg <prompt>       Set negative prompt
      /gpu offload <on/off>   Toggle CPU offload
      /gpu reset              Reset all settings to defaults
    
    Scheduler indices:
      0=DDPM  1=DDIM  2=PNDM  3=LMS  4=Euler
      5=Euler_A  6=DPM++ (default)  7=DPM_SDE  8=Heun  9=UniPC
    
    Model indices:
      0=audioldm2-large (default, best quality)
      1=audioldm2-music (music focused)
      2=audioldm2-full (base model)
    
    Examples:
      /gpu steps 200       -> More steps = better quality
      /gpu cfg 12          -> Higher CFG = stronger prompt adherence
      /gpu sk 6            -> Use DPM++ scheduler
      /gpu device cuda     -> Force NVIDIA GPU
    
    Short aliases: /gpuconfig
    """
    error = _ensure_ai()
    if error:
        return error
    
    # No args - show full status
    if not args:
        return get_gpu_status()
    
    cmd = args[0].lower()
    
    # Test GPU
    if cmd == 'test':
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                # Quick compute test
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.matmul(x, x)
                del x, y
                torch.cuda.empty_cache()
                
                # Memory info
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                
                return (f"✓ GPU TEST PASSED\n"
                        f"  Device: {gpu_name}\n"
                        f"  VRAM: {total:.1f} GB total, {allocated:.2f} GB used\n"
                        f"  CUDA: {torch.version.cuda}\n"
                        f"  PyTorch: {torch.__version__}")
            else:
                return ("✗ GPU NOT AVAILABLE\n"
                        "  CUDA not detected. Install CUDA-enabled PyTorch:\n"
                        "  pip install torch --index-url https://download.pytorch.org/whl/cu118")
        except Exception as e:
            return f"✗ GPU TEST FAILED: {e}"
    
    # Reset to defaults
    if cmd == 'reset':
        settings = get_gpu_settings()
        settings.steps = 150
        settings.cfg_scale = 10.0
        settings.scheduler_index = 6
        settings.model_index = 0
        settings.device = 'cuda'
        settings.force_gpu = True
        settings.fp16 = True
        settings.default_duration = 5.0
        settings.negative_prompt = "low quality, noise, distortion, silence"
        settings.cpu_offload = False
        return "OK: All GPU settings reset to defaults"
    
    # Scheduler
    if cmd in ('sk', 'sched', 'scheduler'):
        if len(args) < 2:
            return list_schedulers()
        if args[1].lower() == 'list':
            return list_schedulers()
        try:
            return set_gpu_setting('scheduler', int(args[1]))
        except ValueError:
            return f"ERROR: Invalid scheduler index '{args[1]}'. Use /gpu sk list"
    
    # Model
    if cmd in ('model', 'mdl', 'm'):
        if len(args) < 2:
            return list_models()
        if args[1].lower() == 'list':
            return list_models()
        try:
            return set_gpu_setting('model', int(args[1]))
        except ValueError:
            return f"ERROR: Invalid model index '{args[1]}'. Use /gpu model list"
    
    # Steps
    if cmd in ('steps', 'step', 's'):
        if len(args) < 2:
            settings = get_gpu_settings()
            return f"Steps: {settings.steps}"
        return set_gpu_setting('steps', args[1])
    
    # CFG
    if cmd in ('cfg', 'guidance', 'g'):
        if len(args) < 2:
            settings = get_gpu_settings()
            return f"CFG Scale: {settings.cfg_scale}"
        return set_gpu_setting('cfg', args[1])
    
    # Device
    if cmd in ('device', 'dev', 'd'):
        if len(args) < 2:
            settings = get_gpu_settings()
            return f"Device: {settings.device} (force_gpu={settings.force_gpu})"
        return set_gpu_setting('device', args[1])
    
    # FP16
    if cmd in ('fp16', 'half'):
        if len(args) < 2:
            settings = get_gpu_settings()
            return f"FP16: {settings.fp16}"
        return set_gpu_setting('fp16', args[1])
    
    # Duration
    if cmd in ('dur', 'duration', 'len'):
        if len(args) < 2:
            settings = get_gpu_settings()
            return f"Default Duration: {settings.default_duration}s"
        return set_gpu_setting('duration', args[1])
    
    # Negative prompt
    if cmd in ('neg', 'negative', 'negprompt'):
        if len(args) < 2:
            settings = get_gpu_settings()
            return f"Negative Prompt: {settings.negative_prompt}"
        return set_gpu_setting('negative', ' '.join(args[1:]))
    
    # CPU Offload
    if cmd in ('offload', 'cpu_offload'):
        if len(args) < 2:
            settings = get_gpu_settings()
            return f"CPU Offload: {settings.cpu_offload}"
        return set_gpu_setting('offload', args[1])
    
    # Info (legacy)
    if cmd in ('info', 'status'):
        return get_gpu_status()
    
    # Settings (legacy)
    if cmd == 'settings':
        gpu_info = detect_gpu()
        settings = get_optimal_settings(gpu_info)
        lines = [
            "=== OPTIMAL AI SETTINGS ===",
            f"Model: {settings['model']}",
            f"Steps: {settings['steps']}",
            f"Batch Size: {settings['batch_size']}",
            f"FP16: {'Yes' if settings['fp16'] else 'No'}",
        ]
        return '\n'.join(lines)
    
    return f"Unknown command: {cmd}. Use /gpu for help."


# ============================================================================
# GENERATION COMMAND
# ============================================================================

def cmd_gen(session: "Session", args: List[str]) -> str:
    """Generate audio from text prompt using AudioLDM2.
    
    Usage:
      /gen <prompt>                    -> Generate 5s audio
      /gen <prompt> <duration>         -> Generate specified duration
      /gen <prompt> <dur> seed=<n>     -> With specific seed
      /gen <prompt> steps=<n>          -> Override steps (default 150)
      /gen <prompt> cfg=<n>            -> Override CFG scale (default 10)
      /gen preset <name>               -> Generate from preset prompt
      /gen presets                     -> List preset prompts
    
    Examples:
      /gen deep kick drum with punch
      /gen aggressive dubstep bass 3.0
      /gen lush pad with reverb steps=200 seed=42
      /gen preset kick_sub
    
    After generation, audio is placed in last_buffer.
    Use /a to append to current buffer.
    """
    error = _ensure_ai()
    if error:
        return error
    
    if not args:
        return ("Usage: /gen <prompt> [duration] [steps=N] [cfg=N] [seed=N]\n"
                "  /gen presets - List preset prompts\n"
                "  /gen preset <name> - Use preset prompt")
    
    # Handle presets
    if args[0] == 'presets':
        lines = ["=== PRESET PROMPTS ===", ""]
        for name, prompt in PRESET_PROMPTS.items():
            lines.append(f"  {name}:")
            lines.append(f"    {prompt}")
        return '\n'.join(lines)
    
    if args[0] == 'preset' and len(args) > 1:
        preset_name = args[1]
        if preset_name not in PRESET_PROMPTS:
            return f"ERROR: preset '{preset_name}' not found. Use /gen presets to list."
        prompt = PRESET_PROMPTS[preset_name]
        args = [prompt] + args[2:]  # Replace with preset prompt
    
    # Parse arguments
    prompt_parts = []
    duration = 5.0
    steps = 150
    cfg_scale = 10.0
    seed = None
    
    for arg in args:
        if '=' in arg:
            key, val = arg.split('=', 1)
            if key == 'steps':
                steps = int(val)
            elif key == 'cfg':
                cfg_scale = float(val)
            elif key == 'seed':
                seed = int(val)
            elif key == 'dur' or key == 'duration':
                duration = float(val)
        else:
            # Try to parse as duration
            try:
                d = float(arg)
                if d < 30:  # Likely duration
                    duration = d
                else:
                    prompt_parts.append(arg)
            except ValueError:
                prompt_parts.append(arg)
    
    prompt = ' '.join(prompt_parts)
    if not prompt:
        return "ERROR: no prompt provided"
    
    # Enhance prompt
    prompt = enhance_prompt(prompt)
    negative = get_negative_prompt()
    
    try:
        print(f"Generating: '{prompt[:50]}...' ({duration}s, {steps} steps)")
        audio, sr = generate_audio(
            prompt=prompt,
            negative_prompt=negative,
            duration=duration,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
        )
        
        # Resample to session rate if needed
        if sr != session.sample_rate:
            # Simple resample
            ratio = session.sample_rate / sr
            new_len = int(len(audio) * ratio)
            x_old = np.linspace(0, 1, len(audio))
            x_new = np.linspace(0, 1, new_len)
            audio = np.interp(x_new, x_old, audio)
        
        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = (audio / peak * 0.95).astype(np.float64)
        
        # Store in last_buffer
        session.last_buffer = audio
        
        # Also add to sample dictionary
        dict_msg = ""
        try:
            from ..core.pack import get_sample_dictionary
            sample_dict = get_sample_dictionary()
            safe_prompt = prompt[:30].replace(' ', '_').replace('/', '-')
            safe_prompt = ''.join(c for c in safe_prompt if c.isalnum() or c in '_-')
            name = f"gen_{safe_prompt}"
            final_name = sample_dict.add(name, audio, session.sample_rate, 
                                         pack='generated', tags=['ai', 'generated'])
            dict_msg = f", stored as '{final_name}'"
        except ImportError:
            pass
        
        actual_dur = len(audio) / session.sample_rate
        seed_str = f", seed={seed}" if seed else ""
        return f"OK: generated {actual_dur:.2f}s audio{seed_str}{dict_msg}. Use /a to append to buffer."
        
    except Exception as e:
        return f"ERROR: generation failed: {e}"


def cmd_genv(session: "Session", args: List[str]) -> str:
    """Generate multiple variations of a prompt.
    
    Usage:
      /genv <prompt> [count] [duration]
    
    Example:
      /genv aggressive kick drum 4 1.0
    
    Variations are stored in buffers 1-N.
    """
    error = _ensure_ai()
    if error:
        return error
    
    if not args:
        return "Usage: /genv <prompt> [count] [duration]"
    
    # Parse args
    prompt_parts = []
    count = 4
    duration = 3.0
    
    for arg in args:
        try:
            n = int(arg)
            if n < 20:
                count = n
            else:
                prompt_parts.append(arg)
        except ValueError:
            try:
                d = float(arg)
                if d < 30:
                    duration = d
                else:
                    prompt_parts.append(arg)
            except ValueError:
                prompt_parts.append(arg)
    
    prompt = ' '.join(prompt_parts)
    if not prompt:
        return "ERROR: no prompt provided"
    
    prompt = enhance_prompt(prompt)
    negative = get_negative_prompt()
    
    try:
        print(f"Generating {count} variations of: '{prompt[:40]}...'")
        
        results = generate_variations(
            prompt=prompt,
            count=count,
            duration=duration,
            steps=100,  # Faster for variations
            cfg_scale=10.0,
            seed_start=42,
        )
        
        # Store in buffers
        for i, (audio, sr, seed) in enumerate(results, 1):
            # Resample if needed
            if sr != session.sample_rate:
                ratio = session.sample_rate / sr
                new_len = int(len(audio) * ratio)
                x_old = np.linspace(0, 1, len(audio))
                x_new = np.linspace(0, 1, new_len)
                audio = np.interp(x_new, x_old, audio)
            
            if not hasattr(session, 'buffers'):
                session.buffers = {}
            session.buffers[i] = audio.astype(np.float64)
        
        return f"OK: generated {count} variations in buffers 1-{count}"
        
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================================
# ANALYSIS COMMANDS
# ============================================================================

def cmd_analyze(session: "Session", args: List[str]) -> str:
    """Deep attribute analysis of current buffer.
    
    Usage:
      /analyze              -> Analyze last_buffer (quick)
      /analyze detailed     -> Full detailed analysis
      /analyze buf <n>      -> Analyze specific buffer
      /analyze top <n>      -> Show top N attributes
      /analyze cat <name>   -> Show category attributes
      /analyze compare <n1> <n2> -> Compare two buffers
    
    Categories: spectral, envelope, dynamics, texture, harmonic, pitch, rhythm, character, spatial
    """
    error = _ensure_ai()
    if error:
        return error
    
    # Get analyzer
    analyzer = get_analyzer(session.sample_rate)
    
    # Determine what to analyze
    if not args:
        audio = session.last_buffer
        if audio is None or len(audio) == 0:
            return "ERROR: no audio in buffer. Generate or load audio first."
        av = analyzer.analyze(audio, detailed=False)
        return format_analysis(av, top_n=15)
    
    if args[0] == 'detailed':
        audio = session.last_buffer
        if audio is None or len(audio) == 0:
            return "ERROR: no audio in buffer"
        av = analyzer.analyze(audio, detailed=True)
        return format_analysis(av, top_n=20)
    
    elif args[0] == 'buf' and len(args) > 1:
        buf_idx = int(args[1])
        if not hasattr(session, 'buffers') or buf_idx not in session.buffers:
            return f"ERROR: buffer {buf_idx} not found"
        audio = session.buffers[buf_idx]
        av = analyzer.analyze(audio, detailed=True, source=f"buffer {buf_idx}")
        return format_analysis(av)
    
    elif args[0] == 'top' and len(args) > 1:
        n = int(args[1])
        audio = session.last_buffer
        if audio is None:
            return "ERROR: no audio"
        av = analyzer.analyze(audio, detailed=True)
        return format_analysis(av, top_n=n)
    
    elif args[0] == 'cat' and len(args) > 1:
        cat = args[1]
        audio = session.last_buffer
        if audio is None:
            return "ERROR: no audio"
        av = analyzer.analyze(audio, detailed=True)
        
        by_cat = av.by_category()
        if cat not in by_cat:
            return f"ERROR: category '{cat}' not found. Available: {', '.join(by_cat.keys())}"
        
        lines = [f"=== {cat.upper()} ATTRIBUTES ===", ""]
        for name, value in sorted(by_cat[cat].items(), key=lambda x: x[1], reverse=True):
            info = ATTRIBUTE_POOL.get(name, {})
            lines.append(f"  {name}: {value:.1f}  ({info.get('desc', '')})")
        return '\n'.join(lines)
    
    elif args[0] == 'compare' and len(args) > 2:
        buf1 = int(args[1])
        buf2 = int(args[2])
        
        if not hasattr(session, 'buffers'):
            return "ERROR: no buffers"
        if buf1 not in session.buffers or buf2 not in session.buffers:
            return f"ERROR: buffer {buf1} or {buf2} not found"
        
        av1 = analyzer.analyze(session.buffers[buf1], source=f"buffer {buf1}")
        av2 = analyzer.analyze(session.buffers[buf2], source=f"buffer {buf2}")
        
        return format_comparison(av1, av2)
    
    return "Unknown subcommand. Use /analyze for help."


def cmd_attr(session: "Session", args: List[str]) -> str:
    """Show attribute information.
    
    Usage:
      /attr                 -> List all attribute categories
      /attr <name>          -> Show info for specific attribute
      /attr list            -> List all attributes
      /attr list <category> -> List attributes in category
    """
    error = _ensure_ai()
    if error:
        return error
    
    if not args:
        # Show categories
        categories = set()
        for info in ATTRIBUTE_POOL.values():
            categories.add(info.get('category', 'other'))
        
        lines = ["=== ATTRIBUTE CATEGORIES ===", ""]
        for cat in sorted(categories):
            count = sum(1 for info in ATTRIBUTE_POOL.values() 
                       if info.get('category') == cat)
            lines.append(f"  {cat}: {count} attributes")
        lines.append("")
        lines.append(f"Total: {len(ATTRIBUTE_POOL)} attributes")
        lines.append("Use /attr list <category> to see attributes")
        return '\n'.join(lines)
    
    if args[0] == 'list':
        cat_filter = args[1] if len(args) > 1 else None
        
        lines = [f"=== {'ALL' if not cat_filter else cat_filter.upper()} ATTRIBUTES ===", ""]
        for name, info in sorted(ATTRIBUTE_POOL.items()):
            if cat_filter and info.get('category') != cat_filter:
                continue
            rng = info.get('range', (0, 100))
            lines.append(f"  {name} [{rng[0]}-{rng[1]}]")
            lines.append(f"    {info.get('desc', '')}")
        return '\n'.join(lines)
    
    # Show specific attribute
    name = args[0]
    if name not in ATTRIBUTE_POOL:
        # Try partial match
        matches = [n for n in ATTRIBUTE_POOL if name.lower() in n.lower()]
        if matches:
            return f"Did you mean: {', '.join(matches[:5])}"
        return f"ERROR: attribute '{name}' not found"
    
    info = ATTRIBUTE_POOL[name]
    lines = [
        f"=== ATTRIBUTE: {name} ===",
        f"Description: {info.get('desc', 'N/A')}",
        f"Category: {info.get('category', 'other')}",
        f"Range: {info.get('range', (0, 100))}",
        f"Weight: {info.get('weight', 1.0)}",
    ]
    return '\n'.join(lines)


# ============================================================================
# BREEDING COMMANDS
# ============================================================================

def cmd_breed(session: "Session", args: List[str]) -> str:
    """Breed two buffers to create children.
    
    Usage:
      /breed <buf1> <buf2>              -> Breed buffers, create 4 children
      /breed <buf1> <buf2> <n>          -> Create n children
      /breed last                       -> Breed buffers 1 and 2
    
    Children are stored in buffers starting at buffer 10.
    
    Examples:
      /breed 1 2                -> Breed buffers 1 and 2
      /breed 1 2 8              -> Create 8 children
    """
    error = _ensure_ai()
    if error:
        return error
    
    if not args:
        return "Usage: /breed <buf1> <buf2> [num_children]"
    
    # Initialize buffers if needed
    if not hasattr(session, 'buffers'):
        session.buffers = {}
    
    # Parse args
    if args[0] == 'last':
        buf1_idx, buf2_idx = 1, 2
        num_children = 4
    else:
        buf1_idx = int(args[0])
        buf2_idx = int(args[1]) if len(args) > 1 else buf1_idx + 1
        num_children = int(args[2]) if len(args) > 2 else 4
    
    # Get parent buffers
    if buf1_idx not in session.buffers or buf2_idx not in session.buffers:
        return f"ERROR: buffer {buf1_idx} or {buf2_idx} not found"
    
    parent_a = session.buffers[buf1_idx]
    parent_b = session.buffers[buf2_idx]
    
    # Breed
    breeder = get_breeder(session.sample_rate)
    children = breeder.breed(parent_a, parent_b, num_children)
    
    # Store children starting at buffer 10
    start_idx = 10
    for i, child in enumerate(children):
        session.buffers[start_idx + i] = child
    
    lines = [
        f"OK: bred {num_children} children from buffers {buf1_idx} and {buf2_idx}",
        f"Children stored in buffers {start_idx}-{start_idx + num_children - 1}",
        "",
        "Children:"
    ]
    for i, child in enumerate(children):
        dur = len(child) / session.sample_rate
        peak = np.max(np.abs(child))
        lines.append(f"  Buffer {start_idx + i}: {dur:.3f}s, peak={peak:.3f}")
    
    return '\n'.join(lines)


def cmd_evolve(session: "Session", args: List[str]) -> str:
    """Evolve population toward target attributes.
    
    Usage:
      /evolve <attrs> [generations]
      /evolve brightness=80 punch=90 10
      /evolve toward <buf> [gens]      -> Evolve toward buffer's attributes
    
    Population is taken from buffers 1-8 (or whatever exists).
    Results stored in buffers 20-27.
    
    Examples:
      /evolve brightness=70 attack=90 warmth=60
      /evolve toward 5 15              -> Evolve toward buffer 5 for 15 gens
    """
    error = _ensure_ai()
    if error:
        return error
    
    if not args:
        return ("Usage: /evolve <attr>=<val> [attr=val...] [generations]\n"
                "       /evolve toward <buf> [generations]")
    
    if not hasattr(session, 'buffers'):
        return "ERROR: no buffers to evolve"
    
    # Parse target
    target_attrs = {}
    generations = 10
    
    if args[0] == 'toward' and len(args) > 1:
        # Evolve toward a buffer's attributes
        target_buf = int(args[1])
        if target_buf not in session.buffers:
            return f"ERROR: buffer {target_buf} not found"
        
        # Analyze target
        analyzer = get_analyzer(session.sample_rate)
        av = analyzer.analyze(session.buffers[target_buf], detailed=True)
        target_attrs = av.attributes.copy()
        
        if len(args) > 2:
            generations = int(args[2])
    else:
        # Parse attribute targets
        for arg in args:
            if '=' in arg:
                key, val = arg.split('=', 1)
                target_attrs[key] = float(val)
            else:
                try:
                    generations = int(arg)
                except ValueError:
                    pass
    
    if not target_attrs:
        return "ERROR: no target attributes specified"
    
    # Get population from buffers 1-10
    population = []
    for i in range(1, 11):
        if i in session.buffers and len(session.buffers[i]) > 0:
            population.append(session.buffers[i])
    
    if len(population) < 2:
        return "ERROR: need at least 2 buffers (1-10) for evolution"
    
    print(f"Evolving {len(population)} samples for {generations} generations...")
    
    # Evolve
    evolved = evolve_samples(
        population=population,
        target_attributes=target_attrs,
        generations=generations,
        sample_rate=session.sample_rate,
    )
    
    # Store results starting at buffer 20
    start_idx = 20
    for i, sample in enumerate(evolved):
        session.buffers[start_idx + i] = sample
    
    lines = [
        f"OK: evolved {len(population)} samples for {generations} generations",
        f"Results stored in buffers {start_idx}-{start_idx + len(evolved) - 1}",
        "",
        "Top targets:"
    ]
    for attr, val in sorted(target_attrs.items(), key=lambda x: x[1], reverse=True)[:5]:
        lines.append(f"  {attr}: {val:.1f}")
    
    return '\n'.join(lines)


def cmd_mutate(session: "Session", args: List[str]) -> str:
    """Apply mutation to buffer.
    
    Usage:
      /mutate [type] [strength]
    
    Types: noise, pitch, time_stretch, freq_shift, envelope, reverse_segment, spectral_smear
    Strength: 0.0-1.0 (default 0.2)
    
    Examples:
      /mutate pitch 0.3
      /mutate spectral_smear
    """
    error = _ensure_ai()
    if error:
        return error
    
    if session.last_buffer is None or len(session.last_buffer) == 0:
        return "ERROR: no audio in buffer"
    
    from ..ai.breeding import apply_mutation, MUTATION_FUNCTIONS
    
    mut_type = None
    strength = 0.2
    
    for arg in args:
        if arg in MUTATION_FUNCTIONS:
            mut_type = arg
        else:
            try:
                strength = float(arg)
            except ValueError:
                pass
    
    if mut_type is None:
        import random
        mut_type = random.choice(list(MUTATION_FUNCTIONS.keys()))
    
    session.last_buffer = apply_mutation(
        session.last_buffer.copy(),
        mutation_type=mut_type,
        strength=strength,
    )
    
    return f"OK: applied {mut_type} mutation (strength={strength:.2f})"


# ============================================================================
# DESCRIPTOR COMMANDS
# ============================================================================

def cmd_describe(session: "Session", args: List[str]) -> str:
    """Generate descriptor profile for audio.
    
    Usage:
      /describe             -> Describe last_buffer
      /describe buf <n>     -> Describe specific buffer
      /describe text        -> Show text summary only
    
    Uses the 200+ descriptor vocabulary to characterize audio
    with fuzzy, probabilistic descriptors.
    """
    error = _ensure_ai()
    if error:
        return error
    
    try:
        from ..ai import (
            get_analyzer, 
            attributes_to_descriptors,
            format_descriptor_profile,
        )
    except ImportError:
        return "ERROR: Descriptor module not available"
    
    # Get audio to analyze
    if args and args[0] == 'buf' and len(args) > 1:
        buf_idx = int(args[1])
        if not hasattr(session, 'buffers') or buf_idx not in session.buffers:
            return f"ERROR: buffer {buf_idx} not found"
        audio = session.buffers[buf_idx]
        source = f"buffer {buf_idx}"
    else:
        audio = session.last_buffer
        source = "last_buffer"
    
    if audio is None or len(audio) == 0:
        return "ERROR: no audio to describe"
    
    # Analyze
    analyzer = get_analyzer(session.sample_rate)
    av = analyzer.analyze(audio, detailed=True, source=source)
    
    # Convert to descriptors
    profile = attributes_to_descriptors(av.attributes)
    profile.duration = av.duration
    profile.source = source
    
    # Text only?
    if args and args[0] == 'text':
        return f"This sound is: {profile.text_summary()}"
    
    return format_descriptor_profile(profile)


# ============================================================================
# ROUTER COMMANDS
# ============================================================================

# Global command table reference for router
_command_table = None

def _get_command_table():
    """Get the command table (set by main launcher)."""
    global _command_table
    return _command_table

def set_command_table(table):
    """Set the command table for router."""
    global _command_table
    _command_table = table


def cmd_ask(session: "Session", args: List[str]) -> str:
    """Ask the AI to interpret a request and suggest commands.
    
    Usage:
      /ask <natural language request>
    
    The AI router will interpret your request and suggest
    MDMA commands to accomplish it.
    
    Examples:
      /ask make me a kick drum
      /ask add some reverb to the buffer
      /ask analyze and breed buffers 1 and 2
    
    Use /high for automatic execution of multi-step plans.
    """
    error = _ensure_ai()
    if error:
        return error
    
    if not args:
        return ("Usage: /ask <natural language request>\n"
                "  /ask make me a bass sound\n"
                "  /ask add distortion\n"
                "Use /high for auto-execution mode")
    
    try:
        from ..ai import get_router, format_plan
    except ImportError:
        return "ERROR: Router module not available"
    
    command_table = _get_command_table()
    if not command_table:
        return "ERROR: Command table not initialized"
    
    router = get_router(command_table)
    user_input = ' '.join(args)
    
    plan = router.route(user_input, session, high_mode=False)
    
    return format_plan(plan)


def cmd_high(session: "Session", args: List[str]) -> str:
    """High-power AI mode: interpret and execute multi-step plans.
    
    Usage:
      /high <natural language request>
    
    /high invokes multi-command execution planning.
    "Build this for me using existing MDMA capabilities."
    
    Characteristics:
    - Always risk_level = high
    - Chains multiple commands
    - Allows intermediate analysis
    - Adapts later steps based on earlier results
    - Requires confirmation for destructive operations
    
    Examples:
      /high make me an aggressive bass with compression
      /high create a lush pad and analyze it
      /high breed buffers 1 and 2 and pick the best
    
    Safety: /high does not bypass validation. All commands
    are validated before execution.
    """
    error = _ensure_ai()
    if error:
        return error
    
    if not args:
        return ("Usage: /high <natural language request>\n"
                "\n"
                "/high is high-power mode that chains multiple\n"
                "commands to build complex sounds.\n"
                "\n"
                "Examples:\n"
                "  /high make me an aggressive bass\n"
                "  /high create a lush pad with reverb\n"
                "  /high analyze buffer 1 and describe it")
    
    try:
        from ..ai import get_router, format_plan
    except ImportError:
        return "ERROR: Router module not available"
    
    command_table = _get_command_table()
    if not command_table:
        return "ERROR: Command table not initialized"
    
    router = get_router(command_table)
    user_input = ' '.join(args)
    
    # Create high-mode plan
    plan = router.route(user_input, session, high_mode=True)
    
    # Show plan first
    output_lines = [format_plan(plan), ""]
    
    if not plan.plan:
        output_lines.append("No executable plan generated.")
        return '\n'.join(output_lines)
    
    # Auto-confirm for low/medium risk, ask for high
    if plan.risk_level.value == 'high' and plan.requires_confirmation:
        output_lines.append("⚠️  HIGH RISK PLAN - Type '/high confirm' to execute")
        output_lines.append("    or '/high cancel' to abort")
        
        # Store pending plan in session
        if not hasattr(session, '_pending_plan'):
            session._pending_plan = None
        session._pending_plan = plan
        
        return '\n'.join(output_lines)
    
    # Execute directly
    output_lines.append("Executing plan...")
    output_lines.append("-" * 40)
    
    results = router.execute_plan(plan, session, confirm_callback=lambda p: True)
    
    for cmd, result in results:
        output_lines.append(f"{cmd}")
        output_lines.append(f"  → {result[:100]}..." if len(result) > 100 else f"  → {result}")
    
    return '\n'.join(output_lines)


def cmd_high_confirm(session: "Session", args: List[str]) -> str:
    """Confirm and execute pending high-risk plan."""
    if not hasattr(session, '_pending_plan') or session._pending_plan is None:
        return "No pending plan to confirm. Use /high <request> first."
    
    try:
        from ..ai import get_router
    except ImportError:
        return "ERROR: Router module not available"
    
    command_table = _get_command_table()
    if not command_table:
        return "ERROR: Command table not initialized"
    
    plan = session._pending_plan
    session._pending_plan = None
    
    router = get_router(command_table)
    
    output_lines = ["Executing confirmed plan...", "-" * 40]
    
    results = router.execute_plan(plan, session, confirm_callback=lambda p: True)
    
    for cmd, result in results:
        output_lines.append(f"{cmd}")
        result_str = result[:100] + "..." if len(result) > 100 else result
        output_lines.append(f"  → {result_str}")
    
    return '\n'.join(output_lines)


def cmd_high_cancel(session: "Session", args: List[str]) -> str:
    """Cancel pending high-risk plan."""
    if hasattr(session, '_pending_plan'):
        session._pending_plan = None
    return "Pending plan cancelled."


# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

AI_COMMANDS = {
    # GPU/Setup
    'gpu': cmd_gpu,
    'g_pu': cmd_gpu,  # alias
    'gpuconfig': cmd_gpu,
    'ai': cmd_gpu,  # ai settings
    
    # Generation
    'gen': cmd_gen,
    'g': cmd_gen,  # short alias (override generator if needed)
    'generate': cmd_gen,
    'genv': cmd_genv,
    'gv': cmd_genv,  # short alias
    'variations': cmd_genv,
    
    # Analysis
    'analyze': cmd_analyze,
    'ana': cmd_analyze,
    'an': cmd_analyze,  # ultra-short
    'attr': cmd_attr,
    'at': cmd_attr,  # short
    'attribute': cmd_attr,
    'describe': cmd_describe,
    'desc': cmd_describe,
    'ds': cmd_describe,  # ultra-short
    
    # Breeding
    'breed': cmd_breed,
    'br': cmd_breed,  # short
    'evolve': cmd_evolve,
    'ev': cmd_evolve,  # short
    'mutate': cmd_mutate,
    'mut': cmd_mutate,  # short
    'mu': cmd_mutate,  # ultra-short
    
    # Router
    'ask': cmd_ask,
    'a_sk': cmd_ask,  # avoid conflict with /a append
    '?': cmd_ask,  # question mark alias
    'high': cmd_high,
    'hi': cmd_high,  # short
    'h_i': cmd_high,  # avoid /h help conflict
    'confirm': cmd_high_confirm,
    'cf': cmd_high_confirm,  # short
    'yes': cmd_high_confirm,  # natural alias
    'cancel': cmd_high_cancel,
    'no': cmd_high_cancel,  # natural alias
    'abort': cmd_high_cancel,
}


def get_ai_commands() -> dict:
    """Return AI commands for registration."""
    return AI_COMMANDS
