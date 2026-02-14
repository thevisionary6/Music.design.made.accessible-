#!/usr/bin/env python
"""MDMA rebuild launcher with advanced command support.

This REPL binds command names starting with '/' to functions in the
mdma_rebuild.commands package. Supports function definition, playback,
and enhanced error help.

Keybindings (readline):
  Ctrl+S   Save project
  Ctrl+O   Open/load project
  Ctrl+N   New project
  Ctrl+K   Clear current input line
  Ctrl+R   Run selected/last command again
  Ctrl+C   Cancel / interrupt
  Tab      Autocomplete command names

BUILD ID: bmdma_v52.0
"""

import sys
import os
import readline
import atexit

# Ensure package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the Session class
from mdma_rebuild.core.session import Session

# Store command modules in a dictionary for safe access
_CMD_MODULES = {}

def _import_module(name):
    """Safely import a command module."""
    try:
        mod = __import__(f'mdma_rebuild.commands.{name}', fromlist=[name])
        _CMD_MODULES[name] = mod
        return mod
    except ImportError as e:
        print(f"Warning: Could not import {name}: {e}")
        return None

# Import all command modules
_import_module('general_cmds')
_import_module('synth_cmds')
_import_module('fx_cmds')
_import_module('render_cmds')
_import_module('advanced_cmds')
_import_module('stub_cmds')
_import_module('pattern_cmds')
_import_module('playback_cmds')
_import_module('buffer_cmds')
_import_module('math_cmds')  # Custom arithmetic and utility commands
_import_module('param_cmds')  # Parameter system v45
_import_module('hq_cmds')     # HQ audio settings v45
_import_module('gen_cmds')    # Phase 4 generative commands

# AI commands are optional
AI_AVAILABLE = False
ai_cmds = _import_module('ai_cmds')
if ai_cmds is not None:
    AI_AVAILABLE = True

# Create convenient global aliases
general_cmds = _CMD_MODULES.get('general_cmds')
synth_cmds = _CMD_MODULES.get('synth_cmds')
fx_cmds = _CMD_MODULES.get('fx_cmds')
render_cmds = _CMD_MODULES.get('render_cmds')
advanced_cmds = _CMD_MODULES.get('advanced_cmds')
stub_cmds = _CMD_MODULES.get('stub_cmds')
pattern_cmds = _CMD_MODULES.get('pattern_cmds')
playback_cmds = _CMD_MODULES.get('playback_cmds')
buffer_cmds = _CMD_MODULES.get('buffer_cmds')
param_cmds = _CMD_MODULES.get('param_cmds')
hq_cmds = _CMD_MODULES.get('hq_cmds')
gen_cmds = _CMD_MODULES.get('gen_cmds')


def build_command_table():
    """Collect all command functions from the commands modules.
    
    COMMAND OWNERSHIP:
    ------------------
    To resolve conflicts where multiple modules define the same command,
    we use explicit ownership. The COMMAND_OWNERS dict specifies which
    module owns each conflicting command. Commands not in this dict
    use last-loaded-wins behavior.
    
    Priority order (for non-owned commands):
    1. stub_cmds (lowest - always overridden)
    2. general_cmds, synth_cmds, fx_cmds, render_cmds
    3. advanced_cmds, adv_cmds
    4. buffer_cmds, pattern_cmds, working_cmds
    5. playback_cmds (high - owns play/stop)
    6. generator_cmds
    7. dj_cmds, perf_cmds
    8. ai_cmds (highest)
    """
    commands = {}
    
    # Explicit ownership for conflicting commands
    # Format: 'command_name': 'owning_module'
    COMMAND_OWNERS = {
        # Playback - playback_cmds owns unified playback
        'play': 'playback_cmds',
        'stop': 'playback_cmds',
        'p': 'playback_cmds',
        'pw': 'playback_cmds',
        'stop_play': 'playback_cmds',
        's': 'playback_cmds',
        
        # Buffer operations - buffer_cmds owns
        'a': 'buffer_cmds',
        'b': 'buffer_cmds',
        'pa': 'buffer_cmds',
        'buf': 'buffer_cmds',
        'bu': 'buffer_cmds',
        'clr': 'buffer_cmds',
        
        # Working buffer - working_cmds owns
        'wb': 'working_cmds',
        'wbc': 'working_cmds',
        'w': 'working_cmds',
        'as': 'working_cmds',
        
        # Pattern - pattern_cmds owns
        'pat': 'pattern_cmds',
        'apat': 'pattern_cmds',
        'chop': 'pattern_cmds',
        'arp': 'pattern_cmds',
        
        # Synth - synth_cmds owns
        'preset': 'synth_cmds',
        'tone': 'synth_cmds',
        'n': 'synth_cmds',
        
        # Effects - fx_cmds owns
        'fx': 'fx_cmds',
        'st': 'fx_cmds',
        'vamp': 'fx_cmds',
        'fc': 'fx_cmds',
        'gg': 'fx_cmds',
        
        # DJ - dj_cmds owns DJ-specific
        'vol': 'dj_cmds',
        'deck': 'dj_cmds',
        'djm': 'dj_cmds',
        'xfade': 'dj_cmds',
        
        # Performance - perf_cmds owns
        'mc': 'perf_cmds',
        'snap': 'perf_cmds',
        'perf': 'perf_cmds',
        
        # General - general_cmds owns
        'save': 'general_cmds',
        'load': 'general_cmds',
        'import': 'general_cmds',
        'bpm': 'general_cmds',
        'help': 'general_cmds',
        'h': 'general_cmds',
        
        # Render - render_cmds owns
        'rn': 'render_cmds',
        'render': 'render_cmds',
        
        # AI - ai_cmds owns
        'ai': 'ai_cmds',
        'enhance': 'ai_cmds',
        'gen': 'ai_cmds',
        'ask': 'ai_cmds',
        'high': 'ai_cmds',
        
        # Audio-rate modulation - audiorate_cmds owns
        'audiorate': 'audiorate_cmds',
        'ar': 'audiorate_cmds',
        'ump': 'audiorate_cmds',
        'impulse': 'audiorate_cmds',
        'chke': 'audiorate_cmds',

        # Phase 3: Convolution & impulse commands - convolution_cmds owns
        'conv': 'convolution_cmds',
        'convolution': 'convolution_cmds',
        'convrev': 'convolution_cmds',
        'impulselfo': 'convolution_cmds',
        'ilfo': 'convolution_cmds',
        'lfoimport': 'convolution_cmds',
        'impenv': 'convolution_cmds',
        'ienv': 'convolution_cmds',
        'envimport': 'convolution_cmds',
        'irenhance': 'convolution_cmds',
        'ire': 'convolution_cmds',
        'irtransform': 'convolution_cmds',
        'irt': 'convolution_cmds',
        'irgranular': 'convolution_cmds',
        'irg': 'convolution_cmds',
        'irgrains': 'convolution_cmds',

        # Phase T: System audit commands - phase_t_cmds owns
        'undo': 'phase_t_cmds',
        'redo': 'phase_t_cmds',
        'snapshot': 'phase_t_cmds',
        'section': 'phase_t_cmds',
        'pchain': 'phase_t_cmds',
        'export': 'phase_t_cmds',
        'master_gain': 'phase_t_cmds',
        'mgain': 'phase_t_cmds',
        'crossover': 'phase_t_cmds',
        'xover': 'phase_t_cmds',
        'dup': 'phase_t_cmds',
        'duplicate': 'phase_t_cmds',
        'metronome': 'phase_t_cmds',
        'metro': 'phase_t_cmds',
        'commit': 'phase_t_cmds',
        'cm': 'phase_t_cmds',
        'autosave': 'phase_t_cmds',
        'pos': 'phase_t_cmds',
        'seek': 'phase_t_cmds',
        'swap': 'phase_t_cmds',
        'filefx': 'phase_t_cmds',

        # Phase 4: Generative commands - gen_cmds owns
        'beat': 'gen_cmds',
        'loop': 'gen_cmds',
        'xform': 'gen_cmds',
        'transform': 'gen_cmds',
        'adapt': 'gen_cmds',
        'theory': 'gen_cmds',
        'gen2': 'gen_cmds',
        'generate': 'gen_cmds',
    }
    
    def get_module_name(module):
        """Extract module name from module object."""
        if module is None:
            return None
        name = getattr(module, '__name__', str(module))
        # Extract just the last part (e.g., 'mdma_rebuild.commands.fx_cmds' -> 'fx_cmds')
        return name.split('.')[-1] if '.' in name else name
    
    def should_register(cmd_name, module):
        """Check if this module should register this command."""
        mod_name = get_module_name(module)
        if cmd_name in COMMAND_OWNERS:
            return COMMAND_OWNERS[cmd_name] == mod_name
        return True  # No ownership rule, allow registration
    
    def register_from_module(module, source_name=None):
        """Register cmd_* functions from a module."""
        if module is None:
            return
        mod_name = source_name or get_module_name(module)
        for attr_name in dir(module):
            if attr_name.startswith('cmd_'):
                func = getattr(module, attr_name)
                if callable(func):
                    cmd_name = attr_name[4:].lower()
                    if should_register(cmd_name, module):
                        commands[cmd_name] = func
    
    def register_from_dict(cmd_dict, mod_name):
        """Register commands from a dictionary."""
        if not isinstance(cmd_dict, dict):
            return
        for cmd_name, func in cmd_dict.items():
            if callable(func):
                # Create a fake module-like check
                if cmd_name in COMMAND_OWNERS:
                    if COMMAND_OWNERS[cmd_name] == mod_name:
                        commands[cmd_name] = func
                else:
                    commands[cmd_name] = func
    
    # ================================================================
    # PHASE 1: Load stub commands first (lowest priority, will be overridden)
    # ================================================================
    stub_mod = _CMD_MODULES.get('stub_cmds')
    if stub_mod is not None and hasattr(stub_mod, 'STUB_COMMANDS'):
        for cmd_name, func in stub_mod.STUB_COMMANDS.items():
            commands[cmd_name] = func
    
    # ================================================================
    # PHASE 2: Load base modules (general, synth, fx, render)
    # ================================================================
    for mod_name in ['general_cmds', 'synth_cmds', 'fx_cmds', 'render_cmds']:
        module = _CMD_MODULES.get(mod_name)
        register_from_module(module, mod_name)
    
    # ================================================================
    # PHASE 3: Load advanced modules
    # ================================================================
    register_from_module(_CMD_MODULES.get('advanced_cmds'), 'advanced_cmds')
    
    # Add ADVANCED_COMMANDS dict
    adv_mod = _CMD_MODULES.get('advanced_cmds')
    if adv_mod is not None and hasattr(adv_mod, 'ADVANCED_COMMANDS'):
        register_from_dict(adv_mod.ADVANCED_COMMANDS, 'advanced_cmds')
    
    # adv_cmds (different from advanced_cmds)
    try:
        from mdma_rebuild.commands.adv_cmds import get_advanced_commands
        register_from_dict(get_advanced_commands(), 'adv_cmds')
    except ImportError:
        pass
    
    # ================================================================
    # PHASE 4: Load buffer, pattern, working modules
    # ================================================================
    register_from_module(_CMD_MODULES.get('buffer_cmds'), 'buffer_cmds')
    register_from_module(_CMD_MODULES.get('pattern_cmds'), 'pattern_cmds')
    
    try:
        from mdma_rebuild.commands.working_cmds import get_working_commands
        register_from_dict(get_working_commands(), 'working_cmds')
    except ImportError:
        pass
    
    # Also register cmd_* from working_cmds module directly
    try:
        from mdma_rebuild.commands import working_cmds as wc_mod
        register_from_module(wc_mod, 'working_cmds')
    except ImportError:
        pass

    # ================================================================
    # PHASE 4.5: Load custom math and utility commands
    # ================================================================
    # Register arithmetic commands defined in math_cmds.  These add
    # /add, /sub, /times, /over, /power, /log, /give and numeric
    # aliases.  They are registered after built‑in commands so they
    # override numeric stack access but before playback commands.
    register_from_module(_CMD_MODULES.get('math_cmds'), 'math_cmds')
    
    # ================================================================
    # PHASE 5: Load playback (high priority for play/stop)
    # ================================================================
    register_from_module(_CMD_MODULES.get('playback_cmds'), 'playback_cmds')
    
    # Also get_playback_commands if it exists
    pb_mod = _CMD_MODULES.get('playback_cmds')
    if pb_mod and hasattr(pb_mod, 'get_playback_commands'):
        try:
            register_from_dict(pb_mod.get_playback_commands(), 'playback_cmds')
        except Exception:
            pass
    
    # ================================================================
    # PHASE 6: Load generators
    # ================================================================
    try:
        from mdma_rebuild.commands.generator_cmds import get_generator_commands
        register_from_dict(get_generator_commands(), 'generator_cmds')
    except ImportError:
        pass
    
    # ================================================================
    # PHASE 7: Load DJ and Performance modules
    # ================================================================
    try:
        from mdma_rebuild.commands.dj_cmds import get_dj_commands
        register_from_dict(get_dj_commands(), 'dj_cmds')
    except ImportError:
        pass
    
    try:
        from mdma_rebuild.commands.perf_cmds import get_perf_commands
        register_from_dict(get_perf_commands(), 'perf_cmds')
    except ImportError:
        pass
    
    # ================================================================
    # PHASE 8: Load pack commands
    # ================================================================
    try:
        from mdma_rebuild.commands.pack_cmds import get_pack_commands
        register_from_dict(get_pack_commands(), 'pack_cmds')
    except ImportError:
        pass
    
    # ================================================================
    # PHASE 8.5: Load audio-rate modulation and umpulse commands
    # ================================================================
    try:
        from mdma_rebuild.commands.audiorate_cmds import get_audiorate_commands
        register_from_dict(get_audiorate_commands(), 'audiorate_cmds')
    except ImportError:
        pass

    # ================================================================
    # PHASE 8.55: Load Phase 3 convolution & impulse commands
    # ================================================================
    try:
        from mdma_rebuild.commands.convolution_cmds import get_convolution_commands
        register_from_dict(get_convolution_commands(), 'convolution_cmds')
    except ImportError:
        pass
    
    # ================================================================
    # PHASE 8.6: Load parameter system commands (v45)
    # ================================================================
    register_from_module(_CMD_MODULES.get('param_cmds'), 'param_cmds')
    param_mod = _CMD_MODULES.get('param_cmds')
    if param_mod and hasattr(param_mod, 'get_param_commands'):
        try:
            register_from_dict(param_mod.get_param_commands(), 'param_cmds')
        except Exception:
            pass
    
    # ================================================================
    # PHASE 8.7: Load HQ audio commands (v45)
    # ================================================================
    register_from_module(_CMD_MODULES.get('hq_cmds'), 'hq_cmds')
    hq_mod = _CMD_MODULES.get('hq_cmds')
    if hq_mod and hasattr(hq_mod, 'get_hq_commands'):
        try:
            register_from_dict(hq_mod.get_hq_commands(), 'hq_cmds')
        except Exception:
            pass
    
    # ================================================================
    # PHASE 8.8: Load Phase 4 generative commands
    # ================================================================
    try:
        from mdma_rebuild.commands.gen_cmds import get_gen_commands
        register_from_dict(get_gen_commands(), 'gen_cmds')
    except ImportError:
        pass

    # ================================================================
    # PHASE 8.9: Load Phase T commands (system audit / song-ready)
    # ================================================================
    try:
        from mdma_rebuild.commands.phase_t_cmds import get_phase_t_commands
        register_from_dict(get_phase_t_commands(), 'phase_t_cmds')
    except ImportError:
        pass

    # ================================================================
    # PHASE 8.95: Load Phase 6 MIDI commands
    # ================================================================
    try:
        from mdma_rebuild.commands.midi_cmds import get_midi_commands
        register_from_dict(get_midi_commands(), 'midi_cmds')
    except ImportError:
        pass

    # ================================================================
    # PHASE 9: Load AI commands (highest priority)
    # ================================================================
    ai_mod = _CMD_MODULES.get('ai_cmds')
    if AI_AVAILABLE and ai_mod is not None:
        register_from_module(ai_mod, 'ai_cmds')
        if hasattr(ai_mod, 'AI_COMMANDS'):
            register_from_dict(ai_mod.AI_COMMANDS, 'ai_cmds')
    
    # ================================================================
    # PHASE 10: Explicit voice parameter commands from synth_cmds
    # ================================================================
    try:
        from mdma_rebuild.commands import synth_cmds
        voice_cmds = {
            'stereo': synth_cmds.cmd_stereo,
            'vphase': synth_cmds.cmd_vphase,
            'venv': synth_cmds.cmd_venv,
            'fenv': synth_cmds.cmd_fenv,
            'menv': synth_cmds.cmd_menv,
        }
        for cmd_name, func in voice_cmds.items():
            commands[cmd_name] = func
        
        # Add synth commands from get_synth_commands if available
        if hasattr(synth_cmds, 'get_synth_commands'):
            register_from_dict(synth_cmds.get_synth_commands(), 'synth_cmds')
    except (ImportError, AttributeError):
        pass
    
    # ================================================================
    # PHASE 10.5: Load MAD DSL commands
    # ================================================================
    try:
        from mdma_rebuild.commands.dsl_cmds import get_dsl_commands
        register_from_dict(get_dsl_commands(), 'dsl_cmds')
    except ImportError:
        pass

    # ================================================================
    # PHASE 10.6: Load SyDef (Synth Definition) commands
    # ================================================================
    try:
        from mdma_rebuild.commands.sydef_cmds import get_sydef_commands
        register_from_dict(get_sydef_commands(), 'sydef_cmds')
    except ImportError:
        pass
    
    # ================================================================
    # PHASE 11: Register 'h' as alias for 'help' if not already set
    # ================================================================
    if 'help' in commands and 'h' not in commands:
        commands['h'] = commands['help']
    
    return commands


def show_help():
    """Display help text."""
    print("\n=== MDMA v52 - ALL COMMANDS ===\n")
    
    print("BUFFER SYSTEM:")
    print("  /buf           Show current buffer info")
    print("  /buf all       Show all buffers")
    print("  /bu <n>        Select buffer (1-indexed)")
    print("  /bu+ <n>       Set buffer count")
    print("  /a [pos]       Insert pending audio (0=start, end=default)")
    print("  /bov [pos]     Overlay (mix on top) at position")
    print("  /ex <sec|cmd>  Extend buffer")
    print()
    
    print("SMART PLAYBACK:")
    print("  /P             Smart play (context-aware)")
    print("  /PB [n]        Play buffer explicitly")
    print("  /PD [n]        Play deck explicitly")
    print("  /PW            Play working buffer")
    print("  /stop          Stop playback")
    print()
    
    print("WORKING BUFFER:")
    print("  /W [src]       Load to working buffer")
    print("  /WBC           Clear working buffer")
    print("  /WFX <fx>      Add effect with metrics")
    print()
    
    print("RENDER:")
    print("  /b [path]      Render current buffer")
    print("  /ba            Render all buffers")
    print("  /bo [path]     Render omni (combined)")
    print()
    
    print("AUTO-CHUNK & REMIX:")
    print("  /CHK [algo]    Auto-chunk buffer")
    print("  /remix [algo]  Remix audio")
    print("  /RPAT <pat>    Rhythmic pattern")
    print()
    
    print("VARIABLES:")
    print("  /= name val    Set variable")
    print("  /GET [name]    Get variable")
    print()
    
    print("MAD DSL:")
    print("  /mel <pat> [hz] [sydef=p]  Melody to working (uses synth engine)")
    print("  /cor <pat> [hz] [sydef=p]  Chord sequence (0,4,7...0,3,7)")
    print("  /out ... /end    Note block to working buffer")
    print("  /wa tone/mel/s   Append to working buffer")
    print("  /wa [n]          Commit working to buffer N")
    print("  /play            Smart play (finds audio automatically)")
    print("  /render [path]   Render to WAV file")
    print("  /loop [n]        Loop working buffer N times")
    print("  /mutate <type>   Transform (reverse/half/double/chop/fade)")
    print("  /b               Show all buffers")
    print()
    
    print("LIVE LOOPS:")
    print("  /live <name>     Start recording a live loop")
    print("    /mel, /wa, etc.  (commands recorded, not executed)")
    print("    /end             Starts the loop playing continuously")
    print("  /live             List active loops")
    print("  /kill <name>     Kill a specific live loop")
    print("  /ka              Kill ALL live loops")
    print()
    
    print("SYNTHS & CHAINS:")
    print("  /synth <name>    Create/select synth instance")
    print("  /chain <id>      Create/select FX chain")
    print("  /<chain> add fx  Add effect to chain")
    print("  /apply <chain>   Apply chain to working buffer")
    print("  /<chain>         Start effect block (apply on /end)")
    print()
    

    print("CHUNK SYSTEM:")
    print("  /ch, /chunk    Chunk command (alias)")
    print("  /ch add        Add buffer as chunk")
    print("  /ch build      Stitch chunks with crossfade")
    print("  /ch xfade <ms> Set crossfade duration")
    print("  /ch play       Play stitched result")
    print()
    
    print("GRANULAR ENGINE:")
    print("  /gr            Show granular status")
    print("  /gr size <ms>  Set grain size")
    print("  /gr density <n> Set grain density")
    print("  /gr pitch <r>  Set pitch ratio")
    print("  /gr process    Process buffer through granular")
    print("  /gr freeze <p> Freeze at position")
    print("  /gr stretch <f> Time stretch")
    print("  /gr shift <st> Pitch shift (semitones)")
    print()
    
    print("PATTERNS:")
    print("  /pat <notes>   Pattern last append")
    print("  /apat <notes>  Pattern entire buffer")
    print("  /pag [n]       List/detail algorithms")
    print("  /arp <chord>   Arpeggio (maj,min,7,maj7,min7)")
    print()

    print("GENERATIVE (Phase 4):")
    print("  /beat <genre> [bars] [bpm]  Generate drum beat")
    print("  /beat fill <type>           Generate fill (buildup/roll/crash)")
    print("  /loop <genre> [layers...]   Generate full loop")
    print("  /gen2 melody [key] [scale]  Generate melody")
    print("  /gen2 chord_prog [key]      Generate chord progression")
    print("  /gen2 bassline [key] [genre] Generate bassline")
    print("  /gen2 arp <chord>           Arpeggiate chord")
    print("  /gen2 drone [key] [dur]     Ambient drone")
    print("  /xform <transform>          Apply musical transform")
    print("  /xform preset <name>        Apply transform preset")
    print("  /adapt key <note> [scale]   Adapt to new key")
    print("  /adapt tempo <bpm>          Adapt to new tempo")
    print("  /theory scales/chords/prog  Music theory queries")
    print()
    
    print("PHASE T — SONG-READY TOOLS:")
    print("  /undo [track N]    Undo last operation")
    print("  /redo [track N]    Redo undone operation")
    print("  /snapshot          Save/restore parameter snapshot")
    print("  /section add/list  Song section markers")
    print("  /pchain <b> <r>... Chain buffers into sequence")
    print("  /export stems      Export all tracks as stems")
    print("  /export track <n>  Export single track")
    print("  /master_gain <dB>  Set master gain")
    print("  /crossover <a> <b> Crossover two buffers")
    print("  /dup [src] [dst]   Duplicate buffer")
    print("  /metronome [bars]  Generate click track")
    print("  /swap <a> <b>      Swap two buffers")
    print("  /commit [track]    Commit working buffer to track")
    print("  /pos [sec|bar]     Show/set write position")
    print("  /seek <pos>        Jump to position")
    print("  /autosave on/off   Toggle auto-save")
    print("  /filefx add/clear  Manage file FX chain")
    print()

    print("FX CHAINS (by position):")
    print("  /bfx           Buffer FX (add/rm/apply/clear)")
    print("  /tfx           Track FX")
    print("  /mfx           Master FX")
    print("  /ffx           File FX")
    print("  /fxall         Show all chains")
    print()
    
    print("EFFECTS:")
    print("  /fx [add|rm]   Effect chain management")
    print("  /fxa <effect>  Add effect")
    print("  /vamp [preset] Vamp/OD (light,medium,heavy,fuzz)")
    print("  /dual [preset] Dual OD (warm,bright,heavy)")
    print("  /conv [preset] Convolution reverb")
    print("  /hfx           Effects help")
    print()
    
    print("FILTERS:")
    print("  /fs <n>        Select filter slot")
    print("  /ft <type>     Set type (lp,hp,bp,acid...)")
    print("  /cut <Hz>      Set cutoff")
    print("  /res <0-100>   Set resonance")
    print("  /hf            Filter help")
    print()
    
    print("SYNTHESIS:")
    print("  /op <n>        Select operator")
    print("  /wm <wave>     Waveform (sine,tri,saw,pulse,noise)")
    print("  /fr <Hz>       Frequency")
    print("  /amp <0-1>     Amplitude")
    print("  /tone <Hz> <b> Generate tone")
    print()
    
    print("MODULATION ROUTING:")
    print("  /fm <s> <t> <amt>  FM routing")
    print("  /am <s> <t> <amt>  AM routing")
    print("  /rm <s> <t> <amt>  Ring mod routing")
    print("  /pm <s> <t> <amt>  Phase mod routing")
    print("  /rt            View/clear routings")
    print()
    
    print("AUDIO-RATE MODULATION:")
    print("  /imod          Interval modulation status")
    print("  /imod <op> lfo <rate> <depth> [wave]")
    print("                 Set LFO interval mod per operator")
    print("  /fmod          Filter modulation status")
    print("  /fmod lfo <rate> <depth>")
    print("                 Set LFO filter modulation")
    print()
    
    print("ROUTING:")
    print("  /route         Show all routings")
    print("  /route add <type> <s> <t> <amt>")
    print("                 Add routing (fm,tfm,am,rm,pm)")
    print("  /route rm <n>  Remove routing")
    print("  /route clear   Clear all")
    print()
    
    print("PRESET BANK:")
    print("  /preset        List presets")
    print("  /preset save <slot> [name]")
    print("  /preset load <slot>")
    print("  /preset del <slot>")
    print()
    
    print("GENERATION:")
    print("  /g <algo>      Generate (sk,ssn,sh,si,pluck,pad)")
    print("  /gen <prompt>  AI generation (if available)")
    print()
    
    print("ENVELOPES:")
    print("  /atk <s>       Attack time")
    print("  /dec <s>       Decay time")
    print("  /sus <0-1>     Sustain level")
    print("  /rel <s>       Release time")
    print("  /adsr          Show envelope")
    print()
    
    print("VOICE ALGORITHM:")
    print("  /vc <n>        Voice count")
    print("  /dt <Hz>       Detune per voice")
    print("  /rand <0-100>  Random variation")
    print("  /vmod <0-100>  Mod depth scaling")
    print()
    
    print("SEED/RANDOM:")
    print("  /sm [0|1|2]    Seed mode (0=off,1=fixed,2=inc)")
    print("  /sm <n>        Set seed (if n>10)")
    print("  /rmode         Random mode (off,fixed,inc)")
    print("  /rn            Random 0-1")
    print()
    
    print("PROJECT:")
    print("  /new [name]    Create new entity")
    print("  /lv [n]        Set/show level")
    print("  /B             Build project")
    print("  /save [path]   Save project")
    print("  /load <path>   Load project")
    print()
    
    print("ROUTING BANKS:")
    print("  /bk            List banks")
    print("  /bk <name>     Select bank (classic_fm, modern_aggressive, etc)")
    print("  /al            Show current algorithm")
    print("  /al <n>        Load algorithm by index")
    print("  /al list       List algorithms in bank")
    print("  /al apply      Apply algorithm to engine")
    print()
    
    print("DYNAMICS:")
    print("  /fc [preset]   Forever Compression (punch,glue,loud,soft,ott)")
    print("  /fc <depth> [low=n] [mid=n] [high=n] [up=n] [down=n]")
    print("  /gg <pattern>  Giga Gate pattern (half,tresillo,1010...)")
    print("  /gg stutter <n> Stutter effect")
    print("  /gg tape stop  Tape stop effect")
    print()
    
    print("PACK SYSTEM:")
    print("  /pack          List installed packs")
    print("  /pack gen <name> <type>  Generate (kicks,snares,hats,bass,leads,pads,all)")
    print("  /pack genai <name> <prompts>  AI-generate from text prompts")
    print("  /pack samples <name>  List samples in pack")
    print("  /pack load <name> <idx>  Load sample to buffer")
    print("  /pack save <name> <name>  Save buffer to pack")
    print("  /pack dict list    List sample dictionary")
    print("  /pack dict get <n> Load from dictionary")
    print("  /pack dict add <n> Save buffer to dictionary")
    print("  /dict              Shortcut for /pack dict")
    print("  /bpa <name>        Export buffers as pack")
    print()
    
    print("DJ MODE:")
    print("  /djm           Toggle DJ mode (/dj)")
    print("  /djm ready     Run readiness check")
    print("  Device Management:")
    print("    /DO          List output devices (/dev)")
    print("    /DOC <id>    Connect master output")
    print("    /HEP         Scan headphone devices")
    print("    /HEPC <id>   Connect headphones/cue")
    print("  Deck Control:")
    print("    /deck        Show deck status (/dk, /d)")
    print("    /deck+ [n]   Add deck / set count (/dk+)")
    print("    /deck- <n>   Remove deck (/dk-)")
    print("    /deck <n> load [buf]  Load buffer to deck")
    print("    /deck <n> save [buf]  Save deck to buffer")
    print("    /deck <n> play/pause  Playback control")
    print("    /deck <n> analyze     AI analysis")
    print("  Mixing:")
    print("    /cue         Cue bus control (/c)")
    print("    /xfade <v>   Crossfader 0-1 (/xf)")
    print("    /sync        Sync deck tempos (/sy)")
    print("  Transitions:")
    print("    /tran        Quick transition (/tr, /x)")
    print("    /tran <d> <style> [dur] Transition to deck")
    print("    /tran styles List transition styles")
    print("    /drop        Instant drop (/drp, /!)")
    print("  Stem Separation (requires demucs):")
    print("    /stem sep <deck>      Separate into stems (/st)")
    print("    /stem <d> <s> <vol>   Set stem volume")
    print("    /stem <d> solo <s>    Solo a stem")
    print("    /stem <d> remix       Remix with levels")
    print("  Sectioning & Chopping:")
    print("    /section <deck>       Auto-detect sections (/sec)")
    print("    /section <d> <idx>    Jump to section")
    print("    /chop <d> [mode] [n]  Chop audio (beat/transient/time)")
    print("    /chop <d> get <idx>   Load chop to buffer")
    print("  Streaming (requires yt-dlp):")
    print("    /stream <d> <url>     Stream to deck (/str, /sc)")
    print("    /stream search <q>    Search SoundCloud")
    print("  Advanced:")
    print("    /VDA         Virtual Audio Device routing")
    print("    /SR          Screen reader (NVDA) routing")
    print("    /fallback    Safety system control (/fb)")
    print()
    
    print("PERFORMANCE MODE (requires DJ Mode):")
    print("  /perf          Toggle Performance Mode (/pf)")
    print("  Macros:")
    print("    /mc new <n>  Create macro")
    print("    /mc add <n> <type> <action>  Add step")
    print("    /mc run <n>  Execute macro")
    print("    /panic       Emergency stop")
    print("  Snapshots:")
    print("    /snap cap <n> [label]  Capture state")
    print("    /snap recall <n>    Recall snapshot")
    print("    /snap home <n>      Set home")
    print("    /snap go               Return to home")
    print("  RNG:")
    print("    /rng seed <n>  Set random seed")
    print("    /rng amount <0-1>  Set randomness")
    print()
    
    print("ADVANCED AUDIO OPS:")
    print("  Auto-Chunking:")
    print("    /CHK [algo]       Chunk buffer (auto,beat,transient,equal,wavetable)")
    print("    /CHK use <idx>    Load chunk to buffer")
    print("  Remix:")
    print("    /remix [algo] [int]  Remix (shuffle,glitch,stutter,evolve,granular)")
    print("  Rhythmic Pattern:")
    print("    /RPAT <pat> [beats]  Apply pattern (x.x.x. or 1 0.5 0 0.8)")
    print("  Buffer Combining:")
    print("    /CBI <idx> ...    Combine/overlay buffers")
    print("    /BAP <src> [dst]  Append buffer to another")
    print("  Wavetable:")
    print("    /WT [frames] [size]  Generate wavetable from audio")
    print()
    
    print("USER VARIABLES:")
    print("  /= name value       Set variable")
    print("  /GET name           Get variable (supports name.key)")
    print("  /DEL name           Delete variable")
    print()
    
    print("ADVANCED MACROS:")
    print("  /MC new <n> [args]  Create macro with arguments")
    print("  /MC run <n> [vals]  Run macro with argument values")
    print("  /MC list            List macros")
    print("  /MC show <n>        Show macro commands")
    print("  /MC del <n>         Delete macro")
    print("  (Use $argname or $1,$2 in macro commands)")
    print()
    
    print("SYNTH DEFINITIONS:")
    print("  /sydef <n> [p=default ...]  Define a synth patch")
    print("    (enter commands, use $param for arguments)")
    print("    /end                         Close definition")
    print("  /use <n> [vals...]          Instantiate a SyDef")
    print("  /use <n> param=val ...      Named arg override")
    print("  /sydef list                    List all SyDefs")
    print("  /sydef show <n>             Inspect a SyDef")
    print("  /sydef del <n>              Delete a SyDef")
    print("  /sydef save <path>             Save to JSON")
    print("  /sydef load <path>             Load from JSON")
    print("  Factory: sine, acid, pad, pluck, kick")
    print()
    
    print("BRIDGE COMMANDS:")
    print("  /PR [buf] [deck]    Print buffer to DJ deck")
    print("  /YT <url> [buf]     Pull YouTube to buffer")
    print("  /SC <url> [buf]     Pull SoundCloud to buffer")
    print("  /DK2BUF [dk] [buf]  Copy deck to buffer")
    print()
    
    print("QUICK DECK LOADING:")
    print("  /DKL <d> <path>  Load local file to deck")
    print("  /DKS <query>     Search for files")
    print("  /DKS <d> <idx>   Load search result to deck")
    print()
    
    print("BLUETOOTH DEVICES:")
    print("  /BL              Scan Bluetooth devices only")
    print("  /BDOC <id>       Connect Bluetooth to master")
    print("  /BTHEP <id>      Connect Bluetooth headphones")
    print()
    
    print("AI FEATURES (requires torch, diffusers):")
    print("  /gpu           Show GPU info")
    print("  /gen <prompt>  Generate audio (/g) - auto-stores to dictionary")
    print("  /genv <p> [n]  Generate variations (/gv)")
    print("  /analyze       Deep analysis (/an)")
    print("  /describe      Descriptor profile (/ds)")
    print("  /breed <b1><b2> Breed samples (/br)")
    print("  /evolve <attr> Evolve toward target (/ev)")
    print("  /mutate        Apply mutation (/mu)")
    print()
    
    print("AI COMMAND ROUTER:")
    print("  /ask <request>   Interpret natural language (/?)")
    print("  /high <request>  Multi-command execution (/hi)")
    print("  /confirm         Execute plan (/cf, /yes)")
    print("  /cancel          Cancel plan (/no, /abort)")
    print()
    
    print("UTILITY:")
    print("  /h [cmd]       Help (detailed if cmd given)")
    print("  /qr            Quick reference")
    print("  /info          Session info")
    print("  /q             Quit")
    print()
    print("Use /h <cmd> for detailed help on specific command")


def show_quick_ref():
    """Quick reference card."""
    print("\n=== QUICK REFERENCE ===")
    print("GENERATE: /tone 440 1  /g sk  /g_kick  /g_snare  /g_hat")
    print("MELODY:   /mel 0.4.7 [hz]  /cor 0,4,7...0,3,7 [hz]")
    print("EFFECTS:  /fx reverb  /fx delay  /fx compress  /fx list")
    print("BUFFER:   /a (commit)  /buf (info)  /clr (clear)  /bu <n>")
    print("PLAYBACK: /p (play)  /s (STOP ALL)  /vol <0-1>")
    print("RENDER:   /b (to file)  /ba (all buffers)  /bo (omni)")
    print("FUNCTION: /def name [args]  /run name [args]  /end")
    print("SYDEF:    /sydef name [p=val]  /use name [args]  /end")
    print("DJ MODE:  /dj  /deck <n>  /cf <ms>  /scratch  /djfx")
    print("REGISTRY: /reg scan  /reg load <id>  /reg search <term>")
    print("SYNTH:    /op <n>  /wm <wave>  /atk /dec /sus /rel")
    print("AI:       /ask <query>  /high <build this>  /analyze")
    print("HELP:     /h (full)  /h <cmd>  /qr (this)  /q (quit)")


def show_preloader():
    """Display preloading screen with system stats."""
    import time
    
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                                                              ║")
    print("║   ███╗   ███╗██████╗ ███╗   ███╗ █████╗                     ║")
    print("║   ████╗ ████║██╔══██╗████╗ ████║██╔══██╗                    ║")
    print("║   ██╔████╔██║██║  ██║██╔████╔██║███████║                    ║")
    print("║   ██║╚██╔╝██║██║  ██║██║╚██╔╝██║██╔══██║                    ║")
    print("║   ██║ ╚═╝ ██║██████╔╝██║ ╚═╝ ██║██║  ██║                    ║")
    print("║   ╚═╝     ╚═╝╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝                    ║")
    print("║                                                              ║")
    print("║   Music Development & Manipulation Application    v52.0    ║")
    print("║                                                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    stats = {
        'modules': 0,
        'commands': 0,
        'banks': 0,
        'algorithms': 0,
        'presets': 0,
        'generators': 0,
        'descriptors': 0,
        'attributes': 0,
        'ai_available': False,
        'playback_available': False,
        'gpu': 'Not detected',
    }
    
    # Count modules
    print("Loading modules...")
    module_names = [
        'core.session', 'core.banks', 'core.user_data',
        'dsp.monolith', 'dsp.effects', 'dsp.granular', 'dsp.pattern',
        'commands.buffer_cmds', 'commands.synth_cmds', 'commands.fx_cmds',
        'commands.pattern_cmds', 'commands.playback_cmds', 'commands.render_cmds',
        'commands.general_cmds', 'commands.advanced_cmds', 'commands.stub_cmds',
    ]
    stats['modules'] = len(module_names)
    
    # Count banks and algorithms
    try:
        from mdma_rebuild.core import banks
        stats['banks'] = len(banks.list_bank_names())
        total_algos = 0
        for bank_name in banks.list_bank_names():
            bank = banks.get_bank(bank_name)
            if bank:
                total_algos += len(bank.get('algorithms', []))
        stats['algorithms'] = total_algos
        print(f"  Banks: {stats['banks']} ({stats['algorithms']} algorithms)")
    except Exception:
        print("  Banks: loading failed")
    
    # Check AI availability
    try:
        from mdma_rebuild.ai import (
            ATTRIBUTE_POOL, 
            DESCRIPTOR_VOCABULARY,
            KNOWN_COMMANDS,
            detect_gpu,
        )
        stats['ai_available'] = True
        stats['attributes'] = len(ATTRIBUTE_POOL)
        stats['descriptors'] = len(DESCRIPTOR_VOCABULARY)
        
        # GPU detection
        gpu_info = detect_gpu()
        if gpu_info.get('detected', False):
            stats['gpu'] = f"{gpu_info.get('name', 'Unknown')} ({gpu_info.get('vram_gb', 0):.1f}GB)"
        else:
            stats['gpu'] = "CPU mode (defaults to RTX 3060 specs)"
        
        print(f"  AI System: READY ({stats['attributes']} attributes, {stats['descriptors']} descriptors)")
        print(f"  GPU: {stats['gpu']}")
    except ImportError as e:
        print(f"  AI System: NOT AVAILABLE (install torch, diffusers)")
        stats['ai_available'] = False
    
    # Check playback
    try:
        if advanced_cmds.PLAYBACK_AVAILABLE:
            stats['playback_available'] = True
            print("  Audio Playback: READY")
        else:
            print(f"  Audio Playback: NOT AVAILABLE")
    except Exception:
        print("  Audio Playback: checking...")
    
    # Count presets (factory)
    try:
        from mdma_rebuild.core.user_data import get_presets_dir
        from pathlib import Path
        preset_dir = get_presets_dir() / 'factory'
        if preset_dir.exists():
            stats['presets'] = len(list(preset_dir.glob('*.json')))
        print(f"  Presets: {stats['presets']} factory")
    except Exception:
        pass
    
    # Count generators
    try:
        from mdma_rebuild.commands.stub_cmds import GENERATOR_ALGORITHMS
        stats['generators'] = len(GENERATOR_ALGORITHMS)
        print(f"  Generators: {stats['generators']} algorithms")
    except Exception:
        stats['generators'] = 26  # Known count
        print(f"  Generators: ~{stats['generators']} algorithms")
    
    print()
    return stats


def main() -> None:
    # Show preloader
    stats = show_preloader()
    
    session = Session()
    commands = build_command_table()
    
    # Store command count
    stats['commands'] = len(commands)
    
    # Set command table for AI router
    if AI_AVAILABLE:
        try:
            from mdma_rebuild.commands.ai_cmds import set_command_table
            set_command_table(commands)
        except Exception:
            pass
    
    # Ensure advanced attributes are initialized
    advanced_cmds._ensure_advanced_attrs(session)

    # ===================================================================
    # READLINE SETUP — history, completion, clipboard, keybindings
    # ===================================================================
    _history_path = os.path.expanduser('~/.mdma_history')
    try:
        readline.read_history_file(_history_path)
        readline.set_history_length(2000)
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file, _history_path)

    # Tab-completion for command names
    _cmd_names = sorted('/' + k for k in commands.keys())

    def _completer(text, state):
        if text.startswith('/'):
            matches = [c for c in _cmd_names if c.startswith(text)]
        else:
            matches = [c for c in _cmd_names if c.startswith('/' + text)]
        if state < len(matches):
            return matches[state]
        return None

    readline.set_completer(_completer)
    readline.set_completer_delims(' \t\n')
    readline.parse_and_bind('tab: complete')

    # OS-specific readline bindings (differs between GNU and libedit)
    _is_libedit = 'libedit' in readline.__doc__ if readline.__doc__ else False

    if _is_libedit:
        # macOS libedit
        readline.parse_and_bind('bind ^K ed-kill-line')       # Ctrl+K: kill to end of line
    else:
        # GNU readline
        readline.parse_and_bind('"\\C-k": kill-line')         # Ctrl+K: kill to end of line
        readline.parse_and_bind('"\\C-u": unix-line-discard') # Ctrl+U: kill whole line
        readline.parse_and_bind('"\\C-y": yank')              # Ctrl+Y: paste from kill ring
        readline.parse_and_bind('"\\C-w": unix-word-rubout')  # Ctrl+W: kill word backward

    # ===================================================================
    # SPECIAL KEYSTROKE COMMANDS
    # ===================================================================
    # Map Ctrl+S / Ctrl+O / Ctrl+N / Ctrl+R to magic bytes that the
    # main loop intercepts after input() returns.

    if _is_libedit:
        readline.parse_and_bind('bind ^S "\\x13"')  # Ctrl+S -> \x13 (save)
        readline.parse_and_bind('bind ^O "\\x0f"')  # Ctrl+O -> \x0f (open)
        readline.parse_and_bind('bind ^N "\\x0e"')  # Ctrl+N -> \x0e (new)
    else:
        readline.parse_and_bind('"\\C-s": "\\x13"')
        readline.parse_and_bind('"\\C-o": "\\x0f"')
        readline.parse_and_bind('"\\C-n": "\\x0e"')

    # ===================================================================
    # AUTO-PERSISTENCE: load user data on startup
    # ===================================================================
    try:
        from mdma_rebuild.core.user_data import (
            load_user_functions,
            load_chains,
        )
        # Load persisted user functions
        persisted_fns = load_user_functions()
        if persisted_fns:
            if not hasattr(session, 'user_functions'):
                session.user_functions = {}
            session.user_functions.update(persisted_fns)
            print(f"  User functions: {len(persisted_fns)} loaded from disk")

        # Load persisted chains
        persisted_chains = load_chains()
        if persisted_chains:
            if not hasattr(session, 'chains'):
                session.chains = {}
            session.chains.update(persisted_chains)
            print(f"  Chains: {len(persisted_chains)} loaded from disk")
    except Exception:
        pass

    def _autosave_functions():
        """Persist user functions whenever they change."""
        try:
            from mdma_rebuild.core.user_data import save_user_functions
            if hasattr(session, 'user_functions') and session.user_functions:
                save_user_functions(dict(session.user_functions))
        except Exception:
            pass

    def _autosave_chains():
        """Persist named chains whenever they change."""
        try:
            from mdma_rebuild.core.user_data import save_chains as _sc
            if hasattr(session, 'chains') and session.chains:
                _sc(dict(session.chains))
        except Exception:
            pass

    def _autosave_sydefs():
        """Persist user SyDef definitions whenever they change."""
        try:
            from mdma_rebuild.commands.sydef_cmds import _autosave_sydefs as _asd
            _asd(session)
        except Exception:
            pass

    def _autosave_all():
        """Persist all user data."""
        _autosave_functions()
        _autosave_chains()
        _autosave_sydefs()

    # Import DSL support
    try:
        from mdma_rebuild.commands.dsl_cmds import (
            get_dsl_state,
            strip_comments,
            preprocess_dsl_line,
            dispatch_dsl_command,
        )
        dsl_support = True
    except ImportError:
        dsl_support = False

    # ===================================================================
    # COMMAND EXECUTION ENGINE
    # ===================================================================

    # Track last executed command for Ctrl+R repeat
    _last_command = [None]

    def execute_command(cmd_line: str) -> str:
        """Execute a single command line. Used for function execution."""
        # Strip DSL comments if DSL support is available
        if dsl_support:
            cmd_line = strip_comments(cmd_line)
            if not cmd_line:
                return ""
        
        if not cmd_line.startswith('/'):
            # In DSL mode, preprocess line
            if dsl_support:
                processed = preprocess_dsl_line(cmd_line)
                if processed and processed.startswith('/'):
                    cmd_line = processed
                else:
                    return f"ERROR: Commands must start with /"
            else:
                return f"ERROR: Commands must start with /"
        
        parts = cmd_line[1:].split()
        if not parts:
            return ""
        
        cmd = parts[0].lower()
        args = parts[1:]

        # ------------------------------------------------------------------
        # DSL: dispatch DSL commands FIRST (they take priority when active)
        # This ensures /play, /mel, /wa, /out, /end, chain names all work
        # even when those command names are owned by other modules.
        # ------------------------------------------------------------------
        if dsl_support:
            dsl_result = dispatch_dsl_command(session, cmd, args)
            if dsl_result is not None:
                return dsl_result

        # ------------------------------------------------------------------
        # DJ deck shorthand: /d1, /d2, /d3, ...
        # ------------------------------------------------------------------
        if len(cmd) >= 2 and cmd[0] == 'd' and cmd[1:].isdigit():
            deck_num = cmd[1:]
            cmd = 'deck'
            args = [deck_num] + args
        
        if cmd in ('tui',):
            return "TUI: Use the standard REPL with readline keybindings."

        if cmd in ('q', 'quit', 'exit'):
            return "EXIT"
        
        # If the command exists in the table (including numeric names) call it first
        func = commands.get(cmd)
        if func is not None:
            try:
                return func(session, args)
            except Exception as exc:
                error_msg = f"ERROR: {exc}"
                # Try to add helpful info
                if hasattr(advanced_cmds, 'get_command_help'):
                    help_text = advanced_cmds.get_command_help(cmd)
                    if help_text:
                        error_msg += f"\n\n--- Help for /{cmd} ---\n{help_text}"
                return error_msg
        
        # Handle numeric stack access if not overridden
        if cmd.isdigit():
            try:
                idx = int(cmd)
                return advanced_cmds.cmd_stack_get(session, args, idx)
            except Exception as exc:
                return f"ERROR: {exc}"

        # Suggest similar commands for unknown input
        similar = [c for c in commands.keys() if cmd in c or c.startswith(cmd[:2])][:5]
        if similar:
            return f"ERROR: Unknown command /{cmd}. Did you mean: {', '.join('/' + s for s in similar)}?"
        return f"ERROR: Unknown command /{cmd}"
    
    # Store executor in session for function execution
    session.command_executor = execute_command

    # System summary
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  MDMA v52 Ready                                              ║")
    print(f"║  Commands: {stats['commands']:3d}  Banks: {stats['banks']:2d}  Algorithms: {stats['algorithms']:3d}  Generators: {stats['generators']:2d}  ║")
    if stats['ai_available']:
        print(f"║  AI: ✓ Attributes: {stats['attributes']:3d}  Descriptors: {stats['descriptors']:3d}                 ║")
    else:
        print(f"║  AI: ✗ (pip install torch diffusers transformers librosa)   ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  ^S save  ^O open  ^N new  ^R rerun  ^K clear  Tab ⇥      ║")
    print("║  ^U kill line  ^W kill word  ^Y paste  /h help  /qr ref   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    # Verify v47 commands are loaded
    v47_cmds = ['parm', 'cha', 'plist', 'hq', 'pt', 'pts']
    missing = [c for c in v47_cmds if c not in commands]
    if missing:
        print(f"\n  WARNING: v47 commands missing: {', '.join(missing)}")
        print("  Try: pip install --upgrade mdma OR re-extract the zip")
    
    print()

    # Import repeat block functions from math_cmds
    try:
        from mdma_rebuild.commands.math_cmds import (
            is_recording_repeat, 
            record_repeat_command,
            cmd_repeat_end,
        )
        repeat_support = True
    except ImportError:
        repeat_support = False

    # Import SyDef block functions
    try:
        from mdma_rebuild.commands.sydef_cmds import (
            is_defining_sydef,
            record_sydef_command,
            end_sydef,
            load_factory_presets,
        )
        sydef_support = True
        # Load factory presets on startup (also loads user SyDefs)
        n_factory = load_factory_presets(session)
        if n_factory:
            print(f"  SyDef: {n_factory} presets loaded (factory + user)")
    except ImportError:
        sydef_support = False

    # ===================================================================
    # MAIN INPUT LOOP
    # ===================================================================

    while True:
        try:
            # Check if we're defining a function
            if session.defining_function:
                prompt = f'[def:{session.defining_function}]> '
            # Check if we're defining a sydef
            elif sydef_support and is_defining_sydef(session):
                prompt = f'[syt:{session.defining_sydef}]> '
            # Check if we're recording a repeat block
            elif repeat_support and is_recording_repeat():
                prompt = '[repeat]> '
            # Check for DSL out block (still supported for /out.../end)
            elif dsl_support and get_dsl_state().out_block:
                prompt = '[out]> '
            # Check for DSL live loop block (still supported)
            elif dsl_support and get_dsl_state().live_block:
                name = get_dsl_state().live_block_name or 'loop'
                prompt = f'[live:{name}]> '
            # Check for DSL effect block (still supported)
            elif dsl_support and get_dsl_state().fx_block:
                chain = get_dsl_state().fx_block_chain or 'fx'
                prompt = f'[{chain}]> '
            # DSL mode (/start.../final) is deprecated - no special prompt
            else:
                prompt = '> '
            
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        
        if not line:
            continue
        
        # ---------------------------------------------------------------
        # KEYBINDING INTERCEPTS
        # ---------------------------------------------------------------
        # Ctrl+S -> \x13 (save project)
        if line == '\x13' or line.startswith('\x13'):
            result = execute_command('/save')
            if result:
                print(result)
            continue

        # Ctrl+O -> \x0f (open/load project)
        if line == '\x0f' or line.startswith('\x0f'):
            print("OPEN: Enter project path (or press Enter for file list):")
            try:
                path = input("  path> ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if path:
                result = execute_command(f'/load {path}')
            else:
                # List .mdma files in current directory
                import glob
                files = sorted(glob.glob('*.mdma'))
                if files:
                    print("  Available projects:")
                    for i, f in enumerate(files, 1):
                        print(f"    {i}. {f}")
                    try:
                        choice = input("  number or path> ").strip()
                        if choice.isdigit():
                            idx = int(choice) - 1
                            if 0 <= idx < len(files):
                                result = execute_command(f'/load {files[idx]}')
                            else:
                                result = "ERROR: invalid selection"
                        elif choice:
                            result = execute_command(f'/load {choice}')
                        else:
                            continue
                    except (EOFError, KeyboardInterrupt):
                        continue
                else:
                    print("  No .mdma files in current directory.")
                    continue
            if result:
                print(result)
            continue

        # Ctrl+N -> \x0e (new project)
        if line == '\x0e' or line.startswith('\x0e'):
            print("NEW PROJECT: Enter name (or press Enter for auto-name):")
            try:
                name = input("  name> ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if name:
                result = execute_command(f'/new {name}')
            else:
                result = execute_command('/new')
            if result:
                print(result)
            continue

        # Ctrl+R -> re-run last command (or run multi-line block)
        if line == '\x12' or line.startswith('\x12'):
            # If there's text after the Ctrl+R byte, treat it as a
            # highlighted selection to execute
            remainder = line.lstrip('\x12').strip()
            if remainder:
                # Execute highlighted selection (may be multi-line)
                for sub_line in remainder.split('\n'):
                    sub_line = sub_line.strip()
                    if sub_line:
                        print(f"  run: {sub_line}")
                        result = execute_command(sub_line)
                        if result:
                            print(result)
                continue
            # Otherwise re-run last command
            if _last_command[0]:
                line = _last_command[0]
                print(f"  re-run: {line}")
            else:
                print("  No previous command to re-run.")
                continue

        # ---------------------------------------------------------------
        # STANDARD INPUT PROCESSING
        # ---------------------------------------------------------------

        # Strip comments (/// ... ///) - this is useful regardless of DSL
        if dsl_support:
            line = strip_comments(line)
            if not line:
                continue
        
        # Commands must start with '/'
        # (DSL slash-free mode is deprecated - all commands need /)
        if not line.startswith('/'):
            print("ERROR: Commands must start with /")
            continue
        
        # If defining a function, record commands (except /end)
        if session.defining_function:
            parts = line[1:].split()
            cmd = parts[0].lower() if parts else ''
            
            if cmd == 'end':
                # End function definition
                fname = session.defining_function
                session.user_functions[fname] = session.function_commands.copy()
                session.defining_function = None
                session.function_commands = []
                _autosave_functions()
                print(f"DEF: function '{fname}' saved with {len(session.user_functions[fname])} commands (auto-saved)")
                continue
            else:
                # Record the command
                session.function_commands.append(line)
                print(f"  recorded: {line}")
                continue

        # If defining a sydef, record commands (except /end)
        if sydef_support and is_defining_sydef(session):
            parts = line[1:].split()
            cmd = parts[0].lower() if parts else ''

            if cmd == 'end':
                result = end_sydef(session)
                print(result)
                _autosave_sydefs()
                continue
            else:
                result = record_sydef_command(session, line)
                print(result)
                continue
        
        # If recording a repeat block, handle specially
        if repeat_support and is_recording_repeat():
            parts = line[1:].split()
            cmd = parts[0].lower() if parts else ''
            
            if cmd == 'end':
                # End repeat block and execute
                result = cmd_repeat_end(session, [])
                if result:
                    print(result)
                continue
            else:
                # Record the command
                result = record_repeat_command(line)
                if result:
                    print(result)
                continue
        
        # Store as last command for Ctrl+R
        _last_command[0] = line

        # Normal command execution
        parts = line[1:].split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ('q', 'quit', 'exit'):
            # Auto-save everything before exit
            _autosave_all()
            break
        
        if cmd in ('h', 'help'):
            if args:
                # Help for specific command
                help_cmd = args[0].lower().lstrip('/')
                if hasattr(advanced_cmds, 'get_command_help'):
                    help_text = advanced_cmds.get_command_help(help_cmd)
                    if help_text:
                        print(help_text)
                        continue
                # Try to get docstring from command function
                func = commands.get(help_cmd)
                if func and func.__doc__:
                    print(f"\n--- Help for /{help_cmd} ---")
                    print(func.__doc__)
                    continue
                print(f"No detailed help for /{help_cmd}")
                continue
            
            show_help()
            continue
        
        if cmd == 'qr':
            show_quick_ref()
            continue
        
        # ---------------------------------------------------------------
        # DSL DISPATCH - DEPRECATED AS PRIMARY INTERFACE (v52)
        # ---------------------------------------------------------------
        # The DSL is preserved but disabled by default. All commands now
        # go through the main command table for unified, debuggable behavior.
        # 
        # DSL can be re-enabled with /dsl enable for specific use cases:
        # - Batch execution, macros, serialization
        # - But it is NOT the primary creative interface
        #
        # See: INTERFACE_TRANSITION_SPEC.md
        # ---------------------------------------------------------------
        
        # DSL only handles block closures (/end) when blocks are active
        if dsl_support:
            state = get_dsl_state()
            # Only intercept if we're in an active block that needs /end
            if state.live_block or state.out_block or state.fx_block:
                dsl_result = dispatch_dsl_command(session, cmd, args)
                if dsl_result is not None:
                    print(dsl_result)
                    continue

        # Main command table - THE SINGLE SOURCE OF TRUTH
        func = commands.get(cmd)
        if func is not None:
            try:
                result = func(session, args)
                if result is not None:
                    print(result)
                    # Auto-save after definition-related commands
                    if cmd in ('fn', 'def', 'define', 'macro'):
                        _autosave_functions()
                    if cmd in ('chain',):
                        _autosave_chains()
                    if cmd in ('sydef', 'sd', 'synthdef', 'syt'):
                        _autosave_sydefs()
                    # Auto-save all on import (may bring in defs)
                    if cmd == 'import':
                        _autosave_all()
            except Exception as exc:
                error_msg = f"ERROR: {exc}"
                # Add helpful info for known commands
                if hasattr(advanced_cmds, 'get_command_help'):
                    help_text = advanced_cmds.get_command_help(cmd)
                    if help_text:
                        error_msg += f"\n\n--- Help for /{cmd} ---\n{help_text}"
                print(error_msg)
            continue

        # Fall back to numeric stack access
        if cmd.isdigit():
            try:
                idx = int(cmd)
                result = advanced_cmds.cmd_stack_get(session, args, idx)
                print(result)
            except Exception as exc:
                print(f"ERROR: {exc}")
            continue

        # Unknown command
        similar = [c for c in commands.keys() if cmd in c or c.startswith(cmd[:2])][:5]
        if similar:
            print(f"ERROR: Unknown command /{cmd}. Did you mean: {', '.join('/' + s for s in similar)}?")
        else:
            print(f"ERROR: Unknown command /{cmd}")


if __name__ == '__main__':
    main()
