# MDMA v52 CHANGELOG

## Version 52.0 (2026-02-03) - Interface Transition + GUI MVP

### Major Changes

#### 1. DSL Mode Deprecated
The `/start`...`/final` DSL mode is no longer the primary interface.
Commands now execute immediately, one at a time.

- Cognitively lighter
- Easier to debug  
- Single execution path
- More compatible with exploratory sound design

#### 2. NEW: GUI MVP (mdma_gui.py)
A wxPython "Action Panel Client" that maps GUI actions to CLI commands.

**Layout:**
```
[ MENU BAR ]
File | View | Engine | Help

---------------------------------------------------------
| LEFT: Object Browser | RIGHT: Action Panel            |
| (tree/list)          | (action dropdown + params)     |
---------------------------------------------------------

[ BOTTOM: Output Console ]
```

**Features:**
- Category browser (Engine, Synth, Filter, Envelope, FX, Presets, Banks)
- Action dropdown per category
- Auto-generated parameter widgets (int, float, enum, string)
- Command preview before execution
- Console output with syntax highlighting
- Run, Copy Command, Reset Params buttons

**Keyboard Shortcuts:**
- Ctrl+R: Run action
- Ctrl+L: Focus console
- Ctrl+F: Focus search
- F5: Refresh browser

**Theme:** Serum-ish muted dark with soft contrast

### Files Added
- `mdma_gui.py` - Complete GUI MVP implementation (~700 lines)
- `GUI_SPEC_v0.2.md` - GUI specification document
- `INTERFACE_TRANSITION_SPEC.md` - Architecture rationale

### CLI Changes
- `/start` now shows deprecation message
- DSL dispatch only activates for block modes (/out, /live, /fx)
- All other commands go through main command table
- No more `[dsl]>` prompt

### Migration Guide

**Before (DSL mode - deprecated):**
```
/start
/tone 440 1
/mel 0.4.7
/play
/final
```

**After (direct execution):**
```
/tone 440 1
/mel 0.4.7
/play
```

**Or use the GUI:**
```
python mdma_gui.py
```

### Requirements for GUI
```
pip install wxPython
```

### What Still Works
- `/sydef ... /end` - Synth definitions
- `/fn ... /end` - User functions  
- `/out ... /end` - Note sequences
- All regular commands

---

## Previous Versions

### v51.0 - Unified dispatch
### v50.0 - SyDef factory override fix
### v49.0 - Default changes (cutoff 4500Hz, FLAC 24-bit)
### v48.0 - DSL management command
