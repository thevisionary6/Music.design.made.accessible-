"""MDMA User Data Management System.

Handles all persistent user data including:
- Constants (immutable user-defined values)
- Preferences (session defaults)
- Packs (sound/sample packs)
- Banks (routing algorithm banks)
- Presets (synth presets)
- Songs (DJ mode song library)
- Outputs (render backups)
- Functions (user-defined functions/macros)
- Variables (persistent user variables)

Section K of MDMA Master Feature List.

User Data Structure:
    ~/Documents/MDMA/
    ├── constants.json       # User-defined constants (immutable)
    ├── preferences.json     # Session defaults
    ├── variables.json       # Persistent user variables
    ├── functions.json       # User-defined functions/macros
    ├── banks/               # Routing algorithm banks
    │   ├── factory/         # Built-in banks (read-only)
    │   └── user/            # User-created banks
    ├── presets/             # Synth presets
    │   ├── factory/         # Built-in presets
    │   └── user/            # User-created presets
    ├── packs/               # Sound/sample packs
    │   └── <pack_name>/     # Individual pack folders
    ├── songs/               # DJ mode song library
    │   ├── library.json     # Song metadata/tags
    │   └── playlists/       # Playlist definitions
    ├── outputs/             # Render backups
    │   ├── index.json       # Output index/metadata
    │   └── *.wav            # Rendered audio files
    └── projects/            # Project files (existing)
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

def get_mdma_root() -> Path:
    """Get the root MDMA user data directory.
    
    Windows: C:\\Users\\<user>\\Documents\\MDMA
    Linux/Mac: ~/Documents/MDMA
    
    Creates the directory if it doesn't exist.
    """
    if os.name == 'nt':
        # Windows - use USERPROFILE environment variable
        docs = Path(os.environ.get('USERPROFILE', str(Path.home()))) / 'Documents'
    else:
        # Linux/Mac
        docs = Path.home() / 'Documents'
    
    mdma_root = docs / 'MDMA'
    mdma_root.mkdir(parents=True, exist_ok=True)
    return mdma_root


def get_constants_path() -> Path:
    """Get path to constants.json file."""
    return get_mdma_root() / 'constants.json'


def get_preferences_path() -> Path:
    """Get path to preferences.json file."""
    return get_mdma_root() / 'preferences.json'


def get_banks_dir(factory: bool = False) -> Path:
    """Get path to banks directory.
    
    Parameters
    ----------
    factory : bool
        If True, return factory banks path (read-only)
        If False, return user banks path
    """
    banks = get_mdma_root() / 'banks'
    if factory:
        path = banks / 'factory'
    else:
        path = banks / 'user'
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_presets_dir(factory: bool = False) -> Path:
    """Get path to presets directory.
    
    Parameters
    ----------
    factory : bool
        If True, return factory presets path
        If False, return user presets path
    """
    presets = get_mdma_root() / 'presets'
    if factory:
        path = presets / 'factory'
    else:
        path = presets / 'user'
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_packs_dir() -> Path:
    """Get path to packs directory."""
    packs = get_mdma_root() / 'packs'
    packs.mkdir(parents=True, exist_ok=True)
    return packs


def get_songs_dir() -> Path:
    """Get path to songs directory for DJ mode."""
    songs = get_mdma_root() / 'songs'
    songs.mkdir(parents=True, exist_ok=True)
    return songs


def get_playlists_dir() -> Path:
    """Get path to playlists directory."""
    playlists = get_songs_dir() / 'playlists'
    playlists.mkdir(parents=True, exist_ok=True)
    return playlists


def get_projects_dir() -> Path:
    """Get path to projects directory."""
    projects = get_mdma_root() / 'projects'
    projects.mkdir(parents=True, exist_ok=True)
    return projects


def get_song_library_path() -> Path:
    """Get path to song library metadata file."""
    return get_songs_dir() / 'library.json'


def get_outputs_dir() -> Path:
    """Get path to outputs directory for render backups."""
    outputs = get_mdma_root() / 'outputs'
    outputs.mkdir(parents=True, exist_ok=True)
    return outputs


def get_outputs_index_path() -> Path:
    """Get path to outputs index file."""
    return get_outputs_dir() / 'index.json'


def get_functions_path() -> Path:
    """Get path to user functions/macros file."""
    return get_mdma_root() / 'functions.json'


def get_variables_path() -> Path:
    """Get path to persistent user variables file."""
    return get_mdma_root() / 'variables.json'


# ============================================================================
# OUTPUT BACKUP SYSTEM
# ============================================================================

def load_outputs_index() -> Dict[str, Any]:
    """Load outputs index from disk."""
    path = get_outputs_index_path()
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {'version': '1.0', 'outputs': [], 'next_id': 1}
    return {'version': '1.0', 'outputs': [], 'next_id': 1}


def save_outputs_index(index: Dict[str, Any]) -> bool:
    """Save outputs index to disk."""
    path = get_outputs_index_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, default=str)
        return True
    except IOError:
        return False


def backup_output(source_path: str, name: str = None, tags: List[str] = None) -> Tuple[bool, str]:
    """Backup a rendered output file to user outputs directory.
    
    Parameters
    ----------
    source_path : str
        Path to the rendered file
    name : str, optional
        Human-readable name for the output
    tags : list, optional
        Tags for categorization
    
    Returns
    -------
    tuple
        (success, message or output_id)
    """
    import shutil
    
    source = Path(source_path)
    if not source.exists():
        return False, f"Source file not found: {source_path}"
    
    index = load_outputs_index()
    output_id = index.get('next_id', 1)
    
    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ext = source.suffix or '.wav'
    dest_name = f"output_{output_id:04d}_{timestamp}{ext}"
    dest_path = get_outputs_dir() / dest_name
    
    try:
        shutil.copy2(source, dest_path)
    except IOError as e:
        return False, f"Copy failed: {e}"
    
    # Update index
    entry = {
        'id': output_id,
        'filename': dest_name,
        'original_name': source.name,
        'name': name or f"Output {output_id}",
        'timestamp': timestamp,
        'tags': tags or [],
        'size_bytes': dest_path.stat().st_size,
    }
    
    index['outputs'].append(entry)
    index['next_id'] = output_id + 1
    
    save_outputs_index(index)
    
    return True, str(output_id)


def list_outputs(limit: int = 20) -> List[Dict]:
    """List recent outputs.
    
    Parameters
    ----------
    limit : int
        Maximum number to return
    
    Returns
    -------
    list
        List of output entries (most recent first)
    """
    index = load_outputs_index()
    outputs = index.get('outputs', [])
    return list(reversed(outputs[-limit:]))


def get_output_by_id(output_id: int) -> Optional[Dict]:
    """Get output entry by ID."""
    index = load_outputs_index()
    for entry in index.get('outputs', []):
        if entry.get('id') == output_id:
            return entry
    return None


def get_output_path(output_id: int) -> Optional[Path]:
    """Get full path to output file by ID."""
    entry = get_output_by_id(output_id)
    if entry:
        return get_outputs_dir() / entry['filename']
    return None


# ============================================================================
# FUNCTIONS/MACROS PERSISTENCE
# ============================================================================

# ============================================================================
# SYDEF PERSISTENCE (auto-save)
# ============================================================================

def get_sydefs_path() -> Path:
    """Get path to persistent SyDef definitions."""
    return get_mdma_root() / 'sydefs.json'


def get_chains_path() -> Path:
    """Get path to persistent chain definitions."""
    return get_mdma_root() / 'chains.json'


def get_user_functions_path() -> Path:
    """Get path to persistent user functions (fn blocks)."""
    return get_mdma_root() / 'user_functions.json'


def save_sydefs(sydefs: dict) -> bool:
    """Auto-save all SyDef definitions to disk.

    Parameters
    ----------
    sydefs : dict
        ``{name: SyDef}`` map from the session.  Each value must
        expose a ``to_dict()`` method.
    """
    path = get_sydefs_path()
    try:
        data = {}
        for name, sd in sydefs.items():
            if hasattr(sd, 'to_dict'):
                data[name] = sd.to_dict()
            elif isinstance(sd, dict):
                data[name] = sd
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except (IOError, TypeError):
        return False


def load_sydefs() -> dict:
    """Load SyDef definitions from disk (raw dicts)."""
    path = get_sydefs_path()
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_chains(chains: dict) -> bool:
    """Auto-save named chain definitions to disk."""
    path = get_chains_path()
    try:
        # chains is {name: [effect_name_list]}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(chains, f, indent=2, default=str)
        return True
    except (IOError, TypeError):
        return False


def load_chains() -> dict:
    """Load named chain definitions from disk."""
    path = get_chains_path()
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_user_functions(functions: dict) -> bool:
    """Auto-save user-defined functions (/fn blocks) to disk."""
    path = get_user_functions_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(functions, f, indent=2, default=str)
        return True
    except (IOError, TypeError):
        return False


def load_user_functions() -> dict:
    """Load user-defined functions from disk."""
    path = get_user_functions_path()
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


# ============================================================================
# LEGACY FUNCTIONS/MACROS PERSISTENCE
# ============================================================================

def load_functions() -> Dict[str, Any]:
    """Load user functions/macros from disk."""
    path = get_functions_path()
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {'version': '1.0', 'functions': {}, 'macros': {}}
    return {'version': '1.0', 'functions': {}, 'macros': {}}


def save_functions(functions: Dict[str, Any]) -> bool:
    """Save user functions/macros to disk."""
    path = get_functions_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(functions, f, indent=2, default=str)
        return True
    except IOError:
        return False


def save_macro(name: str, commands: List[str], args: List[str] = None, 
               description: str = None) -> Tuple[bool, str]:
    """Save a macro to persistent storage.
    
    Parameters
    ----------
    name : str
        Macro name
    commands : list
        List of commands in the macro
    args : list, optional
        Argument names for the macro
    description : str, optional
        Human-readable description
    
    Returns
    -------
    tuple
        (success, message)
    """
    data = load_functions()
    
    data['macros'][name] = {
        'commands': commands,
        'args': args or [],
        'description': description or '',
        'created': datetime.now().isoformat(),
        'modified': datetime.now().isoformat(),
    }
    
    if save_functions(data):
        return True, f"Macro '{name}' saved"
    return False, "Failed to save macro"


def load_macro(name: str) -> Optional[Dict]:
    """Load a macro from persistent storage."""
    data = load_functions()
    return data.get('macros', {}).get(name)


def list_macros() -> List[str]:
    """List all saved macro names."""
    data = load_functions()
    return list(data.get('macros', {}).keys())


def delete_macro(name: str) -> Tuple[bool, str]:
    """Delete a macro from persistent storage."""
    data = load_functions()
    if name in data.get('macros', {}):
        del data['macros'][name]
        if save_functions(data):
            return True, f"Macro '{name}' deleted"
        return False, "Failed to save"
    return False, f"Macro '{name}' not found"


# ============================================================================
# PERSISTENT VARIABLES
# ============================================================================

def load_persistent_variables() -> Dict[str, Any]:
    """Load persistent user variables from disk."""
    path = get_variables_path()
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_persistent_variables(variables: Dict[str, Any]) -> bool:
    """Save persistent user variables to disk."""
    path = get_variables_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(variables, f, indent=2, default=str)
        return True
    except IOError:
        return False


def set_persistent_variable(name: str, value: Any) -> bool:
    """Set a persistent variable."""
    variables = load_persistent_variables()
    variables[name] = value
    return save_persistent_variables(variables)


def get_persistent_variable(name: str, default: Any = None) -> Any:
    """Get a persistent variable."""
    variables = load_persistent_variables()
    return variables.get(name, default)


# ============================================================================
# CONSTANTS MANAGEMENT
# ============================================================================

def load_constants() -> Dict[str, Any]:
    """Load user constants from disk.
    
    Returns
    -------
    dict
        Dictionary of constant name -> value
    """
    path = get_constants_path()
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_constants(constants: Dict[str, Any]) -> bool:
    """Save user constants to disk.
    
    Parameters
    ----------
    constants : dict
        Dictionary of constant name -> value
    
    Returns
    -------
    bool
        True if saved successfully
    """
    path = get_constants_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(constants, f, indent=2, default=str)
        return True
    except IOError:
        return False


def add_constant(name: str, value: Any, constants: Dict[str, Any]) -> Tuple[bool, str]:
    """Add a new constant (immutable - cannot overwrite).
    
    Parameters
    ----------
    name : str
        Constant name
    value : Any
        Constant value
    constants : dict
        Existing constants dictionary (will be modified)
    
    Returns
    -------
    tuple
        (success: bool, message: str)
    """
    name = name.lower()
    if name in constants:
        return False, f"Constant '{name}' already exists and is immutable"
    
    constants[name] = value
    if save_constants(constants):
        return True, f"Constant '{name}' = {value} saved"
    else:
        return False, "Failed to save constants to disk"


# ============================================================================
# PREFERENCES MANAGEMENT
# ============================================================================

DEFAULT_PREFERENCES = {
    'bpm': 128.0,
    'step': 1.0,
    'attack': 0.01,
    'decay': 0.1,
    'sustain': 0.8,
    'release': 0.1,
    'filter_count': 0,
    'voice_count': 1,
    'carrier_count': 1,
    'autoplay': False,
    'note_duration': 1.0,
    'rest_duration': 0.0,
    'random_mode': 'off',
    'random_seed': 42,
    'default_bank': 'classic_fm',
    'dj_crossfade': 2.0,
    'dj_cue_offset': 0.0,
}


def load_preferences() -> Dict[str, Any]:
    """Load user preferences from disk.
    
    Returns
    -------
    dict
        Preferences dictionary with defaults filled in
    """
    prefs = DEFAULT_PREFERENCES.copy()
    path = get_preferences_path()
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                prefs.update(saved)
        except (json.JSONDecodeError, IOError):
            pass
    return prefs


def save_preferences(prefs: Dict[str, Any]) -> bool:
    """Save user preferences to disk.
    
    Parameters
    ----------
    prefs : dict
        Preferences dictionary
    
    Returns
    -------
    bool
        True if saved successfully
    """
    path = get_preferences_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(prefs, f, indent=2)
        return True
    except IOError:
        return False


# ============================================================================
# BANK MANAGEMENT
# ============================================================================

def list_banks(include_factory: bool = True) -> List[Dict[str, Any]]:
    """List all available routing banks.
    
    Parameters
    ----------
    include_factory : bool
        Include factory banks in listing
    
    Returns
    -------
    list
        List of bank info dicts: {name, path, factory, description}
    """
    banks = []
    
    # Factory banks
    if include_factory:
        factory_dir = get_banks_dir(factory=True)
        if factory_dir.exists():
            for f in factory_dir.glob('*.json'):
                try:
                    with open(f, 'r', encoding='utf-8') as fp:
                        data = json.load(fp)
                        banks.append({
                            'name': f.stem,
                            'path': str(f),
                            'factory': True,
                            'description': data.get('description', ''),
                            'algorithm_count': len(data.get('algorithms', [])),
                        })
                except (json.JSONDecodeError, IOError):
                    pass
    
    # User banks
    user_dir = get_banks_dir(factory=False)
    if user_dir.exists():
        for f in user_dir.glob('*.json'):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    banks.append({
                        'name': f.stem,
                        'path': str(f),
                        'factory': False,
                        'description': data.get('description', ''),
                        'algorithm_count': len(data.get('algorithms', [])),
                    })
            except (json.JSONDecodeError, IOError):
                pass
    
    return banks


def load_bank(name: str) -> Optional[Dict[str, Any]]:
    """Load a routing bank by name.
    
    Parameters
    ----------
    name : str
        Bank name (without .json extension)
    
    Returns
    -------
    dict or None
        Bank data or None if not found
    """
    # Try user banks first
    user_path = get_banks_dir(factory=False) / f'{name}.json'
    if user_path.exists():
        try:
            with open(user_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    # Try factory banks
    factory_path = get_banks_dir(factory=True) / f'{name}.json'
    if factory_path.exists():
        try:
            with open(factory_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    return None


def save_bank(name: str, bank_data: Dict[str, Any], factory: bool = False) -> bool:
    """Save a routing bank to disk.
    
    Parameters
    ----------
    name : str
        Bank name
    bank_data : dict
        Bank data including algorithms
    factory : bool
        Save as factory bank (requires special permissions)
    
    Returns
    -------
    bool
        True if saved successfully
    """
    path = get_banks_dir(factory=factory) / f'{name}.json'
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(bank_data, f, indent=2)
        return True
    except IOError:
        return False


# ============================================================================
# PRESET MANAGEMENT
# ============================================================================

def list_presets(category: Optional[str] = None, include_factory: bool = True) -> List[Dict[str, Any]]:
    """List all available synth presets.
    
    Parameters
    ----------
    category : str, optional
        Filter by category (bass, lead, pad, etc.)
    include_factory : bool
        Include factory presets
    
    Returns
    -------
    list
        List of preset info dicts
    """
    presets = []
    
    def scan_dir(base_dir: Path, factory: bool):
        if not base_dir.exists():
            return
        for f in base_dir.glob('*.json'):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    cat = data.get('category', 'misc')
                    if category is None or cat.lower() == category.lower():
                        presets.append({
                            'name': f.stem,
                            'path': str(f),
                            'factory': factory,
                            'category': cat,
                            'description': data.get('description', ''),
                        })
            except (json.JSONDecodeError, IOError):
                pass
    
    if include_factory:
        scan_dir(get_presets_dir(factory=True), factory=True)
    scan_dir(get_presets_dir(factory=False), factory=False)
    
    return presets


def load_preset(name: str) -> Optional[Dict[str, Any]]:
    """Load a synth preset by name."""
    # Try user presets first
    user_path = get_presets_dir(factory=False) / f'{name}.json'
    if user_path.exists():
        try:
            with open(user_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    # Try factory presets
    factory_path = get_presets_dir(factory=True) / f'{name}.json'
    if factory_path.exists():
        try:
            with open(factory_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    return None


def save_preset(name: str, preset_data: Dict[str, Any]) -> bool:
    """Save a synth preset to user directory."""
    path = get_presets_dir(factory=False) / f'{name}.json'
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(preset_data, f, indent=2)
        return True
    except IOError:
        return False


# ============================================================================
# PACK MANAGEMENT
# ============================================================================

def list_packs() -> List[Dict[str, Any]]:
    """List all installed sound packs.
    
    Returns
    -------
    list
        List of pack info dicts
    """
    packs = []
    packs_dir = get_packs_dir()
    
    for d in packs_dir.iterdir():
        if d.is_dir():
            manifest = d / 'pack.json'
            if manifest.exists():
                try:
                    with open(manifest, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        packs.append({
                            'name': data.get('name', d.name),
                            'path': str(d),
                            'author': data.get('author', 'Unknown'),
                            'version': data.get('version', '1.0'),
                            'description': data.get('description', ''),
                            'sample_count': data.get('sample_count', 0),
                        })
                except (json.JSONDecodeError, IOError):
                    # Directory exists but no valid manifest
                    packs.append({
                        'name': d.name,
                        'path': str(d),
                        'author': 'Unknown',
                        'version': '1.0',
                        'description': '',
                        'sample_count': len(list(d.glob('*.wav'))),
                    })
            else:
                # Count audio files
                wav_count = len(list(d.glob('*.wav')))
                packs.append({
                    'name': d.name,
                    'path': str(d),
                    'author': 'Unknown',
                    'version': '1.0',
                    'description': 'No manifest',
                    'sample_count': wav_count,
                })
    
    return packs


def get_pack_samples(pack_name: str) -> List[Path]:
    """Get all sample files in a pack.
    
    Parameters
    ----------
    pack_name : str
        Pack name
    
    Returns
    -------
    list
        List of Path objects for audio files
    """
    pack_dir = get_packs_dir() / pack_name
    if not pack_dir.exists():
        return []
    
    samples = []
    for ext in ('*.wav', '*.mp3', '*.flac', '*.ogg', '*.aiff'):
        samples.extend(pack_dir.glob(ext))
        samples.extend(pack_dir.glob(f'**/{ext}'))  # Recursive
    
    return sorted(set(samples))


def create_pack_manifest(pack_name: str, author: str = '', description: str = '') -> bool:
    """Create or update a pack manifest.
    
    Parameters
    ----------
    pack_name : str
        Pack name (folder name)
    author : str
        Author name
    description : str
        Pack description
    
    Returns
    -------
    bool
        True if created successfully
    """
    pack_dir = get_packs_dir() / pack_name
    if not pack_dir.exists():
        pack_dir.mkdir(parents=True)
    
    samples = get_pack_samples(pack_name)
    
    manifest = {
        'name': pack_name,
        'author': author,
        'version': '1.0',
        'description': description,
        'sample_count': len(samples),
        'created': datetime.now().isoformat(),
    }
    
    manifest_path = pack_dir / 'pack.json'
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        return True
    except IOError:
        return False


# ============================================================================
# SONG LIBRARY MANAGEMENT (DJ MODE)
# ============================================================================

def load_song_library() -> Dict[str, Any]:
    """Load the DJ song library.
    
    Returns
    -------
    dict
        Library data with songs and metadata
    """
    path = get_song_library_path()
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    return {
        'version': '1.0',
        'songs': [],
        'last_updated': None,
    }


def save_song_library(library: Dict[str, Any]) -> bool:
    """Save the DJ song library.
    
    Parameters
    ----------
    library : dict
        Library data
    
    Returns
    -------
    bool
        True if saved successfully
    """
    library['last_updated'] = datetime.now().isoformat()
    path = get_song_library_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(library, f, indent=2)
        return True
    except IOError:
        return False


def add_song_to_library(
    path: str,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    bpm: Optional[float] = None,
    key: Optional[str] = None,
    genre: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """Add a song to the DJ library.
    
    Parameters
    ----------
    path : str
        Path to audio file
    title : str, optional
        Song title (defaults to filename)
    artist : str, optional
        Artist name
    bpm : float, optional
        Song BPM
    key : str, optional
        Musical key (e.g., 'Am', 'C')
    genre : str, optional
        Genre tag
    tags : list, optional
        Additional tags
    
    Returns
    -------
    tuple
        (success: bool, message: str)
    """
    library = load_song_library()
    
    # Check if already in library
    for song in library['songs']:
        if song.get('path') == path:
            return False, "Song already in library"
    
    # Create song entry
    song_entry = {
        'path': path,
        'title': title or Path(path).stem,
        'artist': artist or 'Unknown',
        'bpm': bpm,
        'key': key,
        'genre': genre,
        'tags': tags or [],
        'added': datetime.now().isoformat(),
        'play_count': 0,
        'rating': 0,
        'cue_points': [],
    }
    
    library['songs'].append(song_entry)
    
    if save_song_library(library):
        return True, f"Added '{song_entry['title']}' to library"
    else:
        return False, "Failed to save library"


def list_playlists() -> List[Dict[str, Any]]:
    """List all playlists.
    
    Returns
    -------
    list
        List of playlist info dicts
    """
    playlists = []
    playlists_dir = get_playlists_dir()
    
    for f in playlists_dir.glob('*.json'):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                playlists.append({
                    'name': f.stem,
                    'path': str(f),
                    'song_count': len(data.get('songs', [])),
                    'description': data.get('description', ''),
                })
        except (json.JSONDecodeError, IOError):
            pass
    
    return playlists


def load_playlist(name: str) -> Optional[Dict[str, Any]]:
    """Load a playlist by name."""
    path = get_playlists_dir() / f'{name}.json'
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_playlist(name: str, songs: List[str], description: str = '') -> bool:
    """Save a playlist.
    
    Parameters
    ----------
    name : str
        Playlist name
    songs : list
        List of song paths or IDs
    description : str
        Playlist description
    
    Returns
    -------
    bool
        True if saved successfully
    """
    playlist = {
        'name': name,
        'description': description,
        'songs': songs,
        'created': datetime.now().isoformat(),
    }
    
    path = get_playlists_dir() / f'{name}.json'
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(playlist, f, indent=2)
        return True
    except IOError:
        return False


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_user_data() -> str:
    """Initialize all user data directories and files.
    
    Creates directory structure and default files if they don't exist.
    
    Returns
    -------
    str
        Status message
    """
    root = get_mdma_root()
    
    # Create all directories
    dirs_created = []
    for getter in [get_banks_dir, get_presets_dir, get_packs_dir, 
                   get_songs_dir, get_playlists_dir, get_projects_dir,
                   get_outputs_dir]:
        try:
            if getter == get_banks_dir or getter == get_presets_dir:
                getter(factory=True)
                getter(factory=False)
            else:
                getter()
            dirs_created.append(getter.__name__.replace('get_', '').replace('_dir', ''))
        except Exception:
            pass
    
    # Create default files if they don't exist
    if not get_constants_path().exists():
        save_constants({})
    
    if not get_preferences_path().exists():
        save_preferences(DEFAULT_PREFERENCES)
    
    if not get_song_library_path().exists():
        save_song_library({'version': '1.0', 'songs': []})
    
    if not get_outputs_index_path().exists():
        save_outputs_index({'version': '1.0', 'outputs': [], 'next_id': 1})
    
    if not get_functions_path().exists():
        save_functions({'version': '1.0', 'functions': {}, 'macros': {}})
    
    if not get_variables_path().exists():
        save_persistent_variables({})
    
    return f"MDMA user data initialized at: {root}"


def get_user_data_info() -> str:
    """Get information about user data locations.
    
    Returns
    -------
    str
        Formatted info string
    """
    root = get_mdma_root()
    
    lines = [
        "=== MDMA USER DATA ===",
        f"Root: {root}",
        "",
        "Directories:",
        f"  Constants: {get_constants_path()}",
        f"  Preferences: {get_preferences_path()}",
        f"  Banks: {get_banks_dir()}",
        f"  Presets: {get_presets_dir()}",
        f"  Packs: {get_packs_dir()}",
        f"  Songs: {get_songs_dir()}",
        f"  Projects: {get_projects_dir()}",
        "",
        f"Banks: {len(list_banks())}",
        f"Presets: {len(list_presets())}",
        f"Packs: {len(list_packs())}",
        f"Songs: {len(load_song_library().get('songs', []))}",
        f"Playlists: {len(list_playlists())}",
        f"Outputs: {len(list_outputs())}",
    ]
    
    return '\n'.join(lines)


# ============================================================================
# MIGRATION SYSTEM
# ============================================================================

def migrate_user_data(new_root: str, copy: bool = True) -> Tuple[bool, str]:
    """Migrate user data to a new location.
    
    Parameters
    ----------
    new_root : str
        New root directory path
    copy : bool
        If True, copy files (keeping originals)
        If False, move files (deleting originals)
    
    Returns
    -------
    tuple
        (success, message)
    """
    old_root = get_mdma_root()
    new_root = Path(new_root)
    
    if old_root == new_root:
        return False, "Source and destination are the same"
    
    if not old_root.exists():
        return False, f"Source directory not found: {old_root}"
    
    # Create new root
    try:
        new_root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"Cannot create destination: {e}"
    
    # Items to migrate
    items = [
        'constants.json',
        'preferences.json',
        'variables.json',
        'functions.json',
        'banks',
        'presets',
        'packs',
        'songs',
        'outputs',
        'projects',
    ]
    
    migrated = []
    errors = []
    
    for item in items:
        src = old_root / item
        dst = new_root / item
        
        if not src.exists():
            continue
        
        try:
            if copy:
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            else:
                shutil.move(str(src), str(dst))
            migrated.append(item)
        except Exception as e:
            errors.append(f"{item}: {e}")
    
    # Save migration record
    migration_record = {
        'from': str(old_root),
        'to': str(new_root),
        'timestamp': datetime.now().isoformat(),
        'mode': 'copy' if copy else 'move',
        'items': migrated,
        'errors': errors,
    }
    
    try:
        record_path = new_root / 'migration_log.json'
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(migration_record, f, indent=2)
    except:
        pass
    
    if errors:
        return False, f"Migrated {len(migrated)} items, {len(errors)} errors:\n" + "\n".join(errors)
    
    return True, f"Successfully migrated {len(migrated)} items to {new_root}"


def export_user_data(export_path: str, include_outputs: bool = False) -> Tuple[bool, str]:
    """Export user data to a zip file.
    
    Parameters
    ----------
    export_path : str
        Path for the export zip file
    include_outputs : bool
        Whether to include output renders (can be large)
    
    Returns
    -------
    tuple
        (success, message)
    """
    import zipfile
    
    root = get_mdma_root()
    export_path = Path(export_path)
    
    # Ensure .zip extension
    if not export_path.suffix.lower() == '.zip':
        export_path = export_path.with_suffix('.zip')
    
    items = [
        'constants.json',
        'preferences.json',
        'variables.json',
        'functions.json',
        'banks',
        'presets',
        'packs',
        'songs',
        'projects',
    ]
    
    if include_outputs:
        items.append('outputs')
    
    exported = []
    
    try:
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for item in items:
                src = root / item
                if not src.exists():
                    continue
                
                if src.is_dir():
                    for file_path in src.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(root)
                            zf.write(file_path, arcname)
                            exported.append(str(arcname))
                else:
                    zf.write(src, item)
                    exported.append(item)
        
        size_kb = export_path.stat().st_size // 1024
        return True, f"Exported {len(exported)} files to {export_path} ({size_kb}KB)"
    
    except Exception as e:
        return False, f"Export failed: {e}"


def import_user_data(import_path: str, merge: bool = True) -> Tuple[bool, str]:
    """Import user data from a zip file.
    
    Parameters
    ----------
    import_path : str
        Path to the import zip file
    merge : bool
        If True, merge with existing data
        If False, replace existing data
    
    Returns
    -------
    tuple
        (success, message)
    """
    import zipfile
    
    import_path = Path(import_path)
    root = get_mdma_root()
    
    if not import_path.exists():
        return False, f"Import file not found: {import_path}"
    
    if not zipfile.is_zipfile(import_path):
        return False, f"Not a valid zip file: {import_path}"
    
    imported = []
    
    try:
        with zipfile.ZipFile(import_path, 'r') as zf:
            for name in zf.namelist():
                dst = root / name
                
                # Skip if not merging and file exists
                if not merge and dst.exists():
                    continue
                
                # Create parent directories
                dst.parent.mkdir(parents=True, exist_ok=True)
                
                # Extract
                zf.extract(name, root)
                imported.append(name)
        
        return True, f"Imported {len(imported)} files from {import_path}"
    
    except Exception as e:
        return False, f"Import failed: {e}"


def get_data_size() -> Dict[str, int]:
    """Get size of each user data category in bytes.
    
    Returns
    -------
    dict
        Category name -> size in bytes
    """
    root = get_mdma_root()
    
    categories = {
        'constants': get_constants_path(),
        'preferences': get_preferences_path(),
        'variables': get_variables_path(),
        'functions': get_functions_path(),
        'banks': get_banks_dir(),
        'presets': get_presets_dir(),
        'packs': get_packs_dir(),
        'songs': get_songs_dir(),
        'outputs': get_outputs_dir(),
        'projects': get_projects_dir(),
    }
    
    sizes = {}
    
    for name, path in categories.items():
        if not path.exists():
            sizes[name] = 0
            continue
        
        if path.is_file():
            sizes[name] = path.stat().st_size
        else:
            # Sum all files in directory
            total = 0
            for f in path.rglob('*'):
                if f.is_file():
                    total += f.stat().st_size
            sizes[name] = total
    
    return sizes


def format_data_size() -> str:
    """Get formatted string of user data sizes.
    
    Returns
    -------
    str
        Formatted size report
    """
    sizes = get_data_size()
    total = sum(sizes.values())
    
    def fmt_size(b: int) -> str:
        if b < 1024:
            return f"{b}B"
        elif b < 1024 * 1024:
            return f"{b // 1024}KB"
        else:
            return f"{b // (1024 * 1024)}MB"
    
    lines = ["=== USER DATA SIZE ==="]
    for name, size in sorted(sizes.items(), key=lambda x: -x[1]):
        lines.append(f"  {name:12s}: {fmt_size(size):>8s}")
    lines.append(f"  {'TOTAL':12s}: {fmt_size(total):>8s}")
    
    return '\n'.join(lines)


def cleanup_old_outputs(keep_count: int = 50) -> Tuple[int, int]:
    """Clean up old outputs, keeping only the most recent.
    
    Parameters
    ----------
    keep_count : int
        Number of outputs to keep
    
    Returns
    -------
    tuple
        (deleted_count, freed_bytes)
    """
    index = load_outputs_index()
    outputs = index.get('outputs', [])
    
    if len(outputs) <= keep_count:
        return 0, 0
    
    # Sort by ID (oldest first)
    outputs_sorted = sorted(outputs, key=lambda x: x.get('id', 0))
    
    # Remove oldest
    to_remove = outputs_sorted[:-keep_count]
    to_keep = outputs_sorted[-keep_count:]
    
    deleted = 0
    freed = 0
    
    outputs_dir = get_outputs_dir()
    
    for entry in to_remove:
        filename = entry.get('filename', '')
        file_path = outputs_dir / filename
        
        if file_path.exists():
            freed += file_path.stat().st_size
            try:
                file_path.unlink()
                deleted += 1
            except:
                pass
    
    # Update index
    index['outputs'] = to_keep
    save_outputs_index(index)
    
    return deleted, freed

