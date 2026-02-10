"""MDMA rebuild package.

Music Design Made Accessible - Command-line audio production environment
designed for screen reader accessibility and zero-vision workflows.

VERSION: 45.0
BUILD ID: mdma_v45.0_20260203

FEATURES:
- Multi-buffer system for building complex sounds
- Unified 1-100 parameter scaling
- Enhanced audio-rate pattern modulation with 10 algorithms
- In-house audio playback (no external media player calls)
- Adaptive note duration (equal division based on audio length)
- Beat-based duration notation (D1, D0.5, R1, R0.5)
- Anti-artifact processing (click removal, crossfades)

V45 NEW FEATURES:
- Unified playback system (/P plays working buffer, /PT track, /PTS song)
- Parameter system (/PARM, /CHA) with LFO/envelope modulation
- HQ audio mode (/HQ) with FLAC output support
- 64-bit float internal processing

PLAYBACK COMMANDS:
- /p        - Play working buffer
- /pb <n>   - Play buffer n
- /pt <n>   - Play track n
- /pts      - Play song (all tracks)
- /pd <n>   - Play deck n

PARAMETER COMMANDS:
- /parm     - List/select parameters
- /cha      - Change parameter value or add modulation

HQ AUDIO COMMANDS:
- /hq       - High-quality mode settings
- /bout     - Output with format selection
"""

__version__ = "52.0"
__build__ = "mdma_v52.0_20260203"

__all__ = ["core", "dsp", "commands"]
