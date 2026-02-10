"""MDMA Routing Banks - Preset FM/AM/PM Algorithm Libraries.

Section E of MDMA Master Feature List.

Routing banks provide pre-configured operator routing algorithms
that users can load instead of manually wiring FM synthesis patches.

Bank Structure:
- Each bank contains 8-16 algorithms
- Each algorithm defines operator connections and modulation types
- Algorithms can be loaded into the Monolith engine

Algorithm Format:
{
    "name": "Algorithm name",
    "description": "What this algorithm sounds like",
    "operator_count": 6,
    "carriers": [0, 1],  # Which operators output to audio
    "routings": [
        {"type": "FM", "src": 2, "dst": 0, "amount": 50},
        {"type": "FM", "src": 3, "dst": 1, "amount": 50},
        ...
    ],
    "wave_hints": {
        0: "sine",  # Suggested wave types
        1: "sine",
        2: "sine",
        ...
    }
}
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple


# ============================================================================
# FACTORY BANKS - Built-in Algorithm Libraries
# ============================================================================

# Bank 1: Classic FM (DX7-style algorithms)
BANK_CLASSIC_FM = {
    "name": "classic_fm",
    "description": "Classic 6-operator FM algorithms inspired by DX7",
    "vibe": "Clean, classic, versatile",
    "algorithms": [
        # Algorithm 1: Simple 2-op FM (classic bell/bass)
        {
            "name": "simple_2op",
            "description": "Simple 2-op FM - bell tones, bass",
            "operator_count": 2,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 50}
            ],
            "wave_hints": {0: "sine", 1: "sine"}
        },
        # Algorithm 2: 2-op with feedback
        {
            "name": "2op_feedback",
            "description": "2-op with modulator feedback - edgier",
            "operator_count": 2,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 60},
                {"type": "FM", "src": 1, "dst": 1, "amount": 20}  # Self-feedback
            ],
            "wave_hints": {0: "sine", 1: "sine"}
        },
        # Algorithm 3: Serial 3-op chain
        {
            "name": "serial_3op",
            "description": "3-op serial chain - complex harmonics",
            "operator_count": 3,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 50},
                {"type": "FM", "src": 2, "dst": 1, "amount": 40}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine"}
        },
        # Algorithm 4: Parallel 2-carrier
        {
            "name": "parallel_2car",
            "description": "2 carriers, 2 modulators - thick sound",
            "operator_count": 4,
            "carriers": [0, 1],
            "routings": [
                {"type": "FM", "src": 2, "dst": 0, "amount": 50},
                {"type": "FM", "src": 3, "dst": 1, "amount": 50}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine"}
        },
        # Algorithm 5: Y-stack (2 mods into 1 carrier)
        {
            "name": "y_stack",
            "description": "2 modulators into 1 carrier - rich harmonics",
            "operator_count": 3,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 40},
                {"type": "FM", "src": 2, "dst": 0, "amount": 40}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine"}
        },
        # Algorithm 6: DX7 Algorithm 1 style
        {
            "name": "dx7_algo1",
            "description": "DX7 Algorithm 1 - brass, strings",
            "operator_count": 6,
            "carriers": [0, 1],
            "routings": [
                {"type": "FM", "src": 2, "dst": 1, "amount": 50},
                {"type": "FM", "src": 3, "dst": 0, "amount": 50},
                {"type": "FM", "src": 4, "dst": 3, "amount": 40},
                {"type": "FM", "src": 5, "dst": 4, "amount": 30}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine", 4: "sine", 5: "sine"}
        },
        # Algorithm 7: DX7 Algorithm 5 style
        {
            "name": "dx7_algo5",
            "description": "DX7 Algorithm 5 - electric piano",
            "operator_count": 6,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 50},
                {"type": "FM", "src": 2, "dst": 0, "amount": 30},
                {"type": "FM", "src": 3, "dst": 2, "amount": 40},
                {"type": "FM", "src": 4, "dst": 1, "amount": 35},
                {"type": "FM", "src": 5, "dst": 4, "amount": 25}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine", 4: "sine", 5: "sine"}
        },
        # Algorithm 8: DX7 Algorithm 32 style (all carriers)
        {
            "name": "dx7_algo32",
            "description": "All carriers - organ/additive",
            "operator_count": 6,
            "carriers": [0, 1, 2, 3, 4, 5],
            "routings": [],  # No modulation - pure additive
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine", 4: "sine", 5: "sine"}
        },
        # Algorithm 9: Deep serial
        {
            "name": "deep_serial",
            "description": "6-op serial chain - extreme harmonics",
            "operator_count": 6,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 40},
                {"type": "FM", "src": 2, "dst": 1, "amount": 35},
                {"type": "FM", "src": 3, "dst": 2, "amount": 30},
                {"type": "FM", "src": 4, "dst": 3, "amount": 25},
                {"type": "FM", "src": 5, "dst": 4, "amount": 20}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine", 4: "sine", 5: "sine"}
        },
        # Algorithm 10: Pyramid
        {
            "name": "pyramid",
            "description": "Pyramid structure - wide stereo",
            "operator_count": 6,
            "carriers": [0, 1, 2],
            "routings": [
                {"type": "FM", "src": 3, "dst": 0, "amount": 50},
                {"type": "FM", "src": 4, "dst": 1, "amount": 50},
                {"type": "FM", "src": 5, "dst": 2, "amount": 50},
                {"type": "FM", "src": 5, "dst": 3, "amount": 30},
                {"type": "FM", "src": 5, "dst": 4, "amount": 30}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine", 4: "sine", 5: "sine"}
        },
        # Algorithm 11: Bell
        {
            "name": "bell",
            "description": "Bell/chime optimized - inharmonic",
            "operator_count": 4,
            "carriers": [0, 1],
            "routings": [
                {"type": "FM", "src": 2, "dst": 0, "amount": 70},
                {"type": "FM", "src": 3, "dst": 1, "amount": 60}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine"},
            "freq_ratios": {0: 1.0, 1: 2.0, 2: 3.5, 3: 5.4}  # Inharmonic
        },
        # Algorithm 12: Bass
        {
            "name": "bass",
            "description": "Bass optimized - fat low end",
            "operator_count": 3,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 80},
                {"type": "FM", "src": 2, "dst": 1, "amount": 40}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine"},
            "freq_ratios": {0: 1.0, 1: 1.0, 2: 2.0}
        },
        # Algorithm 13: Pad
        {
            "name": "pad",
            "description": "Pad/string optimized - lush",
            "operator_count": 6,
            "carriers": [0, 1, 2],
            "routings": [
                {"type": "FM", "src": 3, "dst": 0, "amount": 30},
                {"type": "FM", "src": 4, "dst": 1, "amount": 25},
                {"type": "FM", "src": 5, "dst": 2, "amount": 20}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine", 4: "sine", 5: "sine"}
        },
        # Algorithm 14: Pluck
        {
            "name": "pluck",
            "description": "Plucked string optimized",
            "operator_count": 4,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 60},
                {"type": "FM", "src": 2, "dst": 0, "amount": 40},
                {"type": "FM", "src": 3, "dst": 2, "amount": 30}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine"}
        },
        # Algorithm 15: Lead
        {
            "name": "lead",
            "description": "Lead synth - cutting",
            "operator_count": 4,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 70},
                {"type": "FM", "src": 2, "dst": 1, "amount": 50},
                {"type": "FM", "src": 3, "dst": 1, "amount": 30}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine"}
        },
        # Algorithm 16: Keys
        {
            "name": "keys",
            "description": "Electric piano / keys",
            "operator_count": 4,
            "carriers": [0, 1],
            "routings": [
                {"type": "FM", "src": 2, "dst": 0, "amount": 55},
                {"type": "FM", "src": 3, "dst": 1, "amount": 45}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine"},
            "freq_ratios": {0: 1.0, 1: 1.0, 2: 14.0, 3: 7.0}
        }
    ]
}


# Bank 2: Modern Aggressive
BANK_MODERN_AGGRESSIVE = {
    "name": "modern_aggressive",
    "description": "Modern aggressive FM for bass music, dubstep",
    "vibe": "Heavy, growling, aggressive",
    "algorithms": [
        {
            "name": "growl_basic",
            "description": "Basic growl bass",
            "operator_count": 3,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 90},
                {"type": "FM", "src": 2, "dst": 1, "amount": 60}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "saw"}
        },
        {
            "name": "reese",
            "description": "Reese bass - detuned saw",
            "operator_count": 2,
            "carriers": [0, 1],
            "routings": [
                {"type": "FM", "src": 0, "dst": 1, "amount": 20}
            ],
            "wave_hints": {0: "saw", 1: "saw"},
            "detune": {1: 0.5}  # Slight detune
        },
        {
            "name": "neuro",
            "description": "Neuro bass - complex modulation",
            "operator_count": 4,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 100},
                {"type": "FM", "src": 2, "dst": 1, "amount": 80},
                {"type": "FM", "src": 3, "dst": 2, "amount": 60},
                {"type": "FM", "src": 3, "dst": 3, "amount": 30}  # Feedback
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine"}
        },
        {
            "name": "yoi",
            "description": "Yoi bass - vowel-like",
            "operator_count": 3,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 85},
                {"type": "PM", "src": 2, "dst": 0, "amount": 40}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "triangle"}
        },
        {
            "name": "screech",
            "description": "Screech lead - harsh",
            "operator_count": 3,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 120},
                {"type": "FM", "src": 2, "dst": 1, "amount": 100}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine"}
        },
        {
            "name": "tear",
            "description": "Tearout bass",
            "operator_count": 4,
            "carriers": [0, 1],
            "routings": [
                {"type": "FM", "src": 2, "dst": 0, "amount": 95},
                {"type": "FM", "src": 3, "dst": 1, "amount": 85},
                {"type": "RM", "src": 0, "dst": 1, "amount": 30}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine"}
        },
        {
            "name": "riddim",
            "description": "Riddim bass - punchy",
            "operator_count": 2,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 75}
            ],
            "wave_hints": {0: "sine", 1: "sine"},
            "freq_ratios": {0: 1.0, 1: 1.0}
        },
        {
            "name": "foghorn",
            "description": "Foghorn bass - massive",
            "operator_count": 4,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 110},
                {"type": "FM", "src": 2, "dst": 0, "amount": 50},
                {"type": "FM", "src": 3, "dst": 1, "amount": 70}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine"}
        }
    ]
}


# Bank 3: Through-Zero FM
BANK_TFM = {
    "name": "thru_zero_fm",
    "description": "Through-zero FM algorithms for harsh, digital sounds",
    "vibe": "Digital, harsh, metallic",
    "algorithms": [
        {
            "name": "tfm_basic",
            "description": "Basic through-zero FM",
            "operator_count": 2,
            "carriers": [0],
            "routings": [
                {"type": "TFM", "src": 1, "dst": 0, "amount": 50}
            ],
            "wave_hints": {0: "sine", 1: "sine"}
        },
        {
            "name": "tfm_harsh",
            "description": "Harsh TFM - extreme",
            "operator_count": 3,
            "carriers": [0],
            "routings": [
                {"type": "TFM", "src": 1, "dst": 0, "amount": 80},
                {"type": "TFM", "src": 2, "dst": 1, "amount": 60}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine"}
        },
        {
            "name": "tfm_digital",
            "description": "Digital texture",
            "operator_count": 4,
            "carriers": [0, 1],
            "routings": [
                {"type": "TFM", "src": 2, "dst": 0, "amount": 70},
                {"type": "TFM", "src": 3, "dst": 1, "amount": 65},
                {"type": "FM", "src": 3, "dst": 2, "amount": 40}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine"}
        },
        {
            "name": "tfm_glitch",
            "description": "Glitchy TFM",
            "operator_count": 3,
            "carriers": [0],
            "routings": [
                {"type": "TFM", "src": 1, "dst": 0, "amount": 100},
                {"type": "TFM", "src": 2, "dst": 0, "amount": 80}
            ],
            "wave_hints": {0: "sine", 1: "triangle", 2: "sine"}
        }
    ]
}


# Bank 4: Phase Modulation
BANK_PM = {
    "name": "phase_mod",
    "description": "Phase modulation algorithms",
    "vibe": "Clean, warm, Casio-like",
    "algorithms": [
        {
            "name": "pm_basic",
            "description": "Basic phase modulation",
            "operator_count": 2,
            "carriers": [0],
            "routings": [
                {"type": "PM", "src": 1, "dst": 0, "amount": 50}
            ],
            "wave_hints": {0: "sine", 1: "sine"}
        },
        {
            "name": "pm_epiano",
            "description": "PM electric piano",
            "operator_count": 4,
            "carriers": [0, 1],
            "routings": [
                {"type": "PM", "src": 2, "dst": 0, "amount": 45},
                {"type": "PM", "src": 3, "dst": 1, "amount": 40}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine"}
        },
        {
            "name": "pm_organ",
            "description": "PM organ",
            "operator_count": 4,
            "carriers": [0, 1, 2, 3],
            "routings": [],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine"},
            "freq_ratios": {0: 0.5, 1: 1.0, 2: 2.0, 3: 4.0}
        },
        {
            "name": "pm_strings",
            "description": "PM string pad",
            "operator_count": 4,
            "carriers": [0, 1],
            "routings": [
                {"type": "PM", "src": 2, "dst": 0, "amount": 30},
                {"type": "PM", "src": 3, "dst": 1, "amount": 25}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine"}
        }
    ]
}


# Bank 5: Ring Modulation / AM
BANK_RINGMOD = {
    "name": "ring_am",
    "description": "Ring modulation and amplitude modulation",
    "vibe": "Metallic, bell-like, tremolo",
    "algorithms": [
        {
            "name": "rm_bell",
            "description": "Ring mod bell",
            "operator_count": 2,
            "carriers": [0],
            "routings": [
                {"type": "RM", "src": 1, "dst": 0, "amount": 60}
            ],
            "wave_hints": {0: "sine", 1: "sine"},
            "freq_ratios": {0: 1.0, 1: 2.4}
        },
        {
            "name": "rm_metallic",
            "description": "Metallic ring mod",
            "operator_count": 3,
            "carriers": [0],
            "routings": [
                {"type": "RM", "src": 1, "dst": 0, "amount": 70},
                {"type": "RM", "src": 2, "dst": 0, "amount": 50}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine"}
        },
        {
            "name": "am_tremolo",
            "description": "AM tremolo effect",
            "operator_count": 2,
            "carriers": [0],
            "routings": [
                {"type": "AM", "src": 1, "dst": 0, "amount": 40}
            ],
            "wave_hints": {0: "sine", 1: "sine"},
            "freq_ratios": {0: 1.0, 1: 0.1}  # LFO-rate modulator
        },
        {
            "name": "am_pwm",
            "description": "AM pseudo-PWM",
            "operator_count": 2,
            "carriers": [0],
            "routings": [
                {"type": "AM", "src": 1, "dst": 0, "amount": 50}
            ],
            "wave_hints": {0: "pulse", 1: "sine"},
            "freq_ratios": {0: 1.0, 1: 0.05}
        }
    ]
}


# Bank 6: Hybrid (Mixed modulation types)
BANK_HYBRID = {
    "name": "hybrid",
    "description": "Hybrid algorithms combining FM, PM, AM, RM",
    "vibe": "Complex, evolving, unique",
    "algorithms": [
        {
            "name": "fm_rm_combo",
            "description": "FM + Ring Mod combo",
            "operator_count": 3,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 50},
                {"type": "RM", "src": 2, "dst": 0, "amount": 30}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine"}
        },
        {
            "name": "pm_am_layers",
            "description": "PM + AM layered",
            "operator_count": 4,
            "carriers": [0, 1],
            "routings": [
                {"type": "PM", "src": 2, "dst": 0, "amount": 45},
                {"type": "AM", "src": 3, "dst": 1, "amount": 35}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine"}
        },
        {
            "name": "fm_tfm_mix",
            "description": "FM + TFM mixed",
            "operator_count": 4,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 40},
                {"type": "TFM", "src": 2, "dst": 0, "amount": 30},
                {"type": "FM", "src": 3, "dst": 1, "amount": 35}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine"}
        },
        {
            "name": "all_mod_types",
            "description": "All modulation types",
            "operator_count": 5,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 35},
                {"type": "PM", "src": 2, "dst": 0, "amount": 25},
                {"type": "AM", "src": 3, "dst": 0, "amount": 20},
                {"type": "RM", "src": 4, "dst": 0, "amount": 15}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine", 4: "sine"}
        }
    ]
}


# Bank 7: Physical Modeling Leaning
BANK_PHYSICAL = {
    "name": "physical",
    "description": "Algorithms optimized for physical modeling waves",
    "vibe": "Acoustic, organic, natural",
    "algorithms": [
        {
            "name": "phys_mallet",
            "description": "Physical mallet percussion",
            "operator_count": 2,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 40}
            ],
            "wave_hints": {0: "physical", 1: "sine"}
        },
        {
            "name": "phys_string",
            "description": "Physical string model",
            "operator_count": 3,
            "carriers": [0, 1],
            "routings": [
                {"type": "FM", "src": 2, "dst": 0, "amount": 30},
                {"type": "PM", "src": 2, "dst": 1, "amount": 25}
            ],
            "wave_hints": {0: "physical2", 1: "physical", 2: "sine"}
        },
        {
            "name": "phys_piano",
            "description": "Physical piano-like",
            "operator_count": 4,
            "carriers": [0, 1],
            "routings": [
                {"type": "FM", "src": 2, "dst": 0, "amount": 35},
                {"type": "FM", "src": 3, "dst": 1, "amount": 30}
            ],
            "wave_hints": {0: "physical2", 1: "physical2", 2: "sine", 3: "sine"}
        },
        {
            "name": "phys_bell",
            "description": "Physical bell/chime",
            "operator_count": 3,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 50},
                {"type": "RM", "src": 2, "dst": 0, "amount": 25}
            ],
            "wave_hints": {0: "physical", 1: "physical", 2: "sine"}
        }
    ]
}


# Bank 8: Experimental / Glitch / Metallic
BANK_EXPERIMENTAL = {
    "name": "experimental",
    "description": "Experimental, glitchy, extreme algorithms",
    "vibe": "Weird, metallic, harsh, glitchy",
    "algorithms": [
        {
            "name": "exp_chaos",
            "description": "Chaotic feedback",
            "operator_count": 3,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 100},
                {"type": "FM", "src": 2, "dst": 1, "amount": 100},
                {"type": "FM", "src": 0, "dst": 2, "amount": 50}  # Feedback loop
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine"}
        },
        {
            "name": "exp_noise",
            "description": "Noise-modulated",
            "operator_count": 3,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 60},
                {"type": "AM", "src": 2, "dst": 0, "amount": 40}
            ],
            "wave_hints": {0: "sine", 1: "noise", 2: "pink"}
        },
        {
            "name": "exp_metallic",
            "description": "Extreme metallic",
            "operator_count": 4,
            "carriers": [0],
            "routings": [
                {"type": "RM", "src": 1, "dst": 0, "amount": 80},
                {"type": "RM", "src": 2, "dst": 0, "amount": 60},
                {"type": "TFM", "src": 3, "dst": 1, "amount": 50}
            ],
            "wave_hints": {0: "sine", 1: "sine", 2: "sine", 3: "sine"}
        },
        {
            "name": "exp_bit",
            "description": "Bitcrushed texture",
            "operator_count": 2,
            "carriers": [0],
            "routings": [
                {"type": "FM", "src": 1, "dst": 0, "amount": 120}
            ],
            "wave_hints": {0: "pulse", 1: "pulse"}
        }
    ]
}


# ============================================================================
# BANK REGISTRY
# ============================================================================

FACTORY_BANKS = {
    'classic_fm': BANK_CLASSIC_FM,
    'modern_aggressive': BANK_MODERN_AGGRESSIVE,
    'thru_zero_fm': BANK_TFM,
    'phase_mod': BANK_PM,
    'ring_am': BANK_RINGMOD,
    'hybrid': BANK_HYBRID,
    'physical': BANK_PHYSICAL,
    'experimental': BANK_EXPERIMENTAL,
}

# Short aliases
BANK_ALIASES = {
    'classic': 'classic_fm',
    'dx7': 'classic_fm',
    'fm': 'classic_fm',
    'aggro': 'modern_aggressive',
    'bass': 'modern_aggressive',
    'dubstep': 'modern_aggressive',
    'tfm': 'thru_zero_fm',
    'tzfm': 'thru_zero_fm',
    'pm': 'phase_mod',
    'casio': 'phase_mod',
    'rm': 'ring_am',
    'am': 'ring_am',
    'ring': 'ring_am',
    'mix': 'hybrid',
    'phys': 'physical',
    'acoustic': 'physical',
    'exp': 'experimental',
    'glitch': 'experimental',
    'weird': 'experimental',
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_bank(name: str) -> Optional[Dict[str, Any]]:
    """Get a bank by name or alias.
    
    Parameters
    ----------
    name : str
        Bank name or alias
    
    Returns
    -------
    dict or None
        Bank data or None if not found
    """
    name_lower = name.lower()
    
    # Check aliases
    if name_lower in BANK_ALIASES:
        name_lower = BANK_ALIASES[name_lower]
    
    # Check factory banks
    if name_lower in FACTORY_BANKS:
        return FACTORY_BANKS[name_lower]
    
    # Try loading from user data
    from .user_data import load_bank
    return load_bank(name_lower)


def get_algorithm(bank_name: str, algorithm_idx: int) -> Optional[Dict[str, Any]]:
    """Get a specific algorithm from a bank.
    
    Parameters
    ----------
    bank_name : str
        Bank name or alias
    algorithm_idx : int
        Algorithm index (0-based)
    
    Returns
    -------
    dict or None
        Algorithm data or None if not found
    """
    bank = get_bank(bank_name)
    if bank is None:
        return None
    
    algorithms = bank.get('algorithms', [])
    if 0 <= algorithm_idx < len(algorithms):
        return algorithms[algorithm_idx]
    
    return None


def get_algorithm_by_name(bank_name: str, algo_name: str) -> Optional[Dict[str, Any]]:
    """Get an algorithm by name from a bank.
    
    Parameters
    ----------
    bank_name : str
        Bank name or alias
    algo_name : str
        Algorithm name
    
    Returns
    -------
    dict or None
        Algorithm data or None if not found
    """
    bank = get_bank(bank_name)
    if bank is None:
        return None
    
    algo_name_lower = algo_name.lower()
    for algo in bank.get('algorithms', []):
        if algo.get('name', '').lower() == algo_name_lower:
            return algo
    
    return None


def list_bank_names() -> List[str]:
    """List all available bank names."""
    return list(FACTORY_BANKS.keys())


def list_bank_algorithms(bank_name: str) -> List[Tuple[int, str, str]]:
    """List algorithms in a bank.
    
    Parameters
    ----------
    bank_name : str
        Bank name or alias
    
    Returns
    -------
    list
        List of (index, name, description) tuples
    """
    bank = get_bank(bank_name)
    if bank is None:
        return []
    
    return [
        (i, algo.get('name', f'algo_{i}'), algo.get('description', ''))
        for i, algo in enumerate(bank.get('algorithms', []))
    ]


def apply_algorithm_to_engine(engine, algorithm: Dict[str, Any]) -> str:
    """Apply an algorithm's routing to a Monolith engine.
    
    Parameters
    ----------
    engine : MonolithEngine
        The synth engine to configure
    algorithm : dict
        Algorithm data
    
    Returns
    -------
    str
        Status message
    """
    # Clear existing routings
    engine.clear_algorithms()
    
    # Set up operators with wave hints
    op_count = algorithm.get('operator_count', 2)
    wave_hints = algorithm.get('wave_hints', {})
    freq_ratios = algorithm.get('freq_ratios', {})
    
    for i in range(op_count):
        wave = wave_hints.get(i, 'sine')
        freq = 440.0 * freq_ratios.get(i, 1.0) if freq_ratios else 440.0
        engine.set_operator(i, wave_type=wave, freq=freq)
    
    # Apply routings
    routings = algorithm.get('routings', [])
    for routing in routings:
        mod_type = routing.get('type', 'FM')
        src = routing.get('src', 0)
        dst = routing.get('dst', 0)
        amount = routing.get('amount', 50)
        engine.add_algorithm(mod_type, src, dst, amount)
    
    algo_name = algorithm.get('name', 'unnamed')
    return f"Applied algorithm '{algo_name}' ({len(routings)} routings, {op_count} operators)"
