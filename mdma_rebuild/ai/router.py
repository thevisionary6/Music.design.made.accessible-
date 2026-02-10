"""MDMA AI Command Router (ACR).

Translates user intent into validated MDMA command sequences.

The router proposes plans.
MDMA validates and executes.

Router Modes:
- Standard: single-command suggestion
- /high: multi-command execution planning (high-risk, powerful)

Safety Rules:
- AI cannot invent commands
- AI cannot bypass file safety
- Destructive commands require confirmation
- All router actions are logged
- Validation always precedes execution
"""

from __future__ import annotations

import re
import json
from typing import List, Dict, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field, asdict
from enum import Enum

if TYPE_CHECKING:
    from ..core.session import Session


# ============================================================================
# ROUTER TYPES
# ============================================================================

class Intent(Enum):
    EXECUTE = "execute"
    EXPLAIN = "explain"
    ANALYZE = "analyze"
    CLARIFY = "clarify"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class CommandStep:
    """A single step in a command plan."""
    step_id: int
    command: str
    arguments: List[str]
    rationale: str
    dependencies: List[int] = field(default_factory=list)


@dataclass
class RouterPlan:
    """Output schema for the AI router (strict)."""
    intent: Intent
    confidence: float  # 0.0 - 1.0
    risk_level: RiskLevel
    requires_confirmation: bool
    plan: List[CommandStep]
    notes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'intent': self.intent.value,
            'confidence': self.confidence,
            'risk_level': self.risk_level.value,
            'requires_confirmation': self.requires_confirmation,
            'plan': [asdict(step) for step in self.plan],
            'notes': self.notes,
        }
    
    def is_valid(self) -> bool:
        """Check if plan is valid."""
        if not 0.0 <= self.confidence <= 1.0:
            return False
        if not self.plan:
            return False
        return all(step.command for step in self.plan)


# ============================================================================
# COMMAND KNOWLEDGE BASE
# ============================================================================

# Commands the router knows about and can suggest
KNOWN_COMMANDS = {
    # Buffer operations
    'buf': {'category': 'buffer', 'risk': 'low', 'desc': 'Show buffer info'},
    'bu': {'category': 'buffer', 'risk': 'low', 'desc': 'Select buffer by index'},
    'bu+': {'category': 'buffer', 'risk': 'low', 'desc': 'Set buffer count'},
    'a': {'category': 'buffer', 'risk': 'low', 'desc': 'Append audio to buffer'},
    'bov': {'category': 'buffer', 'risk': 'low', 'desc': 'Overlay audio on buffer'},
    'ex': {'category': 'buffer', 'risk': 'low', 'desc': 'Extend buffer'},
    
    # Playback
    'p': {'category': 'playback', 'risk': 'low', 'desc': 'Play buffer'},
    'pa': {'category': 'playback', 'risk': 'low', 'desc': 'Play all buffers'},
    'stop': {'category': 'playback', 'risk': 'low', 'desc': 'Stop playback'},
    
    # Render
    'b': {'category': 'render', 'risk': 'medium', 'desc': 'Render buffer to file'},
    'ba': {'category': 'render', 'risk': 'medium', 'desc': 'Render all buffers'},
    'bo': {'category': 'render', 'risk': 'medium', 'desc': 'Render omni (combined)'},
    'bpa': {'category': 'render', 'risk': 'medium', 'desc': 'Export as sample pack'},
    
    # Synthesis
    'tone': {'category': 'synth', 'risk': 'low', 'desc': 'Generate tone'},
    'op': {'category': 'synth', 'risk': 'low', 'desc': 'Select operator'},
    'wm': {'category': 'synth', 'risk': 'low', 'desc': 'Set waveform'},
    'fr': {'category': 'synth', 'risk': 'low', 'desc': 'Set frequency'},
    'amp': {'category': 'synth', 'risk': 'low', 'desc': 'Set amplitude'},
    'atk': {'category': 'synth', 'risk': 'low', 'desc': 'Set attack'},
    'dec': {'category': 'synth', 'risk': 'low', 'desc': 'Set decay'},
    'sus': {'category': 'synth', 'risk': 'low', 'desc': 'Set sustain'},
    'rel': {'category': 'synth', 'risk': 'low', 'desc': 'Set release'},
    'vc': {'category': 'synth', 'risk': 'low', 'desc': 'Set voice count'},
    'dt': {'category': 'synth', 'risk': 'low', 'desc': 'Set detune'},
    
    # Modulation routing
    'fm': {'category': 'routing', 'risk': 'low', 'desc': 'FM routing'},
    'am': {'category': 'routing', 'risk': 'low', 'desc': 'AM routing'},
    'rm': {'category': 'routing', 'risk': 'low', 'desc': 'Ring mod routing'},
    'pm': {'category': 'routing', 'risk': 'low', 'desc': 'Phase mod routing'},
    'rt': {'category': 'routing', 'risk': 'low', 'desc': 'View/clear routings'},
    
    # Banks and presets
    'bk': {'category': 'preset', 'risk': 'low', 'desc': 'Select bank'},
    'al': {'category': 'preset', 'risk': 'low', 'desc': 'Select algorithm'},
    'preset': {'category': 'preset', 'risk': 'low', 'desc': 'Preset management'},
    
    # Effects
    'fxa': {'category': 'fx', 'risk': 'low', 'desc': 'Add effect'},
    'fx': {'category': 'fx', 'risk': 'low', 'desc': 'Effect chain management'},
    'bfx': {'category': 'fx', 'risk': 'low', 'desc': 'Buffer FX chain'},
    'vamp': {'category': 'fx', 'risk': 'low', 'desc': 'Vamp/overdrive'},
    'dual': {'category': 'fx', 'risk': 'low', 'desc': 'Dual overdrive'},
    'conv': {'category': 'fx', 'risk': 'low', 'desc': 'Convolution reverb'},
    'fc': {'category': 'fx', 'risk': 'low', 'desc': 'Forever Compression'},
    'gg': {'category': 'fx', 'risk': 'low', 'desc': 'Giga Gate'},
    
    # Filters
    'fs': {'category': 'filter', 'risk': 'low', 'desc': 'Select filter slot'},
    'ft': {'category': 'filter', 'risk': 'low', 'desc': 'Set filter type'},
    'cut': {'category': 'filter', 'risk': 'low', 'desc': 'Set cutoff'},
    'res': {'category': 'filter', 'risk': 'low', 'desc': 'Set resonance'},
    
    # Pattern
    'pat': {'category': 'pattern', 'risk': 'low', 'desc': 'Apply pattern'},
    'apat': {'category': 'pattern', 'risk': 'low', 'desc': 'Apply pattern to all'},
    'arp': {'category': 'pattern', 'risk': 'low', 'desc': 'Arpeggio'},
    
    # Generators
    'g': {'category': 'generator', 'risk': 'low', 'desc': 'Sound generator'},
    
    # AI
    'gen': {'category': 'ai', 'risk': 'medium', 'desc': 'AI audio generation'},
    'genv': {'category': 'ai', 'risk': 'medium', 'desc': 'Generate variations'},
    'analyze': {'category': 'ai', 'risk': 'low', 'desc': 'Analyze audio'},
    'breed': {'category': 'ai', 'risk': 'low', 'desc': 'Breed samples'},
    'evolve': {'category': 'ai', 'risk': 'medium', 'desc': 'Evolve samples'},
    'mutate': {'category': 'ai', 'risk': 'low', 'desc': 'Mutate audio'},
    
    # Pack
    'pack': {'category': 'pack', 'risk': 'medium', 'desc': 'Pack management'},
    
    # Granular
    'gr': {'category': 'granular', 'risk': 'low', 'desc': 'Granular engine'},
    
    # Chunk
    'ch': {'category': 'chunk', 'risk': 'low', 'desc': 'Chunk system'},
    
    # Project
    'save': {'category': 'project', 'risk': 'high', 'desc': 'Save project'},
    'load': {'category': 'project', 'risk': 'high', 'desc': 'Load project'},
    'new': {'category': 'project', 'risk': 'medium', 'desc': 'New project'},
}


# Intent patterns for natural language understanding
INTENT_PATTERNS = {
    # Synthesis requests
    r'\b(make|create|generate|synth|build)\s+(a\s+)?(kick|snare|bass|lead|pad|tone|sound)\b': 
        {'intent': 'synth', 'type_match': 4},
    
    # Effect requests
    r'\b(add|apply|put)\s+(some\s+)?(reverb|delay|distortion|compression|filter|fx|effect)\b':
        {'intent': 'fx', 'type_match': 3},
    
    # Analysis requests
    r'\b(analyze|describe|what\s+is|tell\s+me\s+about|show)\s+(the\s+)?(sound|audio|buffer|sample)\b':
        {'intent': 'analyze'},
    
    # Playback requests
    r'\b(play|listen|hear|preview)\b':
        {'intent': 'play'},
    
    # Render requests
    r'\b(render|export|save\s+as|bounce|output)\b':
        {'intent': 'render'},
    
    # Breeding requests
    r'\b(breed|combine|mix|cross|merge)\s+(the\s+)?(samples|buffers|sounds)\b':
        {'intent': 'breed'},
    
    # Evolution requests
    r'\b(evolve|mutate|vary|variate|transform)\b':
        {'intent': 'evolve'},
    
    # Help/explain requests
    r'\b(how\s+do\s+i|explain|help|what\s+does|how\s+to)\b':
        {'intent': 'explain'},
}


# ============================================================================
# COMMAND ROUTER
# ============================================================================

class AICommandRouter:
    """AI-powered command router for MDMA.
    
    Translates natural language and ambiguous input into
    validated MDMA command sequences.
    """
    
    def __init__(self, command_table: Dict[str, callable]):
        """Initialize with the command table.
        
        Parameters
        ----------
        command_table : dict
            Mapping of command names to functions
        """
        self.command_table = command_table
        self.history: List[RouterPlan] = []
    
    def route(
        self,
        user_input: str,
        session: "Session",
        high_mode: bool = False,
    ) -> RouterPlan:
        """Route user input to command plan.
        
        Parameters
        ----------
        user_input : str
            Raw user input (may be natural language)
        session : Session
            Current session for context
        high_mode : bool
            If True, allow multi-command chaining
        
        Returns
        -------
        RouterPlan
            Proposed command plan
        """
        user_input = user_input.strip()
        
        # Check if it's already a valid command
        if user_input.startswith('/'):
            cmd_match = self._parse_command(user_input)
            if cmd_match:
                return self._create_direct_plan(cmd_match)
        
        # Natural language processing
        intent_info = self._detect_intent(user_input)
        
        if high_mode:
            return self._create_high_plan(user_input, intent_info, session)
        else:
            return self._create_standard_plan(user_input, intent_info, session)
    
    def _parse_command(self, cmd_str: str) -> Optional[Tuple[str, List[str]]]:
        """Parse a /command string."""
        if not cmd_str.startswith('/'):
            return None
        
        parts = cmd_str[1:].split()
        if not parts:
            return None
        
        cmd_name = parts[0].lower()
        args = parts[1:]
        
        if cmd_name in self.command_table or cmd_name in KNOWN_COMMANDS:
            return (cmd_name, args)
        
        return None
    
    def _detect_intent(self, text: str) -> Dict[str, Any]:
        """Detect intent from natural language."""
        text_lower = text.lower()
        
        for pattern, info in INTENT_PATTERNS.items():
            match = re.search(pattern, text_lower)
            if match:
                result = {'pattern': pattern, **info}
                if 'type_match' in info:
                    try:
                        result['matched_type'] = match.group(info['type_match'])
                    except IndexError:
                        pass
                return result
        
        return {'intent': 'unknown'}
    
    def _create_direct_plan(self, cmd_match: Tuple[str, List[str]]) -> RouterPlan:
        """Create plan for a directly specified command."""
        cmd_name, args = cmd_match
        cmd_info = KNOWN_COMMANDS.get(cmd_name, {'risk': 'low'})
        
        risk = RiskLevel[cmd_info.get('risk', 'low').upper()]
        
        return RouterPlan(
            intent=Intent.EXECUTE,
            confidence=1.0,
            risk_level=risk,
            requires_confirmation=risk == RiskLevel.HIGH,
            plan=[CommandStep(
                step_id=1,
                command=cmd_name,
                arguments=args,
                rationale="Direct command input",
            )],
        )
    
    def _create_standard_plan(
        self,
        user_input: str,
        intent_info: Dict[str, Any],
    session: "Session",
    ) -> RouterPlan:
        """Create standard single-command plan."""
        intent = intent_info.get('intent', 'unknown')
        
        if intent == 'unknown':
            return RouterPlan(
                intent=Intent.CLARIFY,
                confidence=0.3,
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                plan=[],
                notes={
                    'explanation': "I couldn't understand that request.",
                    'suggestions': [
                        "Try using a / command like /tone 440 0.5",
                        "Or be more specific about what you want to create",
                    ],
                },
            )
        
        # Map intents to commands
        if intent == 'synth':
            matched_type = intent_info.get('matched_type', 'tone')
            return self._plan_synth(matched_type, user_input)
        
        elif intent == 'fx':
            matched_type = intent_info.get('matched_type', 'reverb')
            return self._plan_fx(matched_type)
        
        elif intent == 'analyze':
            return RouterPlan(
                intent=Intent.EXECUTE,
                confidence=0.9,
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                plan=[CommandStep(
                    step_id=1,
                    command='analyze',
                    arguments=['detailed'],
                    rationale="User wants to analyze the current audio",
                )],
            )
        
        elif intent == 'play':
            return RouterPlan(
                intent=Intent.EXECUTE,
                confidence=0.95,
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                plan=[CommandStep(
                    step_id=1,
                    command='p',
                    arguments=[],
                    rationale="User wants to hear the audio",
                )],
            )
        
        elif intent == 'render':
            return RouterPlan(
                intent=Intent.EXECUTE,
                confidence=0.8,
                risk_level=RiskLevel.MEDIUM,
                requires_confirmation=True,
                plan=[CommandStep(
                    step_id=1,
                    command='bo',
                    arguments=[],
                    rationale="User wants to export/render audio",
                )],
            )
        
        elif intent == 'breed':
            return RouterPlan(
                intent=Intent.EXECUTE,
                confidence=0.8,
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                plan=[CommandStep(
                    step_id=1,
                    command='breed',
                    arguments=['1', '2'],
                    rationale="User wants to breed samples",
                )],
                notes={'assumption': "Using buffers 1 and 2 as parents"},
            )
        
        elif intent == 'explain':
            return RouterPlan(
                intent=Intent.EXPLAIN,
                confidence=0.9,
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                plan=[CommandStep(
                    step_id=1,
                    command='h',
                    arguments=[],
                    rationale="User needs help",
                )],
            )
        
        # Default fallback
        return RouterPlan(
            intent=Intent.CLARIFY,
            confidence=0.4,
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            plan=[],
            notes={'explanation': f"Detected intent '{intent}' but not sure how to proceed."},
        )
    
    def _plan_synth(self, sound_type: str, user_input: str) -> RouterPlan:
        """Create synthesis plan based on sound type."""
        sound_type = sound_type.lower()
        
        # Extract any numeric parameters
        numbers = re.findall(r'\d+\.?\d*', user_input)
        
        if sound_type == 'kick':
            return RouterPlan(
                intent=Intent.EXECUTE,
                confidence=0.85,
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                plan=[
                    CommandStep(1, 'g', ['kick'], "Generate kick using built-in algorithm"),
                    CommandStep(2, 'a', [], "Append to buffer", [1]),
                ],
            )
        
        elif sound_type == 'snare':
            return RouterPlan(
                intent=Intent.EXECUTE,
                confidence=0.85,
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                plan=[
                    CommandStep(1, 'g', ['snare'], "Generate snare"),
                    CommandStep(2, 'a', [], "Append to buffer", [1]),
                ],
            )
        
        elif sound_type == 'bass':
            freq = numbers[0] if numbers else '55'
            return RouterPlan(
                intent=Intent.EXECUTE,
                confidence=0.8,
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                plan=[
                    CommandStep(1, 'bk', ['modern_aggressive'], "Select bass-focused bank"),
                    CommandStep(2, 'al', ['0'], "Load algorithm", [1]),
                    CommandStep(3, 'tone', [freq, '0.5', '0.8'], "Generate bass tone", [2]),
                    CommandStep(4, 'a', [], "Append to buffer", [3]),
                ],
            )
        
        elif sound_type in ('pad', 'pads'):
            return RouterPlan(
                intent=Intent.EXECUTE,
                confidence=0.8,
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                plan=[
                    CommandStep(1, 'vc', ['4'], "Set voice count for richness"),
                    CommandStep(2, 'dt', ['3'], "Add detuning"),
                    CommandStep(3, 'atk', ['0.3'], "Slow attack for pad"),
                    CommandStep(4, 'rel', ['0.5'], "Long release"),
                    CommandStep(5, 'tone', ['220', '2.0', '0.6'], "Generate pad", [1, 2, 3, 4]),
                    CommandStep(6, 'a', [], "Append to buffer", [5]),
                ],
            )
        
        elif sound_type == 'lead':
            return RouterPlan(
                intent=Intent.EXECUTE,
                confidence=0.8,
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                plan=[
                    CommandStep(1, 'wm', ['saw'], "Saw wave for lead"),
                    CommandStep(2, 'tone', ['440', '0.5', '0.7'], "Generate lead tone", [1]),
                    CommandStep(3, 'a', [], "Append to buffer", [2]),
                ],
            )
        
        else:
            # Generic tone
            freq = numbers[0] if numbers else '440'
            dur = numbers[1] if len(numbers) > 1 else '0.5'
            return RouterPlan(
                intent=Intent.EXECUTE,
                confidence=0.7,
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                plan=[
                    CommandStep(1, 'tone', [freq, dur, '0.8'], f"Generate {sound_type} tone"),
                    CommandStep(2, 'a', [], "Append to buffer", [1]),
                ],
            )
    
    def _plan_fx(self, fx_type: str) -> RouterPlan:
        """Create effect plan based on type."""
        fx_type = fx_type.lower()
        
        if fx_type == 'reverb':
            return RouterPlan(
                intent=Intent.EXECUTE,
                confidence=0.9,
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                plan=[
                    CommandStep(1, 'bfx', ['add', 'reverb'], "Add reverb to buffer FX chain"),
                ],
            )
        
        elif fx_type in ('distortion', 'distort'):
            return RouterPlan(
                intent=Intent.EXECUTE,
                confidence=0.9,
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                plan=[
                    CommandStep(1, 'vamp', ['medium'], "Add medium overdrive/distortion"),
                ],
            )
        
        elif fx_type == 'compression':
            return RouterPlan(
                intent=Intent.EXECUTE,
                confidence=0.9,
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                plan=[
                    CommandStep(1, 'fc', ['punch'], "Add punchy compression"),
                ],
            )
        
        elif fx_type == 'filter':
            return RouterPlan(
                intent=Intent.EXECUTE,
                confidence=0.8,
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                plan=[
                    CommandStep(1, 'ft', ['lp'], "Set lowpass filter"),
                    CommandStep(2, 'cut', ['1000'], "Set cutoff to 1000Hz", [1]),
                    CommandStep(3, 'res', ['30'], "Add some resonance", [2]),
                ],
            )
        
        return RouterPlan(
            intent=Intent.EXECUTE,
            confidence=0.7,
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            plan=[
                CommandStep(1, 'bfx', ['add', fx_type], f"Add {fx_type} effect"),
            ],
        )
    
    def _create_high_plan(
        self,
        user_input: str,
        intent_info: Dict[str, Any],
        session: "Session",
    ) -> RouterPlan:
        """Create multi-command /high mode plan.
        
        /high mode allows chaining multiple commands to build
        complex sounds from a single request.
        """
        # Always high risk in /high mode
        intent = intent_info.get('intent', 'unknown')
        
        # More ambitious plans for /high mode
        if 'kick' in user_input.lower() or 'drum' in user_input.lower():
            return self._high_plan_drum(user_input)
        
        elif 'bass' in user_input.lower():
            return self._high_plan_bass(user_input)
        
        elif 'pad' in user_input.lower() or 'ambient' in user_input.lower():
            return self._high_plan_pad(user_input)
        
        elif 'lead' in user_input.lower() or 'synth' in user_input.lower():
            return self._high_plan_lead(user_input)
        
        elif 'analyze' in user_input.lower() and 'breed' in user_input.lower():
            return self._high_plan_analyze_breed(user_input)
        
        # Fallback to standard plan
        standard = self._create_standard_plan(user_input, intent_info, session)
        standard.risk_level = RiskLevel.HIGH
        standard.requires_confirmation = True
        standard.notes['high_mode'] = True
        return standard
    
    def _high_plan_drum(self, user_input: str) -> RouterPlan:
        """High mode drum synthesis plan."""
        return RouterPlan(
            intent=Intent.EXECUTE,
            confidence=0.85,
            risk_level=RiskLevel.HIGH,
            requires_confirmation=True,
            plan=[
                CommandStep(1, 'bu', ['1'], "Select buffer 1 for kick"),
                CommandStep(2, 'bk', ['classic_fm'], "Select FM bank"),
                CommandStep(3, 'al', ['0'], "Load algorithm", [2]),
                CommandStep(4, 'g', ['kick'], "Generate kick", [3]),
                CommandStep(5, 'a', [], "Append kick to buffer", [4]),
                CommandStep(6, 'fc', ['punch'], "Add punch compression", [5]),
                CommandStep(7, 'analyze', [], "Analyze result", [6]),
            ],
            notes={
                'explanation': "Building a punchy kick drum with FM synthesis and compression",
                'assumptions': ["Using buffer 1", "Classic FM bank"],
            },
        )
    
    def _high_plan_bass(self, user_input: str) -> RouterPlan:
        """High mode bass synthesis plan."""
        aggressive = 'aggressive' in user_input.lower() or 'growl' in user_input.lower()
        
        steps = [
            CommandStep(1, 'bu', ['1'], "Select buffer 1"),
            CommandStep(2, 'bk', ['modern_aggressive' if aggressive else 'classic_fm'], 
                       "Select appropriate bank"),
            CommandStep(3, 'al', ['growl_basic' if aggressive else 'bass'], "Load bass algorithm", [2]),
            CommandStep(4, 'vc', ['2'], "Add voice layering"),
            CommandStep(5, 'dt', ['5'], "Add detuning for thickness"),
            CommandStep(6, 'tone', ['55', '0.5', '0.9'], "Generate bass", [3, 4, 5]),
            CommandStep(7, 'a', [], "Append to buffer", [6]),
        ]
        
        if aggressive:
            steps.append(CommandStep(8, 'vamp', ['heavy'], "Add heavy saturation", [7]))
            steps.append(CommandStep(9, 'fc', ['loud'], "Compress aggressively", [8]))
        else:
            steps.append(CommandStep(8, 'fc', ['glue'], "Add glue compression", [7]))
        
        steps.append(CommandStep(len(steps) + 1, 'analyze', [], "Analyze result"))
        
        return RouterPlan(
            intent=Intent.EXECUTE,
            confidence=0.85,
            risk_level=RiskLevel.HIGH,
            requires_confirmation=True,
            plan=steps,
            notes={
                'explanation': f"Building {'aggressive' if aggressive else 'smooth'} bass with layering and processing",
            },
        )
    
    def _high_plan_pad(self, user_input: str) -> RouterPlan:
        """High mode pad synthesis plan."""
        return RouterPlan(
            intent=Intent.EXECUTE,
            confidence=0.85,
            risk_level=RiskLevel.HIGH,
            requires_confirmation=True,
            plan=[
                CommandStep(1, 'bu', ['1'], "Select buffer"),
                CommandStep(2, 'vc', ['6'], "6 voices for lush pad"),
                CommandStep(3, 'dt', ['8'], "Heavy detuning"),
                CommandStep(4, 'atk', ['0.5'], "Slow attack"),
                CommandStep(5, 'dec', ['0.3'], "Medium decay"),
                CommandStep(6, 'sus', ['0.7'], "High sustain"),
                CommandStep(7, 'rel', ['1.0'], "Long release"),
                CommandStep(8, 'tone', ['220', '4.0', '0.5'], "Generate 4s pad", [2, 3, 4, 5, 6, 7]),
                CommandStep(9, 'a', [], "Append to buffer", [8]),
                CommandStep(10, 'bfx', ['add', 'reverb'], "Add reverb", [9]),
                CommandStep(11, 'fc', ['soft'], "Soft compression", [10]),
                CommandStep(12, 'analyze', [], "Analyze result", [11]),
            ],
            notes={
                'explanation': "Building a lush ambient pad with 6-voice detuning and reverb",
            },
        )
    
    def _high_plan_lead(self, user_input: str) -> RouterPlan:
        """High mode lead synthesis plan."""
        return RouterPlan(
            intent=Intent.EXECUTE,
            confidence=0.85,
            risk_level=RiskLevel.HIGH,
            requires_confirmation=True,
            plan=[
                CommandStep(1, 'bu', ['1'], "Select buffer"),
                CommandStep(2, 'wm', ['saw'], "Saw wave for lead"),
                CommandStep(3, 'vc', ['2'], "2 voices"),
                CommandStep(4, 'dt', ['5'], "Slight detune"),
                CommandStep(5, 'atk', ['0.01'], "Fast attack"),
                CommandStep(6, 'rel', ['0.2'], "Medium release"),
                CommandStep(7, 'tone', ['440', '0.5', '0.8'], "Generate lead", [2, 3, 4, 5, 6]),
                CommandStep(8, 'a', [], "Append", [7]),
                CommandStep(9, 'ft', ['lp'], "Lowpass filter", [8]),
                CommandStep(10, 'cut', ['3000'], "Set cutoff", [9]),
                CommandStep(11, 'res', ['40'], "Add resonance", [10]),
                CommandStep(12, 'analyze', [], "Analyze", [11]),
            ],
            notes={
                'explanation': "Building a filtered saw lead with detuning",
            },
        )
    
    def _high_plan_analyze_breed(self, user_input: str) -> RouterPlan:
        """High mode analyze and breed plan."""
        return RouterPlan(
            intent=Intent.EXECUTE,
            confidence=0.8,
            risk_level=RiskLevel.HIGH,
            requires_confirmation=True,
            plan=[
                CommandStep(1, 'analyze', ['buf', '1'], "Analyze buffer 1"),
                CommandStep(2, 'analyze', ['buf', '2'], "Analyze buffer 2", [1]),
                CommandStep(3, 'analyze', ['compare', '1', '2'], "Compare buffers", [2]),
                CommandStep(4, 'breed', ['1', '2', '4'], "Breed 4 children", [3]),
            ],
            notes={
                'explanation': "Analyzing both samples, comparing them, then breeding",
            },
        )
    
    def execute_plan(
        self,
        plan: RouterPlan,
        session: "Session",
        confirm_callback: Optional[callable] = None,
    ) -> List[Tuple[str, str]]:
        """Execute a router plan.
        
        Parameters
        ----------
        plan : RouterPlan
            Plan to execute
        session : Session
            Current session
        confirm_callback : callable, optional
            Function to call for confirmation (returns bool)
        
        Returns
        -------
        list
            List of (command, result) tuples
        """
        results = []
        
        # Check for confirmation
        if plan.requires_confirmation:
            if confirm_callback:
                if not confirm_callback(plan):
                    return [("CANCELLED", "Plan execution cancelled by user")]
            else:
                return [("NEEDS_CONFIRM", "Plan requires confirmation")]
        
        # Execute each step
        for step in plan.plan:
            cmd_func = self.command_table.get(step.command)
            
            if cmd_func is None:
                results.append((f"/{step.command}", f"ERROR: Unknown command '{step.command}'"))
                continue
            
            try:
                result = cmd_func(session, step.arguments)
                results.append((f"/{step.command} {' '.join(step.arguments)}", result))
            except Exception as e:
                results.append((f"/{step.command}", f"ERROR: {e}"))
                break  # Stop on error
        
        # Store in history
        self.history.append(plan)
        
        return results


# ============================================================================
# GLOBAL ROUTER
# ============================================================================

_router: Optional[AICommandRouter] = None


def get_router(command_table: Dict[str, callable]) -> AICommandRouter:
    """Get or create global router."""
    global _router
    if _router is None:
        _router = AICommandRouter(command_table)
    return _router


def format_plan(plan: RouterPlan) -> str:
    """Format a router plan for display."""
    lines = [
        "=== AI COMMAND PLAN ===",
        f"Intent: {plan.intent.value}",
        f"Confidence: {plan.confidence:.0%}",
        f"Risk: {plan.risk_level.value}",
        f"Requires Confirmation: {'Yes' if plan.requires_confirmation else 'No'}",
        "",
    ]
    
    if plan.plan:
        lines.append("STEPS:")
        for step in plan.plan:
            deps = f" [after {step.dependencies}]" if step.dependencies else ""
            lines.append(f"  {step.step_id}. /{step.command} {' '.join(step.arguments)}{deps}")
            lines.append(f"     → {step.rationale}")
    
    if plan.notes:
        lines.append("")
        lines.append("NOTES:")
        for key, value in plan.notes.items():
            if isinstance(value, list):
                lines.append(f"  {key}:")
                for item in value:
                    lines.append(f"    - {item}")
            else:
                lines.append(f"  {key}: {value}")
    
    return '\n'.join(lines)
