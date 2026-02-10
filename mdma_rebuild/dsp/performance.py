"""MDMA Performance Mode - Macro-driven performance layer.

Performance Mode sits explicitly on top of DJ Mode, providing:
- Macro scripting for multi-step actions
- Snapshots for state recall
- Controlled randomness
- Evolving fallback (autopilot)
- Deterministic execution

HARD REQUIREMENT: DJ Mode must be active before Performance Mode can be enabled.
"""

from __future__ import annotations

import time
import json
import numpy as np
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import random

if TYPE_CHECKING:
    from ..core.session import Session


# ============================================================================
# SNAPSHOT SYSTEM
# ============================================================================

@dataclass
class Snapshot:
    """Captures full recallable performance state."""
    name: str
    timestamp: float = field(default_factory=time.time)
    
    # Mode flags
    dj_enabled: bool = True
    perf_enabled: bool = True
    
    # Deck states
    deck_states: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    # Active deck
    active_deck: int = 1
    crossfader: float = 0.5
    
    # Effect states
    effect_states: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Routing
    master_device: Optional[str] = None
    headphone_device: Optional[str] = None
    cue_volume: float = 1.0
    master_volume: float = 1.0
    
    # Labels
    label: str = ""  # e.g., "verse", "drop", "break"
    tags: List[str] = field(default_factory=list)


# ============================================================================
# MACRO SYSTEM
# ============================================================================

class MacroStepType(Enum):
    COMMAND = "command"
    SNAPSHOT = "snapshot"
    TRANSITION = "transition"
    WAIT = "wait"
    LOOP = "loop"
    MUTATE = "mutate"
    RANDOM = "random"


@dataclass
class MacroStep:
    """Single step in a macro."""
    step_type: MacroStepType
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    delay_beats: float = 0.0
    condition: Optional[str] = None


@dataclass
class Macro:
    """Named performance script."""
    name: str
    steps: List[MacroStep] = field(default_factory=list)
    
    # Properties
    home_snapshot: Optional[str] = None
    rng_seed: Optional[int] = None
    rng_locked: bool = False
    rng_amount: float = 0.0
    rng_domain: List[str] = field(default_factory=list)
    
    # Mutation scheduling
    mutate_every_bars: int = 0
    resolve_every_bars: int = 0
    mutation_recipe: List[str] = field(default_factory=list)
    
    # State
    armed: bool = False
    running: bool = False
    current_step: int = 0
    started_at: float = 0.0
    
    # Tags
    tags: List[str] = field(default_factory=list)
    description: str = ""


# ============================================================================
# RANDOMNESS CONTROLLER
# ============================================================================

class RandomnessController:
    """Manages controlled randomness."""
    
    def __init__(self):
        self.global_seed: int = 42
        self.locked: bool = False
        self.amount: float = 0.0
        self.domain: List[str] = ['filter', 'gate', 'volume', 'pan']
        self._rng = random.Random(self.global_seed)
        self.history: List[Dict] = []
    
    def set_seed(self, seed: int):
        self.global_seed = seed
        if self.locked:
            self._rng.seed(seed)
    
    def lock(self):
        self.locked = True
        self._rng.seed(self.global_seed)
    
    def unlock(self):
        self.locked = False
    
    def set_amount(self, amount: float):
        self.amount = max(0.0, min(1.0, amount))
    
    def set_domain(self, domain: List[str]):
        self.domain = domain
    
    def random_value(self, base: float, param: str, range_pct: float = 0.2) -> float:
        if param not in self.domain or self.amount == 0:
            return base
        
        deviation = self._rng.uniform(-range_pct, range_pct) * self.amount
        result = base * (1 + deviation)
        
        self.history.append({
            'time': time.time(),
            'param': param,
            'base': base,
            'result': result,
        })
        
        return result


# ============================================================================
# PERFORMANCE MODE ENGINE
# ============================================================================

class PerformanceModeEngine:
    """Core Performance Mode engine."""
    
    def __init__(self, dj_engine):
        self.dj = dj_engine
        self.enabled = False
        
        # Snapshots
        self.snapshots: Dict[str, Snapshot] = {}
        self.home_snapshot: Optional[str] = None
        
        # Macros
        self.macros: Dict[str, Macro] = {}
        self.armed_macros: List[str] = []
        self.running_macros: List[str] = []
        
        # Randomness
        self.rng = RandomnessController()
        
        # Fallback/evolution
        self.evolution_enabled: bool = True
        self.evolution_frozen: bool = False
        self.last_user_input: float = time.time()
        self.evolution_timeout: float = 16.0
        
        # Tempo tracking
        self.tempo: float = 120.0
        
        # Command registry
        self._command_handler: Optional[Callable] = None
        
        # Action log
        self.action_log: List[Dict] = []
    
    def set_command_handler(self, handler: Callable):
        self._command_handler = handler
    
    def enable(self) -> str:
        if not self.dj.enabled:
            return "ERROR: DJ Mode must be enabled first. Use /djm on"
        
        if self.enabled:
            return "Performance Mode already enabled"
        
        self.enabled = True
        self.last_user_input = time.time()
        
        return (f"Performance Mode ENABLED\n"
                f"  DJ Mode: active\n"
                f"  Macros: {len(self.macros)}\n"
                f"  Snapshots: {len(self.snapshots)}\n"
                f"  RNG: {'locked' if self.rng.locked else 'unlocked'} (amount: {self.rng.amount:.0%})")
    
    def disable(self) -> str:
        if not self.enabled:
            return "Performance Mode not enabled"
        
        self._stop_all_macros()
        self.enabled = False
        return "Performance Mode DISABLED"
    
    def check_dj_dependency(self) -> Optional[str]:
        if self.enabled and not self.dj.enabled:
            self.disable()
            return "WARNING: Performance Mode auto-disabled (DJ Mode was disabled)"
        return None
    
    # ========================================================================
    # SNAPSHOT MANAGEMENT
    # ========================================================================
    
    def capture_snapshot(self, name: str, label: str = "") -> str:
        if not self.enabled:
            return "ERROR: Performance Mode not enabled"
        
        snap = Snapshot(
            name=name,
            dj_enabled=self.dj.enabled,
            perf_enabled=self.enabled,
            active_deck=self.dj.active_deck,
            crossfader=self.dj.crossfader,
            master_device=self.dj.master_device,
            headphone_device=self.dj.headphone_device,
            cue_volume=self.dj.cue_volume,
            master_volume=self.dj.master_volume,
            label=label,
        )
        
        for deck_id, deck in self.dj.decks.items():
            snap.deck_states[deck_id] = {
                'has_audio': deck.buffer is not None,
                'position': deck.position,
                'tempo': deck.tempo,
                'pitch': deck.pitch,
                'volume': deck.volume,
                'eq_low': deck.eq_low,
                'eq_mid': deck.eq_mid,
                'eq_high': deck.eq_high,
                'playing': deck.playing,
                'cue_enabled': deck.cue_enabled,
                'stem_levels': dict(deck.active_stems) if deck.stems else {},
                'loop_active': deck.loop_active,
                'loop_start': deck.loop_start,
                'loop_end': deck.loop_end,
            }
        
        self.snapshots[name] = snap
        self._log_action('snapshot_capture', {'name': name, 'label': label})
        
        return f"OK: captured snapshot '{name}'" + (f" [{label}]" if label else "")
    
    def recall_snapshot(self, name: str, soft_land: bool = False) -> str:
        if name not in self.snapshots:
            return f"ERROR: snapshot '{name}' not found"
        
        snap = self.snapshots[name]
        
        for deck_id, state in snap.deck_states.items():
            if deck_id not in self.dj.decks:
                continue
            
            deck = self.dj.decks[deck_id]
            deck.position = state['position']
            deck.tempo = state['tempo']
            deck.pitch = state['pitch']
            deck.volume = state['volume']
            deck.eq_low = state['eq_low']
            deck.eq_mid = state['eq_mid']
            deck.eq_high = state['eq_high']
            deck.cue_enabled = state['cue_enabled']
            deck.loop_active = state['loop_active']
            deck.loop_start = state['loop_start']
            deck.loop_end = state['loop_end']
            
            if state['stem_levels'] and deck.stems:
                deck.active_stems = dict(state['stem_levels'])
        
        self.dj.crossfader = snap.crossfader
        self.dj.cue_volume = snap.cue_volume
        self.dj.master_volume = snap.master_volume
        
        self._log_action('snapshot_recall', {'name': name})
        return f"OK: recalled snapshot '{name}'" + (f" [{snap.label}]" if snap.label else "")
    
    def set_home_snapshot(self, name: str) -> str:
        if name not in self.snapshots:
            return f"ERROR: snapshot '{name}' not found"
        self.home_snapshot = name
        return f"OK: home snapshot set to '{name}'"
    
    def go_home(self) -> str:
        if not self.home_snapshot:
            return "ERROR: no home snapshot set. Use /snap home <n>"
        return self.recall_snapshot(self.home_snapshot, soft_land=True)
    
    def list_snapshots(self) -> str:
        if not self.snapshots:
            return "No snapshots captured. Use /snap cap <n>"
        
        lines = ["=== SNAPSHOTS ===", ""]
        for name, snap in self.snapshots.items():
            home_marker = " [HOME]" if name == self.home_snapshot else ""
            label = f" ({snap.label})" if snap.label else ""
            lines.append(f"  {name}{label}{home_marker}")
        return '\n'.join(lines)
    
    # ========================================================================
    # MACRO MANAGEMENT
    # ========================================================================
    
    def create_macro(self, name: str, description: str = "") -> str:
        if name in self.macros:
            return f"ERROR: macro '{name}' already exists"
        self.macros[name] = Macro(name=name, description=description)
        return f"OK: created macro '{name}'"
    
    def add_macro_step(self, name: str, step_type: str, action: str, **params) -> str:
        if name not in self.macros:
            return f"ERROR: macro '{name}' not found"
        
        try:
            stype = MacroStepType(step_type)
        except ValueError:
            valid = [t.value for t in MacroStepType]
            return f"ERROR: invalid step type. Valid: {valid}"
        
        step = MacroStep(
            step_type=stype,
            action=action,
            params=params,
            delay_beats=params.get('delay', 0.0),
        )
        
        self.macros[name].steps.append(step)
        return f"OK: added {step_type} step to '{name}'"
    
    def set_macro_property(self, name: str, prop: str, value: Any) -> str:
        if name not in self.macros:
            return f"ERROR: macro '{name}' not found"
        
        macro = self.macros[name]
        
        if prop == 'home':
            macro.home_snapshot = value
        elif prop == 'seed':
            macro.rng_seed = int(value)
        elif prop == 'rng_locked':
            macro.rng_locked = str(value).lower() in ('true', '1', 'on', 'yes')
        elif prop == 'rng_amount':
            macro.rng_amount = float(value)
        elif prop == 'rng_domain':
            macro.rng_domain = value.split(',') if isinstance(value, str) else value
        elif prop == 'mutate_bars':
            macro.mutate_every_bars = int(value)
        elif prop == 'resolve_bars':
            macro.resolve_every_bars = int(value)
        elif prop == 'description':
            macro.description = value
        else:
            return f"ERROR: unknown property '{prop}'"
        
        return f"OK: set {prop} = {value}"
    
    def arm_macro(self, name: str) -> str:
        if name not in self.macros:
            return f"ERROR: macro '{name}' not found"
        
        macro = self.macros[name]
        macro.armed = True
        
        if name not in self.armed_macros:
            self.armed_macros.append(name)
        
        return f"OK: armed macro '{name}'"
    
    def run_macro(self, name: str) -> str:
        if name not in self.macros:
            return f"ERROR: macro '{name}' not found"
        
        if not self.enabled:
            return "ERROR: Performance Mode not enabled"
        
        macro = self.macros[name]
        macro.running = True
        macro.current_step = 0
        macro.started_at = time.time()
        
        if name not in self.running_macros:
            self.running_macros.append(name)
        
        if macro.rng_seed is not None:
            self.rng.set_seed(macro.rng_seed)
        if macro.rng_locked:
            self.rng.lock()
        if macro.rng_amount > 0:
            self.rng.set_amount(macro.rng_amount)
        if macro.rng_domain:
            self.rng.set_domain(macro.rng_domain)
        
        self._log_action('macro_run', {'name': name})
        return f"OK: running macro '{name}' ({len(macro.steps)} steps)"
    
    def stop_macro(self, name: str) -> str:
        if name not in self.macros:
            return f"ERROR: macro '{name}' not found"
        
        macro = self.macros[name]
        macro.running = False
        macro.armed = False
        
        if name in self.running_macros:
            self.running_macros.remove(name)
        if name in self.armed_macros:
            self.armed_macros.remove(name)
        
        return f"OK: stopped macro '{name}'"
    
    def _stop_all_macros(self):
        for name in list(self.running_macros):
            self.stop_macro(name)
    
    def list_macros(self) -> str:
        if not self.macros:
            return "No macros defined. Use /mc new <n>"
        
        lines = ["=== MACROS ===", ""]
        for name, macro in self.macros.items():
            status = ""
            if macro.running:
                status = " [RUNNING]"
            elif macro.armed:
                status = " [ARMED]"
            
            steps = len(macro.steps)
            desc = f" - {macro.description}" if macro.description else ""
            lines.append(f"  {name}{status} ({steps} steps){desc}")
        return '\n'.join(lines)
    
    def macro_info(self, name: str) -> str:
        if name not in self.macros:
            return f"ERROR: macro '{name}' not found"
        
        m = self.macros[name]
        lines = [f"=== MACRO: {name} ===", ""]
        
        if m.description:
            lines.append(f"Description: {m.description}")
        
        lines.append(f"Steps: {len(m.steps)}")
        lines.append(f"Home Snapshot: {m.home_snapshot or 'none'}")
        lines.append(f"RNG: seed={m.rng_seed or 'global'}, locked={m.rng_locked}, amount={m.rng_amount:.0%}")
        
        if m.mutate_every_bars:
            lines.append(f"Mutate every: {m.mutate_every_bars} bars")
        
        lines.append("")
        lines.append("Steps:")
        for i, step in enumerate(m.steps):
            marker = "→" if i == m.current_step and m.running else " "
            lines.append(f"  {marker}[{i}] {step.step_type.value}: {step.action}")
        
        return '\n'.join(lines)
    
    # ========================================================================
    # PANIC & SAFETY
    # ========================================================================
    
    def panic(self) -> str:
        self._stop_all_macros()
        self.evolution_frozen = True
        
        result = "⚠️ PANIC: All macros stopped, evolution frozen"
        
        if self.home_snapshot:
            home_result = self.go_home()
            result += f"\n{home_result}"
        
        self._log_action('panic', {})
        return result
    
    def freeze_evolution(self) -> str:
        self.evolution_frozen = True
        return "OK: Evolution/mutation FROZEN"
    
    def unfreeze_evolution(self) -> str:
        self.evolution_frozen = False
        return "OK: Evolution/mutation UNFROZEN"
    
    def user_input(self):
        self.last_user_input = time.time()
    
    def _log_action(self, action_type: str, data: Dict):
        self.action_log.append({
            'time': time.time(),
            'type': action_type,
            'data': data,
        })
        if len(self.action_log) > 1000:
            self.action_log = self.action_log[-500:]
    
    # ========================================================================
    # MACRO STEP EXECUTION WITH TIMING
    # ========================================================================
    
    def tick(self, current_beat: float, current_bar: int) -> List[str]:
        """Process macro timing - call this from playback loop.
        
        Parameters
        ----------
        current_beat : float
            Current beat position (0-based, fractional)
        current_bar : int
            Current bar number (0-based)
        
        Returns
        -------
        List[str]
            List of commands to execute this tick
        """
        commands_to_run = []
        
        for macro_name in list(self.running_macros):
            if macro_name not in self.macros:
                continue
            
            macro = self.macros[macro_name]
            if not macro.running:
                continue
            
            # Check if we need to advance to next step
            while macro.current_step < len(macro.steps):
                step = macro.steps[macro.current_step]
                
                # Check timing
                should_execute = self._check_step_timing(
                    step, macro, current_beat, current_bar
                )
                
                if should_execute:
                    cmd = self._execute_step(step, macro)
                    if cmd:
                        commands_to_run.append(cmd)
                    macro.current_step += 1
                else:
                    break  # Wait for timing
            
            # Check if macro is complete
            if macro.current_step >= len(macro.steps):
                macro.running = False
                if macro_name in self.running_macros:
                    self.running_macros.remove(macro_name)
        
        return commands_to_run
    
    def _check_step_timing(self, step: MacroStep, macro: Macro,
                           current_beat: float, current_bar: int) -> bool:
        """Check if a step should execute based on timing."""
        # If no delay, execute immediately
        if step.delay_beats <= 0:
            return True
        
        # Calculate elapsed time since macro started
        elapsed = time.time() - macro.started_at
        beat_duration = 60.0 / self.tempo
        elapsed_beats = elapsed / beat_duration
        
        # Check if we've waited enough beats
        # Sum delays of all previous steps
        total_delay = sum(
            s.delay_beats for s in macro.steps[:macro.current_step]
        ) + step.delay_beats
        
        return elapsed_beats >= total_delay
    
    def _execute_step(self, step: MacroStep, macro: Macro) -> Optional[str]:
        """Execute a single macro step.
        
        Returns command string to execute, or None.
        """
        if step.step_type == MacroStepType.COMMAND:
            return step.action
        
        elif step.step_type == MacroStepType.SNAPSHOT:
            self.recall_snapshot(step.action)
            return None
        
        elif step.step_type == MacroStepType.TRANSITION:
            # Transition to snapshot with crossfade
            params = step.params
            duration = params.get('duration', 2.0)
            curve = params.get('curve', 'linear')
            return f"/tran {step.action} {duration} {curve}"
        
        elif step.step_type == MacroStepType.WAIT:
            # Wait is handled by delay_beats
            return None
        
        elif step.step_type == MacroStepType.MUTATE:
            # Apply mutation
            if macro.rng_amount > 0:
                # Return mutation command
                return f"/mutate {step.action} {macro.rng_amount}"
            return None
        
        elif step.step_type == MacroStepType.RANDOM:
            # Pick random action from list
            options = step.action.split('|')
            choice = self.rng._rng.choice(options)
            return choice.strip()
        
        elif step.step_type == MacroStepType.LOOP:
            # Loop back to start
            macro.current_step = -1  # Will be incremented to 0
            return None
        
        return None
    
    def execute_macro_immediate(self, name: str, 
                                command_handler: Callable[[str], str]) -> str:
        """Execute all macro steps immediately (no timing).
        
        Parameters
        ----------
        name : str
            Macro name
        command_handler : callable
            Function to execute commands
        
        Returns
        -------
        str
            Execution results
        """
        if name not in self.macros:
            return f"ERROR: macro '{name}' not found"
        
        macro = self.macros[name]
        results = []
        
        for step in macro.steps:
            cmd = self._execute_step(step, macro)
            if cmd:
                try:
                    result = command_handler(cmd)
                    results.append(f"  {cmd} -> {result[:50] if result else 'OK'}")
                except Exception as e:
                    results.append(f"  {cmd} -> ERROR: {e}")
        
        return f"Executed macro '{name}':\n" + "\n".join(results)


# ============================================================================
# MACRO SCHEDULER (for adv_cmds integration)
# ============================================================================

class MacroScheduler:
    """Handles timed execution of macro commands.
    
    Supports timing tokens:
    - @now - immediate execution
    - @beat - execute on next beat boundary
    - @bar - execute on next bar boundary
    - @delay:N - delay N beats
    - @time:HH:MM:SS - execute at wall clock time
    """
    
    def __init__(self, bpm: float = 120.0, sr: int = 48000):
        self.bpm = bpm
        self.sr = sr
        self.pending: List[Dict] = []
        self.start_time: float = time.time()
        self.current_beat: float = 0.0
        self.current_bar: int = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._command_handler: Optional[Callable] = None
    
    def set_command_handler(self, handler: Callable[[str], str]):
        """Set the function that executes commands."""
        self._command_handler = handler
    
    def schedule(self, command: str, timing: str = "@now") -> str:
        """Schedule a command for execution.
        
        Parameters
        ----------
        command : str
            Command to execute
        timing : str
            Timing token (@now, @beat, @bar, @delay:N)
        
        Returns
        -------
        str
            Confirmation message
        """
        event = {
            'command': command,
            'timing': timing,
            'scheduled_at': time.time(),
            'scheduled_beat': self.current_beat,
            'scheduled_bar': self.current_bar,
            'executed': False,
        }
        
        if timing == "@now":
            # Execute immediately
            if self._command_handler:
                result = self._command_handler(command)
                return f"Executed: {command} -> {result[:50] if result else 'OK'}"
            return f"Scheduled (no handler): {command}"
        
        # Parse timing
        if timing == "@beat":
            event['target_beat'] = int(self.current_beat) + 1
            event['target_type'] = 'beat'
        elif timing == "@bar":
            event['target_bar'] = self.current_bar + 1
            event['target_type'] = 'bar'
        elif timing.startswith("@delay:"):
            try:
                delay = float(timing.split(":")[1])
                event['target_beat'] = self.current_beat + delay
                event['target_type'] = 'beat'
            except:
                event['target_beat'] = self.current_beat + 1
                event['target_type'] = 'beat'
        elif timing.startswith("@time:"):
            # Wall clock time (future feature)
            event['target_type'] = 'time'
            event['target_time'] = timing.split(":", 1)[1]
        else:
            # Unknown timing, execute on next beat
            event['target_beat'] = int(self.current_beat) + 1
            event['target_type'] = 'beat'
        
        self.pending.append(event)
        return f"Scheduled at {timing}: {command}"
    
    def tick(self, current_beat: float, current_bar: int) -> List[str]:
        """Process pending events.
        
        Call this from the playback/update loop.
        
        Returns list of results from executed commands.
        """
        self.current_beat = current_beat
        self.current_bar = current_bar
        
        results = []
        still_pending = []
        
        for event in self.pending:
            should_execute = False
            
            target_type = event.get('target_type', 'beat')
            
            if target_type == 'beat':
                if current_beat >= event.get('target_beat', 0):
                    should_execute = True
            elif target_type == 'bar':
                if current_bar >= event.get('target_bar', 0):
                    should_execute = True
            
            if should_execute and not event['executed']:
                event['executed'] = True
                if self._command_handler:
                    try:
                        result = self._command_handler(event['command'])
                        results.append(result)
                    except Exception as e:
                        results.append(f"ERROR: {e}")
            elif not event['executed']:
                still_pending.append(event)
        
        self.pending = still_pending
        return results
    
    def clear(self):
        """Clear all pending events."""
        self.pending = []
    
    def list_pending(self) -> str:
        """List all pending scheduled events."""
        if not self.pending:
            return "No pending scheduled events"
        
        lines = ["=== PENDING EVENTS ==="]
        for i, event in enumerate(self.pending):
            timing = event['timing']
            cmd = event['command'][:40]
            lines.append(f"  {i}: {timing} -> {cmd}")
        return "\n".join(lines)


# Global scheduler instance
_macro_scheduler: Optional[MacroScheduler] = None

def get_macro_scheduler(bpm: float = 120.0) -> MacroScheduler:
    """Get the global macro scheduler."""
    global _macro_scheduler
    if _macro_scheduler is None:
        _macro_scheduler = MacroScheduler(bpm)
    return _macro_scheduler


# ============================================================================
# GLOBAL ENGINE
# ============================================================================

_perf_engine: Optional[PerformanceModeEngine] = None


def get_perf_engine(dj_engine=None) -> Optional[PerformanceModeEngine]:
    global _perf_engine
    if _perf_engine is None and dj_engine is not None:
        _perf_engine = PerformanceModeEngine(dj_engine)
    return _perf_engine


def is_perf_mode_enabled() -> bool:
    return _perf_engine is not None and _perf_engine.enabled
