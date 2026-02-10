"""
Custom arithmetic and utility commands for MDMA rebuild.

This module defines simple math operators and helper commands that
augment the MDMA command set with basic calculation capabilities.
These commands parse numeric literals or user variables (via the
global user stack) and return results as strings.  They are kept
separate from the core MDMA modules to avoid polluting the signal
processing codebase.

Supported commands:

  /add a b        Add two values
  /sub a b        Subtract b from a
  /over a b       Divide a by b (fraction)
  /times a b      Multiply a and b
  /power a b      Raise a to the bth power
  /log x [base]   Logarithm of x (optional base, default e)
  /give name      Print a variable or literal as string
  /out text       Output/print text (joins all args)
  /sleep seconds  Pause execution for N seconds

Repeat block commands:
  /8              Start a repeat block that runs 8 times
  /16             Start a repeat block that runs 16 times
  /4              Start a repeat block that runs 4 times
  /<N>            Start a repeat block that runs N times
  /end            End repeat block and execute commands

Example:
  /8
  /tone 440 1
  /fx reverb
  /end
  -> Executes /tone 440 1 and /fx reverb eight times

Values may be numeric literals (integer or float) or user variable
names defined via the /= command.  If a variable name is provided,
the current value is retrieved from the global user stack.  If a
non‑numeric value is encountered for an arithmetic operation an
informative error is returned.
"""

from __future__ import annotations

import math
from typing import List, Any, Optional

from ..core.session import Session
from ..dsp.advanced_ops import get_user_stack


# ============================================================================
# MODULE-LEVEL STATE FOR REPEAT BLOCKS
# ============================================================================
# These are used to track repeat block state across commands.
# We use module-level state so that /N commands can communicate with /end.

_repeat_count: int = 0  # Number of times to repeat
_recording_repeat: bool = False  # Are we recording commands?
_repeat_commands: List[str] = []  # Commands recorded in the block


def _parse_operand(token: str) -> Any:
    """Resolve an operand token to a numeric value or user variable.

    If the token matches a user variable name, return its value.
    Otherwise attempt to convert to int or float.  If conversion
    fails the original string is returned.
    """
    stack = get_user_stack()
    # variable names are case-insensitive
    if stack.exists(token.lower()):
        return stack.get(token)
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return token


def _binary_op(args: List[str], op_name: str, func) -> str:
    """Generic wrapper for binary numeric operations.

    Parses two operands, applies the function, and returns the result
    as a string.  If arguments are missing or non‑numeric, an error
    string is returned.
    """
    if len(args) < 2:
        return f"ERROR: /{op_name} requires 2 arguments"
    a = _parse_operand(args[0])
    b = _parse_operand(args[1])
    # Ensure both are numeric
    if not isinstance(a, (int, float)):
        return f"ERROR: invalid operand '{args[0]}'"
    if not isinstance(b, (int, float)):
        return f"ERROR: invalid operand '{args[1]}'"
    try:
        result = func(a, b)
    except Exception as exc:
        return f"ERROR: {exc}"
    return str(result)


def cmd_add(session: Session, args: List[str]) -> str:
    """Add two numeric values.

    Usage: /add a b
    """
    return _binary_op(args, 'add', lambda a, b: a + b)


def cmd_sub(session: Session, args: List[str]) -> str:
    """Subtract second value from first.

    Usage: /sub a b
    """
    return _binary_op(args, 'sub', lambda a, b: a - b)


def cmd_over(session: Session, args: List[str]) -> str:
    """Divide first value by second (fraction).

    Usage: /over numerator denominator
    """
    # Division by zero handled naturally by Python
    return _binary_op(args, 'over', lambda a, b: a / b)


def cmd_times(session: Session, args: List[str]) -> str:
    """Multiply two numeric values.

    Usage: /times a b
    """
    return _binary_op(args, 'times', lambda a, b: a * b)


def cmd_power(session: Session, args: List[str]) -> str:
    """Raise first value to the power of second.

    Usage: /power base exponent
    """
    return _binary_op(args, 'power', lambda a, b: a ** b)


def cmd_log(session: Session, args: List[str]) -> str:
    """Compute logarithm.

    Usage: /log x [base]
    Returns the natural logarithm (base e) if no base is provided.
    """
    if not args:
        return "ERROR: /log requires at least 1 argument"
    x = _parse_operand(args[0])
    if not isinstance(x, (int, float)):
        return f"ERROR: invalid operand '{args[0]}'"
    base = math.e
    if len(args) > 1:
        base_arg = _parse_operand(args[1])
        if not isinstance(base_arg, (int, float)):
            return f"ERROR: invalid base '{args[1]}'"
        base = base_arg
    try:
        result = math.log(x, base)
    except Exception as exc:
        return f"ERROR: {exc}"
    return str(result)


def cmd_give(session: Session, args: List[str]) -> str:
    """Print a variable or literal as a string.

    Usage: /give name
    If the name refers to a user variable its value is returned;
    otherwise the literal value is returned.  This is similar to
    /GET but always returns the raw repr() of the value.
    """
    if not args:
        return "ERROR: /give requires a name or value"
    token = args[0]
    # Attempt to resolve variable
    stack = get_user_stack()
    if stack.exists(token.lower()):
        value = stack.get(token)
    else:
        # Try to parse numeric literal
        value = _parse_operand(token)
    return repr(value)


def cmd_out(session: Session, args: List[str]) -> str:
    """Output/print text to the console.

    Usage: /out <text...>
    Joins all arguments with spaces and outputs them.  Useful for
    printing messages, debugging, or creating output in DSL scripts.
    Variables can be resolved by using /give first and storing the
    result.

    Examples:
      /out Hello World        -> "Hello World"
      /out Processing step 1  -> "Processing step 1"
      /out                    -> ""
    """
    if not args:
        return ""
    # Join all arguments with spaces
    return " ".join(args)


# Uppercase alias for /out
cmd_OUT = cmd_out


# ============================================================================
# REPEAT BLOCK SYSTEM
# ============================================================================
# These commands implement /N (like /8, /16, /4) as repeat block starters.
# Use /N followed by commands, then /end to execute them N times.

def _start_repeat_block(count: int) -> str:
    """Start a repeat block that will run 'count' times.
    
    Internal helper used by /8, /16, /4, and numeric commands.
    """
    global _repeat_count, _recording_repeat, _repeat_commands
    
    if _recording_repeat:
        return f"ERROR: already recording a repeat block ({_repeat_count}x). Use /end first."
    
    if count < 1:
        return "ERROR: repeat count must be at least 1"
    if count > 1000:
        return "ERROR: repeat count cannot exceed 1000 (safety limit)"
    
    _repeat_count = count
    _recording_repeat = True
    _repeat_commands = []
    
    return f"REPEAT: started block (will repeat {count} times). Commands will be recorded until /end"


def _record_command(cmd_line: str) -> Optional[str]:
    """Record a command into the current repeat block.
    
    Returns None if recording, or a result string if command should be
    executed immediately (like /end).
    """
    global _recording_repeat, _repeat_commands
    
    if not _recording_repeat:
        return None  # Not recording, let normal dispatch handle it
    
    # Check if this is /end
    parts = cmd_line[1:].split() if cmd_line.startswith('/') else []
    cmd = parts[0].lower() if parts else ''
    
    if cmd == 'end':
        return None  # Let /end be handled normally
    
    # Reject nested repeat blocks - these commands should not be recorded
    # as they would start a new block inside the current one
    nested_block_cmds = {'2', '4', '8', '16', '32', '64', 'repeat'}
    if cmd in nested_block_cmds:
        return f"ERROR: cannot start nested repeat block (/{cmd}) while recording. Use /end first."
    
    # Record the command
    _repeat_commands.append(cmd_line)
    return f"  recorded: {cmd_line}"


def _execute_repeat_block(session: Session) -> str:
    """Execute the recorded repeat block.
    
    Called by cmd_end when ending a repeat block.
    Returns the combined output of all executions.
    """
    global _repeat_count, _recording_repeat, _repeat_commands
    
    if not _recording_repeat:
        return "ERROR: no repeat block to end"
    
    count = _repeat_count
    commands = _repeat_commands.copy()
    
    # Reset state
    _repeat_count = 0
    _recording_repeat = False
    _repeat_commands = []
    
    if not commands:
        return f"REPEAT END: no commands to execute (block was empty)"
    
    # Get the command executor from session
    executor = getattr(session, 'command_executor', None)
    if executor is None:
        # Fallback: just report what would have been executed
        return f"REPEAT END: would execute {len(commands)} commands {count} times (no executor available)"
    
    # Execute commands count times
    outputs = []
    total_executed = 0
    
    for iteration in range(count):
        for cmd_line in commands:
            try:
                result = executor(cmd_line)
                if result and result != "EXIT":
                    outputs.append(f"  [{iteration+1}] {result}")
                total_executed += 1
            except Exception as exc:
                outputs.append(f"  [{iteration+1}] ERROR: {exc}")
    
    summary = f"REPEAT END: executed {len(commands)} commands {count} times ({total_executed} total)"
    if outputs:
        return summary + "\n" + "\n".join(outputs[-10:])  # Show last 10 outputs
    return summary


def is_recording_repeat() -> bool:
    """Check if we're currently recording a repeat block.
    
    Used by the main REPL to know if commands should be recorded.
    """
    return _recording_repeat


def record_repeat_command(cmd_line: str) -> Optional[str]:
    """Public interface to record a command into repeat block.
    
    Returns the result message if command was recorded, None otherwise.
    """
    return _record_command(cmd_line)


def cmd_8(session: Session, args: List[str]) -> str:
    """Start a repeat block that runs 8 times.

    Usage: 
      /8
      <commands>
      /end

    Example:
      /8
      /tone 440 1
      /sleep 0.1
      /end
      -> Generates 8 tones with 0.1s delay between each
    """
    return _start_repeat_block(8)


def cmd_16(session: Session, args: List[str]) -> str:
    """Start a repeat block that runs 16 times.

    Usage: 
      /16
      <commands>
      /end
    """
    return _start_repeat_block(16)


def cmd_4(session: Session, args: List[str]) -> str:
    """Start a repeat block that runs 4 times.

    Usage: 
      /4
      <commands>
      /end
    """
    return _start_repeat_block(4)


def cmd_2(session: Session, args: List[str]) -> str:
    """Start a repeat block that runs 2 times."""
    return _start_repeat_block(2)


def cmd_32(session: Session, args: List[str]) -> str:
    """Start a repeat block that runs 32 times."""
    return _start_repeat_block(32)


def cmd_64(session: Session, args: List[str]) -> str:
    """Start a repeat block that runs 64 times."""
    return _start_repeat_block(64)


def cmd_repeat(session: Session, args: List[str]) -> str:
    """Start a repeat block with custom count.

    Usage: /repeat <count>

    Example:
      /repeat 12
      /tone 440 1
      /end
      -> Generates 12 tones
    """
    if not args:
        return "ERROR: /repeat requires a count. Usage: /repeat <count>"
    
    try:
        count = int(args[0])
    except ValueError:
        return f"ERROR: invalid repeat count '{args[0]}'"
    
    return _start_repeat_block(count)


def cmd_repeat_end(session: Session, args: List[str]) -> str:
    """End a repeat block and execute the recorded commands.

    Usage: /end

    This command ends the current repeat block (started with /8, /16, 
    /4, /repeat, etc.) and executes all recorded commands the specified
    number of times.
    """
    return _execute_repeat_block(session)


# Capitalised alias for math commands: allow uppercase names
cmd_ADD = cmd_add  # alias
cmd_SUB = cmd_sub
cmd_OVER = cmd_over
cmd_TIMES = cmd_times
cmd_POWER = cmd_power
cmd_LOG = cmd_log
cmd_GIVE = cmd_give
cmd_EIGHT = cmd_8
cmd_SIXTEEN = cmd_16
cmd_FOUR = cmd_4
cmd_REPEAT = cmd_repeat


def cmd_sleep(session: Session, args: List[str]) -> str:
    """Pause execution for a specified number of seconds.

    Usage: /sleep <seconds>
    
    Pauses the command processing for the given duration.  Useful for
    scheduling delays in playback or sequencing.  Accepts integer or
    floating point values.  If no duration is provided or the value
    is invalid an error is returned.
    
    The sleep can be interrupted by keyboard interrupt (Ctrl+C).
    
    Examples:
      /sleep 1       -> Sleep for 1 second
      /sleep 0.5     -> Sleep for 500 milliseconds
      /sleep 0.01    -> Sleep for 10 milliseconds
    """
    import time

    if not args:
        return "ERROR: /sleep requires a duration in seconds"
    # Resolve argument to numeric (supports variables)
    duration = _parse_operand(args[0])
    if not isinstance(duration, (int, float)):
        return f"ERROR: invalid duration '{args[0]}'"
    # Enforce non-negative values
    if duration < 0:
        return "ERROR: duration must be non-negative"
    # Safety limit
    if duration > 3600:
        return "ERROR: sleep duration cannot exceed 3600 seconds (1 hour)"
    try:
        time.sleep(float(duration))
    except KeyboardInterrupt:
        return "INTERRUPTED: sleep was cancelled"
    except Exception as exc:
        return f"ERROR: {exc}"
    return f"OK: slept {duration} seconds"


# Uppercase alias for /sleep
cmd_SLEEP = cmd_sleep


# ============================================================================
# COMMAND REGISTRATION HELPER
# ============================================================================

def get_math_commands() -> dict:
    """Return all math commands for registration.
    
    This function is called by the command table builder to register
    all commands defined in this module.
    """
    return {
        # Arithmetic
        'add': cmd_add,
        'sub': cmd_sub,
        'over': cmd_over,
        'times': cmd_times,
        'power': cmd_power,
        'log': cmd_log,
        
        # Output
        'give': cmd_give,
        'out': cmd_out,
        
        # Sleep
        'sleep': cmd_sleep,
        
        # Repeat blocks
        '2': cmd_2,
        '4': cmd_4,
        '8': cmd_8,
        '16': cmd_16,
        '32': cmd_32,
        '64': cmd_64,
        'repeat': cmd_repeat,
    }