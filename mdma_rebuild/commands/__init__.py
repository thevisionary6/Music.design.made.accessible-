"""Command entry points for the MDMA rebuild.

Each module in this package defines functions named ``cmd_<n>``.
These are discovered by the launcher and bound to corresponding
command names invoked via the REPL.  Commands take a session object
and a list of strings (arguments) and return a string message or
``None``.

BUILD ID: commands_v14.4
"""

__all__ = [
    "general_cmds",
    "synth_cmds",
    "fx_cmds",
    "render_cmds",
    "pattern_cmds",
    "playback_cmds",
    "buffer_cmds",
    "param_cmds",
    "hq_cmds",
    "advanced_cmds",
    "adv_cmds",
    "ai_cmds",
    "audiorate_cmds",
    "dj_cmds",
    "dsl_cmds",
    "generator_cmds",
    "math_cmds",
    "pack_cmds",
    "perf_cmds",
    "stub_cmds",
    "sydef_cmds",
    "working_cmds",
    "gen_cmds",
    "midi_cmds",
]
