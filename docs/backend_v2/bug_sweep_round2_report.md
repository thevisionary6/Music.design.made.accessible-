# Bug sweep round 2 — post-effects-expansion

Second-pass audit of the main codebase, run after the first sweep
already fixed the bulk of silent-swallow / buffer-priority / dim-check
issues.

## Findings

Round 2 surfaced **no new actionable bugs**. The agent's candidate
list (unprotected `int()` / `int(float())` calls, unchecked `args[i]`
accesses) was investigated line-by-line; every flagged site turned
out to be either:

- Already wrapped in a matching ``try ... except ValueError`` block
  (e.g. `commands/dj_cmds.py:158-161` for the DJ blocksize argument),
  or
- Gated by an earlier ``if len(args) < N: return ...`` guard
  (e.g. `commands/synth_cmds.py:563` before the `args[0..2]` reads,
  `commands/phase_t_cmds.py:375` before `int(args[1])`).

This is a good result — the argument-validation layer across the
five biggest command modules (``fx_cmds``, ``synth_cmds``, ``dj_cmds``,
``dsl_cmds``, ``advanced_cmds``) is healthier than the agent's
heuristic hits suggested.

## Deferred items (carried over from round 1, no re-audit needed)

These still need your sign-off; they're flagged but not fixed:

1. `fx_cmds.py` numeric filter-type vs cutoff ambiguity.
2. `dsp/effects.py` per-effect normalisation destroying chain
   dynamics (would need full audio-regression testing before
   touching 100+ effects).
3. `dsp/effects.py` "unified 1-100" docstring mismatch with
   metadata defaults.
4. `core/buffer_store.py` undo-stack overflow silence.
5. `commands/buffer_cmds.py` concurrent-modification risk under
   InputController threading.

The first-sweep fixes are still in place: test suite passes
(228 unit tests + 5 opt-in audio tests skipped) after every
subsequent change.
