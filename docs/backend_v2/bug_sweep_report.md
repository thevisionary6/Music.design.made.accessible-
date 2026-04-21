# Bug sweep report — post-Phase-8

Audit scope: effect application, buffer handling, and user-expectation
mismatches in the legacy main codebase (everything outside
`mdma_rebuild/backend/` and `tests/`). Runs ahead of any further DSP
porting per your instruction.

## Fixes applied

| Area | File:line | Severity | What was wrong | Fix |
| ---- | --------- | -------- | -------------- | --- |
| Effect application | `core/session.py:571` | HIGH | `apply_fx_chain` silently swallowed every exception — unknown effects and DSP crashes looked identical to "the effect worked but was subtle." | Split into `ImportError` (clear message about `dsp.effects` missing) and generic-`Exception` paths, both of which print a `[session] fx chain …` line so the user sees what happened. |
| Effect application | `dsp/effects.py:~3880` | MEDIUM | `apply_effects_with_params` silently skipped unknown effect names. | Added `print("[effects] unknown effect …")`. |
| Effect application | `dsp/effects.py:~3925` | MEDIUM | Per-effect failures swallowed. | Added `print("[effects] effect X failed: …")`. |
| Effect application | `dsp/effects.py:~4103` (`remove_dc_offset`) | MEDIUM | `if audio.ndim == 2:` accessed `audio[:, 1]` without checking `shape[1] >= 2`, raising IndexError on `(N, 1)` arrays. Rewritten to loop over channels and degrade `(N, 1)` to mono. |
| Effect application | `dsp/effects.py:~4156` (`gentle_highpass` fallback) | MEDIUM | Same bug pattern. | Same channel-loop fix. |
| Effect application | `dsp/effects.py:~4216` (`gentle_highshelf`) | MEDIUM | Same bug pattern. | Same channel-loop fix. |
| Buffers | `core/buffer_store.py:310-333` (`has_real_working_audio`) | MEDIUM | Relied purely on `working_buffer_source != 'init'`. Legacy commands do `session.working_buffer = arr` directly without updating `working_buffer_source`, which made that audio invisible to every priority resolver (`get_any_audio`, `get_playable_buffer`). | Added a content-check fallback: if the source label says `'init'` but the buffer has peak amplitude > 1e-7, treat it as real. `get_any_audio` and `get_playable_buffer` now route through `has_real_working_audio` instead of inlining the label check. |
| Buffers | `commands/dsl_cmds.py:1965` | LOW | `session.working_buffer_source = saved_src if saved_src else 'init'` — empty strings got coerced to `'init'`. | Switched to `saved_src if saved_src is not None else 'init'`. |
| User expectations | `core/session_params.py:~128` (`set_cutoff`) | LOW | Silently clamped to `[20, 20000]`. | Prints `[session] cutoff X clamped to Y Hz …` when the user's value is out of range, so the outcome is visible. |

## Flagged, not fixed (need your call)

These surfaced in the audit but involve design judgment, so I'd like
your sign-off before touching them:

1. **`fx_cmds.py:2040-2047` — numeric filter-type vs cutoff
   ambiguity.** Small integers select a filter *type* (lpf / hpf /
   etc.), but bigger integers are interpreted as cutoff frequencies.
   Confusing when the user types `10` and means "10 Hz cutoff" or
   "allpass-variant" depending on context. Two clean paths:
   (a) require explicit `type:lpf` / `cut:800` prefixes;
   (b) always use string aliases for type selection. **Your call?**

2. **`dsp/effects.py:~78` + per-effect normalisation — dynamics
   destroyed in chains.** Several effects divide by `max(|x|)` on the
   way out. Chaining two such effects flattens gain relationships
   (effect-A-at-0.5 and effect-B-at-2.0 both become 1.0). Fixing this
   means auditing every effect that calls `_normalise` and deciding
   whether to drop the per-effect normalisation or apply only at
   chain end. Needs audio-testing to confirm listening impact;
   risky to touch 100+ effects without a baseline.

3. **`dsp/effects.py:32-39` — doc claims "unified 1-100 scaling"
   but some effects accept real units (Hz, ms) via metadata
   defaults.** Fix is a docstring audit, not code. Leaving until
   the effect-metadata layer is overhauled (probably as part of
   Phase 8 porting).

4. **`core/buffer_store.py:~323` — undo stack overflow silent.** We
   silently drop the oldest entry when the stack hits
   `_undo_max_depth`. Not a bug per se, but surprising during long
   sessions. Would you like the overflow logged, the depth
   configurable, or both?

5. **`commands/buffer_cmds.py:78` — concurrent modification risk.**
   `for idx in session.buffers:` followed by `session.buffers[idx]`.
   Fine today because commands don't run concurrently; once Phase 7
   InputController handlers fire from a different thread, this path
   could race. Flagged here so we don't forget to revisit when the
   threading audit (per scheduler spec §Threading) happens.

## Notes

Two of the Explore agent's findings turned out to be false positives
on re-reading the code:

- `dsl_cmds.py:1866`'s `except ValueError: pass` leaves `value` as
  the string from line 1857 — intentional fallback, not a bug.
- `playback_cmds.py:232-235`'s error message ("Working buffer
  contains only silence") is accurate; the only path that triggers
  it is `working_buffer_source == 'init' AND peak < 1e-7`, which
  really does mean "nothing has been generated yet".

Test suite green after every fix: 194 unit tests pass, 5 audio
integration tests skipped (opt-in).
