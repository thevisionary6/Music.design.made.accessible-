# Backend V2 — Phases 0 through 8 summary

Branch: `claude/analyze-backend-docs-phase-zero-00xhd`

The full backend roadmap in `Backend Docs V2/SKILL.md` is implemented
up through Phase 8 (selective DSP porting, first two effects). What
follows is the inventory — one file per section — plus the open
questions at the end.

## What shipped, by phase

### Phase 0 — repo prep

| File | Role |
| ---- | ---- |
| `mdma_rebuild/commands/router.py` | Extracted `build_command_table` + `COMMAND_OWNERS` + module loading + `dispatch()` |
| `mdma_rebuild/dsp/buffer_rearranger.py` | Renamed from `dsp/pattern.py` to free the `Pattern` name |
| `mdma_rebuild/core/audio_io.py` | Split from session.py: `_play_buffer`, `_play_via_file`, `play`, `preview_track`, `stop_playback`, `play_last_buffer`, `full_render`, etc. |
| `mdma_rebuild/core/buffer_store.py` | Split from session.py: numbered buffers, working buffer, undo/redo, snapshots |
| `mdma_rebuild/core/session_params.py` | Split from session.py: filter / envelope / voice / operator / tempo setters |
| `docs/backend_v2/phase_0_triage.md` | Per-module triage decisions |

Deletions approved by user: `dsp/visualization.py`, `commands/stub_cmds.py`.

### Phase 1 — Pattern

- `mdma_rebuild/backend/pattern.py` — `Pattern` class with every
  compositional transform (`fast`, `slow`, `t_p`, `l`, `gs`, `ex`,
  `cat`, `pre`) and every synthesis transform (`mod`, `dist`, `bf`,
  `spec`, `ir`, `gate`, `fx`). Immutable: every transform returns a
  new Pattern; self is never mutated.

### Phase 2 — chain application

- `Pattern._apply_chain`, `_apply_chain_with_nodes`,
  `_apply_builtin_filter`, `_apply_gate` — fully implemented.
- Gate uses a binary `(state, duration)` pattern that loops via
  `signalflow.BufferPlayer(loop=True)` for as long as the host event
  is playing.

### Phase 3 — Clock + Scheduler

- `backend/clock.py` — seconds-based clock, BPM convenience,
  bidirectional sync with `Session.bpm` via `session._clock`
  back-reference.
- `backend/scheduler.py` — thread-safe dispatch via per-handle event
  cursor; `Scheduler`, `ScheduleHandle`, `LoopHandle`. No
  `graph.wait(dur)` — concurrency comes for free.
- `Session.set_bpm` added via `session_params.bind_to`.

### Phase 4 — automation

- `backend/automation.py` — `AutomationSource` base + `Constant`,
  `LFO` (sine/saw/square/triangle), `Ramp`, `Envelope`, and
  `BindingHandle`.
- `Scheduler.bind_chain_param` / `bind_tempo` / `bind_master_volume`
  dispatched on every tick. `Pattern._apply_chain_with_nodes` returns
  per-stage node list so chain-param bindings route to the live node
  that owns the parameter.
- Graceful shutdown: `Scheduler.stop` clears bindings and calls
  `source.stop()` on each.

### Phase 5 — render + Session integration

- `backend/render.py` — `produce_pattern_buffer`,
  `produce_schedule_buffer`, `write_wav`, `render_pattern`,
  `render_schedule`. Audio production is cleanly separable from wav
  writing so a future `render_to_buffer` variant can share the
  produce step (spec §Render open question).
- `core/backend_bridge.py` — `Session.backend_graph`,
  `backend_clock`, `backend_scheduler`, `attach_backend_graph`,
  `play_pattern`, `schedule`, `backend_loop`, `render_pattern`.
  `render_pattern` populates `session.last_buffer` (float64) so the
  legacy effects / AI command path still sees the rendered audio
  through the existing buffer contract.

### Phase 6 — PAR + PARScheduler

- `backend/par.py` — `PAR(Pattern)` with `mod_speed`,
  `PARScheduleHandle.set_mod_speed`, `PARLoopHandle`,
  `PARScheduler`, and `Scheduler.bind_mod_speed`. `PARScheduler`
  shares the base scheduler's tick / binding / transport machinery;
  only dispatch semantics differ (effective duration =
  `dur / mod_speed`).

### Phase 7 — InputController

- `backend/input_controller.py` — `InputEvent`, `InputChannel`,
  `Controller`, `InputController`, `HandlerHandle`,
  `ControllerSource`. Thread-safe aggregator with its own poll
  thread; fnmatch-style glob matching for handlers; handler
  exceptions isolated.
- `backend/controllers/keyboard.py` — NVDA-safe, stdin-based (no
  global hook).
- `backend/controllers/midi.py` — via `mido`; note_on/off →
  press/release, CC → `cc_N` continuous channels,
  program_change → trigger.
- `backend/controllers/osc.py` — via `python-osc`; OSC messages →
  trigger events, first float arg populates a continuous channel
  created on demand.

### Phase 8 — selective porting (first two effects)

- `backend/effects/__init__.py` — documents the porting convention.
- `backend/effects/soft_clip.py` — tanh-based distortion for
  `Pattern.dist()`.
- `backend/effects/delay.py` — single-tap feedback delay for
  `Pattern.fx()`.

## Test suite

178 total:

- **173 unit tests** run headless without an audio device:
  `test_pattern.py` (51), `test_scheduler.py` (30),
  `test_automation.py` (27), `test_render.py` (18), `test_par.py` (13),
  `test_input_controller.py` (23), `test_effects.py` (11).
- **5 audio-integration tests** under `tests/audio/` gated on
  `MDMA_AUDIO_TESTS=1` — exercise real `signalflow.SVFilter` and
  looping `BufferPlayer` for the gate path.

Run everything with:

    python -m unittest discover tests

Run the audio integration subset on a host with an audio device:

    MDMA_AUDIO_TESTS=1 python -m unittest tests.audio.test_pattern_dsp

## Files touched / added

Added under the branch:
- `backend/__init__.py`, `backend/pattern.py`, `backend/clock.py`,
  `backend/scheduler.py`, `backend/automation.py`,
  `backend/render.py`, `backend/par.py`,
  `backend/input_controller.py`.
- `backend/controllers/__init__.py`, `keyboard.py`, `midi.py`,
  `osc.py`.
- `backend/effects/__init__.py`, `soft_clip.py`, `delay.py`.
- `core/audio_io.py`, `core/buffer_store.py`, `core/session_params.py`,
  `core/backend_bridge.py`.
- `commands/router.py`.
- `tests/test_pattern.py`, `test_scheduler.py`, `test_automation.py`,
  `test_render.py`, `test_par.py`, `test_input_controller.py`,
  `test_effects.py`, `tests/audio/test_pattern_dsp.py`.
- `docs/backend_v2/phase_0_triage.md`,
  `docs/backend_v2/phases_0_through_8_summary.md` (this file).

Removed:
- `mdma_rebuild/commands/stub_cmds.py`, `mdma_rebuild/dsp/visualization.py`.

Renamed:
- `mdma_rebuild/dsp/pattern.py` → `mdma_rebuild/dsp/buffer_rearranger.py`.

## Scope boundaries respected

The following items from the spec's §"Out of scope for v1" list were
**not** implemented, per the spec's instruction:

- Polyphony within a single pattern.
- Generalising Pattern to Sound/Over primitive dicts.
- Mid-event pattern swap.
- Transform-argument automation (`.fast(live_source)`).
- `render_to_buffer` (architecture preserved — `produce_pattern_buffer`
  and `write_wav` are separable — but the buffer-returning public
  function is not yet a real export).
- Output controllers / LED feedback / OSC-out.
- MIDI clock sync.
- Quantization at schedule time.
- GamepadController (interface is open; implementation deferred).
- Learn mode for automation.
- Wholesale port of all 113 effects from `dsp/effects.py`.
- Wholesale rewrite of `dsp/monolith.py`.
- DJ mode rework to route through Scheduler.

## Open questions

A handful of choices need your call before the merge is "done":

1. **`/playp` or similar command.** Phase 5b's "Done when" mentions a
   new command that plays a Pattern via Session. I wired
   `session.play_pattern(...)` but did not add a slash command for
   it. Which name would you like — `/playp`, `/pp`, `/patplay`? And
   do you want it added to `gen_cmds.py` or a new
   `pattern_backend_cmds.py`?

2. **Phase 8 porting priorities.** Two reference effects shipped
   (`soft_clip`, `delay`). The spec recommends "the most-used effects
   from `dsp/effects.py`" next. From the old codebase that's typically
   reverb / chorus / flanger / phaser / a few filter variants. Which
   are highest priority for your workflow?

3. **`bind_master_volume` graph hook.** I implemented two paths:
   `graph.set_output_level(value)` if present, else
   `graph.output_level = value` attribute write. SignalFlow's stock
   `AudioGraph` doesn't expose either out of the box; do you have a
   preferred wiring (e.g. always multiply the graph's output node by
   a `Mul` tap and automate its `input1`)? Happy to add a
   `GraphOutputLevel` helper if so.

4. **Session backend scheduler lifecycle.** `session.backend_scheduler()`
   lazy-creates the scheduler and clock on first use but never calls
   `scheduler.start()`. Callers that want live dispatch need to call
   `session.backend_scheduler().start()` themselves. Do you want
   `schedule()` / `backend_loop()` to auto-start the scheduler on
   first call, or stay explicit?

5. **`bind_mod_speed` install location.** I installed it onto
   `Scheduler` at `par.py` import time (so base Scheduler stays
   PAR-ignorant at module level). Alternative: move it into
   `scheduler.py` as a stub that raises until PAR is imported, or
   duplicate it on `PARScheduler` only. Fine as-is? The current shape
   means anyone with a backend `Scheduler` and a `PARScheduleHandle`
   can bind, which matches the spec.

6. **Handler logging channel.** `InputController` uses `print` for
   handler-exception and controller-exception reports (intentional
   Phase 7 placeholder). Want a real `logging.getLogger("mdma.input")`
   route, or is the print-based surface fine for live use?

Everything else from the spec is in and tested. Ready for review
when you're ready.
