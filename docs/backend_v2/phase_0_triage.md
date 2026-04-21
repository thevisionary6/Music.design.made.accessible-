# Phase 0 Triage Audit

Per `Backend Docs V2/SKILL.md` §Phase 0.4: each module on the triage list
needs an explicit keep/cleanup/remove decision before Phase 1 begins. This
document records the decisions made during Phase 0 repo-prep. No modules
were actually deleted — decisions are surfaced here for the user to
approve before any cuts happen.

LOC numbers are measured post-session split (`core/session.py` at 1472
LOC after the V2 repo-prep extract).

---

## DSP layer — likely keep

| Module | LOC | Import count | Verdict | Notes |
| ------ | --- | ------------ | ------- | ----- |
| `dsp/streaming.py` | 980 | 6 | **Keep** | SoundCloud/YouTube/local ingestion + float64 conversion. Used by `gen_cmds`, `dsl_cmds`, `playback_cmds`, `buffer_cmds`. Stable API, agnostic to the audio-production paradigm. |
| `dsp/advanced_ops.py` | 1060 | 15 | **Keep** | `UserStack` singleton for cross-command user variables, auto-chunking, remix patterns. Heavily integrated; accessibility-relevant. |
| `dsp/scaling.py` | 515 | 6 | **Keep** | Canonical 1–100 parameter scaling. `core/session.py` depends on it directly. |
| `dsp/music_theory.py` | 438 | 3 | **Keep** | Scales, chords, progressions, voice leading, MIDI helpers. Required by generative layer; symbolic only, no DSP. |
| `dsp/envelopes.py` | 57 | 1 | **Keep** | ADSR envelope class. Tiny; core primitive. |
| `dsp/transforms.py` | 402 | 1 | **Keep** | Musical transforms (retrograde, inversion, augmentation). Only `loop_gen.py` imports it — small audience but clean. |
| `dsp/generators.py` | 1253 | 4 | **Keep** | Sound generators (drums, FX, utility tones). `stub_cmds.py` redirects into this. |

---

## DSP layer — keep with bug pass

| Module | LOC | Import count | Verdict | Notes |
| ------ | --- | ------------ | ------- | ----- |
| `dsp/dj_mode.py` | 3367 | 20 | **Keep; bug pass later** | Deck + crossfader + VDA/NVDA routing. Audio mixing eventually routes through Scheduler (open question in `scheduler_spec.md`); defer that rework. Pare device-probe code that's clearly dead on target OS. |
| `dsp/beat_gen.py` | 784 | 3 | **Keep; symbolic parts only for now** | Drum synthesis + humanization. Buffer-generating code stays until a SignalFlow-node rewrite lands. |
| `dsp/loop_gen.py` | 316 | 1 | **Keep** | Loop-spec composer (bpm, key, mood, layers). Clean. |

---

## DSP layer — pare significantly

| Module | LOC | Import count | Verdict | Notes |
| ------ | --- | ------------ | ------- | ----- |
| `dsp/performance.py` | 852 | 7 | **Keep with cleanup** | Macros, snapshots, controlled-randomness, autopilot. Actively used via `perf_cmds`; cleanup target is the snapshot serializer. |
| `dsp/enhancement.py` | 810 | 3 | **Audit before touching** | AI-powered output-stage processing. Defaults to OFF. Confirm no hidden session-level dependencies, then decide. |
| `dsp/stems.py` | 796 | 1 | **Audit before touching** | Stem separation via Demucs/Spleeter (external ML dependency). Only `dj_cmds` mentions it. Surface the Demucs question to the user before keeping. |
| `dsp/visualization.py` | 330 | 0 | **Proposed remove** | Zero importers in the codebase. Terminal visualization stub (waveform/spectrum/meters/beat indicators), mostly unimplemented. Flagged for removal pending user approval. |

---

## Commands layer — pare significantly

| Module | LOC | Verdict | Notes |
| ------ | --- | ------- | ----- |
| `commands/stub_cmds.py` | 1580 | **Keep with significant cleanup** | 51 `cmd_*` stubs, many of which are overridden by real implementations elsewhere (`generators.py`, `audiorate_cmds`, etc.). The router loads stubs at the lowest priority so real handlers win. Action: walk each stub, confirm a real handler owns it, then delete the stub. Likely target: 10–20 stubs remain. |

---

## Commands layer — audit before touching

These are the largest command modules. Keep everything that has users, but
look for duplicates and dead branches. Do **not** pare aggressively without
user sign-off; these implement mature workflows and users may depend on
obscure commands.

| Module | LOC | Verdict | Notes |
| ------ | --- | ------- | ----- |
| `commands/advanced_cmds.py` | 3025 | **Keep** | Note/temperament (`/nt`), durations (`/d`), chord mode (`/c`), user stack (`/=`), random, math, conditionals, live loop mode (`/lbm`). Core DSL backbone; ~53 `cmd_*` functions. Potential dead code in old preference system (`/up`, `/upl`) — audit, don't assume. |
| `commands/fx_cmds.py` | 3312 | **Keep** | Filter bank + FX aliasing + effect chain setup. ~65 `cmd_*` functions. Brittle aliasing — don't refactor without a full effect-system overhaul. |
| `commands/dj_cmds.py` | 3347 | **Keep** | Deck + crossfader + headphone/cue + VDA/NVDA. ~47 `cmd_*` functions. Verify hardware-dependent fallbacks during Phase 7 InputController work. |
| `commands/dsl_cmds.py` | 2496 | **Keep** | MAD DSL (`/mel`, `/out`, `/wa`, `/play`, `/live`, `/render`). Core user-facing DSL. |
| `commands/synth_cmds.py` | 2659 | **Keep** | Carrier/modulator/voice counts, waveform params, tone gen, envelope settings, presets. ~60 `cmd_*` functions. Possible consolidation opportunity, but no red flags. |

---

## Resolved decisions

1. **`dsp/visualization.py`** — **Removed.** Zero importers; terminal visualization stub that was never wired in.
2. **`dsp/stems.py`** — **Keep.** External ML dependency (Demucs) stays; no code changes.
3. **`dsp/enhancement.py`** — **Keep.** Default-off AI stage; no code changes.
4. **`commands/stub_cmds.py`** — **Removed.** The `STUB_COMMANDS` dict delegated only 10 commands (`vamp`, `fc`, `gg`, `voc`, `spc`, `lfo`, `ump`, `audiorate`, `bbe`, `pack`), every one of which is owned by a real module in `router.COMMAND_OWNERS`. After removal, `build_command_table()` still registers every previously-owned command — only the stub placeholders go. `bmdma.py`'s `GENERATOR_ALGORITHMS` import fell back cleanly to the "26 known count" path that already existed.

After these cuts, `build_command_table()` goes from 620 → 616 commands.
Spot-check: `vamp`, `fc`, `gg`, `ump`, `audiorate` all still resolve.

No other modules were flagged for removal. Everything else lands in
either "Keep" or "Keep with cleanup deferred to Phase 8+ porting work."

---

## Phase 0 repo-prep status

- **0.1 Extract command router.** Done. New module `mdma_rebuild/commands/router.py` owns `build_command_table`, `COMMAND_OWNERS`, module-loading, and a convenience `dispatch()` helper. `bmdma.py` now re-exports from it.
- **0.2 Rename `dsp/pattern.py` → `dsp/buffer_rearranger.py`.** Done. All in-tree imports updated (`commands/pattern_cmds.py`, `commands/gen_cmds.py`, `bmdma.py`, `dsp/__init__.py`).
- **0.3 Pare `core/session.py`.** Done. 2566 LOC → 1472 LOC via three helper modules:
  - `core/audio_io.py` (430 LOC) — `_play_buffer`, `_play_via_file`, `_open_file`, `play`, `preview_track`, `stop_playback`, `playback_status`, `play_last_buffer`, `full_render`.
  - `core/buffer_store.py` (465 LOC) — numbered buffer + working buffer management, undo/redo stacks, snapshot save/restore.
  - `core/session_params.py` (447 LOC) — filter, envelope, voice, operator, and DSP parameter setters; `resonance` / `cutoff` properties.
  Each helper module exposes `bind_to(Session)` and is wired at the bottom of `core/session.py`. External call sites are unchanged.
- **0.4 Triage audit.** This document.

Session is now under the 1500-LOC target. Phase 1 (Pattern class) can
proceed once the items above are approved.
