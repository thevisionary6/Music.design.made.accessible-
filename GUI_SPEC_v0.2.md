MDMA GUI (wxPython) — UPDATED SPEC: ACTION PANEL CLIENT
======================================================

Version: v0.2 (Action Panel Client MVP)
Author: Cyrus
Date: 2026-02-03
Purpose: Define an ultra-fast first GUI implementation that directly calls the
existing MDMA CLI/entrypoint commands (no new engine layers), while preserving
a clean path toward the fuller editor-style GUI spec later.

------------------------------------------------------
WHAT CHANGED (HIGH LEVEL)
------------------------------------------------------

- The DSL is deprecated for now and moved to a future "performance mode / song
  compression" feature track.
- The GUI's first shipping version is NOT a timeline editor.
- The GUI is a thin CLIENT over the reliable MDMA command-line entrypoint:
  it maps GUI actions to existing commands and shows results immediately.

This MVP is designed to be implementable in a couple hours while producing
massive usability gains.

------------------------------------------------------
PRIMARY GOAL OF THIS MVP
------------------------------------------------------

A "two-pane action panel" UI:

LEFT:    Select an object (Track / Preset / Bank / File / FX chain / etc.)
RIGHT:   Choose an action and fill parameters (via native widgets)
BOTTOM:  Output console showing command + engine output + errors

Outcome: Users can perform core MDMA workflows quickly without typing, while
still using the exact same command paths as the CLI.

------------------------------------------------------
NON-GOALS (FOR MVP)
------------------------------------------------------

- No grid/timeline selection
- No waveform view
- No complex track editing (insert/drag/region tools)
- No deep deck UI yet
- No new execution layers or render-plan compiler required

Those remain the end-goal spec (previous v0.1) and are planned as later phases.

------------------------------------------------------
TECH STACK
------------------------------------------------------

Language: Python
GUI Toolkit: wxPython
Engine Invocation: direct calls to MDMA "core entrypoint" (same as CLI)
Command Dispatch: function call OR subprocess (choose simplest stable path)
Logging: redirect engine stdout/stderr into GUI console

Theme: "Serum-ish" muted dark / soft contrast, but mostly native widgets.

------------------------------------------------------
INFORMATION ARCHITECTURE
------------------------------------------------------

Main Window Layout
------------------

[ MENU BAR ]
File | View | Engine | Help

[ MAIN SPLIT ]
---------------------------------------------------------
| LEFT: Object Browser | RIGHT: Action Panel            |
| (tree/list)          | (action dropdown + params)     |
---------------------------------------------------------

[ BOTTOM: Output Console ]
- command echo (what ran)
- engine stdout
- engine stderr
- status (ok/error)

Recommended: 3-way splitter (left / right / bottom).

------------------------------------------------------
LEFT PANEL: OBJECT BROWSER
------------------------------------------------------

A list/tree that shows the things users act on.

Minimum viable object types:
1) Tracks
2) Banks
3) Presets (filtered by bank)
4) Files (project audio assets / recent outputs)
5) FX Chains / Racks (optional; can be "later MVP+")

Object Browser Requirements:
- Keyboard navigation
- Search filter box
- Clear labeling
- Refresh button (re-scan MDMA state, banks, outputs)

Selection Model:
- Exactly one "active object" at a time for MVP
- Later: multi-select for batch actions

------------------------------------------------------
RIGHT PANEL: ACTION PANEL
------------------------------------------------------

Structure:
- Header shows selected object type + name
- Action dropdown (contextual to object type)
- Parameter editor area (auto-generated controls)
- Buttons: Run / Dry Run (optional) / Copy Command / Reset Params

Core Idea:
- Every action maps 1:1 to a real MDMA command path.
- The GUI never invents a new behavior; it only makes it faster to invoke.

------------------------------------------------------
ACTIONS BY OBJECT TYPE (MVP SET)
------------------------------------------------------

TRACK actions:
- Render Track
- Play Track (if supported; otherwise "Open Output")
- Apply Bank+Preset to Track
- Set Algorithm
- Set Voice Count
- Set Voice Algorithm
- Add FX (choose from list)
- Remove FX
- Set FX Parameter (choose FX -> param -> value)
- Export Track Stem

BANK actions:
- List Presets
- Refresh/Rescan Bank
- Set as Active Bank

PRESET actions:
- Preview Metadata (description, tags)
- Apply to Selected Track
- Clone Preset
- Save Track Overrides as New Preset (optional in MVP+)

FILE actions:
- Import into Track (or set track source)
- Open in Explorer / Reveal
- Delete from project outputs (careful prompt)

ENGINE actions (global):
- Rebuild/Index Banks
- Validate Install / Print Version
- Open Output Folder
- Clear Cache (if exists)
- Show Help / Quick Reference

NOTE:
Your exact command names can be wired later; in MVP, it's fine to stub action
definitions and fill them in progressively.

------------------------------------------------------
PARAMETER EDITOR (CRITICAL MVP FEATURE)
------------------------------------------------------

Parameter widgets are generated from a schema per action.

Parameter Types -> Widgets:
- int        -> SpinCtrl
- float      -> SpinCtrlDouble (or text + validation)
- bool       -> Checkbox
- enum       -> Dropdown
- string     -> TextCtrl
- file path  -> TextCtrl + Browse button

Rules:
- Defaults prefilled
- Validation before Run
- "Reset Params" restores defaults
- Every control has label + keyboard access

Accessibility Requirements:
- Predictable tab order
- No "canvas-only" controls
- Status text after run (success/fail)

------------------------------------------------------
COMMAND DISPATCH / EXECUTION
------------------------------------------------------

Two acceptable MVP approaches:

A) Direct function calls
- Import the MDMA entrypoint function and call it with argv-like list.
- Capture stdout/stderr using contextlib redirect.
- Fastest + avoids subprocess overhead.

B) Subprocess calls
- Run the same CLI command via subprocess.
- Capture stdout/stderr.
- Slightly more robust isolation, slower.

Recommendation:
Start with A if the entrypoint is stable and importable without side effects.
If import-time issues occur, switch to B.

Console Output Format:
- Show the command line string the GUI ran
- Then show stdout
- Then stderr (if any)
- Then a final status line

------------------------------------------------------
STATE MODEL (MVP)
------------------------------------------------------

The GUI maintains minimal state:
- Current selection (object type + id)
- Cached lists (tracks, banks, presets, files)
- Recent commands history

MDMA remains authoritative. "Refresh" re-queries / re-lists from MDMA.

No project JSON required for MVP (unless your CLI already has it).
If MDMA already has project persistence, the GUI should call those commands.

------------------------------------------------------
KEYBOARD SHORTCUTS (MVP)
------------------------------------------------------

- Ctrl+R: Run action
- Ctrl+L: Focus Output Console
- Ctrl+F: Focus Search in Object Browser
- F5: Refresh Object Browser
- Ctrl+C: Copy last command
- Esc: Clear selection / reset action parameters

------------------------------------------------------
MVP BUILD STEPS (COUPLE-HOUR IMPLEMENTATION PLAN)
------------------------------------------------------

1) wx Frame + splitters (Left / Right / Bottom)
2) Left object list with mock data
3) Right action dropdown + parameter form (static first)
4) Bottom console widget (multi-line read-only TextCtrl)
5) "Run" button wires to a placeholder dispatcher that prints the command string
6) Swap placeholder dispatcher for real MDMA entrypoint invocation
7) Implement Refresh to repopulate banks/presets/tracks from MDMA queries

That's enough to start daily-driving the GUI immediately.

------------------------------------------------------
ROADMAP: EVOLUTION TO FULL EDITOR SPEC
------------------------------------------------------

Phase 1 (this spec): Action Panel Client
- fast command mapping
- object browser
- parameter UI + console

Phase 2: Track-centric inspector + presets/FX rack improvements
- persistent per-track overrides in GUI view
- bulk actions
- better preset management

Phase 3: Minimal step-grid lane (optional)
- step-based region selection
- right-click insert/generate
- still no waveform required

Phase 4: Full editor spec (previous v0.1)
- event timeline, render plan JSON
- undo/redo on timeline edits
- deck view, performance mode
- optional waveform thumbnails

------------------------------------------------------
DEPRECATION NOTE: DSL
------------------------------------------------------

- The DSL is deprecated for now.
- It may return later as:
  (a) performance mode macro language
  (b) song compression/serialization method
  (c) export format for portable projects
- The GUI MVP intentionally bypasses DSL to reduce bugs and duplication.

------------------------------------------------------
FINAL NOTE
------------------------------------------------------

This MVP is intentionally "small but powerful":
it makes MDMA approachable and fast by turning your existing CLI into a
discoverable, clickable control surface — without risking engine rewrites.
