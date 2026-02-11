MDMA — INTERFACE TRANSITION SPEC
Version: 1.0
Status: Active Transition (Implemented in v52)
Purpose: Clarify execution truth, deprecate DSL safely, and realign testing and development flow

------------------------------------------------------------
1. CONTEXT AND PROBLEM STATEMENT
------------------------------------------------------------

MDMA has reached a maturity point where multiple interfaces (CLI, DSL, future GUI)
exist or are planned. During development, an implicit assumption emerged:

    "The DSL is functionally equivalent to the core MDMA CLI."

This assumption proved false.

As a result:
- Bugs were chased that did not exist in the core engine
- Debug signals were fragmented across interfaces
- Test failures reflected interface divergence, not engine defects
- Cognitive overhead increased rather than decreased

This document defines a controlled transition that:
- Re-establishes a single source of truth
- Preserves forward flexibility
- Avoids premature interface lock-in
- Restores developer trust in test outcomes

------------------------------------------------------------
2. DECLARATION OF TRUTH
------------------------------------------------------------

Effective immediately:

- The MDMA core engine + CLI execution path is the sole behavioral authority.
- All other interfaces are clients of the engine, not peers.
- No interface is assumed equivalent unless mechanically proven.

This is not a value judgment against the DSL or future interfaces.
It is a boundary clarification.

------------------------------------------------------------
3. CORE INSIGHT DRIVING THE TRANSITION
------------------------------------------------------------

One-command-at-a-time interaction has proven to be:
- Cognitively lighter
- Easier to debug
- Easier to reason about
- More compatible with exploratory sound design

Batch execution via DSL introduced:
- Hidden state accumulation
- Manual clearing requirements
- Increased working memory load
- Debug ambiguity across grouped operations

Acceleration in execution did not translate to acceleration in creativity.

------------------------------------------------------------
4. DSL STATUS CHANGE
------------------------------------------------------------

The DSL is formally deprecated as a PRIMARY USER INTERFACE.

This means:
- It is no longer assumed to mirror CLI behavior
- It is no longer used as a debugging reference
- It is no longer required for feature completeness

This does NOT mean:
- Immediate deletion
- Loss of long-term value
- Architectural abandonment

------------------------------------------------------------
5. ACCEPTABLE FUTURE ROLES FOR THE DSL
------------------------------------------------------------

The DSL may be reintroduced later in one or more of the following constrained roles:

1. Performance / Macro Layer
   - Batch execution where intentional
   - Repeatable action sets
   - Explicit, non-interactive runs

2. Serialization / Project Description
   - Save/load engine state
   - Shareable or archival formats
   - Deterministic reproduction

3. Internal Transport / Backend Protocol
   - GUI-to-engine messaging
   - Not exposed as a creative surface

In all cases:
- The DSL must call the same engine entry point as the CLI
- Behavioral parity must be testable, not assumed

------------------------------------------------------------
6. ENGINE EXECUTION SPINE (NON-NEGOTIABLE)
------------------------------------------------------------

All interfaces must converge on a single execution spine.

Conceptually:
- Command construction
- Engine execution
- Explicit result/event return

Interfaces may differ in presentation, not behavior.

No interface may:
- Mutate engine state directly
- Bypass validation layers
- Introduce implicit execution ordering

------------------------------------------------------------
7. TESTING REALIGNMENT
------------------------------------------------------------

Testing must be realigned to prevent phantom debugging.

Minimum guarantees going forward:

1. Interface Isolation
   - Tests validate engine behavior directly
   - Interfaces are tested as thin clients

2. Command Parity Validation (when applicable)
   - If two interfaces exist, parity must be mechanically verified

3. State Visibility Guarantees
   - All audio-holding objects must be discoverable through a single registry
   - No hidden or orphaned execution state

4. Deterministic Rendering Where Possible
   - Same inputs + same commands = same outputs (within defined tolerances)

------------------------------------------------------------
8. GUI DEVELOPMENT BOUNDARY CONDITIONS
------------------------------------------------------------

This spec intentionally does NOT prescribe GUI design.

However, the following constraints apply:

- The GUI is a client, not an engine
- The GUI does not own DSP logic
- The GUI does not introduce new execution semantics
- The GUI calls the same engine commands as the CLI

How those commands are surfaced is explicitly out of scope.

------------------------------------------------------------
9. TRANSITION PHASE SUMMARY
------------------------------------------------------------

Phase 1 — Stabilize Truth (COMPLETE - v52)
- CLI declared authoritative
- DSL deprecated as primary interface
- Engine execution spine isolated

Phase 2 — Reduce Surface Area
- Debugging performed only against engine + CLI
- Tests rewritten to target core behavior

Phase 3 — Forward Expansion
- GUI development proceeds against stable engine API
- DSL reconsidered only in constrained, supporting roles

------------------------------------------------------------
10. SUCCESS CRITERIA
------------------------------------------------------------

This transition is considered successful when:
- Bugs are traceable to a single execution path
- Test failures correspond to real engine defects
- Interface changes do not require engine rewrites
- Creative workflow feels lighter, not heavier

------------------------------------------------------------
END OF SPEC
