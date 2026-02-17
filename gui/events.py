"""MDMA GUI Custom Event Types.

Defines wx event types used for inter-component communication within
the GUI layer.  All events flow through the standard wx event system —
windows never call each other directly.

The Bridge fires these events after completing engine operations.
Windows bind handlers to receive updates.  The Object Registry fires
RegistryEvents (see core/registry.py) which the Bridge translates into
wx events for thread-safe GUI updates.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Window Communication Model

BUILD ID: gui_events_v1.0_phase1
"""

from __future__ import annotations

try:
    import wx
    import wx.lib.newevent
    _WX_AVAILABLE = True
except ImportError:
    _WX_AVAILABLE = False


if _WX_AVAILABLE:
    # ------------------------------------------------------------------
    # Registry bridge events — fired when the object registry changes
    # ------------------------------------------------------------------

    # An object was created in the registry.
    # attrs: object_id (str), obj_type (str), name (str)
    ObjectCreatedEvent, EVT_OBJECT_CREATED = wx.lib.newevent.NewEvent()

    # An object was updated in the registry.
    # attrs: object_id (str), obj_type (str), name (str)
    ObjectUpdatedEvent, EVT_OBJECT_UPDATED = wx.lib.newevent.NewEvent()

    # An object was deleted from the registry.
    # attrs: object_id (str), obj_type (str), name (str)
    ObjectDeletedEvent, EVT_OBJECT_DELETED = wx.lib.newevent.NewEvent()

    # An object was renamed in the registry.
    # attrs: object_id (str), obj_type (str), name (str), old_name (str)
    ObjectRenamedEvent, EVT_OBJECT_RENAMED = wx.lib.newevent.NewEvent()

    # ------------------------------------------------------------------
    # Bridge operation events — fired after DSP/session operations
    # ------------------------------------------------------------------

    # A generation operation completed (beat, melody, loop, etc.).
    # attrs: object_id (str), obj_type (str), command (str), status (str)
    GenerationCompleteEvent, EVT_GENERATION_COMPLETE = wx.lib.newevent.NewEvent()

    # An effect chain was applied.
    # attrs: object_id (str), chain_id (str), status (str)
    EffectAppliedEvent, EVT_EFFECT_APPLIED = wx.lib.newevent.NewEvent()

    # A transform/mutation was applied.
    # attrs: source_id (str), result_id (str), transform (str), status (str)
    TransformCompleteEvent, EVT_TRANSFORM_COMPLETE = wx.lib.newevent.NewEvent()

    # Audio playback state changed.
    # attrs: state (str: 'playing', 'stopped', 'paused'), object_id (str)
    PlaybackStateEvent, EVT_PLAYBACK_STATE = wx.lib.newevent.NewEvent()

    # ------------------------------------------------------------------
    # Console / status events
    # ------------------------------------------------------------------

    # A status message should be displayed in the console/status bar.
    # attrs: message (str), level (str: 'info', 'warning', 'error')
    StatusMessageEvent, EVT_STATUS_MESSAGE = wx.lib.newevent.NewEvent()

    # A CLI command was executed (for console mirror).
    # attrs: command (str), output (str), status (str: 'ok', 'error')
    CommandExecutedEvent, EVT_COMMAND_EXECUTED = wx.lib.newevent.NewEvent()

else:
    # Stubs so the module can be imported without wxPython installed.
    # This allows tests and CLI-only usage to reference event types
    # without crashing.

    class _StubEvent:
        """Placeholder event class when wx is not available."""
        pass

    class _StubBinder:
        """Placeholder event binder when wx is not available."""
        pass

    ObjectCreatedEvent = _StubEvent
    EVT_OBJECT_CREATED = _StubBinder()
    ObjectUpdatedEvent = _StubEvent
    EVT_OBJECT_UPDATED = _StubBinder()
    ObjectDeletedEvent = _StubEvent
    EVT_OBJECT_DELETED = _StubBinder()
    ObjectRenamedEvent = _StubEvent
    EVT_OBJECT_RENAMED = _StubBinder()
    GenerationCompleteEvent = _StubEvent
    EVT_GENERATION_COMPLETE = _StubBinder()
    EffectAppliedEvent = _StubEvent
    EVT_EFFECT_APPLIED = _StubBinder()
    TransformCompleteEvent = _StubEvent
    EVT_TRANSFORM_COMPLETE = _StubBinder()
    PlaybackStateEvent = _StubEvent
    EVT_PLAYBACK_STATE = _StubBinder()
    StatusMessageEvent = _StubEvent
    EVT_STATUS_MESSAGE = _StubBinder()
    CommandExecutedEvent = _StubEvent
    EVT_COMMAND_EXECUTED = _StubBinder()
