"""MDMA GUI Bridge — Session/DSP Adapter.

The Bridge is the single adapter object that all GUI windows call for
session and DSP operations.  It translates GUI actions into engine calls
and publishes results as wx events.

Design rules:
- No window holds a reference to another window.
- No window calls DSP or session methods directly.
- All engine interaction goes through Bridge methods.
- The Bridge owns the ObjectRegistry and fires events on change.
- Thread safety: long-running DSP operations are dispatched to a
  worker thread; results are posted back as wx events.

See: docs/specs/GUI_WINDOW_ARCHITECTURE_SPEC.md — Window Communication Model

BUILD ID: gui_bridge_v1.0_phase1
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from mdma_rebuild.core.registry import (
    ObjectRegistry,
    RegistryEvent,
    RegistryEventType,
)
from mdma_rebuild.core.objects import (
    MDMAObject,
    Pattern,
    BeatPattern,
    Loop,
    AudioClip,
    Patch,
    EffectChain,
    Track,
    Song,
)

logger = logging.getLogger(__name__)

# Try to import wx for event posting — Bridge can also run headless
# for testing and CLI integration.
try:
    import wx
    from .events import (
        ObjectCreatedEvent,
        ObjectUpdatedEvent,
        ObjectDeletedEvent,
        ObjectRenamedEvent,
        GenerationCompleteEvent,
        StatusMessageEvent,
        CommandExecutedEvent,
    )
    _WX_AVAILABLE = True
except ImportError:
    _WX_AVAILABLE = False


class Bridge:
    """Adapter between GUI windows and the MDMA engine.

    All GUI windows receive a reference to the same Bridge instance.
    They call Bridge methods to perform operations; the Bridge calls
    the Session/DSP layer and updates the ObjectRegistry.  Registry
    change events are translated into wx events and posted to the
    GUI event target (usually the top-level shell frame).

    Parameters:
        session: The MDMA Session instance (from mdma_rebuild.core).
        registry: The shared ObjectRegistry instance.
        event_target: A wx.EvtHandler to post events to (usually the
            shell frame).  Can be None for headless/test usage.
    """

    def __init__(
        self,
        session: Any,
        registry: Optional[ObjectRegistry] = None,
        event_target: Any = None,
    ) -> None:
        self.session = session
        self.registry = registry or ObjectRegistry()
        self._event_target = event_target

        # Subscribe to registry events so we can relay them as wx events
        self.registry.subscribe(self._on_registry_event)

    # ---- Registry passthrough -----------------------------------------------

    def register_object(self, obj: MDMAObject, auto_name: bool = True) -> str:
        """Register a new object in the shared registry."""
        return self.registry.register(obj, auto_name=auto_name)

    def get_object(self, object_id: str) -> Optional[MDMAObject]:
        """Retrieve an object by ID."""
        return self.registry.get(object_id)

    def get_object_by_name(self, name: str, obj_type: Optional[str] = None) -> Optional[MDMAObject]:
        """Retrieve an object by name."""
        return self.registry.get_by_name(name, obj_type)

    def list_objects(self, obj_type: Optional[str] = None) -> list[MDMAObject]:
        """List all objects, optionally filtered by type."""
        return self.registry.list_objects(obj_type)

    def delete_object(self, object_id: str) -> bool:
        """Delete an object from the registry."""
        return self.registry.delete(object_id)

    def rename_object(self, object_id: str, new_name: str) -> Optional[MDMAObject]:
        """Rename an object."""
        return self.registry.rename(object_id, new_name)

    def duplicate_object(self, object_id: str, new_name: str = "") -> Optional[MDMAObject]:
        """Duplicate an object."""
        return self.registry.duplicate(object_id, new_name)

    # ---- Command execution --------------------------------------------------

    def execute_command(self, command: str, args: Optional[list[str]] = None) -> str:
        """Execute a CLI command through the session.

        This is the primary interface for GUI actions — each button press
        or parameter change translates into a command string that the
        Bridge dispatches to the session's command table.

        Returns the command output string.
        """
        args = args or []
        full_cmd = command if not args else f"{command} {' '.join(args)}"
        logger.info("Bridge executing: %s", full_cmd)

        # TODO: Phase 3+ — dispatch through the session's command table
        # and capture output.  For now, return a placeholder.
        output = f"[bridge] Command queued: {full_cmd}"
        self._post_status(output)
        return output

    # ---- Status / feedback --------------------------------------------------

    def post_status(self, message: str, level: str = "info") -> None:
        """Post a status message to the GUI."""
        self._post_status(message, level)

    # ---- Event relay --------------------------------------------------------

    def _on_registry_event(self, event: RegistryEvent) -> None:
        """Translate a RegistryEvent into a wx event and post it."""
        if not _WX_AVAILABLE or self._event_target is None:
            return

        evt_map = {
            RegistryEventType.OBJECT_CREATED: ObjectCreatedEvent,
            RegistryEventType.OBJECT_UPDATED: ObjectUpdatedEvent,
            RegistryEventType.OBJECT_DELETED: ObjectDeletedEvent,
            RegistryEventType.OBJECT_RENAMED: ObjectRenamedEvent,
        }

        evt_cls = evt_map.get(event.event_type)
        if evt_cls is None:
            return

        wx_event = evt_cls(
            object_id=event.object_id,
            obj_type=event.obj_type,
            name=event.name,
        )
        # Add extra data for rename events
        if event.event_type == RegistryEventType.OBJECT_RENAMED:
            wx_event.old_name = event.data.get("old_name", "")

        wx.PostEvent(self._event_target, wx_event)

    def _post_status(self, message: str, level: str = "info") -> None:
        """Post a StatusMessageEvent to the GUI event target."""
        logger.log(
            logging.WARNING if level == "warning"
            else logging.ERROR if level == "error"
            else logging.INFO,
            message,
        )
        if _WX_AVAILABLE and self._event_target is not None:
            evt = StatusMessageEvent(message=message, level=level)
            wx.PostEvent(self._event_target, evt)

    # ---- Lifecycle ----------------------------------------------------------

    def set_event_target(self, target: Any) -> None:
        """Set or update the wx event target (e.g. after frame creation)."""
        self._event_target = target
