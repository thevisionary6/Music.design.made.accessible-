"""MDMA Object Registry.

The single authoritative store for all first-class objects in a project.
Both the CLI and the GUI read from and write to the registry. Nothing
exists outside the registry except transient intermediate computation.

The registry provides:
- Registration, retrieval, update, and deletion of objects
- Name-based and type-based lookup
- Duplicate and search operations
- Dependency tracking (which objects reference a given object)
- Event subscriptions so GUI windows update automatically
- JSON serialisation for project persistence

This module is Phase 2 of the Object Model migration. Attaching it to
the Session object does not modify any existing code paths.

See: docs/specs/OBJECT_MODEL_SPEC.md for the full specification.

BUILD ID: registry_v1.0_phase2
"""

from __future__ import annotations

import json
import copy
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable, Optional

from .objects import MDMAObject, OBJECT_TYPE_MAP, _new_id, _now

logger = logging.getLogger(__name__)


# ============================================================================
# REGISTRY EVENTS
# ============================================================================

class RegistryEventType:
    """Event type constants for registry subscriptions."""
    OBJECT_CREATED = "object_created"
    OBJECT_UPDATED = "object_updated"
    OBJECT_DELETED = "object_deleted"
    OBJECT_RENAMED = "object_renamed"


class RegistryEvent:
    """Payload for a registry change event.

    Attributes:
        event_type: One of the RegistryEventType constants.
        object_id: ID of the affected object.
        obj_type: Type tag of the affected object.
        name: Current name of the affected object.
        data: Optional extra data (e.g. old name on rename).
    """

    __slots__ = ("event_type", "object_id", "obj_type", "name", "data")

    def __init__(
        self,
        event_type: str,
        object_id: str,
        obj_type: str = "",
        name: str = "",
        data: Optional[dict] = None,
    ) -> None:
        self.event_type = event_type
        self.object_id = object_id
        self.obj_type = obj_type
        self.name = name
        self.data = data or {}


# ============================================================================
# AUTO-NAMER
# ============================================================================

class _AutoNamer:
    """Generates sequential names per object type.

    Pattern: ``{type_tag}_{sequence:03d}`` — e.g. ``beat_001``,
    ``melody_002``, ``patch_003``.
    """

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}

    def next_name(self, obj_type: str) -> str:
        count = self._counters.get(obj_type, 0) + 1
        self._counters[obj_type] = count
        return f"{obj_type}_{count:03d}"

    def reset(self) -> None:
        self._counters.clear()

    def sync_from_objects(self, objects: list[MDMAObject]) -> None:
        """Update counters to avoid collisions with existing objects."""
        import re
        pattern = re.compile(r"^(.+)_(\d{3,})$")
        for obj in objects:
            m = pattern.match(obj.name)
            if m:
                prefix, num = m.group(1), int(m.group(2))
                if prefix == obj.obj_type:
                    current = self._counters.get(obj.obj_type, 0)
                    if num > current:
                        self._counters[obj.obj_type] = num


# ============================================================================
# OBJECT REGISTRY
# ============================================================================

class ObjectRegistry:
    """Central object store for an MDMA project.

    All windows, CLI commands, and bridge operations interact with the
    same registry instance. The registry owns object lifecycle and
    fires events that subscribers (GUI windows, inspector, etc.) can
    listen to for automatic view updates.
    """

    def __init__(self) -> None:
        self._objects: dict[str, MDMAObject] = {}
        self._subscribers: dict[str, list[Callable[[RegistryEvent], None]]] = {}
        self._namer = _AutoNamer()

    # ---- Core CRUD ----------------------------------------------------------

    def register(self, obj: MDMAObject, auto_name: bool = True) -> str:
        """Add an object to the registry.

        If ``obj.name`` is empty and ``auto_name`` is True, a sequential
        name is generated automatically.

        Returns the object's ID.
        """
        if not obj.id:
            obj.id = _new_id()

        if auto_name and not obj.name:
            obj.name = self._namer.next_name(obj.obj_type)

        # Ensure name uniqueness within type
        if self._name_exists(obj.obj_type, obj.name, exclude_id=obj.id):
            base = obj.name
            suffix = 2
            while self._name_exists(obj.obj_type, f"{base}_{suffix}", exclude_id=obj.id):
                suffix += 1
            obj.name = f"{base}_{suffix}"

        self._objects[obj.id] = obj
        self._fire(RegistryEvent(
            RegistryEventType.OBJECT_CREATED,
            obj.id,
            obj.obj_type,
            obj.name,
        ))
        logger.info("Registered %s: %s (%s)", obj.obj_type, obj.name, obj.id)
        return obj.id

    def get(self, object_id: str) -> Optional[MDMAObject]:
        """Retrieve an object by ID. Returns None if not found."""
        return self._objects.get(object_id)

    def get_by_name(self, name: str, obj_type: Optional[str] = None) -> Optional[MDMAObject]:
        """Retrieve the first object matching the given name.

        If ``obj_type`` is provided, restricts the search to that type.
        """
        for obj in self._objects.values():
            if obj.name == name:
                if obj_type is None or obj.obj_type == obj_type:
                    return obj
        return None

    def list_objects(self, obj_type: Optional[str] = None) -> list[MDMAObject]:
        """List all objects, optionally filtered by type."""
        if obj_type is None:
            return list(self._objects.values())
        return [o for o in self._objects.values() if o.obj_type == obj_type]

    def update(self, object_id: str, updates: dict) -> Optional[MDMAObject]:
        """Apply a dict of field updates to an existing object.

        Calls ``touch()`` to bump the version counter and timestamp.
        Returns the updated object, or None if not found.
        """
        obj = self._objects.get(object_id)
        if obj is None:
            return None

        for key, value in updates.items():
            if hasattr(obj, key) and key not in ("id", "obj_type", "created_at"):
                setattr(obj, key, value)

        obj.touch()
        self._fire(RegistryEvent(
            RegistryEventType.OBJECT_UPDATED,
            obj.id,
            obj.obj_type,
            obj.name,
        ))
        return obj

    def delete(self, object_id: str) -> bool:
        """Remove an object from the registry.

        Returns True if the object was found and deleted.
        """
        obj = self._objects.pop(object_id, None)
        if obj is None:
            return False

        self._fire(RegistryEvent(
            RegistryEventType.OBJECT_DELETED,
            object_id,
            obj.obj_type,
            obj.name,
        ))
        logger.info("Deleted %s: %s (%s)", obj.obj_type, obj.name, object_id)
        return True

    def rename(self, object_id: str, new_name: str) -> Optional[MDMAObject]:
        """Rename an object. Returns the object, or None if not found."""
        obj = self._objects.get(object_id)
        if obj is None:
            return None

        if self._name_exists(obj.obj_type, new_name, exclude_id=object_id):
            raise ValueError(
                f"Name '{new_name}' already in use for type '{obj.obj_type}'"
            )

        old_name = obj.name
        obj.name = new_name
        obj.touch()
        self._fire(RegistryEvent(
            RegistryEventType.OBJECT_RENAMED,
            obj.id,
            obj.obj_type,
            obj.name,
            data={"old_name": old_name},
        ))
        return obj

    # ---- Convenience --------------------------------------------------------

    def duplicate(self, object_id: str, new_name: str = "") -> Optional[MDMAObject]:
        """Create a deep copy of an object with a new ID and name.

        The new object records the source object in ``source_object_ids``.
        Returns the new object, or None if the source was not found.
        """
        source = self._objects.get(object_id)
        if source is None:
            return None

        dup = copy.deepcopy(source)
        dup.id = _new_id()
        dup.name = new_name  # Will be auto-named in register() if empty
        dup.created_at = _now()
        dup.modified_at = _now()
        dup.version = 1
        dup.source_object_ids = [object_id]

        self.register(dup, auto_name=not new_name)
        return dup

    def search(self, query: str, obj_type: Optional[str] = None) -> list[MDMAObject]:
        """Search objects by name or tag substring match."""
        query_lower = query.lower()
        results = []
        for obj in self._objects.values():
            if obj_type and obj.obj_type != obj_type:
                continue
            if query_lower in obj.name.lower():
                results.append(obj)
                continue
            if any(query_lower in tag.lower() for tag in obj.tags):
                results.append(obj)
        return results

    def get_dependents(self, object_id: str) -> list[MDMAObject]:
        """Find all objects that reference the given object ID.

        Checks ``source_object_ids``, ``layers`` (Loop), ``placements``
        (Track), ``track_ids`` (Song), and string-typed ID fields.
        """
        dependents = []
        for obj in self._objects.values():
            if object_id in getattr(obj, "source_object_ids", []):
                dependents.append(obj)
                continue
            # Loop layers
            if hasattr(obj, "layers") and object_id in obj.layers.values():
                dependents.append(obj)
                continue
            # Track placements
            if hasattr(obj, "placements"):
                if any(p.object_id == object_id for p in obj.placements):
                    dependents.append(obj)
                    continue
            # Song track list
            if hasattr(obj, "track_ids") and object_id in obj.track_ids:
                dependents.append(obj)
                continue
            # Patch/Track effect chain reference
            for attr in ("effects_chain_id", "render_source_id",
                         "render_patch_id", "patch_id"):
                if getattr(obj, attr, "") == object_id:
                    dependents.append(obj)
                    break
        return dependents

    # ---- Event system -------------------------------------------------------

    def subscribe(
        self,
        callback: Callable[[RegistryEvent], None],
        event_type: Optional[str] = None,
    ) -> None:
        """Subscribe to registry events.

        If ``event_type`` is None, the callback receives all events.
        """
        key = event_type or "__all__"
        self._subscribers.setdefault(key, []).append(callback)

    def unsubscribe(
        self,
        callback: Callable[[RegistryEvent], None],
        event_type: Optional[str] = None,
    ) -> None:
        """Remove a previously registered callback."""
        key = event_type or "__all__"
        listeners = self._subscribers.get(key, [])
        if callback in listeners:
            listeners.remove(callback)

    def _fire(self, event: RegistryEvent) -> None:
        """Dispatch an event to all matching subscribers."""
        for cb in self._subscribers.get(event.event_type, []):
            try:
                cb(event)
            except Exception:
                logger.exception("Error in registry subscriber for %s", event.event_type)
        for cb in self._subscribers.get("__all__", []):
            try:
                cb(event)
            except Exception:
                logger.exception("Error in registry subscriber (__all__)")

    # ---- Persistence (JSON) -------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise registry contents to a JSON-compatible dict.

        AudioClip ``data`` (numpy arrays) is excluded — audio is stored
        as separate WAV files referenced by object ID.
        """
        result = {}
        for obj_id, obj in self._objects.items():
            d = {}
            for key, value in obj.__dict__.items():
                if key == "data":
                    continue  # Skip numpy audio data
                if isinstance(value, datetime):
                    d[key] = value.isoformat()
                else:
                    d[key] = value
            result[obj_id] = d
        return result

    def from_dict(self, data: dict) -> None:
        """Restore registry contents from a serialised dict.

        Objects are reconstructed using OBJECT_TYPE_MAP. Unknown types
        are stored as base MDMAObject instances.
        """
        self._objects.clear()
        for obj_id, obj_data in data.items():
            obj_type = obj_data.get("obj_type", "base")
            cls = OBJECT_TYPE_MAP.get(obj_type, MDMAObject)
            # Reconstruct — only pass known fields
            init_fields = {}
            for key, value in obj_data.items():
                if hasattr(cls, key) or key in ("id", "name", "obj_type",
                        "created_at", "modified_at", "tags", "source_params",
                        "source_object_ids", "notes", "version", "is_template"):
                    if key in ("created_at", "modified_at") and isinstance(value, str):
                        try:
                            value = datetime.fromisoformat(value)
                        except (ValueError, TypeError):
                            pass
                    init_fields[key] = value
            try:
                obj = cls(**init_fields)
            except TypeError:
                # Fall back to setting attrs after bare init
                obj = cls()
                for k, v in init_fields.items():
                    setattr(obj, k, v)
            obj.id = obj_id
            self._objects[obj_id] = obj

        # Sync auto-namer counters
        self._namer.sync_from_objects(list(self._objects.values()))
        logger.info("Restored %d objects from project data", len(self._objects))

    # ---- Export / Import ----------------------------------------------------

    def export(self, object_id: str, path: str) -> bool:
        """Export a single object to a JSON file.

        AudioClip ``data`` (numpy arrays) is exported as a WAV file at
        ``{path}.wav`` alongside the JSON metadata.  Returns True on
        success.
        """
        obj = self._objects.get(object_id)
        if obj is None:
            return False

        obj_data: dict = {}
        for key, value in obj.__dict__.items():
            if key == "data":
                continue
            if isinstance(value, datetime):
                obj_data[key] = value.isoformat()
            else:
                obj_data[key] = value

        import os
        # Write metadata JSON
        json_path = path if path.endswith('.json') else f"{path}.json"
        with open(json_path, 'w') as f:
            json.dump(obj_data, f, indent=2, default=str)

        # If AudioClip with data, write WAV alongside
        if obj.obj_type == "audio_clip" and getattr(obj, "data", None) is not None:
            try:
                import numpy as np
                import soundfile as sf
                wav_path = json_path.replace('.json', '.wav')
                data = obj.data
                if data.ndim == 1:
                    data = np.column_stack([data, data])
                sf.write(wav_path, data, obj.sample_rate)
                logger.info("Exported audio to %s", wav_path)
            except ImportError:
                logger.warning("soundfile not available — audio data not exported")

        logger.info("Exported %s to %s", obj.name, json_path)
        return True

    def import_file(self, path: str) -> Optional[MDMAObject]:
        """Import an object from a JSON file.

        If a companion WAV file exists alongside the JSON, its audio
        data is loaded into the AudioClip's ``data`` field.

        Returns the imported object, or None on failure.
        """
        import os
        json_path = path if path.endswith('.json') else f"{path}.json"

        if not os.path.exists(json_path):
            logger.error("Import file not found: %s", json_path)
            return None

        with open(json_path, 'r') as f:
            obj_data = json.load(f)

        obj_type = obj_data.get("obj_type", "base")
        cls = OBJECT_TYPE_MAP.get(obj_type, MDMAObject)

        # Reconstruct object
        init_fields = {}
        for key, value in obj_data.items():
            if key in ("created_at", "modified_at") and isinstance(value, str):
                try:
                    value = datetime.fromisoformat(value)
                except (ValueError, TypeError):
                    pass
            init_fields[key] = value

        try:
            obj = cls(**init_fields)
        except TypeError:
            obj = cls()
            for k, v in init_fields.items():
                setattr(obj, k, v)

        # Assign new ID for imported object to avoid collisions
        old_id = obj.id
        obj.id = _new_id()

        # Load companion WAV if AudioClip
        if obj.obj_type == "audio_clip":
            wav_path = json_path.replace('.json', '.wav')
            if os.path.exists(wav_path):
                try:
                    import soundfile as sf
                    data, sr = sf.read(wav_path)
                    obj.data = data
                    obj.sample_rate = sr
                    logger.info("Loaded audio from %s", wav_path)
                except ImportError:
                    logger.warning("soundfile not available — audio data not loaded")

        self.register(obj, auto_name=not obj.name)
        logger.info("Imported %s from %s (old ID: %s → new ID: %s)",
                     obj.name, json_path, old_id[:8], obj.id[:8])
        return obj

    # ---- Stats / introspection ----------------------------------------------

    def count(self, obj_type: Optional[str] = None) -> int:
        """Return the number of registered objects."""
        if obj_type is None:
            return len(self._objects)
        return sum(1 for o in self._objects.values() if o.obj_type == obj_type)

    def types_summary(self) -> dict[str, int]:
        """Return a dict of type -> count for all registered objects."""
        summary: dict[str, int] = {}
        for obj in self._objects.values():
            summary[obj.obj_type] = summary.get(obj.obj_type, 0) + 1
        return summary

    def clear(self) -> None:
        """Remove all objects and reset counters."""
        self._objects.clear()
        self._namer.reset()

    # ---- Internal -----------------------------------------------------------

    def _name_exists(
        self, obj_type: str, name: str, exclude_id: str = ""
    ) -> bool:
        """Check if a name is already in use for the given type."""
        for obj in self._objects.values():
            if obj.obj_type == obj_type and obj.name == name and obj.id != exclude_id:
                return True
        return False
