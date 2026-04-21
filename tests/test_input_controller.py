"""Tests for Phase 7: InputController stack.

Uses a `_FakeController` test double so tests don't need any real
hardware (MIDI device, audio device, keyboard focus). The concrete
controllers have separate smoke tests that import them lazily so
missing optional deps (mido, python-osc) don't break CI.

Run with::

    python -m unittest tests.test_input_controller
"""

from __future__ import annotations

import unittest

from mdma_rebuild.backend.input_controller import (
    Controller,
    ControllerSource,
    HandlerHandle,
    InputChannel,
    InputController,
    InputEvent,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeChannel(InputChannel):
    """Writable continuous channel for tests."""

    def __init__(self, key: str, value: float = 0.0) -> None:
        super().__init__(key)
        self._value = float(value)

    def read(self) -> float:
        return self._value

    def write(self, value: float) -> None:
        self._value = float(value)


class _FakeController(Controller):
    """Emits whatever events the test enqueues, exposes a dict of channels."""

    name = "Fake"

    def __init__(self) -> None:
        self.events: list[InputEvent] = []
        self.start_calls: int = 0
        self.stop_calls: int = 0
        self._channels: dict[str, InputChannel] = {}

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def poll(self):
        out = list(self.events)
        self.events.clear()
        return out

    def channels(self):
        return dict(self._channels)

    def add_channel(self, key: str, value: float = 0.0) -> _FakeChannel:
        ch = _FakeChannel(key, value)
        self._channels[key] = ch
        return ch


# ---------------------------------------------------------------------------
# Registration and lookup
# ---------------------------------------------------------------------------


class TestRegistration(unittest.TestCase):
    def test_add_and_remove_controller(self):
        ic = InputController()
        c = _FakeController()
        ic.add_controller(c, "fake")
        self.assertIn("fake", ic._controllers)
        ic.remove_controller("fake")
        self.assertNotIn("fake", ic._controllers)

    def test_duplicate_alias_raises(self):
        ic = InputController()
        ic.add_controller(_FakeController(), "a")
        with self.assertRaises(ValueError):
            ic.add_controller(_FakeController(), "a")

    def test_remove_unknown_alias_is_noop(self):
        ic = InputController()
        ic.remove_controller("nonexistent")  # no raise


class TestChannelLookup(unittest.TestCase):
    def test_channel_path_resolves(self):
        ic = InputController()
        c = _FakeController()
        ch = c.add_channel("cc_7", value=0.5)
        ic.add_controller(c, "mk1")
        self.assertIs(ic.channel("mk1.cc_7"), ch)

    def test_unknown_alias_raises(self):
        ic = InputController()
        with self.assertRaises(KeyError):
            ic.channel("nope.cc_7")

    def test_unknown_key_raises(self):
        ic = InputController()
        c = _FakeController()
        c.add_channel("cc_7")
        ic.add_controller(c, "mk1")
        with self.assertRaises(KeyError):
            ic.channel("mk1.cc_99")

    def test_malformed_path_raises(self):
        ic = InputController()
        with self.assertRaises(KeyError):
            ic.channel("no_dot")


# ---------------------------------------------------------------------------
# Polling and namespacing
# ---------------------------------------------------------------------------


class TestPolling(unittest.TestCase):
    def test_poll_namespaces_event_keys(self):
        ic = InputController()
        c1 = _FakeController()
        c2 = _FakeController()
        ic.add_controller(c1, "mk1")
        ic.add_controller(c2, "mk2")

        c1.events.append(InputEvent(0.0, "press", "note_60"))
        c2.events.append(InputEvent(0.0, "press", "note_60"))

        events = ic.poll()
        self.assertEqual(
            sorted(e.key for e in events),
            ["mk1.note_60", "mk2.note_60"],
        )

    def test_poll_preserves_value_and_meta(self):
        ic = InputController()
        c = _FakeController()
        c.events.append(
            InputEvent(
                timestamp=0.0,
                kind="press",
                key="note_42",
                value=0.8,
                meta={"channel": 3},
            )
        )
        ic.add_controller(c, "kb")
        events = ic.poll()
        self.assertEqual(events[0].value, 0.8)
        self.assertEqual(events[0].meta, {"channel": 3})

    def test_poll_swallows_controller_exceptions(self):
        class _Angry(Controller):
            def poll(self):
                raise RuntimeError("boom")

        ic = InputController()
        ic.add_controller(_Angry(), "bad")
        ic.add_controller(_FakeController(), "good")
        # Must not propagate the RuntimeError.
        self.assertEqual(ic.poll(), [])


# ---------------------------------------------------------------------------
# Handler registration and dispatch
# ---------------------------------------------------------------------------


class TestHandlers(unittest.TestCase):
    def test_exact_match_fires_handler(self):
        ic = InputController()
        c = _FakeController()
        ic.add_controller(c, "mk1")

        seen: list[str] = []
        ic.on("press", "mk1.note_60", lambda e: seen.append(e.key))

        c.events.append(InputEvent(0.0, "press", "note_60"))
        ic.poll()
        self.assertEqual(seen, ["mk1.note_60"])

    def test_glob_matches_wildcard(self):
        ic = InputController()
        c = _FakeController()
        ic.add_controller(c, "mk1")

        seen: list[str] = []
        ic.on("press", "mk1.note_*", lambda e: seen.append(e.key))

        c.events.append(InputEvent(0.0, "press", "note_60"))
        c.events.append(InputEvent(0.0, "press", "note_72"))
        c.events.append(InputEvent(0.0, "press", "other"))
        ic.poll()
        self.assertEqual(
            sorted(seen), ["mk1.note_60", "mk1.note_72"]
        )

    def test_kind_filter(self):
        ic = InputController()
        c = _FakeController()
        ic.add_controller(c, "mk1")

        seen: list[str] = []
        ic.on("press", "*", lambda e: seen.append(e.key))

        c.events.append(InputEvent(0.0, "press", "note_60"))
        c.events.append(InputEvent(0.0, "release", "note_60"))
        ic.poll()
        self.assertEqual(seen, ["mk1.note_60"])

    def test_handler_exception_does_not_kill_loop(self):
        ic = InputController()
        c = _FakeController()
        ic.add_controller(c, "mk1")

        good_seen: list[str] = []

        def angry(_):
            raise RuntimeError("handler fail")

        ic.on("press", "mk1.note_60", angry)
        ic.on("press", "mk1.note_60", lambda e: good_seen.append(e.key))

        c.events.append(InputEvent(0.0, "press", "note_60"))
        ic.poll()
        # The other handler still ran.
        self.assertEqual(good_seen, ["mk1.note_60"])

    def test_handler_remove(self):
        ic = InputController()
        c = _FakeController()
        ic.add_controller(c, "mk1")

        seen: list[str] = []
        handle = ic.on("press", "mk1.note_60", lambda e: seen.append(e.key))
        handle.remove()
        handle.remove()  # idempotent

        c.events.append(InputEvent(0.0, "press", "note_60"))
        ic.poll()
        self.assertEqual(seen, [])

    def test_handler_registered_before_controller(self):
        """Handlers can be registered before any matching controller is
        added — the registration sticks and fires once events arrive."""
        ic = InputController()
        seen: list[str] = []
        ic.on("press", "mk1.note_60", lambda e: seen.append(e.key))

        c = _FakeController()
        ic.add_controller(c, "mk1")
        c.events.append(InputEvent(0.0, "press", "note_60"))
        ic.poll()
        self.assertEqual(seen, ["mk1.note_60"])


# ---------------------------------------------------------------------------
# ControllerSource
# ---------------------------------------------------------------------------


class TestControllerSource(unittest.TestCase):
    def test_reads_current_channel_value(self):
        ic = InputController()
        c = _FakeController()
        ch = c.add_channel("cc_7", 0.25)
        ic.add_controller(c, "mk1")

        src = ControllerSource(ic, "mk1.cc_7")
        self.assertEqual(src.value_at(0.0), 0.25)

        ch.write(0.75)
        self.assertEqual(src.value_at(100.0), 0.75)

    def test_unknown_path_raises_at_construction(self):
        ic = InputController()
        ic.add_controller(_FakeController(), "mk1")
        with self.assertRaises(KeyError):
            ControllerSource(ic, "mk1.missing")

    def test_channel_read_failure_returns_zero(self):
        class _BadChannel(InputChannel):
            def read(self):
                raise RuntimeError("broken")

        ic = InputController()
        c = _FakeController()
        c._channels["bad"] = _BadChannel("bad")
        ic.add_controller(c, "mk1")
        src = ControllerSource(ic, "mk1.bad")
        self.assertEqual(src.value_at(0.0), 0.0)


# ---------------------------------------------------------------------------
# Start / stop lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle(unittest.TestCase):
    def test_start_propagates_to_controllers(self):
        ic = InputController()
        c = _FakeController()
        ic.add_controller(c, "mk1")
        ic.start()
        try:
            self.assertEqual(c.start_calls, 1)
        finally:
            ic.stop()
        self.assertEqual(c.stop_calls, 1)

    def test_add_controller_while_running_starts_it(self):
        ic = InputController()
        ic.start()
        try:
            c = _FakeController()
            ic.add_controller(c, "late")
            self.assertEqual(c.start_calls, 1)
        finally:
            ic.stop()

    def test_start_is_idempotent(self):
        ic = InputController()
        c = _FakeController()
        ic.add_controller(c, "mk1")
        ic.start()
        ic.start()  # second start should be a no-op
        try:
            self.assertEqual(c.start_calls, 1)
        finally:
            ic.stop()


# ---------------------------------------------------------------------------
# Integration with command router (handler -> scheduler or router)
# ---------------------------------------------------------------------------


class TestRouterIntegrationExample(unittest.TestCase):
    """A MIDI pad press fires a slash command through the router.

    This is the handler path described in the spec's "Integration
    with Session and the command router" section. We use a stub
    router here so the test is hermetic.
    """

    def test_handler_invokes_router(self):
        ic = InputController()
        c = _FakeController()
        ic.add_controller(c, "mk1")

        router_calls: list[tuple] = []

        def fake_dispatch(session, cmd, args):
            router_calls.append((cmd, args))

        session = object()

        def pad_handler(_event):
            fake_dispatch(session, "beat", ["trap", "4"])

        ic.on("press", "mk1.note_36", pad_handler)

        c.events.append(InputEvent(0.0, "press", "note_36"))
        ic.poll()
        self.assertEqual(router_calls, [("beat", ["trap", "4"])])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
