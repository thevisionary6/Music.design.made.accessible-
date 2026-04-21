"""Tests for the V2 backend slash commands (``/patn`` and siblings)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from mdma_rebuild.commands.backend_cmds import cmd_patn


class _FakeSession:
    """Minimal session stand-in for the command-layer tests."""

    def __init__(self) -> None:
        self.played: list = []
        self._raise_on_graph: Exception | None = None

    def backend_graph(self):
        if self._raise_on_graph is not None:
            raise self._raise_on_graph
        return object()

    def play_pattern(self, pattern, shape):
        self.played.append((pattern, shape))


class TestPatnCommand(unittest.TestCase):
    def test_no_args_prints_usage(self):
        session = _FakeSession()
        out = cmd_patn(session, [])
        self.assertIn("Usage", out)
        self.assertEqual(session.played, [])

    def test_parses_and_plays(self):
        session = _FakeSession()
        out = cmd_patn(session, ["60,0.25", "64,0.25", "67,0.5"])
        self.assertTrue(out.startswith("OK"))
        self.assertEqual(len(session.played), 1)
        pattern, _shape = session.played[0]
        self.assertEqual(
            pattern.events,
            [(60.0, 0.25), (64.0, 0.25), (67.0, 0.5)],
        )

    def test_malformed_token_returns_error(self):
        session = _FakeSession()
        out = cmd_patn(session, ["oops"])
        self.assertTrue(out.startswith("ERROR"))
        self.assertEqual(session.played, [])

    def test_non_numeric_note_returns_error(self):
        session = _FakeSession()
        out = cmd_patn(session, ["x,0.25"])
        self.assertTrue(out.startswith("ERROR"))

    def test_non_positive_duration_returns_error(self):
        session = _FakeSession()
        out = cmd_patn(session, ["60,0"])
        self.assertTrue(out.startswith("ERROR"))
        out = cmd_patn(session, ["60,-0.1"])
        self.assertTrue(out.startswith("ERROR"))

    def test_missing_signalflow_is_clear_message(self):
        session = _FakeSession()
        session._raise_on_graph = ImportError("no signalflow")
        out = cmd_patn(session, ["60,0.25"])
        self.assertIn("signalflow", out.lower())

    def test_generic_graph_failure_is_surfaced(self):
        session = _FakeSession()
        session._raise_on_graph = RuntimeError("no audio device")
        out = cmd_patn(session, ["60,0.25"])
        self.assertIn("backend graph unavailable", out)

    def test_command_is_discovered_by_router(self):
        from mdma_rebuild.commands import router

        table = router.build_command_table()
        self.assertIn("patn", table)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
