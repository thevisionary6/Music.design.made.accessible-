"""Tests for the drop-in custom-effect loader + /loadfx + /listfx."""

from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path

from mdma_rebuild.backend.fx_loader import (
    LoadResult,
    format_summary,
    load_from,
)
from mdma_rebuild.commands.backend_cmds import cmd_listfx, cmd_loadfx


class _FakeSession:
    pass


def _write(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body))


# ---------------------------------------------------------------------------
# load_from — the Python-level API
# ---------------------------------------------------------------------------


class TestLoadFrom(unittest.TestCase):
    def test_loads_matching_callables(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "my_fx.py").write_text(textwrap.dedent("""
                def boost(input_node, gain=2.0):
                    return input_node * gain

                def weird(input_node, **params):
                    return input_node
            """))
            registry: dict = {}
            results = load_from(registry, d)
            self.assertIn("boost", registry)
            self.assertIn("weird", registry)
            self.assertEqual(len(results), 1)
            self.assertEqual(sorted(results[0].loaded), ["boost", "weird"])

    def test_skips_private_and_misnamed(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "x.py").write_text(textwrap.dedent("""
                def _private(input_node):
                    return input_node

                def bad(node):
                    # first param isn't named 'input_node'
                    return node

                def good(input_node):
                    return input_node
            """))
            registry: dict = {}
            results = load_from(registry, d)
            self.assertIn("good", registry)
            self.assertNotIn("_private", registry)
            self.assertNotIn("bad", registry)
            skipped_names = [name for name, _ in results[0].skipped]
            self.assertIn("bad", skipped_names)

    def test_single_file_path_works(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "one.py"
            f.write_text("def only(input_node):\n    return input_node\n")
            registry: dict = {}
            results = load_from(registry, f)
            self.assertEqual(results[0].loaded, ["only"])

    def test_import_error_reported_per_file(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "broken.py").write_text("this is not python !!\n")
            (Path(d) / "ok.py").write_text(
                "def works(input_node):\n    return input_node\n"
            )
            registry: dict = {}
            results = load_from(registry, d)
            errors = [r for r in results if r.error]
            self.assertEqual(len(errors), 1)
            self.assertIn("broken.py", errors[0].path)
            # The healthy file still loaded.
            self.assertIn("works", registry)

    def test_nonexistent_path_reports_error(self):
        registry: dict = {}
        results = load_from(registry, "/definitely/does/not/exist")
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].error.startswith("path does not exist"))

    def test_reload_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "rel.py"
            f.write_text(
                "def rel(input_node):\n    return input_node  # v1\n"
            )
            registry: dict = {}
            load_from(registry, f)
            v1 = registry["rel"]
            # Rewrite file, reload — the cached sys.modules entry must
            # not leak from the previous load.
            f.write_text(
                "def rel(input_node):\n    return input_node  # v2\n"
            )
            load_from(registry, f)
            v2 = registry["rel"]
            self.assertIsNot(v1, v2)

    def test_init_py_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "__init__.py").write_text(
                "def pkg_init(input_node):\n    return input_node\n"
            )
            (Path(d) / "real.py").write_text(
                "def real(input_node):\n    return input_node\n"
            )
            registry: dict = {}
            load_from(registry, d)
            self.assertIn("real", registry)
            self.assertNotIn("pkg_init", registry)


class TestFormatSummary(unittest.TestCase):
    def test_renders_loaded_and_errors(self):
        results = [
            LoadResult(path="/tmp/a.py", loaded=["one", "two"]),
            LoadResult(path="/tmp/b.py", error="SyntaxError: bad"),
        ]
        text = format_summary(results)
        self.assertIn("Loaded 2 effect(s)", text)
        self.assertIn("one, two", text)
        self.assertIn("ERROR", text)


# ---------------------------------------------------------------------------
# /loadfx and /listfx commands
# ---------------------------------------------------------------------------


class TestLoadfxCommand(unittest.TestCase):
    def test_help_flag(self):
        session = _FakeSession()
        self.assertIn("Usage", cmd_loadfx(session, ["--help"]))

    def test_loads_into_session_registry(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "fx.py").write_text(
                "def boost(input_node, gain=3.0):\n"
                "    return input_node * gain\n"
            )
            session = _FakeSession()
            out = cmd_loadfx(session, [d])
            self.assertIn("boost", out)
            self.assertIn("boost", session.custom_effects)

    def test_missing_path_is_surfaced_cleanly(self):
        session = _FakeSession()
        out = cmd_loadfx(session, ["/nope/nope"])
        self.assertIn("does not exist", out)


class TestListfxCommand(unittest.TestCase):
    def test_empty_registry(self):
        session = _FakeSession()
        self.assertIn("No custom effects", cmd_listfx(session, []))

    def test_lists_loaded(self):
        session = _FakeSession()
        session.custom_effects = {
            "boost": lambda input_node, gain=2: input_node * gain,
        }
        session.custom_effects["boost"].__doc__ = "Gain booster."
        out = cmd_listfx(session, [])
        self.assertIn("boost", out)
        self.assertIn("Gain booster", out)

    def test_router_registers_loadfx_and_listfx(self):
        from mdma_rebuild.commands.router import build_command_table
        table = build_command_table()
        self.assertIn("loadfx", table)
        self.assertIn("listfx", table)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
