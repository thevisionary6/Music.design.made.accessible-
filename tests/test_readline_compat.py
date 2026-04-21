"""Tests for the cross-platform readline shim.

The shim has to behave sensibly on three hosts:

1. Linux / macOS with stdlib readline — the common case.
2. Windows with pyreadline3 installed — the supported NVDA path.
3. Anything without readline at all (misconfigured Linux, stripped
   macOS, Windows without pyreadline3) — the REPL must still run,
   just without history / completion / bindings.

We can't actually import pyreadline3 here (its console backend
refuses to load on non-Windows), so the pyreadline3 branch is
exercised via a synthetic stub injected into ``sys.modules``.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest


def _reload_compat():
    """Reload the shim so it picks up whatever is in ``sys.modules``."""
    sys.modules.pop("mdma_rebuild.core.readline_compat", None)
    return importlib.import_module("mdma_rebuild.core.readline_compat")


class TestStdlibBranch(unittest.TestCase):
    """On a host with stdlib readline, the shim must prefer it."""

    def test_picks_stdlib_readline(self):
        if sys.platform == "win32":  # pragma: no cover - CI matrix
            self.skipTest("Windows path exercised in TestPyreadlineBranch")
        compat = _reload_compat()
        self.assertTrue(compat.HAS_READLINE)
        self.assertFalse(compat.IS_PYREADLINE3)
        # Our stdlib branch sets IS_GNU OR IS_LIBEDIT, never both.
        self.assertTrue(compat.IS_GNU ^ compat.IS_LIBEDIT)

    def test_parse_and_bind_returns_bool(self):
        compat = _reload_compat()
        self.assertIsInstance(compat.parse_and_bind("tab: complete"), bool)

    def test_platform_summary_mentions_backend(self):
        compat = _reload_compat()
        summary = compat.platform_summary()
        self.assertIn("readline", summary)


class TestPyreadlineBranch(unittest.TestCase):
    """Force the stdlib-readline import to fail and inject a stub
    pyreadline3 so the shim falls back to the Windows path."""

    def setUp(self):
        self._prev_readline = sys.modules.get("readline")
        self._prev_pyreadline3 = sys.modules.get("pyreadline3")

        # Block stdlib readline for the duration of this test.
        class _ReadlineBlocker:
            def find_spec(self, name, path=None, target=None):
                if name == "readline":
                    raise ImportError("blocked in test")
                return None

        self._blocker = _ReadlineBlocker()
        sys.meta_path.insert(0, self._blocker)
        sys.modules.pop("readline", None)

        # Inject a pyreadline3 stub.
        fake = types.ModuleType("pyreadline3")
        fake.parsed: list[str] = []  # type: ignore[attr-defined]

        def _pab(spec):
            fake.parsed.append(spec)  # type: ignore[attr-defined]

        fake.parse_and_bind = _pab
        fake.set_completer = lambda _: None
        fake.set_completer_delims = lambda _: None
        fake.read_history_file = lambda _p: None
        fake.write_history_file = lambda _p: None
        fake.set_history_length = lambda _n: None
        sys.modules["pyreadline3"] = fake

    def tearDown(self):
        try:
            sys.meta_path.remove(self._blocker)
        except ValueError:
            pass
        if self._prev_readline is None:
            sys.modules.pop("readline", None)
        else:
            sys.modules["readline"] = self._prev_readline
        if self._prev_pyreadline3 is None:
            sys.modules.pop("pyreadline3", None)
        else:
            sys.modules["pyreadline3"] = self._prev_pyreadline3
        sys.modules.pop("mdma_rebuild.core.readline_compat", None)

    def test_falls_back_to_pyreadline3(self):
        compat = _reload_compat()
        self.assertTrue(compat.HAS_READLINE)
        self.assertTrue(compat.IS_PYREADLINE3)
        self.assertFalse(compat.IS_LIBEDIT)
        self.assertFalse(compat.IS_GNU)

    def test_parse_and_bind_goes_through_pyreadline3(self):
        compat = _reload_compat()
        self.assertTrue(compat.parse_and_bind('"\\C-k": kill-line'))
        stub = sys.modules["pyreadline3"]
        self.assertIn('"\\C-k": kill-line', stub.parsed)

    def test_summary_mentions_pyreadline3(self):
        compat = _reload_compat()
        self.assertIn("pyreadline3", compat.platform_summary())


class TestNoReadlineBranch(unittest.TestCase):
    """Block both stdlib readline and pyreadline3 so the shim lands on
    the None branch. The REPL still has to function without crashing.
    """

    def setUp(self):
        self._prev_readline = sys.modules.get("readline")
        self._prev_pyreadline3 = sys.modules.get("pyreadline3")

        class _BothBlocker:
            def find_spec(self, name, path=None, target=None):
                if name in ("readline", "pyreadline3"):
                    raise ImportError("blocked in test")
                return None

        self._blocker = _BothBlocker()
        sys.meta_path.insert(0, self._blocker)
        sys.modules.pop("readline", None)
        sys.modules.pop("pyreadline3", None)

    def tearDown(self):
        try:
            sys.meta_path.remove(self._blocker)
        except ValueError:
            pass
        if self._prev_readline is None:
            sys.modules.pop("readline", None)
        else:
            sys.modules["readline"] = self._prev_readline
        if self._prev_pyreadline3 is None:
            sys.modules.pop("pyreadline3", None)
        else:
            sys.modules["pyreadline3"] = self._prev_pyreadline3
        sys.modules.pop("mdma_rebuild.core.readline_compat", None)

    def test_degrades_gracefully(self):
        compat = _reload_compat()
        self.assertFalse(compat.HAS_READLINE)
        self.assertIsNone(compat.readline)
        # parse_and_bind returns False rather than raising, so
        # bmdma can keep walking through the binding list.
        self.assertFalse(compat.parse_and_bind("tab: complete"))
        self.assertIn("not available", compat.platform_summary())


class TestBmdmaImport(unittest.TestCase):
    """bmdma imports the shim at module scope; make sure the
    combination of the shim + the existing command-table loader
    still produces a working command table on this host."""

    def test_bmdma_still_builds_command_table(self):
        import bmdma

        table = bmdma.build_command_table()
        self.assertGreater(len(table), 500)
        # Cross-check one known V2 command is registered.
        self.assertIn("patn", table)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
