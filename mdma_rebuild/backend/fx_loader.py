"""Drop-in loader for custom SignalFlow effect functions.

Extension model: a user puts Python files in a directory of their
choosing (``~/.mdma/effects/`` is the default) and runs ``/loadfx``
or calls :func:`load_from` programmatically. Every top-level
callable that matches the spec's DSP signature::

    def my_effect(input_node, **params) -> Node:
        ...

gets registered under its function name on
:attr:`Session.custom_effects` (a dict on the session object).

**Registration semantics**

- ``Session.custom_effects`` is created lazily on first load.
- Re-loading a file overwrites whatever names it previously defined.
- Private helpers (names starting with ``_``) are ignored.
- Functions are validated by signature — the first positional
  parameter must be named ``input_node``. Anything that doesn't
  match is skipped (with a report so users can see what was
  rejected).

**Why a separate module, not just ``importlib.import_module``**

- Files can live outside ``sys.path`` so users don't have to fiddle
  with PYTHONPATH.
- Re-loading an already-loaded file works cleanly (``importlib.reload``
  requires the module to already be in ``sys.modules`` under the
  same name; we avoid that wart).
- The Session session is the source of truth for which effects are
  "active" — a failed import leaves existing registrations alone.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List


DEFAULT_USER_DIR = Path.home() / ".mdma" / "effects"


@dataclass
class LoadResult:
    """Per-file outcome from :func:`load_from`."""

    path: str
    loaded: List[str] = field(default_factory=list)
    skipped: List[tuple] = field(default_factory=list)  # (name, reason)
    error: str = ""  # non-empty if the file itself failed to import


def _is_candidate(name: str, obj) -> tuple[bool, str]:
    """Return ``(is_callable_effect, reason_if_not)``."""
    if name.startswith("_"):
        return False, "private name"
    if not callable(obj):
        return False, "not callable"
    if inspect.isclass(obj):
        return False, "is a class"
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return False, "couldn't inspect signature"
    params = list(sig.parameters.values())
    if not params:
        return False, "takes no args"
    if params[0].name != "input_node":
        return False, f"first param is {params[0].name!r}, want 'input_node'"
    return True, ""


def _import_file(path: Path):
    """Import a Python file as a throwaway module, without polluting
    ``sys.modules`` permanently. Returns the module object."""
    module_name = f"_mdma_fx_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Register briefly so ``from __future__`` etc. resolve, then
    # remove to avoid stale references between reloads.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        sys.modules.pop(module_name, None)
    return module


def _iter_files(path: Path) -> Iterable[Path]:
    """Yield every ``*.py`` file at ``path`` (or the file itself if
    ``path`` is a .py file). Hidden files and ``__init__.py`` are
    skipped."""
    if path.is_file():
        if path.suffix == ".py":
            yield path
        return
    if not path.is_dir():
        return
    for entry in sorted(path.iterdir()):
        if entry.name.startswith(".") or entry.name == "__init__.py":
            continue
        if entry.is_file() and entry.suffix == ".py":
            yield entry


def load_from(
    registry: Dict[str, Callable],
    path: str | os.PathLike,
) -> List[LoadResult]:
    """Load every effect callable in ``path`` into ``registry``.

    ``path`` may point to one file or a directory; directories are
    walked one level deep (non-recursive) so a user's effects folder
    stays flat and predictable.

    ``registry`` is typically ``session.custom_effects``. Returns a
    list of :class:`LoadResult` records — one per file — so callers
    can surface a summary.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return [LoadResult(path=str(p), error=f"path does not exist: {p}")]

    results: list[LoadResult] = []
    for file in _iter_files(p):
        result = LoadResult(path=str(file))
        try:
            module = _import_file(file)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            results.append(result)
            continue
        for name in dir(module):
            obj = getattr(module, name)
            ok, reason = _is_candidate(name, obj)
            if ok:
                registry[name] = obj
                result.loaded.append(name)
            elif not name.startswith("_") and callable(obj) and not inspect.ismodule(obj):
                # Only report near-misses; don't spam about every
                # integer, string, or imported module.
                result.skipped.append((name, reason))
        results.append(result)
    return results


def format_summary(results: List[LoadResult]) -> str:
    """Render a :func:`load_from` result list as a user-readable string."""
    lines: list[str] = []
    total_loaded = 0
    for r in results:
        if r.error:
            lines.append(f"  {r.path}: ERROR — {r.error}")
            continue
        total_loaded += len(r.loaded)
        if r.loaded:
            lines.append(
                f"  {r.path}: loaded {len(r.loaded)} — " + ", ".join(r.loaded)
            )
        else:
            lines.append(f"  {r.path}: no matching callables found")
        for name, reason in r.skipped:
            lines.append(f"      skipped {name} ({reason})")
    header = f"Loaded {total_loaded} effect(s)"
    return header + "\n" + "\n".join(lines) if lines else header


__all__ = [
    "DEFAULT_USER_DIR",
    "LoadResult",
    "format_summary",
    "load_from",
]
