"""MDMA backend V2 package.

Hosts the SignalFlow-native Pattern / Clock / Scheduler / PAR /
PARScheduler / InputController stack defined by ``Backend Docs V2``.
Lives alongside the existing numpy-buffer DSP layer; migration is
additive, not a rewrite.

Phase 1 exports only the :class:`Pattern` class. Subsequent phases
append Clock, Scheduler, render helpers, PAR support, and the input
controller.
"""

from .pattern import Pattern

__all__ = ["Pattern"]
