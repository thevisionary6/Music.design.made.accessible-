"""MDMA backend V2 package.

Hosts the SignalFlow-native Pattern / Clock / Scheduler / PAR /
PARScheduler / InputController stack defined by ``Backend Docs V2``.
Lives alongside the existing numpy-buffer DSP layer; migration is
additive, not a rewrite.

Phase 3 exports: :class:`Pattern`, :class:`Clock`, :class:`Scheduler`,
:class:`ScheduleHandle`, :class:`LoopHandle`. Automation, PAR, render
helpers, and the input controller arrive in Phase 4+.
"""

from .pattern import Pattern
from .clock import Clock
from .scheduler import LoopHandle, ScheduleHandle, Scheduler

__all__ = [
    "Pattern",
    "Clock",
    "Scheduler",
    "ScheduleHandle",
    "LoopHandle",
]
