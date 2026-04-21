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
from .automation import (
    AutomationSource,
    BindingHandle,
    Constant,
    Envelope,
    LFO,
    Ramp,
)
from .render import render_pattern, render_schedule
from .par import PAR, PARLoopHandle, PARScheduleHandle, PARScheduler
from .input_controller import (
    Controller,
    ControllerSource,
    HandlerHandle,
    InputChannel,
    InputController,
    InputEvent,
)
from . import effects

__all__ = [
    "Pattern",
    "Clock",
    "Scheduler",
    "ScheduleHandle",
    "LoopHandle",
    "AutomationSource",
    "BindingHandle",
    "Constant",
    "LFO",
    "Ramp",
    "Envelope",
    "render_pattern",
    "render_schedule",
    "PAR",
    "PARScheduler",
    "PARScheduleHandle",
    "PARLoopHandle",
    "InputController",
    "Controller",
    "ControllerSource",
    "HandlerHandle",
    "InputChannel",
    "InputEvent",
]
