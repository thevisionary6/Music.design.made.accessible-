"""Core functionality for the MDMA rebuild.

This subpackage contains the Session class, which maintains state for
projects, operators, filters, envelopes and playback.  It also
provides minimal support for tone generation and writing WAV files.
"""

from .session import Session  # noqa: F401