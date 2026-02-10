"""Envelope processing for the MDMA rebuild.

This module defines a simple attack/decay/sustain/release (ADSR)
envelope class that can be applied to one‑dimensional audio
buffers.  Durations are specified in seconds and the envelope is
computed at instantiation time each time it is applied.
"""

from __future__ import annotations

import numpy as np  # type: ignore


class ADSREnvelope:
    """Attack‑Decay‑Sustain‑Release envelope.

    Parameters are floats expressed in seconds for attack, decay and
    release and in the range [0, 1] for sustain.  The envelope
    generates a curve that ramps from 0 to 1 during the attack
    period, decays to the sustain level, holds that level, and then
    linearly releases to zero.
    """

    def __init__(self, attack: float = 0.01, decay: float = 0.1, sustain: float = 0.8,
                 release: float = 0.1, sample_rate: int = 48_000) -> None:
        self.attack = max(0.0, float(attack))
        self.decay = max(0.0, float(decay))
        self.sustain = max(0.0, min(1.0, float(sustain)))
        self.release = max(0.0, float(release))
        self.sample_rate = sample_rate

    def apply(self, buffer: np.ndarray) -> np.ndarray:
        n = len(buffer)
        # Convert times to samples
        a = int(self.attack * self.sample_rate)
        d = int(self.decay * self.sample_rate)
        r = int(self.release * self.sample_rate)
        # Sustain duration fills the remainder
        s = max(n - (a + d + r), 0)
        env = np.ones(n, dtype=np.float64)
        pos = 0
        # Attack segment: ramp 0 -> 1
        if a > 0 and pos < n:
            env[pos:pos + a] = np.linspace(0.0, 1.0, a, endpoint=False)
            pos += a
        # Decay segment: 1 -> sustain level
        if d > 0 and pos < n:
            env[pos:pos + d] = np.linspace(1.0, self.sustain, d, endpoint=False)
            pos += d
        # Sustain segment
        if s > 0 and pos < n:
            env[pos:pos + s] = self.sustain
            pos += s
        # Release segment: sustain level -> 0
        rem = n - pos
        if rem > 0:
            env[pos:] = np.linspace(self.sustain, 0.0, rem, endpoint=True)
        return (buffer.astype(np.float64) * env).astype(np.float64)