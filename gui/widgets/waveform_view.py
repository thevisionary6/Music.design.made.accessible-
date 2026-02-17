"""Text-Representable Waveform Display Widget.

Displays audio waveform data in a format accessible to both sighted
users (visual rendering) and screen reader users (text-based summary).

The text representation includes: duration, peak level, RMS level,
and a simple ASCII amplitude graph.  No information is conveyed by
visual shape alone.

BUILD ID: widget_waveform_view_v1.0
"""

from __future__ import annotations

from typing import Any, Optional

try:
    import wx
    _WX_AVAILABLE = True
except ImportError:
    _WX_AVAILABLE = False

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False


if _WX_AVAILABLE:

    class WaveformView(wx.Panel):
        """Waveform display with text-accessible representation.

        When audio data is loaded, generates both a visual overview
        and a text summary.  Screen readers see the text summary via
        the accessible name/description.

        Parameters:
            parent: Parent wx window.
            sample_rate: Sample rate of audio data.
        """

        def __init__(
            self,
            parent: wx.Window,
            sample_rate: int = 48000,
            **kwargs: Any,
        ) -> None:
            super().__init__(parent, **kwargs)

            self._sample_rate = sample_rate
            self._data: Optional[Any] = None  # np.ndarray or None

            sizer = wx.BoxSizer(wx.VERTICAL)

            # Text summary (always visible, screen reader accessible)
            self._summary = wx.StaticText(
                self, label="No audio loaded",
                name="Waveform Summary",
            )
            sizer.Add(self._summary, 0, wx.ALL | wx.EXPAND, 5)

            # ASCII waveform (text-based, accessible)
            self._ascii_display = wx.TextCtrl(
                self,
                style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL,
                size=(-1, 80),
                name="Waveform ASCII Display",
            )
            sizer.Add(self._ascii_display, 1, wx.EXPAND | wx.ALL, 2)

            self.SetSizer(sizer)
            self.SetName("Waveform View")

        def set_audio(self, data: Any, sample_rate: int = 0) -> None:
            """Load audio data and update the display.

            Parameters:
                data: Numpy array of audio samples (mono or stereo).
                sample_rate: Sample rate (0 = use default).
            """
            if not _NP_AVAILABLE:
                self._summary.SetLabel("numpy not available")
                return

            self._data = data
            if sample_rate > 0:
                self._sample_rate = sample_rate

            if data is None or len(data) == 0:
                self._summary.SetLabel("No audio loaded")
                self._ascii_display.SetValue("")
                return

            # Compute summary statistics
            if data.ndim == 2:
                mono = np.mean(data, axis=1)
            else:
                mono = data

            duration = len(mono) / self._sample_rate
            peak = float(np.max(np.abs(mono)))
            rms = float(np.sqrt(np.mean(mono ** 2)))
            peak_db = 20 * np.log10(peak) if peak > 0 else -120.0
            rms_db = 20 * np.log10(rms) if rms > 0 else -120.0

            summary = (
                f"Duration: {duration:.2f}s | "
                f"Peak: {peak_db:.1f} dB | "
                f"RMS: {rms_db:.1f} dB | "
                f"Samples: {len(mono)} | "
                f"Rate: {self._sample_rate} Hz"
            )
            self._summary.SetLabel(summary)
            self._summary.SetName(f"Waveform: {summary}")

            # Generate ASCII waveform (60 columns)
            ascii_wave = self._generate_ascii(mono, width=60, height=8)
            self._ascii_display.SetValue(ascii_wave)

        def _generate_ascii(
            self, mono: Any, width: int = 60, height: int = 8
        ) -> str:
            """Generate an ASCII representation of the waveform."""
            if not _NP_AVAILABLE or mono is None or len(mono) == 0:
                return ""

            # Downsample to fit width
            chunk_size = max(1, len(mono) // width)
            peaks = []
            for i in range(0, len(mono), chunk_size):
                chunk = mono[i:i + chunk_size]
                peaks.append(float(np.max(np.abs(chunk))))

            if not peaks:
                return ""

            max_peak = max(peaks) if max(peaks) > 0 else 1.0

            # Build ASCII grid
            lines = []
            for row in range(height):
                threshold = (height - row) / height
                line = ""
                for p in peaks[:width]:
                    normalised = p / max_peak
                    if normalised >= threshold:
                        line += "#"
                    else:
                        line += " "
                lines.append(line.rstrip())

            return "\n".join(lines)

        def clear(self) -> None:
            """Clear the display."""
            self._data = None
            self._summary.SetLabel("No audio loaded")
            self._ascii_display.SetValue("")

else:

    class WaveformView:  # type: ignore[no-redef]
        """Stub when wx is not available."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("wxPython required")
