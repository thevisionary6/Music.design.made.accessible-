"""MDMA GUI Panels.

Panels are the internal building blocks of sub-windows. Each panel
is a self-contained wx.Panel subclass that owns a specific set of
controls and communicates with the engine exclusively through the
Bridge.

Panel sub-packages mirror the window structure:
- generation/   — Beat, melody, loop, theory panels
- mutation/     — Transform, adapt, pattern editor, splice panels
- effects/      — Effect browser, chain builder, convolution, param inspector
- synthesis/    — Operator, waveform, envelope, modulation, presets
- arrangement/  — Track list, pattern lane, song settings
- mixing/       — Deck, crossfader, master, stem panels
- inspector/    — Object tree, parameter inspector, console
"""
