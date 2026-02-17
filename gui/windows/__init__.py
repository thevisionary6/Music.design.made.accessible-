"""MDMA GUI Sub-Windows.

Each module defines a wx.Frame subclass representing one workflow
window as specified in GUI_WINDOW_ARCHITECTURE_SPEC.md:

- generation_window: Create new objects (beats, melodies, loops)
- mutation_window: Transform existing objects
- effects_window: Apply signal processing
- synthesis_window: Sound design / patch editing
- arrangement_window: Track and song assembly
- mixing_window: DJ decks and master output
- inspector_window: Object tree, console, status (always available)
"""
