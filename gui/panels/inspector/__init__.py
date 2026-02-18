"""Inspector window panels.

Panels:
- ObjectTreePanel — hierarchical registry browser
- ConsolePanel — CLI mirror with direct command input
- ParameterInspectorPanel — detailed object parameter view
"""

from .object_tree import ObjectTreePanel
from .console_panel import ConsolePanel
from .parameter_inspector import ParameterInspectorPanel

__all__ = [
    'ObjectTreePanel',
    'ConsolePanel',
    'ParameterInspectorPanel',
]
