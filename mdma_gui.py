#!/usr/bin/env python3
"""
MDMA GUI - Action Panel Client MVP
==================================

A thin wxPython client over the MDMA CLI entrypoint.
Maps GUI actions to existing commands and shows results immediately.

Version: 0.2.0
Author: Based on spec by Cyrus
Date: 2026-02-03

Requirements:
    pip install wxPython

Usage:
    python mdma_gui.py

BUILD ID: mdma_gui_v0.2.0
"""

import wx
import wx.lib.agw.aui as aui
import sys
import os
import io
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

# Add MDMA to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# THEME / COLORS (Serum-ish muted dark)
# ============================================================================

class Theme:
    """Muted dark theme inspired by Serum."""
    BG_DARK = wx.Colour(30, 30, 35)
    BG_PANEL = wx.Colour(40, 40, 48)
    BG_INPUT = wx.Colour(50, 50, 58)
    FG_TEXT = wx.Colour(220, 220, 225)
    FG_DIM = wx.Colour(180, 180, 190)  # Raised from (140,140,150) for WCAG AA contrast
    ACCENT = wx.Colour(100, 180, 255)
    SUCCESS = wx.Colour(100, 200, 120)
    ERROR = wx.Colour(255, 100, 100)
    WARNING = wx.Colour(255, 200, 100)


# ============================================================================
# ACTION DEFINITIONS
# ============================================================================

@dataclass
class ActionParam:
    """Definition of a parameter for an action."""
    name: str
    label: str
    param_type: str  # 'int', 'float', 'bool', 'enum', 'string', 'file'
    default: Any = None
    choices: List[str] = field(default_factory=list)  # For enum type
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    

@dataclass 
class ActionDef:
    """Definition of an action that can be performed."""
    name: str
    label: str
    command_template: str  # e.g., "/tone {freq} {duration}"
    params: List[ActionParam] = field(default_factory=list)
    description: str = ""


# Actions organized by object type
ACTIONS: Dict[str, List[ActionDef]] = {
    'engine': [
        ActionDef(
            name='tone',
            label='Generate Tone',
            command_template='/tone {freq} {duration}',
            params=[
                ActionParam('freq', 'Frequency (Hz)', 'float', 440.0, min_val=20, max_val=20000),
                ActionParam('duration', 'Duration (beats)', 'float', 1.0, min_val=0.1, max_val=32),
            ],
            description='Generate a simple tone'
        ),
        ActionDef(
            name='mel',
            label='Generate Melody',
            command_template='/mel {pattern} {root_hz}',
            params=[
                ActionParam('pattern', 'Note Pattern', 'string', '0.4.7'),
                ActionParam('root_hz', 'Root Frequency', 'float', 440.0, min_val=20, max_val=8000),
            ],
            description='Generate a melody from note pattern'
        ),
        ActionDef(
            name='play',
            label='Play Audio',
            command_template='/play',
            params=[],
            description='Play the current audio buffer'
        ),
        ActionDef(
            name='render',
            label='Render to File',
            command_template='/render {filename}',
            params=[
                ActionParam('filename', 'Output File', 'string', 'output.flac'),
            ],
            description='Render audio to file'
        ),
        ActionDef(
            name='bpm',
            label='Set Tempo',
            command_template='/bpm {tempo}',
            params=[
                ActionParam('tempo', 'BPM', 'float', 128.0, min_val=20, max_val=400),
            ],
            description='Set the tempo in beats per minute'
        ),
        ActionDef(
            name='version',
            label='Show Version',
            command_template='/version',
            params=[],
            description='Show MDMA version info'
        ),
    ],
    'synth': [
        ActionDef(
            name='wave',
            label='Set Waveform',
            command_template='/wm {waveform}',
            params=[
                ActionParam('waveform', 'Waveform', 'enum', 'sine', 
                           choices=['sine', 'saw', 'square', 'triangle', 'noise']),
            ],
            description='Set the oscillator waveform'
        ),
        ActionDef(
            name='freq',
            label='Set Frequency',
            command_template='/fr {freq}',
            params=[
                ActionParam('freq', 'Frequency (Hz)', 'float', 440.0, min_val=20, max_val=20000),
            ],
            description='Set oscillator frequency'
        ),
        ActionDef(
            name='amp',
            label='Set Amplitude',
            command_template='/amp {level}',
            params=[
                ActionParam('level', 'Amplitude (0-1)', 'float', 0.8, min_val=0, max_val=1),
            ],
            description='Set oscillator amplitude'
        ),
        ActionDef(
            name='voices',
            label='Set Voice Count',
            command_template='/vc {count}',
            params=[
                ActionParam('count', 'Voices', 'int', 1, min_val=1, max_val=16),
            ],
            description='Set number of voices for unison'
        ),
        ActionDef(
            name='detune',
            label='Set Detune',
            command_template='/dt {cents}',
            params=[
                ActionParam('cents', 'Detune (cents)', 'float', 0, min_val=0, max_val=100),
            ],
            description='Set voice detuning in cents'
        ),
    ],
    'filter': [
        ActionDef(
            name='filter_type',
            label='Set Filter Type',
            command_template='/ft {filter_type}',
            params=[
                ActionParam('filter_type', 'Filter Type', 'enum', 'lowpass',
                           choices=['lowpass', 'highpass', 'bandpass', 'notch', 'moog', 'acid']),
            ],
            description='Set filter type'
        ),
        ActionDef(
            name='cutoff',
            label='Set Cutoff',
            command_template='/cut {freq}',
            params=[
                ActionParam('freq', 'Cutoff (Hz)', 'float', 4500.0, min_val=20, max_val=20000),
            ],
            description='Set filter cutoff frequency'
        ),
        ActionDef(
            name='resonance',
            label='Set Resonance',
            command_template='/res {amount}',
            params=[
                ActionParam('amount', 'Resonance (0-100)', 'float', 50.0, min_val=0, max_val=100),
            ],
            description='Set filter resonance'
        ),
    ],
    'envelope': [
        ActionDef(
            name='attack',
            label='Set Attack',
            command_template='/atk {time}',
            params=[
                ActionParam('time', 'Attack (sec)', 'float', 0.01, min_val=0, max_val=10),
            ],
            description='Set envelope attack time'
        ),
        ActionDef(
            name='decay',
            label='Set Decay',
            command_template='/dec {time}',
            params=[
                ActionParam('time', 'Decay (sec)', 'float', 0.1, min_val=0, max_val=10),
            ],
            description='Set envelope decay time'
        ),
        ActionDef(
            name='sustain',
            label='Set Sustain',
            command_template='/sus {level}',
            params=[
                ActionParam('level', 'Sustain (0-1)', 'float', 0.8, min_val=0, max_val=1),
            ],
            description='Set envelope sustain level'
        ),
        ActionDef(
            name='release',
            label='Set Release',
            command_template='/rel {time}',
            params=[
                ActionParam('time', 'Release (sec)', 'float', 0.1, min_val=0, max_val=10),
            ],
            description='Set envelope release time'
        ),
    ],
    'fx': [
        ActionDef(
            name='fx_add',
            label='Add Effect',
            command_template='/fx {effect}',
            params=[
                ActionParam('effect', 'Effect', 'enum', 'reverb',
                           choices=['reverb', 'delay', 'chorus', 'distortion', 'phaser', 
                                   'flanger', 'compressor', 'eq', 'bitcrush']),
            ],
            description='Add an effect to the chain'
        ),
        ActionDef(
            name='fx_clear',
            label='Clear Effects',
            command_template='/fxc',
            params=[],
            description='Clear all effects'
        ),
    ],
    'preset': [
        ActionDef(
            name='use_preset',
            label='Use Preset',
            command_template='/use {name}',
            params=[
                ActionParam('name', 'Preset Name', 'string', 'saw'),
            ],
            description='Load and use a synth preset'
        ),
        ActionDef(
            name='list_presets',
            label='List Presets',
            command_template='/sydef list',
            params=[],
            description='List all available presets'
        ),
    ],
    'bank': [
        ActionDef(
            name='bank_select',
            label='Select Bank',
            command_template='/bank {number}',
            params=[
                ActionParam('number', 'Bank Number', 'int', 1, min_val=1, max_val=8),
            ],
            description='Select a sound bank'
        ),
        ActionDef(
            name='bank_list',
            label='List Banks',
            command_template='/banks',
            params=[],
            description='List all available banks'
        ),
    ],
}


# ============================================================================
# COMMAND EXECUTOR
# ============================================================================

class CommandExecutor:
    """Executes MDMA commands and captures output."""
    
    def __init__(self):
        self.session = None
        self.commands = None
        self._init_engine()
    
    def _init_engine(self):
        """Initialize the MDMA engine."""
        try:
            from mdma_rebuild.core.session import Session
            import bmdma
            
            self.session = Session()
            self.commands = bmdma.build_command_table()
            
            # Load factory presets
            try:
                from mdma_rebuild.commands.sydef_cmds import load_factory_presets
                load_factory_presets(self.session)
            except (ImportError, AttributeError):
                pass

            self.init_error = None

        except ImportError as e:
            self.init_error = str(e)
            print(f"Warning: Could not import MDMA engine: {e}")
            self.session = None
            self.commands = {}
    
    def execute(self, command: str) -> tuple[str, str, bool]:
        """Execute a command and return (stdout, stderr, success)."""
        if not command.startswith('/'):
            command = '/' + command
            
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        success = True
        
        try:
            # Parse command
            parts = command[1:].split()
            if not parts:
                return "", "Empty command", False
            
            cmd = parts[0].lower()
            args = parts[1:]
            
            # Look up and execute
            if self.commands and cmd in self.commands:
                func = self.commands[cmd]
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    result = func(self.session, args)
                if result:
                    stdout_capture.write(str(result) + "\n")
            else:
                stderr_capture.write(f"Unknown command: {cmd}\n")
                success = False
                
        except Exception as e:
            stderr_capture.write(f"Error: {e}\n")
            success = False
        
        return stdout_capture.getvalue(), stderr_capture.getvalue(), success
    
    def get_presets(self) -> List[str]:
        """Get list of available presets."""
        if self.session and hasattr(self.session, 'sydefs'):
            return list(self.session.sydefs.keys())
        return ['saw', 'square', 'sine', 'bass', 'lead', 'pad']
    
    def get_banks(self) -> List[str]:
        """Get list of available banks."""
        return [f"Bank {i}" for i in range(1, 9)]


# ============================================================================
# GUI COMPONENTS
# ============================================================================

class ObjectBrowser(wx.Panel):
    """Left panel - tree/list of objects to act on."""
    
    def __init__(self, parent, on_select_callback):
        super().__init__(parent)
        self.on_select = on_select_callback
        
        self.SetBackgroundColour(Theme.BG_PANEL)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Search box
        search_sizer = wx.BoxSizer(wx.HORIZONTAL)
        search_label = wx.StaticText(self, label="Search:")
        search_label.SetForegroundColour(Theme.FG_TEXT)
        self.search_box = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        self.search_box.SetBackgroundColour(Theme.BG_INPUT)
        self.search_box.SetForegroundColour(Theme.FG_TEXT)
        self.search_box.Bind(wx.EVT_TEXT, self.on_search)
        
        search_sizer.Add(search_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        search_sizer.Add(self.search_box, 1, wx.EXPAND)
        
        sizer.Add(search_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Tree control
        self.tree = wx.TreeCtrl(self, style=wx.TR_DEFAULT_STYLE | wx.TR_HIDE_ROOT)
        self.tree.SetBackgroundColour(Theme.BG_INPUT)
        self.tree.SetForegroundColour(Theme.FG_TEXT)
        self.tree.Bind(wx.EVT_TREE_SEL_CHANGED, self.on_tree_select)
        
        sizer.Add(self.tree, 1, wx.EXPAND | wx.ALL, 5)
        
        # Refresh button
        refresh_btn = wx.Button(self, label="Refresh (F5)")
        refresh_btn.Bind(wx.EVT_BUTTON, self.on_refresh)
        sizer.Add(refresh_btn, 0, wx.EXPAND | wx.ALL, 5)
        
        self.SetSizer(sizer)
        self.populate_tree()
    
    def populate_tree(self):
        """Populate the tree with object categories."""
        self.tree.DeleteAllItems()
        
        root = self.tree.AddRoot("MDMA")
        
        # Categories matching ACTIONS
        categories = [
            ('engine', 'Engine', ['Generate', 'Playback', 'Settings']),
            ('synth', 'Synthesizer', ['Oscillator', 'Voices']),
            ('filter', 'Filter', ['Type', 'Cutoff', 'Resonance']),
            ('envelope', 'Envelope', ['ADSR']),
            ('fx', 'Effects', ['Add', 'Manage']),
            ('preset', 'Presets', ['Factory', 'User']),
            ('bank', 'Banks', ['Select', 'Browse']),
        ]
        
        self.category_items = {}
        
        for cat_id, cat_name, subcats in categories:
            cat_item = self.tree.AppendItem(root, cat_name)
            self.tree.SetItemData(cat_item, ('category', cat_id))
            self.category_items[cat_id] = cat_item
            
            for subcat in subcats:
                sub_item = self.tree.AppendItem(cat_item, subcat)
                self.tree.SetItemData(sub_item, ('subcategory', cat_id))
        
        self.tree.ExpandAll()
    
    def on_tree_select(self, event):
        """Handle tree selection."""
        item = event.GetItem()
        if item.IsOk():
            data = self.tree.GetItemData(item)
            if data:
                obj_type, obj_id = data
                self.on_select(obj_type, obj_id)
    
    def on_search(self, event):
        """Filter tree by highlighting matching items and collapsing non-matches."""
        query = self.search_box.GetValue().lower().strip()
        if not query:
            self.tree.ExpandAll()
            return

        root = self.tree.GetRootItem()
        if not root.IsOk():
            return

        # Walk categories and expand/collapse based on match
        item, cookie = self.tree.GetFirstChild(root)
        while item.IsOk():
            cat_text = self.tree.GetItemText(item).lower()
            cat_match = query in cat_text
            child_match = False

            child, c_cookie = self.tree.GetFirstChild(item)
            while child.IsOk():
                if query in self.tree.GetItemText(child).lower():
                    child_match = True
                child, c_cookie = self.tree.GetNextChild(item, c_cookie)

            if cat_match or child_match:
                self.tree.Expand(item)
            else:
                self.tree.Collapse(item)

            item, cookie = self.tree.GetNextChild(root, cookie)
    
    def on_refresh(self, event):
        """Refresh the tree."""
        self.populate_tree()


class ActionPanel(wx.Panel):
    """Right panel - action selection and parameter editing."""
    
    def __init__(self, parent, executor: CommandExecutor, console_callback):
        super().__init__(parent)
        self.executor = executor
        self.console_callback = console_callback
        self.current_category = 'engine'
        self.current_action: Optional[ActionDef] = None
        self.param_controls: Dict[str, wx.Control] = {}
        
        self.SetBackgroundColour(Theme.BG_PANEL)
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Header
        self.header = wx.StaticText(self, label="Select an object from the browser")
        self.header.SetForegroundColour(Theme.ACCENT)
        font = self.header.GetFont()
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.header.SetFont(font)
        self.sizer.Add(self.header, 0, wx.ALL, 10)
        
        # Action dropdown
        action_sizer = wx.BoxSizer(wx.HORIZONTAL)
        action_label = wx.StaticText(self, label="Action:")
        action_label.SetForegroundColour(Theme.FG_TEXT)
        self.action_choice = wx.Choice(self)
        self.action_choice.SetBackgroundColour(Theme.BG_INPUT)
        self.action_choice.Bind(wx.EVT_CHOICE, self.on_action_select)
        
        action_sizer.Add(action_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        action_sizer.Add(self.action_choice, 1, wx.EXPAND)
        self.sizer.Add(action_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Description
        self.description = wx.StaticText(self, label="")
        self.description.SetForegroundColour(Theme.FG_DIM)
        self.sizer.Add(self.description, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        
        # Parameters panel (scrollable)
        self.params_panel = wx.ScrolledWindow(self)
        self.params_panel.SetBackgroundColour(Theme.BG_PANEL)
        self.params_panel.SetScrollRate(0, 20)
        self.params_sizer = wx.BoxSizer(wx.VERTICAL)
        self.params_panel.SetSizer(self.params_sizer)
        self.sizer.Add(self.params_panel, 1, wx.EXPAND | wx.ALL, 5)
        
        # Command preview
        preview_label = wx.StaticText(self, label="Command:")
        preview_label.SetForegroundColour(Theme.FG_DIM)
        self.sizer.Add(preview_label, 0, wx.LEFT | wx.TOP, 10)
        
        self.command_preview = wx.TextCtrl(self, style=wx.TE_READONLY)
        self.command_preview.SetBackgroundColour(Theme.BG_INPUT)
        self.command_preview.SetForegroundColour(Theme.ACCENT)
        self.sizer.Add(self.command_preview, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        
        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.run_btn = wx.Button(self, label="Run (Ctrl+R)")
        self.run_btn.SetBackgroundColour(Theme.ACCENT)
        self.run_btn.Bind(wx.EVT_BUTTON, self.on_run)
        
        self.copy_btn = wx.Button(self, label="Copy Command")
        self.copy_btn.Bind(wx.EVT_BUTTON, self.on_copy)
        
        self.reset_btn = wx.Button(self, label="Reset Params")
        self.reset_btn.Bind(wx.EVT_BUTTON, self.on_reset)
        
        btn_sizer.Add(self.run_btn, 1, wx.RIGHT, 5)
        btn_sizer.Add(self.copy_btn, 1, wx.RIGHT, 5)
        btn_sizer.Add(self.reset_btn, 1)
        
        self.sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        self.SetSizer(self.sizer)
        
        # Initialize with engine actions
        self.set_category('engine')
    
    def set_category(self, category: str):
        """Set the current category and update action list."""
        self.current_category = category
        self.header.SetLabel(f"{category.upper()} Actions")
        
        # Populate action dropdown
        self.action_choice.Clear()
        if category in ACTIONS:
            for action in ACTIONS[category]:
                self.action_choice.Append(action.label)
            if ACTIONS[category]:
                self.action_choice.SetSelection(0)
                self.on_action_select(None)
    
    def on_action_select(self, event):
        """Handle action selection."""
        idx = self.action_choice.GetSelection()
        if idx >= 0 and self.current_category in ACTIONS:
            actions = ACTIONS[self.current_category]
            if idx < len(actions):
                self.current_action = actions[idx]
                self.description.SetLabel(self.current_action.description)
                self.build_param_controls()
                self.update_command_preview()
    
    def build_param_controls(self):
        """Build parameter controls for current action."""
        # Clear existing
        self.params_sizer.Clear(True)
        self.param_controls.clear()
        
        if not self.current_action:
            return
        
        for param in self.current_action.params:
            row_sizer = wx.BoxSizer(wx.HORIZONTAL)
            
            # Label
            label = wx.StaticText(self.params_panel, label=f"{param.label}:")
            label.SetForegroundColour(Theme.FG_TEXT)
            label.SetMinSize((120, -1))
            row_sizer.Add(label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
            
            # Control based on type
            ctrl = None
            if param.param_type == 'int':
                ctrl = wx.SpinCtrl(self.params_panel, min=int(param.min_val or 0), 
                                   max=int(param.max_val or 100))
                ctrl.SetValue(int(param.default or 0))
                ctrl.Bind(wx.EVT_SPINCTRL, lambda e: self.update_command_preview())
            
            elif param.param_type == 'float':
                ctrl = wx.SpinCtrlDouble(self.params_panel, min=param.min_val or 0,
                                         max=param.max_val or 100, inc=0.1)
                ctrl.SetValue(float(param.default or 0))
                ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, lambda e: self.update_command_preview())
            
            elif param.param_type == 'bool':
                ctrl = wx.CheckBox(self.params_panel)
                ctrl.SetValue(bool(param.default))
                ctrl.Bind(wx.EVT_CHECKBOX, lambda e: self.update_command_preview())
            
            elif param.param_type == 'enum':
                ctrl = wx.Choice(self.params_panel, choices=param.choices)
                if param.default in param.choices:
                    ctrl.SetSelection(param.choices.index(param.default))
                else:
                    ctrl.SetSelection(0)
                ctrl.Bind(wx.EVT_CHOICE, lambda e: self.update_command_preview())
            
            elif param.param_type == 'file':
                file_sizer = wx.BoxSizer(wx.HORIZONTAL)
                ctrl = wx.TextCtrl(self.params_panel, value=str(param.default or ''))
                ctrl.Bind(wx.EVT_TEXT, lambda e: self.update_command_preview())
                browse_btn = wx.Button(self.params_panel, label="...", size=(30, -1))
                browse_btn.Bind(wx.EVT_BUTTON, lambda e, c=ctrl: self.on_browse_file(c))
                file_sizer.Add(ctrl, 1, wx.EXPAND | wx.RIGHT, 5)
                file_sizer.Add(browse_btn, 0)
                row_sizer.Add(file_sizer, 1, wx.EXPAND)
                self.param_controls[param.name] = ctrl
                self.params_sizer.Add(row_sizer, 0, wx.EXPAND | wx.ALL, 5)
                continue
            
            else:  # string
                ctrl = wx.TextCtrl(self.params_panel, value=str(param.default or ''))
                ctrl.Bind(wx.EVT_TEXT, lambda e: self.update_command_preview())
            
            if ctrl:
                ctrl.SetBackgroundColour(Theme.BG_INPUT)
                ctrl.SetForegroundColour(Theme.FG_TEXT)
                row_sizer.Add(ctrl, 1, wx.EXPAND)
                self.param_controls[param.name] = ctrl
            
            self.params_sizer.Add(row_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        self.params_panel.FitInside()
        self.Layout()
    
    def on_browse_file(self, text_ctrl):
        """Handle file browse button."""
        dlg = wx.FileDialog(self, "Select file", wildcard="All files (*.*)|*.*")
        if dlg.ShowModal() == wx.ID_OK:
            text_ctrl.SetValue(dlg.GetPath())
        dlg.Destroy()
    
    def get_param_values(self) -> Dict[str, Any]:
        """Get current parameter values."""
        values = {}
        if not self.current_action:
            return values
        
        for param in self.current_action.params:
            ctrl = self.param_controls.get(param.name)
            if ctrl:
                if param.param_type == 'int':
                    values[param.name] = ctrl.GetValue()
                elif param.param_type == 'float':
                    values[param.name] = ctrl.GetValue()
                elif param.param_type == 'bool':
                    values[param.name] = ctrl.GetValue()
                elif param.param_type == 'enum':
                    idx = ctrl.GetSelection()
                    values[param.name] = param.choices[idx] if idx >= 0 else param.default
                else:
                    values[param.name] = ctrl.GetValue()
        
        return values
    
    def build_command(self) -> str:
        """Build the command string from current action and params."""
        if not self.current_action:
            return ""
        
        values = self.get_param_values()
        try:
            return self.current_action.command_template.format(**values)
        except KeyError as e:
            self.console_callback(f"Warning: missing parameter {e} in command template\n", 'warning')
            return self.current_action.command_template
    
    def update_command_preview(self):
        """Update the command preview text."""
        cmd = self.build_command()
        self.command_preview.SetValue(cmd)
    
    def on_run(self, event):
        """Execute the current command."""
        cmd = self.build_command()
        if cmd:
            self.console_callback(f">>> {cmd}\n", 'command')
            stdout, stderr, success = self.executor.execute(cmd)
            if stdout:
                self.console_callback(stdout, 'stdout')
            if stderr:
                self.console_callback(stderr, 'stderr')
            status = "OK" if success else "ERROR"
            color = 'success' if success else 'error'
            self.console_callback(f"[{status}]\n\n", color)
    
    def on_copy(self, event):
        """Copy command to clipboard."""
        cmd = self.build_command()
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(cmd))
            wx.TheClipboard.Close()
            self.console_callback(f"Copied: {cmd}\n", 'info')
    
    def on_reset(self, event):
        """Reset parameters to defaults."""
        if self.current_action:
            for param in self.current_action.params:
                ctrl = self.param_controls.get(param.name)
                if ctrl:
                    if param.param_type in ('int', 'float'):
                        ctrl.SetValue(param.default or 0)
                    elif param.param_type == 'bool':
                        ctrl.SetValue(bool(param.default))
                    elif param.param_type == 'enum':
                        if param.default in param.choices:
                            ctrl.SetSelection(param.choices.index(param.default))
                    else:
                        ctrl.SetValue(str(param.default or ''))
            self.update_command_preview()


class ConsolePanel(wx.Panel):
    """Bottom panel - output console."""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        self.SetBackgroundColour(Theme.BG_DARK)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Header
        header_sizer = wx.BoxSizer(wx.HORIZONTAL)
        header = wx.StaticText(self, label="Console Output")
        header.SetForegroundColour(Theme.FG_DIM)
        
        clear_btn = wx.Button(self, label="Clear", size=(60, -1))
        clear_btn.Bind(wx.EVT_BUTTON, self.on_clear)
        
        header_sizer.Add(header, 1, wx.ALIGN_CENTER_VERTICAL)
        header_sizer.Add(clear_btn, 0)
        
        sizer.Add(header_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Console text
        self.console = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY | 
                                   wx.TE_RICH2 | wx.HSCROLL)
        self.console.SetBackgroundColour(Theme.BG_DARK)
        self.console.SetForegroundColour(Theme.FG_TEXT)
        
        # Use monospace font
        font = wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.console.SetFont(font)
        
        sizer.Add(self.console, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        
        self.SetSizer(sizer)
        
        # Color mapping
        self.colors = {
            'command': Theme.ACCENT,
            'stdout': Theme.FG_TEXT,
            'stderr': Theme.ERROR,
            'success': Theme.SUCCESS,
            'error': Theme.ERROR,
            'info': Theme.FG_DIM,
            'warning': Theme.WARNING,
        }
    
    def append(self, text: str, style: str = 'stdout'):
        """Append text to console with color."""
        color = self.colors.get(style, Theme.FG_TEXT)
        
        # Get current position
        pos = self.console.GetLastPosition()
        
        # Append text
        self.console.AppendText(text)
        
        # Apply color to new text
        new_pos = self.console.GetLastPosition()
        self.console.SetStyle(pos, new_pos, wx.TextAttr(color))
        
        # Scroll to end
        self.console.ShowPosition(new_pos)
    
    def on_clear(self, event):
        """Clear the console."""
        self.console.Clear()


# ============================================================================
# MAIN FRAME
# ============================================================================

class MDMAFrame(wx.Frame):
    """Main application window."""
    
    def __init__(self):
        super().__init__(None, title="MDMA - Action Panel", size=(1200, 800))
        
        self.SetBackgroundColour(Theme.BG_DARK)
        
        # Initialize executor
        self.executor = CommandExecutor()
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create main layout with splitters
        self.create_layout()
        
        # Set up keyboard shortcuts
        self.setup_shortcuts()
        
        # Status bar
        self.CreateStatusBar()
        self.SetStatusText("MDMA GUI v0.2.0 - Ready")
        
        # Center on screen
        self.Centre()
        
        # Welcome message
        self.console.append("MDMA GUI v0.2.0 - Action Panel Client\n", 'info')
        self.console.append("="*50 + "\n", 'info')

        # Warn if engine failed to load
        if self.executor.init_error:
            self.console.append(f"WARNING: MDMA engine failed to load: {self.executor.init_error}\n", 'error')
            self.console.append("Commands will not execute. Check your installation.\n\n", 'error')
            self.SetStatusText("MDMA GUI v0.2.0 - ENGINE NOT LOADED")
        else:
            self.console.append("Select a category from the left panel, choose an action,\n", 'info')
            self.console.append("set parameters, and click Run (Ctrl+R) to execute.\n\n", 'info')
    
    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = wx.MenuBar()
        
        # File menu
        file_menu = wx.Menu()
        file_menu.Append(wx.ID_NEW, "&New Session\tCtrl+N")
        file_menu.Append(wx.ID_OPEN, "&Open Project\tCtrl+O")
        file_menu.Append(wx.ID_SAVE, "&Save Project\tCtrl+S")
        file_menu.AppendSeparator()
        file_menu.Append(wx.ID_EXIT, "E&xit\tAlt+F4")
        menubar.Append(file_menu, "&File")
        
        # View menu
        view_menu = wx.Menu()
        view_menu.Append(wx.ID_ANY, "&Refresh\tF5")
        view_menu.Append(wx.ID_ANY, "Focus &Console\tCtrl+L")
        view_menu.Append(wx.ID_ANY, "Focus &Search\tCtrl+F")
        menubar.Append(view_menu, "&View")
        
        # Engine menu
        engine_menu = wx.Menu()
        engine_menu.Append(wx.ID_ANY, "&Version Info")
        engine_menu.Append(wx.ID_ANY, "&Help / Quick Reference")
        engine_menu.AppendSeparator()
        engine_menu.Append(wx.ID_ANY, "Open &Output Folder")
        menubar.Append(engine_menu, "&Engine")
        
        # Help menu
        help_menu = wx.Menu()
        help_menu.Append(wx.ID_ABOUT, "&About")
        menubar.Append(help_menu, "&Help")
        
        self.SetMenuBar(menubar)
        
        # Bind events
        self.Bind(wx.EVT_MENU, self.on_exit, id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU, self.on_about, id=wx.ID_ABOUT)
    
    def create_layout(self):
        """Create the main layout with splitters."""
        # Main vertical splitter (top panels / bottom console)
        self.main_splitter = wx.SplitterWindow(self)
        
        # Top horizontal splitter (left browser / right action panel)
        self.top_splitter = wx.SplitterWindow(self.main_splitter)
        
        # Create panels
        self.browser = ObjectBrowser(self.top_splitter, self.on_object_select)
        self.action_panel = ActionPanel(self.top_splitter, self.executor, self.console_append)
        self.console = ConsolePanel(self.main_splitter)
        
        # Configure splitters
        self.top_splitter.SplitVertically(self.browser, self.action_panel, 250)
        self.top_splitter.SetMinimumPaneSize(200)
        
        self.main_splitter.SplitHorizontally(self.top_splitter, self.console, -200)
        self.main_splitter.SetMinimumPaneSize(150)
    
    def setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        accel_entries = [
            (wx.ACCEL_CTRL, ord('R'), wx.ID_ANY),  # Run
            (wx.ACCEL_CTRL, ord('L'), wx.ID_ANY),  # Focus console
            (wx.ACCEL_CTRL, ord('F'), wx.ID_ANY),  # Focus search
            (wx.ACCEL_NORMAL, wx.WXK_F5, wx.ID_ANY),  # Refresh
        ]
        
        # Create IDs and bind
        self.run_id = wx.NewIdRef()
        self.console_id = wx.NewIdRef()
        self.search_id = wx.NewIdRef()
        self.refresh_id = wx.NewIdRef()
        
        accel_table = wx.AcceleratorTable([
            (wx.ACCEL_CTRL, ord('R'), self.run_id),
            (wx.ACCEL_CTRL, ord('L'), self.console_id),
            (wx.ACCEL_CTRL, ord('F'), self.search_id),
            (wx.ACCEL_NORMAL, wx.WXK_F5, self.refresh_id),
        ])
        self.SetAcceleratorTable(accel_table)
        
        self.Bind(wx.EVT_MENU, lambda e: self.action_panel.on_run(e), id=self.run_id)
        self.Bind(wx.EVT_MENU, lambda e: self.console.console.SetFocus(), id=self.console_id)
        self.Bind(wx.EVT_MENU, lambda e: self.browser.search_box.SetFocus(), id=self.search_id)
        self.Bind(wx.EVT_MENU, lambda e: self.browser.on_refresh(e), id=self.refresh_id)
    
    def on_object_select(self, obj_type: str, obj_id: str):
        """Handle object selection from browser."""
        if obj_type in ('category', 'subcategory'):
            self.action_panel.set_category(obj_id)
            self.SetStatusText(f"Category: {obj_id}")
    
    def console_append(self, text: str, style: str = 'stdout'):
        """Append text to console (callback for action panel)."""
        self.console.append(text, style)
    
    def on_exit(self, event):
        """Handle exit."""
        self.Close()
    
    def on_about(self, event):
        """Show about dialog."""
        info = wx.adv.AboutDialogInfo()
        info.SetName("MDMA GUI")
        info.SetVersion("0.2.0")
        info.SetDescription("Action Panel Client for MDMA\n\n"
                           "A thin GUI client over the MDMA CLI.\n"
                           "Maps GUI actions to existing commands.")
        info.SetCopyright("(C) 2026")
        wx.adv.AboutBox(info)


# ============================================================================
# APPLICATION
# ============================================================================

class MDMAApp(wx.App):
    """Main application."""
    
    def OnInit(self):
        frame = MDMAFrame()
        frame.Show()
        return True


def main():
    """Entry point."""
    app = MDMAApp()
    app.MainLoop()


if __name__ == '__main__':
    main()
