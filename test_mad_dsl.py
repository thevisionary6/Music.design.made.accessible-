#!/usr/bin/env python
"""
MAD DSL Test Script - simulates the DSL dummy document flow.

This tests the complete DSL implementation against the spec.
"""

import sys
import os

# Setup path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mdma_rebuild.core.session import Session
from mdma_rebuild.dsp.advanced_ops import get_user_stack
from mdma_rebuild.commands.dsl_cmds import (
    reset_dsl_state, get_dsl_state, strip_comments,
    cmd_start, cmd_final, cmd_update,
    cmd_n, cmd_ni, cmd_mel,
    cmd_out_block, process_out_line, end_out_block,
    cmd_synth, cmd_use, cmd_chain, handle_chain_command,
    handle_object_command, cmd_eq_expr,
)
from bmdma import build_command_table


def test_dsl():
    """Run DSL tests matching the dummy document."""
    
    print("=" * 60)
    print("MAD DSL TEST SUITE")
    print("=" * 60)
    print()
    
    session = Session()
    commands = build_command_table()
    
    # Create executor
    def execute(line):
        if not line.startswith('/'):
            return 'ERROR: must start with /'
        parts = line[1:].split()
        cmd, args = parts[0].lower(), parts[1:]
        func = commands.get(cmd)
        if func:
            return func(session, args)
        return f'Unknown: {cmd}'
    
    session.command_executor = execute
    reset_dsl_state()
    get_user_stack().clear()
    
    passed = 0
    failed = 0
    
    def test(name, result, expected_contains=None):
        nonlocal passed, failed
        success = True
        if expected_contains and expected_contains not in str(result):
            success = False
        if success:
            passed += 1
            print(f"✓ {name}")
            print(f"  -> {result}")
        else:
            failed += 1
            print(f"✗ {name}")
            print(f"  -> {result}")
            print(f"  Expected to contain: {expected_contains}")
        print()
    
    # === Test 1: Comments ===
    print("--- Testing Comments ---")
    result = strip_comments("/// this is a comment ///")
    test("Comment stripping (full)", result == "", None)
    
    result = strip_comments("/bpm 140 /// set tempo ///")
    test("Comment stripping (partial)", result, "/bpm 140")
    
    # === Test 2: /start block ===
    print("--- Testing /start Block ---")
    result = cmd_start(session, [])
    test("/start", result, "DSL MODE")
    
    # === Test 3: BPM setting ===
    print("--- Testing BPM ---")
    result = execute("/bpm 140")
    test("/bpm 140", result, "140")
    
    # === Test 4: /update ===
    print("--- Testing /update ---")
    result = cmd_update(session, [])
    test("/update", result, "UPDATE")
    
    # === Test 5: Note buffer ===
    print("--- Testing Note Buffers ---")
    result = cmd_n(session, ['a'])
    test("/n a", result, "created 'a'")
    
    result = cmd_ni(session, ['a'])
    test("/ni a", result, "selected 'a'")
    
    # === Test 6: Melody ===
    print("--- Testing Melody ---")
    result = cmd_mel(session, ['0...2..3.2.0...', 'a', 'RUTE'])
    test("/mel pattern", result, "notes")
    
    # === Test 7: Output block ===
    print("--- Testing /out Block ---")
    reset_dsl_state()
    
    result = cmd_out_block(session, [])
    test("/out", result, "started")
    
    result = process_out_line(session, "/60 /amp 1.2 /atk 1")
    test("/60 /amp 1.2 /atk 1", result, "note: 60")
    
    result = process_out_line(session, "/55")
    test("/55", result, "note: 55")
    
    result = process_out_line(session, "/53 /cut 1200")
    test("/53 /cut 1200", result, "note: 53")
    
    result = end_out_block(session)
    test("/end (out)", result, "OUT BLOCK END")
    
    # === Test 8: Chords ===
    print("--- Testing Chords ---")
    reset_dsl_state()
    
    result = cmd_out_block(session, [])
    result = process_out_line(session, "/c 60 63 67")
    test("/c 60 63 67", result, "chord")
    
    result = process_out_line(session, "/c")
    test("/c (repeat)", result, "repeat")
    
    end_out_block(session)
    
    # === Test 9: Synth instances ===
    print("--- Testing Synth Instances ---")
    reset_dsl_state()
    
    result = cmd_synth(session, ['bass_man'])
    test("/synth bass_man", result, "created 'bass_man'")
    
    result = cmd_use(session, ['synth', '1'])
    test("/use synth 1", result, "bass_man")
    
    result = handle_object_command(session, 'bass_man', ['fc', '2'])
    test("/bass_man fc 2", result, "filter_count = 2")
    
    # === Test 10: FX chains ===
    print("--- Testing FX Chains ---")
    reset_dsl_state()
    
    result = cmd_chain(session, ['cook'])
    test("/chain cook", result, "created 'cook'")
    
    result = handle_chain_command(session, 'cook', ['add', 'cr5', '/and', 'add', 'vamp5'])
    test("/cook add cr5 /and add vamp5", result, "added")
    
    # === Test 11: Variables ===
    print("--- Testing Variables ---")
    get_user_stack().clear()
    
    result = cmd_eq_expr(session, ['jake', '8'])
    test("/= jake 8", result, "jake = 8")
    
    result = cmd_eq_expr(session, ['kaj', '9'])
    test("/= kaj 9", result, "kaj = 9")
    
    result = cmd_eq_expr(session, ['a_good_time', 'jake', '/add', 'kaj'])
    test("/= a_good_time jake /add kaj", result, "= 17")
    
    # Check variable value
    stack = get_user_stack()
    val = stack.get('a_good_time')
    test("/give a_good_time (value check)", val == 17, None)
    
    # === Test 12: /final ===
    print("--- Testing /final ---")
    reset_dsl_state()
    cmd_start(session, [])
    get_dsl_state().dsl_commands = ['/bpm 120', '/out Hello']
    result = cmd_final(session, [])
    test("/final", result, "compiled")
    
    # === Summary ===
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = test_dsl()
    sys.exit(0 if success else 1)
