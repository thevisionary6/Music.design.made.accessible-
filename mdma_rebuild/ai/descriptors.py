"""MDMA Descriptor Vocabulary Engine.

The shared language for audio analysis, genetic breeding, sample pack curation,
text-to-preset generation, and AI command routing.

Descriptors are:
- Fuzzy (probabilistic, not absolute)
- Overlapping (multiple can apply)
- Interpretable (human-readable)
- Composable (can combine)

AI systems may not invent descriptors outside this vocabulary.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import numpy as np


# ============================================================================
# MASTER DESCRIPTOR VOCABULARY (A-Z)
# ============================================================================

DESCRIPTOR_VOCABULARY = {
    # A
    'airy': {'category': 'spatial', 'opposites': ['dense', 'heavy'], 'related': ['light', 'open', 'spacious']},
    'aggressive': {'category': 'character', 'opposites': ['gentle', 'soft'], 'related': ['harsh', 'intense', 'powerful']},
    'analog': {'category': 'character', 'opposites': ['digital', 'synthetic'], 'related': ['warm', 'organic', 'natural']},
    'abrasive': {'category': 'texture', 'opposites': ['smooth', 'soft'], 'related': ['harsh', 'gritty', 'rough']},
    'alive': {'category': 'movement', 'opposites': ['static', 'lifeless'], 'related': ['dynamic', 'evolving', 'organic']},
    'ambient': {'category': 'spatial', 'opposites': ['dry', 'intimate'], 'related': ['spacious', 'atmospheric', 'reverberant']},
    'articulated': {'category': 'envelope', 'opposites': ['smeared', 'muddy'], 'related': ['crisp', 'defined', 'precise']},
    'asymmetrical': {'category': 'structure', 'opposites': ['balanced', 'symmetrical'], 'related': ['uneven', 'irregular']},
    'atmospheric': {'category': 'spatial', 'opposites': ['dry', 'direct'], 'related': ['ambient', 'spacious', 'ethereal']},
    
    # B
    'bright': {'category': 'spectral', 'opposites': ['dark', 'dull'], 'related': ['airy', 'crisp', 'shimmering']},
    'brittle': {'category': 'texture', 'opposites': ['soft', 'rubbery'], 'related': ['fragile', 'glassy', 'crystalline']},
    'bloomy': {'category': 'spatial', 'opposites': ['dry', 'tight'], 'related': ['lush', 'reverberant', 'soft']},
    'blunt': {'category': 'envelope', 'opposites': ['sharp', 'crisp'], 'related': ['soft', 'rounded', 'dull']},
    'buzzing': {'category': 'texture', 'opposites': ['clean', 'pure'], 'related': ['noisy', 'electric', 'vibrating']},
    'breathy': {'category': 'texture', 'opposites': ['clean', 'tonal'], 'related': ['airy', 'noisy', 'soft']},
    'boomy': {'category': 'spectral', 'opposites': ['tight', 'controlled'], 'related': ['deep', 'resonant', 'muddy']},
    'broken': {'category': 'texture', 'opposites': ['smooth', 'clean'], 'related': ['distorted', 'glitchy', 'fractured']},
    'balanced': {'category': 'spectral', 'opposites': ['unbalanced', 'lopsided'], 'related': ['neutral', 'even', 'full']},
    'bending': {'category': 'pitch', 'opposites': ['stable', 'fixed'], 'related': ['sliding', 'warped', 'modulated']},
    
    # C
    'cold': {'category': 'character', 'opposites': ['warm', 'hot'], 'related': ['icy', 'sterile', 'clinical']},
    'cloudy': {'category': 'spectral', 'opposites': ['clear', 'crisp'], 'related': ['muddy', 'hazy', 'veiled']},
    'crunchy': {'category': 'texture', 'opposites': ['smooth', 'clean'], 'related': ['gritty', 'distorted', 'saturated']},
    'clean': {'category': 'texture', 'opposites': ['dirty', 'distorted'], 'related': ['pure', 'clear', 'pristine']},
    'complex': {'category': 'structure', 'opposites': ['simple', 'minimal'], 'related': ['layered', 'dense', 'rich']},
    'chaotic': {'category': 'movement', 'opposites': ['controlled', 'ordered'], 'related': ['erratic', 'unpredictable', 'wild']},
    'controlled': {'category': 'movement', 'opposites': ['chaotic', 'wild'], 'related': ['precise', 'tight', 'restrained']},
    'crystalline': {'category': 'texture', 'opposites': ['muddy', 'cloudy'], 'related': ['glassy', 'clear', 'bright']},
    'compressed': {'category': 'dynamics', 'opposites': ['dynamic', 'open'], 'related': ['squashed', 'dense', 'loud']},
    'cavernous': {'category': 'spatial', 'opposites': ['intimate', 'dry'], 'related': ['spacious', 'reverberant', 'deep']},
    
    # D
    'dark': {'category': 'spectral', 'opposites': ['bright', 'airy'], 'related': ['deep', 'heavy', 'low']},
    'dense': {'category': 'texture', 'opposites': ['sparse', 'airy'], 'related': ['thick', 'heavy', 'layered']},
    'drifting': {'category': 'movement', 'opposites': ['static', 'locked'], 'related': ['floating', 'wandering', 'evolving']},
    'distorted': {'category': 'texture', 'opposites': ['clean', 'pure'], 'related': ['saturated', 'crunchy', 'broken']},
    'dry': {'category': 'spatial', 'opposites': ['wet', 'reverberant'], 'related': ['direct', 'intimate', 'tight']},
    'deep': {'category': 'spectral', 'opposites': ['thin', 'shallow'], 'related': ['low', 'subby', 'weighty']},
    'dynamic': {'category': 'dynamics', 'opposites': ['compressed', 'flat'], 'related': ['expressive', 'alive', 'moving']},
    'dull': {'category': 'spectral', 'opposites': ['bright', 'sharp'], 'related': ['muffled', 'dark', 'filtered']},
    'delicate': {'category': 'character', 'opposites': ['heavy', 'aggressive'], 'related': ['gentle', 'soft', 'fragile']},
    'detuned': {'category': 'pitch', 'opposites': ['tuned', 'pure'], 'related': ['thick', 'wobbly', 'chorused']},
    
    # E
    'evolving': {'category': 'movement', 'opposites': ['static', 'fixed'], 'related': ['moving', 'changing', 'alive']},
    'energetic': {'category': 'character', 'opposites': ['lifeless', 'dull'], 'related': ['alive', 'dynamic', 'powerful']},
    'elastic': {'category': 'texture', 'opposites': ['rigid', 'stiff'], 'related': ['rubbery', 'bouncy', 'flexible']},
    'eerie': {'category': 'character', 'opposites': ['warm', 'comforting'], 'related': ['ghostly', 'haunting', 'unsettling']},
    'empty': {'category': 'texture', 'opposites': ['full', 'dense'], 'related': ['sparse', 'hollow', 'minimal']},
    'expressive': {'category': 'movement', 'opposites': ['static', 'mechanical'], 'related': ['dynamic', 'alive', 'emotional']},
    'enveloping': {'category': 'spatial', 'opposites': ['narrow', 'focused'], 'related': ['immersive', 'wide', 'surrounding']},
    'erratic': {'category': 'movement', 'opposites': ['stable', 'predictable'], 'related': ['chaotic', 'unpredictable', 'jittery']},
    'ethereal': {'category': 'character', 'opposites': ['grounded', 'heavy'], 'related': ['ghostly', 'airy', 'floating']},
    'exaggerated': {'category': 'character', 'opposites': ['subtle', 'restrained'], 'related': ['extreme', 'dramatic', 'bold']},
    
    # F
    'full': {'category': 'spectral', 'opposites': ['thin', 'empty'], 'related': ['rich', 'dense', 'complete']},
    'flat': {'category': 'dynamics', 'opposites': ['dynamic', 'expressive'], 'related': ['compressed', 'lifeless', 'even']},
    'fuzzy': {'category': 'texture', 'opposites': ['clean', 'crisp'], 'related': ['soft', 'warm', 'blurred']},
    'fragile': {'category': 'character', 'opposites': ['robust', 'powerful'], 'related': ['delicate', 'brittle', 'soft']},
    'filtered': {'category': 'spectral', 'opposites': ['full', 'open'], 'related': ['muffled', 'narrow', 'processed']},
    'focused': {'category': 'spatial', 'opposites': ['diffuse', 'wide'], 'related': ['tight', 'precise', 'narrow']},
    'flowing': {'category': 'movement', 'opposites': ['choppy', 'staccato'], 'related': ['smooth', 'continuous', 'liquid']},
    'fractured': {'category': 'texture', 'opposites': ['smooth', 'whole'], 'related': ['broken', 'glitchy', 'scattered']},
    'fat': {'category': 'spectral', 'opposites': ['thin', 'narrow'], 'related': ['thick', 'full', 'heavy']},
    'forward': {'category': 'spatial', 'opposites': ['distant', 'recessed'], 'related': ['present', 'direct', 'upfront']},
    
    # G
    'gritty': {'category': 'texture', 'opposites': ['smooth', 'clean'], 'related': ['rough', 'dirty', 'raw']},
    'glassy': {'category': 'texture', 'opposites': ['soft', 'warm'], 'related': ['crystalline', 'bright', 'clear']},
    'granular': {'category': 'texture', 'opposites': ['smooth', 'continuous'], 'related': ['grainy', 'textured', 'particulate']},
    'grounded': {'category': 'character', 'opposites': ['floating', 'ethereal'], 'related': ['solid', 'stable', 'heavy']},
    'glowing': {'category': 'character', 'opposites': ['dark', 'dull'], 'related': ['warm', 'radiant', 'bright']},
    'gated': {'category': 'envelope', 'opposites': ['sustained', 'flowing'], 'related': ['chopped', 'rhythmic', 'staccato']},
    'growling': {'category': 'character', 'opposites': ['smooth', 'pure'], 'related': ['aggressive', 'modulated', 'bass']},
    'gentle': {'category': 'character', 'opposites': ['aggressive', 'harsh'], 'related': ['soft', 'delicate', 'subtle']},
    'geometric': {'category': 'structure', 'opposites': ['organic', 'flowing'], 'related': ['angular', 'precise', 'mechanical']},
    'ghostly': {'category': 'character', 'opposites': ['solid', 'present'], 'related': ['ethereal', 'eerie', 'distant']},
    
    # H
    'harsh': {'category': 'texture', 'opposites': ['smooth', 'soft'], 'related': ['aggressive', 'abrasive', 'bright']},
    'hollow': {'category': 'spectral', 'opposites': ['full', 'dense'], 'related': ['empty', 'thin', 'resonant']},
    'heavy': {'category': 'spectral', 'opposites': ['light', 'airy'], 'related': ['deep', 'dense', 'massive']},
    'hot': {'category': 'character', 'opposites': ['cold', 'cool'], 'related': ['warm', 'saturated', 'driven']},
    'hazy': {'category': 'spatial', 'opposites': ['clear', 'focused'], 'related': ['foggy', 'diffuse', 'soft']},
    'harmonic': {'category': 'spectral', 'opposites': ['inharmonic', 'noisy'], 'related': ['tonal', 'musical', 'rich']},
    'hostile': {'category': 'character', 'opposites': ['friendly', 'warm'], 'related': ['aggressive', 'harsh', 'cold']},
    'hovering': {'category': 'spatial', 'opposites': ['grounded', 'heavy'], 'related': ['floating', 'suspended', 'ethereal']},
    'hybrid': {'category': 'structure', 'opposites': ['pure', 'simple'], 'related': ['mixed', 'layered', 'complex']},
    'hypnotic': {'category': 'movement', 'opposites': ['erratic', 'chaotic'], 'related': ['repetitive', 'trance', 'mesmerizing']},
    
    # I
    'icy': {'category': 'character', 'opposites': ['warm', 'hot'], 'related': ['cold', 'crystalline', 'bright']},
    'intimate': {'category': 'spatial', 'opposites': ['distant', 'spacious'], 'related': ['close', 'dry', 'direct']},
    'intense': {'category': 'character', 'opposites': ['mild', 'subtle'], 'related': ['powerful', 'aggressive', 'extreme']},
    'irregular': {'category': 'rhythm', 'opposites': ['regular', 'steady'], 'related': ['erratic', 'asymmetrical', 'unpredictable']},
    'industrial': {'category': 'character', 'opposites': ['organic', 'natural'], 'related': ['mechanical', 'metallic', 'harsh']},
    'immersive': {'category': 'spatial', 'opposites': ['narrow', 'focused'], 'related': ['enveloping', 'wide', 'atmospheric']},
    'isolated': {'category': 'structure', 'opposites': ['layered', 'complex'], 'related': ['sparse', 'minimal', 'solo']},
    'inflating': {'category': 'envelope', 'opposites': ['deflating', 'decaying'], 'related': ['swelling', 'rising', 'building']},
    'impulsive': {'category': 'envelope', 'opposites': ['sustained', 'gradual'], 'related': ['transient', 'punchy', 'sharp']},
    
    # J
    'jagged': {'category': 'texture', 'opposites': ['smooth', 'rounded'], 'related': ['sharp', 'angular', 'rough']},
    'jittery': {'category': 'movement', 'opposites': ['smooth', 'stable'], 'related': ['nervous', 'erratic', 'unstable']},
    'juicy': {'category': 'character', 'opposites': ['dry', 'thin'], 'related': ['fat', 'rich', 'saturated']},
    'jerky': {'category': 'movement', 'opposites': ['smooth', 'flowing'], 'related': ['choppy', 'staccato', 'abrupt']},
    'jumpy': {'category': 'movement', 'opposites': ['smooth', 'stable'], 'related': ['nervous', 'erratic', 'bouncy']},
    
    # K
    'kinetic': {'category': 'movement', 'opposites': ['static', 'still'], 'related': ['energetic', 'dynamic', 'moving']},
    'knocky': {'category': 'envelope', 'opposites': ['soft', 'sustained'], 'related': ['punchy', 'percussive', 'transient']},
    
    # L
    'lush': {'category': 'character', 'opposites': ['sparse', 'dry'], 'related': ['rich', 'full', 'reverberant']},
    'loose': {'category': 'rhythm', 'opposites': ['tight', 'precise'], 'related': ['relaxed', 'swinging', 'flowing']},
    'liquid': {'category': 'texture', 'opposites': ['solid', 'rigid'], 'related': ['flowing', 'smooth', 'wet']},
    'layered': {'category': 'structure', 'opposites': ['isolated', 'minimal'], 'related': ['complex', 'dense', 'rich']},
    'lifeless': {'category': 'character', 'opposites': ['alive', 'dynamic'], 'related': ['static', 'dull', 'flat']},
    'light': {'category': 'spectral', 'opposites': ['heavy', 'dark'], 'related': ['airy', 'bright', 'thin']},
    'long': {'category': 'envelope', 'opposites': ['short', 'tight'], 'related': ['sustained', 'decaying', 'reverberant']},
    'looping': {'category': 'movement', 'opposites': ['linear', 'one-shot'], 'related': ['repetitive', 'cyclic', 'hypnotic']},
    'lopsided': {'category': 'structure', 'opposites': ['balanced', 'symmetrical'], 'related': ['uneven', 'asymmetrical']},
    
    # M
    'metallic': {'category': 'texture', 'opposites': ['woody', 'organic'], 'related': ['bell-like', 'ringing', 'inharmonic']},
    'murky': {'category': 'spectral', 'opposites': ['clear', 'bright'], 'related': ['muddy', 'dark', 'cloudy']},
    'massive': {'category': 'character', 'opposites': ['small', 'thin'], 'related': ['huge', 'heavy', 'powerful']},
    'mellow': {'category': 'character', 'opposites': ['harsh', 'aggressive'], 'related': ['soft', 'warm', 'gentle']},
    'modulated': {'category': 'movement', 'opposites': ['static', 'fixed'], 'related': ['moving', 'wobbling', 'dynamic']},
    'moving': {'category': 'movement', 'opposites': ['static', 'still'], 'related': ['dynamic', 'evolving', 'alive']},
    'minimal': {'category': 'structure', 'opposites': ['complex', 'dense'], 'related': ['sparse', 'simple', 'clean']},
    'muddy': {'category': 'spectral', 'opposites': ['clear', 'crisp'], 'related': ['murky', 'cloudy', 'boomy']},
    'mechanical': {'category': 'character', 'opposites': ['organic', 'natural'], 'related': ['robotic', 'precise', 'industrial']},
    'melodic': {'category': 'pitch', 'opposites': ['atonal', 'noisy'], 'related': ['tonal', 'musical', 'harmonic']},
    
    # N
    'noisy': {'category': 'texture', 'opposites': ['clean', 'tonal'], 'related': ['dirty', 'textured', 'hissy']},
    'narrow': {'category': 'spectral', 'opposites': ['wide', 'full'], 'related': ['thin', 'focused', 'filtered']},
    'neutral': {'category': 'spectral', 'opposites': ['colored', 'extreme'], 'related': ['balanced', 'flat', 'natural']},
    'nasal': {'category': 'spectral', 'opposites': ['full', 'round'], 'related': ['honky', 'middy', 'thin']},
    'nervous': {'category': 'movement', 'opposites': ['calm', 'stable'], 'related': ['jittery', 'anxious', 'erratic']},
    'natural': {'category': 'character', 'opposites': ['synthetic', 'artificial'], 'related': ['organic', 'acoustic', 'real']},
    'nuanced': {'category': 'character', 'opposites': ['flat', 'simple'], 'related': ['subtle', 'detailed', 'complex']},
    'nebulous': {'category': 'spatial', 'opposites': ['focused', 'clear'], 'related': ['hazy', 'diffuse', 'vague']},
    
    # O
    'organic': {'category': 'character', 'opposites': ['synthetic', 'digital'], 'related': ['natural', 'acoustic', 'alive']},
    'open': {'category': 'spectral', 'opposites': ['filtered', 'muffled'], 'related': ['bright', 'full', 'clear']},
    'overdriven': {'category': 'texture', 'opposites': ['clean', 'pure'], 'related': ['distorted', 'saturated', 'hot']},
    'oscillating': {'category': 'movement', 'opposites': ['static', 'stable'], 'related': ['wobbling', 'pulsing', 'vibrating']},
    'oppressive': {'category': 'character', 'opposites': ['light', 'airy'], 'related': ['heavy', 'dense', 'dark']},
    'opaque': {'category': 'texture', 'opposites': ['transparent', 'clear'], 'related': ['dense', 'thick', 'solid']},
    
    # P
    'punchy': {'category': 'envelope', 'opposites': ['soft', 'weak'], 'related': ['impactful', 'transient', 'tight']},
    'pulsing': {'category': 'movement', 'opposites': ['static', 'sustained'], 'related': ['rhythmic', 'throbbing', 'beating']},
    'percussive': {'category': 'envelope', 'opposites': ['sustained', 'smooth'], 'related': ['transient', 'punchy', 'rhythmic']},
    'polished': {'category': 'texture', 'opposites': ['raw', 'rough'], 'related': ['smooth', 'clean', 'refined']},
    'prickly': {'category': 'texture', 'opposites': ['smooth', 'soft'], 'related': ['sharp', 'harsh', 'pointy']},
    'powerful': {'category': 'character', 'opposites': ['weak', 'thin'], 'related': ['strong', 'massive', 'impactful']},
    'pressured': {'category': 'dynamics', 'opposites': ['relaxed', 'open'], 'related': ['compressed', 'tight', 'intense']},
    'processed': {'category': 'character', 'opposites': ['natural', 'raw'], 'related': ['synthetic', 'artificial', 'effected']},
    'playful': {'category': 'character', 'opposites': ['serious', 'dark'], 'related': ['bouncy', 'light', 'fun']},
    'primitive': {'category': 'character', 'opposites': ['complex', 'refined'], 'related': ['raw', 'simple', 'basic']},
    
    # Q
    'quiet': {'category': 'dynamics', 'opposites': ['loud', 'powerful'], 'related': ['soft', 'subtle', 'whispered']},
    'quantized': {'category': 'rhythm', 'opposites': ['loose', 'human'], 'related': ['precise', 'locked', 'mechanical']},
    'quivering': {'category': 'movement', 'opposites': ['stable', 'steady'], 'related': ['trembling', 'vibrating', 'nervous']},
    'quick': {'category': 'envelope', 'opposites': ['slow', 'long'], 'related': ['fast', 'snappy', 'short']},
    'quirky': {'category': 'character', 'opposites': ['normal', 'standard'], 'related': ['unusual', 'weird', 'playful']},
    
    # R
    'rough': {'category': 'texture', 'opposites': ['smooth', 'polished'], 'related': ['gritty', 'raw', 'textured']},
    'resonant': {'category': 'spectral', 'opposites': ['damped', 'dry'], 'related': ['ringing', 'sustaining', 'harmonic']},
    'roomy': {'category': 'spatial', 'opposites': ['dry', 'intimate'], 'related': ['spacious', 'reverberant', 'ambient']},
    'rhythmic': {'category': 'rhythm', 'opposites': ['atonal', 'sustained'], 'related': ['pulsing', 'groovy', 'percussive']},
    'raw': {'category': 'texture', 'opposites': ['polished', 'processed'], 'related': ['unprocessed', 'rough', 'natural']},
    'restrained': {'category': 'character', 'opposites': ['extreme', 'exaggerated'], 'related': ['subtle', 'controlled', 'understated']},
    'rolling': {'category': 'movement', 'opposites': ['static', 'staccato'], 'related': ['flowing', 'continuous', 'wave-like']},
    'ringing': {'category': 'envelope', 'opposites': ['damped', 'short'], 'related': ['resonant', 'sustaining', 'metallic']},
    'reactive': {'category': 'movement', 'opposites': ['static', 'fixed'], 'related': ['responsive', 'dynamic', 'alive']},
    'rubbery': {'category': 'texture', 'opposites': ['rigid', 'brittle'], 'related': ['elastic', 'bouncy', 'flexible']},
    
    # S
    'smooth': {'category': 'texture', 'opposites': ['rough', 'gritty'], 'related': ['polished', 'clean', 'flowing']},
    'sharp': {'category': 'envelope', 'opposites': ['soft', 'blunt'], 'related': ['crisp', 'transient', 'precise']},
    'smeared': {'category': 'texture', 'opposites': ['crisp', 'articulated'], 'related': ['blurred', 'diffuse', 'washy']},
    'spacious': {'category': 'spatial', 'opposites': ['tight', 'narrow'], 'related': ['wide', 'open', 'reverberant']},
    'static': {'category': 'movement', 'opposites': ['dynamic', 'evolving'], 'related': ['fixed', 'unchanging', 'still']},
    'synthetic': {'category': 'character', 'opposites': ['organic', 'natural'], 'related': ['artificial', 'digital', 'electronic']},
    'soft': {'category': 'character', 'opposites': ['hard', 'aggressive'], 'related': ['gentle', 'mellow', 'quiet']},
    'stiff': {'category': 'texture', 'opposites': ['loose', 'flexible'], 'related': ['rigid', 'mechanical', 'tight']},
    'saturated': {'category': 'texture', 'opposites': ['clean', 'pure'], 'related': ['warm', 'colored', 'driven']},
    'scattered': {'category': 'spatial', 'opposites': ['focused', 'centered'], 'related': ['diffuse', 'spread', 'wide']},
    
    # T
    'thick': {'category': 'spectral', 'opposites': ['thin', 'narrow'], 'related': ['fat', 'dense', 'full']},
    'thin': {'category': 'spectral', 'opposites': ['thick', 'fat'], 'related': ['narrow', 'light', 'weak']},
    'textured': {'category': 'texture', 'opposites': ['smooth', 'clean'], 'related': ['granular', 'rough', 'complex']},
    'tight': {'category': 'envelope', 'opposites': ['loose', 'roomy'], 'related': ['controlled', 'punchy', 'precise']},
    'trembling': {'category': 'movement', 'opposites': ['stable', 'steady'], 'related': ['vibrating', 'quivering', 'nervous']},
    'tonal': {'category': 'pitch', 'opposites': ['noisy', 'atonal'], 'related': ['pitched', 'harmonic', 'melodic']},
    'transient': {'category': 'envelope', 'opposites': ['sustained', 'soft'], 'related': ['punchy', 'sharp', 'percussive']},
    'tense': {'category': 'character', 'opposites': ['relaxed', 'calm'], 'related': ['anxious', 'tight', 'nervous']},
    'transparent': {'category': 'texture', 'opposites': ['opaque', 'dense'], 'related': ['clear', 'open', 'airy']},
    'turbulent': {'category': 'movement', 'opposites': ['calm', 'smooth'], 'related': ['chaotic', 'rough', 'wild']},
    
    # U
    'unstable': {'category': 'movement', 'opposites': ['stable', 'steady'], 'related': ['wobbly', 'shaky', 'unpredictable']},
    'uneven': {'category': 'structure', 'opposites': ['even', 'balanced'], 'related': ['asymmetrical', 'irregular', 'lopsided']},
    'understated': {'category': 'character', 'opposites': ['exaggerated', 'bold'], 'related': ['subtle', 'restrained', 'quiet']},
    'unfiltered': {'category': 'spectral', 'opposites': ['filtered', 'processed'], 'related': ['raw', 'open', 'full']},
    'unnatural': {'category': 'character', 'opposites': ['natural', 'organic'], 'related': ['synthetic', 'artificial', 'weird']},
    'unpredictable': {'category': 'movement', 'opposites': ['predictable', 'regular'], 'related': ['chaotic', 'erratic', 'random']},
    
    # V/W (combined as in original)
    'warm': {'category': 'character', 'opposites': ['cold', 'icy'], 'related': ['soft', 'analog', 'rich']},
    'wide': {'category': 'spatial', 'opposites': ['narrow', 'mono'], 'related': ['stereo', 'spacious', 'immersive']},
    'vibrant': {'category': 'character', 'opposites': ['dull', 'lifeless'], 'related': ['alive', 'colorful', 'energetic']},
    'volatile': {'category': 'movement', 'opposites': ['stable', 'steady'], 'related': ['unstable', 'explosive', 'unpredictable']},
    'vocal': {'category': 'character', 'opposites': ['instrumental'], 'related': ['voice-like', 'human', 'formant']},
    'veiled': {'category': 'spectral', 'opposites': ['clear', 'open'], 'related': ['muffled', 'hazy', 'filtered']},
    'viscous': {'category': 'texture', 'opposites': ['thin', 'watery'], 'related': ['thick', 'dense', 'heavy']},
    'vivid': {'category': 'character', 'opposites': ['dull', 'muted'], 'related': ['bright', 'colorful', 'intense']},
    'wet': {'category': 'spatial', 'opposites': ['dry', 'direct'], 'related': ['reverberant', 'effected', 'processed']},
    'wobbly': {'category': 'movement', 'opposites': ['stable', 'steady'], 'related': ['unstable', 'modulated', 'detuned']},
    'warped': {'category': 'pitch', 'opposites': ['stable', 'pure'], 'related': ['bent', 'distorted', 'twisted']},
    'weighty': {'category': 'spectral', 'opposites': ['light', 'thin'], 'related': ['heavy', 'deep', 'massive']},
    'whispery': {'category': 'texture', 'opposites': ['loud', 'bold'], 'related': ['soft', 'breathy', 'quiet']},
    'washed': {'category': 'spatial', 'opposites': ['dry', 'direct'], 'related': ['reverberant', 'blurred', 'smeared']},
    'wandering': {'category': 'movement', 'opposites': ['fixed', 'stable'], 'related': ['drifting', 'evolving', 'moving']},
    'wooly': {'category': 'texture', 'opposites': ['crisp', 'clean'], 'related': ['fuzzy', 'soft', 'warm']},
    
    # X/Y/Z
    'extreme': {'category': 'character', 'opposites': ['mild', 'subtle'], 'related': ['intense', 'exaggerated', 'bold']},
    'experimental': {'category': 'character', 'opposites': ['conventional', 'standard'], 'related': ['unusual', 'weird', 'avant-garde']},
    'yielding': {'category': 'texture', 'opposites': ['rigid', 'hard'], 'related': ['soft', 'flexible', 'gentle']},
    'zippy': {'category': 'envelope', 'opposites': ['slow', 'sluggish'], 'related': ['fast', 'snappy', 'energetic']},
    'zoned': {'category': 'character', 'opposites': ['clear', 'focused'], 'related': ['hazy', 'dreamy', 'hypnotic']},
}


# Categories for organization
DESCRIPTOR_CATEGORIES = {
    'spectral': 'Frequency content and tonal balance',
    'texture': 'Surface quality and grain',
    'envelope': 'Attack, decay, and dynamics over time',
    'dynamics': 'Loudness and compression',
    'spatial': 'Stereo width, depth, and reverb',
    'movement': 'Evolution and modulation',
    'character': 'Overall feel and mood',
    'pitch': 'Pitch clarity and stability',
    'rhythm': 'Rhythmic content and feel',
    'structure': 'Arrangement and complexity',
}


# ============================================================================
# DESCRIPTOR PROFILE
# ============================================================================

@dataclass
class DescriptorProfile:
    """A probabilistic profile of descriptors for a sample."""
    
    descriptors: Dict[str, float] = field(default_factory=dict)  # name -> confidence (0-1)
    duration: float = 0.0
    source: str = ""
    
    def set(self, name: str, confidence: float) -> None:
        """Set descriptor confidence (0-1)."""
        if name in DESCRIPTOR_VOCABULARY:
            self.descriptors[name] = max(0.0, min(1.0, confidence))
    
    def get(self, name: str) -> float:
        """Get descriptor confidence."""
        return self.descriptors.get(name, 0.0)
    
    def top(self, n: int = 10, min_confidence: float = 0.3) -> List[Tuple[str, float]]:
        """Get top N descriptors above minimum confidence."""
        filtered = [(k, v) for k, v in self.descriptors.items() if v >= min_confidence]
        return sorted(filtered, key=lambda x: x[1], reverse=True)[:n]
    
    def by_category(self, category: str) -> List[Tuple[str, float]]:
        """Get descriptors in a category."""
        return [
            (name, conf) for name, conf in self.descriptors.items()
            if DESCRIPTOR_VOCABULARY.get(name, {}).get('category') == category
            and conf > 0.2
        ]
    
    def text_summary(self, max_descriptors: int = 8) -> str:
        """Generate natural language summary."""
        top_descs = self.top(max_descriptors, min_confidence=0.4)
        if not top_descs:
            return "No strong characteristics detected."
        
        # Group by confidence level
        high = [d for d, c in top_descs if c >= 0.7]
        medium = [d for d, c in top_descs if 0.5 <= c < 0.7]
        low = [d for d, c in top_descs if c < 0.5]
        
        parts = []
        if high:
            parts.append(f"Strongly {', '.join(high)}")
        if medium:
            parts.append(f"moderately {', '.join(medium)}")
        if low:
            parts.append(f"slightly {', '.join(low)}")
        
        return "; ".join(parts) + "."
    
    def similarity(self, other: 'DescriptorProfile') -> float:
        """Calculate similarity to another profile (0-1)."""
        common = set(self.descriptors.keys()) & set(other.descriptors.keys())
        if not common:
            # Use all keys from both
            all_keys = set(self.descriptors.keys()) | set(other.descriptors.keys())
            if not all_keys:
                return 0.0
            common = all_keys
        
        vec1 = np.array([self.descriptors.get(k, 0) for k in common])
        vec2 = np.array([other.descriptors.get(k, 0) for k in common])
        
        # Cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))


# ============================================================================
# ATTRIBUTE TO DESCRIPTOR MAPPING
# ============================================================================

def attributes_to_descriptors(attributes: Dict[str, float]) -> DescriptorProfile:
    """Convert numeric attributes to descriptor profile.
    
    Maps measured attributes (brightness=75, attack=90) to fuzzy
    descriptors (bright, punchy, aggressive).
    """
    profile = DescriptorProfile()
    
    # Spectral mappings
    brightness = attributes.get('brightness', 50)
    darkness = attributes.get('darkness', 50)
    warmth = attributes.get('warmth', 50)
    harshness = attributes.get('harshness', 50)
    
    if brightness > 70:
        profile.set('bright', (brightness - 70) / 30)
        profile.set('airy', (brightness - 70) / 40)
    if brightness < 30:
        profile.set('dark', (30 - brightness) / 30)
        profile.set('dull', (30 - brightness) / 40)
    if darkness > 60:
        profile.set('deep', (darkness - 60) / 40)
        profile.set('heavy', (darkness - 60) / 50)
    if warmth > 60:
        profile.set('warm', (warmth - 60) / 40)
        profile.set('analog', (warmth - 60) / 50)
    if warmth < 40:
        profile.set('cold', (40 - warmth) / 40)
    if harshness > 60:
        profile.set('harsh', (harshness - 60) / 40)
        profile.set('aggressive', (harshness - 60) / 50)
    
    # Envelope mappings
    attack = attributes.get('attack', 50)
    transient = attributes.get('transient_strength', 50)
    sustain = attributes.get('sustain', 50)
    release = attributes.get('release', 50)
    
    if attack > 70:
        profile.set('punchy', (attack - 70) / 30)
        profile.set('sharp', (attack - 70) / 35)
        profile.set('transient', (attack - 70) / 40)
    if attack < 30:
        profile.set('soft', (30 - attack) / 30)
        profile.set('gentle', (30 - attack) / 40)
    if transient > 70:
        profile.set('percussive', (transient - 70) / 30)
        profile.set('impulsive', (transient - 70) / 40)
    if sustain > 60:
        profile.set('sustained', (sustain - 60) / 40)
    if release > 60:
        profile.set('long', (release - 60) / 40)
        profile.set('reverberant', (release - 60) / 50)
    
    # Texture mappings
    noisiness = attributes.get('noisiness', 50)
    distortion = attributes.get('distortion', 50)
    complexity = attributes.get('complexity', 50)
    granularity = attributes.get('granularity', 50)
    
    if noisiness > 60:
        profile.set('noisy', (noisiness - 60) / 40)
        profile.set('textured', (noisiness - 60) / 50)
    if noisiness < 30:
        profile.set('clean', (30 - noisiness) / 30)
        profile.set('tonal', (30 - noisiness) / 40)
    if distortion > 50:
        profile.set('distorted', (distortion - 50) / 50)
        profile.set('saturated', (distortion - 50) / 60)
    if complexity > 60:
        profile.set('complex', (complexity - 60) / 40)
        profile.set('layered', (complexity - 60) / 50)
    if granularity > 60:
        profile.set('granular', (granularity - 60) / 40)
        profile.set('gritty', (granularity - 60) / 50)
    
    # Character derivations
    aggressive = attributes.get('aggressive', 50)
    metallic = attributes.get('metallic', 50)
    digital = attributes.get('digital', 50)
    
    if aggressive > 60:
        profile.set('aggressive', (aggressive - 60) / 40)
        profile.set('intense', (aggressive - 60) / 50)
    if aggressive < 40:
        profile.set('gentle', (40 - aggressive) / 40)
        profile.set('soft', (40 - aggressive) / 50)
    if metallic > 60:
        profile.set('metallic', (metallic - 60) / 40)
        profile.set('ringing', (metallic - 60) / 50)
    if digital > 60:
        profile.set('digital', (digital - 60) / 40)
        profile.set('synthetic', (digital - 60) / 50)
    if digital < 40:
        profile.set('analog', (40 - digital) / 40)
        profile.set('organic', (40 - digital) / 50)
    
    # Movement mappings
    spectral_flux = attributes.get('spectral_flux', 50)
    rhythmic = attributes.get('rhythmic', 50)
    
    if spectral_flux > 60:
        profile.set('evolving', (spectral_flux - 60) / 40)
        profile.set('dynamic', (spectral_flux - 60) / 50)
    if spectral_flux < 30:
        profile.set('static', (30 - spectral_flux) / 30)
    if rhythmic > 60:
        profile.set('rhythmic', (rhythmic - 60) / 40)
        profile.set('pulsing', (rhythmic - 60) / 50)
    
    return profile


def format_descriptor_profile(profile: DescriptorProfile) -> str:
    """Format descriptor profile as readable text."""
    lines = [
        "=== DESCRIPTOR PROFILE ===",
        f"Source: {profile.source or 'buffer'}",
        "",
        "SUMMARY:",
        f"  {profile.text_summary()}",
        "",
        "TOP DESCRIPTORS:",
    ]
    
    for desc, conf in profile.top(12, min_confidence=0.3):
        bar = "█" * int(conf * 10)
        info = DESCRIPTOR_VOCABULARY.get(desc, {})
        cat = info.get('category', '?')
        lines.append(f"  {desc}: {conf:.0%} {bar} [{cat}]")
    
    lines.append("")
    lines.append("BY CATEGORY:")
    for cat in ['spectral', 'texture', 'envelope', 'character', 'movement']:
        cat_descs = profile.by_category(cat)
        if cat_descs:
            desc_str = ", ".join(f"{d}({c:.0%})" for d, c in cat_descs[:4])
            lines.append(f"  {cat}: {desc_str}")
    
    return '\n'.join(lines)
