"""Generator Commands for MDMA - Audio Synthesis One-Shots.

These commands generate various sound effects, percussion, and utility sounds.
All generators output to the working buffer or can be appended to buffers.

BUILD ID: generator_cmds_v1.0
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.session import Session


def _set_audio(session: "Session", audio: np.ndarray, source: str) -> str:
    """Set generated audio to working buffer and last_buffer."""
    # Set to working buffer if available
    try:
        from .working_cmds import get_working_buffer
        wb = get_working_buffer()
        wb.set(audio, source, session.sample_rate)
    except ImportError:
        pass
    
    # Also set last_buffer for compatibility
    session.last_buffer = audio
    
    # Calculate metrics
    dur = len(audio) / session.sample_rate
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    
    return f"OK: {source}\n  {dur:.3f}s, peak={peak:.3f}, rms={rms:.3f}"


def _envelope(length: int, attack: float, decay: float, sr: int) -> np.ndarray:
    """Generate simple attack-decay envelope."""
    attack_samples = int(attack * sr)
    decay_samples = int(decay * sr)
    
    env = np.ones(length)
    
    # Attack
    if attack_samples > 0 and attack_samples < length:
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Decay
    if decay_samples > 0:
        decay_start = attack_samples
        decay_end = min(length, decay_start + decay_samples)
        if decay_end > decay_start:
            env[decay_start:decay_end] = np.linspace(1, 0, decay_end - decay_start)
            env[decay_end:] = 0
    
    return env


def _noise(length: int) -> np.ndarray:
    """Generate white noise."""
    return np.random.randn(length)


def _pink_noise(length: int) -> np.ndarray:
    """Generate pink noise using Voss-McCartney algorithm."""
    # Simple approximation
    white = np.random.randn(length)
    
    # Apply lowpass filtering via cumulative sum
    pink = np.zeros(length)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    
    # Simple IIR approximation
    from scipy.signal import lfilter
    pink = lfilter([0.04, 0.1, 0.3, 0.5, 0.3, 0.1], [1.0], white)
    
    # Normalize
    peak = np.max(np.abs(pink))
    if peak > 0:
        pink /= peak
    
    return pink


# ============================================================================
# PERCUSSION GENERATORS
# ============================================================================

def cmd_g_kick(session: "Session", args: List[str]) -> str:
    """Generate kick drum.
    
    Usage:
      /g_kick [type]     Generate kick
      
    Types: 1=808, 2=punchy, 3=sub, 4=acoustic
    """
    sr = session.sample_rate
    kick_type = int(args[0]) if args and args[0].isdigit() else 1
    
    duration = 0.5
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    if kick_type == 1:  # 808
        # Pitch drop from 150Hz to 50Hz
        freq = 150 * np.exp(-t * 10) + 50
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = np.sin(phase)
        audio *= _envelope(length, 0.002, 0.4, sr)
        
    elif kick_type == 2:  # Punchy
        freq = 200 * np.exp(-t * 15) + 60
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = np.sin(phase)
        # Add click
        click = _noise(int(0.01 * sr)) * 0.3
        audio[:len(click)] += click
        audio *= _envelope(length, 0.001, 0.3, sr)
        
    elif kick_type == 3:  # Sub
        freq = 80 * np.exp(-t * 5) + 40
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = np.sin(phase)
        audio *= _envelope(length, 0.005, 0.5, sr)
        
    else:  # Acoustic-ish
        freq = 180 * np.exp(-t * 12) + 55
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = np.sin(phase)
        # Add harmonics
        audio += 0.3 * np.sin(2 * phase) * np.exp(-t * 20)
        audio *= _envelope(length, 0.001, 0.35, sr)
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    return _set_audio(session, audio, f"kick_{kick_type}")


def cmd_g_snare(session: "Session", args: List[str]) -> str:
    """Generate snare drum.
    
    Usage:
      /g_snare [type]    Generate snare
      
    Types: 1=808, 2=acoustic, 3=tight, 4=clicky
    """
    sr = session.sample_rate
    snare_type = int(args[0]) if args and args[0].isdigit() else 1
    
    duration = 0.3
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    # Body - pitched sine
    body_freq = 180 if snare_type == 1 else 200
    body = np.sin(2 * np.pi * body_freq * t) * np.exp(-t * 25)
    
    # Snares - filtered noise
    noise = _noise(length)
    noise_env = np.exp(-t * 15) if snare_type == 1 else np.exp(-t * 30)
    snares = noise * noise_env
    
    # Mix
    if snare_type == 1:  # 808
        audio = body * 0.6 + snares * 0.5
    elif snare_type == 2:  # Acoustic
        audio = body * 0.4 + snares * 0.7
    elif snare_type == 3:  # Tight
        audio = body * 0.5 + snares * 0.4
        audio *= _envelope(length, 0.001, 0.15, sr)
    else:  # Clicky
        click = np.zeros(length)
        click[:int(0.005 * sr)] = 1.0
        audio = body * 0.3 + snares * 0.5 + click * 0.3
    
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    return _set_audio(session, audio, f"snare_{snare_type}")


def cmd_g_hat(session: "Session", args: List[str]) -> str:
    """Generate hi-hat.
    
    Usage:
      /g_hat [type]      Generate hi-hat
      
    Types: 1=closed, 2=open, 3=pedal, 4=crispy
    """
    sr = session.sample_rate
    hat_type = int(args[0]) if args and args[0].isdigit() else 1
    
    if hat_type == 2:  # Open
        duration = 0.5
    else:
        duration = 0.1
    
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    # Metallic noise
    noise = _noise(length)
    
    # Highpass by differencing
    noise = np.diff(np.concatenate([[0], noise]))
    
    # Envelope
    if hat_type == 1:  # Closed
        env = np.exp(-t * 50)
    elif hat_type == 2:  # Open
        env = np.exp(-t * 8)
    elif hat_type == 3:  # Pedal
        env = np.exp(-t * 30)
    else:  # Crispy
        env = np.exp(-t * 80)
    
    audio = noise * env
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return _set_audio(session, audio, f"hat_{hat_type}")


def cmd_g_tom(session: "Session", args: List[str]) -> str:
    """Generate tom drum.
    
    Usage:
      /g_tom [type]      Generate tom
      
    Types: 1=low, 2=mid, 3=high, 4=electronic
    """
    sr = session.sample_rate
    tom_type = int(args[0]) if args and args[0].isdigit() else 2
    
    duration = 0.4
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    # Pitch based on type
    if tom_type == 1:  # Low
        freq = 80
    elif tom_type == 2:  # Mid  
        freq = 120
    elif tom_type == 3:  # High
        freq = 180
    else:  # Electronic
        freq = 100
    
    # Pitch drop
    freq_mod = freq * (1 + 0.5 * np.exp(-t * 15))
    phase = 2 * np.pi * np.cumsum(freq_mod) / sr
    
    audio = np.sin(phase)
    
    if tom_type == 4:  # Electronic - add harmonics
        audio += 0.3 * np.sin(2 * phase)
    
    audio *= _envelope(length, 0.002, 0.35, sr)
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    return _set_audio(session, audio, f"tom_{tom_type}")


def cmd_g_clp(session: "Session", args: List[str]) -> str:
    """Generate clap sound.
    
    Usage:
      /g_clp [type]      Generate clap
      
    Types: 1=808, 2=acoustic, 3=layered, 4=tight
    """
    sr = session.sample_rate
    clap_type = int(args[0]) if args and args[0].isdigit() else 1
    
    duration = 0.3
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    # Multiple noise bursts for clap texture
    audio = np.zeros(length)
    
    if clap_type == 1:  # 808
        # Several short bursts
        for i, offset in enumerate([0, 0.01, 0.02, 0.03]):
            start = int(offset * sr)
            burst_len = int(0.02 * sr)
            if start + burst_len < length:
                burst = _noise(burst_len) * (0.7 + 0.3 * (i / 3))
                audio[start:start + burst_len] += burst
        
        # Tail
        tail_start = int(0.04 * sr)
        tail = _noise(length - tail_start) * np.exp(-np.linspace(0, 1, length - tail_start) * 10)
        audio[tail_start:] += tail * 0.5
        
    elif clap_type == 2:  # Acoustic
        # Filtered noise
        noise = _noise(length)
        audio = noise * np.exp(-t * 20)
        
    elif clap_type == 3:  # Layered
        for offset in [0, 0.005, 0.015]:
            start = int(offset * sr)
            audio[start:] += _noise(length - start)[:length - start] * np.exp(-t[:length - start] * 15)
        audio *= 0.5
        
    else:  # Tight
        audio = _noise(length) * np.exp(-t * 40)
    
    audio = audio / np.max(np.abs(audio)) * 0.85
    
    return _set_audio(session, audio, f"clap_{clap_type}")


def cmd_g_cym(session: "Session", args: List[str]) -> str:
    """Generate cymbal sound.
    
    Usage:
      /g_cym [type]      Generate cymbal
      
    Types: 1=crash, 2=ride, 3=china, 4=splash
    """
    sr = session.sample_rate
    cym_type = int(args[0]) if args and args[0].isdigit() else 1
    
    if cym_type == 1:  # Crash
        duration = 2.0
        decay = 3
    elif cym_type == 2:  # Ride
        duration = 1.5
        decay = 5
    elif cym_type == 3:  # China
        duration = 1.0
        decay = 4
    else:  # Splash
        duration = 0.5
        decay = 8
    
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    # Metallic noise
    noise = _noise(length)
    
    # Add some pitched components for shimmer
    shimmer = np.sin(2 * np.pi * 3000 * t) * 0.1
    shimmer += np.sin(2 * np.pi * 5000 * t) * 0.05
    shimmer += np.sin(2 * np.pi * 7000 * t) * 0.03
    
    audio = noise * 0.8 + shimmer
    audio *= np.exp(-t * decay)
    
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return _set_audio(session, audio, f"cymbal_{cym_type}")


def cmd_g_snp(session: "Session", args: List[str]) -> str:
    """Generate finger snap.
    
    Usage:
      /g_snp [type]      Generate snap
      
    Types: 1=dry, 2=reverb
    """
    sr = session.sample_rate
    snap_type = int(args[0]) if args and args[0].isdigit() else 1
    
    duration = 0.15 if snap_type == 1 else 0.4
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    # Short click with body
    click_len = int(0.005 * sr)
    click = _noise(click_len)
    
    # Body - resonant
    body = np.sin(2 * np.pi * 1500 * t) * np.exp(-t * 50)
    
    audio = np.zeros(length)
    audio[:click_len] = click * 0.5
    audio += body * 0.8
    
    if snap_type == 2:  # Add reverb-like tail
        tail = _noise(length) * np.exp(-t * 10) * 0.2
        audio += tail
    
    audio = audio / np.max(np.abs(audio)) * 0.85
    
    return _set_audio(session, audio, f"snap_{snap_type}")


def cmd_g_shk(session: "Session", args: List[str]) -> str:
    """Generate shaker.
    
    Usage:
      /g_shk [type]      Generate shaker
      
    Types: 1=16th, 2=8th, 3=triplet
    """
    sr = session.sample_rate
    shk_type = int(args[0]) if args and args[0].isdigit() else 1
    
    duration = 0.1
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    # Filtered noise
    noise = _noise(length)
    
    # Different envelopes for different feels
    if shk_type == 1:  # 16th - tight
        env = np.exp(-t * 40)
    elif shk_type == 2:  # 8th - moderate
        env = np.exp(-t * 25)
    else:  # Triplet - looser
        env = np.exp(-t * 20)
    
    audio = noise * env
    audio = audio / np.max(np.abs(audio)) * 0.7
    
    return _set_audio(session, audio, f"shaker_{shk_type}")


# ============================================================================
# FX GENERATORS
# ============================================================================

def cmd_g_zap(session: "Session", args: List[str]) -> str:
    """Generate zap/laser sound.
    
    Usage:
      /g_zap [type]      Generate zap
      
    Types: 1=short, 2=long, 3=sweep, 4=retro
    """
    sr = session.sample_rate
    zap_type = int(args[0]) if args and args[0].isdigit() else 1
    
    if zap_type == 1:
        duration = 0.1
    elif zap_type == 2:
        duration = 0.3
    else:
        duration = 0.2
    
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    if zap_type == 4:  # Retro - square wave
        # Pitch drop
        freq = 2000 * np.exp(-t * 20) + 100
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = np.sign(np.sin(phase))
    else:
        # Sine with pitch drop
        if zap_type == 1:
            freq = 3000 * np.exp(-t * 30) + 200
        elif zap_type == 2:
            freq = 2000 * np.exp(-t * 10) + 150
        else:  # Sweep
            freq = 4000 * np.exp(-t * 15) + 100
        
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = np.sin(phase)
    
    audio *= _envelope(length, 0.001, duration * 0.8, sr)
    audio = audio / np.max(np.abs(audio)) * 0.85
    
    return _set_audio(session, audio, f"zap_{zap_type}")


def cmd_g_lsr(session: "Session", args: List[str]) -> str:
    """Generate laser FX.
    
    Usage:
      /g_lsr [type]      Generate laser
      
    Types: 1=pew, 2=beam, 3=charge
    """
    sr = session.sample_rate
    lsr_type = int(args[0]) if args and args[0].isdigit() else 1
    
    if lsr_type == 3:  # Charge
        duration = 1.0
    else:
        duration = 0.3
    
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    if lsr_type == 1:  # Pew
        freq = 5000 * np.exp(-t * 15) + 200
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = np.sin(phase) * np.exp(-t * 10)
        
    elif lsr_type == 2:  # Beam
        freq = 1000 + 500 * np.sin(2 * np.pi * 30 * t)  # Wobble
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = np.sin(phase) * np.exp(-t * 3)
        
    else:  # Charge
        # Rising pitch
        freq = 100 * np.exp(t * 3)
        freq = np.minimum(freq, 8000)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = np.sin(phase)
        audio *= np.linspace(0, 1, length)  # Fade in
    
    audio = audio / np.max(np.abs(audio)) * 0.85
    
    return _set_audio(session, audio, f"laser_{lsr_type}")


def cmd_g_bel(session: "Session", args: List[str]) -> str:
    """Generate bell/ping.
    
    Usage:
      /g_bel [type]      Generate bell
      
    Types: 1=bright, 2=dark, 3=tubular, 4=glass
    """
    sr = session.sample_rate
    bel_type = int(args[0]) if args and args[0].isdigit() else 1
    
    duration = 2.0
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    if bel_type == 1:  # Bright
        freq = 880
        harmonics = [1, 2.4, 3.8, 5.2]
        amps = [1, 0.6, 0.4, 0.2]
    elif bel_type == 2:  # Dark
        freq = 440
        harmonics = [1, 1.5, 2.1, 2.8]
        amps = [1, 0.4, 0.3, 0.1]
    elif bel_type == 3:  # Tubular
        freq = 220
        harmonics = [1, 2.76, 5.4, 8.93]
        amps = [1, 0.5, 0.3, 0.2]
    else:  # Glass
        freq = 1200
        harmonics = [1, 2.2, 3.4]
        amps = [1, 0.3, 0.15]
    
    audio = np.zeros(length)
    for h, a in zip(harmonics, amps):
        decay = 3 + h * 2  # Higher harmonics decay faster
        audio += a * np.sin(2 * np.pi * freq * h * t) * np.exp(-t * decay)
    
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return _set_audio(session, audio, f"bell_{bel_type}")


def cmd_g_bas(session: "Session", args: List[str]) -> str:
    """Generate bass hit.
    
    Usage:
      /g_bas [type]      Generate bass
      
    Types: 1=808, 2=sub, 3=punch, 4=growl
    """
    sr = session.sample_rate
    bas_type = int(args[0]) if args and args[0].isdigit() else 1
    
    duration = 0.5
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    if bas_type == 1:  # 808
        freq = 60
        audio = np.sin(2 * np.pi * freq * t)
        audio *= _envelope(length, 0.002, 0.4, sr)
        
    elif bas_type == 2:  # Sub
        freq = 40
        audio = np.sin(2 * np.pi * freq * t)
        audio *= _envelope(length, 0.005, 0.5, sr)
        
    elif bas_type == 3:  # Punch
        freq = 80
        audio = np.sin(2 * np.pi * freq * t)
        # Add click
        click = _noise(int(0.01 * sr)) * 0.3
        audio[:len(click)] += click
        audio *= _envelope(length, 0.001, 0.2, sr)
        
    else:  # Growl
        freq = 55
        # FM for growl
        mod = np.sin(2 * np.pi * 30 * t) * 20
        audio = np.sin(2 * np.pi * (freq + mod) * t)
        audio *= _envelope(length, 0.005, 0.4, sr)
    
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    return _set_audio(session, audio, f"bass_{bas_type}")


def cmd_g_rsr(session: "Session", args: List[str]) -> str:
    """Generate riser.
    
    Usage:
      /g_rsr [type]      Generate riser
      
    Types: 1=noise, 2=tonal, 3=sweep, 4=tension
    """
    sr = session.sample_rate
    rsr_type = int(args[0]) if args and args[0].isdigit() else 1
    
    duration = 4.0
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    if rsr_type == 1:  # Noise
        audio = _pink_noise(length)
        # Rising filter
        audio *= np.linspace(0.1, 1, length)
        
    elif rsr_type == 2:  # Tonal
        # Rising pitch
        freq = 100 * (2 ** (t / duration * 3))  # 3 octaves
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = np.sin(phase) + 0.5 * np.sin(2 * phase)
        audio *= np.linspace(0.3, 1, length)
        
    elif rsr_type == 3:  # Sweep
        freq = 50 * (2 ** (t / duration * 4))
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = np.sin(phase)
        audio *= np.linspace(0.2, 1, length)
        
    else:  # Tension
        # Noise with rising filter + tone
        noise = _pink_noise(length) * np.linspace(0.1, 0.5, length)
        freq = 200 * (2 ** (t / duration * 2))
        phase = 2 * np.pi * np.cumsum(freq) / sr
        tone = np.sin(phase) * np.linspace(0.2, 0.8, length)
        audio = noise + tone
    
    audio = audio / np.max(np.abs(audio)) * 0.85
    
    return _set_audio(session, audio, f"riser_{rsr_type}")


def cmd_g_dwn(session: "Session", args: List[str]) -> str:
    """Generate downlifter.
    
    Usage:
      /g_dwn [type]      Generate downlifter
      
    Types: 1=drop, 2=sweep, 3=impact, 4=reverse
    """
    sr = session.sample_rate
    dwn_type = int(args[0]) if args and args[0].isdigit() else 1
    
    duration = 2.0
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    if dwn_type == 1:  # Drop
        freq = 2000 * np.exp(-t * 2) + 50
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = np.sin(phase)
        audio *= np.exp(-t * 1.5)
        
    elif dwn_type == 2:  # Sweep
        freq = 4000 * np.exp(-t * 3) + 30
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = np.sin(phase)
        audio *= np.exp(-t * 1)
        
    elif dwn_type == 3:  # Impact
        # Short boom with tail
        freq = 200 * np.exp(-t * 5) + 40
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = np.sin(phase)
        audio *= np.exp(-t * 3)
        # Add noise tail
        audio += _pink_noise(length) * np.exp(-t * 2) * 0.3
        
    else:  # Reverse
        # Generate forward then reverse
        freq = 50 * (2 ** (t / duration * 2))
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = np.sin(phase)
        audio *= np.linspace(0, 1, length)
        audio = audio[::-1]  # Reverse
    
    audio = audio / np.max(np.abs(audio)) * 0.85
    
    return _set_audio(session, audio, f"downlifter_{dwn_type}")


def cmd_g_wsh(session: "Session", args: List[str]) -> str:
    """Generate whoosh.
    
    Usage:
      /g_wsh [type]      Generate whoosh
      
    Types: 1=fast, 2=slow, 3=textured
    """
    sr = session.sample_rate
    wsh_type = int(args[0]) if args and args[0].isdigit() else 1
    
    if wsh_type == 1:
        duration = 0.5
    elif wsh_type == 2:
        duration = 1.5
    else:
        duration = 1.0
    
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    # Filtered noise
    noise = _pink_noise(length)
    
    # Bell curve envelope
    mid = duration / 2
    env = np.exp(-((t - mid) ** 2) / (2 * (duration / 4) ** 2))
    
    audio = noise * env
    
    if wsh_type == 3:  # Textured - add some pitch
        freq = 500 + 300 * np.sin(2 * np.pi * 2 * t)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio += np.sin(phase) * env * 0.3
    
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return _set_audio(session, audio, f"whoosh_{wsh_type}")


def cmd_g_glt(session: "Session", args: List[str]) -> str:
    """Generate glitch.
    
    Usage:
      /g_glt [type]      Generate glitch
      
    Types: 1=stutter, 2=buffer, 3=bitcrush, 4=random
    """
    sr = session.sample_rate
    glt_type = int(args[0]) if args and args[0].isdigit() else 1
    
    duration = 0.5
    length = int(duration * sr)
    
    if glt_type == 1:  # Stutter
        # Repeating short segment
        base = _noise(int(0.05 * sr))
        audio = np.tile(base, length // len(base) + 1)[:length]
        
    elif glt_type == 2:  # Buffer
        # Scrambled chunks
        chunk_size = int(0.02 * sr)
        chunks = [_noise(chunk_size) for _ in range(length // chunk_size)]
        np.random.shuffle(chunks)
        audio = np.concatenate(chunks)[:length]
        
    elif glt_type == 3:  # Bitcrush
        # Reduced bit depth effect
        t = np.linspace(0, duration, length)
        audio = np.sin(2 * np.pi * 440 * t)
        # Quantize
        bits = 4
        audio = np.round(audio * (2 ** bits)) / (2 ** bits)
        # Downsample effect
        factor = 8
        audio = np.repeat(audio[::factor], factor)[:length]
        
    else:  # Random
        # Random mix of effects
        audio = _noise(length)
        # Random amplitude jumps
        for _ in range(10):
            start = np.random.randint(0, length - 1000)
            audio[start:start + 500] *= np.random.uniform(0, 2)
    
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return _set_audio(session, audio, f"glitch_{glt_type}")


def cmd_g_vnl(session: "Session", args: List[str]) -> str:
    """Generate vinyl texture.
    
    Usage:
      /g_vnl [type]      Generate vinyl
      
    Types: 1=crackle, 2=hiss
    """
    sr = session.sample_rate
    vnl_type = int(args[0]) if args and args[0].isdigit() else 1
    
    duration = 2.0
    length = int(duration * sr)
    
    if vnl_type == 1:  # Crackle
        audio = np.zeros(length)
        # Random clicks
        num_clicks = int(duration * 20)  # 20 clicks per second
        for _ in range(num_clicks):
            pos = np.random.randint(0, length - 100)
            audio[pos:pos + 50] = np.random.randn(50) * 0.3
    else:  # Hiss
        audio = _noise(length) * 0.1
        # Lowpass to make it warmer
        audio = np.convolve(audio, np.ones(5) / 5, mode='same')
    
    return _set_audio(session, audio, f"vinyl_{vnl_type}")


def cmd_g_wnd(session: "Session", args: List[str]) -> str:
    """Generate wind/noise sweep.
    
    Usage:
      /g_wnd [type]      Generate wind
      
    Types: 1=gentle, 2=harsh, 3=filtered
    """
    sr = session.sample_rate
    wnd_type = int(args[0]) if args and args[0].isdigit() else 1
    
    duration = 3.0
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    if wnd_type == 1:  # Gentle
        audio = _pink_noise(length)
        # Slow modulation
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)
        audio *= mod
        
    elif wnd_type == 2:  # Harsh
        audio = _noise(length)
        mod = 0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 0.5 * t))
        audio *= mod
        
    else:  # Filtered
        audio = _pink_noise(length)
        # Moving filter (simplified)
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t)
        audio *= mod
    
    audio = audio / np.max(np.abs(audio)) * 0.6
    
    return _set_audio(session, audio, f"wind_{wnd_type}")


# ============================================================================
# UTILITY GENERATORS
# ============================================================================

def cmd_g_sil(session: "Session", args: List[str]) -> str:
    """Generate silence/spacer.
    
    Usage:
      /g_sil [duration]  Generate silence (seconds)
    """
    sr = session.sample_rate
    
    duration = float(args[0]) if args else 1.0
    length = int(duration * sr)
    
    audio = np.zeros(length, dtype=np.float64)
    
    return _set_audio(session, audio, f"silence_{duration}s")


def cmd_g_clk(session: "Session", args: List[str]) -> str:
    """Generate click track.
    
    Usage:
      /g_clk [bpm] [bars] Generate click track
    """
    sr = session.sample_rate
    
    bpm = float(args[0]) if args else 120.0
    bars = int(args[1]) if len(args) > 1 else 4
    
    beat_duration = 60 / bpm
    total_beats = bars * 4
    duration = total_beats * beat_duration
    length = int(duration * sr)
    
    audio = np.zeros(length, dtype=np.float64)
    
    # Click parameters
    click_len = int(0.01 * sr)
    click = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.01, click_len))
    click *= np.exp(-np.linspace(0, 1, click_len) * 10)
    
    downbeat = np.sin(2 * np.pi * 1500 * np.linspace(0, 0.01, click_len))
    downbeat *= np.exp(-np.linspace(0, 1, click_len) * 10)
    
    for beat in range(total_beats):
        pos = int(beat * beat_duration * sr)
        if pos + click_len < length:
            if beat % 4 == 0:  # Downbeat
                audio[pos:pos + click_len] = downbeat
            else:
                audio[pos:pos + click_len] = click * 0.7
    
    return _set_audio(session, audio, f"click_{bpm}bpm_{bars}bars")


def cmd_g_cal(session: "Session", args: List[str]) -> str:
    """Generate calibration tone.
    
    Usage:
      /g_cal [freq] [duration]  Generate sine wave
    """
    sr = session.sample_rate
    
    freq = float(args[0]) if args else 1000.0
    duration = float(args[1]) if len(args) > 1 else 1.0
    
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    audio = np.sin(2 * np.pi * freq * t) * 0.5
    
    # Fade in/out
    fade = int(0.01 * sr)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    
    return _set_audio(session, audio, f"cal_{freq}hz")


def cmd_g_swp(session: "Session", args: List[str]) -> str:
    """Generate frequency sweep.
    
    Usage:
      /g_swp [f1] [f2] [duration]  Generate sweep
    """
    sr = session.sample_rate
    
    f1 = float(args[0]) if args else 20.0
    f2 = float(args[1]) if len(args) > 1 else 20000.0
    duration = float(args[2]) if len(args) > 2 else 5.0
    
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    # Logarithmic sweep
    freq = f1 * (f2 / f1) ** (t / duration)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    
    audio = np.sin(phase) * 0.5
    
    # Fade in/out
    fade = int(0.05 * sr)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    
    return _set_audio(session, audio, f"sweep_{f1}-{f2}hz")


def cmd_g_blp(session: "Session", args: List[str]) -> str:
    """Generate bleep/blip.
    
    Usage:
      /g_blp [type]      Generate bleep
      
    Types: 1=short, 2=pitched, 3=glitch, 4=zap
    """
    sr = session.sample_rate
    blp_type = int(args[0]) if args and args[0].isdigit() else 1
    
    duration = 0.1
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    if blp_type == 1:  # Short
        audio = np.sin(2 * np.pi * 1000 * t)
        audio *= np.exp(-t * 50)
        
    elif blp_type == 2:  # Pitched
        audio = np.sin(2 * np.pi * 880 * t)
        audio *= _envelope(length, 0.001, 0.08, sr)
        
    elif blp_type == 3:  # Glitch
        audio = np.sin(2 * np.pi * 2000 * t)
        # Chop it up
        audio[int(0.02 * sr):int(0.03 * sr)] = 0
        audio[int(0.05 * sr):int(0.06 * sr)] = 0
        audio *= np.exp(-t * 30)
        
    else:  # Zap
        freq = 3000 * np.exp(-t * 40) + 500
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = np.sin(phase) * np.exp(-t * 30)
    
    audio = audio / np.max(np.abs(audio)) * 0.85
    
    return _set_audio(session, audio, f"bleep_{blp_type}")


def cmd_g_stb(session: "Session", args: List[str]) -> str:
    """Generate stab.
    
    Usage:
      /g_stb [type]      Generate stab
      
    Types: 1=chord, 2=brass, 3=synth, 4=orchestral
    """
    sr = session.sample_rate
    stb_type = int(args[0]) if args and args[0].isdigit() else 1
    
    duration = 0.3
    length = int(duration * sr)
    t = np.linspace(0, duration, length)
    
    if stb_type == 1:  # Chord
        # C major
        freqs = [261.63, 329.63, 392.00]
        audio = sum(np.sin(2 * np.pi * f * t) for f in freqs) / 3
        
    elif stb_type == 2:  # Brass
        freq = 220
        audio = np.sin(2 * np.pi * freq * t)
        audio += 0.5 * np.sin(2 * np.pi * freq * 2 * t)
        audio += 0.3 * np.sin(2 * np.pi * freq * 3 * t)
        
    elif stb_type == 3:  # Synth
        freq = 440
        # Saw-ish
        audio = 2 * (t * freq % 1) - 1
        
    else:  # Orchestral
        freqs = [130.81, 196.00, 261.63, 329.63]  # C major spread
        audio = sum(np.sin(2 * np.pi * f * t) * 0.3 for f in freqs)
    
    audio *= _envelope(length, 0.01, 0.2, sr)
    audio = audio / np.max(np.abs(audio)) * 0.85
    
    return _set_audio(session, audio, f"stab_{stb_type}")


# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

def get_generator_commands() -> dict:
    """Return all generator commands."""
    return {
        # Drums
        'g_kick': cmd_g_kick,
        'g_snare': cmd_g_snare,
        'g_hat': cmd_g_hat,
        'g_tom': cmd_g_tom,
        'g_clp': cmd_g_clp,
        'g_cym': cmd_g_cym,
        'g_snp': cmd_g_snp,
        'g_shk': cmd_g_shk,
        
        # FX
        'g_zap': cmd_g_zap,
        'g_lsr': cmd_g_lsr,
        'g_bel': cmd_g_bel,
        'g_bas': cmd_g_bas,
        'g_rsr': cmd_g_rsr,
        'g_dwn': cmd_g_dwn,
        'g_wsh': cmd_g_wsh,
        'g_glt': cmd_g_glt,
        'g_vnl': cmd_g_vnl,
        'g_wnd': cmd_g_wnd,
        'g_blp': cmd_g_blp,
        'g_stb': cmd_g_stb,
        
        # Utility
        'g_sil': cmd_g_sil,
        'g_clk': cmd_g_clk,
        'g_cal': cmd_g_cal,
        'g_swp': cmd_g_swp,
    }
