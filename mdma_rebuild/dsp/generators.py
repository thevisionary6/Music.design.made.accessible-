"""Sound Generators for MDMA.

Implements algorithmic sound generation for:
- Drums (kick, snare, hat, tom, clap, etc.)
- FX (zaps, risers, downlifters, whooshes)
- Utility (silence, clicks, calibration tones, sweeps)
- Textures (vinyl, wind, glitch)

All generators return numpy arrays at the session sample rate.

BUILD ID: generators_v1.0
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List

SAMPLE_RATE = 48000


# ============================================================================
# ENVELOPE HELPERS
# ============================================================================

def exp_decay(samples: int, decay_time: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Exponential decay envelope."""
    t = np.arange(samples) / sr
    return np.exp(-t / max(0.001, decay_time))


def linear_decay(samples: int) -> np.ndarray:
    """Linear decay envelope."""
    return np.linspace(1, 0, samples)


def adsr_envelope(samples: int, attack: float, decay: float, 
                  sustain: float, release: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """ADSR envelope."""
    attack_samples = int(attack * sr)
    decay_samples = int(decay * sr)
    release_samples = int(release * sr)
    sustain_samples = samples - attack_samples - decay_samples - release_samples
    
    if sustain_samples < 0:
        # Adjust proportionally
        total = attack_samples + decay_samples + release_samples
        scale = samples / max(1, total)
        attack_samples = int(attack_samples * scale)
        decay_samples = int(decay_samples * scale)
        release_samples = samples - attack_samples - decay_samples
        sustain_samples = 0
    
    env = np.zeros(samples)
    
    # Attack
    if attack_samples > 0:
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Decay
    start = attack_samples
    if decay_samples > 0:
        env[start:start+decay_samples] = np.linspace(1, sustain, decay_samples)
    
    # Sustain
    start += decay_samples
    if sustain_samples > 0:
        env[start:start+sustain_samples] = sustain
    
    # Release
    start += sustain_samples
    if release_samples > 0:
        env[start:start+release_samples] = np.linspace(sustain, 0, release_samples)
    
    return env


# ============================================================================
# KICK GENERATOR
# ============================================================================

def generate_kick(pitch: float = 60, duration: float = 0.5,
                  decay: float = 0.2, variant: int = 1,
                  sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate kick drum sound.
    
    Variants:
        1 = 808-style (deep sub bass, long decay)
        2 = Punchy (tight attack, shorter tail)
        3 = Sub (very low, minimal click)
        4 = Acoustic (more natural attack)
    """
    samples = int(duration * sr)
    t = np.arange(samples) / sr
    
    # Variant settings
    if variant == 1:  # 808
        freq_start, freq_end = pitch * 2, pitch * 0.5
        click_amt = 0.3
        sub_amt = 0.9
        decay_time = decay * 1.5
    elif variant == 2:  # Punchy
        freq_start, freq_end = pitch * 3, pitch * 0.8
        click_amt = 0.6
        sub_amt = 0.7
        decay_time = decay * 0.7
    elif variant == 3:  # Sub
        freq_start, freq_end = pitch * 1.2, pitch * 0.4
        click_amt = 0.1
        sub_amt = 1.0
        decay_time = decay * 2.0
    else:  # Acoustic (variant 4)
        freq_start, freq_end = pitch * 4, pitch
        click_amt = 0.5
        sub_amt = 0.6
        decay_time = decay
    
    # Pitch envelope (fast drop)
    pitch_env = np.exp(-t * 40) * (freq_start - freq_end) + freq_end
    
    # Generate sine with pitch envelope
    phase = np.cumsum(2 * np.pi * pitch_env / sr)
    body = np.sin(phase)
    
    # Click/transient
    click_samples = int(0.005 * sr)
    click = np.zeros(samples)
    if click_samples > 0:
        click[:click_samples] = np.random.randn(click_samples) * exp_decay(click_samples, 0.001, sr)
        # Highpass the click
        click[:click_samples] *= np.linspace(1, 0.3, click_samples)
    
    # Amplitude envelope
    amp_env = exp_decay(samples, decay_time, sr)
    
    # Mix
    out = (body * sub_amt + click * click_amt) * amp_env
    
    # Soft clip
    out = np.tanh(out * 1.5) * 0.9
    
    return out


# ============================================================================
# SNARE GENERATOR
# ============================================================================

def generate_snare(pitch: float = 200, duration: float = 0.25,
                   snap: float = 0.5, variant: int = 1,
                   sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate snare drum sound.
    
    Variants:
        1 = 808-style (electronic, punchy)
        2 = Acoustic (natural, roomy)
        3 = Tight (short, crisp)
        4 = Clicky (emphasized attack)
    """
    samples = int(duration * sr)
    t = np.arange(samples) / sr
    
    # Variant settings
    if variant == 1:  # 808
        body_freq = pitch
        noise_amt = 0.6
        body_amt = 0.5
        decay_body = 0.08
        decay_noise = 0.12
    elif variant == 2:  # Acoustic
        body_freq = pitch * 0.8
        noise_amt = 0.7
        body_amt = 0.4
        decay_body = 0.06
        decay_noise = 0.15
    elif variant == 3:  # Tight
        body_freq = pitch * 1.2
        noise_amt = 0.5
        body_amt = 0.6
        decay_body = 0.04
        decay_noise = 0.08
    else:  # Clicky (variant 4)
        body_freq = pitch * 1.5
        noise_amt = 0.4
        body_amt = 0.7
        decay_body = 0.05
        decay_noise = 0.1
    
    # Body (pitched component)
    body_env = exp_decay(samples, decay_body, sr)
    # Pitch drop
    freq_env = body_freq * np.exp(-t * 30) + body_freq * 0.5
    phase = np.cumsum(2 * np.pi * freq_env / sr)
    body = np.sin(phase) * body_env
    
    # Noise (snare wires)
    noise = np.random.randn(samples)
    noise_env = exp_decay(samples, decay_noise, sr)
    
    # Bandpass the noise (snare character)
    from scipy import signal
    try:
        b, a = signal.butter(2, [3000, 10000], btype='band', fs=sr)
        noise = signal.filtfilt(b, a, noise)
    except:
        pass  # Skip filtering if it fails
    
    noise = noise * noise_env
    
    # Mix
    out = body * body_amt + noise * noise_amt
    
    # Add snap/attack
    snap_samples = int(0.003 * sr)
    if snap_samples > 0 and snap_samples < samples:
        out[:snap_samples] += np.random.randn(snap_samples) * snap * exp_decay(snap_samples, 0.001, sr)
    
    # Normalize
    max_val = np.max(np.abs(out))
    if max_val > 0:
        out = out / max_val * 0.9
    
    return out


# ============================================================================
# HIHAT GENERATOR
# ============================================================================

def generate_hihat(duration: float = 0.1, brightness: float = 0.7,
                   variant: int = 1, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate hi-hat sound.
    
    Variants:
        1 = Closed (short, tight)
        2 = Open (longer decay, shimmer)
        3 = Pedal (medium, muted)
        4 = Crispy (bright, digital)
    """
    # Adjust duration based on variant
    if variant == 1:  # Closed
        duration = min(duration, 0.08)
        decay_time = 0.03
    elif variant == 2:  # Open
        duration = max(duration, 0.3)
        decay_time = 0.2
    elif variant == 3:  # Pedal
        duration = 0.15
        decay_time = 0.08
    else:  # Crispy (variant 4)
        duration = min(duration, 0.1)
        decay_time = 0.04
    
    samples = int(duration * sr)
    t = np.arange(samples) / sr
    
    # Base: filtered noise
    noise = np.random.randn(samples)
    
    # Highpass filter for metallic character
    from scipy import signal
    try:
        # High cutoff for hi-hat character
        cutoff = 5000 + brightness * 7000  # 5kHz to 12kHz
        b, a = signal.butter(2, cutoff, btype='high', fs=sr)
        noise = signal.filtfilt(b, a, noise)
    except:
        pass
    
    # Add metallic tones (multiple inharmonic frequencies)
    freqs = [7000, 9500, 12000, 14500]  # Metallic partials
    for freq in freqs:
        phase = 2 * np.pi * freq * t + np.random.rand() * 2 * np.pi
        noise += np.sin(phase) * 0.15 * brightness
    
    # Amplitude envelope
    env = exp_decay(samples, decay_time, sr)
    out = noise * env
    
    # Variant-specific processing
    if variant == 4:  # Crispy - add more high end
        try:
            b, a = signal.butter(1, 8000, btype='high', fs=sr)
            out = signal.filtfilt(b, a, out) * 0.5 + out * 0.5
        except:
            pass
    
    # Normalize
    max_val = np.max(np.abs(out))
    if max_val > 0:
        out = out / max_val * 0.85
    
    return out


# ============================================================================
# TOM GENERATOR
# ============================================================================

def generate_tom(pitch: float = 80, duration: float = 0.3, 
                 decay: float = 0.15, variant: int = 1,
                 sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate tom drum sound.
    
    Variants:
        1 = Low tom (floor)
        2 = Mid tom
        3 = High tom
        4 = Electronic/808 tom
    """
    samples = int(duration * sr)
    t = np.arange(samples) / sr
    
    # Pitch based on variant
    base_freqs = {1: 60, 2: 100, 3: 150, 4: 80}
    freq = base_freqs.get(variant, 80) * (pitch / 80)
    
    # Pitch envelope (descending)
    pitch_env = np.exp(-t / 0.05) * 0.5 + 0.5
    freq_curve = freq * (1 + pitch_env * 0.5)
    
    # Generate waveform
    phase = np.cumsum(2 * np.pi * freq_curve / sr)
    
    if variant == 4:
        # Electronic - pure sine
        wave = np.sin(phase)
    else:
        # Acoustic - add harmonics
        wave = np.sin(phase) * 0.7
        wave += np.sin(phase * 2) * 0.2
        wave += np.sin(phase * 3) * 0.1
    
    # Amplitude envelope
    env = exp_decay(samples, decay, sr)
    
    # Add attack transient
    click_samples = int(0.005 * sr)
    if click_samples < samples:
        click = np.random.randn(click_samples) * 0.3
        click *= np.linspace(1, 0, click_samples)
        wave[:click_samples] += click
    
    return wave * env * 0.8


# ============================================================================
# CYMBAL GENERATOR
# ============================================================================

def generate_cymbal(duration: float = 0.5, brightness: float = 0.7,
                    variant: int = 1, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate cymbal sound.
    
    Variants:
        1 = Crash
        2 = Ride
        3 = China
        4 = Splash
    """
    durations = {1: 1.0, 2: 0.4, 3: 0.8, 4: 0.3}
    dur = durations.get(variant, 0.5) * duration
    samples = int(dur * sr)
    t = np.arange(samples) / sr
    
    # Metallic frequencies (inharmonic)
    if variant == 1:  # Crash
        freqs = [300, 587, 1174, 2349, 3520, 5274, 8000]
    elif variant == 2:  # Ride
        freqs = [400, 800, 1600, 2400, 3600, 5000]
    elif variant == 3:  # China
        freqs = [250, 500, 1000, 2000, 3500, 6000, 9000]
    else:  # Splash
        freqs = [500, 1000, 2000, 4000, 6000]
    
    # Generate metallic texture
    wave = np.zeros(samples)
    for i, f in enumerate(freqs):
        amp = (1.0 - i * 0.1) * brightness
        decay = 0.3 / (i + 1)
        wave += np.sin(2 * np.pi * f * t) * amp * exp_decay(samples, decay, sr)
    
    # Add noise component
    noise = np.random.randn(samples)
    # Highpass filter noise
    noise_hp = np.diff(np.concatenate([[0], noise]))
    noise_hp = np.diff(np.concatenate([[0], noise_hp]))
    wave += noise_hp[:samples] * 0.3 * brightness
    
    # Envelope
    env = exp_decay(samples, dur * 0.8, sr)
    
    # Normalize
    wave = wave * env
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * 0.7
    
    return wave


# ============================================================================
# CLAP GENERATOR
# ============================================================================

def generate_clap(duration: float = 0.15, variant: int = 1,
                  sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate clap sound.
    
    Variants:
        1 = 808 style
        2 = Acoustic
        3 = Layered
        4 = Tight
    """
    samples = int(duration * sr)
    
    # Number of "hands" in the clap
    num_hands = {1: 3, 2: 5, 3: 7, 4: 2}[variant]
    
    wave = np.zeros(samples)
    
    for i in range(num_hands):
        # Slight timing offset for each hand
        offset = int(np.random.uniform(0, 0.01) * sr)
        
        # Each hand is bandpassed noise
        hand_samples = samples - offset
        if hand_samples <= 0:
            continue
            
        noise = np.random.randn(hand_samples)
        
        # Bandpass around 1-2kHz
        from scipy.signal import butter, lfilter
        nyq = sr / 2
        low = 800 / nyq
        high = min(0.99, 2500 / nyq)
        b, a = butter(2, [low, high], btype='band')
        hand = lfilter(b, a, noise)
        
        # Envelope
        env = exp_decay(hand_samples, 0.03 + i * 0.01, sr)
        hand = hand * env * (1 - i * 0.1)
        
        wave[offset:offset + len(hand)] += hand
    
    # Normalize
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * 0.8
    
    return wave


# ============================================================================
# SNAP GENERATOR
# ============================================================================

def generate_snap(duration: float = 0.08, variant: int = 1,
                  sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate finger snap sound.
    
    Variants:
        1 = Dry
        2 = With reverb tail
    """
    samples = int(duration * sr)
    
    # Core snap - very short transient
    t = np.arange(samples) / sr
    
    # Mix of click and tonal component
    click = np.random.randn(samples) * exp_decay(samples, 0.005, sr)
    
    # Tonal component around 2-3kHz
    tone = np.sin(2 * np.pi * 2500 * t) * exp_decay(samples, 0.01, sr) * 0.3
    
    wave = click * 0.7 + tone
    
    if variant == 2:
        # Add reverb-like tail
        tail = np.random.randn(samples) * 0.1 * exp_decay(samples, 0.05, sr)
        wave += tail
    
    # Normalize
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * 0.7
    
    return wave


# ============================================================================
# SHAKER GENERATOR
# ============================================================================

def generate_shaker(duration: float = 0.1, pattern: int = 1,
                    sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate shaker sound.
    
    Patterns:
        1 = 16th notes
        2 = 8th notes
        3 = Triplets
    """
    samples = int(duration * sr)
    
    # Filtered noise
    noise = np.random.randn(samples)
    
    # Highpass filter
    from scipy.signal import butter, lfilter
    nyq = sr / 2
    b, a = butter(2, 3000 / nyq, btype='high')
    wave = lfilter(b, a, noise)
    
    # Envelope - quick attack, moderate decay
    env = adsr_envelope(samples, 0.002, 0.02, 0.3, 0.03, sr)
    wave = wave * env
    
    # Normalize
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * 0.5
    
    return wave


# ============================================================================
# BLEEP/BLIP GENERATOR
# ============================================================================

def generate_bleep(freq: float = 1000, duration: float = 0.05,
                   variant: int = 1, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate bleep/blip sound.
    
    Variants:
        1 = Short
        2 = Pitched (with vibrato)
        3 = Glitch
        4 = Zap
    """
    samples = int(duration * sr)
    t = np.arange(samples) / sr
    
    if variant == 1:  # Short
        wave = np.sin(2 * np.pi * freq * t)
        env = exp_decay(samples, duration * 0.3, sr)
        
    elif variant == 2:  # Pitched with vibrato
        vibrato = np.sin(2 * np.pi * 6 * t) * 0.02
        phase = np.cumsum(2 * np.pi * freq * (1 + vibrato) / sr)
        wave = np.sin(phase)
        env = adsr_envelope(samples, 0.01, 0.05, 0.5, 0.1, sr)
        
    elif variant == 3:  # Glitch
        wave = np.sign(np.sin(2 * np.pi * freq * t))  # Square
        # Add bitcrushing
        wave = np.round(wave * 4) / 4
        env = exp_decay(samples, duration * 0.5, sr)
        
    else:  # Zap
        freq_env = np.exp(-t / 0.02)
        phase = np.cumsum(2 * np.pi * freq * (1 + freq_env * 2) / sr)
        wave = np.sin(phase)
        env = exp_decay(samples, duration * 0.3, sr)
    
    return wave * env * 0.7


# ============================================================================
# STAB GENERATOR
# ============================================================================

def generate_stab(freq: float = 200, duration: float = 0.2,
                  variant: int = 1, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate stab sound.
    
    Variants:
        1 = Chord (major)
        2 = Brass-like
        3 = Synth
        4 = Orchestral
    """
    samples = int(duration * sr)
    t = np.arange(samples) / sr
    
    wave = np.zeros(samples)
    
    if variant == 1:  # Chord
        ratios = [1, 1.26, 1.5]  # Major triad
        for r in ratios:
            wave += np.sin(2 * np.pi * freq * r * t) * 0.33
        
    elif variant == 2:  # Brass
        # Sawtooth with filter sweep
        saw = 2 * (t * freq % 1) - 1
        # Simple lowpass sweep
        cutoff_env = 0.2 + 0.8 * exp_decay(samples, 0.1, sr)
        # Apply as amplitude modulation approximation
        wave = saw * cutoff_env
        
    elif variant == 3:  # Synth
        # Detuned saws
        for detune in [-0.02, 0, 0.02]:
            f = freq * (1 + detune)
            wave += (2 * (t * f % 1) - 1) * 0.33
            
    else:  # Orchestral
        # String-like with harmonics
        for h in range(1, 8):
            amp = 1.0 / h
            wave += np.sin(2 * np.pi * freq * h * t) * amp
        wave = wave / np.max(np.abs(wave)) * 0.8
    
    env = adsr_envelope(samples, 0.01, 0.1, 0.3, 0.1, sr)
    return wave * env * 0.7


# ============================================================================
# ZAP/LASER GENERATOR
# ============================================================================

def generate_zap(start_freq: float = 2000, end_freq: float = 100,
                 duration: float = 0.15, variant: int = 1,
                 sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate zap/laser sound.
    
    Variants:
        1 = Short
        2 = Long
        3 = Sweep
        4 = Retro
    """
    dur_mults = {1: 0.5, 2: 2.0, 3: 1.0, 4: 0.8}
    dur = duration * dur_mults.get(variant, 1.0)
    samples = int(dur * sr)
    t = np.arange(samples) / sr
    
    # Frequency sweep
    freq_env = np.exp(-t / (dur * 0.3))
    freq_curve = end_freq + (start_freq - end_freq) * freq_env
    
    # Generate waveform
    phase = np.cumsum(2 * np.pi * freq_curve / sr)
    
    if variant == 4:  # Retro - square wave
        wave = np.sign(np.sin(phase))
    else:
        wave = np.sin(phase)
    
    # Envelope
    env = exp_decay(samples, dur * 0.8, sr)
    
    return wave * env * 0.7


def generate_laser(duration: float = 0.2, variant: int = 1,
                   sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate laser FX.
    
    Variants:
        1 = Pew
        2 = Beam
        3 = Charge
    """
    if variant == 1:  # Pew
        return generate_zap(3000, 200, duration * 0.3, 1, sr)
    elif variant == 2:  # Beam
        return generate_zap(1000, 1000, duration, 2, sr)
    else:  # Charge
        # Reversed zap
        zap = generate_zap(200, 3000, duration, 2, sr)
        return zap[::-1]


# ============================================================================
# BELL GENERATOR
# ============================================================================

def generate_bell(freq: float = 440, duration: float = 1.0,
                  variant: int = 1, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate bell sound.
    
    Variants:
        1 = Bright
        2 = Dark
        3 = Tubular
        4 = Glass
    """
    samples = int(duration * sr)
    t = np.arange(samples) / sr
    
    wave = np.zeros(samples)
    
    # Bell-like inharmonic partials
    if variant == 1:  # Bright
        partials = [(1, 1.0), (2.4, 0.5), (3.0, 0.4), (4.5, 0.25), (6.7, 0.15)]
    elif variant == 2:  # Dark
        partials = [(1, 1.0), (2.0, 0.6), (2.5, 0.3), (3.5, 0.2)]
    elif variant == 3:  # Tubular
        partials = [(1, 1.0), (2.76, 0.5), (5.4, 0.3), (8.9, 0.15)]
    else:  # Glass
        partials = [(1, 1.0), (2.2, 0.7), (4.1, 0.4), (6.3, 0.2), (9.0, 0.1)]
    
    for ratio, amp in partials:
        decay = duration / (ratio * 0.5)
        env = exp_decay(samples, decay, sr)
        wave += np.sin(2 * np.pi * freq * ratio * t) * amp * env
    
    # Normalize
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * 0.7
    
    return wave


# ============================================================================
# BASS HIT GENERATOR
# ============================================================================

def generate_bass(freq: float = 60, duration: float = 0.3,
                  variant: int = 1, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate bass hit.
    
    Variants:
        1 = 808
        2 = Sub
        3 = Punch
        4 = Growl
    """
    samples = int(duration * sr)
    t = np.arange(samples) / sr
    
    if variant == 1:  # 808
        # Pitch envelope
        pitch_env = np.exp(-t / 0.05)
        freq_curve = freq * (1 + pitch_env * 0.5)
        phase = np.cumsum(2 * np.pi * freq_curve / sr)
        wave = np.sin(phase)
        env = exp_decay(samples, duration * 0.7, sr)
        
    elif variant == 2:  # Sub
        wave = np.sin(2 * np.pi * freq * t)
        env = adsr_envelope(samples, 0.01, 0.1, 0.8, 0.2, sr)
        
    elif variant == 3:  # Punch
        # Mix sine and click
        wave = np.sin(2 * np.pi * freq * t)
        click = np.random.randn(min(500, samples)) * 0.3
        click *= np.linspace(1, 0, len(click))
        wave[:len(click)] += click
        env = exp_decay(samples, duration * 0.5, sr)
        
    else:  # Growl
        # Distorted bass
        wave = np.sin(2 * np.pi * freq * t)
        wave = np.tanh(wave * 3) / 2
        wave += np.sin(2 * np.pi * freq * 2 * t) * 0.3
        env = adsr_envelope(samples, 0.02, 0.1, 0.6, 0.2, sr)
    
    return wave * env * 0.8


# ============================================================================
# RISER GENERATOR
# ============================================================================

def generate_riser(duration: float = 2.0, variant: int = 1,
                   sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate riser sound.
    
    Variants:
        1 = Noise
        2 = Tonal
        3 = Sweep
        4 = Tension
    """
    samples = int(duration * sr)
    t = np.arange(samples) / sr
    
    if variant == 1:  # Noise
        wave = np.random.randn(samples)
        # Rising filter
        from scipy.signal import butter, lfilter
        nyq = sr / 2
        # Process in chunks with increasing cutoff
        chunk_size = samples // 20
        output = np.zeros(samples)
        for i in range(20):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, samples)
            cutoff = 200 + (i / 19) * 8000
            b, a = butter(2, min(0.99, cutoff / nyq), btype='low')
            output[start:end] = lfilter(b, a, wave[start:end])
        wave = output
        env = np.linspace(0.3, 1.0, samples)
        
    elif variant == 2:  # Tonal
        freq_start, freq_end = 100, 800
        freq_curve = freq_start + (freq_end - freq_start) * (t / duration) ** 2
        phase = np.cumsum(2 * np.pi * freq_curve / sr)
        wave = np.sin(phase)
        env = np.linspace(0.5, 1.0, samples)
        
    elif variant == 3:  # Sweep
        freq_start, freq_end = 20, 5000
        freq_curve = freq_start * (freq_end / freq_start) ** (t / duration)
        phase = np.cumsum(2 * np.pi * freq_curve / sr)
        wave = np.sin(phase)
        env = np.linspace(0.3, 1.0, samples)
        
    else:  # Tension
        # Multiple risers combined
        wave = np.zeros(samples)
        for mult in [1, 1.5, 2]:
            freq_curve = 100 * mult + 400 * mult * (t / duration) ** 1.5
            phase = np.cumsum(2 * np.pi * freq_curve / sr)
            wave += np.sin(phase) * 0.33
        env = np.linspace(0.4, 1.0, samples)
    
    wave = wave * env
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * 0.7
    
    return wave


# ============================================================================
# DOWNLIFTER GENERATOR
# ============================================================================

def generate_downlifter(duration: float = 1.0, variant: int = 1,
                        sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate downlifter sound.
    
    Variants:
        1 = Drop
        2 = Sweep
        3 = Impact
        4 = Reverse
    """
    if variant == 4:  # Reverse riser
        riser = generate_riser(duration, 3, sr)
        return riser[::-1]
    
    samples = int(duration * sr)
    t = np.arange(samples) / sr
    
    if variant == 1:  # Drop
        freq_start, freq_end = 1000, 40
        freq_curve = freq_start * (freq_end / freq_start) ** (t / duration)
        phase = np.cumsum(2 * np.pi * freq_curve / sr)
        wave = np.sin(phase)
        env = exp_decay(samples, duration * 0.6, sr)
        
    elif variant == 2:  # Sweep
        freq_start, freq_end = 5000, 50
        freq_curve = freq_start * (freq_end / freq_start) ** (t / duration)
        phase = np.cumsum(2 * np.pi * freq_curve / sr)
        wave = np.sin(phase)
        env = np.linspace(1.0, 0.2, samples)
        
    else:  # Impact
        # Noise burst with lowpass sweep down
        wave = np.random.randn(samples)
        env = exp_decay(samples, duration * 0.3, sr)
        # Add low thump
        thump = np.sin(2 * np.pi * 60 * t) * exp_decay(samples, 0.15, sr)
        wave = wave * 0.5 + thump * 0.8
    
    wave = wave * env
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * 0.7
    
    return wave


# ============================================================================
# WHOOSH GENERATOR
# ============================================================================

def generate_whoosh(duration: float = 0.5, variant: int = 1,
                    sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate whoosh sound.
    
    Variants:
        1 = Fast
        2 = Slow
        3 = Textured
    """
    dur_mults = {1: 0.5, 2: 2.0, 3: 1.0}
    dur = duration * dur_mults.get(variant, 1.0)
    samples = int(dur * sr)
    t = np.arange(samples) / sr
    
    # Base noise
    wave = np.random.randn(samples)
    
    # Envelope - peaks in middle
    env = np.sin(np.pi * t / dur) ** 0.5
    
    if variant == 3:  # Textured
        # Add some tonal content
        tone = np.sin(2 * np.pi * 500 * t) * 0.2
        wave = wave * 0.8 + tone
    
    # Bandpass filter
    from scipy.signal import butter, lfilter
    nyq = sr / 2
    b, a = butter(2, [500 / nyq, min(0.99, 8000 / nyq)], btype='band')
    wave = lfilter(b, a, wave)
    
    wave = wave * env
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * 0.6
    
    return wave


# ============================================================================
# GLITCH GENERATOR
# ============================================================================

def generate_glitch(duration: float = 0.2, variant: int = 1,
                    sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate glitch sound.
    
    Variants:
        1 = Stutter
        2 = Buffer
        3 = Bitcrush
        4 = Random
    """
    samples = int(duration * sr)
    
    if variant == 1:  # Stutter
        # Repeated micro-segment
        chunk = int(0.01 * sr)
        base = np.random.randn(chunk)
        repeats = samples // chunk + 1
        wave = np.tile(base, repeats)[:samples]
        
    elif variant == 2:  # Buffer
        # Random jumps in a buffer
        base = np.sin(2 * np.pi * 440 * np.arange(samples) / sr)
        wave = np.zeros(samples)
        pos = 0
        while pos < samples:
            chunk = np.random.randint(100, 1000)
            src_pos = np.random.randint(0, samples - chunk) if samples > chunk else 0
            end = min(pos + chunk, samples)
            wave[pos:end] = base[src_pos:src_pos + (end - pos)]
            pos = end
            
    elif variant == 3:  # Bitcrush
        wave = np.sin(2 * np.pi * 440 * np.arange(samples) / sr)
        bits = 4
        wave = np.round(wave * (2 ** bits)) / (2 ** bits)
        
    else:  # Random
        wave = np.zeros(samples)
        pos = 0
        while pos < samples:
            chunk = np.random.randint(50, 500)
            end = min(pos + chunk, samples)
            freq = np.random.uniform(100, 2000)
            t = np.arange(end - pos) / sr
            wave[pos:end] = np.sin(2 * np.pi * freq * t)
            pos = end
    
    env = adsr_envelope(samples, 0.005, 0.02, 0.8, 0.05, sr)
    wave = wave * env
    
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * 0.6
    
    return wave


# ============================================================================
# VINYL TEXTURE GENERATOR
# ============================================================================

def generate_vinyl(duration: float = 1.0, variant: int = 1,
                   sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate vinyl texture.
    
    Variants:
        1 = Crackle
        2 = Hiss
    """
    samples = int(duration * sr)
    
    if variant == 1:  # Crackle
        wave = np.zeros(samples)
        # Random clicks
        num_clicks = int(duration * 50)  # ~50 clicks per second
        for _ in range(num_clicks):
            pos = np.random.randint(0, samples)
            click_len = np.random.randint(5, 50)
            end = min(pos + click_len, samples)
            wave[pos:end] += np.random.randn(end - pos) * np.random.uniform(0.1, 0.5)
        
    else:  # Hiss
        wave = np.random.randn(samples) * 0.1
        # Highpass filter
        from scipy.signal import butter, lfilter
        nyq = sr / 2
        b, a = butter(2, 5000 / nyq, btype='high')
        wave = lfilter(b, a, wave)
    
    return wave * 0.3


# ============================================================================
# WIND/NOISE GENERATOR
# ============================================================================

def generate_wind(duration: float = 2.0, variant: int = 1,
                  sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate wind/noise sweep.
    
    Variants:
        1 = Gentle
        2 = Harsh
        3 = Filtered
    """
    samples = int(duration * sr)
    t = np.arange(samples) / sr
    
    # Base noise
    wave = np.random.randn(samples)
    
    from scipy.signal import butter, lfilter
    nyq = sr / 2
    
    if variant == 1:  # Gentle
        b, a = butter(2, 2000 / nyq, btype='low')
        wave = lfilter(b, a, wave)
        env = np.sin(np.pi * t / duration) ** 0.3
        
    elif variant == 2:  # Harsh
        # Less filtering
        b, a = butter(1, 5000 / nyq, btype='low')
        wave = lfilter(b, a, wave)
        env = np.sin(np.pi * t / duration) ** 0.5
        
    else:  # Filtered sweep
        # Time-varying filter
        chunk_size = samples // 20
        output = np.zeros(samples)
        for i in range(20):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, samples)
            # Sweep cutoff
            cutoff = 500 + 3000 * np.sin(np.pi * i / 19)
            b, a = butter(2, min(0.99, cutoff / nyq), btype='low')
            output[start:end] = lfilter(b, a, wave[start:end])
        wave = output
        env = np.ones(samples) * 0.8
    
    wave = wave * env
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * 0.4
    
    return wave


# ============================================================================
# UTILITY GENERATORS
# ============================================================================

def generate_silence(duration: float = 1.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate silence."""
    return np.zeros(int(duration * sr))


def generate_click(bpm: float = 120, bars: int = 4, 
                   sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate click track.
    
    Parameters
    ----------
    bpm : float
        Tempo in BPM
    bars : int
        Number of bars (4 beats per bar)
    """
    beat_samples = int(60 / bpm * sr)
    total_beats = bars * 4
    total_samples = beat_samples * total_beats
    
    wave = np.zeros(total_samples)
    
    # Click sound
    click_len = int(0.01 * sr)
    click = np.sin(2 * np.pi * 1000 * np.arange(click_len) / sr)
    click *= np.linspace(1, 0, click_len)
    
    # Accent click (downbeat)
    accent = np.sin(2 * np.pi * 1500 * np.arange(click_len) / sr)
    accent *= np.linspace(1, 0, click_len)
    
    for beat in range(total_beats):
        pos = beat * beat_samples
        if beat % 4 == 0:
            wave[pos:pos + click_len] = accent
        else:
            wave[pos:pos + click_len] = click
    
    return wave * 0.5


def generate_calibration(freq: float = 1000, duration: float = 1.0,
                         sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate calibration tone (pure sine)."""
    samples = int(duration * sr)
    t = np.arange(samples) / sr
    return np.sin(2 * np.pi * freq * t) * 0.5


def generate_sweep(start_freq: float = 20, end_freq: float = 20000,
                   duration: float = 5.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate frequency sweep."""
    samples = int(duration * sr)
    t = np.arange(samples) / sr
    
    # Logarithmic sweep
    freq_curve = start_freq * (end_freq / start_freq) ** (t / duration)
    phase = np.cumsum(2 * np.pi * freq_curve / sr)
    
    return np.sin(phase) * 0.5


# ============================================================================
# GENERATOR DISPATCHER
# ============================================================================

GENERATORS = {
    # Drums - short names
    'kick': generate_kick,
    'snare': generate_snare,
    'hat': generate_hihat,
    'tom': generate_tom,
    'cym': generate_cymbal,
    'clp': generate_clap,
    'snp': generate_snap,
    'shk': generate_shaker,
    
    # Drums - long names (aliases)
    'hihat': generate_hihat,
    'cymbal': generate_cymbal,
    'clap': generate_clap,
    'snap': generate_snap,
    'shaker': generate_shaker,
    
    # FX/Synth - short names
    'blp': generate_bleep,
    'stb': generate_stab,
    'zap': generate_zap,
    'lsr': generate_laser,
    'bel': generate_bell,
    'bas': generate_bass,
    'rsr': generate_riser,
    'dwn': generate_downlifter,
    'wsh': generate_whoosh,
    'glt': generate_glitch,
    
    # FX/Synth - long names (aliases)
    'bleep': generate_bleep,
    'stab': generate_stab,
    'laser': generate_laser,
    'bell': generate_bell,
    'bass': generate_bass,
    'riser': generate_riser,
    'downlifter': generate_downlifter,
    'whoosh': generate_whoosh,
    'glitch': generate_glitch,
    
    # Textures
    'vnl': generate_vinyl,
    'wnd': generate_wind,
    'vinyl': generate_vinyl,
    'wind': generate_wind,
    
    # Utility
    'sil': generate_silence,
    'clk': generate_click,
    'cal': generate_calibration,
    'swp': generate_sweep,
    'silence': generate_silence,
    'click': generate_click,
    'calibration': generate_calibration,
    'sweep': generate_sweep,
}


def generate(name: str, variant: int = 1, duration: float = None,
             freq: float = None, sr: int = SAMPLE_RATE, **kwargs) -> Optional[np.ndarray]:
    """Generate sound by name.
    
    Parameters
    ----------
    name : str
        Generator name (tom, cym, clp, etc.)
    variant : int
        Variant/preset number
    duration : float, optional
        Override duration
    freq : float, optional
        Override frequency (where applicable)
    **kwargs : dict
        Additional generator-specific parameters
    
    Returns
    -------
    np.ndarray or None
        Generated audio, or None if generator not found
    """
    if name not in GENERATORS:
        return None
    
    gen_func = GENERATORS[name]
    
    # Build kwargs
    gen_kwargs = {'sr': sr}
    if variant is not None:
        gen_kwargs['variant'] = variant
    if duration is not None:
        gen_kwargs['duration'] = duration
    if freq is not None:
        gen_kwargs['freq'] = freq
    gen_kwargs.update(kwargs)
    
    # Call generator (it will ignore unknown kwargs)
    import inspect
    sig = inspect.signature(gen_func)
    valid_kwargs = {k: v for k, v in gen_kwargs.items() if k in sig.parameters}
    
    return gen_func(**valid_kwargs)


def list_generators() -> List[str]:
    """List all available generator names."""
    return list(GENERATORS.keys())
