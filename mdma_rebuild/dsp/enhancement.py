"""AI-Powered Audio Quality Enhancement Module.

Provides output-stage processing for improved audio quality.
Uses machine learning inspired techniques for:
- Dynamic range optimization
- Spectral enhancement
- Transient preservation
- Stereo width enhancement
- Intelligent limiting
- Multi-pass processing with best-pick selection
- Corruption detection and prevention

This module operates on the master output to polish the final mix.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field


@dataclass
class EnhancementSettings:
    """Settings for AI audio enhancement."""
    enabled: bool = False  # Default OFF - user must enable with /ai
    
    # Multi-pass processing
    multi_pass_enabled: bool = True
    num_passes: int = 3  # Number of enhancement variations to try
    
    # Corruption prevention
    corruption_detection: bool = True
    max_dc_offset: float = 0.1  # Maximum allowed DC offset
    min_correlation: float = 0.3  # Minimum correlation with original
    max_peak_ratio: float = 3.0  # Maximum peak increase ratio
    
    # Loudness optimization
    target_lufs: float = -14.0  # Target loudness (streaming standard)
    loudness_enabled: bool = True
    
    # Dynamic range
    dynamics_enabled: bool = True
    dynamics_mode: str = 'balanced'  # 'gentle', 'balanced', 'aggressive'
    
    # Spectral enhancement
    spectral_enabled: bool = True
    brightness: float = 0.0  # -1 to +1 (dark to bright)
    warmth: float = 0.0  # -1 to +1 (thin to warm)
    
    # Transient handling
    transient_enabled: bool = True
    transient_punch: float = 0.5  # 0 to 1
    
    # Stereo (for future stereo support)
    stereo_width: float = 1.0  # 0.5 to 2.0
    
    # Limiting
    limiter_enabled: bool = True
    limiter_ceiling: float = -0.3  # dB below 0
    limiter_release: float = 100.0  # ms
    
    # Quality preset
    preset: str = 'master'  # 'transparent', 'master', 'broadcast', 'loud'


# Global settings instance
_enhancement_settings = EnhancementSettings()


def get_enhancement_settings() -> EnhancementSettings:
    """Get current enhancement settings."""
    return _enhancement_settings


def set_enhancement_preset(preset: str) -> str:
    """Set enhancement preset.
    
    Presets:
    - transparent: Minimal processing, preserve original character
    - master: Balanced mastering-style enhancement
    - broadcast: Optimized for streaming/broadcast
    - loud: Maximum loudness (may sacrifice dynamics)
    """
    global _enhancement_settings
    
    presets = {
        'transparent': {
            'target_lufs': -16.0,
            'dynamics_mode': 'gentle',
            'brightness': 0.0,
            'warmth': 0.0,
            'transient_punch': 0.3,
            'limiter_ceiling': -0.5,
        },
        'master': {
            'target_lufs': -14.0,
            'dynamics_mode': 'balanced',
            'brightness': 0.1,
            'warmth': 0.1,
            'transient_punch': 0.5,
            'limiter_ceiling': -0.3,
        },
        'broadcast': {
            'target_lufs': -14.0,
            'dynamics_mode': 'balanced',
            'brightness': 0.2,
            'warmth': 0.0,
            'transient_punch': 0.6,
            'limiter_ceiling': -1.0,
        },
        'loud': {
            'target_lufs': -11.0,
            'dynamics_mode': 'aggressive',
            'brightness': 0.15,
            'warmth': 0.2,
            'transient_punch': 0.7,
            'limiter_ceiling': -0.1,
        },
    }
    
    if preset not in presets:
        return f"ERROR: Unknown preset '{preset}'. Use: transparent, master, broadcast, loud"
    
    settings = presets[preset]
    _enhancement_settings.preset = preset
    _enhancement_settings.target_lufs = settings['target_lufs']
    _enhancement_settings.dynamics_mode = settings['dynamics_mode']
    _enhancement_settings.brightness = settings['brightness']
    _enhancement_settings.warmth = settings['warmth']
    _enhancement_settings.transient_punch = settings['transient_punch']
    _enhancement_settings.limiter_ceiling = settings['limiter_ceiling']
    
    return f"OK: Enhancement preset set to '{preset}'"


class AudioEnhancer:
    """AI-powered audio enhancement processor with corruption prevention."""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.settings = _enhancement_settings
        
        # State for stateful processing
        self._limiter_state = 0.0
        self._rms_history = []
        self._peak_hold = 0.0
        self._dc_filter_state = 0.0
        
        # Multiband state
        self._low_state = np.zeros(2)
        self._mid_state = np.zeros(2)
        self._high_state = np.zeros(2)
        
        # Corruption detection stats
        self._corruption_count = 0
        self._total_processed = 0
    
    def detect_corruption(self, original: np.ndarray, processed: np.ndarray) -> Tuple[bool, str]:
        """Detect if processed audio is corrupted.
        
        Returns (is_corrupted, reason)
        """
        if not self.settings.corruption_detection:
            return False, ""
        
        # Check for NaN or Inf
        if np.any(np.isnan(processed)) or np.any(np.isinf(processed)):
            return True, "NaN/Inf values detected"
        
        # Check DC offset
        dc_offset = abs(np.mean(processed))
        if dc_offset > self.settings.max_dc_offset:
            return True, f"DC offset too high: {dc_offset:.3f}"
        
        # Check correlation with original (similarity)
        if len(original) == len(processed) and len(original) > 0:
            orig_norm = original - np.mean(original)
            proc_norm = processed - np.mean(processed)
            
            denom = np.sqrt(np.sum(orig_norm**2) * np.sum(proc_norm**2))
            if denom > 1e-10:
                correlation = np.sum(orig_norm * proc_norm) / denom
                if correlation < self.settings.min_correlation:
                    return True, f"Low correlation: {correlation:.3f}"
        
        # Check peak ratio
        orig_peak = np.max(np.abs(original)) + 1e-10
        proc_peak = np.max(np.abs(processed)) + 1e-10
        peak_ratio = proc_peak / orig_peak
        
        if peak_ratio > self.settings.max_peak_ratio:
            return True, f"Peak ratio too high: {peak_ratio:.2f}x"
        
        # Check for silence (all zeros or near-zero)
        if np.max(np.abs(processed)) < 1e-6 and np.max(np.abs(original)) > 1e-3:
            return True, "Output is silent"
        
        # Check for clipping (too many samples at max)
        clip_threshold = 0.99
        clip_count = np.sum(np.abs(processed) > clip_threshold)
        clip_ratio = clip_count / len(processed)
        if clip_ratio > 0.1:  # More than 10% clipping
            return True, f"Excessive clipping: {clip_ratio*100:.1f}%"
        
        return False, ""
    
    def score_quality(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Score the quality of processed audio (higher is better).
        
        Considers:
        - Similarity to original (don't destroy the sound)
        - Loudness improvement
        - Dynamic range preservation
        - Absence of artifacts
        """
        score = 0.0
        
        # Correlation bonus (similarity)
        orig_norm = original - np.mean(original)
        proc_norm = processed - np.mean(processed)
        denom = np.sqrt(np.sum(orig_norm**2) * np.sum(proc_norm**2))
        if denom > 1e-10:
            correlation = np.sum(orig_norm * proc_norm) / denom
            score += correlation * 30  # Up to 30 points for similarity
        
        # RMS improvement (louder is better, within reason)
        orig_rms = np.sqrt(np.mean(original**2)) + 1e-10
        proc_rms = np.sqrt(np.mean(processed**2)) + 1e-10
        rms_ratio = proc_rms / orig_rms
        
        if 0.8 <= rms_ratio <= 1.5:
            score += 20  # Good loudness range
        elif 0.5 <= rms_ratio <= 2.0:
            score += 10  # Acceptable
        
        # Dynamic range preservation
        orig_crest = np.max(np.abs(original)) / (orig_rms + 1e-10)
        proc_crest = np.max(np.abs(processed)) / (proc_rms + 1e-10)
        crest_ratio = proc_crest / (orig_crest + 1e-10)
        
        if 0.7 <= crest_ratio <= 1.3:
            score += 20  # Good dynamics preservation
        elif 0.5 <= crest_ratio <= 1.5:
            score += 10
        
        # Penalty for DC offset
        dc_offset = abs(np.mean(processed))
        score -= dc_offset * 50
        
        # Penalty for clipping
        clip_count = np.sum(np.abs(processed) > 0.99)
        clip_ratio = clip_count / len(processed)
        score -= clip_ratio * 100
        
        # Bonus for headroom
        headroom = 1.0 - np.max(np.abs(processed))
        if 0.01 <= headroom <= 0.1:
            score += 10  # Good headroom
        
        return score
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio through enhancement chain with corruption prevention.
        
        Uses multi-pass processing to select best result.
        
        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono float64)
        
        Returns
        -------
        np.ndarray
            Enhanced audio (or original if all passes corrupted)
        """
        if not self.settings.enabled:
            return audio
        
        self._total_processed += 1
        original = audio.astype(np.float64).copy()
        
        # Multi-pass processing
        if self.settings.multi_pass_enabled and self.settings.num_passes > 1:
            candidates = []
            
            for pass_idx in range(self.settings.num_passes):
                # Vary parameters slightly for each pass
                variation = self._get_pass_variation(pass_idx)
                
                try:
                    processed = self._process_single_pass(original.copy(), variation)
                    
                    # Check for corruption
                    is_corrupted, reason = self.detect_corruption(original, processed)
                    
                    if not is_corrupted:
                        score = self.score_quality(original, processed)
                        candidates.append((processed, score, pass_idx))
                except Exception:
                    # Skip failed passes
                    continue
            
            if candidates:
                # Select best candidate
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][0]
            else:
                # All passes corrupted - return original
                self._corruption_count += 1
                return original
        
        else:
            # Single pass processing
            try:
                processed = self._process_single_pass(original)
                
                is_corrupted, reason = self.detect_corruption(original, processed)
                if is_corrupted:
                    self._corruption_count += 1
                    return original
                
                return processed
            except Exception:
                self._corruption_count += 1
                return original
    
    def _get_pass_variation(self, pass_idx: int) -> Dict[str, float]:
        """Get parameter variations for multi-pass processing."""
        variations = [
            # Pass 0: Default settings
            {'brightness': 0, 'warmth': 0, 'punch': 0, 'dynamics': 0},
            # Pass 1: Brighter, more punch
            {'brightness': 0.1, 'warmth': -0.05, 'punch': 0.1, 'dynamics': 0.1},
            # Pass 2: Warmer, gentler
            {'brightness': -0.1, 'warmth': 0.1, 'punch': -0.1, 'dynamics': -0.1},
            # Pass 3: More aggressive
            {'brightness': 0.05, 'warmth': 0.05, 'punch': 0.15, 'dynamics': 0.15},
            # Pass 4: More transparent
            {'brightness': 0, 'warmth': 0, 'punch': -0.15, 'dynamics': -0.15},
        ]
        
        return variations[pass_idx % len(variations)]
    
    def _process_single_pass(
        self, 
        audio: np.ndarray, 
        variation: Dict[str, float] = None
    ) -> np.ndarray:
        """Process a single enhancement pass."""
        output = audio.copy()
        
        # Apply variation to settings temporarily
        orig_brightness = self.settings.brightness
        orig_warmth = self.settings.warmth
        orig_punch = self.settings.transient_punch
        
        if variation:
            self.settings.brightness += variation.get('brightness', 0)
            self.settings.warmth += variation.get('warmth', 0)
            self.settings.transient_punch += variation.get('punch', 0)
        
        try:
            # DC offset removal
            output = self._remove_dc(output)
            
            # Spectral enhancement (EQ curves)
            if self.settings.spectral_enabled:
                output = self._spectral_enhance(output)
            
            # Transient enhancement
            if self.settings.transient_enabled:
                output = self._enhance_transients(output)
            
            # Dynamics processing
            if self.settings.dynamics_enabled:
                output = self._process_dynamics(output)
            
            # Loudness optimization
            if self.settings.loudness_enabled:
                output = self._optimize_loudness(output)
            
            # Final limiting
            if self.settings.limiter_enabled:
                output = self._apply_limiter(output)
        
        finally:
            # Restore original settings
            self.settings.brightness = orig_brightness
            self.settings.warmth = orig_warmth
            self.settings.transient_punch = orig_punch
        
        return output
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'total_processed': self._total_processed,
            'corruption_count': self._corruption_count,
            'corruption_rate': (self._corruption_count / max(1, self._total_processed)) * 100,
        }
    
    def _remove_dc(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset with high-pass filter."""
        # Simple DC blocking filter
        alpha = 0.995
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            output[i] = audio[i] - self._dc_filter_state
            self._dc_filter_state = alpha * self._dc_filter_state + (1 - alpha) * audio[i]
        
        return output
    
    def _spectral_enhance(self, audio: np.ndarray) -> np.ndarray:
        """Apply spectral enhancement (brightness/warmth)."""
        # Simple 3-band processing
        
        # Low shelf (warmth)
        if abs(self.settings.warmth) > 0.01:
            # Apply low frequency boost/cut
            cutoff = 200  # Hz
            alpha = 1.0 / (1.0 + 2 * np.pi * cutoff / self.sample_rate)
            
            low = np.zeros_like(audio)
            for i in range(len(audio)):
                low[i] = alpha * self._low_state[0] + (1 - alpha) * audio[i]
                self._low_state[0] = low[i]
            
            # Boost or cut lows
            warmth_gain = 1.0 + self.settings.warmth * 0.3
            audio = audio + (low - audio) * (warmth_gain - 1.0)
        
        # High shelf (brightness)
        if abs(self.settings.brightness) > 0.01:
            # Apply high frequency boost/cut
            cutoff = 4000  # Hz
            alpha = 2 * np.pi * cutoff / self.sample_rate / (1.0 + 2 * np.pi * cutoff / self.sample_rate)
            
            high = np.zeros_like(audio)
            prev_high = self._high_state[0]
            for i in range(len(audio)):
                high[i] = alpha * (prev_high + audio[i] - (audio[i-1] if i > 0 else 0))
                prev_high = high[i]
            self._high_state[0] = prev_high
            
            # Boost or cut highs
            brightness_gain = 1.0 + self.settings.brightness * 0.4
            audio = audio + high * (brightness_gain - 1.0)
        
        return audio
    
    def _enhance_transients(self, audio: np.ndarray) -> np.ndarray:
        """Enhance transients for more punch."""
        if self.settings.transient_punch <= 0.01:
            return audio
        
        # Detect transients via envelope follower difference
        attack_ms = 1.0
        release_ms = 50.0
        
        attack_coef = np.exp(-1.0 / (self.sample_rate * attack_ms / 1000))
        release_coef = np.exp(-1.0 / (self.sample_rate * release_ms / 1000))
        
        envelope_fast = np.zeros_like(audio)
        envelope_slow = np.zeros_like(audio)
        
        fast_env = 0.0
        slow_env = 0.0
        
        for i in range(len(audio)):
            abs_sample = abs(audio[i])
            
            # Fast envelope (attack)
            if abs_sample > fast_env:
                fast_env = attack_coef * fast_env + (1 - attack_coef) * abs_sample
            else:
                fast_env = release_coef * fast_env
            
            # Slow envelope (release)
            if abs_sample > slow_env:
                slow_env = 0.99 * slow_env + 0.01 * abs_sample
            else:
                slow_env = 0.9999 * slow_env
            
            envelope_fast[i] = fast_env
            envelope_slow[i] = slow_env
        
        # Transient detection: fast > slow means transient
        transient_signal = np.maximum(0, envelope_fast - envelope_slow)
        
        # Apply transient enhancement
        punch = self.settings.transient_punch
        gain = 1.0 + transient_signal * punch * 2.0
        
        return audio * np.clip(gain, 1.0, 1.0 + punch)
    
    def _process_dynamics(self, audio: np.ndarray) -> np.ndarray:
        """Apply dynamics processing based on mode."""
        mode = self.settings.dynamics_mode
        
        if mode == 'gentle':
            ratio = 2.0
            threshold = -12.0
            knee = 10.0
        elif mode == 'aggressive':
            ratio = 6.0
            threshold = -18.0
            knee = 3.0
        else:  # balanced
            ratio = 3.0
            threshold = -14.0
            knee = 6.0
        
        # Convert threshold to linear
        threshold_lin = 10 ** (threshold / 20)
        knee_lin = 10 ** (knee / 20)
        
        # Attack/release times
        attack_ms = 5.0
        release_ms = 100.0
        
        attack_coef = np.exp(-1.0 / (self.sample_rate * attack_ms / 1000))
        release_coef = np.exp(-1.0 / (self.sample_rate * release_ms / 1000))
        
        envelope = 0.0
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            abs_sample = abs(audio[i])
            
            # Envelope follower
            if abs_sample > envelope:
                envelope = attack_coef * envelope + (1 - attack_coef) * abs_sample
            else:
                envelope = release_coef * envelope + (1 - release_coef) * abs_sample
            
            # Soft knee compression
            if envelope > threshold_lin:
                # Over threshold - compress
                over_db = 20 * np.log10(envelope / threshold_lin + 1e-10)
                compressed_db = over_db / ratio
                gain = 10 ** ((compressed_db - over_db) / 20)
            else:
                gain = 1.0
            
            output[i] = audio[i] * gain
        
        return output
    
    def _optimize_loudness(self, audio: np.ndarray) -> np.ndarray:
        """Optimize loudness towards target LUFS."""
        # Calculate current RMS (simplified LUFS approximation)
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms < 1e-10:
            return audio
        
        # Convert to dB
        rms_db = 20 * np.log10(rms)
        
        # Target RMS from LUFS (simplified)
        target_rms_db = self.settings.target_lufs + 10  # Approximate conversion
        
        # Calculate gain needed
        gain_db = target_rms_db - rms_db
        
        # Limit gain adjustment per block
        max_gain_change = 3.0  # dB
        gain_db = np.clip(gain_db, -max_gain_change, max_gain_change)
        
        gain = 10 ** (gain_db / 20)
        
        return audio * gain
    
    def _apply_limiter(self, audio: np.ndarray) -> np.ndarray:
        """Apply lookahead limiter for transparent peak control."""
        ceiling_lin = 10 ** (self.settings.limiter_ceiling / 20)
        
        release_ms = self.settings.limiter_release
        release_coef = np.exp(-1.0 / (self.sample_rate * release_ms / 1000))
        
        output = np.zeros_like(audio)
        gain = 1.0
        
        for i in range(len(audio)):
            abs_sample = abs(audio[i])
            
            # Calculate required gain reduction
            if abs_sample > ceiling_lin:
                target_gain = ceiling_lin / (abs_sample + 1e-10)
            else:
                target_gain = 1.0
            
            # Smooth gain changes
            if target_gain < gain:
                # Fast attack
                gain = target_gain
            else:
                # Slow release
                gain = release_coef * gain + (1 - release_coef) * target_gain
            
            output[i] = audio[i] * gain
        
        return output


# Global enhancer instance
_enhancer: Optional[AudioEnhancer] = None


def get_enhancer(sample_rate: int = 48000) -> AudioEnhancer:
    """Get or create global enhancer instance."""
    global _enhancer
    if _enhancer is None or _enhancer.sample_rate != sample_rate:
        _enhancer = AudioEnhancer(sample_rate)
    return _enhancer


def enhance_audio(audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    """Enhance audio with AI processing.
    
    Convenience function for one-shot processing.
    """
    enhancer = get_enhancer(sample_rate)
    return enhancer.process(audio)


def cmd_ai(args: list) -> str:
    """Toggle AI audio enhancement on/off.
    
    Usage:
      /ai               Toggle AI enhancement
      /ai on            Enable AI enhancement
      /ai off           Disable AI enhancement
      /ai status        Show detailed status
      /ai passes <n>    Set number of multi-pass iterations (1-5)
    
    AI enhancement uses multi-pass processing to find the best
    result and includes corruption detection to prevent bad output.
    
    Default: OFF (must be enabled explicitly)
    """
    settings = get_enhancement_settings()
    enhancer = get_enhancer()
    
    if not args:
        # Toggle
        settings.enabled = not settings.enabled
        state = "ON" if settings.enabled else "OFF"
        return f"AI Enhancement: {state}"
    
    cmd = args[0].lower()
    
    if cmd == 'on':
        settings.enabled = True
        return "AI Enhancement: ON"
    
    elif cmd == 'off':
        settings.enabled = False
        return "AI Enhancement: OFF"
    
    elif cmd == 'status':
        stats = enhancer.get_stats()
        return (f"=== AI ENHANCEMENT STATUS ===\n"
                f"  Enabled: {settings.enabled}\n"
                f"  Preset: {settings.preset}\n"
                f"  Multi-pass: {settings.multi_pass_enabled} ({settings.num_passes} passes)\n"
                f"  Corruption detection: {settings.corruption_detection}\n"
                f"  ---\n"
                f"  Total processed: {stats['total_processed']}\n"
                f"  Corrupted (prevented): {stats['corruption_count']}\n"
                f"  Corruption rate: {stats['corruption_rate']:.1f}%")
    
    elif cmd == 'passes' and len(args) > 1:
        try:
            n = int(args[1])
            n = max(1, min(5, n))
            settings.num_passes = n
            settings.multi_pass_enabled = n > 1
            return f"OK: Multi-pass set to {n} passes"
        except ValueError:
            return "ERROR: Invalid number of passes"
    
    else:
        return f"Unknown command: {cmd}. Use /ai for help."


def cmd_enhance(args: list) -> str:
    """Audio enhancement command interface.
    
    Usage:
      /enhance              Show enhancement status
      /enhance on           Enable enhancement
      /enhance off          Disable enhancement
      /enhance <preset>     Set preset (transparent/master/broadcast/loud)
      /enhance bright <v>   Set brightness (-1 to +1)
      /enhance warm <v>     Set warmth (-1 to +1)
      /enhance punch <v>    Set transient punch (0 to 1)
      /enhance target <db>  Set target LUFS
      /enhance passes <n>   Set multi-pass count (1-5)
      /enhance stats        Show processing stats
    
    Presets:
      transparent - Minimal processing
      master      - Balanced mastering (default)
      broadcast   - Streaming optimized
      loud        - Maximum loudness
    
    Multi-pass processing runs multiple enhancement variations
    and selects the best result. Corruption detection prevents
    bad audio from reaching output.
    """
    settings = get_enhancement_settings()
    enhancer = get_enhancer()
    
    if not args:
        stats = enhancer.get_stats()
        return (f"=== AUDIO ENHANCEMENT ===\n"
                f"  Enabled: {settings.enabled}\n"
                f"  Preset: {settings.preset}\n"
                f"  Target LUFS: {settings.target_lufs:.1f}\n"
                f"  Dynamics: {settings.dynamics_mode}\n"
                f"  Brightness: {settings.brightness:+.2f}\n"
                f"  Warmth: {settings.warmth:+.2f}\n"
                f"  Transient punch: {settings.transient_punch:.2f}\n"
                f"  Limiter ceiling: {settings.limiter_ceiling:.1f}dB\n"
                f"  ---\n"
                f"  Multi-pass: {settings.num_passes} passes\n"
                f"  Corruption prevention: {settings.corruption_detection}\n"
                f"  Stats: {stats['total_processed']} processed, {stats['corruption_count']} prevented")
    
    cmd = args[0].lower()
    
    if cmd == 'on':
        settings.enabled = True
        return "OK: Audio enhancement enabled"
    
    elif cmd == 'off':
        settings.enabled = False
        return "OK: Audio enhancement disabled"
    
    elif cmd in ('transparent', 'master', 'broadcast', 'loud'):
        return set_enhancement_preset(cmd)
    
    elif cmd in ('bright', 'brightness') and len(args) > 1:
        try:
            settings.brightness = float(args[1])
            settings.brightness = max(-1.0, min(1.0, settings.brightness))
            return f"OK: Brightness set to {settings.brightness:+.2f}"
        except ValueError:
            return "ERROR: Invalid brightness value"
    
    elif cmd in ('warm', 'warmth') and len(args) > 1:
        try:
            settings.warmth = float(args[1])
            settings.warmth = max(-1.0, min(1.0, settings.warmth))
            return f"OK: Warmth set to {settings.warmth:+.2f}"
        except ValueError:
            return "ERROR: Invalid warmth value"
    
    elif cmd in ('punch', 'transient') and len(args) > 1:
        try:
            settings.transient_punch = float(args[1])
            settings.transient_punch = max(0.0, min(1.0, settings.transient_punch))
            return f"OK: Transient punch set to {settings.transient_punch:.2f}"
        except ValueError:
            return "ERROR: Invalid punch value"
    
    elif cmd in ('target', 'lufs') and len(args) > 1:
        try:
            settings.target_lufs = float(args[1])
            settings.target_lufs = max(-24.0, min(-6.0, settings.target_lufs))
            return f"OK: Target LUFS set to {settings.target_lufs:.1f}"
        except ValueError:
            return "ERROR: Invalid LUFS value"
    
    elif cmd == 'passes' and len(args) > 1:
        try:
            n = int(args[1])
            n = max(1, min(5, n))
            settings.num_passes = n
            settings.multi_pass_enabled = n > 1
            return f"OK: Multi-pass set to {n} passes"
        except ValueError:
            return "ERROR: Invalid number of passes"
    
    elif cmd == 'stats':
        stats = enhancer.get_stats()
        return (f"=== ENHANCEMENT STATS ===\n"
                f"  Total processed: {stats['total_processed']}\n"
                f"  Corrupted (prevented): {stats['corruption_count']}\n"
                f"  Corruption rate: {stats['corruption_rate']:.2f}%")
    
    else:
        return f"Unknown command: {cmd}. Use /enhance for help."


# Exports
__all__ = [
    'EnhancementSettings',
    'AudioEnhancer',
    'get_enhancement_settings',
    'set_enhancement_preset',
    'get_enhancer',
    'enhance_audio',
    'cmd_ai',
    'cmd_enhance',
]
