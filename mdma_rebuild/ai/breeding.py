"""MDMA Genetic Sample Breeding System.

Combines samples using genetic algorithms to create hybrid children.
Uses attribute vectors for fitness evaluation and guided breeding.

Section N3 of MDMA Master Feature List.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, Dict, Callable, Any
from dataclasses import dataclass
import random

from .analysis import AttributeVector, DeepAnalyzer, get_analyzer


# ============================================================================
# BREEDING CONFIGURATION
# ============================================================================

@dataclass
class BreedingConfig:
    """Configuration for genetic breeding operations."""
    
    # Crossover settings
    crossover_probability: float = 0.8
    crossover_type: str = 'multi'  # 'single', 'multi', 'uniform', 'blend'
    crossover_points: int = 2
    
    # Mutation settings
    mutation_probability: float = 0.15
    mutation_strength: float = 0.2  # 0-1, how much to mutate
    mutation_types: List[str] = None  # Types of mutations to use
    
    # Selection settings
    elite_count: int = 2
    tournament_size: int = 3
    
    # Population settings
    population_size: int = 8
    generations: int = 10
    
    # Target attribute weights (for fitness)
    target_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.mutation_types is None:
            self.mutation_types = [
                'noise', 'pitch', 'time_stretch', 'freq_shift',
                'envelope', 'reverse_segment', 'spectral_smear'
            ]
        if self.target_weights is None:
            self.target_weights = {}


# ============================================================================
# CROSSOVER OPERATIONS
# ============================================================================

def crossover_temporal(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    crossover_point: float = 0.5,
    fade_samples: int = 512,
) -> np.ndarray:
    """Time-domain crossover at specified point.
    
    Parameters
    ----------
    parent_a, parent_b : np.ndarray
        Parent audio arrays (same length)
    crossover_point : float
        Where to split (0-1)
    fade_samples : int
        Crossfade length at junction
    
    Returns
    -------
    np.ndarray
        Child audio
    """
    split = int(len(parent_a) * crossover_point)
    fade = min(fade_samples, split // 4, (len(parent_a) - split) // 4)
    
    child = np.zeros_like(parent_a)
    
    # Copy sections
    child[:split - fade] = parent_a[:split - fade]
    child[split + fade:] = parent_b[split + fade:]
    
    # Crossfade
    if fade > 0:
        fade_out = np.linspace(1, 0, 2 * fade)
        fade_in = 1 - fade_out
        start = split - fade
        end = split + fade
        child[start:end] = (
            parent_a[start:end] * fade_out +
            parent_b[start:end] * fade_in
        )
    
    return child


def crossover_spectral(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    crossover_freq: float = 0.5,
    blend_width: int = 50,
) -> np.ndarray:
    """Frequency-domain crossover.
    
    Parameters
    ----------
    parent_a, parent_b : np.ndarray
        Parent audio
    crossover_freq : float
        Crossover point in spectrum (0-1)
    blend_width : int
        FFT bins to blend at crossover
    
    Returns
    -------
    np.ndarray
        Child audio
    """
    spec_a = np.fft.rfft(parent_a)
    spec_b = np.fft.rfft(parent_b)
    
    split = int(len(spec_a) * crossover_freq)
    blend = min(blend_width, split // 4, (len(spec_a) - split) // 4)
    
    child_spec = np.zeros_like(spec_a)
    
    # Copy frequency bands
    child_spec[:split - blend] = spec_a[:split - blend]
    child_spec[split + blend:] = spec_b[split + blend:]
    
    # Blend transition
    if blend > 0:
        blend_curve = np.linspace(1, 0, 2 * blend)
        start = split - blend
        end = split + blend
        child_spec[start:end] = (
            spec_a[start:end] * blend_curve +
            spec_b[start:end] * (1 - blend_curve)
        )
    
    return np.fft.irfft(child_spec, len(parent_a))


def crossover_blend(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    blend_factor: float = 0.5,
) -> np.ndarray:
    """Simple linear blend of parents.
    
    Parameters
    ----------
    parent_a, parent_b : np.ndarray
        Parent audio
    blend_factor : float
        0 = all A, 1 = all B
    
    Returns
    -------
    np.ndarray
        Blended child
    """
    return parent_a * (1 - blend_factor) + parent_b * blend_factor


def crossover_morphological(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    env_blend: float = 0.5,
    spec_blend: float = 0.5,
) -> np.ndarray:
    """Separate envelope and spectral content crossover.
    
    Parameters
    ----------
    parent_a, parent_b : np.ndarray
        Parent audio
    env_blend : float
        Envelope blend (0=A, 1=B)
    spec_blend : float
        Spectral blend (0=A, 1=B)
    
    Returns
    -------
    np.ndarray
        Morphed child
    """
    # Extract envelopes
    env_a = np.abs(parent_a)
    env_b = np.abs(parent_b)
    
    # Smooth envelopes
    win = 256
    env_a = np.convolve(env_a, np.ones(win)/win, mode='same')
    env_b = np.convolve(env_b, np.ones(win)/win, mode='same')
    
    # Blend envelope
    child_env = env_a * (1 - env_blend) + env_b * env_blend
    
    # Get spectral content
    spec_a = np.fft.rfft(parent_a)
    spec_b = np.fft.rfft(parent_b)
    
    # Blend magnitudes
    mag_a = np.abs(spec_a)
    mag_b = np.abs(spec_b)
    child_mag = mag_a * (1 - spec_blend) + mag_b * spec_blend
    
    # Take phase from one parent (or average)
    if np.random.random() < 0.5:
        child_phase = np.angle(spec_a)
    else:
        child_phase = np.angle(spec_b)
    
    child_spec = child_mag * np.exp(1j * child_phase)
    child = np.fft.irfft(child_spec, len(parent_a))
    
    # Apply blended envelope
    child_current_env = np.abs(child) + 0.001
    child = child * (child_env / child_current_env)
    
    return child


def crossover_multi_point(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    num_points: int = 3,
    fade_samples: int = 256,
) -> np.ndarray:
    """Multi-point temporal crossover.
    
    Parameters
    ----------
    parent_a, parent_b : np.ndarray
        Parent audio
    num_points : int
        Number of crossover points
    fade_samples : int
        Crossfade at each junction
    
    Returns
    -------
    np.ndarray
        Child audio
    """
    points = sorted([random.random() for _ in range(num_points)])
    points = [0.0] + points + [1.0]
    
    child = np.zeros_like(parent_a)
    use_a = True
    
    for i in range(len(points) - 1):
        start = int(len(parent_a) * points[i])
        end = int(len(parent_a) * points[i + 1])
        
        source = parent_a if use_a else parent_b
        child[start:end] = source[start:end]
        
        # Crossfade at junction
        if start > fade_samples:
            prev_source = parent_b if use_a else parent_a
            fade = min(fade_samples, (end - start) // 2)
            if fade > 0:
                curve = np.linspace(0, 1, fade)
                child[start:start+fade] = (
                    prev_source[start:start+fade] * (1 - curve) +
                    source[start:start+fade] * curve
                )
        
        use_a = not use_a
    
    return child


# ============================================================================
# MUTATION OPERATIONS
# ============================================================================

def mutate_noise(audio: np.ndarray, strength: float = 0.1) -> np.ndarray:
    """Add random noise mutation."""
    noise = np.random.randn(len(audio)) * strength * 0.1
    return audio + noise


def mutate_pitch(audio: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """Pitch shift mutation via resampling."""
    ratio = 1.0 + (random.random() - 0.5) * strength
    new_len = int(len(audio) / ratio)
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, new_len)
    resampled = np.interp(x_new, x_old, audio)
    
    # Pad/trim to original length
    if len(resampled) > len(audio):
        return resampled[:len(audio)]
    else:
        return np.pad(resampled, (0, len(audio) - len(resampled)))


def mutate_time_stretch(audio: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """Stretch a random segment."""
    start = int(random.random() * len(audio) * 0.5)
    length = int(len(audio) * 0.3)
    end = min(start + length, len(audio))
    
    segment = audio[start:end]
    stretch = 1.0 + (random.random() - 0.5) * strength
    new_len = int(len(segment) * stretch)
    
    if new_len < 10:
        return audio
    
    x_old = np.linspace(0, 1, len(segment))
    x_new = np.linspace(0, 1, new_len)
    stretched = np.interp(x_new, x_old, segment)
    
    # Reconstruct
    result = np.concatenate([
        audio[:start],
        stretched,
        audio[end:]
    ])
    
    # Pad/trim
    if len(result) > len(audio):
        return result[:len(audio)]
    else:
        return np.pad(result, (0, len(audio) - len(result)))


def mutate_freq_shift(audio: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """Shift frequencies up or down."""
    spec = np.fft.rfft(audio)
    shift = int(len(spec) * strength * 0.1 * (2 * random.random() - 1))
    
    if shift > 0:
        shifted = np.concatenate([np.zeros(shift), spec[:-shift]])
    elif shift < 0:
        shifted = np.concatenate([spec[-shift:], np.zeros(-shift)])
    else:
        return audio
    
    return np.fft.irfft(shifted, len(audio))


def mutate_envelope(audio: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """Modify amplitude envelope."""
    # Random envelope modulation
    env_points = np.random.random(8) * strength + (1 - strength/2)
    env = np.interp(
        np.linspace(0, 1, len(audio)),
        np.linspace(0, 1, len(env_points)),
        env_points
    )
    
    # Smooth
    win = 512
    env = np.convolve(env, np.ones(win)/win, mode='same')
    
    return audio * env


def mutate_reverse_segment(audio: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """Reverse a random segment."""
    seg_len = int(len(audio) * strength * 0.3)
    if seg_len < 100:
        return audio
    
    start = int(random.random() * (len(audio) - seg_len))
    end = start + seg_len
    
    result = audio.copy()
    result[start:end] = result[start:end][::-1]
    return result


def mutate_spectral_smear(audio: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """Blur/smear spectral content."""
    spec = np.fft.rfft(audio)
    mag = np.abs(spec)
    phase = np.angle(spec)
    
    # Convolve magnitude with gaussian
    kernel_size = int(len(mag) * strength * 0.02) + 1
    kernel = np.exp(-np.linspace(-2, 2, kernel_size) ** 2)
    kernel = kernel / kernel.sum()
    
    smeared_mag = np.convolve(mag, kernel, mode='same')
    
    new_spec = smeared_mag * np.exp(1j * phase)
    return np.fft.irfft(new_spec, len(audio))


MUTATION_FUNCTIONS = {
    'noise': mutate_noise,
    'pitch': mutate_pitch,
    'time_stretch': mutate_time_stretch,
    'freq_shift': mutate_freq_shift,
    'envelope': mutate_envelope,
    'reverse_segment': mutate_reverse_segment,
    'spectral_smear': mutate_spectral_smear,
}


def apply_mutation(
    audio: np.ndarray,
    mutation_type: Optional[str] = None,
    strength: float = 0.2,
) -> np.ndarray:
    """Apply a mutation to audio.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio to mutate
    mutation_type : str, optional
        Type of mutation (random if None)
    strength : float
        Mutation strength (0-1)
    
    Returns
    -------
    np.ndarray
        Mutated audio
    """
    if mutation_type is None:
        mutation_type = random.choice(list(MUTATION_FUNCTIONS.keys()))
    
    func = MUTATION_FUNCTIONS.get(mutation_type, mutate_noise)
    return func(audio, strength)


# ============================================================================
# SAMPLE BREEDER
# ============================================================================

class SampleBreeder:
    """Genetic algorithm-based sample breeding system."""
    
    def __init__(
        self,
        sample_rate: int = 48000,
        config: Optional[BreedingConfig] = None,
    ):
        self.sample_rate = sample_rate
        self.config = config or BreedingConfig()
        self.analyzer = get_analyzer(sample_rate)
    
    def _normalize_length(
        self,
        samples: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Pad/trim samples to same length."""
        max_len = max(len(s) for s in samples)
        result = []
        for s in samples:
            if len(s) < max_len:
                result.append(np.pad(s, (0, max_len - len(s))))
            elif len(s) > max_len:
                result.append(s[:max_len])
            else:
                result.append(s)
        return result
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to -1,1 range."""
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio / peak * 0.95
        return audio
    
    def breed(
        self,
        parent_a: np.ndarray,
        parent_b: np.ndarray,
        num_children: int = 4,
    ) -> List[np.ndarray]:
        """Breed two parents to create children.
        
        Parameters
        ----------
        parent_a, parent_b : np.ndarray
            Parent audio samples
        num_children : int
            Number of children to create
        
        Returns
        -------
        list
            List of child audio arrays
        """
        # Normalize lengths
        parent_a, parent_b = self._normalize_length([parent_a, parent_b])
        
        children = []
        
        for i in range(num_children):
            # Choose crossover method
            method = i % 5
            
            if method == 0:
                # Temporal crossover
                cp = random.random()
                child = crossover_temporal(parent_a, parent_b, cp)
            elif method == 1:
                # Spectral crossover
                cp = random.random()
                child = crossover_spectral(parent_a, parent_b, cp)
            elif method == 2:
                # Blend crossover
                blend = random.random()
                child = crossover_blend(parent_a, parent_b, blend)
            elif method == 3:
                # Morphological crossover
                env_b = random.random()
                spec_b = random.random()
                child = crossover_morphological(parent_a, parent_b, env_b, spec_b)
            else:
                # Multi-point crossover
                child = crossover_multi_point(parent_a, parent_b, 3)
            
            # Apply mutation with probability
            if random.random() < self.config.mutation_probability:
                mut_type = random.choice(self.config.mutation_types)
                child = apply_mutation(child, mut_type, self.config.mutation_strength)
            
            # Normalize
            child = self._normalize_audio(child)
            children.append(child.astype(np.float64))
        
        return children
    
    def breed_targeted(
        self,
        parent_a: np.ndarray,
        parent_b: np.ndarray,
        target_attributes: Dict[str, float],
        num_children: int = 8,
    ) -> List[Tuple[np.ndarray, float]]:
        """Breed with fitness toward target attributes.
        
        Parameters
        ----------
        parent_a, parent_b : np.ndarray
            Parent audio
        target_attributes : dict
            Target attribute values to aim for
        num_children : int
            Number of children
        
        Returns
        -------
        list
            List of (child_audio, fitness_score) tuples, sorted by fitness
        """
        children = self.breed(parent_a, parent_b, num_children)
        
        # Create target vector
        target = AttributeVector()
        for name, value in target_attributes.items():
            target.set(name, value)
        
        # Score each child
        scored = []
        for child in children:
            av = self.analyzer.analyze(child, detailed=False)
            fitness = av.similarity(target)
            scored.append((child, fitness))
        
        # Sort by fitness (descending)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def evolve(
        self,
        population: List[np.ndarray],
        fitness_fn: Callable[[np.ndarray], float],
        generations: Optional[int] = None,
        callback: Optional[Callable[[int, List[np.ndarray], List[float]], None]] = None,
    ) -> List[np.ndarray]:
        """Evolve a population over generations.
        
        Parameters
        ----------
        population : list
            Initial population of audio samples
        fitness_fn : callable
            Function: audio -> fitness score (higher = better)
        generations : int, optional
            Number of generations (uses config default if None)
        callback : callable, optional
            Called after each generation: (gen, population, fitness_scores)
        
        Returns
        -------
        list
            Final evolved population, sorted by fitness
        """
        generations = generations or self.config.generations
        population = self._normalize_length(population)
        
        for gen in range(generations):
            # Evaluate fitness
            scored = [(sample, fitness_fn(sample)) for sample in population]
            scored.sort(key=lambda x: x[1], reverse=True)
            
            if callback:
                callback(gen, [s for s, _ in scored], [f for _, f in scored])
            
            # Keep elite
            new_pop = [s for s, _ in scored[:self.config.elite_count]]
            
            # Breed to fill population
            while len(new_pop) < self.config.population_size:
                # Tournament selection for parents
                parent_a = self._tournament_select(scored)
                parent_b = self._tournament_select(scored)
                
                # Breed
                if random.random() < self.config.crossover_probability:
                    children = self.breed(parent_a, parent_b, num_children=2)
                    new_pop.extend(children[:2])
                else:
                    # Just mutate parents
                    new_pop.append(apply_mutation(parent_a.copy(), strength=self.config.mutation_strength))
            
            population = new_pop[:self.config.population_size]
        
        # Final sort
        scored = [(sample, fitness_fn(sample)) for sample in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [s for s, _ in scored]
    
    def _tournament_select(
        self,
        scored: List[Tuple[np.ndarray, float]],
    ) -> np.ndarray:
        """Tournament selection."""
        k = min(self.config.tournament_size, len(scored))
        tournament = random.sample(range(len(scored)), k)
        winner = max(tournament, key=lambda i: scored[i][1])
        return scored[winner][0]
    
    def evolve_toward_target(
        self,
        population: List[np.ndarray],
        target_attributes: Dict[str, float],
        generations: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Evolve population toward target attributes.
        
        Parameters
        ----------
        population : list
            Initial population
        target_attributes : dict
            Target attribute values
        generations : int, optional
            Number of generations
        
        Returns
        -------
        list
            Evolved population
        """
        target = AttributeVector()
        for name, value in target_attributes.items():
            target.set(name, value)
        
        def fitness_fn(audio: np.ndarray) -> float:
            av = self.analyzer.analyze(audio, detailed=False)
            return av.similarity(target)
        
        return self.evolve(population, fitness_fn, generations)


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

_breeder: Optional[SampleBreeder] = None

def get_breeder(sample_rate: int = 48000) -> SampleBreeder:
    """Get or create global breeder."""
    global _breeder
    if _breeder is None or _breeder.sample_rate != sample_rate:
        _breeder = SampleBreeder(sample_rate)
    return _breeder


def breed_samples(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    num_children: int = 4,
    sample_rate: int = 48000,
) -> List[np.ndarray]:
    """Quick breed two samples.
    
    Parameters
    ----------
    parent_a, parent_b : np.ndarray
        Parent samples
    num_children : int
        Number of children
    sample_rate : int
        Sample rate
    
    Returns
    -------
    list
        Child samples
    """
    breeder = get_breeder(sample_rate)
    return breeder.breed(parent_a, parent_b, num_children)


def evolve_samples(
    population: List[np.ndarray],
    target_attributes: Dict[str, float],
    generations: int = 10,
    sample_rate: int = 48000,
) -> List[np.ndarray]:
    """Evolve samples toward target attributes.
    
    Parameters
    ----------
    population : list
        Initial population
    target_attributes : dict
        Target attribute values
    generations : int
        Number of generations
    sample_rate : int
        Sample rate
    
    Returns
    -------
    list
        Evolved population
    """
    breeder = get_breeder(sample_rate)
    return breeder.evolve_toward_target(population, target_attributes, generations)


def format_breeding_result(
    children: List[np.ndarray],
    fitness_scores: Optional[List[float]] = None,
    sample_rate: int = 48000,
) -> str:
    """Format breeding results as text.
    
    Parameters
    ----------
    children : list
        Child samples
    fitness_scores : list, optional
        Fitness scores for each child
    sample_rate : int
        Sample rate
    
    Returns
    -------
    str
        Formatted result
    """
    lines = [
        "=== BREEDING RESULTS ===",
        f"Children: {len(children)}",
        "",
    ]
    
    for i, child in enumerate(children):
        dur = len(child) / sample_rate
        peak = np.max(np.abs(child))
        
        score_str = ""
        if fitness_scores and i < len(fitness_scores):
            score_str = f" (fitness: {fitness_scores[i]:.3f})"
        
        lines.append(f"  Child {i+1}: {dur:.3f}s, peak={peak:.3f}{score_str}")
    
    return '\n'.join(lines)
