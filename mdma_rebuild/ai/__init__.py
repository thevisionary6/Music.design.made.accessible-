"""MDMA AI Systems.

Audio generation, analysis, breeding, descriptors, and command routing.

Modules:
- generation: AudioLDM2 text-to-audio generation
- analysis: Deep attribute analysis with 80+ attributes
- breeding: Genetic algorithm sample breeding
- descriptors: Descriptor vocabulary engine (200+ descriptors)
- router: AI Command Router (ACR) with /high mode

Usage:
    from mdma_rebuild.ai import generate_audio, analyze_audio, breed_samples
"""

from .generation import (
    detect_gpu,
    get_optimal_settings,
    generate_audio,
    generate_and_store,
    generate_variations,
    enhance_prompt,
    get_negative_prompt,
    PRESET_PROMPTS,
    # GPU Settings
    SCHEDULERS,
    MODELS,
    GPUSettings,
    get_gpu_settings,
    set_gpu_setting,
    get_gpu_status,
    list_schedulers,
    list_models,
)

from .analysis import (
    ATTRIBUTE_POOL,
    AttributeVector,
    DeepAnalyzer,
    get_analyzer,
    format_analysis,
    format_comparison,
)

from .breeding import (
    BreedingConfig,
    SampleBreeder,
    get_breeder,
    breed_samples,
    evolve_samples,
    format_breeding_result,
    # Crossover functions
    crossover_temporal,
    crossover_spectral,
    crossover_blend,
    crossover_morphological,
    # Mutation functions
    apply_mutation,
    MUTATION_FUNCTIONS,
)

from .descriptors import (
    DESCRIPTOR_VOCABULARY,
    DESCRIPTOR_CATEGORIES,
    DescriptorProfile,
    attributes_to_descriptors,
    format_descriptor_profile,
)

from .router import (
    AICommandRouter,
    RouterPlan,
    CommandStep,
    Intent,
    RiskLevel,
    KNOWN_COMMANDS,
    get_router,
    format_plan,
)

__all__ = [
    # Generation
    'detect_gpu',
    'get_optimal_settings',
    'generate_audio',
    'generate_variations',
    'enhance_prompt',
    'get_negative_prompt',
    'PRESET_PROMPTS',
    # GPU Settings
    'SCHEDULERS',
    'MODELS',
    'GPUSettings',
    'get_gpu_settings',
    'set_gpu_setting',
    'get_gpu_status',
    'list_schedulers',
    'list_models',
    
    # Analysis
    'ATTRIBUTE_POOL',
    'AttributeVector',
    'DeepAnalyzer',
    'get_analyzer',
    'format_analysis',
    'format_comparison',
    
    # Breeding
    'BreedingConfig',
    'SampleBreeder',
    'get_breeder',
    'breed_samples',
    'evolve_samples',
    'format_breeding_result',
    'crossover_temporal',
    'crossover_spectral',
    'crossover_blend',
    'crossover_morphological',
    'apply_mutation',
    'MUTATION_FUNCTIONS',
    
    # Descriptors
    'DESCRIPTOR_VOCABULARY',
    'DESCRIPTOR_CATEGORIES',
    'DescriptorProfile',
    'attributes_to_descriptors',
    'format_descriptor_profile',
    
    # Router
    'AICommandRouter',
    'RouterPlan',
    'CommandStep',
    'Intent',
    'RiskLevel',
    'KNOWN_COMMANDS',
    'get_router',
    'format_plan',
]
