"""MDMA AI Audio Generation Module.

Uses AudioLDM2 for text-to-audio generation with GPU auto-detection.

Section N of MDMA Master Feature List.

Default Settings:
- Steps: 150
- Scheduler: DPM++ (discrete)
- CFG Scale: 10
- Model: audioldm2-large (or audioldm2-music for musical content)
"""

from __future__ import annotations

import os
import sys
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from dataclasses import dataclass, field


# ============================================================================
# GPU/AI SETTINGS
# ============================================================================

# Available schedulers (index-mapped for easy command access)
SCHEDULERS = {
    0: ('ddpm', 'DDPMScheduler', 'Original DDPM (slow but stable)'),
    1: ('ddim', 'DDIMScheduler', 'DDIM (faster, good quality)'),
    2: ('pndm', 'PNDMScheduler', 'PNDM (fast)'),
    3: ('lms', 'LMSDiscreteScheduler', 'LMS Discrete'),
    4: ('euler', 'EulerDiscreteScheduler', 'Euler (fast, good)'),
    5: ('euler_a', 'EulerAncestralDiscreteScheduler', 'Euler Ancestral (creative)'),
    6: ('dpm', 'DPMSolverMultistepScheduler', 'DPM++ Multistep (recommended)'),
    7: ('dpm_sde', 'DPMSolverSDEScheduler', 'DPM SDE (high quality)'),
    8: ('heun', 'HeunDiscreteScheduler', 'Heun (accurate)'),
    9: ('unipc', 'UniPCMultistepScheduler', 'UniPC (very fast)'),
}

# Available models
MODELS = {
    0: ('audioldm2-large', 'cvssp/audioldm2-large', 'Large model (best quality)'),
    1: ('audioldm2-music', 'cvssp/audioldm2-music', 'Music-focused model'),
    2: ('audioldm2-full', 'cvssp/audioldm2', 'Full/base model'),
}


@dataclass
class GPUSettings:
    """Global GPU and AI generation settings."""
    # Device settings
    device: str = 'cuda'  # Force CUDA by default
    force_gpu: bool = True  # Always try GPU first
    device_index: int = 0  # GPU index for multi-GPU systems
    
    # Generation settings
    steps: int = 150
    cfg_scale: float = 10.0
    scheduler_index: int = 6  # DPM++ by default
    model_index: int = 0  # audioldm2-large by default
    
    # Memory settings
    fp16: bool = True
    attention_slicing: bool = True
    vae_slicing: bool = True
    cpu_offload: bool = False
    
    # Audio settings
    default_duration: float = 5.0
    negative_prompt: str = "low quality, noise, distortion, silence"
    
    def get_scheduler_name(self) -> str:
        """Get current scheduler name."""
        return SCHEDULERS.get(self.scheduler_index, SCHEDULERS[6])[0]
    
    def get_scheduler_class_name(self) -> str:
        """Get current scheduler class name."""
        return SCHEDULERS.get(self.scheduler_index, SCHEDULERS[6])[1]
    
    def get_model_name(self) -> str:
        """Get current model name."""
        return MODELS.get(self.model_index, MODELS[0])[0]
    
    def get_model_id(self) -> str:
        """Get current model HuggingFace ID."""
        return MODELS.get(self.model_index, MODELS[0])[1]


# Global settings instance
_gpu_settings = GPUSettings()

# Global model cache (needs to be defined early for set_gpu_setting)
_model_cache = {
    'pipe': None,
    'model_name': None,
    'device': None,
}


def get_gpu_settings() -> GPUSettings:
    """Get global GPU settings."""
    return _gpu_settings


def set_gpu_setting(key: str, value: Any) -> str:
    """Set a GPU setting.
    
    Parameters
    ----------
    key : str
        Setting name (steps, cfg, scheduler, model, device, etc.)
    value : Any
        New value
    
    Returns
    -------
    str
        Status message
    """
    global _gpu_settings, _model_cache
    
    key = key.lower().strip()
    
    if key in ('steps', 'step', 's'):
        _gpu_settings.steps = max(1, min(500, int(value)))
        return f"OK: Steps set to {_gpu_settings.steps}"
    
    elif key in ('cfg', 'cfg_scale', 'guidance', 'g'):
        _gpu_settings.cfg_scale = max(1.0, min(30.0, float(value)))
        return f"OK: CFG scale set to {_gpu_settings.cfg_scale}"
    
    elif key in ('scheduler', 'sched', 'sk'):
        idx = int(value)
        if idx not in SCHEDULERS:
            return f"ERROR: Invalid scheduler index {idx}. Use 0-{len(SCHEDULERS)-1}"
        _gpu_settings.scheduler_index = idx
        name, cls, desc = SCHEDULERS[idx]
        return f"OK: Scheduler set to [{idx}] {name} ({desc})"
    
    elif key in ('model', 'mdl', 'm'):
        idx = int(value)
        if idx not in MODELS:
            return f"ERROR: Invalid model index {idx}. Use 0-{len(MODELS)-1}"
        _gpu_settings.model_index = idx
        name, hf_id, desc = MODELS[idx]
        # Clear model cache to force reload
        _model_cache['pipe'] = None
        return f"OK: Model set to [{idx}] {name} ({desc})\n  Note: Model will reload on next generation"
    
    elif key in ('device', 'dev', 'd'):
        val = str(value).lower()
        if val in ('cuda', 'gpu', '0', 'nvidia'):
            _gpu_settings.device = 'cuda'
            _gpu_settings.force_gpu = True
        elif val in ('cpu',):
            _gpu_settings.device = 'cpu'
            _gpu_settings.force_gpu = False
        elif val in ('mps', 'apple', 'metal'):
            _gpu_settings.device = 'mps'
            _gpu_settings.force_gpu = True
        else:
            return f"ERROR: Unknown device '{val}'. Use cuda/cpu/mps"
        # Clear model cache to force reload
        _model_cache['pipe'] = None
        return f"OK: Device set to {_gpu_settings.device} (force_gpu={_gpu_settings.force_gpu})"
    
    elif key in ('fp16', 'half'):
        _gpu_settings.fp16 = str(value).lower() in ('true', '1', 'on', 'yes')
        return f"OK: FP16 set to {_gpu_settings.fp16}"
    
    elif key in ('duration', 'dur', 'len'):
        _gpu_settings.default_duration = max(1.0, min(30.0, float(value)))
        return f"OK: Default duration set to {_gpu_settings.default_duration}s"
    
    elif key in ('negative', 'neg', 'negprompt'):
        _gpu_settings.negative_prompt = str(value)
        return f"OK: Negative prompt set to: {_gpu_settings.negative_prompt}"
    
    elif key in ('offload', 'cpu_offload'):
        _gpu_settings.cpu_offload = str(value).lower() in ('true', '1', 'on', 'yes')
        return f"OK: CPU offload set to {_gpu_settings.cpu_offload}"
    
    else:
        return f"ERROR: Unknown setting '{key}'"


def get_gpu_status() -> str:
    """Get comprehensive GPU and settings status."""
    lines = ["=== GPU/AI SETTINGS ===", ""]
    
    # Device info
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            lines.append(f"GPU: {gpu_name} ({vram:.1f} GB)")
            lines.append(f"CUDA: Available ✓")
        else:
            lines.append("GPU: Not detected")
            lines.append("CUDA: Not available")
    except ImportError:
        lines.append("PyTorch: Not installed")
    
    lines.append("")
    lines.append(f"Device: {_gpu_settings.device} (force_gpu={_gpu_settings.force_gpu})")
    lines.append(f"FP16: {_gpu_settings.fp16}")
    
    # Generation settings
    lines.append("")
    lines.append("--- Generation ---")
    lines.append(f"Steps: {_gpu_settings.steps}")
    lines.append(f"CFG Scale: {_gpu_settings.cfg_scale}")
    
    sched_idx = _gpu_settings.scheduler_index
    sched_name, sched_cls, sched_desc = SCHEDULERS.get(sched_idx, SCHEDULERS[6])
    lines.append(f"Scheduler: [{sched_idx}] {sched_name}")
    
    model_idx = _gpu_settings.model_index
    model_name, model_id, model_desc = MODELS.get(model_idx, MODELS[0])
    lines.append(f"Model: [{model_idx}] {model_name}")
    
    lines.append(f"Duration: {_gpu_settings.default_duration}s")
    
    lines.append("")
    lines.append("--- Memory ---")
    lines.append(f"Attention Slicing: {_gpu_settings.attention_slicing}")
    lines.append(f"VAE Slicing: {_gpu_settings.vae_slicing}")
    lines.append(f"CPU Offload: {_gpu_settings.cpu_offload}")
    
    return '\n'.join(lines)


def list_schedulers() -> str:
    """List available schedulers."""
    lines = ["=== SCHEDULERS ===", ""]
    current = _gpu_settings.scheduler_index
    for idx, (name, cls, desc) in SCHEDULERS.items():
        marker = "▶" if idx == current else " "
        lines.append(f"  {marker} [{idx}] {name:12s} - {desc}")
    lines.append("")
    lines.append("Use /gpu sk <index> to select")
    return '\n'.join(lines)


def list_models() -> str:
    """List available models."""
    lines = ["=== MODELS ===", ""]
    current = _gpu_settings.model_index
    for idx, (name, hf_id, desc) in MODELS.items():
        marker = "▶" if idx == current else " "
        lines.append(f"  {marker} [{idx}] {name:18s} - {desc}")
    lines.append("")
    lines.append("Use /gpu model <index> to select")
    return '\n'.join(lines)


# ============================================================================
# GPT2 MONKEY PATCH - CRITICAL: Apply BEFORE any transformers/diffusers import
# ============================================================================
# This fixes missing method errors in GPT2Model when used with AudioLDM2:
# - '_get_initial_cache_position'
# - '_update_model_kwargs_for_generation'
#
# These methods are expected by the diffusers pipeline but missing in some
# transformers versions.

def _apply_gpt2_patch_aggressive():
    """Apply GPT2 monkey patch aggressively to all possible locations.
    
    This patches:
    1. The GPT2 module classes directly
    2. The transformers top-level exports
    3. Any already-instantiated models in the cache
    """
    patched = []
    
    # Method 1: _get_initial_cache_position - returns model_kwargs dict
    def _get_initial_cache_position(self, input_ids=None, model_kwargs=None, *args, **kwargs):
        if model_kwargs is not None:
            return model_kwargs
        if kwargs:
            return kwargs
        return {}
    
    def get_initial_cache_position(self, input_ids=None, model_kwargs=None, *args, **kwargs):
        if model_kwargs is not None:
            return model_kwargs
        if kwargs:
            return kwargs
        return {}
    
    # Method 2: _update_model_kwargs_for_generation - updates and returns model_kwargs
    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, **kwargs):
        # Update model_kwargs with any new values from outputs
        # This is called after each generation step
        if model_kwargs is None:
            model_kwargs = {}
        
        # Handle past_key_values if present
        if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
            model_kwargs['past_key_values'] = outputs.past_key_values
        elif isinstance(outputs, tuple) and len(outputs) > 1:
            model_kwargs['past_key_values'] = outputs[1]
        
        # Update attention mask if needed
        if 'attention_mask' in model_kwargs:
            attention_mask = model_kwargs['attention_mask']
            if attention_mask is not None:
                import torch
                model_kwargs['attention_mask'] = torch.cat([
                    attention_mask,
                    attention_mask.new_ones((attention_mask.shape[0], 1))
                ], dim=-1)
        
        return model_kwargs
    
    # 1. Patch transformers.models.gpt2.modeling_gpt2 module
    try:
        import transformers.models.gpt2.modeling_gpt2 as gpt2_module
        
        classes_to_patch = [
            'GPT2Model',
            'GPT2LMHeadModel', 
            'GPT2DoubleHeadsModel',
            'GPT2ForSequenceClassification',
            'GPT2ForTokenClassification',
            'GPT2PreTrainedModel',
            'GPT2Block',
        ]
        
        for class_name in classes_to_patch:
            if hasattr(gpt2_module, class_name):
                cls = getattr(gpt2_module, class_name)
                # Patch _get_initial_cache_position
                if not hasattr(cls, '_get_initial_cache_position'):
                    cls._get_initial_cache_position = _get_initial_cache_position
                    patched.append(f"gpt2_module.{class_name}._get_initial_cache_position")
                if not hasattr(cls, 'get_initial_cache_position'):
                    cls.get_initial_cache_position = get_initial_cache_position
                # Patch _update_model_kwargs_for_generation
                if not hasattr(cls, '_update_model_kwargs_for_generation'):
                    cls._update_model_kwargs_for_generation = _update_model_kwargs_for_generation
                    patched.append(f"gpt2_module.{class_name}._update_model_kwargs_for_generation")
    except ImportError:
        pass
    except Exception as e:
        print(f"GPT2 patch warning (module level): {e}")
    
    # 2. Patch transformers top-level
    try:
        import transformers
        
        for attr_name in ['GPT2Model', 'GPT2LMHeadModel', 'GPT2PreTrainedModel']:
            if hasattr(transformers, attr_name):
                cls = getattr(transformers, attr_name)
                if not hasattr(cls, '_get_initial_cache_position'):
                    cls._get_initial_cache_position = _get_initial_cache_position
                    patched.append(f"transformers.{attr_name}._get_initial_cache_position")
                if not hasattr(cls, 'get_initial_cache_position'):
                    cls.get_initial_cache_position = get_initial_cache_position
                if not hasattr(cls, '_update_model_kwargs_for_generation'):
                    cls._update_model_kwargs_for_generation = _update_model_kwargs_for_generation
                    patched.append(f"transformers.{attr_name}._update_model_kwargs_for_generation")
    except ImportError:
        pass
    except Exception as e:
        print(f"GPT2 patch warning (transformers level): {e}")
    
    # 3. Patch via PreTrainedModel base if possible
    try:
        from transformers import PreTrainedModel
        if not hasattr(PreTrainedModel, '_get_initial_cache_position'):
            PreTrainedModel._get_initial_cache_position = _get_initial_cache_position
            patched.append("PreTrainedModel._get_initial_cache_position")
        if not hasattr(PreTrainedModel, 'get_initial_cache_position'):
            PreTrainedModel.get_initial_cache_position = get_initial_cache_position
        if not hasattr(PreTrainedModel, '_update_model_kwargs_for_generation'):
            PreTrainedModel._update_model_kwargs_for_generation = _update_model_kwargs_for_generation
            patched.append("PreTrainedModel._update_model_kwargs_for_generation")
    except Exception:
        pass
    
    # 4. Try to patch GenerationMixin which might also need it
    try:
        from transformers.generation.utils import GenerationMixin
        if not hasattr(GenerationMixin, '_get_initial_cache_position'):
            GenerationMixin._get_initial_cache_position = _get_initial_cache_position
            patched.append("GenerationMixin._get_initial_cache_position")
        if not hasattr(GenerationMixin, '_update_model_kwargs_for_generation'):
            GenerationMixin._update_model_kwargs_for_generation = _update_model_kwargs_for_generation
            patched.append("GenerationMixin._update_model_kwargs_for_generation")
    except Exception:
        pass
    
    return patched


def _patch_pipeline_models(pipe):
    """Patch any GPT2 models inside an already-loaded pipeline.
    
    Uses types.MethodType to properly bind methods to instances.
    """
    import types
    
    # Method that returns model_kwargs dict
    def _get_initial_cache_position(self, input_ids=None, model_kwargs=None, *args, **kwargs):
        if model_kwargs is not None:
            return model_kwargs
        if kwargs:
            return kwargs
        return {}
    
    def get_initial_cache_position(self, input_ids=None, model_kwargs=None, *args, **kwargs):
        if model_kwargs is not None:
            return model_kwargs
        if kwargs:
            return kwargs
        return {}
    
    # Method that updates model_kwargs after generation step
    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, **kwargs):
        if model_kwargs is None:
            model_kwargs = {}
        
        # Handle past_key_values if present
        if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
            model_kwargs['past_key_values'] = outputs.past_key_values
        elif isinstance(outputs, tuple) and len(outputs) > 1:
            model_kwargs['past_key_values'] = outputs[1]
        
        # Update attention mask if needed
        if 'attention_mask' in model_kwargs:
            attention_mask = model_kwargs['attention_mask']
            if attention_mask is not None:
                import torch
                model_kwargs['attention_mask'] = torch.cat([
                    attention_mask,
                    attention_mask.new_ones((attention_mask.shape[0], 1))
                ], dim=-1)
        
        return model_kwargs
    
    patched = []
    
    # Helper to patch an object
    def patch_object(obj, name_prefix):
        obj._get_initial_cache_position = types.MethodType(_get_initial_cache_position, obj)
        obj.get_initial_cache_position = types.MethodType(get_initial_cache_position, obj)
        obj._update_model_kwargs_for_generation = types.MethodType(_update_model_kwargs_for_generation, obj)
        patched.append(f'{name_prefix}._get_initial_cache_position')
        patched.append(f'{name_prefix}._update_model_kwargs_for_generation')
        
        # Also patch the class
        cls = type(obj)
        if not hasattr(cls, '_get_initial_cache_position'):
            cls._get_initial_cache_position = _get_initial_cache_position
            cls.get_initial_cache_position = get_initial_cache_position
        if not hasattr(cls, '_update_model_kwargs_for_generation'):
            cls._update_model_kwargs_for_generation = _update_model_kwargs_for_generation
    
    # Check language_model (GPT2 in AudioLDM2)
    if hasattr(pipe, 'language_model') and pipe.language_model is not None:
        patch_object(pipe.language_model, 'language_model')
    
    # Check text_encoder_2 (GPT2 in AudioLDM2)
    if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
        te2 = pipe.text_encoder_2
        patch_object(te2, 'text_encoder_2')
        
        # Patch the inner model instance
        if hasattr(te2, 'model') and te2.model is not None:
            patch_object(te2.model, 'text_encoder_2.model')
        
        # Patch transformer if exists
        if hasattr(te2, 'transformer') and te2.transformer is not None:
            patch_object(te2.transformer, 'text_encoder_2.transformer')
        
        # Patch text_model if exists (CLAP style)
        if hasattr(te2, 'text_model') and te2.text_model is not None:
            patch_object(te2.text_model, 'text_encoder_2.text_model')
    
    return patched


# Apply patch immediately at module load - BEFORE any other imports
_gpt2_patch_result = _apply_gpt2_patch_aggressive()


# ============================================================================
# GPU DETECTION
# ============================================================================

def detect_gpu() -> Dict[str, Any]:
    """Auto-detect user's GPU and return capabilities.
    
    Returns
    -------
    dict
        GPU info: name, vram_gb, cuda_available, device
    """
    gpu_info = {
        'name': 'Unknown',
        'vram_gb': 12.0,  # Default to 3060 specs
        'cuda_available': False,
        'device': 'cpu',
        'detected': False,
        'force_gpu': True,  # Always try GPU first
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['device'] = 'cuda'
            gpu_info['detected'] = True
            
            # Get GPU name and VRAM
            gpu_info['name'] = torch.cuda.get_device_name(0)
            
            # Get VRAM in GB
            total_mem = torch.cuda.get_device_properties(0).total_memory
            gpu_info['vram_gb'] = total_mem / (1024 ** 3)
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon
            gpu_info['cuda_available'] = False
            gpu_info['device'] = 'mps'
            gpu_info['name'] = 'Apple Silicon (MPS)'
            gpu_info['vram_gb'] = 16.0  # Assume unified memory
            gpu_info['detected'] = True
            
    except ImportError:
        pass
    
    # Default to 3060 specs if not detected - still try CUDA
    if not gpu_info['detected']:
        gpu_info['name'] = 'Default (RTX 3060 assumed)'
        gpu_info['vram_gb'] = 12.0
        # Still set cuda as device - let it fail gracefully if not available
        gpu_info['device'] = 'cuda'
        gpu_info['cuda_available'] = True  # Assume available
    
    return gpu_info


def get_optimal_settings(gpu_info: Dict[str, Any]) -> Dict[str, Any]:
    """Get optimal generation settings based on GPU.
    
    Parameters
    ----------
    gpu_info : dict
        GPU information from detect_gpu()
    
    Returns
    -------
    dict
        Optimal settings for AudioLDM2
    """
    vram = gpu_info.get('vram_gb', 12.0)
    
    if vram >= 16:
        # High VRAM - full quality
        return {
            'model': 'audioldm2-large',
            'steps': 150,
            'batch_size': 2,
            'fp16': True,
        }
    elif vram >= 10:
        # Medium VRAM (3060, 3080, etc)
        return {
            'model': 'audioldm2-large',
            'steps': 150,
            'batch_size': 1,
            'fp16': True,
        }
    elif vram >= 6:
        # Low VRAM
        return {
            'model': 'audioldm2-music',  # Smaller model
            'steps': 100,
            'batch_size': 1,
            'fp16': True,
        }
    else:
        # Very low VRAM / CPU
        return {
            'model': 'audioldm2-music',
            'steps': 50,
            'batch_size': 1,
            'fp16': False,
        }


# ============================================================================
# AUDIOLDM2 GENERATION
# ============================================================================

# Note: _model_cache is defined earlier in this file (after _gpu_settings)


def load_audioldm2(
    model_name: str = None,
    device: str = None,
    fp16: bool = None,
) -> Any:
    """Load AudioLDM2 model.
    
    Parameters
    ----------
    model_name : str, optional
        Model variant (uses settings if None)
    device : str, optional
        Device to use (uses settings if None)
    fp16 : bool, optional
        Use half precision (uses settings if None)
    
    Returns
    -------
    pipeline
        AudioLDM2 pipeline ready for generation
    """
    global _model_cache, _gpu_settings
    
    # Use settings for defaults
    if model_name is None:
        model_name = _gpu_settings.get_model_name()
    if device is None:
        device = _gpu_settings.device
    if fp16 is None:
        fp16 = _gpu_settings.fp16
    
    # CRITICAL: Ensure GPT2 patch is applied BEFORE loading any model
    _apply_gpt2_patch_aggressive()
    
    # Ensure device is a string
    device = str(device) if device else 'cuda'
    
    # Return cached model if same config
    if (_model_cache.get('pipe') is not None and 
        _model_cache.get('model_name') == model_name and
        _model_cache.get('device') == device):
        return _model_cache['pipe']
    
    try:
        import torch
        
        # FORCE GPU: Set CUDA device before anything else
        if _gpu_settings.force_gpu and device == 'cuda':
            if torch.cuda.is_available():
                # Force specific GPU device
                torch.cuda.set_device(_gpu_settings.device_index)
                print(f"Forcing CUDA device {_gpu_settings.device_index}: {torch.cuda.get_device_name()}")
            else:
                print("WARNING: CUDA requested but not available!")
                print("  Make sure you have: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        
        # Re-apply patch right before importing diffusers
        _apply_gpt2_patch_aggressive()
        
        from diffusers import AudioLDM2Pipeline
        
        # Apply patch AGAIN after importing diffusers (it may have imported transformers)
        _apply_gpt2_patch_aggressive()
        
        # Model mapping
        model_ids = {
            'audioldm2-large': 'cvssp/audioldm2-large',
            'audioldm2-music': 'cvssp/audioldm2-music',
            'audioldm2-full': 'cvssp/audioldm2',
        }
        
        model_id = model_ids.get(model_name, model_ids['audioldm2-large'])
        
        # Determine actual device - FORCE CUDA if available
        actual_device = device
        if device == 'cuda':
            if torch.cuda.is_available():
                actual_device = 'cuda'
                print(f"Using CUDA: {torch.cuda.get_device_name()}")
            else:
                print("ERROR: CUDA not available! Install CUDA-enabled PyTorch:")
                print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
                if _gpu_settings.force_gpu:
                    raise RuntimeError("CUDA required but not available. Set /gpu device cpu to use CPU.")
                actual_device = 'cpu'
                fp16 = False
        elif device == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                actual_device = 'mps'
            else:
                print("MPS not available, falling back to CPU")
                actual_device = 'cpu'
                fp16 = False
        
        # Load pipeline
        dtype = torch.float16 if fp16 and actual_device != 'cpu' else torch.float32
        
        print(f"Loading AudioLDM2 model: {model_id}")
        print(f"  Device: {actual_device}, Dtype: {dtype}")
        
        pipe = AudioLDM2Pipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        
        # CRITICAL: Patch the pipeline's internal models AFTER loading
        patched_models = _patch_pipeline_models(pipe)
        if patched_models:
            print(f"Patched pipeline models: {len(patched_models)} methods")
        
        # Set up scheduler based on settings
        scheduler_name = _gpu_settings.get_scheduler_class_name()
        try:
            scheduler_class = _get_scheduler_class(scheduler_name)
            if scheduler_class:
                # Special handling for DPM++
                if 'DPMSolver' in scheduler_name:
                    pipe.scheduler = scheduler_class.from_config(
                        pipe.scheduler.config,
                        algorithm_type="dpmsolver++",
                        use_karras_sigmas=True,
                    )
                else:
                    pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
                print(f"Scheduler: {scheduler_name}")
        except Exception as e:
            print(f"Note: Could not set scheduler {scheduler_name}, using default: {e}")
        
        # Move to device FIRST
        pipe = pipe.to(actual_device)
        
        # Apply memory optimizations AFTER moving to device
        if actual_device == 'cuda':
            if _gpu_settings.attention_slicing:
                try:
                    pipe.enable_attention_slicing()
                    print("  Attention slicing: enabled")
                except Exception:
                    pass
            
            if _gpu_settings.vae_slicing:
                try:
                    pipe.enable_vae_slicing()
                    print("  VAE slicing: enabled")
                except Exception:
                    pass
            
            if _gpu_settings.cpu_offload:
                try:
                    pipe.enable_sequential_cpu_offload()
                    print("  CPU offload: enabled")
                except Exception:
                    pass
        
        # Cache the pipeline
        _model_cache['pipe'] = pipe
        _model_cache['model_name'] = model_name
        _model_cache['device'] = actual_device
        
        print(f"AudioLDM2 loaded successfully on {actual_device}")
        return pipe
        
    except ImportError as e:
        raise ImportError(
            f"AudioLDM2 requires: pip install diffusers transformers torch\n"
            f"For CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
            f"Error: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load AudioLDM2: {e}")


def _get_scheduler_class(scheduler_name: str):
    """Get scheduler class by name."""
    try:
        import diffusers
        
        scheduler_map = {
            'DDPMScheduler': diffusers.DDPMScheduler,
            'DDIMScheduler': diffusers.DDIMScheduler,
            'PNDMScheduler': diffusers.PNDMScheduler,
            'LMSDiscreteScheduler': diffusers.LMSDiscreteScheduler,
            'EulerDiscreteScheduler': diffusers.EulerDiscreteScheduler,
            'EulerAncestralDiscreteScheduler': diffusers.EulerAncestralDiscreteScheduler,
            'DPMSolverMultistepScheduler': diffusers.DPMSolverMultistepScheduler,
            'HeunDiscreteScheduler': diffusers.HeunDiscreteScheduler,
        }
        
        # Try direct lookup
        if scheduler_name in scheduler_map:
            return scheduler_map[scheduler_name]
        
        # Try getattr
        if hasattr(diffusers, scheduler_name):
            return getattr(diffusers, scheduler_name)
        
        return None
    except Exception:
        return None


def generate_audio(
    prompt: str,
    negative_prompt: str = None,
    duration: float = None,
    steps: int = None,
    cfg_scale: float = None,
    seed: Optional[int] = None,
    model: str = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """Generate audio from text prompt using AudioLDM2.
    
    Parameters
    ----------
    prompt : str
        Text description of desired audio
    negative_prompt : str, optional
        What to avoid in generation (uses settings if None)
    duration : float, optional
        Length in seconds (uses settings if None)
    steps : int, optional
        Number of diffusion steps (uses settings if None)
    cfg_scale : float, optional
        Classifier-free guidance scale (uses settings if None)
    seed : int, optional
        Random seed for reproducibility
    model : str, optional
        Model variant to use (uses settings if None)
    device : str, optional
        Device override (uses settings if None)
    
    Returns
    -------
    tuple
        (audio_array, sample_rate)
    """
    global _gpu_settings
    import torch
    
    # CRITICAL: Ensure GPT2 patch is applied before ANY model operations
    _apply_gpt2_patch_aggressive()
    
    # Use global settings for defaults
    if negative_prompt is None:
        negative_prompt = _gpu_settings.negative_prompt
    if duration is None:
        duration = _gpu_settings.default_duration
    if steps is None:
        steps = _gpu_settings.steps
    if cfg_scale is None:
        cfg_scale = _gpu_settings.cfg_scale
    if model is None:
        model = _gpu_settings.get_model_name()
    if device is None:
        device = _gpu_settings.device
    
    # Ensure types
    cfg_scale = float(cfg_scale)
    steps = int(steps)
    duration = float(duration)
    
    # Force GPU mode - use settings
    if _gpu_settings.force_gpu:
        if torch.cuda.is_available():
            device = 'cuda'
            # Force specific device index
            torch.cuda.set_device(_gpu_settings.device_index)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
    
    # Ensure device is a string
    device = str(device) if device is not None else 'cuda'
    
    # Load model (this also applies the patch and patches pipeline models)
    pipe = load_audioldm2(model, device)
    
    # Set seed if provided
    generator = None
    if seed is not None:
        seed = int(seed)
        gen_device = device if device != 'mps' else 'cpu'
        generator = torch.Generator(device=gen_device).manual_seed(seed)
    
    # Generate
    # AudioLDM2 works in ~10.24s chunks, so we need to calculate audio_length_in_s
    audio_length = max(1.0, min(30.0, duration))  # Clamp to valid range
    
    # Extra safety: patch the pipeline's internal models one more time
    _patch_pipeline_models(pipe)
    
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            audio_length_in_s=audio_length,
            generator=generator,
        )
    except AttributeError as e:
        error_str = str(e)
        if 'get_initial_cache_position' in error_str or '_get_initial_cache_position' in error_str or '_update_model_kwargs_for_generation' in error_str:
            # Last resort: patch the ACTUAL INSTANCE directly
            print(f"GPT2 method error caught, applying emergency instance patch...")
            
            # Define the missing methods - MUST return model_kwargs dict!
            def _get_initial_cache_position(self, input_ids=None, model_kwargs=None, *args, **kwargs):
                if model_kwargs is not None:
                    return model_kwargs
                if kwargs:
                    return kwargs
                return {}
            
            def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, **kwargs):
                if model_kwargs is None:
                    model_kwargs = {}
                if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                    model_kwargs['past_key_values'] = outputs.past_key_values
                elif isinstance(outputs, tuple) and len(outputs) > 1:
                    model_kwargs['past_key_values'] = outputs[1]
                if 'attention_mask' in model_kwargs:
                    attention_mask = model_kwargs['attention_mask']
                    if attention_mask is not None:
                        model_kwargs['attention_mask'] = torch.cat([
                            attention_mask,
                            attention_mask.new_ones((attention_mask.shape[0], 1))
                        ], dim=-1)
                return model_kwargs
            
            import types
            
            # Helper to patch an object
            def patch_obj(obj, name):
                obj._get_initial_cache_position = types.MethodType(_get_initial_cache_position, obj)
                obj.get_initial_cache_position = types.MethodType(_get_initial_cache_position, obj)
                obj._update_model_kwargs_for_generation = types.MethodType(_update_model_kwargs_for_generation, obj)
                print(f"  Patched {name}: {type(obj).__name__}")
            
            # Patch the pipeline's language_model directly
            if hasattr(pipe, 'language_model') and pipe.language_model is not None:
                patch_obj(pipe.language_model, 'language_model')
            
            # Also patch text_encoder_2 
            if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
                te2 = pipe.text_encoder_2
                patch_obj(te2, 'text_encoder_2')
                if hasattr(te2, 'model') and te2.model is not None:
                    patch_obj(te2.model, 'text_encoder_2.model')
            
            # Also patch the class itself for future instances
            _apply_gpt2_patch_aggressive()
            
            # Retry generation
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                audio_length_in_s=audio_length,
                generator=generator,
            )
        else:
            raise
    
    audio = result.audios[0]
    sample_rate = 16000  # AudioLDM2 outputs at 16kHz
    
    # Convert to numpy if needed
    if hasattr(audio, 'numpy'):
        audio = audio.numpy()
    
    return audio.astype(np.float64), sample_rate


def generate_and_store(
    prompt: str,
    session: "Session",
    duration: float = 5.0,
    steps: int = 150,
    cfg_scale: float = 10.0,
    seed: Optional[int] = None,
    name: Optional[str] = None,
    add_to_dict: bool = True,
) -> Tuple[np.ndarray, str]:
    """Generate audio and store in session buffer and dictionary.
    
    Parameters
    ----------
    prompt : str
        Text description
    session : Session
        Current session
    duration : float
        Duration in seconds
    steps : int
        Diffusion steps
    cfg_scale : float
        Guidance scale
    seed : int, optional
        Random seed
    name : str, optional
        Name for dictionary entry
    add_to_dict : bool
        Add to sample dictionary
    
    Returns
    -------
    tuple
        (audio_array, status_message)
    """
    # Generate audio
    audio, sr = generate_audio(
        prompt=prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
    )
    
    # Resample to session rate if needed
    if sr != session.sample_rate:
        ratio = session.sample_rate / sr
        new_len = int(len(audio) * ratio)
        x_old = np.linspace(0, 1, len(audio))
        x_new = np.linspace(0, 1, new_len)
        audio = np.interp(x_new, x_old, audio)
    
    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak * 0.95).astype(np.float64)
    
    # Store in session buffer
    session.last_buffer = audio
    
    # Add to dictionary
    if add_to_dict:
        try:
            from ..core.pack import get_sample_dictionary
            sample_dict = get_sample_dictionary()
            
            if name is None:
                # Generate name from prompt
                safe_prompt = prompt[:30].replace(' ', '_').replace('/', '-')
                safe_prompt = ''.join(c for c in safe_prompt if c.isalnum() or c in '_-')
                name = f"gen_{safe_prompt}"
            
            final_name = sample_dict.add(name, audio, session.sample_rate, 
                                         pack='generated', tags=['ai', 'generated'])
            dict_msg = f", added to dictionary as '{final_name}'"
        except ImportError:
            dict_msg = ""
    else:
        dict_msg = ""
    
    actual_dur = len(audio) / session.sample_rate
    seed_str = f", seed={seed}" if seed else ""
    
    return audio, f"OK: generated {actual_dur:.2f}s audio{seed_str}{dict_msg}"


def generate_variations(
    prompt: str,
    count: int = 4,
    duration: float = 5.0,
    steps: int = 150,
    cfg_scale: float = 10.0,
    seed_start: int = 42,
    model: str = 'audioldm2-large',
) -> List[Tuple[np.ndarray, int, int]]:
    """Generate multiple variations of the same prompt.
    
    Parameters
    ----------
    prompt : str
        Text description
    count : int
        Number of variations to generate
    duration : float
        Length in seconds
    steps : int
        Diffusion steps
    cfg_scale : float
        Guidance scale
    seed_start : int
        Starting seed (increments for each variation)
    model : str
        Model variant
    
    Returns
    -------
    list
        List of (audio_array, sample_rate, seed) tuples
    """
    results = []
    for i in range(count):
        seed = seed_start + i
        audio, sr = generate_audio(
            prompt=prompt,
            duration=duration,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            model=model,
        )
        results.append((audio, sr, seed))
    
    return results


# ============================================================================
# AUDIO ENHANCEMENT PROMPTS
# ============================================================================

def enhance_prompt(base_prompt: str, style: str = 'general') -> str:
    """Enhance a basic prompt with quality modifiers.
    
    Parameters
    ----------
    base_prompt : str
        User's basic prompt
    style : str
        Style category: 'general', 'music', 'sfx', 'ambient', 'percussion'
    
    Returns
    -------
    str
        Enhanced prompt with quality modifiers
    """
    quality_suffixes = {
        'general': ", high quality, professional recording, clear audio",
        'music': ", high quality music production, professional mix, studio recording",
        'sfx': ", high quality sound effect, clean recording, professional foley",
        'ambient': ", immersive atmosphere, high quality field recording, spatial audio",
        'percussion': ", punchy drums, professional drum recording, tight and clean",
    }
    
    suffix = quality_suffixes.get(style, quality_suffixes['general'])
    return base_prompt.strip() + suffix


def get_negative_prompt(style: str = 'general') -> str:
    """Get appropriate negative prompt for style.
    
    Parameters
    ----------
    style : str
        Style category
    
    Returns
    -------
    str
        Negative prompt
    """
    negatives = {
        'general': "low quality, noise, distortion, muffled, amateur recording",
        'music': "low quality, noise, distortion, off-key, out of tune, amateur",
        'sfx': "low quality, background noise, distortion, reverb, music",
        'ambient': "low quality, harsh, digital artifacts, music, voice",
        'percussion': "low quality, muddy, boomy, thin, amateur recording, noise",
    }
    
    return negatives.get(style, negatives['general'])


# ============================================================================
# PRESET PROMPTS
# ============================================================================

PRESET_PROMPTS = {
    # Drums
    'kick_sub': "deep subby kick drum, 808 style, punchy low end",
    'kick_acoustic': "acoustic kick drum, natural room sound, punchy",
    'kick_punchy': "punchy electronic kick drum, tight attack, deep sub",
    'snare_crack': "snappy snare drum, bright crack, punchy attack",
    'snare_fat': "fat layered snare, punchy with body",
    'snare_tight': "tight electronic snare, short decay, punchy",
    'hihat_closed': "crisp closed hi-hat, tight and clean",
    'hihat_open': "open hi-hat, shimmering decay",
    'clap': "big clap sound, layered hands, reverb tail",
    'rim': "rimshot, tight and bright, percussive",
    
    # Bass
    'bass_sub': "deep sub bass, clean sine wave, powerful low end",
    'bass_growl': "aggressive dubstep growl bass, modulated, heavy",
    'bass_reese': "reese bass, detuned saw waves, dark and powerful",
    'bass_fm': "fm bass, metallic and punchy, electronic",
    'bass_808': "808 bass, long sustain, deep sub frequencies",
    
    # Synths
    'lead_saw': "bright saw lead synth, cutting through the mix",
    'lead_square': "retro square wave lead, video game style",
    'lead_pluck': "plucky synth lead, short decay, bright",
    'pad_lush': "lush evolving pad, warm and atmospheric",
    'pad_dark': "dark ambient pad, mysterious and deep",
    'pad_strings': "string pad, orchestral texture, warm",
    'arp': "arpeggiated synth pattern, rhythmic, electronic",
    
    # FX
    'riser': "tension building riser, sweeping upward, dramatic",
    'impact': "cinematic impact hit, deep boom with crack",
    'sweep': "filter sweep, bright to dark transition",
    'whoosh': "fast whoosh sound effect, dramatic movement",
    'noise_white': "white noise texture, hissing, static",
    'noise_pink': "pink noise, softer, ambient texture",
    
    # Textures
    'texture_metallic': "metallic texture, industrial, resonant",
    'texture_organic': "organic texture, natural sounds, evolving",
    'texture_digital': "digital glitch texture, electronic artifacts",
}
