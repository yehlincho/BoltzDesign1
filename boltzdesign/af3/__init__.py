"""Warm, batched AlphaFold 3 validation for BoltzDesign.

AlphaFold 3 itself is not bundled: it is licensed CC BY-NC-SA 4.0 and its weights
require agreeing to DeepMind's terms. Install it separately and point the pipeline at
it with --alphafold_dir (or $AF3_ROOT); see runtime.py.
"""

from .base import ValidationInput, ValidationResult
from .validator import AF3Validator

__all__ = ["AF3Validator", "ValidationInput", "ValidationResult"]
