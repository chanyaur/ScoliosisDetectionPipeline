"""
Inference components for scoliosis prediction
"""

from .model_loader import ModelLoader
from .predictor import ScoliosisPredictor, EnsemblePredictor

__all__ = [
    'ModelLoader',
    'ScoliosisPredictor',
    'EnsemblePredictor'
]
