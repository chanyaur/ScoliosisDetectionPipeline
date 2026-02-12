"""
Load and manage trained ScoNet models for inference
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import sys

# Add parent directory to path for model imports
sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Load and manage ScoNet models for inference
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize model loader
        
        Args:
            device: Device to load models on ('cpu' or 'cuda')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        logger.info(f"Model loader initialized on device: {self.device}")
        
    def load_sconet(self, checkpoint_path: str) -> nn.Module:  # just loads up the pretrained model from checkpoint
        """
        Load ScoNet model from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Loaded model in eval mode
        """

        try:
            # Import ScoNet model
            from scoliosis_app.models.sconet import ScoNet
            
            # Initialize model
            model = ScoNet(num_classes=3, n_frames=30)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)  # restore the trained ScoNet parameters w/out retraining
            # this was changed to false!! TAKE NOTE OF THIS
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:  # state dict gets the parameters
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])  # it may be named either one of the two so check both for compatibility
            else:
                model.load_state_dict(checkpoint)
                
            model.to(self.device)
            model.eval()
            
            logger.info(f"Loaded ScoNet model from {checkpoint_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load ScoNet model: {e}")
            raise
            
    def load_sconet_mt(self, checkpoint_path: str) -> nn.Module:  # loads up pretrained ScoNet-MT
        """
        Load ScoNet-MT model from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Loaded model in eval mode
        """
        try:
            # Import ScoNet-MT model
            from scoliosis_app.models.sconet import ScoNetMT
            
            # Initialize model
            model = ScoNetMT(num_classes=3, n_frames=30)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.to(self.device)
            model.eval()
            
            logger.info(f"Loaded ScoNet-MT model from {checkpoint_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load ScoNet-MT model: {e}")
            raise
            
    def load_model(self, model_type: str, checkpoint_path: str) -> nn.Module:  # ez just run whichever one of the two functions above to load desired model
        """
        Load model based on type
        
        Args:
            model_type: Type of model ('sconet' or 'sconet_mt')
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Loaded model
        """
        if model_type == 'sconet':
            return self.load_sconet(checkpoint_path)
        elif model_type == 'sconet_mt':
            return self.load_sconet_mt(checkpoint_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def preprocess_silhouettes(self, silhouettes: np.ndarray) -> torch.Tensor:  # takes your video silhouettes, gets 30 of them, and converts them to 0 and 1 tensor input to prep for training
        
        """
        Preprocess silhouette sequence for model input
        
        Args:
            silhouettes: Silhouette sequence (300, 64, 64)
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Sample 30 frames from 300 - because we instantiate the model with 30 frames to match what it was trained on.
        indices = np.linspace(0, len(silhouettes) - 1, 30, dtype=int)
        sampled = silhouettes[indices]
        
        # Convert to tensor
        tensor = torch.from_numpy(sampled).float()
        
        # Ensure values are in [0, 1]
        if tensor.max() > 1.0:
            tensor = tensor / 255.0  # not black and white but 0 1 so can pass into model
            
        # Add batch dimension - neural networks just expect this
        tensor = tensor.unsqueeze(0)  # (1, 30, 64, 64)
        
        return tensor.to(self.device)
        
    def get_class_names(self) -> Dict[int, str]:  # simple int index - str name mapping
        """
        Get mapping of class indices to names
        
        Returns:
            Dictionary mapping class index to name
        """
        return {
            0: 'Positive (Scoliosis)',
            1: 'Neutral (Borderline)',
            2: 'Negative (Healthy)'
        }
