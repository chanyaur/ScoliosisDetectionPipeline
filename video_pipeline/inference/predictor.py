"""
Run inference on processed silhouettes using trained models
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path

from .model_loader import ModelLoader

logger = logging.getLogger(__name__)


class ScoliosisPredictor:
    """
    Predict scoliosis classification from silhouette sequences
    """
    
    def __init__(self,
                 model_path: str,
                 model_type: str = 'sconet',
                 device: str = 'cpu'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: Type of model ('sconet', 'sconet_mt', 'sconet_binary', or 'sconet_mt_binary')
            device: Device for inference
        """
        self.model_type = model_type
        self.device = device
        
        # Initialize model loader
        self.model_loader = ModelLoader(device)
        
        # Load model
        self.model = self.model_loader.load_model(model_type, model_path)
        
        # Get class names based on model type
        if 'binary' in model_type:
            self.class_names = self.model_loader.get_class_names_binary()
        else:
            self.class_names = self.model_loader.get_class_names()
        
        logger.info(f"Predictor initialized with {model_type} model")
        
    def predict(self, silhouettes: np.ndarray) -> Dict:  # takes one silhouette sequence, preprocesses it, and runs it into selected model for prediction. Outputs pred and probability
        """
        Predict scoliosis classification
        
        Args:
            silhouettes: Silhouette sequence (300, 64, 64)
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess silhouettes
        input_tensor = self.model_loader.preprocess_silhouettes(silhouettes)
        
        # Run inference
        with torch.no_grad():  # doesn't update the gradient: forward pass but no learning
            if self.model_type == 'sconet':
                logits = self.model(input_tensor)
                angle_pred = None
            elif self.model_type == 'sconet_mt':
                logits, angle_pred = self.model(input_tensor)
                angle_pred = angle_pred.cpu().numpy()[0]
            elif self.model_type == 'sconet_binary':
                logits = self.model(input_tensor)
                angle_pred = None
            elif self.model_type == 'sconet_mt_binary':
                logits, angle_pred = self.model(input_tensor)
                angle_pred = angle_pred.cpu().numpy()[0]
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
                
        # Get probabilities
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # returns array with probabilities of each class
        
        # Get predicted class
        pred_class = np.argmax(probs)
        pred_confidence = probs[pred_class]
        
        # Prepare results
        results = {
            'predicted_class': int(pred_class),
            'predicted_label': self.class_names[pred_class],
            'confidence': float(pred_confidence),
            'probabilities': {
                self.class_names[i]: float(probs[i])
                for i in range(len(probs))
            }
        }
        
        # Add angle prediction if available
        if angle_pred is not None:
            results['predicted_angle'] = float(angle_pred)
            
        return results
    
    def predict_batch(self, silhouette_list: List[np.ndarray]) -> List[Dict]:  # runs predict for multiple sequences
        """
        Predict for multiple silhouette sequences
        
        Args:
            silhouette_list: List of silhouette sequences
            
        Returns:
            List of prediction results
        """
        results = []
        
        for silhouettes in silhouette_list:
            try:
                result = self.predict(silhouettes)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict: {e}")
                results.append(None)
                
        return results
    
    def get_risk_assessment(self, prediction: Dict) -> str:  # tells the user what further action to take based on classification and probability
        """
        Get risk assessment based on prediction
        
        Args:
            prediction: Prediction results
            
        Returns:
            Risk assessment message
        """
        pred_class = prediction['predicted_class']
        confidence = prediction['confidence']
        
        if pred_class == 0:  # Positive
            if confidence > 0.8:
                return "HIGH RISK: Strong indication of scoliosis. Medical evaluation recommended."
            else:
                return "MODERATE RISK: Possible scoliosis. Further screening recommended."
            
        elif pred_class == 2 or "binary" in self.model_type:  # Negative
            if confidence > 0.8:
                return "LOW RISK: No significant signs of scoliosis detected."
            else:
                return "LOW RISK: Unlikely scoliosis, but confidence is moderate."
            
        else:  # Neutral
            return "BORDERLINE: Monitor closely. Follow-up screening recommended in 6 months."
                
    def explain_prediction(self, prediction: Dict) -> str:  # shows the user all probabilities, and adds angle and medical significance (if ScoNet-MT)
        """
        Generate explanation for prediction
        
        Args:
            prediction: Prediction results
            
        Returns:
            Explanation text
        """
        explanation = []
        
        # Add main prediction
        explanation.append(f"Classification: {prediction['predicted_label']}")
        explanation.append(f"Confidence: {prediction['confidence']:.1%}")
        
        # Add probability breakdown
        explanation.append("\nProbability Breakdown:")
        for class_name, prob in prediction['probabilities'].items():
            explanation.append(f"  - {class_name}: {prob:.1%}")
            
        # Add angle if available
        if 'predicted_angle' in prediction:
            angle = prediction['predicted_angle']
            explanation.append(f"\nEstimated Cobb Angle: {angle:.1f}°")
            
            # Interpret angle
            if angle < 10:
                explanation.append("  (Normal range)")
            elif angle < 25:
                explanation.append("  (Mild scoliosis)")
            elif angle < 40:
                explanation.append("  (Moderate scoliosis)")
            else:
                explanation.append("  (Severe scoliosis)")
                
        return "\n".join(explanation)


class EnsemblePredictor:
    """
    Ensemble predictions from multiple models
    """
    
    def __init__(self,
                 sconet_path: str,
                 sconet_mt_path: str,
                 device: str = 'cpu'):
        """
        Initialize ensemble predictor
        
        Args:
            sconet_path: Path to ScoNet checkpoint
            sconet_mt_path: Path to ScoNet-MT checkpoint
            device: Device for inference
        """
        # Initialize both predictors
        self.sconet_predictor = ScoliosisPredictor(
            sconet_path, 'sconet', device
        )
        self.sconet_mt_predictor = ScoliosisPredictor(
            sconet_mt_path, 'sconet_mt', device
        )
        
    def predict(self, silhouettes: np.ndarray) -> Dict:  # run both models for prediction and average their probabilities to get results. Uses angle from ScoNet-MT
        """
        Ensemble prediction using both models
        
        Args:
            silhouettes: Silhouette sequence
            
        Returns:
            Combined prediction results
        """
        # Get predictions from both models
        sconet_pred = self.sconet_predictor.predict(silhouettes)
        sconet_mt_pred = self.sconet_mt_predictor.predict(silhouettes)
        
        # Average probabilities
        avg_probs = {}
        for class_name in sconet_pred['probabilities'].keys():
            avg_probs[class_name] = (
                sconet_pred['probabilities'][class_name] +
                sconet_mt_pred['probabilities'][class_name]
            ) / 2
            
        # Get ensemble prediction
        pred_class = max(avg_probs, key=avg_probs.get)
        pred_idx = list(self.sconet_predictor.class_names.values()).index(pred_class)
        
        # Prepare ensemble results
        results = {
            'predicted_class': pred_idx,
            'predicted_label': pred_class,
            'confidence': avg_probs[pred_class],
            'probabilities': avg_probs,
            'sconet_prediction': sconet_pred,
            'sconet_mt_prediction': sconet_mt_pred,
            'ensemble_method': 'average'
        }
        
        # Use angle from MT model
        if 'predicted_angle' in sconet_mt_pred:
            results['predicted_angle'] = sconet_mt_pred['predicted_angle']
            
        return results
