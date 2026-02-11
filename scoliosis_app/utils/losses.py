"""
Loss functions for scoliosis detection training
Including focal loss for handling class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in scoliosis detection
    Based on "Focal Loss for Dense Object Detection" by Lin et al.
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', num_classes=3):
        """
        Args:
            alpha: Class weights (tensor or list). If None, all classes weighted equally.
            gamma: Focusing parameter for modulating loss (default: 2.0)
            reduction: 'none', 'mean', or 'sum'
            num_classes: Number of classes
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
            
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions from model (before softmax), shape (N, C)
            targets: Ground truth labels, shape (N,)
        """
        # Get softmax probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get class probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get the probability of the true class
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Calculate focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t).pow(self.gamma)
        
        # Calculate focal loss
        loss = focal_term * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != loss.device:
                self.alpha = self.alpha.to(loss.device)
            
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task learning (ScoNet-MT)
    Combines classification loss with angle regression loss
    """
    
    def __init__(self, class_weight=1.0, angle_weight=0.5, 
                 alpha=None, gamma=2.0):
        """
        Args:
            class_weight: Weight for classification loss
            angle_weight: Weight for angle regression loss
            alpha: Class weights for focal loss
            gamma: Focusing parameter for focal loss
        """
        super().__init__()
        self.class_weight = class_weight
        self.angle_weight = angle_weight
        
        # Classification loss (focal loss for imbalanced classes)
        self.class_loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        
        # Regression loss for angle prediction
        self.angle_loss_fn = nn.SmoothL1Loss()
        
    def forward(self, class_pred, angle_pred, class_target, angle_target=None):
        """
        Args:
            class_pred: Classification predictions (N, C)
            angle_pred: Angle predictions (N,)
            class_target: Classification targets (N,)
            angle_target: Angle targets (N,), optional
        """
        # Classification loss
        class_loss = self.class_loss_fn(class_pred, class_target)
        
        # Angle regression loss (only for positive cases if angle_target provided)
        if angle_target is not None:
            # Only compute angle loss for positive cases
            positive_mask = class_target == 0  # Assuming 0 is positive class
            
            if positive_mask.sum() > 0:
                angle_loss = self.angle_loss_fn(
                    angle_pred[positive_mask], 
                    angle_target[positive_mask]
                )
            else:
                angle_loss = torch.tensor(0.0, device=class_pred.device)
        else:
            angle_loss = torch.tensor(0.0, device=class_pred.device)
        
        # Combined loss
        total_loss = self.class_weight * class_loss + self.angle_weight * angle_loss
        
        return total_loss, class_loss, angle_loss


def get_class_weights(dataset_root='Scoliosis1K-pkl'):
    """
    Calculate class weights for the Scoliosis1K dataset
    Based on the class distribution mentioned in the paper
    """
    # Class distribution from paper
    # Positive: 493 (0-492)
    # Negative: 800 (493-1292) 
    # Neutral: 200 (1293-1492)
    
    n_positive = 493
    n_negative = 800
    n_neutral = 200
    total = n_positive + n_negative + n_neutral
    
    # Calculate weights (inverse of frequency)
    weights = {
        'positive': total / (3 * n_positive),  # class 0
        'neutral': total / (3 * n_neutral),    # class 1
        'negative': total / (3 * n_negative),  # class 2
    }
    
    # Return as tensor in correct order
    weight_tensor = torch.tensor([
        weights['positive'],
        weights['neutral'],
        weights['negative']
    ], dtype=torch.float32)
    
    return weight_tensor


if __name__ == "__main__":
    # Test focal loss
    batch_size = 8
    num_classes = 3
    
    # Create dummy predictions and targets
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Get class weights
    class_weights = get_class_weights()
    print(f"Class weights: {class_weights}")
    
    # Test focal loss
    focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
    loss = focal_loss(predictions, targets)
    print(f"Focal loss: {loss.item():.4f}")
    
    # Test combined loss for multi-task
    angle_predictions = torch.randn(batch_size)
    angle_targets = torch.randn(batch_size) * 30  # Angles in degrees
    
    combined_loss = CombinedLoss(alpha=class_weights)
    total_loss, class_loss, angle_loss = combined_loss(
        predictions, angle_predictions, targets, angle_targets
    )
    print(f"Combined loss: {total_loss.item():.4f}")
    print(f"  Classification loss: {class_loss.item():.4f}")
    print(f"  Angle loss: {angle_loss.item():.4f}")
