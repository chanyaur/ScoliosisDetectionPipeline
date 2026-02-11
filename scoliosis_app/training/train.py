"""
Training script for ScoNet models on Scoliosis1K dataset
Implements training loop with proper evaluation metrics
"""

import os
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sconet import create_sconet
 
from dataset.data_loader import get_data_loaders
from utils.losses import FocalLoss, CombinedLoss, get_class_weights


class Trainer:
    """Trainer class for ScoNet models"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint directory
        self.checkpoint_dir = self.output_dir / 'checkpoints'  # just configs the output folder name: sconet_[date]_number
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')  # initializes TensorBoard logging and can store in folder
        
        # Initialize best metrics
        self.best_val_acc = 0.0
        self.best_val_sensitivity = 0.0
        self.best_epoch = 0
        
    def setup_model(self):
        """Initialize model"""
        self.model = create_sconet(
            model_type=self.config['model_type'],
            num_classes=self.config['num_classes'],
            n_frames=self.config['n_frames']
        )
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    def setup_data(self):
        """Setup data loaders"""
        loaders = get_data_loaders(
            data_root=self.config['data_root'],
            batch_size=self.config['batch_size'],
            n_frames=self.config['n_frames'],
            num_workers=self.config['num_workers']
        )
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']
        self.test_loader = loaders['test']
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"Test batches: {len(self.test_loader)}")
        
    def setup_loss_and_optimizer(self):
        """Setup loss function and optimizer"""
        # Get class weights
        class_weights = get_class_weights(self.config['data_root'])
        
        if self.config['model_type'] == 'base':
            self.criterion = FocalLoss(
                alpha=class_weights,
                gamma=self.config['focal_gamma']
            )
        else:  # multi-task
            self.criterion = CombinedLoss(
                class_weight=self.config['class_weight'],
                angle_weight=self.config['angle_weight'],
                alpha=class_weights,
                gamma=self.config['focal_gamma']
            )
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup learning rate scheduler
        
        print(torch.__version__)


        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, #verbose=True  # verbose just used for logging
        )
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            if self.config['model_type'] == 'base':
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
            else:  # multi-task
                class_outputs, angle_outputs = self.model(data)
                # For now, we don't have angle labels, so pass None
                loss, class_loss, angle_loss = self.criterion(
                    class_outputs, angle_outputs, labels, None
                )
                outputs = class_outputs
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Calculate per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=[0, 1, 2]
        )
        
        # Log epoch metrics
        self.writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
        self.writer.add_scalar('Train/Accuracy', accuracy, epoch)
        
        return avg_loss, accuracy, recall[0]  # Return sensitivity for positive class
    
    def validate(self, epoch, loader, phase='Val'):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in tqdm(loader, desc=f'{phase}'):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.config['model_type'] == 'base':
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)
                else:  # multi-task
                    class_outputs, angle_outputs = self.model(data)
                    loss, _, _ = self.criterion(
                        class_outputs, angle_outputs, labels, None
                    )
                    outputs = class_outputs
                
                # Track metrics
                total_loss += loss.item()
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Calculate per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=[0, 1, 2]
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Log metrics
        self.writer.add_scalar(f'{phase}/Loss', avg_loss, epoch)
        self.writer.add_scalar(f'{phase}/Accuracy', accuracy, epoch)
        self.writer.add_scalar(f'{phase}/Sensitivity_Positive', recall[0], epoch)
        self.writer.add_scalar(f'{phase}/Sensitivity_Neutral', recall[1], epoch)
        self.writer.add_scalar(f'{phase}/Sensitivity_Negative', recall[2], epoch)
        
        # Print detailed report
        if phase == 'Val':
            print(f"\n{phase} Metrics - Epoch {epoch+1}")
            print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            print(f"Sensitivity - Positive: {recall[0]:.4f}, Neutral: {recall[1]:.4f}, Negative: {recall[2]:.4f}")
            print("\nClassification Report:")
            print(classification_report(
                all_labels, all_preds,
                labels=[0, 1, 2],
                target_names=['Positive', 'Neutral', 'Negative'],
                zero_division=0
        ))

        
        return avg_loss, accuracy, recall[0], cm
    
    def save_checkpoint(self, epoch, val_acc, val_sensitivity):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_acc,
            'val_sensitivity': val_sensitivity,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if improved
        if val_sensitivity > self.best_val_sensitivity:
            self.best_val_sensitivity = val_sensitivity
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with sensitivity: {val_sensitivity:.4f}")
    
    def train(self):
        """Main training loop"""
        print("\nStarting training...")
        
        # Setup
        self.setup_model()
        self.setup_data()
        self.setup_loss_and_optimizer()
        
        # Training loop
        for epoch in range(self.config['epochs']):
            # Train
            train_loss, train_acc, train_sens = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_sens, val_cm = self.validate(epoch, self.val_loader)
            
            # Update learning rate
            self.scheduler.step(val_sens)
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_acc, val_sens)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} Summary:")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Sens: {train_sens:.4f}")
            print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Sens: {val_sens:.4f}")
            print(f"Best Val Sensitivity: {self.best_val_sensitivity:.4f} (Epoch {self.best_epoch+1})")
            
            test(self, epoch)
            
            # Early stopping
            # if epoch - self.best_epoch > self.config['patience']:  # so instead of adjusting the learning rate we stop entirely?
            #     print(f"\nEarly stopping triggered after {epoch+1} epochs")
            #     break
        
        print("\nTraining completed!")
        print(f"Best validation sensitivity: {self.best_val_sensitivity:.4f}")
    
def test(self, epoch):
    """Test the best model"""
    print("\nEvaluating best model on test set...")

    # Evaluate
    self.model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []  # store predicted probabilities for AUC

    with torch.no_grad():
        for data, labels in tqdm(self.test_loader, desc='Test'):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass (unchanged)
            if self.config['model_type'] == 'base':
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
            else:
                class_outputs, angle_outputs = self.model(data)
                loss, _, _ = self.criterion(class_outputs, angle_outputs, labels, None)
                outputs = class_outputs
            
            total_loss += loss.item()

            # 🆕 compute full softmax probabilities for all 3 classes
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
    # 🆕 concatenate probability arrays from all batches
    all_probs = np.concatenate(all_probs, axis=0)

    # Calculate metrics (unchanged structure)
    avg_loss = total_loss / len(self.test_loader)
    test_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1, 2]
    )
    test_cm = confusion_matrix(all_labels, all_preds)

    # 🆕 Compute multi-class AUC safely
    from sklearn.metrics import roc_auc_score
    try:
        test_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError:
        test_auc = float('nan')
    
    # Save test results (same structure)
    results = {
        'test_auc': float(test_auc),                   # 🆕 same key as before
        'test_accuracy': float(test_acc),
        'test_sensitivity': float(recall[0]),          # class 0 = Positive
        'test_loss': float(avg_loss),
        'confusion_matrix': test_cm.tolist(),
        'best_epoch': self.best_epoch + 1,
        'config': self.config
    }
    
    with open(self.output_dir / f'test_results_{epoch}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results (identical output)
    print("BEST MODEL")
    print(f"\nTest Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Sensitivity (Positive): {recall[0]:.4f}")  # still prints only class 0
    print(f"AUC: {test_auc:.4f}")
    print(f"Confusion Matrix:")
    print(test_cm)
    
    return test_acc, recall[0]



def get_config():
    """Get training configuration"""
    config = {
        # Model
        'model_type': 'mt',  # 'base' or 'mt'
        'num_classes': 3,
        'n_frames': 30,
        
        # Data
        'data_root': './Scoliosis1K-pkl',  # Updated path
        'batch_size': 16,
        'num_workers': 0,  # Disable multiprocessing to avoid segfaults
        
        # Training
        'epochs': 50,  # EPOCH NUM - change this later
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'patience': 20,
        
        # Loss
        'focal_gamma': 2.0,
        'class_weight': 1.0,  # For multi-task
        'angle_weight': 0.5,  # For multi-task
        
        # Output
        'output_dir': f'scoliosis_app/experiments/sconet_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    return config


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train ScoNet on Scoliosis1K')
    parser.add_argument('--model_type', type=str, default='base',
                       choices=['base', 'mt'], help='Model type')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')  # EPOCH NUM
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Get config and update with args
    config = get_config()
    config['model_type'] = args.model_type
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    
    # Print config
    print("Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()