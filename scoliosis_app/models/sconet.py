"""
ScoNet Model Implementation for Scoliosis Detection
Based on the Scoliosis1K paper architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalPooling(nn.Module):
    """Temporal pooling module for aggregating temporal features"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        
        # B = batch size
        # C = number of channels (like RGB or feature maps)
        # T = temporal length (number of frames in a clip)
        # H, W = spatial height and width

        b, c, t, h, w = x.size()
        
        # Reshape for 1D convolution along temporal dimension
        x = x.permute(0, 3, 4, 1, 2)  # (B, H, W, C, T) -- basically just rearrange how the info is presented (dimensions go in what order?)
        x = x.reshape(b * h * w, c, t)  # (B*H*W, C, T) -- We now have BHW sequences, each of shape (C, T). -- why??
        
        # Apply temporal convolution
        x = self.conv1d(x)  # applies linear function across every number (?)
        x = self.bn(x)  # batch normalization
        x = self.relu(x)  # introduces non-linearity by outputting the input directly if positive, else output 0 if negative. non-linear activation functions typically perform better
        
        # Max pooling along  temporal dimension
        x = F.max_pool1d(x, kernel_size=x.size(-1))  # (B*H*W, C_out, 1)
        # What this does is take a sliding window and takes the max value PER PIXEL PER FEATURE CHANNEL (just one) PER TIME
        
        # Reshape back
        x = x.squeeze(-1)  # (B*H*W, C_out)
        x = x.reshape(b, h, w, -1)  # (B, H, W, C_out)
        x = x.permute(0, 3, 1, 2)  # (B, C_out, H, W)
        
        return x


class HorizontalPooling(nn.Module):
    """Horizontal pooling module for part-based feature extraction"""
    
    def __init__(self, n_parts=16):
        super().__init__()
        self.n_parts = n_parts  # number of horizontal bands
        
    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        
        # Split horizontally into parts
        part_h = h // self.n_parts  # split each frame into 16 horizontal bands - so basically the amount in each part
        parts = []
        
        for i in range(self.n_parts):
            start_h = i * part_h
            end_h = start_h + part_h if i < self.n_parts - 1 else h  # get the coords of start and end points
            
            # Extract part and apply max pooling
            part = x[:, :, start_h:end_h, :]  # (B, C, part_h, W) -- colons = TAKE ALL, start_h:end_h = TAKE ONLY
            part_pooled = F.max_pool2d(part, kernel_size=(part.size(2), part.size(3)))  # (B, C, 1, 1) -- collapse into a single (max) number per channel -- are these numbers usually 0/1?
            parts.append(part_pooled.squeeze(-1).squeeze(-1))  # (B, C)
        
        # Stack parts
        x_parts = torch.stack(parts, dim=2)  # (B, C, n_parts) -- stacking them allows the model to analyze each portion of body separately
        
        return x_parts


class ScoNet(nn.Module):
    """
    ScoNet model for scoliosis detection from gait silhouettes
    Based on the architecture described in the Scoliosis1K paper
    """
    
    def __init__(self, num_classes=3, input_channels=1, n_frames=30):  # initialization
        super().__init__()
        self.num_classes = num_classes
        self.n_frames = n_frames
        
        # Backbone CNN for feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),  # what is this "inplace"
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),  # why is the output dimension increasing by 2 each time
            nn.ReLU(inplace=True),
        )
        
        # Temporal pooling module
        self.temporal_pool = TemporalPooling(128, 256, kernel_size=3)
        
        # Horizontal pooling module
        self.horizontal_pool = HorizontalPooling(n_parts=16)
        
        # Feature processing -- what is this (come back to understand)
        self.feature_conv = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16, 512),  # 16 parts
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x: (B, T, H, W) or (B, T, C, H, W)
        
        # Handle different input formats
        if x.dim() == 4:
            # Add channel dimension if not present
            x = x.unsqueeze(2)  # (B, T, 1, H, W)
        
        b, t, c, h, w = x.size()
        
        # Process each frame through backbone
        frame_features = []
        for i in range(t):
            frame = x[:, i]  # (B, C, H, W)
            
            # Extract features
            feat = self.conv1(frame)
            feat = self.conv2(feat)
            feat = self.conv3(feat)
            
            frame_features.append(feat)
        
        # Stack temporal features
        temporal_features = torch.stack(frame_features, dim=2)  # (B, C, T, H, W)
        
        # Apply temporal pooling
        pooled_features = self.temporal_pool(temporal_features)  # (B, C, H, W)
        
        # Apply horizontal pooling
        part_features = self.horizontal_pool(pooled_features)  # (B, C, n_parts)
        
        # Process part features -- what is this - come back
        processed_features = self.feature_conv(part_features)  # (B, C', n_parts)
        
        # Flatten and classify
        features_flat = processed_features.view(b, -1)  # (B, C' * n_parts)
        logits = self.classifier(features_flat)
        
        return logits


class ScoNetMT(ScoNet):
    """
    Multi-task version of ScoNet with auxiliary angle prediction
    """
    
    def __init__(self, num_classes=3, input_channels=1, n_frames=30):
        super().__init__(num_classes, input_channels, n_frames)
        
        # Additional angle regression head
        self.angle_regressor = nn.Sequential(  # HUH what is this explain this
            nn.Linear(128 * 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)  # Single angle output
        )
        
    def forward(self, x):
        # Get base features
        if x.dim() == 4:
            x = x.unsqueeze(2)
        
        b, t, c, h, w = x.size()
        
        # Process through backbone
        frame_features = []
        for i in range(t):
            frame = x[:, i]
            feat = self.conv1(frame)
            feat = self.conv2(feat)
            feat = self.conv3(feat)
            frame_features.append(feat)
        
        temporal_features = torch.stack(frame_features, dim=2)
        pooled_features = self.temporal_pool(temporal_features)
        part_features = self.horizontal_pool(pooled_features)
        processed_features = self.feature_conv(part_features)
        features_flat = processed_features.view(b, -1)
        
        # Classification output
        class_logits = self.classifier(features_flat)
        
        # up until here is the same
        
        # Angle regression output
        angle_pred = self.angle_regressor(features_flat)
        
        return class_logits, angle_pred.squeeze(-1)


def create_sconet(model_type='base', num_classes=2, n_frames=30):
    """
    Factory function to create ScoNet models
    
    Args:
        model_type: 'base' for ScoNet or 'mt' for ScoNet-MT
        num_classes: Number of classification classes
        n_frames: Number of input frames
        
    Returns:
        Initialized model
    """
    if model_type == 'base':
        return ScoNet(num_classes=num_classes, n_frames=n_frames)
    elif model_type == 'mt':
        return ScoNetMT(num_classes=num_classes, n_frames=n_frames)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":  # Only run the following code if this file is executed directly, not imported as a module (dummy test)
    # Test the model
    model = create_sconet('base', num_classes=3, n_frames=30)
    
    # Create dummy input (batch_size=2, n_frames=30, height=64, width=64)
    x = torch.randn(2, 30, 64, 64)
    
    # Forward pass
    output = model(x)
    print(f"Base model output shape: {output.shape}")
    
    # Test multi-task model
    model_mt = create_sconet('mt', num_classes=3, n_frames=30)
    class_out, angle_out = model_mt(x)
    print(f"MT model class output shape: {class_out.shape}")
    print(f"MT model angle output shape: {angle_out.shape}")
