"""
Data Loader with Balanced Sampling for Scoliosis1K Dataset
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Optional, Dict
import numpy as np
from .scoliosis_dataset import Scoliosis1KDataset


def create_weighted_sampler(dataset: Scoliosis1KDataset) -> WeightedRandomSampler:
    """Create a weighted random sampler for balanced training"""
    
    # Get labels for all samples
    labels = [item['label'] for item in dataset.data]
    
    # Calculate class counts
    class_counts = np.bincount(labels)
    
    # Calculate weight for each class (inverse frequency)
    class_weights = 1.0 / class_counts
    
    # Assign weight to each sample
    weights = [class_weights[label] for label in labels]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler


def get_data_loaders(
    data_root: str = 'Scoliosis1K-pkl',
    batch_size: int = 32,
    num_workers: int = 4,
    temporal_sampling: str = 'uniform',
    n_frames: int = 30,
    balanced_sampling: bool = True,
    cache_data: bool = False
) -> Dict[str, DataLoader]:
    """
    Create data loaders for train, validation, and test sets
    
    Args:
        data_root: Path to the Scoliosis1K-pkl directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        temporal_sampling: Strategy for sampling frames
        n_frames: Number of frames to sample
        balanced_sampling: Whether to use balanced sampling for training
        cache_data: Whether to cache data in memory
    
    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        # Create dataset
        dataset = Scoliosis1KDataset(
            data_root=data_root,
            split=split,
            temporal_sampling=temporal_sampling,
            n_frames=n_frames,
            normalize=True,
            cache_data=cache_data,
            binary=True
        )
        
        # Create sampler for training set if balanced sampling is enabled
        sampler = None
        shuffle = True
        
        if split == 'train' and balanced_sampling:
            sampler = create_weighted_sampler(dataset)
            shuffle = False  # Don't shuffle when using sampler
        elif split != 'train':
            shuffle = False  # Don't shuffle validation/test sets
        
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == 'train')  # Drop last incomplete batch for training
        )
        
        loaders[split] = loader
    
    return loaders


def collate_fn_padding(batch):
    """
    Custom collate function to handle variable length sequences
    Pads sequences to the maximum length in the batch
    """
    sequences = []
    labels = []
    infos = []
    
    for seq, label, info in batch:
        sequences.append(seq)
        labels.append(label)
        infos.append(info)
    
    # Find max temporal length in batch
    max_t = max(seq.shape[1] for seq in sequences)
    
    # Pad sequences
    padded_sequences = []
    for seq in sequences:
        if seq.shape[1] < max_t:
            # Pad with zeros
            padding = torch.zeros(
                seq.shape[0], 
                max_t - seq.shape[1],
                seq.shape[2],
                seq.shape[3]
            )
            seq = torch.cat([seq, padding], dim=1)
        padded_sequences.append(seq)
    
    # Stack into batch
    batch_sequences = torch.stack(padded_sequences, dim=0)
    batch_labels = torch.tensor(labels, dtype=torch.long)
    
    return batch_sequences, batch_labels, infos


if __name__ == "__main__":
    print("Testing Data Loaders...")
    
    # Create data loaders
    loaders = get_data_loaders(
        batch_size=8,
        num_workers=2,
        temporal_sampling='uniform',
        n_frames=30,
        balanced_sampling=True
    )
    
    # Test each loader
    for split_name, loader in loaders.items():
        print(f"\n{split_name.upper()} Loader:")
        print(f"  Number of batches: {len(loader)}")
        
        # Get a sample batch
        for batch_idx, (sequences, labels, infos) in enumerate(loader):
            print(f"  Batch shape: {sequences.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Batch labels: {labels[:8]}")  # Show first 8 labels
            
            # Show class distribution in this batch
            unique, counts = torch.unique(labels, return_counts=True)
            print(f"  Class distribution in batch: {dict(zip(unique.tolist(), counts.tolist()))}")
            break  # Only show first batch
