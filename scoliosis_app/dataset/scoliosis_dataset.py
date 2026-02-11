"""
Scoliosis1K Dataset Class
Handles loading and preprocessing of silhouette sequences for scoliosis detection.
Supports 3-class (positive/neutral/negative) and binary (positive/negative) modes.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
from typing import Tuple, Optional, Dict, List


class Scoliosis1KDataset(Dataset):
    """PyTorch Dataset for Scoliosis1K dataset"""

    def __init__(
        self,
        data_root: str = 'Scoliosis1K-pkl',
        split: str = 'train',
        transform=None,
        temporal_sampling: str = 'full',  # 'full', 'uniform', 'random'
        n_frames: int = 30,               # Number of frames to sample if not using full
        normalize: bool = True,
        cache_data: bool = False,         # Whether to cache loaded sequences in memory
        binary: bool = False,             # NEW: train only Positive/Negative if True
        positive_first: bool = True       # NEW: map Positive->0, Negative->1 (if False, reversed)
    ):
        """
        Args:
            data_root: Path to the Scoliosis1K-pkl directory
            split: 'train', 'val', or 'test'
            transform: Optional transforms to apply
            temporal_sampling: Strategy for sampling frames from sequences
            n_frames: Number of frames to sample (if not using full sequence)
            normalize: Whether to normalize pixel values to [0, 1]
            cache_data: Whether to cache sequences in memory (faster but uses more RAM)
            binary: If True, drop 'neutral' and remap to 2 classes
            positive_first: If True, Positive->0, Negative->1 (else the opposite)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.temporal_sampling = temporal_sampling
        self.n_frames = n_frames
        self.normalize = normalize
        self.cache_data = cache_data
        self.binary = binary
        self.positive_first = positive_first

        # Original 3-class mapping (used while indexing)
        self.class_to_idx = {'positive': 0, 'neutral': 1, 'negative': 2}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Build a flat index of all available samples
        self.data_index = self._create_data_index()

        # If binary mode, filter out 'neutral' and remap labels to two classes
        if self.binary:
            if self.positive_first:
                bin_map = {'positive': 0, 'negative': 1}
                ordered_names = ['positive', 'negative']
            else:
                bin_map = {'positive': 1, 'negative': 0}
                ordered_names = ['negative', 'positive']

            filtered = []
            for it in self.data_index:
                if it['class'] in bin_map:  # keep only positive/negative
                    it2 = dict(it)
                    it2['label'] = bin_map[it['class']]
                    filtered.append(it2)
            self.data_index = filtered

            # Replace the class maps with 2-class maps
            self.class_to_idx = {ordered_names[0]: 0, ordered_names[1]: 1}
            self.idx_to_class = {0: ordered_names[0], 1: ordered_names[1]}

        # Number of classes now depends on mode
        self.num_classes = len(self.class_to_idx)

        # Create stratified splits
        self._create_splits()

        # Optional in-memory cache
        self.cache = {} if cache_data else None

        # Class weights & counts (computed on the selected split)
        self.class_weights = self._calculate_class_weights()
        self.class_counts = {i: 0 for i in range(self.num_classes)}
        for item in self.data:
            self.class_counts[item['label']] += 1

    # ----------------------
    # Indexing & Splitting
    # ----------------------
    def _create_data_index(self) -> List[Dict]:
        """Create index of all available data samples by scanning the directory tree."""
        data_index = []

        if not self.data_root.exists():
            print(f"Warning: Data root {self} does not exist")
            return data_index

        for subject_dir in sorted(self.data_root.iterdir()):
            if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
                continue

            # Parse subject ID from directory name if possible
            try:
                subject_id = int(subject_dir.name)
            except ValueError:
                continue

            # Check for class subdirectories (we stop at the first that exists)
            for class_name in ['positive', 'negative', 'neutral']:
                pkl_path = subject_dir / class_name / "000_180" / "000_180.pkl"
                if pkl_path.exists():
                    data_index.append({
                        'path': pkl_path,
                        'subject_id': subject_id,
                        'class': class_name,
                        'label': self.class_to_idx[class_name]
                    })
                    break  # assume each subject has only one class folder

        print(f"Found {len(data_index)} samples in {self.data_root}")
        return data_index

    def _create_splits(self):
        """Create train/val/test splits (70/15/15) with stratification."""
        # Group by label for stratification
        class_groups: Dict[int, List[Dict]] = {}
        for it in self.data_index:
            class_groups.setdefault(it['label'], []).append(it)

        train_data, val_data, test_data = [], [], []
        rng = random.Random(42)  # reproducible shuffling

        for label, items in class_groups.items():
            rng.shuffle(items)
            n_total = len(items)
            n_train = int(0.70 * n_total)
            n_val = int(0.15 * n_total)
            train_data.extend(items[:n_train])
            val_data.extend(items[n_train:n_train + n_val])
            test_data.extend(items[n_train + n_val:])

        if self.split == 'train':
            self.data = train_data
        elif self.split == 'val':
            self.data = val_data
        elif self.split == 'test':
            self.data = test_data
        else:
            raise ValueError(f"Invalid split: {self.split}")

        rng.shuffle(self.data)

    # ----------------------
    # Weights & Stats
    # ----------------------
    def _calculate_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for the current split."""
        counts = {i: 0 for i in range(self.num_classes)}
        for it in self.data:
            counts[it['label']] += 1

        total = sum(counts.values())
        weights = []
        for label in range(self.num_classes):
            if counts[label] > 0:
                weights.append(total / (self.num_classes * counts[label]))
            else:
                weights.append(1.0)
        return torch.FloatTensor(weights)

    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in this split."""
        dist = {i: 0 for i in range(self.num_classes)}
        for it in self.data:
            dist[it['label']] += 1
        return {self.idx_to_class[k]: v for k, v in dist.items()}

    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for balanced training."""
        return self.class_weights

    # ----------------------
    # I/O and Sampling
    # ----------------------
    def _load_sequence(self, path: Path) -> np.ndarray:
        """Load a silhouette sequence from pickle file."""
        if self.cache_data and str(path) in self.cache:
            return self.cache[str(path)]

        with open(path, 'rb') as f:
            sequence = pickle.load(f)

        sequence = np.array(sequence, dtype=np.float32)

        if self.cache_data:
            self.cache[str(path)] = sequence

        return sequence

    def _sample_frames(self, sequence: np.ndarray) -> np.ndarray:
        """Sample frames from sequence based on temporal_sampling strategy."""
        T, H, W = sequence.shape

        if self.temporal_sampling == 'full':
            return sequence

        elif self.temporal_sampling == 'uniform':
            idx = np.linspace(0, T - 1, self.n_frames, dtype=int)
            return sequence[idx]

        elif self.temporal_sampling == 'random':
            if T <= self.n_frames:
                idx = np.arange(T)
                extra = np.random.choice(T, self.n_frames - T, replace=True)
                idx = np.concatenate([idx, extra])
            else:
                idx = np.sort(np.random.choice(T, self.n_frames, replace=False))
            return sequence[idx]

        else:
            raise ValueError(f"Invalid temporal_sampling: {self.temporal_sampling}")

    def _normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Normalize pixel values to [0, 1]."""
        return sequence / 255.0 if self.normalize else sequence

    # ----------------------
    # Dataset protocol
    # ----------------------
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            sequence: Tensor of shape (T, H, W) for model input
            label: Integer class label (0..num_classes-1)
        """
        item = self.data[idx]

        # Load sequence
        sequence = self._load_sequence(item['path'])

        # Temporal sampling
        sequence = self._sample_frames(sequence)

        # Normalize
        sequence = self._normalize_sequence(sequence)

        # Optional transforms
        if self.transform:
            sequence = self.transform(sequence)

        # To tensor (T, H, W); model will add channel dimension as needed
        sequence = torch.from_numpy(sequence).float()

        return sequence, item['label']


if __name__ == "__main__":
    # Quick smoke test
    print("Testing Scoliosis1K Dataset...")

    for split in ['train', 'val', 'test']:
        # Try binary=True (Positive/Negative only)
        ds = Scoliosis1KDataset(
            data_root='Scoliosis1K-pkl',
            split=split,
            temporal_sampling='uniform',
            n_frames=30,
            binary=True,           # set False to use 3-class
            positive_first=True    # Positive->0, Negative->1
        )

        print(f"\n{split.upper()} Split:")
        print(f"  Total samples: {len(ds)}")
        print(f"  Class distribution: {ds.get_class_distribution()}")
        print(f"  Class weights: {ds.get_class_weights()}")
        if len(ds) > 0:
            x, y = ds[0]
            print(f"  Sample shape: {x.shape}")
            # map index to class name
            print(f"  Label: {y} ({ds.idx_to_class[y]})")
