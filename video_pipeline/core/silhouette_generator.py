"""
Generate silhouettes from segmented frames for model input
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SilhouetteGenerator:
    """
    Convert segmented frames to silhouette sequences for model input
    """
    
    def __init__(self,
                 output_size: Tuple[int, int] = (64, 64),
                 sequence_length: int = 300,
                 binary_threshold: int = 128,
                 normalize: bool = True):
        """
        Initialize silhouette generator
        
        Args:
            output_size: Target silhouette size (width, height)
            sequence_length: Target sequence length
            binary_threshold: Threshold for binary silhouette
            normalize: Whether to normalize pixel values
        """
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.binary_threshold = binary_threshold
        self.normalize = normalize
        
    def generate_silhouette(self,
                           frame: np.ndarray,
                           mask: np.ndarray,
                           crop_to_person: bool = True) -> np.ndarray:
        """
        Generate single silhouette from frame and mask
        
        Args:
            frame: Original frame (H, W, C)
            mask: Binary mask (H, W)
            crop_to_person: Whether to crop to person bounding box
            
        Returns:
            Silhouette image of shape output_size
        """
        # Create binary silhouette
        if len(frame.shape) == 3:
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray_frame = frame
            
        # Apply mask to create silhouette
        silhouette = np.where(mask > 0, 255, 0).astype(np.uint8)
        
        if crop_to_person:
            # Find bounding box of person
            bbox = self._get_bbox_from_mask(mask)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                silhouette = silhouette[y1:y2, x1:x2]
            
        # Resize to target size
        silhouette = self._resize_with_padding(silhouette, self.output_size)
        
        # Apply binary threshold if needed - question - why would we need this? Due to line 60 isn't it already only comprised of 0 and 255?
        if self.binary_threshold > 0:
            _, silhouette = cv2.threshold(
                silhouette, self.binary_threshold, 255, cv2.THRESH_BINARY
            )
            
        # Normalize if requested
        if self.normalize:  # converts to float inputs of 0 and 1
            silhouette = silhouette.astype(np.float32) / 255.0
            
        return silhouette
    
    def generate_sequence(self,  # sample the frames, get the silhouette of each of them, and return that list
                         frames: np.ndarray,
                         masks: np.ndarray,
                         sampling_strategy: str = "uniform") -> np.ndarray:
        """
        Generate silhouette sequence from frames and masks
        
        Args:
            frames: Array of frames (N, H, W, C)
            masks: Array of masks (N, H, W)
            sampling_strategy: How to sample frames ('uniform', 'random', 'adaptive')
            
        Returns:
            Silhouette sequence of shape (sequence_length, output_size[1], output_size[0])
        """
        n_frames = len(frames)
        
        # Sample frame indices
        indices = self._sample_frames(n_frames, self.sequence_length, sampling_strategy)
        
        # Generate silhouettes for sampled frames
        silhouettes = []
        for idx in indices:
            silhouette = self.generate_silhouette(
                frames[idx], masks[idx], crop_to_person=True
            )
            silhouettes.append(silhouette)
            
        # Stack into sequence
        sequence = np.stack(silhouettes, axis=0)  # combines the list of arrays into one array
        
        logger.info(f"Generated silhouette sequence: {sequence.shape}")
        
        return sequence
    
    def _sample_frames(self,  # this is sample within a temporal sequence
                      n_frames: int,
                      target_length: int,
                      strategy: str) -> List[int]:
        """
        Sample frame indices based on strategy
        
        Args:
            n_frames: Total number of available frames
            target_length: Target sequence length
            strategy: Sampling strategy
            
        Returns:
            List of frame indices
        """
        if strategy == "uniform":
            # Uniform sampling
            if n_frames >= target_length:
                indices = np.linspace(0, n_frames - 1, target_length, dtype=int)
            else:
                # Repeat frames if not enough
                indices = np.arange(n_frames)
                indices = np.resize(indices, target_length)
                
        elif strategy == "random":
            # Random sampling with replacement
            indices = np.random.choice(n_frames, target_length, replace=True)
            indices = np.sort(indices)  # Keep temporal order
            
        elif strategy == "adaptive":
            # Adaptive sampling based on motion
            # For now, fallback to uniform
            indices = self._sample_frames(n_frames, target_length, "uniform")
            
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
            
        return indices.tolist()
    
    def _resize_with_padding(self,  # returns the resized image with either the same height or the same width as the target. May not be perfectly square tho
                            image: np.ndarray,
                            target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size with padding to maintain aspect ratio
        
        Args:
            image: Input image
            target_size: Target (width, height)
            
        Returns:
            Resized and padded image
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale to fit within target size
        scale = min(target_w / w, target_h / h)  # so this should be the scale factor. It's the min of the two bec you need to scale it by the larger one
        
        # Calculate new size
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded = np.zeros((target_h, target_w), dtype=image.dtype)
        
        # Calculate padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        return padded
    
    def _get_bbox_from_mask(self, mask: np.ndarray) -> Optional[List[int]]:  # get bounding box from mask
        """
        Get bounding box from mask
        
        Args:
            mask: Binary mask
            
        Returns:
            Bounding box [x1, y1, x2, y2] or None
        """
        # Find non-zero pixels
        points = np.argwhere(mask > 0)
        
        if len(points) == 0:
            return None
            
        # Get bounding box
        y1, x1 = points.min(axis=0)
        y2, x2 = points.max(axis=0)
        
        # Add small padding
        padding = 5
        h, w = mask.shape
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        return [x1, y1, x2, y2]
    
    def process_video_to_silhouettes(self,  # process entire video to silhouette sequence (binary)?
                                    frames: np.ndarray,
                                    masks: np.ndarray,
                                    bbox_list: Optional[List[List[int]]] = None) -> np.ndarray:
        """
        Process entire video to silhouette sequence
        
        Args:
            frames: Video frames
            masks: Segmentation masks
            bbox_list: Optional list of bounding boxes
            
        Returns:
            Silhouette sequence ready for model input
        """
        # Generate full silhouette sequence
        sequence = self.generate_sequence(frames, masks, sampling_strategy="uniform")
        
        # Ensure correct shape for model input - question - i thought that it only resized it based on one dimension? where did we pad the rest of it?
        # Expected: (sequence_length, height, width)
        if sequence.shape != (self.sequence_length, self.output_size[1], self.output_size[0]):
            logger.warning(f"Unexpected sequence shape: {sequence.shape}")
            
        return sequence
    
    def visualize_silhouettes(self,  # takes one sequence, n_display is the # of images to display in that sequence, and linear spaces them
                             silhouettes: np.ndarray,
                             n_display: int = 10) -> np.ndarray:
        """
        Create visualization of silhouette sequence
        
        Args:
            silhouettes: Silhouette sequence (N, H, W)
            n_display: Number of silhouettes to display
            
        Returns:
            Visualization image
        """
        n_frames = len(silhouettes)
        display_indices = np.linspace(0, n_frames - 1, n_display, dtype=int)
        
        # Create grid of silhouettes
        display_silhouettes = []
        for idx in display_indices:
            sil = silhouettes[idx]
            
            # Convert to uint8 if normalized
            if sil.max() <= 1.0:
                sil = (sil * 255).astype(np.uint8)  # convert back to black and white (0 and 255)
            else:
                sil = sil.astype(np.uint8)
                
            display_silhouettes.append(sil)
            
        # Concatenate horizontally
        visualization = np.hstack(display_silhouettes)
        
        return visualization
    
    def save_silhouette_sequence(self,  # so that it can directly be loaded into the input pipeline later
                                silhouettes: np.ndarray,
                                output_path: str) -> None:
        """
        Save silhouette sequence to file
        
        Args:
            silhouettes: Silhouette sequence
            output_path: Path to save file
        """
        # Convert to uint8 if needed
        if silhouettes.dtype != np.uint8:
            if silhouettes.max() <= 1.0:
                silhouettes = (silhouettes * 255).astype(np.uint8)
            else:
                silhouettes = silhouettes.astype(np.uint8)
                
        # Save as numpy file
        np.save(output_path, silhouettes)
        logger.info(f"Saved silhouette sequence to {output_path}")
        
    def validate_sequence(self, sequence: np.ndarray) -> bool:  # check if the sequence is okay or if the user should do it again (or if pipeline error bec of shape ?? )
        """
        Validate silhouette sequence for model input
        
        Args:
            sequence: Silhouette sequence
            
        Returns:
            True if valid, False otherwise
        """
        # Check shape
        expected_shape = (self.sequence_length, self.output_size[1], self.output_size[0])
        if sequence.shape != expected_shape:
            logger.error(f"Invalid shape: {sequence.shape}, expected {expected_shape}")
            return False
            
        # Check for empty frames
        empty_frames = np.sum(np.sum(sequence, axis=(1, 2)) == 0)
        if empty_frames > self.sequence_length * 0.5:
            logger.error(f"Too many empty frames: {empty_frames}/{self.sequence_length}")
            return False
            
        return True
