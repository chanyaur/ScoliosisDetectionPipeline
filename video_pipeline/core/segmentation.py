"""
Human segmentation using MediaPipe and other methods
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)


class HumanSegmenter:  # segmenting = separate foreground (human) from bg
    """
    Segment human from background using various methods
    """
    
    def __init__(self, 
                 backend: str = "mediapipe",  # seems like      and bg subtraction are 2 ways of doing the same thing - segmenting
                 model_selection: int = 1,  # From MediaPipe selfie segmentation. 0 is for like selfies but 1 is for full body views
                 confidence_threshold: float = 0.7):  # the cutoff to turn the per-pixel probability map into binary segmentation mask. Higher --> may miss some foreground, lower --> may include some background
        """
        Initialize segmenter
        
        Args:
            backend: Segmentation backend ('mediapipe', 'background_subtraction')
            model_selection: Model selection for MediaPipe (0: general, 1: landscape)
            confidence_threshold: Confidence threshold for segmentation
        """
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        
        if backend == "mediapipe":
            # Initialize MediaPipe selfie segmentation
            self.mp_selfie = mp.solutions.selfie_segmentation  # the module from MediaPipe that allows the model to be created in the future
            self.segmentor = self.mp_selfie.SelfieSegmentation(
                model_selection=model_selection
            )
            logger.info(f"Initialized MediaPipe segmentation (model={model_selection})")
            
        elif backend == "background_subtraction":
            # Initialize background subtractor
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True  # why would we want to identify shadows?
            )
            logger.info("Initialized background subtraction")
            
        else:
            raise ValueError(f"Unknown segmentation backend: {backend}")
            
    def segment_person(self,  # this just calls the media pipe or bg sub
                      frame: np.ndarray,
                      bbox: Optional[List[int]] = None) -> np.ndarray:
        """
        Segment person from frame
        
        Args:
            frame: Input frame (H, W, C)
            bbox: Optional bounding box [x1, y1, x2, y2] to focus segmentation
            
        Returns:
            Binary mask (H, W) with person pixels as 1
        """
        if self.backend == "mediapipe":
            return self._segment_mediapipe(frame, bbox)
        elif self.backend == "background_subtraction":
            return self._segment_background_subtraction(frame, bbox)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
            
    def _segment_mediapipe(self,  # mediapipe subtract method of segmentation
                          frame: np.ndarray,  # np.ndarray is the type, representing a n-dimensional array
                          bbox: Optional[List[int]] = None) -> np.ndarray:  # can not be provided when detector returns no boxes
        """
        Segment using MediaPipe
        
        Args:
            frame: Input frame
            bbox: Optional bounding box
            
        Returns:
            Binary mask
        """
        # If bbox provided, crop frame
        if bbox:
            x1, y1, x2, y2 = bbox
            cropped_frame = frame[y1:y2, x1:x2]
            
            # Process cropped region
            results = self.segmentor.process(cropped_frame)  # this runs the pretrained model to segment the object
                                                             # input: RGB image, output: float32 array representing the mask
            
            if results.segmentation_mask is not None:
                # Convert to binary mask
                mask = results.segmentation_mask > self.confidence_threshold  # for each pixel, only take it if probability > confidence
                
                # Resize back to original frame size
                full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)  # set an array of all 0s first
                full_mask[y1:y2, x1:x2] = mask  # and then take the bounding box frame and then set these pixels to true or false, depending on the mask
                
                return full_mask.astype(np.uint8)
            else:
                return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)  # dtype - data type
                                                                                   # uint8 can store 8-bit integers [0-255]
        else:
            # Process full frame
            results = self.segmentor.process(frame)
            
            if results.segmentation_mask is not None:
                # Convert to binary mask
                mask = results.segmentation_mask > self.confidence_threshold
                return mask.astype(np.uint8)
            else:
                return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                
    def _segment_background_subtraction(self,  # bg subtract method of segmentation
                                       frame: np.ndarray,
                                       bbox: Optional[List[int]] = None) -> np.ndarray:
        """
        Segment using background subtraction
        
        Args:
            frame: Input frame
            bbox: Optional bounding box
            
        Returns:
            Binary mask
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (shadows have value 127 in MOG2)  # question - if we are removing shadows why did we set shadows = True in the first place?
        fg_mask[fg_mask == 127] = 0
        fg_mask[fg_mask > 0] = 1
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
# Creates a 5×5 elliptical structuring element (the neighborhood used for morphology). 
# Ellipse is usually better for natural shapes (people) than a square.

# OPEN - Removes small isolated foreground pixels (salt noise) and 
# disconnects tiny spurious blobswhile keeping larger shapes.

# CLOSE - Fills small holes/gaps inside foreground regions 
# (removes pepper noise inside blobs) and smooths boundaries.

# in summary: first, make an elliptical segmentor, then keep just the larger shapes, and finally clean up the segmentation (by filling gaps and smoothing boundary)
        
        # If bbox provided, zero out areas outside bbox
        if bbox:
            x1, y1, x2, y2 = bbox
            mask = np.zeros_like(fg_mask)
            mask[y1:y2, x1:x2] = fg_mask[y1:y2, x1:x2]
            return mask
            
        return fg_mask
    
    # segment returns a binary mask (0 and 1)
    # extract_silhouette returns a color image (with RGB tuple) - background black, foreground keeps RGB
    # create_binary_silhouette takes the mask and makes it to 0 and 255, which is suitable for visualization/saving binary img (?)
    
    def extract_silhouette(self,  # returns a color image (with RGB tuple) - background black, foreground keeps RGB
                          frame: np.ndarray,
                          mask: np.ndarray,
                          background_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:  # this is probably (R, G, B)
        """
        Extract silhouette from frame using mask
        
        Args:
            frame: Input frame
            mask: Binary mask
            background_color: Color for background pixels
            
        Returns:
            Frame with background removed
        """
        # Create output frame
        silhouette = frame.copy()
        
        # Apply mask
        silhouette[mask == 0] = background_color
        
        return silhouette
    
    def create_binary_silhouette(self,  # takes the ask and makes it to 0 and 255, which is suitable for visualization/saving binary img (?)
                                mask: np.ndarray,
                                foreground_value: int = 255,
                                background_value: int = 0) -> np.ndarray:
        """
        Create binary silhouette from mask
        
        Args:
            mask: Binary mask
            foreground_value: Value for foreground pixels
            background_value: Value for background pixels
            
        Returns:
            Binary silhouette image
        """
        silhouette = np.where(mask > 0, foreground_value, background_value)
        return silhouette.astype(np.uint8)  # question - come back to this and see where it is referenced
    
    def segment_batch(self,  # basically segments a bunch of frames into masks and stores in np array "masks"
                     frames: np.ndarray,
                     bboxes: Optional[List[List[int]]] = None) -> np.ndarray:
        """
        Segment batch of frames
        
        Args:
            frames: Batch of frames (N, H, W, C)
            bboxes: Optional list of bounding boxes
            
        Returns:
            Batch of masks (N, H, W)
        """
        masks = []
        
        for i, frame in enumerate(frames):
            bbox = bboxes[i] if bboxes else None
            mask = self.segment_person(frame, bbox)
            masks.append(mask)
            
        return np.array(masks)
    
    def refine_mask_with_bbox(self,  # pads to slightly expand the region kept (so it's not just white on the edge and it looks more like real input data)
                             mask: np.ndarray,
                             bbox: List[int],
                             padding: int = 10) -> np.ndarray:
        """
        Refine segmentation mask using bounding box
        
        Args:
            mask: Initial segmentation mask
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Padding around bbox
            
        Returns:
            Refined mask
        """
        refined_mask = np.zeros_like(mask)
        
        # Add padding to bbox
        x1 = max(0, bbox[0] - padding)
        y1 = max(0, bbox[1] - padding)
        x2 = min(mask.shape[1], bbox[2] + padding)
        y2 = min(mask.shape[0], bbox[3] + padding)
        
        # Copy only the mask within padded bbox
        refined_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        
        return refined_mask
    
    def get_mask_bbox(self, mask: np.ndarray) -> Optional[List[int]]:  # question why do we need this...
        """
        Get bounding box from mask
        
        Args:
            mask: Binary mask
            
        Returns:
            Bounding box [x1, y1, x2, y2] or None if mask is empty
        """
        # Find non-zero pixels
        points = np.argwhere(mask > 0)
        
        if len(points) == 0:
            return None
            
        # Get bounding box
        y1, x1 = points.min(axis=0)
        y2, x2 = points.max(axis=0)
        
        return [int(x1), int(y1), int(x2), int(y2)]
    
    def cleanup_mask(self,  # filters out small clusters not part of the mask, and also flood fills the mask in case any missing pixels in there
                    mask: np.ndarray,
                    min_area: int = 1000,
                    fill_holes: bool = True) -> np.ndarray:
        """
        Clean up segmentation mask using morphological operations
        
        Args:
            mask: Binary mask
            min_area: Minimum area for connected components
            fill_holes: Whether to fill holes in mask
            
        Returns:
            Cleaned mask
        """
        cleaned_mask = mask.copy()
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned_mask, connectivity=8
        )
        
        # Keep only components larger than min_area
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]  # the number of foreground pixels in that component
            if area < min_area:
                cleaned_mask[labels == i] = 0  # filters out noise
                
        # A component = a connected group of foreground pixels in the binary mask (pixels > 0).
        # the above clears that entire connected component when it is smaller than min_area. (noise)
                
        # Fill holes if requested
        if fill_holes:
            # Use flood fill from edges to find background
            h, w = cleaned_mask.shape
            temp_mask = cleaned_mask.copy()
            cv2.floodFill(temp_mask, None, (0, 0), 255)
            
            # Invert to get holes
            holes = cv2.bitwise_not(temp_mask)
            
            # Add holes back to mask
            cleaned_mask = cv2.bitwise_or(cleaned_mask, holes)
            
        return cleaned_mask
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'segmentor') and self.backend == "mediapipe":
            self.segmentor.close()
