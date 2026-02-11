"""
Person detection using YOLOv8
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from ultralytics import YOLO
import logging
import cv2

logger = logging.getLogger(__name__)


class PersonDetector:
    """
    Detect persons in video frames using YOLOv8
    """
    
    def __init__(self, 
                 model_path: str = "yolov8m.pt",
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 device: str = 'cpu'):
        """
        Initialize person detector
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-max suppression threshold
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        
        # Load YOLO model
        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        
        # Person class ID in COCO dataset
        self.person_class_id = 0  # COCO is a large scale image detection, segmentation, and captioning dataset
                                  # person is actually defined as category ID 1, but this is zero indexed so person is 0
        
    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect all persons in a single frame
        
        Args:
            frame: Input frame (H, W, C)  --> height, width, channels
            
        Returns:
            List of detection dictionaries with keys:
                - bbox: [x1, y1, x2, y2] coordinates
                - confidence: Detection confidence
                - area: Bounding box area
        """
        # Run detection
        results = self.model(frame, conf=self.confidence_threshold, 
                            iou=self.nms_threshold, verbose=False)  # this returns a Results object with all the bounding boxes for one frame
        
        # IOU (Intersection over Union): measures overlap between bounding boxes - to decide how many to have/which ones to keep.
        # During Non-Maximum Suppression (NMS):
        #   - If IoU < NMS threshold → boxes are considered separate → keep both.
        #   - If IoU ≥ NMS threshold → boxes overlap too much → keep the one with higher confidence.

        detections = []  # list of each detection, each entry stored as a dictionary with bbox, confidence, and area parameters
        
        for result in results:
            if result.boxes is None:  # normally this never happens, and normally len(result.boxes) == 0, in which case the code wouldn't crash. just for safety check
                continue
                
            boxes = result.boxes
            
            # result.plot
            
            for i in range(len(boxes)):
                # Check if detection is a person
                if int(boxes.cls[i]) == self.person_class_id:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu())
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'area': (x2 - x1) * (y2 - y1)
                    }
                    detections.append(detection)
                    
        # Sort by area (largest first)
        detections.sort(key=lambda x: x['area'], reverse=True)
        
        return detections
    
    def detect_primary_person(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect the primary person in frame (largest/most centered)
        
        Args:
            frame: Input frame
            
        Returns:
            Primary person detection or None if no person found
        """
        detections = self.detect_persons(frame)
        
        if not detections:
            return None
            
        # If only one person, return it
        if len(detections) == 1:
            return detections[0]
            
        # Score detections based on size and centrality
        h, w = frame.shape[:2]
        frame_center_x = w / 2
        frame_center_y = h / 2
        
        best_detection = None
        best_score = -1
        
        for detection in detections:
            bbox = detection['bbox']
            
            # Calculate center of bounding box
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            
            # Calculate distance from frame center (normalized)
            dist_from_center = np.sqrt(
                ((bbox_center_x - frame_center_x) / w) ** 2 +
                ((bbox_center_y - frame_center_y) / h) ** 2
            )
            
            # Calculate normalized area
            norm_area = detection['area'] / (w * h)  # how much of the frame is taken up by the person? (prioritizes bigger people)
            
            # Combined score: prioritize size and centrality
            score = norm_area * (1 - dist_from_center) * detection['confidence']
            
            if score > best_score:
                best_score = score  # goes thru all people and updates their score
                best_detection = detection  # and if it's greater then this person is listed as the best person
                
        return best_detection
    
    def detect_batch(self, frames: np.ndarray) -> List[List[Dict]]:  # basically runs the two functions created above for the whole video, then stores in a list "batch_detections"
        """
        Detect persons in batch of frames
        
        Args:
            frames: Batch of frames (N, H, W, C)
            
        Returns:
            List of detection lists for each frame
        """
        batch_detections = []
        
        for frame in frames:
            detections = self.detect_persons(frame)
            batch_detections.append(detections)
            
        return batch_detections
    
    def track_primary_person(self, frames: np.ndarray) -> List[Optional[Dict]]:
        """
        Simple tracking of primary person across frames
        
        Args:
            frames: Sequence of frames
            
        Returns:
            List of primary person detections for each frame
        """
        primary_detections = []
        last_valid_bbox = None
        
        for i, frame in enumerate(frames):
            detection = self.detect_primary_person(frame)
            
            # If no detection, try to use last valid bbox area
            if detection is None and last_valid_bbox is not None:  # i think this is redundant bec it can't happen unless detect_persons is also none?
                # Look for person near last position
                all_detections = self.detect_persons(frame)
                
                if all_detections:
                    # Find detection closest to last position
                    last_center = [
                        (last_valid_bbox[0] + last_valid_bbox[2]) / 2,
                        (last_valid_bbox[1] + last_valid_bbox[3]) / 2
                    ]
                    
                    min_dist = float('inf')
                    closest_detection = None
                    
                    for det in all_detections:
                        bbox = det['bbox']
                        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                        
                        dist = np.sqrt((center[0] - last_center[0])**2 + 
                                      (center[1] - last_center[1])**2)
                        
                        if dist < min_dist:
                            min_dist = dist
                            closest_detection = det
                            
                    # Use closest detection if it's reasonably close
                    if min_dist < 100:  # Threshold in pixels
                        detection = closest_detection
                        
            if detection:
                last_valid_bbox = detection['bbox']
                
            primary_detections.append(detection)
            
        return primary_detections  # i think that if it detects no one that it'll return an empty list?
    
    def visualize_detections(self, frame: np.ndarray,   # just draw the bounding boxes
                            detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with drawn bounding boxes
        """
        vis_frame = frame.copy()
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for primary
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), color, 2)
            
            # Draw confidence
            label = f"Person: {confidence:.2f}"
            cv2.putText(vis_frame, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
        return vis_frame
    
    def filter_detections_by_size(self, detections: List[Dict], 
                                 min_area_ratio: float = 0.1) -> List[Dict]:
        """
        Filter detections by minimum size
        
        Args:
            detections: List of detections
            min_area_ratio: Minimum ratio of bbox area to frame area
            
        Returns:
            Filtered detections
        """
        if not detections:
            return []
            
        # Assume all detections are from same frame size
        # Get approximate frame area from largest detection
        frame_area_estimate = detections[0]['area'] * 10  # Rough estimate
        
        # question review this bec i dont get it
        filtered = []
        for detection in detections:
            if detection['area'] >= min_area_ratio * frame_area_estimate:
                filtered.append(detection)
                
        return filtered
