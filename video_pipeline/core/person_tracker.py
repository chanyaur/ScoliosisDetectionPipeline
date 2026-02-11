"""
Person tracking across video frames
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PersonTracker:
    """
    Simple person tracker to maintain consistency across frames
    """
    
    def __init__(self,
                 max_distance: float = 100,  # this is a fixed integer pixel value to represent the max distance for a match to be considered.
                 min_confidence: float = 0.3,
                 track_buffer: int = 30):
        """
        Initialize tracker
        
        Args:
            max_distance: Maximum distance between a track's center and a detection's center for track association (in pixels)
            min_confidence: Minimum confidence for valid track
            track_buffer: Number of frames to keep lost tracks
        """
        self.max_distance = max_distance
        self.min_confidence = min_confidence
        self.track_buffer = track_buffer
        
        self.tracks = []
        self.track_id_counter = 0  # just for indexing track ID i think. So simple integer counter ensuring a unique ID for every track upon initialization
        self.frame_count = 0  # very simple integer counter just representing the frame index. Used to timestamp stuff like first_seen or last_seen
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of person detections
            
        Returns:
            Updated detections with track IDs
        """
        self.frame_count += 1
        
        if not detections:  # if you don't detect anything
            # Update the count of the tracks lost so that if they're lost for only a few frames, they are not lost completely
            self._update_lost_tracks()
            return []
            
        # Calculate cost matrix between tracks and detections
        cost_matrix = self._calculate_cost_matrix(detections)
        
        # Associate detections with tracks
        matched_pairs, unmatched_detections, unmatched_tracks = self._associate(
            cost_matrix, detections
        )
        
        # Update matched tracks
        tracked_detections = []
        for track_idx, det_idx in matched_pairs:
            track = self.tracks[track_idx]
            detection = detections[det_idx]
            
            # Update track
            track['bbox'] = detection['bbox']  # as you go through frames, updates the bbox of the track to be equal to the bbox of the last frame you saw
            track['confidence'] = detection['confidence']
            track['last_seen'] = self.frame_count
            track['age'] += 1
            
            # Add track ID to detection
            detection['track_id'] = track['track_id']
            tracked_detections.append(detection)
            
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            
            # Create new track
            new_track = {
                'track_id': self.track_id_counter,
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'first_seen': self.frame_count,
                'last_seen': self.frame_count,
                'age': 1
            }
            self.tracks.append(new_track)
            
            # Add track ID to detection
            detection['track_id'] = self.track_id_counter
            tracked_detections.append(detection)
            
            self.track_id_counter += 1
            
        # Mark unmatched tracks as lost
        for track_idx in unmatched_tracks:
            self.tracks[track_idx]['last_seen'] = self.frame_count - 1
            
        # Remove old lost tracks
        self._remove_old_tracks()
        
        return tracked_detections
    
    def get_primary_track(self) -> Optional[Dict]:  # referenced in another file
        """
        Get the primary (most stable) track
        
        Returns:
            Primary track or None
        """
        if not self.tracks:
            return None
            
        # Find track with highest score (age * confidence)
        best_track = None
        best_score = -1
        
        for track in self.tracks:
            if self.frame_count - track['last_seen'] > 5:
                continue  # Skip lost tracks
                
            score = track['age'] * track['confidence']
            if score > best_score:
                best_score = score
                best_track = track
                
        return best_track
    
    def _calculate_cost_matrix(self, detections: List[Dict]) -> np.ndarray:
        """
        Calculate cost matrix between tracks and detections
        
        Args:
            detections: Current detections
            
        Returns:
            Cost matrix
        """
        n_tracks = len(self.tracks)
        n_detections = len(detections)
        
        cost_matrix = np.full((n_tracks, n_detections), self.max_distance)  # Return a new array of shape (n_tracks, n_detections), filled with self.max_distance
        
        for i, track in enumerate(self.tracks):  # i is the index, track is the track object
            # Skip very old tracks
            if self.frame_count - track['last_seen'] > self.track_buffer:
                continue
                
            track_center = self._get_bbox_center(track['bbox'])
            
            for j, detection in enumerate(detections):
                det_center = self._get_bbox_center(detection['bbox'])
                
                # Calculate Euclidean distance
                distance = np.linalg.norm(
                    np.array(track_center) - np.array(det_center)
                )
                
                cost_matrix[i, j] = distance
                
        return cost_matrix  # smaller distance --> more likely it is the same person, so lower cost
    
    def _associate(self,  # question come back to understand this
                   cost_matrix: np.ndarray,
                   detections: List[Dict]) -> Tuple[List, List, List]:
        """
        Associate detections with tracks using simple nearest neighbor
        
        Args:
            cost_matrix: Cost matrix
            detections: Current detections
            
        Returns:
            matched_pairs: List of (track_idx, det_idx) pairs
            unmatched_detections: List of detection indices
            unmatched_tracks: List of track indices
        """
        n_tracks = cost_matrix.shape[0]
        n_detections = cost_matrix.shape[1]  # this seems right bec it's how cost_matrix was defined
        
        matched_pairs = []
        used_tracks = set()
        used_detections = set()
        
        # Simple greedy matching
        while True:
            # Find minimum cost
            min_cost = self.max_distance  # initialize it with the max value
            min_track = -1
            min_det = -1
            
            # So the inner for-loop finds one best match per iteration;
            # the outer while repeats until all viable matches are found.
            
            for i in range(n_tracks):  # nested for loop running thru all tracks and conditions
                if i in used_tracks:
                    continue
                    
                for j in range(n_detections):
                    if j in used_detections:
                        continue
                        
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        min_track = i
                        min_det = j  # effectively matches track with detection
                        
            # If no valid match found, break
            if min_track == -1 or min_cost >= self.max_distance:
                break
                
            # Add match
            matched_pairs.append((min_track, min_det))
            used_tracks.add(min_track)
            used_detections.add(min_det)
            
        # Find unmatched
        unmatched_detections = [i for i in range(n_detections) if i not in used_detections]
        unmatched_tracks = [i for i in range(n_tracks) if i not in used_tracks]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _get_bbox_center(self, bbox: List[int]) -> Tuple[float, float]:
        """
        Get center of bounding box
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Center (x, y)
        """
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        return (x, y)
    
    def _update_lost_tracks(self) -> None:  # functions to increment the age of something
        # in the case it's lost for a split second in one frame and then comes back in the subsequent ones, the age will still be accurate
        """
        Update tracks when no detections
        """
        for track in self.tracks:
            track['age'] += 1  # why would this be incremented if no detection?
            
    def _remove_old_tracks(self) -> None:
        """
        Remove tracks that have been lost for too long
        """
        self.tracks = [
            track for track in self.tracks
            if self.frame_count - track['last_seen'] <= self.track_buffer
        ]
        
    def reset(self) -> None:
        """
        Reset tracker state
        """
        self.tracks = []
        self.track_id_counter = 0
        self.frame_count = 0
        
    def get_track_history(self, track_id: int) -> Optional[Dict]:
        """
        Get history of a specific track
        
        Args:
            track_id: Track ID
            
        Returns:
            Track information or None
        """
        for track in self.tracks:
            if track['track_id'] == track_id:
                return track
        return None
