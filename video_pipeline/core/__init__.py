"""
Core components for video processing pipeline
"""

from .video_loader import VideoLoader
from .person_detector import PersonDetector
from .person_tracker import PersonTracker
from .segmentation import HumanSegmenter
from .silhouette_generator import SilhouetteGenerator

__all__ = [
    'VideoLoader',
    'PersonDetector', 
    'PersonTracker',
    'HumanSegmenter',
    'SilhouetteGenerator'
]
