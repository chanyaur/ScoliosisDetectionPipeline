"""
Main video processing pipeline for converting phone videos to silhouettes
"""


import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging
from tqdm import tqdm
import warnings

from .core.video_loader import VideoLoader  # added dots to these: why?
from .core.person_detector import PersonDetector  
from .core.person_tracker import PersonTracker
from .core.segmentation import HumanSegmenter
from .core.silhouette_generator import SilhouetteGenerator
# for regular, no dot, for website, dot
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


class VideoToSilhouettePipeline:
    """
    Complete pipeline for processing phone videos to silhouette sequences
    """
    
    def __init__(self, config_path: str = "video_pipeline\config.yaml"):  # reads yaml file for config - this won't work?
        """
        Initialize pipeline with configuration
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        
        print("CWD:", Path.cwd())
        print("__file__ dir:", Path(__file__).parent)


        BASE_DIR = Path(__file__).resolve().parent

        CONFIG_PATH = BASE_DIR / "config.yaml"

        with open(CONFIG_PATH, "r") as f:
            self.config = yaml.safe_load(f)
    
        # with open(config_path, 'r') as f:
        #     self.config = yaml.safe_load(f)
            
        logger.info("Initializing Video Processing Pipeline")
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):  # initialize components (video loader, person detector, person tracker, human segmentor, silhouette generator)
        """Initialize all pipeline components"""
        
        # Video loader
        self.video_loader = VideoLoader(
            target_fps=self.config['video']['target_fps']
        )
        
        # Person detector
        try:
            self.person_detector = PersonDetector(
                model_path=self.config['detection']['model'],
                confidence_threshold=self.config['detection']['confidence_threshold'],
                nms_threshold=self.config['detection']['nms_threshold'],
                device='cuda' if self.config['model']['device'] == 'cuda' else 'cpu'
            )
            logger.info("Person detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize person detector: {e}")
            logger.info("Using fallback detection method")
            self.person_detector = None
            
        # Person tracker
        self.person_tracker = PersonTracker(
            max_distance=self.config['tracking']['max_distance'],
            min_confidence=self.config['tracking']['min_confidence'],
            track_buffer=self.config['tracking']['track_buffer']
        )
        
        # Human segmenter
        self.segmenter = HumanSegmenter(
            backend=self.config['segmentation']['backend'],
            model_selection=self.config['segmentation']['model_selection'],
            confidence_threshold=self.config['segmentation']['confidence_threshold']
        )
        
        # Silhouette generator
        self.silhouette_generator = SilhouetteGenerator(
            output_size=tuple(self.config['silhouette']['output_size']),
            sequence_length=self.config['silhouette']['sequence_length'],
            binary_threshold=self.config['silhouette']['binary_threshold'],
            normalize=self.config['silhouette']['normalize']
        )
        
    def process_video(self, 
                     video_path: str,
                     output_dir: Optional[str] = None) -> Dict:
        """
        Process video file to silhouette sequence
        
        Args:
            video_path: Path to input video
            output_dir: Optional directory to save outputs
            
        Returns:
            Dictionary with results including silhouettes and metadata
        """
        logger.info(f"Processing video: {video_path}")
        
        # Check if video exists
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        # Load video
        logger.info("Loading video frames...")
        frames, metadata = self.video_loader.load_video(video_path)
        logger.info(f"Loaded {len(frames)} frames")
        
        # Validate video
        if not self._validate_video(frames, metadata):
            raise ValueError("Video does not meet quality requirements")
            
        # Detect and track person
        logger.info("Detecting and tracking person...")
        person_bboxes = self._detect_and_track_person(frames)
        
        if not person_bboxes or all(b is None for b in person_bboxes):
            raise ValueError("No person detected in video")
            
        # Segment person
        logger.info("Segmenting person from background...")
        masks = self._segment_person(frames, person_bboxes)
        
        # Generate silhouettes
        logger.info("Generating silhouette sequence...")
        silhouettes = self.silhouette_generator.process_video_to_silhouettes(
            frames, masks
        )
        
        # Validate silhouettes
        if not self.silhouette_generator.validate_sequence(silhouettes):
            logger.warning("Silhouette sequence validation failed")
            
        # Prepare results
        results = {
            'silhouettes': silhouettes,
            'video_metadata': metadata,
            'num_frames_processed': len(frames),
            'num_valid_detections': sum(1 for b in person_bboxes if b is not None),
            'sequence_shape': silhouettes.shape
        }
        
        # Save outputs if requested
        if output_dir:
            self._save_outputs(results, output_dir, Path(video_path).stem)
            
        logger.info("Video processing complete")
        return results
    
    def _validate_video(self, frames: np.ndarray, metadata: Dict) -> bool:  # check duration and resolution
        """
        Validate video meets quality requirements
        
        Args:
            frames: Video frames
            metadata: Video metadata
            
        Returns:
            True if valid, False otherwise
        """
        # Check duration
        min_duration = self.config['video']['min_duration']
        max_duration = self.config['video']['max_duration']
        
        if metadata['duration'] < min_duration:
            logger.error(f"Video too short: {metadata['duration']:.1f}s < {min_duration}s")
            return False
            
        if metadata['duration'] > max_duration:
            logger.warning(f"Video too long: {metadata['duration']:.1f}s > {max_duration}s")
            # Continue processing but warn
            
        # Check resolution
        min_res = self.config['quality']['min_resolution']
        if metadata['height'] < min_res[0] or metadata['width'] < min_res[1]:
            logger.error(f"Resolution too low: {metadata['height']}x{metadata['width']}")
            return False
            
        return True
    
    def _detect_and_track_person(self, frames: np.ndarray) -> List[Optional[List[int]]]:  # takes the primary detection, detects and tracks it, interpolates its bboxes, and puts each bbox into a sequential list
        """
        Detect and track primary person across frames
        
        Args:
            frames: Video frames
            
        Returns:
            List of bounding boxes for each frame
        """
        person_bboxes = []
        
        # Reset tracker
        self.person_tracker.reset()
        
        # Process frames with progress bar
        for frame in tqdm(frames, desc="Detecting person"):  # tqdm just adds a progress bar  # 100%|███████████████████████████| 100/100 [00:02<00:00, 40.1it/s]
            if self.person_detector:
                # Detect persons in frame
                detections = self.person_detector.detect_persons(frame)  # use YOLO v8 to detect each person in the frame
                
                # Update tracker
                tracked_detections = self.person_tracker.update(detections)  # remove lost tracks, and update the bboxes and count of ongoing ones
                
                # Get primary person
                if tracked_detections:
                    # Use first tracked detection (primary)
                    bbox = tracked_detections[0]['bbox']
                else:
                    bbox = None
            else:
                # Fallback: assume person is in center
                h, w = frame.shape[:2]
                bbox = [w//4, h//4, 3*w//4, 3*h//4]
                
            person_bboxes.append(bbox)
            
        # Fill gaps in detections using interpolation
        person_bboxes = self._interpolate_bboxes(person_bboxes)  # tracker tracks it but doesn't interpolate bbs. this does
        
        return person_bboxes
    
    def _segment_person(self,  # use either MediaPipe or background subtractor to segment the person as a binary mask, filling in any gaps or removing clusters that may occur
                       frames: np.ndarray,
                       bboxes: List[Optional[List[int]]]) -> np.ndarray:
        """
        Segment person from background
        
        Args:
            frames: Video frames
            bboxes: Bounding boxes for each frame
            
        Returns:
            Segmentation masks
        """
        masks = []
        
        for frame, bbox in tqdm(zip(frames, bboxes), 
                               total=len(frames),
                               desc="Segmenting person"):
            # Segment person
            mask = self.segmenter.segment_person(frame, bbox)
            
            # Clean up mask
            mask = self.segmenter.cleanup_mask(mask, min_area=500)
            
            masks.append(mask)
            
        return np.array(masks)
    
    def _interpolate_bboxes(self, bboxes: List[Optional[List[int]]]) -> List[List[int]]:
        """
        Interpolate missing bounding boxes
        
        Args:
            bboxes: List of bounding boxes with possible None values
            
        Returns:
            List of bounding boxes with interpolated values
        """
        # Find valid bbox indices
        valid_indices = [i for i, b in enumerate(bboxes) if b is not None]
        
        if not valid_indices:
            return bboxes
            
        # Interpolate missing bboxes
        interpolated = []
        for i in range(len(bboxes)):
            if bboxes[i] is not None:
                interpolated.append(bboxes[i])
            else:
                # Find nearest valid bboxes
                prev_idx = max([idx for idx in valid_indices if idx < i], default=None)
                next_idx = min([idx for idx in valid_indices if idx > i], default=None)
                
                if prev_idx is not None and next_idx is not None:
                    # Linear interpolation
                    prev_bbox = bboxes[prev_idx]
                    next_bbox = bboxes[next_idx]
                    
                    alpha = (i - prev_idx) / (next_idx - prev_idx)  # a fraction from 0 → 1 representing how far i is between the two key frames.
                    interp_bbox = [
                        int(prev_bbox[j] * (1 - alpha) + next_bbox[j] * alpha)  # interpolate linearly
                        for j in range(4)
                    ]
                    interpolated.append(interp_bbox)
                    
                elif prev_idx is not None:
                    # Use previous bbox
                    interpolated.append(bboxes[prev_idx])
                    
                elif next_idx is not None:
                    # Use next bbox
                    interpolated.append(bboxes[next_idx])
                    
                else:
                    # No valid bbox found
                    interpolated.append(None)
                    
        return interpolated
    
    def _save_outputs(self, results: Dict, output_dir: str, video_name: str):
        """
        Save processing outputs
        
        Args:
            results: Processing results
            output_dir: Output directory
            video_name: Original video name
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save silhouette sequence
        silhouette_path = output_path / f"{video_name}_silhouettes.npy"
        np.save(silhouette_path, results['silhouettes'])
        logger.info(f"Saved silhouettes to {silhouette_path}")
        
        # Save visualization
        if self.config['output']['visualization']:
            vis_path = output_path / f"{video_name}_visualization.png"
            visualization = self.silhouette_generator.visualize_silhouettes(
                results['silhouettes'], n_display=10
            )
            import cv2
            cv2.imwrite(str(vis_path), visualization)
            logger.info(f"Saved visualization to {vis_path}")
            
   
        # Save metadata
        import json
        metadata_path = output_path / f"{video_name}_metadata.json"
        metadata = {
            'video_metadata': results['video_metadata'],
            'num_frames_processed': results['num_frames_processed'],
            'num_valid_detections': results['num_valid_detections'],
            'sequence_shape': list(results['sequence_shape'])
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        
    def process_batch(self, video_paths: List[str], output_dir: str) -> List[Dict]:  # process multiple videos
        """
        Process multiple videos
        
        Args:
            video_paths: List of video paths
            output_dir: Output directory
            
        Returns:
            List of results for each video
        """
        results = []
        
        for video_path in video_paths:
            try:
                result = self.process_video(video_path, output_dir)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                results.append(None)
                
        return results


if __name__ == "__main__":
    # Example usage
    pipeline = VideoToSilhouettePipeline("config.yaml")
    
    # # Process single video
    results = pipeline.process_video("IMG_6123 (1).mov", "output/")
    
    print("Video processing pipeline initialized successfully!")
