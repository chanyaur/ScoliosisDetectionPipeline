"""
Video loading and basic processing utilities
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator
import logging

logger = logging.getLogger(__name__)


class VideoLoader:
    """
    Load and process video files with various utilities
    """
    
    def __init__(self, target_fps: int = 15, max_frames: Optional[int] = None):  # scoliosis 1k also does 15 fps
        """
        Initialize video loader
        
        Args:
            target_fps: Target frame rate for processing
            max_frames: Maximum number of frames to load (None for all)
        """
        self.target_fps = target_fps
        self.max_frames = max_frames
        
    def load_video(self, video_path: str) -> Tuple[np.ndarray, dict]:  # load vid in RGB, making sure it runs at 15 fps (stored in an array called frames)
        """
        Load video and extract frames
        
        Args:
            video_path: Path to video file
            
        Returns:
            frames: Array of shape (N, H, W, C)
            metadata: Dictionary with video information
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            
        # Get video metadata
        metadata = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        logger.info(f"Loading video: {video_path.name}")
        logger.info(f"Video metadata: {metadata}")
        
        # Calculate frame sampling rate
        sample_rate = max(1, int(metadata['fps'] / self.target_fps))  # each x [sample_rate] frames, sample one. keeps it at target FPS. If it's lower, clamp it to 1
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:  # if not successfully read
                break
                
            # Sample frames based on target FPS
            if frame_idx % sample_rate == 0:  # get the relevant frames and make it into an RGB sequence, with each frame in frames
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                if self.max_frames and len(frames) >= self.max_frames:
                    break
                    
            frame_idx += 1
            
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video")
            
        frames_array = np.array(frames)
        logger.info(f"Loaded {len(frames)} frames with shape {frames_array.shape}")
        
        return frames_array, metadata
    
    def load_video_generator(self, video_path: str) -> Generator[np.ndarray, None, None]:  # version of the above that yields one frame at a time
        """
        Load video frames using generator for memory efficiency
        
        Args:
            video_path: Path to video file
            
        Yields:
            frame: Single frame as numpy array
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_rate = max(1, int(fps / self.target_fps))
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_rgb  # memory efficient as it returns one frame at a time, and then keeps going when caller asks
                
            frame_idx += 1
            
        cap.release()
        
    def extract_frame_range(self, video_path: str, start_frame: int,  # extract only the amount of frames you need from a video
                           end_frame: int) -> np.ndarray:
        """
        Extract specific frame range from video
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            end_frame: Ending frame index
            
        Returns:
            frames: Array of extracted frames
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            
        # Set to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for i in range(start_frame, min(end_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
        cap.release()
        
        return np.array(frames)
    
    def save_frames_as_video(self, frames: np.ndarray, output_path: str,  # converts the RGB video to BGR so that openCV can process it
                           fps: int = 30) -> None:
        """
        Save frames array as video file
        
        Args:
            frames: Array of frames (N, H, W, C)
            output_path: Path for output video
            fps: Frame rate for output video
        """
        if len(frames) == 0:
            raise ValueError("No frames to save")
            
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # tells openCV which encoder/decoder to use when writing video
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV (uses BGR by default)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
        out.release()
        logger.info(f"Saved video to {output_path}")
        
    def get_video_info(self, video_path: str) -> dict:  # return metadata (characteristics) abt the video
        """
        Get video information without loading frames
        
        Args:
            video_path: Path to video file
            
        Returns:
            metadata: Dictionary with video information
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            
        metadata = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        cap.release()
        
        return metadata
    
    @staticmethod
    def resize_frame(frame: np.ndarray, target_size: Tuple[int, int],  # resizes frame to target size
                    maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize frame to target size
        
        Args:
            frame: Input frame
            target_size: Target (width, height)
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized frame
        """
        if maintain_aspect:
            h, w = frame.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Pad to target size
            pad_w = target_w - new_w
            pad_h = target_h - new_h
            
            top = pad_h // 2  # adds the same amount on top and bottom (may be 0)
            bottom = pad_h - top
            left = pad_w // 2  # adds the same amount on left and right (may be 0)
            right = pad_w - left
            
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
            return padded
        else:
            return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
