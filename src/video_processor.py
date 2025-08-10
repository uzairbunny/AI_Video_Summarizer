"""
Video processing module for scene detection and audio extraction
"""

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from typing import List, Tuple, Dict
import logging
from pathlib import Path
import tempfile

from .config import (
    SCENE_THRESHOLD, 
    MIN_SCENE_LENGTH, 
    FRAME_RATE, 
    AUDIO_SAMPLE_RATE,
    TEMP_DIR
)

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self):
        self.current_video = None
        self.scenes = []
        
    def load_video(self, video_path: str) -> bool:
        """Load video file and validate format"""
        try:
            self.current_video = VideoFileClip(video_path)
            logger.info(f"Video loaded: {video_path}")
            logger.info(f"Duration: {self.current_video.duration:.2f}s, FPS: {self.current_video.fps}")
            return True
        except Exception as e:
            logger.error(f"Failed to load video {video_path}: {str(e)}")
            return False
    
    def extract_audio(self, output_path: str = None) -> str:
        """Extract audio from video and save as WAV file"""
        if not self.current_video:
            raise ValueError("No video loaded")
        
        if not output_path:
            output_path = TEMP_DIR / f"audio_{hash(str(self.current_video.filename))}.wav"
        
        try:
            # Extract audio and resample
            audio = self.current_video.audio.set_fps(AUDIO_SAMPLE_RATE)
            audio.write_audiofile(str(output_path), verbose=False, logger=None)
            logger.info(f"Audio extracted to: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to extract audio: {str(e)}")
            raise
    
    def detect_scenes(self) -> List[Tuple[float, float]]:
        """
        Detect scene changes in video using histogram comparison
        Returns list of (start_time, end_time) tuples
        """
        if not self.current_video:
            raise ValueError("No video loaded")
        
        scenes = []
        frame_times = []
        histograms = []
        
        # Extract frames at specified frame rate
        duration = self.current_video.duration
        timestamps = np.arange(0, duration, 1.0 / FRAME_RATE)
        
        logger.info(f"Analyzing {len(timestamps)} frames for scene detection...")
        
        for i, t in enumerate(timestamps):
            try:
                # Get frame at timestamp
                frame = self.current_video.get_frame(t)
                
                # Convert to grayscale and calculate histogram
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                frame_times.append(t)
                histograms.append(hist)
                
            except Exception as e:
                logger.warning(f"Failed to process frame at {t:.2f}s: {str(e)}")
                continue
        
        # Compare consecutive histograms to detect scene changes
        scene_starts = [0.0]  # Video always starts with a scene
        
        for i in range(1, len(histograms)):
            # Calculate correlation between consecutive frames
            correlation = cv2.compareHist(
                histograms[i-1], 
                histograms[i], 
                cv2.HISTCMP_CORREL
            )
            
            # If correlation is low, it's likely a scene change
            if correlation < (100 - SCENE_THRESHOLD) / 100:
                scene_starts.append(frame_times[i])
        
        # Add video end as final scene boundary
        scene_starts.append(duration)
        
        # Create scene segments, filtering out too-short scenes
        for i in range(len(scene_starts) - 1):
            start_time = scene_starts[i]
            end_time = scene_starts[i + 1]
            
            if end_time - start_time >= MIN_SCENE_LENGTH:
                scenes.append((start_time, end_time))
        
        self.scenes = scenes
        logger.info(f"Detected {len(scenes)} scenes")
        
        return scenes
    
    def extract_key_frames(self, num_frames: int = 5) -> List[np.ndarray]:
        """Extract key frames from video for thumbnail generation"""
        if not self.current_video:
            raise ValueError("No video loaded")
        
        duration = self.current_video.duration
        timestamps = np.linspace(0, duration * 0.95, num_frames)  # Avoid very end
        
        key_frames = []
        for t in timestamps:
            try:
                frame = self.current_video.get_frame(t)
                key_frames.append(frame)
            except Exception as e:
                logger.warning(f"Failed to extract frame at {t:.2f}s: {str(e)}")
        
        return key_frames
    
    def get_video_info(self) -> Dict:
        """Get comprehensive video information"""
        if not self.current_video:
            raise ValueError("No video loaded")
        
        return {
            'filename': Path(self.current_video.filename).name,
            'duration': self.current_video.duration,
            'fps': self.current_video.fps,
            'size': self.current_video.size,
            'aspect_ratio': self.current_video.aspect_ratio,
            'has_audio': self.current_video.audio is not None,
            'scenes_detected': len(self.scenes)
        }
    
    def create_highlight_reel(self, highlights: List[Tuple[float, float]], 
                            output_path: str, max_duration: float = 60.0) -> str:
        """
        Create a highlight reel from specified time segments
        
        Args:
            highlights: List of (start, end) time tuples
            output_path: Output video file path
            max_duration: Maximum duration of highlight reel
        """
        if not self.current_video:
            raise ValueError("No video loaded")
        
        clips = []
        total_duration = 0
        
        for start, end in highlights:
            if total_duration >= max_duration:
                break
            
            segment_duration = min(end - start, max_duration - total_duration)
            if segment_duration < 1.0:  # Skip very short segments
                continue
            
            try:
                clip = self.current_video.subclip(start, start + segment_duration)
                clips.append(clip)
                total_duration += segment_duration
            except Exception as e:
                logger.warning(f"Failed to create clip {start}-{end}: {str(e)}")
        
        if not clips:
            raise ValueError("No valid clips to create highlight reel")
        
        # Concatenate clips
        from moviepy.editor import concatenate_videoclips
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Write output video
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None
        )
        
        # Clean up
        final_video.close()
        for clip in clips:
            clip.close()
        
        logger.info(f"Highlight reel created: {output_path}")
        return output_path
    
    def close(self):
        """Clean up resources"""
        if self.current_video:
            self.current_video.close()
            self.current_video = None


def validate_video_file(file_path: str) -> bool:
    """Validate if file is a supported video format"""
    from .config import SUPPORTED_FORMATS
    
    path = Path(file_path)
    if not path.exists():
        return False
    
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        return False
    
    # Try to open with OpenCV to verify it's a valid video
    try:
        cap = cv2.VideoCapture(str(file_path))
        ret, _ = cap.read()
        cap.release()
        return ret
    except:
        return False
