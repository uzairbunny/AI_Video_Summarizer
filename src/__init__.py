"""
AI Video Content Summarizer

A comprehensive system for processing video files to generate:
- Speech-to-text transcription using Whisper AI
- Content summarization using BART/T5 models
- Scene detection and analysis
- Highlight reel creation

Main Components:
- VideoProcessor: Handles video loading, scene detection, and audio extraction
- SpeechToTextProcessor: Transcribes audio using Whisper models
- TextSummarizer: Creates summaries using transformer models
- VideoSummarizer: Main coordinator class

Usage:
    from src.main_processor import VideoSummarizer, process_video_file
    
    # Simple usage
    results = process_video_file("video.mp4")
    
    # Advanced usage
    summarizer = VideoSummarizer()
    results = summarizer.process_video(
        "video.mp4",
        language="en",
        detect_scenes=True,
        create_highlights=True
    )
"""

__version__ = "1.0.0"
__author__ = "AI Video Summarizer Team"

# Import main classes for easy access
from .main_processor import VideoSummarizer, process_video_file
from .video_processor import VideoProcessor
from .speech_to_text import SpeechToTextProcessor
from .summarizer import TextSummarizer

__all__ = [
    'VideoSummarizer',
    'process_video_file',
    'VideoProcessor',
    'SpeechToTextProcessor',
    'TextSummarizer'
]
