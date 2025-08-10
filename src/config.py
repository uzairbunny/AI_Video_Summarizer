"""
Configuration settings for AI Video Summarizer
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Model configurations
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"  # Alternative: "t5-small"
MAX_SUMMARY_LENGTH = 300
MIN_SUMMARY_LENGTH = 50

# Video processing settings
MAX_VIDEO_SIZE_MB = 500
SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
FRAME_RATE = 1  # Extract 1 frame per second for scene detection
MIN_SCENE_DURATION = 2  # Minimum seconds for a scene

# Audio settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

# Scene detection settings
SCENE_THRESHOLD = 30.0  # Threshold for scene change detection
MIN_SCENE_LENGTH = 5  # Minimum scene length in seconds

# Summarization settings
CHUNK_SIZE = 512  # Token chunk size for processing long texts
OVERLAP_SIZE = 50   # Overlap between chunks

# Web app settings
MAX_UPLOAD_SIZE = 16 * 1024 * 1024 * 1024  # 16GB
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}

# API settings
API_TIMEOUT = 300  # 5 minutes timeout for processing

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Model cache directory
CACHE_DIR = BASE_DIR / "model_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Environment variables
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
