"""
Speech-to-text module using OpenAI Whisper
"""

import whisper
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

from .config import WHISPER_MODEL, CACHE_DIR

logger = logging.getLogger(__name__)


class SpeechToTextProcessor:
    def __init__(self, model_name: str = WHISPER_MODEL):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load Whisper model"""
        if self.model is None:
            try:
                logger.info(f"Loading Whisper model: {self.model_name}")
                self.model = whisper.load_model(
                    self.model_name,
                    device=self.device,
                    download_root=str(CACHE_DIR)
                )
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {str(e)}")
                raise
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> Dict:
        """
        Transcribe audio file using Whisper
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es', 'fr'). Auto-detect if None
        
        Returns:
            Dictionary with transcription results including segments and metadata
        """
        if self.model is None:
            self.load_model()
        
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Transcribe with word-level timestamps
            result = self.model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                verbose=False
            )
            
            # Process segments to include more detailed timing
            processed_segments = []
            for segment in result['segments']:
                processed_segment = {
                    'id': segment['id'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                    'confidence': segment.get('avg_logprob', 0.0),
                    'words': []
                }
                
                # Add word-level timestamps if available
                if 'words' in segment:
                    for word in segment['words']:
                        processed_segment['words'].append({
                            'word': word['word'],
                            'start': word['start'],
                            'end': word['end'],
                            'confidence': word.get('probability', 0.0)
                        })
                
                processed_segments.append(processed_segment)
            
            transcription_result = {
                'text': result['text'].strip(),
                'language': result['language'],
                'segments': processed_segments,
                'duration': max([seg['end'] for seg in result['segments']]) if result['segments'] else 0,
                'confidence': np.mean([seg.get('avg_logprob', 0.0) for seg in result['segments']]) if result['segments'] else 0.0
            }
            
            logger.info(f"Transcription completed. Language: {result['language']}, Duration: {transcription_result['duration']:.2f}s")
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
    
    def transcribe_with_scenes(self, audio_path: str, scenes: List[Tuple[float, float]], 
                             language: str = None) -> Dict:
        """
        Transcribe audio with scene-based segmentation
        
        Args:
            audio_path: Path to audio file
            scenes: List of (start_time, end_time) tuples for scenes
            language: Language code for transcription
        
        Returns:
            Dictionary with scene-based transcription results
        """
        full_transcription = self.transcribe_audio(audio_path, language)
        
        # Map segments to scenes
        scene_transcriptions = []
        for i, (scene_start, scene_end) in enumerate(scenes):
            scene_segments = []
            scene_text_parts = []
            
            for segment in full_transcription['segments']:
                # Check if segment overlaps with scene
                if (segment['start'] < scene_end and segment['end'] > scene_start):
                    # Adjust segment timing relative to scene
                    adjusted_segment = segment.copy()
                    adjusted_segment['scene_start'] = max(0, segment['start'] - scene_start)
                    adjusted_segment['scene_end'] = min(scene_end - scene_start, segment['end'] - scene_start)
                    
                    scene_segments.append(adjusted_segment)
                    scene_text_parts.append(segment['text'])
            
            scene_transcription = {
                'scene_id': i + 1,
                'start_time': scene_start,
                'end_time': scene_end,
                'duration': scene_end - scene_start,
                'text': ' '.join(scene_text_parts).strip(),
                'segments': scene_segments,
                'word_count': len(' '.join(scene_text_parts).split()) if scene_text_parts else 0
            }
            
            scene_transcriptions.append(scene_transcription)
        
        return {
            'full_transcription': full_transcription,
            'scene_transcriptions': scene_transcriptions,
            'total_scenes': len(scenes)
        }
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extract important keywords from transcribed text
        Simple implementation using word frequency
        """
        import re
        from collections import Counter
        
        # Simple keyword extraction using frequency
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'what', 'where', 'when',
            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
        }
        
        # Clean and tokenize text
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count frequency and return top keywords
        word_freq = Counter(keywords)
        return [word for word, _ in word_freq.most_common(top_k)]
    
    def save_transcription(self, transcription: Dict, output_path: str, format: str = 'json'):
        """
        Save transcription to file in specified format
        
        Args:
            transcription: Transcription dictionary
            output_path: Output file path
            format: Output format ('json', 'txt', 'srt')
        """
        output_path = Path(output_path)
        
        try:
            if format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(transcription, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    if 'scene_transcriptions' in transcription:
                        # Scene-based format
                        for scene in transcription['scene_transcriptions']:
                            f.write(f"Scene {scene['scene_id']} ({scene['start_time']:.1f}s - {scene['end_time']:.1f}s):\n")
                            f.write(f"{scene['text']}\n\n")
                    else:
                        # Simple format
                        f.write(transcription['text'])
            
            elif format.lower() == 'srt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    segments = transcription.get('segments', [])
                    for i, segment in enumerate(segments, 1):
                        start_time = self._seconds_to_srt_time(segment['start'])
                        end_time = self._seconds_to_srt_time(segment['end'])
                        f.write(f"{i}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"{segment['text']}\n\n")
            
            logger.info(f"Transcription saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save transcription: {str(e)}")
            raise
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'model_loaded': self.model is not None,
            'cuda_available': torch.cuda.is_available()
        }
