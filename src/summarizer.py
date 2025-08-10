"""
Text summarization module using BART and T5 models from Hugging Face
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    BartTokenizer, BartForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration,
    pipeline
)
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import re
from collections import Counter

from .config import (
    SUMMARIZATION_MODEL, 
    MAX_SUMMARY_LENGTH, 
    MIN_SUMMARY_LENGTH,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    CACHE_DIR
)

logger = logging.getLogger(__name__)


class TextSummarizer:
    def __init__(self, model_name: str = SUMMARIZATION_MODEL):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
    
    def load_model(self):
        """Load summarization model and tokenizer"""
        if self.model is None:
            try:
                logger.info(f"Loading summarization model: {self.model_name}")
                
                # Load model and tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=str(CACHE_DIR)
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    cache_dir=str(CACHE_DIR)
                )
                
                # Move model to device
                self.model.to(self.device)
                
                # Create pipeline for easier use
                self.pipeline = pipeline(
                    "summarization",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1
                )
                
                logger.info("Summarization model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load summarization model: {str(e)}")
                raise
    
    def summarize_text(self, text: str, max_length: int = MAX_SUMMARY_LENGTH, 
                      min_length: int = MIN_SUMMARY_LENGTH) -> Dict:
        """
        Summarize input text
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
        
        Returns:
            Dictionary with summary and metadata
        """
        if self.pipeline is None:
            self.load_model()
        
        if not text.strip():
            return {
                'summary': '',
                'original_length': 0,
                'summary_length': 0,
                'compression_ratio': 0.0
            }
        
        try:
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Handle long texts by chunking
            if len(self.tokenizer.encode(cleaned_text)) > CHUNK_SIZE:
                summary = self._summarize_long_text(cleaned_text, max_length, min_length)
            else:
                # Single pass summarization
                result = self.pipeline(
                    cleaned_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )
                summary = result[0]['summary_text']
            
            # Calculate metrics
            original_length = len(text.split())
            summary_length = len(summary.split())
            compression_ratio = summary_length / original_length if original_length > 0 else 0.0
            
            return {
                'summary': summary.strip(),
                'original_length': original_length,
                'summary_length': summary_length,
                'compression_ratio': compression_ratio
            }
            
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            raise
    
    def _summarize_long_text(self, text: str, max_length: int, min_length: int) -> str:
        """
        Summarize long text by chunking and combining summaries
        """
        # Split text into sentences for better chunking
        sentences = self._split_into_sentences(text)
        chunks = self._create_chunks(sentences)
        
        if len(chunks) == 1:
            # If only one chunk, summarize directly
            result = self.pipeline(
                chunks[0],
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            return result[0]['summary_text']
        
        # Summarize each chunk
        chunk_summaries = []
        chunk_max_length = max(50, max_length // len(chunks))
        chunk_min_length = max(10, min_length // len(chunks))
        
        for i, chunk in enumerate(chunks):
            try:
                result = self.pipeline(
                    chunk,
                    max_length=chunk_max_length,
                    min_length=chunk_min_length,
                    do_sample=False,
                    truncation=True
                )
                chunk_summaries.append(result[0]['summary_text'])
                logger.info(f"Summarized chunk {i+1}/{len(chunks)}")
            except Exception as e:
                logger.warning(f"Failed to summarize chunk {i+1}: {str(e)}")
                continue
        
        if not chunk_summaries:
            raise ValueError("Failed to summarize any chunks")
        
        # Combine chunk summaries
        combined_summary = ' '.join(chunk_summaries)
        
        # If combined summary is still too long, summarize it again
        if len(self.tokenizer.encode(combined_summary)) > CHUNK_SIZE:
            result = self.pipeline(
                combined_summary,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            return result[0]['summary_text']
        
        return combined_summary
    
    def summarize_scenes(self, scene_transcriptions: List[Dict], 
                        max_length_per_scene: int = 100) -> Dict:
        """
        Summarize each scene individually and create an overall summary
        
        Args:
            scene_transcriptions: List of scene transcription dictionaries
            max_length_per_scene: Maximum length for each scene summary
        
        Returns:
            Dictionary with scene summaries and overall summary
        """
        scene_summaries = []
        all_text = []
        
        for scene in scene_transcriptions:
            if not scene['text'].strip():
                scene_summaries.append({
                    'scene_id': scene['scene_id'],
                    'start_time': scene['start_time'],
                    'end_time': scene['end_time'],
                    'summary': '',
                    'key_points': []
                })
                continue
            
            # Summarize individual scene
            scene_summary = self.summarize_text(
                scene['text'], 
                max_length=max_length_per_scene,
                min_length=20
            )
            
            # Extract key points
            key_points = self.extract_key_points(scene['text'], top_k=3)
            
            scene_summaries.append({
                'scene_id': scene['scene_id'],
                'start_time': scene['start_time'],
                'end_time': scene['end_time'],
                'summary': scene_summary['summary'],
                'key_points': key_points,
                'compression_ratio': scene_summary['compression_ratio']
            })
            
            all_text.append(scene['text'])
        
        # Create overall summary from all scenes
        full_text = ' '.join(all_text)
        overall_summary = self.summarize_text(full_text)
        
        # Extract overall key points
        overall_key_points = self.extract_key_points(full_text, top_k=10)
        
        return {
            'scene_summaries': scene_summaries,
            'overall_summary': overall_summary['summary'],
            'overall_key_points': overall_key_points,
            'total_scenes': len(scene_summaries),
            'overall_compression_ratio': overall_summary['compression_ratio']
        }
    
    def extract_key_points(self, text: str, top_k: int = 5) -> List[str]:
        """
        Extract key points/sentences from text using simple scoring
        """
        sentences = self._split_into_sentences(text)
        if len(sentences) <= top_k:
            return sentences
        
        # Score sentences based on word frequency and position
        word_freq = self._calculate_word_frequency(text)
        sentence_scores = []
        
        for i, sentence in enumerate(sentences):
            words = re.findall(r'\w+', sentence.lower())
            
            # Calculate sentence score based on word frequencies
            word_score = sum(word_freq.get(word, 0) for word in words) / len(words) if words else 0
            
            # Boost score for sentences at the beginning (often contain main topics)
            position_score = 1.0 - (i / len(sentences)) * 0.3
            
            # Boost score for longer sentences (often contain more information)
            length_score = min(len(words) / 20, 1.0)
            
            total_score = word_score * position_score * length_score
            sentence_scores.append((sentence.strip(), total_score))
        
        # Sort by score and return top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        return [sentence for sentence, _ in sentence_scores[:top_k]]
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for summarization"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Ensure text ends with punctuation
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting based on punctuation
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _create_chunks(self, sentences: List[str]) -> List[str]:
        """Create text chunks that fit within token limits"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(self.tokenizer.encode(sentence))
            
            if current_length + sentence_length > CHUNK_SIZE - OVERLAP_SIZE:
                # Finalize current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                if len(current_chunk) > 1:
                    # Keep last sentence for overlap
                    current_chunk = [current_chunk[-1], sentence]
                    current_length = len(self.tokenizer.encode(current_chunk[-1])) + sentence_length
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _calculate_word_frequency(self, text: str) -> Dict[str, float]:
        """Calculate normalized word frequency"""
        words = re.findall(r'\w+', text.lower())
        word_count = Counter(words)
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Filter stop words and calculate frequency
        filtered_words = {word: count for word, count in word_count.items() 
                         if word not in stop_words and len(word) > 2}
        
        # Normalize frequencies
        max_count = max(filtered_words.values()) if filtered_words else 1
        return {word: count / max_count for word, count in filtered_words.items()}
    
    def generate_highlight_timestamps(self, scene_summaries: List[Dict], 
                                    max_highlights: int = 5) -> List[Tuple[float, float]]:
        """
        Generate timestamps for video highlights based on scene summaries
        
        Args:
            scene_summaries: List of scene summary dictionaries
            max_highlights: Maximum number of highlights to generate
        
        Returns:
            List of (start_time, end_time) tuples for highlights
        """
        # Score scenes based on summary quality and content richness
        scored_scenes = []
        
        for scene in scene_summaries:
            if not scene['summary']:
                continue
            
            # Score based on summary length and compression ratio
            summary_length_score = min(len(scene['summary'].split()) / 20, 1.0)
            compression_score = 1.0 - scene.get('compression_ratio', 0.5)
            key_points_score = len(scene.get('key_points', [])) / 5
            
            # Duration score (prefer medium-length scenes)
            duration = scene['end_time'] - scene['start_time']
            duration_score = min(duration / 60, 1.0) if duration < 120 else max(0.5, 120 / duration)
            
            total_score = (summary_length_score + compression_score + 
                          key_points_score + duration_score) / 4
            
            scored_scenes.append((scene, total_score))
        
        # Sort by score and select top scenes
        scored_scenes.sort(key=lambda x: x[1], reverse=True)
        
        highlights = []
        for scene, _ in scored_scenes[:max_highlights]:
            highlights.append((scene['start_time'], scene['end_time']))
        
        # Sort highlights by time
        highlights.sort(key=lambda x: x[0])
        
        return highlights
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'model_loaded': self.model is not None,
            'cuda_available': torch.cuda.is_available()
        }
