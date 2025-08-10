"""
Main video summarizer processor that coordinates all components
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time

from .video_processor import VideoProcessor, validate_video_file
from .speech_to_text import SpeechToTextProcessor
from .summarizer import TextSummarizer
from .config import OUTPUT_DIR, TEMP_DIR

logger = logging.getLogger(__name__)


class VideoSummarizer:
    """
    Main class that coordinates video processing, transcription, and summarization
    """
    
    def __init__(self, whisper_model: str = None, summarization_model: str = None):
        self.video_processor = VideoProcessor()
        self.speech_processor = SpeechToTextProcessor(whisper_model) if whisper_model else SpeechToTextProcessor()
        self.text_summarizer = TextSummarizer(summarization_model) if summarization_model else TextSummarizer()
        
        # Processing state
        self.current_video_path = None
        self.processing_results = {}
        
    def process_video(self, video_path: str, 
                     language: Optional[str] = None,
                     detect_scenes: bool = True,
                     create_highlights: bool = True,
                     max_highlight_duration: float = 60.0,
                     output_formats: List[str] = None) -> Dict:
        """
        Complete video processing pipeline
        
        Args:
            video_path: Path to input video file
            language: Language code for transcription (auto-detect if None)
            detect_scenes: Whether to perform scene detection
            create_highlights: Whether to create highlight reel
            max_highlight_duration: Maximum duration for highlight reel
            output_formats: List of output formats ['json', 'txt', 'srt']
        
        Returns:
            Dictionary with all processing results
        """
        if output_formats is None:
            output_formats = ['json', 'txt']
        
        start_time = time.time()
        logger.info(f"Starting video processing: {video_path}")
        
        try:
            # Step 1: Validate and load video
            if not validate_video_file(video_path):
                raise ValueError(f"Invalid video file: {video_path}")
            
            if not self.video_processor.load_video(video_path):
                raise ValueError(f"Failed to load video: {video_path}")
            
            self.current_video_path = video_path
            video_info = self.video_processor.get_video_info()
            logger.info(f"Video info: {video_info}")
            
            # Step 2: Extract audio
            logger.info("Extracting audio...")
            audio_path = self.video_processor.extract_audio()
            
            # Step 3: Scene detection (optional)
            scenes = []
            if detect_scenes:
                logger.info("Detecting scenes...")
                scenes = self.video_processor.detect_scenes()
                logger.info(f"Detected {len(scenes)} scenes")
            
            # Step 4: Transcribe audio
            logger.info("Transcribing audio...")
            if scenes:
                transcription_result = self.speech_processor.transcribe_with_scenes(
                    audio_path, scenes, language
                )
            else:
                full_transcription = self.speech_processor.transcribe_audio(audio_path, language)
                transcription_result = {
                    'full_transcription': full_transcription,
                    'scene_transcriptions': [],
                    'total_scenes': 0
                }
            
            # Step 5: Generate summaries
            logger.info("Generating summaries...")
            if scenes and transcription_result['scene_transcriptions']:
                summary_result = self.text_summarizer.summarize_scenes(
                    transcription_result['scene_transcriptions']
                )
            else:
                # Fallback to simple summarization
                full_summary = self.text_summarizer.summarize_text(
                    transcription_result['full_transcription']['text']
                )
                summary_result = {
                    'scene_summaries': [],
                    'overall_summary': full_summary['summary'],
                    'overall_key_points': self.text_summarizer.extract_key_points(
                        transcription_result['full_transcription']['text']
                    ),
                    'total_scenes': 0,
                    'overall_compression_ratio': full_summary['compression_ratio']
                }
            
            # Step 6: Generate highlights (optional)
            highlight_video_path = None
            highlight_timestamps = []
            
            if create_highlights and summary_result['scene_summaries']:
                logger.info("Generating highlights...")
                highlight_timestamps = self.text_summarizer.generate_highlight_timestamps(
                    summary_result['scene_summaries']
                )
                
                if highlight_timestamps:
                    output_name = Path(video_path).stem
                    highlight_video_path = OUTPUT_DIR / f"{output_name}_highlights.mp4"
                    
                    try:
                        self.video_processor.create_highlight_reel(
                            highlight_timestamps,
                            str(highlight_video_path),
                            max_highlight_duration
                        )
                        logger.info(f"Highlight reel created: {highlight_video_path}")
                    except Exception as e:
                        logger.warning(f"Failed to create highlight reel: {str(e)}")
                        highlight_video_path = None
            
            # Step 7: Extract key frames
            logger.info("Extracting key frames...")
            try:
                key_frames = self.video_processor.extract_key_frames(5)
                key_frames_info = {
                    'count': len(key_frames),
                    'extracted': True
                }
            except Exception as e:
                logger.warning(f"Failed to extract key frames: {str(e)}")
                key_frames_info = {
                    'count': 0,
                    'extracted': False
                }
            
            # Compile final results
            processing_time = time.time() - start_time
            
            results = {
                'video_info': video_info,
                'processing_time': processing_time,
                'transcription': transcription_result,
                'summary': summary_result,
                'highlights': {
                    'timestamps': highlight_timestamps,
                    'video_path': str(highlight_video_path) if highlight_video_path else None,
                    'duration': sum([end - start for start, end in highlight_timestamps])
                },
                'key_frames': key_frames_info,
                'scenes': scenes,
                'metadata': {
                    'whisper_model': self.speech_processor.model_name,
                    'summarization_model': self.text_summarizer.model_name,
                    'language_detected': transcription_result['full_transcription']['language'],
                    'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'scene_detection_enabled': detect_scenes,
                    'highlights_created': highlight_video_path is not None
                }
            }
            
            # Step 8: Save results in requested formats
            output_base_name = Path(video_path).stem
            saved_files = []
            
            for format_type in output_formats:
                try:
                    output_file = self._save_results(results, output_base_name, format_type)
                    saved_files.append(output_file)
                    logger.info(f"Results saved: {output_file}")
                except Exception as e:
                    logger.warning(f"Failed to save in {format_type} format: {str(e)}")
            
            results['output_files'] = saved_files
            
            # Clean up temporary files
            try:
                Path(audio_path).unlink(missing_ok=True)
                logger.info("Temporary files cleaned up")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {str(e)}")
            
            logger.info(f"Video processing completed in {processing_time:.2f} seconds")
            self.processing_results = results
            
            return results
            
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            raise
        finally:
            # Ensure video processor is closed
            self.video_processor.close()
    
    def _save_results(self, results: Dict, base_name: str, format_type: str) -> str:
        """Save processing results in specified format"""
        
        if format_type.lower() == 'json':
            output_path = OUTPUT_DIR / f"{base_name}_summary.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
        elif format_type.lower() == 'txt':
            output_path = OUTPUT_DIR / f"{base_name}_summary.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                self._write_text_summary(f, results)
        
        elif format_type.lower() == 'srt':
            # Save transcription as SRT
            output_path = OUTPUT_DIR / f"{base_name}_transcript.srt"
            self.speech_processor.save_transcription(
                results['transcription']['full_transcription'],
                str(output_path),
                'srt'
            )
        
        elif format_type.lower() == 'html':
            output_path = OUTPUT_DIR / f"{base_name}_summary.html"
            with open(output_path, 'w', encoding='utf-8') as f:
                self._write_html_summary(f, results)
        
        else:
            raise ValueError(f"Unsupported output format: {format_type}")
        
        return str(output_path)
    
    def _write_text_summary(self, file, results: Dict):
        """Write results in text format"""
        video_info = results['video_info']
        summary = results['summary']
        transcription = results['transcription']
        
        file.write("=" * 80 + "\n")
        file.write("AI VIDEO CONTENT SUMMARY\n")
        file.write("=" * 80 + "\n\n")
        
        # Video Information
        file.write("VIDEO INFORMATION:\n")
        file.write("-" * 40 + "\n")
        file.write(f"Filename: {video_info['filename']}\n")
        file.write(f"Duration: {video_info['duration']:.2f} seconds ({video_info['duration']/60:.1f} minutes)\n")
        file.write(f"Resolution: {video_info['size']}\n")
        file.write(f"FPS: {video_info['fps']:.2f}\n")
        file.write(f"Language: {transcription['full_transcription']['language']}\n")
        file.write(f"Scenes Detected: {video_info['scenes_detected']}\n\n")
        
        # Overall Summary
        file.write("OVERALL SUMMARY:\n")
        file.write("-" * 40 + "\n")
        file.write(f"{summary['overall_summary']}\n\n")
        
        # Key Points
        if summary['overall_key_points']:
            file.write("KEY POINTS:\n")
            file.write("-" * 40 + "\n")
            for i, point in enumerate(summary['overall_key_points'], 1):
                file.write(f"{i}. {point}\n")
            file.write("\n")
        
        # Scene Summaries
        if summary['scene_summaries']:
            file.write("SCENE-BY-SCENE BREAKDOWN:\n")
            file.write("-" * 40 + "\n")
            for scene in summary['scene_summaries']:
                file.write(f"Scene {scene['scene_id']} ({scene['start_time']:.1f}s - {scene['end_time']:.1f}s):\n")
                file.write(f"Summary: {scene['summary']}\n")
                if scene['key_points']:
                    file.write("Key Points:\n")
                    for point in scene['key_points']:
                        file.write(f"  • {point}\n")
                file.write("\n")
        
        # Highlights
        if results['highlights']['timestamps']:
            file.write("HIGHLIGHTS:\n")
            file.write("-" * 40 + "\n")
            for i, (start, end) in enumerate(results['highlights']['timestamps'], 1):
                file.write(f"{i}. {start:.1f}s - {end:.1f}s ({end-start:.1f}s duration)\n")
            file.write(f"\nTotal highlight duration: {results['highlights']['duration']:.1f}s\n")
            if results['highlights']['video_path']:
                file.write(f"Highlight video: {results['highlights']['video_path']}\n")
            file.write("\n")
        
        # Full Transcript
        file.write("FULL TRANSCRIPT:\n")
        file.write("-" * 40 + "\n")
        file.write(transcription['full_transcription']['text'])
        file.write("\n\n")
        
        # Processing Info
        file.write("PROCESSING INFORMATION:\n")
        file.write("-" * 40 + "\n")
        file.write(f"Processing Time: {results['processing_time']:.2f} seconds\n")
        file.write(f"Whisper Model: {results['metadata']['whisper_model']}\n")
        file.write(f"Summarization Model: {results['metadata']['summarization_model']}\n")
        file.write(f"Processing Date: {results['metadata']['processing_date']}\n")
    
    def _write_html_summary(self, file, results: Dict):
        """Write results in HTML format"""
        video_info = results['video_info']
        summary = results['summary']
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Summary - {video_info['filename']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .scene {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .highlight {{ background-color: #e8f4fd; padding: 10px; border-left: 4px solid #2196F3; }}
        ul {{ padding-left: 20px; }}
        .timestamp {{ color: #666; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AI Video Content Summary</h1>
        <h2>{video_info['filename']}</h2>
        <p><strong>Duration:</strong> {video_info['duration']:.2f} seconds ({video_info['duration']/60:.1f} minutes)</p>
        <p><strong>Language:</strong> {results['transcription']['full_transcription']['language']}</p>
        <p><strong>Scenes:</strong> {video_info['scenes_detected']}</p>
    </div>
    
    <div class="section">
        <h3>Overall Summary</h3>
        <div class="highlight">
            <p>{summary['overall_summary']}</p>
        </div>
    </div>
    
    <div class="section">
        <h3>Key Points</h3>
        <ul>
        """
        
        for point in summary['overall_key_points']:
            html_content += f"<li>{point}</li>\n"
        
        html_content += "</ul></div>"
        
        # Scene summaries
        if summary['scene_summaries']:
            html_content += '<div class="section"><h3>Scene Breakdown</h3>'
            for scene in summary['scene_summaries']:
                html_content += f"""
                <div class="scene">
                    <h4>Scene {scene['scene_id']} <span class="timestamp">({scene['start_time']:.1f}s - {scene['end_time']:.1f}s)</span></h4>
                    <p>{scene['summary']}</p>
                </div>
                """
            html_content += "</div>"
        
        html_content += """
    </body>
    </html>
    """
        
        file.write(html_content)
    
    def get_processing_summary(self) -> Dict:
        """Get a summary of the last processing operation"""
        if not self.processing_results:
            return {"status": "No processing completed yet"}
        
        results = self.processing_results
        return {
            "status": "completed",
            "video_filename": results['video_info']['filename'],
            "duration": f"{results['video_info']['duration']:.1f} seconds",
            "language": results['metadata']['language_detected'],
            "scenes_detected": len(results['scenes']),
            "processing_time": f"{results['processing_time']:.2f} seconds",
            "summary_length": len(results['summary']['overall_summary'].split()),
            "highlights_created": results['metadata']['highlights_created'],
            "output_files": results['output_files']
        }
    
    def cleanup(self):
        """Clean up resources and temporary files"""
        self.video_processor.close()
        
        # Clean up temporary files
        try:
            import shutil
            if TEMP_DIR.exists():
                for temp_file in TEMP_DIR.glob("*"):
                    temp_file.unlink(missing_ok=True)
            logger.info("Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {str(e)}")


def process_video_file(video_path: str, **kwargs) -> Dict:
    """
    Convenience function to process a single video file
    
    Args:
        video_path: Path to video file
        **kwargs: Additional arguments for processing
    
    Returns:
        Processing results dictionary
    """
    summarizer = VideoSummarizer()
    try:
        return summarizer.process_video(video_path, **kwargs)
    finally:
        summarizer.cleanup()
