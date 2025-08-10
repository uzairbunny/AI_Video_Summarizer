"""
Example script demonstrating basic usage of AI Video Summarizer
"""

import sys
from pathlib import Path
import logging

# Configure logging to see progress
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Import our video processing functions
from src.main_processor import VideoSummarizer, process_video_file


def simple_example():
    """
    Simple example using the convenience function
    """
    print("🎥 Simple Video Processing Example")
    print("=" * 50)
    
    # Example video path (replace with your video)
    video_path = "sample_video.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        print("Please place a video file named 'sample_video.mp4' in the current directory")
        return
    
    try:
        # Process video with default settings
        print(f"Processing: {video_path}")
        results = process_video_file(video_path)
        
        # Print summary
        print("\n📊 Results Summary:")
        print(f"Duration: {results['video_info']['duration']:.1f} seconds")
        print(f"Language: {results['metadata']['language_detected']}")
        print(f"Scenes: {len(results['scenes'])}")
        print(f"\nSummary: {results['summary']['overall_summary'][:200]}...")
        
        if results['highlights']['video_path']:
            print(f"Highlights created: {Path(results['highlights']['video_path']).name}")
        
        print(f"\n✅ Processing completed in {results['processing_time']:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def advanced_example():
    """
    Advanced example with custom configuration
    """
    print("\n🔧 Advanced Video Processing Example")
    print("=" * 50)
    
    # Example video path
    video_path = "sample_video.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    try:
        # Create summarizer with custom models
        summarizer = VideoSummarizer(
            whisper_model="small",  # Better accuracy than base
            summarization_model="facebook/bart-large-cnn"
        )
        
        print(f"Processing: {video_path}")
        print("Using enhanced models for better accuracy...")
        
        # Process with custom settings
        results = summarizer.process_video(
            video_path,
            language="en",  # Force English
            detect_scenes=True,
            create_highlights=True,
            max_highlight_duration=90,  # 90-second highlights
            output_formats=["json", "txt", "srt", "html"]  # All formats
        )
        
        # Detailed results
        print("\n📊 Detailed Results:")
        video_info = results['video_info']
        summary = results['summary']
        
        print(f"📺 Video: {video_info['filename']}")
        print(f"⏱️  Duration: {video_info['duration']:.1f}s ({video_info['duration']/60:.1f} min)")
        print(f"🎞️  FPS: {video_info['fps']:.1f}")
        print(f"📐 Resolution: {video_info['size']}")
        print(f"🌍 Language: {results['metadata']['language_detected']}")
        
        print(f"\n📝 Transcription:")
        print(f"Word count: {len(results['transcription']['full_transcription']['text'].split())}")
        print(f"Confidence: {results['transcription']['full_transcription']['confidence']:.2f}")
        
        print(f"\n📊 Summary:")
        print(f"Original words: {len(results['transcription']['full_transcription']['text'].split())}")
        print(f"Summary words: {len(summary['overall_summary'].split())}")
        print(f"Compression ratio: {summary['overall_compression_ratio']:.2f}")
        
        print(f"\n🎬 Scenes: {len(results['scenes'])} detected")
        if summary['scene_summaries']:
            print("Scene summaries:")
            for i, scene in enumerate(summary['scene_summaries'][:3], 1):
                duration = scene['end_time'] - scene['start_time']
                print(f"  {i}. {scene['start_time']:.1f}s-{scene['end_time']:.1f}s ({duration:.1f}s)")
                print(f"     {scene['summary'][:80]}...")
        
        print(f"\n⭐ Highlights: {len(results['highlights']['timestamps'])} segments")
        if results['highlights']['timestamps']:
            total_duration = sum(end - start for start, end in results['highlights']['timestamps'])
            print(f"Total highlight duration: {total_duration:.1f}s")
            for i, (start, end) in enumerate(results['highlights']['timestamps'], 1):
                print(f"  {i}. {start:.1f}s - {end:.1f}s")
        
        print(f"\n📁 Output files: {len(results['output_files'])}")
        for file_path in results['output_files']:
            file_size = Path(file_path).stat().st_size / 1024
            print(f"  {Path(file_path).name} ({file_size:.1f} KB)")
        
        print(f"\n✅ Advanced processing completed in {results['processing_time']:.1f} seconds")
        
        # Clean up
        summarizer.cleanup()
        
    except Exception as e:
        print(f"❌ Error: {e}")


def model_info_example():
    """
    Show information about available models
    """
    print("\n🔍 Model Information")
    print("=" * 50)
    
    try:
        from src.speech_to_text import SpeechToTextProcessor
        from src.summarizer import TextSummarizer
        
        # Speech-to-text model info
        stt = SpeechToTextProcessor()
        stt_info = stt.get_model_info()
        
        print("🗣️ Speech-to-Text (Whisper):")
        print(f"  Default model: {stt_info['model_name']}")
        print(f"  Device: {stt_info['device']}")
        print(f"  CUDA available: {stt_info['cuda_available']}")
        
        # Summarization model info  
        summarizer = TextSummarizer()
        sum_info = summarizer.get_model_info()
        
        print("\n📝 Summarization:")
        print(f"  Default model: {sum_info['model_name']}")
        print(f"  Device: {sum_info['device']}")
        
        print("\n📊 Available Whisper Models:")
        models = ["tiny", "base", "small", "medium", "large"]
        sizes = ["~39 MB", "~142 MB", "~466 MB", "~1.5 GB", "~2.9 GB"]
        
        for model, size in zip(models, sizes):
            print(f"  {model:8} - {size}")
        
        print("\n💡 Model Selection Tips:")
        print("  • Use 'tiny' for testing or very fast processing")
        print("  • Use 'base' for good balance (default)")
        print("  • Use 'small' or 'medium' for better accuracy")
        print("  • Use 'large' for maximum accuracy (if you have time/resources)")
        print("  • GPU acceleration speeds up all models significantly")
        
    except Exception as e:
        print(f"❌ Error getting model info: {e}")


if __name__ == "__main__":
    print("🎥 AI Video Content Summarizer - Examples")
    print("=" * 60)
    
    # Show model information
    model_info_example()
    
    # Run simple example
    simple_example()
    
    # Run advanced example
    advanced_example()
    
    print("\n" + "=" * 60)
    print("👉 To test with your own video:")
    print("   1. Place a video file named 'sample_video.mp4' in this directory")
    print("   2. Or edit the video_path variable in this script")
    print("   3. Run: python example.py")
    print("\n👉 For command-line usage:")
    print("   python cli.py your_video.mp4")
    print("\n👉 For web interface:")
    print("   streamlit run app.py")
    print("=" * 60)
