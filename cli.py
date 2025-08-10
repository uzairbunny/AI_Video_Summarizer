"""
Command-line interface for AI Video Content Summarizer
"""

import argparse
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from src.main_processor import process_video_file
from src.config import (
    WHISPER_MODEL, 
    SUMMARIZATION_MODEL,
    MAX_SUMMARY_LENGTH,
    MIN_SUMMARY_LENGTH
)


def main():
    parser = argparse.ArgumentParser(
        description='AI Video Content Summarizer - Generate transcriptions and summaries from videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4                                    # Basic processing
  %(prog)s video.mp4 --language en --no-scenes         # English only, no scene detection
  %(prog)s video.mp4 --highlights --max-highlight 120  # Create 2-minute highlights
  %(prog)s video.mp4 --format json txt srt            # Multiple output formats
  %(prog)s video.mp4 --whisper-model large            # Use large Whisper model
        """
    )
    
    # Required arguments
    parser.add_argument(
        'video_path',
        help='Path to the video file to process'
    )
    
    # Model selection
    parser.add_argument(
        '--whisper-model',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        default=WHISPER_MODEL,
        help=f'Whisper model to use for transcription (default: {WHISPER_MODEL})'
    )
    
    parser.add_argument(
        '--summarization-model',
        default=SUMMARIZATION_MODEL,
        help=f'Model for summarization (default: {SUMMARIZATION_MODEL})'
    )
    
    # Processing options
    parser.add_argument(
        '--language',
        help='Language code for transcription (e.g., en, es, fr). Auto-detect if not specified'
    )
    
    parser.add_argument(
        '--no-scenes',
        action='store_true',
        help='Disable scene detection'
    )
    
    parser.add_argument(
        '--no-highlights',
        action='store_true',
        help='Disable highlight reel creation'
    )
    
    parser.add_argument(
        '--max-highlight',
        type=float,
        default=60.0,
        help='Maximum duration for highlight reel in seconds (default: 60)'
    )
    
    # Summary options
    parser.add_argument(
        '--max-summary-length',
        type=int,
        default=MAX_SUMMARY_LENGTH,
        help=f'Maximum length of summary (default: {MAX_SUMMARY_LENGTH})'
    )
    
    parser.add_argument(
        '--min-summary-length',
        type=int,
        default=MIN_SUMMARY_LENGTH,
        help=f'Minimum length of summary (default: {MIN_SUMMARY_LENGTH})'
    )
    
    # Output options
    parser.add_argument(
        '--format',
        nargs='+',
        choices=['json', 'txt', 'srt', 'html'],
        default=['json', 'txt'],
        help='Output formats (default: json txt)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for results (default: ./outputs/)'
    )
    
    # Debug options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Validate input file
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    if not video_path.is_file():
        print(f"Error: Path is not a file: {video_path}")
        sys.exit(1)
    
    # Check file extension
    from src.config import SUPPORTED_FORMATS
    if video_path.suffix.lower() not in SUPPORTED_FORMATS:
        print(f"Error: Unsupported video format: {video_path.suffix}")
        print(f"Supported formats: {SUPPORTED_FORMATS}")
        sys.exit(1)
    
    # Prepare processing parameters
    processing_params = {
        'language': args.language,
        'detect_scenes': not args.no_scenes,
        'create_highlights': not args.no_highlights,
        'max_highlight_duration': args.max_highlight,
        'output_formats': args.format
    }
    
    # Display configuration
    print("🎥 AI Video Content Summarizer")
    print("=" * 50)
    print(f"Video: {video_path.name}")
    print(f"Size: {video_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Whisper Model: {args.whisper_model}")
    print(f"Summarization Model: {args.summarization_model.split('/')[-1]}")
    print(f"Language: {args.language or 'Auto-detect'}")
    print(f"Scene Detection: {'Yes' if processing_params['detect_scenes'] else 'No'}")
    print(f"Create Highlights: {'Yes' if processing_params['create_highlights'] else 'No'}")
    print(f"Output Formats: {', '.join(args.format)}")
    print()
    
    # Process the video
    try:
        print("Starting video processing...")
        print("This may take several minutes depending on video length and selected models.")
        print()
        
        results = process_video_file(
            str(video_path),
            **processing_params
        )
        
        # Display results summary
        print_results_summary(results)
        
        # List output files
        if results.get('output_files'):
            print("\n📁 Generated Files:")
            print("-" * 30)
            for file_path in results['output_files']:
                file_obj = Path(file_path)
                if file_obj.exists():
                    size_kb = file_obj.stat().st_size / 1024
                    print(f"  {file_obj.name} ({size_kb:.1f} KB)")
        
        # Highlight video
        if results['highlights']['video_path']:
            highlight_path = Path(results['highlights']['video_path'])
            if highlight_path.exists():
                size_mb = highlight_path.stat().st_size / 1024 / 1024
                print(f"  {highlight_path.name} ({size_mb:.1f} MB)")
        
        print(f"\n✅ Processing completed successfully!")
        print(f"⏱️  Total time: {results['processing_time']:.1f} seconds")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def print_results_summary(results):
    """Print a summary of processing results"""
    
    print("\n📊 Processing Results")
    print("=" * 50)
    
    # Video info
    video_info = results['video_info']
    print(f"Duration: {video_info['duration']:.1f} seconds ({video_info['duration']/60:.1f} minutes)")
    print(f"Language: {results['metadata']['language_detected']}")
    print(f"Scenes detected: {len(results['scenes'])}")
    
    # Summary stats
    summary = results['summary']
    transcription = results['transcription']['full_transcription']
    
    print(f"Original words: {len(transcription['text'].split())}")
    print(f"Summary words: {len(summary['overall_summary'].split())}")
    print(f"Compression ratio: {summary['overall_compression_ratio']:.2f}")
    
    # Overall summary
    print(f"\n📝 Summary:")
    print("-" * 30)
    print(summary['overall_summary'])
    
    # Key points
    if summary['overall_key_points']:
        print(f"\n🔑 Key Points:")
        print("-" * 30)
        for i, point in enumerate(summary['overall_key_points'][:5], 1):
            print(f"{i}. {point}")
        
        if len(summary['overall_key_points']) > 5:
            print(f"   ... and {len(summary['overall_key_points']) - 5} more")
    
    # Highlights
    if results['highlights']['timestamps']:
        print(f"\n⭐ Highlights ({len(results['highlights']['timestamps'])} segments):")
        print("-" * 30)
        total_highlight_duration = 0
        for i, (start, end) in enumerate(results['highlights']['timestamps'], 1):
            duration = end - start
            total_highlight_duration += duration
            print(f"{i}. {start:.1f}s - {end:.1f}s ({duration:.1f}s)")
        
        print(f"Total highlight duration: {total_highlight_duration:.1f}s")
    
    # Scene summaries (show first few)
    if summary['scene_summaries']:
        print(f"\n🎬 Scene Breakdown (showing first 3 of {len(summary['scene_summaries'])}):")
        print("-" * 30)
        
        for scene in summary['scene_summaries'][:3]:
            duration = scene['end_time'] - scene['start_time']
            print(f"Scene {scene['scene_id']} ({scene['start_time']:.1f}s-{scene['end_time']:.1f}s, {duration:.1f}s):")
            print(f"  {scene['summary'][:100]}{'...' if len(scene['summary']) > 100 else ''}")
            print()


if __name__ == '__main__':
    main()
