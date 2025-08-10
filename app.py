"""
Streamlit web application for AI Video Content Summarizer
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
import time
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from src.main_processor import VideoSummarizer
    from src.config import (
        MAX_VIDEO_SIZE_MB, 
        ALLOWED_EXTENSIONS, 
        WHISPER_MODEL, 
        SUMMARIZATION_MODEL,
        OUTPUT_DIR
    )
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.stop()


def main():
    st.set_page_config(
        page_title="AI Video Content Summarizer",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 AI Video Content Summarizer")
    st.markdown("""
    Upload a video file to automatically generate:
    - 📝 Text transcription using Whisper AI
    - 📊 Content summary using BART/T5
    - 🎬 Scene detection and analysis
    - ⭐ Highlight reel creation
    """)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Model selection
        st.subheader("AI Models")
        whisper_model = st.selectbox(
            "Whisper Model",
            ["tiny", "base", "small", "medium", "large"],
            index=1,  # Default to "base"
            help="Larger models are more accurate but slower"
        )
        
        summarization_model = st.selectbox(
            "Summarization Model",
            ["facebook/bart-large-cnn", "t5-small", "t5-base"],
            index=0,
            help="BART is generally better for summarization"
        )
        
        # Processing options
        st.subheader("Processing Options")
        language = st.selectbox(
            "Language (auto-detect if not specified)",
            ["Auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
            index=0
        )
        
        detect_scenes = st.checkbox("Enable scene detection", value=True)
        create_highlights = st.checkbox("Create highlight reel", value=True)
        
        max_highlight_duration = st.slider(
            "Max highlight duration (seconds)",
            30, 300, 60
        )
        
        # Output formats
        st.subheader("Output Formats")
        output_formats = []
        if st.checkbox("JSON", value=True):
            output_formats.append("json")
        if st.checkbox("Text Summary", value=True):
            output_formats.append("txt")
        if st.checkbox("SRT Subtitles", value=False):
            output_formats.append("srt")
        if st.checkbox("HTML Report", value=False):
            output_formats.append("html")
        
        if not output_formats:
            output_formats = ["json", "txt"]
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Video")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=list(ALLOWED_EXTENSIONS),
            help=f"Maximum file size: {MAX_VIDEO_SIZE_MB}MB"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.info(f"File: {uploaded_file.name}")
            st.info(f"Size: {uploaded_file.size / 1024 / 1024:.1f} MB")
            
            # Check file size
            if uploaded_file.size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
                st.error(f"File too large! Maximum size is {MAX_VIDEO_SIZE_MB}MB")
                return
            
            # Process button
            if st.button("🚀 Process Video", type="primary"):
                process_video(
                    uploaded_file, 
                    whisper_model, 
                    summarization_model,
                    language if language != "Auto" else None,
                    detect_scenes,
                    create_highlights,
                    max_highlight_duration,
                    output_formats
                )
    
    with col2:
        st.header("System Info")
        
        # Display system information
        import torch
        
        system_info = {
            "CUDA Available": torch.cuda.is_available(),
            "Device": "GPU" if torch.cuda.is_available() else "CPU",
            "Default Whisper Model": WHISPER_MODEL,
            "Default Summary Model": SUMMARIZATION_MODEL.split("/")[-1]
        }
        
        for key, value in system_info.items():
            st.metric(key, value)
        
        # Model status
        st.subheader("Current Settings")
        st.json({
            "Whisper Model": whisper_model,
            "Summary Model": summarization_model.split("/")[-1],
            "Language": language,
            "Scene Detection": detect_scenes,
            "Create Highlights": create_highlights,
            "Output Formats": output_formats
        })


def process_video(uploaded_file, whisper_model, summarization_model, language, 
                 detect_scenes, create_highlights, max_highlight_duration, 
                 output_formats):
    """Process the uploaded video file"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_video_path = tmp_file.name
    
    try:
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize processor
        status_text.text("Initializing AI models...")
        progress_bar.progress(10)
        
        summarizer = VideoSummarizer(
            whisper_model=whisper_model,
            summarization_model=summarization_model
        )
        
        # Process video with progress updates
        status_text.text("Loading video...")
        progress_bar.progress(20)
        
        start_time = time.time()
        
        # Custom processing with progress callbacks would be ideal,
        # but for now we'll do it in one go
        status_text.text("Processing video (this may take several minutes)...")
        progress_bar.progress(30)
        
        results = summarizer.process_video(
            temp_video_path,
            language=language,
            detect_scenes=detect_scenes,
            create_highlights=create_highlights,
            max_highlight_duration=max_highlight_duration,
            output_formats=output_formats
        )
        
        progress_bar.progress(100)
        status_text.text("Processing completed!")
        
        # Display results
        display_results(results)
        
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        logger.error(f"Processing error: {str(e)}", exc_info=True)
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_video_path)
        except:
            pass


def display_results(results):
    """Display processing results in the UI"""
    
    st.success("✅ Video processing completed!")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Summary", 
        "📝 Transcript", 
        "🎬 Scenes", 
        "⭐ Highlights", 
        "📁 Files"
    ])
    
    with tab1:
        st.header("Video Summary")
        
        # Video info
        col1, col2, col3, col4 = st.columns(4)
        video_info = results['video_info']
        
        with col1:
            st.metric("Duration", f"{video_info['duration']:.1f}s")
        with col2:
            st.metric("FPS", f"{video_info['fps']:.1f}")
        with col3:
            st.metric("Language", results['metadata']['language_detected'])
        with col4:
            st.metric("Scenes", len(results['scenes']))
        
        # Overall summary
        st.subheader("Overall Summary")
        st.write(results['summary']['overall_summary'])
        
        # Key points
        if results['summary']['overall_key_points']:
            st.subheader("Key Points")
            for i, point in enumerate(results['summary']['overall_key_points'], 1):
                st.write(f"{i}. {point}")
        
        # Processing stats
        st.subheader("Processing Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processing Time", f"{results['processing_time']:.1f}s")
        with col2:
            st.metric("Compression Ratio", f"{results['summary']['overall_compression_ratio']:.2f}")
        with col3:
            st.metric("Word Count", len(results['transcription']['full_transcription']['text'].split()))
    
    with tab2:
        st.header("Full Transcript")
        
        # Language and confidence info
        transcription = results['transcription']['full_transcription']
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Language", transcription['language'])
        with col2:
            st.metric("Confidence", f"{transcription['confidence']:.2f}")
        
        # Full text
        st.text_area(
            "Transcript",
            value=transcription['text'],
            height=400,
            disabled=True
        )
        
        # Segments with timestamps
        if st.checkbox("Show detailed segments"):
            st.subheader("Timestamped Segments")
            for segment in transcription['segments']:
                with st.expander(f"{segment['start']:.1f}s - {segment['end']:.1f}s"):
                    st.write(segment['text'])
                    if segment.get('words'):
                        st.json(segment['words'][:5])  # Show first 5 words with timing
    
    with tab3:
        if results['scenes']:
            st.header("Scene Analysis")
            
            # Scene summaries
            for scene in results['summary']['scene_summaries']:
                with st.expander(
                    f"Scene {scene['scene_id']}: {scene['start_time']:.1f}s - {scene['end_time']:.1f}s"
                ):
                    st.write("**Summary:**")
                    st.write(scene['summary'])
                    
                    if scene['key_points']:
                        st.write("**Key Points:**")
                        for point in scene['key_points']:
                            st.write(f"• {point}")
                    
                    # Scene stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Duration", f"{scene['end_time'] - scene['start_time']:.1f}s")
                    with col2:
                        st.metric("Compression", f"{scene.get('compression_ratio', 0):.2f}")
        else:
            st.info("Scene detection was not enabled or no scenes were detected.")
    
    with tab4:
        if results['highlights']['timestamps']:
            st.header("Video Highlights")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Highlights", len(results['highlights']['timestamps']))
            with col2:
                st.metric("Total Duration", f"{results['highlights']['duration']:.1f}s")
            
            # Highlight segments
            for i, (start, end) in enumerate(results['highlights']['timestamps'], 1):
                st.write(f"**Highlight {i}:** {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")
            
            # Highlight video file
            if results['highlights']['video_path']:
                st.success(f"Highlight reel created: {Path(results['highlights']['video_path']).name}")
                
                # Provide download button if file exists
                highlight_path = Path(results['highlights']['video_path'])
                if highlight_path.exists():
                    with open(highlight_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Highlight Reel",
                            data=f.read(),
                            file_name=highlight_path.name,
                            mime="video/mp4"
                        )
        else:
            st.info("No highlights were generated.")
    
    with tab5:
        st.header("Generated Files")
        
        if results['output_files']:
            for file_path in results['output_files']:
                file_path_obj = Path(file_path)
                
                if file_path_obj.exists():
                    # File info
                    st.write(f"**{file_path_obj.name}**")
                    st.write(f"Size: {file_path_obj.stat().st_size / 1024:.1f} KB")
                    
                    # Download button
                    with open(file_path_obj, 'rb') as f:
                        mime_type = "application/json" if file_path_obj.suffix == ".json" else "text/plain"
                        st.download_button(
                            label=f"📥 Download {file_path_obj.suffix.upper()}",
                            data=f.read(),
                            file_name=file_path_obj.name,
                            mime=mime_type,
                            key=f"download_{file_path_obj.name}"
                        )
                    
                    st.divider()
        else:
            st.info("No output files were generated.")
        
        # Raw JSON results
        if st.checkbox("Show raw JSON results"):
            st.json(results, expanded=False)


if __name__ == "__main__":
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Run the app
    main()
