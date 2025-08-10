"""
Flask API for AI Video Content Summarizer
"""

from flask import Flask, request, jsonify, send_file
import os
import tempfile
from pathlib import Path
import uuid
import json
import time
import logging
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from src.main_processor import VideoSummarizer, process_video_file
from src.config import (
    MAX_VIDEO_SIZE_MB, 
    ALLOWED_EXTENSIONS, 
    OUTPUT_DIR,
    API_TIMEOUT
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_VIDEO_SIZE_MB * 1024 * 1024  # MB to bytes

# Store processing jobs in memory (use Redis/database for production)
processing_jobs = {}


@app.route('/', methods=['GET'])
def home():
    """API information endpoint"""
    return jsonify({
        "service": "AI Video Content Summarizer",
        "version": "1.0.0",
        "description": "Upload videos to get automatic transcription and summarization",
        "endpoints": {
            "POST /upload": "Upload and process video",
            "GET /status/<job_id>": "Get processing status",
            "GET /results/<job_id>": "Get processing results",
            "GET /download/<job_id>/<file_type>": "Download result files",
            "GET /health": "Health check"
        },
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": MAX_VIDEO_SIZE_MB
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        import torch
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "cuda_available": torch.cuda.is_available(),
            "device": "GPU" if torch.cuda.is_available() else "CPU"
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route('/upload', methods=['POST'])
def upload_video():
    """Upload and process video endpoint"""
    
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Validate file extension
    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {list(ALLOWED_EXTENSIONS)}"
        }), 400
    
    # Get processing parameters
    params = {
        'language': request.form.get('language'),
        'detect_scenes': request.form.get('detect_scenes', 'true').lower() == 'true',
        'create_highlights': request.form.get('create_highlights', 'true').lower() == 'true',
        'max_highlight_duration': float(request.form.get('max_highlight_duration', 60)),
        'whisper_model': request.form.get('whisper_model', 'base'),
        'summarization_model': request.form.get('summarization_model', 'facebook/bart-large-cnn'),
        'output_formats': request.form.get('output_formats', 'json,txt').split(',')
    }
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    temp_video_path = OUTPUT_DIR / f"{job_id}_{filename}"
    
    try:
        file.save(str(temp_video_path))
        
        # Initialize job status
        processing_jobs[job_id] = {
            'status': 'queued',
            'filename': filename,
            'started_at': time.time(),
            'params': params,
            'video_path': str(temp_video_path),
            'results': None,
            'error': None
        }
        
        # Start processing in background (in production, use Celery or similar)
        import threading
        thread = threading.Thread(target=process_video_async, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "job_id": job_id,
            "status": "queued",
            "message": "Video uploaded successfully and processing started",
            "estimated_time": "5-10 minutes depending on video length"
        }), 202
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get processing status for a job"""
    
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = processing_jobs[job_id]
    
    response = {
        "job_id": job_id,
        "status": job['status'],
        "filename": job['filename'],
        "started_at": job['started_at'],
        "elapsed_time": time.time() - job['started_at']
    }
    
    if job['status'] == 'completed':
        response['completed_at'] = job.get('completed_at')
        response['processing_time'] = job.get('processing_time', 0)
        response['results_available'] = True
    elif job['status'] == 'failed':
        response['error'] = job.get('error')
    elif job['status'] == 'processing':
        response['estimated_remaining'] = max(0, 600 - response['elapsed_time'])  # Rough estimate
    
    return jsonify(response)


@app.route('/results/<job_id>', methods=['GET'])
def get_results(job_id):
    """Get processing results for a completed job"""
    
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = processing_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({
            "error": "Job not completed yet",
            "current_status": job['status']
        }), 400
    
    if not job['results']:
        return jsonify({"error": "Results not available"}), 500
    
    # Return summary results (not the full detailed results)
    results = job['results']
    
    summary_response = {
        "job_id": job_id,
        "video_info": results['video_info'],
        "processing_time": results['processing_time'],
        "summary": {
            "text": results['summary']['overall_summary'],
            "key_points": results['summary']['overall_key_points'],
            "compression_ratio": results['summary']['overall_compression_ratio']
        },
        "transcription": {
            "language": results['transcription']['full_transcription']['language'],
            "text": results['transcription']['full_transcription']['text'],
            "confidence": results['transcription']['full_transcription']['confidence']
        },
        "scenes": {
            "count": len(results['scenes']),
            "summaries": [
                {
                    "scene_id": scene['scene_id'],
                    "start_time": scene['start_time'],
                    "end_time": scene['end_time'],
                    "summary": scene['summary'],
                    "key_points": scene['key_points']
                }
                for scene in results['summary']['scene_summaries']
            ]
        },
        "highlights": {
            "count": len(results['highlights']['timestamps']),
            "timestamps": results['highlights']['timestamps'],
            "total_duration": results['highlights']['duration'],
            "video_available": results['highlights']['video_path'] is not None
        },
        "metadata": results['metadata'],
        "available_downloads": {
            "json": f"/download/{job_id}/json",
            "txt": f"/download/{job_id}/txt",
            "srt": f"/download/{job_id}/srt" if 'srt' in job['params']['output_formats'] else None,
            "html": f"/download/{job_id}/html" if 'html' in job['params']['output_formats'] else None,
            "highlight_video": f"/download/{job_id}/highlights" if results['highlights']['video_path'] else None
        }
    }
    
    return jsonify(summary_response)


@app.route('/download/<job_id>/<file_type>', methods=['GET'])
def download_file(job_id, file_type):
    """Download result files"""
    
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = processing_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({
            "error": "Job not completed yet",
            "current_status": job['status']
        }), 400
    
    results = job['results']
    if not results:
        return jsonify({"error": "Results not available"}), 500
    
    try:
        if file_type == 'highlights':
            # Download highlight video
            if results['highlights']['video_path']:
                return send_file(
                    results['highlights']['video_path'],
                    as_attachment=True,
                    download_name=f"{job['filename']}_highlights.mp4"
                )
            else:
                return jsonify({"error": "Highlight video not available"}), 404
        
        else:
            # Download text files
            output_files = results.get('output_files', [])
            
            for file_path in output_files:
                file_path_obj = Path(file_path)
                if file_type in file_path_obj.name:
                    return send_file(
                        file_path,
                        as_attachment=True,
                        download_name=file_path_obj.name
                    )
            
            return jsonify({"error": f"File type '{file_type}' not found"}), 404
    
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return jsonify({"error": f"Download failed: {str(e)}"}), 500


@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all jobs (for debugging/monitoring)"""
    
    job_list = []
    for job_id, job in processing_jobs.items():
        job_summary = {
            "job_id": job_id,
            "filename": job['filename'],
            "status": job['status'],
            "started_at": job['started_at'],
            "elapsed_time": time.time() - job['started_at']
        }
        if job['status'] == 'completed':
            job_summary['completed_at'] = job.get('completed_at')
        job_list.append(job_summary)
    
    return jsonify({
        "total_jobs": len(job_list),
        "jobs": job_list
    })


@app.route('/cleanup/<job_id>', methods=['DELETE'])
def cleanup_job(job_id):
    """Clean up job files and data"""
    
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = processing_jobs[job_id]
    
    try:
        # Clean up video file
        video_path = Path(job['video_path'])
        if video_path.exists():
            video_path.unlink()
        
        # Clean up result files
        if job['results'] and job['results'].get('output_files'):
            for file_path in job['results']['output_files']:
                Path(file_path).unlink(missing_ok=True)
        
        # Clean up highlight video
        if job['results'] and job['results']['highlights']['video_path']:
            Path(job['results']['highlights']['video_path']).unlink(missing_ok=True)
        
        # Remove from job tracking
        del processing_jobs[job_id]
        
        return jsonify({"message": f"Job {job_id} cleaned up successfully"})
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        return jsonify({"error": f"Cleanup failed: {str(e)}"}), 500


def allowed_file(filename):
    """Check if file has allowed extension"""
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def process_video_async(job_id):
    """Process video asynchronously"""
    
    job = processing_jobs[job_id]
    job['status'] = 'processing'
    
    try:
        logger.info(f"Starting processing for job {job_id}")
        
        # Process the video
        results = process_video_file(
            job['video_path'],
            language=job['params']['language'],
            detect_scenes=job['params']['detect_scenes'],
            create_highlights=job['params']['create_highlights'],
            max_highlight_duration=job['params']['max_highlight_duration'],
            output_formats=job['params']['output_formats']
        )
        
        # Update job status
        job['status'] = 'completed'
        job['completed_at'] = time.time()
        job['processing_time'] = results['processing_time']
        job['results'] = results
        
        logger.info(f"Processing completed for job {job_id} in {results['processing_time']:.2f}s")
        
    except Exception as e:
        logger.error(f"Processing failed for job {job_id}: {str(e)}")
        job['status'] = 'failed'
        job['error'] = str(e)
        job['completed_at'] = time.time()
    
    finally:
        # Clean up original video file
        try:
            Path(job['video_path']).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to clean up video file: {str(e)}")


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        "error": f"File too large. Maximum size is {MAX_VIDEO_SIZE_MB}MB"
    }), 413


@app.errorhandler(Exception)
def handle_exception(e):
    """Handle unexpected errors"""
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500


if __name__ == '__main__':
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
