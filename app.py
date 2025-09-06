"""
Flask web application for ShuttleSense video upload and analysis visualization
"""

import os
import json
import time
from pathlib import Path
from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import logging

# Import the existing pipeline and modules
from src.pipeline import ShuttleSensePipeline
from src.hit_detection.detector import HitPointDetector
from src.video_segmentation.segmenter import VideoSegmenter
from src.annotation.annotator import VideoAnnotator
import yaml

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
hit_detector = HitPointDetector(config['hit_detection'])
segmenter = VideoSegmenter(config['video_segmentation'])
annotator = VideoAnnotator(config['annotation'])
pipeline = ShuttleSensePipeline(hit_detector, segmenter, annotator, config)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page for video upload"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and processing"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a video file.'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing uploaded video: {filename}")
        
        # Process video through pipeline
        result_dir = pipeline.run(filepath)
        
        if result_dir is None:
            return jsonify({'error': 'Failed to process video'}), 500
        
        # Return result information
        result_info = {
            'session_id': result_dir.name,
            'video_filename': filename,
            'status': 'completed'
        }
        
        return jsonify(result_info)
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

@app.route('/results/<session_id>')
def view_results(session_id):
    """Display results page for a processed video"""
    try:
        session_dir = Path(config["paths"]["output_dir"]) / session_id
        
        if not session_dir.exists():
            return "Session not found", 404
        
        # Load session data
        segments_info_path = session_dir / "segments_info.json"
        annotations_path = session_dir / "annotations.json"
        hit_points_path = session_dir / "hit_points.json"
        
        session_data = {
            'session_id': session_id,
            'segments': [],
            'hit_points': [],
            'annotations': {}
        }
        
        # Load hit points
        if hit_points_path.exists():
            with open(hit_points_path, 'r') as f:
                hit_data = json.load(f)
                session_data['hit_points'] = hit_data.get('hit_timestamps', [])
        
        # Load segments info
        if segments_info_path.exists():
            with open(segments_info_path, 'r') as f:
                segments_data = json.load(f)
                session_data['segments'] = segments_data.get('segments', [])
        
        # Load annotations
        if annotations_path.exists():
            with open(annotations_path, 'r') as f:
                session_data['annotations'] = json.load(f)
        
        return render_template('results.html', data=session_data)
        
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        return f"Error loading results: {str(e)}", 500

@app.route('/video/<session_id>/<segment_name>')
def serve_video(session_id, segment_name):
    """Serve video segments"""
    try:
        session_dir = Path(config["paths"]["output_dir"]) / session_id
        video_path = session_dir / "segments" / segment_name
        
        if not video_path.exists():
            return "Video not found", 404
        
        return send_file(video_path)
        
    except Exception as e:
        logger.error(f"Error serving video: {str(e)}")
        return f"Error serving video: {str(e)}", 500

@app.route('/api/sessions')
def list_sessions():
    """API endpoint to list all processing sessions"""
    try:
        output_dir = Path(config["paths"]["output_dir"])
        sessions = []
        
        for session_dir in output_dir.iterdir():
            if session_dir.is_dir():
                session_info = {
                    'session_id': session_dir.name,
                    'created_at': session_dir.stat().st_mtime,
                    'has_segments': (session_dir / "segments").exists(),
                    'has_annotations': (session_dir / "annotations.json").exists()
                }
                sessions.append(session_info)
        
        # Sort by creation time, newest first
        sessions.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify(sessions)
        
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)