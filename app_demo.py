"""
Flask web application for ShuttleSense video upload and analysis visualization (Demo Version)
"""

import os
import json
import time
from pathlib import Path
from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import logging
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
    """Handle video upload and create mock results for demo"""
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
        
        # Create mock results for demo
        session_id = f"{Path(filename).stem}_{timestamp}"
        result_dir = Path(config["paths"]["output_dir"]) / session_id
        result_dir.mkdir(exist_ok=True, parents=True)
        
        # Create mock data
        mock_hit_points = [5.2, 12.8, 18.3, 25.1, 31.7]
        mock_segments = [
            {"filename": "segment_1.mp4", "start_time": 0.0, "end_time": 8.8},
            {"filename": "segment_2.mp4", "start_time": 8.8, "end_time": 15.6},
            {"filename": "segment_3.mp4", "start_time": 15.6, "end_time": 21.7},
            {"filename": "segment_4.mp4", "start_time": 21.7, "end_time": 28.4},
            {"filename": "segment_5.mp4", "start_time": 28.4, "end_time": 35.0}
        ]
        
        mock_annotations = {
            "segment_1.mp4": {
                "foundational_actions": ["正手高远球", "后场移动", "击球准备动作"],
                "tactical_semantics": ["防守反击", "拉开对方", "争取主动"],
                "brief_evaluation": "击球技术标准，但移动速度稍慢，建议加强步法训练"
            },
            "segment_2.mp4": {
                "foundational_actions": ["反手挑球", "网前步法", "快速回位"],
                "tactical_semantics": ["被动防守", "化解压力", "寻找反击机会"],
                "brief_evaluation": "反手技术需要改进，击球点偏低，影响回球质量"
            },
            "segment_3.mp4": {
                "foundational_actions": ["正手杀球", "起跳动作", "落地缓冲"],
                "tactical_semantics": ["进攻得分", "压迫对手", "主动进攻"],
                "brief_evaluation": "杀球动作流畅有力，角度刁钻，是很好的得分机会"
            },
            "segment_4.mp4": {
                "foundational_actions": ["网前搓球", "手腕控制", "身体平衡"],
                "tactical_semantics": ["技巧性处理", "改变节奏", "调动对手"],
                "brief_evaluation": "网前技术细腻，但可以更加贴网，增加对手难度"
            },
            "segment_5.mp4": {
                "foundational_actions": ["反手平抽", "侧身击球", "连续对抽"],
                "tactical_semantics": ["平衡局面", "控制中场", "寻找破绽"],
                "brief_evaluation": "平抽节奏掌握良好，但需要更多变化来打破僵局"
            }
        }
        
        # Save mock data
        with open(result_dir / "hit_points.json", 'w', encoding='utf-8') as f:
            json.dump({"hit_timestamps": mock_hit_points}, f, indent=2, ensure_ascii=False)
        
        with open(result_dir / "segments_info.json", 'w', encoding='utf-8') as f:
            json.dump({"segments": mock_segments}, f, indent=2, ensure_ascii=False)
        
        with open(result_dir / "annotations.json", 'w', encoding='utf-8') as f:
            json.dump(mock_annotations, f, indent=2, ensure_ascii=False)
        
        # Return result information
        result_info = {
            'session_id': session_id,
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
            with open(hit_points_path, 'r', encoding='utf-8') as f:
                hit_data = json.load(f)
                session_data['hit_points'] = hit_data.get('hit_timestamps', [])
        
        # Load segments info
        if segments_info_path.exists():
            with open(segments_info_path, 'r', encoding='utf-8') as f:
                segments_data = json.load(f)
                session_data['segments'] = segments_data.get('segments', [])
        
        # Load annotations
        if annotations_path.exists():
            with open(annotations_path, 'r', encoding='utf-8') as f:
                session_data['annotations'] = json.load(f)
        
        return render_template('results.html', data=session_data)
        
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        return f"Error loading results: {str(e)}", 500

@app.route('/video/<session_id>/<segment_name>')
def serve_video(session_id, segment_name):
    """Serve original uploaded video for demo (since we don't actually segment)"""
    try:
        # For demo purposes, we'll serve the original uploaded video
        # In real implementation, this would serve the actual segments
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        
        # Find the original video file for this session
        for video_file in upload_dir.glob(f"*{session_id.split('_')[-1]}*"):
            return send_file(video_file)
        
        return "Video not found", 404
        
    except Exception as e:
        logger.error(f"Error serving video: {str(e)}")
        return f"Error serving video: {str(e)}", 500

@app.route('/api/sessions')
def list_sessions():
    """API endpoint to list all processing sessions"""
    try:
        output_dir = Path(config["paths"]["output_dir"])
        if not output_dir.exists():
            return jsonify([])
            
        sessions = []
        
        for session_dir in output_dir.iterdir():
            if session_dir.is_dir():
                session_info = {
                    'session_id': session_dir.name,
                    'created_at': session_dir.stat().st_mtime,
                    'has_segments': (session_dir / "segments_info.json").exists(),
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