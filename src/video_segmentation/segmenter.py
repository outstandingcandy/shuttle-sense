"""
Video Segmenter
Segments a badminton video into clips based on detected hit points.
"""

import logging
import os
import json
import subprocess
from pathlib import Path
import cv2
from tqdm import tqdm

class VideoSegmenter:
    """Segments video into clips based on hit points."""
    
    def __init__(self, config):
        """
        Initialize the video segmenter.
        
        Args:
            config: Configuration dictionary for video segmentation
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.pre_hit_frames = config["pre_hit_frames"]
        self.post_hit_frames = config["post_hit_frames"]
    
    def segment_video(self, video_path, hit_points, output_dir):
        """
        Segment the video based on hit points.
        
        Args:
            video_path: Path to the input video file
            hit_points: List of hit point timestamps in seconds
            output_dir: Directory where segments will be saved
            
        Returns:
            Dictionary containing information about the segments
        """
        self.logger.info(f"Segmenting video: {video_path} into {len(hit_points)} segments")
        
        # Ensure output directory exists
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        cap.release()
        
        self.logger.info(f"Video properties: {width}x{height}, {fps} FPS, {duration:.2f} seconds")
        
        # Prepare segment info
        segments = []
        
        # Calculate pre and post hit time deltas in seconds
        pre_hit_delta = self.pre_hit_frames / fps
        post_hit_delta = self.post_hit_frames / fps
        
        # Create segments between consecutive hit points
        for i, hit_time in enumerate(hit_points[:-1]):
            next_hit_time = hit_points[i + 1]
            
            # Calculate start and end times
            start_time = max(0, hit_time - pre_hit_delta)
            end_time = min(duration, next_hit_time + post_hit_delta)
            
            # Skip if segment is too short
            if end_time - start_time < 0.5:
                self.logger.warning(f"Skipping too short segment between {hit_time:.2f}s and {next_hit_time:.2f}s")
                continue
            
            # Create segment file name
            segment_filename = f"segment_{i+1:03d}.mp4"
            output_path = os.path.join(output_dir, segment_filename)
            
            # Extract segment using FFmpeg
            self._extract_segment(video_path, start_time, end_time, output_path)
            
            # Add segment info
            segments.append({
                "id": i + 1,
                "filename": segment_filename,
                "start_time": start_time,
                "end_time": end_time,
                "hit_time": hit_time,
                "next_hit_time": next_hit_time,
                "duration": end_time - start_time
            })
        
        # Create a summary of segments
        segments_info = {
            "video": {
                "path": video_path,
                "width": width,
                "height": height,
                "fps": fps,
                "duration": duration,
                "total_frames": total_frames
            },
            "segments": segments
        }
        
        self.logger.info(f"Created {len(segments)} video segments")
        
        return segments_info
    
    def _extract_segment(self, video_path, start_time, end_time, output_path):
        """
        Extract a segment from the video using FFmpeg.
        
        Args:
            video_path: Path to the input video file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path where the segment will be saved
        """
        duration = end_time - start_time
        
        # Construct FFmpeg command
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ss", f"{start_time:.3f}",
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",  # Overwrite output file if exists
            output_path
        ]
        
        try:
            # Execute FFmpeg command
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=True
            )
            
            self.logger.debug(f"Created segment: {output_path}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error creating segment {output_path}: {e.stderr}")
            raise