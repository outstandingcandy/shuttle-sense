"""
Hit Point Detector
Detects hit points in badminton videos using VideoMAE model.
"""

import logging
import os
import numpy as np
import torch
from torch import nn
import cv2
from transformers import VideoMAEModel, VideoMAEFeatureExtractor, VideoMAEForVideoClassification
from tqdm import tqdm

class HitPointDetector:
    """Detects badminton hit points using the VideoMAE model."""
    
    def __init__(self, config):
        """
        Initialize the hit point detector.
        
        Args:
            config: Configuration for the hit detector
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model_id = config["model_id"]
        self.confidence_threshold = config["confidence_threshold"]
        self.sample_rate = config["sample_rate"]
        self.window_size = config["window_size"]
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize the VideoMAE model and feature extractor."""
        self.logger.info(f"Loading VideoMAE model: {self.model_id}")
        
        try:
            # Load the feature extractor and model
            self.feature_extractor = VideoMAEFeatureExtractor.from_pretrained(self.model_id)
            self.model = VideoMAEForVideoClassification.from_pretrained(self.model_id)
            
            # Add classification head for hit detection
            self.classifier = nn.Linear(self.model.config.hidden_size, 2)  # 2 classes: hit/no-hit
            
            # Set model to evaluation mode
            self.model.eval()
            self.classifier.eval()
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")
            self.model.to(self.device)
            self.classifier.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _extract_frames(self, video_path):
        """
        Extract frames from video at the specified sample rate.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (frames, frame_indices, fps)
        """
        self.logger.info(f"Extracting frames from video: {video_path}")
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.logger.info(f"Video stats: {total_frames} frames, {fps} FPS")
        
        # Extract frames at the specified sample rate
        frames = []
        frame_indices = []
        
        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % self.sample_rate == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    frame_indices.append(frame_idx)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        self.logger.info(f"Extracted {len(frames)} frames")
        
        return frames, frame_indices, fps
    
    def _process_window(self, frames_window):
        """
        Process a window of frames with the VideoMAE model.
        
        Args:
            frames_window: List of frames to process
            
        Returns:
            Probability of hit in the center frame
        """
        # Prepare inputs
        inputs = self.feature_extractor(frames_window, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Return predicted label
        predicted_label = outputs.logits.argmax(-1).item()
        return predicted_label
    
    def detect(self, video_path):
        """
        Detect hit points in the video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of timestamps (in seconds) where hits were detected
        """
        self.logger.info(f"Detecting hit points in: {video_path}")
        
        # Extract frames
        frames, frame_indices, fps = self._extract_frames(video_path)
        
        # Process frames in sliding windows
        half_window = self.window_size // 2
        hit_points = []
        predicted_labels = []
        
        with tqdm(total=len(frames), desc="Detecting hits") as pbar:
            for i in range(half_window, len(frames) - half_window):
                window_start = i - half_window
                window_end = i + half_window
                frames_window = frames[window_start:window_end]
                
                # Process window
                predicted_label = self._process_window(frames_window)
                predicted_labels.append(predicted_label)
                
                # If probability exceeds threshold, mark as hit point
                if predicted_label > 0:
                    frame_idx = frame_indices[i]
                    timestamp = frame_idx / fps
                    hit_points.append(timestamp)
                
                pbar.update(1)
        
        # Post-process hit points (remove duplicates close in time)
        hit_points = self._post_process_hits(hit_points, fps)
        
        self.logger.info(f"Detected {len(hit_points)} hit points")
        return hit_points
    
    def _post_process_hits(self, hit_points, fps):
        """
        Post-process hit points to remove duplicates that are close in time.
        
        Args:
            hit_points: List of hit point timestamps
            fps: Frames per second of the video
            
        Returns:
            Filtered list of hit points
        """
        if not hit_points:
            return []
        
        # Sort hit points by time
        hit_points = sorted(hit_points)
        
        # Minimum time between consecutive hits (in seconds)
        min_hit_interval = 0.5  # Assume at least 0.5s between consecutive hits
        
        # Filter hit points
        filtered_hits = [hit_points[0]]
        for hit in hit_points[1:]:
            if hit - filtered_hits[-1] >= min_hit_interval:
                filtered_hits.append(hit)
        
        return filtered_hits