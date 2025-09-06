"""
Enhanced Hit Point Detector with Action Classification
Combines hit detection with trained action classification model
"""

import json
import logging
import os
import numpy as np
import torch
from torch import nn
import cv2
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from tqdm import tqdm
from pathlib import Path

class EnhancedHitPointDetector:
    """Enhanced hit detector with action classification capability."""
    
    def __init__(self, config):
        """
        Initialize the enhanced hit point detector.
        
        Args:
            config: Configuration for the hit detector
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Original hit detection parameters
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.sample_rate = config.get("sample_rate", 2)
        self.window_size = config.get("window_size", 16)
        
        # Check if we should use custom trained model or fallback to original
        self.use_custom_model = config.get("use_custom_model", True)
        self.custom_model_path = config.get("custom_model_path", "models/checkpoints/action_classification/best_model")
        
        # Initialize models
        self._init_models()
        
    def _init_models(self):
        """Initialize the models (custom trained or fallback)."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        if self.use_custom_model and Path(self.custom_model_path).exists():
            self._load_custom_model()
        else:
            self._load_fallback_model()
    
    def _load_custom_model(self):
        """Load the custom trained action classification model."""
        try:
            self.logger.info(f"Loading custom trained model from {self.custom_model_path}")
            
            # Load class mapping
            class_mapping_path = Path(self.custom_model_path) / "class_mapping.json"
            with open(class_mapping_path, 'r') as f:
                self.class_mapping = json.load(f)
            
            self.action_classes = self.class_mapping['action_classes']
            self.num_classes = len(self.action_classes)
            
            # Load model and processor
            self.processor = VideoMAEImageProcessor.from_pretrained(self.custom_model_path)
            self.model = VideoMAEForVideoClassification.from_pretrained(self.custom_model_path)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Successfully loaded custom model with {self.num_classes} action classes")
            self.model_type = "action_classification"
            
        except Exception as e:
            self.logger.error(f"Failed to load custom model: {e}")
            self.logger.info("Falling back to original hit detection model")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load the original hit detection model as fallback."""
        try:
            # Use the original model configuration
            original_model_id = self.config.get("model_id", "yexter/videomae-base-Badminton_strokes-finetuned-stroke-classification")
            
            self.logger.info(f"Loading fallback model: {original_model_id}")
            
            self.processor = VideoMAEImageProcessor.from_pretrained(original_model_id)
            self.model = VideoMAEForVideoClassification.from_pretrained(original_model_id)
            
            self.model.to(self.device)
            self.model.eval()
            
            # For original model, we treat any prediction > threshold as a hit
            self.model_type = "binary_hit_detection"
            self.logger.info("Successfully loaded fallback hit detection model")
            
        except Exception as e:
            self.logger.error(f"Error loading fallback model: {str(e)}")
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
        Process a window of frames with the model.
        
        Args:
            frames_window: List of frames to process
            
        Returns:
            Dictionary with prediction results
        """
        # Prepare inputs
        inputs = self.processor(frames_window, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        if self.model_type == "action_classification":
            # For action classification model
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = self.action_classes[predicted_class_idx]
            confidence = probabilities[0, predicted_class_idx].item()
            
            return {
                'is_hit': confidence > 0,
                'hit_confidence': confidence,
                'action_class': predicted_class,
                'action_confidence': confidence,
                'all_probabilities': probabilities[0].cpu().numpy()
            }
        
        else:
            # For binary hit detection model
            predicted_label = logits.argmax(-1).item()
            hit_confidence = probabilities[0, predicted_label].item() if predicted_label > 0 else 0.0
            
            return {
                'is_hit': predicted_label > 0,
                'hit_confidence': hit_confidence,
                'action_class': 'Hit' if predicted_label > 0 else 'No Hit',
                'action_confidence': hit_confidence,
                'all_probabilities': probabilities[0].cpu().numpy()
            }
    
    def detect(self, video_path):
        """
        Detect hit points and classify actions in the video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing hit points and action classifications
        """
        self.logger.info(f"Processing video: {video_path}")
        
        # Extract frames
        frames, frame_indices, fps = self._extract_frames(video_path)
        
        # Process frames in sliding windows
        half_window = self.window_size // 2
        results = []
        
        with tqdm(total=len(frames), desc="Analyzing video") as pbar:
            for i in range(half_window, len(frames) - half_window):
                window_start = i - half_window
                window_end = i + half_window
                frames_window = frames[window_start:window_end]
                
                # Process window
                result = self._process_window(frames_window)
                
                # Add temporal information
                frame_idx = frame_indices[i]
                timestamp = frame_idx / fps
                
                result.update({
                    'frame_idx': frame_idx,
                    'timestamp': timestamp
                })
                
                results.append(result)
                pbar.update(1)
        
        # Post-process to extract hit points
        hit_points = self._extract_hit_points(results, fps)
        
        # Compile final results
        final_results = {
            'hit_points': hit_points,
            'detailed_results': results,
            'video_info': {
                'path': video_path,
                'fps': fps,
                'total_frames': len(frame_indices),
                'duration': len(frame_indices) / fps if fps > 0 else 0
            },
            'model_info': {
                'type': self.model_type,
                'custom_model': self.use_custom_model,
                'model_path': self.custom_model_path if self.use_custom_model else None
            }
        }
        
        self.logger.info(f"Analysis complete. Found {len(hit_points)} hit points")
        return final_results
    
    def _extract_hit_points(self, results, fps):
        """
        Extract hit points from the detailed results.
        
        Args:
            results: List of detailed analysis results
            fps: Video frames per second
            
        Returns:
            List of hit point dictionaries
        """
        hit_points = []
        
        # Find peaks in hit confidence
        for i, result in enumerate(results):
            if not result['is_hit'] or result['hit_confidence'] < self.confidence_threshold:
                continue
            
            # Check if this is a local maximum
            is_peak = True
            window_size = 5  # Check 5 frames around
            
            for j in range(max(0, i - window_size), min(len(results), i + window_size + 1)):
                if j != i and results[j]['hit_confidence'] > result['hit_confidence']:
                    is_peak = False
                    break
            
            if is_peak:
                hit_point = {
                    'timestamp': result['timestamp'],
                    'frame_idx': result['frame_idx'],
                    'confidence': result['hit_confidence'],
                    'action_class': result['action_class'],
                    'action_confidence': result['action_confidence']
                }
                hit_points.append(hit_point)
        
        # Remove hits that are too close together
        filtered_hits = self._filter_close_hits(hit_points)
        
        return filtered_hits
    
    def _filter_close_hits(self, hit_points, min_interval=0.5):
        """
        Filter out hit points that are too close in time.
        
        Args:
            hit_points: List of hit point dictionaries
            min_interval: Minimum time interval between hits (seconds)
            
        Returns:
            Filtered list of hit points
        """
        if not hit_points:
            return []
        
        # Sort by timestamp
        sorted_hits = sorted(hit_points, key=lambda x: x['timestamp'])
        
        # Filter
        filtered = [sorted_hits[0]]
        for hit in sorted_hits[1:]:
            if hit['timestamp'] - filtered[-1]['timestamp'] >= min_interval:
                filtered.append(hit)
        
        return filtered
    
    def get_model_info(self):
        """Get information about the loaded model."""
        info = {
            'model_type': self.model_type,
            'device': str(self.device),
            'use_custom_model': self.use_custom_model
        }
        
        if self.model_type == "action_classification":
            info.update({
                'num_classes': self.num_classes,
                'action_classes': self.action_classes,
                'model_path': self.custom_model_path
            })
        
        return info