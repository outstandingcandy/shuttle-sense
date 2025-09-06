"""
Qwen-VL Integration Module
Integrate Qwen-VL LoRA model with existing ShuttleSense pipeline
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

# Import existing components
from src.hit_detection.enhanced_detector import EnhancedHitDetector
from src.qwen_vl_lora.inference_qwen_vl import QwenVLActionRecognizer

# Setup logging
logger = logging.getLogger(__name__)

class QwenVLEnhancedDetector:
    """Enhanced detector that integrates Qwen-VL with existing hit detection"""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 qwen_vl_model_path: str = None,
                 use_qwen_vl: bool = True):
        """
        Initialize enhanced detector with Qwen-VL integration
        
        Args:
            config: ShuttleSense configuration
            qwen_vl_model_path: Path to trained Qwen-VL LoRA model
            use_qwen_vl: Whether to use Qwen-VL for action recognition
        """
        self.config = config
        self.use_qwen_vl = use_qwen_vl
        
        # Initialize existing enhanced detector
        self.hit_detector = EnhancedHitDetector(config)
        
        # Initialize Qwen-VL recognizer if enabled
        self.qwen_vl_recognizer = None
        if self.use_qwen_vl and qwen_vl_model_path:
            self._initialize_qwen_vl(qwen_vl_model_path)
    
    def _initialize_qwen_vl(self, model_path: str):
        """Initialize Qwen-VL action recognizer"""
        try:
            if Path(model_path).exists():
                self.qwen_vl_recognizer = QwenVLActionRecognizer(
                    model_path=model_path,
                    config_path="configs/qwen_vl_lora_config.yaml"
                )
                logger.info("✅ Qwen-VL model initialized successfully")
            else:
                logger.warning(f"Qwen-VL model not found at {model_path}, using only VideoMAE")
                self.use_qwen_vl = False
        except Exception as e:
            logger.error(f"Failed to initialize Qwen-VL model: {str(e)}")
            self.use_qwen_vl = False
    
    def detect_hits(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Detect hits using existing pipeline
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of hit detection results
        """
        return self.hit_detector.detect_hits(video_path)
    
    def classify_action_videomae(self, video_segment: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Classify action using VideoMAE model
        
        Args:
            video_segment: Path to video segment or video frames array
            
        Returns:
            Action classification result
        """
        return self.hit_detector.classify_action(video_segment)
    
    def classify_action_qwen_vl(self, video_path: str, question: str = None) -> Dict[str, Any]:
        """
        Classify action using Qwen-VL model
        
        Args:
            video_path: Path to video segment
            question: Custom question for action recognition
            
        Returns:
            Action classification result from Qwen-VL
        """
        if not self.qwen_vl_recognizer:
            return {
                "action": "unknown",
                "confidence": 0.0,
                "method": "qwen_vl",
                "error": "Qwen-VL model not available"
            }
        
        try:
            result = self.qwen_vl_recognizer.recognize_action(video_path, question)
            
            return {
                "action": result.get("predicted_action", "unknown"),
                "confidence": result.get("confidence", 1.0),
                "method": "qwen_vl",
                "model_response": result.get("model_response", ""),
                "num_frames_processed": result.get("num_frames_processed", 0),
                "raw_result": result
            }
            
        except Exception as e:
            logger.error(f"Qwen-VL classification failed: {str(e)}")
            return {
                "action": "unknown",
                "confidence": 0.0,
                "method": "qwen_vl",
                "error": str(e)
            }
    
    def classify_action_ensemble(self, video_path: str, question: str = None) -> Dict[str, Any]:
        """
        Classify action using ensemble of VideoMAE and Qwen-VL
        
        Args:
            video_path: Path to video segment
            question: Custom question for Qwen-VL
            
        Returns:
            Ensemble classification result
        """
        results = {}
        
        # Get VideoMAE result
        videomae_result = self.classify_action_videomae(video_path)
        results["videomae"] = videomae_result
        
        # Get Qwen-VL result if available
        if self.use_qwen_vl and self.qwen_vl_recognizer:
            qwen_vl_result = self.classify_action_qwen_vl(video_path, question)
            results["qwen_vl"] = qwen_vl_result
        
        # Simple ensemble strategy: prefer Qwen-VL if available and confident
        if self.use_qwen_vl and "qwen_vl" in results and results["qwen_vl"]["confidence"] > 0.8:
            primary_result = results["qwen_vl"]
            primary_result["ensemble_method"] = "qwen_vl_primary"
        else:
            primary_result = results["videomae"]
            primary_result["ensemble_method"] = "videomae_primary"
        
        # Add all results for reference
        primary_result["all_results"] = results
        
        return primary_result
    
    def process_video_with_action_recognition(self, 
                                            video_path: str,
                                            use_ensemble: bool = True,
                                            save_segments: bool = True,
                                            output_dir: str = "output/segments") -> Dict[str, Any]:
        """
        Complete pipeline: detect hits and classify actions
        
        Args:
            video_path: Path to input video
            use_ensemble: Whether to use ensemble classification
            save_segments: Whether to save video segments
            output_dir: Directory to save segments
            
        Returns:
            Complete processing results
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.info(f"Processing video with Qwen-VL integration: {video_path}")
        
        # Step 1: Detect hits
        hits = self.detect_hits(str(video_path))
        
        if not hits:
            return {
                "video_path": str(video_path),
                "hits": [],
                "segments": [],
                "total_hits": 0,
                "processing_method": "qwen_vl_integrated"
            }
        
        # Step 2: Create segments and classify actions
        segments = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, hit in enumerate(hits):
            segment_info = self._create_and_classify_segment(
                video_path, hit, i, output_dir, use_ensemble, save_segments
            )
            segments.append(segment_info)
        
        result = {
            "video_path": str(video_path),
            "hits": hits,
            "segments": segments,
            "total_hits": len(hits),
            "total_segments": len(segments),
            "processing_method": "qwen_vl_integrated",
            "qwen_vl_enabled": self.use_qwen_vl,
            "ensemble_used": use_ensemble
        }
        
        logger.info(f"✅ Video processing completed: {len(hits)} hits, {len(segments)} segments")
        
        return result
    
    def _create_and_classify_segment(self, 
                                   video_path: Path,
                                   hit: Dict[str, Any],
                                   segment_index: int,
                                   output_dir: Path,
                                   use_ensemble: bool,
                                   save_segment: bool) -> Dict[str, Any]:
        """Create video segment and classify action"""
        
        # Extract segment parameters
        hit_frame = hit.get("frame_index", 0)
        pre_frames = self.config.get("video_segmentation", {}).get("pre_hit_frames", 15)
        post_frames = self.config.get("video_segmentation", {}).get("post_hit_frames", 30)
        
        start_frame = max(0, hit_frame - pre_frames)
        end_frame = hit_frame + post_frames
        
        segment_filename = f"{video_path.stem}_segment_{segment_index:03d}.mp4"
        segment_path = output_dir / segment_filename
        
        try:
            # Create video segment
            if save_segment:
                self._extract_video_segment(str(video_path), str(segment_path), start_frame, end_frame)
            
            # Classify action
            if use_ensemble:
                action_result = self.classify_action_ensemble(str(segment_path))
            else:
                # Use only Qwen-VL if available, otherwise VideoMAE
                if self.use_qwen_vl:
                    action_result = self.classify_action_qwen_vl(str(segment_path))
                else:
                    action_result = self.classify_action_videomae(str(segment_path))
            
            return {
                "segment_index": segment_index,
                "segment_path": str(segment_path) if save_segment else None,
                "hit_frame": hit_frame,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "hit_confidence": hit.get("confidence", 0.0),
                "action_classification": action_result,
                "timestamp": hit.get("timestamp", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Failed to process segment {segment_index}: {str(e)}")
            return {
                "segment_index": segment_index,
                "segment_path": None,
                "hit_frame": hit_frame,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "hit_confidence": hit.get("confidence", 0.0),
                "action_classification": {
                    "action": "processing_failed",
                    "confidence": 0.0,
                    "error": str(e)
                },
                "timestamp": hit.get("timestamp", 0.0)
            }
    
    def _extract_video_segment(self, input_path: str, output_path: str, start_frame: int, end_frame: int):
        """Extract video segment between specified frames"""
        
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Extract frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        
        while current_frame <= end_frame:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            out.write(frame)
            current_frame += 1
        
        cap.release()
        out.release()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "enhanced_detector": True,
            "videomae_available": self.hit_detector.use_custom_model,
            "qwen_vl_available": self.use_qwen_vl and self.qwen_vl_recognizer is not None
        }
        
        if info["videomae_available"]:
            info["videomae_model_path"] = self.config.get("hit_detection", {}).get("custom_model_path")
        
        if info["qwen_vl_available"]:
            info["qwen_vl_model_path"] = str(self.qwen_vl_recognizer.model_path)
        
        return info

def create_qwen_vl_enhanced_detector(config: Dict[str, Any], 
                                   qwen_vl_model_path: str = None) -> QwenVLEnhancedDetector:
    """
    Factory function to create Qwen-VL enhanced detector
    
    Args:
        config: ShuttleSense configuration
        qwen_vl_model_path: Path to Qwen-VL LoRA model
        
    Returns:
        QwenVLEnhancedDetector instance
    """
    
    # Default model path if not provided
    if not qwen_vl_model_path:
        qwen_vl_model_path = config.get("qwen_vl", {}).get("model_path", 
                                                           "models/checkpoints/qwen_vl_lora/final_model")
    
    # Check if Qwen-VL should be enabled
    use_qwen_vl = config.get("qwen_vl", {}).get("enabled", True)
    
    return QwenVLEnhancedDetector(
        config=config,
        qwen_vl_model_path=qwen_vl_model_path,
        use_qwen_vl=use_qwen_vl
    )