"""
ShuttleSense Pipeline
Coordinates the workflow between different modules in the ShuttleSense system.
"""

import json
import logging
import os
from pathlib import Path
import time

class ShuttleSensePipeline:
    """Main pipeline for ShuttleSense that orchestrates the entire workflow."""
    
    def __init__(self, hit_detector, segmenter, annotator, config):
        """
        Initialize the pipeline with its components.
        
        Args:
            hit_detector: The HitPointDetector instance
            segmenter: The VideoSegmenter instance
            annotator: The VideoAnnotator instance
            config: Configuration dictionary
        """
        self.hit_detector = hit_detector
        self.segmenter = segmenter
        self.annotator = annotator
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set up output paths
        self.output_dir = Path(config["paths"]["output_dir"])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def run(self, video_path):
        """
        Run the complete pipeline on the input video.
        
        Args:
            video_path: Path to the input video file
        
        Returns:
            Path to the directory containing all outputs
        """
        video_name = Path(video_path).stem
        self.logger.info(f"Processing video: {video_name}")
        
        # Create session directory
        session_dir = self.output_dir / f"{video_name}_{int(time.time())}"
        session_dir.mkdir(exist_ok=True)
        
        # Step 1: Hit Point Detection
        hit_points_path = session_dir / "hit_points.json"
        detailed_results_path = session_dir / "detailed_analysis.json"
        
        if self.hit_detector:
            self.logger.info("Starting hit point detection...")
            
            # Get detection results (enhanced detector returns more info)
            if hasattr(self.hit_detector, 'model_type'):
                # Enhanced detector
                results = self.hit_detector.detect(video_path)
                hit_points = [hit['timestamp'] for hit in results['hit_points']]
                
                # Save detailed results
                with open(detailed_results_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                # Save hit points in original format for compatibility
                with open(hit_points_path, 'w') as f:
                    json.dump({"hit_timestamps": hit_points}, f, indent=2)
                
                self.logger.info(f"Enhanced detection completed. Found {len(hit_points)} hit points with action classification")
                
            else:
                # Original detector
                hit_points = self.hit_detector.detect(video_path)
                
                # Save hit points
                with open(hit_points_path, 'w') as f:
                    json.dump({"hit_timestamps": hit_points}, f, indent=2)
                
                self.logger.info(f"Detected {len(hit_points)} hit points")
        else:
            self.logger.info("Skipping hit point detection")
            # Try to load hit points from existing file
            if hit_points_path.exists():
                with open(hit_points_path, 'r') as f:
                    hit_points = json.load(f)["hit_timestamps"]
            else:
                self.logger.error("No hit points available. Cannot continue.")
                return None
        
        # Step 2: Video Segmentation
        segments_dir = session_dir / "segments"
        segments_info_path = session_dir / "segments_info.json"
        
        if self.segmenter:
            self.logger.info("Starting video segmentation...")
            segments_dir.mkdir(exist_ok=True)
            segments_info = self.segmenter.segment_video(
                video_path,
                hit_points,
                segments_dir
            )
            
            # Save segments info
            with open(segments_info_path, 'w') as f:
                json.dump(segments_info, f, indent=2)
                
            self.logger.info(f"Created {len(segments_info['segments'])} video segments")
        else:
            self.logger.info("Skipping video segmentation")
            # Try to load segments info from existing file
            if segments_info_path.exists():
                with open(segments_info_path, 'r') as f:
                    segments_info = json.load(f)
            else:
                self.logger.error("No segments information available. Cannot continue.")
                return None
        
        # Step 3: Annotation Generation
        annotations_path = session_dir / "annotations.json"
        
        if self.annotator:
            self.logger.info("Starting annotation generation...")
            annotations = self.annotator.annotate_segments(
                segments_dir,
                segments_info["segments"]
            )
            
            # Save annotations
            with open(annotations_path, 'w') as f:
                json.dump(annotations, f, indent=2)
                
            self.logger.info(f"Generated annotations for {len(annotations)} video segments")
        else:
            self.logger.info("Skipping annotation generation")
        
        self.logger.info(f"Pipeline completed. Results available at: {session_dir}")
        return session_dir