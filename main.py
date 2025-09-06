#!/usr/bin/env python3
"""
ShuttleSense: An Intelligent Badminton Video Analysis and Annotation System
Main entry point for the ShuttleSense pipeline.
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from hit_detection.detector import HitPointDetector
from hit_detection.enhanced_detector import EnhancedHitPointDetector
from video_segmentation.segmenter import VideoSegmenter
from annotation.annotator import VideoAnnotator
from pipeline import ShuttleSensePipeline

def setup_logging(log_level):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("shuttle_sense.log")
        ]
    )

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main entry point for ShuttleSense."""
    parser = argparse.ArgumentParser(description="ShuttleSense: Badminton Video Analysis System")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")
    parser.add_argument("--skip-detection", action="store_true", help="Skip hit point detection")
    parser.add_argument("--skip-segmentation", action="store_true", help="Skip video segmentation")
    parser.add_argument("--skip-annotation", action="store_true", help="Skip annotation generation")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    
    # Set up logging
    setup_logging(config["system"]["log_level"])
    logger = logging.getLogger(__name__)
    logger.info("Starting ShuttleSense pipeline")
    
    # Ensure output directories exist
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create pipeline components
    if args.skip_detection:
        hit_detector = None
    else:
        # Choose detector based on configuration
        hit_config = config["hit_detection"]
        use_enhanced = hit_config.get("use_enhanced_detector", False)
        
        if use_enhanced:
            logger.info("Using enhanced hit detector with action classification")
            hit_detector = EnhancedHitPointDetector(hit_config)
        else:
            logger.info("Using standard hit detector")
            hit_detector = HitPointDetector(hit_config)
    
    segmenter = None if args.skip_segmentation else VideoSegmenter(config["video_segmentation"])
    annotator = None if args.skip_annotation else VideoAnnotator(config["annotation"])
    
    # Create and run pipeline
    pipeline = ShuttleSensePipeline(
        hit_detector=hit_detector,
        segmenter=segmenter,
        annotator=annotator,
        config=config
    )
    
    # Run the pipeline
    pipeline.run(args.video)
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()