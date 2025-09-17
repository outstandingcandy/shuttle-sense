#!/usr/bin/env python3
"""
Test script for video slicing inference functionality
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qwen_vl_lora.inference_qwen_vl import QwenVLActionRecognizer

def test_video_slicing():
    """Test video slicing functionality"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Check if we have a test video
    test_video = Path("test.mp4")
    if not test_video.exists():
        logger.error(f"Test video not found: {test_video}")
        logger.info("Please place a test video file named 'test.mp4' in the root directory")
        return
    
    # Check if we have a model (for testing we can use a dummy path)
    model_path = "dummy_model_path"  # This will use base model without LoRA
    
    try:
        # Initialize recognizer
        logger.info("Initializing QwenVL Action Recognizer...")
        
        # Use a minimal config for testing
        recognizer = QwenVLActionRecognizer(
            model_path=model_path,
            config_path=None  # Use default config
        )
        
        # Test 1: Video slicing without overlap
        logger.info("\n=== Test 1: Video slicing (5s intervals, no overlap) ===")
        result1 = recognizer.recognize_action_on_slices(
            video_path=str(test_video),
            interval_seconds=5.0,
            overlap_seconds=0.0
        )
        
        print(f"Video: {result1['video_path']}")
        print(f"Total slices: {result1['total_slices']}")
        print(f"Successful slices: {result1['processing_summary']['successful_slices']}")
        print(f"Failed slices: {result1['processing_summary']['failed_slices']}")
        
        for slice_result in result1['slice_results'][:3]:  # Show first 3 slices
            print(f"  Slice {slice_result['slice_index']} ({slice_result['start_time']:.1f}s-{slice_result['end_time']:.1f}s): {slice_result['predicted_action']}")
        
        # Test 2: Video slicing with overlap
        logger.info("\n=== Test 2: Video slicing (3s intervals, 1s overlap) ===")
        result2 = recognizer.recognize_action_on_slices(
            video_path=str(test_video),
            interval_seconds=3.0,
            overlap_seconds=1.0
        )
        
        print(f"Video: {result2['video_path']}")
        print(f"Total slices: {result2['total_slices']}")
        print(f"Successful slices: {result2['processing_summary']['successful_slices']}")
        print(f"Failed slices: {result2['processing_summary']['failed_slices']}")
        
        for slice_result in result2['slice_results'][:3]:  # Show first 3 slices
            print(f"  Slice {slice_result['slice_index']} ({slice_result['start_time']:.1f}s-{slice_result['end_time']:.1f}s): {slice_result['predicted_action']}")
        
        # Test 3: Just slice extraction (without inference)
        logger.info("\n=== Test 3: Video slice extraction only ===")
        slices = recognizer.slice_video_with_overlap(
            video_path=str(test_video),
            interval_seconds=2.0,
            overlap_seconds=0.5
        )
        
        print(f"Extracted {len(slices)} slices:")
        for slice_info in slices[:5]:  # Show first 5 slices
            print(f"  Slice {slice_info['slice_index']}: {slice_info['start_time']:.1f}s-{slice_info['end_time']:.1f}s ({slice_info['num_frames']} frames)")
            
        logger.info("✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_video_slicing()