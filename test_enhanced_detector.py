#!/usr/bin/env python3
"""
Test script for Enhanced Hit Detector with Action Classification
Tests the integration of custom trained model with hit detection
"""

import sys
import os
import logging
import yaml
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from hit_detection.enhanced_detector import EnhancedHitPointDetector

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config():
    """Load configuration from config.yaml."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_model_loading():
    """Test model loading functionality."""
    print("🧪 Testing Enhanced Hit Detector...")
    print("=" * 50)
    
    try:
        # Load configuration
        config = load_config()
        hit_config = config['hit_detection']
        
        print("📋 Configuration:")
        print(f"  Use enhanced detector: {hit_config.get('use_enhanced_detector', False)}")
        print(f"  Use custom model: {hit_config.get('use_custom_model', False)}")
        print(f"  Custom model path: {hit_config.get('custom_model_path', 'None')}")
        
        # Initialize enhanced detector
        print("\n🔄 Initializing Enhanced Hit Detector...")
        detector = EnhancedHitPointDetector(hit_config)
        
        # Get model info
        model_info = detector.get_model_info()
        print("\n✅ Model successfully loaded!")
        print("📊 Model Information:")
        print(f"  Model type: {model_info['model_type']}")
        print(f"  Device: {model_info['device']}")
        print(f"  Using custom model: {model_info['use_custom_model']}")
        
        if model_info['model_type'] == 'action_classification':
            print(f"  Number of classes: {model_info['num_classes']}")
            print(f"  Action classes: {len(model_info['action_classes'])} classes")
            print("  Sample classes:")
            for i, class_name in enumerate(model_info['action_classes'][:5]):
                print(f"    {i}: {class_name}")
            if len(model_info['action_classes']) > 5:
                print(f"    ... and {len(model_info['action_classes']) - 5} more classes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_video_analysis(video_path=None):
    """Test video analysis with sample video."""
    if not video_path:
        # Look for sample video
        sample_videos = [
            "424_raw.MOV",
            "sample_video.mp4",
            "test_video.mp4"
        ]
        
        for video in sample_videos:
            if os.path.exists(video):
                video_path = video
                break
        
        if not video_path:
            print("⚠️  No sample video found for testing. Skipping video analysis test.")
            print("   Available sample videos to try:")
            print("   - 424_raw.MOV")
            print("   - sample_video.mp4") 
            print("   - test_video.mp4")
            return True
    
    print(f"\n🎬 Testing video analysis with: {video_path}")
    print("-" * 50)
    
    try:
        # Load configuration and create detector
        config = load_config()
        hit_config = config['hit_detection']
        detector = EnhancedHitPointDetector(hit_config)
        
        print("🔍 Analyzing video... (this may take a while)")
        
        # Analyze video
        results = detector.detect(video_path)
        
        # Display results
        print("✅ Analysis completed!")
        print(f"📊 Results:")
        print(f"  Video: {results['video_info']['path']}")
        print(f"  Duration: {results['video_info']['duration']:.2f} seconds")
        print(f"  FPS: {results['video_info']['fps']:.2f}")
        print(f"  Hit points found: {len(results['hit_points'])}")
        
        if results['hit_points']:
            print("\n🎯 Hit Points Detected:")
            for i, hit in enumerate(results['hit_points'][:5]):
                print(f"  {i+1}. Time: {hit['timestamp']:.2f}s, "
                      f"Action: {hit['action_class']}, "
                      f"Confidence: {hit['confidence']:.3f}")
            
            if len(results['hit_points']) > 5:
                print(f"  ... and {len(results['hit_points']) - 5} more hit points")
        
        return True
        
    except Exception as e:
        print(f"❌ Video analysis failed: {e}")
        return False

def main():
    """Main test function."""
    setup_logging()
    
    print("🏸 Enhanced Hit Detector Integration Test")
    print("=" * 60)
    
    # Test 1: Model loading
    print("\n1️⃣  Testing Model Loading")
    model_test_passed = test_model_loading()
    
    if not model_test_passed:
        print("\n❌ Model loading test failed. Please check your configuration and model files.")
        return False
    
    # Test 2: Video analysis (if sample video available)
    print("\n2️⃣  Testing Video Analysis")
    video_test_passed = test_video_analysis()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Integration Test Summary")
    print("=" * 60)
    print(f"Model loading: {'✅ PASS' if model_test_passed else '❌ FAIL'}")
    print(f"Video analysis: {'✅ PASS' if video_test_passed else '❌ FAIL'}")
    
    if model_test_passed and video_test_passed:
        print("\n🎊 All tests passed! The enhanced hit detector is ready to use.")
        print("\n💡 You can now use the enhanced detector in your pipeline:")
        print("   python main.py --video your_video.mp4")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    return model_test_passed and video_test_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)