#!/usr/bin/env python3
"""
Action Classification Training Script
Train VideoMAE on VideoBadminton_Dataset for 18-class action classification
"""

import os
import sys
import argparse
import logging
from pathlib import Path

def setup_logging(log_level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('action_classification_training.log')
        ]
    )

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = {
        'torch': 'torch', 
        'transformers': 'transformers', 
        'cv2': 'opencv-python or opencv-python-headless',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn', 
        'tqdm': 'tqdm', 
        'matplotlib': 'matplotlib', 
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"uv pip install {' '.join(missing_packages)}")
        return False
    
    print("✓ All required packages are installed")
    return True

def check_dataset(dataset_path):
    """Check if dataset exists and has correct structure"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        return False
    
    # Check for action class directories
    action_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if len(action_dirs) == 0:
        print(f"❌ No action class directories found in {dataset_path}")
        return False
    
    print(f"✓ Found {len(action_dirs)} action classes:")
    
    total_videos = 0
    for action_dir in sorted(action_dirs)[:5]:  # Show first 5
        video_files = list(action_dir.glob("*.mp4"))
        total_videos += len(video_files)
        print(f"   - {action_dir.name}: {len(video_files)} videos")
    
    if len(action_dirs) > 5:
        remaining_videos = 0
        for action_dir in sorted(action_dirs)[5:]:
            video_files = list(action_dir.glob("*.mp4"))
            remaining_videos += len(video_files)
        total_videos += remaining_videos
        print(f"   - ... and {len(action_dirs)-5} more classes with {remaining_videos} videos")
    
    print(f"✓ Total: {total_videos} videos across {len(action_dirs)} action classes")
    
    if total_videos < 100:
        print("⚠️  Warning: Dataset seems small. Consider getting more data for better performance.")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Train VideoMAE for Badminton Action Classification")
    parser.add_argument("--config", default="configs/action_classification_config.yaml", 
                       help="Path to training configuration file")
    parser.add_argument("--dataset", default="/home/ubuntu/Project/shuttle-sense/VideoBadminton_Dataset",
                       help="Path to VideoBadminton_Dataset directory")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true",
                       help="Perform all checks but don't start training")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    print("🏸 VideoMAE Action Classification Training")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check dataset
    if not check_dataset(args.dataset):
        sys.exit(1)
    
    # Check configuration file
    if not Path(args.config).exists():
        print(f"⚠️  Configuration file not found: {args.config}")
        print("Using default configuration...")
    else:
        print(f"✓ Using configuration: {args.config}")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name} ({gpu_count} GPUs)")
        else:
            print("⚠️  No GPU detected. Training will use CPU (much slower)")
    except ImportError:
        print("⚠️  PyTorch not available")
    
    if args.dry_run:
        print("✅ All checks passed! Ready for training.")
        print("Remove --dry-run to start actual training.")
        return
    
    print("\n🚀 Starting action classification training...")
    print("=" * 60)
    
    try:
        # Import and run training
        sys.path.append(str(Path(__file__).parent))
        from src.training.train_action_classification import main as train_main
        
        # Override dataset path in environment
        os.environ['DATASET_PATH'] = args.dataset
        os.environ['CONFIG_PATH'] = args.config
        
        # Start training
        train_main()
        
        print("\n🎉 Action classification training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()