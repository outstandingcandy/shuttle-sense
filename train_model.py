#!/usr/bin/env python3
"""
Training Script Launcher for VideoMAE Fine-tuning
Convenient script to start model training with different configurations
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
            logging.FileHandler('training.log')
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

def check_dataset():
    """Check if dataset is properly configured"""
    from src.training.prepare_dataset import VideoBadmintonDatasetPreparator
    
    preparator = VideoBadmintonDatasetPreparator()
    
    # Check if raw data exists
    raw_annotations = Path("data/VideoBadminton_Dataset/raw/annotations.json")
    if not raw_annotations.exists():
        print("⚠️  No dataset found. Creating sample dataset...")
        preparator.create_sample_annotations()
        
        print("📁 Sample dataset created at data/VideoBadminton_Dataset/raw/")
        print("   Please replace with real VideoBadminton_Dataset")
        return False
    
    # Check if split data exists
    train_dir = Path("data/VideoBadminton_Dataset/train")
    if not train_dir.exists():
        print("📊 Splitting dataset...")
        preparator.split_dataset()
    
    print("✓ Dataset is ready")
    return True

def main():
    parser = argparse.ArgumentParser(description="Train VideoMAE for Badminton Hit Detection")
    parser.add_argument("--config", default="configs/training_config.yaml", 
                       help="Path to training configuration file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--dataset-only", action="store_true", 
                       help="Only prepare dataset, don't start training")
    parser.add_argument("--dry-run", action="store_true",
                       help="Perform all checks but don't start training")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    print("🏸 VideoMAE Training for Badminton Hit Detection")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check dataset
    if not check_dataset():
        if not args.dataset_only:
            print("\n❌ Dataset not ready. Please prepare the dataset first.")
            sys.exit(1)
    
    if args.dataset_only:
        print("✅ Dataset preparation completed!")
        return
    
    # Check configuration file
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    print(f"✓ Using configuration: {args.config}")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name} ({gpu_count} GPUs)")
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower)")
    except ImportError:
        print("⚠️  PyTorch not available")
    
    if args.dry_run:
        print("✅ All checks passed! Ready for training.")
        print("Remove --dry-run to start actual training.")
        return
    
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    try:
        # Import and run training
        sys.path.append(str(Path(__file__).parent))
        from src.training.train_videomae import main as train_main
        
        # Override config path
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add resume checkpoint if provided
        if args.resume:
            config['training']['resume_from'] = args.resume
        
        # Save modified config temporarily
        temp_config = Path("temp_config.yaml")
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        # Set config path environment variable
        os.environ['TRAINING_CONFIG'] = str(temp_config)
        
        # Start training
        train_main()
        
        # Cleanup
        if temp_config.exists():
            temp_config.unlink()
        
        print("\n🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()