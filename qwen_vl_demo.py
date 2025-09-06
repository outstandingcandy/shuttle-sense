#!/usr/bin/env python3
"""
Qwen-VL LoRA Demo Script
Demonstration of Qwen-VL LoRA fine-tuning and inference for badminton action recognition
"""

import argparse
import logging
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_dataset_preparation():
    """Demonstrate dataset preparation for Qwen-VL training"""
    logger.info("🏸 Demo: Dataset Preparation")
    
    from src.qwen_vl_lora import QwenVLDatasetPreparator
    
    try:
        preparator = QwenVLDatasetPreparator()
        success = preparator.prepare_dataset()
        
        if success:
            logger.info("✅ Dataset preparation completed successfully!")
        else:
            logger.error("❌ Dataset preparation failed")
            
    except Exception as e:
        logger.error(f"Dataset preparation error: {str(e)}")

def demo_training(epochs: int = 2):
    """Demonstrate LoRA training (with minimal epochs for demo)"""
    logger.info("🏸 Demo: LoRA Training")
    
    from src.qwen_vl_lora.train_qwen_vl_lora import QwenVLLoRATrainer, QwenVLTrainingConfig
    
    try:
        # Create minimal config for demo
        config = QwenVLTrainingConfig(
            epochs=epochs,
            batch_size=1,
            dataset_path="data/qwen_vl_dataset",
            output_dir="models/checkpoints/qwen_vl_lora_demo"
        )
        
        trainer = QwenVLLoRATrainer(config)
        trainer.train()
        
        logger.info("✅ LoRA training demo completed!")
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        logger.info("💡 Note: Training requires significant GPU memory. Consider using smaller batch size or gradient checkpointing.")

def demo_inference(model_path: str, video_path: str = None):
    """Demonstrate inference with trained model"""
    logger.info("🏸 Demo: Qwen-VL Inference")
    
    from src.qwen_vl_lora import QwenVLActionRecognizer
    
    if not video_path:
        logger.warning("No video path provided, skipping inference demo")
        return
    
    try:
        recognizer = QwenVLActionRecognizer(
            model_path=model_path,
            config_path="configs/qwen_vl_lora_config.yaml"
        )
        
        logger.info(f"Processing video: {video_path}")
        result = recognizer.recognize_action(video_path)
        
        logger.info("📊 Inference Results:")
        logger.info(f"  Predicted Action: {result['predicted_action']}")
        logger.info(f"  Confidence: {result['confidence']:.4f}")
        logger.info(f"  Frames Processed: {result['num_frames_processed']}")
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")

def demo_integration(video_path: str = None):
    """Demonstrate integration with existing ShuttleSense pipeline"""
    logger.info("🏸 Demo: Pipeline Integration")
    
    from src.qwen_vl_lora import create_qwen_vl_enhanced_detector
    import yaml
    
    if not video_path:
        logger.warning("No video path provided, skipping integration demo")
        return
    
    try:
        # Load config
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Create enhanced detector with Qwen-VL integration
        detector = create_qwen_vl_enhanced_detector(config)
        
        logger.info(f"Processing video with integrated pipeline: {video_path}")
        results = detector.process_video_with_action_recognition(
            video_path=video_path,
            use_ensemble=True,
            save_segments=False  # Skip saving for demo
        )
        
        logger.info("📊 Integration Results:")
        logger.info(f"  Total Hits: {results['total_hits']}")
        logger.info(f"  Total Segments: {results['total_segments']}")
        logger.info(f"  Qwen-VL Enabled: {results['qwen_vl_enabled']}")
        logger.info(f"  Ensemble Used: {results['ensemble_used']}")
        
        # Show action classifications for first few segments
        for i, segment in enumerate(results['segments'][:3]):
            action_class = segment['action_classification']
            logger.info(f"  Segment {i+1}: {action_class['action']} (confidence: {action_class['confidence']:.3f})")
        
    except Exception as e:
        logger.error(f"Integration error: {str(e)}")

def demo_evaluation(dataset_path: str = "data/qwen_vl_dataset", model_path: str = None):
    """Demonstrate model evaluation"""
    logger.info("🏸 Demo: Model Evaluation")
    
    from src.qwen_vl_lora import QwenVLActionRecognizer
    
    if not model_path or not Path(model_path).exists():
        logger.warning("Model path not provided or doesn't exist, skipping evaluation demo")
        return
    
    try:
        recognizer = QwenVLActionRecognizer(
            model_path=model_path,
            config_path="configs/qwen_vl_lora_config.yaml"
        )
        
        logger.info(f"Evaluating on validation set: {dataset_path}")
        results = recognizer.evaluate_on_dataset(dataset_path, split="val")
        
        logger.info("📊 Evaluation Results:")
        logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        logger.info(f"  Correct: {results['correct_predictions']}/{results['total_samples']}")
        
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Qwen-VL LoRA Demo for Badminton Action Recognition")
    
    parser.add_argument("--demo", 
                       choices=["prepare", "train", "inference", "integration", "evaluation", "all"],
                       default="all",
                       help="Which demo to run")
    
    parser.add_argument("--video", type=str, help="Path to video file for inference/integration demo")
    parser.add_argument("--model", type=str, 
                       default="models/checkpoints/qwen_vl_lora/final_model",
                       help="Path to trained LoRA model")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for training demo")
    parser.add_argument("--dataset", type=str, default="data/qwen_vl_dataset", 
                       help="Path to Qwen-VL dataset")
    
    args = parser.parse_args()
    
    logger.info("🏸 Starting Qwen-VL LoRA Demo")
    logger.info("=" * 50)
    
    if args.demo == "prepare" or args.demo == "all":
        demo_dataset_preparation()
        logger.info("")
    
    if args.demo == "train" or args.demo == "all":
        demo_training(args.epochs)
        logger.info("")
    
    if args.demo == "inference" or args.demo == "all":
        demo_inference(args.model, args.video)
        logger.info("")
    
    if args.demo == "integration" or args.demo == "all":
        demo_integration(args.video)
        logger.info("")
    
    if args.demo == "evaluation" or args.demo == "all":
        demo_evaluation(args.dataset, args.model)
        logger.info("")
    
    logger.info("✅ Demo completed!")
    logger.info("\n🎯 Next Steps:")
    logger.info("1. Prepare dataset: python qwen_vl_demo.py --demo prepare")
    logger.info("2. Train LoRA model: python src/qwen_vl_lora/train_qwen_vl_lora.py")
    logger.info("3. Test inference: python src/qwen_vl_lora/inference_qwen_vl.py --video path/to/video.mp4")
    logger.info("4. Use in web app: The integration is automatically available in the web interface")

if __name__ == "__main__":
    main()