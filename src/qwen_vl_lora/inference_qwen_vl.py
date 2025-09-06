"""
Qwen-VL LoRA Inference Pipeline for Badminton Action Recognition
Use trained LoRA model to perform action classification on video segments
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import yaml
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenVLActionRecognizer:
    """Qwen-VL LoRA model for badminton action recognition"""
    
    def __init__(self, 
                 model_path: str,
                 base_model_name: str = "Qwen/Qwen-VL-Chat",
                 config_path: str = None,
                 device: str = "auto"):
        """
        Initialize the action recognizer
        
        Args:
            model_path: Path to the trained LoRA model
            base_model_name: Base Qwen-VL model name
            config_path: Path to configuration file
            device: Device to run on ("auto", "cuda", "cpu")
        """
        self.model_path = Path(model_path)
        self.base_model_name = base_model_name
        self.device = self._setup_device(device)
        self.config = self._load_config(config_path)
        
        # Load model components
        self.tokenizer = None
        self.model = None
        self.processor = None
        
        self._load_model()
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computing device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        return torch.device(device)
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            "max_frames": 8,
            "frame_size": [224, 224],
            "max_new_tokens": 50,
            "temperature": 0.3,
            "top_p": 0.8,
            "do_sample": True,
            "frame_sampling": "uniform"
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                
            # Update with config values
            for key, value in yaml_config.get('inference', {}).items():
                default_config[key] = value
                
            # Add data config
            for key, value in yaml_config.get('data', {}).items():
                default_config[key] = value
        
        return default_config
    
    def _load_model(self):
        """Load the trained LoRA model"""
        logger.info(f"Loading Qwen-VL LoRA model from {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
                use_fast=False
            )
            
            # Load processor
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.base_model_name,
                    trust_remote_code=True
                )
            except:
                self.processor = None
                logger.warning("Could not load processor, using tokenizer only")
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Load LoRA weights
            if self.model_path.exists():
                self.model = PeftModel.from_pretrained(
                    base_model,
                    str(self.model_path),
                    device_map="auto" if self.device.type == "cuda" else None
                )
                logger.info("✅ LoRA model loaded successfully")
            else:
                logger.warning(f"LoRA model not found at {self.model_path}, using base model")
                self.model = base_model
            
            # Set to evaluation mode
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def extract_video_frames(self, video_path: str) -> List[Image.Image]:
        """Extract frames from video"""
        video_path = Path(video_path)
        
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return []
        
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = self.config.get("max_inference_frames", self.config.get("max_frames", 8))
        frame_size = tuple(self.config.get("frame_size", [224, 224]))
        
        # Calculate frame indices to extract
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            if self.config.get("frame_sampling", "uniform") == "uniform":
                # Evenly sample frames
                frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            else:
                # Random sampling
                frame_indices = np.random.choice(total_frames, max_frames, replace=False)
                frame_indices.sort()
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize frame
                frame = cv2.resize(frame, frame_size)
                # Convert to PIL Image
                pil_frame = Image.fromarray(frame)
                frames.append(pil_frame)
        
        cap.release()
        
        return frames[:max_frames]
    
    def recognize_action(self, 
                        video_path: str, 
                        question: str = None) -> Dict[str, Any]:
        """
        Recognize action in video
        
        Args:
            video_path: Path to video file
            question: Custom question, defaults to standard action recognition question
            
        Returns:
            Dictionary containing prediction results
        """
        if not question:
            question = "请识别视频中的羽毛球动作是什么？"
        
        # Extract video frames
        frames = self.extract_video_frames(video_path)
        
        if not frames:
            return {
                "video_path": video_path,
                "predicted_action": "无法处理视频",
                "confidence": 0.0,
                "error": "Could not extract frames from video"
            }
        
        # Prepare input
        input_text = f"用户: {question}\n助手:"
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # Move to device
            if self.device.type == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get("max_new_tokens", 50),
                    temperature=self.config.get("temperature", 0.3),
                    top_p=self.config.get("top_p", 0.8),
                    do_sample=self.config.get("do_sample", True),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            return {
                "video_path": video_path,
                "question": question,
                "predicted_action": response,
                "confidence": 1.0,  # Confidence scoring would require additional implementation
                "num_frames_processed": len(frames),
                "model_response": response
            }
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            return {
                "video_path": video_path,
                "predicted_action": "推理失败",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def batch_recognize(self, 
                       video_paths: List[str],
                       question: str = None) -> List[Dict[str, Any]]:
        """
        Recognize actions in multiple videos
        
        Args:
            video_paths: List of video file paths
            question: Custom question for all videos
            
        Returns:
            List of prediction results
        """
        results = []
        
        for video_path in tqdm(video_paths, desc="Processing videos"):
            result = self.recognize_action(video_path, question)
            results.append(result)
        
        return results
    
    def evaluate_on_dataset(self, 
                           dataset_path: str,
                           split: str = "val") -> Dict[str, Any]:
        """
        Evaluate model on validation dataset
        
        Args:
            dataset_path: Path to dataset directory
            split: Dataset split to evaluate on ("val", "test")
            
        Returns:
            Evaluation results
        """
        dataset_path = Path(dataset_path)
        data_file = dataset_path / f"{split}.json"
        
        if not data_file.exists():
            logger.error(f"Dataset file not found: {data_file}")
            return {"error": f"Dataset file not found: {data_file}"}
        
        # Load dataset
        with open(data_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        results = []
        correct_predictions = 0
        total_predictions = 0
        
        logger.info(f"Evaluating on {len(conversations)} samples")
        
        for conversation in tqdm(conversations, desc=f"Evaluating {split} set"):
            video_path = dataset_path / conversation["video"]
            
            # Extract question and ground truth
            user_message = conversation["conversations"][0]["value"]
            ground_truth = conversation["conversations"][1]["value"]
            question = user_message.replace("<video>", "").strip()
            
            # Get prediction
            prediction = self.recognize_action(str(video_path), question)
            
            # Simple exact match evaluation
            predicted_action = prediction["predicted_action"]
            is_correct = predicted_action.strip() == ground_truth.strip()
            
            if is_correct:
                correct_predictions += 1
            
            total_predictions += 1
            
            results.append({
                "conversation_id": conversation.get("id", "unknown"),
                "ground_truth": ground_truth,
                "predicted_action": predicted_action,
                "is_correct": is_correct,
                "video_path": str(video_path)
            })
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        evaluation_results = {
            "split": split,
            "total_samples": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "detailed_results": results
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
        
        return evaluation_results

def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen-VL LoRA inference for badminton action recognition")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained LoRA model")
    parser.add_argument("--video", type=str, help="Path to single video for inference")
    parser.add_argument("--video-dir", type=str, help="Directory of videos for batch inference")
    parser.add_argument("--evaluate", type=str, help="Evaluate on dataset (provide dataset path)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split for evaluation")
    parser.add_argument("--question", type=str, help="Custom question for inference")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    if not args.model_path:
        logger.error("Model path is required")
        return
    
    # Initialize recognizer
    recognizer = QwenVLActionRecognizer(
        model_path=args.model_path,
        config_path=args.config
    )
    
    results = []
    
    try:
        if args.video:
            # Single video inference
            logger.info(f"Processing single video: {args.video}")
            result = recognizer.recognize_action(args.video, args.question)
            results.append(result)
            
            print(f"Video: {args.video}")
            print(f"Predicted Action: {result['predicted_action']}")
            if 'confidence' in result:
                print(f"Confidence: {result['confidence']:.4f}")
            
        elif args.video_dir:
            # Batch inference on directory
            video_dir = Path(args.video_dir)
            video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
            
            logger.info(f"Processing {len(video_files)} videos from {video_dir}")
            results = recognizer.batch_recognize([str(f) for f in video_files], args.question)
            
            # Print results
            for result in results:
                print(f"Video: {Path(result['video_path']).name}")
                print(f"Predicted Action: {result['predicted_action']}")
                print("---")
        
        elif args.evaluate:
            # Evaluate on dataset
            logger.info(f"Evaluating on dataset: {args.evaluate}")
            evaluation_results = recognizer.evaluate_on_dataset(args.evaluate, args.split)
            
            print(f"Evaluation Results ({args.split} set):")
            print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
            print(f"Correct: {evaluation_results['correct_predictions']}/{evaluation_results['total_samples']}")
            
            results = [evaluation_results]
        
        else:
            logger.error("Please specify --video, --video-dir, or --evaluate")
            return
        
        # Save results if output path provided
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to: {args.output}")
        
        logger.info("✅ Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()