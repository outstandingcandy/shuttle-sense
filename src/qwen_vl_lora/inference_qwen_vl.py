"""
Qwen2.5-VL LoRA Inference Pipeline for Badminton Action Recognition
Use trained LoRA model to perform action classification on video segments
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import torch
import cv2
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import yaml
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenVLActionRecognizer:
    """Qwen2.5-VL LoRA model for badminton action recognition"""
    
    def __init__(self, 
                 model_path: str,
                 base_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 config_path: str = None,
                 device: str = "auto"):
        """
        Initialize the action recognizer
        
        Args:
            model_path: Path to the trained LoRA model
            base_model_name: Base Qwen2.5-VL model name
            config_path: Path to configuration file
            device: Device to run on ("auto", "cuda", "cpu")
        """
        self.model_path = Path(model_path)
        self.base_model_name = base_model_name
        self.device = self._setup_device(device)
        self.config = self._load_config(config_path)
        
        # Load model components
        self.processor = None
        self.model = None
        
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
        logger.info(f"Loading Qwen2.5-VL LoRA model from {self.model_path}")
        
        try:
            # Load processor (includes tokenizer)
            self.processor = AutoProcessor.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
                use_fast=False  # Use slow processor for compatibility
            )
            logger.info("✅ Processor loaded successfully")
            
            # Load base model
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.base_model_name,
                torch_dtype="auto",
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
                
                # Resize frame while preserving aspect ratio
                h, w = frame.shape[:2]
                target_size = frame_size[0]  # Use the first dimension as target
                
                # Calculate scaling factor to fit the larger dimension to target_size
                scale = target_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                
                # Resize maintaining aspect ratio
                frame = cv2.resize(frame, (new_w, new_h))
                
                # Pad to make it square
                pad_h = target_size - new_h
                pad_w = target_size - new_w
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                # Add padding (black borders)
                frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, 
                                         cv2.BORDER_CONSTANT, value=[0, 0, 0])
                
                # Convert to PIL Image
                pil_frame = Image.fromarray(frame)
                frames.append(pil_frame)
        
        cap.release()
        
        return frames[:max_frames]
        
    def _perform_inference_on_frames(self, 
                                    frames: List[Image.Image], 
                                    question: str) -> Dict[str, Any]:
        """
        Core inference method that takes frames and returns prediction
        
        Args:
            frames: List of PIL Image frames
            question: Question to ask about the frames
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Format messages using Qwen2.5-VL conversation pattern
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Apply chat template
            input_text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process with processor
            inputs = self.processor(
                text=input_text,
                videos=frames,
                padding=True,
                return_tensors="pt"
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
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            return {
                "predicted_action": response,
                "confidence": 1.0,  # Confidence scoring would require additional implementation
                "num_frames_processed": len(frames),
                "model_response": response,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            return {
                "predicted_action": "推理失败",
                "confidence": 0.0,
                "error": str(e),
                "success": False
            }
    
    def _save_frames_to_disk(self, 
                           frames: List[Image.Image], 
                           output_dir: Path, 
                           prefix: str = "frame") -> List[str]:
        """
        Save frames to disk
        
        Args:
            frames: List of PIL Image frames
            output_dir: Directory to save frames
            prefix: Filename prefix
            
        Returns:
            List of saved frame paths
        """
        saved_paths = []
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame_path = output_dir / f"{prefix}_{i:03d}.jpg"
            frame.save(frame_path, "JPEG", quality=95)
            saved_paths.append(str(frame_path))
        
        return saved_paths
    
    def recognize_action(self, 
                        video_path: str, 
                        question: str = None,
                        save_frames: bool = False,
                        frames_output_dir: str = None) -> Dict[str, Any]:
        """
        Recognize action in video
        
        Args:
            video_path: Path to video file
            question: Custom question, defaults to standard action recognition question
            save_frames: Whether to save extracted frames to disk
            frames_output_dir: Directory to save frames (defaults to "inference_frames")
            
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
        
        # Save frames if requested
        saved_frame_paths = []
        video_frames_dir = None
        if save_frames:
            if frames_output_dir is None:
                frames_output_dir = "inference_frames"
            
            # Create output directory
            output_dir = Path(frames_output_dir)
            video_name = Path(video_path).stem
            video_frames_dir = output_dir / video_name
            
            saved_frame_paths = self._save_frames_to_disk(frames, video_frames_dir)
            logger.info(f"Saved {len(frames)} frames to {video_frames_dir}")
        
        # Perform inference
        inference_result = self._perform_inference_on_frames(frames, question)
        
        # Build result
        result = {
            "video_path": video_path,
            "question": question,
            **inference_result  # Unpack inference results
        }
        
        # Add frame paths if frames were saved
        if save_frames:
            result["saved_frame_paths"] = saved_frame_paths
            result["frames_output_dir"] = str(video_frames_dir)
        
        return result
    
    def slice_video_with_overlap(self, 
                                   video_path: str, 
                                   interval_seconds: float, 
                                   overlap_seconds: float = 0.0) -> List[Dict[str, Any]]:
        """
        Slice video into overlapping segments based on time intervals
        
        Args:
            video_path: Path to video file
            interval_seconds: Duration of each slice in seconds
            overlap_seconds: Overlap duration between slices in seconds
            
        Returns:
            List of dictionaries containing slice info (start_time, end_time, frames)
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return []
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        
        logger.info(f"Video info: {total_frames} frames, {fps} FPS, {total_duration:.2f}s duration")
        
        slices = []
        current_start = 0.0
        slice_index = 0
        
        while current_start < total_duration:
            current_end = min(current_start + interval_seconds, total_duration)
            
            # Calculate frame indices for this slice
            start_frame = int(current_start * fps)
            end_frame = int(current_end * fps)
            
            # Extract frames for this slice
            frames = []
            frame_size = tuple(self.config.get("frame_size", [224, 224]))
            max_frames = self.config.get("max_inference_frames", self.config.get("max_frames", 8))
            
            # Get frame indices for this slice
            slice_frame_count = end_frame - start_frame
            if slice_frame_count <= max_frames:
                frame_indices = list(range(start_frame, end_frame))
            else:
                # Uniformly sample frames from this slice
                frame_indices = np.linspace(start_frame, end_frame - 1, max_frames, dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB and resize
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize frame while preserving aspect ratio
                    h, w = frame.shape[:2]
                    target_size = frame_size[0]
                    
                    scale = target_size / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    
                    frame = cv2.resize(frame, (new_w, new_h))
                    
                    # Pad to make it square
                    pad_h = target_size - new_h
                    pad_w = target_size - new_w
                    pad_top = pad_h // 2
                    pad_bottom = pad_h - pad_top
                    pad_left = pad_w // 2
                    pad_right = pad_w - pad_left
                    
                    frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, 
                                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    
                    pil_frame = Image.fromarray(frame)
                    frames.append(pil_frame)
            
            if frames:
                slice_info = {
                    "slice_index": slice_index,
                    "start_time": current_start,
                    "end_time": current_end,
                    "duration": current_end - current_start,
                    "frames": frames,
                    "num_frames": len(frames)
                }
                slices.append(slice_info)
                slice_index += 1
            
            # Move to next slice position (with overlap)
            current_start += (interval_seconds - overlap_seconds)
            
            # Prevent infinite loop if overlap >= interval
            if overlap_seconds >= interval_seconds:
                logger.warning("Overlap duration >= interval duration, adjusting to prevent infinite loop")
                current_start += 0.1  # Add small increment
        
        cap.release()
        logger.info(f"Created {len(slices)} video slices")
        
        return slices

    def recognize_action_on_slices(self, 
                                  video_path: str,
                                  interval_seconds: float,
                                  overlap_seconds: float = 0.0,
                                  question: str = None,
                                  save_frames: bool = False,
                                  frames_output_dir: str = None) -> Dict[str, Any]:
        """
        Perform inference on video slices with specified intervals and overlap
        
        Args:
            video_path: Path to video file
            interval_seconds: Duration of each slice in seconds
            overlap_seconds: Overlap duration between slices in seconds  
            question: Custom question, defaults to standard action recognition question
            save_frames: Whether to save extracted frames from each slice
            frames_output_dir: Directory to save frames (defaults to "inference_frames")
            
        Returns:
            Dictionary containing results for all slices
        """
        if not question:
            question = "请识别视频中的羽毛球动作是什么？并输出置信度，如果置信度小于0.5，则输出'无法识别'"
        
        logger.info(f"Processing video with {interval_seconds}s intervals and {overlap_seconds}s overlap")
        
        # Get video slices
        slices = self.slice_video_with_overlap(video_path, interval_seconds, overlap_seconds)
        
        if not slices:
            return {
                "video_path": video_path,
                "interval_seconds": interval_seconds,
                "overlap_seconds": overlap_seconds,
                "total_slices": 0,
                "slice_results": [],
                "error": "Could not create video slices"
            }
        
        slice_results = []
        
        # Setup frame saving if requested
        video_frames_dir = None
        if save_frames:
            if frames_output_dir is None:
                frames_output_dir = "inference_frames"
            
            output_dir = Path(frames_output_dir)
            video_name = Path(video_path).stem
            video_frames_dir = output_dir / f"{video_name}_slices"
        
        for slice_info in tqdm(slices, desc="Processing video slices"):
            try:
                # Save frames for this slice if requested
                saved_frame_paths = []
                slice_frames_dir = None
                if save_frames and video_frames_dir:
                    slice_frames_dir = video_frames_dir / f"slice_{slice_info['slice_index']:03d}"
                    saved_frame_paths = self._save_frames_to_disk(
                        slice_info["frames"], 
                        slice_frames_dir, 
                        prefix="frame"
                    )
                
                # Perform inference using core method
                inference_result = self._perform_inference_on_frames(slice_info["frames"], question)
                
                # Build slice result
                slice_result = {
                    "slice_index": slice_info["slice_index"],
                    "start_time": slice_info["start_time"], 
                    "end_time": slice_info["end_time"],
                    "duration": slice_info["duration"],
                    **inference_result  # Unpack inference results
                }
                
                # Add frame paths if frames were saved
                if save_frames:
                    slice_result["saved_frame_paths"] = saved_frame_paths
                    if slice_frames_dir:
                        slice_result["slice_frames_dir"] = str(slice_frames_dir)
                
                slice_results.append(slice_result)
                
            except Exception as e:
                logger.error(f"Error processing slice {slice_info['slice_index']}: {str(e)}")
                slice_result = {
                    "slice_index": slice_info["slice_index"],
                    "start_time": slice_info["start_time"],
                    "end_time": slice_info["end_time"], 
                    "duration": slice_info["duration"],
                    "predicted_action": "推理失败",
                    "confidence": 0.0,
                    "error": str(e),
                    "success": False
                }
                slice_results.append(slice_result)
        
        result = {
            "video_path": video_path,
            "question": question,
            "interval_seconds": interval_seconds,
            "overlap_seconds": overlap_seconds,
            "total_slices": len(slices),
            "slice_results": slice_results,
            "processing_summary": {
                "successful_slices": len([r for r in slice_results if r.get("success", False)]),
                "failed_slices": len([r for r in slice_results if not r.get("success", False)]),
                "total_processing_time": sum([r["duration"] for r in slice_results])
            }
        }
        
        # Add frame saving info
        if save_frames and video_frames_dir:
            result["frames_saved"] = True
            result["frames_output_dir"] = str(video_frames_dir)
            total_saved_frames = sum([len(r.get("saved_frame_paths", [])) for r in slice_results])
            result["total_saved_frames"] = total_saved_frames
            logger.info(f"Saved total of {total_saved_frames} frames to {video_frames_dir}")
        
        return result

    def batch_recognize(self, 
                       video_paths: List[str],
                       question: str = None,
                       save_frames: bool = False,
                       frames_output_dir: str = None) -> List[Dict[str, Any]]:
        """
        Recognize actions in multiple videos
        
        Args:
            video_paths: List of video file paths
            question: Custom question for all videos
            save_frames: Whether to save extracted frames from each video
            frames_output_dir: Directory to save frames
            
        Returns:
            List of prediction results
        """
        results = []
        
        for video_path in tqdm(video_paths, desc="Processing videos"):
            result = self.recognize_action(
                video_path, 
                question,
                save_frames=save_frames,
                frames_output_dir=frames_output_dir
            )
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
    
    # Video slicing arguments
    parser.add_argument("--slice-video", action="store_true", help="Enable video slicing mode")
    parser.add_argument("--interval", type=float, default=5.0, help="Slice interval in seconds (default: 5.0)")
    parser.add_argument("--overlap", type=float, default=0.0, help="Overlap duration in seconds (default: 0.0)")
    
    # Frame saving arguments
    parser.add_argument("--save-frames", action="store_true", help="Save extracted video frames to disk")
    parser.add_argument("--frames-dir", type=str, help="Directory to save frames (default: inference_frames)")
    
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
            
            if args.slice_video:
                # Video slicing mode
                logger.info(f"Using video slicing: {args.interval}s intervals, {args.overlap}s overlap")
                result = recognizer.recognize_action_on_slices(
                    args.video, 
                    args.interval, 
                    args.overlap,
                    args.question,
                    save_frames=args.save_frames,
                    frames_output_dir=args.frames_dir
                )
                results.append(result)
                
                print(f"Video: {args.video}")
                print(f"Slicing: {args.interval}s intervals, {args.overlap}s overlap")
                print(f"Total slices processed: {result['total_slices']}")
                print(f"Successful slices: {result['processing_summary']['successful_slices']}")
                print(f"Failed slices: {result['processing_summary']['failed_slices']}")
                if args.save_frames and 'frames_output_dir' in result:
                    print(f"Frames saved to: {result['frames_output_dir']}")
                    print(f"Total frames saved: {result.get('total_saved_frames', 0)}")
                print("\nSlice results:")
                for slice_result in result['slice_results']:
                    print(f"  Slice {slice_result['slice_index']} ({slice_result['start_time']:.1f}s-{slice_result['end_time']:.1f}s): {slice_result['predicted_action']}")
            else:
                # Standard single video inference
                result = recognizer.recognize_action(
                    args.video, 
                    args.question,
                    save_frames=args.save_frames,
                    frames_output_dir=args.frames_dir
                )
                results.append(result)
                
                print(f"Video: {args.video}")
                print(f"Predicted Action: {result['predicted_action']}")
                if 'confidence' in result:
                    print(f"Confidence: {result['confidence']:.4f}")
                if args.save_frames and 'frames_output_dir' in result:
                    print(f"Frames saved to: {result['frames_output_dir']}")
                    print(f"Number of frames saved: {len(result.get('saved_frame_paths', []))}")
            
        elif args.video_dir:
            # Batch inference on directory
            video_dir = Path(args.video_dir)
            video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
            
            logger.info(f"Processing {len(video_files)} videos from {video_dir}")
            
            if args.slice_video:
                # Batch video slicing mode
                logger.info(f"Using video slicing: {args.interval}s intervals, {args.overlap}s overlap")
                results = []
                for video_file in tqdm(video_files, desc="Processing videos with slicing"):
                    result = recognizer.recognize_action_on_slices(
                        str(video_file),
                        args.interval,
                        args.overlap,
                        args.question,
                        save_frames=args.save_frames,
                        frames_output_dir=args.frames_dir
                    )
                    results.append(result)
                
                # Print results summary
                for result in results:
                    print(f"Video: {Path(result['video_path']).name}")
                    print(f"  Total slices: {result['total_slices']}")
                    print(f"  Successful: {result['processing_summary']['successful_slices']}")
                    print(f"  Failed: {result['processing_summary']['failed_slices']}")
                    print("---")
            else:
                # Standard batch inference
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