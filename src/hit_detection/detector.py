"""
ShuttleSense Hit Point Detector
Redesigned based on Qwen2.5-VL official best practices
"""

import logging
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

logger = logging.getLogger(__name__)

class HitPointDetector:
    """
    Modern hit point detector using Qwen2.5-VL with proper vision-language integration.
    Follows Hugging Face official best practices for Qwen2.5-VL-3B-Instruct.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the detector with optimized Qwen2.5-VL configuration."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
        self.sample_rate = config.get("sample_rate", 4)
        self.window_size = config.get("window_size", 8)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Vision processing parameters (official recommendations)
        self.min_pixels = config.get("min_pixels", 256 * 28 * 28)
        self.max_pixels = config.get("max_pixels", 1280 * 28 * 28)
        
        # Action classes for badminton
        self.action_classes = [
            "clear", "drop", "drive", "lob", "net", "smash", "serve", "no_action"
        ]
        
        # Model components
        self.model = None
        self.processor = None
        self.model_type = None
        
        # Detection state
        self._detailed_results = {"hit_points": [], "video_info": {}, "model_info": {}}
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Qwen2.5-VL model following official best practices."""
        try:
            self._load_qwen_vl_model()
        except Exception as e:
            self.logger.error(f"Failed to load Qwen-VL model: {e}")
            raise RuntimeError("Failed to load detection model")
    
    def _load_qwen_vl_model(self) -> bool:
        """Load Qwen2.5-VL model following official recommendations."""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from peft import PeftModel
            
            # Check for LoRA checkpoint
            checkpoint_path = "models/checkpoints/qwen_vl_lora/checkpoint-200"
            if Path(checkpoint_path).exists():
                self.logger.info(f"Loading Qwen2.5-VL LoRA model from {checkpoint_path}")
                
                # Load base model with official settings
                base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    torch_dtype="auto",  # Official recommendation
                    device_map="auto",   # Official recommendation
                    trust_remote_code=True
                )
                
                # Load LoRA adapter
                self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
                
                # Load processor with vision parameters
                self.processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels
                )
                
                self.model_type = "qwen_vl_lora"
                self.logger.info("Successfully loaded Qwen2.5-VL LoRA model")
                
            else:
                # Load original model without LoRA
                self.logger.info("Loading original Qwen2.5-VL model")
                
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True
                )
                
                self.processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels
                )
                
                self.model_type = "qwen_vl_base"
                self.logger.info("Successfully loaded base Qwen2.5-VL model")
            
            self.model.eval()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load Qwen2.5-VL model: {e}")
            return False
    
    def detect(self, video_path: str) -> List[float]:
        """
        Detect hit points in video using modern Qwen2.5-VL approach.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of timestamps where hits are detected
        """
        self.logger.info(f"Starting hit detection on {video_path}")
        
        # Reset results
        self._detailed_results = {
            "hit_points": [],
            "video_info": {},
            "model_info": {
                "model_type": self.model_type,
                "device": str(self.device),
                "qwen_vl_available": self.model_type.startswith("qwen_vl")
            }
        }
        
        # Extract frames
        frames, frame_indices, fps = self._extract_frames(video_path)
        if not frames:
            self.logger.warning("No frames extracted from video")
            return []
        
        # Update video info
        self._detailed_results["video_info"] = {
            "total_frames": len(frames),
            "fps": fps,
            "duration": len(frames) / fps if fps > 0 else 0,
            "sample_rate": self.sample_rate
        }
        
        # Process frames in windows
        hit_timestamps = []
        windows = self._create_windows(frames, frame_indices, fps)
        
        self.logger.info(f"Processing {len(windows)} video windows")
        
        for window_data in tqdm(windows, desc="Analyzing video"):
            try:
                result = self._process_window(window_data)
                
                if result['is_hit']:
                    timestamp = window_data['timestamp']
                    hit_timestamps.append(timestamp)
                    
                    # Store detailed results
                    self._detailed_results["hit_points"].append({
                        'timestamp': timestamp,
                        'confidence': result.get('hit_confidence', 0.0),
                        'action_class': result.get('action_class', 'unknown'),
                        'action_confidence': result.get('action_confidence', 0.0),
                        'frame_index': window_data['center_frame_idx']
                    })
                    
            except Exception as e:
                self.logger.warning(f"Error processing window at {window_data['timestamp']:.1f}s: {e}")
                continue
        
        self.logger.info(f"Detection completed. Found {len(hit_timestamps)} hit points")
        return hit_timestamps
    
    def _extract_frames(self, video_path: str) -> tuple:
        """Extract frames from video with proper error handling."""
        frames = []
        frame_indices = []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video file: {video_path}")
            return [], [], 0
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Extracting frames: FPS={fps:.2f}, Total={total_frames}")
        
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Extracting frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.sample_rate == 0:
                # Convert BGR to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_indices.append(frame_idx)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        self.logger.info(f"Extracted {len(frames)} frames from {total_frames} total frames")
        return frames, frame_indices, fps
    
    def _create_windows(self, frames: List, frame_indices: List, fps: float) -> List[Dict]:
        """Create sliding windows for analysis."""
        windows = []
        step_size = max(1, self.window_size // 2)  # 50% overlap
        
        for i in range(0, len(frames) - self.window_size + 1, step_size):
            window_frames = frames[i:i + self.window_size]
            window_indices = frame_indices[i:i + self.window_size]
            
            # Use center frame for timestamp
            center_idx = len(window_frames) // 2
            center_frame_idx = window_indices[center_idx]
            timestamp = center_frame_idx / fps if fps > 0 else 0
            
            windows.append({
                'frames': window_frames,
                'indices': window_indices,
                'timestamp': timestamp,
                'center_frame_idx': center_frame_idx
            })
        
        return windows
    
    def _process_window(self, window_data: Dict) -> Dict[str, Any]:
        """Process a window of frames using Qwen-VL model."""
        return self._process_window_qwen_vl(window_data)
    
    def _process_window_qwen_vl(self, window_data: Dict) -> Dict[str, Any]:
        """Process window using Qwen2.5-VL with multi-frame image analysis following official best practices."""
        try:
            # Use multiple frames for better context - take first, middle, and last frames
            frames = window_data['frames']
            frame_indices = [0, len(frames)//2, len(frames)-1] if len(frames) >= 3 else [len(frames)//2]
            selected_frames = [frames[i] for i in frame_indices]
            images = [Image.fromarray(frame) for frame in selected_frames]
            
            # Prepare messages following official format with multiple images
            action_classes_str = ", ".join(self.action_classes[:-1])  # Exclude "no_action"
            
            # Create content with multiple images
            content = []
            content.append({"type": "video", "video": [""]})
            for img in images:
                content.append({"type": "image", "image": img})
            
            content.append({
                "type": "text", 
                "text": f"Analyze these sequential badminton frames. Is there a hit/stroke happening in this sequence? If yes, classify the stroke type from: {action_classes_str}. Answer with just the stroke type or 'no_action'."
            })
            
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            # Apply chat template (official recommendation)
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs with multiple images (stable approach)
            inputs = self.processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Generate with controlled parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=10,  # Short response expected
                    do_sample=False
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip().lower()
            
            # Parse response
            predicted_class = "no_action"
            confidence = 0.3
            
            # Enhanced matching with Chinese terms
            action_mapping = {
                "clear": ["clear", "高远球", "挑球"],
                "drop": ["drop", "吊球", "放网"],
                "drive": ["drive", "平抽", "平球"],
                "lob": ["lob", "挑球"],
                "net": ["net", "搓球", "放网"],
                "smash": ["smash", "杀球", "扣球"],
                "serve": ["serve", "发球"]
            }
            
            for action, keywords in action_mapping.items():
                if any(keyword in output_text for keyword in keywords):
                    predicted_class = action
                    confidence = 0.85
                    break
            
            is_hit = predicted_class != "no_action"
            
            return {
                'is_hit': is_hit,
                'hit_confidence': confidence if is_hit else 0.0,
                'action_class': predicted_class,
                'action_confidence': confidence,
                'raw_response': output_text
            }
            
        except Exception as e:
            self.logger.warning(f"Qwen-VL processing failed: {e}")
            return {'is_hit': False, 'hit_confidence': 0.0, 'action_class': 'no_action', 'action_confidence': 0.0}
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """Get detailed detection results."""
        return self._detailed_results.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': self.model_type,
            'device': str(self.device),
            'qwen_vl_available': self.model_type.startswith("qwen_vl") if self.model_type else False,
            'action_classes': self.action_classes,
            'config': {
                'confidence_threshold': self.confidence_threshold,
                'sample_rate': self.sample_rate,
                'window_size': self.window_size,
                'min_pixels': self.min_pixels,
                'max_pixels': self.max_pixels
            }
        }