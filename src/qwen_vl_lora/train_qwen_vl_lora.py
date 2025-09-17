"""
Qwen-VL LoRA Fine-tuning for Badminton Action Recognition
Fine-tune Qwen-VL using LoRA (Low-Rank Adaptation) for action classification
"""

import json
import logging
from pathlib import Path
from typing import List
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from peft import LoraConfig, get_peft_model, TaskType
import yaml
import cv2
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QwenVLTrainingConfig:
    """Configuration for Qwen2.5-VL LoRA fine-tuning following official best practices"""
    
    # Model configuration - Official Qwen2.5-VL model
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # LoRA configuration - Following official recommendations
    lora_r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    # Target modules for Qwen2.5-VL (will be auto-detected)
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training configuration - Optimized for vision-language tasks
    epochs: int = 5
    batch_size: int = 2  # Reduced for vision-language processing
    gradient_accumulation_steps: int = 8  # Increased to maintain effective batch size
    learning_rate: float = 5e-5  # Reduced from 1e-4 for stability
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 2048  # Increased for vision-language conversations
    max_grad_norm: float = 1.0  # Add gradient clipping
    
    # Vision processing configuration
    max_frames: int = 8  # Optimal for Qwen2.5-VL
    frame_size: tuple = (224, 224)
    vision_processing_batch_size: int = 1  # Process videos one at a time
    
    # Data configuration
    dataset_path: str = "data/qwen_vl_dataset"
    
    # Output configuration
    output_dir: str = "models/checkpoints/qwen_vl_lora"
    logging_steps: int = 10
    save_steps: int = 100
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Checkpoint configuration
    resume_from_checkpoint: str = None  # Path to checkpoint to resume from
    auto_find_last_checkpoint: bool = True  # Automatically find last checkpoint in output_dir
    
    # Hardware configuration - Optimized for vision-language training
    fp16: bool = False  # Disabled for numerical stability
    bf16: bool = True   # Use bf16 for better stability than fp16
    gradient_checkpointing: bool = False  # Disabled for video processing stability
    dataloader_num_workers: int = 4  # Increased for video processing
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
    
    # Advanced training settings
    remove_unused_columns: bool = False  # Keep all columns for vision data
    ddp_find_unused_parameters: bool = False
    seed: int = 42

class QwenVLActionDataset(Dataset):
    """Dataset for Qwen-VL action recognition fine-tuning"""
    
    def __init__(self, data_path: str, processor, tokenizer, config: QwenVLTrainingConfig, split: str = "train"):
        self.data_path = Path(data_path)
        self.processor = processor
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        # Cache for extracted frames to avoid repeated video decoding
        self.frame_cache = {}
        
        # Load conversation data
        data_file = self.data_path / f"{split}.json"
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            self.conversations = json.load(f)
        
        logger.info(f"Loaded {len(self.conversations)} conversations for {split}")
        
        # Pre-cache frames for first epoch to improve training speed
        if len(self.conversations) < 1000:  # Only for smaller datasets
            logger.info("Pre-caching video frames for faster training...")
            self._precache_frames()
    
    def __len__(self):
        return len(self.conversations)
    
    def _precache_frames(self):
        """Pre-cache frames for faster training."""
        from tqdm import tqdm
        for conversation in tqdm(self.conversations[:100], desc="Pre-caching frames"):  # Cache first 100
            video_path = conversation["video"]
            if video_path not in self.frame_cache:
                try:
                    frames = self._extract_frames_cached(video_path)
                    self.frame_cache[video_path] = frames
                except Exception as e:
                    logger.warning(f"Failed to cache frames for {video_path}: {e}")
    
    def _extract_frames_cached(self, video_path: str) -> List[Image.Image]:
        """Extract frames with caching."""
        if video_path in self.frame_cache:
            return self.frame_cache[video_path]
        
        frames = self.extract_video_frames(video_path)
        # Cache small datasets
        if len(self.conversations) < 1000:
            self.frame_cache[video_path] = frames
        return frames
    
    def extract_video_frames(self, video_path: str) -> List[Image.Image]:
        """Extract frames from video"""
        full_path = self.data_path / video_path
        
        if not full_path.exists():
            logger.warning(f"Video not found: {full_path}")
            # Return dummy frames
            dummy_frame = Image.new('RGB', self.config.frame_size, (0, 0, 0))
            return [dummy_frame] * min(self.config.max_frames, 4)
        
        cap = cv2.VideoCapture(str(full_path))
        frames = []
        
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {full_path}")
            dummy_frame = Image.new('RGB', self.config.frame_size, (0, 0, 0))
            return [dummy_frame] * min(self.config.max_frames, 4)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract
        if total_frames <= self.config.max_frames:
            frame_indices = list(range(total_frames))
        else:
            # Evenly sample frames
            frame_indices = np.linspace(0, total_frames - 1, self.config.max_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame while preserving aspect ratio
                h, w = frame.shape[:2]
                target_size = self.config.frame_size[0]  # Use the first dimension as target
                
                # Calculate scaling factor to fit the larger dimension to target_size
                scale = target_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                
                # Resize maintaining aspect ratio
                frame = cv2.resize(frame, (new_w, new_h))
                
                # Pad to make it square (224x224)
                # Calculate padding
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
        
        # Ensure we have at least some frames
        if not frames:
            dummy_frame = Image.new('RGB', self.config.frame_size, (0, 0, 0))
            frames = [dummy_frame] * min(self.config.max_frames, 4)
        
        return frames[:self.config.max_frames]
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        
        # Extract video frames with caching
        video_frames = self._extract_frames_cached(conversation["video"])
        
        # Get conversation pairs
        user_message = conversation["conversations"][0]["value"]
        assistant_message = conversation["conversations"][1]["value"]
        
        # Create proper vision-language conversation format following official pattern
        question = user_message.replace("<video>", "").strip()
        
        # Format messages using official Qwen2.5-VL conversation pattern
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_frames},  # Pass video frames directly
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant", 
                "content": assistant_message
            }
        ]
        
        # Use processor's chat template (official approach)
        try:
            # Apply chat template with both user and assistant messages
            full_text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # Apply chat template for input only (user message)
            input_text = self.processor.apply_chat_template(
                messages[:1], 
                tokenize=False, 
                add_generation_prompt=True
            )
            
        except Exception as e:
            logger.warning(f"Chat template failed: {e}, falling back to manual formatting")
            # Fallback to manual formatting
            input_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            full_text = f"{input_text}{assistant_message}<|im_end|>"
        
        # Process with the official processor
        try:
            # For training, we process the full conversation
            inputs = self.processor(
                text=full_text,  # Pass as single string, not list
                videos=video_frames,  # Pass frames directly without list wrapper
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            # Get input-only encoding for label masking
            input_only = self.processor(
                text=input_text,  # Pass as single string, not list
                videos=video_frames,  # Pass frames directly without list wrapper
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
        except Exception as e:
            logger.warning(f"Processor failed: {e}, using tokenizer only")
            # Fallback to tokenizer only
            inputs = self.processor.tokenizer(
                full_text,
                truncation=True,
                max_length=self.config.max_length,
                padding=False,
                return_tensors="pt"
            )
            input_only = self.processor.tokenizer(
                input_text,
                truncation=True,
                max_length=self.config.max_length,
                padding=False,
                return_tensors="pt"
            )
        
        # Extract tensors and convert to lists for data collator
        try:
            if torch.is_tensor(inputs["input_ids"]):
                # Handle tensor shape properly - only squeeze if batch dimension exists
                input_ids_tensor = inputs["input_ids"]
                attention_mask_tensor = inputs["attention_mask"]
                
                if input_ids_tensor.dim() > 1:
                    input_ids = input_ids_tensor.squeeze(0).tolist()
                    attention_mask = attention_mask_tensor.squeeze(0).tolist()
                else:
                    input_ids = input_ids_tensor.tolist()
                    attention_mask = attention_mask_tensor.tolist()
            else:
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
        except (KeyError, TypeError) as e:
            logger.error(f"Failed to extract input_ids: {e}")
            raise ValueError(f"Invalid processor output: {e}")
        
        # Create labels (mask input tokens) - ensure length matches input_ids
        labels = input_ids[:]
        try:
            if torch.is_tensor(input_only["input_ids"]):
                input_only_tensor = input_only["input_ids"]
                if input_only_tensor.dim() > 1:
                    input_len = len(input_only_tensor.squeeze(0).tolist())
                else:
                    input_len = len(input_only_tensor.tolist())
            else:
                input_len = len(input_only["input_ids"])
        except (KeyError, TypeError):
            input_len = len(input_ids) // 2  # Rough estimate
        
        # Ensure labels and input_ids have the same length
        if len(labels) != len(input_ids):
            logger.warning(f"Length mismatch: input_ids={len(input_ids)}, labels={len(labels)}, adjusting...")
            if len(labels) < len(input_ids):
                # Pad labels to match input_ids length
                labels.extend([-100] * (len(input_ids) - len(labels)))
            else:
                # Truncate labels to match input_ids length  
                labels = labels[:len(input_ids)]
        
        # Mask input tokens in labels
        labels[:input_len] = [-100] * input_len
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "conversation_id": conversation.get("id", f"conv_{idx}")
        }
        
        # Add vision features if available - include all video-specific tensors
        video_tensor_keys = ["pixel_values", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]
        
        for key in video_tensor_keys:
            if key in inputs and inputs[key] is not None:
                result[key] = inputs[key]
        
        return result

class QwenVLLoRATrainer:
    """Trainer for Qwen-VL LoRA fine-tuning"""
    
    def __init__(self, config: QwenVLTrainingConfig):
        self.config = config
        self.setup_model_and_tokenizer()
        self.setup_datasets()
        
    def setup_model_and_tokenizer(self):
        """Setup model and processor following official Qwen2.5-VL patterns"""
        logger.info(f"Loading model and processor: {self.config.model_name}")
        
        # Load processor first (official recommended approach)
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                use_fast=False  # Use slow processor for compatibility with saved checkpoints
            )
            logger.info("✅ Processor loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load processor: {e}")
            raise RuntimeError(f"Could not load Qwen2VL processor: {e}")
        
        # Access tokenizer through processor (official pattern)
        self.tokenizer = self.processor.tokenizer
        
        # Load model with official recommended settings
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype="auto",  # Official recommendation
                device_map="auto",   # Official recommendation
                trust_remote_code=True
            )
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model with auto settings: {e}")
            logger.info("Trying fallback loading...")
            # Fallback without device_map
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        # Setup LoRA
        self.setup_lora()
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
            # Enable input gradients for gradient checkpointing compatibility
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                
                self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
            logger.info("✅ Gradient checkpointing enabled with input gradients")
            
    def setup_lora(self):
        """Setup LoRA configuration"""
        logger.info("Setting up LoRA configuration")
        
        # Print all available modules for debugging
        available_modules = []
        logger.info("Available model modules with weights:")
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                available_modules.append(name)
                if any(pattern in name for pattern in ['proj', 'linear', 'fc', 'attn']):
                    logger.info(f"  {name}: {type(module).__name__} - {module.weight.shape if hasattr(module, 'weight') else 'No weight'}")
        
        # Find correct target modules for Qwen2.5-VL
        potential_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        target_modules = []
        
        for pattern in potential_targets:
            matching = [name for name in available_modules if pattern in name and 'language_model' in name]
            if matching:
                target_modules.extend(matching)
                logger.info(f"Found modules for {pattern}: {matching[:3]}...")  # Show first 3
        
        # If no language_model modules found, try without that constraint
        if not target_modules:
            logger.warning("No language_model modules found, trying broader search...")
            for pattern in ["q_proj", "v_proj"]:  # Conservative fallback
                matching = [name for name in available_modules if pattern in name]
                if matching:
                    target_modules.extend(matching[:5])  # Limit to avoid too many
        
        if not target_modules:
            # Ultimate fallback - find any linear layers
            logger.warning("No projection modules found, using any Linear layers...")
            target_modules = [name for name in available_modules if 'Linear' in str(type(self.model.get_submodule(name)))][:10]
        
        logger.info(f"Selected target modules ({len(target_modules)}): {target_modules[:5]}...")
        
        if not target_modules:
            raise RuntimeError("Could not find any suitable modules for LoRA adaptation!")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        # Apply LoRA to model
        try:
            self.model = get_peft_model(self.model, lora_config)
            logger.info("LoRA applied successfully")
        except Exception as e:
            logger.error(f"Failed to apply LoRA: {e}")
            # Try with just the first few modules
            logger.info("Trying with reduced target modules...")
            lora_config.target_modules = target_modules[:3] if len(target_modules) > 3 else target_modules
            self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        # Explicitly set model to training mode
        self.model.train()
        
        # Verify we have trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        if trainable_params == 0:
            raise RuntimeError("No trainable parameters found after LoRA setup!")
        
        logger.info(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        # Log some trainable parameter names for verification
        trainable_names = [name for name, param in self.model.named_parameters() if param.requires_grad]
        logger.info(f"Trainable parameter examples: {trainable_names[:5]}")
        
    def verify_gradients(self):
        """Verify that model has trainable parameters with gradients"""
        trainable_params = []
        frozen_params = []
        trainable_param_count = 0
        frozen_param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
                trainable_param_count += param.numel()
            else:
                frozen_params.append(name)
                frozen_param_count += param.numel()
        
        total_param_count = trainable_param_count + frozen_param_count
        trainable_percentage = 100 * trainable_param_count / total_param_count if total_param_count > 0 else 0
        
        logger.info(f"Gradient verification:")
        logger.info(f"  Trainable parameters: {len(trainable_params):,} layers, {trainable_param_count:,} elements ({trainable_percentage:.2f}%)")
        logger.info(f"  Frozen parameters: {len(frozen_params):,} layers, {frozen_param_count:,} elements")
        
        if not trainable_params:
            raise RuntimeError("No parameters require gradients! Training will fail.")
        
        # Show some examples
        logger.info(f"  Trainable examples: {trainable_params[:5]}")
        if len(trainable_params) > 5:
            logger.info(f"    ... and {len(trainable_params) - 5} more")
        
        # Verify specific LoRA parameters exist
        lora_params = [name for name in trainable_params if 'lora_' in name]
        lora_param_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'lora_' in name:
                lora_param_count += param.numel()
                
        if lora_params:
            logger.info(f"  LoRA parameters found: {len(lora_params)} layers, {lora_param_count:,} elements")
            logger.info(f"  LoRA examples: {lora_params[:3]}")
        else:
            logger.warning("  No LoRA parameters found in trainable parameters!")
        
        return len(trainable_params) > 0
    
    def find_last_checkpoint(self, output_dir: str) -> str:
        """Find the last checkpoint in the output directory"""
        output_path = Path(output_dir)
        
        if not output_path.exists():
            return None
        
        # Look for checkpoint directories
        checkpoint_dirs = []
        for item in output_path.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step_num = int(item.name.split("-")[1])
                    checkpoint_dirs.append((step_num, item))
                except (ValueError, IndexError):
                    continue
        
        if not checkpoint_dirs:
            return None
        
        # Return the checkpoint with the highest step number
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: x[0])[1]
        
        # Verify it's a valid checkpoint
        if (latest_checkpoint / "pytorch_model.bin").exists() or \
           (latest_checkpoint / "model.safetensors").exists() or \
           (latest_checkpoint / "adapter_model.safetensors").exists():
            return str(latest_checkpoint)
        
        return None
    
    def load_from_checkpoint(self, checkpoint_path: str):
        """Load model state from checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        try:
            # Load LoRA weights if available
            adapter_model_path = checkpoint_path / "adapter_model.bin"
            adapter_config_path = checkpoint_path / "adapter_config.json"
            
            if adapter_model_path.exists() and adapter_config_path.exists():
                logger.info("Loading LoRA adapter weights...")
                
                # Load the adapter config to verify compatibility
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                
                # Load adapter weights
                adapter_weights = torch.load(adapter_model_path, map_location="cpu")
                
                # Apply weights to the model
                missing_keys, unexpected_keys = self.model.load_state_dict(adapter_weights, strict=False)
                
                if missing_keys:
                    logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
                
                logger.info("✅ LoRA adapter weights loaded successfully")
            else:
                logger.warning("LoRA adapter weights not found in checkpoint, using base model")
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def get_resume_checkpoint_path(self) -> str:
        """Get the checkpoint path to resume from"""
        if self.config.resume_from_checkpoint:
            # Explicit checkpoint path provided
            checkpoint_path = self.config.resume_from_checkpoint
            if Path(checkpoint_path).exists():
                logger.info(f"Using explicit checkpoint: {checkpoint_path}")
                return checkpoint_path
            else:
                logger.warning(f"Explicit checkpoint not found: {checkpoint_path}")
        
        if self.config.auto_find_last_checkpoint:
            # Auto-find last checkpoint
            last_checkpoint = self.find_last_checkpoint(self.config.output_dir)
            if last_checkpoint:
                logger.info(f"Auto-found last checkpoint: {last_checkpoint}")
                return last_checkpoint
            else:
                logger.info("No previous checkpoints found, starting fresh training")
        
        return None
    
    def list_checkpoints(self, output_dir: str = None) -> List[str]:
        """List all available checkpoints in the output directory"""
        if output_dir is None:
            output_dir = self.config.output_dir
            
        output_path = Path(output_dir)
        
        if not output_path.exists():
            return []
        
        checkpoints = []
        for item in output_path.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step_num = int(item.name.split("-")[1])
                    # Verify it's a valid checkpoint
                    if (item / "pytorch_model.bin").exists() or \
                       (item / "model.safetensors").exists() or \
                       (item / "adapter_model.bin").exists():
                        checkpoints.append((step_num, str(item)))
                except (ValueError, IndexError):
                    continue
        
        # Sort by step number
        checkpoints.sort(key=lambda x: x[0])
        return [ckpt[1] for ckpt in checkpoints]
    
    def print_checkpoint_info(self):
        """Print information about available checkpoints"""
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            logger.info("No checkpoints found in output directory")
            return
        
        logger.info(f"Found {len(checkpoints)} checkpoints:")
        for i, ckpt_path in enumerate(checkpoints):
            ckpt_name = Path(ckpt_path).name
            step_num = ckpt_name.split("-")[1]
            
            # Check checkpoint size and modification time
            try:
                ckpt_stat = Path(ckpt_path).stat()
                mod_time = ckpt_stat.st_mtime
                from datetime import datetime
                mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                
                is_latest = (i == len(checkpoints) - 1)
                status = " (LATEST)" if is_latest else ""
                
                logger.info(f"  {i+1:2d}. {ckpt_name} - Step {step_num} - {mod_time_str}{status}")
            except:
                logger.info(f"  {i+1:2d}. {ckpt_name} - Step {step_num}")
        
        # Show which one will be used for auto-resume
        if self.config.auto_find_last_checkpoint:
            latest = self.find_last_checkpoint(self.config.output_dir)
            if latest:
                logger.info(f"Auto-resume will use: {Path(latest).name}")
        
        return checkpoints
        
    def setup_datasets(self):
        """Setup training and validation datasets"""
        logger.info("Setting up datasets")
        
        self.train_dataset = QwenVLActionDataset(
            self.config.dataset_path,
            self.processor,
            self.tokenizer,
            self.config,
            split="train"
        )
        
        self.eval_dataset = QwenVLActionDataset(
            self.config.dataset_path,
            self.processor,
            self.tokenizer,
            self.config,
            split="val"
        )
        
    def data_collator(self, batch):
        """Custom data collator for vision-language training"""
        # Extract text features
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Find max length
        max_len = max(len(ids) for ids in input_ids)
        
        # Pad sequences to max length
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for i in range(len(batch)):
            ids = input_ids[i]
            mask = attention_masks[i]
            label = labels[i]
            
            pad_len = max_len - len(ids)
            
            if pad_len > 0:
                # Pad with appropriate tokens
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                ids = ids + [pad_token_id] * pad_len
                mask = mask + [0] * pad_len
                label = label + [-100] * pad_len
            
            padded_input_ids.append(ids)
            padded_attention_masks.append(mask)
            padded_labels.append(label)
        
        # Create result dictionary
        result = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long)
        }

        # Handle vision features if present
        pixel_values = [item.get("pixel_values") for item in batch]
        if any(pv is not None for pv in pixel_values):
            # Stack pixel values if present
            valid_pixel_values = [pv for pv in pixel_values if pv is not None]
            if valid_pixel_values:
                try:
                    # Convert to tensors and stack
                    if torch.is_tensor(valid_pixel_values[0]):
                        result["pixel_values"] = torch.stack(valid_pixel_values)
                    else:
                        result["pixel_values"] = torch.stack([torch.tensor(pv) for pv in valid_pixel_values])
                except Exception as e:
                    logger.warning(f"Failed to stack pixel values: {e}")

        # Handle video-specific tensors for Qwen2.5-VL
        video_tensors = ["pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]
        
        for tensor_name in video_tensors:
            tensor_values = [item.get(tensor_name) for item in batch]
            if any(tv is not None for tv in tensor_values):
                valid_tensor_values = [tv for tv in tensor_values if tv is not None]
                if valid_tensor_values:
                    try:
                        # Special handling for video_grid_thw and metadata tensors
                        if tensor_name in ["video_grid_thw", "second_per_grid_ts"]:
                            if torch.is_tensor(valid_tensor_values[0]):
                                # Ensure metadata tensors don't require gradients
                                if tensor_name == "video_grid_thw":
                                    result[tensor_name] = torch.cat(valid_tensor_values, dim=0).detach()
                                else:
                                    result[tensor_name] = torch.stack(valid_tensor_values).detach()
                            else:
                                if tensor_name == "video_grid_thw":
                                    result[tensor_name] = torch.cat([torch.tensor(tv, requires_grad=False) for tv in valid_tensor_values], dim=0)
                                else:
                                    result[tensor_name] = torch.stack([torch.tensor(tv, requires_grad=False) for tv in valid_tensor_values])
                        else:
                            # For other tensors (like pixel_values_videos), preserve gradient flow
                            if torch.is_tensor(valid_tensor_values[0]):
                                # Ensure pixel_values_videos can flow gradients for training
                                if tensor_name == "pixel_values_videos":
                                    result[tensor_name] = torch.stack(valid_tensor_values, dim=0).squeeze(1)
                                else:
                                    result[tensor_name] = torch.stack(valid_tensor_values)
                            else:
                                result[tensor_name] = torch.stack([torch.tensor(tv) for tv in valid_tensor_values])
                    except Exception as e:
                        # For non-stackable tensors, try concatenation
                        try:
                            if torch.is_tensor(valid_tensor_values[0]):
                                result[tensor_name] = torch.cat(valid_tensor_values, dim=0)
                            else:
                                result[tensor_name] = torch.cat([torch.tensor(tv) for tv in valid_tensor_values], dim=0)
                        except Exception as e2:
                            logger.warning(f"Failed to process {tensor_name}: {e}, {e2}")

        # Handle video grid metadata if present (legacy support)
        video_grid_thw = [item.get("video_grid_thw") for item in batch]
        if any(vgt is not None for vgt in video_grid_thw) and "video_grid_thw" not in result:
            valid_video_grid = [vgt for vgt in video_grid_thw if vgt is not None]
            if valid_video_grid:
                try:
                    if torch.is_tensor(valid_video_grid[0]):
                        result["video_grid_thw"] = torch.stack(valid_video_grid)
                    else:
                        result["video_grid_thw"] = torch.stack([torch.tensor(vgt) for vgt in valid_video_grid])
                except Exception as e:
                    logger.warning(f"Failed to stack video grid: {e}")
        
        return result
        
    def train(self):
        """Start training"""
        logger.info("Starting Qwen-VL LoRA training")
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for resume checkpoint
        resume_checkpoint = self.get_resume_checkpoint_path()
        
        # Print checkpoint information
        self.print_checkpoint_info()
        
        # Save config
        config_dict = {
            "model_name": self.config.model_name,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "target_modules": self.config.target_modules,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "resume_from_checkpoint": resume_checkpoint,
        }
        
        with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        # Training arguments - Following official best practices
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,  # Add gradient clipping
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            dataloader_prefetch_factor=self.config.dataloader_prefetch_factor,
            report_to=[],  # Disable external logging for simplicity
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Advanced settings
            remove_unused_columns=self.config.remove_unused_columns,
            ddp_find_unused_parameters=self.config.ddp_find_unused_parameters,
            seed=self.config.seed,
            # Memory optimizations
            optim="adamw_torch",  # Official recommendation
            lr_scheduler_type="cosine",  # Better for fine-tuning
            # Checkpoint settings
            resume_from_checkpoint=resume_checkpoint,
        )
        
        # Create trainer with proper processor integration
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.processor,  # Use processor instead of tokenizer
            data_collator=self.data_collator,
        )
        
        # Start training
        logger.info(f"Training on {len(self.train_dataset)} samples")
        logger.info(f"Validation on {len(self.eval_dataset)} samples")
        
        if resume_checkpoint:
            logger.info(f"🔄 Resuming training from checkpoint: {resume_checkpoint}")
        else:
            logger.info("🆕 Starting fresh training")
        
        # Verify gradients before training
        logger.info("Performing final gradient verification...")
        if not self.verify_gradients():
            raise RuntimeError("Gradient verification failed!")
        
        # Start training (will automatically resume from checkpoint if provided)
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        # Save final model
        final_model_path = output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        
        # Save processor (includes tokenizer)
        self.processor.save_pretrained(str(final_model_path))
        
        logger.info(f"Training completed! Model and processor saved to {final_model_path}")
        
        return trainer

def load_training_config(config_path: str = None) -> QwenVLTrainingConfig:
    """Load training configuration from YAML file"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Update config with values from YAML
        config = QwenVLTrainingConfig()
        for key, value in config_dict.get('qwen_vl_lora', {}).items():
            if hasattr(config, key):
                # Handle type conversion for specific fields
                if key == 'learning_rate' and isinstance(value, str):
                    value = float(value)
                elif key == 'frame_size' and isinstance(value, list):
                    value = tuple(value)
                elif key in ['resume_from_checkpoint'] and value == "":
                    value = None  # Handle empty strings as None
                elif key in ['auto_find_last_checkpoint'] and isinstance(value, str):
                    value = value.lower() in ['true', '1', 'yes', 'on']  # Convert string to bool
                setattr(config, key, value)
        
        return config
    else:
        return QwenVLTrainingConfig()

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Qwen-VL LoRA for badminton action recognition")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--dataset-path", type=str, help="Dataset path")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    
    # Checkpoint arguments
    parser.add_argument("--resume-from-checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--auto-find-checkpoint", action="store_true", help="Automatically find and resume from last checkpoint")
    parser.add_argument("--no-auto-find-checkpoint", action="store_true", help="Disable automatic checkpoint detection")
    parser.add_argument("--list-checkpoints", action="store_true", help="List available checkpoints and exit")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_training_config(args.config)
    
    # Override with command line arguments (only if provided)
    if args.dataset_path:
        config.dataset_path = args.dataset_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    # Handle checkpoint arguments
    if args.resume_from_checkpoint:
        config.resume_from_checkpoint = args.resume_from_checkpoint
        config.auto_find_last_checkpoint = False  # Disable auto-find when explicit path provided
    elif args.auto_find_checkpoint:
        config.auto_find_last_checkpoint = True
    elif args.no_auto_find_checkpoint:
        config.auto_find_last_checkpoint = False
    
    # Handle list checkpoints option
    if args.list_checkpoints:
        logger.info("Listing available checkpoints...")
        try:
            # Create a temporary trainer just to list checkpoints
            trainer_obj = QwenVLLoRATrainer(config)
            checkpoints = trainer_obj.print_checkpoint_info()
            if not checkpoints:
                logger.info("No checkpoints found in the output directory.")
            return
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return
    
    logger.info(f"Training configuration:")
    logger.info(f"  Dataset path: {config.dataset_path}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  LoRA r: {config.lora_r}")
    logger.info(f"  Resume from checkpoint: {config.resume_from_checkpoint}")
    logger.info(f"  Auto-find last checkpoint: {config.auto_find_last_checkpoint}")
    
    try:
        # Create trainer and start training
        trainer_obj = QwenVLLoRATrainer(config)
        trainer_obj.train()
        
        logger.info("✅ Qwen-VL LoRA training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()