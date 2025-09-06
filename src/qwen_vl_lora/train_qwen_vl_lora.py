"""
Qwen-VL LoRA Fine-tuning for Badminton Action Recognition
Fine-tune Qwen-VL using LoRA (Low-Rank Adaptation) for action classification
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
import yaml
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QwenVLTrainingConfig:
    """Configuration for Qwen-VL LoRA training"""
    
    # Model configuration
    model_name: str = "Qwen/Qwen-VL-Chat"
    
    # LoRA configuration
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj"])
    
    # Training configuration
    epochs: int = 5
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 2048
    
    # Data configuration
    dataset_path: str = "data/qwen_vl_dataset"
    max_frames: int = 8  # Maximum frames to extract from video
    frame_size: tuple = (224, 224)
    
    # Output configuration
    output_dir: str = "models/checkpoints/qwen_vl_lora"
    logging_steps: int = 10
    save_steps: int = 100
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Hardware configuration
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4

class QwenVLActionDataset(Dataset):
    """Dataset for Qwen-VL action recognition fine-tuning"""
    
    def __init__(self, data_path: str, processor, tokenizer, config: QwenVLTrainingConfig, split: str = "train"):
        self.data_path = Path(data_path)
        self.processor = processor
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        # Load conversation data
        data_file = self.data_path / f"{split}.json"
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            self.conversations = json.load(f)
        
        logger.info(f"Loaded {len(self.conversations)} conversations for {split}")
    
    def __len__(self):
        return len(self.conversations)
    
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
                # Resize frame
                frame = cv2.resize(frame, self.config.frame_size)
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
        
        # Extract video frames
        video_frames = self.extract_video_frames(conversation["video"])
        
        # Get conversation pairs
        user_message = conversation["conversations"][0]["value"]
        assistant_message = conversation["conversations"][1]["value"]
        
        # Create input text (remove <video> token as we'll handle images separately)
        question = user_message.replace("<video>", "").strip()
        
        # Format input and target
        input_text = f"用户: {question}\n助手:"
        target_text = f"{assistant_message}"
        full_text = f"{input_text} {target_text}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
            return_tensors=None
        )
        
        targets = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
            return_tensors=None
        )
        
        # Create labels (mask input tokens)
        labels = targets["input_ids"].copy()
        input_len = len(inputs["input_ids"])
        labels[:input_len] = [-100] * input_len  # Ignore input tokens in loss
        
        return {
            "input_ids": targets["input_ids"],
            "attention_mask": targets["attention_mask"],
            "labels": labels,
            "images": video_frames,  # Pass frames for potential future use
            "conversation_id": conversation.get("id", f"conv_{idx}")
        }

class QwenVLLoRATrainer:
    """Trainer for Qwen-VL LoRA fine-tuning"""
    
    def __init__(self, config: QwenVLTrainingConfig):
        self.config = config
        self.setup_model_and_tokenizer()
        self.setup_datasets()
        
    def setup_model_and_tokenizer(self):
        """Setup model, tokenizer, and processor"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Load processor (for potential image handling)
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
        except:
            self.processor = None
            logger.warning("Could not load processor, using tokenizer only")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Setup LoRA
        self.setup_lora()
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
    def setup_lora(self):
        """Setup LoRA configuration"""
        logger.info("Setting up LoRA configuration")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
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
        """Custom data collator for batching"""
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        attention_masks = [torch.tensor(item["attention_mask"]) for item in batch]
        labels = [torch.tensor(item["labels"]) for item in batch]
        
        # Pad sequences
        max_len = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for i in range(len(batch)):
            ids = input_ids[i]
            mask = attention_masks[i]
            label = labels[i]
            
            pad_len = max_len - len(ids)
            
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
                mask = torch.cat([mask, torch.zeros(pad_len)])
                label = torch.cat([label, torch.full((pad_len,), -100)])
            
            padded_input_ids.append(ids)
            padded_attention_masks.append(mask)
            padded_labels.append(label)
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "labels": torch.stack(padded_labels)
        }
        
    def train(self):
        """Start training"""
        logger.info("Starting Qwen-VL LoRA training")
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
        }
        
        with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            report_to=[],  # Disable wandb for now
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        
        # Start training
        logger.info(f"Training on {len(self.train_dataset)} samples")
        logger.info(f"Validation on {len(self.eval_dataset)} samples")
        
        trainer.train()
        
        # Save final model
        final_model_path = output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        logger.info(f"Training completed! Model saved to {final_model_path}")
        
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
                setattr(config, key, value)
        
        return config
    else:
        return QwenVLTrainingConfig()

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Qwen-VL LoRA for badminton action recognition")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--dataset-path", type=str, default="data/qwen_vl_dataset", help="Dataset path")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints/qwen_vl_lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_training_config(args.config)
    
    # Override with command line arguments
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
    
    logger.info(f"Training configuration:")
    logger.info(f"  Dataset path: {config.dataset_path}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  LoRA r: {config.lora_r}")
    
    try:
        # Create trainer and start training
        trainer_obj = QwenVLLoRATrainer(config)
        trainer = trainer_obj.train()
        
        logger.info("✅ Qwen-VL LoRA training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()