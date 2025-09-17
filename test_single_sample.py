#!/usr/bin/env python3
"""
Test video processing with a single sample
"""

import sys
from pathlib import Path
import torch
import cv2
from PIL import Image
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qwen_vl_lora.train_qwen_vl_lora import QwenVLActionDataset, QwenVLTrainingConfig

def test_single_video_sample():
    """Test processing a single video sample"""
    
    print("Setting up config and dataset...")
    config = QwenVLTrainingConfig()
    config.dataset_path = "data/qwen_vl_dataset"
    config.batch_size = 1
    
    # Create dataset
    from transformers import Qwen2VLProcessor
    processor = Qwen2VLProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", 
        use_fast=False  # Use slow processor for compatibility
    )
    tokenizer = processor.tokenizer
    
    dataset = QwenVLActionDataset(
        config.dataset_path,
        processor,
        tokenizer,
        config,
        split="train"
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a single sample
    print("\n--- Processing sample 0 ---")
    sample = dataset[100]
    
    print("Sample keys:", list(sample.keys()))
    for key, value in sample.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape}, dtype={value.dtype}, requires_grad={value.requires_grad}")
            if key == "labels":
                print(f"    Labels content: {value[:20]}...")  # First 20 elements
        else:
            print(f"  {key}: {type(value)} - {value}")
    
    # Test if labels have valid values
    labels = sample.get("labels", [])
    if torch.is_tensor(labels):
        valid_labels = (labels != -100).sum()
        total_labels = len(labels)
        print(f"  Labels: {valid_labels}/{total_labels} valid ({100*valid_labels/total_labels:.1f}%)")
        
        # Check if we have any valid labels
        if valid_labels == 0:
            print("  WARNING: No valid labels found - this will cause gradient issues!")

if __name__ == "__main__":
    test_single_video_sample()