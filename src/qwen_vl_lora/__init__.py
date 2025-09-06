"""
Qwen-VL LoRA Module for Badminton Action Recognition
==================================================

This module provides LoRA (Low-Rank Adaptation) fine-tuning capabilities for Qwen-VL 
to perform badminton action recognition. It includes:

- Dataset preparation for Qwen-VL format
- LoRA training with PEFT
- Inference pipeline for action recognition  
- Integration with existing ShuttleSense pipeline

Usage:
    from src.qwen_vl_lora import QwenVLActionRecognizer, QwenVLEnhancedDetector
    
    # For inference only
    recognizer = QwenVLActionRecognizer("models/checkpoints/qwen_vl_lora/final_model")
    result = recognizer.recognize_action("path/to/video.mp4")
    
    # For integrated pipeline
    detector = QwenVLEnhancedDetector(config)
    results = detector.process_video_with_action_recognition("path/to/video.mp4")

Components:
-----------
- prepare_dataset.py: Convert VideoBadminton_Dataset to Qwen-VL training format
- train_qwen_vl_lora.py: LoRA training script with PEFT integration
- inference_qwen_vl.py: Inference pipeline for trained models
- qwen_vl_integration.py: Integration with existing ShuttleSense pipeline

Configuration:
--------------
See configs/qwen_vl_lora_config.yaml for training and inference settings.

Action Classes:
---------------
The module recognizes 18 badminton action classes in Chinese:
短发球, 斜线飞行, 挑球, 点杀, 挡球, 吊球, 推球, 过渡切球, 切球, 
扑球, 防守高远球, 防守平抽, 高远球, 长发球, 杀球, 平射球, 后场平抽, 短平射
"""

from .inference_qwen_vl import QwenVLActionRecognizer
from .qwen_vl_integration import QwenVLEnhancedDetector, create_qwen_vl_enhanced_detector
from .prepare_dataset import QwenVLDatasetPreparator

__version__ = "1.0.0"
__author__ = "ShuttleSense"

__all__ = [
    "QwenVLActionRecognizer",
    "QwenVLEnhancedDetector", 
    "create_qwen_vl_enhanced_detector",
    "QwenVLDatasetPreparator"
]

# Action class mappings
ACTION_CLASSES = {
    "00_Short Serve": "短发球",
    "01_Cross Court Flight": "斜线飞行",
    "02_Lift": "挑球",
    "03_Tap Smash": "点杀",
    "04_Block": "挡球",
    "05_Drop Shot": "吊球",
    "06_Push Shot": "推球",
    "07_Transitional Slice": "过渡切球",
    "08_Cut": "切球",
    "09_Rush Shot": "扑球",
    "10_Defensive Clear": "防守高远球",
    "11_Defensive Drive": "防守平抽",
    "12_Clear": "高远球",
    "13_Long Serve": "长发球",
    "14_Smash": "杀球",
    "15_Flat Shot": "平射球",
    "16_Rear Court Flat Drive": "后场平抽",
    "17_Short Flat Shot": "短平射"
}

# Default question templates
QUESTION_TEMPLATES = [
    "请识别视频中的羽毛球动作是什么？",
    "视频中展示的是什么羽毛球技术动作？",
    "这个羽毛球视频片段显示的动作类型是？",
    "请分析这个羽毛球动作并给出名称。",
    "请问视频中的球员做的是什么技术动作？",
    "识别一下这个羽毛球动作的类型。",
    "这是什么羽毛球技术？",
    "请描述视频中的羽毛球动作名称。"
]