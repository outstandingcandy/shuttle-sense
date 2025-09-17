"""
Hit Detection Module for ShuttleSense
Unified hit point detector with Qwen-VL LoRA as primary model and VideoMAE as fallback
"""

from .detector import HitPointDetector

__all__ = ['HitPointDetector']

__version__ = "2.1.0"