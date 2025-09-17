"""
Data Preparation for Qwen-VL LoRA Fine-tuning
Convert VideoBadminton_Dataset to Qwen-VL training format
"""

import json
import os
import random
from pathlib import Path
import cv2
from tqdm import tqdm
import shutil

class QwenVLDatasetPreparator:
    """Prepare VideoBadminton_Dataset for Qwen-VL LoRA fine-tuning"""
    
    def __init__(self, 
                 source_dataset_path="/home/ubuntu/shuttle-sense/VideoBadminton_Dataset",
                 output_path="data/qwen_vl_dataset"):
        self.source_path = Path(source_dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        # Action class mapping (simplified names for better recognition)
        self.action_mapping = {
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
        
        # Various question templates for data augmentation
        self.question_templates = [
            "请识别视频中的羽毛球动作是什么？",
            "视频中展示的是什么羽毛球技术动作？",
            "这个羽毛球视频片段显示的动作类型是？",
            "请分析这个羽毛球动作并给出名称。",
            "请问视频中的球员做的是什么技术动作？",
            "识别一下这个羽毛球动作的类型。",
            "这是什么羽毛球技术？",
            "请描述视频中的羽毛球动作名称。"
        ]
    
    def validate_video(self, video_path):
        """Validate if video file is readable"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return False
            
            # Try to read first frame
            ret, frame = cap.read()
            cap.release()
            return ret and frame is not None
        except:
            return False
    
    def copy_and_validate_videos(self):
        """Copy videos to output directory and validate them"""
        videos_dir = self.output_path / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        valid_videos = []
        invalid_videos = []
        
        print("📹 Processing and validating videos...")
        
        for action_class in tqdm(self.action_mapping.keys(), desc="Processing classes"):
            class_dir = self.source_path / action_class
            if not class_dir.exists():
                continue
            
            video_files = list(class_dir.glob("*.mp4"))
            
            for video_file in tqdm(video_files, desc=f"Processing {action_class}", leave=False):
                # Create new filename with class prefix for uniqueness
                new_filename = f"{action_class}_{video_file.name}"
                dest_path = videos_dir / new_filename
                
                # Copy video
                if not dest_path.exists():
                    shutil.copy2(video_file, dest_path)
                
                # Validate video
                if self.validate_video(dest_path):
                    valid_videos.append({
                        'path': f"videos/{new_filename}",
                        'action_class': action_class,
                        'action_name': self.action_mapping[action_class]
                    })
                else:
                    invalid_videos.append(str(dest_path))
                    if dest_path.exists():
                        dest_path.unlink()  # Remove invalid video
        
        print(f"✅ Valid videos: {len(valid_videos)}")
        print(f"❌ Invalid videos: {len(invalid_videos)}")
        
        return valid_videos
    
    def create_training_data(self, valid_videos, train_ratio=0.8):
        """Create training data in Qwen-VL format"""
        
        # Split data
        random.shuffle(valid_videos)
        split_idx = int(len(valid_videos) * train_ratio)
        
        train_videos = valid_videos[:split_idx]
        val_videos = valid_videos[split_idx:]
        
        print(f"📊 Dataset split:")
        print(f"  Training: {len(train_videos)} videos")
        print(f"  Validation: {len(val_videos)} videos")
        
        # Create training data
        train_data = self._create_conversation_data(train_videos, "training")
        val_data = self._create_conversation_data(val_videos, "validation")
        
        # Save training data
        train_file = self.output_path / "train.json"
        val_file = self.output_path / "val.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Training data saved to: {train_file}")
        print(f"💾 Validation data saved to: {val_file}")
        
        return train_data, val_data
    
    def _create_conversation_data(self, videos, split_type):
        """Create conversation format data for Qwen-VL"""
        conversations = []
        
        for video in tqdm(videos, desc=f"Creating {split_type} data"):
            # Create multiple examples per video with different questions
            num_questions = random.randint(1, 3) if split_type == "training" else 1
            
            for _ in range(num_questions):
                question = random.choice(self.question_templates)
                answer = video['action_name']
                
                # Create different answer variations for training
                if split_type == "training":
                    answer_variations = [
                        answer,
                        f"这是{answer}",
                        f"视频中的动作是{answer}",
                        f"这个动作叫做{answer}",
                        f"球员执行的是{answer}动作"
                    ]
                    answer = random.choice(answer_variations)
                
                conversation = {
                    "id": f"{video['action_class']}_{Path(video['path']).stem}_{random.randint(1000, 9999)}",
                    "video": video['path'],
                    "conversations": [
                        {
                            "from": "user",
                            "value": f"<video>{question}"
                        },
                        {
                            "from": "assistant", 
                            "value": answer
                        }
                    ]
                }
                conversations.append(conversation)
        
        return conversations
    
    def create_class_statistics(self, valid_videos):
        """Create statistics about the dataset"""
        stats = {}
        for video in valid_videos:
            action_class = video['action_class']
            action_name = video['action_name']
            
            if action_name not in stats:
                stats[action_name] = {
                    'class_id': action_class,
                    'count': 0,
                    'videos': []
                }
            
            stats[action_name]['count'] += 1
            stats[action_name]['videos'].append(video['path'])
        
        # Save statistics
        stats_file = self.output_path / "dataset_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n📈 Dataset Statistics:")
        for action_name, stat in sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"  {action_name}: {stat['count']} videos")
        
        return stats
    
    def prepare_dataset(self):
        """Main function to prepare the dataset"""
        print("🏸 Preparing VideoBadminton_Dataset for Qwen-VL LoRA fine-tuning")
        print("=" * 70)
        
        # Step 1: Copy and validate videos
        valid_videos = self.copy_and_validate_videos()
        
        if not valid_videos:
            print("❌ No valid videos found!")
            return False
        
        # Step 2: Create training data
        train_data, val_data = self.create_training_data(valid_videos)
        
        # Step 3: Create statistics
        stats = self.create_class_statistics(valid_videos)
        
        # Step 4: Create README
        self._create_readme(len(train_data), len(val_data), len(stats))
        
        print("\n✅ Dataset preparation completed!")
        print(f"📁 Output directory: {self.output_path}")
        print(f"📊 Training samples: {len(train_data)}")
        print(f"📊 Validation samples: {len(val_data)}")
        
        return True
    
    def _create_readme(self, train_count, val_count, class_count):
        """Create README file for the dataset"""
        readme_content = f"""# Qwen-VL 羽毛球动作识别数据集

## 数据集概述

本数据集基于 VideoBadminton_Dataset 创建，专门用于 Qwen-VL 模型的 LoRA 微调，实现羽毛球动作识别功能。

## 数据统计

- **训练样本**: {train_count} 个
- **验证样本**: {val_count} 个
- **动作类别**: {class_count} 种羽毛球技术动作

## 数据格式

每个训练样本包含：
- 视频路径
- 问答对话（用户问题 + 模型回答）

示例格式：
```json
{{
  "id": "unique_id",
  "video": "videos/video_file.mp4",
  "conversations": [
    {{
      "from": "user",
      "value": "<video>请识别视频中的羽毛球动作是什么？"
    }},
    {{
      "from": "assistant",
      "value": "杀球"
    }}
  ]
}}
```

## 文件结构

```
{self.output_path}/
├── videos/              # 视频文件目录
├── train.json          # 训练数据
├── val.json            # 验证数据
├── dataset_stats.json  # 数据集统计信息
└── README.md           # 本文件
```

## 动作类别

包含 18 种羽毛球技术动作：
{chr(10).join([f'- {chinese}' for chinese in self.action_mapping.values()])}

## 使用说明

1. 确保所有视频文件在 `videos/` 目录下
2. 使用 `train.json` 进行 LoRA 微调训练
3. 使用 `val.json` 进行验证和测试

## 数据质量

- 所有视频文件都已验证可读性
- 问题模板多样化，增强模型泛化能力
- 答案格式标准化，便于模型学习
"""
        
        readme_file = self.output_path / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)

def main():
    """Main function"""
    preparator = QwenVLDatasetPreparator()
    success = preparator.prepare_dataset()
    
    if success:
        print("\n🎯 下一步：")
        print("1. 检查生成的数据：data/qwen_vl_dataset/")
        print("2. 开始 LoRA 微调训练")
        print("3. 运行推理测试")
    else:
        print("❌ 数据集准备失败，请检查源数据路径")

if __name__ == "__main__":
    main()