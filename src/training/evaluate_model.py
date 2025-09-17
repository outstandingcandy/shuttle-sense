"""
Model Evaluation and Testing Utilities for Qwen-VL
Tools for evaluating trained Qwen-VL models on test data
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from torch.utils.data import Dataset
import yaml
import logging
from tqdm import tqdm
import cv2
from PIL import Image

class QwenVLDataset(Dataset):
    """Dataset class for Qwen-VL evaluation"""
    
    def __init__(self, data_file, videos_dir, processor, tokenizer):
        """
        Initialize dataset
        
        Args:
            data_file: Path to JSON file containing test data
            videos_dir: Directory containing video files
            processor: Qwen-VL processor
            tokenizer: Qwen-VL tokenizer
        """
        self.videos_dir = Path(videos_dir)
        self.processor = processor
        self.tokenizer = tokenizer
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Create action class mapping
        self.action_mapping = {
            "短发球": 0, "斜线飞行": 1, "挑球": 2, "点杀": 3, "挡球": 4,
            "吊球": 5, "推球": 6, "过渡切球": 7, "切球": 8, "扑球": 9,
            "防守高远球": 10, "防守平抽": 11, "高远球": 12, "长发球": 13,
            "杀球": 14, "平射球": 15, "后场平抽": 16, "短平射": 17
        }
        
        # Reverse mapping for evaluation
        self.id_to_action = {v: k for k, v in self.action_mapping.items()}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load video
        video_path = self.videos_dir / item['video']
        frames = self._load_video_frames(video_path)
        
        # Get question and answer
        conversation = item['conversations']
        question = conversation[0]['value'].replace('<video>', '')
        answer = conversation[1]['value']
        
        # Convert answer to class ID
        label = self._extract_action_from_answer(answer)
        
        return {
            'frames': frames,
            'question': question,
            'answer': answer,
            'label': label,
            'video_path': str(video_path)
        }
    
    def _load_video_frames(self, video_path, max_frames=8):
        """Load video frames"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frames.append(frame_pil)
        
        cap.release()
        return frames
    
    def _extract_action_from_answer(self, answer):
        """Extract action class from answer text"""
        for action, class_id in self.action_mapping.items():
            if action in answer:
                return class_id
        return -1  # Unknown action

class ModelEvaluator:
    """Evaluate trained Qwen-VL models"""
    
    def __init__(self, model_path, config_path="config.yaml"):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
        """
        self.model_path = Path(model_path)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model, tokenizer, and processor
        self._load_model()
        
        # Action mapping for evaluation
        self.action_mapping = {
            "短发球": 0, "斜线飞行": 1, "挑球": 2, "点杀": 3, "挡球": 4,
            "吊球": 5, "推球": 6, "过渡切球": 7, "切球": 8, "扑球": 9,
            "防守高远球": 10, "防守平抽": 11, "高远球": 12, "长发球": 13,
            "杀球": 14, "平射球": 15, "后场平抽": 16, "短平射": 17
        }
        self.id_to_action = {v: k for k, v in self.action_mapping.items()}
        
    def _load_model(self):
        """Load the trained model, tokenizer, and processor"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.config.get('qwen_vl_lora', {}).get('bf16', True) else torch.float16,
                device_map="auto"
            )
            self.model.eval()
            
            self.logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def evaluate_on_test_set(self, test_data_file=None, videos_dir=None):
        """
        Evaluate model on test dataset
        
        Args:
            test_data_file: Path to test JSON file
            videos_dir: Directory containing test videos
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Use config paths if not provided
        if test_data_file is None:
            test_data_file = self.config.get('qwen_vl_lora', {}).get('dataset_path', 'data/qwen_vl_dataset') + '/val.json'
        if videos_dir is None:
            videos_dir = self.config.get('qwen_vl_lora', {}).get('dataset_path', 'data/qwen_vl_dataset') + '/videos'
        
        # Create test dataset
        test_dataset = QwenVLDataset(
            data_file=test_data_file,
            videos_dir=videos_dir,
            processor=self.processor,
            tokenizer=self.tokenizer
        )
        
        # Evaluate
        all_predictions = []
        all_predicted_actions = []
        all_targets = []
        all_target_actions = []
        
        self.logger.info("Starting evaluation on test set...")
        
        for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
            item = test_dataset[i]
            
            # Skip items with unknown labels
            if item['label'] == -1:
                continue
            
            # Prepare input
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": item['frames']},
                        {"type": "text", "text": item['question']}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text], 
                videos=[item['frames']], 
                padding=True, 
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.1
                )
                
                # Decode response
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            
            # Extract predicted action
            predicted_label = self._extract_action_from_response(response)
            predicted_action = self.id_to_action.get(predicted_label, "未知动作")
            target_action = self.id_to_action[item['label']]
            
            all_predictions.append(predicted_label)
            all_predicted_actions.append(predicted_action)
            all_targets.append(item['label'])
            all_target_actions.append(target_action)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_targets, all_predictions)
        
        self.logger.info("Evaluation completed!")
        self._print_metrics(metrics)
        
        return {
            'metrics': metrics,
            'predictions': all_predictions,
            'predicted_actions': all_predicted_actions,
            'targets': all_targets,
            'target_actions': all_target_actions
        }
    
    def _extract_action_from_response(self, response):
        """Extract action class from model response"""
        for action, class_id in self.action_mapping.items():
            if action in response:
                return class_id
        
        return -1  # Unknown action
    
    def _calculate_metrics(self, targets, predictions):
        """Calculate evaluation metrics"""
        # Filter out unknown predictions
        valid_mask = (predictions != -1)
        targets_valid = targets[valid_mask]
        predictions_valid = predictions[valid_mask]
        
        if len(targets_valid) == 0:
            self.logger.warning("No valid predictions found!")
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(targets_valid, predictions_valid)
        metrics['precision'] = precision_score(targets_valid, predictions_valid, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(targets_valid, predictions_valid, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(targets_valid, predictions_valid, average='weighted', zero_division=0)
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(targets_valid, predictions_valid, average=None, zero_division=0)
        metrics['recall_per_class'] = recall_score(targets_valid, predictions_valid, average=None, zero_division=0)
        metrics['f1_per_class'] = f1_score(targets_valid, predictions_valid, average=None, zero_division=0)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(targets_valid, predictions_valid)
        
        # Classification report
        action_names = [self.id_to_action[i] for i in range(len(self.action_mapping))]
        metrics['classification_report'] = classification_report(
            targets_valid, predictions_valid, 
            target_names=action_names,
            output_dict=True,
            zero_division=0
        )
        
        return metrics
    
    def _print_metrics(self, metrics):
        """Print evaluation metrics"""
        if not metrics:
            return
            
        self.logger.info("=" * 50)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("=" * 50)
        
        self.logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        self.logger.info(f"Precision: {metrics['precision']:.4f}")
        self.logger.info(f"Recall:    {metrics['recall']:.4f}")
        self.logger.info(f"F1 Score:  {metrics['f1']:.4f}")
        
        self.logger.info("\nTop-5 Action Performance:")
        f1_scores = metrics.get('f1_per_class', [])
        if len(f1_scores) > 0:
            # Get top 5 performing actions
            sorted_indices = np.argsort(f1_scores)[::-1][:5]
            for i, idx in enumerate(sorted_indices):
                action_name = self.id_to_action.get(idx, f"Action_{idx}")
                f1_score_val = f1_scores[idx]
                self.logger.info(f"{i+1:2d}. {action_name:12} - F1: {f1_score_val:.4f}")
    
    def plot_results(self, results, save_dir="evaluation_plots"):
        """
        Create visualization plots for evaluation results
        
        Args:
            results: Results from evaluate_on_test_set()
            save_dir: Directory to save plots
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        targets = results['targets']
        predictions = results['predictions']
        metrics = results['metrics']
        
        if not metrics:
            self.logger.warning("No metrics to plot")
            return
        
        # Set up plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        # 1. Confusion Matrix
        plt.figure(figsize=(12, 10))
        action_names = [self.id_to_action[i] for i in range(len(self.action_mapping))]
        
        # Filter confusion matrix to only include classes that appear in the data
        cm = metrics['confusion_matrix']
        unique_labels = np.unique(np.concatenate([targets, predictions]))
        filtered_action_names = [action_names[i] for i in unique_labels if i < len(action_names)]
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            xticklabels=filtered_action_names,
            yticklabels=filtered_action_names,
            cmap='Blues'
        )
        plt.title('羽毛球动作识别混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Action Performance Bar Chart
        f1_scores = metrics.get('f1_per_class', [])
        if len(f1_scores) > 0:
            plt.figure(figsize=(15, 8))
            
            # Only plot actions that appear in the data
            plot_actions = []
            plot_f1_scores = []
            for i in unique_labels:
                if i < len(action_names) and i < len(f1_scores):
                    plot_actions.append(action_names[i])
                    plot_f1_scores.append(f1_scores[i])
            
            bars = plt.bar(range(len(plot_actions)), plot_f1_scores)
            plt.xlabel('羽毛球动作')
            plt.ylabel('F1分数')
            plt.title('各动作类别F1分数表现')
            plt.xticks(range(len(plot_actions)), plot_actions, rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, plot_f1_scores)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'action_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Plots saved to {save_dir}")

def main():
    """Main evaluation function"""
    logging.basicConfig(level=logging.INFO)
    
    # Update these paths based on your actual model location
    model_path = "models/checkpoints/qwen_vl_lora_512"
    
    if not Path(model_path).exists():
        logging.error(f"Model not found at {model_path}")
        logging.info("Please train a model first or update the model path")
        return
    
    evaluator = ModelEvaluator(model_path)
    
    # Evaluate on test set
    results = evaluator.evaluate_on_test_set()
    
    # Create plots
    if results['metrics']:
        evaluator.plot_results(results)

if __name__ == "__main__":
    main()