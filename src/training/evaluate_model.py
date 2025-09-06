"""
Model Evaluation and Testing Utilities
Tools for evaluating trained VideoMAE models on test data
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from torch.utils.data import DataLoader
import yaml
import logging
from tqdm import tqdm
import cv2

from train_videomae import VideoBadmintonDataset

class ModelEvaluator:
    """Evaluate trained VideoMAE models"""
    
    def __init__(self, model_path, config_path="configs/training_config.yaml"):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model
            config_path: Path to training configuration
        """
        self.model_path = Path(model_path)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model and processor
        self._load_model()
        
    def _load_model(self):
        """Load the trained model and processor"""
        try:
            self.processor = VideoMAEImageProcessor.from_pretrained(self.model_path)
            self.model = VideoMAEForVideoClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def evaluate_on_test_set(self, test_data_dir=None, test_annotations=None):
        """
        Evaluate model on test dataset
        
        Args:
            test_data_dir: Path to test videos
            test_annotations: Path to test annotations
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Use config paths if not provided
        if test_data_dir is None:
            test_data_dir = self.config['data']['test_dir']
        if test_annotations is None:
            test_annotations = self.config['data']['test_annotations']
        
        # Create test dataset
        test_dataset = VideoBadmintonDataset(
            data_dir=test_data_dir,
            annotations_file=test_annotations,
            processor=self.processor,
            window_size=self.config['data']['window_size'],
            sample_rate=self.config['data']['sample_rate'],
            is_training=False
        )
        
        # Create data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4
        )
        
        # Evaluate
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        self.logger.info("Starting evaluation on test set...")
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(pixel_values=pixel_values)
                
                # Get predictions and probabilities
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = logits.argmax(dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_targets, all_predictions, all_probabilities)
        
        self.logger.info("Evaluation completed!")
        self._print_metrics(metrics)
        
        return {
            'metrics': metrics,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'targets': all_targets
        }
    
    def _calculate_metrics(self, targets, predictions, probabilities):
        """Calculate evaluation metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(targets, predictions)
        metrics['precision'] = precision_score(targets, predictions, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(targets, predictions, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(targets, predictions, average=None, zero_division=0)
        metrics['recall_per_class'] = recall_score(targets, predictions, average=None, zero_division=0)
        metrics['f1_per_class'] = f1_score(targets, predictions, average=None, zero_division=0)
        
        # AUC-ROC (for hit class)
        if probabilities.shape[1] > 1:
            hit_probs = probabilities[:, 1]  # Probability of hit class
            metrics['auc_roc'] = roc_auc_score(targets, hit_probs)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(targets, predictions)
        
        # Classification report
        metrics['classification_report'] = classification_report(
            targets, predictions, 
            target_names=['No Hit', 'Hit'],
            output_dict=True
        )
        
        return metrics
    
    def _print_metrics(self, metrics):
        """Print evaluation metrics"""
        self.logger.info("=" * 50)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("=" * 50)
        
        self.logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        self.logger.info(f"Precision: {metrics['precision']:.4f}")
        self.logger.info(f"Recall:    {metrics['recall']:.4f}")
        self.logger.info(f"F1 Score:  {metrics['f1']:.4f}")
        
        if 'auc_roc' in metrics:
            self.logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        
        self.logger.info("\nPer-class metrics:")
        classes = ['No Hit', 'Hit']
        for i, class_name in enumerate(classes):
            if i < len(metrics['precision_per_class']):
                self.logger.info(f"{class_name:8} - P: {metrics['precision_per_class'][i]:.4f}, "
                               f"R: {metrics['recall_per_class'][i]:.4f}, "
                               f"F1: {metrics['f1_per_class'][i]:.4f}")
    
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
        probabilities = results['probabilities']
        metrics = results['metrics']
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            metrics['confusion_matrix'], 
            annot=True, 
            fmt='d',
            xticklabels=['No Hit', 'Hit'],
            yticklabels=['No Hit', 'Hit'],
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        
        # 2. ROC Curve
        if probabilities.shape[1] > 1:
            from sklearn.metrics import roc_curve, auc
            
            fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / 'roc_curve.png', dpi=300)
            plt.close()
        
        # 3. Prediction Distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        hit_probs = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        plt.hist(hit_probs[targets == 0], bins=30, alpha=0.7, label='No Hit', color='blue')
        plt.hist(hit_probs[targets == 1], bins=30, alpha=0.7, label='Hit', color='red')
        plt.xlabel('Hit Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.bar(['No Hit', 'Hit'], [np.sum(targets == 0), np.sum(targets == 1)], alpha=0.7)
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'distributions.png', dpi=300)
        plt.close()
        
        self.logger.info(f"Plots saved to {save_dir}")
    
    def test_on_single_video(self, video_path, output_path=None):
        """
        Test model on a single video and output hit predictions
        
        Args:
            video_path: Path to video file
            output_path: Path to save results (optional)
            
        Returns:
            Dictionary with predictions and timestamps
        """
        self.logger.info(f"Testing on single video: {video_path}")
        
        # Extract frames from video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process video in sliding windows
        window_size = self.config['data']['window_size']
        sample_rate = self.config['data']['sample_rate']
        
        frames = []
        frame_timestamps = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_timestamps.append(frame_idx / fps)
            
            frame_idx += 1
        
        cap.release()
        
        # Process frames in sliding windows
        predictions = []
        timestamps = []
        
        half_window = window_size // 2
        
        for i in tqdm(range(half_window, len(frames) - half_window), desc="Processing"):
            window_frames = frames[i - half_window:i + half_window]
            
            # Process with model
            inputs = self.processor(window_frames, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                hit_prob = probabilities[0, 1].item()  # Probability of hit
                
                predictions.append(hit_prob)
                timestamps.append(frame_timestamps[i])
        
        # Find hit points (peaks in probability)
        threshold = 0.5  # Configurable threshold
        hit_points = []
        
        for i, (timestamp, prob) in enumerate(zip(timestamps, predictions)):
            if prob > threshold:
                # Check if this is a local maximum
                is_peak = True
                for j in range(max(0, i-5), min(len(predictions), i+6)):
                    if j != i and predictions[j] > prob:
                        is_peak = False
                        break
                
                if is_peak:
                    hit_points.append({
                        'timestamp': timestamp,
                        'confidence': prob
                    })
        
        result = {
            'video_path': video_path,
            'hit_points': hit_points,
            'all_predictions': predictions,
            'all_timestamps': timestamps,
            'fps': fps
        }
        
        # Save results if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            self.logger.info(f"Results saved to {output_path}")
        
        self.logger.info(f"Found {len(hit_points)} hit points")
        return result

def main():
    """Main evaluation function"""
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    model_path = "models/checkpoints/best_model"
    
    if not Path(model_path).exists():
        logging.error(f"Model not found at {model_path}")
        logging.info("Please train a model first using train_videomae.py")
        return
    
    evaluator = ModelEvaluator(model_path)
    
    # Evaluate on test set
    results = evaluator.evaluate_on_test_set()
    
    # Create plots
    evaluator.plot_results(results)
    
    # Test on single video (if available)
    sample_video = "data/sample_video.mp4"
    if Path(sample_video).exists():
        single_result = evaluator.test_on_single_video(
            sample_video, 
            "single_video_results.json"
        )

if __name__ == "__main__":
    main()