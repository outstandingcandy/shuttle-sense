"""
VideoMAE Model Training for Badminton Hit Detection
Fine-tuning VideoMAE on VideoBadminton_Dataset for hit point detection
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    VideoMAEImageProcessor, 
    VideoMAEForVideoClassification,
    VideoMAEConfig,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import yaml
import wandb
from datetime import datetime

class VideoBadmintonDataset(Dataset):
    """Dataset class for VideoBadminton_Dataset"""
    
    def __init__(self, data_dir, annotations_file, processor, window_size=16, sample_rate=2, is_training=True):
        """
        Initialize the dataset
        
        Args:
            data_dir: Path to video data directory
            annotations_file: Path to annotations JSON file
            processor: VideoMAE image processor
            window_size: Number of frames per video clip
            sample_rate: Frame sampling rate
            is_training: Whether this is training dataset
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.is_training = is_training
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.samples = self._prepare_samples()
        
    def _prepare_samples(self):
        """Prepare training samples from annotations"""
        samples = []
        
        for video_id, annotation in self.annotations.items():
            video_path = self.data_dir / f"{video_id}.mp4"
            if not video_path.exists():
                continue
                
            # Extract hit points and non-hit segments
            hit_points = annotation.get('hit_points', [])
            video_duration = annotation.get('duration', 0)
            
            # Create positive samples (hit points)
            for hit_time in hit_points:
                samples.append({
                    'video_path': str(video_path),
                    'start_time': max(0, hit_time - 1.0),  # 1 second before hit
                    'end_time': min(video_duration, hit_time + 1.0),  # 1 second after hit
                    'label': 1,  # Hit
                    'hit_time': hit_time
                })
            
            # Create negative samples (non-hit segments)
            for i in range(len(hit_points) + 1):
                start_time = hit_points[i-1] + 2.0 if i > 0 else 0
                end_time = hit_points[i] - 2.0 if i < len(hit_points) else video_duration
                
                if end_time - start_time > 2.0:  # Only if segment is long enough
                    # Sample random non-hit segment
                    segment_start = np.random.uniform(start_time, end_time - 2.0)
                    samples.append({
                        'video_path': str(video_path),
                        'start_time': segment_start,
                        'end_time': segment_start + 2.0,
                        'label': 0,  # Non-hit
                        'hit_time': None
                    })
        
        return samples
    
    def _extract_video_frames(self, video_path, start_time, end_time):
        """Extract frames from video segment"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for frame_idx in range(start_frame, end_frame, self.sample_rate):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        
        # Ensure we have exactly window_size frames
        if len(frames) < self.window_size:
            # Repeat last frame if needed
            while len(frames) < self.window_size:
                frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        elif len(frames) > self.window_size:
            # Sample uniformly
            indices = np.linspace(0, len(frames) - 1, self.window_size).astype(int)
            frames = [frames[i] for i in indices]
        
        return frames
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Extract frames
            frames = self._extract_video_frames(
                sample['video_path'],
                sample['start_time'],
                sample['end_time']
            )
            
            # Process frames
            inputs = self.processor(frames, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)  # Remove batch dimension
            
            return {
                'pixel_values': pixel_values,
                'labels': torch.tensor(sample['label'], dtype=torch.long)
            }
            
        except Exception as e:
            # Return a dummy sample in case of error
            dummy_frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.window_size)]
            inputs = self.processor(dummy_frames, return_tensors="pt")
            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'labels': torch.tensor(0, dtype=torch.long)
            }

class VideoMAETrainer:
    """Trainer class for VideoMAE fine-tuning"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'shuttlesense-training'),
                name=f"videomae-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=config
            )
        
        self._setup_model()
        self._setup_data()
    
    def _setup_model(self):
        """Initialize model and optimizer"""
        model_config = self.config['model']
        
        # Load pre-trained model
        self.processor = VideoMAEImageProcessor.from_pretrained(model_config['base_model'])
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_config['base_model'],
            num_labels=2,  # Hit / No-hit
            ignore_mismatched_sizes=True
        )
        
        self.model.to(self.device)
        
        # Setup optimizer
        training_config = self.config['training']
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def _setup_data(self):
        """Setup data loaders"""
        data_config = self.config['data']
        
        # Training dataset
        self.train_dataset = VideoBadmintonDataset(
            data_dir=data_config['train_dir'],
            annotations_file=data_config['train_annotations'],
            processor=self.processor,
            window_size=data_config['window_size'],
            sample_rate=data_config['sample_rate'],
            is_training=True
        )
        
        # Validation dataset
        self.val_dataset = VideoBadmintonDataset(
            data_dir=data_config['val_dir'],
            annotations_file=data_config['val_annotations'],
            processor=self.processor,
            window_size=data_config['window_size'],
            sample_rate=data_config['sample_rate'],
            is_training=False
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup scheduler
        total_steps = len(self.train_loader) * self.config['training']['epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Collect predictions
            preds = outputs.logits.argmax(dim=-1)
            predictions.extend(preds.cpu().tolist())
            targets.extend(labels.cpu().tolist())
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                
                preds = outputs.logits.argmax(dim=-1)
                predictions.extend(preds.cpu().tolist())
                targets.extend(labels.cpu().tolist())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best.pth')
            # Also save the model for inference
            self.model.save_pretrained(checkpoint_dir / 'best_model')
            self.processor.save_pretrained(checkpoint_dir / 'best_model')
    
    def train(self):
        """Main training loop"""
        best_f1 = 0
        
        for epoch in range(self.config['training']['epochs']):
            self.logger.info(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")
            
            # Train
            train_metrics = self.train_epoch()
            self.logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                           f"F1: {train_metrics['f1']:.4f}, "
                           f"Acc: {train_metrics['accuracy']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            self.logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                           f"F1: {val_metrics['f1']:.4f}, "
                           f"Acc: {val_metrics['accuracy']:.4f}")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'train_f1': train_metrics['f1'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_f1': val_metrics['f1'],
                    'val_accuracy': val_metrics['accuracy'],
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
            
            # Save checkpoint
            is_best = val_metrics['f1'] > best_f1
            if is_best:
                best_f1 = val_metrics['f1']
            
            if epoch % self.config['training']['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
        
        self.logger.info(f"Training completed! Best F1: {best_f1:.4f}")

def main():
    """Main training function"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_path = "configs/training_config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer and start training
    trainer = VideoMAETrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()