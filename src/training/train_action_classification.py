"""
VideoMAE Training for Badminton Action Classification
Training script adapted for VideoBadminton_Dataset with action categories
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import (
    VideoMAEImageProcessor, 
    VideoMAEForVideoClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import yaml
from datetime import datetime
from sklearn.model_selection import train_test_split

class VideoBadmintonActionDataset(Dataset):
    """Dataset class for VideoBadminton action classification dataset"""
    
    def __init__(self, data_dir, processor, split='train', window_size=16, sample_rate=2, train_ratio=0.8):
        """
        Initialize the dataset
        
        Args:
            data_dir: Path to VideoBadminton_Dataset directory
            processor: VideoMAE image processor
            split: 'train', 'val', or 'test'
            window_size: Number of frames per video clip
            sample_rate: Frame sampling rate
            train_ratio: Ratio for train/val split
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.split = split
        
        # Define action classes
        self.action_classes = [
            "00_Short Serve", "01_Cross Court Flight", "02_Lift", "03_Tap Smash",
            "04_Block", "05_Drop Shot", "06_Push Shot", "07_Transitional Slice",
            "08_Cut", "09_Rush Shot", "10_Defensive Clear", "11_Defensive Drive",
            "12_Clear", "13_Long Serve", "14_Smash", "15_Flat Shot",
            "16_Rear Court Flat Drive", "17_Short Flat Shot"
        ]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.action_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load video files
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load all video samples and create train/val/test splits"""
        all_samples = []
        
        # Collect all video files
        for action_class in self.action_classes:
            class_dir = self.data_dir / action_class
            if not class_dir.exists():
                continue
                
            video_files = list(class_dir.glob("*.mp4"))
            
            for video_file in video_files:
                all_samples.append({
                    'video_path': str(video_file),
                    'label': self.class_to_idx[action_class],
                    'class_name': action_class
                })
        
        # Split data
        if len(all_samples) == 0:
            return []
            
        # Group by class for stratified split
        samples_by_class = {}
        for sample in all_samples:
            class_name = sample['class_name']
            if class_name not in samples_by_class:
                samples_by_class[class_name] = []
            samples_by_class[class_name].append(sample)
        
        # Create stratified splits
        train_samples = []
        val_samples = []
        test_samples = []
        
        for class_name, class_samples in samples_by_class.items():
            n_samples = len(class_samples)
            if n_samples < 3:  # Need at least 3 samples per class
                # Put all in train if too few samples
                train_samples.extend(class_samples)
                continue
            
            # Split: 70% train, 15% val, 15% test
            n_train = max(1, int(n_samples * 0.7))
            n_val = max(1, int(n_samples * 0.15))
            
            # Ensure we have at least 1 sample in each split if possible
            if n_samples >= 6:  # Only do full split if we have enough samples
                np.random.seed(42)  # For reproducibility
                np.random.shuffle(class_samples)
                
                train_samples.extend(class_samples[:n_train])
                val_samples.extend(class_samples[n_train:n_train+n_val])
                test_samples.extend(class_samples[n_train+n_val:])
            else:
                # For small classes, put most in train, 1 in val
                np.random.seed(42)
                np.random.shuffle(class_samples)
                train_samples.extend(class_samples[:-1])
                val_samples.extend(class_samples[-1:])
        
        # Return samples for requested split
        if self.split == 'train':
            return train_samples
        elif self.split == 'val':
            return val_samples
        else:  # test
            return test_samples
    
    def _extract_video_frames(self, video_path):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_count += 1
        
        cap.release()
        
        # Ensure we have exactly window_size frames
        if len(frames) == 0:
            # Create dummy frames if video is empty
            frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.window_size)]
        elif len(frames) < self.window_size:
            # Repeat frames if too few
            while len(frames) < self.window_size:
                frames.extend(frames[:min(len(frames), self.window_size - len(frames))])
        elif len(frames) > self.window_size:
            # Sample uniformly if too many
            indices = np.linspace(0, len(frames) - 1, self.window_size).astype(int)
            frames = [frames[i] for i in indices]
        
        return frames[:self.window_size]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Extract frames
            frames = self._extract_video_frames(sample['video_path'])
            
            # Process frames
            inputs = self.processor(frames, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)  # Remove batch dimension
            
            return {
                'pixel_values': pixel_values,
                'labels': torch.tensor(sample['label'], dtype=torch.long),
                'video_path': sample['video_path']
            }
            
        except Exception as e:
            # Return a dummy sample in case of error
            print(f"Error loading {sample['video_path']}: {e}")
            dummy_frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.window_size)]
            inputs = self.processor(dummy_frames, return_tensors="pt")
            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'labels': torch.tensor(sample['label'], dtype=torch.long),
                'video_path': sample['video_path']
            }

class VideoMAEActionTrainer:
    """Trainer class for VideoMAE action classification"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize wandb if enabled (skip if wandb not available)
        if config.get('use_wandb', False):
            try:
                import wandb
                wandb.init(
                    project=config.get('wandb_project', 'shuttlesense-action-classification'),
                    name=f"videomae-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    config=config
                )
                self.use_wandb = True
            except ImportError:
                self.logger.warning("wandb not available, skipping W&B logging")
                self.use_wandb = False
        else:
            self.use_wandb = False
        
        self._setup_model()
        self._setup_data()
    
    def _setup_model(self):
        """Initialize model and optimizer"""
        model_config = self.config['model']
        
        # Get number of action classes
        dataset_path = Path(self.config['data']['dataset_path'])
        action_classes = [d.name for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        self.num_classes = len(action_classes)
        self.action_classes = sorted(action_classes)
        
        self.logger.info(f"Found {self.num_classes} action classes: {self.action_classes}")
        
        # Load pre-trained model
        self.processor = VideoMAEImageProcessor.from_pretrained(model_config['base_model'])
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_config['base_model'],
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        )
        
        self.model.to(self.device)
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Setup optimizer - ensure all config values are proper types
        training_config = self.config['training']
        learning_rate = float(training_config['learning_rate'])
        weight_decay = float(training_config['weight_decay'])
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def _setup_data(self):
        """Setup data loaders"""
        data_config = self.config['data']
        dataset_path = data_config['dataset_path']
        
        # Create datasets
        self.train_dataset = VideoBadmintonActionDataset(
            data_dir=dataset_path,
            processor=self.processor,
            split='train',
            window_size=int(data_config['window_size']),
            sample_rate=int(data_config['sample_rate'])
        )
        
        self.val_dataset = VideoBadmintonActionDataset(
            data_dir=dataset_path,
            processor=self.processor,
            split='val',
            window_size=int(data_config['window_size']),
            sample_rate=int(data_config['sample_rate'])
        )
        
        self.test_dataset = VideoBadmintonActionDataset(
            data_dir=dataset_path,
            processor=self.processor,
            split='test',
            window_size=int(data_config['window_size']),
            sample_rate=int(data_config['sample_rate'])
        )
        
        # Log dataset statistics
        self.logger.info(f"Dataset statistics:")
        self.logger.info(f"  Train: {len(self.train_dataset)} samples")
        self.logger.info(f"  Val: {len(self.val_dataset)} samples")
        self.logger.info(f"  Test: {len(self.test_dataset)} samples")
        
        # Data loaders
        batch_size = int(self.config['training']['batch_size'])
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,  # Reduce workers to avoid memory issues
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Setup scheduler
        epochs = int(self.config['training']['epochs'])
        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
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
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
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
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'predictions': predictions,
            'targets': targets
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
            'config': self.config,
            'action_classes': self.action_classes,
            'num_classes': self.num_classes
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best.pth')
            # Also save the model for inference
            self.model.save_pretrained(checkpoint_dir / 'best_model')
            self.processor.save_pretrained(checkpoint_dir / 'best_model')
            
            # Save class mapping
            with open(checkpoint_dir / 'best_model' / 'class_mapping.json', 'w') as f:
                json.dump({
                    'action_classes': self.action_classes,
                    'class_to_idx': {cls: idx for idx, cls in enumerate(self.action_classes)},
                    'idx_to_class': {str(idx): cls for idx, cls in enumerate(self.action_classes)}
                }, f, indent=2)
    
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
            
            # Print classification report for validation
            if epoch % 10 == 0:  # Every 10 epochs
                report = classification_report(
                    val_metrics['targets'], 
                    val_metrics['predictions'],
                    target_names=[cls.split('_', 1)[1] for cls in self.action_classes],
                    zero_division=0,
                    output_dict=False
                )
                self.logger.info(f"Validation Classification Report:\n{report}")
            
            # Log to wandb
            if self.use_wandb:
                try:
                    import wandb
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
                except:
                    pass  # Skip if wandb fails
            
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
    
    # Get configuration from environment or use default
    config_path = os.environ.get('CONFIG_PATH', "configs/action_classification_config.yaml")
    dataset_path = os.environ.get('DATASET_PATH', "/home/ubuntu/Project/shuttle-sense/VideoBadminton_Dataset")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        logger.info("Using default configuration...")
        # Create default config
        config = {
            'model': {
                'base_model': "MCG-NJU/videomae-base-finetuned-kinetics"
            },
            'data': {
                'dataset_path': dataset_path,
                'window_size': 16,
                'sample_rate': 2
            },
            'training': {
                'epochs': 30,
                'batch_size': 4,
                'learning_rate': 1e-5,
                'weight_decay': 0.01,
                'save_every': 5,
                'checkpoint_dir': "models/checkpoints/action_classification"
            },
            'use_wandb': False
        }
    
    # Override dataset path if provided in environment
    config['data']['dataset_path'] = dataset_path
    
    # Create trainer and start training
    trainer = VideoMAEActionTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()