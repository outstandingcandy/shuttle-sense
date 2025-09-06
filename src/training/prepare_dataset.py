"""
Data Preparation Utilities for VideoBadminton_Dataset
Utilities for downloading, preprocessing, and organizing the dataset
"""

import os
import json
import requests
import zipfile
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import yaml
import logging
from sklearn.model_selection import train_test_split
import shutil

class VideoBadmintonDatasetPreparator:
    """Class for preparing VideoBadminton_Dataset for training"""
    
    def __init__(self, config_path="configs/training_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data/VideoBadminton_Dataset")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
    def download_dataset(self, dataset_url=None):
        """
        Download VideoBadminton_Dataset
        Note: Replace with actual dataset URL
        """
        if dataset_url is None:
            self.logger.info("Please provide the dataset URL or download manually")
            self.logger.info("Expected structure:")
            self.logger.info("data/VideoBadminton_Dataset/raw/")
            self.logger.info("├── videos/")
            self.logger.info("│   ├── video1.mp4")
            self.logger.info("│   ├── video2.mp4")
            self.logger.info("│   └── ...")
            self.logger.info("└── annotations.json")
            return False
        
        # Download and extract dataset
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.logger.info(f"Downloading dataset from {dataset_url}")
            response = requests.get(dataset_url, stream=True)
            response.raise_for_status()
            
            dataset_zip = self.raw_dir / "dataset.zip"
            with open(dataset_zip, 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading"):
                    f.write(chunk)
            
            # Extract
            self.logger.info("Extracting dataset...")
            with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
                zip_ref.extractall(self.raw_dir)
            
            # Clean up zip file
            dataset_zip.unlink()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading dataset: {e}")
            return False
    
    def create_sample_annotations(self):
        """Create sample annotations file for testing"""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample annotation format - create more videos for proper splitting
        sample_annotations = {}
        
        for i in range(1, 11):  # Create 10 sample videos
            video_id = f"video_{i:03d}"
            duration = np.random.uniform(60, 180)  # Random duration 60-180s
            
            # Generate random hit points
            num_hits = np.random.randint(5, 15)
            hit_points = sorted(np.random.uniform(5, duration-5, num_hits))
            
            sample_annotations[video_id] = {
                "duration": round(duration, 1),
                "hit_points": [round(h, 1) for h in hit_points],
                "metadata": {
                    "match_type": np.random.choice(["singles", "doubles"]),
                    "court_type": np.random.choice(["indoor", "outdoor"]),
                    "players": [f"Player_{chr(65+j)}" for j in range(np.random.randint(2, 5))]
                }
            }
        
        annotations_file = self.raw_dir / "annotations.json"
        with open(annotations_file, 'w') as f:
            json.dump(sample_annotations, f, indent=2)
        
        self.logger.info(f"Created sample annotations for {len(sample_annotations)} videos at {annotations_file}")
    
    def analyze_dataset(self):
        """Analyze the raw dataset and provide statistics"""
        annotations_file = self.raw_dir / "annotations.json"
        videos_dir = self.raw_dir / "videos"
        
        if not annotations_file.exists():
            self.logger.error("Annotations file not found")
            return
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        stats = {
            'total_videos': len(annotations),
            'total_hit_points': 0,
            'video_durations': [],
            'hit_point_intervals': [],
            'missing_videos': []
        }
        
        for video_id, data in annotations.items():
            video_path = videos_dir / f"{video_id}.mp4"
            
            if not video_path.exists():
                stats['missing_videos'].append(video_id)
                continue
            
            duration = data.get('duration', 0)
            hit_points = data.get('hit_points', [])
            
            stats['video_durations'].append(duration)
            stats['total_hit_points'] += len(hit_points)
            
            # Calculate intervals between hit points
            for i in range(1, len(hit_points)):
                interval = hit_points[i] - hit_points[i-1]
                stats['hit_point_intervals'].append(interval)
        
        # Calculate statistics
        if stats['video_durations']:
            stats['avg_duration'] = np.mean(stats['video_durations'])
            stats['total_duration'] = np.sum(stats['video_durations'])
        
        if stats['hit_point_intervals']:
            stats['avg_hit_interval'] = np.mean(stats['hit_point_intervals'])
            stats['min_hit_interval'] = np.min(stats['hit_point_intervals'])
            stats['max_hit_interval'] = np.max(stats['hit_point_intervals'])
        
        # Print statistics
        self.logger.info("Dataset Statistics:")
        self.logger.info(f"Total videos: {stats['total_videos']}")
        self.logger.info(f"Total hit points: {stats['total_hit_points']}")
        self.logger.info(f"Average video duration: {stats.get('avg_duration', 0):.2f}s")
        self.logger.info(f"Total dataset duration: {stats.get('total_duration', 0):.2f}s")
        self.logger.info(f"Average hit interval: {stats.get('avg_hit_interval', 0):.2f}s")
        self.logger.info(f"Missing videos: {len(stats['missing_videos'])}")
        
        return stats
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """Split dataset into train/val/test sets"""
        annotations_file = self.raw_dir / "annotations.json"
        
        if not annotations_file.exists():
            self.logger.error("Annotations file not found")
            return
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        video_ids = list(annotations.keys())
        
        # First split: train + val, test
        train_val_ids, test_ids = train_test_split(
            video_ids, 
            test_size=test_ratio, 
            random_state=seed
        )
        
        # Second split: train, val
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_ratio / (train_ratio + val_ratio),
            random_state=seed
        )
        
        # Create split directories
        splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
        
        for split_name, video_list in splits.items():
            split_dir = self.data_dir / split_name
            split_videos_dir = split_dir / "videos"
            split_videos_dir.mkdir(parents=True, exist_ok=True)
            
            # Create split annotations
            split_annotations = {vid: annotations[vid] for vid in video_list if vid in annotations}
            
            with open(split_dir / "annotations.json", 'w') as f:
                json.dump(split_annotations, f, indent=2)
            
            # Copy videos (optional - can use symlinks to save space)
            for video_id in video_list:
                src_video = self.raw_dir / "videos" / f"{video_id}.mp4"
                dst_video = split_videos_dir / f"{video_id}.mp4"
                
                if src_video.exists() and not dst_video.exists():
                    # Use symlink to save disk space
                    try:
                        dst_video.symlink_to(src_video.absolute())
                    except:
                        # Fallback to copying if symlink fails
                        shutil.copy2(src_video, dst_video)
        
        self.logger.info("Dataset split completed:")
        self.logger.info(f"Train: {len(train_ids)} videos")
        self.logger.info(f"Val: {len(val_ids)} videos")
        self.logger.info(f"Test: {len(test_ids)} videos")
        
        return splits
    
    def validate_videos(self):
        """Validate video files and check for corruption"""
        results = {'valid': [], 'invalid': [], 'missing': []}
        
        for split in ['train', 'val', 'test']:
            split_dir = self.data_dir / split
            annotations_file = split_dir / "annotations.json"
            
            if not annotations_file.exists():
                continue
            
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            
            for video_id in tqdm(annotations.keys(), desc=f"Validating {split}"):
                video_path = split_dir / "videos" / f"{video_id}.mp4"
                
                if not video_path.exists():
                    results['missing'].append(str(video_path))
                    continue
                
                # Try to open video with OpenCV
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    # Check if we can read at least one frame
                    ret, frame = cap.read()
                    if ret:
                        results['valid'].append(str(video_path))
                    else:
                        results['invalid'].append(str(video_path))
                else:
                    results['invalid'].append(str(video_path))
                
                cap.release()
        
        self.logger.info("Video validation results:")
        self.logger.info(f"Valid: {len(results['valid'])}")
        self.logger.info(f"Invalid: {len(results['invalid'])}")
        self.logger.info(f"Missing: {len(results['missing'])}")
        
        return results

def main():
    """Main function for dataset preparation"""
    logging.basicConfig(level=logging.INFO)
    
    preparator = VideoBadmintonDatasetPreparator()
    
    # Create sample data for testing
    preparator.create_sample_annotations()
    
    # Analyze dataset
    stats = preparator.analyze_dataset()
    
    # Split dataset
    if stats and stats['total_videos'] > 0:
        splits = preparator.split_dataset()
        
        # Validate videos
        validation_results = preparator.validate_videos()

if __name__ == "__main__":
    main()