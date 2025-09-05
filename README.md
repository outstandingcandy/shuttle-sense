# ShuttleSense

An Intelligent Badminton Video Analysis and Annotation System

## Overview

ShuttleSense is an automated system for analyzing badminton match videos. It processes full-length badminton videos by:

1. Automatically detecting hit points
2. Segmenting the video into meaningful clips
3. Generating detailed tactical and technical annotations for each segment

## Features

- **Hit Point Detection**: Identifies precise moments when the shuttle is hit using a fine-tuned VideoMAE model
- **Smart Video Segmentation**: Creates short video clips between consecutive hits
- **AI-Powered Annotations**: Generates detailed insights about techniques, tactics, and quality for each rally segment
- **End-to-End Pipeline**: Fully automated workflow from raw video to annotated segments
- **YouTube Video Downloader**: Download badminton training videos directly from YouTube

## Requirements

- Python 3.8+
- FFmpeg
- CUDA-compatible GPU (recommended for faster processing)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/shuttle-sense.git
cd shuttle-sense
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up API keys for multimodal models:

Create a `.env` file in the project root with your API keys:
```
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Basic Usage

Process a video with default settings:

```bash
python main.py --video /path/to/your/badminton_match.mp4
```

### Advanced Options

```bash
# Skip certain pipeline stages
python main.py --video /path/to/video.mp4 --skip-detection
python main.py --video /path/to/video.mp4 --skip-segmentation
python main.py --video /path/to/video.mp4 --skip-annotation

# Specify custom output directory
python main.py --video /path/to/video.mp4 --output-dir /path/to/results

# Use custom configuration
python main.py --video /path/to/video.mp4 --config custom_config.yaml
```

### Downloading Training Videos

ShuttleSense includes a YouTube downloader to help you collect badminton training videos:

```bash
# Interactive downloader with menu options
python scripts/download_training_videos.py

# Or use the downloader directly
python utils/youtube_downloader.py --search "badminton match BWF" --max-videos 5
python utils/youtube_downloader.py --url "https://youtube.com/watch?v=VIDEO_ID"
python utils/youtube_downloader.py --playlist "https://youtube.com/playlist?list=PLAYLIST_ID"
```

The downloader will:
- Filter for badminton-related content
- Save videos in appropriate quality (up to 720p)
- Generate download logs for tracking
- Organize files in the `data/training_videos/` directory

## Configuration

You can customize ShuttleSense by modifying `config.yaml`. Key settings include:

- Hit detection sensitivity
- Video segmentation parameters
- Annotation model selection (Gemini or GPT-4o)
- System resources allocation

## Example Output

For each processed video, ShuttleSense generates:

1. A JSON file with detected hit points
2. Video segments for each rally section
3. Detailed annotations in JSON format with:
   - Foundational Actions (technical movements)
   - Tactical Semantics (strategic elements)
   - Brief Evaluation (quality assessment)

## Project Structure

```
shuttle-sense/
├── src/
│   ├── hit_detection/      # Hit point detection module
│   ├── video_segmentation/ # Video segmentation module
│   ├── annotation/         # AI annotation module
│   └── pipeline.py         # Main processing pipeline
├── data/                   # Input data directory
├── output/                 # Generated outputs
├── config.yaml             # Configuration file
├── main.py                 # Entry point
└── requirements.txt        # Dependencies
```

## Inspiration

This project was inspired by the research paper "FineBadminton: A Multi-Level Dataset for Fine-Grained Badminton Video Understanding".

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.