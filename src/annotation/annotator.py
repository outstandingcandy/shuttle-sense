"""
Video Annotator
Generates semantic annotations for badminton video segments using multimodal LLMs.
"""

import logging
import os
import json
import base64
from pathlib import Path
import time
import cv2
from tqdm import tqdm
import requests
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

class VideoAnnotator:
    """Generates annotations for badminton video segments using multimodal LLMs."""
    
    def __init__(self, config):
        """
        Initialize the video annotator.
        
        Args:
            config: Configuration for annotation
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.provider = config["provider"]
        self.model = config["model"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        
        # Load API keys from environment
        if self.provider == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                self.logger.warning("GEMINI_API_KEY not found in environment")
        elif self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                self.logger.warning("OPENAI_API_KEY not found in environment")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Initialize API client if needed
        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "openai":
            self._init_openai()
    
    def _init_gemini(self):
        """Initialize Google Gemini API client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai
            self.logger.info("Initialized Google Gemini API client")
        except ImportError:
            self.logger.error("Failed to import google.generativeai. Please install with: pip install google-generativeai")
            raise
    
    def _init_openai(self):
        """Initialize OpenAI API client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.logger.info("Initialized OpenAI API client")
        except ImportError:
            self.logger.error("Failed to import openai. Please install with: pip install openai")
            raise
    
    def annotate_segments(self, segments_dir, segments_info):
        """
        Annotate video segments using multimodal LLM.
        
        Args:
            segments_dir: Directory containing video segments
            segments_info: List of segment information dictionaries
            
        Returns:
            List of annotations for each segment
        """
        self.logger.info(f"Annotating {len(segments_info)} video segments using {self.provider}/{self.model}")
        
        annotations = []
        for segment in tqdm(segments_info, desc="Annotating segments"):
            segment_path = os.path.join(segments_dir, segment["filename"])
            if not os.path.exists(segment_path):
                self.logger.warning(f"Segment file not found: {segment_path}")
                continue
            
            try:
                # Extract frames from the segment for annotation
                frames = self._extract_key_frames(segment_path, num_frames=5)
                
                # Generate annotation
                annotation = self._generate_annotation(frames, segment)
                
                # Add to results
                annotations.append({
                    "segment_id": segment["id"],
                    "filename": segment["filename"],
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "annotation": annotation
                })
                
                # Rate limiting to avoid API throttling
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error annotating segment {segment['filename']}: {str(e)}")
        
        self.logger.info(f"Completed annotations for {len(annotations)} segments")
        return {"annotations": annotations}
    
    def _extract_key_frames(self, video_path, num_frames=5):
        """
        Extract key frames from video for annotation.
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to extract
            
        Returns:
            List of extracted frame images (as NumPy arrays)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract
        if frame_count <= num_frames:
            # If video has fewer frames than requested, take all frames
            indices = list(range(frame_count))
        else:
            # Take frames at regular intervals
            indices = [int(i * (frame_count - 1) / (num_frames - 1)) for i in range(num_frames)]
        
        # Extract frames
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    def _generate_annotation(self, frames, segment):
        """
        Generate annotation for the segment using multimodal LLM.
        
        Args:
            frames: List of frame images
            segment: Segment information dictionary
            
        Returns:
            Dictionary containing the generated annotation
        """
        if self.provider == "gemini":
            return self._generate_with_gemini(frames, segment)
        elif self.provider == "openai":
            return self._generate_with_openai(frames, segment)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_with_gemini(self, frames, segment):
        """Generate annotation using Google Gemini."""
        from PIL import Image
        import google.generativeai as genai
        
        # Convert frames to PIL images
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Craft the prompt
        prompt = """
You are a professional badminton tactics analyst. Please analyze this set of frames from a badminton rally segment between two consecutive hits.
Provide a detailed analysis report with these three sections:

1. FOUNDATIONAL ACTIONS: Describe the player's key hitting action type (e.g., clear, smash, drop shot, net shot) and their footwork/movement.

2. TACTICAL SEMANTICS: Analyze the tactical intent (offensive, defensive, transitional), describe the shuttle's trajectory (straight, cross-court), and the player's court positioning.

3. BRIEF EVALUATION: Provide a brief quality assessment of this shot (e.g., high quality, precise placement, error, opportunity).

Only include these three sections with clear headings. Be precise and use proper badminton terminology.
"""
        
        # Generate content
        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            }
        )
        
        response = model.generate_content([prompt] + pil_frames)
        
        # Extract sections from response
        text = response.text
        
        # Parse the response into sections
        sections = {
            "foundational_actions": "",
            "tactical_semantics": "",
            "brief_evaluation": ""
        }
        
        current_section = None
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            if "FOUNDATIONAL ACTIONS" in line.upper():
                current_section = "foundational_actions"
                continue
            elif "TACTICAL SEMANTICS" in line.upper():
                current_section = "tactical_semantics"
                continue
            elif "BRIEF EVALUATION" in line.upper():
                current_section = "brief_evaluation"
                continue
                
            if current_section and current_section in sections:
                if sections[current_section]:
                    sections[current_section] += " " + line
                else:
                    sections[current_section] = line
        
        return sections
    
    def _generate_with_openai(self, frames, segment):
        """Generate annotation using OpenAI GPT-4o."""
        import io
        from PIL import Image
        
        # Encode frames as base64
        encoded_frames = []
        for frame in frames:
            pil_image = Image.fromarray(frame)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            encoded_frames.append(encoded_image)
        
        # Prepare the messages
        messages = [
            {
                "role": "system",
                "content": "You are a professional badminton tactics analyst that provides precise, technical analysis of badminton plays."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Please analyze these frames from a badminton rally segment between two consecutive hits.
Provide a detailed analysis report with these three sections:

1. FOUNDATIONAL ACTIONS: Describe the player's key hitting action type (e.g., clear, smash, drop shot, net shot) and their footwork/movement.

2. TACTICAL SEMANTICS: Analyze the tactical intent (offensive, defensive, transitional), describe the shuttle's trajectory (straight, cross-court), and the player's court positioning.

3. BRIEF EVALUATION: Provide a brief quality assessment of this shot (e.g., high quality, precise placement, error, opportunity).

Only include these three sections with clear headings. Be precise and use proper badminton terminology."""
                    }
                ]
            }
        ]
        
        # Add image content to the user message
        for encoded_image in encoded_frames:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            })
        
        # Call the API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        # Extract the generated text
        text = response.choices[0].message.content
        
        # Parse the response into sections
        sections = {
            "foundational_actions": "",
            "tactical_semantics": "",
            "brief_evaluation": ""
        }
        
        current_section = None
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            if "FOUNDATIONAL ACTIONS" in line.upper():
                current_section = "foundational_actions"
                continue
            elif "TACTICAL SEMANTICS" in line.upper():
                current_section = "tactical_semantics"
                continue
            elif "BRIEF EVALUATION" in line.upper():
                current_section = "brief_evaluation"
                continue
                
            if current_section and current_section in sections:
                if sections[current_section]:
                    sections[current_section] += " " + line
                else:
                    sections[current_section] = line
        
        return sections