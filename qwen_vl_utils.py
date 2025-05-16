from typing import Dict, List, Optional, Tuple, Union
import re
import numpy as np
from PIL import Image
import torch

def process_vision_info(messages):
    """
    Process image and video data from messages.
    
    Args:
        messages (List[Dict]): List of message dictionaries containing content.
        
    Returns:
        Tuple: Processed image and video inputs.
    """
    image_inputs = []
    video_inputs = []

    for message in messages:
        if message.get("role") == "user":
            for content in message.get("content", []):
                if isinstance(content, dict):
                    if content.get("type") == "image":
                        image_data = content.get("image")
                        if isinstance(image_data, Image.Image):
                            image_inputs.append(image_data)
                    elif content.get("type") == "video":
                        video_data = content.get("video")
                        if video_data is not None:
                            video_inputs.append(video_data)

    return image_inputs, video_inputs

def process_text_for_arabic(text):
    """
    Process text to handle Arabic characters properly.
    
    Args:
        text (str): Input text.
        
    Returns:
        str: Processed text.
    """
    # Add any specific Arabic text processing if needed
    return text
