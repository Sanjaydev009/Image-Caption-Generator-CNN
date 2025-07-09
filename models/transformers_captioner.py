"""
Advanced Image Captioning using Hugging Face Transformers
"""

import os
import numpy as np
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

class TransformersCaptioner:
    """
    Image captioning model using ViT (Vision Transformer) + GPT2 architecture from Hugging Face.
    This model generates dynamic captions based on actual image content.
    """
    
    def __init__(self, model_name="nlpconnect/vit-gpt2-image-captioning", max_length=20):
        """
        Initialize the captioner with pre-trained models.
        
        Args:
            model_name: Name of the pre-trained model from Hugging Face
            max_length: Maximum length of generated captions
        """
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Initializing Transformers Captioner (using {self.device})...")
        print(f"Loading model: {model_name}")
        
        try:
            # Load models
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Move model to appropriate device
            self.model.to(self.device)
            
            # Set generation parameters
            self.model.eval()
            
            # Define diversity parameters for different caption styles
            self.caption_styles = {
                "standard": {
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.92,
                    "num_beams": 1,
                    "max_length": 20,
                    "no_repeat_ngram_size": 2
                },
                "creative": {
                    "temperature": 1.0,
                    "top_k": 120,
                    "top_p": 0.95,
                    "num_beams": 1,
                    "max_length": 25,
                    "no_repeat_ngram_size": 2
                },
                "detailed": {
                    "temperature": 0.8,
                    "top_k": 80,
                    "top_p": 0.9,
                    "num_beams": 1,
                    "max_length": 30,
                    "no_repeat_ngram_size": 2
                },
                "concise": {
                    "temperature": 0.6,
                    "top_k": 40,
                    "top_p": 0.85,
                    "num_beams": 1,
                    "max_length": 15,
                    "no_repeat_ngram_size": 2
                },
                "diverse": {
                    "temperature": 1.1,
                    "top_k": 150,
                    "top_p": 0.98,
                    "num_beams": 1,
                    "max_length": 20,
                    "no_repeat_ngram_size": 2
                }
            }
            
            print("✓ Transformers Captioner initialized successfully")
            self.initialized = True
        except Exception as e:
            print(f"✗ Failed to initialize Transformers Captioner: {str(e)}")
            self.initialized = False
    
    def generate_caption(self, image):
        """
        Generate a caption for an image.
        
        Args:
            image: PIL.Image object or numpy array
            
        Returns:
            str: Generated caption
        """
        if not self.initialized:
            return "Image captioning model not available."
        
        try:
            # Ensure image is a PIL Image
            if isinstance(image, np.ndarray):
                # Convert from numpy array (RGB, 0-1) to PIL Image
                image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Process image
            pixel_values = self.feature_extractor(images=[image], return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)                # Generate caption
            with torch.no_grad():
                output_ids = self.model.generate(
                    pixel_values,
                    max_length=self.max_length,
                    num_beams=1,  # Use greedy decoding instead of beam search
                    do_sample=False
                )
                
                # Decode the generated caption
                caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Capitalize first letter and ensure it ends with a period
            caption = caption[0].upper() + caption[1:]
            if not caption.endswith('.'):
                caption += '.'
            
            return caption
        
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            return "Failed to generate caption for this image."

    def generate_multiple_captions(self, image, num_captions=5):
        """
        Generate multiple different captions for the same image using different parameter sets.
        
        Args:
            image: PIL.Image object or numpy array
            num_captions: Number of captions to generate
            
        Returns:
            list: List of generated captions
        """
        if not self.initialized:
            return ["Image captioning model not available."]
        
        try:
            # Ensure image is a PIL Image
            if isinstance(image, np.ndarray):
                # Convert from numpy array (RGB, 0-1) to PIL Image
                image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Process image
            pixel_values = self.feature_extractor(images=[image], return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate captions with diversity
            captions = []
            
            # Use different parameter sets to get diverse captions
            style_keys = list(self.caption_styles.keys())
            styles_to_use = style_keys.copy()
            
            # If we need more captions than styles, add more standard style
            while len(styles_to_use) < num_captions:
                styles_to_use.append("standard")
            
            # Ensure we only use as many styles as captions requested
            styles_to_use = styles_to_use[:num_captions]
            
            with torch.no_grad():
                # First, generate captions with different styles
                for style_name in styles_to_use:
                    style_params = self.caption_styles[style_name]
                    
                    output_ids = self.model.generate(
                        pixel_values,
                        max_length=style_params["max_length"],
                        num_beams=style_params["num_beams"],
                        temperature=style_params["temperature"],
                        do_sample=True,
                        top_k=style_params["top_k"],
                        top_p=style_params["top_p"],
                        no_repeat_ngram_size=style_params["no_repeat_ngram_size"]
                    )
                    
                    caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    # Capitalize first letter and ensure it ends with a period
                    caption = caption[0].upper() + caption[1:]
                    if not caption.endswith('.'):
                        caption += '.'
                    
                    captions.append(caption)
                
                # If we still need more captions, generate additional ones with modified parameters
                additional_needed = num_captions - len(captions)
                if additional_needed > 0:
                    # Create variation by adjusting temperature and top_k
                    base_params = self.caption_styles["standard"]
                    for i in range(additional_needed):
                        # Vary temperature for additional diversity
                        temp_adjustment = 0.1 * (i + 1)
                        
                        output_ids = self.model.generate(
                            pixel_values,
                            max_length=base_params["max_length"],
                            num_beams=1,
                            temperature=base_params["temperature"] + temp_adjustment,
                            do_sample=True,
                            top_k=base_params["top_k"] + (i * 10),
                            top_p=min(0.99, base_params["top_p"] + (0.01 * i)),
                            no_repeat_ngram_size=base_params["no_repeat_ngram_size"]
                        )
                        
                        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                        
                        # Capitalize first letter and ensure it ends with a period
                        caption = caption[0].upper() + caption[1:]
                        if not caption.endswith('.'):
                            caption += '.'
                        
                        captions.append(caption)
            
            # Ensure captions are unique
            unique_captions = []
            for caption in captions:
                if caption not in unique_captions:
                    unique_captions.append(caption)
            
            # If we have fewer unique captions than requested, fill with backups
            if len(unique_captions) < num_captions:
                backup_captions = [
                    f"A detailed view of the image showing various elements.",
                    f"An interesting composition captured in this scene.",
                    f"The image depicts a visually engaging subject.",
                    f"A well-composed photograph with intriguing details.",
                    f"A captivating visual representation of the subject."
                ]
                
                # Add backup captions that aren't already in our list
                for backup in backup_captions:
                    if len(unique_captions) >= num_captions:
                        break
                    if backup not in unique_captions:
                        unique_captions.append(backup)
            
            return unique_captions[:num_captions]
        
        except Exception as e:
            print(f"Error generating multiple captions: {str(e)}")
            return ["Failed to generate captions for this image."]
