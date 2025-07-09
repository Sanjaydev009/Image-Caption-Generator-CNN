"""
Ensemble Captioning System that combines multiple captioning approaches
for better quality and diversity in generated captions
"""

import os
import numpy as np
import random
import re
from PIL import Image
import time

class EnsembleCaptioner:
    """
    An ensemble captioning system that combines multiple captioning approaches
    to generate more diverse and higher quality captions.
    """
    
    def __init__(self, captioner_models=None):
        """
        Initialize the ensemble captioner with multiple captioning models.
        
        Args:
            captioner_models: Dictionary of captioning models to use
        """
        self.captioner_models = captioner_models or {}
        self.description_enhancers = self._load_description_enhancers()
        self.style_templates = self._load_style_templates()
        
        print(f"Initialized Ensemble Captioner with {len(self.captioner_models)} models")
    
    def add_captioner(self, name, captioner):
        """Add a captioner to the ensemble."""
        self.captioner_models[name] = captioner
        print(f"Added {name} to the ensemble captioner")
    
    def _load_description_enhancers(self):
        """Load description enhancement patterns."""
        return {
            "artistic": [
                "with beautiful lighting", 
                "with stunning composition", 
                "artistically framed", 
                "expertly captured",
                "showing amazing detail"
            ],
            "emotional": [
                "creating a peaceful atmosphere", 
                "evoking a sense of calm", 
                "with a dramatic effect",
                "with a nostalgic feel",
                "with a cheerful mood"
            ],
            "technical": [
                "in high resolution", 
                "with sharp focus", 
                "with a shallow depth of field",
                "with balanced exposure",
                "captured with a wide-angle lens"
            ],
            "temporal": [
                "during golden hour", 
                "at sunset", 
                "in the early morning light",
                "under afternoon sun",
                "at twilight"
            ]
        }
    
    def _load_style_templates(self):
        """Load caption style templates."""
        return {
            "descriptive": "{caption}",
            "artistic": "A captivating image showing {caption}",
            "technical": "This photograph depicts {caption}",
            "narrative": "In this scene, {caption}",
            "minimalist": "{simple_caption}.",
            "detailed": "A detailed view of {caption}, {enhancement}.",
            "professional": "A professional photograph of {caption}",
            "documentary": "Documentary-style image of {caption}",
            "vibrant": "A vibrant scene featuring {caption}",
            "elegant": "An elegant composition showing {caption}",
            "dynamic": "A dynamic shot of {caption} in motion",
            "atmospheric": "An atmospheric scene with {caption}",
            "candid": "A candid moment capturing {caption}",
            "dramatic": "A dramatic perspective of {caption}",
            "serene": "A serene view of {caption}"
        }
    
    def generate_captions(self, image, scene_type=None, objects=None, colors=None, num_captions=5):
        """
        Generate multiple high-quality captions using all available models.
        
        Args:
            image: PIL.Image object or numpy array
            scene_type: Type of scene in the image (optional)
            objects: Detected objects in the image (optional)
            colors: Dominant colors in the image (optional)
            num_captions: Number of captions to generate
            
        Returns:
            list: List of generated captions
        """
        all_captions = []
        start_time = time.time()
        
        # Try all available captioning models
        for model_name, captioner in self.captioner_models.items():
            try:
                print(f"Generating captions with {model_name}...")
                
                # Each captioner might have different methods for generating captions
                if model_name == "transformers" and hasattr(captioner, "generate_multiple_captions"):
                    # Try to get more captions than needed for better filtering later
                    model_captions = captioner.generate_multiple_captions(image, min(num_captions + 2, 10))
                    for caption in model_captions:
                        all_captions.append({"caption": caption, "source": model_name, "score": 0.95})
                        
                elif model_name == "enhanced" and hasattr(captioner, "generate_caption"):
                    # Generate multiple captions with the enhanced captioner
                    for _ in range(min(3, num_captions)):
                        caption = captioner.generate_caption(
                            image, scene_type=scene_type, 
                            detected_objects=objects, dominant_colors=colors,
                            has_faces="Yes" if objects and "person" in objects else "No"
                        )
                        all_captions.append({"caption": caption, "source": model_name, "score": 0.9})
                
                elif hasattr(captioner, "generate_caption"):
                    # Generic single caption generator
                    caption = captioner.generate_caption(image)
                    if caption:
                        all_captions.append({"caption": caption, "source": model_name, "score": 0.8})
                
                else:
                    print(f"Unsupported captioner type: {model_name}")
                    
            except Exception as e:
                print(f"Error generating captions with {model_name}: {e}")
        
        # Generate enhanced variations of the existing captions
        enhanced_captions = self._generate_enhanced_variations(all_captions)
        all_captions.extend(enhanced_captions)
        
        # Score and filter captions
        scored_captions = self._score_captions(all_captions)
        unique_captions = self._filter_similar_captions(scored_captions)
        
        # Sort by score and take the top N
        top_captions = sorted(unique_captions, key=lambda x: x["score"], reverse=True)[:num_captions]
        
        print(f"Generated {len(top_captions)} captions in {time.time() - start_time:.2f} seconds")
        
        # Return just the caption strings
        return [item["caption"] for item in top_captions]
    
    def _generate_enhanced_variations(self, base_captions, num_variations=5):
        """Generate enhanced variations of existing captions."""
        variations = []
        
        # Process the best captions
        for caption_data in base_captions[:5]:
            base_caption = caption_data["caption"]
            
            # Remove ending period for easier manipulation
            if base_caption.endswith('.'):
                base_caption = base_caption[:-1]
            
            # Apply different style templates - more styles for more diversity
            # Randomly select multiple styles but ensure we get different ones
            styles_to_use = random.sample(list(self.style_templates.items()), 
                                        min(num_variations, len(self.style_templates)))
            
            for style_name, template in styles_to_use:
                try:
                    # For minimalist style, simplify the caption
                    if style_name == "minimalist":
                        simple_caption = self._simplify_caption(base_caption)
                        variation = template.format(simple_caption=simple_caption)
                    
                    # For detailed style, add an enhancement
                    elif style_name == "detailed":
                        enhancement_type = random.choice(list(self.description_enhancers.keys()))
                        enhancement = random.choice(self.description_enhancers[enhancement_type])
                        variation = template.format(caption=base_caption.lower(), enhancement=enhancement)
                    
                    # For atmospheric styles, add more enhancements
                    elif style_name in ["atmospheric", "dramatic", "serene", "elegant"]:
                        # Choose artistic or emotional enhancers for these styles
                        enhancer_type = "artistic" if random.random() < 0.6 else "emotional"
                        enhancement = random.choice(self.description_enhancers[enhancer_type])
                        caption_with_enhancement = f"{base_caption.lower()}, {enhancement}"
                        variation = template.format(caption=caption_with_enhancement)
                    
                    # For technical styles, add technical details
                    elif style_name in ["technical", "professional", "documentary"]:
                        # Choose technical enhancers
                        enhancement = random.choice(self.description_enhancers["technical"])
                        caption_with_enhancement = f"{base_caption.lower()}, {enhancement}"
                        variation = template.format(caption=caption_with_enhancement)
                    
                    # For other styles, just apply the template
                    else:
                        variation = template.format(caption=base_caption.lower())
                    
                    # Ensure the caption ends with a period
                    if not variation.endswith('.'):
                        variation += '.'
                    
                    # Ensure the first letter is capitalized
                    variation = variation[0].upper() + variation[1:]
                    
                    # Avoid exact duplicates
                    if not any(var["caption"] == variation for var in variations):
                        # Add the variation
                        variations.append({
                            "caption": variation,
                            "source": f"{caption_data['source']}-{style_name}",
                            "score": caption_data["score"] * 0.98  # Slightly lower score for variations
                        })
                except Exception as e:
                    print(f"Error creating caption variation: {e}")
        
        return variations
    
    def _simplify_caption(self, caption):
        """Simplify a caption to its core elements."""
        # Remove common phrases like "a photo of", "an image of", etc.
        simplifiers = [
            r"^(a|an|the) (photo|picture|image|snapshot) (of|showing|displaying|with) ",
            r"^(a|an|the) ",
            r"(in|on|at) (a|an|the) (background|foreground)$"
        ]
        
        simplified = caption
        for pattern in simplifiers:
            simplified = re.sub(pattern, "", simplified, flags=re.IGNORECASE)
        
        # Further simplification can be done here
        
        return simplified.strip()
    
    def _score_captions(self, captions):
        """Score captions based on various quality metrics."""
        scored_captions = []
        
        for caption_data in captions:
            caption = caption_data["caption"]
            initial_score = caption_data.get("score", 0.7)  # Default score
            
            # Scoring factors
            length_score = min(1.0, len(caption.split()) / 20.0)  # Prefer longer captions up to a point
            diversity_score = len(set(caption.lower().split())) / max(1, len(caption.split()))  # Lexical diversity
            
            # Penalize very short captions
            if len(caption.split()) < 5:
                length_score *= 0.7
            
            # Penalize generic captions
            generic_phrases = ["an image", "a picture", "a photo", "this image", "this picture"]
            generic_penalty = 0.0
            for phrase in generic_phrases:
                if phrase in caption.lower():
                    generic_penalty += 0.05
            
            # Calculate final score
            final_score = initial_score * 0.6 + length_score * 0.2 + diversity_score * 0.2 - generic_penalty
            
            # Ensure score is in [0,1] range
            final_score = max(0.0, min(1.0, final_score))
            
            scored_captions.append({
                "caption": caption,
                "source": caption_data["source"],
                "score": final_score
            })
        
        return scored_captions
    
    def _filter_similar_captions(self, captions, similarity_threshold=0.7):
        """Filter out very similar captions."""
        unique_captions = []
        
        for caption_data in captions:
            caption = caption_data["caption"].lower()
            
            # Check if this caption is too similar to any we've already kept
            is_too_similar = False
            for unique in unique_captions:
                similarity = self._caption_similarity(caption, unique["caption"].lower())
                if similarity > similarity_threshold:
                    is_too_similar = True
                    # Keep the higher-scored caption
                    if caption_data["score"] > unique["score"]:
                        unique_captions.remove(unique)
                        unique_captions.append(caption_data)
                    break
            
            if not is_too_similar:
                unique_captions.append(caption_data)
        
        return unique_captions
    
    def _caption_similarity(self, caption1, caption2):
        """Calculate similarity between two captions."""
        # Simple word overlap measure
        words1 = set(caption1.lower().split())
        words2 = set(caption2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
