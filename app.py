#!/usr/bin/env python3
"""
Web interface for the Image Caption Generator
"""

import os
import sys
import numpy as np
import random
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io
import cv2
import json
import time
from googletrans import Translator

# Import custom modules
from models.cnn_feature_extractor import CNNFeatureExtractor
from models.object_detector import ObjectDetector
from models.enhanced_captioner import EnhancedImageCaptioner
from models.scene_captioner import SceneUnderstandingCaptioner
from models.ensemble_captioner import EnsembleCaptioner
try:
    from models.transformers_captioner import TransformersCaptioner
    transformers_available = True
except ImportError as e:
    print(f"Transformers not available: {e}")
    transformers_available = False

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize models
feature_extractor = None
object_detector = None
transformers_captioner = None
enhanced_captioner = None
scene_captioner = None
ensemble_captioner = None

try:
    feature_extractor = CNNFeatureExtractor(model_name='vgg16', feature_dim=2048)
    print("✓ Feature extractor initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize feature extractor: {e}")
    # We'll continue anyway and show an error in the web interface

# Try to initialize the transformer-based captioner (preferred method)
if transformers_available:
    try:
        print("Initializing transformer-based captioning model (this may take a moment)...")
        transformers_captioner = TransformersCaptioner()
        if transformers_captioner.initialized:
            print("✓ Transformer captioning model initialized successfully")
        else:
            transformers_captioner = None
    except Exception as e:
        print(f"✗ Failed to initialize transformer captioner: {e}")
        transformers_captioner = None

try:
    object_detector = ObjectDetector()
    print("✓ Object detector initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize object detector: {e}")
    object_detector = None
    object_detector = None

try:
    enhanced_captioner = EnhancedImageCaptioner()
    print("✓ Enhanced captioner initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize enhanced captioner: {e}")
    enhanced_captioner = None

try:
    scene_captioner = SceneUnderstandingCaptioner()
    print("✓ Scene captioner initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize scene captioner: {e}")
    scene_captioner = None

# Initialize the ensemble captioner with all available captioning models
try:
    captioner_models = {}
    if transformers_captioner and transformers_captioner.initialized:
        captioner_models["transformers"] = transformers_captioner
    if enhanced_captioner:
        captioner_models["enhanced"] = enhanced_captioner
    if scene_captioner:
        captioner_models["scene"] = scene_captioner
        
    ensemble_captioner = EnsembleCaptioner(captioner_models)
    print(f"✓ Ensemble captioner initialized successfully with {len(captioner_models)} models")
except Exception as e:
    print(f"✗ Failed to initialize ensemble captioner: {e}")
    ensemble_captioner = None

def allowed_file(filename):
    """Check if the filename extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path):
    """Process image and generate caption."""
    try:
        start_time = time.time()
        
        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        img_for_analysis = img.resize((224, 224))
        img_array = np.array(img_for_analysis) / 255.0
        
        # Analyze image content for color composition and patterns
        dominant_colors = analyze_image_colors(img_array)
        image_brightness = np.mean(img_array)
        image_contrast = np.std(img_array)
        has_faces = detect_faces(img_array)
        scene_type = analyze_scene(img_array)
        
        print(f"Image analysis: Brightness={image_brightness:.2f}, Contrast={image_contrast:.2f}, Scene={scene_type}")
        print(f"Dominant colors: {dominant_colors}")
        
        # Detect objects in the image
        detected_objects = {}
        if object_detector is not None:
            detected_objects = object_detector.detect_objects(img_array)
            print(f"Detected objects: {json.dumps(detected_objects, indent=2)}")
        
        # Prepare for feature extraction
        img_array_batch = np.expand_dims(img_array, axis=0)
        
        # Extract features
        if feature_extractor is None:
            return {"error": "Feature extractor not initialized. Check server logs."}, 500
        
        print(f"Extracting CNN features for image: {os.path.basename(image_path)}")
        features = feature_extractor.extract_features(img_array_batch)
        print(f"Features extracted, shape: {features.shape}")
        
        # Feature statistics
        feature_stats = {
            "shape": features.shape,
            "min": float(features.min()),
            "max": float(features.max()),
            "mean": float(features.mean()),
            "std": float(np.std(features)),
            "sample": [float(x) for x in features[0, :5].tolist() if x is not None]
        }
        
        # Generate captions - use different approaches and return multiple options
        captions = []
        caption_sources = []
        
        # 1. Try the ensemble captioner first (best quality and diversity)
        if ensemble_captioner is not None:
            try:
                print("Generating captions using ensemble captioner model...")
                # Generate 10 different captions with the ensemble captioner
                ensemble_captions = ensemble_captioner.generate_captions(
                    img,
                    scene_type=scene_type,
                    objects=detected_objects,
                    colors=dominant_colors,
                    num_captions=10  # Increased for more diversity
                )
                
                for caption in ensemble_captions:
                    if caption and caption not in captions:
                        captions.append(caption)
                        caption_sources.append("ensemble")
                
                print(f"Generated {len(captions)} captions with ensemble model")
            except Exception as e:
                print(f"Error generating ensemble captions: {e}")
        
        # 2. If ensemble failed, try the transformer directly
        if (not captions or len(captions) < 3) and transformers_captioner and transformers_captioner.initialized:
            try:
                print("Generating captions using transformer model...")
                # Generate 3 different captions with the transformer model
                transformer_captions = transformers_captioner.generate_multiple_captions(img, 3)
                for caption in transformer_captions:
                    if caption and caption not in captions:
                        captions.append(caption)
                        caption_sources.append("transformer")
                print(f"Generated {len(transformer_captions)} captions with transformer model")
            except Exception as e:
                print(f"Error generating transformer captions: {e}")
        
        # 3. Try scene understanding captioner if available
        if (not captions or len(captions) < 4) and scene_captioner is not None:
            try:
                # Use scene captioning
                scene_caption = scene_captioner.generate_caption(
                    img_array,
                    scene_type=scene_type,
                    detected_objects=detected_objects,
                    dominant_colors=dominant_colors,
                    brightness=image_brightness,
                    contrast=image_contrast,
                    has_faces=has_faces
                )
                
                if scene_caption and scene_caption not in captions:
                    captions.append(scene_caption)
                    caption_sources.append("scene")
                    print(f"Caption generated by scene captioner: {scene_caption}")
                
                # Generate multiple variations with different styles
                scene_styles = ["descriptive", "poetic", "technical", "casual", "detailed"]
                for style in scene_styles[:2]:  # Use 2 different styles
                    scene_caption_var = scene_captioner.generate_caption(
                        img_array,
                        scene_type=scene_type,
                        detected_objects=detected_objects,
                        dominant_colors=dominant_colors,
                        brightness=image_brightness,
                        contrast=image_contrast,
                        has_faces=has_faces,
                        style=style
                    )
                    
                    if scene_caption_var and scene_caption_var not in captions:
                        captions.append(scene_caption_var)
                        caption_sources.append("scene")
                        print(f"Generated {style} caption with scene captioner")
            except Exception as e:
                print(f"Error with scene captioning: {e}")
        
        # 4. Try enhanced captioner (produces very good results)
        if (not captions or len(captions) < 5) and enhanced_captioner is not None:
            try:
                # Use enhanced captioning
                enhanced_caption = enhanced_captioner.generate_caption(
                    img_array,
                    scene_type,
                    dominant_colors,
                    detected_objects,
                    has_faces
                )
                if enhanced_caption and enhanced_caption not in captions:
                    captions.append(enhanced_caption)
                    caption_sources.append("enhanced")
                    print(f"Caption generated by enhanced captioner: {enhanced_caption}")
                    
                # Generate a second variation with the enhanced captioner
                enhanced_caption2 = enhanced_captioner.generate_caption(
                    img_array,
                    scene_type,
                    dominant_colors,
                    detected_objects,
                    has_faces
                )
                if enhanced_caption2 and enhanced_caption2 != enhanced_caption and enhanced_caption2 not in captions:
                    captions.append(enhanced_caption2)
                    caption_sources.append("enhanced")
            except Exception as e:
                print(f"Error with enhanced captioning: {e}")
        
        # 5. Final fallback to category-based approach
        if not captions:
            try:
                chosen_category = determine_image_category(
                    features=features, 
                    dominant_colors=dominant_colors,
                    brightness=image_brightness,
                    contrast=image_contrast,
                    has_faces=has_faces,
                    scene_type=scene_type
                )
                
                # Sample captions database
                sample_captions = load_sample_captions()
                
                # Select appropriate caption from the chosen category
                import random
                caption = random.choice(sample_captions.get(chosen_category, ["An interesting image"]))
                captions.append(caption)
                caption_sources.append("category")
                print(f"Determined image category: {chosen_category}")
                print(f"Generated caption from category: {caption}")
            except Exception as e:
                print(f"Error with category-based captioning: {e}")
        
        # If still no captions, add a generic one
        if not captions:
            captions = ["An image worth exploring."]
            caption_sources = ["fallback"]
            
        # Remove any duplicate captions
        unique_captions = []
        unique_sources = []
        for i, caption in enumerate(captions):
            if caption not in unique_captions:
                unique_captions.append(caption)
                unique_sources.append(caption_sources[i])
        
        captions = unique_captions
        caption_sources = unique_sources
        
        # Structure for detected objects to return in API
        objects_list = []
        for obj_name, confidence in detected_objects.items():
            objects_list.append({
                "name": obj_name,
                "confidence": float(confidence)
            })
        
        processing_time = time.time() - start_time
        print(f"Total processing time: {processing_time:.2f} seconds")
        
        return {
            "success": True,
            "caption": captions[0],  # Primary caption (best one)
            "alternative_captions": captions[1:] if len(captions) > 1 else [],  # Alternative options
            "caption_source": caption_sources[0],  # Source of the primary caption
            "caption_sources": caption_sources[1:] if len(caption_sources) > 1 else [],  # Sources for alternative captions
            "features": feature_stats,
            "analysis": {
                "brightness": float(image_brightness),
                "contrast": float(image_contrast),
                "dominant_colors": dominant_colors,
                "scene_type": scene_type,
                "has_faces": has_faces,
                "detected_objects": objects_list,
                "processing_time": f"{processing_time:.2f} seconds"
            }
        }
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "success": False
        }

def analyze_image_colors(img_array):
    """Analyze dominant colors in the image with more precise color detection."""
    # Convert the image to a list of pixels
    pixels = img_array.reshape(-1, 3)
    
    # More refined color categorization with overlapping ranges
    color_ranges = {
        'red': ([0.5, 0.0, 0.0], [1.0, 0.4, 0.4]),
        'bright_red': ([0.7, 0.0, 0.0], [1.0, 0.3, 0.3]),
        'green': ([0.0, 0.5, 0.0], [0.4, 1.0, 0.4]),
        'bright_green': ([0.0, 0.7, 0.0], [0.3, 1.0, 0.3]),
        'grass_green': ([0.2, 0.5, 0.0], [0.4, 0.8, 0.3]),
        'blue': ([0.0, 0.0, 0.5], [0.4, 0.4, 1.0]),
        'sky_blue': ([0.2, 0.5, 0.7], [0.5, 0.8, 1.0]),
        'yellow': ([0.7, 0.7, 0.0], [1.0, 1.0, 0.4]),
        'orange': ([0.8, 0.4, 0.0], [1.0, 0.7, 0.3]),
        'sunset_orange': ([0.8, 0.3, 0.2], [1.0, 0.6, 0.4]),
        'purple': ([0.4, 0.0, 0.4], [0.8, 0.3, 0.8]),
        'pink': ([0.7, 0.3, 0.5], [1.0, 0.8, 0.9]),
        'brown': ([0.3, 0.15, 0.0], [0.7, 0.5, 0.3]),
        'beige': ([0.6, 0.5, 0.4], [0.9, 0.8, 0.7]),
        'cream': ([0.8, 0.8, 0.6], [1.0, 1.0, 0.8]),
        'white': ([0.8, 0.8, 0.8], [1.0, 1.0, 1.0]),
        'light_gray': ([0.6, 0.6, 0.6], [0.8, 0.8, 0.8]),
        'gray': ([0.3, 0.3, 0.3], [0.6, 0.6, 0.6]),
        'dark_gray': ([0.1, 0.1, 0.1], [0.3, 0.3, 0.3]),
        'black': ([0.0, 0.0, 0.0], [0.15, 0.15, 0.15]),
    }
    
    # Count pixels in each color range
    color_counts = {color: 0 for color in color_ranges}
    unclassified = 0
    
    for pixel in pixels:
        classified = False
        for color, (lower, upper) in color_ranges.items():
            if all(lower[i] <= pixel[i] <= upper[i] for i in range(3)):
                color_counts[color] += 1
                classified = True
                break
        if not classified:
            unclassified += 1
    
    # Get top colors
    total_pixels = len(pixels)
    color_percentages = {color: count / total_pixels for color, count in color_counts.items()}
    top_colors = sorted(color_percentages.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Simplified color mapping to reduce redundancy
    simplified_colors = []
    color_mapping = {
        'bright_red': 'red', 
        'bright_green': 'green',
        'grass_green': 'green', 
        'sky_blue': 'blue',
        'sunset_orange': 'orange',
        'beige': 'brown',
        'cream': 'white',
        'light_gray': 'white',
        'dark_gray': 'gray'
    }
    
    # Get colors that occupy a significant portion of the image (>8%)
    significant_colors = []
    for color, percentage in top_colors:
        if percentage > 0.08:
            simplified_color = color_mapping.get(color, color)
            if simplified_color not in significant_colors:
                significant_colors.append(simplified_color)
    
    # Special combinations
    if 'red' in significant_colors and 'orange' in significant_colors:
        if 'yellow' not in significant_colors:
            significant_colors.append('yellow')  # Likely sunset or warm scene
    
    if 'blue' in significant_colors and 'white' in significant_colors and color_percentages.get('blue', 0) > 0.15:
        significant_colors.append('sky')  # Likely sky
        
    if 'green' in significant_colors and color_percentages.get('green', 0) > 0.25:
        significant_colors.append('grass')  # Likely grass or forest
        
    if 'blue' in significant_colors and 'orange' in significant_colors:
        significant_colors.append('sunset')  # Likely sunset
        
    return significant_colors

def detect_faces(img_array):
    """
    Enhanced face and person detection heuristic based on multiple cues.
    This is still a heuristic approach; in a production system, you'd use proper
    face detection models like those in OpenCV, TensorFlow, or specialized libraries.
    """
    height, width, _ = img_array.shape
    
    # Check multiple regions for better detection
    face_regions = [
        img_array[height//4:3*height//4, width//3:2*width//3, :],  # Center face region
        img_array[height//5:2*height//5, width//3:2*width//3, :]   # Upper face region
    ]
    
    # More sophisticated skin tone detection with multiple skin tone ranges
    def check_skin_tone(region):
        # Light skin tones
        light_skin = np.logical_and(
            np.logical_and(region[:, :, 0] > 0.5, region[:, :, 0] < 0.9),
            np.logical_and(
                np.logical_and(region[:, :, 1] > 0.3, region[:, :, 1] < 0.7),
                np.logical_and(region[:, :, 2] > 0.2, region[:, :, 2] < 0.6)
            )
        )
        
        # Medium skin tones
        medium_skin = np.logical_and(
            np.logical_and(region[:, :, 0] > 0.4, region[:, :, 0] < 0.7),
            np.logical_and(
                np.logical_and(region[:, :, 1] > 0.2, region[:, :, 1] < 0.5),
                np.logical_and(region[:, :, 2] > 0.1, region[:, :, 2] < 0.4)
            )
        )
        
        # Darker skin tones
        dark_skin = np.logical_and(
            np.logical_and(region[:, :, 0] > 0.2, region[:, :, 0] < 0.6),
            np.logical_and(
                np.logical_and(region[:, :, 1] > 0.1, region[:, :, 1] < 0.4),
                np.logical_and(region[:, :, 2] > 0.05, region[:, :, 2] < 0.3)
            )
        )
        
        # Combine all skin tone masks
        skin_mask = np.logical_or(np.logical_or(light_skin, medium_skin), dark_skin)
        skin_percentage = np.sum(skin_mask) / (region.shape[0] * region.shape[1])
        
        return skin_percentage
    
    # Check for symmetrical patterns (faces tend to be symmetrical)
    def check_symmetry(region):
        # Calculate symmetry by comparing left and right halves
        left_half = region[:, :region.shape[1]//2, :]
        right_half = region[:, region.shape[1]//2:, :]
        right_half_flipped = np.flip(right_half, axis=1)
        
        # Get minimum width to handle odd dimensions
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        symmetry_diff = np.mean(np.abs(
            left_half[:, :min_width, :] - right_half_flipped[:, :min_width, :]
        ))
        
        # Lower difference means higher symmetry
        symmetry_score = 1.0 - min(symmetry_diff * 2.0, 1.0)
        return symmetry_score
    
    # Calculate scores
    skin_scores = [check_skin_tone(region) for region in face_regions]
    symmetry_scores = [check_symmetry(region) for region in face_regions]
    
    # Combine scores - needs both skin tones and symmetry
    combined_scores = [skin * (0.5 + 0.5 * sym) for skin, sym in zip(skin_scores, symmetry_scores)]
    max_score = max(combined_scores)
    
    # Also check for person body shape
    middle_lower_region = img_array[height//3:, width//4:3*width//4, :]
    vertical_edges = np.mean(np.abs(np.diff(middle_lower_region, axis=0)))
    vertical_structure = vertical_edges > 0.05  # Vertical lines suggest standing person
    
    # Make final determination based on combined evidence
    if max_score > 0.15 or (max_score > 0.1 and vertical_structure):
        confidence = min(max_score * 100, 99)
        if confidence > 70:
            return "Yes (High confidence)"
        elif confidence > 40:
            return "Yes (Medium confidence)"
        else:
            return "Yes (Low confidence)"
    else:
        return "No"

def analyze_scene(img_array):
    """Determine scene type with more precise classification."""
    # Get basic image properties
    brightness = np.mean(img_array)
    contrast = np.std(img_array)
    
    # Divide image into regions for more detailed analysis
    height, width, _ = img_array.shape
    top_region = img_array[:height//3, :, :]
    middle_region = img_array[height//3:2*height//3, :, :]
    bottom_region = img_array[2*height//3:, :, :]
    
    # Check for sky (blue in top region)
    blue_channel_top = np.mean(top_region[:, :, 2])
    red_channel_top = np.mean(top_region[:, :, 0])
    green_channel_top = np.mean(top_region[:, :, 1])
    has_sky = (blue_channel_top > red_channel_top) and (blue_channel_top > green_channel_top)
    sky_intensity = blue_channel_top - (red_channel_top + green_channel_top) / 2
    
    # Check for grass/ground (green in bottom region)
    green_channel_bottom = np.mean(bottom_region[:, :, 1])
    red_channel_bottom = np.mean(bottom_region[:, :, 0])
    blue_channel_bottom = np.mean(bottom_region[:, :, 2])
    has_grass = (green_channel_bottom > red_channel_bottom) and (green_channel_bottom > blue_channel_bottom)
    grass_intensity = green_channel_bottom - (red_channel_bottom + blue_channel_bottom) / 2
    
    # Check for sunset (orange/red in top/middle)
    red_channel_middle = np.mean(middle_region[:, :, 0])
    green_channel_middle = np.mean(middle_region[:, :, 1])
    has_sunset_colors = (red_channel_top > 0.5) and (red_channel_top > blue_channel_top * 1.5)
    
    # Check for uniform color (potential close-up)
    region_stds = [
        np.std(top_region), 
        np.std(middle_region), 
        np.std(bottom_region)
    ]
    color_uniformity = np.mean(region_stds)
    
    # Check for lines/edges (buildings, structures)
    horizontal_diff = np.mean(np.abs(np.diff(img_array, axis=1)))
    vertical_diff = np.mean(np.abs(np.diff(img_array, axis=0)))
    edge_intensity = (horizontal_diff + vertical_diff) / 2
    
    # Determine scene type based on combined factors
    if brightness < 0.3 and contrast < 0.15:
        return "dark"
    elif color_uniformity < 0.1 and contrast < 0.15:
        return "close-up"
    elif has_sky and sky_intensity > 0.1:
        if has_sunset_colors:
            return "sunset"
        elif has_grass and grass_intensity > 0.05:
            return "outdoor-nature"
        else:
            return "outdoor-sky"
    elif edge_intensity > 0.15 and brightness > 0.4:
        if has_grass:
            return "outdoor-urban"
        else:
            return "indoor-structured"
    elif brightness < 0.4:
        return "indoor-dim"
    elif contrast > 0.25:
        if color_uniformity < 0.15:
            return "food"  # High contrast with uniform regions often indicates food
        else:
            return "detailed"
    else:
        # Default cases
        if has_grass:
            return "outdoor-nature"
        elif brightness > 0.6:
            return "bright-scene"
        else:
            return "indoor"

def determine_image_category(features, dominant_colors, brightness, contrast, has_faces, scene_type):
    """Determine the most likely image category based on extracted features and image analysis."""
    # Use all available information to make a more informed decision
    
    # Convert has_faces string to boolean for logic
    has_faces_bool = (has_faces == "Yes")
    
    # Some basic heuristic rules
    if "red" in dominant_colors and "yellow" in dominant_colors and scene_type == "close-up":
        return "pizza"
        
    if "blue" in dominant_colors and brightness > 0.6 and scene_type == "outdoor":
        if "orange" in dominant_colors or "yellow" in dominant_colors:
            return "sunset"
        else:
            return "bird"
    
    if "green" in dominant_colors and scene_type == "outdoor":
        if has_faces_bool:
            return "children"
        else:
            return "dog"
    
    if has_faces_bool and scene_type == "indoor":
        return "reading"
        
    if "brown" in dominant_colors and scene_type == "indoor":
        return "cat"
        
    if "gray" in dominant_colors and scene_type == "outdoor":
        if brightness > 0.5:
            return "car"
        else:
            return "mountain"
            
    if len(dominant_colors) >= 3 and brightness > 0.5:
        return "flowers"
    
    # If none of the rules match, use feature statistics for a fallback decision
    features_mean = features.mean()
    features_std = float(np.std(features))
    
    if features_std > 0.3:
        if scene_type == "outdoor":
            return "mountain" if brightness < 0.5 else "flowers"
        else:
            return "pizza" if "red" in dominant_colors else "cat"
    else:
        if scene_type == "outdoor":
            return "car" if "black" in dominant_colors or "gray" in dominant_colors else "dog"
        else:
            return "reading"

def load_sample_captions():
    """Return hardcoded default captions for fallback scenarios."""
    default_captions = {
        "person": ["A person standing in front of a scenic background",
                  "Someone posing for a photograph outdoors"],
        "dog": ["A golden retriever dog running through a green park on a sunny day",
               "A happy dog playing in the grass with a red ball"],
        "cat": ["A white cat sitting peacefully on a wooden chair by the window",
               "A cute cat resting comfortably on furniture indoors"],
        "bird": ["A colorful bird flying high in the clear blue sky",
                "A beautiful bird soaring through the air with wings spread wide"],
        "car": ["A red sports car parked on a busy city street",
               "A shiny automobile parked alongside other vehicles in an urban area"],
        "food": ["A delicious meal presented on a plate with garnishes",
                "A tasty dish prepared with colorful ingredients"],
        "pizza": ["A delicious pizza with cheese and pepperoni on a wooden table",
                 "A hot pizza with various toppings served on a plate"],
        "sunset": ["A beautiful sunset over calm ocean waves at the beach",
                  "The sun setting over the water with orange and pink colors in the sky"],
        "reading": ["A young woman reading a book while sitting in a comfortable armchair",
                   "A person enjoying a good book in a cozy indoor reading space"],
        "mountain": ["A tall mountain peak covered with snow against a clear blue sky",
                    "A majestic snow-capped mountain rising high above the landscape"],
        "forest": ["A dense forest with tall trees and filtered sunlight",
                  "A path winding through a lush green forest"],
        "beach": ["A sandy beach with waves washing onto the shore",
                 "A tropical beach with palm trees at sunset"],
        "flowers": ["A colorful flower garden with various blooms in full bloom",
                   "A beautiful garden filled with different types of flowers and plants"],
        "building": ["A modern building with glass facades reflecting the sky",
                    "An architectural structure with interesting design elements"],
        "city": ["A cityscape with tall buildings and busy streets",
                "An urban environment with people and buildings"]
    }
    
    return default_captions

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process image."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        result = process_image(filepath)
        return jsonify(result)
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/translate', methods=['POST'])
def translate_text():
    """Translate text to the specified language."""
    try:
        data = request.json
        text = data.get('text', '')
        target_lang = data.get('target_lang', 'en')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Initialize translator
        translator = Translator()
        
        # Translate the text
        translated = translator.translate(text, dest=target_lang)
        
        return jsonify({
            "translated_text": translated.text,
            "source_lang": translated.src,
            "target_lang": target_lang
        })
    
    except Exception as e:
        print(f"Translation error: {e}")
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Server will be available at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    
    # Start the Flask development server
    app.run(debug=True, host='127.0.0.1', port=5000)