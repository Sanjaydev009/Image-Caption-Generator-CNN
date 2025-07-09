"""
Advanced Scene Understanding Caption Generator
"""

import numpy as np
import random
from collections import defaultdict

class SceneUnderstandingCaptioner:
    """
    A captioning system that focuses on advanced scene understanding
    and contextual relationships between objects in the image.
    """
    
    def __init__(self):
        """Initialize the scene understanding captioner."""
        self.relationship_templates = self._load_relationship_templates()
        self.activity_templates = self._load_activity_templates()
        self.scene_templates = self._load_scene_templates()
        self.adjectives = self._load_adjectives()
        
    def _load_relationship_templates(self):
        """Load templates for describing relationships between objects."""
        return {
            "spatial": [
                "{subject} {position} {object}",
                "{subject} positioned {position} {object}",
                "{subject} located {position} {object}"
            ],
            "interaction": [
                "{subject} {action} {object}",
                "{subject} is {action} {object}",
                "{subject} {action} with {object}"
            ],
            "possession": [
                "{subject} with {object}",
                "{subject} containing {object}",
                "{subject} holding {object}"
            ]
        }
    
    def _load_activity_templates(self):
        """Load templates for activities based on object combinations."""
        return {
            # People-based activities
            "person+outdoors": [
                "A person enjoying time outdoors in {scene}",
                "Someone spending time outside in {scene}",
                "A person exploring {scene}"
            ],
            "person+food": [
                "A person enjoying {food}",
                "Someone having a meal with {food}",
                "A person with {food} ready to eat"
            ],
            "person+animal": [
                "A person with {animal}",
                "Someone interacting with {animal}",
                "A person spending time with {animal}"
            ],
            
            # Nature scenes
            "landscape+sky": [
                "A landscape view with {landscape} under {sky}",
                "A scenic view showing {landscape} beneath {sky}",
                "{sky} above {landscape} in a natural setting"
            ],
            "water+nature": [
                "{water} surrounded by {nature}",
                "A natural scene with {water} and {nature}",
                "{nature} near {water} in a peaceful setting"
            ],
            
            # Urban scenes
            "building+urban": [
                "{building} in an urban environment",
                "A city view featuring {building}",
                "An urban landscape with {building}"
            ],
            
            # Default
            "default": [
                "A scene featuring {objects}",
                "An image showing {objects}",
                "A view containing {objects}"
            ]
        }
    
    def _load_scene_templates(self):
        """Load templates for different scene types."""
        return {
            "outdoor-nature": [
                "A {adj} natural landscape with {elements}",
                "An outdoor scene showing {elements} in a {adj} setting",
                "A {adj} view of nature featuring {elements}"
            ],
            "outdoor-urban": [
                "An urban {adj} scene with {elements}",
                "A city view showing {elements} in a {adj} environment",
                "A {adj} street scene with {elements}"
            ],
            "indoor": [
                "An indoor {adj} space with {elements}",
                "A {adj} interior showing {elements}",
                "A room featuring {elements} in a {adj} setting"
            ],
            "close-up": [
                "A close-up view of {elements} with {adj} details",
                "A detailed {adj} shot of {elements}",
                "A {adj} macro perspective of {elements}"
            ],
            "food": [
                "A {adj} food presentation with {elements}",
                "A {adj} dish featuring {elements}",
                "{elements} presented in a {adj} manner"
            ],
            "portrait": [
                "A {adj} portrait showing {elements}",
                "A {adj} view of {elements} in a portrait style",
                "A portrait-style image of {elements} with {adj} tones"
            ]
        }
    
    def _load_adjectives(self):
        """Load adjectives for different image attributes."""
        return {
            # Scene adjectives
            "scene": {
                "outdoor-nature": ["lush", "scenic", "picturesque", "serene", "tranquil"],
                "outdoor-urban": ["busy", "vibrant", "modern", "lively", "metropolitan"],
                "indoor": ["cozy", "comfortable", "intimate", "spacious", "homey"],
                "close-up": ["detailed", "intricate", "textured", "crisp", "sharp"],
                "food": ["appetizing", "delicious", "tempting", "tasty", "fresh"],
                "sunset": ["golden", "warm", "atmospheric", "glowing", "colorful"],
                "bright-scene": ["bright", "sunlit", "radiant", "airy", "light-filled"]
            },
            
            # Color adjectives
            "color": {
                "red": ["vibrant red", "ruby", "crimson", "scarlet", "red-toned"],
                "green": ["verdant", "lush green", "emerald", "leafy", "green-tinted"],
                "blue": ["azure", "cobalt blue", "sky blue", "deep blue", "blue-toned"],
                "yellow": ["golden", "sunny yellow", "bright yellow", "amber", "yellow-tinged"],
                "white": ["pure white", "snow white", "pristine", "bright white", "alabaster"],
                "black": ["deep black", "charcoal", "midnight", "ebony", "jet black"]
            },
            
            # Brightness/contrast adjectives
            "brightness": {
                "high": ["bright", "well-lit", "luminous", "radiant", "gleaming"],
                "medium": ["moderately lit", "balanced", "evenly lit", "clear", "distinct"],
                "low": ["dim", "subdued", "muted", "moody", "atmospheric"]
            },
            
            # Contrast adjectives
            "contrast": {
                "high": ["high-contrast", "dramatic", "bold", "striking", "vivid"],
                "medium": ["balanced", "defined", "clear", "distinct", "well-defined"],
                "low": ["soft", "gentle", "subtle", "delicate", "mellow"]
            }
        }
    
    def generate_caption(self, img_array=None, scene_type=None, detected_objects=None, dominant_colors=None, 
                        brightness=None, contrast=None, has_faces=None, style=None):
        """
        Generate a detailed caption based on scene understanding.
        
        Args:
            img_array: Image array (optional, not used directly)
            scene_type: Detected scene type
            detected_objects: Dictionary of detected objects and confidence scores
            dominant_colors: List of dominant colors in the image
            brightness: Image brightness value
            contrast: Image contrast value
            has_faces: Whether faces are detected in the image
            style: Optional style to influence caption generation ('descriptive', 'poetic', 'technical', 'casual', etc.)
            
        Returns:
            str: Generated caption
        """
        try:
            # If style is not specified, randomly select one for diversity
            if not style:
                available_styles = ["descriptive", "poetic", "technical", "casual", "detailed", "simple"]
                style = random.choice(available_styles)
                
            # Convert detected_objects to a list if it's a dictionary
            objects_list = []
            if isinstance(detected_objects, dict):
                for obj, conf in detected_objects.items():
                    objects_list.append(obj)
            elif isinstance(detected_objects, list):
                objects_list = detected_objects
            
            # Default values if none provided
            scene_type = scene_type or "general"
            objects_list = objects_list or ["subject"]
            dominant_colors = dominant_colors or ["colorful"]
            has_faces = has_faces or "No"
            
            # Get brightness/contrast category
            brightness_level = "medium"
            if brightness is not None:
                if brightness > 0.6:
                    brightness_level = "high"
                elif brightness < 0.4:
                    brightness_level = "low"
                    
            contrast_level = "medium"
            if contrast is not None:
                if contrast > 0.25:
                    contrast_level = "high"
                elif contrast < 0.15:
                    contrast_level = "low"
            
            # Select appropriate caption strategy based on content
            has_person = "person" in objects_list or "Yes" in has_faces
            
            if has_person and len(objects_list) > 1:
                # Person interacting with objects
                caption = self._generate_person_interaction_caption(
                    objects_list, scene_type, dominant_colors, brightness_level)
            elif scene_type in self.scene_templates:
                # Scene-based caption
                caption = self._generate_scene_caption(
                    scene_type, objects_list, dominant_colors, brightness_level)
            else:
                # Generic objects caption
                caption = self._generate_objects_caption(
                    objects_list, dominant_colors, brightness_level, contrast_level)
            
            # Ensure the caption starts with a capital letter and ends with a period
            caption = caption[0].upper() + caption[1:]
            if not caption.endswith('.'):
                caption += '.'
                
            return caption
            
        except Exception as e:
            print(f"Error generating scene understanding caption: {e}")
            return "An interesting image with various visual elements."
    
    def _generate_person_interaction_caption(self, objects, scene_type, colors, brightness_level):
        """Generate caption for person interacting with objects."""
        # Remove person from objects for interaction descriptions
        other_objects = [obj for obj in objects if obj != "person"]
        
        if not other_objects:
            # Only person in image
            if scene_type.startswith("outdoor"):
                template = random.choice(self.activity_templates["person+outdoors"])
                return template.format(scene=self._describe_scene(scene_type, colors))
            else:
                return f"A person in a {self._get_scene_adjective(scene_type, colors, brightness_level)} {scene_type} setting."
        
        # Choose a primary object for interaction
        primary_object = random.choice(other_objects)
        
        # Check for specific combinations
        if primary_object in ["dog", "cat", "bird", "animal"]:
            template = random.choice(self.activity_templates["person+animal"])
            return template.format(animal=primary_object)
        elif primary_object in ["food", "pizza", "meal", "dish"]:
            template = random.choice(self.activity_templates["person+food"])
            return template.format(food=primary_object)
        else:
            # Generic interaction
            interaction_type = random.choice(list(self.relationship_templates.keys()))
            template = random.choice(self.relationship_templates[interaction_type])
            
            if interaction_type == "spatial":
                position = random.choice(["next to", "beside", "near", "in front of"])
                return template.format(subject="A person", position=position, object=f"a {primary_object}")
            elif interaction_type == "interaction":
                action = random.choice(["interacting with", "looking at", "examining"])
                return template.format(subject="A person", action=action, object=f"a {primary_object}")
            else:
                return template.format(subject="A person", object=f"a {primary_object}")
    
    def _generate_scene_caption(self, scene_type, objects, colors, brightness_level):
        """Generate a scene-based caption."""
        templates = self.scene_templates.get(scene_type, self.scene_templates.get("outdoor-nature"))
        template = random.choice(templates)
        
        # Get adjective for the scene
        adjective = self._get_scene_adjective(scene_type, colors, brightness_level)
        
        # Describe elements in the scene
        if objects and objects[0] != "subject":
            elements = self._describe_objects(objects, colors)
        else:
            elements = self._describe_scene(scene_type, colors)
            
        return template.format(adj=adjective, elements=elements)
    
    def _generate_objects_caption(self, objects, colors, brightness_level, contrast_level):
        """Generate a caption focusing on objects in the image."""
        if not objects or objects[0] == "subject":
            color_adj = self._get_color_adjective(colors)
            brightness_adj = random.choice(self.adjectives["brightness"][brightness_level])
            return f"A {color_adj}, {brightness_adj} image with various elements."
        
        # Format the objects list
        if len(objects) == 1:
            objects_text = f"a {objects[0]}"
        elif len(objects) == 2:
            objects_text = f"a {objects[0]} and a {objects[1]}"
        else:
            objects_text = f"various items including {', '.join(['a ' + obj for obj in objects[:2]])}"
        
        # Select a template based on object combinations
        template_key = "default"
        for key in self.activity_templates:
            if key != "default" and all(obj_type in '+'.join(objects) for obj_type in key.split('+')):
                template_key = key
                break
                
        templates = self.activity_templates[template_key]
        template = random.choice(templates)
        
        if template_key == "default":
            return template.format(objects=objects_text)
        
        # Handle specific template keys
        if template_key == "landscape+sky":
            landscape = next((obj for obj in objects if obj in ["mountain", "field", "forest", "landscape"]), "landscape")
            sky = next((obj for obj in objects if obj in ["sky", "clouds", "sunset"]), "sky")
            return template.format(landscape=landscape, sky=sky)
        
        elif template_key == "water+nature":
            water = next((obj for obj in objects if obj in ["ocean", "lake", "river", "water"]), "water")
            nature = next((obj for obj in objects if obj in ["trees", "forest", "flowers", "greenery"]), "natural elements")
            return template.format(water=water, nature=nature)
        
        elif template_key == "building+urban":
            building = next((obj for obj in objects if obj in ["building", "house", "skyscraper", "structure"]), "buildings")
            return template.format(building=building)
        
        else:
            # For other specific templates
            placeholders = {}
            for part in template_key.split('+'):
                placeholders[part] = next((obj for obj in objects if part in obj), part)
            return template.format(**placeholders)
    
    def _describe_scene(self, scene_type, colors):
        """Generate a description for a scene."""
        if scene_type.startswith("outdoor-nature"):
            elements = random.choice(["natural landscape", "outdoor setting", "natural scenery"])
        elif scene_type.startswith("outdoor-urban"):
            elements = random.choice(["urban environment", "city scene", "metropolitan area"])
        elif scene_type == "indoor":
            elements = random.choice(["interior space", "indoor setting", "room"])
        else:
            elements = f"{scene_type} scene"
            
        # Add color if available
        if colors and colors[0] != "colorful":
            color = colors[0]
            return f"{color} {elements}"
        
        return elements
    
    def _describe_objects(self, objects, colors):
        """Generate a description for objects."""
        if not objects:
            return "various elements"
            
        if len(objects) == 1:
            obj = objects[0]
            if colors and colors[0] != "colorful":
                return f"a {colors[0]} {obj}"
            return f"a {obj}"
            
        elif len(objects) == 2:
            return f"a {objects[0]} and a {objects[1]}"
            
        else:
            main_objects = objects[:2]
            return f"{main_objects[0]} and {main_objects[1]}"
    
    def _get_scene_adjective(self, scene_type, colors, brightness_level):
        """Get an appropriate adjective for the scene."""
        scene_base = scene_type.split('-')[0] if '-' in scene_type else scene_type
        
        # Try to get a scene-specific adjective
        if scene_base in self.adjectives["scene"]:
            scene_adj = random.choice(self.adjectives["scene"][scene_base])
        else:
            scene_adj = random.choice(self.adjectives["scene"]["outdoor-nature"])
        
        # Combine with color or brightness if appropriate
        if random.random() < 0.5 and colors and colors[0] != "colorful":
            color = colors[0]
            if color in self.adjectives["color"]:
                color_adj = random.choice(self.adjectives["color"][color])
                return f"{color_adj} {scene_adj}"
        
        return scene_adj
    
    def _get_color_adjective(self, colors):
        """Get an adjective based on color."""
        if not colors or not colors[0] or colors[0] == "colorful":
            return "colorful"
            
        color = colors[0]
        if color in self.adjectives["color"]:
            return random.choice(self.adjectives["color"][color])
        
        return color
