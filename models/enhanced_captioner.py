"""
Enhanced image captioning using a template-based approach with detailed image analysis
"""

import numpy as np
import cv2
from PIL import Image

class EnhancedImageCaptioner:
    """
    A more sophisticated rule-based image captioning system that uses
    detailed image analysis to generate relevant captions.
    """
    
    def __init__(self):
        """Initialize the captioner with templates and visual word mappings"""
        self.templates = self._load_templates()
        self.color_descriptions = self._load_color_descriptions()
        self.scene_descriptions = self._load_scene_descriptions()
        self.object_descriptions = self._load_object_descriptions()
    
    def _load_templates(self):
        """Load caption templates for different scenarios"""
        return {
            "person": [
                "A {color_adj} {scene_adj} showing a person {action}.",
                "A person {action} in a {color_adj} {scene_type} setting.",
                "{scene_desc} with a person {action}."
            ],
            "group": [
                "A group of people {action} in a {scene_adj} {scene_type}.",
                "Several people {action} together in a {color_adj} environment.",
                "People {action} in a {scene_desc}."
            ],
            "animal": [
                "A {color_adj} {object} in a {scene_adj} {scene_type}.",
                "A {object} {action} in a {scene_desc}.",
                "The image shows a {object} in a {color_adj} {scene_type} setting."
            ],
            "nature": [
                "A {color_adj} {scene_type} landscape with {object_desc}.",
                "A beautiful view of a {scene_adj} {scene_type} {scene_desc}.",
                "A {color_adj} natural scene showing {object_desc}."
            ],
            "urban": [
                "A {scene_adj} urban setting with {object_desc}.",
                "A city view showing {object_desc} with {color_adj} tones.",
                "An urban {scene_type} with {object_desc}."
            ],
            "indoor": [
                "An interior scene showing {object_desc} in a {color_adj} room.",
                "A {color_adj} indoor space with {object_desc}.",
                "A {scene_adj} interior with {object_desc}."
            ],
            "food": [
                "A {color_adj} {object} presented {scene_desc}.",
                "A delicious-looking {object} with {object_desc}.",
                "Food consisting of {object_desc} with {color_adj} appearance."
            ],
            "abstract": [
                "An image with {color_adj} tones and {scene_adj} composition.",
                "A {color_adj} {scene_type} composition with abstract elements.",
                "A {scene_adj} image with {color_adj} colors."
            ]
        }
    
    def _load_color_descriptions(self):
        """Load color-related descriptions"""
        return {
            "red": ["vibrant red", "warm red", "reddish", "red-tinted"],
            "green": ["lush green", "verdant", "greenish", "natural green"],
            "blue": ["cool blue", "azure", "bluish", "deep blue"],
            "yellow": ["bright yellow", "golden", "yellowish", "sunny yellow"],
            "orange": ["warm orange", "amber", "orange-tinted", "sunset orange"],
            "purple": ["rich purple", "violet", "purplish", "lavender"],
            "pink": ["soft pink", "rosy", "pinkish", "light pink"],
            "brown": ["earthy brown", "chocolate", "brownish", "wooden brown"],
            "white": ["clean white", "bright", "pale", "light"],
            "gray": ["neutral gray", "silver", "grayish", "muted"],
            "black": ["deep black", "dark", "shadow", "charcoal"]
        }
    
    def _load_scene_descriptions(self):
        """Load scene-related descriptions"""
        return {
            "outdoor-nature": ["natural", "scenic", "pastoral", "wild", "untamed"],
            "outdoor-urban": ["urban", "busy", "metropolitan", "city", "built-up"],
            "outdoor-sky": ["open", "clear", "airy", "expansive", "vast"],
            "indoor": ["comfortable", "cozy", "interior", "enclosed", "domestic"],
            "indoor-dim": ["atmospheric", "intimate", "dimly lit", "shadowy", "moody"],
            "indoor-structured": ["organized", "structured", "geometric", "designed", "arranged"],
            "bright-scene": ["bright", "well-lit", "sunny", "radiant", "luminous"],
            "dark": ["dark", "low-light", "shadowy", "dim", "moody"],
            "close-up": ["detailed", "close-up", "macro", "intricate", "zoomed-in"],
            "sunset": ["sunset", "golden hour", "evening", "dusk", "twilight"],
            "food": ["culinary", "appetizing", "gastronomic", "tasteful", "gourmet"],
            "detailed": ["complex", "intricate", "detailed", "elaborate", "sophisticated"]
        }
    
    def _load_object_descriptions(self):
        """Load object-related descriptions"""
        return {
            "person": ["individual", "subject", "figure", "person"],
            "dog": ["canine", "pet dog", "furry companion", "dog"],
            "cat": ["feline", "pet cat", "domestic cat", "kitty"],
            "car": ["vehicle", "automobile", "car", "motor vehicle"],
            "tree": ["tall tree", "foliage", "woody plant", "tree"],
            "building": ["structure", "building", "architectural construction", "edifice"],
            "flower": ["blooming flower", "floral element", "colorful bloom", "plant flower"],
            "mountain": ["elevated landform", "peak", "mountain range", "highland"],
            "sky": ["blue sky", "cloudy atmosphere", "overhead expanse", "celestial view"],
            "water": ["body of water", "liquid surface", "aquatic element", "water feature"],
            "food": ["culinary item", "edible arrangement", "dish", "meal"],
            "object": ["item", "object", "element", "subject"]
        }
    
    def _get_person_action(self, scene_type, has_faces, brightness):
        """Determine likely person action based on scene context"""
        if "outdoor" in scene_type:
            if brightness > 0.6:
                return np.random.choice(["enjoying the outdoors", "spending time outside", "posing for a photo outside"])
            else:
                return np.random.choice(["walking", "standing", "exploring the area"])
        elif "indoor" in scene_type:
            return np.random.choice(["sitting comfortably", "standing indoors", "posing for a photo", "relaxing"])
        else:
            return np.random.choice(["appearing in the scene", "shown in the image", "featured in the photo"])
    
    def _get_color_description(self, dominant_colors):
        """Get appropriate color description based on dominant colors"""
        if not dominant_colors:
            return np.random.choice(["colorful", "visually interesting", "textured"])
        
        # Get a description for the most dominant color
        primary_color = dominant_colors[0]
        if primary_color in self.color_descriptions:
            return np.random.choice(self.color_descriptions[primary_color])
        return primary_color
    
    def _get_scene_description(self, scene_type):
        """Get scene description based on scene type"""
        scene_base = scene_type.split('-')[0] if '-' in scene_type else scene_type
        
        if scene_base in self.scene_descriptions:
            adj = np.random.choice(self.scene_descriptions[scene_base])
            
            if scene_base == "outdoor":
                return f"{adj} outdoor scene"
            elif scene_base == "indoor":
                return f"{adj} indoor setting"
            else:
                return f"{adj} {scene_base}"
        
        return scene_type
    
    def _get_object_description(self, detected_objects):
        """Get object description based on detected objects"""
        if not detected_objects:
            return "various elements"
        
        # Get the most confident object
        top_object = max(detected_objects.items(), key=lambda x: x[1])
        obj_name = top_object[0]
        
        if obj_name in self.object_descriptions:
            return np.random.choice(self.object_descriptions[obj_name])
        
        return obj_name
    
    def generate_caption(self, img_array, scene_type, dominant_colors, detected_objects, has_faces):
        """
        Generate a dynamic caption based on image analysis
        
        Args:
            img_array: Numpy array of the image
            scene_type: Detected scene type
            dominant_colors: List of dominant colors
            detected_objects: Dictionary of detected objects and confidence scores
            has_faces: Whether faces were detected
            
        Returns:
            String containing the generated caption
        """
        try:
            # Basic image properties
            brightness = np.mean(img_array)
            
            # Determine category based on image content
            category = self._determine_category(scene_type, detected_objects, has_faces)
            
            # Get template for the category
            templates = self.templates.get(category, self.templates["abstract"])
            template = np.random.choice(templates)
            
            # Fill in template variables
            color_adj = self._get_color_description(dominant_colors)
            scene_adj = np.random.choice(self.scene_descriptions.get(scene_type, ["interesting"]))
            scene_desc = self._get_scene_description(scene_type)
            
            # Get primary object
            main_object = "subject"
            obj_conf = 0
            for obj, conf in detected_objects.items():
                if conf > obj_conf:
                    main_object = obj
                    obj_conf = conf
            
            object_desc = self._get_object_description(detected_objects)
            action = self._get_person_action(scene_type, has_faces, brightness)
            
            # Format the caption
            caption = template.format(
                color_adj=color_adj,
                scene_adj=scene_adj,
                scene_type=scene_type,
                scene_desc=scene_desc,
                object=main_object,
                object_desc=object_desc,
                action=action
            )
            
            return caption
        
        except Exception as e:
            print(f"Error generating enhanced caption: {e}")
            return "An interesting image worth exploring."
    
    def _determine_category(self, scene_type, detected_objects, has_faces):
        """Determine the most appropriate caption category"""
        # Check for people
        has_person = "Yes" in has_faces
        
        if has_person:
            if len(detected_objects) > 1:
                return "group"
            else:
                return "person"
        
        # Check for animals
        for obj in detected_objects:
            if obj in ["dog", "cat", "bird", "animal"]:
                return "animal"
        
        # Scene-based categories
        if "outdoor" in scene_type:
            if "nature" in scene_type or scene_type == "outdoor-sky":
                return "nature"
            else:
                return "urban"
        
        if "indoor" in scene_type:
            return "indoor"
        
        if scene_type == "food":
            return "food"
        
        # Default to abstract for unclassified images
        return "abstract"
