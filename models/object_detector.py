import numpy as np
import cv2
from PIL import Image

class ObjectDetector:
    """
    A class for detecting objects in images.
    This implementation uses simple heuristics for demonstration,
    but could be replaced with a more sophisticated model like YOLO or SSD.
    """
    
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the object detector.
        
        Args:
            confidence_threshold: Threshold for object detection confidence
        """
        self.confidence_threshold = confidence_threshold
        self.common_objects = [
            'person', 'dog', 'cat', 'car', 'bicycle', 'bird', 'flower',
            'tree', 'building', 'mountain', 'beach', 'food', 'book'
        ]
        
    def detect_objects(self, img_array):
        """
        Detect objects in an image using color and texture features.
        This is a simplified implementation that uses heuristics.
        
        Args:
            img_array: Numpy array of the image (RGB)
            
        Returns:
            Dictionary of detected objects and their confidence scores
        """
        # Convert to different color spaces for analysis
        height, width, _ = img_array.shape
        img_hsv = cv2.cvtColor(np.uint8(img_array * 255), cv2.COLOR_RGB2HSV)
        img_gray = cv2.cvtColor(np.uint8(img_array * 255), cv2.COLOR_RGB2GRAY)
        
        # Calculate image features
        avg_hsv = np.mean(img_hsv, axis=(0, 1))
        std_hsv = np.std(img_hsv, axis=(0, 1))
        
        # Calculate texture features using Haralick texture
        try:
            texture_features = self._calculate_texture_features(img_gray)
        except Exception as e:
            print(f"Error calculating texture features: {e}")
            texture_features = np.zeros(5)
        
        # Calculate edge features
        edge_features = self._calculate_edge_features(img_gray)
        
        # Detect objects based on features
        objects_detected = {}
        
        # Person detection
        if self._detect_person(img_array, img_hsv, texture_features):
            objects_detected['person'] = 0.8
        
        # Animal detection
        animal_score, animal_type = self._detect_animal(img_array, img_hsv, texture_features)
        if animal_score > self.confidence_threshold:
            objects_detected[animal_type] = animal_score
        
        # Vehicle detection
        vehicle_score, vehicle_type = self._detect_vehicle(img_array, edge_features)
        if vehicle_score > self.confidence_threshold:
            objects_detected[vehicle_type] = vehicle_score
        
        # Nature elements detection
        nature_score, nature_type = self._detect_nature_elements(img_array, img_hsv)
        if nature_score > self.confidence_threshold:
            objects_detected[nature_type] = nature_score
        
        # Food detection
        if self._detect_food(img_array, img_hsv, texture_features):
            objects_detected['food'] = 0.7
        
        # If no specific objects detected, add generic object
        if not objects_detected:
            objects_detected['object'] = 0.5
        
        return objects_detected
    
    def _calculate_texture_features(self, img_gray):
        """Calculate Haralick texture features."""
        from skimage.feature import graycomatrix, graycoprops
        
        # Normalize and convert to uint8
        img_gray_norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
        img_gray_uint8 = img_gray_norm.astype(np.uint8)
        
        # Calculate GLCM
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(
            img_gray_uint8, 
            distances=distances, 
            angles=angles, 
            symmetric=True, 
            normed=True
        )
        
        # Calculate properties
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        
        return np.array([contrast, dissimilarity, homogeneity, energy, correlation])
    
    def _calculate_edge_features(self, img_gray):
        """Calculate edge features using Canny edge detector."""
        # Apply Canny edge detector
        edges = cv2.Canny(img_gray.astype(np.uint8), 50, 150)
        
        # Calculate edge density and statistics
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255)
        
        # Calculate horizontal and vertical edge strength
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        horizontal_strength = np.mean(np.abs(sobelx))
        vertical_strength = np.mean(np.abs(sobely))
        ratio = horizontal_strength / (vertical_strength + 1e-5)
        
        return np.array([edge_density, horizontal_strength, vertical_strength, ratio])
    
    def _detect_person(self, img_array, img_hsv, texture_features):
        """Detect if a person is in the image."""
        height, width, _ = img_array.shape
        
        # Check for skin tones
        skin_mask = self._get_skin_mask(img_hsv)
        skin_percentage = np.sum(skin_mask) / (height * width)
        
        # Check for facial features - simplified heuristic
        face_region = img_array[height//4:3*height//4, width//3:2*width//3, :]
        face_hsv = cv2.cvtColor(np.uint8(face_region * 255), cv2.COLOR_RGB2HSV)
        face_skin_mask = self._get_skin_mask(face_hsv)
        face_skin_percentage = np.sum(face_skin_mask) / (face_region.shape[0] * face_region.shape[1])
        
        # Check texture features - skin has specific texture characteristics
        skin_texture = texture_features[2] > 0.6  # High homogeneity for skin
        
        # Combine evidence
        if (skin_percentage > 0.1 and face_skin_percentage > 0.3) or (skin_percentage > 0.2 and skin_texture):
            return True
        return False
    
    def _get_skin_mask(self, img_hsv):
        """Create a mask for skin tones in HSV space."""
        # Skin tone ranges in HSV
        lower_skin1 = np.array([0, 20, 70])
        upper_skin1 = np.array([20, 150, 255])
        
        lower_skin2 = np.array([170, 20, 70])
        upper_skin2 = np.array([180, 150, 255])
        
        # Create masks
        mask1 = np.logical_and(
            np.logical_and(img_hsv[:, :, 0] >= lower_skin1[0], img_hsv[:, :, 0] <= upper_skin1[0]),
            np.logical_and(
                np.logical_and(img_hsv[:, :, 1] >= lower_skin1[1], img_hsv[:, :, 1] <= upper_skin1[1]),
                np.logical_and(img_hsv[:, :, 2] >= lower_skin1[2], img_hsv[:, :, 2] <= upper_skin1[2])
            )
        )
        
        mask2 = np.logical_and(
            np.logical_and(img_hsv[:, :, 0] >= lower_skin2[0], img_hsv[:, :, 0] <= upper_skin2[0]),
            np.logical_and(
                np.logical_and(img_hsv[:, :, 1] >= lower_skin2[1], img_hsv[:, :, 1] <= upper_skin2[1]),
                np.logical_and(img_hsv[:, :, 2] >= lower_skin2[2], img_hsv[:, :, 2] <= upper_skin2[2])
            )
        )
        
        return np.logical_or(mask1, mask2)
    
    def _detect_animal(self, img_array, img_hsv, texture_features):
        """Detect if an animal (dog or cat) is in the image."""
        # Simplified heuristics for animal detection
        # Dogs often have higher contrast and texture
        dog_score = texture_features[0] * 0.8  # Higher contrast for dogs
        
        # Cats often have smoother texture
        cat_score = texture_features[2] * 0.7  # Higher homogeneity for cats
        
        if dog_score > cat_score and dog_score > 0.5:
            return dog_score, 'dog'
        elif cat_score > 0.5:
            return cat_score, 'cat'
        else:
            return 0.0, 'unknown'
    
    def _detect_vehicle(self, img_array, edge_features):
        """Detect if a vehicle is in the image."""
        # Vehicles typically have strong edges and geometric patterns
        edge_density, horizontal, vertical, ratio = edge_features
        
        # Cars typically have strong horizontal edges
        car_score = edge_density * (horizontal / (vertical + 1e-5)) * 0.7
        
        if car_score > 0.4:
            return car_score, 'car'
        else:
            return 0.0, 'unknown'
    
    def _detect_nature_elements(self, img_array, img_hsv):
        """Detect nature elements like mountains, trees, sky, etc."""
        height, width, _ = img_array.shape
        
        # Check for blue sky
        sky_region = img_array[:height//3, :, :]
        sky_hsv = cv2.cvtColor(np.uint8(sky_region * 255), cv2.COLOR_RGB2HSV)
        blue_mask = (sky_hsv[:, :, 0] >= 100) & (sky_hsv[:, :, 0] <= 140) & (sky_hsv[:, :, 1] >= 40)
        sky_percentage = np.sum(blue_mask) / (sky_region.shape[0] * sky_region.shape[1])
        
        # Check for green vegetation
        green_mask = (img_hsv[:, :, 0] >= 35) & (img_hsv[:, :, 0] <= 85) & (img_hsv[:, :, 1] >= 40)
        green_percentage = np.sum(green_mask) / (height * width)
        
        # Check for mountains (typically at the horizon with characteristic shapes)
        mountain_region = img_array[height//4:height//2, :, :]
        mountain_gray = cv2.cvtColor(np.uint8(mountain_region * 255), cv2.COLOR_RGB2GRAY)
        mountain_edges = cv2.Canny(mountain_gray, 50, 150)
        mountain_edge_density = np.sum(mountain_edges) / (mountain_edges.shape[0] * mountain_edges.shape[1] * 255)
        
        # Determine nature element
        if sky_percentage > 0.4:
            return 0.8, 'sky'
        elif green_percentage > 0.3:
            return 0.8, 'trees'
        elif mountain_edge_density > 0.1:
            return 0.7, 'mountain'
        else:
            return 0.0, 'unknown'
    
    def _detect_food(self, img_array, img_hsv, texture_features):
        """Detect if food is in the image."""
        # Food typically has rich colors and complex textures
        avg_saturation = np.mean(img_hsv[:, :, 1])
        color_variety = np.std(img_hsv[:, :, 0])
        
        # High contrast and energy are common in food images
        contrast = texture_features[0]
        energy = texture_features[3]
        
        # Combine evidence
        food_score = (avg_saturation / 255) * color_variety * contrast * energy * 5
        
        return food_score > 0.4
