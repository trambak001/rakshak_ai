import cv2
import numpy as np
from ultralytics import YOLO
import time

class HazardDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        # Class names for standard YOLOv8
        self.target_classes = {
            'person': 0,
            'bicycle': 1,
            'car': 2,
            'motorcycle': 3,
            'bus': 5,
            'truck': 7,
            'cow': 19,
            'dog': 16
        }
        
    def preprocess_environment(self, frame, mode='night'):
        """Enhances visibility for night/rain using CLAHE and denoising."""
        # 1. Contrast Enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # 2. Denoising for Rain (removes some streaks)
        if mode == 'rain':
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            
        return enhanced

    def analyze_weather(self, frame):
        """Simple heuristic to detect rain/night."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        # Rain often increases edge noise in specific patterns
        # Night is characterized by low average brightness
        is_night = avg_brightness < 80
        
        # For demo purposes, we'll return a score
        return {
            'is_night': is_night,
            'brightness': avg_brightness,
            'status': 'NIGHT' if is_night else 'DAYLIGHT'
        }

    def detect_water_reflections(self, frame, roi_y_start):
        """Detects water-filled areas by analyzing reflections and brightness patterns."""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]
        
        # Focus on road area (bottom 40%)
        roi = hsv[int(height*0.6):, :]
        roi_bgr = frame[int(height*0.6):, :]
        
        # Water typically reflects sky (bright) or lights (high saturation spots)
        # Look for bright patches with low saturation (water reflecting sky)
        v_channel = roi[:, :, 2]
        s_channel = roi[:, :, 1]
        
        # Detect bright areas (potential water reflections)
        _, bright_mask = cv2.threshold(v_channel, 150, 255, cv2.THRESH_BINARY)
        
        # Detect low saturation (water is often grayish/bluish)
        _, low_sat_mask = cv2.threshold(s_channel, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Combine: bright + low saturation = potential water
        water_mask = cv2.bitwise_and(bright_mask, low_sat_mask)
        
        return water_mask, roi_bgr
    
    def detect_edge_gradients(self, roi_gray):
        """Detects potholes using edge gradient analysis - water creates distinct edges."""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        
        # Sobel edge detection (horizontal and vertical)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_mag = np.uint8(gradient_mag / gradient_mag.max() * 255)
        
        # Potholes have strong circular/elliptical edges
        _, edge_thresh = cv2.threshold(gradient_mag, 50, 255, cv2.THRESH_BINARY)
        
        return edge_thresh
    
    def detect_texture_anomalies(self, roi_gray):
        """Detects areas with different texture (water has smoother texture than asphalt)."""
        # Calculate local variance (texture measure)
        # Water-filled areas have lower variance (smoother)
        kernel_size = 15
        mean = cv2.blur(roi_gray, (kernel_size, kernel_size))
        mean_sq = cv2.blur(roi_gray**2, (kernel_size, kernel_size))
        variance = mean_sq - mean**2
        
        # Normalize variance
        variance = np.uint8(variance / variance.max() * 255)
        
        # Low variance areas (smooth = potential water)
        _, smooth_mask = cv2.threshold(variance, 80, 255, cv2.THRESH_BINARY_INV)
        
        return smooth_mask

    def detect_road_hazards(self, frame):
        """Advanced pothole detection including water-filled potholes in rain."""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[int(height*0.6):, :]
        
        # Method 1: Traditional dark spot detection (dry potholes)
        _, dark_thresh = cv2.threshold(roi_gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Method 2: Water reflection detection (water-filled potholes)
        water_mask, roi_bgr = self.detect_water_reflections(frame, int(height*0.6))
        
        # Method 3: Edge gradient analysis
        edge_mask = self.detect_edge_gradients(roi_gray)
        
        # Method 4: Texture anomaly detection
        texture_mask = self.detect_texture_anomalies(roi_gray)
        
        # Combine all methods with weighted approach
        # Dark spots OR (water reflections AND texture anomalies)
        combined_mask = cv2.bitwise_or(dark_thresh, 
                                       cv2.bitwise_and(water_mask, texture_mask))
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        road_hazards = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 8000:  # Increased upper limit for water-filled potholes
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Calculate aspect ratio (potholes are roughly circular)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Filter based on shape (0.5 to 2.0 for reasonable potholes)
                if 0.4 < aspect_ratio < 2.5:
                    # Adjust coordinates back to original frame
                    real_y = y + int(height*0.6)
                    
                    # Calculate confidence based on multiple factors
                    roi_section = roi_gray[y:y+h, x:x+w]
                    avg_intensity = np.mean(roi_section)
                    
                    # Check if it's in water mask (higher confidence for water-filled)
                    water_section = water_mask[y:y+h, x:x+w]
                    water_percentage = np.sum(water_section > 0) / (w * h) if (w * h) > 0 else 0
                    
                    # Determine type and confidence
                    if water_percentage > 0.3:
                        label = 'water-filled pothole'
                        confidence = min(0.75 + water_percentage * 0.2, 0.95)
                    else:
                        label = 'pothole/drainage'
                        confidence = 0.65
                    
                    road_hazards.append({
                        'label': label,
                        'confidence': confidence,
                        'box': [x, real_y, x+w, real_y+h],
                        'area': area,
                        'distance_index': 1000 / (h + 1),
                        'water_filled': water_percentage > 0.3
                    })
        
        return road_hazards

    def detect_hazards(self, frame, enhance=False):
        weather = self.analyze_weather(frame)
        
        if enhance:
            # Auto-select mode based on analysis
            mode = 'night' if weather['is_night'] else 'rain'
            frame = self.preprocess_environment(frame, mode=mode)
            
        results = self.model(frame, verbose=False)[0]
        
        detections = []
        # 1. Standard AI Detections
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            label = self.model.names[cls]
            
            if label in ['person', 'car', 'bus', 'truck', 'cow', 'motorcycle']:
                height = xyxy[3] - xyxy[1]
                detections.append({
                    'label': label,
                    'confidence': conf,
                    'box': xyxy,
                    'area': (xyxy[2]-xyxy[0]) * height,
                    'distance_index': 1000 / (height + 1)
                })
        
        # 2. Add Road Condition Analysis (Potholes/Drainage)
        road_hazards = self.detect_road_hazards(frame)
        detections.extend(road_hazards)
            
        return detections, frame, weather

    def check_accident(self, detections):
        """Simulates accident detection based on proximity and overlap."""
        # Simple logic: if two vehicles have high overlap and are large (close)
        # Or if we find a 'car' that is rotated (hard with standard YOLO)
        # For now, let's just return False unless we see a collision pattern
        vehicles = [d for d in detections if d['label'] in ['car', 'truck', 'bus']]
        if len(vehicles) >= 2:
            # Check overlap
            # This is a simplified placeholder for accident detection logic
            pass
        return False
