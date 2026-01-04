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

    def detect_road_hazards(self, frame):
        """Traditional CV fallback to detect potholes/open drainage."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Focus on the bottom half (road)
        height, width = gray.shape
        roi = gray[int(height*0.6):, :]
        
        # Threshold for dark spots (potholes/open drainage)
        _, thresh = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        road_hazards = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 5000: # Filter small noise and large shadows
                x, y, w, h = cv2.boundingRect(cnt)
                # Adjust coordinates back to original frame
                real_y = y + int(height*0.6)
                road_hazards.append({
                    'label': 'pothole/drainage',
                    'confidence': 0.65,
                    'box': [x, real_y, x+w, real_y+h],
                    'area': area,
                    'distance_index': 1000 / (h + 1)
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
