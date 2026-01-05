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
        
        # 2. Denoising for Rain
        if mode == 'rain':
            # CHANGED: fastNlMeans is too slow (1-2 FPS). Used Bilateral Filter instead.
            # Keeps edges sharp but removes noise, much faster.
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
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
        """Detects water-filled areas - Returns mask AND roi for debug."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]
        roi = hsv[int(height*0.6):, :]
        roi_bgr = frame[int(height*0.6):, :]
        v_channel = roi[:, :, 2]
        s_channel = roi[:, :, 1]
        
        _, bright_mask = cv2.threshold(v_channel, 150, 255, cv2.THRESH_BINARY)
        _, low_sat_mask = cv2.threshold(s_channel, 60, 255, cv2.THRESH_BINARY_INV)
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
        """Advanced pothole detection with debug output."""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[int(height*0.6):, :]
        
        # - Method 1: Adaptive Dark Spot
        dark_thresh = cv2.adaptiveThreshold(
            roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 51, 15
        )
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_thresh = cv2.erode(dark_thresh, kernel_clean, iterations=1)
        
        # - Method 2, 3, 4
        water_mask, _ = self.detect_water_reflections(frame, int(height*0.6))
        edge_mask = self.detect_edge_gradients(roi_gray)
        texture_mask = self.detect_texture_anomalies(roi_gray)
        
        # - Combination
        water_pothole_candidate = cv2.bitwise_and(water_mask, texture_mask)
        water_pothole_candidate = cv2.bitwise_and(water_pothole_candidate, edge_mask)
        combined_mask = cv2.bitwise_or(dark_thresh, water_pothole_candidate)
        
        # - Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        road_hazards = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if 600 < area < 4000:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Filter 1: Aspect Ratio
                aspect_ratio = float(w) / h
                if aspect_ratio < 0.5 or aspect_ratio > 2.0: continue
                
                # Filter 2: Solidity
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                if solidity < 0.7: continue
                
                # Filter 3: REFLECTOR / HEADLIGHT FILTER (New)
                # Dividers/Reflectors are: Small, Compact, EXTREMELY Bright
                roi_section = roi_gray[y:y+h, x:x+w]
                avg_intensity = np.mean(roi_section)
                max_intensity = np.max(roi_section)
                
                # If area is small-ish and it's practically glowing white -> Reflector
                if area < 1000 and max_intensity > 240:
                    continue 

                # Filter 4: Lane Markings
                water_section = water_mask[y:y+h, x:x+w]
                water_percentage = np.sum(water_section > 0) / (w * h) if (w * h) > 0 else 0
                is_water_filled = water_percentage > 0.4
                
                if avg_intensity > 160 and not is_water_filled:
                    continue 
                
                # --- Valid Detection ---
                real_y = y + int(height*0.6)
                
                if is_water_filled:
                    label = 'water-filled pothole'
                    confidence = min(0.80 + water_percentage * 0.15, 0.98)
                else:
                    label = 'pothole/drainage'
                    confidence = 0.60 + (solidity * 0.2)
                
                road_hazards.append({
                    'label': label,
                    'confidence': confidence,
                    'box': [x, real_y, x+w, real_y+h],
                    'area': area,
                    'distance_index': 1000 / (h + 1),
                    'water_filled': is_water_filled
                })
        
        # Return hazardous items AND the debug mask for UI
        return road_hazards, combined_mask

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
        
        # 2. Add Road Condition Analysis
        # Now unpacks both hazards and the debug mask
        road_hazards, debug_mask = self.detect_road_hazards(frame)
        detections.extend(road_hazards)
            
        return detections, frame, weather, debug_mask

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
