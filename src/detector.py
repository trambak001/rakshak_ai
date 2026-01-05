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
            'train': 6,
            'truck': 7,
            'cow': 19,
            'dog': 16
        }
        
        # STATIC OBJECT FILTERING
        # Tracks position of hazards to ignore things that don't move (like AC vents, dashboard)
        self.history = {} # ID -> {'box': [x,y,w,h], 'static_count': 0}
        self.next_id = 0
        
    def reset_history(self):
        self.history = {}
        self.next_id = 0

    def merge_train_cars(self, detections):
        """
        Fixes the glitch where a train is detected as a line of trucks.
        If multiple trucks/buses are aligned horizontally and close, merge them into one 'TRAIN'.
        """
        # Filter for heavy vehicles
        heavies = [d for d in detections if d['label'] in ['truck', 'bus', 'train']]
        others = [d for d in detections if d['label'] not in ['truck', 'bus', 'train']]
        
        if len(heavies) < 2:
            return detections
            
        # Sort by X position
        heavies.sort(key=lambda x: x['box'][0])
        
        merged_heavies = []
        skip_indices = set()
        
        for i in range(len(heavies)):
            if i in skip_indices: continue
            
            # Start a potential train cluster
            cluster = [heavies[i]]
            
            for j in range(i + 1, len(heavies)):
                if j in skip_indices: continue
                
                prev = cluster[-1]
                curr = heavies[j]
                
                # Check horizontal gap (should be small)
                gap = curr['box'][0] - prev['box'][2]
                
                # Check vertical alignment (y-centers should be close)
                prev_cy = (prev['box'][1] + prev['box'][3]) / 2
                curr_cy = (curr['box'][1] + curr['box'][3]) / 2
                y_diff = abs(prev_cy - curr_cy)
                
                # If gap is small (< 50px) and aligned vertically (< 50px)
                if gap < 50 and y_diff < 50:
                    cluster.append(curr)
                    skip_indices.add(j)
                else:
                    break # Gap too big, chain broken
            
            # If cluster has 3+ segments OR explicitly contains a 'train' detection
            has_train_label = any(d['label'] == 'train' for d in cluster)
            
            if len(cluster) >= 3 or (len(cluster) >= 2 and has_train_label):
                # MERGE INTO ONE GIANT TRAIN
                min_x = min(d['box'][0] for d in cluster)
                min_y = min(d['box'][1] for d in cluster)
                max_x = max(d['box'][2] for d in cluster)
                max_y = max(d['box'][3] for d in cluster)
                
                # Calculate aggregated confidence
                avg_conf = sum(d['confidence'] for d in cluster) / len(cluster)
                
                merged_heavies.append({
                    'label': 'TRAIN', # Uppercase to standout
                    'confidence': avg_conf,
                    'box': [min_x, min_y, max_x, max_y],
                    'area': (max_x - min_x) * (max_y - min_y),
                    'distance_index': cluster[0]['distance_index'] # Use closest distance
                })
            else:
                # Not a train, just individual vehicles
                merged_heavies.extend(cluster)
                
        return others + merged_heavies
        
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

    def detect_road_hazards(self, frame, dashboard_mask_ratio=0.0, roi_start_ratio=0.6):
        """
        Advanced pothole detection with debug output.
        dashboard_mask_ratio: Float 0.0 to 0.4 (ignores bottom X% of screen)
        roi_start_ratio: Float 0.0 to 1.0 (start detection at this vertical %)
        """
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ROI: Start at roi_start_ratio, End before the dashboard mask
        y_start = int(height * roi_start_ratio)
        y_end = int(height * (1.0 - dashboard_mask_ratio))
        
        # Ensure we have a valid ROI
        if y_end <= y_start:
            # Fallback for weird configs: minimal 100px ROI
            if y_start < height - 100:
                y_end = height
            else:
                y_start = int(height * 0.5)
                y_end = height
            
        roi_gray = gray[y_start:y_end, :]
        roi_height = y_end - y_start # For relative coord calcs
        
        # - Method 1: Adaptive Dark Spot
        dark_thresh = cv2.adaptiveThreshold(
            roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 51, 15
        )
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_thresh = cv2.erode(dark_thresh, kernel_clean, iterations=1)
        
        # - Method 2: Water Reflection
        # Pass the dynamic y_start so it calculates brightness/saturation correctly for the zone
        water_mask_full, _ = self.detect_water_reflections(frame, y_start)
        water_mask = water_mask_full[:roi_height, :] # Crop to match current strict ROI
        
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
                if aspect_ratio < 0.5 or aspect_ratio > 2.5: continue
                
                # Filter 2: Solidity
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                if solidity < 0.7: continue
                
                # Filter 3: Glare/Reflector
                roi_section = roi_gray[y:y+h, x:x+w]
                avg_intensity = np.mean(roi_section)
                max_intensity = np.max(roi_section)
                if area < 1500 and max_intensity > 220: continue 

                # Filter 4: Bottom Edge (Dashboard bleed)
                # Stricter: ignore if touches bottom 5 pixels
                if (y + h) >= (roi_height - 5): continue

                # Filter 5: Lane Markings
                water_section = water_mask[y:y+h, x:x+w]
                water_percentage = np.sum(water_section > 0) / (w * h) if (w * h) > 0 else 0
                is_water_filled = water_percentage > 0.4
                if avg_intensity > 160 and not is_water_filled: continue 
                
                # --- Valid Detection ---
                real_y = y + y_start
                
                if is_water_filled:
                    label = 'water-filled pothole'
                    confidence = min(0.80 + water_percentage * 0.15, 0.98)
                else:
                    label = 'pothole/drainage'
                    confidence = 0.60 + (solidity * 0.2)
                
                # --- STATIC OBJECT FILTERING LOGIC ---
                current_box = [x, real_y, w, h]
                matched_id = None
                
                for hid, data in self.history.items():
                    hx, hy, hw, hh = data['box']
                    xA = max(x, hx)
                    yA = max(real_y, hy)
                    xB = min(x + w, hx + hw)
                    yB = min(real_y + h, hy + hh)
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    boxArea = w * h
                    histArea = hw * hh
                    iou = interArea / float(boxArea + histArea - interArea)
                    
                    if iou > 0.6: 
                        matched_id = hid
                        break
                
                if matched_id is not None:
                    prev_y = self.history[matched_id]['box'][1]
                    if abs(real_y - prev_y) < 10: # More lenient static check
                         self.history[matched_id]['static_count'] += 1
                         self.history[matched_id]['box'] = current_box
                    else:
                         self.history[matched_id]['static_count'] = max(0, self.history[matched_id]['static_count'] - 1)
                         self.history[matched_id]['box'] = current_box
                         
                    if self.history[matched_id]['static_count'] > 5: # Faster rejection (5 frames)
                        continue 
                else:
                    self.history[self.next_id] = {'box': current_box, 'static_count': 0}
                    self.next_id += 1
                    if len(self.history) > 50: self.history = {}

                road_hazards.append({
                    'label': label,
                    'confidence': confidence,
                    'box': [x, real_y, x+w, real_y+h],
                    'area': area,
                    'distance_index': 1000 / (h + 1),
                    'water_filled': is_water_filled
                })
        
        # Debug Mask
        full_debug_mask = np.zeros((height, width), dtype=np.uint8)
        full_debug_mask[y_start:y_end, :] = combined_mask
        if dashboard_mask_ratio > 0:
            cv2.rectangle(full_debug_mask, (0, y_end), (width, height), 50, -1)
            
        return road_hazards, full_debug_mask

    def detect_hazards(self, frame, enhance=False, dashboard_mask_ratio=0.0, roi_start_ratio=0.6):
        weather = self.analyze_weather(frame)
        
        if enhance:
            mode = 'night' if weather['is_night'] else 'rain'
            frame = self.preprocess_environment(frame, mode=mode)
            
        results = self.model(frame, verbose=False)[0]
        
        detections = []
        # 1. Standard AI Detections (YOLO)
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            label = self.model.names[cls]
            
            # YOLO Filter: Reject things in Dashboard Zone
            # Using same logic as CV: ignore if center of box is below our ROI end
            box_cy = (xyxy[1] + xyxy[3]) / 2
            roi_limit = frame.shape[0] * (1.0 - dashboard_mask_ratio)
            
            if box_cy > roi_limit:
                continue 
            
            if label in ['person', 'car', 'bus', 'truck', 'train', 'cow', 'motorcycle']:
                height = xyxy[3] - xyxy[1]
                detections.append({
                    'label': label,
                    'confidence': conf,
                    'box': xyxy,
                    'area': (xyxy[2]-xyxy[0]) * height,
                    'distance_index': 1000 / (height + 1)
                })
        
        # FIX: Merge trucks into trains if aligned
        detections = self.merge_train_cars(detections)
        
        # 2. Add Road Condition Analysis
        road_hazards, debug_mask = self.detect_road_hazards(
            frame, 
            dashboard_mask_ratio=dashboard_mask_ratio,
            roi_start_ratio=roi_start_ratio
        )
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
