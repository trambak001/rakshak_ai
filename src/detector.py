import cv2
import numpy as np
from ultralytics import YOLO
import time

class HazardDetector:
    def __init__(self, model_path='yolov8s.pt'):
        self.model = YOLO(model_path)
        self.process_width = 640 # Optimization: Downscale for faster processing
        # Global Standard Dimensions (Meters)
        self.standards = {
            'car': {'height': 1.5, 'width': 1.8},
            'truck': {'height': 3.5, 'width': 2.5},
            'bus': {'height': 3.2, 'width': 2.5},
            'person': {'height': 1.7, 'width': 0.6},
            'motorcycle': {'height': 1.2, 'width': 0.8},
            'pothole': {'height': 0.1, 'width': 0.5},
            'speed_breaker': {'height': 0.1, 'width': 3.0}
        }
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
        
        # --- FIXED: PERSPECTIVE MASK (No Potholes in Sky/Buildings/Dividers) ---
        # Force the detection to be within a STRICT road-shaped trapezoid
        h_mask, w_mask = combined_mask.shape
        road_perspective_mask = np.zeros_like(combined_mask)
        # Trapezoid points: Narrower to exclude dividers/service lanes
        poly_pts = np.array([[
            (int(w_mask * 0.10), h_mask),    # Bottom Left (Cut 10% side)
            (int(w_mask * 0.35), 0),         # Top Left (Focus on center horizon)
            (int(w_mask * 0.65), 0),         # Top Right
            (int(w_mask * 0.90), h_mask)     # Bottom Right (Cut 10% side)
        ]], np.int32)
        cv2.fillPoly(road_perspective_mask, [poly_pts], 255)
        
        # Apply the mask: Everything outside the trapezoid becomes 0
        combined_mask = cv2.bitwise_and(combined_mask, road_perspective_mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        road_hazards = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if 300 < area < 8000: # Relaxed area constraints
                x, y, w, h = cv2.boundingRect(cnt)
                real_y = y + y_start # Absolute Y coordinate
                
                # Filter 1: Aspect Ratio
                aspect_ratio = float(w) / h
                if aspect_ratio < 0.3 or aspect_ratio > 4.0: continue # Relaxed aspect ratio
                
                # Filter 2: Solidity (Relaxed for irregular potholes)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                if solidity < 0.5: continue
                
                roi_section = roi_gray[y:y+h, x:x+w]
                avg_intensity = np.mean(roi_section)
                max_intensity = np.max(roi_section)
                # Allow higher intensity for wet roads (glare is common)
                if area < 1000 and max_intensity > 240: continue 

                # Filter 4: Bottom Edge (Dashboard bleed)
                # Stricter: ignore if touches bottom 5 pixels
                if (y + h) >= (roi_height - 5): continue

                # Filter 5: Lane Markings
                water_section = water_mask[y:y+h, x:x+w]
                water_percentage = np.sum(water_section > 0) / (w * h) if (w * h) > 0 else 0
                is_water_filled = water_percentage > 0.3 # Lower threshold for "water-filled"
                
                # Only filter out bright things if they are NOT water
                if avg_intensity > 180 and not is_water_filled: continue 
                
                # --- POTHOLE LEVEL & SEVERITY LOGIC ---
                # Calculate Severity Score (0-10) based on multiple factors
                
                # 1. Size Factor (0-4 points)
                # Bigger = More dangerous
                size_score = min(4, area / 800)
                
                # 2. Depth Factor (Darkness) (0-3 points)
                # Darker = Deeper. Avg intensity is 0(black) to 255(white).
                # We want lower intensity to give higher score.
                depth_score = 0
                if avg_intensity < 50: depth_score = 3     # Very Deep
                elif avg_intensity < 90: depth_score = 2   # Moderate
                elif avg_intensity < 130: depth_score = 1  # Shallow
                
                # 3. Water Factor (Critical Multiplier)
                # Water hides depth, making it inherently dangerous (Level 3 potential)
                water_bonus = 5 if is_water_filled else 0
                
                total_severity = size_score + depth_score + water_bonus
                
                # Assign Levels
                if total_severity >= 5.0 or is_water_filled:
                    level = 3
                    desc = "CRITICAL"
                elif total_severity >= 3.0:
                    level = 2
                    desc = "MODERATE"
                else:
                    level = 1
                    desc = "MINOR"
                
                # --- SPEED BREAKER HEURISTIC ---
                # Speed breakers are typically very wide (> 3x height)
                if aspect_ratio > 3.0 and area > 1200:
                    label = "Speed Breaker"
                    level = 1 # Usually minor unless at high speed
                    desc = "BUMP"
                else:
                    label = f"Pothole L{level}"
                
                if is_water_filled:
                    label = f"Water Pit L{level}"
                    confidence = min(0.80 + water_percentage * 0.15, 0.98)
                else:
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
                    'water_filled': is_water_filled,
                    'pothole_level': level,
                    'pothole_desc': desc
                })
        
        # Debug Mask
        full_debug_mask = np.zeros((height, width), dtype=np.uint8)
        full_debug_mask[y_start:y_end, :] = combined_mask
        if dashboard_mask_ratio > 0:
            cv2.rectangle(full_debug_mask, (0, y_end), (width, height), 50, -1)
            
        return road_hazards, full_debug_mask

    def detect_active_lanes(self, frame):
        """
        Detects lane lines to dynamically determine the drive path.
        Returns (left_boundary_x, right_boundary_x) at the bottom of the frame relative to 640px width.
        """
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Contrast enhancement for better line visualization
        gray = cv2.equalizeHist(gray)
        
        # Edge Detection
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # ROI Mask (Trapezoid focusing on road)
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height),
            (int(width * 0.3), int(height * 0.6)), # Horizon
            (int(width * 0.7), int(height * 0.6)),
            (width, height)
        ]], np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Hough Lines
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=100)
        
        if lines is None:
            return None, None # Fallback
            
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1: continue # Vertical
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter by slope (expecting diagonals)
            if slope < -0.4: # Left lane (negative slope)
                left_lines.append(line[0])
            elif slope > 0.4: # Right lane (positive slope)
                right_lines.append(line[0])
                
        # Calculate Average X intercept at bottom (y=height)
        def get_bottom_x(line_group):
            if not line_group: return None
            x_coords = []
            weights = []
            for l in line_group:
                x1, y1, x2, y2 = l
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                # x = (y - b) / m
                bottom_x = (height - intercept) / slope
                x_coords.append(bottom_x)
                weights.append(length) # Longer lines have more weight
            return np.average(x_coords, weights=weights)
            
        left_x = get_bottom_x(left_lines)
        right_x = get_bottom_x(right_lines)
        
        return left_x, right_x

    def detect_hazards(self, frame, enhance=False, dashboard_mask_ratio=0.0, roi_start_ratio=0.6):
        orig_h, orig_w = frame.shape[:2]
        
        # Optimization: Process at lower resolution
        scale = self.process_width / orig_w
        proc_frame = cv2.resize(frame, (self.process_width, int(orig_h * scale)))
        
        weather = self.analyze_weather(proc_frame)
        
        if enhance:
            mode = 'night' if weather['is_night'] else 'rain'
            proc_frame = self.preprocess_environment(proc_frame, mode=mode)
            
        results = self.model(proc_frame, verbose=False)[0]
        
        detections = []
        # 1. Standard AI Detections (YOLO)
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            # Scale boxes back to original size
            xyxy = (box.xyxy[0] / scale).tolist()
            label = self.model.names[cls]
            
            # YOLO Filter: Reject things in Dashboard Zone OR Far Sides (Divider/Service Lane)
            box_cy = (xyxy[1] + xyxy[3]) / 2
            box_cx = (xyxy[0] + xyxy[2]) / 2
            roi_limit = orig_h * (1.0 - dashboard_mask_ratio)
            
            # 1. Dashboard Check
            if box_cy > roi_limit: continue 
            # 2. Divider/Side Check (Ignore far 10% on edges)
            if box_cx < (orig_w * 0.10) or box_cx > (orig_w * 0.90): continue
            
            if label in ['person', 'car', 'bus', 'truck', 'train', 'cow', 'motorcycle', 'bicycle', 'dog']:
                h = xyxy[3] - xyxy[1]
                w = xyxy[2] - xyxy[0]
                
                # Accuracy: filter small noise
                if h < 20 or w < 20: continue
                
                # Distance calculation using global standards
                obj_std_h = self.standards.get(label, self.standards['car'])['height']
                # focal_length_simulated: assuming 700px focal length for HD
                dist_m = (obj_std_h * 700) / (h + 1)
                
                detections.append({
                    'label': label,
                    'confidence': conf,
                    'box': xyxy,
                    'area': w * h,
                    'distance_m': round(dist_m, 1),
                    'distance_index': 1000 / (h + 1)
                })
        
        detections = self.merge_train_cars(detections)
        
        # 2. Add Road Condition Analysis (on proc_frame for speed)
        road_hazards, debug_mask_small = self.detect_road_hazards(
            proc_frame, 
            dashboard_mask_ratio=dashboard_mask_ratio,
            roi_start_ratio=roi_start_ratio
        )
        
        # Scale road hazards back to original
        for rh in road_hazards:
            box = rh['box']
            rh['box'] = [b / scale for b in box]
            detections.append(rh)
        
        # Scale debug mask back
        debug_mask = cv2.resize(debug_mask_small, (orig_w, orig_h))
        
        # --- DYNAMIC LANE ANALYSIS ---
        # 1. Detect lanes on the small frame
        l_x, r_x = self.detect_active_lanes(proc_frame)
        
        # 2. Scale up to original resolution
        # Default Logic (Fallback): 30% | 40% | 30% split
        lane_left_x = orig_w * 0.35
        lane_right_x = orig_w * 0.65
        dynamic_lanes_found = False
        
        if l_x is not None or r_x is not None:
            # If we only found one line, infer the other based on standard lane width (~50% of screen at bottom?)
            # Heuristic: Lane width usually ~40-50% of screen width at bottom
            inferred_width = orig_w * 0.45
            
            # Scale up detected points
            cur_lx = l_x / scale if l_x is not None else None
            cur_rx = r_x / scale if r_x is not None else None
            
            if cur_lx is not None and cur_rx is not None:
                # Found both!
                lane_left_x = cur_lx
                lane_right_x = cur_rx
                dynamic_lanes_found = True
            elif cur_lx is not None:
                # Found Left only
                lane_left_x = cur_lx
                lane_right_x = cur_lx + inferred_width
            elif cur_rx is not None:
                # Found Right only
                lane_right_x = cur_rx
                lane_left_x = cur_rx - inferred_width
                
            # Sanity Limits: Don't let lanes cross or go off screen
            lane_left_x = max(0, min(lane_left_x, orig_w * 0.45))
            lane_right_x = min(orig_w, max(lane_right_x, orig_w * 0.55))

        # 3. Post-Process: Dynamic Lane Classification
        for d in detections:
            box = d['box']
            center_x = (box[0] + box[2]) / 2
            # norm_x = center_x / orig_w # Legacy
            
            # Dynamic Lane Assignment
            # < Left_Line : Left Lane
            # Left_Line < x < Right_Line : Center (Ego) Lane
            # > Right_Line : Right Lane
            
            # Additional 'Shoulder' buffers (15% of width)
            shoulder_buffer = orig_w * 0.15
            
            if center_x < (lane_left_x - shoulder_buffer):
                d['lane'] = 'Left Shoulder'
                d['lane_id'] = 0
            elif center_x < lane_left_x:
                d['lane'] = 'Left Lane'
                d['lane_id'] = 1
            elif center_x < lane_right_x:
                d['lane'] = 'Ego Lane' # DRIVING PATH
                d['lane_id'] = 2
            elif center_x < (lane_right_x + shoulder_buffer):
                d['lane'] = 'Right Lane'
                d['lane_id'] = 3
            else:
                d['lane'] = 'Right Shoulder'
                d['lane_id'] = 4
            
            # Unified Distance & TTC
            dist_m = d.get('distance_m')
            if dist_m is None:
                dist_m = 1000 / ((box[3]-box[1]) + 1) * 0.5 
                d['distance_m'] = round(dist_m, 1)
            
            closing_speed = 15.0 
            d['ttc'] = round(dist_m / closing_speed, 2)
            
            # --- INTELLIGENT SEVERITY LOGIC ---
            severity = 0
            
            # Path Logic:
            if d['lane_id'] == 2: # DIRECTLY IN PATH
                severity = 6
                if d['ttc'] < 2.5: severity += 4 # CRITICAL (10)
            elif d['lane_id'] in [1, 3]: # ADJACENT LANES
                severity = 3
                # If very close and moving laterally (heuristic), bump it
                if d['ttc'] < 1.0: severity += 2 # Warning (5)
            else: # SHOULDERS
                severity = 1 # Observation only
                
            # Vulnerable Road User Logic
            # People/Animals are critical if they are anywhere near the road (lanes 1,2,3)
            # But NOT if they are safe on the shoulder
            is_living = d['label'] in ['person', 'cow', 'dog', 'bicycle']
            if is_living:
                if d['lane_id'] in [1, 2, 3]: # On the road
                    severity = max(severity, 7) # Automatic High Alert
                elif d['lane_id'] in [0, 4]: # On shoulder
                    severity = 2 # Safe but monitor
            
            d['severity'] = min(10, severity)

        return detections, frame, weather, debug_mask

    def check_accident(self, detections):
        """Simulates accident detection based on proximity and high overlap."""
        vehicles = [d for d in detections if d['label'].lower() in ['car', 'truck', 'bus', 'train']]
        if len(vehicles) >= 2:
            for i in range(len(vehicles)):
                for j in range(i + 1, len(vehicles)):
                    v1, v2 = vehicles[i], vehicles[j]
                    b1, b2 = v1['box'], v2['box']
                    
                    # Calculate IOU
                    xA = max(b1[0], b2[0])
                    yA = max(b1[1], b2[1])
                    xB = min(b1[2], b2[2])
                    yB = min(b1[3], b2[3])
                    
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    v1_area = (b1[2]-b1[0]) * (b1[3]-b1[1])
                    v2_area = (b2[2]-b2[0]) * (b2[3]-b2[1])
                    
                    iou = interArea / float(v1_area + v2_area - interArea) if (v1_area + v2_area - interArea) > 0 else 0
                    
                    # If two close vehicles overlap significantly, it's a potential crash
                    if iou > 0.45 and v1['distance_index'] > 5:
                        return True
        return False
