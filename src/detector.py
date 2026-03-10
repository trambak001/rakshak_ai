"""
Rakshak AI - HazardDetector
CPU-Optimized for Intel i3 (no GPU)
─────────────────────────────────────────────────────────────────────────────
Key optimizations applied:
  1. OpenVINO IR model loader  — ~3x faster CPU inference than PyTorch .pt
  2. 320×320 inference input   — 4x fewer pixels vs 640×640 (significant speedup)
  3. Frame-skip counter        — caller can use skip_frame() to process every 3rd
  4. Pothole ROI = bottom 40%  — matches Rakshak AI methodology (section 3.4.2)
─────────────────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

# ── OpenVINO optional import ──────────────────────────────────────────────────
try:
    from openvino.runtime import Core as OVCore
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

# ── Model paths priority list ─────────────────────────────────────────────────
OPENVINO_MODEL_PATH = 'models/rakshak_openvino/best.xml'   # Preferred: 3x CPU speed
PYTORCH_MODEL_PATH  = 'models/rakshak_best.pt'             # Fallback 1: custom trained
NANO_MODEL_PATH     = 'yolov8n.pt'                          # Fallback 2: YOLOv8 Nano


class HazardDetector:
    """
    Rakshak AI hazard detector.
    Optimized for real-time inference on Intel i3 CPU without a GPU.
    """

    def __init__(self, model_path=None):
        """
        Load model with priority:
          1. OpenVINO IR (.xml) — fastest on Intel CPU
          2. Custom .pt         — YOLOv8 nano trained model
          3. yolov8n.pt         — stock Nano model (fallback)
          4. caller-supplied path
        """
        self.use_openvino = False
        self.ov_compiled   = None   # OpenVINO compiled model
        self.ov_input_name = None
        self.model         = None   # YOLO / PyTorch model

        # ── Frame-skip state ──────────────────────────────────────────────────
        # Every 3rd frame is processed; others return the last result.
        # This cuts CPU load by ~66% without affecting human-perceivable accuracy.
        self._frame_counter  = 0
        self._SKIP_N         = 3       # Process 1 out of every N frames
        self._last_result    = None    # Cache of previous detection result

        # ── Inference resolution ──────────────────────────────────────────────
        # 320×320 = 4x fewer pixels than 640×640, dramatically faster on CPU.
        # Display is always at original resolution (we scale boxes back up).
        self.infer_size = 320

        if model_path is not None:
            self._load_model(model_path)
        else:
            self._auto_load_model()

        # Standard object dimensions (meters) for distance estimation
        self.standards = {
            'car':           {'height': 1.5, 'width': 1.8},
            'truck':         {'height': 3.5, 'width': 2.5},
            'bus':           {'height': 3.2, 'width': 2.5},
            'person':        {'height': 1.7, 'width': 0.6},
            'motorcycle':    {'height': 1.2, 'width': 0.8},
            'auto-rickshaw': {'height': 1.8, 'width': 1.4},
            'cow':           {'height': 1.4, 'width': 0.7},
            'pothole':       {'height': 0.1, 'width': 0.5},
            'speed_breaker': {'height': 0.1, 'width': 3.0},
        }

        # Target class labels (Rakshak AI v2 — matches methodology)
        self.target_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'bus',
            'truck', 'cow', 'auto-rickshaw', 'dog',
            # COCO fallback IDs kept for stock yolov8n.pt
            'train',
        ]

        # Static object filtering (ignore things stuck in one place = dashboard artifacts)
        self.history  = {}
        self.next_id  = 0

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _auto_load_model(self):
        """Auto-detect and load the best available model."""
        if OPENVINO_AVAILABLE and os.path.exists(OPENVINO_MODEL_PATH):
            self._load_openvino(OPENVINO_MODEL_PATH)
        elif os.path.exists(PYTORCH_MODEL_PATH):
            self._load_pytorch(PYTORCH_MODEL_PATH)
            print("✅ Using custom-trained model: models/rakshak_best.pt")
        elif os.path.exists(NANO_MODEL_PATH):
            self._load_pytorch(NANO_MODEL_PATH)
            print("✅ Using YOLOv8n (Nano) — optimized for Intel i3 CPU")
        else:
            # Last resort: download Nano on-demand
            self._load_pytorch('yolov8n.pt')
            print("⚡ Downloading YOLOv8n (Nano) model — this runs on CPU without a GPU")

    def _load_openvino(self, xml_path):
        """Load OpenVINO IR model for maximum CPU throughput."""
        try:
            ie = OVCore()
            net = ie.read_model(xml_path)
            self.ov_compiled   = ie.compile_model(net, device_name='CPU')
            self.ov_input_name = self.ov_compiled.input(0).any_name
            self.use_openvino  = True
            print(f"🚀 OpenVINO model loaded: {xml_path}")
            print("   Intel CPU optimizations active (~3x faster inference)")
        except Exception as e:
            print(f"⚠️  OpenVINO load failed ({e}). Falling back to PyTorch.")
            self.use_openvino = False
            self._load_pytorch(PYTORCH_MODEL_PATH if os.path.exists(PYTORCH_MODEL_PATH) else NANO_MODEL_PATH)

    def _load_pytorch(self, model_path):
        """Load standard YOLO / PyTorch model."""
        self.model = YOLO(model_path)
        self.use_openvino = False

    def _load_model(self, path):
        """Load model from an explicit caller-supplied path."""
        if path.endswith('.xml') and OPENVINO_AVAILABLE:
            self._load_openvino(path)
        else:
            self._load_pytorch(path)

    # ── Frame-Skip Helper ─────────────────────────────────────────────────────

    def skip_frame(self):
        """
        Returns True if the current frame should be SKIPPED (not processed).
        Increments internal counter on every call.

        Usage in main.py detection loop:
            if detector.skip_frame():
                dets, proc_frame, weather, dbg = detector.get_last_result(frame)
            else:
                dets, proc_frame, weather, dbg = detector.detect_hazards(frame, ...)
        """
        self._frame_counter += 1
        should_skip = (self._frame_counter % self._SKIP_N) != 0
        return should_skip

    def get_last_result(self, frame):
        """
        Return cached result from previous detection frame.
        Used when current frame is skipped.
        Falls back to a blank result if no prior detection has run yet.
        """
        if self._last_result is not None:
            dets, _, weather, dbg = self._last_result
            return dets, frame.copy(), weather, dbg
        # No prior result — return empty
        empty_weather = {'is_night': False, 'brightness': 128.0, 'status': 'DAYLIGHT'}
        return [], frame.copy(), empty_weather, None

    # ── Static-object filter helpers ─────────────────────────────────────────

    def reset_history(self):
        self.history  = {}
        self.next_id  = 0

    # ── Environment preprocessing ─────────────────────────────────────────────

    def preprocess_environment(self, frame, mode='night'):
        """Enhances visibility for night/rain using CLAHE and denoising."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        if mode == 'rain':
            # Bilateral filter — keeps edges sharp while removing rain noise
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

        return enhanced

    def analyze_weather(self, frame):
        """Simple heuristic to detect rain/night from frame brightness."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        is_night = avg_brightness < 80
        return {
            'is_night':   is_night,
            'brightness': avg_brightness,
            'status':     'NIGHT' if is_night else 'DAYLIGHT',
        }

    # ── Pothole / Road-hazard detection ──────────────────────────────────────

    def detect_water_reflections(self, frame, roi_y_start):
        """Detect water-filled areas — returns mask AND roi for debug."""
        hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]
        roi    = hsv[int(height * 0.6):, :]
        roi_bgr = frame[int(height * 0.6):, :]
        v_channel = roi[:, :, 2]
        s_channel = roi[:, :, 1]

        _, bright_mask  = cv2.threshold(v_channel, 140, 255, cv2.THRESH_BINARY)
        _, low_sat_mask = cv2.threshold(s_channel, 70,  255, cv2.THRESH_BINARY_INV)
        water_mask = cv2.bitwise_and(bright_mask, low_sat_mask)
        return water_mask, roi_bgr

    def detect_edge_gradients(self, roi_gray):
        """Detect potholes using Sobel edge gradient analysis."""
        blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        max_val = gradient_mag.max()
        if max_val == 0:
            return np.zeros_like(roi_gray, dtype=np.uint8)
        gradient_mag = np.uint8(gradient_mag / max_val * 255)
        _, edge_thresh = cv2.threshold(gradient_mag, 50, 255, cv2.THRESH_BINARY)
        return edge_thresh

    def detect_texture_anomalies(self, roi_gray):
        """Detect water via smooth texture (water = low local variance)."""
        kernel_size = 15
        roi_f = roi_gray.astype(np.float32)
        mean   = cv2.blur(roi_f, (kernel_size, kernel_size))
        mean_sq= cv2.blur(roi_f**2, (kernel_size, kernel_size))
        variance = mean_sq - mean**2
        max_var = variance.max()
        if max_var == 0:
            return np.zeros_like(roi_gray, dtype=np.uint8)
        variance = np.uint8(variance / max_var * 255)
        _, smooth_mask = cv2.threshold(variance, 80, 255, cv2.THRESH_BINARY_INV)
        return smooth_mask

    def detect_road_hazards(self, frame, dashboard_mask_ratio=0.0, roi_start_ratio=0.60):
        """
        Pothole + road hazard detection using classical CV.

        ┌─────────────────────────────────────────────────────────┐
        │ METHODOLOGY §3.4.2 — Pothole ROI                       │
        │ Only the BOTTOM 40% of screen is scanned for potholes. │
        │ This eliminates false positives from sky/buildings.    │
        │ roi_start_ratio is therefore clamped to ≥ 0.60.        │
        └─────────────────────────────────────────────────────────┘

        dashboard_mask_ratio : Float 0–0.4 — ignore bottom X% (dashboard)
        roi_start_ratio      : Float 0–1.0 — start scan at this Y% of frame
                               Clamped to max(0.60, roi_start_ratio) so that
                               detection never scans above the bottom 40%.
        """
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Bottom-40% ROI enforcement ────────────────────────────────────────
        # Per Rakshak AI methodology: potholes are only in the near-road zone
        # which occupies the bottom 40% of the dashcam frame.
        roi_start_ratio = max(0.60, roi_start_ratio)   # Never scan above 60% mark

        y_start = int(height * roi_start_ratio)
        y_end   = int(height * (1.0 - dashboard_mask_ratio))

        if y_end <= y_start:
            if y_start < height - 80:
                y_end = height
            else:
                y_start = int(height * 0.60)
                y_end   = height

        roi_gray   = gray[y_start:y_end, :]
        roi_height = y_end - y_start

        # Method 1: Adaptive dark-spot thresholding (dry potholes)
        dark_thresh = cv2.adaptiveThreshold(
            roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 15
        )
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_thresh = cv2.erode(dark_thresh, kernel_clean, iterations=1)

        # Method 2: Water reflection detection
        water_mask_full, _ = self.detect_water_reflections(frame, y_start)
        water_mask = water_mask_full[:roi_height, :]

        edge_mask    = self.detect_edge_gradients(roi_gray)
        texture_mask = self.detect_texture_anomalies(roi_gray)

        # Combine: water candidate = water ∩ texture ∩ edges
        water_pothole_candidate = cv2.bitwise_and(water_mask, texture_mask)
        water_pothole_candidate = cv2.bitwise_and(water_pothole_candidate, edge_mask)
        combined_mask = cv2.bitwise_or(dark_thresh, water_pothole_candidate)

        # Morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,   kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel, iterations=1)

        # Perspective mask (trapezoid) — excludes dividers and side lanes
        h_mask, w_mask = combined_mask.shape
        road_mask = np.zeros_like(combined_mask)
        poly_pts = np.array([[
            (int(w_mask * 0.10), h_mask),
            (int(w_mask * 0.35), 0),
            (int(w_mask * 0.65), 0),
            (int(w_mask * 0.90), h_mask),
        ]], np.int32)
        cv2.fillPoly(road_mask, [poly_pts], 255)
        combined_mask = cv2.bitwise_and(combined_mask, road_mask)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        road_hazards = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (400 < area < 7000):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            real_y = y + y_start

            aspect_ratio = float(w) / (h + 1e-5)
            if aspect_ratio < 0.4 or aspect_ratio > 3.0:
                continue

            hull      = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity  = float(area) / hull_area if hull_area > 0 else 0
            if solidity < 0.55:
                continue

            roi_section   = roi_gray[y:y+h, x:x+w]
            avg_intensity = np.mean(roi_section)
            max_intensity = np.max(roi_section)

            if area < 1000 and max_intensity > 240:
                continue
            if (y + h) >= (roi_height - 5):
                continue

            water_section    = water_mask[y:y+h, x:x+w]
            water_percentage = (np.sum(water_section > 0) / (w * h)) if (w * h) > 0 else 0
            is_water_filled  = water_percentage > 0.3

            if avg_intensity > 180 and not is_water_filled:
                continue

            # Severity scoring (0–10)
            size_score  = min(4, area / 800)
            depth_score = (3 if avg_intensity < 50 else 2 if avg_intensity < 90
                           else 1 if avg_intensity < 130 else 0)
            water_bonus    = 5 if is_water_filled else 0
            total_severity = size_score + depth_score + water_bonus

            if total_severity >= 5.0 or is_water_filled:
                level, desc = 3, "CRITICAL"
            elif total_severity >= 3.0:
                level, desc = 2, "MODERATE"
            else:
                level, desc = 1, "MINOR"

            # Speed-breaker heuristic
            if aspect_ratio > 3.0 and area > 1200:
                label = "Speed Breaker"
                level, desc = 1, "BUMP"
            else:
                label = f"Pothole L{level}"

            # Confidence
            if is_water_filled:
                label      = f"Water Pit L{level}"
                confidence = min(0.70 + water_percentage*0.3 + solidity*0.25
                                 + min(area/5000, 1.0)*0.15, 0.98)
            else:
                confidence = min(0.60 + solidity*0.30
                                 + min(area/5000, 1.0)*0.15
                                 + (1.0 - avg_intensity/255.0)*0.20, 0.95)

            # Static object filtering (skip things stuck in the same position)
            current_box = [x, real_y, w, h]
            matched_id  = None
            for hid, data in self.history.items():
                hx, hy, hw, hh = data['box']
                xA = max(x, hx);  yA = max(real_y, hy)
                xB = min(x+w, hx+hw); yB = min(real_y+h, hy+hh)
                interArea = max(0, xB-xA) * max(0, yB-yA)
                boxArea   = w * h;  histArea = hw * hh
                iou       = interArea / float(boxArea + histArea - interArea) if (boxArea + histArea - interArea) > 0 else 0
                if iou > 0.6:
                    matched_id = hid
                    break

            if matched_id is not None:
                prev_y = self.history[matched_id]['box'][1]
                if abs(real_y - prev_y) < 10:
                    self.history[matched_id]['static_count'] += 1
                    self.history[matched_id]['box'] = current_box
                else:
                    self.history[matched_id]['static_count'] = max(0, self.history[matched_id]['static_count'] - 1)
                    self.history[matched_id]['box'] = current_box
                if self.history[matched_id]['static_count'] > 5:
                    continue
            else:
                self.history[self.next_id] = {'box': current_box, 'static_count': 0}
                self.next_id += 1
                if len(self.history) > 50:
                    self.history = {}

            road_hazards.append({
                'label':        label,
                'confidence':   confidence,
                'box':          [x, real_y, x+w, real_y+h],
                'area':         area,
                'distance_index': 1000 / (h + 1),
                'water_filled': is_water_filled,
                'pothole_level': level,
                'pothole_desc': desc,
            })

        # Build full-size debug mask
        full_debug_mask = np.zeros((height, width), dtype=np.uint8)
        full_debug_mask[y_start:y_end, :] = combined_mask
        if dashboard_mask_ratio > 0:
            cv2.rectangle(full_debug_mask, (0, y_end), (width, height), 50, -1)

        return road_hazards, full_debug_mask

    # ── YOLO / OpenVINO inference ─────────────────────────────────────────────

    def _run_yolo(self, proc_frame):
        """
        Run YOLO inference on a pre-processed (already-resized) frame.
        Returns raw ultralytics Results object.
        """
        return self.model(proc_frame, verbose=False)[0]

    def _run_openvino(self, proc_frame, orig_h, orig_w):
        """
        Run OpenVINO inference.
        Input: BGR frame already resized to self.infer_size × self.infer_size.
        Returns list of dicts matching YOLO boxes format.
        """
        # Normalize + transpose to NCHW
        blob = cv2.dnn.blobFromImage(
            proc_frame.astype(np.float32) / 255.0,
            scalefactor=1.0,
            size=(self.infer_size, self.infer_size),
            swapRB=True,
            crop=False,
        )
        outputs   = self.ov_compiled({self.ov_input_name: blob})
        out_key   = list(outputs.keys())[0]
        raw_out   = outputs[out_key][0]   # shape: [num_classes+4, num_anchors] or similar

        detections = []
        # YOLOv8 OpenVINO output is (num_anchors, 4+nc)
        if raw_out.ndim == 2 and raw_out.shape[1] > 4:
            for det in raw_out.T:
                bbox        = det[:4]
                class_probs = det[4:]
                cls_id      = int(np.argmax(class_probs))
                conf        = float(class_probs[cls_id])
                if conf < 0.25:
                    continue
                # Decode cx,cy,w,h → x1,y1,x2,y2 in original resolution
                cx, cy, bw, bh = bbox
                sx = orig_w / self.infer_size
                sy = orig_h / self.infer_size
                x1 = (cx - bw / 2) * sx
                y1 = (cy - bh / 2) * sy
                x2 = (cx + bw / 2) * sx
                y2 = (cy + bh / 2) * sy
                detections.append({
                    'cls':  cls_id,
                    'conf': conf,
                    'xyxy': [x1, y1, x2, y2],
                })
        return detections

    # ── Lane detection ────────────────────────────────────────────────────────

    def merge_train_cars(self, detections):
        """Merge overlapping trucks/buses detected as a single train."""
        heavies = [d for d in detections if d['label'] in ['truck', 'bus', 'train']]
        others  = [d for d in detections if d['label'] not in ['truck', 'bus', 'train']]

        if len(heavies) < 2:
            return detections

        heavies.sort(key=lambda x: x['box'][0])
        merged_heavies = []
        skip_indices   = set()

        for i in range(len(heavies)):
            if i in skip_indices:
                continue
            cluster = [heavies[i]]
            for j in range(i + 1, len(heavies)):
                if j in skip_indices:
                    continue
                prev, curr = cluster[-1], heavies[j]
                gap  = curr['box'][0] - prev['box'][2]
                prev_cy = (prev['box'][1] + prev['box'][3]) / 2
                curr_cy = (curr['box'][1] + curr['box'][3]) / 2
                if gap < 50 and abs(prev_cy - curr_cy) < 50:
                    cluster.append(curr)
                    skip_indices.add(j)
                else:
                    break

            has_train = any(d['label'] == 'train' for d in cluster)
            if len(cluster) >= 3 or (len(cluster) >= 2 and has_train):
                min_x = min(d['box'][0] for d in cluster)
                min_y = min(d['box'][1] for d in cluster)
                max_x = max(d['box'][2] for d in cluster)
                max_y = max(d['box'][3] for d in cluster)
                avg_conf = sum(d['confidence'] for d in cluster) / len(cluster)
                merged_heavies.append({
                    'label': 'TRAIN',
                    'confidence': avg_conf,
                    'box': [min_x, min_y, max_x, max_y],
                    'area': (max_x - min_x) * (max_y - min_y),
                    'distance_index': cluster[0]['distance_index'],
                })
            else:
                merged_heavies.extend(cluster)

        return others + merged_heavies

    def detect_active_lanes(self, frame):
        """
        Detect lane boundaries using Canny + Hough lines.
        Returns (left_boundary_x, right_boundary_x) at bottom of frame.
        """
        height, width = frame.shape[:2]
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height),
            (int(width*0.3), int(height*0.6)),
            (int(width*0.7), int(height*0.6)),
            (width, height),
        ]], np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50,
                                 minLineLength=40, maxLineGap=100)
        if lines is None:
            return None, None

        left_lines, right_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if slope < -0.4:
                left_lines.append(line[0])
            elif slope > 0.4:
                right_lines.append(line[0])

        def get_bottom_x(line_group):
            if not line_group:
                return None
            x_coords, weights = [], []
            for l in line_group:
                x1, y1, x2, y2 = l
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                slope  = (y2-y1) / (x2-x1)
                intercept = y1 - slope * x1
                bottom_x  = (height - intercept) / slope
                x_coords.append(bottom_x)
                weights.append(length)
            return np.average(x_coords, weights=weights)

        return get_bottom_x(left_lines), get_bottom_x(right_lines)

    # ── Main detect_hazards ───────────────────────────────────────────────────

    def detect_hazards(self, frame, enhance=False, dashboard_mask_ratio=0.0, roi_start_ratio=0.6):
        """
        Full detection pipeline.

        Steps:
          1. Downscale frame to 320×320 for inference (CPU speedup)
          2. Optional night/rain enhancement
          3. YOLO (OpenVINO or PyTorch) inference
          4. Classical CV pothole detection (bottom 40% ROI)
          5. Lane boundary detection
          6. Per-detection: lane classification, distance, TTC, severity
          7. Cache result for frame-skip retrieval
          8. Return at original display resolution
        """
        orig_h, orig_w = frame.shape[:2]

        # ── Step 1: Downscale to 320×320 for inference ────────────────────────
        proc_frame = cv2.resize(frame, (self.infer_size, self.infer_size))

        # ── Step 2: Weather analysis + optional enhancement ───────────────────
        weather = self.analyze_weather(proc_frame)
        if enhance:
            mode       = 'night' if weather['is_night'] else 'rain'
            proc_frame = self.preprocess_environment(proc_frame, mode=mode)

        # ── Step 3: Object detection ──────────────────────────────────────────
        detections = []
        raw_scale  = orig_w / self.infer_size   # used to scale boxes → original res

        if self.use_openvino:
            # OpenVINO inference path
            ov_dets = self._run_openvino(proc_frame, orig_h, orig_w)
            for d in ov_dets:
                cls_id = d['cls']
                conf   = d['conf']
                xyxy   = d['xyxy']

                # Resolve class name from loaded YOLO model names or generic COCO names
                try:
                    label = self.model.names[cls_id] if self.model else f"class{cls_id}"
                except Exception:
                    label = f"class{cls_id}"

                if label not in self.target_classes:
                    continue

                h = xyxy[3] - xyxy[1]
                w = xyxy[2] - xyxy[0]
                if h < 20 or w < 20:
                    continue

                obj_std_h = self.standards.get(label, self.standards['car'])['height']
                dist_m    = (obj_std_h * 700) / (h + 1)
                detections.append({
                    'label':          label,
                    'confidence':     conf,
                    'box':            xyxy,
                    'area':           w * h,
                    'distance_m':     round(dist_m, 1),
                    'distance_index': 1000 / (h + 1),
                })
        else:
            # PyTorch / YOLO inference path
            results = self._run_yolo(proc_frame)
            for box in results.boxes:
                cls   = int(box.cls[0])
                conf  = float(box.conf[0])
                xyxy  = (box.xyxy[0] * raw_scale).tolist()
                label = self.model.names[cls]

                box_cy  = (xyxy[1] + xyxy[3]) / 2
                box_cx  = (xyxy[0] + xyxy[2]) / 2
                roi_lim = orig_h * (1.0 - dashboard_mask_ratio)

                if box_cy > roi_lim:
                    continue
                if box_cx < (orig_w * 0.10) or box_cx > (orig_w * 0.90):
                    continue

                if label not in self.target_classes:
                    continue

                h = xyxy[3] - xyxy[1]
                w = xyxy[2] - xyxy[0]
                if h < 20 or w < 20:
                    continue

                obj_std_h = self.standards.get(label, self.standards['car'])['height']
                dist_m    = (obj_std_h * 700) / (h + 1)
                detections.append({
                    'label':          label,
                    'confidence':     conf,
                    'box':            xyxy,
                    'area':           w * h,
                    'distance_m':     round(dist_m, 1),
                    'distance_index': 1000 / (h + 1),
                })

        detections = self.merge_train_cars(detections)

        # ── Step 4: Classical pothole detection (bottom 40% ROI) ──────────────
        road_hazards, debug_mask_small = self.detect_road_hazards(
            proc_frame,
            dashboard_mask_ratio=dashboard_mask_ratio,
            roi_start_ratio=roi_start_ratio,   # internally clamped to ≥ 0.60
        )

        for rh in road_hazards:
            # Scale pothole boxes from inference res back to display res
            rh['box'] = [b * raw_scale for b in rh['box']]
            detections.append(rh)

        # Scale debug mask back to display resolution
        debug_mask = cv2.resize(debug_mask_small, (orig_w, orig_h))

        # ── Step 5: Lane detection ────────────────────────────────────────────
        l_x, r_x = self.detect_active_lanes(proc_frame)

        lane_left_x  = orig_w * 0.35
        lane_right_x = orig_w * 0.65

        if l_x is not None or r_x is not None:
            inferred_width = orig_w * 0.45
            cur_lx = l_x * raw_scale if l_x is not None else None
            cur_rx = r_x * raw_scale if r_x is not None else None

            if cur_lx is not None and cur_rx is not None:
                lane_left_x, lane_right_x = cur_lx, cur_rx
            elif cur_lx is not None:
                lane_left_x  = cur_lx
                lane_right_x = cur_lx + inferred_width
            elif cur_rx is not None:
                lane_right_x = cur_rx
                lane_left_x  = cur_rx - inferred_width

            lane_left_x  = max(0,       min(lane_left_x,  orig_w * 0.45))
            lane_right_x = min(orig_w,  max(lane_right_x, orig_w * 0.55))

        # ── Step 6: Per-detection classification ─────────────────────────────
        shoulder_buffer = orig_w * 0.15
        for d in detections:
            box      = d['box']
            center_x = (box[0] + box[2]) / 2

            if   center_x < (lane_left_x - shoulder_buffer):
                d['lane'], d['lane_id'] = 'Left Shoulder', 0
            elif center_x < lane_left_x:
                d['lane'], d['lane_id'] = 'Left Lane',     1
            elif center_x < lane_right_x:
                d['lane'], d['lane_id'] = 'Ego Lane',      2
            elif center_x < (lane_right_x + shoulder_buffer):
                d['lane'], d['lane_id'] = 'Right Lane',    3
            else:
                d['lane'], d['lane_id'] = 'Right Shoulder', 4

            dist_m = d.get('distance_m')
            if dist_m is None:
                dist_m = 1000 / ((box[3] - box[1]) + 1) * 0.5
                d['distance_m'] = round(dist_m, 1)

            d['ttc'] = round(dist_m / 15.0, 2)

            # Severity (0–10)
            severity = 0
            if d['lane_id'] == 2:
                severity = 6
                if d['ttc'] < 2.5:
                    severity += 4
            elif d['lane_id'] in [1, 3]:
                severity = 3
                if d['ttc'] < 1.0:
                    severity += 2
            else:
                severity = 1

            is_living = d['label'] in ['person', 'cow', 'dog', 'bicycle', 'auto-rickshaw']
            if is_living:
                severity = max(severity, 7) if d['lane_id'] in [1, 2, 3] else 2

            d['severity'] = min(10, severity)

        # ── Step 7: Upscale proc_frame back for display ───────────────────────
        output_frame = cv2.resize(proc_frame, (orig_w, orig_h))

        # ── Step 8: Cache result for frame-skip ──────────────────────────────
        self._last_result = (detections, output_frame, weather, debug_mask)

        return detections, output_frame, weather, debug_mask

    # ── Accident detection ────────────────────────────────────────────────────

    def check_accident(self, detections):
        """Simulates accident detection based on IOU overlap of nearby vehicles."""
        vehicles = [d for d in detections if d['label'].lower()
                    in ['car', 'truck', 'bus', 'train']]
        if len(vehicles) < 2:
            return False

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1, v2 = vehicles[i], vehicles[j]
                b1, b2 = v1['box'], v2['box']
                xA = max(b1[0], b2[0]); yA = max(b1[1], b2[1])
                xB = min(b1[2], b2[2]); yB = min(b1[3], b2[3])
                interArea = max(0, xB-xA) * max(0, yB-yA)
                v1_area   = (b1[2]-b1[0]) * (b1[3]-b1[1])
                v2_area   = (b2[2]-b2[0]) * (b2[3]-b2[1])
                denom     = float(v1_area + v2_area - interArea)
                iou       = interArea / denom if denom > 0 else 0
                if iou > 0.45 and v1['distance_index'] > 5:
                    return True
        return False
