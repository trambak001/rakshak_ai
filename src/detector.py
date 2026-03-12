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
import math

# ── OpenVINO optional import ──────────────────────────────────────────────────
try:
    from openvino.runtime import Core as OVCore
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

# ── Model paths priority list ─────────────────────────────────────────────────
# OpenVINO: point to the DIRECTORY exported by `model.export(format='openvino')`.
# Ultralytics YOLO natively loads the whole directory with task='detect'.
# This is the recommended approach — no need for raw openvino.runtime.Core.
OPENVINO_MODEL_DIR  = 'models/rakshak_openvino'          # ← directory, not .xml
OPENVINO_MODEL_XML  = 'models/rakshak_openvino/best.xml'  # ← fallback raw path
PYTORCH_MODEL_PATH  = 'models/rakshak_best.pt'            # Fallback 1: custom .pt
NANO_MODEL_PATH     = 'yolov8m.pt'                        # Fallback 2: stock Medium


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
        self.use_openvino        = False   # True = raw OVCore path (ov_compiled set)
        self.use_openvino_native  = False   # True = Ultralytics-native OpenVINO dir load
        self.ov_compiled          = None    # Only used by _load_openvino_raw / _run_openvino
        self.ov_input_name        = None
        self.model                = None    # YOLO / PyTorch / Ultralytics-OpenVINO model

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
        
        # Cross-frame tracking for Smart Mute / Traffic Mode logic
        self.object_tracks = {}
        self.next_track_id = 0

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _auto_load_model(self):
        """Auto-detect and load the best available model.

        Priority:
          1. models/rakshak_openvino/best.xml via openvino.runtime.Core
             — fastest on Intel CPU (requires: pip install openvino)
          2. models/rakshak_best.pt  — custom-trained YOLOv8n PyTorch
          3. yolov8n.pt              — stock Nano (auto-download)
        """
        if OPENVINO_AVAILABLE and os.path.exists(OPENVINO_MODEL_XML):
            # ── PRIMARY: Raw openvino.runtime.Core ─────────────────────────────
            # YOLO() rejects .xml in newer Ultralytics builds; OVCore is reliable.
            self._load_openvino_raw(OPENVINO_MODEL_XML)
            # Also load a YOLO model for class names if OVCore succeeded
            if self.use_openvino and self.model is None:
                try:
                    self.model = YOLO(NANO_MODEL_PATH if os.path.exists(NANO_MODEL_PATH) else 'yolov8n.pt')
                except Exception:
                    pass   # class names won't resolve — falls back to f"class{id}"
        elif os.path.exists(PYTORCH_MODEL_PATH):
            self._load_pytorch(PYTORCH_MODEL_PATH)
            print("✅ Using custom-trained model: models/rakshak_best.pt")
        elif os.path.exists(NANO_MODEL_PATH):
            self._load_pytorch(NANO_MODEL_PATH)
            print("✅ Using YOLOv8m (Medium) — optimized for Intel CPU")
        else:
            self._load_pytorch('yolov8m.pt')
            print("⚡ Downloading YOLOv8m (Medium) — CPU-only, no GPU required")

    def _load_openvino_native(self, model_xml_path):
        """
        Load OpenVINO model via Ultralytics YOLO engine.

        IMPORTANT: Ultralytics YOLO() needs the .xml FILE path, NOT the directory.
          Correct:   YOLO('models/rakshak_openvino/best.xml', task='detect')
          Wrong:     YOLO('models/rakshak_openvino/', task='detect')  ← TypeError

        self.model is a standard Ultralytics YOLO object — inference via _run_yolo().
        self.ov_compiled is NOT used here.
        """
        try:
            self.model              = YOLO(model_xml_path, task='detect')
            # Warm up once to force AutoBackend to load the OpenVINO runtime
            import numpy as np
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)   # triggers actual OpenVINO compile
            self.use_openvino_native = True    # route to _run_yolo (Ultralytics handles OV)
            self.use_openvino        = False   # ov_compiled is NOT set in this path
            print(f"🚀 OpenVINO model loaded (Ultralytics native): {model_xml_path}")
            print("   Intel CPU optimizations active — ~3x faster than PyTorch .pt")
        except Exception as e:
            print(f"⚠️  OpenVINO native load failed ({e}).")
            self.use_openvino_native = False
            self.use_openvino        = False
            self.model               = None
            # Try raw OVCore next
            if OPENVINO_AVAILABLE and os.path.exists(model_xml_path):
                print("   Falling back to raw openvino.runtime.Core ...")
                try:
                    self._load_openvino_raw(model_xml_path)
                    return
                except Exception as e2:
                    print(f"   Raw OVCore also failed ({e2}). Falling back to PyTorch.")
            fallback = PYTORCH_MODEL_PATH if os.path.exists(PYTORCH_MODEL_PATH) else NANO_MODEL_PATH
            self._load_pytorch(fallback)


    def _load_openvino_raw(self, xml_path):
        """Load OpenVINO IR via raw openvino.runtime.Core.
        ov_compiled IS set here.  detect_hazards routes to _run_openvino().
        """
        try:
            ie = OVCore()
            net = ie.read_model(xml_path)
            self.ov_compiled         = ie.compile_model(net, device_name='CPU')
            self.ov_input_name       = self.ov_compiled.input(0).any_name
            self.use_openvino        = True   # route to _run_openvino (ov_compiled used)
            self.use_openvino_native = False
            print(f"🚀 OpenVINO model loaded (raw Core): {xml_path}")
        except Exception as e:
            print(f"⚠️  OpenVINO raw load failed ({e}). Falling back to PyTorch.")
            self.use_openvino        = False
            self.use_openvino_native = False
            self._load_pytorch(PYTORCH_MODEL_PATH if os.path.exists(PYTORCH_MODEL_PATH) else NANO_MODEL_PATH)

    # Keep old name as alias for backward compat
    def _load_openvino(self, path):
        """Alias: route to native or raw loader based on path type."""
        if path.endswith('.xml'):
            self._load_openvino_native(path)    # pass .xml directly to YOLO()
        elif os.path.isdir(path):
            # If given a directory, look for .xml inside it
            xml = os.path.join(path, 'best.xml')
            if os.path.exists(xml):
                self._load_openvino_native(xml)
            else:
                self._load_pytorch(PYTORCH_MODEL_PATH if os.path.exists(PYTORCH_MODEL_PATH) else NANO_MODEL_PATH)
        else:
            self._load_openvino_native(path)

    def _load_model(self, path):
        """Load model from an explicit caller-supplied path."""
        if path.endswith('.xml') and os.path.exists(path):
            self._load_openvino_native(path)
        elif os.path.isdir(path):
            # Directory → look for .xml inside
            xml = os.path.join(path, 'best.xml')
            if os.path.exists(xml):
                self._load_openvino_native(xml)
            else:
                self._load_pytorch(path)   # might be a PyTorch saved model dir
        else:
            self._load_pytorch(path)

    def _load_pytorch(self, model_path):
        """Load standard YOLO / PyTorch model."""
        self.model = YOLO(model_path)
        self.use_openvino = False
        print(f"✅ Loaded PyTorch model: {model_path}")



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
        return self.model.track(proc_frame, persist=True, tracker="botsort.yaml", verbose=False)[0]

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

        if self.use_openvino and self.ov_compiled is not None:
            # ── Raw OVCore path: ov_compiled is set by _load_openvino_raw() ────────
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

                box_cy = (xyxy[1] + xyxy[3]) / 2.0
                
                # Dynamic Horizon Filter: Target ground objects only
                is_pothole_label = label in ['pothole', 'water-pothole', 'crack', 'drainage', 'bump']
                if is_pothole_label and box_cy < (orig_h * 0.60):
                    continue

                is_water_filled = False
                if is_pothole_label and w >= 10 and h >= 10:
                    x1, y1, x2, y2 = map(int, xyxy)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(orig_w, x2), min(orig_h, y2)
                    crop_bgr = frame[y1:y2, x1:x2]
                    if crop_bgr.size > 0:
                        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                        dark_thresh = cv2.adaptiveThreshold(crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
                        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
                        _, bright_mask = cv2.threshold(hsv[:,:,2], 140, 255, cv2.THRESH_BINARY)
                        _, low_sat_mask = cv2.threshold(hsv[:,:,1], 70, 255, cv2.THRESH_BINARY_INV)
                        water_mask = cv2.bitwise_and(bright_mask, low_sat_mask)
                        
                        blurred = cv2.GaussianBlur(crop_gray, (5, 5), 0)
                        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
                        max_val = gradient_mag.max()
                        edge_thresh = np.uint8(gradient_mag / max_val * 255) if max_val > 0 else np.zeros_like(crop_gray)
                        _, edge_thresh = cv2.threshold(edge_thresh, 50, 255, cv2.THRESH_BINARY)

                        roi_f = crop_gray.astype(np.float32)
                        mean = cv2.blur(roi_f, (15, 15))
                        mean_sq = cv2.blur(roi_f**2, (15, 15))
                        variance = mean_sq - mean**2
                        max_var = variance.max()
                        variance = np.uint8(variance / max_var * 255) if max_var > 0 else np.zeros_like(crop_gray)
                        _, texture_mask = cv2.threshold(variance, 80, 255, cv2.THRESH_BINARY_INV)

                        water_pothole_candidate = cv2.bitwise_and(water_mask, texture_mask)
                        water_pothole_candidate = cv2.bitwise_and(water_pothole_candidate, edge_thresh)
                        combined_mask = cv2.bitwise_or(dark_thresh, water_pothole_candidate)
                        
                        water_percentage = np.sum(water_mask > 0) / (w * h)
                        is_water_filled = bool(water_percentage > 0.3)

                obj_std_h = self.standards.get(label, self.standards['car'])['height']
                dist_m    = (obj_std_h * 700) / (h + 1)
                detections.append({
                    'label':          label,
                    'confidence':     conf,
                    'box':            xyxy,
                    'area':           w * h,
                    'distance_m':     round(dist_m, 1),
                    'distance_index': 1000 / (h + 1),
                    'water_filled':   is_water_filled,
                })

        else:
            # ── Ultralytics path: covers ──────────────────────────────────────
            #   a) Native OpenVINO dir load  (use_openvino_native=True, ov_compiled=None)
            #      Ultralytics runs OpenVINO internally via self.model()
            #   b) Standard PyTorch .pt load (use_openvino=False)
            # Both cases return an Ultralytics Results object from _run_yolo()
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

                # Dynamic Horizon Filter: Target ground objects only
                is_pothole_label = label in ['pothole', 'water-pothole', 'crack', 'drainage', 'bump']
                if is_pothole_label and box_cy < (orig_h * 0.60):
                    continue

                is_water_filled = False
                if is_pothole_label and w >= 10 and h >= 10:
                    x1, y1, x2, y2 = map(int, xyxy)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(orig_w, x2), min(orig_h, y2)
                    crop_bgr = frame[y1:y2, x1:x2]
                    if crop_bgr.size > 0:
                        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                        dark_thresh = cv2.adaptiveThreshold(crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
                        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
                        _, bright_mask = cv2.threshold(hsv[:,:,2], 140, 255, cv2.THRESH_BINARY)
                        _, low_sat_mask = cv2.threshold(hsv[:,:,1], 70, 255, cv2.THRESH_BINARY_INV)
                        water_mask = cv2.bitwise_and(bright_mask, low_sat_mask)
                        
                        blurred = cv2.GaussianBlur(crop_gray, (5, 5), 0)
                        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
                        max_val = gradient_mag.max()
                        edge_thresh = np.uint8(gradient_mag / max_val * 255) if max_val > 0 else np.zeros_like(crop_gray)
                        _, edge_thresh = cv2.threshold(edge_thresh, 50, 255, cv2.THRESH_BINARY)

                        roi_f = crop_gray.astype(np.float32)
                        mean = cv2.blur(roi_f, (15, 15))
                        mean_sq = cv2.blur(roi_f**2, (15, 15))
                        variance = mean_sq - mean**2
                        max_var = variance.max()
                        variance = np.uint8(variance / max_var * 255) if max_var > 0 else np.zeros_like(crop_gray)
                        _, texture_mask = cv2.threshold(variance, 80, 255, cv2.THRESH_BINARY_INV)

                        water_pothole_candidate = cv2.bitwise_and(water_mask, texture_mask)
                        water_pothole_candidate = cv2.bitwise_and(water_pothole_candidate, edge_thresh)
                        combined_mask = cv2.bitwise_or(dark_thresh, water_pothole_candidate)
                        
                        water_percentage = np.sum(water_mask > 0) / (w * h)
                        is_water_filled = bool(water_percentage > 0.3)

                obj_std_h = self.standards.get(label, self.standards['car'])['height']
                dist_m    = (obj_std_h * 700) / (h + 1)
                detections.append({
                    'label':          label,
                    'confidence':     conf,
                    'box':            xyxy,
                    'area':           w * h,
                    'distance_m':     round(dist_m, 1),
                    'distance_index': 1000 / (h + 1),
                    'water_filled':   is_water_filled,
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


        # ── Step 5 & 6: Geometric Lane, Distance, TTC, and Severity Logic ───────
        lane_w = orig_w / 3.0
        frame_area = orig_w * orig_h

        # Attempt to get speed from session state if available, else default to 15m/s (54km/h)
        speed_m_s = 15.0
        try:
            import streamlit as st
            if 'simulated_speed' in st.session_state and st.session_state.simulated_speed > 0:
                speed_m_s = st.session_state.simulated_speed / 3.6
        except Exception:
            pass

        # Update object tracks for Smart Mute / Traffic Mode logic
        current_tracks = {}
        for d in detections:
            box = d['box']
            label = d.get('label', '')
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            ymax = box[3]
            
            # IoU matching with previous tracks
            matched_id = None
            max_iou = 0
            for tid, tdata in self.object_tracks.items():
                tbox = tdata['box']
                if tdata['label'] != label: continue
                
                xA = max(box[0], tbox[0]); yA = max(box[1], tbox[1])
                xB = min(box[2], tbox[2]); yB = min(box[3], tbox[3])
                interArea = max(0, xB-xA) * max(0, yB-yA)
                tbox_area = (tbox[2]-tbox[0]) * (tbox[3]-tbox[1])
                iou = interArea / float(box_area + tbox_area - interArea) if (box_area + tbox_area - interArea) > 0 else 0
                
                if iou > 0.45 and iou > max_iou:
                    matched_id = tid
                    max_iou = iou
            
            if matched_id is not None:
                prev_area = self.object_tracks[matched_id]['area']
                prev_ymax = self.object_tracks[matched_id]['ymax']
                expansion_rate = box_area / prev_area if prev_area > 0 else 1.0
                ymax_shift = ymax - prev_ymax
                current_tracks[matched_id] = {
                    'box': box, 'label': label, 'area': box_area, 'ymax': ymax,
                    'expansion_rate': expansion_rate, 'ymax_shift': ymax_shift
                }
                d['expansion_rate'] = expansion_rate
                d['ymax_shift'] = ymax_shift
            else:
                tid = self.next_track_id
                self.next_track_id += 1
                current_tracks[tid] = {
                    'box': box, 'label': label, 'area': box_area, 'ymax': ymax,
                    'expansion_rate': 1.0, 'ymax_shift': 0.0
                }
                d['expansion_rate'] = 1.0
                d['ymax_shift'] = 0.0
                
        self.object_tracks = current_tracks

        for d in detections:
            box      = d['box']
            center_x = (box[0] + box[2]) / 2.0
            
            # Smart Distance & TTC based on Area
            w = box[2] - box[0]
            h = box[3] - box[1]
            box_area = w * h
            
            # Approximate distance (m) based on area proportion to frame
            area_ratio = box_area / frame_area
            # Clamped inverse relationship: larger box = closer 
            dist_m = max(1.0, 1000.0 / (math.sqrt(box_area) + 1.0)) if box_area > 0 else 50.0
            
            ttc = dist_m / speed_m_s
            
            d['distance_m'] = round(dist_m, 1)
            d['ttc'] = round(ttc, 2)

            # Strict 3-Zone Lane Splitting
            if center_x < lane_w:
                d['lane'], d['lane_id'] = 'Left Side', 1
            elif center_x > (2 * lane_w):
                d['lane'], d['lane_id'] = 'Right Side', 3
            else:
                d['lane'], d['lane_id'] = 'My Lane', 2

            # Dynamic Severity Prioritization (1-10)
            label = d.get('label', '')
            severity = 1
            
            if d['lane_id'] == 2: # My Lane
                # Base severity maps closer distance to higher severity securely inside My Lane
                severity = max(6, int(11 - dist_m))
                
                # Boost if TTC < 2.5s
                if ttc < 2.5:
                    severity = max(severity, 9) # or 10
                    
                # ── Selective Hazard Suppression (Smart Mute / Traffic Mode) ──
                expansion_rate = d.get('expansion_rate', 1.0)
                ymax_shift = d.get('ymax_shift', 0.0)
                
                if label in ['car', 'truck', 'bus', 'auto-rickshaw']:
                    # Traffic logic: Constant speed following means area doesn't change much
                    if 0.95 <= expansion_rate <= 1.05 and dist_m > 3.0:
                        severity = max(2, severity - 5)  # Suppress alarm
                        
                    # Fallback Y-Coordinate Shift Override
                    # Fast descending ymax towards ego hood
                    if ymax_shift > 5.0 and dist_m < 10.0:
                        severity = max(severity, 8) 
                        
                # Absolute Overrides for Vulnerables / Static hazards
                # NEVER suppress these!
                if label in ['pothole', 'water-pothole', 'cow', 'dog', 'person']:
                    severity = 10
            else: # Left or Right Side
                # Max out at 5 for Left/Right sides as per instructions
                severity = min(5, int(8 - dist_m))
                
            # Absolute Pothole Override anywhere if water filled
            if label == 'pothole' and d.get('water_filled', False):
                severity = 10

            # Soft Sign Override for minor road anomalies
            if label in ['bump', 'crack', 'drainage']:
                severity = min(severity, 4)

            d['severity'] = min(10, max(1, severity)) # Clamp 1-10


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
