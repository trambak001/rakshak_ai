import numpy as np
import threading
import cv2
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import json
import sys

# Audio libraries (Might fail on Colab/Cloud)
try:
    import pygame
    import pyttsx3
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: Audio libraries not found. Running in Headless Mode.")

class AlertManager:
    def __init__(self):
        self.headless = True # TEMPORARILY DISABLED VOICE FOR STABILITY
        
        if not self.headless and AUDIO_AVAILABLE:
            try:
                pygame.mixer.init()
                self.engine = pyttsx3.init()
                self._set_voice()
            except Exception as e:
                print(f"Audio init failed: {e}. Switching to Headless Mode.")
                self.headless = True
                
        self.is_speaking = False

    def _set_voice(self):
        if self.headless: return
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('rate', 150)

    def play_beep(self, frequency=1000, duration=0.1):
        if self.headless:
            print(f"[BEEP] {frequency}Hz for {duration}s")
            return
            
        sample_rate = 44100
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)
        wave = np.sin(frequency * t * 2 * np.pi)
        audio = (wave * 32767).astype(np.int16)
        audio_stereo = np.column_stack((audio, audio))
        sound = pygame.sndarray.make_sound(audio_stereo)
        sound.play()

    def speak(self, text):
        if self.headless:
            print(f"[SPEAK] {text}")
            return
            
        if not self.is_speaking:
            self.is_speaking = True
            def run():
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except:
                    pass
                self.is_speaking = False
            threading.Thread(target=run, daemon=True).start()

    def trigger_hazard_alert(self, hazard_type, is_water_filled=False, lane=""):
        """Combination of beep and voice with lane/level details."""
        if is_water_filled or 'water' in hazard_type.lower():
            # More urgent alert for water-filled potholes
            self.play_beep(frequency=1800, duration=0.3)
            voice_text = f"Danger: Water filled pothole in {lane or 'your path'}. Level 3 danger. Slow down!"
            self.speak(voice_text)
        elif 'pothole' in hazard_type.lower():
            self.play_beep(frequency=1500, duration=0.2)
            voice_text = f"Warning: {hazard_type} in {lane or 'road'}."
            self.speak(voice_text)
        elif 'speed breaker' in hazard_type.lower() or 'bump' in hazard_type.lower():
            self.play_beep(frequency=900, duration=0.1)
            self.speak(f"Speed breaker ahead. Slow down.")
        else:
            self.play_beep(frequency=1200, duration=0.15)
            # Shortened labels for voice e.g. "Car Center Lane"
            lane_info = f" in {lane}" if lane else ""
            self.speak(f"{hazard_type}{lane_info}")



alert_manager = AlertManager()


def draw_tesla_visualization(detections, width=400, height=500):
    """
    Realistic 2D 'Digital Twin' view.
    - Width: 400px (approx 12m road width)
    - Height: 500px (approx 50m forward view)
    - Scale: 10 pixels per meter
    """
    canvas = np.full((height, width, 3), 30, dtype=np.uint8) # Dark mode Tesla UI
    
    # Draw Road (Centered)
    road_width_m = 10 # Total road width in meters
    road_w_px = road_width_m * 10
    road_x = (width - road_w_px) // 2
    
    # Asphalt
    cv2.rectangle(canvas, (road_x, 0), (road_x + road_w_px, height), (50, 50, 50), -1)
    
    # Lane Lines (Dashed)
    lane_count = 3
    lane_w_px = road_w_px // lane_count
    for i in range(1, lane_count):
        lx = road_x + i * lane_w_px
        for y in range(0, height, 40):
            cv2.line(canvas, (lx, y), (lx, y+20), (150, 150, 150), 1)
            
    # Road Edges (Solid)
    cv2.line(canvas, (road_x, 0), (road_x, height), (200, 200, 200), 2)
    cv2.line(canvas, (road_x + road_w_px, 0), (road_x + road_w_px, height), (200, 200, 200), 2)
    
    # Ego Car (Global Standard: ~1.8m x 4.5m)
    # Positioning at bottom-center
    ego_w_px = int(1.8 * 10)
    ego_h_px = int(4.5 * 10)
    ego_x = (width - ego_w_px) // 2
    ego_y = height - ego_h_px - 20
    
    # Ego car shape (modern grey accent)
    cv2.rectangle(canvas, (ego_x, ego_y), (ego_x + ego_w_px, ego_y + ego_h_px), (180, 180, 180), -1)
    cv2.rectangle(canvas, (ego_x+2, ego_y+5), (ego_x+ego_w_px-2, ego_y+15), (100, 100, 100), -1) # Windshield
    
    # Process Detections
    for d in detections:
        label = d['label']
        dist_m = d.get('distance_m', 0)
        lane_id = d.get('lane_id', 2)
        
        # Vertical mapping: Closer (0m) -> Down (ego_y), Far (50m) -> Up (0)
        # Scale: 1m = 10px
        obj_y = ego_y - int(dist_m * 10)
        
        # Horizontal mapping: Base on Lane ID
        # lane_id mappings in detector: 0:L-Shoulder, 1:L-Lane, 2:Center, 3:R-Lane, 4:R-Shoulder
        lane_centers = [
            road_x - 15,                     # Shoulder Left
            road_x + lane_w_px // 2,        # Left Lane
            road_x + 3 * lane_w_px // 2,    # Center Lane (Wait, logic check)
            # Fix: 3 lanes -> centers at road_x + 0.5w, 1.5w, 2.5w
            # Let's use a simpler center-based mapping
        ]
        
        # Correctly map 5 lane markers into our 3-lane visualizer
        # We'll map them laterally based on lane_id
        lane_map_px = [
            road_x - 10,                            # 0
            road_x + lane_w_px // 2,               # 1
            road_x + road_w_px // 2,               # 2
            road_x + road_w_px - lane_w_px // 2,    # 3
            road_x + road_w_px + 10                # 4
        ]
        obj_x_center = lane_map_px[lane_id]
        
        # Object Size based on Global Standards
        # Standards from detector if we want to be precise, or generic
        o_w, o_h = 1.8, 4.0 # Default car
        color = (200, 200, 200) # Neutral
        
        lower_label = label.lower()
        if 'pothole' in lower_label or 'water pit' in lower_label:
            o_w, o_h = 0.6, 0.4
            # Color coding same as main view
            if 'l3' in lower_label or 'critical' in lower_label or 'water' in lower_label: color = (0, 0, 255)
            elif 'l2' in lower_label: color = (0, 165, 255)
            else: color = (0, 255, 255)
        elif 'speed breaker' in lower_label:
            o_w, o_h = 4.0, 0.3
            color = (0, 255, 100)
        elif lower_label in ['person', 'cow', 'dog']:
            o_w, o_h = 0.6, 0.6
            color = (255, 0, 255)
        elif lower_label in ['truck', 'bus', 'train']:
            o_w, o_h = 2.5, 8.0
            color = (150,150,150)
            
        # Draw Object
        ow_px = max(4, int(o_w * 10))
        oh_px = max(4, int(o_h * 10))
        
        # Visibility check
        if obj_y < 0: continue
        
        cv2.rectangle(canvas, (obj_x_center - ow_px//2, obj_y - oh_px), (obj_x_center + ow_px//2, obj_y), color, -1)
        
        # Add Distance Text for nearby objects
        if dist_m < 25:
            cv2.putText(canvas, f"{int(dist_m)}m", (obj_x_center - 10, obj_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

    return canvas
