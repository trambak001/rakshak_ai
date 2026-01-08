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
        self.headless = not AUDIO_AVAILABLE
        
        if not self.headless:
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
        else:
            self.play_beep(frequency=1200, duration=0.15)
            # Shortened labels for voice e.g. "Car Center Lane"
            lane_info = f" in {lane}" if lane else ""
            self.speak(f"{hazard_type}{lane_info}")

    def track_accident(self, detections, lat=19.0760, lon=72.8777):
        """Simulates accident detection and emergency contact."""
        # This is a simplified logic for the college project
        # In reality, this would use impact sensors or complex motion analysis
        accident_detected = False
        
        # Mock logic: if we see multiple cars heavily overlapping and static
        # For demonstration, we'll just provide a method to trigger it
        return accident_detected

    def contact_emergency(self, lat, lon):
        """Mocks contacting emergency services with location."""
        # Simulated emergency centers in an Indian city (Mumbai as example)
        centers = [
            {"name": "City Hospital", "lat": 19.0765, "lon": 72.8780},
            {"name": "Highway Trauma Center", "lat": 19.1000, "lon": 72.9000},
            {"name": "Emergency Police Station", "lat": 19.0500, "lon": 72.8500}
        ]
        
        # Find nearest
        user_loc = (lat, lon)
        nearest = centers[0]
        min_dist = geodesic(user_loc, (centers[0]['lat'], centers[0]['lon'])).km
        
        for c in centers:
            d = geodesic(user_loc, (c['lat'], c['lon'])).km
            if d < min_dist:
                min_dist = d
                nearest = c
        
        alert_msg = f"EMERGENCY: Accident detected at {lat}, {lon}. Alerting {nearest['name']} ({min_dist:.2f}km away)."
        self.speak("Emergency! Accident detected. Contacting nearest medical center.")
        return alert_msg

alert_manager = AlertManager()


def draw_tesla_visualization(detections, width=400, height=400):
    """
    Creates a Tesla-style 'Gray Vision' 2D visualization.
    detections: List of detection dicts
    """
    # Create canvas (Light Gray background like Tesla)
    canvas = np.full((height, width, 3), 240, dtype=np.uint8)
    
    # Draw Road (White lane area)
    road_color = (255, 255, 255)
    lane_width = int(width * 0.6)
    lane_x = (width - lane_width) // 2
    cv2.rectangle(canvas, (lane_x, 0), (lane_x + lane_width, height), road_color, -1)
    
    # Draw Lane Lines
    for i in range(0, height, 40):
        cv2.line(canvas, (width//2, i), (width//2, i+20), (200, 200, 200), 2)
    cv2.line(canvas, (lane_x, 0), (lane_x, height), (180, 180, 180), 2)
    cv2.line(canvas, (lane_x + lane_width, 0), (lane_x + lane_width, height), (180, 180, 180), 2)
    
    # Draw Ego Car (Your car)
    ego_w, ego_h = 40, 70
    ego_x = (width - ego_w) // 2
    ego_y = height - ego_h - 20
    # Tesla Blue/Grey
    cv2.fillPoly(canvas, [np.array([
        [ego_x, ego_y], [ego_x+ego_w, ego_y], 
        [ego_x+ego_w, ego_y+ego_h], [ego_x, ego_y+ego_h]
    ])], (200, 100, 50)) # Blueish tint
    
    # Draw detected objects
    for d in detections:
        label = d['label']
        box = d['box'] # xyxy
        w_px = box[2] - box[0]
        # h_px = box[3] - box[1]
        
        # Mapping Logic:
        # X: map center of box to lateral position relative to center of frame
        # Y: map 'distance_index' inversely to Y position
        # Note: distance_index was 1000 / (h+1). Larger index = Closer.
        
        dist_idx = d.get('distance_index', 0)
        
        # Simple heuristic mapping for visual demo
        # Far (small h) -> Top of screen (y=0)
        # Close (large h) -> Bottom of screen (y=height)
        
        # Normalize lateral position (-1 left, 0 center, 1 right)
        # Using fixed input frame width assumption (e.g., 640 or 1280)
        # We can guess relative position
        center_x_ratio = ((box[0] + box[2]) / 2) / 1280.0 # Assuming HD input, doesn't matter much relative
        
        # Re-center (0.5 is center)
        # If input is 640, adjust logic. Better to use relative to generic center.
        # Let's assume input frame center is width/2.
        
        # Map: 0 -> left lane, 1 -> right lane
        # We'll map detecting frame X (0-100%) to Canvas X (20%-80%)
        # But we don't have frame width here. Let's assume standard behavior.
        # We can pass frame_width/height or just guess based on boxes.
        # Hack: use box center vs an assumed center.
        
        # Simpler: Map relative X.
        # But we need to know the source frame width.
        # For this function to be pure, we'll just assume the box coordinates are roughly standard.
        # Actually, let's pass frame width in 'd' or map loosely.
        
        # We'll rely on the fact that YOLO inputs are usually resized.
        # Let's just place them based on rough spread.
        # We'll map "center of box" to "center of canvas + offset"
        
        # Y Position: 
        # dist_idx range is roughly 0 to 20+.
        # 10 is very close. 1 is far.
        vis_y = height - int(min(dist_idx * 30, height - 50))
        if vis_y < 50: vis_y = 50 # Horizon clamp
        
        vis_x = int((box[0] + box[2]) / 2 / 1280 * width) # Rough standard map
        if vis_x < lane_x: vis_x = lane_x + 10 # Clamp to road ish for cleanliness
        if vis_x > lane_x + width: vis_x = lane_x + width - 10
        
        # Draw Object Block
        color = (100, 100, 100) # Gray cars
        if label in ['pothole', 'water-filled pothole']:
            color = (0, 165, 255) # Orange/Red for hazards
        elif label == 'TRAIN':
            color = (50, 50, 50) # Dark gray for Train
            
        # Draw 3D-ish block
        obj_w = 40 if label == 'TRAIN' else 35
        obj_h = 120 if label == 'TRAIN' else 60
        if label in ['pothole', 'water-filled pothole']:
            obj_w, obj_h = 20, 20
            
        # Refine Y if it's really far
        if vis_y < 100: obj_w, obj_h = obj_w//2, obj_h//2
        
        top_left = (vis_x - obj_w//2, vis_y - obj_h)
        bottom_right = (vis_x + obj_w//2, vis_y)
        
        cv2.rectangle(canvas, top_left, bottom_right, color, -1)
        # Add label
        cv2.putText(canvas, label[:4].upper(), (vis_x - 15, vis_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50,50,50), 1)

    return canvas
