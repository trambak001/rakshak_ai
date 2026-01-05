import numpy as np
import pygame
import pyttsx3
import threading
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import json

class AlertManager:
    def __init__(self):
        pygame.mixer.init()
        self.engine = pyttsx3.init()
        self._set_voice()
        self.is_speaking = False

    def _set_voice(self):
        voices = self.engine.getProperty('voices')
        # Try to find a clear voice
        self.engine.setProperty('rate', 150)

    def play_beep(self, frequency=1000, duration=0.1):
        sample_rate = 44100
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)
        wave = np.sin(frequency * t * 2 * np.pi)
        audio = (wave * 32767).astype(np.int16)
        
        # Convert to stereo
        audio_stereo = np.column_stack((audio, audio))
        
        sound = pygame.sndarray.make_sound(audio_stereo)
        sound.play()

    def speak(self, text):
        if not self.is_speaking:
            self.is_speaking = True
            def run():
                self.engine.say(text)
                self.engine.runAndWait()
                self.is_speaking = False
            threading.Thread(target=run, daemon=True).start()

    def trigger_hazard_alert(self, hazard_type, is_water_filled=False):
        """Combination of beep and voice."""
        if is_water_filled or 'water' in hazard_type.lower():
            # More urgent alert for water-filled potholes
            self.play_beep(frequency=1800, duration=0.3)
            self.speak(f"Danger: Water filled pothole ahead. Slow down!")
        else:
            self.play_beep(frequency=1500, duration=0.2)
            self.speak(f"Warning: {hazard_type} ahead")

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
