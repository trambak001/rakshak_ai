"""
Rakshak AI — Alert Manager & Utility Functions
Non-Blocking Voice Alerts via Python threading
──────────────────────────────────────────────────────────────────────────────
Key change: AlertManager.speak() always runs pyttsx3 in a daemon thread so
the main detection loop NEVER blocks waiting for TTS to finish.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import threading
import cv2
import json
import sys

# Audio libraries (optional — falls back silently on headless/cloud systems)
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

AUDIO_AVAILABLE = PYGAME_AVAILABLE or PYTTSX3_AVAILABLE
if not AUDIO_AVAILABLE:
    print("ℹ️  Audio libraries not found — running in headless/silent mode.")


class AlertManager:
    """
    Non-blocking audio alert manager for Rakshak AI.

    Voice alerts run in background daemon threads so the OpenCV/Streamlit
    detection loop is never stalled by TTS synthesis.
    Beeps (pygame) are fire-and-forget and also non-blocking.
    """

    def __init__(self):
        # Headless mode: True → print-only, no actual audio.
        # Auto-detected based on library availability.
        self.headless    = not AUDIO_AVAILABLE
        self.is_speaking = False
        self._tts_lock   = threading.Lock()   # Prevent overlapping TTS calls
        self._tts_engine = None               # Lazily initialized

        if not self.headless:
            if PYGAME_AVAILABLE:
                try:
                    pygame.mixer.init()
                except Exception as e:
                    print(f"⚠️  pygame mixer init failed: {e}")

    # ── TTS engine (lazy, per-thread) ─────────────────────────────────────────

    def _get_tts_engine(self):
        """
        Create a pyttsx3 engine instance.
        pyttsx3 is NOT thread-safe when sharing a single instance, so we
        create a fresh engine per speech thread call.
        """
        if not PYTTSX3_AVAILABLE:
            return None
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)   # Words per minute
            # Prefer a slightly lower pitch for clarity
            voices = engine.getProperty('voices')
            if voices:
                engine.setProperty('voice', voices[0].id)
            return engine
        except Exception as e:
            print(f"⚠️  TTS engine init failed: {e}")
            return None

    # ── Beep ─────────────────────────────────────────────────────────────────

    def play_beep(self, frequency=1000, duration=0.1):
        """
        Play a synthesized beep at the given frequency.
        Uses pygame mixer — non-blocking (plays in mixer's own thread).
        Falls back to console print in headless mode.
        """
        if self.headless or not PYGAME_AVAILABLE:
            print(f"[BEEP] {frequency}Hz × {duration}s")
            return

        try:
            sample_rate = 44100
            n_samples   = int(sample_rate * duration)
            t           = np.linspace(0, duration, n_samples, False)
            wave        = np.sin(frequency * t * 2 * np.pi)
            audio       = (wave * 32767).astype(np.int16)
            audio_stereo = np.column_stack((audio, audio))
            sound = pygame.sndarray.make_sound(audio_stereo)
            sound.play()   # Non-blocking: pygame handles playback in background
        except Exception as e:
            print(f"[BEEP] {frequency}Hz (fallback — {e})")

    # ── Non-blocking speak ────────────────────────────────────────────────────

    def speak(self, text):
        """
        Speak `text` in a background daemon thread.

        ✅ The calling thread (detection loop / Streamlit) returns immediately.
        ✅ If already speaking, the new alert is silently dropped to avoid a
           queue of stale alerts building up.
        ✅ daemon=True ensures the thread dies when the main program exits.
        """
        if self.headless or not PYTTSX3_AVAILABLE:
            print(f"[SPEAK] {text}")
            return

        # Non-blocking acquisition — drop alert if already speaking
        acquired = self._tts_lock.acquire(blocking=False)
        if not acquired:
            # TTS is busy; don't queue — drop this alert to prevent lag
            return

        self.is_speaking = True

        def _tts_worker():
            """Worker function — runs inside daemon thread."""
            try:
                engine = self._get_tts_engine()
                if engine:
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
            except Exception as e:
                print(f"[TTS error] {e}")
            finally:
                self.is_speaking = False
                self._tts_lock.release()   # Always release lock

        # daemon=True: thread dies with the process, no zombie threads
        t = threading.Thread(target=_tts_worker, daemon=True, name="RakshakTTS")
        t.start()

    # ── Hazard alert dispatcher ───────────────────────────────────────────────

    def trigger_hazard_alert(self, hazard_type, is_water_filled=False, lane=""):
        """
        Dispatch a combined beep + voice alert for a detected hazard.
        Both beep and voice are non-blocking (fire-and-forget).

        hazard_type    : Label string (e.g. 'person', 'Pothole L2', 'cow')
        is_water_filled: True if this is a water-filled pothole
        lane           : Lane string (e.g. 'My Lane', 'Left Side')
        """
        lane_info = lane if lane else "your path"

        if is_water_filled or 'water' in hazard_type.lower():
            self.play_beep(frequency=1800, duration=0.3)
            self.speak(
                f"Danger! Water filled pothole in {lane_info}. "
                f"Level 3. Reduce speed immediately."
            )

        elif 'pothole' in hazard_type.lower():
            self.play_beep(frequency=1500, duration=0.2)
            self.speak(f"Warning: {hazard_type} in {lane_info}.")

        elif 'speed breaker' in hazard_type.lower() or 'bump' in hazard_type.lower():
            self.play_beep(frequency=900, duration=0.1)
            self.speak("Speed breaker ahead. Slow down.")

        elif hazard_type.lower() in ['person', 'cow', 'auto-rickshaw']:
            self.play_beep(frequency=1400, duration=0.25)
            self.speak(f"Caution! {hazard_type} in {lane_info}.")

        else:
            self.play_beep(frequency=1200, duration=0.15)
            self.speak(f"{hazard_type} in {lane_info}.")


# ── Module-level singleton (imported by main.py) ──────────────────────────────
alert_manager = AlertManager()


# ── Tesla-style 2D visualization canvas ──────────────────────────────────────

def draw_tesla_visualization(detections, width=400, height=500):
    """
    Realistic 2D 'Digital Twin' top-down view.
    Scale: 10 pixels per metre · Road width: 10 m · View: 50 m ahead.
    """
    canvas = np.full((height, width, 3), 30, dtype=np.uint8)

    road_width_m = 10
    road_w_px    = road_width_m * 10
    road_x       = (width - road_w_px) // 2

    # Asphalt
    cv2.rectangle(canvas, (road_x, 0), (road_x + road_w_px, height), (50, 50, 50), -1)

    # Dashed lane lines
    lane_count = 3
    lane_w_px  = road_w_px // lane_count
    for i in range(1, lane_count):
        lx = road_x + i * lane_w_px
        for y in range(0, height, 40):
            cv2.line(canvas, (lx, y), (lx, y + 20), (150, 150, 150), 1)

    # Road edges
    cv2.line(canvas, (road_x, 0),             (road_x, height),             (200, 200, 200), 2)
    cv2.line(canvas, (road_x + road_w_px, 0), (road_x + road_w_px, height), (200, 200, 200), 2)

    # Ego car
    ego_w_px = int(1.8 * 10)
    ego_h_px = int(4.5 * 10)
    ego_x    = (width - ego_w_px) // 2
    ego_y    = height - ego_h_px - 20
    cv2.rectangle(canvas, (ego_x, ego_y), (ego_x + ego_w_px, ego_y + ego_h_px), (180, 180, 180), -1)
    cv2.rectangle(canvas, (ego_x+2, ego_y+5), (ego_x+ego_w_px-2, ego_y+15), (100, 100, 100), -1)

    lane_map_px = [
        road_x - 10,
        road_x + lane_w_px // 2,
        road_x + road_w_px // 2,
        road_x + road_w_px - lane_w_px // 2,
        road_x + road_w_px + 10,
    ]

    for d in detections:
        label    = d['label']
        dist_m   = d.get('distance_m', 0)
        lane_id  = d.get('lane_id', 2)

        obj_y = ego_y - int(dist_m * 10)
        if obj_y < 0:
            continue

        lane_id = max(0, min(lane_id, 4))
        obj_x_center = lane_map_px[lane_id]

        o_w, o_h = 1.8, 4.0
        color    = (200, 200, 200)
        lower    = label.lower()

        if 'pothole' in lower or 'water pit' in lower:
            o_w, o_h = 0.6, 0.4
            color = ((0, 0, 255)   if ('l3' in lower or 'water' in lower)
                     else (0, 165, 255) if 'l2' in lower
                     else (0, 255, 255))
        elif 'speed breaker' in lower:
            o_w, o_h = 4.0, 0.3
            color = (0, 255, 100)
        elif lower in ['person', 'cow', 'dog', 'auto-rickshaw']:
            o_w, o_h = 0.6, 0.6
            color = (255, 0, 255)
        elif lower in ['truck', 'bus', 'train']:
            o_w, o_h = 2.5, 8.0
            color = (150, 150, 150)

        ow_px = max(4, int(o_w * 10))
        oh_px = max(4, int(o_h * 10))
        cv2.rectangle(
            canvas,
            (obj_x_center - ow_px // 2, obj_y - oh_px),
            (obj_x_center + ow_px // 2, obj_y),
            color, -1,
        )

        if dist_m < 25:
            cv2.putText(canvas, f"{int(dist_m)}m",
                        (obj_x_center - 10, obj_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return canvas
