"""Utility helpers for alert audio dispatch in the Streamlit app.

This module exposes a singleton `alert_manager` expected by `main.py`.
The implementation is defensive: if optional audio dependencies are missing
or fail at runtime, calls return silently without crashing the app thread.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

try:
    import winsound  # Windows-only standard library module
except Exception:  # pragma: no cover
    winsound = None

try:
    import pyttsx3
except Exception:  # pragma: no cover
    pyttsx3 = None


@dataclass
class AlertManager:
    """Handles non-blocking hazard audio cues (beep + optional TTS)."""

    tts_enabled: bool = True

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._engine = None
        if self.tts_enabled and pyttsx3 is not None:
            try:
                self._engine = pyttsx3.init()
                # Keep voice output concise for rapid hazard announcements.
                self._engine.setProperty("rate", 185)
            except Exception:
                self._engine = None

    def _play_beep(self, is_water: bool) -> None:
        if winsound is None:
            return

        try:
            # Water-filled potholes get a lower-frequency warning tone.
            tone = 880 if is_water else 1200
            winsound.Beep(tone, 180)
        except Exception:
            # Never allow alert audio failures to crash detection loop.
            return

    def _speak(self, message: str) -> None:
        if self._engine is None:
            return

        try:
            with self._lock:
                self._engine.say(message)
                self._engine.runAndWait()
        except Exception:
            return

    def trigger_hazard_alert(self, label: str, is_water: bool = False, lane: str = "CENTER") -> None:
        """Emit an immediate beep and optional voice warning for a hazard."""
        self._play_beep(is_water=is_water)

        if not self.tts_enabled:
            return

        hazard_name = label.replace("_", " ").strip() if label else "hazard"
        water_prefix = "water filled " if is_water else ""
        spoken = f"Warning. {water_prefix}{hazard_name} detected in {lane} lane."
        self._speak(spoken)


alert_manager = AlertManager()
