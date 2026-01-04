# RAKSHAK AI: Advanced Indian Road Hazard Detection System

🛡️ **Protecting Lives with Intelligent Vision**

Rakshak AI is a state-of-the-art driver assistance system specifically engineered for the unique challenges of Indian roads. Traditional ADAS systems often fail in diverse Indian conditions like heavy rain, stray animals, and potholes. Rakshak AI bridges this gap.

## 🌟 Key Features

*   **Real-time AI Detection**: Optimized YOLO algorithms for detecting Cows, Pedestrians, Tuk-tuks, and specialized Indian vehicles.
*   **Pothole & Drainage Detection**: Innovative computer vision fallback to identify hazardous road irregularities, even those obscured by rainwater.
*   **Night & Rain Vision**: Integrated CLAHE image enhancement for superior visibility in adverse weather and low-light conditions.
*   **Proximity Alert System**: Dynamic distance estimation with audio-visual "Siren" alerts and voice feedback.
*   **Emergency Response System**: Automated accident detection logic with simulated GPS-based emergency service contact.
*   **Dual Mode Feed**: Support for both live USB Camera monitoring and pre-recorded Video File analysis.

## 🛠️ Technology Stack

*   **Intelligence**: YOLOv8/v11 (Ultralytics)
*   **Vision**: OpenCV & PIL
*   **Interface**: Streamlit (Premium Glassmorphism Design)
*   **Alerts**: Pygame Mixer & pyttsx3 (Voice synthesis)
*   **Navigation**: Geopy & Geocoding

## 🚀 Presentation Mode: How to Run

1.  Open PowerShell in the project directory.
2.  Run the launcher:
    ```powershell
    .\run_rakshak.ps1
    ```
3.  The dashboard will open in your default browser.
4.  Select "Video File" to show pre-recorded scenarios or "Live Camera" for a real-time demo.

## 📈 Roadmap for Future Development
- Integration with LiDAR for precision depth mapping.
- V2X communication for vehicle-to-infrastructure alerts.
- Dedicated hardware deployment (NVIDIA Jetson / Raspberry Pi).

---
*Developed for Academic Excellence | Final Year Project Prototype*
