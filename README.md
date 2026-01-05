# RAKSHAK AI: Advanced Indian Road Hazard Detection System

🛡️ **Protecting Lives with Intelligent Vision**

Rakshak AI is a state-of-the-art driver assistance system specifically engineered for the unique challenges of Indian roads. Traditional ADAS systems often fail in diverse Indian conditions like heavy rain, stray animals, and potholes. Rakshak AI bridges this gap.

## 🌟 Key Features

*   **Real-time AI Detection**: Optimized YOLO algorithms for detecting Cows, Pedestrians, Tuk-tuks, and specialized Indian vehicles.
*   **🌧️ Advanced Water-Filled Pothole Detection**: Multi-method computer vision system specifically designed for Indian monsoon conditions:
    - **HSV Water Reflection Analysis**: Detects water surfaces by analyzing brightness and saturation patterns
    - **Edge Gradient Detection**: Uses Sobel operators to identify pothole boundaries even when filled with water
    - **Texture Anomaly Analysis**: Distinguishes smooth water surfaces from rough asphalt
    - **Morphological Filtering**: Eliminates false positives with intelligent shape-based filtering
    - **Differentiated Alerts**: More urgent warnings for water-filled potholes (75-95% confidence)
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

## 🧪 Testing Water-Filled Pothole Detection

To visualize how the water-filled pothole detection works:

1. Place a test image in the project directory (road image with potholes or water)
2. Run the test script:
   ```powershell
   python test_water_detection.py path/to/your/image.jpg
   ```
3. The script will show:
   - Water reflection mask
   - Edge gradient mask
   - Texture anomaly mask
   - Combined detection result
   - Final bounding boxes (Blue = water-filled, Red = dry)

**Technical Documentation**: See `WATER_POTHOLE_DETECTION.md` for detailed explanation of the 4-method detection approach.

## 📈 Roadmap for Future Development
- Integration with LiDAR for precision depth mapping.
- V2X communication for vehicle-to-infrastructure alerts.
- Dedicated hardware deployment (NVIDIA Jetson / Raspberry Pi).

---
*Developed for Academic Excellence | Final Year Project Prototype*
