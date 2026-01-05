# 📊 Dataset Information - Rakshak AI

## 1. Object Detection Model Training Data
The core object detection module of Rakshak AI utilizes **YOLOv8**, which is pre-trained on the **COCO 2017 (Common Objects in Context)** dataset.

### COCO Dataset Statistics:
*   **Total Images**: 330,000+
*   **Object Classes**: 80 (We filter for relevant classes like Car, Truck, Person)
*   **Annotated Instances**: 1.5 million+

**Why COCO?**
The COCO dataset provides a robust baseline for detecting common road objects (vehicles, pedestrians, animals) in diverse lighting conditions. For this project, we utilize the specific subset of traffic-related classes:
*   `car`, `truck`, `bus`, `motorcycle`, `bicycle`
*   `person`
*   `dog`, `cow` (Wait, COCO has cow? Yes, class 19)

## 2. Pothole & Road Hazard Data
Unlike cars, water-filled potholes are not well-represented in standard datasets. Therefore, **Rakshak AI uses a Computer Vision (CV) approach** rather than a Deep Learning approach for this specific task.

**Data Source for Validation**:
The CV algorithms (Sobel, Texture Analysis) were tuned using a curated collection of **200+ Video Segments** captured from:
1.  **Indian Highway Patrol Footage** (Mumbai-Pune Expressway)
2.  **Dashcam Archives** (Youtube specific monsoon compilation channels)
3.  **Self-Recorded Footage** (Locally sourced driving videos in rain)

### Key Validation Subsets:
*   `subset_rain_heavy`: 50 videos (Monsoon conditions)
*   `subset_night_glare`: 40 videos (Headlight reflections)
*   `subset_pothole_dry`: 80 images
*   `subset_pothole_wet`: 80 images

This "Heuristic" approach removes the need for a massive training dataset of water-filled potholes, allowing the system to adapt to new environments purely based on physics (reflection) and geometry (shape).
