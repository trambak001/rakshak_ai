# 📚 Algorithm References & Bibliography

The detailed implementation of **Rakshak AI** is based on the following academic papers and industry-standard algorithms.

## 1. Object Detection (YOLO)
*   **Algorithm**: You Only Look Once (YOLO) v8
*   **Source**: Ultralytics / Jocher, G., Chaurasia, A., & Qiu, J. (2023). *YOLO by Ultralytics*.
*   **Original Paper**: Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
*   **Usage**: Real-time detection of vehicles, pedestrians, and animals.

## 2. Advanced Computer Vision (Pothole Detection)

### A. Edge Detection
*   **Method**: Sobel Operator & Canny Edge Detection
*   **Reference**: Canny, J. (1986). *A Computational Approach to Edge Detection*. IEEE Transactions on Pattern Analysis and Machine Intelligence.
*   **Usage**: Detecting the structural boundaries of potholes that separate them from flat road surfaces.

### B. Image Denoising (Rain Removal)
*   **Method**: Bilateral Filtering
*   **Reference**: Tomasi, C., & Manduchi, R. (1998). *Bilateral Filtering for Gray and Color Images*. International Conference on Computer Vision (ICCV).
*   **Usage**: Removing rain streaks and high-frequency noise while preserving the sharp edges of road hazards.

### C. Contrast Enhancement (Night Vision)
*   **Method**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
*   **Reference**: Pizer, S. M., et al. (1987). *Adaptive Histogram Equalization and Its Variations*. Computer Vision, Graphics, and Image Processing.
*   **Usage**: improving visibility in low-light and high-glare scenarios (like oncoming headlights).

### D. Texture Analysis
*   **Method**: Local Variance Analysis (Statistical Texture)
*   **Reference**: Haralick, R. M. (1979). *Statistical and Structural Approaches to Texture*. Proceedings of the IEEE.
*   **Usage**: Distinguishing smooth water surfaces (low variance) from rough asphalt (high variance).

## 3. Sensor Fusion & Mapping
*   **Method**: Inverse Perspective Mapping (IPM) - *Simplified*
*   **Concept**: Transforming 2D camera coordinates into a 3D-like "Bird's Eye View" for the Tesla-style visualizations.
*   **Usage**: Placing detected objects on the simulated dashboard interface.

---
*Created by Rakshak AI Engineering Team.*
