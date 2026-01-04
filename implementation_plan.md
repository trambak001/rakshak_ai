# Implementation Plan - Indian Road Hazard Detection System (Rakshak AI)

This project aims to build a real-time AI-powered hazard detection system specifically for Indian roads, capable of identifying potholes, animals (cows), pedestrians, debris, and open drainage in various conditions like rain and night.

## Phase 1: Environment Setup & Project Structure
- [ ] Initialize project directory.
- [ ] Create a Python virtual environment.
- [ ] Install dependencies: `ultralytics`, `opencv-python`, `streamlit`, `pygame` (for siren), `geopy` (for location).

## Phase 2: Core Detection Engine
- [ ] Integrate YOLOv8/v11 for real-time object detection.
- [ ] Configure detection for standard objects (Cow, Person, Vehicle, Obstacle).
- [ ] Implement/Source a specialized model for Pothole and Open Drainage detection.
- [ ] Implement distance estimation to trigger alerts when hazards are "too close".

## Phase 3: Environmental Robustness (Rain/Night)
- [ ] Apply image enhancement techniques (e.g., CLAHE) for better night visibility.
- [ ] Test detection performance under simulated/actual rain conditions.

## Phase 4: Emergency Service Integration
- [ ] Implement accident detection logic (speed change, impact detection simulation).
- [ ] Mock GPS location fetching.
- [ ] Implement a mock "Emergency Contact" system (simulating sending SMS/Email to local services).

## Phase 5: Streamlit Web UI
- [ ] Create a modern, user-friendly dashboard.
- [ ] Add "Live Camera" and "Upload Video" functionality.
- [ ] Add real-time visual overlays and hazard logs.

## Phase 6: Alert System
- [ ] Integrate audio feedback (Siren/Voice alerts) using `pygame`.
- [ ] Add visual flashing alerts for critical hazards.

## Phase 7: Final Polish
- [ ] Documentation and "How to Run" guide for the college presentation.
