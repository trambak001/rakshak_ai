# Presentation Speech & Demo Guide for Rakshak AI

This guide helps you explain the project during your college viva or presentation.

## 1. Introduction (The Problem)
"Good morning, everyone. Today, I'm presenting **Rakshak AI**, an AI-powered safety system designed for Indian road conditions. Standard cars today have safety systems designed for Western highways, but they fail in India when a cow suddenly crosses the road, or when there's a deep pothole filled with rain water. Our project solves this."

## 2. Technical Strategy (How it works)
"We use a Deep Learning model called **YOLO (You Only Look Once)**. It can see objects in real-time, just like a human eye but faster. 
- We've trained/configured it to recognize standard hazards: Cows, Dogs, Pedestrians, and Cars.
- Since potholes are hard to see at night or in rain, we've added a **Digital Image Enhancement** layer that brightens up low-light scenes.
- We also developed a custom **Computer Vision Fallback** to detect dark depressions on the road, which likely represent potholes or open drains."

## 3. Key Innovation: Alert & Emergency
"The system doesn't just 'see'; it 'acts'. 
- **Proximity Alerts**: If a hazard is too close, it sounds a siren and gives a voice warning.
- **Emergency Rescue**: If the system detects a high-impact collision pattern, it automatically locates the nearest hospital using GPS logic and mocks an emergency alert."

## 4. Live Demo Steps
1.  **Launch the App**: Run `run_rakshak.ps1`.
2.  **Toggle Settings**: Show how 'Night Vision' improves the image (mention CLAHE algorithm).
3.  **Upload Video**: Use a video clip of an Indian road. Show the bounding boxes appearing.
4.  **Show Alert**: Point out when the box turns RED and the voice alarm triggers.

## 5. Potential Questions (Q&A)
- **Q: How will this work in a real car?**
  - *Answer*: This can be deployed on an edge device like an NVIDIA Jetson, connected to a dashboard camera and the car's speakers.
- **Q: How does it detect potholes in rain?**
  - *Answer*: It looks for light reflections and dark patch gradients on the road surface which are typical of submerged holes.
