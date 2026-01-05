"""
Test script to visualize water-filled pothole detection
Shows the different detection masks and final result
"""

import cv2
import numpy as np
from src.detector import HazardDetector
import matplotlib.pyplot as plt

def test_water_detection(image_path):
    """Test water-filled pothole detection on a single image"""
    
    # Load detector
    detector = HazardDetector()
    
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Testing on image: {image_path}")
    print(f"Image size: {frame.shape}")
    
    # Get detections
    detections, processed_frame, weather = detector.detect_hazards(frame, enhance=True)
    
    # Print results
    print(f"\nWeather Status: {weather['status']}")
    print(f"Brightness: {weather['brightness']:.2f}")
    print(f"\nDetections found: {len(detections)}")
    
    for i, det in enumerate(detections):
        print(f"\n{i+1}. {det['label']}")
        print(f"   Confidence: {det['confidence']:.2%}")
        print(f"   Water-filled: {det.get('water_filled', False)}")
        print(f"   Box: {det['box']}")
        print(f"   Distance Index: {det['distance_index']:.2f}")
    
    # Visualize detection process
    height, width = frame.shape[:2]
    roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[int(height*0.6):, :]
    
    # Get individual masks
    water_mask, _ = detector.detect_water_reflections(frame, int(height*0.6))
    edge_mask = detector.detect_edge_gradients(roi_gray)
    texture_mask = detector.detect_texture_anomalies(roi_gray)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Water-Filled Pothole Detection Process', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Water reflection mask
    axes[0, 1].imshow(water_mask, cmap='hot')
    axes[0, 1].set_title('Water Reflection Mask\n(Bright + Low Saturation)')
    axes[0, 1].axis('off')
    
    # Edge gradient mask
    axes[0, 2].imshow(edge_mask, cmap='hot')
    axes[0, 2].set_title('Edge Gradient Mask\n(Sobel Operators)')
    axes[0, 2].axis('off')
    
    # Texture anomaly mask
    axes[1, 0].imshow(texture_mask, cmap='hot')
    axes[1, 0].set_title('Texture Anomaly Mask\n(Low Variance = Smooth)')
    axes[1, 0].axis('off')
    
    # Combined mask
    dark_thresh = cv2.threshold(roi_gray, 50, 255, cv2.THRESH_BINARY_INV)[1]
    combined = cv2.bitwise_or(dark_thresh, cv2.bitwise_and(water_mask, texture_mask))
    axes[1, 1].imshow(combined, cmap='hot')
    axes[1, 1].set_title('Combined Detection Mask')
    axes[1, 1].axis('off')
    
    # Final result with bounding boxes
    result_img = frame.copy()
    for det in detections:
        box = det['box']
        label = det['label']
        conf = det['confidence']
        is_water = det.get('water_filled', False)
        
        # Color: Blue for water-filled, Red for regular
        color = (255, 0, 0) if is_water else (0, 0, 255)
        
        cv2.rectangle(result_img, 
                     (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), 
                     color, 3)
        
        # Label
        text = f"{label} ({conf:.0%})"
        cv2.putText(result_img, text, 
                   (int(box[0]), int(box[1]-10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    axes[1, 2].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Final Detection Result\n(Blue=Water-filled, Red=Dry)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = image_path.replace('.', '_analysis.')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python test_water_detection.py <image_path>")
        print("\nExample: python test_water_detection.py test_images/rainy_road.jpg")
        print("\nTip: You can use any road image with potholes or water puddles")
        sys.exit(1)
    
    test_water_detection(image_path)
