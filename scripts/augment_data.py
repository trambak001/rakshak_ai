"""
Data Augmentation Script for Rakshak AI
Applies monsoon/rain and night-time augmentation to enhance dataset
"""

import cv2
import numpy as np
import os
import random
import yaml
from pathlib import Path
import glob

def add_rain_effect(image, intensity='medium'):
    """Add realistic rain effect to image"""
    h, w = image.shape[:2]
    
    # Rain intensity settings
    if intensity == 'light':
        num_drops = 500
        length_range = (10, 20)
    elif intensity == 'heavy':
        num_drops = 2000
        length_range = (15, 30)
    else:  # medium
        num_drops = 1000
        length_range = (12, 25)
    
    rain_img = image.copy()
    
    # Create rain drops
    for _ in range(num_drops):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        length = random.randint(*length_range)
        thickness = 1
        
        # Draw rain drop
        cv2.line(rain_img, (x, y), (x + 2, y + length), (200, 200, 200), thickness)
    
    # Blend with original
    alpha = 0.7
    result = cv2.addWeighted(image, alpha, rain_img, 1 - alpha, 0)
    
    # Add slight blur for realism
    result = cv2.GaussianBlur(result, (3, 3), 0)
    
    return result

def add_water_puddle_effect(image, bbox_list):
    """Add water puddle/reflection effect to pothole regions"""
    result = image.copy()
    h, w = result.shape[:2]
    
    for bbox in bbox_list:
        # Parse YOLO format: class_id x_center y_center width height
        parts = bbox.strip().split()
        if len(parts) < 5:
            continue
        
        class_id = int(parts[0])
        if class_id != 0:  # Only process potholes (class 0)
            continue
        
        # Convert normalized coords to pixels
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        bbox_w = float(parts[3]) * w
        bbox_h = float(parts[4]) * h
        
        # Calculate corners
        x1 = int(x_center - bbox_w / 2)
        y1 = int(y_center - bbox_h / 2)
        x2 = int(x_center + bbox_w / 2)
        y2 = int(y_center + bbox_h / 2)
        
        # Ensure within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Extract region
        region = result[y1:y2, x1:x2].copy()
        
        # Add water reflection (brighten and blue tint)
        water_overlay = region.copy()
        water_overlay[:, :, 0] = np.clip(water_overlay[:, :, 0] * 1.3, 0, 255)  # Blue channel
        water_overlay[:, :, 1] = np.clip(water_overlay[:, :, 1] * 1.1, 0, 255)  # Green
        water_overlay[:, :, 2] = np.clip(water_overlay[:, :, 2] * 1.2, 0, 255)  # Red
        
        # Blend
        alpha = 0.5
        region = cv2.addWeighted(region, alpha, water_overlay, 1 - alpha, 10)
        
        # Add slight blur for water surface
        region = cv2.GaussianBlur(region, (5, 5), 0)
        
        # Put back
        result[y1:y2, x1:x2] = region
    
    return result

def adjust_for_night(image, brightness_factor=0.4):
    """Simulate night-time conditions"""
    # Reduce brightness
    night_img = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    
    # Add slight blue tint (moonlight effect)
    night_img[:, :, 0] = np.clip(night_img[:, :, 0] * 1.2, 0, 255)  # Blue
    
    # Add noise for low-light grain
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    night_img = cv2.add(night_img, noise)
    
    return night_img

def augment_dataset(manifest_path='data/processed/augmentation_manifest.yaml',
                   output_suffix='_aug'):
    """Apply augmentation to tagged images"""
    print("="*60)
    print("RAKSHAK AI - Data Augmentation")
    print("="*60)
    
    # Load manifest
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        print("Please run prepare_dataset.py first")
        return
    
    with open(manifest_path, 'r') as f:
        manifest = yaml.safe_load(f)
    
    base_path = 'data/processed'
    splits = ['train']  # Only augment training data
    
    augmented_count = 0
    
    for split in splits:
        images_dir = os.path.join(base_path, 'images', split)
        labels_dir = os.path.join(base_path, 'labels', split)
        
        # Process rain candidates
        print(f"\n🌧️ Adding rain effects to {len(manifest.get('rain_candidates', []))} images...")
        for img_name in manifest.get('rain_candidates', [])[:100]:  # Limit to 100
            img_path = os.path.join(images_dir, img_name)
            label_path = os.path.join(labels_dir, img_name.replace('.jpg', '.txt'))
            
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Add rain
            rain_img = add_rain_effect(img, intensity='medium')
            
            # Save augmented image
            aug_name = img_name.replace('.jpg', f'{output_suffix}_rain.jpg')
            aug_path = os.path.join(images_dir, aug_name)
            cv2.imwrite(aug_path, rain_img)
            
            # Copy label
            if os.path.exists(label_path):
                aug_label_path = os.path.join(labels_dir, aug_name.replace('.jpg', '.txt'))
                with open(label_path, 'r') as src, open(aug_label_path, 'w') as dst:
                    dst.write(src.read())
            
            augmented_count += 1
        
        # Process monsoon/pothole candidates (water-filled effect)
        print(f"💧 Adding water-filled pothole effects to {len(manifest.get('monsoon_candidates', []))} images...")
        for img_name in manifest.get('monsoon_candidates', [])[:100]:
            img_path = os.path.join(images_dir, img_name)
            label_path = os.path.join(labels_dir, img_name.replace('.jpg', '.txt'))
            
            if not os.path.exists(img_path) or not os.path.exists(label_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Read labels
            with open(label_path, 'r') as f:
                labels = f.readlines()
            
            # Add water puddle effect
            water_img = add_water_puddle_effect(img, labels)
            
            # Also add light rain
            water_img = add_rain_effect(water_img, intensity='light')
            
            # Save
            aug_name = img_name.replace('.jpg', f'{output_suffix}_water.jpg')
            aug_path = os.path.join(images_dir, aug_name)
            cv2.imwrite(aug_path, water_img)
            
            # Update label (change pothole class to water-pothole)
            aug_label_path = os.path.join(labels_dir, aug_name.replace('.jpg', '.txt'))
            with open(aug_label_path, 'w') as f:
                for label in labels:
                    parts = label.strip().split()
                    if len(parts) >= 5 and parts[0] == '0':  # Pothole class
                        # Change to class 1 (water-pothole)
                        f.write(f"1 {' '.join(parts[1:])}\n")
                    else:
                        f.write(label)
            
            augmented_count += 1
        
        # Process night candidates
        print(f"🌙 Adding night effects to {len(manifest.get('night_candidates', []))} images...")
        for img_name in manifest.get('night_candidates', [])[:50]:  # Limit to 50
            img_path = os.path.join(images_dir, img_name)
            label_path = os.path.join(labels_dir, img_name.replace('.jpg', '.txt'))
            
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Add night effect
            night_img = adjust_for_night(img, brightness_factor=0.5)
            
            # Save
            aug_name = img_name.replace('.jpg', f'{output_suffix}_night.jpg')
            aug_path = os.path.join(images_dir, aug_name)
            cv2.imwrite(aug_path, night_img)
            
            # Copy label
            if os.path.exists(label_path):
                aug_label_path = os.path.join(labels_dir, aug_name.replace('.jpg', '.txt'))
                with open(label_path, 'r') as src, open(aug_label_path, 'w') as dst:
                    dst.write(src.read())
            
            augmented_count += 1
    
    print("\n" + "="*60)
    print(f"Augmentation Complete! Created {augmented_count} new augmented images")
    print("="*60)
    print("Dataset is now ready for training!")
    print("\nNext step: Run 'python scripts/train_yolo.py' to start training")

if __name__ == "__main__":
    augment_dataset()
