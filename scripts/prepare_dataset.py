"""
Dataset Preparation Script for Rakshak AI
Merges multiple datasets, converts to unified YOLO format, and creates train/val/test splits
"""

import os
import shutil
import glob
import random
from pathlib import Path
import yaml
from collections import defaultdict
import cv2

# Class mapping from different datasets to our unified classes
CLASS_MAPPING = {
    # Indian Roads Dataset
    'pothole': 'pothole',
    'potholes': 'pothole',
    'speed-breaker': None,  # We don't include speed breakers
    'unpaved': None,
    
    # Road Hazards Dataset
    'crack': 'crack',
    'manhole': 'drainage',
    'open manhole': 'drainage',
    
    # Indian Vehicles Dataset
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'auto': 'auto-rickshaw',
    'autorickshaw': 'auto-rickshaw',
    'rikshaw': 'auto-rickshaw',
    'person': 'person',
    'pedestrian': 'person',
    'animal': 'cow',
    'cow': 'cow',
}

# Our unified class names (must match config/data.yaml)
TARGET_CLASSES = [
    'pothole', 'water-pothole', 'car', 'truck', 'bus',
    'auto-rickshaw', 'person', 'cow', 'drainage', 'crack'
]

def load_config():
    """Load the dataset configuration"""
    with open('config/data.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def convert_annotation(annotation_line, class_map, target_classes):
    """Convert annotation line to unified format"""
    parts = annotation_line.strip().split()
    if len(parts) < 5:
        return None
    
    # Parse original class
    try:
        class_id = int(parts[0])
        # For now, assume all datasets use numeric class IDs
        # You'll need to map these based on each dataset's classes
        
        # Simple conversion - this is a placeholder
        # You'll need to implement proper class mapping per dataset
        if class_id < len(target_classes):
            new_class_id = class_id
        else:
            return None
        
        # Keep bbox as is (already normalized YOLO format)
        x_center, y_center, width, height = parts[1:5]
        
        return f"{new_class_id} {x_center} {y_center} {width} {height}\n"
    except ValueError:
        return None

def process_dataset(dataset_name, source_path, output_path, split_ratio=(0.7, 0.2, 0.1)):
    """Process a single dataset and split into train/val/test"""
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(source_path, '**', ext), recursive=True))
    
    print(f"Found {len(images)} images")
    
    # Shuffle and split
    random.shuffle(images)
    train_size = int(len(images) * split_ratio[0])
    val_size = int(len(images) * split_ratio[1])
    
    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]
    
    # Process each split
    stats = {'train': 0, 'val': 0, 'test': 0}
    for split_name, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
        for img_path in split_images:
            # Find corresponding label file
            label_path = img_path.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
            label_path = label_path.replace('.JPG', '.txt').replace('.PNG', '.txt').replace('.JPEG', '.txt')
            
            # Common patterns for label directories
            if not os.path.exists(label_path):
                # Try labels/ directory
                label_path = img_path.replace('/images/', '/labels/')
                label_path = os.path.splitext(label_path)[0] + '.txt'
            
            if not os.path.exists(label_path):
                print(f"Warning: No label found for {img_path}")
                continue
            
            # Copy image and label to output
            img_basename = f"{dataset_name}_{os.path.basename(img_path)}"
            label_basename = os.path.splitext(img_basename)[0] + '.txt'
            
            output_img_path = os.path.join(output_path, 'images', split_name, img_basename)
            output_label_path = os.path.join(output_path, 'labels', split_name, label_basename)
            
            shutil.copy2(img_path, output_img_path)
            shutil.copy2(label_path, output_label_path)
            stats[split_name] += 1
    
    print(f"Processed: Train={stats['train']}, Val={stats['val']}, Test={stats['test']}")
    return stats

def apply_augmentation_tags(output_path):
    """Tag images that should receive monsoon/rain augmentation"""
    print("\nAnalyzing images for augmentation tagging...")
    
    # We'll create a simple manifest file
    manifest = {
        'rain_candidates': [],      # Images that could benefit from rain augmentation
        'night_candidates': [],     # Images for night augmentation
        'monsoon_candidates': []    # Prime candidates for water-filled pothole simulation
    }
    
    # Analyze train images for brightness (simple check)
    train_images = glob.glob(os.path.join(output_path, 'images', 'train', '*.jpg'))
    
    for img_path in train_images[:100]:  # Sample first 100
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Calculate average brightness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg_brightness = gray.mean()
        
        if avg_brightness < 80:
            manifest['night_candidates'].append(os.path.basename(img_path))
        elif avg_brightness > 120:
            manifest['rain_candidates'].append(os.path.basename(img_path))
        
        # Check if it has pothole labels
        label_path = img_path.replace('/images/', '/labels/').replace('.jpg', '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = f.readlines()
                for label in labels:
                    parts = label.strip().split()
                    if len(parts) > 0 and parts[0] == '0':  # Class 0 = pothole
                        manifest['monsoon_candidates'].append(os.path.basename(img_path))
                        break
    
    # Save manifest
    manifest_path = os.path.join(output_path, 'augmentation_manifest.yaml')
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest, f)
    
    print(f"Tagged {len(manifest['rain_candidates'])} rain candidates")
    print(f"Tagged {len(manifest['night_candidates'])} night candidates")
    print(f"Tagged {len(manifest['monsoon_candidates'])} monsoon candidates")

def main():
    print("="*60)
    print("RAKSHAK AI - Dataset Preparation")
    print("="*60)
    
    # Define source datasets
    datasets = [
        ('indian_roads', 'data/raw/indian_roads'),
        ('potholes_yolov8', 'data/raw/potholes_yolov8'),
        ('road_hazards', 'data/raw/road_hazards'),
        ('indian_vehicles', 'data/raw/indian_vehicles'),
    ]
    
    output_path = 'data/processed'
    
    # Process each dataset
    total_stats = defaultdict(int)
    for dataset_name, source_path in datasets:
        if not os.path.exists(source_path):
            print(f"Warning: Dataset '{dataset_name}' not found at {source_path}, skipping...")
            continue
        
        stats = process_dataset(dataset_name, source_path, output_path)
        for split, count in stats.items():
            total_stats[split] += count
    
    # Apply augmentation tagging
    apply_augmentation_tags(output_path)
    
    # Summary
    print("\n" + "="*60)
    print("Dataset Preparation Complete!")
    print("="*60)
    print(f"Total Train: {total_stats['train']}")
    print(f"Total Val: {total_stats['val']}")
    print(f"Total Test: {total_stats['test']}")
    print(f"Total: {sum(total_stats.values())}")
    print("\nNext step: Run 'python scripts/train_yolo.py' to start training")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
