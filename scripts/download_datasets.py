"""
Dataset Download Script for Rakshak AI
Downloads all required datasets from Kaggle using Kaggle API
"""

import os
import subprocess
import sys

# Kaggle dataset identifiers
DATASETS = {
    'indian_roads': 'datacluster/indian-roads-dataset',
    'potholes_yolov8': 'vipulgote4/potholes-detection-yolov8',
    'road_hazards': 'andrewmvd/road-hazards',
    'indian_vehicles': 'datacluster/indian-vehicle-image-dataset'
}

def check_kaggle_api():
    """Check if Kaggle API is configured"""
    try:
        result = subprocess.run(['kaggle', '--version'], 
                               capture_output=True, text=True, check=True)
        print(f"✓ Kaggle API found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Kaggle API not found or not configured")
        print("\nTo set up Kaggle API:")
        print("1. pip install kaggle")
        print("2. Go to https://www.kaggle.com/account")
        print("3. Click 'Create New API Token'")
        print("4. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\YourName\\.kaggle\\ (Windows)")
        return False

def download_dataset(name, identifier, output_dir='data/raw'):
    """Download a single dataset from Kaggle"""
    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"Source: {identifier}")
    print(f"{'='*60}")
    
    output_path = os.path.join(output_dir, name)
    os.makedirs(output_path, exist_ok=True)
    
    try:
        cmd = ['kaggle', 'datasets', 'download', '-d', identifier, '-p', output_path, '--unzip']
        subprocess.run(cmd, check=True)
        print(f"✓ Successfully downloaded {name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download {name}: {e}")
        return False

def main():
    print("="*60)
    print("RAKSHAK AI - Dataset Download Utility")
    print("="*60)
    
    # Check if Kaggle API is configured
    if not check_kaggle_api():
        sys.exit(1)
    
    # Create output directory
    os.makedirs('data/raw', exist_ok=True)
    
    # Download all datasets
    success_count = 0
    for name, identifier in DATASETS.items():
        if download_dataset(name, identifier):
            success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"Download Summary: {success_count}/{len(DATASETS)} datasets")
    print("="*60)
    
    if success_count == len(DATASETS):
        print("✓ All datasets downloaded successfully!")
        print("\nNext step: Run 'python scripts/prepare_dataset.py' to merge datasets")
    else:
        print("⚠ Some datasets failed to download. Please check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
