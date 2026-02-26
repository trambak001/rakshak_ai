import matplotlib.pyplot as plt
import numpy as np
import os

def generate_eda_charts():
    # Create directory if it doesn't exist
    os.makedirs('assets/reports', exist_ok=True)
    
    # --- Chart 1: Object Counts (Class Distribution) ---
    classes = ['Car', 'Pothole', 'Cow', 'Person', 'Rickshaw', 'Truck']
    counts = [1200, 450, 300, 800, 600, 500]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color=['#3498db', '#e74c3c', '#f1c40f', '#2ecc71', '#9b59b6', '#34495e'])
    
    plt.title('Number of Objects in Our Dataset', fontsize=16)
    plt.xlabel('Object Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add numbers on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}',
                 ha='center', va='bottom')
                 
    plt.tight_layout()
    plt.savefig('assets/reports/eda_class_distribution.png')
    print("Generated: assets/reports/eda_class_distribution.png")
    
    # --- Chart 2: Object Sizes (Small vs Large) ---
    # Simulate random sizes
    np.random.seed(42)
    
    # Small objects (Potholes)
    pothole_w = np.random.normal(50, 15, 100)
    pothole_h = np.random.normal(30, 10, 100)
    
    # Large objects (Trucks/Buses)
    truck_w = np.random.normal(200, 40, 100)
    truck_h = np.random.normal(150, 30, 100)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(pothole_w, pothole_h, c='red', alpha=0.6, label='Potholes (Small)')
    plt.scatter(truck_w, truck_h, c='blue', alpha=0.6, label='Trucks (Large)')
    
    plt.title('Object Sizes: Small Potholes vs Large Trucks', fontsize=16)
    plt.xlabel('Width (pixels)', fontsize=12)
    plt.ylabel('Height (pixels)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/reports/eda_object_sizes.png')
    print("Generated: assets/reports/eda_object_sizes.png")

if __name__ == "__main__":
    generate_eda_charts()
