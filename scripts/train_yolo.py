"""
YOLOv8 Training Script for Rakshak AI
Trains custom model on Indian road hazard detection dataset
"""

import os
import argparse
from ultralytics import YOLO
import yaml
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Train Rakshak AI Model')
    parser.add_argument('--model', type=str, default='yolov8m.pt', 
                       help='Base model (yolov8n/s/m/l/x.pt)')
    parser.add_argument('--data', type=str, default='config/data.yaml',
                       help='Dataset config file')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use (0 for GPU, cpu for CPU)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of dataloader workers')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Save results to project/name')
    parser.add_argument('--name', type=str, default='rakshak_ai',
                       help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    return parser.parse_args()

def load_config(config_path):
    """Load dataset configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    args = parse_args()
    
    print("="*60)
    print("RAKSHAK AI - Model Training")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch}")
    print(f"Image Size: {args.imgsz}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Load dataset config to verify
    config = load_config(args.data)
    print(f"\nDataset Classes ({config['nc']}):")
    for class_id, class_name in config['names'].items():
        print(f"  {class_id}: {class_name}")
    
    # Check if processed dataset exists
    dataset_path = config.get('path', 'data/processed')
    train_path = os.path.join(dataset_path, config.get('train', 'images/train'))
    if not os.path.exists(train_path):
        print(f"\n❌ Error: Training dataset not found at {train_path}")
        print("Please run 'python scripts/prepare_dataset.py' first to prepare the dataset")
        return
    
    # Count training images
    train_images = len([f for f in os.listdir(train_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"\nTraining images found: {train_images}")
    
    if train_images == 0:
        print("❌ Error: No training images found!")
        return
    
    # Load model
    print(f"\n📥 Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Training configuration
    print("\n🚀 Starting training...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Train the model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=50,           # Early stopping patience
        save=True,             # Save checkpoints
        plots=True,            # Create training plots
        verbose=True,          # Verbose output
        
        # Optimized hyperparameters for Indian road conditions
        lr0=0.01,              # Initial learning rate
        lrf=0.01,              # Final learning rate (fraction of lr0)
        momentum=0.937,        # SGD momentum
        weight_decay=0.0005,   # Weight decay
        warmup_epochs=3.0,     # Warmup epochs
        warmup_momentum=0.8,   # Warmup initial momentum
        
        # Augmentation (from config/data.yaml)
        hsv_h=0.015,           # HSV-Hue augmentation
        hsv_s=0.7,             # HSV-Saturation
        hsv_v=0.4,             # HSV-Value
        degrees=10,            # Rotation
        translate=0.1,         # Translation
        scale=0.5,             # Scaling
        shear=5.0,             # Shear
        flipud=0.0,            # Vertical flip
        fliplr=0.5,            # Horizontal flip
        mosaic=1.0,            # Mosaic augmentation
        mixup=0.1,             # MixUp augmentation
        
        # Additional settings
        close_mosaic=10,       # Disable mosaic in last N epochs
        resume=args.resume,    # Resume training
    )
    
    print("\n✅ Training completed!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get best results
    best_map = results.results_dict.get('metrics/mAP50(B)', 0)
    print(f"\n📊 Best mAP@50: {best_map:.4f} ({best_map*100:.2f}%)")
    
    # Save best model to models directory
    best_model_path = os.path.join(args.project, args.name, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        output_path = 'models/rakshak_best.pt'
        os.makedirs('models', exist_ok=True)
        import shutil
        shutil.copy2(best_model_path, output_path)
        print(f"\n✅ Best model saved to: {output_path}")
    
    # Display training results location
    print(f"\n📁 Training results saved to: {os.path.join(args.project, args.name)}")
    print("   - weights/best.pt: Best checkpoint")
    print("   - weights/last.pt: Last checkpoint")
    print("   - results.png: Training metrics plot")
    print("   - confusion_matrix.png: Classification confusion matrix")
    
    # Generate report
    print("\n📝 Generating evaluation report...")
    try:
        from src.evaluate import generate_training_report
        generate_training_report()
    except Exception as e:
        print(f"Warning: Could not generate report: {e}")
    
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Review training plots in the results directory")
    print("2. Test the model: python main.py (will use models/rakshak_best.pt)")
    print("3. Fine-tune if needed by adjusting hyperparameters")
    print("="*60)

if __name__ == "__main__":
    main()
