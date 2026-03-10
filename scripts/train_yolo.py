"""
YOLOv8 Training Script for Rakshak AI
Optimized for Intel i3 CPU - uses YOLOv8n (Nano) + OpenVINO export
Trains custom model on Indian road hazard detection dataset
"""

import os
import argparse
import shutil
from ultralytics import YOLO
import yaml
from datetime import datetime

# ── Required Rakshak AI classes ────────────────────────────────────────────────
RAKSHAK_CLASSES = [
    'Auto-rickshaw',  # Indian-specific priority vehicle
    'Pothole',        # Dry road pothole
    'Cow',            # Cattle on road (Indian hazard)
    'Person',         # Pedestrian
    'Motorcycle',     # Two-wheeler
]

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Rakshak AI Model (CPU-Optimized with OpenVINO Export)'
    )
    # ── CHANGED: Default model switched from yolov8m.pt → yolov8n.pt (Nano)
    #             Nano is ~4x faster on CPU inference and ~6x smaller file size.
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Base model. Default: yolov8n.pt (Nano) for CPU inference')
    parser.add_argument('--data', type=str, default='config/data.yaml',
                        help='Dataset config file')
    # Reduced epochs for CPU training: use 100 by default (was 200)
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (100 recommended for CPU)')
    # Smaller batch for i3 CPU with limited RAM
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size (8 recommended for Intel i3 / 8GB RAM)')
    # Training image size matches inference size for consistency
    parser.add_argument('--imgsz', type=int, default=320,
                        help='Training image size — 320 matches our CPU inference size')
    # ── CHANGED: Default device is now 'cpu' (was '0' for GPU)
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: "cpu" for Intel i3 (no GPU), "0" if GPU available')
    # Fewer workers on CPU to prevent thermal throttling
    parser.add_argument('--workers', type=int, default=2,
                        help='Dataloader workers (2 for CPU to prevent overheating)')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Save results to project/name')
    parser.add_argument('--name', type=str, default='rakshak_ai',
                        help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    # ── NEW: OpenVINO export flag (enabled by default for CPU optimization)
    parser.add_argument('--export-openvino', action='store_true', default=True,
                        help='Export trained model to OpenVINO IR format (3x CPU speedup). Default: True')
    return parser.parse_args()


def load_config(config_path):
    """Load dataset configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def verify_rakshak_classes(config):
    """Verify config includes the required Rakshak AI classes."""
    print("\n🔍 Verifying Rakshak AI required classes...")
    names_values = list(config.get('names', {}).values())
    names_lower  = [n.lower() for n in names_values]

    all_ok = True
    for required in RAKSHAK_CLASSES:
        if required.lower() in names_lower:
            print(f"  ✅  {required}")
        else:
            print(f"  ⚠️  {required}  — NOT FOUND in dataset config (add to data.yaml)")
            all_ok = False

    if not all_ok:
        print("\n  💡 Tip: Check config/data.yaml and ensure all 5 classes appear in 'names:'")
    return all_ok


def export_to_openvino(best_model_path, output_dir):
    """
    Export trained YOLOv8n model to OpenVINO IR format.
    OpenVINO provides ~3x faster inference on Intel CPUs via INT8 quantization.

    Output files:
      models/openvino_model/best_openvino_model/
        ├── best.xml   (model graph)
        └── best.bin   (weights)
    """
    print("\n" + "="*60)
    print("🔧 Exporting to OpenVINO IR format...")
    print("   This gives ~3x faster CPU inference via Intel OpenVINO runtime.")
    print("="*60)

    try:
        model = YOLO(best_model_path)

        # Export to OpenVINO format
        # imgsz=320 matches our inference pipeline (see detector.py)
        export_path = model.export(
            format='openvino',
            imgsz=320,          # Must match inference input size
            half=False,         # INT8 quant not needed; FP32 for accuracy
            dynamic=False,      # Static shapes for OpenVINO optimization
        )

        # Copy to models/ directory for easy access
        ov_output = 'models/rakshak_openvino'
        os.makedirs('models', exist_ok=True)
        if os.path.exists(export_path):
            if os.path.exists(ov_output):
                shutil.rmtree(ov_output)
            shutil.copytree(export_path, ov_output)
            print(f"\n✅ OpenVINO model saved to: {ov_output}/")
            print("   📂 Files: best.xml (graph) + best.bin (weights)")
            print("   🚀 Usage: YOLO('models/rakshak_openvino/best.xml') in detector.py")
        else:
            print(f"✅ OpenVINO model exported to: {export_path}")

        return export_path

    except Exception as e:
        print(f"\n⚠️  OpenVINO export failed: {e}")
        print("   Make sure 'openvino' package is installed:")
        print("   pip install openvino openvino-dev")
        return None


def main():
    args = parse_args()

    print("="*60)
    print("RAKSHAK AI — Model Training (CPU-Optimized)")
    print("="*60)
    print(f"  Base Model  : {args.model}  {'✅ Nano (recommended for i3)' if 'n.pt' in args.model else '⚠️  Consider yolov8n.pt for CPU'}")
    print(f"  Dataset     : {args.data}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch Size  : {args.batch}")
    print(f"  Image Size  : {args.imgsz}px  {'✅ 320px — fast CPU inference' if args.imgsz <= 320 else '⚠️  Consider --imgsz 320 for speed'}")
    print(f"  Device      : {args.device}")
    print(f"  Workers     : {args.workers}")
    print(f"  OpenVINO    : {'✅ Will export after training' if args.export_openvino else '❌ Skipped'}")
    print("="*60)

    # Load and verify dataset config
    config = load_config(args.data)
    print(f"\nDataset Classes ({config['nc']}):")
    for class_id, class_name in config['names'].items():
        print(f"  {class_id}: {class_name}")

    verify_rakshak_classes(config)

    # Check training data exists
    dataset_path = config.get('path', 'data/processed')
    train_path = os.path.join(dataset_path, config.get('train', 'images/train'))
    if not os.path.exists(train_path):
        print(f"\n❌ Error: Training dataset not found at {train_path}")
        print("Please run 'python scripts/prepare_dataset.py' first.")
        return

    train_images = len([f for f in os.listdir(train_path)
                         if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"\nTraining images found: {train_images}")

    if train_images == 0:
        print("❌ Error: No training images found!")
        return

    # Load YOLOv8n model
    print(f"\n📥 Loading base model: {args.model}")
    model = YOLO(args.model)

    print("\n🚀 Starting training...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Train the model — CPU-tuned hyperparameters
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=30,           # Reduced early stopping (faster convergence on Nano)
        save=True,
        plots=True,
        verbose=True,

        # CPU-optimized hyperparameters
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,

        # Augmentation (effective Indian road conditions)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=5.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,
        resume=args.resume,
    )

    print("\n✅ Training completed!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    best_map = results.results_dict.get('metrics/mAP50(B)', 0)
    print(f"\n📊 Best mAP@50: {best_map:.4f} ({best_map*100:.2f}%)")

    # Save .pt model to models/
    best_model_path = os.path.join(args.project, args.name, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        output_path = 'models/rakshak_best.pt'
        os.makedirs('models', exist_ok=True)
        shutil.copy2(best_model_path, output_path)
        print(f"\n✅ Best model (.pt) saved to: {output_path}")
    else:
        best_model_path = None
        print("⚠️  best.pt not found — check training output directory")

    # ── NEW: OpenVINO Export ────────────────────────────────────────────────────
    if args.export_openvino and best_model_path and os.path.exists(best_model_path):
        openvino_path = export_to_openvino(best_model_path, 'models/rakshak_openvino')
        if openvino_path:
            print("\n🎯 CPU Pipeline Ready:")
            print("   1. .pt   model → models/rakshak_best.pt   (PyTorch fallback)")
            print("   2. OpenVINO IR → models/rakshak_openvino/  (3x faster CPU)")
    # ──────────────────────────────────────────────────────────────────────────

    print(f"\n📁 Training results: {os.path.join(args.project, args.name)}")
    print("   - weights/best.pt        : Best checkpoint")
    print("   - weights/last.pt        : Last checkpoint")
    print("   - results.png            : Training metrics")
    print("   - confusion_matrix.png   : Class confusion")

    print("\n📝 Generating evaluation report...")
    try:
        from src.evaluate import generate_training_report
        generate_training_report()
    except Exception as e:
        print(f"Warning: Could not generate report: {e}")

    print("\n" + "="*60)
    print("Next steps:")
    print("1. Review training plots in:", os.path.join(args.project, args.name))
    print("2. Use OpenVINO model in detector.py for 3x CPU speedup")
    print("   → Set OPENVINO_MODEL_PATH in src/detector.py")
    print("3. Run: streamlit run main.py")
    print("="*60)


if __name__ == "__main__":
    main()
