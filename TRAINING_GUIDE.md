# Rakshak AI - Model Training Guide

## Quick Start

Follow these steps to train your custom model:

### 1. Setup Kaggle API (One-time setup)

```powershell
# Install Kaggle package
pip install kaggle

# Get your API key from https://www.kaggle.com/account
# Click "Create New API Token" - this downloads kaggle.json

# Place kaggle.json in: C:\Users\YourName\.kaggle\
# Make sure it's not in Downloads folder!
```

### 2. Download Datasets (~5-10 GB)

```powershell
python scripts/download_datasets.py
```

This downloads:
- Indian Roads Dataset (4000+ images)
- Potholes Detection YOLOv8 (1977 images)
- Road Hazards Dataset (2700 images)
- Indian Vehicle Dataset (35,000+ images)

### 3. Prepare Dataset (Merge & Split)

```powershell
python scripts/prepare_dataset.py
```

This will:
- Merge all datasets into unified format
- Create train/val/test splits (70/20/10)
- Tag images for augmentation

### 4. Apply Augmentation (Optional but Recommended)

```powershell
python scripts/augment_data.py
```

This adds:
- Rain effects (monsoon conditions)
- Water-filled pothole simulation
- Night-time effects

### 5. Train Model (4-8 hours)

```powershell
# Basic training (GPU recommended)
python scripts/train_yolo.py --epochs 200 --batch 16

# CPU training (slower, smaller batch)
python scripts/train_yolo.py --epochs 100 --batch 4 --device cpu

# Resume from checkpoint
python scripts/train_yolo.py --resume
```

### 6. Evaluate Results

Training will automatically:
- Save best model to `models/rakshak_best.pt`
- Generate training plots in `runs/train/rakshak_ai/`
- Create confusion matrix
- Calculate mAP@50 accuracy

### 7. Test Your Model

```powershell
# Run the main application with your custom model
.\run_rakshak.ps1
```

The app will automatically use `models/rakshak_best.pt` if it exists.

---

## Training Configuration

### Model Options
- `yolov8n.pt` - Nano (fastest, 85-88% mAP)
- `yolov8s.pt` - Small (faster, 88-90% mAP)
- `yolov8m.pt` - Medium (balanced, 90-92% mAP) **[RECOMMENDED]**
- `yolov8l.pt` - Large (accurate, 91-93% mAP)
- `yolov8x.pt` - Extra Large (best, 92-94% mAP, very slow)

### Expected Training Time
- **GPU (RTX 3060/4060)**: 4-6 hours for 200 epochs
- **GPU (RTX 4090)**: 2-3 hours for 200 epochs
- **CPU**: 24-48 hours for 100 epochs (not recommended)

### Expected Accuracy
- **Target**: 90.7% mAP@50
- **Realistic**: 88-92% mAP@50 (depending on dataset quality)
- **Per-Class**:
  - Cars/Vehicles: 95-98%
  - Potholes: 85-90%
  - Water-filled potholes: 75-85%
  - Auto-rickshaw: 88-92%

---

## Troubleshooting

### "No training images found"
→ Run `python scripts/prepare_dataset.py` first

### "Kaggle API not configured"
→ Download `kaggle.json` from kaggle.com/account and place in `~/.kaggle/`

### "CUDA out of memory"
→ Reduce batch size: `--batch 8` or `--batch 4`
→ Or use smaller model: `--model yolov8s.pt`

### "Training too slow"
→ Use GPU if available
→ Reduce epochs: `--epochs 100`
→ Reduce workers: `--workers 4`

---

## What Gets Updated After Training

1. **Model**: `models/rakshak_best.pt` (custom trained model)
2. **Detector**: `src/detector.py` (auto-detects custom model)
3. **Evaluation**: `src/evaluate.py` (load real training metrics)
4. **Documentation**: Update README and presentation files with actual stats

---

## Need Help?

Check:
1. Training logs in `runs/train/rakshak_ai/`
2. Validation results in `runs/val/`
3. Confusion matrix for per-class accuracy
4. TensorBoard: `tensorboard --logdir runs/train`
