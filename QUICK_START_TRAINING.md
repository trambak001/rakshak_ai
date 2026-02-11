# 🚀 Quick Start: Train Your Model on Google Colab

## TL;DR - 3 Steps to Trained Model

### 1. Open Notebook in Colab (2 minutes)
- Open **[Rakshak_AI_Training.ipynb](Rakshak_AI_Training.ipynb)**
- Click "Open in Colab" button **OR**
- Go to https://colab.research.google.com/ → Upload `Rakshak_AI_Training.ipynb`

### 2. Enable GPU (1 minute)
- **Runtime → Change runtime type → GPU (T4)**
- Click Save

### 3. Run All Cells (~ 5-7 hours total, mostly automated)
- **Cell 1-2**: Setup (2 min) ✅ 
- **Cell 3**: Upload `kaggle.json` (1 min) - [Get it here](https://www.kaggle.com/account)
- **Cell 4**: Download datasets (30-40 min) ⏳ *Coffee break!*
- **Cell 5-6**: Prepare data (15 min) ✅
- **Cell 7**: **TRAIN** (4-6 hours) 🚀 *Go work on something else!*
- **Cell 8**: View results (1 min) ✅
- **Cell 9**: **Download `rakshak_best.pt`** (1 min) 💾

---

## After Training

### Place the Model
```
your_project/
├── models/
│   └── rakshak_best.pt  ← Put downloaded file here
```

### Test It
```powershell
.\run_rakshak.ps1
```

The app will **automatically detect and use** your custom model!

---

## Expected Results

✅ **mAP@50**: 88-92% (targeting 90.7%)  
✅ **Training Time**: 4-6 hours  
✅ **Model Size**: ~50 MB  
✅ **Inference Speed**: 50-80ms  

### Per-Class Accuracy:
- Cars: 95-98%
- Potholes: 85-90%
- Water-filled potholes: 75-85%
- Auto-rickshaw: 88-92%

---

## What Datasets Are Used?

Training uses **100,000+ images** from:
1. Indian Roads Dataset (4000+ images)
2. Potholes Detection YOLOv8 (1977 images)
3. Road Hazards (2700 images) 
4. Indian Vehicles (35,000+ images)
5. DriveIndia (66,986 images)

All downloaded automatically by the notebook!

---

## Tips

💡 **Keep browser tab open** during training  
💡 **Free tier is sufficient** - don't need Colab Pro  
💡 **Training can't be paused** - but checkpoints save every epoch  
💡 **Download model immediately** after training completes  

---

## Need More Details?

See [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md) for:
- Detailed step-by-step instructions
- Troubleshooting tips
- What to do if Colab disconnects
- How to resume training
- Expected training metrics

---

**Ready to train? Upload the notebook to Colab and press play! 🚀**
