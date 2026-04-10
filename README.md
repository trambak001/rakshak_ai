# 🛡️ RAKSHAK AI — Indian Road Hazard Detection System

> **Protecting Lives with Intelligent Vision** · CPU-only · Real-time · Built for Indian Roads

Rakshak AI is an Advanced Driver Assistance System (ADAS) engineered specifically for the chaos of Indian roads — potholes, monsoon water pits, stray cattle, auto-rickshaws, and mixed traffic — running entirely on an Intel i3 CPU with no GPU.

---

## 🗂️ Project File Architecture

How every file connects and what it does:

```mermaid
graph TD
    %% ── Entry Points ──────────────────────────────────────────
    PS["🖥️ run_rakshak.ps1<br/><i>PowerShell launcher</i><br/>Activates .venv, runs streamlit"]
    MAIN["🧠 main.py<br/><i>Streamlit app · Core UI</i><br/>Video loop · HUD · Alert log · Analytics tab"]

    %% ── Source Modules ────────────────────────────────────────
    DET["🔍 src/detector.py<br/><i>HazardDetector class</i><br/>YOLO + OpenVINO + CV dual pipeline<br/>TTC · Lane · Severity · Smart Mute"]
    UTILS["🔔 src/utils.py<br/><i>AlertManager class</i><br/>Non-blocking TTS + beep<br/>Thread-local pyttsx3 engine"]

    %% ── AI Models ─────────────────────────────────────────────
    OV["⚡ models/rakshak_openvino/<br/><i>best.xml + best.bin + metadata.yaml</i><br/>Custom OpenVINO IR model<br/>Fastest CPU inference path"]

    %% ── Config & Data ─────────────────────────────────────────
    CFG["⚙️ config/data.yaml<br/><i>YOLO training config</i><br/>10 Indian road classes<br/>Augmentation hyperparams"]
    STCFG["🎨 .streamlit/config.toml<br/><i>Streamlit theme config</i><br/>Dark mode · Layout width"]

    %% ── Documentation ─────────────────────────────────────────
    README["📖 README.md<br/><i>This file</i><br/>Architecture · Setup · Usage"]
    REPORT["📄 rakshak_project_report.md<br/><i>Full B.Tech project report</i><br/>1059 lines · 12 chapters"]
    REFS["📚 references/ALGORITHM_REFERENCES.md<br/><i>Academic citations</i><br/>YOLO · OpenCV · CLAHE · Sobel"]
    DATA_INFO["📊 data_info/DATASET_DETAILS.md<br/><i>Dataset documentation</i><br/>COCO · IDD · Dashcam footage"]

    %% ── Config Files ──────────────────────────────────────────
    GITIGNORE["🚫 .gitignore<br/><i>Excludes venv · *.pt · temp_video.mp4</i>"]
    REQS["📦 requirements.txt<br/><i>pip dependencies</i><br/>ultralytics · opencv · streamlit · openvino"]
    NOTEBOOK["📓 Rakshak_AI_Training.ipynb<br/><i>Colab training pipeline</i><br/>Dataset prep · YOLO fine-tune"]

    %% ── Runtime Connections ───────────────────────────────────
    PS -->|"activates venv<br/>runs streamlit"| MAIN
    MAIN -->|"imports HazardDetector"| DET
    MAIN -->|"imports AlertManager"| UTILS
    DET -->|"loads IR model<br/>Priority 1: fastest"| OV
    DET -->|"fallback: downloads<br/>yolov8m.pt if needed"| YOLO_DL["☁️ yolov8m.pt<br/><i>auto-downloaded from Ultralytics</i><br/>NOT committed to git"]
    MAIN -->|"reads theme"| STCFG
    NOTEBOOK -->|"exports trained weights<br/>converted to IR format"| OV
    CFG -->|"used during training"| NOTEBOOK

    %% ── Documentation Links ──────────────────────────────────
    README -.->|"documents"| MAIN
    README -.->|"documents"| DET
    README -.->|"documents"| UTILS
    REPORT -.->|"academic write-up of"| DET
    REFS -.->|"cites algorithms used in"| DET
    DATA_INFO -.->|"describes training data for"| OV

    %% ── Styles ────────────────────────────────────────────────
    style MAIN fill:#1e3a5f,stroke:#00d4ff,color:#fff
    style DET  fill:#1a3a2a,stroke:#00e676,color:#fff
    style UTILS fill:#1a3a2a,stroke:#00e676,color:#fff
    style OV   fill:#3a1a2a,stroke:#ff6090,color:#fff
    style PS   fill:#2a2a1a,stroke:#ffcc00,color:#fff
    style YOLO_DL fill:#222,stroke:#555,color:#999
    style README fill:#1a1a3a,stroke:#a78bfa,color:#fff
    style REPORT fill:#1a1a3a,stroke:#a78bfa,color:#fff
    style REFS fill:#1a1a3a,stroke:#a78bfa,color:#fff
    style DATA_INFO fill:#1a1a3a,stroke:#a78bfa,color:#fff
    style CFG fill:#2a1a1a,stroke:#fb923c,color:#fff
    style STCFG fill:#2a1a1a,stroke:#fb923c,color:#fff
    style REQS fill:#2a1a1a,stroke:#fb923c,color:#fff
    style GITIGNORE fill:#1a1a1a,stroke:#444,color:#888
    style NOTEBOOK fill:#1a2a3a,stroke:#38bdf8,color:#fff
```

### File Role Summary

| File | Role | Called By |
|------|------|-----------|
| `run_rakshak.ps1` | Launch script — activates venv, starts Streamlit | User (terminal) |
| `main.py` | Core app — UI, video loop, HUD, session state | `run_rakshak.ps1` |
| `src/detector.py` | AI brain — YOLO + CV dual pipeline, TTC, severity | `main.py` |
| `src/utils.py` | Audio alerts — non-blocking TTS + beep threads | `main.py` |
| `models/rakshak_openvino/` | Custom OpenVINO IR model (best.xml/.bin) | `src/detector.py` |
| `config/data.yaml` | YOLO training class config (10 Indian classes) | Training notebook |
| `.streamlit/config.toml` | Streamlit dark theme settings | Streamlit runtime |
| `requirements.txt` | Python dependencies | pip install |
| `Rakshak_AI_Training.ipynb` | Colab training pipeline | Manual / Google Colab |
| `rakshak_project_report.md` | Full B.Tech project report (12 chapters) | Reference |
| `references/ALGORITHM_REFERENCES.md` | Academic citations for algorithms | Reference |
| `data_info/DATASET_DETAILS.md` | Dataset strategy documentation | Reference |

---

## 🚀 Quick Start

```powershell
# 1. Clone and enter
git clone https://github.com/trambak001/rakshak_ai.git
cd rakshak_ai

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
.\run_rakshak.ps1
```

The dashboard opens at **http://localhost:8501** automatically.

---

## 🧠 Detection Pipeline (How It Works)

```
Camera / Video
      │
      ▼
Downscale 320×320 ──► Frame Skip (every 3rd) ──► CLAHE / Bilateral Filter
      │
      ├──► YOLO / OpenVINO ──► Vehicles, Pedestrians, Animals
      │                              │
      └──► CV Pothole Pipeline ──► Potholes, Water Pits
                 │                        │
           (Bottom 40% ROI)               │
           HSV water mask                 │
           Sobel edge mask                │
           Texture variance mask          │
           White paint rejection          │
                 │                        │
                 └──────── Merge ─────────┘
                                │
                    Lane Assignment (L / My Lane / R)
                                │
                    Distance Estimation (Pinhole Camera)
                                │
                    TTC Scoring (dist ÷ speed_m/s)
                                │
                    Severity Score 1–10
                                │
                    Smart Mute (IoU expansion rate)
                                │
                   Alert Dispatch ──► TTS + Beep (daemon thread)
                                │
                   Streamlit HUD Update
```

---

## 🌟 Key Features

| Feature | Detail |
|---------|--------|
| **Dual Pipeline** | YOLO for objects + Classical CV for road surface — best of both |
| **CPU-only** | Runs on Intel i3 with OpenVINO IR model at 8+ FPS |
| **Water Pothole Detection** | HSV + Sobel + Texture Variance 4-method fusion |
| **White Paint Rejection** | Skips zebra crossings / road markings (>25% white pixel ratio) |
| **Smart Mute** | Suppresses alarms for vehicles maintaining constant following distance |
| **Lane-Aware Alerts** | L / My Lane / R classification using perspective-corrected geometry |
| **Non-blocking Audio** | TTS + beep in daemon threads — never stalls video loop |
| **Live HUD** | System Status, FPS, Latency update in real-time via `st.empty()` placeholders |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Overall mAP@50 | 91.3% |
| Hardware | Intel Core i3 (no GPU) |
| Avg FPS | 8–12 FPS |
| Avg Latency | ~120 ms/frame |
| False Positive Reduction (Smart Mute) | 74% in dense traffic |

---

## 📁 What's NOT in git

These are excluded from the repo (see `.gitignore`):

- `yolov8m.pt` / `yolov8s.pt` — standard Ultralytics models, auto-downloaded on first run
- `temp_video.mp4` — runtime temp file created during video upload
- `.venv/` — Python virtual environment
- `__pycache__/` — compiled bytecode

---

<<<<<<< Updated upstream
*Rakshak AI · B.Sc.IT Final Year Project · 2025–2026*
=======
*Rakshak AI · B.Sc.IT Data Science Final Year Project · 2025–2026*
>>>>>>> Stashed changes
