# 🚀 Stampede Prediction & Early Warning System (EWS)

An advanced real-time crowd density estimation, motion analysis, and stampede prediction system. By leveraging deep learning models for crowd density heatmap prediction combined with a multi-threaded processing pipeline and a modern web dashboard, this platform enables event managers and security personnel to proactively detect overcrowding and coordinate responses before incidents occur.

---

## 🏗️ System Architecture

The project consists of three core components working in tandem:

```
                  ┌──────────────────────────────────────────┐
                  │          Real-time Video Feed            │
                  │   (RTSP Stream / Video Files / Camera)   │
                  └────────────────────┬─────────────────────┘
                                       │
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │         Inference Engine (v2)            │
                  │   - Heatmap Density Model (.keras)       │
                  │   - Vectorized Grid Density Analysis     │
                  │   - Multi-threaded Queue Pipeline        │
                  └────────────────────┬─────────────────────┘
                                       │ (Density & FPS metrics)
                                       ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                     Unified Backend Server                       │
    │         - FastAPI Web Server (src/app/server.py)                 │
    │         - SQLite Database (stampede.db)                          │
    │         - WebSocket Alert Broadcasting & Sound Indicators        │
    └───────────────────┬──────────────────────────────▲───────────────┘
                        │ (Live updates)               │ (Commands & Config)
                        ▼                              │
    ┌──────────────────────────────────────────────────┴───────────────┐
    │                      Web Admin Panel                             │
    │        - Live Monitor Dashboard (HTML/CSS/JS)                    │
    │        - Dynamic Worker Management & Hot-Reload Settings         │
    │        - Volunteer Tracking & Area Assignment                    │
    └──────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```filepath
Stampede_Prediction/
├── config.py                 # Central configuration for directories & training defaults
├── requirements.txt          # Minimal Python dependencies for local development
├── stampede.db               # SQLite database mapping area settings & volunteer records
├── src/
│   ├── app/                  # Unified v2 Web Application (Backend & Dashboard)
│   │   ├── index.html        # Premium Admin Panel with Live Monitor, Feeds & Volunteers tabs
│   │   ├── server.py         # FastAPI backend managing concurrent in-process background worker threads
│   │   └── requirements.txt  # FastAPI & Web-specific dependencies
│   │
│   ├── model/                # Crowd density and counting models
│   │   ├── heatmap_model/    # Heatmap-based crowd density estimation CNN model (.keras)
│   │   └── Yolo/             # YOLO-based crowd counting notebook
│   │
│   ├── preprocessing/        # Data pipelines & preparation notebooks
│   │   └── data_preprocessing.ipynb
│   │
│   ├── risk_detection/       # Local prediction & risk assessment
│   │   ├── video_inference_server_1.py  # Single-feed inference client
│   │   ├── video_inference_server_2.py  # Vectorized, multi-threaded inference client with queue buffers
│   │   └── risk_scoring.py   # Placeholder for future step-based risk scoring algorithm
│   │
│   └── utils/                # Environment testing & utilities
│       └── env_check.py      # Basic check script to confirm CUDA/GPU accessibility
```

---

## ⚡ Key Features

### 1. Advanced Density & Overcrowding Detection
- **Focal Loss CNN Model**: Predicts precise continuous density maps (heatmaps) from input frames.
- **Vectorized Spatial Grid Grid**: Divides each frame area into customizable grids (e.g., $8 \times 8$). NumPy-vectorized operations calculate the absolute density ($people/m^2$) for all cells simultaneously rather than iterating sequentially.
- **Dynamic HUD Overlay**: Renders real-time count overlays, scene configurations, and bright red visual boundary borders for overcrowded grid cells.

### 2. Multi-Threaded Inference Pipeline (`video_inference_server_2.py`)
- Employs a triple-thread architecture (Capture Thread → Inference Thread → Display Thread) separated by thread-safe, latency-bounded queues.
- Discards stale frames instead of stalling processing to maintain real-time frame rates during heavy computation load.

### 3. Unified Web Dashboard & In-Process Workers (`src/app/`)
- **No Manual Configuration Files**: Manage all camera feeds, models, and threshold rules directly from a graphical Web UI.
- **Concurrent Camera Feeds**: Spin up multiple camera feeds in independent background daemon threads controlled directly through the backend.
- **Hot-Reloadable Parameters**: Adjust density thresholds, prediction confidence, or push intervals on-the-fly without stopping or restarting processes.
- **Real-Time Dashboards**:
  - **Live Monitor**: Colorful status indicators, live FPS counts, and immediate audio alert buzzers.
  - **Feeds & Cameras**: Live metrics tables showing details for active and idle worker threads.
  - **Volunteers**: Quick assignment interfaces to coordinate volunteers to overcrowded sectors.

---

## 🚀 Quick Setup & Usage

### Prerequisites
- Python 3.10+
- A compatible webcam, video files, or RTSP network camera stream.

### Installation

1. Clone this repository to your local workspace:
   ```bash
   git clone <repository_url>
   cd Stampede_Prediction
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   pip install fastapi uvicorn tensorflow opencv-python numpy
   ```

### Running the Unified Web Dashboard (v2 Upgrade)

1. Navigate to the web application directory and launch the server:
   ```bash
   cd src/app
   python server.py
   ```
   *This starts the FastAPI backend server on `http://localhost:8000`.*

2. Open `index.html` in your web browser.

3. Sign in using the default administrator credentials:
   - **Username**: `admin`
   - **Password**: `admin123`

4. Add a Camera Feed:
   - Go to the **Feeds & Cameras** tab.
   - Enter your Area Name, Camera Source (can be an RTSP URL, a video file path, or `0` for your local webcam).
   - The Model Path field is pre-populated with the default model: `../model/heatmap_model/217k_relu/217k_relu.keras` (change this only if you want to use a different model).
   - Enter scene dimensions and other configuration parameters.
   - Click **Create Area & Save**.

5. Start Real-time Detection:
   - Click the **▶ Start** button next to your feed in the table. The worker state indicator will turn green (`RUNNING`).
   - Switch to the **Live Monitor** tab to view the live dashboard, real-time crowd metrics, FPS counters, and overcrowding warnings.

---

## 🔬 Model Training

Model training notebooks, metrics, and dataset preparation steps can be found in `src/preprocessing/` and `src/model/`.
- Training can be performed locally or via high-performance GPU instances like Google Colab using the provided Jupyter Notebooks.
- The default model weights (`217k_relu.keras`) are loaded automatically when starting the camera workers.
