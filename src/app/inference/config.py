"""
Stampede Early Warning System — Inference Server Config
Edit this file instead of using environment variables.
"""

# ── Backend connection ───────────────────────────────────────────────────────
BACKEND_URL      = "http://localhost:8000"
INFERENCE_SECRET = "stampede-secret-2025"

# ── Area: just set the name, the UUID is looked up automatically ─────────────
AREA_NAME        = "Gate a"          # Must match exactly what you typed in Admin Panel
                                     # Leave blank "" to run without pushing to backend

# ── Video source ─────────────────────────────────────────────────────────────
CAMERA_SOURCE    = "0"   # File path, RTSP URL, or 0 for webcam

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH       = "../../model/heatmap_model/67_precision49_recall.keras"

# ── Scene dimensions (real-world size the camera covers) ─────────────────────
SCENE_WIDTH_M    = 30.0
SCENE_HEIGHT_M   = 30.0

# ── Detection settings ────────────────────────────────────────────────────────
DENSITY_THRESHOLD = 4.0    # persons/m² — cells above this are flagged red
PRED_THRESHOLD    = 0.2    # model output values below this are zeroed out
ALPHA             = 0.6    # heatmap overlay opacity

# ── Push rate ─────────────────────────────────────────────────────────────────
PUSH_INTERVAL    = 1.0     # seconds between backend pushes
