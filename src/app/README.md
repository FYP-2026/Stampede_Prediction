# Stampede Early Warning System
### End-to-End Setup Guide

---

## System Architecture

```
┌─────────────────┐      WebSocket/REST      ┌──────────────────────┐
│  Admin Panel    │ ◄──────────────────────► │                      │
│  (browser)      │                          │   Backend Server     │
└─────────────────┘                          │   FastAPI + SQLite   │
                                             │   (server.py)        │
┌─────────────────┐      WebSocket/REST      │                      │
│ Volunteer Panel │ ◄──────────────────────► │                      │
│  (browser/PWA)  │                          └──────────┬───────────┘
└─────────────────┘                                     │ POST /inference/update
                                                        │
                                             ┌──────────▼───────────┐
                                             │  Inference Server    │
                                             │  TensorFlow + cv2    │
                                             │  (video_inference_   │
                                             │   server.py)         │
                                             └──────────────────────┘
                                                        │
                                                  CCTV / Video file
```

---

## Directory Structure

```
stampede_system/
├── backend/
│   ├── server.py          ← FastAPI backend (REST + WebSocket)
│   └── requirements.txt
├── frontend/
│   ├── admin/
│   │   └── index.html     ← Admin Panel (open in browser)
│   └── volunteer/
│       └── index.html     ← Volunteer Portal (mobile browser / PWA)
└── inference/
    └── video_inference_server.py  ← Updated inference + push to backend
```

---

## Step 1 — Start the Backend

```bash
cd stampede_system/backend

# Install dependencies (Python 3.11+)
pip install -r requirements.txt

# Run the server
python server.py
# → Listening on http://localhost:8000
```

Default admin credentials:
- **Username:** `admin`
- **Password:** `admin123`

> **Production note:** Replace the in-memory token store with Redis,
> use a proper secret manager for INFERENCE_SECRET, and run behind nginx+SSL.

---

## Step 2 — Open the Admin Panel

Open `frontend/admin/index.html` in a browser (or serve it from any static file server).

```bash
# Quick static server (Python)
cd frontend/admin
python -m http.server 8080
# → http://localhost:8080
```

Sign in with `admin / admin123`.

**Admin workflow:**
1. Create areas (name, camera source, scene dimensions, density threshold)
2. Wait for volunteers to register
3. Assign volunteers to areas
4. Monitor live density grid; approve close/reopen alerts

---

## Step 3 — Open the Volunteer Portal

Open `frontend/volunteer/index.html` in a browser or phone browser.

```bash
cd frontend/volunteer
python -m http.server 8081
# → http://localhost:8081
```

**Volunteer workflow:**
1. Register an account
2. Wait for admin to assign an area
3. Receive push instructions via WebSocket
4. Act on close/open instructions; report area safe when crowd clears

---

## Step 4 — Run the Inference Server

No UUIDs or environment variables needed. Just edit `config.py` once.

```bash
cd stampede_system/inference

# Install inference deps
pip install tensorflow opencv-python matplotlib
```

Open `inference/config.py` and set your values:

```python
BACKEND_URL   = "http://localhost:8000"
AREA_NAME     = "Gate A"              # Must match exactly what you typed in the Admin Panel
CAMERA_SOURCE = "Sparse_crowd.mp4"   # File path, RTSP URL, or 0 for webcam
MODEL_PATH    = "../model/heatmap_model/67_precision49_recall.keras"
```

Scene size and density threshold are synced automatically from the Admin Panel —
you don't need to set them here unless you're running without a backend.

Then simply run:

```bash
python video_inference_server.py
```

At startup the server logs into the backend, looks up the area UUID by name,
and pulls its settings (scene dimensions, density threshold) from the database.
The video window shows `-> Pushing: Gate A` when connected successfully.

The inference server:
- Runs your existing CNN model on the video/CCTV stream
- Displays the heatmap overlay window (unchanged from original)
- **Pushes crowd count + density grid to the backend every 1 second**
- The backend propagates live metrics to the Admin Panel via WebSocket

---

## Environment Variables Reference

| Variable            | Default                           | Description                        |
|---------------------|-----------------------------------|------------------------------------|
| `AREA_ID`           | *(empty)*                         | Area UUID from admin panel         |
| `BACKEND_URL`       | `http://localhost:8000`           | Backend server URL                 |
| `INFERENCE_SECRET`  | `stampede-secret-2025`            | Shared secret for API auth         |
| `MODEL_PATH`        | `../model/heatmap_model/...keras` | Path to your .keras model          |
| `CAMERA_SOURCE`     | `Sparse_crowd.mp4`                | Video file, RTSP URL, or `0`       |
| `SCENE_WIDTH_M`     | `30.0`                            | Real-world scene width (metres)    |
| `SCENE_HEIGHT_M`    | `30.0`                            | Real-world scene height (metres)   |
| `DENSITY_THRESHOLD` | `4.0`                             | Persons/m² to flag as risky        |
| `DB_PATH`           | `stampede.db`                     | SQLite database file path          |

---

## API Endpoints Summary

| Method | Path                        | Auth    | Description                         |
|--------|-----------------------------|---------|-------------------------------------|
| POST   | `/auth/register`            | None    | Volunteer self-registration         |
| POST   | `/auth/login`               | None    | Login (volunteer or admin)          |
| GET    | `/volunteers`               | Admin   | List all volunteers                 |
| GET    | `/volunteers/me`            | Bearer  | Get own profile + area              |
| POST   | `/areas`                    | Admin   | Create a monitoring area            |
| GET    | `/areas`                    | Bearer  | List all areas with live stats      |
| POST   | `/areas/{id}/assign`        | Admin   | Assign volunteer to area            |
| POST   | `/inference/update`         | Secret  | Push density data from CNN server   |
| GET    | `/alerts`                   | Bearer  | Recent alerts                       |
| POST   | `/alerts/approve`           | Admin   | Approve close/open action           |
| GET    | `/messages/me`              | Bearer  | Messages for current volunteer      |
| WS     | `/ws/admin/{token}`         | Token   | Live admin WebSocket feed           |
| WS     | `/ws/volunteer/{id}/{token}`| Token   | Live volunteer WebSocket feed       |

---

## What Changed in video_inference_server.py

The original file was extended with:

1. **`push_to_backend()`** — Posts `{area_id, count, density, grid_flags, secret}`
   to `POST /inference/update` via `urllib` (no new dependencies).

2. **Background thread push** — A `threading.Thread` runs the HTTP POST so the
   cv2 display loop is never blocked.

3. **Rate limiting** — Pushes happen at most once per `PUSH_INTERVAL` second
   (default: 1 s) rather than every frame.

4. **Environment variable config** — All parameters (model path, scene size,
   thresholds, camera source, backend URL) are now overridable via env vars,
   making the server deployable without code edits.

5. **Everything else is unchanged** — Heatmap overlay, grid flagging,
   HUD text, and model loading are identical to the original.

---

## Production Checklist

- [ ] Replace SQLite with PostgreSQL for concurrent writes
- [ ] Replace in-memory token store with Redis (TTL-based)
- [ ] Add HTTPS / WSS (nginx reverse proxy + Let's Encrypt)
- [ ] Change `INFERENCE_SECRET` to a strong random value
- [ ] Enable CORS only for your specific frontend origins
- [ ] Add rate limiting to the inference push endpoint
- [ ] Set up process manager (systemd / pm2 / Docker)
- [ ] Add SMS/push notifications as fallback for WebSocket
