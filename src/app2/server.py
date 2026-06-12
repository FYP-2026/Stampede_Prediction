"""
Stampede Early Warning System — Backend Server v2
FastAPI + WebSocket + SQLite

New in v2:
  • POST /inference/start   — launch a camera feed worker in-process (no extra terminal)
  • POST /inference/stop    — stop a running worker
  • GET  /inference/workers — list all running workers + status
  • Each worker runs video_inference_worker() in a background asyncio thread
"""

import asyncio
import json
import sqlite3
import uuid
import hashlib
import os
import threading
import time
import queue
from datetime import datetime
from typing import Dict, List, Optional, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# ── Optional heavy deps (only needed if running inference in-process) ─────────
try:
    import cv2
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as _plt
    HAS_INFERENCE = True
except ImportError:
    HAS_INFERENCE = False

# ── Compiled inference helper ─────────────────────────────────────────────────

def _make_inference_fn(model, image_size):
    """
    Returns a @tf.function-compiled inference callable.
    Threshold is NOT baked in so hot-reloaded values take effect immediately;
    it is applied in NumPy after the call.
    """
    h, w = image_size

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1, h, w, 3), dtype=tf.float32)
    ])
    def infer(img_batch):
        pred = model(img_batch, training=False)
        return pred[0, ..., 0]   # (H, W)

    return infer

# ── Database ──────────────────────────────────────────────────────────────────

DB_PATH = os.environ.get("DB_PATH", "stampede.db")

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS volunteers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            password_hash TEXT NOT NULL,
            area_id TEXT,
            status TEXT DEFAULT 'offline',
            registered_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS areas (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            camera_source TEXT,
            model_path TEXT DEFAULT '',
            scene_width_m REAL DEFAULT 20.0,
            scene_height_m REAL DEFAULT 20.0,
            density_threshold REAL DEFAULT 4.0,
            pred_threshold REAL DEFAULT 0.3,
            push_interval REAL DEFAULT 1.0,
            alpha REAL DEFAULT 0.5,
            status TEXT DEFAULT 'open',
            current_count REAL DEFAULT 0,
            current_density REAL DEFAULT 0,
            risk_flagged INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            area_id TEXT NOT NULL,
            area_name TEXT,
            alert_type TEXT NOT NULL,
            density REAL,
            count REAL,
            admin_approved INTEGER DEFAULT 0,
            resolved INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            resolved_at TEXT
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            volunteer_id TEXT,
            area_id TEXT,
            content TEXT NOT NULL,
            sender TEXT DEFAULT 'system',
            read_at TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS admins (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        );
    """)
    admin_pw = hashlib.sha256("admin123".encode()).hexdigest()
    cur.execute("INSERT OR IGNORE INTO admins VALUES (?,?,?)",
                ("admin-001", "admin", admin_pw))
    conn.commit()
    conn.close()

# ── Pydantic models ───────────────────────────────────────────────────────────

class VolunteerRegister(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str
    role: str

class AreaCreate(BaseModel):
    name: str
    camera_source: str
    model_path: str = ""
    scene_width_m: float = 20.0
    scene_height_m: float = 20.0
    density_threshold: float = 4.0
    pred_threshold: float = 0.3
    push_interval: float = 1.0
    alpha: float = 0.5

class AreaUpdate(BaseModel):
    name: Optional[str] = None
    camera_source: Optional[str] = None
    model_path: Optional[str] = None
    scene_width_m: Optional[float] = None
    scene_height_m: Optional[float] = None
    density_threshold: Optional[float] = None
    pred_threshold: Optional[float] = None
    push_interval: Optional[float] = None
    alpha: Optional[float] = None

class AreaAssign(BaseModel):
    volunteer_id: str

class AlertApproval(BaseModel):
    alert_id: str
    action: str

class DensityUpdate(BaseModel):
    area_id: str
    count: float
    density: float
    grid_flags: List[dict]
    secret: str

class WorkerStart(BaseModel):
    area_id: str

INFERENCE_SECRET = os.environ.get("INFERENCE_SECRET", "stampede-secret-2025")

# ── In-process worker registry ────────────────────────────────────────────────

class InferenceWorker:
    """Runs one camera feed in a background daemon thread."""

    IMAGE_SIZE = (400, 400)
    GRID_COLS  = 8
    GRID_ROWS  = 8

    def __init__(self, area: dict, push_fn):
        self.area_id   = area["id"]
        self.area_name = area["name"]
        self.push_fn   = push_fn          # coroutine-safe async fn
        self._stop     = threading.Event()
        self._thread   = None
        self.status    = "idle"           # idle | running | stopped | error
        self.error     = ""
        self.fps       = 0.0
        self.last_count   = 0.0
        self.last_density = 0.0
        # live config — refreshed from DB each push cycle
        self._area = dict(area)
        # Latest annotated JPEG for MJPEG streaming (maxsize=1 → always newest)
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self.status = "stopped"

    def _load_model(self, path):
        if not HAS_INFERENCE:
            raise RuntimeError("TensorFlow / cv2 not installed on this machine")

        def _wfl(gamma=2.0, pos_weight=800.0):
            bce = tf.keras.losses.BinaryFocalCrossentropy(gamma=gamma)
            def loss(y_true, y_pred):
                weights = 1.0 + y_true * (pos_weight - 1.0)
                return tf.reduce_mean(bce(y_true, y_pred) * tf.squeeze(weights, -1))
            return loss

        return tf.keras.models.load_model(path, custom_objects={"loss": _wfl()})

    def _run(self):
        self.status = "loading"
        try:
            model_path = self._area.get("model_path", "")
            if not model_path:
                raise ValueError("No model_path configured for this area")
            model = self._load_model(model_path)
        except Exception as e:
            self.status = "error"
            self.error  = str(e)
            return

        source = self._area["camera_source"]
        try:
            source = int(source)
        except (TypeError, ValueError):
            pass

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.status = "error"
            self.error  = f"Cannot open: {source}"
            return

        self.status = "running"
        IH, IW = self.IMAGE_SIZE
        GR, GC = self.GRID_ROWS, self.GRID_COLS
        cell_h = IH // GR
        cell_w = IW // GC

        # ── Compile TF graph once (warm-up with a dummy batch) ────────────────
        infer = _make_inference_fn(model, self.IMAGE_SIZE)
        infer(tf.zeros((1, IH, IW, 3), dtype=tf.float32))

        # ── Precompute vectorised grid index arrays ───────────────────────────
        cell_y0 = (np.arange(GR) * cell_h).astype(np.int32)
        cell_x0 = (np.arange(GC) * cell_w).astype(np.int32)
        cell_y1 = cell_y0 + cell_h
        cell_x1 = cell_x0 + cell_w

        # ── Colormap LUT (hot palette, 256 entries) ───────────────────────────
        try:
            _cmap_lut = (_plt.get_cmap("hot")(np.linspace(0, 1, 256))[..., :3] * 255
                         ).astype(np.uint8)
        except Exception:
            # Fallback: black → red → yellow gradient
            _cmap_lut = np.zeros((256, 3), dtype=np.uint8)
            _cmap_lut[:128, 0] = np.linspace(0, 255, 128, dtype=np.uint8)
            _cmap_lut[128:, 0] = 255
            _cmap_lut[128:, 1] = np.linspace(0, 255, 128, dtype=np.uint8)

        # ── Pre-allocated per-frame buffers (zero heap churn) ─────────────────
        img_batch  = np.empty((1, IH, IW, 3), dtype=np.float32)
        lut_idx    = np.empty((IH, IW),        dtype=np.uint8)
        _scale_buf = np.empty((IH, IW),        dtype=np.float32)

        # ── Inter-thread queues ───────────────────────────────────────────────
        # maxsize=2: capture can't flood inference; inference can't flood push.
        capture_q  = queue.Queue(maxsize=2)
        last_push  = [0.0]   # list so the closure can mutate it

        # ── Thread A: Capture ─────────────────────────────────────────────────
        def _capture():
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    # Loop video files; treat live-camera EOF as a hard stop.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        self._stop.set()
                        break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (IW, IH))
                try:
                    capture_q.put(frame_rgb, timeout=0.1)
                except queue.Full:
                    pass   # drop frame rather than stall

        # ── Thread B: Inference + annotation + push ───────────────────────────
        def _inference():
            frame_t = time.time()
            while not self._stop.is_set():
                try:
                    frame_rgb = capture_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                now = time.time()
                self.fps = round(1.0 / max(now - frame_t, 1e-6), 1)
                frame_t  = now

                # Read latest hot-reloaded config
                pred_threshold    = float(self._area.get("pred_threshold",    0.3))
                density_threshold = float(self._area.get("density_threshold", 4.0))
                alpha             = float(self._area.get("alpha",             0.5))
                scene_w           = float(self._area.get("scene_width_m",    20.0))
                scene_h           = float(self._area.get("scene_height_m",   20.0))

                # Preprocess into pre-allocated buffer (no new allocation)
                np.multiply(frame_rgb, 1.0 / 255.0, out=img_batch[0])

                # Infer (compiled graph) then apply threshold in NumPy so
                # hot-reloaded values take effect without recompiling the graph
                pred_map = infer(img_batch).numpy()
                pred_map[pred_map < pred_threshold] = 0.0
                total_count = float(pred_map.sum())

                # ── Heatmap overlay (LUT path, no per-pixel Python) ───────────
                vmax = float(pred_map.max()) or 1e-6
                np.multiply(pred_map, 255.0 / vmax, out=_scale_buf)
                np.clip(_scale_buf, 0, 255, out=_scale_buf)
                np.copyto(lut_idx, _scale_buf, casting='unsafe')
                heatmap_rgb = _cmap_lut[lut_idx]
                overlay = cv2.addWeighted(frame_rgb, 1.0 - alpha,
                                          heatmap_rgb, alpha, 0)

                # ── Vectorised grid density (single reshape + sum, no loop) ───
                cell_area_m2 = ((cell_h * scene_h / IH) *
                                (cell_w * scene_w / IW))
                grid = (pred_map
                        .reshape(GR, cell_h, GC, cell_w)
                        .sum(axis=(1, 3)) / cell_area_m2)          # (GR, GC)

                flagged_mask = grid >= density_threshold
                any_flagged  = bool(flagged_mask.any())

                grid_flags = []
                for row, col in zip(*np.where(flagged_mask)):
                    row, col = int(row), int(col)
                    y0_, x0_ = int(cell_y0[row]), int(cell_x0[col])
                    y1_, x1_ = int(cell_y1[row]), int(cell_x1[col])
                    density  = float(grid[row, col])
                    grid_flags.append({"row": row, "col": col,
                                       "density": round(density, 2)})
                    cv2.rectangle(overlay, (x0_, y0_), (x1_, y1_),
                                  (255, 0, 0), 2)
                    cv2.putText(overlay, f"{density:.1f}/m\u00b2",
                                org=(x0_ + 4, y0_ + 20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.45, color=(255, 0, 0),
                                thickness=1, lineType=cv2.LINE_AA)

                max_density       = max((f["density"] for f in grid_flags), default=0.0)
                self.last_count   = round(total_count, 1)
                self.last_density = round(max_density, 2)

                # ── HUD ───────────────────────────────────────────────────────
                status_text  = "!! OVERCROWDED !!" if any_flagged else "OK"
                status_color = (255, 0, 0)         if any_flagged else (0, 255, 0)
                cv2.putText(overlay, f"Count: {total_count:.1f}",
                            (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 80, 0), 2, cv2.LINE_AA)
                cv2.putText(overlay, f"FPS: {self.fps}",
                            (15, 65), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(overlay, status_text,
                            (15, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, status_color, 2, cv2.LINE_AA)

                # ── Encode JPEG for MJPEG stream ──────────────────────────────
                try:
                    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    ok, buf = cv2.imencode(".jpg", bgr,
                                          [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        jpeg_bytes = buf.tobytes()
                        try:
                            self.frame_queue.get_nowait()   # evict stale frame
                        except queue.Empty:
                            pass
                        self.frame_queue.put_nowait(jpeg_bytes)
                except Exception:
                    pass   # never let display errors kill the inference loop

                # ── Push metrics at configured interval ───────────────────────
                push_interval = float(self._area.get("push_interval", 1.0))
                if now - last_push[0] >= push_interval:
                    last_push[0] = now
                    asyncio.run_coroutine_threadsafe(
                        self.push_fn(self.area_id, total_count,
                                     max_density, grid_flags),
                        _loop
                    )

        # ── Launch threads; _run itself blocks until both finish ──────────────
        t_capture   = threading.Thread(target=_capture,   daemon=True)
        t_inference = threading.Thread(target=_inference, daemon=True)
        t_capture.start()
        t_inference.start()
        t_capture.join()
        t_inference.join()
        cap.release()
        if self.status != "stopped":
            self.status = "stopped"

    def refresh_area(self, area: dict):
        """Hot-reload config without restart."""
        self._area = dict(area)


# Global registry
_workers: Dict[str, InferenceWorker] = {}   # area_id -> worker
_loop: asyncio.AbstractEventLoop = None      # set in lifespan

# ── WebSocket manager ─────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.admin_connections: Set[WebSocket] = set()
        self.volunteer_connections: Dict[str, WebSocket] = {}

    async def connect_admin(self, ws: WebSocket):
        await ws.accept()
        self.admin_connections.add(ws)

    async def connect_volunteer(self, ws: WebSocket, volunteer_id: str):
        await ws.accept()
        self.volunteer_connections[volunteer_id] = ws
        db = get_db()
        db.execute("UPDATE volunteers SET status='standby' WHERE id=?", (volunteer_id,))
        db.commit(); db.close()

    def disconnect_admin(self, ws: WebSocket):
        self.admin_connections.discard(ws)

    def disconnect_volunteer(self, vid: str):
        self.volunteer_connections.pop(vid, None)
        db = get_db()
        db.execute("UPDATE volunteers SET status='offline' WHERE id=?", (vid,))
        db.commit(); db.close()

    async def broadcast_admin(self, payload: dict):
        dead = set()
        for ws in self.admin_connections:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.add(ws)
        self.admin_connections -= dead

    async def send_volunteer(self, vid: str, payload: dict):
        ws = self.volunteer_connections.get(vid)
        if ws:
            try:
                await ws.send_json(payload)
            except Exception:
                self.disconnect_volunteer(vid)

    async def broadcast_area_volunteers(self, area_id: str, payload: dict):
        db = get_db()
        rows = db.execute("SELECT id FROM volunteers WHERE area_id=?", (area_id,)).fetchall()
        db.close()
        for row in rows:
            await self.send_volunteer(row["id"], payload)

manager = ConnectionManager()

# ── Auth ──────────────────────────────────────────────────────────────────────

security = HTTPBearer(auto_error=False)
active_tokens: Dict[str, dict] = {}

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def create_token(user_id, role):
    raw = f"{user_id}:{role}:{os.urandom(8).hex()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32] + f"_{user_id}_{role}"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(401, "Missing token")
    info = active_tokens.get(credentials.credentials)
    if not info:
        raise HTTPException(401, "Invalid or expired token")
    return info

# ── Internal push function (called by workers) ────────────────────────────────

async def _push_density(area_id, count, density, grid_flags):
    """Same logic as the /inference/update route but called in-process."""
    db = get_db()
    area = db.execute("SELECT * FROM areas WHERE id=?", (area_id,)).fetchone()
    if not area:
        db.close()
        return

    risk = density >= area["density_threshold"]
    db.execute(
        "UPDATE areas SET current_count=?,current_density=?,risk_flagged=? WHERE id=?",
        (count, density, int(risk), area_id)
    )

    if risk and not area["risk_flagged"]:
        alert_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO alerts (id,area_id,area_name,alert_type,density,count) VALUES (?,?,?,?,?,?)",
            (alert_id, area_id, area["name"], "risk_detected", density, count)
        )
        db.commit()
        await manager.broadcast_admin({
            "type": "risk_alert",
            "alert_id": alert_id,
            "area_id": area_id,
            "area_name": area["name"],
            "density": round(density, 2),
            "count": round(count, 1),
            "grid_flags": grid_flags,
            "timestamp": datetime.now().isoformat()
        })
    else:
        db.commit()

    # Refresh worker config from DB
    if area_id in _workers:
        fresh = db.execute("SELECT * FROM areas WHERE id=?", (area_id,)).fetchone()
        if fresh:
            _workers[area_id].refresh_area(dict(fresh))

    await manager.broadcast_admin({
        "type": "metric_update",
        "area_id": area_id,
        "area_name": area["name"],
        "count": round(count, 1),
        "density": round(density, 2),
        "risk": risk,
        "timestamp": datetime.now().isoformat()
    })
    db.close()

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop
    _loop = asyncio.get_running_loop()
    init_db()
    yield
    for w in _workers.values():
        w.stop()

app = FastAPI(title="Stampede EWS v2", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── Auth routes ───────────────────────────────────────────────────────────────

@app.post("/auth/register")
async def register(data: VolunteerRegister):
    db = get_db()
    vid = str(uuid.uuid4())
    try:
        db.execute(
            "INSERT INTO volunteers (id,name,email,phone,password_hash) VALUES (?,?,?,?,?)",
            (vid, data.name, data.email, data.phone, hash_pw(data.password))
        )
        db.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Email already registered")
    finally:
        db.close()
    return {"message": "Registered"}

@app.post("/auth/login")
async def login(data: LoginRequest):
    db = get_db()
    if data.role == "admin":
        row = db.execute(
            "SELECT * FROM admins WHERE username=? AND password_hash=?",
            (data.username, hash_pw(data.password))
        ).fetchone()
        db.close()
        if not row:
            raise HTTPException(401, "Bad credentials")
        token = create_token(row["id"], "admin")
        active_tokens[token] = {"id": row["id"], "role": "admin", "name": "Admin"}
        return {"token": token, "role": "admin", "name": "Admin"}
    else:
        row = db.execute(
            "SELECT * FROM volunteers WHERE email=? AND password_hash=?",
            (data.username, hash_pw(data.password))
        ).fetchone()
        db.close()
        if not row:
            raise HTTPException(401, "Bad credentials")
        token = create_token(row["id"], "volunteer")
        active_tokens[token] = {"id": row["id"], "role": "volunteer", "name": row["name"]}
        return {"token": token, "role": "volunteer", "id": row["id"], "name": row["name"]}

# ── Volunteer routes ──────────────────────────────────────────────────────────

@app.get("/volunteers")
async def list_volunteers(auth=Depends(verify_token)):
    if auth["role"] != "admin":
        raise HTTPException(403, "Admin only")
    db = get_db()
    rows = db.execute("""
        SELECT v.*, a.name as area_name
        FROM volunteers v LEFT JOIN areas a ON v.area_id = a.id
    """).fetchall()
    db.close()
    return [dict(r) for r in rows]

@app.get("/volunteers/me")
async def my_profile(auth=Depends(verify_token)):
    db = get_db()
    row = db.execute("""
        SELECT v.*, a.name as area_name, a.density_threshold, a.current_density
        FROM volunteers v LEFT JOIN areas a ON v.area_id = a.id
        WHERE v.id=?
    """, (auth["id"],)).fetchone()
    db.close()
    return dict(row) if row else {}

# ── Area routes ───────────────────────────────────────────────────────────────

@app.post("/areas")
async def create_area(data: AreaCreate, auth=Depends(verify_token)):
    if auth["role"] != "admin":
        raise HTTPException(403, "Admin only")
    db = get_db()
    aid = str(uuid.uuid4())
    db.execute(
        """INSERT INTO areas
           (id,name,camera_source,model_path,scene_width_m,scene_height_m,
            density_threshold,pred_threshold,push_interval,alpha)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (aid, data.name, data.camera_source, data.model_path,
         data.scene_width_m, data.scene_height_m, data.density_threshold,
         data.pred_threshold, data.push_interval, data.alpha)
    )
    db.commit()
    db.close()
    return {"id": aid, "message": f"Area '{data.name}' created"}

@app.patch("/areas/{area_id}")
async def update_area(area_id: str, data: AreaUpdate, auth=Depends(verify_token)):
    if auth["role"] != "admin":
        raise HTTPException(403, "Admin only")
    db = get_db()
    fields = {k: v for k, v in data.dict().items() if v is not None}
    if fields:
        set_clause = ", ".join(f"{k}=?" for k in fields)
        db.execute(f"UPDATE areas SET {set_clause} WHERE id=?",
                   (*fields.values(), area_id))
        db.commit()
        # Hot-reload running worker
        area = db.execute("SELECT * FROM areas WHERE id=?", (area_id,)).fetchone()
        if area and area_id in _workers:
            _workers[area_id].refresh_area(dict(area))
    db.close()
    return {"message": "Updated"}

@app.delete("/areas/{area_id}")
async def delete_area(area_id: str, auth=Depends(verify_token)):
    if auth["role"] != "admin":
        raise HTTPException(403, "Admin only")
    if area_id in _workers:
        _workers[area_id].stop()
        del _workers[area_id]
    db = get_db()
    db.execute("DELETE FROM areas WHERE id=?", (area_id,))
    db.commit(); db.close()
    return {"message": "Deleted"}

@app.get("/areas")
async def list_areas(auth=Depends(verify_token)):
    db = get_db()
    rows = db.execute("SELECT * FROM areas").fetchall()
    db.close()
    areas = [dict(r) for r in rows]
    # Annotate with live worker status
    for a in areas:
        w = _workers.get(a["id"])
        a["worker_status"] = w.status if w else "stopped"
        a["worker_fps"]    = w.fps    if w else 0
    return areas

@app.post("/areas/{area_id}/assign")
async def assign_volunteer(area_id: str, data: AreaAssign, auth=Depends(verify_token)):
    if auth["role"] != "admin":
        raise HTTPException(403, "Admin only")
    db = get_db()
    db.execute("UPDATE volunteers SET area_id=? WHERE id=?", (area_id, data.volunteer_id))
    db.commit(); db.close()
    await manager.send_volunteer(data.volunteer_id, {"type": "area_assigned", "area_id": area_id})
    return {"message": "Assigned"}

# ── Inference worker management routes ───────────────────────────────────────

@app.post("/inference/start")
async def start_worker(data: WorkerStart, auth=Depends(verify_token)):
    if auth["role"] != "admin":
        raise HTTPException(403, "Admin only")
    if data.area_id in _workers and _workers[data.area_id].status == "running":
        return {"message": "Already running"}
    db = get_db()
    area = db.execute("SELECT * FROM areas WHERE id=?", (data.area_id,)).fetchone()
    db.close()
    if not area:
        raise HTTPException(404, "Area not found")
    w = InferenceWorker(dict(area), _push_density)
    _workers[data.area_id] = w
    w.start()
    return {"message": f"Worker started for '{area['name']}'"}

@app.post("/inference/stop")
async def stop_worker(data: WorkerStart, auth=Depends(verify_token)):
    if auth["role"] != "admin":
        raise HTTPException(403, "Admin only")
    w = _workers.get(data.area_id)
    if not w:
        return {"message": "No worker found"}
    w.stop()
    return {"message": "Worker stopped"}

@app.get("/inference/workers")
async def list_workers(auth=Depends(verify_token)):
    if auth["role"] != "admin":
        raise HTTPException(403, "Admin only")
    return [
        {
            "area_id":      aid,
            "area_name":    w.area_name,
            "status":       w.status,
            "fps":          w.fps,
            "last_count":   w.last_count,
            "last_density": w.last_density,
            "error":        w.error,
        }
        for aid, w in _workers.items()
    ]

# ── External inference push (legacy / remote) ─────────────────────────────────

@app.post("/inference/update")
async def inference_update(data: DensityUpdate):
    if data.secret != INFERENCE_SECRET:
        raise HTTPException(403, "Invalid secret")
    db = get_db()
    area = db.execute("SELECT * FROM areas WHERE id=?", (data.area_id,)).fetchone()
    if not area:
        db.close()
        raise HTTPException(404, "Area not found")
    db.close()
    await _push_density(data.area_id, data.count, data.density, data.grid_flags)
    return {"received": True}

# ── Alert routes ──────────────────────────────────────────────────────────────

@app.get("/alerts")
async def list_alerts(auth=Depends(verify_token)):
    db = get_db()
    rows = db.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT 100").fetchall()
    db.close()
    return [dict(r) for r in rows]

@app.post("/alerts/approve")
async def approve_alert(data: AlertApproval, auth=Depends(verify_token)):
    if auth["role"] != "admin":
        raise HTTPException(403, "Admin only")
    db = get_db()
    alert = db.execute("SELECT * FROM alerts WHERE id=?", (data.alert_id,)).fetchone()
    if not alert:
        db.close()
        raise HTTPException(404, "Alert not found")
    area_id = alert["area_id"]
    area = db.execute("SELECT * FROM areas WHERE id=?", (area_id,)).fetchone()
    new_status = "closed" if data.action == "close_area" else "open"
    msg = (f"⚠️ URGENT: Close and redirect crowd in {area['name']}. Density too high."
           if data.action == "close_area"
           else f"✅ {area['name']} is safe. Reopen access.")
    db.execute("UPDATE areas SET status=? WHERE id=?", (new_status, area_id))
    db.execute("UPDATE alerts SET admin_approved=1,resolved=1,resolved_at=datetime('now') WHERE id=?",
               (data.alert_id,))
    mid = str(uuid.uuid4())
    db.execute("INSERT INTO messages (id,area_id,content,sender) VALUES (?,?,?,?)",
               (mid, area_id, msg, "admin"))
    db.commit(); db.close()
    await manager.broadcast_area_volunteers(area_id, {
        "type": "instruction", "action": data.action,
        "area_status": new_status, "message": msg,
        "timestamp": datetime.now().isoformat()
    })
    await manager.broadcast_admin({
        "type": "area_status_changed", "area_id": area_id,
        "status": new_status, "timestamp": datetime.now().isoformat()
    })
    return {"message": f"Area {new_status}, volunteers notified"}

# ── Messages ──────────────────────────────────────────────────────────────────

@app.get("/messages/me")
async def my_messages(auth=Depends(verify_token)):
    db = get_db()
    vol = db.execute("SELECT area_id FROM volunteers WHERE id=?", (auth["id"],)).fetchone()
    if not vol or not vol["area_id"]:
        db.close(); return []
    rows = db.execute(
        "SELECT * FROM messages WHERE area_id=? ORDER BY created_at DESC LIMIT 20",
        (vol["area_id"],)
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]

# ── MJPEG video stream ────────────────────────────────────────────────────────

@app.get("/video/{area_id}")
async def video_stream(area_id: str):
    """
    MJPEG stream of the annotated camera feed for a running worker.
    Open directly in an <img> tag: <img src="http://localhost:8000/video/{id}">
    No auth header needed — the area_id acts as the token for browser img tags.
    """
    w = _workers.get(area_id)
    if not w or w.status not in ("running", "loading"):
        raise HTTPException(404, "No active worker for this area")

    boundary = b"--frame"

    async def generate():
        while True:
            worker = _workers.get(area_id)
            if not worker or worker.status not in ("running", "loading"):
                break
            try:
                # wait up to 200 ms for a fresh frame, then send a keepalive
                jpeg = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: worker.frame_queue.get(timeout=0.2)
                )
            except queue.Empty:
                # send an empty comment to keep the TCP connection alive
                yield boundary + b"\r\n\r\n"
                continue
            yield (
                boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                b"\r\n" + jpeg + b"\r\n"
            )
            await asyncio.sleep(0)   # yield control back to event loop

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/admin/{token}")
async def admin_ws(ws: WebSocket, token: str):
    info = active_tokens.get(token)
    if not info or info["role"] != "admin":
        await ws.close(code=4001); return
    await manager.connect_admin(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_admin(ws)

@app.websocket("/ws/volunteer/{volunteer_id}/{token}")
async def volunteer_ws(ws: WebSocket, volunteer_id: str, token: str):
    info = active_tokens.get(token)
    if not info or info["id"] != volunteer_id:
        await ws.close(code=4001); return
    await manager.connect_volunteer(ws, volunteer_id)
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "area_safe_confirmed":
                db = get_db()
                vol = db.execute("SELECT area_id FROM volunteers WHERE id=?",
                                 (volunteer_id,)).fetchone()
                if vol and vol["area_id"]:
                    area = db.execute("SELECT * FROM areas WHERE id=?",
                                      (vol["area_id"],)).fetchone()
                    db.close()
                    if area:
                        db2 = get_db()
                        aid = str(uuid.uuid4())
                        db2.execute(
                            "INSERT INTO alerts (id,area_id,area_name,alert_type) VALUES (?,?,?,?)",
                            (aid, area["id"], area["name"], "safe_reported")
                        )
                        db2.commit(); db2.close()
                        await manager.broadcast_admin({
                            "type": "safe_report", "alert_id": aid,
                            "area_id": area["id"], "area_name": area["name"],
                            "volunteer_id": volunteer_id,
                            "timestamp": datetime.now().isoformat()
                        })
                else:
                    db.close()
    except WebSocketDisconnect:
        manager.disconnect_volunteer(volunteer_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
