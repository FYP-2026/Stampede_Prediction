"""
Stampede Early Warning System - Backend Server
FastAPI + WebSocket + SQLite
"""

import asyncio
import json
import sqlite3
import uuid
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Optional, Set
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    HTTPException, Depends, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# ── Database Setup ──────────────────────────────────────────────────────────

DB_PATH = os.environ.get("DB_PATH", "stampede.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
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
            status TEXT DEFAULT 'offline',   -- offline | standby | active
            registered_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS areas (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            camera_source TEXT,              -- CCTV URL / device index
            scene_width_m REAL DEFAULT 20.0,
            scene_height_m REAL DEFAULT 20.0,
            density_threshold REAL DEFAULT 4.0,
            status TEXT DEFAULT 'open',      -- open | closed | monitoring
            current_count REAL DEFAULT 0,
            current_density REAL DEFAULT 0,
            risk_flagged INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            area_id TEXT NOT NULL,
            area_name TEXT,
            alert_type TEXT NOT NULL,        -- risk_detected | area_closed | area_opened
            density REAL,
            count REAL,
            admin_approved INTEGER DEFAULT 0,
            resolved INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            resolved_at TEXT
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            volunteer_id TEXT,               -- NULL = broadcast to all in area
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

    # Seed default admin if not exists
    admin_pw = hashlib.sha256("admin123".encode()).hexdigest()
    cur.execute(
        "INSERT OR IGNORE INTO admins VALUES (?, ?, ?)",
        ("admin-001", "admin", admin_pw)
    )
    conn.commit()
    conn.close()

# ── Pydantic Models ─────────────────────────────────────────────────────────

class VolunteerRegister(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    password: str

class LoginRequest(BaseModel):
    username: str   # email for volunteers, username for admin
    password: str
    role: str       # 'volunteer' | 'admin'

class AreaCreate(BaseModel):
    name: str
    camera_source: str
    scene_width_m: float = 20.0
    scene_height_m: float = 20.0
    density_threshold: float = 4.0

class AreaAssign(BaseModel):
    volunteer_id: str

class AlertApproval(BaseModel):
    alert_id: str
    action: str     # 'close_area' | 'open_area'

class DensityUpdate(BaseModel):
    area_id: str
    count: float
    density: float
    grid_flags: List[dict]
    secret: str     # shared secret so only the inference server can POST

INFERENCE_SECRET = os.environ.get("INFERENCE_SECRET", "stampede-secret-2025")

# ── WebSocket Manager ───────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.admin_connections: Set[WebSocket] = set()
        self.volunteer_connections: Dict[str, WebSocket] = {}  # volunteer_id -> ws

    async def connect_admin(self, ws: WebSocket):
        await ws.accept()
        self.admin_connections.add(ws)

    async def connect_volunteer(self, ws: WebSocket, volunteer_id: str):
        await ws.accept()
        self.volunteer_connections[volunteer_id] = ws
        # Update volunteer status
        conn = get_db()
        conn.execute(
            "UPDATE volunteers SET status='standby' WHERE id=?", (volunteer_id,)
        )
        conn.commit()
        conn.close()

    def disconnect_admin(self, ws: WebSocket):
        self.admin_connections.discard(ws)

    def disconnect_volunteer(self, volunteer_id: str):
        self.volunteer_connections.pop(volunteer_id, None)
        conn = get_db()
        conn.execute(
            "UPDATE volunteers SET status='offline' WHERE id=?", (volunteer_id,)
        )
        conn.commit()
        conn.close()

    async def broadcast_admin(self, payload: dict):
        dead = set()
        for ws in self.admin_connections:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.add(ws)
        self.admin_connections -= dead

    async def send_volunteer(self, volunteer_id: str, payload: dict):
        ws = self.volunteer_connections.get(volunteer_id)
        if ws:
            try:
                await ws.send_json(payload)
            except Exception:
                self.disconnect_volunteer(volunteer_id)

    async def broadcast_area_volunteers(self, area_id: str, payload: dict):
        conn = get_db()
        rows = conn.execute(
            "SELECT id FROM volunteers WHERE area_id=?", (area_id,)
        ).fetchall()
        conn.close()
        for row in rows:
            await self.send_volunteer(row["id"], payload)

manager = ConnectionManager()

# ── Auth Helpers ────────────────────────────────────────────────────────────

security = HTTPBearer(auto_error=False)

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def create_token(user_id: str, role: str) -> str:
    raw = f"{user_id}:{role}:{os.urandom(8).hex()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32] + f"_{user_id}_{role}"

# In-memory token store (replace with Redis in production)
active_tokens: Dict[str, dict] = {}

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing token")
    token = credentials.credentials
    info = active_tokens.get(token)
    if not info:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return info

# ── App Lifecycle ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Stampede Early Warning System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth Routes ─────────────────────────────────────────────────────────────

@app.post("/auth/register")
async def register_volunteer(data: VolunteerRegister):
    conn = get_db()
    try:
        vid = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO volunteers (id,name,email,phone,password_hash) VALUES (?,?,?,?,?)",
            (vid, data.name, data.email, data.phone, hash_pw(data.password))
        )
        conn.commit()
        return {"id": vid, "message": "Registered successfully. Await area assignment."}
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Email already registered")
    finally:
        conn.close()

@app.post("/auth/login")
async def login(data: LoginRequest):
    conn = get_db()
    try:
        pw_hash = hash_pw(data.password)
        if data.role == "admin":
            row = conn.execute(
                "SELECT id FROM admins WHERE username=? AND password_hash=?",
                (data.username, pw_hash)
            ).fetchone()
            if not row:
                raise HTTPException(401, "Invalid credentials")
            token = create_token(row["id"], "admin")
            active_tokens[token] = {"id": row["id"], "role": "admin"}
            return {"token": token, "role": "admin", "id": row["id"]}
        else:
            row = conn.execute(
                "SELECT id,name,area_id FROM volunteers WHERE email=? AND password_hash=?",
                (data.username, pw_hash)
            ).fetchone()
            if not row:
                raise HTTPException(401, "Invalid credentials")
            token = create_token(row["id"], "volunteer")
            active_tokens[token] = {"id": row["id"], "role": "volunteer", "name": row["name"]}
            return {
                "token": token, "role": "volunteer",
                "id": row["id"], "name": row["name"],
                "area_id": row["area_id"]
            }
    finally:
        conn.close()

# ── Volunteer Routes ─────────────────────────────────────────────────────────

@app.get("/volunteers")
async def list_volunteers(auth=Depends(verify_token)):
    if auth["role"] != "admin":
        raise HTTPException(403, "Admin only")
    conn = get_db()
    rows = conn.execute(
        "SELECT v.*, a.name as area_name FROM volunteers v LEFT JOIN areas a ON v.area_id=a.id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/volunteers/me")
async def get_my_profile(auth=Depends(verify_token)):
    conn = get_db()
    row = conn.execute(
        "SELECT v.*, a.name as area_name, a.status as area_status FROM volunteers v "
        "LEFT JOIN areas a ON v.area_id=a.id WHERE v.id=?", (auth["id"],)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Not found")
    return dict(row)

# ── Area Routes ─────────────────────────────────────────────────────────────

@app.post("/areas")
async def create_area(data: AreaCreate, auth=Depends(verify_token)):
    if auth["role"] != "admin":
        raise HTTPException(403, "Admin only")
    conn = get_db()
    aid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO areas (id,name,camera_source,scene_width_m,scene_height_m,density_threshold) "
        "VALUES (?,?,?,?,?,?)",
        (aid, data.name, data.camera_source, data.scene_width_m,
         data.scene_height_m, data.density_threshold)
    )
    conn.commit()
    conn.close()
    return {"id": aid, "message": f"Area '{data.name}' created"}

@app.get("/areas")
async def list_areas(auth=Depends(verify_token)):
    conn = get_db()
    rows = conn.execute("SELECT * FROM areas").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/areas/{area_id}/assign")
async def assign_volunteer(area_id: str, data: AreaAssign, auth=Depends(verify_token)):
    if auth["role"] != "admin":
        raise HTTPException(403, "Admin only")
    conn = get_db()
    conn.execute(
        "UPDATE volunteers SET area_id=? WHERE id=?", (area_id, data.volunteer_id)
    )
    conn.commit()
    conn.close()
    await manager.send_volunteer(data.volunteer_id, {
        "type": "area_assigned",
        "area_id": area_id
    })
    return {"message": "Volunteer assigned"}

# ── Inference Push Route ────────────────────────────────────────────────────

@app.post("/inference/update")
async def inference_update(data: DensityUpdate):
    """Called by the inference server each frame with crowd metrics."""
    if data.secret != INFERENCE_SECRET:
        raise HTTPException(403, "Invalid secret")

    conn = get_db()
    area = conn.execute("SELECT * FROM areas WHERE id=?", (data.area_id,)).fetchone()
    if not area:
        conn.close()
        raise HTTPException(404, "Area not found")

    risk = data.density >= area["density_threshold"]

    conn.execute(
        "UPDATE areas SET current_count=?, current_density=?, risk_flagged=? WHERE id=?",
        (data.count, data.density, int(risk), data.area_id)
    )

    # Create alert if newly risky
    if risk and not area["risk_flagged"]:
        alert_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO alerts (id,area_id,area_name,alert_type,density,count) VALUES (?,?,?,?,?,?)",
            (alert_id, data.area_id, area["name"], "risk_detected", data.density, data.count)
        )
        conn.commit()

        payload = {
            "type": "risk_alert",
            "alert_id": alert_id,
            "area_id": data.area_id,
            "area_name": area["name"],
            "density": round(data.density, 2),
            "count": round(data.count, 1),
            "grid_flags": data.grid_flags,
            "timestamp": datetime.now().isoformat()
        }
        await manager.broadcast_admin(payload)
    else:
        conn.commit()

    # Always push live metrics to admin
    await manager.broadcast_admin({
        "type": "metric_update",
        "area_id": data.area_id,
        "area_name": area["name"],
        "count": round(data.count, 1),
        "density": round(data.density, 2),
        "risk": risk,
        "timestamp": datetime.now().isoformat()
    })

    conn.close()
    return {"received": True, "risk": risk}

# ── Admin Action Routes ─────────────────────────────────────────────────────

@app.get("/alerts")
async def list_alerts(auth=Depends(verify_token)):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM alerts ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/alerts/approve")
async def approve_alert(data: AlertApproval, auth=Depends(verify_token)):
    if auth["role"] != "admin":
        raise HTTPException(403, "Admin only")

    conn = get_db()
    alert = conn.execute(
        "SELECT * FROM alerts WHERE id=?", (data.alert_id,)
    ).fetchone()
    if not alert:
        conn.close()
        raise HTTPException(404, "Alert not found")

    area_id = alert["area_id"]
    area = conn.execute("SELECT * FROM areas WHERE id=?", (area_id,)).fetchone()

    if data.action == "close_area":
        new_status = "closed"
        msg = f"⚠️ URGENT: Please close and redirect crowd in {area['name']}. Density too high."
        alert_type = "area_closed"
    else:
        new_status = "open"
        msg = f"✅ Area {area['name']} is now safe. You may reopen access."
        alert_type = "area_opened"

    conn.execute("UPDATE areas SET status=? WHERE id=?", (new_status, area_id))
    conn.execute(
        "UPDATE alerts SET admin_approved=1, resolved=1, resolved_at=datetime('now') WHERE id=?",
        (data.alert_id,)
    )

    msg_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO messages (id,area_id,content,sender) VALUES (?,?,?,?)",
        (msg_id, area_id, msg, "admin")
    )
    conn.commit()
    conn.close()

    vol_payload = {
        "type": "instruction",
        "action": data.action,
        "area_status": new_status,
        "message": msg,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast_area_volunteers(area_id, vol_payload)
    await manager.broadcast_admin({
        "type": "area_status_changed",
        "area_id": area_id,
        "status": new_status,
        "timestamp": datetime.now().isoformat()
    })

    return {"message": f"Area {new_status}, volunteers notified"}

@app.get("/messages/me")
async def my_messages(auth=Depends(verify_token)):
    conn = get_db()
    vol = conn.execute(
        "SELECT area_id FROM volunteers WHERE id=?", (auth["id"],)
    ).fetchone()
    if not vol or not vol["area_id"]:
        conn.close()
        return []
    rows = conn.execute(
        "SELECT * FROM messages WHERE area_id=? ORDER BY created_at DESC LIMIT 20",
        (vol["area_id"],)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ── WebSocket Endpoints ─────────────────────────────────────────────────────

@app.websocket("/ws/admin/{token}")
async def admin_ws(ws: WebSocket, token: str):
    info = active_tokens.get(token)
    if not info or info["role"] != "admin":
        await ws.close(code=4001)
        return
    await manager.connect_admin(ws)
    try:
        while True:
            await ws.receive_text()  # Keep alive; admin is receive-only
    except WebSocketDisconnect:
        manager.disconnect_admin(ws)

@app.websocket("/ws/volunteer/{volunteer_id}/{token}")
async def volunteer_ws(ws: WebSocket, volunteer_id: str, token: str):
    info = active_tokens.get(token)
    if not info or info["id"] != volunteer_id:
        await ws.close(code=4001)
        return
    await manager.connect_volunteer(ws, volunteer_id)
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            # Handle volunteer "safe" confirmation
            if msg.get("type") == "area_safe_confirmed":
                conn = get_db()
                vol = conn.execute(
                    "SELECT area_id FROM volunteers WHERE id=?", (volunteer_id,)
                ).fetchone()
                if vol and vol["area_id"]:
                    area = conn.execute(
                        "SELECT * FROM areas WHERE id=?", (vol["area_id"],)
                    ).fetchone()
                    conn.close()
                    if area:
                        # Trigger admin notification for re-open approval
                        alert_id = str(uuid.uuid4())
                        conn2 = get_db()
                        conn2.execute(
                            "INSERT INTO alerts (id,area_id,area_name,alert_type) VALUES (?,?,?,?)",
                            (alert_id, area["id"], area["name"], "safe_reported")
                        )
                        conn2.commit()
                        conn2.close()
                        await manager.broadcast_admin({
                            "type": "safe_report",
                            "alert_id": alert_id,
                            "area_id": area["id"],
                            "area_name": area["name"],
                            "volunteer_id": volunteer_id,
                            "timestamp": datetime.now().isoformat()
                        })
                else:
                    conn.close()
    except WebSocketDisconnect:
        manager.disconnect_volunteer(volunteer_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
