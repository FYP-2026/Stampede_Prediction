# Stampede EWS — v1 Issues, Upgrade Plan & v2 Changelog

---

## Part 1 — Issues in the Current Version (v1)

### Issue 1 · Manual config.py editing required for every camera feed

**What happens:**
Every time you want to add a new camera, you must open `config.py` in a text
editor and manually set `AREA_NAME`, `CAMERA_SOURCE`, `MODEL_PATH`,
`SCENE_WIDTH_M`, `SCENE_HEIGHT_M`, `DENSITY_THRESHOLD`, and several other
variables.

**Why it's a problem:**
- Requires developer-level access to the file system on the machine running
  inference.
- Easy to make typos (e.g. wrong path separator, mismatched area name) that
  silently break inference.
- Non-technical operators — security staff, event managers — cannot add or
  change feeds without help.
- There is no validation: a wrong value only fails at runtime, usually mid-event.

---

### Issue 2 · One terminal window per video feed (no parallel management)

**What happens:**
`video_inference_server.py` is a single-feed script. To monitor three gates
simultaneously you must open three Command Prompt windows, navigate to the
correct directory in each, and run the script three times.

**Why it's a problem:**
- Completely unscalable. A 10-camera event needs 10 open terminals.
- Each terminal is an independent process with its own state; there is no
  unified view of which feeds are running, which have crashed, or which are
  healthy.
- Closing a terminal window (accidentally or deliberately) silently kills
  inference for that area with no alert to the admin.
- Restarting a crashed feed requires manually re-opening a terminal and re-
  running the script — impossible to do quickly in an emergency.

---

### Issue 3 · No live configuration changes — restart required for every change

**What happens:**
If you change `DENSITY_THRESHOLD` or `SCENE_WIDTH_M` in config.py after the
server is running, you must stop the terminal and restart the script for the
change to take effect.

**Why it's a problem:**
- In a live event environment the density threshold needs to be tunable in
  real-time as conditions change.
- Restarting loses any in-progress frame-by-frame state and can cause a gap in
  monitoring.

---

### Issue 4 · Area name must match exactly — brittle area lookup

**What happens:**
`video_inference_server.py` calls `lookup_area_id()` which does a
case-insensitive string match of `AREA_NAME` against the database. If the name
in config.py and the name in the Admin Panel differ by even one character (e.g.
`"Gate A"` vs `"Gate A "` with a trailing space), the server runs in
standalone mode and **pushes nothing** — silently.

**Why it's a problem:**
- The failure mode is silent: the video window says `NOT CONNECTED TO BACKEND`
  in small grey text, easy to miss under pressure.
- No error is shown in the Admin Panel.

---

### Issue 5 · No way to see feed health from the Admin Panel

**What happens:**
The Admin Panel has no indication of which camera feeds are actually running.
An area card shows `Count: 0.0` and `Density: 0.00` whether the inference
server is running and producing zeros, or whether it has crashed entirely.

**Why it's a problem:**
- An operator has no way to distinguish "this area is empty" from "the camera
  feed for this area has died."
- During a critical event, a dead feed looks identical to a safe area.

---

### Issue 6 · Model path is hardcoded per-machine

**What happens:**
`MODEL_PATH` in config.py is an absolute file-system path. Moving the project
to a different machine, different OS, or different directory requires editing
the file again.

**Why it's a problem:**
- Increases setup time when deploying to a new machine.
- Easy to forget when doing a quick re-deployment.

---

## Part 2 — Plan to Fix All Issues

### Solution for Issues 1, 2, 4, 6 · In-process worker management via GUI

**Approach:**
Move inference from a standalone script to in-process background threads
managed by the backend server (`server.py`). Add three new API routes:

| Route | Purpose |
|---|---|
| `POST /inference/start` | Start a worker thread for an area |
| `POST /inference/stop`  | Stop a running worker thread |
| `GET /inference/workers`| List all workers with status/fps/count |

Each `InferenceWorker` object receives its configuration directly from the
database (no config.py), is addressed by `area_id` UUID (no name matching), and
is started/stopped from the Admin Panel by clicking a button.

**Result:**
- Zero config files. Everything is typed once into the Admin Panel form.
- One server process, unlimited concurrent feeds, each in its own daemon thread.
- Area lookup is by UUID — no string matching, no silent failures.
- Model path is stored in the database per-area, not per-machine-config.

---

### Solution for Issue 3 · Hot-reload of settings

**Approach:**
Every time a worker completes a push cycle it calls `refresh_area()` which
re-reads its area's row from the database. The `PATCH /areas/{id}` route
updates the database and immediately calls `refresh_area()` on the running
worker.

**Result:**
Density threshold, pred threshold, and push interval changes take effect on
the next frame cycle — no restart needed.

---

### Solution for Issue 5 · Worker status visible in Admin Panel

**Approach:**
`GET /areas` now returns two extra fields per area: `worker_status` (idle /
loading / running / stopped / error) and `worker_fps`. These are rendered:

- On area cards in the Live Monitor tab as a coloured dot + fps badge.
- In the Feeds & Cameras tab as a full status table with per-area start/stop
  controls and an error message field.

**Result:**
An operator can immediately see whether each feed is running, what fps it is
processing at, and the last reported count — without leaving the browser.

---

## Part 3 — What Changed in v2 (File-by-File)

### server.py

| Area | Change |
|---|---|
| `areas` table | Added columns: `model_path`, `pred_threshold`, `push_interval`, `alpha` |
| `AreaCreate` / `AreaUpdate` | New fields to match the expanded schema |
| `InferenceWorker` class | New: full in-process camera+model thread |
| `_workers` registry | New: `Dict[area_id, InferenceWorker]` |
| `_push_density()` | New: shared logic called by both in-process workers and the legacy `/inference/update` HTTP route |
| `POST /inference/start` | New route — starts a worker for an area |
| `POST /inference/stop`  | New route — stops a worker |
| `GET /inference/workers`| New route — lists all workers with live metrics |
| `PATCH /areas/{id}`     | New route — live-edits area config and hot-reloads worker |
| `DELETE /areas/{id}`    | New route — removes area and stops its worker |
| `GET /areas`            | Extended: now includes `worker_status` and `worker_fps` |
| `POST /inference/update`| Kept for backward compat with external inference servers |
| `lifespan`              | Captures the event loop so worker threads can post to it |

### index.html (Admin Panel)

| Area | Change |
|---|---|
| **Feeds & Cameras tab** | New tab: full area creation form (all settings in GUI), workers status table with Start/Stop/Edit/Delete per row |
| **Area creation form** | All 8 parameters configurable from UI — name, camera source, model path, dimensions, thresholds, push interval, heatmap alpha |
| **Edit modal** | Click ✏ on any feed to edit settings live; changes hot-reload into the running worker |
| **Live Monitor — area cards** | Now show worker status dot + fps badge + inline Start/Stop button |
| **Feeds & Cameras tab** | Status table shows FPS, last count, last density, and worker errors per area |
| **Volunteers tab** | Moved from inline sidebar to full tab with area-selector dropdown for assignment |
| **WebSocket auto-reconnect** | Retries every 3 s if disconnected |
| **Alert sound** | Web Audio API beep on risk_alert |

### video_inference_server.py

No changes required. It continues to work as a standalone script for remote
machines that run inference on separate hardware and push via HTTP to
`POST /inference/update`.

---

## Part 4 — Setup Instructions for v2

```
1. pip install fastapi uvicorn tensorflow opencv-python numpy
   (tensorflow + cv2 only needed if running inference on the same machine)

2. python server.py
   → http://localhost:8000

3. Open index.html in a browser.
   Sign in: admin / admin123

4. Click "Feeds & Cameras" tab.
   Fill in the form — area name, RTSP URL or file path, model path,
   scene dimensions, thresholds.
   Click "Create Area & Save".

5. The new area appears in the table below.
   Click ▶ Start to begin inference. The dot turns green.

6. Switch to "Live Monitor" to watch density in real time.
   Area cards show worker status and FPS.

7. To change any setting (e.g. lower the density threshold live):
   Click ✏ Edit → change value → Save Changes.
   The running worker picks up the new value on the next push cycle.
```

No config.py files. No extra terminal windows. One server, one browser tab.
