"""
Stampede Early Warning System — Inference Server

Setup:
  1. Edit config.py with your settings (area name, camera, model path, etc.)
  2. python video_inference_server.py

No environment variables needed.
"""

import threading
import time
import json
import urllib.request
import urllib.error

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import config

IMAGE_SIZE = (400, 400)
GRID_COLS  = 8
GRID_ROWS  = 8


# ── Loss (same as training) ──────────────────────────────────────────────────

def weighted_focal_loss(gamma=2.0, pos_weight=800.0):
    bce = tf.keras.losses.BinaryFocalCrossentropy(gamma=gamma)
    def loss(y_true, y_pred):
        weights = 1.0 + y_true * (pos_weight - 1.0)
        return tf.reduce_mean(bce(y_true, y_pred) * tf.squeeze(weights, -1))
    return loss


# ── Area lookup — get UUID from area name ────────────────────────────────────

def lookup_area_id(area_name: str):
    """
    Logs in as admin and fetches all areas, then returns the UUID
    matching area_name. Called once at startup.
    """
    if not area_name:
        return None

    # Step 1: login to get token
    try:
        login_payload = json.dumps({
            "username": "admin",
            "password": "admin123",
            "role": "admin"
        }).encode()
        req = urllib.request.Request(
            f"{config.BACKEND_URL}/auth/login",
            data=login_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            login_data = json.loads(resp.read())
        token = login_data["token"]
    except Exception as e:
        print(f"[config] Could not login to backend: {e}")
        return None

    # Step 2: fetch areas
    try:
        req = urllib.request.Request(
            f"{config.BACKEND_URL}/areas",
            headers={"Authorization": f"Bearer {token}"},
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            areas = json.loads(resp.read())
    except Exception as e:
        print(f"[config] Could not fetch areas: {e}")
        return None

    # Step 3: find matching area (case-insensitive)
    for area in areas:
        if area["name"].strip().lower() == area_name.strip().lower():
            print(f"[config] Found area '{area['name']}' -> ID: {area['id']}")
            # Sync settings from DB so admin panel is the single source of truth
            config.DENSITY_THRESHOLD = area.get("density_threshold", config.DENSITY_THRESHOLD)
            config.SCENE_WIDTH_M     = area.get("scene_width_m",     config.SCENE_WIDTH_M)
            config.SCENE_HEIGHT_M    = area.get("scene_height_m",    config.SCENE_HEIGHT_M)
            return area["id"]

    print(f"[config] ERROR: No area named '{area_name}' found.")
    print(f"[config] Available areas: {[a['name'] for a in areas]}")
    return None


# ── HTTP push ────────────────────────────────────────────────────────────────

def push_to_backend(area_id, count, density, grid_flags):
    payload = json.dumps({
        "area_id":    area_id,
        "count":      count,
        "density":    density,
        "grid_flags": grid_flags,
        "secret":     config.INFERENCE_SECRET,
    }).encode()
    req = urllib.request.Request(
        f"{config.BACKEND_URL}/inference/update",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            _ = resp.read()
    except Exception as e:
        print(f"[push] Failed: {e}")


# ── Main inference + display loop ────────────────────────────────────────────

def predict_video(model, area_id):
    source = config.CAMERA_SOURCE
    try:
        source = int(source)
    except (TypeError, ValueError):
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        return

    scene_w = config.SCENE_WIDTH_M
    scene_h = config.SCENE_HEIGHT_M

    meters_per_pixel_x = scene_w / IMAGE_SIZE[1]
    meters_per_pixel_y = scene_h / IMAGE_SIZE[0]

    cell_h = IMAGE_SIZE[0] // GRID_ROWS
    cell_w = IMAGE_SIZE[1] // GRID_COLS
    cell_area_m2 = (cell_h * meters_per_pixel_y) * (cell_w * meters_per_pixel_x)

    last_push   = 0.0
    push_thread = None

    print("[inference] Starting. Press Q in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, IMAGE_SIZE)
        img_batch = (frame_rgb / 255.0)[np.newaxis, ...].astype(np.float32)

        # 2. Predict
        pred_map = model(img_batch, training=False).numpy()[0, ..., 0]
        pred_map[pred_map < config.PRED_THRESHOLD] = 0
        total_count = float(pred_map.sum())

        # 3. Heatmap overlay
        vmax = pred_map.max() or 1e-6
        heatmap_rgb = (plt.get_cmap("hot")(pred_map / vmax)[..., :3] * 255).astype(np.uint8)
        overlay = cv2.addWeighted(frame_rgb.astype(np.uint8), 1.0, heatmap_rgb, config.ALPHA, 0)

        # 4. Grid density check
        any_flagged = False
        grid_flags  = []
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                y0, y1 = row * cell_h, (row + 1) * cell_h
                x0, x1 = col * cell_w, (col + 1) * cell_w
                cell_count   = float(pred_map[y0:y1, x0:x1].sum())
                cell_density = cell_count / cell_area_m2
                if cell_density >= config.DENSITY_THRESHOLD:
                    any_flagged = True
                    grid_flags.append({"row": row, "col": col, "density": round(cell_density, 2)})
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    cv2.putText(overlay, f"{cell_density:.1f}/m2",
                                (x0+4, y0+20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (255, 0, 0), 1, cv2.LINE_AA)

        # 5. HUD
        max_density  = max((f["density"] for f in grid_flags), default=0.0)
        status       = "!! OVERCROWDED !!" if any_flagged else "OK"
        status_color = (255, 0, 0)         if any_flagged else (0, 255, 0)
        push_label   = f"-> Pushing: {config.AREA_NAME}" if area_id else "NOT CONNECTED TO BACKEND"
        push_color   = (0, 200, 255) if area_id else (80, 80, 80)

        cv2.putText(overlay, f"Count: {total_count:.1f}",
                    (15, 35),  cv2.FONT_HERSHEY_SIMPLEX, 1.0,  (255, 80, 0),    2)
        cv2.putText(overlay, f"Scene: {scene_w}m x {scene_h}m",
                    (15, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (200, 200, 200), 1)
        cv2.putText(overlay, f"Max density: {max_density:.1f}/m2",
                    (15, 95),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(overlay, status,
                    (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.0,  status_color,    2)
        cv2.putText(overlay, push_label,
                    (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, push_color,      1)

        out_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imshow("Crowd Counter  |  Q to quit", out_bgr)

        # 6. Push (rate-limited, background thread)
        if area_id:
            now = time.time()
            if now - last_push >= config.PUSH_INTERVAL:
                last_push = now
                if push_thread is None or not push_thread.is_alive():
                    push_thread = threading.Thread(
                        target=push_to_backend,
                        args=(area_id, total_count, max_density, grid_flags),
                        daemon=True,
                    )
                    push_thread.start()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Stampede EWS - Inference Server")
    print("=" * 50)
    print(f"  Backend : {config.BACKEND_URL}")
    print(f"  Area    : {config.AREA_NAME or '(none - no push)'}")
    print(f"  Camera  : {config.CAMERA_SOURCE}")
    print(f"  Model   : {config.MODEL_PATH}")
    print("=" * 50)

    area_id = None
    if config.AREA_NAME:
        print(f"[startup] Looking up area '{config.AREA_NAME}' in backend...")
        area_id = lookup_area_id(config.AREA_NAME)
        if not area_id:
            print("[startup] Will run WITHOUT pushing to backend.")
    else:
        print("[startup] No AREA_NAME in config.py — running standalone.")

    print("[startup] Loading model (this may take a moment)...")
    model = tf.keras.models.load_model(
        config.MODEL_PATH,
        custom_objects={"loss": weighted_focal_loss()}
    )
    print("[startup] Model ready.")

    predict_video(model, area_id=area_id)
