import threading
import queue
import time
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import onnxruntime as ort


def make_inference_fn(model, pred_threshold, image_size):
    h, w = image_size

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1, h, w, 3), dtype=tf.float32)
    ])
    def infer(img_batch):
        pred = model(img_batch, training=False)
        pred = pred[0, ..., 0]
        pred = tf.where(pred < pred_threshold,
                        tf.zeros_like(pred), pred)
        return pred

    return infer


def predict_video(model, source=0, alpha=0.6, pred_threshold=0.2,
                  grid_cols=8, grid_rows=8,
                  density_threshold=4.0,
                  scene_width_m=20.0,
                  scene_height_m=20.0,
                  IMAGE_SIZE=(400, 400)):

    H, W = IMAGE_SIZE

    if isinstance(source, str) and not os.path.isabs(source) and not os.path.exists(source):
        possible_path = os.path.join(os.path.dirname(__file__), source)
        if os.path.exists(possible_path):
            source = possible_path

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        return

    # ── ONNX Session Initialization ───────────────────────────────────────────
    onnx_path = os.path.join(os.path.dirname(__file__), "67_precision49_recall.onnx")
    session = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    # ── Provider Diagnostics ──────────────────────────────────────────────────
    print("Available Providers:", ort.get_available_providers())
    print("Active Providers:", session.get_providers())


    # ── Precompute grid constants ─────────────────────────────────────────────
    meters_per_pixel_x = scene_width_m  / W
    meters_per_pixel_y = scene_height_m / H
    cell_h = H // grid_rows
    cell_w = W // grid_cols
    cell_area_m2 = (cell_h * meters_per_pixel_y) * (cell_w * meters_per_pixel_x)

    # ── Vectorised grid: precompute all cell slice indices as arrays ──────────
    # Instead of a nested Python for-loop, we reshape pred_map into cells
    # and call .sum() once across all 64 cells simultaneously.
    # pred_map (H, W) → (grid_rows, cell_h, grid_cols, cell_w)
    #                  → (grid_rows, grid_cols)  after summing axes 1,3
    # cell_coords[row, col] = (y0, x0) for rectangle drawing
    row_idx = np.arange(grid_rows)
    col_idx = np.arange(grid_cols)
    cell_y0 = (row_idx * cell_h).astype(np.int32)
    cell_x0 = (col_idx * cell_w).astype(np.int32)
    cell_y1 = cell_y0 + cell_h
    cell_x1 = cell_x0 + cell_w

    # ── Colormap LUT ──────────────────────────────────────────────────────────
    _cmap_lut = (plt.get_cmap("hot")(np.linspace(0, 1, 256))[..., :3] * 255).astype(np.uint8)

    # ── Pre-allocate per-frame buffers ────────────────────────────────────────
    img_batch   = np.empty((1, H, W, 3), dtype=np.float32)
    lut_idx     = np.empty((H, W),       dtype=np.uint8)
    _scale_buf  = np.empty((H, W),       dtype=np.float32)

    # ── Thread-safe queues ────────────────────────────────────────────────────
    # capture  → inference  (raw resized uint8 frames)
    # inference → display   (fully annotated BGR frames)
    # maxsize=2 keeps latency bounded — inference can't get more than
    # 2 frames ahead of display, and capture can't flood inference.
    capture_q = queue.Queue(maxsize=2)
    display_q = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    # ── Thread 1: Capture ─────────────────────────────────────────────────────
    def capture_thread():
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                stop_event.set()
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (W, H))
            try:
                capture_q.put(frame_rgb, timeout=0.1)
            except queue.Full:
                pass   # drop frame rather than stall — keeps capture live

    # ── Thread 2: Inference + annotation ─────────────────────────────────────
    def inference_thread():


        while not stop_event.is_set():
            try:
                frame_rgb = capture_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Preprocess into pre-allocated buffer (zero new allocations)
            np.multiply(frame_rgb, 1.0 / 255.0, out=img_batch[0])

            # 2. ONNX Inference
            pred_map = session.run(
                None,
                {input_name: img_batch}
            )[0]

            pred_map = pred_map[0, :, :, 0]
            # Apply Thresholding Equivalent to TensorFlow Version
            pred_map[pred_map < pred_threshold] = 0

            # 3. Heatmap Generation
            total_count = float(pred_map.sum())

            # Heatmap — write into pre-allocated buffers
            vmax = float(pred_map.max()) or 1e-6
            np.multiply(pred_map, 255.0 / vmax, out=_scale_buf)
            np.clip(_scale_buf, 0, 255, out=_scale_buf)
            np.copyto(lut_idx, _scale_buf, casting='unsafe')
            heatmap_rgb = _cmap_lut[lut_idx]
            overlay = cv2.addWeighted(frame_rgb, 1.0, heatmap_rgb, alpha, 0)

            # 4. Grid Density Calculation
            # ── Vectorised grid density ───────────────────────────────────────
            # Reshape pred_map into (grid_rows, cell_h, grid_cols, cell_w),
            # then sum axes 1 and 3 to get (grid_rows, grid_cols) densities
            # in a single NumPy call — replaces 64-iteration Python loop.
            grid = (pred_map
                    .reshape(grid_rows, cell_h, grid_cols, cell_w)
                    .sum(axis=(1, 3)) / cell_area_m2)             # (R, C)

            flagged_mask = grid >= density_threshold
            any_flagged  = bool(flagged_mask.any())

            # 5. Overlay Rendering
            for row, col in zip(*np.where(flagged_mask)):
                y0, x0 = int(cell_y0[row]), int(cell_x0[col])
                y1, x1 = int(cell_y1[row]), int(cell_x1[col])
                density = float(grid[row, col])
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), 2)
                cv2.putText(overlay, f"{density:.1f}/m\u00b2",
                            org=(x0 + 4, y0 + 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.45, color=(255, 0, 0),
                            thickness=1, lineType=cv2.LINE_AA)

            # HUD
            status       = "!! OVERCROWDED !!" if any_flagged else "OK"
            status_color = (255, 0, 0)         if any_flagged else (0, 255, 0)
            cv2.putText(overlay, f"Count: {total_count:.1f}",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 80, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, f"Scene: {scene_width_m}m x {scene_height_m}m",
                        (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(overlay, status,
                        (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)

            out_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

            try:
                display_q.put(out_bgr, timeout=0.1)
            except queue.Full:
                pass   # drop rather than stall inference

            # Count frames for overall FPS (shared counter)
            total_frames[0] += 1

    # ── Thread 3: Display (main thread owns the OpenCV window) ───────────────
    # Shared counter and start time for overall FPS measurement
    total_frames = [0]
    start_time = time.perf_counter()

    t_capture   = threading.Thread(target=capture_thread,   daemon=True)
    t_inference = threading.Thread(target=inference_thread, daemon=True)
    t_capture.start()
    t_inference.start()

    while not stop_event.is_set():
        try:
            out_bgr = display_q.get(timeout=0.1)
        except queue.Empty:
            continue
        cv2.imshow("Crowd Counter  |  Q to quit", out_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    t_capture.join()
    t_inference.join()
    cap.release()
    cv2.destroyAllWindows()

    # Overall FPS (frames processed by inference thread / elapsed seconds)
    elapsed = time.perf_counter() - start_time
    fps = float(total_frames[0]) / elapsed if elapsed > 0 else 0.0
    print(f"Processed {total_frames[0]} frames in {elapsed:.2f}s — FPS: {fps:.2f}")


def weighted_focal_loss(gamma=2.0, pos_weight=800.0):
    bce = tf.keras.losses.BinaryFocalCrossentropy(gamma=gamma)
    def loss(y_true, y_pred):
        weights = 1.0 + y_true * (pos_weight - 1.0)
        return tf.reduce_mean(bce(y_true, y_pred) * tf.squeeze(weights, -1))
    return loss


custom_objects = {'loss': weighted_focal_loss}

keras_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'heatmap_model', '67_precision49_recall.keras'))
model = tf.keras.models.load_model(
    keras_model_path,
    custom_objects=custom_objects
)

predict_video(model, source='concert_crowd.mp4', alpha=0.6, pred_threshold=0.2,
              grid_cols=8, grid_rows=8,
              density_threshold=4.0,
              scene_width_m=30.0,
              scene_height_m=30.0)