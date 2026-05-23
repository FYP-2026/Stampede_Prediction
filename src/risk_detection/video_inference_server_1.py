import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np


def make_inference_fn(model, pred_threshold, image_size):
    """
    Builds a tf.function-compiled inference + preprocess pipeline.
    Traced once on first call, then runs as a static graph every frame.
    input_shape must be fixed so TF can trace a single concrete function.
    """
    h, w = image_size

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1, h, w, 3), dtype=tf.float32)
    ])
    def infer(img_batch):
        pred = model(img_batch, training=False)          # (1, H, W, 1)
        pred = pred[0, ..., 0]                           # (H, W)
        pred = tf.where(pred < pred_threshold,
                        tf.zeros_like(pred), pred)       # threshold in-graph
        return pred

    return infer


def predict_video(model, source=0, alpha=0.6, pred_threshold=0.2,
                  grid_cols=8, grid_rows=8,
                  density_threshold=4.0,
                  scene_width_m=20.0,
                  scene_height_m=20.0,
                  IMAGE_SIZE=(400, 400)):

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        return

    # ── Build the compiled inference function once, outside the loop ─────────
    infer = make_inference_fn(model, pred_threshold, IMAGE_SIZE)

    # Warm-up: forces tracing + XLA compilation before the loop starts,
    # so the first real frame isn't penalised by graph-build time.
    dummy = tf.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32)
    infer(dummy)
    print("Graph compiled — starting capture loop.")

    # ── Precompute grid constants ─────────────────────────────────────────────
    meters_per_pixel_x = scene_width_m  / IMAGE_SIZE[1]
    meters_per_pixel_y = scene_height_m / IMAGE_SIZE[0]

    cell_h = IMAGE_SIZE[0] // grid_rows
    cell_w = IMAGE_SIZE[1] // grid_cols
    cell_area_m2 = (cell_h * meters_per_pixel_y) * (cell_w * meters_per_pixel_x)

    # Pre-build colormap LUT (256 entries) to avoid per-frame matplotlib call
    _lut_indices = np.linspace(0, 1, 256)
    _cmap_lut = (plt.get_cmap("hot")(_lut_indices)[..., :3] * 255).astype(np.uint8)  # (256, 3)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 1. Preprocess (NumPy — stays on CPU, feeds straight to GPU) ───────
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, IMAGE_SIZE)
        # Use np.multiply instead of / 255.0  — avoids a Python float cast
        img_batch = np.multiply(frame_rgb, 1.0 / 255.0, dtype=np.float32)[np.newaxis]

        # ── 2. Predict (runs as compiled graph) ───────────────────────────────
        pred_map = infer(img_batch).numpy()      # already thresholded in-graph
        total_count = pred_map.sum()

        # ── 3. Heatmap overlay (LUT-based, no matplotlib per frame) ───────────
        vmax = pred_map.max() or 1e-6
        lut_idx = (pred_map / vmax * 255).clip(0, 255).astype(np.uint8)
        heatmap_rgb = _cmap_lut[lut_idx]        # vectorised index into LUT
        overlay = cv2.addWeighted(frame_rgb, 1.0, heatmap_rgb, alpha, 0)

        # ── 4. Grid density check & flagging ──────────────────────────────────
        any_flagged = False
        for row in range(grid_rows):
            for col in range(grid_cols):
                y0, y1 = row * cell_h, (row + 1) * cell_h
                x0, x1 = col * cell_w, (col + 1) * cell_w

                cell_density = pred_map[y0:y1, x0:x1].sum() / cell_area_m2

                if cell_density >= density_threshold:
                    any_flagged = True
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    cv2.putText(
                        overlay,
                        f"{cell_density:.1f}/m\u00b2",
                        org=(x0 + 4, y0 + 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.45,
                        color=(255, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

        # ── 5. HUD text ───────────────────────────────────────────────────────
        status = "!! OVERCROWDED !!" if any_flagged else "OK"
        status_color = (255, 0, 0) if any_flagged else (0, 255, 0)

        cv2.putText(overlay, f"Count: {total_count:.1f}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 80, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Scene: {scene_width_m}m x {scene_height_m}m",
                    (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(overlay, status,
                    (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)

        out_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imshow("Crowd Counter  |  Q to quit", out_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def weighted_focal_loss(gamma=2.0, pos_weight=800.0):
    bce = tf.keras.losses.BinaryFocalCrossentropy(gamma=gamma)
    def loss(y_true, y_pred):
        weights = 1.0 + y_true * (pos_weight - 1.0)
        return tf.reduce_mean(bce(y_true, y_pred) * tf.squeeze(weights, -1))
    return loss


custom_objects = {'loss': weighted_focal_loss}

model = tf.keras.models.load_model(
    '../model/heatmap_model/67_precision49_recall.keras',
    custom_objects=custom_objects
)

predict_video(model, source='concert_crowd.mp4', alpha=0.6, pred_threshold=0.2,
              grid_cols=8, grid_rows=8,
              density_threshold=4.0,
              scene_width_m=30.0,
              scene_height_m=30.0)