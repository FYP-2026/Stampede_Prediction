import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

def predict_video(model, source=0, alpha=0.6, pred_threshold=0.2,
                  grid_cols=8, grid_rows=8,
                  density_threshold=4.0,
                  scene_width_m=20.0,    # real-world width the camera covers
                  scene_height_m=20.0,
          	  IMAGE_SIZE=(400, 400)):  # real-world height the camera covers
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        return

    # Derive meters-per-pixel from scene dimensions
    meters_per_pixel_x = scene_width_m  / IMAGE_SIZE[1]
    meters_per_pixel_y = scene_height_m / IMAGE_SIZE[0]

    cell_h = IMAGE_SIZE[0] // grid_rows
    cell_w = IMAGE_SIZE[1] // grid_cols
    cell_area_m2 = (cell_h * meters_per_pixel_y) * (cell_w * meters_per_pixel_x)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 1. Preprocess ────────────────────────────────────────────────────
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, IMAGE_SIZE)
        img = frame_rgb / 255.0
        img_batch = img[np.newaxis, ...].astype(np.float32)

        # ── 2. Predict ───────────────────────────────────────────────────────
        pred_map = model(img_batch, training=False).numpy()[0, ..., 0]
        pred_map[pred_map < pred_threshold] = 0
        total_count = pred_map.sum()

        # ── 3. Heatmap overlay ───────────────────────────────────────────────
        vmax = pred_map.max() or 1e-6
        heatmap_rgb = (plt.get_cmap("hot")(pred_map / vmax)[..., :3] * 255).astype(np.uint8)
        overlay = cv2.addWeighted(frame_rgb.astype(np.uint8), 1.0, heatmap_rgb, alpha, 0)

        # ── 4. Grid density check & flagging ─────────────────────────────────
        any_flagged = False
        for row in range(grid_rows):
            for col in range(grid_cols):
                y0, y1 = row * cell_h, (row + 1) * cell_h
                x0, x1 = col * cell_w, (col + 1) * cell_w

                cell_count   = pred_map[y0:y1, x0:x1].sum()
                cell_density = cell_count / cell_area_m2

                if cell_density >= density_threshold:
                    any_flagged = True
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    cv2.putText(
                        overlay,
                        f"{cell_density:.1f}/m²",
                        org=(x0 + 4, y0 + 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.45,
                        color=(255, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

        # ── 5. HUD text ──────────────────────────────────────────────────────
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


# Usage
# predict_video(model, source=0, scene_width_m=15.0, scene_height_m=10.0)
def weighted_focal_loss(gamma=2.0, pos_weight=800.0):
    bce = tf.keras.losses.BinaryFocalCrossentropy(gamma=gamma)
    def loss(y_true, y_pred):
        weights = 1.0 + y_true * (pos_weight - 1.0)  # upweight positives
        return tf.reduce_mean(bce(y_true, y_pred) * tf.squeeze(weights, -1))
    return loss


custom_objects = {
    'loss': weighted_focal_loss
}

model = tf.keras.models.load_model('../model/heatmap_model/67_precision49_recall.keras',
                                   custom_objects=custom_objects)

predict_video(model, source='concert_crowd.mp4', alpha=0.6, pred_threshold=0.2,
                  grid_cols=8, grid_rows=8,
                  density_threshold=4.0,
                  scene_width_m=30.0,    # real-world width the camera covers
                  scene_height_m=30.0
        	)