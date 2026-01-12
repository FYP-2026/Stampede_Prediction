import os
import cv2
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter

# Paths
RAW_TRAIN_PATH = "datasets/raw/UCF-QNRF_ECCV18/Train"
SAVE_IMAGE_PATH = "datasets/density/UCF-QNRF/images"
SAVE_DENSITY_PATH = "datasets/density/UCF-QNRF/density_maps"

os.makedirs(SAVE_IMAGE_PATH, exist_ok=True)
os.makedirs(SAVE_DENSITY_PATH, exist_ok=True)

def generate_density_map(image, points, sigma=15):
    """
    image: input image
    points: list of head coordinates
    sigma: Gaussian kernel size
    """
    h, w = image.shape[:2]
    density_map = np.zeros((h, w), dtype=np.float32)

    for point in points:
        x = int(point[0])
        y = int(point[1])

        if x >= w or y >= h:
            continue

        density_map[y, x] += 1

    density_map = gaussian_filter(density_map, sigma=sigma)
    return density_map


def process_dataset():
    for file in os.listdir(RAW_TRAIN_PATH):
        if file.lower().endswith(".jpg"):
            img_path = os.path.join(RAW_TRAIN_PATH, file)

            # Correct annotation file name
            mat_name = file.replace(".jpg", "_ann.mat")
            mat_path = os.path.join(RAW_TRAIN_PATH, mat_name)

            if not os.path.exists(mat_path):
                print(f"Annotation missing for {file}")
                continue

            # Load image
            image = cv2.imread(img_path)

            # Load annotation
            mat = sio.loadmat(mat_path)
            points = mat["annPoints"]

            # Generate density map
            density_map = generate_density_map(image, points)

            # Save image
            cv2.imwrite(os.path.join(SAVE_IMAGE_PATH, file), image)

            # Save density map
            np.save(
                os.path.join(
                    SAVE_DENSITY_PATH, file.replace(".jpg", ".npy")
                ),
                density_map,
            )

            print(f"Processed {file}")



if __name__ == "__main__":
    process_dataset()
