# Central config for paths and hyperparameters

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"
SAMPLES_DIR = DATASETS_DIR / "samples"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Density estimation defaults
IMG_SIZE = (800, 576)  # (H, W)
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-5
