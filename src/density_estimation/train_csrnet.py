import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from csrnet import CSRNet
from dataset import CrowdDataset

# ----------------------------
# Configuration
# ----------------------------
IMG_DIR = "datasets/density/UCF-QNRF/images"
DEN_DIR = "datasets/density/UCF-QNRF/density_maps"

BATCH_SIZE = 1          # CSRNet standard
EPOCHS = 10             # Increase later if GPU time allows
LEARNING_RATE = 1e-5
IMG_SIZE = (800, 576)   # (H, W) fixed size

MODEL_SAVE_PATH = "csrnet_ucf_qnrf.pth"

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Dataset & Loader
# ----------------------------
dataset = CrowdDataset(
    img_dir=IMG_DIR,
    den_dir=DEN_DIR,
    img_size=IMG_SIZE
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,   # IMPORTANT for Colab
    pin_memory=False
)


# ----------------------------
# Model, Loss, Optimizer
# ----------------------------
model = CSRNet().to(device)
criterion = nn.MSELoss(reduction="sum")  # standard for density regression
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ----------------------------
# Training Loop
# ----------------------------
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for images, densities in dataloader:
        images = images.to(device)
        densities = densities.to(device)

        outputs = model(images)
        loss = criterion(outputs, densities)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {epoch_loss:.4f}"
    )

# ----------------------------
# Save Model
# ----------------------------
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved at: {MODEL_SAVE_PATH}")
