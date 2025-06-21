# Step2-Train-MaskRCNN.py

import os
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load training data safely ---
train_data = torch.load("d:/temp/train_data.pt", weights_only=True)
train_loader = DataLoader(
    train_data,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

# --- Model output folder ---
model_dir = "d:/temp/models/lungs"
os.makedirs(model_dir, exist_ok=True)

# --- Build the Mask R-CNN model ---
def get_model(num_classes):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)

    # Replace classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

model = get_model(num_classes=2)
model.to(device)

# --- Optimizer ---
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.0005)

# --- Training settings ---
num_epochs = 50
patience = 10
best_loss = float("inf")
epochs_without_improvement = 0

# --- Training loop ---
for epoch in range(num_epochs):
    print(f"\n📘 Epoch {epoch + 1}/{num_epochs}")
    model.train()
    epoch_loss = 0.0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        progress_bar.set_postfix(loss=total_loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"🔹 Epoch {epoch+1:02d} | Avg Loss: {avg_loss:.4f}")

    # --- Save best model ---
    best_model_path = os.path.join(model_dir, "maskrcnn_best.pth")
    last_model_path = os.path.join(model_dir, "maskrcnn_last.pth")

    if avg_loss < best_loss:
        best_loss = avg_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), best_model_path)
        print("✅ Best model saved.")
    else:
        epochs_without_improvement += 1
        print(f"⚠️  No improvement for {epochs_without_improvement} epoch(s).")

    # Always save last model
    torch.save(model.state_dict(), last_model_path)

    # --- Early stopping ---
    if epochs_without_improvement >= patience:
        print(f"\n⏹ Early stopping triggered after {epoch+1} epochs.")
        break

print("\n🎉 Training complete.")
