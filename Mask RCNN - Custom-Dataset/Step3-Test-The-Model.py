# Step3-Infer-Random-Test-Images.py

import os
import random
import torch
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as TF

# --- Paths ---
model_path = "d:/temp/models/lungs/maskrcnn_best.pth"
base_dir = "D:/Data-Sets-Object-Segmentation/Lung Image Segmentation Dataset"
csv_path = os.path.join(base_dir, "test.csv")  # your image list CSV

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
def get_model(num_classes):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

model = get_model(num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

model.to(device)
model.eval()

# --- Read test image list from CSV ---
df = pd.read_csv(csv_path)
image_paths = df["images"].tolist()
random_images = random.sample(image_paths, 3)

# --- Run inference ---
originals = []
masked_results = []

### Create output directory if it doesn't exist.
output_dir = "d:/temp/inference_results"
os.makedirs(output_dir, exist_ok=True)


for rel_path in random_images:
    img_path = os.path.join(base_dir, rel_path)
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = TF.to_tensor(image_rgb).to(device)

    with torch.no_grad():
        output = model([image_tensor])[0]

    # Draw predicted masks
    masked_image = image_rgb.copy()
    for i in range(len(output["masks"])):
        if output["scores"][i] > 0.5:
            mask = output["masks"][i, 0].mul(255).byte().cpu().numpy()
            color = np.random.randint(0, 255, (1, 3), dtype=np.uint8).tolist()[0]
            masked_image[mask > 128] = color

    originals.append(image_rgb)
    masked_results.append(masked_image)

    ### Save the final image with masks drawn on it.
    output_filename = os.path.splitext(os.path.basename(rel_path))[0] + "_predicted.png"
    output_path = os.path.join(output_dir, output_filename)

    ### Convert from RGB to BGR before saving with OpenCV.
    cv2.imwrite(output_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
    print(f"🖼 Saved: {output_path}")

# --- Display all 6 images in 2 rows ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i in range(3):
    axes[0, i].imshow(originals[i])
    axes[0, i].set_title(f"Original: {os.path.basename(random_images[i])}")
    axes[0, i].axis('off')

    axes[1, i].imshow(masked_results[i])
    axes[1, i].set_title("Predicted Mask")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()
