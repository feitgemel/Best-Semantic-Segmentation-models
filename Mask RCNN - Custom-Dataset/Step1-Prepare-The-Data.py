# Link to the dataset : https://www.kaggle.com/datasets/beosup/lung-segment/data

# Part 1: Data Preparation (save PyTorch-ready dataset info)
# Prepare the dataset for training a Mask R-CNN model on lung image segmentation.
# prepare_data.py

import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

def prepare_dataset(csv_path, base_dir, output_path, resize=(512, 512)):
    df = pd.read_csv(csv_path)
    dataset = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(base_dir, row['images'])
        mask_path = os.path.join(base_dir, row['masks'])

        # Load and resize
        image = Image.open(image_path).convert("RGB").resize(resize)
        mask = Image.open(mask_path).convert("L").resize(resize, resample=Image.NEAREST)

        # Convert to tensors
        image_tensor = TF.to_tensor(image)                  # shape: [3, H, W], float32
        mask_tensor = torch.from_numpy(np.array(mask))      # shape: [H, W], int64

        # Only 1 class, foreground = 1, background = 0
        obj_ids = torch.unique(mask_tensor)
        obj_ids = obj_ids[obj_ids != 0]  # skip background

        if len(obj_ids) == 0:
            continue  # skip empty masks

        masks = mask_tensor.unsqueeze(0) == obj_ids[:, None, None]  # [N, H, W]

        boxes = []
        for m in masks:
            pos = (m.nonzero(as_tuple=True))
            xmin = pos[1].min().item()
            xmax = pos[1].max().item()
            ymin = pos[0].min().item()
            ymax = pos[0].max().item()
            boxes.append([xmin, ymin, xmax, ymax])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.ones((len(boxes),), dtype=torch.int64),
            'masks': masks.type(torch.uint8),
            'image_id': torch.tensor([len(dataset)]),
            'area': (masks.sum(dim=(1, 2))).float(),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
        }

        dataset.append((image_tensor, target))

    torch.save(dataset, output_path)
    print(f"Saved dataset to: {output_path}")

# Paths
base_dir = "D:/Data-Sets-Object-Segmentation/Lung Image Segmentation Dataset"
prepare_dataset(os.path.join(base_dir, "train.csv"), base_dir, "d:/temp/train_data.pt", resize=(512, 512))
