# Dataset : Fiber Segmentation : https://www.kaggle.com/datasets/franzwagner/pe-fibers

from tqdm import tqdm
import os
import shutil

def copy_and_rename_files(src_root, dest_root):
    os.makedirs(dest_root, exist_ok=True)
    
    for subfolder in tqdm(os.listdir(src_root)):
        subfolder_path = os.path.join(src_root, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                src_file = os.path.join(subfolder_path, filename)
                if os.path.isfile(src_file):
                    new_filename = f"{subfolder}_{filename}"
                    dest_file = os.path.join(dest_root, new_filename)
                    shutil.copy2(src_file, dest_file)
                    #print(f"Copied: {src_file} -> {dest_file}")

# Define source paths for training data
base_path = r"D:\Data-Sets-Object-Segmentation\Fiber Segmentation\fibers_geometric_aug\fibers"
images_src = os.path.join(base_path, "images")
masks_src = os.path.join(base_path, "masks")

TARGET_PATH = r"D:\Data-Sets-Object-Segmentation\Fiber Segmentation\ALL_DATA"
# Define destination paths
train_path = os.path.join(TARGET_PATH, "train")
images_dest = os.path.join(train_path, "images")
masks_dest = os.path.join(train_path, "masks")

# Copy and rename images
copy_and_rename_files(images_src, images_dest)
copy_and_rename_files(masks_src, masks_dest)


# Defince source path for validation data
base_path = r"D:\Data-Sets-Object-Segmentation\Fiber Segmentation\fibers_geometric_aug\fibers\validation"
images_src = os.path.join(base_path, "images")
masks_src = os.path.join(base_path, "masks")

TARGET_PATH = r"D:\Data-Sets-Object-Segmentation\Fiber Segmentation\ALL_DATA"
# Define destination paths
train_path = os.path.join(TARGET_PATH, "validation")
images_dest = os.path.join(train_path, "images")
masks_dest = os.path.join(train_path, "masks")

# Copy and rename images
copy_and_rename_files(images_src, images_dest)
copy_and_rename_files(masks_src, masks_dest)



# Defince source path for test data
base_path = r"D:\Data-Sets-Object-Segmentation\Fiber Segmentation\fibers_geometric_aug\fibers\test"
images_src = os.path.join(base_path, "images")
masks_src = os.path.join(base_path, "masks")

TARGET_PATH = r"D:\Data-Sets-Object-Segmentation\Fiber Segmentation\ALL_DATA"
# Define destination paths
train_path = os.path.join(TARGET_PATH, "test")
images_dest = os.path.join(train_path, "images")
masks_dest = os.path.join(train_path, "masks")

# Copy and rename images
copy_and_rename_files(images_src, images_dest)
copy_and_rename_files(masks_src, masks_dest)





print("All files copied and renamed successfully!")
