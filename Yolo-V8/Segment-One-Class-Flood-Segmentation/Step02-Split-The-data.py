import os
import random
import shutil

# split 80% train and 20% validations

def split_data(source, destination_train, destination_val, split_ratio):
    # Create train and val directories if they don't exist
    if not os.path.exists(destination_train):
        os.makedirs(destination_train)
    if not os.path.exists(destination_val):
        os.makedirs(destination_val)
    
    # Create image and labels directories within train and val directories
    train_image_dir = os.path.join(destination_train, "images")
    train_label_dir = os.path.join(destination_train, "labels")
    val_image_dir = os.path.join(destination_val, "images")
    val_label_dir = os.path.join(destination_val, "labels")
    if not os.path.exists(train_image_dir):
        os.makedirs(train_image_dir)
    if not os.path.exists(train_label_dir):
        os.makedirs(train_label_dir)
    if not os.path.exists(val_image_dir):
        os.makedirs(val_image_dir)
    if not os.path.exists(val_label_dir):
        os.makedirs(val_label_dir)
    
    # Get list of images
    images = os.listdir(os.path.join(source, "image"))
    # Shuffle the list
    random.shuffle(images)
    
    # Calculate split index
    split_index = int(len(images) * split_ratio)
    
    # Split images
    train_images = images[:split_index]
    val_images = images[split_index:]
    
    # Copy images and corresponding labels to train folder
    print("Copying files to train folder:")
    for image_name in train_images:
        # Copy image
        shutil.copy(os.path.join(source, "image", image_name), train_image_dir)
        print(f"Copied {image_name} to {train_image_dir}")
        # Copy label
        label_name = os.path.splitext(image_name)[0] + ".txt"
        shutil.copy(os.path.join(source, "labels", label_name), train_label_dir)
        print(f"Copied {label_name} to {train_label_dir}")
    
    # Copy images and corresponding labels to val folder
    print("Copying files to val folder:")
    for image_name in val_images:
        # Copy image
        shutil.copy(os.path.join(source, "image", image_name), val_image_dir)
        print(f"Copied {image_name} to {val_image_dir}")
        # Copy label
        label_name = os.path.splitext(image_name)[0] + ".txt"
        shutil.copy(os.path.join(source, "labels", label_name), val_label_dir)
        print(f"Copied {label_name} to {val_label_dir}")

# Define source directory
source_dir = "C:/Data-sets/Flood Area Segmentation"

# Define destination directories
destination_train = os.path.join(source_dir, "Train")
destination_val = os.path.join(source_dir, "Val")

# Split data with 80% for training
split_data(source_dir, destination_train, destination_val, split_ratio=0.8)
