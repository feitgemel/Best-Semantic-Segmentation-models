import os 
import numpy as np
import cv2 
from glob import glob
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# import the model 
from model import build_u2net, build_u2net_lite

# Image dimensions
H = 256
W = 256 

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_dataset(path):
    train_x = sorted(glob(os.path.join(path, "train", "blurred_image", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "train", "mask", "*.png")))

    valid_x = sorted(glob(os.path.join(path, "validation", "P3M-500-NP","original_image", "*.jpg")))
    valid_y = sorted(glob(os.path.join(path, "validation", "P3M-500-NP","mask", "*.png")))

    return (train_x, train_y), (valid_x, valid_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0 
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x / 255.0 
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x,y):
    def _parse(x, y):
        x = read_image(x)
        mask = read_mask(y)
        # return list containing input image and all masks
        return( x, mask , mask , mask , mask , mask , mask , mask)
    
    # Define output types for all tensors
    output_types = (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
    outputs = tf.numpy_function(_parse, [x, y], output_types)

    # Set shapes for all tensors
    outputs[0].set_shape((H, W, 3)) # input image
    for i in range(1, 8): # 7 masks
        outputs[i].set_shape((H, W, 1))

    # return image and dictionary of outpus
    return outputs[0], {f"y{i}": outputs[i+1] for i in range(7)}

def tf_dataset (X, Y , batch=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse).batch(batch).prefetch(10)
    return ds

import tensorflow as tf

if __name__ == "__main__":

    np.random.seed(42)
    tf.random.set_seed(42)

    path_for_model_weights = "/mnt/d/temp/Models/U2Net-weights"
    create_dir(path_for_model_weights)

    # Hyperparameters
    batch_size = 4
    lr = 1e-4
    num_epochs = 500
    model_path  = os.path.join(path_for_model_weights, "u2net-model.keras")
    csv_path = os.path.join(path_for_model_weights, "u2net-training-log.csv")

    #Dataset 
    dataset_path = "/mnt/d/Data-Sets-Object-Segmentation/P3M-10k"
    (train_x, train_y), (valid_x, valid_y) = load_dataset(dataset_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)


    #Build Model
    model = build_u2net_lite((H, W, 3)) 


    losses = {
        "y0": "binary_crossentropy",
        "y1": "binary_crossentropy",
        "y2": "binary_crossentropy",
        "y3": "binary_crossentropy",
        "y4": "binary_crossentropy",
        "y5": "binary_crossentropy",
        "y6": "binary_crossentropy"
    }

    loss_weights = {
        "y0": 1.0,
        "y1": 0.4,
        "y2": 0.4,
        "y3": 0.4,
        "y4": 0.4,
        "y5": 0.4,
        "y6": 0.4
    }    


    model.compile(optimizer=Adam(learning_rate=lr), loss=losses, loss_weights=loss_weights) 

    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True),
        EarlyStopping(patience=10, monitor="val_y0_loss", restore_best_weights=False, mode="min"),
        ReduceLROnPlateau(monitor="val_y0_loss", factor=0.1, patience=5, min_lr=1e-7 , verbose=1),
        CSVLogger(csv_path)]
    
    model.fit(
        train_dataset,
        epochs = num_epochs,
        validation_data = valid_dataset,
        callbacks = callbacks,
    )


    














