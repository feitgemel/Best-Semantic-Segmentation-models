import os 
import numpy as np
import cv2 
import tensorflow as tf
import matplotlib.pyplot as plt

H = 256
W = 256

test_img = "Best-Semantic-Segmentation-models/U-Net/U2Net-Background Removal-TensorFlow Image Matting Tutorial/test1.jpg"

# load the model
model_path = os.path.join("/mnt/d/temp/Models/U2Net-weights","u2net-model.keras")
model = tf.keras.models.load_model(model_path)

# load the image 
image = cv2.imread(test_img, cv2.IMREAD_COLOR)
x = cv2.resize(image, (W, H))
x = x / 255.0
x = np.expand_dims(x, axis=0)  # add batch dimension


# Prediction 
pred = model.predict(x, verbose=0)

# Convert the predictiions to a grayscale image and normalize

pred_list = [] 
for item in pred:
    p = item[0]* 255 
    p = np.concatenate((p, p, p), axis=-1)  # Convert to 3-channel image
    pred_list.append(p)


# Display first set of predictions using pyplot (7 images in one step)
fig , ax = plt.subplots(1, len(pred_list), figsize=(20, 5))
for i , img in enumerate(pred_list):
    ax[i].imshow(img.astype(np.uint8))
    ax[i].axis('off')

plt.tight_layout()
plt.show()

# Save final mask and display it using pyplot (3 images side by side)

image_h , image_w , _ = image.shape

y0 = pred[0][0] 
y0 = cv2.resize(y0, (image_w, image_h))
y0 = np.expand_dims(y0, axis=-1)  # Add channel dimension
y0 = np.concatenate((y0, y0, y0), axis=-1)  # Convert to 3-channel image

final_images = [image , y0 * 255, image * y0]  # Original, Mask, Original * Mask

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
titles = ['Original Image', 'Mask', 'Image * Mask']

for i , img in enumerate(final_images):
    ax[i].imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax[i].set_title(titles[i])
    ax[i].axis('off')

plt.tight_layout()
plt.savefig("/mnt/d/temp/final_output.png")
plt.show()



