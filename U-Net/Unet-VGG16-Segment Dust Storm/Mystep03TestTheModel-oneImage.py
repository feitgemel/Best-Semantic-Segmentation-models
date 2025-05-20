import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt 

#load the model
best_model_file = "/mnt/d/temp/models/Dust-Storm/VGG16-Dust-Storm.keras"
model = tf.keras.models.load_model(best_model_file)
print(model.summary())


Height = 128
Width = 128

#-----------------------------------------------------------------------------
# show one image just for test

path_test_image = "Best-Semantic-Segmentation-models/U-Net/Unet-VGG16-Segment Dust Storm/dust_storm_test_img.jpg"
img = cv2.imread(path_test_image, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)

#img = cv2.imread("U-net\Human Image Segmentation - Binary - Good !!\Group.jpg", cv2.IMREAD_COLOR)


img2 = cv2.resize(img, (Width, Height))
img2 = img2 / 255.0
imgForModel = np.expand_dims(img2, axis=0)

p = model.predict(imgForModel)
resultMask = p[0]

print(resultMask.shape)


# Since it is a binary classification so any value above 0.5 means predict to 1
#and every value under 0.5 is predicted to 0
# So, we will update the values of the predicted mask :
# -> under 0.5 to black , and above 0.5 to white
resultMask[resultMask <= 0.5] = 0
resultMask[resultMask > 0.5] = 255

scale_percent = 25 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
mask = cv2.resize(resultMask,dim, interpolation = cv2.INTER_AREA)

cv2.imwrite("Best-Semantic-Segmentation-models/U-Net/Unet-VGG16-Segment Dust Storm/dust_storm_test_mask.jpg",mask)


# Create a figure and set the subplots
plt.figure(figsize=(10, 5))

# Display the first image
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')  # Remove axis for better visualization

# Display the second image
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title("Mask Image")
plt.axis('off')  # Remove axis for better visualization

# Show the images
plt.tight_layout()
plt.show()




