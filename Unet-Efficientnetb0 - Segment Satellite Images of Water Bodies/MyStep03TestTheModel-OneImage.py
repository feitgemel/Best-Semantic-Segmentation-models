import numpy as np
import tensorflow as tf 
import cv2 
import matplotlib.pyplot as plt 

# load the model 
best_model_file = "/mnt/d/temp/models/efficientnetb0_unet_Water_bodies.keras"
model = tf.keras.models.load_model(best_model_file)
print(model.summary() )

# ---------------------------------------
Width = 128
Height = 128 
# Show one image for test 

path_test_image = "Best-Semantic-Segmentation-models/Unet-Efficientnetb0 - Segment Satellite Images of Water Bodies/test_img.jpg"
img = cv2.imread(path_test_image, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

img2 = cv2.resize(img , (Width , Height))
img2 = img2 / 255.0 
imgForModel = np.expand_dims(img2 , axis=0)

p = model.predict(imgForModel)
resultMask = p[0]
print(resultMask.shape)

# Since it is a binary segmentation , so any value above 0.5 means predict to 1, 
# and every value under 0.5 is predicted to 0 
# So , lets update the values of the predicted mask : under 0.5 to black , and above to white 

resultMask[resultMask <= 0.5] = 0
resultMask[resultMask > 0.5 ] = 255 

scale_precent = 25 
width = int(img.shape[1] * scale_precent / 100)
height = int(img.shape[0] * scale_precent / 100)
dim = (width, height)

img = cv2.resize(img , dim , interpolation= cv2.INTER_AREA)
mask = cv2.resize(resultMask , dim , interpolation= cv2.INTER_AREA)

cv2.imwrite("/mnt/d/temp/water_predicted_mask.png", mask)

# Create a figure and set the subplots
plt.figure(figsize=(10,5))

# Display the first image 
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Origianl image")
plt.axis('off')


# Display the second image
plt.subplot(1,2,2)
plt.imshow(mask , cmap='gray')
plt.title("Mask image")
plt.axis('off')

plt.tight_layout()
plt.show()