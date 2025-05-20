import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


# Pixel value = 0 -> Background
# Pixel Value = 255 -> Water


Height = 128 
Width = 128

allImages = [] 
MaskImages = []
AllValidImages = []
MaskValidImages = []

path = "/mnt/d/Data-Sets-Object-Segmentation/Water Bodies Dataset"
imagesPath = path + "/Images"
masksPath = path + "/Masks"


img = cv2.imread(imagesPath + "/water_body_7.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask = cv2.imread(masksPath + "/water_body_7.jpg", cv2.IMREAD_GRAYSCALE)

# Create a figure and set subplot
plt.figure(figsize=(10, 5))

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# Display the mask
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.axis('off')


# Show the plot
plt.tight_layout()
plt.show()

######################################################

mask16 = cv2.resize(mask, (16,16))
print(mask16)

# 0 is the background and 255 is the object (white object)
# We prefer that the values will be 0 and 1 

mask16[mask16 > 0 ] = 1
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(mask16)
print(mask16.shape)

# Augmentation  

# crete a sample of the augmentation 

import imgaug as ia 
import imgaug.augmenters as iaa

hflip = iaa.Fliplr(p=1.0)
hflipImg = hflip.augment_image(img)

vflip = iaa.Flipud(p=1.0)
vflipImg = vflip.augment_image(img)

rot1 = iaa.Affine(rotate=(-50,20))
rot1Img = rot1.augment_image(img)

# Display all augmented images 
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# Horizontal flip
plt.subplot(1, 4, 2)
plt.imshow(hflipImg)
plt.title('Horizontal Flip')
plt.axis('off')


# Vertical flip
plt.subplot(1, 4, 3)
plt.imshow(vflipImg)
plt.title('Vertical Flip')
plt.axis('off')


# Rotation
plt.subplot(1, 4, 4)
plt.imshow(rot1Img)
plt.title('Rotation')
plt.axis('off')

# Show the plot
plt.tight_layout()
plt.show()


# Load the train images and masks 

print( "Start loading the images and masks")

images_path = os.path.join(path, "Images")
masks_path = os.path.join(path, "Masks")


# list all files in the folder 
images_file_list = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
masks_file_list = [f for f in os.listdir(masks_path) if os.path.isfile(os.path.join(masks_path, f))]
print("Number of images: ", len(images_file_list))


for file in tqdm(images_file_list, desc="Proccessing images"):

    filePathForImage = images_path + "/" + file
    filePathForMask = masks_path + "/" + file

    # Create the numpy data for the image
    img = cv2.imread(filePathForImage, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (Height, Width))
    img = img / 255.0 
    img = img.astype(np.float32)
    allImages.append(img)

    # Create the numpy data for the mask
    mask = cv2.imread(filePathForMask, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (Height, Width))
    mask[mask > 0 ] = 1
    MaskImages.append(mask)


    # Augmentation
    hflip = iaa.Fliplr(p=1.0)
    hflipImg = hflip.augment_image(img)
    hflipMask = hflip.augment_image(mask)
    allImages.append(hflipImg)
    MaskImages.append(hflipMask)

    vflip = iaa.Flipud(p=1.0)
    vflipImg = vflip.augment_image(img)
    vflipMask = vflip.augment_image(mask)
    allImages.append(vflipImg)
    MaskImages.append(vflipMask)

    rot1 = iaa.Affine(rotate=(-50,20))
    rot1Img = rot1.augment_image(img)
    rot1Mask = rot1.augment_image(mask)
    allImages.append(rot1Img)
    MaskImages.append(rot1Mask)

print("Number of images after augmentation: ", len(allImages))
print("Number of masks after augmentation: ", len(MaskImages))

print("------------------------------------------------------")
print("Start convert all the lists to numpy arrays")


allImagesNP = np.array(allImages)
MaskImagesNP = np.array(MaskImages)
MaskImagesNP = MaskImagesNP.astype(int)
print("------------------------------------------------------")
print("Shape of allImagesNP: ", allImagesNP.shape)
print("Shape of MaskImagesNP: ", MaskImagesNP.shape)
print("Shape of MaskImagesNP: ", MaskImagesNP.dtype)

print("Save the numpy arrays to disk")
np.save('/mnt/d/temp/Water Bodies-Images.npy', allImagesNP)
np.save('/mnt/d/temp/Water Bodies-Masks.npy', MaskImagesNP)
print("Finished saving the numpy arrays to disk")


























