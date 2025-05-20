#Dataset : https://www.kaggle.com/datasets/nuwansribandara/elai-dust-storm-dataset-from-modis

#   Pixel Value = 0 - Background
#   Pixel Value = 255 - Water    

import cv2
import numpy as np
import os
#import pandas as pd
import matplotlib.pyplot as plt 
from tqdm import tqdm

Height = 128 # Reduce if there are memory errors messages
Width = 128 # Reduce if there are memory errors messages

allImages = []
maskImages = []
allValidateImages = []
maskValidateImages = []


path = "/mnt/d/Data-Sets-Object-Segmentation/ELAI Dust Storm Dataset from MODIS"
imagespath = path + "/images"
maskPath = path + "/annotations"


# TrainFile = path+"segmentation/train.txt"
# validateFIle =  path+"segmentation/val.txt"

# # train
# df = pd.read_csv(TrainFile, sep=" ", header=None)
# filesList = df[0].values # get the list out of the pandas varaiable

#load one image and one mask

img = cv2.imread(imagespath+"/13.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
mask = cv2.imread(maskPath+"/13_GT.png" , cv2.IMREAD_GRAYSCALE)

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


mask16 = cv2.resize(mask, (16, 16))
print(mask16)

# 0 is the background and 255 is the Object (white object)
#We prefer than the values will be 0 and 1
mask16[mask16 > 0] = 1 

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(mask16)

print(mask16.shape)

# ========================================================================================================

# A U G M E N T A T I O N

# step2 - create sample of augmentation :
import imgaug as ia
import imgaug.augmenters as iaa

hflip= iaa.Fliplr(p=1.0)
hflipImg = hflip.augment_image(img)

vflip= iaa.Flipud(p=1.0) 
vflipImg= vflip.augment_image(img)

rot1 = iaa.Affine(rotate=(-50,20))
rotImg = rot1.augment_image(img)

# Display all images side-by-side
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')

# Horizontally Flipped Image
plt.subplot(1, 4, 2)
plt.imshow(hflipImg)
plt.title("Horizontally Flipped")
plt.axis('off')

# Vertically Flipped Image
plt.subplot(1, 4, 3)
plt.imshow(vflipImg)
plt.title("Vertically Flipped")
plt.axis('off')

# Rotated Image
plt.subplot(1, 4, 4)
plt.imshow(rotImg)
plt.title("Rotated Image")
plt.axis('off')

# Show the images
plt.tight_layout()
plt.show()






# load the train images and masks
print("Start loading the train images and masks .................")


#images_path = os.path.join(path,"images")
#masks_path = os.path.join(path,"masks")

# List all files in the folder
images_file_list = [f for f in os.listdir(imagespath) if os.path.isfile(os.path.join(imagespath, f))]
masks_file_list = [f for f in os.listdir(maskPath) if os.path.isfile(os.path.join(maskPath, f))]

# Print the file list
#print("Files in folder:", images_file_list)
print(str(len(images_file_list)) + " images found")



for file in tqdm(images_file_list, desc="Processing images"):
    filePathForImage = imagespath+"/"+file

    # Create the name for the mask file XXX_GT.png instead of XXX.jpg
    name, ext = os.path.splitext(file)
    new_filename_for_mask = f"{name}_GT.png"
    filePathForMask = maskPath+"/"+new_filename_for_mask
    #print(filePathForImage)
    #print(filePathForMask)
    
    # create the NumpyData for images
    img = cv2.imread(filePathForImage, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (Width, Height))
    img = img / 255.0
    img = img.astype(np.float32)
    allImages.append(img)

    # create the NumpyData for masks
    mask = cv2.imread(filePathForMask, cv2.IMREAD_GRAYSCALE) # gray scale image
    mask = cv2.resize(mask, (Width, Height))
    mask[mask > 0] = 1 
    maskImages.append(mask)

    #DataAugmentaion
    hflip= iaa.Fliplr(p=1.0)
    hflipImg = hflip.augment_image(img)
    hflipMask = hflip.augment_image(mask)
    allImages.append(hflipImg)
    maskImages.append(hflipMask)


    vflip= iaa.Flipud(p=1.0) 
    vflipImg= vflip.augment_image(img)
    vflipMask = vflip.augment_image(mask)
    allImages.append(vflipImg)
    maskImages.append(vflipMask)

    rot1 = iaa.Affine(rotate=(-50,20))
    rotImg = rot1.augment_image(img)
    rotMask = rot1.augment_image(mask)
    allImages.append(rotImg)
    maskImages.append(rotMask)

    
print("Total images : " + str(len(allImages)))
print("Total masks : " + str(len(maskImages)))


print("Start Convert all the lists to Numpy Arrays")
allImageNP = np.array(allImages)
maskImagesNP = np.array(maskImages)
maskImagesNP = maskImagesNP.astype(int) # convert to integer


print("shapes of train images and masks :")
print(allImageNP.shape)
print(maskImagesNP.shape)
print(maskImagesNP.dtype)



print("Save the data ....." )    
np.save('/mnt/d/temp/Dust-Storm-Images.npy', allImageNP)
np.save('/mnt/d/temp/Dust-Storm-Masks.npy', maskImagesNP)
print("Finish save the data ....." ) 


