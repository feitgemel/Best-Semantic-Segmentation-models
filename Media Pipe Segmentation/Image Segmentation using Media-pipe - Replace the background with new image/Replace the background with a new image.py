import cv2 
PathToImage = "Best-Semantic-Segmentation-models/Media Pipe Segmentation/Image Segmentation using Media-pipe - Replace the background with new image/lilach.jpg" 
img = cv2.imread(PathToImage)

new_bg_path = "Best-Semantic-Segmentation-models/Media Pipe Segmentation/Image Segmentation using Media-pipe - Replace the background with new image/Desert.jpg"
new_bg = cv2.imread(new_bg_path)
new_bg = cv2.resize(new_bg, (img.shape[1], img.shape[0]))

cv2.imshow("img", img)

x = 0.5 
y = 0.5 

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers 

RegionOfInterest = vision.InteractiveSegmenterRegionOfInterest
NormalizedKeypoint = containers.keypoint.NormalizedKeypoint

# Create the options that will be used for InteractiveSegmentation
base_options = python.BaseOptions(model_asset_path="D:/Temp/Models/MediaPipe/deeplab_v3.tflite") 

options = vision.InteractiveSegmenterOptions(base_options=base_options,
                                             output_category_mask=True)

# Generate another visualation image where we highlist the selected object

OVERLAY_COLOR = (255,0,0) # Blue

# Create a segnentor 
with python.vision.InteractiveSegmenter.create_from_options(options) as segmentor:

    # Create the media pipe Image 
    image2 = mp.Image.create_from_file(PathToImage)

    #retrieve the category masks for the image
    roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT , 
                           keypoint = NormalizedKeypoint(x,y))
    segmenation_result = segmentor.segment(image2,roi)
    category_mask = segmenation_result.category_mask

    # Convert the BGR to RGB 
    image_data = cv2.cvtColor(image2.numpy_view(), cv2.COLOR_BGR2RGB)

    # Create an overlay image with the desired color
    overlay_image = np.zeros(image_data.shape, dtype=np.uint8)
    overlay_image[:] = OVERLAY_COLOR

    # Create the condition from the category_masks array
    alpha = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1 

    # Create an alpha channal from the condition with the desired opacty (70%)
    alpha = alpha.astype(float) * 0.7


    # Blend the original image with the overlay image using the alpha channel
    output_image2 = image_data * (1-alpha) + overlay_image * alpha
    output_image2 = output_image2.astype(np.uint8)


    # replace the background with the new image
    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1

    image_with_new_bg = np.where(condition, img , new_bg) # Replace the background using the mask 
    cv2.imwrite("Best-Semantic-Segmentation-models/Media Pipe Segmentation/Image Segmentation using Media-pipe - Replace the background with new image/image_with_new_bg.jpg", image_with_new_bg) # Save the overlay image

    cv2.imshow("output_image2", output_image2) # Show the overlay image
    cv2.imshow("image_with_new_bg", image_with_new_bg) # Show the image with new background

cv2.waitKey(0)
cv2.destroyAllWindows()

