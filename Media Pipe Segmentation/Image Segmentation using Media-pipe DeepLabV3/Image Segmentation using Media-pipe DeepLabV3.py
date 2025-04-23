import cv2 
pathToImage = "Best-Semantic-Segmentation-models/Media Pipe Segmentation/Image Segmentation using Media-pipe DeepLabV3/Inbal-Midbar 768.jpg"

img = cv2.imread(pathToImage)

cv2.imshow("img", img)



# Segementation using MediaPipe DeepLabV3
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BG_COLOR = (192, 192, 192)  # Gray
MASK_COLOR = (255, 255, 255)  # White

# Initialize MediaPipe DeepLabV3 model
base_options = python.BaseOptions(model_asset_path='D:/Temp/Models/MediaPipe/deeplab_v3.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)

# Create the image segmenter
with vision.ImageSegmenter.create_from_options(options) as segmenter:
    image = mp.Image.create_from_file(pathToImage)

    # retrieve the masks for the segmented image
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask


    # Generate solid color images for showing the output segmentation mask

    image_data = image.numpy_view()
    fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR

    condition  = np.stack((category_mask.numpy_view(),) *3 , axis=-1) > 0.2
    output_image = np.where(condition , fg_image, bg_image) 

    cv2.imshow("Segmentation Mask", output_image)   




    # Blur the background of the original image
    image_data = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)

    # Apply effect
    blurred_image = cv2.GaussianBlur(image_data, (55, 55), 0)
    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
    blur_image = np.where(condition, image_data, blurred_image)

    cv2.imshow("Blurred Background", blur_image)


cv2.waitKey(0)

cv2.destroyAllWindows()