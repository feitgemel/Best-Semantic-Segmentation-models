import cv2 

pathToImage = "Best-Semantic-Segmentation-models/Media Pipe Segmentation/image Segmentation and  Highlight the object with color/lilach.jpg"
img = cv2.imread(pathToImage)
cv2.imshow("img", img)

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers  

x = 0.5 #@param {type:"slider" , min:0 , max:1 , step:0.01}
y = 0.5 #@param {type:"slider" , min:0 , max:1 , step:0.01}

region_of_interest = vision.InteractiveSegmenterRegionOfInterest
normalized_keypoint = containers.keypoint.NormalizedKeypoint 

# Create the options that will be used for Interactive segmentation.
# Hightlight the background  

base_options = python.BaseOptions(model_asset_path='D:/Temp/Models/MediaPipe/deeplab_v3.tflite')

# Highlight the object 

options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)

OVERLAY_COLOR = (255,0,0) # Blue color

# Create the image segmenter.
with python.vision.InteractiveSegmenter.create_from_options(options) as segmenter:

    # Create the Mediapipe image 
    image2 = mp.Image.create_from_file(pathToImage)

    # Retrrieve the category masks for the image 

    roi = region_of_interest(format = region_of_interest.Format.KEYPOINT, 
                             keypoint = normalized_keypoint(x,y))
    segementation_result = segmenter.segment(image2, roi)
    category_mask = segementation_result.category_mask


    # Convert the BGR to RGB format
    image_data = cv2.cvtColor(image2.numpy_view(), cv2.COLOR_BGR2RGB)

    # Create an overlay image with the desired color 
    overlay_image = np.zeros(image_data.shape , dtype=np.uint8)
    overlay_image[:] = OVERLAY_COLOR

    # Create the condition from the category mask array 
    alpha = np.stack((category_mask.numpy_view(),) * 3 , axis=-1) > 0.1 

    # Create an akpha channel from the condition with the desired opacity ( e.g. , 0.7 for 70%)
    alpha = alpha.astype(float) * 0.7 

    # Blend the original image and the overlay image using the alpha channel
    output_image2 = image_data * (1-alpha) + overlay_image * alpha
    output_image2 = output_image2.astype(np.uint8)

    cv2.imwrite("Best-Semantic-Segmentation-models/Media Pipe Segmentation/image Segmentation and  Highlight the object with color/output_image2.jpg", output_image2)
   

    cv2.imshow("output_image2", output_image2)
  
cv2.waitKey(0)

cv2.destroyAllWindows()

