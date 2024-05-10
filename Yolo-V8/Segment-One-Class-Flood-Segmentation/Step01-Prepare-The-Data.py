
# =================================================================================
# data set : Flood Area Segmentation : https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation
# we need to convert the masks to another format .


#The script takes binary mask images, which are basically black and white images where 
#the white areas represent the regions affected by floods. 
#These masks are typically in a format where the white parts represent the flooded areas, 
#and the black parts represent everything else.

#The conversion process involves turning these binary masks into a format called YOLO format, 
#which is commonly used for training models to detect objects in images. 
#YOLO format represents objects in images by specifying their bounding boxes or shapes.

#Here's a simplified breakdown of the conversion process:

# 1. Read Mask Image: The script reads each binary mask image.
# 2. Thresholding: It converts the mask into a purely black and white image, where white pixels represent the flood-affected areas.
# 3. Find Contours: It identifies the boundaries of the white areas, essentially finding the outline of the flooded regions.
# 4. Convert Contours to Polygons: Each contour, representing a flooded region, is converted into a polygon. A polygon is like a shape with multiple straight sides.
# 5. Write to Text File: Finally, the script saves the coordinates of these polygons into a text file. Each line in the text file represents a polygon, and the coordinates of the polygon are written in a specific order. This text file follows the YOLO format, making it suitable for training models to detect flooded areas in images.
# So, in essence, the conversion process changes the representation of flood-affected areas 
#from binary masks to a format that's more suitable for training machine learning 
#models to recognize and locate these flooded regions within images.


import os

import cv2

# source images
input_dir = 'C:/Data-sets/Flood Area Segmentation/Mask'

# tatget folder to create labels 
output_dir = 'C:/Data-sets/Flood Area Segmentation/labels'
if not os.path.exists(output_dir):
    # Create the directory
    os.makedirs(output_dir)


for j in os.listdir(input_dir):
    image_path = os.path.join(input_dir, j)
    # load the binary mask and get its contours
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    H, W = mask.shape
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # convert the contours to polygons
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            polygon = []
            for point in cnt:
                x, y = point[0]
                polygon.append(x / W)
                polygon.append(y / H)
            polygons.append(polygon)

    # print the polygons
    with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f:
        for polygon in polygons:
            for p_, p in enumerate(polygon):
                if p_ == len(polygon) - 1:
                    f.write('{}\n'.format(p))
                elif p_ == 0:
                    f.write('0 {} '.format(p))
                else:
                    f.write('{} '.format(p))

        f.close()
