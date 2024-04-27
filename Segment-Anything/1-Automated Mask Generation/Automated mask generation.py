from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch 

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
pathForSamModel = "e:/temp/sam_vit_h_4b8939.pth" # the downloaded model 

sam = sam_model_registry[MODEL_TYPE](checkpoint=pathForSamModel).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

import cv2 
import supervision as sv 

image_bgr = cv2.imread("Best-Semantic-Segmentation-models/Segment-Anything/1-Automated Mask Generation/brain-MRI.jpg")
image_rgb = cv2.cvtColor(image_bgr , cv2.COLOR_BGR2RGB)

sam_result = mask_generator.generate(image_bgr)

print(sam_result[0].keys())

# dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'])

mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
detections = sv.Detections.from_sam(sam_result=sam_result)

anotated_image = mask_annotator.annotate(scene=image_bgr.copy() , detections=detections)

sv.plot_images_grid(
    images = [image_bgr, anotated_image],
    grid_size=(1,2),
    titles=['source image', 'segmented image']
)

# Sort from the large area to the smallest

sorted_sam_result = sorted(sam_result, key=lambda x: x['area'], reverse=True)

masks = [] 

# itrate through each dictionary in the sorted list
for mask in sorted_sam_result:
    # extreact the segmentation value from the current dictionary 
    segmentation_value = mask['segmentation']

    # append the value into the 'masks' list
    masks. append(segmentation_value)


import math 
# calculate the number of rows and columns neede for plotting the masks :

num_masks = len(masks)
num_cols = 8 # number of the columns you want to display 
num_rows = math.ceil(num_masks / num_cols)

# plot :

sv.plot_images_grid(
    images=masks,
    grid_size=(num_rows, num_cols),
    size=(16,16)
)


