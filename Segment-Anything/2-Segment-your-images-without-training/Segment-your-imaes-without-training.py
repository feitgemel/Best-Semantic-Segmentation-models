import torch 
print ("Cuda is available : " , torch.cuda.is_available())

import numpy as np  
import matplotlib.pyplot as plt 
import cv2 

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam_checkpoint = "e:/temp/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

image = cv2.imread("Best-Semantic-Segmentation-models/Segment-Anything/2-Segment-your-images-without-training/Rahaf.jpg")

print(image.shape)

scale_precent = 40 
width = int(image.shape[1]* scale_precent / 100)
height = int(image.shape[0]* scale_precent / 100)
dim = (width, height)

# resize the image 
image = cv2.resize(image , dim , interpolation= cv2.INTER_AREA)

image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('off')
plt.show()


# Run the segment anything model :

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


mask_generator_ = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.96,
    stability_score_thresh=0.96, 
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

masks = mask_generator_.generate(image)
print("Total mask discovered : " + str(len(masks)))


# show the results :

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = [] 

    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1,3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        
        ax.imshow(np.dstack((img, m*0.35)))


plt.figure(figsize=(10,10))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig("Best-Semantic-Segmentation-models/Segment-Anything/2-Segment-your-images-without-training/output2.png")
plt.show()



