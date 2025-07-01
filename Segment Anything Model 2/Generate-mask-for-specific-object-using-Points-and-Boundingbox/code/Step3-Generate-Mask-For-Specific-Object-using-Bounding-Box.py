import numpy as np
import torch 
import matplotlib.pyplot as plt
import cv2 

# Te target is to generate all the possible masks for the given image

np.random.seed(3)

# Select the decive for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Copy the functions from here :
# https://github.com/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 
                        
    ax.imshow(img)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()



# Load the image
image = cv2.imread("code/Elephant2.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB

#plt.figure(figsize=(10, 10))
#plt.imshow(image)
#plt.axis('off')
#plt.show()

#First, load the SAM 2 model and predictor. 
# Change the path below to point to the SAM 2 checkpoint. 
# Running on CUDA and using the default model are recommended for best results.

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt" # download it in the install part 
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" # part of the SAM2 repo

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device) 
predictor = SAM2ImagePredictor(sam2_model) 

# Process the image to produce an image embedding by calling SAM2ImagePredictor.set_image. 
# SAM2ImagePredictor remembers this embedding and will use it for subsequent mask prediction.

predictor.set_image(image)

input_box = np.array([1280, 650, 1630 , 1190])

# Predict whith SAM2ImagePredictor.predict 
# The model returns a list of masks, scores, and point coordinates.

masks , scores , logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,)  # Set to True to get multiple masks


show_masks(image, masks, scores, box_coords=input_box)

# Combining points and boxes
# ==========================
input_point = np.array([[1449, 811], [1383, 799] , [1490, 810], [1574, 887], [1581, 993], [1491, 992], [1420, 992], [1346, 973]])
input_label = np.array([1, 1, 1, 1, 1, 1, 1, 1])

input_box = np.array([1280, 650, 1630, 1190])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=input_box,
    multimask_output=False,
)

show_masks(image, masks, scores, box_coords=input_box, point_coords=input_point, input_labels=input_label)

