import numpy as np
import torch 
import matplotlib.pyplot as plt
import cv2 

# Te target is to generate all the possible masks for the given image

np.random.seed(3)

# Select the decive for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Copy some help functions from the SAM2 examples 
# https://github.com/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb


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
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


# Load the image
image = cv2.imread("code/Elephant2.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
#plt.show()


# Segmentation model
from sam2.build_sam import build_sam2 
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt" # download it in the install part 
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" # part of the SAM2 repo

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2) 


# Generate masks : tun the generator on the image
masks = mask_generator.generate(image)


# Show the masks
print(f"Generated {len(masks)} masks")
print(f"First mask area: {masks[0]['area']}")
print(masks[0].keys() )

# Display the masks
plt.figure(figsize=(20, 20))
plt.imshow(image) # show the image first
show_anns(masks) # image with masks
plt.axis('off')
plt.show()













































