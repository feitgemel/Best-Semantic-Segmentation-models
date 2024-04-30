# pip uninstall opencv-python-headless
# pip uninstall opencv-python
# pip install opencv-python

import numpy as np 
import torch 
import matplotlib.pyplot as plt 
import cv2 

# copy these functions from this link :
#https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   




imagePath = "Best-Semantic-Segmentation-models/Segment-Anything/3-Segment Anything with one mouse click/Dori.jpg"
image = cv2.imread(imagePath)
image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

def get_clicked_point(image_path):

    chosenX = 0
    chosenY = 0 

    img = cv2.imread(image_path)
    cv2.putText(img, "Choose point with the mouse , and click 'q' to exit :", (20,50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    
    # Callback funtion for mouse event 

    def mouse_callback(event , x, y, flags, param):
        nonlocal chosenX, chosenY

        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img , (x,y), 10 , (0,255,0), -1) # draw a filled circle in the clicked point
            print(x,y)
            chosenX = x 
            chosenY = y


    # Create a window and set the mouse callback
    cv2.namedWindow("Select Point")
    cv2.setMouseCallback("Select Point" , mouse_callback)

    while True :
        cv2.imshow("Select Point", img)
        key = cv2.waitKey(1) & 0xFF 

        # break the loop if the key 'q' was pressed 
        if key == ord('q') :
            break

    cv2.destroyAllWindows()

    return chosenX, chosenY

# run :
x, y = get_clicked_point(imagePath)
print ("Chosen clicked point : ", str(x) + "," + str(y))


# Run the SAM model 
pathForSAMModel = "e:/temp/sam_vit_h_4b8939.pth"
device = "cuda"
model_type = "default"

from segment_anything import sam_model_registry , SamPredictor
sam = sam_model_registry[model_type](checkpoint=pathForSAMModel)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image) # the one converted to RGB

# We have to choose a point over the image
# This the x,y format with label : that 1 is forground and 0 is background

input_point = np.array([[x,y]])
input_label = np.array([1])

plt.figure(figsize=(10,10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()

# Segment the area 

masks , scores, logits = predictor.predict(point_coords=input_point, 
                                           point_labels=input_label, multimask_output=True)

# multimask output = True -> The SAM model outputs 3 masks with a score of the quailty of the mask

print("Masks shapes : ")
print(masks.shape) # we have 3 masks with 1600X1200 shape (gray images )

# lets see the 3 masks :

for i , (mask , score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image) 
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mak {i+1}, Score: {score:.3f}" , fontsize=18)
    plt.axis('off')
    filename = "Best-Semantic-Segmentation-models/Segment-Anything/3-Segment Anything with one mouse click/output"+str(i)+".png"
    plt.savefig(filename)
    plt.show()