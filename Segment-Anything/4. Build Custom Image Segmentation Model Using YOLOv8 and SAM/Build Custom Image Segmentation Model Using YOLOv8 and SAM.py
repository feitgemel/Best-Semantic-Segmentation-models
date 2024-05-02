# Step1 : Object Detection using YoloV8
# We will mark the bounding box over the object (a dog )
# we will use the box dimentions for the Segmentation Anything model to put a mask on the object and extract it 

import numpy as np 
import torch 
import matplotlib.pyplot as plt 
import cv2 

# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb

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


imagePath = "Best-Semantic-Segmentation-models/Segment-Anything/4. Build Custom Image Segmentation Model Using YOLOv8 and SAM/Dori.jpg"
image = cv2.imread(imagePath)
cv2.imshow("image", image)
cv2.waitKey(0)

# Yolo 8
# pip install ultralytics 

from ultralytics import YOLO
import pickle 

model = YOLO('yolov8n.pt')
names = model.names
print(names)

objects = model(image)
print(objects)

print("For loop : ")
for obj in objects:
    for c in obj.boxes.cls:
        print(names[int(c)])


# class 16 is a dog 
print ("======================================")
# lets run the only for a dog
objects = model(image , classes=[16])

from segment_anything import sam_model_registry, SamPredictor

# plot a rectangle around the dog in the image
for obj in objects:
    boxes = obj.boxes # grab the boxes object
    print(boxes)
    cls = boxes.cls
    print(cls)

    output_index = cls 
    class_name = names[int(output_index)]
    print(class_name)


    if output_index == 16 : # Dog
        # get the coordinates 
        xyxy_coordinates = boxes.xyxy.cpu().numpy()

        # extract x1, y1 , x2 , y2 
        x1, y1 , x2, y2 = xyxy_coordinates[0]

        print("x1:", x1)
        print("y1:", y1)
        print("x2:", x2)
        print("y2", y2)

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.rectangle(image , (x1,y1 , x2, y2) , (0,255,0) , 2)
        cv2.putText(image , class_name, (x1+10 , y1+30), cv2.FONT_HERSHEY_COMPLEX, 1 , (0,0,255), 2)

        image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

        #Download the default trained model: 
        #https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
        pathForSamMODEL = "e:/temp/sam_vit_h_4b8939.pth"
        device = "cuda"
        model_type = "vit_h"

        sam = sam_model_registry[model_type](checkpoint=pathForSamMODEL)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        predictor.set_image(image)

        input_box = np.array([x1 , y1 , x2, y2])
        print(input_box)
        print(input_box[None, :])

        # segment the bounding box area

        masks , scores , logits = predictor.predict(point_coords=None, point_labels=None,
                                                    multimask_output=False,
                                                    box = input_box[None, :])
        
        plt.figure(figsize=(10,10))
        plt.imshow(image)

        # show the mask , and show the bounding boxes 
        show_mask(masks[0], plt.gca())
        show_box(input_box, plt.gca())
        plt.axis('off')
        plt.savefig("Best-Semantic-Segmentation-models/Segment-Anything/4. Build Custom Image Segmentation Model Using YOLOv8 and SAM/out.jpg")
        plt.show()
