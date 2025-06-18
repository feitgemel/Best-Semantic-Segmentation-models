import torch 
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights


weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

model.eval()  # Set the model to evaluation mode
if torch.cuda.is_available():
    model = model.to('cuda')  # Move the model to GPU if available

COCO_INSTLANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' ]

print(len(COCO_INSTLANCE_CATEGORY_NAMES))  # Print the number of categories

from PIL import Image
from torchvision import transforms as T
import numpy as np
import requests
from io import BytesIO

# The io and requests libraries are used to handle image data from URLs 

def get_prediction(img_path, threshold=0.5 , url=False):
    if url: # we have requested an image from a URL
        response = requests.get(img_path)
        img = Image.open(BytesIO(response.content))
    else:  # we have a local image file
        img = Image.open(img_path)
    
    transform = T.Compose([T.ToTensor()])  # Define the transformation to convert the image to a tensor
    img = transform(img)  # Apply the transformation to the image
    img = img.cuda()
    pred = model([img])  # Send the image to the model for prediction

    pred_score = list(pred[0]['scores'].detach().cpu().numpy())  # Get the prediction scores
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] 
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()  # Get the masks from the predictions
    pred_class = [COCO_INSTLANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].detach().cpu().numpy())]  # Get the predicted classes
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Get the bounding boxes

    masks = masks[:pred_t + 1]  # Select the masks up to the threshold
    pred_boxes = pred_boxes[:pred_t + 1]  # Select the bounding boxes up to the threshold
    pred_class = pred_class[:pred_t + 1]  # Select the classes up to the threshold

    return masks, pred_boxes, pred_class  # Return the masks, bounding boxes, and classes



import matplotlib.pyplot as plt
import cv2 
import random 


from urllib.request import urlopen

def url_to_image(url , readFlag=cv2.IMREAD_COLOR):
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)  # Decode the image from the URL
    return image  # Return the decoded image

def random_color_masks(image):

    # List of random colors for the masks
    colors= [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180], [250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]

    r = np.zeros_like(image, dtype=np.uint8)  # Create a red channel for the mask
    g = np.zeros_like(image, dtype=np.uint8)  # Create a green channel for the mask
    b = np.zeros_like(image, dtype=np.uint8)  # Create a blue channel for the mask

    r[image==1], g[image==1], b[image==1] = colors[random.randrange(0, 10)]
    colored_mask = np.stack([b, g, r], axis=2)
    return colored_mask  # Stack the channels to create a color mask


def instance_segmentation(img_path , threshold=0.6, rect_th=1, text_size=1, text_th=1, url=False):
    masks , boxes , pred_cls = get_prediction(img_path, threshold=threshold, url=url)  # Get the predictions

    if url:
        img = url_to_image(img_path)  # Load the image from the URL
    else:
        img = cv2.imread(img_path)  # Read the image from the local path

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB format

    for i in range(len(masks)):
        rgb_mask = random_color_masks(masks[i])  # Get a random color mask for the instance
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)  # Overlay the mask on the image
        pt1 = tuple(int(x) for x in boxes[i][0])  # Get the top-left corner of the bounding box
        pt2 = tuple(int(x) for x in boxes[i][1])  # Get the bottom-right corner of the bounding box
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), thickness=rect_th)  # Draw the bounding box on the image
        cv2.putText(img, pred_cls[i], pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)  # Add the class label to the image

    return img, pred_cls, masks[i]


# Run the Segmenation
img_path = "Best-Semantic-Segmentation-models\Mask RCNN/the-last-of-us.jpg"
img , pred_classes , masks = instance_segmentation(img_path, rect_th=5 , text_th=4)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB format
cv2.imshow("Instance Segmentation", img)  # Display the segmented image
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close all OpenCV windows
# Save the segmented image
cv2.imwrite("d:/temp/instance_segmentation_result.jpg", img)  # Save the segmented image to disk


