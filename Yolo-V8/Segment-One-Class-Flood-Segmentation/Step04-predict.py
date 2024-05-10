from ultralytics import YOLO
import numpy as np 
import cv2 

model_path = "C:/Data-sets/Flood Area Segmentation/My-Flood-Model/weights/best.pt"
image_path = "Best-Semantic-Segmentation-models/Yolo-V8/Segment-One-Class-Flood-Segmentation/test-flood.jpg"

img = cv2.imread(image_path)
H , W , _ = img.shape

model = YOLO(model_path)

# predict
results = model(img)
result = results[0]

#print(result)

# get names of classess
names = model.names
print(names)

# Create an empty mask tp accumulate all the masks to one image

final_mask = np.zeros((H,W) , dtype=np.uint8)
predicted_classes = result.boxes.cls.cpu().numpy()
print(predicted_classes)

for j, mask in enumerate(result.masks.data):

    mask = mask.cpu().numpy() * 255 
    classId = int(predicted_classes[j])

    print("Object "+str(j) + " detected as " + str(classId) + " - " + names[classId])

    mask = cv2.resize(mask, (W,H))

    # accumulate the masks
    final_mask = np.maximum(final_mask, mask)

    file_name = "output"+str(j)+".png"

    # save each mask
    cv2.imwrite("C:/Data-sets/Flood Area Segmentation/My-Flood-Model/"+file_name , mask)

# save the final mask 
cv2.imwrite("C:/Data-sets/Flood Area Segmentation/My-Flood-Model/final_mask.png" , final_mask)


# show the results
scale_precent = 30
width = int(img.shape[1] * scale_precent / 100)
height = int(img.shape[0] * scale_precent / 100)
dim=(width, height)

# resize the image
resized = cv2.resize(img , dim , interpolation= cv2.INTER_AREA)
resized_mask = cv2.resize(final_mask , dim , interpolation= cv2.INTER_AREA)

cv2.imshow("img", resized)
cv2.imshow("final mask ", resized_mask)
cv2.waitKey(0)










