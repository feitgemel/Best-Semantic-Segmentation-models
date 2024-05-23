from ultralytics import YOLO
import numpy as np 
import cv2 

model_path = "C:/Data-sets/Dog segmentation YoloV8/My-model2/weights/best.pt"
image_path = "Best-Semantic-Segmentation-models/Yolo-V8/Segment-Dog-with-New-background/German_Shepherd.jpg"

img = cv2.imread(image_path)
H, W, _ = img.shape 
model = YOLO(model_path)

# run the predictions
results = model(img)

# get classes names
names = model.names
print(names)


result = results[0]

# create an empty mask to accumulate the results 
final_mask = np.zeros((H,W), dtype = np.uint8)
predicted_classes = result.boxes.cls.cpu().numpy()
print(predicted_classes) # We discovered two masks

for j , mask in enumerate(result.masks.data):

    mask = mask.cpu().numpy()*255
    classId = int(predicted_classes[j])

    print( "Object " + str(j) + " detected as " + str(classId) + " -  " + names[classId])

    mask = cv2.resize(mask , (W,H))

    # accumlate masks
    final_mask = np.maximum(final_mask , mask)

    file_name = "output"+str(j)+".png"

    cv2.imwrite("C:/Data-sets/Dog segmentation YoloV8/My-model2/"+file_name , mask)

cv2.imwrite("C:/Data-sets/Dog segmentation YoloV8/My-model2/final_mask.png", final_mask)

# ensure the mask is binary (itf it's not already)
_ , threshold_mask = cv2.threshold(final_mask, 128 , 255 , cv2.THRESH_BINARY)

# convert the mask to the appropraite data type
threshold_mask = threshold_mask.astype(np.uint8)

# Extract the region of interest (ROI) from the original image using the mask 

result = cv2.bitwise_and(img , img , mask=threshold_mask)


cv2.imshow("img" , img)
cv2.imshow("final mask ", final_mask)
cv2.imshow("Result", result )

cv2.waitKey(0)
cv2.destroyAllWindows()



