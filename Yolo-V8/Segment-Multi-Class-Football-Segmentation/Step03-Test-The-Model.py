from ultralytics import YOLO
import numpy as np 
import cv2 
import random

model_path = "C:/Data-sets/Football-Seg.v1i.yolov8/My-model/weights/best.pt"
image_path = "Best-Semantic-Segmentation-models/Yolo-V8/Segment-Multi-Class-Football-Segmentation/football_test.png"

img = cv2.imread(image_path)
H, W, _ = img.shape

print (img.shape)


model = YOLO(model_path)
results = model(img)

# get the result data
result = results[0]

# get the classes names
names = model.names
print(names)

num_classes = 10
random_colors = [] 

for _ in range(num_classes):

    blue = random.randint(0, 255)
    green = random.randint(0, 255)
    red = random.randint(0, 255)

    color = blue , green , red 
    random_colors.append(color)

print(random_colors)

# create an empty mask to accumulate results
final_mask = np.zeros((H,W,3), dtype=np.uint8)
predicted_classes = result.boxes.cls.cpu().numpy()
print(predicted_classes)


for j , mask in enumerate(result.masks.data):


    mask = mask.cpu().numpy() * 255
    classId = int(predicted_classes[j])
    print("Object " + str(j) + " detected as " + str(classId) + " - " + names[classId])

    mask = cv2.resize(mask , (W, H))

    # assign color based class ID
    color = random_colors[classId]

    # apply color to the mask
    colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
    colored_mask = np.zeros_like(colored_mask)
    colored_mask[mask > 0] = color


    # accumulate masks
    final_mask = np.maximum(final_mask, colored_mask)
    file_name = "output"+str(j)+".png"

    cv2.imwrite("C:/Data-sets/Football-Seg.v1i.yolov8/My-model/"+file_name, colored_mask)

cv2.imwrite("C:/Data-sets/Football-Seg.v1i.yolov8/My-model/final_mask.png",final_mask)

cv2.imshow("img", img)
cv2.imshow("final mask ", final_mask)
cv2.waitKey(0)

# show using pyplot
import matplotlib.pyplot as plt 

# ctrate a subplot with 1 row and 2 columns
fig , axes = plt.subplots(1 , 2, figsize=(12,6))

# plot the original image on the left side
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original image")
axes[0].axis('off')

# plot the final mask on the right side
axes[1].imshow(cv2.cvtColor(final_mask, cv2.COLOR_BGR2RGB))
axes[1].set_title("Final mask ")
axes[1].axis('off')

plt.show()








