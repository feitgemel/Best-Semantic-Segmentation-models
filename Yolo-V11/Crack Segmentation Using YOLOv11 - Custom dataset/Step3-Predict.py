from ultralytics import YOLO
import numpy as np
import cv2
import os 

model_path = "D:/Temp/Models/Crack-Segmentation-Using-YOLOv11/My-model-S/weights/best.pt"  # Path to the trained model
image_path = "Best-Semantic-Segmentation-models/Yolo-V11/Crack Segmentation Using YOLOv11 - Custom dataset/test_image.jpg"
os.makedirs("d:/temp/Fiber-Segment", exist_ok=True)

img = cv2.imread(image_path)  # Read the input image

H, W, _ = img.shape  # Get the dimensions of the image

model = YOLO(model_path)  # Load the trained YOLOv11 model
results = model(img)  # Perform inference on the input image

result = results[0]  # Get the first result from the inference

# get model classes
names = model.names  # Get the class names from the model
print("Classes:", names)  # Print the class names


# Create an empty mask for the segmentation
final_mask = np.zeros((H, W), dtype=np.uint8)

predicted_classes = result.boxes.cls.cpu().numpy()  # Get the predicted classes
print("Predicted classes:", predicted_classes)  # Print the predicted classes

for j , mask in enumerate(result.masks.data):

    mask = mask.cpu().numpy()* 255 # Convert the mask to a numpy array and scale it to 255
    classID = int(predicted_classes[j])  # Get the class ID for the mask
    print("Object "+ str(j) + " detected as " + str(classID) + " - " + names[classID])  # Print the class ID and name

    mask = cv2.resize(mask, (W, H))  # Resize the mask to match the original image dimensions

    # Accumulate the mask for the final output
    final_mask = np.maximum(final_mask , mask) 

    file_name = "output" +str(j) + ".png"  # Create a filename for the mask
    cv2.imwrite("d:/temp/Fiber-Segment/" + file_name, mask)  # Save the mask to disk

# Save the final mask to disk
cv2.imwrite("d:/temp/Fiber-Segment/final_mask.png", final_mask)  # Save the final mask

# Display the final mask
cv2.imshow("Final Mask", final_mask)  # Show the final mask
cv2.imshow("Input Image", img)  # Show the input image
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close all OpenCV windows


