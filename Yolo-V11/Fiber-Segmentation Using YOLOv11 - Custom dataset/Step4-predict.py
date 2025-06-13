from ultralytics import YOLO
import numpy as np
import cv2


model_path = "D:/Temp/Models/Fiber-Segmentation/My-Model-n/weights/best.pt"
model = YOLO(model_path)  # Load the trained YOLOv11 model

image_path = "Best-Semantic-Segmentation-models/Yolo-V11/Fiber-Segmentation Using YOLOv11 - Custom dataset/test_image.png"
image_true_mask_path = "Best-Semantic-Segmentation-models/Yolo-V11/Fiber-Segmentation Using YOLOv11 - Custom dataset/test_mask.png"

img = cv2.imread(image_path)  # Read the input image
img_true_mask = cv2.imread(image_true_mask_path)

H, W, _ = img.shape  # Get the dimensions of the image

# Predict the segmentation mask using the model
results = model(img)

result = results[0]  # Get the first result (the only one in this case)

# Get the model classes
names = model.names  # Get the class names from the model

print(f"names: {names}")

# Create an empty mask with the same dimensions as the input image
final_mask = np.zeros((H, W), dtype=np.uint8)
predicted_classes = result.boxes.cls.cpu().numpy()  # Get the predicted classes as a numpy array

print(f"Predicted classes: {predicted_classes}")

for j , mask in enumerate(result.masks.data):

    mask = mask.cpu().numpy() * 255  # Convert the mask to a numpy array and scale it to 0-255
    class_id = int(predicted_classes[j])  # Get the class ID of the mask
    print("Object " +str(j) + " detected as " + str(class_id) + " = " +names[class_id])

    mask = cv2.resize(mask, (W, H))  # Resize the mask to match the input image dimensions

    # Accumulate the mask into the final mask
    final_mask = np.maximum(final_mask, mask) 

    file_name = "d:/temp/output" + str(j) +".png"

    cv2.imwrite(file_name, mask)  # Save the individual mask to a file

# Save the final mask to a file
final_mask_file = "d:/temp/final_mask.png"
cv2.imwrite(final_mask_file, final_mask)

cv2.imshow("Final Mask", final_mask)  # Display the final mask
cv2.imshow("Input Image", img)  # Display the input image
cv2.imshow("True Mask", img_true_mask)  # Display the true mask
cv2.waitKey(0)  # Wait for a key press to close the windows

cv2.destroyAllWindows()  # Close all OpenCV windows

