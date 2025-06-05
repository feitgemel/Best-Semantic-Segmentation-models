import cv2 
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import os  


model = YOLO("yolo11l-seg.pt")  # Load a custom YOLOv11 model trained for segmentation
names = model.model.names  # Get the class names from the model
print("Class Names:", names)

cap = cv2.VideoCapture("Best-Semantic-Segmentation-models/Yolo-V11/Auto segment any Object/town.mp4") 
output_folder = "d:/temp/output_segment_yoloV11" 
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

out = cv2.VideoWriter(os.path.join(output_folder, "output_segment_yoloV11.mp4"),
                      cv2.VideoWriter_fourcc(*'MJPG'), 30 , (int(cap.get(3)), int(cap.get(4))))  # Define the codec and create VideoWriter object

while True:
    ret , im0 = cap.read()  # Read a frame from the video
    if not ret:
        print("No frame read from video. Exiting...")
        break  # Break the loop if no frame is read

    results = model.predict(im0)
    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()  # 
        masks = results[0].masks.xy 

        annotator = Annotator(im0, line_width=2) 

        for idx , (mask, cls) in enumerate(zip(masks, clss)):
            det_label = names[int(cls)]  # Get the class name for the detected object
            #print(int(cls))

            annotator.seg_bbox(mask = mask ,
                               mask_color=colors(int(cls), True),
                               label = det_label)
            
            # Save each instance segmented object 
            instance_folder = os.path.join(output_folder, det_label)
            os.makedirs(instance_folder, exist_ok=True)
            instance_path = os.path.join(instance_folder, f"{det_label}_{idx}.png")
            cv2.imwrite(instance_path, im0)

    cv2.imshow("Result", im0)  # Display the result
    out.write(im0)  # Write the frame to the output video

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the video capture object
out.release()  # Release the video writer object
cv2.destroyAllWindows()  # Close all OpenCV windows

