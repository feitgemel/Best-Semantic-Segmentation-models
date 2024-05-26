import cv2 
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator , colors
import os  


model = YOLO('yolov8n-seg.pt')
names = model.model.names
print(names)

cap = cv2.VideoCapture("Best-Semantic-Segmentation-models/Yolo-V8/Autolabel-Segmentation/test.mp4")

out = cv2.VideoWriter("Best-Semantic-Segmentation-models/Yolo-V8/Autolabel-Segmentation/out.avi",
                      cv2.VideoWriter_fourcc(*'MJPG'), 30 ,
                      (int(cap.get(3)), int(cap.get(4))))

numerator = 0

while True:
    ret , img = cap.read() 
    numerator = numerator + 1

    if not ret:
        print("Break the loop")
        break 

    results = model.predict(img)
    #print(results[0].masks) # just to see mask detection

    if results[0].masks is not None :
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy 

        annotator = Annotator(img , line_width=2)

        for idx,  (mask , cls) in enumerate(zip(masks,clss)):
            det_label = names[int(cls)]
            annotator.seg_bbox(mask = mask,
                               mask_color = colors(int(cls), True),
                               det_label=det_label)


            # Save each instance - segmented object
            instance_folder = os.path.join("Best-Semantic-Segmentation-models/Yolo-V8/Autolabel-Segmentation", det_label)
            os.makedirs(instance_folder, exist_ok=True)
            instance_path = os.path.join(instance_folder, f"{det_label}_{str(numerator)}.png")
            cv2.imwrite(instance_path, img)


    cv2.imshow("img", img)
    out.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 


cap.release()
out.release()
cv2.destroyAllWindows()