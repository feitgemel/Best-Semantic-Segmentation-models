import cv2 
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

model = YOLO("yolo11l-seg.pt")  # Load a pretrained YOLOv11 segmentation model
cap = cv2.VideoCapture("Best-Semantic-Segmentation-models/Yolo-V11/Segmentation and Tracking using YoloV11/dogs.mp4")
w, h, fps = (int(cap.get(x)) for x in(cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("Best-Semantic-Segmentation-models/Yolo-V11/Segmentation and Tracking using YoloV11/output_video.avi", 
                      cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h) )

while True:
    ret , img = cap.read()

    if not ret:
        print("Video ended or error reading video")
        break


    annotator = Annotator(img, line_width=2) 

    results = model.track(img , persist=True)

    if results[0].boxes is not None and results[0].masks is not None:
        
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for mask , track_id in zip(masks, track_ids):
            color = colors(int(track_id) , True)
            txt_color = annotator.get_txt_color(color)
            annotator.seg_bbox(mask = mask , mask_color = color , 
                               label=str(track_id) , txt_color = txt_color)
            
        out.write(img)
        cv2.imshow("Result", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

out.release()
cap.release()
cv2.destroyAllWindows()
print("Video processing completed and saved to output_video.mp4")

