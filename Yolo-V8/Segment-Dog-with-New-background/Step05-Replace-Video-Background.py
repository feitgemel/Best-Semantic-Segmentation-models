from ultralytics import YOLO
import numpy as np 
import cv2 

video_path = "Best-Semantic-Segmentation-models/Yolo-V8/Segment-Dog-with-New-background/dog.mp4"
model_path = "C:/Data-sets/Dog segmentation YoloV8/My-model2/weights/best.pt"
background_image_path = "Best-Semantic-Segmentation-models/Yolo-V8/Segment-Dog-with-New-background/new_background.jpg"

background_image = cv2.imread(background_image_path)

# open the video file 
cap = cv2.VideoCapture(video_path)

# check if the video was opened successfuly
if not cap.isOpened():
    print("Error open video file.")
    exit()

# read and disply the frames until the video ends or the user presses 'q' key

while True:
    ret , frame = cap.read()

    scale_precent = 30
    width = int(frame.shape[1] * scale_precent / 100)
    height = int(frame.shape[0] * scale_precent / 100) 
    dim= (width, height)

    # resize image
    frame = cv2.resize(frame , dim , interpolation = cv2.INTER_AREA)

    H , W , _ = frame.shape

    # load our model 
    model = YOLO(model_path)

    # predict each frame
    results = model(frame)
    result = results[0]

    # get classe names
    names = model.names

    # create variable to keep track of the best mask and its confidence

    best_mask = None
    best_confidence = 0.5 

    predicted_classes = result.boxes.cls.cpu().numpy()
    confidence_values = result.boxes.conf.cpu().numpy()


    # run through the masks :
    for j , mask in enumerate(result.masks.data):

        mask = mask.cpu().numpy() * 255
        classId = int(predicted_classes[j])
        confidence = confidence_values[j]

        mask = cv2.resize(mask , (W,H))

        # update the best mask if the current (inside the loop) mask has higher confidence 
        if confidence > best_confidence :
            best_mask = mask
            best_confidence = confidence

    # Ensure the mask is binary 
    _ , threshold_mask = cv2.threshold(best_mask , 128 , 255 , cv2.THRESH_BINARY)

    # convert the mask to the appropiate data type
    threshold_mask = threshold_mask.astype(np.uint8)

    # extract the region of interest (ROI) from the original image using the mask
    result = cv2.bitwise_and(frame , frame , mask = threshold_mask)


    # merge the background image with the extract image

    # find the center coordinates of the large image
    center_y , center_x = background_image.shape[0] //2 , background_image.shape[1] //2

    # calculate the starting and ending ccordinates for the small image in the large image
    start_y , start_x = center_y - result.shape[0] //2 , center_x - result.shape[1] //2 
    end_y , end_x = start_y + result.shape[0] , start_x +result.shape[1]

    # create a boolean mask for pixels in the resized small image that are not equal to [0,0,0]
    non_black_pixels = (result != [0,0,0]).any(axis=2)

    # Replace non-black pixels in the large image with corresponding pixels from the resized small image
    final = background_image.copy()
    final[start_y:end_y , start_x:end_x][non_black_pixels]= result[non_black_pixels]

    scale_precent= 60
    width = int(background_image.shape[1] * scale_precent / 100)
    height = int(background_image.shape[0] * scale_precent / 100)
    dim = (width, height)

    # resize image
    final = cv2.resize(final , dim , interpolation = cv2.INTER_AREA)

    # break the loop if the video ends
    if not ret:
        print("Video playback completed.")
        break

    # display the frames 

    cv2.imshow("original", frame)
    cv2.imshow("result", result)
    cv2.imshow("Final", final )

    if cv2.waitKey(25) & 0xFF == ord('q') :
        break 




# release the windows
cap.release()
cv2.destroyAllWindows()























