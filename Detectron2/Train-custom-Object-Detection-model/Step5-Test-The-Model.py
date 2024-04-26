from detectron2.engine import DefaultPredictor 
import os 
import pickle 

cfg_saved_path = "My-Train-Detectron2/IS_cfg.pickle" # IS -> Instance segmentation
with open (cfg_saved_path , 'rb') as f :
    cfg = pickle.load(f) # get the configuration file 


output_dir = "My-Train-Detectron2" 

cfg.MODEL.WEIGHTS = os.path.join(output_dir , "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # over 50% , which object should display
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)

CLASSES = ["Apple", "Strawberry", "Ornage" , "Grapes" , "Banana", "Lemon"]

#path of the test images 

image_path_1 = "Train-custom-Object-Detection-model/Fruits_for_detectron2/Test/apples-vs-bananas.jpg"
image_path_2 = "Train-custom-Object-Detection-model/Fruits_for_detectron2/Test/pexels-pixabay-70746.jpg"

# show the predicitions on the test image

import cv2 
import numpy as np 
from detectron2.utils.visualizer import Visualizer 

im = cv2.imread(image_path_1)
outputs = predictor(im)

print("=========================================")
print(outputs)
print("=========================================")

pred_classes = outputs['instances'].pred_classes.cpu() 
print("Pred Classes : ")
print(pred_classes)

pred_classes = pred_classes.numpy()

flag = np.size(pred_classes)
print("Flag :")
print(flag)

if flag > 0 :
    pred_classes = pred_classes[0] # grab the first elemnet 
    print("pred_classes:")
    print(pred_classes)

    print(CLASSES[pred_classes])

    img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    v = Visualizer(img_rgb, metadata={}, scale=0.6) # init the visualizer
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img_bgr = cv2.cvtColor(v.get_image(), cv2.COLOR_RGB2BGR)

    cv2.imshow("v", img_bgr)
    cv2.waitKey(0)
else:
    print("Pred_classes is empty")

cv2.destroyAllWindows()
