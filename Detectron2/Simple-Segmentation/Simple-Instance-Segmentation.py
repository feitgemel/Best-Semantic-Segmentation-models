import torch
import detectron2

import numpy as np 
import os 
import json 
import cv2 
import random 

# import common detectron2 utilities 
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg 
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

imagePath = "pexels-brett-sayles-1115171.jpg"
myImage = cv2.imread(imagePath)
scale_precent = 30
width = int(myImage.shape[1] * scale_precent / 100)
height = int(myImage.shape[0] * scale_precent / 100)
dim = (width, height)

#resize image
myImage = cv2.resize(myImage , dim , interpolation=cv2.INTER_AREA)



# Segmentations : You can find the models here :
#https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

#run inference 
cfg_inst = get_cfg()
cfg_inst.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg_inst.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg_inst.MODEL.DEVICE = "cpu" # -> if you have Cuda on a Linux machine you dont need this line

predictor = DefaultPredictor(cfg_inst)
outputs = predictor(myImage)

v = Visualizer(myImage[:, :, ::-1] ,MetadataCatalog.get(cfg_inst.DATASETS.TRAIN[0]) , scale=1.0)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

img = out.get_image()[:, :, ::-1]

cv2.imshow("img", myImage)
cv2.imshow("predict", img)

cv2.imwrite("e:/temp/segmented.png", img)

cv2.waitKey(0)