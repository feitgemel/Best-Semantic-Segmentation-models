import torch
import detectron2

import numpy as np 
import cv2 

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

imagePath = "pexels-brett-sayles-1115171.jpg"
myNewImage = cv2.imread(imagePath)
scale_precent = 30 
width = int(myNewImage.shape[1] * scale_precent / 100)
height = int(myNewImage.shape[0] * scale_precent / 100)
dim = (width, height)

# resize image
myNewImage = cv2.resize(myNewImage, dim , interpolation=cv2.INTER_AREA)

# Panoptic Segmentation = Instance Segmentation + Semnatic Segmentation 

cfg_pan = get_cfg()
cfg_pan.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg_pan.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
cfg_pan.MODEL.DEVICE = "cpu" # if you have Cuda , dont need this line

predictor = DefaultPredictor(cfg_pan)
panoptic_seg , segments_info = predictor(myNewImage)["panoptic_seg"]
v = Visualizer(myNewImage[:, :, ::-1],MetadataCatalog.get(cfg_pan.DATASETS.TRAIN[0]), scale=1.0 )
out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
img = out.get_image()[:, :, ::-1]


cv2.imshow("img", myNewImage)
cv2.imshow("predict", img)
cv2.imwrite("e:/temp/panoptic.png", img)

cv2.waitKey(0)