#WSL - Lynux
# create WSL Ubunto enviroment

# open WSL in c:
#conda create -n detectorn99 python=3.9
#conda activate detectorn99

#install Pytorch
#conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia


#cd detectron2 # inside the git clone folder
#cd Train Folder 

#python setup.py build develop

# run the python train code:
#python Step4-Train-The-Model-Ubunto-WSL.py





# register the dataset

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os
import pickle


# go to this link :
# https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
# you can see many models in the table : "Instance segmentation"

# object detection 

config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

output_dir = r"My-Train-DetectRon2-Ubuntu"
num_classes = 6

device = "cuda"

#train
train_dataset_name = "LP_train"
#train_images_path = r"/mnt/c/Python-Code/Best-Object-Detection-models/Detectron2/Fruits_for_detectron2/Train"
#train_json_annot_path = r"/mnt/c/Python-Code/Best-Object-Detection-models/Detectron2/Fruits_for_detectron2/Train/labels_my-project-name_2023-12-04-07-26-09.json"
train_images_path = r"Fruits_for_detectron2/Train"
train_json_annot_path = r"Fruits_for_detectron2/Train/labels_my-project-name_2023-12-04-07-26-09.json"


#Validation
val_dataset_name = "LP_test"
val_images_path = r"Fruits_for_detectron2/Validate"
val_json_annot_path = r"Fruits_for_detectron2/Validate/labels_my-project-name_2023-12-04-07-39-25.json"


# register the dataset:
# =====================

# register the train
register_coco_instances(name = train_dataset_name, metadata={}, json_file=train_json_annot_path, image_root=train_images_path)

# register the test
register_coco_instances(name = val_dataset_name, metadata={}, json_file=val_json_annot_path, image_root=val_images_path)


from detectron2.config import get_cfg
from detectron2 import model_zoo

def get_train_cfg(a_config_file_path , a_model_name , a_train_dataset_name, a_test_dataset_name, a_num_classes, device, output_dir) :
    cfg = get_cfg() 

    cfg.merge_from_file(model_zoo.get_config_file(a_config_file_path))  # get the "config_file_path"
    
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(a_model_name)
    #model_zoo.get_checkpoint_url(a_model_name)
    cfg.DATASETS.TRAIN = (a_train_dataset_name, )
    cfg.DATASETS.TEST = (a_test_dataset_name, )

    # number of workers -> how to stress the CPU
    cfg.DATALOADER.NUM_WORKERS = 2 
    
    cfg.SOLVER.IMS_PER_BATCH = 2 # how many images per batch
    cfg.SOLVER.BASE_LR = 0.00025 # learning rate
    cfg.SOLVER.MAX_ITER = 3500 # how many iterations (like epohcs)  !!!!!!!!!!!!
    cfg.SOLVER.STEPS = [] # this will not reduce the learning rate during training

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = a_num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir

    return cfg



def main():
    
    os.makedirs(output_dir, exist_ok=True)
    
    cfg = get_train_cfg(config_file_path, model_name, train_dataset_name, val_dataset_name, num_classes, device, output_dir)

    
    cfg_save_path = "My-Train-DetectRon2-Ubuntu/IS_cfg-Ubunto.pickle" #IS -> Instance segmentation

    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg , f, protocol=pickle.HIGHEST_PROTOCOL) # this will save our cfg


    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # After training, add a testing step
    #trainer.test(ckpt=None)  # Use the last checkpoint by default 


if __name__ == '__main__' :
    main()

# After train :
#Copy the My-Train-DetectRon2 folder to the Windows enviroment to continute to the test new image





