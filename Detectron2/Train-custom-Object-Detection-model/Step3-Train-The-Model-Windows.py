from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor , DefaultTrainer
import os 
import pickle


# go to the model link :
# https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

output_dir = "My-Train-Detectron2"
num_classes= 6
device = "cpu"

# Train
train_dataset_name = "LP_train"
train_images_path = r"Train-custom-Object-Detection-model/Fruits_for_detectron2/Train"
train_json_annot_path = r"Train-custom-Object-Detection-model/Fruits_for_detectron2/Train/labels_my-project-name_2023-12-04-07-26-09.json"

# Validate
val_dataset_name = "LP_Test"
val_images_path = r"Train-custom-Object-Detection-model/Fruits_for_detectron2/Validate"
val_json_annot_path = r"Train-custom-Object-Detection-model/Fruits_for_detectron2/Validate/labels_my-project-name_2023-12-04-07-39-25.json"

# Register the dataset

#Register the train:
register_coco_instances(name = train_dataset_name, metadata={}, json_file=train_json_annot_path, image_root=train_images_path)

#Register the Validation:
register_coco_instances(name = val_dataset_name, metadata={}, json_file=val_json_annot_path, image_root=val_images_path)


from detectron2.config import get_cfg 
from detectron2 import model_zoo


def get_train_cfg (a_config_file_path , a_model_name , a_train_dataset_name , a_test_dataset_name , a_num_classes, device , output_dir):

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(a_config_file_path))

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(a_model_name)
    cfg.DATASETS.TRAIN = (a_train_dataset_name, )
    cfg.DATASETS.TEST = (a_test_dataset_name, )

    # number of workers (How to stress the CPU)
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 2 # how many images per batch
    cfg.SOLVER.BASE_LR = 0.00025 # learning rate
    cfg.SOLVER.MAX_ITER = 1000 # how many iterations (like epochs)
    cfg.SOLVER.STEPS = [] # this will reduce the learning rate during the training 

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = a_num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir

    return cfg 



# main 

def main() :

    os.makedirs(output_dir, exist_ok=True)
    cfg = get_train_cfg(config_file_path, model_name, train_dataset_name, val_dataset_name, num_classes , device, output_dir)

    cfg_save_path = "My-Train-Detectron2/IS_cfg.pickle" # IS -> Instance segmentation

    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL) # save the cfg 

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__' :
    main()








