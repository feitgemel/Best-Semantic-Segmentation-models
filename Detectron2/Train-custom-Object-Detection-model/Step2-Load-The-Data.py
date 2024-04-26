import detectron2
import cv2 
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances 

# register datasets 


# Train dataset
register_coco_instances("my_dataset_train", {}, "Train-custom-Object-Detection-model/Fruits_for_detectron2/Train/labels_my-project-name_2023-12-04-07-26-09.json",
                        "Train-custom-Object-Detection-model/Fruits_for_detectron2/Train")


# Validate dataset
register_coco_instances("my_dataset_val", {}, "Train-custom-Object-Detection-model/Fruits_for_detectron2/Validate/labels_my-project-name_2023-12-04-07-39-25.json",
                        "Train-custom-Object-Detection-model/Fruits_for_detectron2/Validate")

# extract the Metadat and the data dictionaries for both : Train and Validation datasets.

train_metedata = MetadataCatalog.get("my_dataset_train")
train_datasets_dicts = DatasetCatalog.get("my_dataset_train")

val_metedata = MetadataCatalog.get("my_dataset_val")
val_datasets_dicts = DatasetCatalog.get("my_dataset_val")


# lets show the first image :

first_dict = train_datasets_dicts[0]
print(first_dict)

file_name = first_dict['file_name']
height = first_dict['height']
width = first_dict['width']
image_id = first_dict['image_id']
annotations = first_dict['annotations']

img = cv2.imread(file_name)
visual = Visualizer(img[:, :, ::-1] , metadata=train_metedata, scale = 0.5)
vis = visual.draw_dataset_dict(first_dict)

img2 = vis.get_image()
img_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

cv2.imshow("img_rgb", img_rgb)
cv2.waitKey(0)


