from ultralytics import YOLO

def main(): 

    model = YOLO("yolo11s-seg.pt")  # Load a pretrained YOLOv11 segmentation model
    project = "d:/temp/models/Crack-Segmentation-Using-YOLOv11"  # Define the project directory
    experiment = "My-model-S"
    batch_size = 8 


    resutls = model.train(data = "Best-Semantic-Segmentation-models/Yolo-V11/Crack Segmentation Using YOLOv11 - Custom dataset/config.yaml",
                          epochs = 50,  # Set the number of training epochs
                          project = project,  # Specify the project directory
                          name = experiment,  # Name of the experiment
                          batch = batch_size,  # Set the batch size
                          imgsz = 416,  # Set the image size for training
                          device = "0",  # Specify the device to use (0 for GPU)
                          patience = 5,  # Set the patience for early stopping   
                          verbose = True,  # Enable verbose output
                          val = True,)
    
if __name__ == "__main__" :  # Enable validation during training            
        main()          