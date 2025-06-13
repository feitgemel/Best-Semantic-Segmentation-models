from ultralytics import YOLO


def main():
    
    model = YOLO('yolo11n-seg.pt')  # Load a pretrained YOLOv11 model
    project = "d:/temp/models/Fiber-Segmentation"
    experiment = "My-Model-n"
    batch_size = 8 

    data_config_file_path = "Best-Semantic-Segmentation-models/Yolo-V11/Fiber-Segmentation Using YOLOv11 - Custom dataset/config.yaml" 

    results = model.train(
        data = data_config_file_path,  # Path to the data configuration file
        epochs=100,  # Number of training epochs
        imgsz=640,  # Image size for training
        batch=batch_size,  # Batch size for training
        project=project,  # Project directory to save results
        name=experiment,  # Experiment name for saving results
        device='0',  # Specify the device to use (e.g., '0' for GPU 0)
        verbose=True,  # Print detailed training logs
        val=True,
        patience=5)   # Validation during training with early stopping patience
    
if __name__ == "__main__":
    main()  

    