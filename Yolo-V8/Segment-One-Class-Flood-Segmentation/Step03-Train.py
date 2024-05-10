from ultralytics import YOLO

def main():
    model = YOLO('yolov8l-seg.pt') # load the YoloV8 large model !!
    project = "c:/Data-sets/Flood Area Segmentation"
    experiment = "My-Flood-Model"
    batch_size = 16 

    config_file = "Best-Semantic-Segmentation-models/Yolo-V8/Segment-One-Class-Flood-Segmentation/config.yaml" 

    results = model.train(data = config_file,
                          epochs=100,
                          project=project,
                          name=experiment,
                          batch=batch_size, 
                          device=0,
                          imgsz=640,
                          patience=30,
                          verbose=True,
                          val=True)
    
if __name__ == "__main__":
    main()