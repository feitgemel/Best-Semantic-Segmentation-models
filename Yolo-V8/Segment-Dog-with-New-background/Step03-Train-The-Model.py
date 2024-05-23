from ultralytics import YOLO 

def main():
    model = YOLO("yolov8m-seg.pt")
    project = "C:/Data-sets/Dog segmentation YoloV8"
    experiment = "My-model"
    batch_size = 16 # about 8 giga gpu card

    results = model.train(data = "Best-Semantic-Segmentation-models/Yolo-V8/Segment-Dog-with-New-background/config.yaml",
                          epochs = 100,
                          project = project,
                          name = experiment, 
                          batch = batch_size, 
                          device = 0 ,
                          imgsz = 640 ,
                          patience = 40 ,
                          verbose = True , 
                          val = True )            
                          
if __name__ == "__main__":
    main()                          
                          
                          
                          
                          
