from ultralytics import YOLO

def main():
    model = YOLO('yolov8l-seg.pt') 
    project = "C:/Data-sets/Football-Seg.v1i.yolov8"
    experiment = "My-model"
    batchSize = 16 

    results = model.train(data = "Best-Semantic-Segmentation-models/Yolo-V8/Segment-Multi-Class-Football-Segmentation/config.yaml" ,
                          epochs=200,
                          project = project,
                          name= experiment, 
                          batch = batchSize ,
                          device = 0 , 
                          imgsz = 640 , 
                          patience = 30 ,
                          verbose=True , 
                          val = True)
                          
                          
                          
if __name__ == '__main__':
    main()
    
                              
                          
                          