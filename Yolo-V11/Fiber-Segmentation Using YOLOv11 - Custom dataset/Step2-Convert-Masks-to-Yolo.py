import os
import cv2
from tqdm import tqdm

def convert_masks_to_labels(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for j in tqdm(os.listdir(input_dir)):
        image_path = os.path.join(input_dir, j)
        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        H, W = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) >= 2:
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append(x / W)
                    polygon.append(y / H)
                polygons.append(polygon)

        label_path = os.path.join(output_dir, j.replace('.png', '.txt'))  # Adjust file extension if needed
        with open(label_path, 'w') as f:
            for polygon in polygons:
                for p_, p in enumerate(polygon):
                    if p_ == len(polygon) - 1:
                        f.write('{}\n'.format(p))
                    elif p_ == 0:
                        f.write('0 {} '.format(p))
                    else:
                        f.write('{} '.format(p))

# Convert the train data
print("Convert the train data...")

convert_masks_to_labels(
    input_dir='D:/Data-Sets-Object-Segmentation/Fiber Segmentation/ALL_DATA/train/masks',
    output_dir='D:/Data-Sets-Object-Segmentation/Fiber Segmentation/ALL_DATA/train/labels'
)

# Convert the validation data
print("Convert the validation data...")

convert_masks_to_labels(
    input_dir='D:/Data-Sets-Object-Segmentation/Fiber Segmentation/ALL_DATA/validation/masks',
    output_dir='D:/Data-Sets-Object-Segmentation/Fiber Segmentation/ALL_DATA/validation/labels'
)

# Convert the test data
print("Convert the test data...")

convert_masks_to_labels(
    input_dir='D:/Data-Sets-Object-Segmentation/Fiber Segmentation/ALL_DATA/test/masks',
    output_dir='D:/Data-Sets-Object-Segmentation/Fiber Segmentation/ALL_DATA/test/labels'
)