from ultralytics import YOLO
from PIL import Image , ImageDraw 

model = YOLO("yolo11m-seg.pt")  # Medium size 

img_path = "Best-Semantic-Segmentation-models/Yolo-V11/How to segment different objects in image python/test_img.jpg" 
#img_path = "Best-Semantic-Segmentation-models/Yolo-V11/How to segment different objects in image python/test_img2.jpg" 


img = Image.open(img_path)
img.show()

# Segmenation
results = model.predict(img)
result = results[0]
masks = result.masks 

print("Masks:")
print(len(masks))  # Number of masks detected

# Extract the first mask and polygon 
mask1 = masks[0]
mask = mask1.data[0].cpu().numpy()  # Convert to numpy array
polygon = mask1.xy[0]

mask_img = Image.fromarray(mask,"I")
mask_img.show()

# Draw a polygon of the mask1 on the image 
draw = ImageDraw.Draw(img)
draw.polygon(polygon, outline="#00FF00", width=6) # Green color
img.show()

# Extract the second mask and polygon
mask2 = masks[1]
mask = mask2.data[0].cpu().numpy()  # Convert to numpy array
polygon = mask2.xy[0]

mask_img = Image.fromarray(mask,"I")
mask_img.show()

draw.polygon(polygon, outline="#ff9100", width=6) # Orange color
img.show()

# Save the final image
img.save("Best-Semantic-Segmentation-models/Yolo-V11/How to segment different objects in image python/segmented_image.jpg")




