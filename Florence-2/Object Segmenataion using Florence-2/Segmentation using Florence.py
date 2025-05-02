import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
import cv2
import torch

# Choose a model 
model_id = "microsoft/florence-2-large"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Init the model 
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
model.to(device) # Move the model to the device (GPU or CPU)
model.eval()

# Init the processor
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def run_florence2(task_prompt , images , text_input=None):

    if text_input is None: # If no text was send to the function 
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input


    # Run the processor and get the tensors 
    inputs = processor(text=prompt, images=image, return_tensors="pt") # The output will be tensors
    inputs.to(device) # Move the tensors to the device (GPU or CPU)

    # Generate the output text from the tensors (Inside the inputs)

    generated_ids = model.generate( 
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens = 1024,
        early_stopping=False ,
        do_sample=False,
        num_beams=3,
    )

    # Get the data in readable format
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height))
    
    return parsed_answer

# Load the image
image = Image.open("Best-Semantic-Segmentation-models/Florence-2/Object Segmenataion using Florence-2/Parrot.jpg")


task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>" # The task prompt for the model

results = run_florence2(task_prompt, image, text_input="a parrot") # Run the model with the image and the task prompt

data = results["<REFERRING_EXPRESSION_SEGMENTATION>"]
print(data)


# Display the image with the segmentation mask

# Convert to Opencv format
open_cv_image = np.array(image)
open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

# Line thickness
thickness = 2

#Put the result text on the image
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
fill_mask = True 


# Plot the result 

for polygons, label in zip(data['polygons'], data['labels']):

    for _polygon in polygons:

        _polygon = np.array(_polygon).reshape(-1,2).astype(int)
        if len(_polygon) < 3:
            print("Invlid polygon", _polygon)
            continue

        _polygon = (_polygon).astype(int)

        if fill_mask: # Should be True for fill and False for only outline
            #Draw filled polygon
            cv2.fillPoly(open_cv_image, [np.array(_polygon)] ,color=(0,255,255))

        # Draw the outline of the polygon
        cv2.polylines(open_cv_image, [np.array(_polygon)] ,color=(255,255,0), thickness=thickness, isClosed=True)


        # Draw the label text near the polygon
        text_postion_x = _polygon[0][0] + 8 
        text_postion_y = _polygon[0][1] + 2
        text_position = (text_postion_x, text_postion_y)

        cv2.putText(open_cv_image, label, text_position, font, font_scale,color= (0, 0, 0), thickness=thickness) 

# Display the result 

scale_precent = 30 # The scale percent to resize the image
width = int(open_cv_image.shape[1] * scale_precent / 100)
height = int(open_cv_image.shape[0] * scale_precent / 100)
dim = (width, height)

# Resize the image
open_cv_image = cv2.resize(open_cv_image, dim, interpolation = cv2.INTER_AREA)

cv2.imwrite("Best-Semantic-Segmentation-models/Florence-2/Object Segmenataion using Florence-2/Parrot_with_mask.jpg",open_cv_image)

cv2.imshow("Image", open_cv_image) # Display the image
cv2.waitKey(0) # Wait for a key press to close the image
cv2.destroyAllWindows()





    
