import torch 
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from unet import UNet
from MyTestDatasetClass import CarvanaTestDataset

# We will generate 2 functios :  
# Pred_show_image_grid : This function will predict multiple images and show (Original image , Original mask and predicted mask)
# Single_image_inference : This function will predict a single image and show the original image and the predicted mask

def single_image_inference(image_path , model_pth, device):

    # Create the model 
    model = UNet(in_channels=3, num_classes=1).to(device)

    # load the model weights
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    # Convert the image to tensor and resize it to 512x512
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    img = transform(Image.open(image_path)).float().to(device)

    # add a batch of 1 dimension to the image
    img = img.unsqueeze(0)

    # Predict the mask by sending the image to the model
    pred_mask = model(img)

    # Convert the image for display 
    img = img.squeeze(0).cpu().detach() # remove the batch dimension
    
    # Switch the channal to the last for plot the image 
    img = img.permute(1,2,0) 

    # Do the same for the mask
    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1,2,0)

    # Convert the mask to binary and all gray will be converted to 0 and all white wiil be converted to 1
    pred_mask[pred_mask < 0] = 0 
    pred_mask[pred_mask > 0] = 1

    fig = plt.figure() 
    for i in range(1,3):
        fig.add_subplot(1, 2, i)
        if i == 1:
            plt.imshow(img , cmap='gray')
        else:
            plt.imshow(pred_mask , cmap='gray')
    plt.show()


if __name__ == "__main__": 
    single_img_path = "Best-Semantic-Segmentation-models/U-Net/Car Segmentation - U-NET Image Segmentation using Pytorch/test_Img.jpg" 
    model_pth = "D:/temp/models/CarvanaCarSegmentation/Car-unet.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    single_image_inference(single_img_path , model_pth, device)