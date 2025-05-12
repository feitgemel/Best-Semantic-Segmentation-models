import torch 
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from unet import UNet
from MyTestDatasetClass import CarvanaTestDataset

# Copy some images to a test folder and tun the following code to test the model on multiple images

def pred_show_image_grid(data_path, model_pth, device): 
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))


    # load the test dataset 
    image_dataset = CarvanaTestDataset(data_path)
    images = [] 
    pred_masks = []

    for img in image_dataset :
        img = img.float().to(device)
        img = img.unsqueeze(0)

        pred_mask = model(img)

        img = img.squeeze(0).cpu().detach() # remove the batch dimension
        img = img.permute(1,2,0)

        pred_mask = pred_mask.squeeze(0).cpu().detach()
        pred_mask = pred_mask.permute(1,2,0)

        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 0] = 1


        images.append(img)
        pred_masks.append(pred_mask)


    # Plot the results

    fig , axes = plt.subplots(len(images), 2, figsize=(10, 5 * len(images)))

    for i in range(len(images)):

        # Plot original image
        axes[i, 0].imshow(images[i].numpy())
        axes[i,0].axis('off')

        # Plot predicted mask
        axes[i, 1].imshow(pred_masks[i].numpy(), cmap='gray')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    data_path = "D:/Data-Sets-Object-Segmentation/Carvana Image Masking Challenge" 
    model_path = "D:/temp/models/CarvanaCarSegmentation/Car-unet.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run the multiple images prediction function
    pred_show_image_grid(data_path, model_path, device)

    



