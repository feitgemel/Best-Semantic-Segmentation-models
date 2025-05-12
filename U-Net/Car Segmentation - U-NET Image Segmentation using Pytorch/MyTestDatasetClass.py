import os 
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

IMG_SIZE = 256 

class CarvanaTestDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path 

        self.images = sorted([root_path+"/test/" + i for i in os.listdir(root_path + "/test/")])
        #self.masks = sorted([root_path+"/train_masks/" + i for i in os.listdir(root_path + "/train_masks/")])

        # Resize image and mask to IMG_SIZE and transform to tensor
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])

    def __getitem__(self , index): 
        img = Image.open(self.images[index]).convert("RGB")
        #mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img)
    

    def __len__(self):
        return len(self.images)
    
    
