import torch 
from torch import nn, optim 
from torch.utils.data import DataLoader, random_split 
from tqdm import tqdm
import os 

from unet import UNet
from MyTrainDatasetClass import CarvanaTrainDataset 

if __name__ == '__main__':

    # Hyperparameters
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 8
    EPOCHS = 10
    DATA_PATH = "D:/Data-Sets-Object-Segmentation/Carvana Image Masking Challenge"
    MODEL_SAVE_PATH = "D:/temp/models/CarvanaCarSegmentation/Car-unet.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the train dataset object
    train_dataset = CarvanaTrainDataset(DATA_PATH)

    generator = torch.Generator().manual_seed(42)

    # Split the dataset into train and validation sets
    train_dataset , val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,)
    
    val_dataloader = DataLoader(dataset=val_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,)
    
    model = UNet(in_channels=3, num_classes=1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with logits

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0.0

        for idx , img_mask in enumerate(tqdm(train_dataloader)):

            # 0 index is image and 1 index is mask

            img = img_mask[0].float().to(DEVICE)
            mask = img_mask[1].float().to(DEVICE)

            y_pred = model(img)
            optimizer.zero_grad()

            # Calculate the loss
            loss = criterion(y_pred, mask)
            #update the training loss
            train_running_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)

        # check the model weights after each epoch with the validation set

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            # same proecess as above but with the validation set
            for idx , img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(DEVICE)
                mask = img_mask[1].float().to(DEVICE)

                y_pred = model(img)
                # Calculate the loss
                loss = criterion(y_pred, mask)

                #update the training loss
                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

        print("-"*30)
        print(f"Train loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Validation loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*30)

    # Create a folder to save the model if it does not exist
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"Model saved to {MODEL_SAVE_PATH}")

print("Training complete.")









