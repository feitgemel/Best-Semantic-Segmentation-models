import numpy as np
from sklearn.model_selection import train_test_split
import cv2


# Load the numpy array from the file
print("Start loading the train data ...........")
allImageNP = np.load('/mnt/d/temp/Dust-Storm-Images.npy')
maskImagesNP = np.load('/mnt/d/temp/Dust-Storm-Masks.npy')

print(allImageNP.shape)
print(maskImagesNP.shape)

# Split the data into training and validation sets
allImageNP, allValidateImageNP, maskImagesNP, maskValidateImages = train_test_split(
        allImageNP, maskImagesNP, test_size=0.2, random_state=42)

print("Training data shape:", allImageNP.shape, maskImagesNP.shape)
print("Validation data shape:", allValidateImageNP.shape, maskValidateImages.shape)


Height = 128 # Reduce if there are memory errors messages
Width = 128 # Reduce if there are memory errors messages

# build the model 
import tensorflow as tf
from vgg16_unet import build_vgg16_unet
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

shape = (Height, Width, 3)

lr = 1e-4 # 0.0001
batch_size = 4 # Increase the value if your have more than 12Giga GPU card 
epochs = 200


model = build_vgg16_unet(shape)
print(model.summary())
opt = tf.keras.optimizers.Adam(lr)
model.compile(loss="binary_crossentropy", optimizer=opt,  metrics=['accuracy'])

stepsPerEpoch = int(np.ceil(len(allImageNP) / batch_size))
validationSteps = int(np.ceil(len(allValidateImageNP) / batch_size))

best_model_file = "/mnt/d/temp/models/Dust-Storm/VGG16-Dust-Storm.keras"

callbacks = [
        ModelCheckpoint(best_model_file, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=20, verbose=1) ]

history = model.fit(allImageNP,maskImagesNP,
                    batch_size= batch_size,
                    epochs= epochs,
                    verbose=1,
                    validation_data=(allValidateImageNP, maskValidateImages),
                    validation_steps = validationSteps,
                    steps_per_epoch = stepsPerEpoch,
                    shuffle=True,
                    callbacks=callbacks) 


# show the results of the train
import matplotlib.pyplot as plt

acc= history.history['accuracy']
val_acc =  history.history['val_accuracy'] 
loss = history.history['loss'] 
val_loss = history.history['val_loss'] 

epochs = range(len(acc))

# training and validation chart
plt.plot(epochs, acc, 'r' , label="Trainig accuracy")
plt.plot(epochs, val_acc, 'b' , label="Validation accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Training and validation accuracy")
plt.legend(loc='lower right')
plt.show()

#loss and validation loss chart :

plt.plot(epochs, loss, 'r' , label="Trainig loss")
plt.plot(epochs, val_loss, 'b' , label="Validation loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Training and validation loss")
plt.legend(loc='upper right')
plt.show()





