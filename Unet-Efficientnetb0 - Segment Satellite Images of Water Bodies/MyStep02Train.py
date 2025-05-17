import numpy as np
from sklearn.model_selection import train_test_split
import cv2


# load the numpy arrays from the disk 

print("Load the numpy arrays from the disk")
allImages = np.load('/mnt/d/temp/Water Bodies-Images.npy')
MaskImages = np.load('/mnt/d/temp/Water Bodies-Masks.npy')

print(allImages.shape)
print(MaskImages.shape)
print("Finished loading the numpy arrays from the disk")
print("------------------------------------------------------")


# Split the data into training and validation sets
allImageNP , allValidateImageNP , MaskImageNP , MaskValidateImageNP = train_test_split(allImages, MaskImages, test_size=0.2, random_state=42)

print("Train data shape: ", allImageNP.shape, MaskImageNP.shape)
print("Validation data shape: ", allValidateImageNP.shape, MaskValidateImageNP.shape)
print("------------------------------------------------------")

Height = 128
Width = 128

# Build the model
import tensorflow as tf
from EfficientNetB0_Unet import build_effienet_unet
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

shape = (Height, Width, 3)

lr = 1e-4
batch_size = 4 # Increae the value if you have more GPU memory
epochs = 200 

model = build_effienet_unet(shape)
print(model.summary())

opt = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

stepsPerEpoch = int(np.ceil(len(allImageNP) / batch_size))
validationSteps = int(np.ceil(len(allValidateImageNP) / batch_size))

best_model_file = "/mnt/d/temp/models/efficientnetb0_unet_Water_bodies.keras"

callbacks = [
    ModelCheckpoint(best_model_file, verbose=1,  save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6),
    EarlyStopping(monitor='val_loss', patience=20, verbose=1)
]

history = model.fit(allImageNP, MaskImageNP,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(allValidateImageNP, MaskValidateImageNP),
                    steps_per_epoch=stepsPerEpoch,
                    validation_steps=validationSteps,
                    callbacks=callbacks,
                    shuffle=True )

print("Finished training the model")

#Show the results
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

# Training and validation accuracy chart
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')
plt.show()

# Training and validation loss chart
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc = 'upper right')
plt.show()
print("Finished showing the results")



