from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x

def build_effienet_unet(input_shape): 

    # Input layer
    inputs = Input(shape=input_shape)

    # Pre trained Encoder  
    encoder = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)

    s1 = encoder.get_layer('input_layer').output ## 256 
    s2 = encoder.get_layer('block2a_expand_activation').output ## 128
    s3 = encoder.get_layer('block3a_expand_activation').output ## 64
    s4 = encoder.get_layer('block4a_expand_activation').output ## 32

    # Bottleneck
    b1 = encoder.get_layer('block5a_expand_activation').output ## 16

    # Decoder
    d1 = decoder_block(b1, s4, 512) ## 32
    d2 = decoder_block(d1, s3, 256) ## 64
    d3 = decoder_block(d2, s2, 128) ## 128
    d4 = decoder_block(d3, s1, 64) ## 256

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(d4)

    model = Model(inputs , outputs , name = "EfficientNetB0_Unet")
    return model



# Main function to test the model
if __name__ == "__main__":
    input_shape = (256, 256, 3)  # Example input shape
    model = build_effienet_unet(input_shape)
    model.summary()
    

