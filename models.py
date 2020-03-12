import loss
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inputs)
    conv1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool3)
    conv4 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool4)
    conv5 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge6)
    conv6 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv6)

    up7 = Conv2D(256, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv7)

    up8 = Conv2D(128, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge8)
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv8)

    up9 = Conv2D(64, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge9)
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)
    conv9 = Conv2D(2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)
    conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr = 1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    # model.compile(optimizer=Adam(lr = 1e-4), loss=loss.dice_coef_loss, metrics=["accuracy"])
    # model.compile(optimizer=Adam(lr = 1e-4), loss=loss.binary_focal_loss(), metrics=["accuracy"])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def segnet(pretrained_weights=None, num_classes=2, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    #Encoder
    # 224x224
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1, mask1 =  MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv1)
    # # 112x112
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2, mask2 =  MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv2)
    # 56x56
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3, mask3 =  MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv3)
    # 28x28
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4, mask4 =  MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv4)
    # 14x14

    # decode
    up7 = MaxUnpooling2D()([pool4, mask4])
    #up7 = concatenate([up7, conv4], axis=-1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    #up8 = concatenate([up8, conv3], axis=-1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = MaxUnpooling2D()([conv8, mask3])
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    #up9 = concatenate([up9, conv2], axis=-1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = MaxUnpooling2D()([conv9, mask2])
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    #up10 = concatenate([up10, conv1], axis=-1)
    conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = MaxUnpooling2D()([conv10, mask1])
    conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv10)
    conv10 = BatchNormalization()(conv10)

    # outputHeight = Model(inputs, conv10).output_shape[1]
    # outputWidth = Model(inputs, conv10).output_shape[2]
    # conv11 = Conv2D(num_classes, (1, 1), padding='same')(conv10)
    # conv11 = (Reshape((outputHeight*outputWidth, num_classes)))(conv11)
    # conv11 = Activation('softmax')(conv11)
    conv12 = Conv2D(2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv10)
    conv13 = Conv2D(1, 1, activation="sigmoid")(conv12)

    model = Model(inputs=inputs, outputs=conv13)
    # model.outputWidth = outputWidth
    # model.outputHeight = outputHeight

    model.compile(optimizer=Adam(lr = 1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    # model.compile(optimizer=Adam(lr = 1e-4), loss=loss.dice_coef_loss, metrics=["accuracy"])
    # model.compile(optimizer=Adam(lr = 1e-4), loss=loss.binary_focal_loss(), metrics=["accuracy"])

    if pretrained_weights:
        model.load_weights(pretrained_weights)
	
    return model