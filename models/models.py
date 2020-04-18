from keras.models import *
from keras.layers import *
from models.layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


def unet2D(classes=2, input_size=(256, 256, 1)):
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

    up6 = UpSampling2D(size=(2, 2))(drop5)
    up6 = Conv2D(512, 2, activation="relu", padding="same", kernel_initializer="he_normal")(up6)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge6)
    conv6 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(256, 2, activation="relu", padding="same", kernel_initializer="he_normal")(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(128, 2, activation="relu", padding="same", kernel_initializer="he_normal")(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge8)
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(64, 2, activation="relu", padding="same", kernel_initializer="he_normal")(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge9)
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)
    conv9 = Conv2D(classes, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)
    conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

    return Model(inputs=inputs, outputs=conv10)


def unet3D(classes=2, input_size=(9, 256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv3D(4, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inputs)
    conv1 = Conv3D(4, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2))(conv1)
    conv1 = Reshape((256, 256, 36))(conv1)

    conv2 = Conv3D(8, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool1)
    conv2 = Conv3D(8, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv2)
    pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
    conv2 = Reshape((128, 128, 72))(conv2)

    conv3 = Conv3D(16, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool2)
    conv3 = Conv3D(16, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv3)
    pool3 = MaxPooling3D(pool_size=(1, 2, 2))(conv3)
    conv3 = Reshape((64, 64, 144))(conv3)

    conv4 = Conv3D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool3)
    conv4 = Conv3D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(1, 2, 2))(drop4)
    drop4 = Reshape((32, 32, 288))(drop4)
    pool4 = Reshape((16, 16, 288))(pool4)

    conv5 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool4)
    conv5 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = UpSampling2D(size=(2, 2))(drop5)
    up6 = Conv2D(512, 2, activation="relu", padding="same", kernel_initializer="he_normal")(up6)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge6)
    conv6 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(256, 2, activation="relu", padding="same", kernel_initializer="he_normal")(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(128, 2, activation="relu", padding="same", kernel_initializer="he_normal")(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge8)
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(64, 2, activation="relu", padding="same", kernel_initializer="he_normal")(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge9)
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)
    conv9 = Conv2D(classes, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)
    conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

    return Model(inputs=inputs, outputs=conv10)


def segnet2D(classes=2, input_size=(256, 256, 1)):
    inputs = Input(input_size)

    conv1 = Conv2D(32, 3, activation="relu", padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, 3, activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1, mask1 =  MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation="relu", padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, 3, activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2, mask2 =  MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation="relu", padding="same")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, 3, activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3, mask3 =  MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation="relu", padding="same")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, 3, activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4, mask4 =  MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv4)

    up7 = MaxUnpooling2D()([pool4, mask4])
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(conv7)
    conv7 = BatchNormalization()(conv7)

    conv8 = Conv2D(128, 3, activation="relu", padding="same")(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = MaxUnpooling2D()([conv8, mask3])
    conv8 = Conv2D(128, 3, activation="relu", padding="same")(conv8)
    conv8 = BatchNormalization()(conv8)

    conv9 = Conv2D(64, 3, activation="relu", padding="same")(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = MaxUnpooling2D()([conv9, mask2])
    conv9 = Conv2D(64, 3, activation="relu", padding="same")(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(32, 3, activation="relu", padding="same")(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = MaxUnpooling2D()([conv10, mask1])
    conv10 = Conv2D(32, 3, activation="relu", padding="same")(conv10)
    conv10 = BatchNormalization()(conv10)

    conv11 = Conv2D(classes, 1, activation="relu")(conv10)
    conv12 = Conv2D(1, 1, activation="sigmoid")(conv11)

    return Model(inputs=inputs, outputs=conv12)