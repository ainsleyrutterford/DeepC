import models.models as M
import models.losses as L
from keras.optimizers import Adam, SGD

def compile(arch="unet2D", loss="binary_cross", weights=None, size=256, lr=0.00005, abl=False):
    """
    Compile a Keras model.

    Args:
        arch: (str) defines which architecture to compile ("unet2D" | "unet3D" | "segnet2D")
        loss: (str) defines the loss function to be used ("binary_cross" | "focal")
        weights (str) the name of the file that the weights are saved in.
        size: (int) the x and y dimensions of the images.
        lr: (float) the initial learning rate.
        abl: (bool) whether or not to define the ablated architecture.
    Returns:
        model: (Keras Model) a compiled Keras model containing the specified architecture.
    """

    model = getattr(M, arch)(size=size, ablated=abl)

    if loss == "focal":
        model.compile(optimizer=Adam(lr=lr), loss=L.binary_focal_loss(), metrics=["accuracy"])
    else:
        model.compile(optimizer=Adam(lr=lr), loss="binary_crossentropy", metrics=["accuracy"])

    # Load pretrained weights if they are available.
    if weights:
        model.load_weights(weights)

    return model