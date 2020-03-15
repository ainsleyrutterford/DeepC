import models.models as M
import models.losses as L
from keras.optimizers import Adam

def compile(arch='unet2D', loss='binary_cross', pretrained_weights=None):
    model = getattr(M, arch)()

    if loss == 'focal':
        model.compile(optimizer=Adam(lr = 1e-4), loss=L.binary_focal_loss(), metrics=["accuracy"])
    elif loss == 'dice':
        model.compile(optimizer=Adam(lr = 1e-4), loss=L.dice_coef_loss, metrics=["accuracy"])
    else:
        model.compile(optimizer=Adam(lr = 1e-4), loss="binary_crossentropy", metrics=["accuracy"])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model