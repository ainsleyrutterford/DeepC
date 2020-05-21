import models.models as M
import models.losses as L
from keras.optimizers import Adam, SGD

def compile(arch='unet2D', loss='binary_cross', pretrained_weights=None, size=256, lr=0.00005):
    model = getattr(M, arch)(size=size)

    if loss == 'focal':
        model.compile(optimizer=Adam(lr=lr), loss=L.binary_focal_loss(), metrics=["accuracy"])
    elif loss == 'dice':
        model.compile(optimizer=Adam(lr=lr), loss=L.dice_coef_loss, metrics=["accuracy"])
    else:
        model.compile(optimizer=Adam(lr=lr), loss="binary_crossentropy", metrics=["accuracy"])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model