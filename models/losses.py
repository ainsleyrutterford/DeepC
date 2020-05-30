import tensorflow as tf
from keras import backend as K

def binary_focal_loss(gamma=2., alpha=.25):
    """
        An implementation of the focal loss function introduced in:

        https://arxiv.org/abs/1708.02002

        This Keras implementation is taken from:

        https://github.com/mkocabas/focal-loss-keras

        Args:
            gamma: (float) the "focusing" parameter
            alpha: (float) the weighting factor
        Returns:
            loss: (arr) a tensor containing the loss values.
        """

    def binary_focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed