# metrics.py

import tensorflow as tf

smooth = 1e-7

def dice_coef(y_true, y_pred):
    y_true = tf.reshape(y_true,   shape=[-1])
    y_pred = tf.reshape(y_pred,   shape=[-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    # binarize at 0.5
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f  = tf.cast(y_true > 0.5, tf.float32)

    # compute per-sample intersection & union
    axes = [1,2,3]  # sum over H, W, channels
    intersection = tf.reduce_sum(y_true_f * y_pred_bin, axis=axes)
    union        = tf.reduce_sum(y_true_f, axis=axes) \
                 + tf.reduce_sum(y_pred_bin, axis=axes) \
                 - intersection

    iou_per_sample = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou_per_sample)



import tensorflow as tf
from tensorflow.keras import backend as K

def soft_cldice_loss(iter_=3, smooth=1.0):
    def soft_skel(x):
        def erode(x):
            return 1.0 - tf.nn.max_pool2d(1.0 - x, ksize=3, strides=1, padding='SAME')
        skel = x
        for _ in range(iter_):
            skel = skel * erode(skel)
        return skel

    def loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, 0.0, 1.0)
        S_true = soft_skel(y_true)
        S_pred = soft_skel(y_pred)

        inter1 = K.sum(y_pred * S_true)
        inter2 = K.sum(y_true * S_pred)
        sum1   = K.sum(y_pred) + K.sum(S_true)
        sum2   = K.sum(y_true) + K.sum(S_pred)

        C = (2.0 * inter1 + smooth) / (sum1 + smooth)
        D = (2.0 * inter2 + smooth) / (sum2 + smooth)

        return 1.0 - (C * D)

    return loss_fn

bce = tf.keras.losses.BinaryCrossentropy()
def combined_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)           # or 1 – dice_coef
    cld  = soft_cldice_loss(iter_=3, smooth=1.0)(y_true, y_pred)
    return 1.0*bce(y_true, y_pred) + 1.0*dice + 1.0*cld

