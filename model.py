import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras import backend as K
import tensorflow as tf





def dice_pred(y_true,y_pred,smooth=1):
    intersection = keras.sum(y_true * y_pred, axis=[1,2,3,4])
    union = keras.sum(y_true, axis=[1,2,3,4]) + keras.sum(y_pred, axis=[1,2,3,4])
    return keras.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def spatial_attention3D(input_feature, kernel_size=3):
        avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input_feature)
        max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input_feature)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        cbam_feature = Conv3D(filters=1,
                              kernel_size=kernel_size,
                              strides=1,
                              padding='same',
                              activation='sigmoid',
                              kernel_initializer='he_normal',
                              use_bias=False)(concat)
        return multiply([input_feature, cbam_feature])

'''
Definitions of loss and evaluation metrices
'''
def dice_coef(y_true, y_pred):
    smooth=1
    beta=5
    intersection = keras.sum(y_true * y_pred, axis=[1,2,3,4])
    union =keras.sum(y_true, axis=[1,2,3,4]) + keras.sum(y_pred, axis=[1,2,3,4])
    # return keras.mean( (2. * intersection + smooth) / (2*(1-beta)*intersection+beta*union + smooth), axis=0)
    return keras.mean( (2*intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

#def binary_crossentropy(y_true, y_pred):
 #   return Keras.mean(Keras.binary_crossentropy(y_pred, y_true), axis=-1)

def binary_crossentropy(y_true, y_pred):
    return keras.binary_crossentropy(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return 0.8*keras.binary_crossentropy(y_true, y_pred) + 0.2*dice_coef_loss(y_true, y_pred)

'''
Definitions of networks
'''   
def Test(input_size = (64,64,64,1), nClass = 2):
    inputs = Input(input_size)
    conv1 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    d1 = spatial_attention3D(conv1)

    pool1 = MaxPooling3D(pool_size=(2, 2 ,2))(conv1)
    conv2 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    d2 = spatial_attention3D(conv2)

    pool2 = MaxPooling3D(pool_size=(2, 2 ,2))(conv2)
    conv3 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    d3 = spatial_attention3D(conv3)

    pool3 = MaxPooling3D(pool_size=(2, 2 ,2))(conv3)
    conv4 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    d4 = spatial_attention3D(drop4)

    pool4 = MaxPooling3D(pool_size=(2, 2 ,2))(drop4)
    conv5 = Conv3D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv3D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv3D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(drop5))
    merge6 = concatenate([d4,up6], axis = 4)
    conv6 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)


    up7 = Conv3D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv6))
    merge7 = concatenate([d3,up7], axis = 4)
    conv7 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)


    up8 = Conv3D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv7))
    merge8 = concatenate([d2,up8], axis = 4)
    conv8 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)


    up9 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv8))
    merge9 = concatenate([d1,up9], axis = 4)
    conv9 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)  

 
    conv10 = Conv3D(nClass, 3, activation = 'softmax', padding = 'same', kernel_initializer = 'he_normal')(conv9) 
    model = Model(input = inputs, output = conv10)

    model.compile(optimizer =Adam(lr = 1e-5), loss = bce_dice_loss, metrics = ['accuracy',dice_pred])
  

    return model


