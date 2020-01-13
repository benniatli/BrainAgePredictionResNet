from keras.models import Sequential
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge, add
from keras.layers.convolutional import Conv3D
from keras.layers.pooling import MaxPooling3D,AveragePooling3D
from keras.regularizers import l2 as L2
from keras.layers.normalization import BatchNormalization as BatchNorm
from keras.models import Model
from batch_renorm import BatchRenormalization
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam, SGD,Adagrad
from keras.layers.advanced_activations import LeakyReLU,ELU
from keras.layers.merge import concatenate
from keras.layers import GlobalAveragePooling3D
import tensorflow as tf
import pandas as pd
import os
import scipy.ndimage as nd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from keras.models import load_model
sys.path.append('./SRC')
import random
import glob
import nibabel as nib
from collections import defaultdict

def regression_hinge(y_true, y_pred):
    epsilon = 2
    return K.mean(K.maximum(K.abs(y_true - y_pred) - epsilon, 0.), axis=-1)

def covarModel(learningRate,decayRate=0.03,momentum=0.9,inputShape=dataShape,paddingType = 'same',initType='he_uniform',regAmount=0.00005):
    K.__dict__["gradients"] = gradients_speed
    t1Input = Input(inputShape+(1,), name='T1_Img')
    scanner  = Input((1,), name='Scanner')
    gender  = Input((1,), name='Gender')
        
    with tf.name_scope('ResBlock0'):
        inputs = t1Input
        features = 8
        hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=L2(regAmount),kernel_initializer=initType)(inputs)
        hidden = BatchRenormalization(axis=-1)(hidden)
        hidden = ELU(alpha=1.0)(hidden)
        hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=L2(regAmount),kernel_initializer=initType)(hidden)
        hidden = BatchRenormalization(axis=-1)(hidden)
        shortcut = Conv3D(features, (1,1,1), strides=(1,1,1), padding=paddingType,kernel_initializer=initType)(inputs)
        hidden = add([shortcut,hidden])
        outputs = ELU(alpha=1.0)(hidden)
        
    pooling = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2), padding=paddingType)(outputs)
    
    with tf.name_scope('ResBlock1'):
        inputs = pooling
        features = 16
        hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=L2(regAmount),kernel_initializer=initType)(inputs)
        hidden = BatchRenormalization(axis=-1)(hidden)
        hidden = ELU(alpha=1.0)(hidden)
        hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=L2(regAmount),kernel_initializer=initType)(hidden)
        hidden = BatchRenormalization(axis=-1)(hidden)
        shortcut = Conv3D(features, (1,1,1), strides=(1,1,1), padding=paddingType,kernel_initializer=initType)(inputs)
        hidden = add([shortcut,hidden])
        outputs = ELU(alpha=1.0)(hidden)
        
    pooling = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2), padding=paddingType)(outputs)

    with tf.name_scope('ResBlock2'):
        inputs = pooling
        features = 32
        hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=L2(regAmount),kernel_initializer=initType)(inputs)
        hidden = BatchRenormalization(axis=-1)(hidden)
        hidden = ELU(alpha=1.0)(hidden)
        hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=L2(regAmount),kernel_initializer=initType)(hidden)
        hidden = BatchRenormalization(axis=-1)(hidden)
        shortcut = Conv3D(features, (1,1,1), strides=(1,1,1), padding=paddingType,kernel_initializer=initType)(inputs)
        hidden = add([shortcut,hidden])
        outputs = ELU(alpha=1.0)(hidden)
        
    pooling = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2), padding=paddingType)(outputs)
    
    with tf.name_scope('ResBlock3'):
        inputs = pooling
        features = 64
        hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=L2(regAmount),kernel_initializer=initType)(inputs)
        hidden = BatchRenormalization(axis=-1)(hidden)
        hidden = ELU(alpha=1.0)(hidden)
        hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=L2(regAmount),kernel_initializer=initType)(hidden)
        hidden = BatchRenormalization(axis=-1)(hidden)
        shortcut = Conv3D(features, (1,1,1), strides=(1,1,1), padding=paddingType,kernel_initializer=initType)(inputs)
        hidden = add([shortcut,hidden])
        outputs = ELU(alpha=1.0)(hidden)
        
        
    pooling = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2), padding=paddingType)(outputs)
    
    with tf.name_scope('ResBlock4'):
        inputs = pooling
        features = 128
        hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=L2(regAmount),kernel_initializer=initType)(inputs)
        hidden = BatchRenormalization(axis=-1)(hidden)
        hidden = ELU(alpha=1.0)(hidden)
        hidden = Conv3D(features, (3, 3, 3), padding=paddingType,kernel_regularizer=L2(regAmount),kernel_initializer=initType)(hidden)
        hidden = BatchRenormalization(axis=-1)(hidden)
        shortcut = Conv3D(features, (1,1,1), strides=(1,1,1), padding=paddingType,kernel_initializer=initType)(inputs)
        hidden = add([shortcut,hidden])
        outputs= ELU(alpha=1.0)(hidden)
        
    pooling = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2), padding=paddingType)(outputs)
        
    hidden = Flatten()(pooling)
    
    hidden = Dense(128,kernel_regularizer=L2(regAmount),kernel_initializer=initType,name='FullyConnectedLayer')(hidden)
    hidden = ELU(alpha=1.0)(hidden)
    hidden = Dropout(0.2)(hidden)
    
    hidden = concatenate([scanner,gender,hidden])
    
    prediction = Dense(1,kernel_regularizer=L2(regAmount), name='AgePrediction')(hidden)
    model = Model(inputs=[t1Input,scanner,gender],outputs=prediction)
    adam = Adam(lr=learningRate, decay=decayRate)
    
    model.compile(loss='mean_absolute_error',optimizer=adam,metrics=['mae','mse'])
    return model