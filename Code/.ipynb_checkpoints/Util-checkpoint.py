import pandas as pd
import os
import scipy.ndimage as nd
from sklearn.metrics import r2_score,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from tensorflow.keras.models import load_model
import random
import glob
import nibabel as nib
from collections import defaultdict
from DataLoader import processing

dataShape = (121,145,121)

def loadHeader(ID,idToPath):
    header = nib.load(idToPath[ID][0]).header
    return header

def loadMR(path):
    #img = nib.load(path).get_data()
    img = nib.load(path).get_fdata()
    return img

def calculateMeanImg(paths):
    imgSum = np.zeros(loadMR(paths[0]).shape)
    for i,path in enumerate(paths):
        if i%250==0:
            print(str(i)+' done')
        img = loadMR(path)
        imgSum += img
    return imgSum/float(paths.shape[0])

def plotLayerWeights(model,layerNr=1):
    #x1w = model.get_layer(index=layerNr).get_weights()[0][:,:,:,0,:]
    x1w = model.get_layer(index=layerNr).get_weights()[0][:,:,:,:,:]
    numberOfFeatureMaps = x1w[:,:,1,:,:].shape[3]
    grayOrWhite = ['Gray Matter','White Matter']
    for i in range(0,numberOfFeatureMaps):
        for k in range(2):
            print('Kernel Nr. '+str(i+1)+' :'+grayOrWhite[k])
            for j in range(0,3):
                plt.subplot(1,3,j+1)
                plt.imshow(x1w[:,:,j,k,i],interpolation="nearest",cmap="gray")
            plt.show()
        

def getFeatureMaps(model,features,layerNr=27,subjectNr=1,figSize=(20,10),max_iLen=None,max_jLen=None,c=0,d=1,inputShape=dataShape):
    intermediate_layer_model = Model(input = model.input, output = model.get_layer(index=layerNr).output)
    t1_img,scanner,gender = getBatchData(features[subjectNr:subjectNr+1,:],1,resizeImg=True)
    intermediate_output = intermediate_layer_model.predict([t1_img,scanner,gender])
    print(intermediate_output.shape)
    iLen = intermediate_output.shape[4]
    jLen = intermediate_output.shape[3]
    if max_iLen!=None:
        iLen = max_iLen
    if max_iLen!=None:
        jLen = max_jLen
    for i in range(iLen):
        print('Feature map nr. '+str(i))
        plt.figure(figsize=figSize)
        for j in range(jLen):
            plt.subplot(1,jLen,j+1)
            plt.imshow(intermediate_output[0,:,:,d*(j+c),i],interpolation="spline16",cmap="gray")
        plt.show()

        
def plotData(batch_data,subjectNr=1,c=5,d = 12,nSlices = 10,figsize=(20,10),inputShape=dataShape,augment=False,training=False,resize=True):
    fig,axs = plt.subplots(1, nSlices,figsize=figsize)
    for i in range(nSlices):
        axs[i].imshow(batch_data[0,:,:,d*(i+c),0],interpolation="spline16",cmap='gray')
    plt.show()        

    
def getBatchData(features, batch_size,inputShape=dataShape,resizeImg=False,augment=False):
    inputShape=(121, 145, 121)
    batch_T1 = np.zeros((batch_size,)+inputShape+(1,))
    scanner = np.zeros((batch_size,1))
    gender = np.zeros((batch_size,1))
    batch_labels = np.zeros((batch_size,1))
    index=0
    for i in range(batch_size):
        batch_T1[i],scanner[i],gender[i] = processing(features[i,:],inputShape,resizeImg,augment)
        index=1+index
    return batch_T1,scanner,gender

def getPredictions(model,X,inputShape=dataShape,batchSize=1,resizeImg=False,roundPredictions=False):
    predictions = np.array([])
    X = np.array_split(X, X.shape[0]/batchSize)
    for i,batch in enumerate(X):
        #plotData(X,subjectNr=i,augment=False,resize=False,c=2,d=10,nSlices=8)
        if i%250==0:
            print(i)
        img,scanner,gender = getBatchData(batch,batch.shape[0],resizeImg=resizeImg)
        if roundPredictions:
            predictions = np.append(predictions,int(model.predict([img,scanner,gender],1)))
        else:
            pred = model.predict([img,scanner,gender],batch_size=batch.shape[0])
            #print(pred)
            predictions = np.append(predictions,pred)
    return predictions