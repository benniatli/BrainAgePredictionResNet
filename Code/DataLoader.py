import threading
from transformations import rotation_matrix
from random import gauss
from scipy.ndimage.interpolation import map_coordinates
import operator
from scipy.ndimage.interpolation import shift,rotate
import pandas as pd
import os
import scipy.ndimage as nd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from tensorflow.keras.models import load_model
sys.path.append('./SRC')
import random
import glob
import nibabel as nib
from collections import defaultdict
from tensorflow.keras.utils import Sequence

def loadMR(path):
    #img = nib.load(path).get_data()
    img = nib.load(path).get_fdata()
    return img

def getIcelandicData(imageType):
    path = '../Data/IcelandicData/'
    if imageType == 'RawT1':
        t1Paths  = glob.glob(path+'*/anatomy/CAT/mri/wm*.nii')
        par1 = 2
    elif imageType == 'GrayMatter':
        t1Paths  = glob.glob(path+'*/anatomy/CAT/mri/mwp1*.nii')
        par1 = 4
    elif imageType == 'WhiteMatter':
        t1Paths  = glob.glob(path+'*/anatomy/CAT/mri/mwp2*.nii')
        par1 = 4
    elif imageType == 'Jacobian':
        t1Paths  = glob.glob(path+'*/anatomy/CAT/mri/wj*.nii')
        par1 = 2
    uniqeKeyToPath = defaultdict(str)
    uniqeKeyToAsc = defaultdict(str)
    for i in t1Paths:
        img_info = i.split('/')[-1].split('_')
        uniqeKey = img_info[0][par1:]+'_'+img_info[3]
        uniqeKeyToPath[uniqeKey] = i
        uniqeKeyToAsc[uniqeKey] = img_info[0][par1:]

    uniqeKey = pd.DataFrame.from_dict(uniqeKeyToAsc,orient='index')
    uniqeKey = uniqeKey.rename(columns={0: "Asc"})
    uniqeKey['Asc'] = uniqeKey['Asc'].astype(int)
    uniqeKey['Key'] = uniqeKey.index

    pnData = getPNData()
    data = pd.merge(uniqeKey,pnData,how='inner',on='Asc')
    data['Loc'] = [uniqeKeyToPath[x] for x in data['Key'].values]
    data = data[data['PsychCNV'].isin(['0'])]
    data = data[data['Case'].isin(['0','ADHD'])]
    data.drop_duplicates('PN')
    return data

def getPNData():
    subjectInfoPath = "../Data/MASTER_SUBJECT_INFO_1619PN_20112017.csv"
    subjectInfo = pd.read_csv(subjectInfoPath)
    subjectInfo = subjectInfo[['PN','Asc','Gender','YOB','YoScan','Case','Scanner','PsychCNV']].dropna()
    subjectInfo['Age'] = subjectInfo['YoScan']-subjectInfo['YOB'].values
    subjectInfo = subjectInfo.drop(['YOB','YoScan'],1)
    subjectInfo['Scanner'] = np.zeros(subjectInfo.shape[0])
    subjectInfo['Gender'] = subjectInfo['Gender'].astype('category').cat.codes 
    return subjectInfo

def getIXIData(imageType):
    path = '../Data/IXI/RawData/'
    if imageType == 'RawT1':
        paths = glob.glob(path+'*/T1/mri/wm*.nii')
    elif imageType == 'GrayMatter':
        paths = glob.glob(path+'*/T1/mri/mwp1*.nii')
    elif imageType == 'WhiteMatter':
        paths = glob.glob(path+'*/T1/mri/mwp2*.nii')
    elif imageType == 'Jacobian':
        t1Paths  = glob.glob(path+'*/T1/mri/wj*.nii')
    idToPath = defaultdict(list)
    
    for i in paths:
        idToPath[i.split('/')[4][:7]] = idToPath[i.split('/')[4][:7]]+[i]
    
    idFrame = pd.DataFrame.from_dict(idToPath,orient='index')
    idFrame['ID'] = idFrame.index
    
    subjectInfoPath = '../Data/T1_Demographic_Information.txt'
    subjectInfo = pd.read_csv(subjectInfoPath,delim_whitespace=True).dropna()
    subjectInfo = subjectInfo[['ID','Scanner','Gender','Age']]
    subjectInfo = subjectInfo.rename(columns={'Scanner':'scanner','Gender':'gender'})
    subjectInfo['gender'] = subjectInfo['gender'].astype(int).astype('category').cat.codes
    subjectInfo['scanner'] = np.zeros(subjectInfo.shape[0])
    data = pd.merge(subjectInfo,idFrame,how='inner',on='ID')
    data.columns = ['ID', 'Scanner', 'Gender', 'Age', 'Loc']
    train,val = train_test_split(data,test_size = 0.2,random_state=257572)
    return train,val

def getUKBData(imageType):
    path = '../Data/CAT_UK_Biobank/'
    if imageType == 'RawT1':
        paths = glob.glob(path+'*/mri/wm*.nii')
    elif imageType == 'GrayMatter':
        paths = glob.glob(path+'*/mri/mwp1*.nii')
    elif imageType == 'WhiteMatter':
        paths = glob.glob(path+'*/mri/mwp2*.nii')
    elif imageType == 'Jacobian':
        paths = glob.glob(path+'*/mri/wj*.nii')
    idToPath = defaultdict(list)
    for i in paths:
        idToPath[i.split('/')[3][:7]] = idToPath[i.split('/')[3][:7]]+[i]
    ids = list(idToPath.keys())
    idFrame = pd.DataFrame.from_dict(idToPath,orient='index')
    idFrame['ID'] = idFrame.index.astype(int)
    subjectInfoPath = '../Data/ID_Gender_YOB_MOB.txt'
    ageInfo = pd.read_csv('../Data/UK_Biobank_ImagingVisitAge2.txt',delim_whitespace=True)
    ageInfo.columns = ['ID','Age']
    subjectInfo = pd.read_csv(subjectInfoPath,delim_whitespace=True)
    subjectInfo = subjectInfo.drop(subjectInfo.index[0])
    subjectInfo.columns = ['ID','gender','YOB','month']
    subjectInfo = pd.merge(subjectInfo,ageInfo,how='inner',on='ID')
    tmp = pd.read_csv('../Data/ukb2489_id2pn.txt',header=None,sep='\t')
    tmp.columns  = ['ID','PN']
    subjectInfo = pd.merge(tmp,subjectInfo,on='ID',how='inner')
    data = pd.merge(subjectInfo,idFrame,how='inner',on=['ID','ID'])
    data['scanner'] = np.zeros(data.shape[0])
    data['gender'] = 1 - data['gender']
    data = data.rename(columns={'scanner':'Scanner','gender':'Gender',0:'Loc'})
    return data

class dataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, features, labels, batch_size=32,meanImg=None, dim=(121, 145, 121),maxAngle=40,maxShift=10, shuffle=True,augment=False):
        'Initialization'
        self.batch_size = batch_size
        self.features = features
        self.labels = labels
        self.dim = dim
        self.meanImg = meanImg
        self.augment = augment
        self.maxAngle = maxAngle
        self.maxShift = maxShift
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.features[0].shape[0] / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.features[0]))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #print(index)
        index = index%self.__len__()
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        #features_temp = [self.features[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X = np.empty((self.batch_size, self.dim[0],self.dim[1],self.dim[2]),dtype=np.uint8)
        X = np.empty((self.batch_size, self.dim[0],self.dim[1],self.dim[2], 1))
        age = np.empty((self.batch_size))
        sex = np.empty((self.batch_size))
        scanner = np.empty((self.batch_size))
        # Generate data
        for i, index in enumerate(indexes):
            X[i,:,:,:,:] = processing(self.features[0][index],self.dim,self.meanImg,augment=self.augment)
            scanner[i] = self.features[1][index]
            sex[i] = self.features[2][index]
            age[i] = self.labels[index]
            
        return [X,scanner,sex], [age]


def resize3d(image,new_shape,order=3):
    real_resize_factor = tuple(map(operator.truediv, new_shape, image.shape))
    return nd.zoom(image, real_resize_factor, order=order)

def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def coordinateTransformWrapper(X_T1,maxDeg=40,maxShift=7.5):
    randomAngle = np.radians(maxDeg*2*(random.random()-0.5))
    unitVec = tuple(make_rand_vector(3))
    shiftVec = [maxShift*2*(random.random()-0.5),
                maxShift*2*(random.random()-0.5),
                maxShift*2*(random.random()-0.5)]
    X_T1 = coordinateTransform(X_T1,randomAngle,unitVec,shiftVec)
    return X_T1

def coordinateTransform(vol,randomAngle,unitVec,shiftVec,order=1,mode='constant'):
    #from transformations import rotation_matrix
    ax = (list(vol.shape))
    ax = [ ax[i] for i in [1,0,2]]
    coords=np.meshgrid(np.arange(ax[0]),np.arange(ax[1]),np.arange(ax[2]))

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz=np.vstack([coords[0].reshape(-1)-float(ax[0])/2,     # x coordinate, centered
               coords[1].reshape(-1)-float(ax[1])/2,     # y coordinate, centered
               coords[2].reshape(-1)-float(ax[2])/2,     # z coordinate, centered
               np.ones((ax[0],ax[1],ax[2])).reshape(-1)])    # 1 for homogeneous coordinates
    
    # create transformation matrix
    mat=rotation_matrix(randomAngle,unitVec)

    # apply transformation
    transformed_xyz=np.dot(mat, xyz)

    # extract coordinates, don't use transformed_xyz[3,:] that's the homogeneous coordinate, always 1
    x=transformed_xyz[0,:]+float(ax[0])/2+shiftVec[0]
    y=transformed_xyz[1,:]+float(ax[1])/2+shiftVec[1]
    z=transformed_xyz[2,:]+float(ax[2])/2+shiftVec[2]
    x=x.reshape((ax[1],ax[0],ax[2]))
    y=y.reshape((ax[1],ax[0],ax[2]))
    z=z.reshape((ax[1],ax[0],ax[2]))
    new_xyz=[y,x,z]
    new_vol=map_coordinates(vol,new_xyz, order=order,mode=mode)
    return new_vol


def processing(features,inputShape,meanImg,maxAngle=40,maxShift=10,resizeImg=False,augment=False,training=False):
    X_T1 = loadMR(features)
    if meanImg is not None:
        X_T1 = X_T1-meanImg
    if augment:
        X_T1 = coordinateTransformWrapper(X_T1,maxDeg=maxAngle,maxShift=maxShift)
    if resizeImg:
        inputShape = (121, 145, 121)
        X_T1 = resize3d(X_T1,inputShape)
    return X_T1.reshape(inputShape+(1,))