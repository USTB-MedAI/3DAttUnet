#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import SimpleITK as sitk
import itk
import numpy as np
'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
'''

def load_mhd(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)   
    ct_scan = sitk.GetArrayFromImage(itkimage) 
    origin = itkimage.GetOrigin()
    spacing = itkimage.GetSpacing()
    return ct_scan, origin, spacing
    

def save_mhd(image,filename,origin,spacing):
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename, True)  

def NormalizeImageData(image):
    maxValue =  np.max(image)
    minVale = np.min(image)
    image = (image - minVale) / (maxValue - minVale)
   
    return image


def NormalizeTrainData(train):
    train = NormalizeImageData(train)
    train = np.expand_dims(train,axis=-1)
   
    return train

def expand_dim(data):
    data = np.expand_dims(data,axis=0)
   
    return data

#seperate label date into 4 classes of background, left lung, right lung and vessel
def SeperateLabelData(label, nClass=4):
    mutliLabel = np.zeros((label.shape[0],label.shape[1],label.shape[2],nClass))
    mutliLabel[(label==0),0] = 1   #background
    mutliLabel[(label==3),1] = 1   #left lung
    mutliLabel[(label==4),2] = 1   #right lung
    mutliLabel[(label==5),3] = 1   #vessel

    return mutliLabel

def SeperateBinaryLabelData(label, nClass=2):
    mutliLabel = np.zeros((label.shape[0],label.shape[1],label.shape[2],nClass))
    mutliLabel[(label==0),0] = 1   #background
    mutliLabel[(label==1),1] = 1   #lung


    return mutliLabel

def SeperateBinaryLabelData1(label, nClass=2):
    mutliLabel = np.zeros((label.shape[0],label.shape[1],label.shape[2],label.shape[3],nClass))
    mutliLabel[(label==0),0] = 1   #background
    mutliLabel[(label==1),1] = 1   #lung


    return mutliLabel

def ComposeLabelData(inputMultiChanelLabel):
    
    outputMutliLabel = np.zeros((inputMultiChanelLabel.shape[0],inputMultiChanelLabel.shape[1],inputMultiChanelLabel.shape[2]))
    
    outputMutliLabel[np.argmax(inputMultiChanelLabel, axis = 3) == 1] = 3
    outputMutliLabel[np.argmax(inputMultiChanelLabel, axis = 3) == 2] = 4
    outputMutliLabel[np.argmax(inputMultiChanelLabel, axis = 3) == 3] = 5

    return outputMutliLabel

def ComposeBinaryLabelData(inputMultiChanelLabel):
    
    outputMutliLabel = np.zeros((inputMultiChanelLabel.shape[0],inputMultiChanelLabel.shape[1],inputMultiChanelLabel.shape[2]))   
    outputMutliLabel[np.argmax(inputMultiChanelLabel, axis = 3) == 1] = 1   

    return outputMutliLabel

def ComposeBinary3DLabelData(inputMultiChanelLabel):
    
    outputMutliLabel = np.zeros((inputMultiChanelLabel.shape[0],inputMultiChanelLabel.shape[1],inputMultiChanelLabel.shape[2]),inputMultiChanelLabel.shape[3])
    
    outputMutliLabel[np.argmax(inputMultiChanelLabel, axis = 4) == 1] = 1

    return outputMutliLabel

def reduce_dim(data):
    data = np.squeeze(data,axis=0)   
    print('test= ', data.shape)
    data = ComposeBinaryLabelData(data)    

    return data

def extract_3Dseg(data):
    data = np.squeeze(data,axis=0)
    print('data = ', data.shape)
    data = data[:,:,:,1]
    print('data = ', data.shape)
    return data

def extract_seg(data):
    # data = np.squeeze(data,axis=0)
    data = data[:,:,:,1]

    return data
     
   
