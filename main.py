#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 

print('using GPU1')

import tensorflow
import keras
from keras.models import load_model
from model import *
from BasicFunction import *
from unet import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils.data_utils import Sequence
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from augmented import *


def readTrainingData():

	file_pre = 'Cov19Data/train/train/64/train_'	
	file_end = '.mhd'
	data, origin, spacing = load_mhd(file_pre  + '1' + file_end)
	data = NormalizeTrainData(data)   
	print('data.shape = ', data.shape)
	data = expand_dim(data)
	trainImage = data
	
	for index_file in range(2,46):   
		if index_file < 10:
			imagefilename = file_pre + str(index_file) + file_end
			print(imagefilename)
			data, origin, spacing = load_mhd(imagefilename)
			data = NormalizeTrainData(data)
			print('data.shape = ', data.shape)
			data = expand_dim(data)   
			trainImage = np.concatenate((trainImage,data), axis = 0)
		else:
			imagefilename = file_pre + str(index_file) + file_end
			print(imagefilename)
			data, origin, spacing = load_mhd(imagefilename)
			data = NormalizeTrainData(data)
			print('data.shape = ', data.shape)
			data = expand_dim(data)
			trainImage = np.concatenate((trainImage,data), axis = 0)
	
	print(trainImage.shape)

	return trainImage

def readTrainingLabel():

	file_pre = 'Cov19Data/train/label/64/label_'	
	file_end = '.mhd'
	label, origin, spacing = load_mhd(file_pre + '1' + file_end)
	nClass = 2
	label = SeperateBinaryLabelData(label, nClass)  
	label = expand_dim(label)	
	labelImage = label

	for index_file in range(2,46):   
		if index_file < 10:
			imagefilename = file_pre + str(index_file) + file_end
			print(imagefilename)
			label, origin, spacing = load_mhd(imagefilename)
			nClass = 2
			label = SeperateBinaryLabelData(label, nClass)
			label = expand_dim(label)
			print('label.shape = ', label.shape)
			labelImage = np.concatenate((labelImage,label), axis = 0)
		else:
			imagefilename = file_pre + str(index_file) + file_end
			print(imagefilename)
			label, origin, spacing = load_mhd(imagefilename)
			nClass = 2
			label = SeperateBinaryLabelData(label, nClass)
			label = expand_dim(label)
			print('label.shape = ', label.shape)
			labelImage = np.concatenate((labelImage,label), axis = 0)
	
	print(labelImage.shape)

	return labelImage



if __name__== "__main__":

	print('start!')    
	print('read training Image')
	trainImage = readTrainingData()

	print('read label Image')
	labelImage = readTrainingLabel()
	print('prepare complete!')

	# data augment
	image_aug = customImageDataGenerator(rotation_range = 20)
	mask_aug = customImageDataGenerator(rotation_range = 20)

	trainImage_nd = image_aug.flow(trainImage, batch_size= 4, seed= 42) 
	labelImag_nd = mask_aug.flow(labelImage, batch_size = 4, seed= 42) 

	input_test = zip(trainImage_nd,labelImag_nd)
	
	model = Test(input_size = (64,64,64,1), nClass = 2)   
	hist = model.fit_generator(input_test,steps_per_epoch=trainImage.shape[0]//4,epochs=2000)

	with open('AttThreeDSC.txt','w') as f:
		f.write(str(hist.history))
	model.save('AttThreeDSC.h5')  
	model.summary()

	
	print('sucess!')


