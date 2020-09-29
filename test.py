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
from medpy.io import load,save
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


'''
Definitions of loss and evaluation metrices
'''
if __name__== "__main__":

	model = load_model('Our 1000.h5',custom_objects={'bce_dice_loss':bce_dice_loss,'dice_pred':dice_pred})  
	file_pre = 'Cov19Data/test/train/64/train_' 
	file_end = '.mhd'

	file_end1 = '_Unet_save.mhd'

	savefile_pre = 'Results/AttThreeDSC/nodule_'
	for index_file in range(1,45):   
		
		if index_file < 10:
			imagefilename = file_pre + str(index_file) + file_end
			data, origin, spacing = load_mhd(imagefilename)
			data = NormalizeTrainData(data)
			data = expand_dim(data)
			print('data.shape = ', data.shape)
		else:
			imagefilename = file_pre + str(index_file) + file_end
			data, origin, spacing = load_mhd(imagefilename)
			data = NormalizeTrainData(data)
			data = expand_dim(data)
			print('data.shape = ', data.shape)
		

		predictResult1 = model.predict(x=data,batch_size=4)  
		print('predictResult1.shape = ', predictResult1.shape)

		saveResult1 = reduce_dim(predictResult1) 
		
		print('saveResult1 = ', saveResult1.shape)

		saveimagefilename1 = savefile_pre + str(index_file) + file_end1
		save_mhd(saveResult1, saveimagefilename1,origin,spacing)


	print('sucess!')
